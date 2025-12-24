"""
Matting Inference Nodes (Split Version)

- RVMInference: RobustVideoMatting inference
- MODNetInference: MODNet inference
- U2NetInference: U-2-Net inference
"""

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from skimage import transform as skimage_transform
from tqdm import tqdm
import comfy.utils


# =============================================================================
# RVM Inference
# =============================================================================

class RVMInference:
    """RobustVideoMatting inference node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rvm_model": ("RVM_MODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, rvm_model, images):
        """Process batch of images through RVM

        Args:
            rvm_model: RVM_MODEL dict with model, device, downsample_ratio
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = rvm_model["model"]
        device = rvm_model["device"]
        downsample_ratio = rvm_model["downsample_ratio"]

        # Convert ComfyUI IMAGE [B,H,W,C] to numpy [B,H,W,C] RGB
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        num_frames = len(images_np)

        transform = transforms.ToTensor()
        alphas = []
        rec = [None] * 4  # RVM recurrent state

        pbar = comfy.utils.ProgressBar(num_frames)

        with torch.no_grad():
            for i, frame_rgb in enumerate(tqdm(images_np, desc='RVM')):
                src = transform(frame_rgb).to(device).unsqueeze(0).unsqueeze(0)
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                alpha = pha[0, 0, 0].cpu().numpy()
                alphas.append(alpha)
                pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(alphas, axis=0)).float()
        return (alpha_tensor,)


# =============================================================================
# MODNet Inference
# =============================================================================

class MODNetInference:
    """MODNet inference node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modnet_model": ("MODNET_MODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, modnet_model, images):
        """Process batch of images through MODNet

        Args:
            modnet_model: MODNET_MODEL dict with model, device, input_size
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = modnet_model["model"]
        device = modnet_model["device"]
        input_size = modnet_model["input_size"]

        # Convert ComfyUI IMAGE [B,H,W,C] to numpy [B,H,W,C] RGB
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        b, h, w, c = images_np.shape

        modnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Calculate MODNet processing size
        scale = input_size / max(w, h)
        modnet_w = int(w * scale) // 32 * 32
        modnet_h = int(h * scale) // 32 * 32

        alphas = []
        pbar = comfy.utils.ProgressBar(b)

        with torch.no_grad():
            for frame_rgb in tqdm(images_np, desc='MODNet'):
                frame_resized = cv2.resize(frame_rgb, (modnet_w, modnet_h))
                frame_pil = Image.fromarray(frame_resized)
                frame_tensor = modnet_transform(frame_pil).unsqueeze(0).to(device)

                _, _, matte = model(frame_tensor, True)
                alpha_small = matte[0, 0].cpu().numpy()
                alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
                alphas.append(alpha)
                pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(alphas, axis=0)).float()
        return (alpha_tensor,)


# =============================================================================
# U2Net Inference
# =============================================================================

class U2NetInference:
    """U-2-Net inference node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "u2net_model": ("U2NET_MODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, u2net_model, images):
        """Process batch of images through U-2-Net

        Args:
            u2net_model: U2NET_MODEL dict with model, device, input_size
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = u2net_model["model"]
        device = u2net_model["device"]
        input_size = u2net_model["input_size"]

        # Convert ComfyUI IMAGE [B,H,W,C] to numpy [B,H,W,C] RGB
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        num_frames = len(images_np)

        alphas = []
        pbar = comfy.utils.ProgressBar(num_frames)

        with torch.no_grad():
            for frame_rgb in tqdm(images_np, desc='U2Net'):
                h, w = frame_rgb.shape[:2]

                # U2Net preprocessing
                image_resized = skimage_transform.resize(
                    frame_rgb, (input_size, input_size), mode='constant'
                )
                tmpImg = np.zeros((input_size, input_size, 3))
                image_norm = image_resized / np.max(image_resized)
                tmpImg[:, :, 0] = (image_norm[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image_norm[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image_norm[:, :, 2] - 0.406) / 0.225
                tmpImg = tmpImg.transpose((2, 0, 1))
                input_tensor = torch.from_numpy(tmpImg[None, ...]).float().to(device)

                # Inference
                d1, *_ = model(input_tensor)
                pred = d1[:, 0, :, :]
                ma, mi = torch.max(pred), torch.min(pred)
                pred = (pred - mi) / (ma - mi + 1e-8)
                pred = pred.squeeze().cpu().numpy()

                # Resize to original size
                alpha = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                alphas.append(alpha)
                pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(alphas, axis=0)).float()
        return (alpha_tensor,)
