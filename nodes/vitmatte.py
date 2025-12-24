"""
ViTMatte Refine Node

Standalone ViTMatte refinement for workflows without Cutie.
Use case: Pure MattingInference + ViTMatte, single image refinement.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
import comfy.utils


class ViTMatteRefine:
    """ViTMatte edge refinement node with built-in trimap generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vitmatte_model": ("VITMATTE_MODEL",),
                "images": ("IMAGE",),
                "alpha": ("MASK",),
            },
            "optional": {
                "max_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "Max size for ViTMatte (0=original)"}),
                "fg_thresh": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.05, "tooltip": "Foreground confidence threshold"}),
                "bg_thresh": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Background confidence threshold"}),
                "erode_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2, "tooltip": "Erosion kernel size for FG core"}),
                "erode_iter": ("INT", {"default": 1, "min": 0, "max": 5, "tooltip": "Erosion iterations"}),
                "dilate_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2, "tooltip": "Dilation kernel size for unknown band"}),
                "dilate_iter": ("INT", {"default": 2, "min": 0, "max": 5, "tooltip": "Dilation iterations"}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("alpha", "trimap")
    FUNCTION = "refine"
    CATEGORY = "Video Matting"

    def refine(self, vitmatte_model, images, alpha, max_size=0,
               fg_thresh=0.95, bg_thresh=0.05,
               erode_size=5, erode_iter=1, dilate_size=5, dilate_iter=2):
        """Refine alpha using ViTMatte with built-in trimap generation

        Args:
            vitmatte_model: VITMATTE_MODEL dict
            images: [B, H, W, C] tensor (ComfyUI IMAGE format)
            alpha: [B, H, W] tensor (input alpha for trimap generation)
            max_size: downscale if larger (0=original)
            fg_thresh, bg_thresh: thresholds for trimap generation
            erode_size/iter, dilate_size/iter: morphology params

        Returns:
            alpha: [B, H, W] refined alpha tensor
            trimap: [B, H, W] generated trimap tensor (for debugging)
        """
        model = vitmatte_model["model"]
        device = vitmatte_model["device"]
        use_fp16 = vitmatte_model["use_fp16"]

        # Convert to numpy
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        alpha_np = alpha.cpu().numpy()

        b, h, w, c = images_np.shape
        refined_alphas = []
        trimaps = []

        pbar = comfy.utils.ProgressBar(b)

        with torch.no_grad():
            for i in tqdm(range(b), desc='ViTMatte'):
                # Generate trimap from input alpha
                trimap = self._generate_trimap(
                    alpha_np[i], fg_thresh, bg_thresh,
                    erode_size, erode_iter, dilate_size, dilate_iter
                )
                trimaps.append(trimap)

                # Run ViTMatte
                refined = self._process_frame(
                    model, device, use_fp16,
                    images_np[i], trimap, max_size
                )
                refined_alphas.append(refined)
                pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(refined_alphas, axis=0)).float()
        trimap_tensor = torch.from_numpy(np.stack(trimaps, axis=0)).float()
        return (alpha_tensor, trimap_tensor)

    def _generate_trimap(self, alpha, fg_thresh, bg_thresh,
                         erode_size, erode_iter, dilate_size, dilate_iter):
        """Generate trimap from soft alpha"""
        h, w = alpha.shape
        trimap = np.ones((h, w), dtype=np.uint8) * 128

        fg_mask = (alpha > fg_thresh)
        bg_mask = (alpha < bg_thresh)

        trimap[fg_mask] = 255
        trimap[bg_mask] = 0

        # Get unknown region and dilate
        unknown_mask = (trimap == 128).astype(np.uint8)
        fg_mask_uint8 = (trimap == 255).astype(np.uint8)

        # Erode FG to get core
        if erode_iter > 0:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            fg_core = cv2.erode(fg_mask_uint8, erode_kernel, iterations=erode_iter)
        else:
            fg_core = fg_mask_uint8

        # Connected components: only keep unknown connected to FG
        fg_unknown = (fg_mask_uint8 | unknown_mask).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(fg_unknown)
        fg_labels = set(np.unique(labels[fg_mask_uint8 == 1]))
        unknown_connected = np.zeros_like(unknown_mask)
        for label in fg_labels:
            if label == 0:
                continue
            unknown_connected |= ((labels == label) & (unknown_mask == 1)).astype(np.uint8)

        # Dilate unknown band
        if dilate_iter > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            unknown_dilated = cv2.dilate(unknown_connected, dilate_kernel, iterations=dilate_iter)
        else:
            unknown_dilated = unknown_connected

        # Build final trimap
        trimap_final = np.zeros((h, w), dtype=np.uint8)
        trimap_final[unknown_dilated == 1] = 128
        trimap_final[fg_core == 1] = 255

        return trimap_final

    def _process_frame(self, model, device, use_fp16, image_rgb, trimap, max_size):
        """Process single frame through ViTMatte"""
        orig_h, orig_w = image_rgb.shape[:2]

        if max_size > 0 and max(orig_h, orig_w) > max_size:
            scale = max_size / max(orig_h, orig_w)
            new_w = int(orig_w * scale) // 32 * 32
            new_h = int(orig_h * scale) // 32 * 32
            image_small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            trimap_small = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            image_small = image_rgb
            trimap_small = trimap

        image_tensor = TF.to_tensor(Image.fromarray(image_small)).unsqueeze(0).to(device)
        trimap_tensor = TF.to_tensor(Image.fromarray(trimap_small).convert('L')).unsqueeze(0).to(device)

        if use_fp16 and device.type in ('cuda', 'mps'):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model({'image': image_tensor, 'trimap': trimap_tensor})
        else:
            output = model({'image': image_tensor, 'trimap': trimap_tensor})

        alpha = output['phas'].flatten(0, 2).float().cpu().numpy()

        if max_size > 0 and max(orig_h, orig_w) > max_size:
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return alpha


class TrimapVisualize:
    """Debug node: visualize trimap with traffic light overlay"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trimap": ("MASK",),
            },
            "optional": {
                "overlay_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Overlay transparency"}),
                "output_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.25, "tooltip": "Output scale (0.5=half size)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "Video Matting/Debug"

    def visualize(self, images, trimap, overlay_alpha=0.4, output_scale=1.0):
        """Visualize trimap with traffic light overlay

        Colors:
            Green = Foreground (255)
            Yellow = Unknown (128)
            Red (light) = Background (0)

        Args:
            images: [B, H, W, C] tensor (ComfyUI IMAGE format)
            trimap: [B, H, W] tensor (0/128/255 values)
            overlay_alpha: transparency of overlay
            output_scale: downsample output

        Returns:
            visualization: [B, H, W, C] tensor
        """
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        trimap_np = trimap.cpu().numpy()

        b, h, w, c = images_np.shape
        visualizations = []

        for i in range(b):
            vis = self._create_visualization(
                images_np[i], trimap_np[i], overlay_alpha, output_scale
            )
            visualizations.append(vis)

        vis_np = np.stack(visualizations, axis=0)
        vis_tensor = torch.from_numpy(vis_np).float() / 255.0
        return (vis_tensor,)

    def _create_visualization(self, image_rgb, trimap, overlay_alpha, output_scale):
        """Create single visualization"""
        vis = image_rgb.copy().astype(np.float32)

        # Foreground - Green
        fg_mask = (trimap > 200)  # ~255
        vis[fg_mask] = vis[fg_mask] * (1 - overlay_alpha) + np.array([0, 255, 0]) * overlay_alpha

        # Background - Red (light)
        bg_mask = (trimap < 50)  # ~0
        vis[bg_mask] = vis[bg_mask] * (1 - overlay_alpha) + np.array([180, 80, 80]) * overlay_alpha

        # Unknown - Yellow
        unknown_mask = (trimap >= 50) & (trimap <= 200)  # ~128
        vis[unknown_mask] = vis[unknown_mask] * (1 - overlay_alpha) + np.array([255, 255, 0]) * overlay_alpha

        vis = vis.astype(np.uint8)

        # Downsample if needed
        if output_scale < 1.0:
            h, w = vis.shape[:2]
            new_w = int(w * output_scale)
            new_h = int(h * output_scale)
            vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return vis
