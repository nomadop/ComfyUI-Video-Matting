"""
RMBG-2.0 Nodes

Background removal using BRIA AI's RMBG-2.0 model.
Uses transformers library - no source repo clone needed.

License: CC BY-NC 4.0 (Non-Commercial)
"""

import os
import torch
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import comfy.utils

# Load config for HuggingFace token
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
CONFIG = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f) or {}

# Lazy import for optional dependency
_rmbg_available = None


def check_rmbg_available():
    """Check if transformers is available for RMBG"""
    global _rmbg_available
    if _rmbg_available is None:
        try:
            from transformers import AutoModelForImageSegmentation
            _rmbg_available = True
        except ImportError:
            _rmbg_available = False
    return _rmbg_available


def get_hf_token():
    """Get HuggingFace token from config"""
    token = CONFIG.get('huggingface_token')
    if token and token != 'your_token_here':
        return token
    return None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class RMBGLoader:
    """RMBG-2.0 model loader (downloads from HuggingFace)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "briaai/RMBG-2.0",
                    "tooltip": "HuggingFace model ID"
                }),
            }
        }

    RETURN_TYPES = ("RMBG_MODEL",)
    RETURN_NAMES = ("rmbg_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, model_id):
        if not check_rmbg_available():
            raise ImportError(
                "RMBG-2.0 requires transformers. "
                "Install with: pip install transformers kornia"
            )

        from transformers import AutoModelForImageSegmentation
        from huggingface_hub import login

        device = get_device()

        # Login with HuggingFace token if available
        hf_token = get_hf_token()
        if hf_token:
            login(token=hf_token)

        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token
        )
        model.eval()
        model.to(device)

        # Standard normalization for RMBG
        transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return ({
            "model": model,
            "device": device,
            "transform": transform,
            "model_id": model_id,
        },)


class RMBGInference:
    """RMBG-2.0 inference node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rmbg_model": ("RMBG_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "process_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Processing resolution (model trained on 1024)"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, rmbg_model, images, process_size=1024):
        """Process batch of images through RMBG-2.0

        Args:
            rmbg_model: RMBG_MODEL dict
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)
            process_size: resolution for processing

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = rmbg_model["model"]
        device = rmbg_model["device"]
        transform = rmbg_model["transform"]

        b, h, w, c = images.shape
        alphas = []

        pbar = comfy.utils.ProgressBar(b)

        with torch.no_grad():
            for i in tqdm(range(b), desc='RMBG'):
                # Get single frame [H, W, C] -> PIL
                frame = images[i].cpu().numpy()
                frame_uint8 = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame_uint8)

                # Resize for processing
                orig_size = pil_image.size  # (W, H)
                resized = pil_image.resize((process_size, process_size), Image.BILINEAR)

                # To tensor and normalize
                tensor = transforms.ToTensor()(resized)  # [3, H, W]
                tensor = transform(tensor)  # normalize
                tensor = tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

                # Inference
                pred = model(tensor)[-1].sigmoid().cpu()  # [1, 1, H, W]
                pred = pred[0, 0]  # [H, W]

                # Resize back to original
                pred_pil = transforms.ToPILImage()(pred)
                pred_resized = pred_pil.resize(orig_size, Image.BILINEAR)
                alpha = np.array(pred_resized).astype(np.float32) / 255.0

                alphas.append(alpha)
                pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(alphas, axis=0)).float()
        return (alpha_tensor,)
