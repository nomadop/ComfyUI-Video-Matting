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
    """RMBG-2.0 inference node with batch support"""

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
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Batch size for inference (higher = faster but more VRAM)"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, rmbg_model, images, process_size=1024, batch_size=4):
        """Process batch of images through RMBG-2.0 with batch inference

        Args:
            rmbg_model: RMBG_MODEL dict
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)
            process_size: resolution for processing
            batch_size: number of frames to process at once

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = rmbg_model["model"]
        device = rmbg_model["device"]
        transform = rmbg_model["transform"]

        b, h, w, c = images.shape
        num_batches = (b + batch_size - 1) // batch_size

        pbar = comfy.utils.ProgressBar(num_batches)
        all_alphas = []

        # Precompute normalization mean/std on device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc='RMBG'):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, b)
                batch_frames = images[start_idx:end_idx]  # [batch, H, W, C]
                current_batch_size = batch_frames.shape[0]

                # Batch preprocessing on GPU
                # [batch, H, W, C] -> [batch, C, H, W]
                batch_tensor = batch_frames.permute(0, 3, 1, 2).to(device)

                # Resize on GPU using interpolate
                batch_resized = torch.nn.functional.interpolate(
                    batch_tensor,
                    size=(process_size, process_size),
                    mode='bilinear',
                    align_corners=False
                )

                # Normalize on GPU
                batch_normalized = (batch_resized - mean) / std

                # Batch inference
                pred = model(batch_normalized)[-1].sigmoid()  # [batch, 1, H, W]

                # Resize back to original on GPU
                pred_resized = torch.nn.functional.interpolate(
                    pred,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )  # [batch, 1, H, W]

                # Remove channel dim and move to CPU
                batch_alphas = pred_resized.squeeze(1).cpu()  # [batch, H, W]
                all_alphas.append(batch_alphas)

                pbar.update(1)

        # Concatenate all batches
        alpha_tensor = torch.cat(all_alphas, dim=0).float()  # [B, H, W]
        return (alpha_tensor,)
