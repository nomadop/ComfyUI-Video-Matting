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

# Lazy import for optional dependencies
_rmbg_available = None
_torchao_available = None


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


def check_torchao_available():
    """Check if torchao is available for FP8 quantization"""
    global _torchao_available
    if _torchao_available is None:
        try:
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            _torchao_available = True
        except ImportError:
            _torchao_available = False
    return _torchao_available


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

    QUANTIZATION_MODES = ["none", "fp8"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "briaai/RMBG-2.0",
                    "tooltip": "HuggingFace model ID"
                }),
            },
            "optional": {
                "quantization": (cls.QUANTIZATION_MODES, {
                    "default": "none",
                    "tooltip": "fp8 = 4x faster on RTX 4090/5090 (requires torchao)"
                }),
                "compile_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use torch.compile for extra speed (slower first run)"
                }),
            }
        }

    RETURN_TYPES = ("RMBG_MODEL",)
    RETURN_NAMES = ("rmbg_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, model_id, quantization="none", compile_model=False):
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

        # Apply FP8 quantization if requested
        if quantization == "fp8":
            if not check_torchao_available():
                raise ImportError(
                    "FP8 quantization requires torchao. "
                    "Install with: pip install torchao"
                )
            if device.type != "cuda":
                raise ValueError("FP8 quantization only supported on CUDA devices")

            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            print(f"[RMBG] Applying FP8 quantization...")
            quantize_(model, float8_dynamic_activation_float8_weight())
            print(f"[RMBG] FP8 quantization complete")

        # Optionally compile model
        if compile_model and device.type == "cuda":
            print(f"[RMBG] Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[RMBG] Model compilation complete")

        # Standard normalization for RMBG
        transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return ({
            "model": model,
            "device": device,
            "transform": transform,
            "model_id": model_id,
            "quantization": quantization,
        },)


class RMBGInference:
    """RMBG-2.0 inference node with batch support"""

    PRECISION_MODES = ["fp32", "fp16", "bf16"]

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
                "precision": (cls.PRECISION_MODES, {
                    "default": "fp16",
                    "tooltip": "fp16/bf16 = 2x faster, half VRAM. bf16 better precision."
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "process"
    CATEGORY = "Video Matting/Inference"

    def process(self, rmbg_model, images, process_size=1024, batch_size=4, precision="fp16"):
        """Process batch of images through RMBG-2.0 with batch inference

        Args:
            rmbg_model: RMBG_MODEL dict
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, 0-1 float, RGB)
            process_size: resolution for processing
            batch_size: number of frames to process at once
            precision: fp32, fp16, or bf16

        Returns:
            alpha: [B, H, W] tensor (0-1 float)
        """
        model = rmbg_model["model"]
        device = rmbg_model["device"]
        quantization = rmbg_model.get("quantization", "none")

        b, h, w, c = images.shape
        num_batches = (b + batch_size - 1) // batch_size

        # For FP8 quantized models, use fp32 input (quantization handles the rest)
        # For non-quantized models, use specified precision
        if quantization == "fp8":
            dtype = torch.float32
            use_autocast = False
            precision_label = "fp8"
        elif precision == "fp16" and device.type in ("cuda", "mps"):
            dtype = torch.float16
            use_autocast = True
            precision_label = "fp16"
        elif precision == "bf16" and device.type == "cuda":
            dtype = torch.bfloat16
            use_autocast = True
            precision_label = "bf16"
        else:
            dtype = torch.float32
            use_autocast = False
            precision_label = "fp32"

        pbar = comfy.utils.ProgressBar(num_batches)
        all_alphas = []

        # Precompute normalization mean/std on device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

        with torch.no_grad():
            # Use autocast only for fp16/bf16, not for fp8 (already quantized)
            autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.inference_mode()

            with autocast_ctx:
                for batch_idx in tqdm(range(num_batches), desc=f'RMBG({precision_label})'):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, b)
                    batch_frames = images[start_idx:end_idx]  # [batch, H, W, C]

                    # Batch preprocessing on GPU
                    # [batch, H, W, C] -> [batch, C, H, W]
                    batch_tensor = batch_frames.permute(0, 3, 1, 2).to(device=device, dtype=dtype)

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

                    # Remove channel dim and move to CPU as float32
                    batch_alphas = pred_resized.squeeze(1).float().cpu()  # [batch, H, W]
                    all_alphas.append(batch_alphas)

                    pbar.update(1)

        # Concatenate all batches
        alpha_tensor = torch.cat(all_alphas, dim=0).float()  # [B, H, W]
        return (alpha_tensor,)
