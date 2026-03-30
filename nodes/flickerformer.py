"""
Flickerformer Deflicker Nodes

- FlickerformerLoader: Load Flickerformer model
- FlickerformerDeflicker: Temporal deflickering of alpha/mask sequences
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import comfy.utils
import folder_paths


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class FlickerformerLoader:
    """Load Flickerformer model for video deflickering"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "use_fp16": ("BOOLEAN", {"default": False,
                                         "tooltip": "Use FP16 for inference (faster on CUDA, may not help on MPS)"}),
            }
        }

    RETURN_TYPES = ("FLICKERFORMER_MODEL",)
    RETURN_NAMES = ("flickerformer_model",)
    FUNCTION = "load"
    CATEGORY = "Video Matting"

    def load(self, use_fp16=False):
        from ..models.flickerformer_arch import Flickerformer

        device = _get_device()

        # Look for weights in ComfyUI models directory
        weights_path = os.path.join(folder_paths.models_dir, "flickerformer", "Flickerformer.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Flickerformer weights not found at {weights_path}. "
                "Please download Flickerformer.pth and place it in models/flickerformer/"
            )

        model = Flickerformer()

        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        # strict=False: pytorch_wavelets' fixed Haar filter buffers (h0_col etc.)
        # are registered in the checkpoint but rebuilt by DWTForward/DWTInverse
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        print(f"Flickerformer loaded on {device}" + (" (FP16)" if use_fp16 else ""))
        return ({"model": model, "device": device, "use_fp16": use_fp16},)


class FlickerformerDeflicker:
    """Temporal deflickering using Flickerformer.

    Uses 3-frame sliding window: for each frame, feeds [prev, current, next]
    through Flickerformer which outputs a residual correction for the center frame.

    Alpha adaptation: replicates single-channel alpha to 3 channels for input,
    averages output channels back to single channel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flickerformer_model": ("FLICKERFORMER_MODEL",),
                "alpha": ("MASK",),
            },
            "optional": {
                "process_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 128,
                                         "tooltip": "Processing resolution (model default: 512). Input is resized to this, residual is upscaled back."}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Blend between original (0) and deflickered (1) alpha"}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1,
                                   "tooltip": "Number of deflicker passes. Each pass expands effective temporal window by ±1 frame (e.g. 2 passes = ±2 frames)"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "deflicker"
    CATEGORY = "Video Matting"

    def deflicker(self, flickerformer_model, alpha, process_size=512, blend_strength=1.0, passes=1):
        if blend_strength == 0:
            return (alpha,)

        model = flickerformer_model["model"]
        device = flickerformer_model["device"]
        use_fp16 = flickerformer_model["use_fp16"]

        b = alpha.shape[0]
        if b < 2:
            return (alpha,)

        orig_h, orig_w = alpha.shape[1], alpha.shape[2]
        proc_h, proc_w = self._compute_proc_size(orig_h, orig_w, process_size)

        current_alpha = alpha
        pbar = comfy.utils.ProgressBar(b * passes)

        with torch.no_grad():
            for p in range(passes):
                output_alphas = []
                desc = f"Flickerformer pass {p+1}/{passes}" if passes > 1 else "Flickerformer"

                for i in tqdm(range(b), desc=desc):
                    idx_prev = max(0, i - 1)
                    idx_next = min(b - 1, i + 1)

                    frame_prev = current_alpha[idx_prev].unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(device)
                    frame_curr = current_alpha[i].unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(device)
                    frame_next = current_alpha[idx_next].unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(device)

                    frame_prev_s = F.interpolate(frame_prev, size=(proc_h, proc_w), mode='bilinear', align_corners=False)
                    frame_curr_s = F.interpolate(frame_curr, size=(proc_h, proc_w), mode='bilinear', align_corners=False)
                    frame_next_s = F.interpolate(frame_next, size=(proc_h, proc_w), mode='bilinear', align_corners=False)

                    inp = torch.cat([frame_prev_s, frame_curr_s, frame_next_s], dim=1)

                    if use_fp16 and device.type in ('cuda', 'mps'):
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            out = model(inp)
                        out = out.float()
                    else:
                        out = model(inp)

                    out_alpha = out.mean(dim=1, keepdim=True)

                    if proc_h != orig_h or proc_w != orig_w:
                        residual = out_alpha - frame_curr_s.mean(dim=1, keepdim=True)
                        residual_up = F.interpolate(residual, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                        result = current_alpha[i].unsqueeze(0).unsqueeze(0).to(device) + residual_up
                    else:
                        result = out_alpha

                    result = result.clamp(0.0, 1.0).squeeze(0).squeeze(0)
                    output_alphas.append(result.cpu())
                    pbar.update(1)

                current_alpha = torch.stack(output_alphas, dim=0)

        # Blend with original
        if blend_strength < 1.0:
            current_alpha = (1 - blend_strength) * alpha + blend_strength * current_alpha

        return (current_alpha,)

    @staticmethod
    def _compute_proc_size(orig_h, orig_w, max_size):
        """Compute processing size as multiple of 64, fitting within max_size.

        Must be multiple of 64 because: 2 downsample levels (/4) + DWT (/2) + window_size 8
        → spatial dims at deepest attention layer = proc_size / 64.
        """
        scale = max_size / max(orig_h, orig_w) if max(orig_h, orig_w) > max_size else 1.0
        proc_h = int(orig_h * scale) // 64 * 64
        proc_w = int(orig_w * scale) // 64 * 64
        proc_h = max(proc_h, 64)
        proc_w = max(proc_w, 64)
        return proc_h, proc_w
