"""
Alpha Operation Nodes

- AlphaCombine: Combine multiple alpha channels
- TwoPassBlend: Blend two alpha sequences from 2-pass propagation
"""

import torch
import numpy as np


class AlphaCombine:
    """Combine multiple alpha channels"""

    COMBINE_MODES = ["avg", "max", "min", "multiply"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_a": ("MASK",),
                "mode": (cls.COMBINE_MODES, {"default": "avg"}),
                "enable_a": ("BOOLEAN", {"default": True}),
                "enable_b": ("BOOLEAN", {"default": True}),
                "enable_c": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha_b": ("MASK",),
                "alpha_c": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "combine"
    CATEGORY = "Video Matting"

    def combine(self, alpha_a, mode, enable_a=True, enable_b=True, enable_c=True, alpha_b=None, alpha_c=None):
        """Combine alpha channels

        Args:
            alpha_a: [B, H, W] tensor (required)
            mode: combination mode
            enable_a, enable_b, enable_c: toggle switches for each channel
            alpha_b, alpha_c: optional additional alphas

        Returns:
            combined: [B, H, W] tensor
        """
        alphas = []
        if enable_a:
            alphas.append(alpha_a)
        if alpha_b is not None and enable_b:
            alphas.append(alpha_b)
        if alpha_c is not None and enable_c:
            alphas.append(alpha_c)

        # If no channels enabled, return zeros
        if len(alphas) == 0:
            return (torch.zeros_like(alpha_a),)

        if len(alphas) == 1:
            return (alphas[0],)

        # Stack alphas [N, B, H, W]
        stacked = torch.stack(alphas, dim=0)

        if mode == "avg":
            result = torch.mean(stacked, dim=0)
        elif mode == "max":
            result = torch.max(stacked, dim=0)[0]
        elif mode == "min":
            result = torch.min(stacked, dim=0)[0]
        elif mode == "multiply":
            result = torch.prod(stacked, dim=0)
        else:
            result = torch.mean(stacked, dim=0)

        return (result,)


class TwoPassBlend:
    """Blend two alpha sequences from 2-pass propagation"""

    BLEND_MODES = ["distance_blend", "avg", "max", "min", "multiply", "bwd_dominant"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_fwd": ("MASK",),
                "alpha_bwd": ("MASK",),
                "blend_mode": (cls.BLEND_MODES, {"default": "distance_blend"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "blend"
    CATEGORY = "Video Matting"

    def blend(self, alpha_fwd, alpha_bwd, blend_mode):
        """Blend two aligned alpha sequences

        Args:
            alpha_fwd: [B, H, W] tensor (frame 0→B-1)
            alpha_bwd: [B, H, W] tensor (frame 0→B-1, aligned)
            blend_mode: blend mode string

        Returns:
            alpha: [B, H, W] blended tensor
        """
        alphas_fwd = alpha_fwd.cpu().numpy()
        alphas_bwd = alpha_bwd.cpu().numpy()
        B = alphas_fwd.shape[0]

        result = []
        for i in range(B):
            a_fwd = alphas_fwd[i]
            a_bwd = alphas_bwd[i]

            if blend_mode == "distance_blend":
                w = i / (B - 1) if B > 1 else 0.5
                blended = w * a_fwd + (1 - w) * a_bwd
            elif blend_mode == "avg":
                blended = (a_fwd + a_bwd) / 2
            elif blend_mode == "max":
                blended = np.maximum(a_fwd, a_bwd)
            elif blend_mode == "min":
                blended = np.minimum(a_fwd, a_bwd)
            elif blend_mode == "multiply":
                blended = a_fwd * a_bwd
            elif blend_mode == "bwd_dominant":
                blended = 0.3 * a_fwd + 0.7 * a_bwd
            else:
                blended = (a_fwd + a_bwd) / 2

            result.append(blended)

        alpha_tensor = torch.from_numpy(np.stack(result, axis=0)).float()
        return (alpha_tensor,)
