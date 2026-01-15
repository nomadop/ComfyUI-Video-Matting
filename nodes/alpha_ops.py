"""
Alpha Operation Nodes

- AlphaCombine: Combine multiple alpha channels
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
