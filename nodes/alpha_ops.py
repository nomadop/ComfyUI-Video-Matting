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

    def combine(self, alpha_a, mode, alpha_b=None, alpha_c=None):
        """Combine alpha channels

        Args:
            alpha_a: [B, H, W] tensor (required)
            mode: combination mode
            alpha_b, alpha_c: optional additional alphas

        Returns:
            combined: [B, H, W] tensor
        """
        alphas = [alpha_a]
        if alpha_b is not None:
            alphas.append(alpha_b)
        if alpha_c is not None:
            alphas.append(alpha_c)

        if len(alphas) == 1:
            return (alpha_a,)

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
