"""
Output Nodes

- ApplyAlpha: Apply alpha to images for final output
- FrameSelector: Select single frame from batch for efficient preview
- PreviewSlider: Preview sequence with slider (no re-run needed)
"""

import os
import random
import torch
import numpy as np
from PIL import Image
import folder_paths


class ApplyAlpha:
    """Apply alpha to images"""

    OUTPUT_MODES = ["rgba", "composite"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha": ("MASK",),
                "output_mode": (cls.OUTPUT_MODES, {"default": "rgba"}),
            },
            "optional": {
                "bg_color": ("STRING", {"default": "#000000"}),
                "checker_size": ("INT", {"default": 15, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "apply"
    CATEGORY = "Video Matting"

    def apply(self, images, alpha, output_mode, bg_color="#000000", checker_size=15):
        """Apply alpha to images

        Args:
            images: [B, H, W, C] tensor (ComfyUI IMAGE format, RGB)
            alpha: [B, H, W] tensor
            output_mode: "rgba" or "composite"
            bg_color: hex color for solid background
            checker_size: checkerboard square size (0=solid color)

        Returns:
            image: [B, H, W, C] tensor
            alpha: pass-through [B, H, W] tensor
        """
        b, h, w, c = images.shape

        if output_mode == "rgba":
            # Add alpha channel to images
            alpha_expanded = alpha.unsqueeze(-1)  # [B, H, W, 1]
            rgba = torch.cat([images, alpha_expanded], dim=-1)  # [B, H, W, 4]
            return (rgba, alpha)

        elif output_mode == "composite":
            # Composite over background
            if checker_size > 0:
                bg = self._create_checkerboard(h, w, checker_size)
            else:
                bg = self._parse_color(bg_color, h, w)

            bg_tensor = torch.from_numpy(bg).float() / 255.0  # [H, W, 3]
            bg_tensor = bg_tensor.unsqueeze(0).expand(b, -1, -1, -1)  # [B, H, W, 3]
            bg_tensor = bg_tensor.to(images.device)  # Match device

            alpha_expanded = alpha.unsqueeze(-1).to(images.device)  # [B, H, W, 1]
            composite = images * alpha_expanded + bg_tensor * (1 - alpha_expanded)

            return (composite, alpha)

        return (images, alpha)

    def _create_checkerboard(self, h, w, square_size):
        """Create checkerboard background"""
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, square_size):
            for x in range(0, w, square_size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    checker[y:y+square_size, x:x+square_size] = [200, 200, 200]
                else:
                    checker[y:y+square_size, x:x+square_size] = [255, 255, 255]
        return checker

    def _parse_color(self, hex_color, h, w):
        """Parse hex color to numpy array"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :] = [r, g, b]
        return bg


class FrameSelector:
    """Select single frame from batch for efficient preview

    Useful for debugging video sequences without overloading the UI.
    Only the selected frame is passed to preview nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "Frame index to select (0-based)"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "select"
    CATEGORY = "Video Matting/Debug"

    def select(self, frame_index, images=None, mask=None):
        """Select single frame from batch

        Args:
            frame_index: 0-based frame index
            images: optional [B, H, W, C] tensor
            mask: optional [B, H, W] tensor

        Returns:
            image: [1, H, W, C] single frame (or None)
            mask: [1, H, W] single frame (or None)
            info: string with frame info
        """
        out_image = None
        out_mask = None
        info_parts = []

        if images is not None:
            b = images.shape[0]
            idx = min(frame_index, b - 1)
            out_image = images[idx:idx+1]
            info_parts.append(f"Images: {b} frames, showing [{idx}]")

        if mask is not None:
            b = mask.shape[0]
            idx = min(frame_index, b - 1)
            out_mask = mask[idx:idx+1]
            info_parts.append(f"Mask: {b} frames, showing [{idx}]")

        info = " | ".join(info_parts) if info_parts else "No input"

        return (out_image, out_mask, info)


class PreviewSlider:
    """Preview image/mask sequence with slider - no re-run needed to scrub frames

    Saves all frames to temp, frontend slider controls which frame to display.
    Much more efficient than gallery mode for long sequences.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "_slider_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Video Matting/Debug"

    def preview(self, images=None, mask=None, prompt=None, extra_pnginfo=None):
        """Save all frames and return paths for slider preview

        Args:
            images: optional [B, H, W, C] tensor
            mask: optional [B, H, W] tensor

        Returns:
            UI dict with all frame paths
        """
        frames = []

        # Process images
        if images is not None:
            b, h, w, c = images.shape
            for i in range(b):
                frame = images[i].cpu().numpy()
                img = Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8))

                filename = f"img{self.prefix}_{i:05d}.png"
                filepath = os.path.join(self.output_dir, filename)
                img.save(filepath, compress_level=self.compress_level)

                frames.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                })

        # Process mask (if no images provided)
        elif mask is not None:
            b, h, w = mask.shape
            for i in range(b):
                frame = mask[i].cpu().numpy()
                mask_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(mask_uint8, mode='L').convert('RGB')

                filename = f"mask{self.prefix}_{i:05d}.png"
                filepath = os.path.join(self.output_dir, filename)
                img.save(filepath, compress_level=self.compress_level)

                frames.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                })

        # Return only frames for slider (no images to avoid default preview)
        return {"ui": {"frames": frames}}
