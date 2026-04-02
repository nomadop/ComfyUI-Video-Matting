"""
Output Nodes

- ApplyAlpha: Apply alpha to images for final output
- FrameSelector: Select single frame from batch for efficient preview
- PreviewSlider: Preview sequence with slider + mask editing via built-in mask editor
- ImageSequencePackager: Pack image sequence to ZIP for download
"""

import os
import json
import random
import zipfile
import time
import torch
import numpy as np
import comfy.utils
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
    """Preview image/mask sequence with slider + interactive mask editing.

    Saves frames as RGBA temp PNGs with mask editor alpha convention
    (alpha=0 → masked/foreground, alpha=255 → not masked/background).
    Frontend provides:
    - Slider scrubbing with overlay preview (click to toggle B&W mask)
    - Edit Mask button opens ComfyUI's built-in mask editor per frame
    - Edited masks output as MASK tensor for downstream use
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "_slider_" + ''.join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edited_masks": ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("edited_mask",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Video Matting/Debug"

    def preview(self, edited_masks="{}", images=None, mask=None, prompt=None,
                extra_pnginfo=None):
        if images is None and mask is None:
            return {"ui": {"frames": []}, "result": (torch.zeros(1, 64, 64),)}

        has_images = images is not None
        has_mask = mask is not None

        if has_images:
            B, H, W, _ = images.shape
        else:
            B, H, W = mask.shape

        # Validate shapes match if both provided
        if has_images and has_mask:
            if mask.shape[0] != B:
                raise ValueError(
                    f"mask has {mask.shape[0]} frames, expected {B} (same as images)"
                )
            if mask.shape[1:] != (H, W):
                raise ValueError(
                    f"mask spatial size {mask.shape[1:]} does not match images ({H}, {W})"
                )

        # Save temp PNGs
        frames = []
        pbar = comfy.utils.ProgressBar(B)
        for i in range(B):
            if has_images and has_mask:
                # RGBA: RGB=image, A=inverted mask (mask editor convention:
                # alpha=255 means "not masked", alpha=0 means "masked")
                rgb = np.clip(images[i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                a = np.clip(mask[i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                rgba = np.dstack([rgb, 255 - a])
                img = Image.fromarray(rgba, 'RGBA')
            elif has_images:
                # RGB only (no mask yet — fully opaque = "not masked")
                rgb = np.clip(images[i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(rgb, 'RGB')
            else:
                # Mask only: checkerboard background for mask editor visibility
                a = np.clip(mask[i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                sq = 16
                yy, xx = np.mgrid[:H, :W]
                checker = np.where(((yy // sq) + (xx // sq)) % 2 == 0, 180, 220)
                bg = np.stack([checker, checker, checker], axis=-1).astype(np.uint8)
                rgba = np.dstack([bg, 255 - a])
                img = Image.fromarray(rgba, 'RGBA')

            filename = f"ps{self.prefix}_{i:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath, compress_level=self.compress_level)

            frames.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type,
            })
            pbar.update(1)

        # Build output MASK tensor
        edited = {}
        try:
            edited = json.loads(edited_masks) if edited_masks else {}
        except (json.JSONDecodeError, TypeError):
            pass

        result_mask = []
        for i in range(B):
            if str(i) in edited:
                # Load edited mask file (may be in input/clipspace or temp)
                ref = edited[str(i)]
                ref_type = ref.get("type", "input")
                base_dir = folder_paths.get_directory_by_type(ref_type)
                if base_dir is None:
                    base_dir = folder_paths.get_input_directory()
                mask_path = os.path.join(
                    base_dir,
                    ref.get("subfolder", ""),
                    ref["filename"],
                )
                try:
                    with Image.open(mask_path) as mask_img:
                        if mask_img.mode == 'RGBA':
                            alpha = np.array(mask_img.getchannel('A'))
                        elif mask_img.mode == 'L':
                            alpha = np.array(mask_img)
                        else:
                            alpha = np.array(mask_img.convert('L'))
                        # Invert: mask editor convention (0=masked) → matting (1.0=foreground)
                        alpha_t = 1.0 - torch.from_numpy(alpha.astype(np.float32) / 255.0)
                        result_mask.append(alpha_t)
                except (FileNotFoundError, OSError):
                    # Fallback if edited file missing
                    if has_mask:
                        result_mask.append(mask[i].cpu())
                    else:
                        result_mask.append(torch.ones(H, W))
            elif has_mask:
                result_mask.append(mask[i].cpu())
            else:
                result_mask.append(torch.ones(H, W))

        mask_tensor = torch.stack(result_mask, dim=0)

        return {
            "ui": {
                "frames": frames,
                "has_images": [has_images],
                "has_mask": [has_mask],
            },
            "result": (mask_tensor,),
        }


class ImageSequencePackager:
    """Pack image sequence to ZIP file for download

    Saves all frames to a ZIP file in the output directory.
    Returns download URL accessible via ComfyUI web server.
    """

    FORMAT_OPTIONS = ["png", "jpg", "webp"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "sequence"}),
                "format": (cls.FORMAT_OPTIONS, {"default": "png"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK", {"tooltip": "Pack masks as grayscale PNG (ignores format/quality, always PNG)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Quality for JPG/WebP (1-100)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("zip_path", "download_url")
    FUNCTION = "pack"
    OUTPUT_NODE = True
    CATEGORY = "Video Matting/Output"

    def pack(self, filename_prefix, format, images=None, masks=None, quality=95):
        """Pack image/mask sequence to ZIP

        Args:
            filename_prefix: prefix for ZIP filename
            format: image format (png/jpg/webp)
            images: optional [B, H, W, C] tensor (ComfyUI IMAGE format)
            masks: optional [B, H, W] tensor (packed as grayscale PNG)
            quality: quality for lossy formats

        Returns:
            zip_path: absolute path to ZIP file
            download_url: full URL for downloading via ComfyUI server
        """
        from tqdm import tqdm
        from io import BytesIO

        if images is None and masks is None:
            raise ValueError("At least one of 'images' or 'masks' must be provided")

        # Prefer masks if provided (grayscale PNG mode), otherwise use images
        use_mask = masks is not None
        if use_mask:
            b, h, w = masks.shape
        else:
            b, h, w, c = images.shape

        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{filename_prefix}_{timestamp}.zip"
        zip_path = os.path.join(self.output_dir, zip_filename)

        # Create ZIP file
        source_label = "mask frames" if use_mask else "frames"
        print(f"Packing {b} {source_label} to {zip_filename}...")

        pbar = comfy.utils.ProgressBar(b)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i in tqdm(range(b), desc="Packing"):
                buffer = BytesIO()

                if use_mask:
                    # Mask mode: always grayscale PNG
                    frame = masks[i].cpu().numpy()
                    img_array = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                    frame_filename = f"{i:05d}.png"
                    img.save(buffer, format="PNG", compress_level=6)
                else:
                    # Image mode: respect format setting
                    frame = images[i].cpu().numpy()

                    if c == 4:
                        img_array = np.clip(frame * 255, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='RGBA')
                    else:
                        img_array = np.clip(frame * 255, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='RGB')

                    frame_filename = f"{i:05d}.{format}"

                    if format == "png":
                        img.save(buffer, format="PNG", compress_level=6)
                    elif format == "jpg":
                        if c == 4:
                            img = img.convert('RGB')
                        img.save(buffer, format="JPEG", quality=quality)
                    elif format == "webp":
                        img.save(buffer, format="WEBP", quality=quality)

                # Write to ZIP
                buffer.seek(0)
                zf.writestr(frame_filename, buffer.read())
                pbar.update(1)

        # Generate download URL (relative path, frontend will resolve with current origin)
        download_url = f"/view?filename={zip_filename}&type=output"

        print(f"ZIP created: {zip_path}")
        print(f"Download URL: {download_url}")
        print(f"Click or copy the URL above to download")

        return {
            "ui": {
                "images": [{
                    "filename": zip_filename,
                    "subfolder": "",
                    "type": self.type,
                }],
                "text": [
                    f"Packed {b} {source_label} to {zip_filename}",
                    f"Download: {download_url}"
                ],
            },
            "result": (zip_path, download_url)
        }
