"""
Cutie Process Node

Video object segmentation with optional alpha correction and ViTMatte refinement.
Handles stateful processing internally within the node.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import functional as TF
from tqdm import tqdm
import comfy.utils


class CutieProcess:
    """Cutie video propagation with optional ViTMatte refinement"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cutie_model": ("CUTIE_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "init_mask": ("MASK",),
                "keyframe_masks": ("MASK",),
                "correction_alphas": ("MASK",),
                "vitmatte_model": ("VITMATTE_MODEL",),
                "keyframe_indices": ("STRING", {"default": "0", "tooltip": "Comma-separated keyframe indices, e.g. '0,50,100'"}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Threshold for binary mask conversion"}),
                "fg_thresh": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.05, "tooltip": "Foreground confidence threshold"}),
                "bg_thresh": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Background confidence threshold"}),
                "conflict_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Conflict detection threshold (soft region lower bound)"}),
                "erode_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2, "tooltip": "Erosion kernel size for FG core"}),
                "erode_iter": ("INT", {"default": 1, "min": 0, "max": 5, "tooltip": "Erosion iterations"}),
                "dilate_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2, "tooltip": "Dilation kernel size for unknown band"}),
                "dilate_iter": ("INT", {"default": 2, "min": 0, "max": 5, "tooltip": "Dilation iterations"}),
                "vitmatte_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "Max size for ViTMatte (0=original)"}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask", "alpha", "trimap")
    FUNCTION = "process"
    CATEGORY = "Video Matting"

    def process(self, cutie_model, images,
                init_mask=None, keyframe_masks=None,
                correction_alphas=None, vitmatte_model=None,
                keyframe_indices="0", mask_threshold=0.5,
                fg_thresh=0.95, bg_thresh=0.05, conflict_thresh=0.1,
                erode_size=5, erode_iter=1, dilate_size=5, dilate_iter=2,
                vitmatte_size=0):
        """
        Process video frames through Cutie with optional refinement.

        Args:
            cutie_model: CUTIE_MODEL dict
            images: [B, H, W, C] tensor (ComfyUI IMAGE format)
            init_mask: [H, W] or [1, H, W] initial mask for frame 0
            keyframe_masks: [K, H, W] masks for keyframes
            correction_alphas: [B, H, W] pre-computed alphas for trimap generation
            vitmatte_model: VITMATTE_MODEL for per-frame refinement
            keyframe_indices: comma-separated frame indices "0,50,100"

        Returns:
            mask: [B, H, W] propagated masks
            alpha: [B, H, W] refined alphas (same as mask if no vitmatte)
            trimap: [B, H, W] generated trimaps (for debugging)
        """
        processor = cutie_model["processor"]
        device = cutie_model["device"]

        # Parse keyframe indices
        kf_indices = [int(x.strip()) for x in keyframe_indices.split(',') if x.strip()]

        # Convert images to numpy
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        b, h, w, c = images_np.shape

        # Prepare masks
        if init_mask is not None:
            init_mask_np = init_mask.cpu().numpy()
            if init_mask_np.ndim == 3:
                init_mask_np = init_mask_np[0]

        if keyframe_masks is not None:
            kf_masks_np = keyframe_masks.cpu().numpy()
        else:
            kf_masks_np = None

        if correction_alphas is not None:
            correction_np = correction_alphas.cpu().numpy()
        else:
            correction_np = None

        output_masks = []
        output_alphas = []
        output_trimaps = []

        pbar = comfy.utils.ProgressBar(b)

        with torch.no_grad():
            for frame_idx in tqdm(range(b), desc='Cutie'):
                frame_rgb = images_np[frame_idx]
                image_tensor = to_tensor(frame_rgb).to(device).float()

                # Check if this is a keyframe
                if frame_idx in kf_indices:
                    # Get mask for this keyframe
                    if frame_idx == 0 and init_mask is not None:
                        kf_mask = init_mask_np
                    elif kf_masks_np is not None:
                        kf_idx = kf_indices.index(frame_idx)
                        if kf_idx < len(kf_masks_np):
                            kf_mask = kf_masks_np[kf_idx]
                        else:
                            kf_mask = None
                    else:
                        kf_mask = None

                    if kf_mask is not None:
                        binary_mask = (kf_mask > mask_threshold).astype(np.uint8)
                        mask_tensor = torch.from_numpy(binary_mask).to(device)
                        _ = processor.step(image_tensor, mask_tensor, objects=[1])
                        cutie_mask = binary_mask.astype(np.float32)
                    else:
                        # No mask for this keyframe, just propagate
                        output_prob = processor.step(image_tensor)
                        propagated = processor.output_prob_to_mask(output_prob)
                        cutie_mask = propagated.cpu().numpy().astype(np.float32)
                else:
                    # Propagate from previous frame
                    output_prob = processor.step(image_tensor)
                    propagated = processor.output_prob_to_mask(output_prob)
                    cutie_mask = propagated.cpu().numpy().astype(np.float32)

                output_masks.append(cutie_mask)

                # Generate trimap (always, for debugging output)
                if correction_np is not None:
                    trimap = self._generate_hybrid_trimap(
                        cutie_mask, correction_np[frame_idx],
                        erode_size, erode_iter, dilate_size, dilate_iter,
                        fg_thresh, conflict_thresh
                    )
                else:
                    trimap = self._generate_trimap_from_mask(
                        cutie_mask, erode_size, erode_iter, dilate_size, dilate_iter
                    )
                output_trimaps.append(trimap)

                # Optionally refine with ViTMatte
                if vitmatte_model is not None:
                    # ViTMatte refinement
                    refined_alpha = self._run_vitmatte(
                        vitmatte_model, frame_rgb, trimap, vitmatte_size
                    )
                    output_alphas.append(refined_alpha)

                    # Update Cutie memory with refined result
                    refined_mask = (refined_alpha > mask_threshold).astype(np.uint8)
                    mask_tensor = torch.from_numpy(refined_mask).to(device)
                    _ = processor.step(image_tensor, mask_tensor, objects=[1])
                else:
                    output_alphas.append(cutie_mask)

                pbar.update(1)

        # Convert to tensors
        mask_tensor = torch.from_numpy(np.stack(output_masks, axis=0)).float()
        alpha_tensor = torch.from_numpy(np.stack(output_alphas, axis=0)).float()
        trimap_tensor = torch.from_numpy(np.stack(output_trimaps, axis=0)).float()

        return (mask_tensor, alpha_tensor, trimap_tensor)

    def _generate_trimap_from_mask(self, mask, erode_size, erode_iter, dilate_size, dilate_iter):
        """Generate trimap from binary mask using morphology"""
        binary = (mask > 0.5).astype(np.uint8)
        h, w = binary.shape

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        fg_core = cv2.erode(binary, erode_kernel, iterations=erode_iter)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        fg_outer = cv2.dilate(binary, dilate_kernel, iterations=dilate_iter)

        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[fg_outer == 1] = 128
        trimap[fg_core == 1] = 255

        return trimap

    def _generate_hybrid_trimap(self, cutie_mask, correction_alpha,
                                 erode_size, erode_iter, dilate_size, dilate_iter,
                                 fg_thresh, conflict_thresh):
        """Generate trimap from Cutie mask + correction alpha"""
        cutie_binary = (cutie_mask > 0.5).astype(np.uint8)
        h, w = cutie_binary.shape

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

        # Cutie: erode for FG core, dilate for outer
        cutie_fg_core = cv2.erode(cutie_binary, erode_kernel, iterations=erode_iter)
        cutie_fg_outer = cv2.dilate(cutie_binary, dilate_kernel, iterations=dilate_iter)

        # Correction alpha: threshold and erode
        corr_fg_binary = (correction_alpha > fg_thresh).astype(np.uint8)
        corr_fg_core = cv2.erode(corr_fg_binary, erode_kernel, iterations=erode_iter)
        corr_soft = (correction_alpha > conflict_thresh).astype(np.uint8)
        corr_bg = (correction_alpha < conflict_thresh).astype(np.uint8)

        soft_dilated = cv2.dilate(corr_soft, dilate_kernel, iterations=dilate_iter)

        # Conflict: correction says BG but Cutie says FG
        conflict = ((corr_bg == 1) & (cutie_fg_core == 1)).astype(np.uint8)

        # Union FG
        fg_union_raw = ((cutie_fg_core == 1) | (corr_fg_core == 1)).astype(np.uint8)
        fg_union = ((fg_union_raw == 1) & (conflict == 0)).astype(np.uint8)

        # Union outer
        fg_outer_union = ((cutie_fg_outer == 1) | (soft_dilated == 1) | (conflict == 1)).astype(np.uint8)

        # Build trimap
        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[fg_outer_union == 1] = 128
        trimap[fg_union == 1] = 255

        return trimap

    def _run_vitmatte(self, vitmatte_model, image_rgb, trimap, max_size):
        """Run ViTMatte on single frame"""
        model = vitmatte_model["model"]
        device = vitmatte_model["device"]
        use_fp16 = vitmatte_model["use_fp16"]

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

        with torch.no_grad():
            if use_fp16 and device.type in ('cuda', 'mps'):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model({'image': image_tensor, 'trimap': trimap_tensor})
            else:
                output = model({'image': image_tensor, 'trimap': trimap_tensor})

        alpha = output['phas'].flatten(0, 2).float().cpu().numpy()

        if max_size > 0 and max(orig_h, orig_w) > max_size:
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return alpha
