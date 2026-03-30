"""
Cutie Process Node

Video object segmentation with optional alpha correction, ViTMatte refinement,
and integrated two-pass blending.
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
    """Cutie video propagation with optional ViTMatte refinement and integrated blending"""

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
                "enable_2pass": ("BOOLEAN", {"default": False, "tooltip": "Enable 2-pass bidirectional propagation"}),
                "blend_mode": (["none", "distance_blend", "avg", "max", "min", "multiply", "bwd_dominant"],
                               {"default": "distance_blend",
                                "tooltip": "Blend mode for 2-pass alpha fusion (replaces TwoPassBlend node). 'none' returns unblended forward alpha."}),
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
                vitmatte_size=0, enable_2pass=False, blend_mode="distance_blend"):
        processor = cutie_model["processor"]
        device = cutie_model["device"]
        b = images.shape[0]

        kf_indices = [int(x.strip()) for x in keyframe_indices.split(',') if x.strip()]

        # Prepare masks (single-channel, much smaller than images)
        init_mask_np = None
        if init_mask is not None:
            init_mask_np = init_mask.cpu().numpy()
            if init_mask_np.ndim == 3:
                init_mask_np = init_mask_np[0]

        kf_masks_np = keyframe_masks.cpu().numpy() if keyframe_masks is not None else None
        correction_np = correction_alphas.cpu().numpy() if correction_alphas is not None else None

        trimap_params = {
            "erode_size": erode_size,
            "erode_iter": erode_iter,
            "dilate_size": dilate_size,
            "dilate_iter": dilate_iter,
            "fg_thresh": fg_thresh,
            "conflict_thresh": conflict_thresh,
        }

        if not enable_2pass:
            masks, alphas, trimaps = self._run_pass(
                processor, device, images,
                init_mask=init_mask_np,
                kf_indices=kf_indices,
                kf_masks_np=kf_masks_np,
                correction_np=correction_np,
                vitmatte_model=vitmatte_model,
                mask_threshold=mask_threshold,
                trimap_params=trimap_params,
                vitmatte_size=vitmatte_size,
                pass_name="Cutie"
            )

            mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).float()
            alpha_tensor = torch.from_numpy(np.stack(alphas, axis=0)).float()
            trimap_tensor = torch.from_numpy(np.stack(trimaps, axis=0)).float()
            return (mask_tensor, alpha_tensor, trimap_tensor)

        # === 2-Pass Mode ===

        # Forward pass (0 → B-1)
        masks_fwd, alphas_fwd, trimaps_fwd = self._run_pass(
            processor, device, images,
            init_mask=init_mask_np,
            kf_indices=kf_indices,
            kf_masks_np=kf_masks_np,
            correction_np=correction_np,
            vitmatte_model=vitmatte_model,
            mask_threshold=mask_threshold,
            trimap_params=trimap_params,
            vitmatte_size=vitmatte_size,
            pass_name="Cutie[fwd]"
        )

        pass2_init_mask = masks_fwd[-1]
        processor.clear_memory()

        # Backward pass (B-1 → 0) — reverse iteration, no array copy
        _, alphas_bwd, _ = self._run_pass(
            processor, device, images,
            reverse=True,
            alpha_only=True,
            init_mask=pass2_init_mask,
            kf_indices=[b - 1],
            kf_masks_np=None,
            correction_np=correction_np,
            vitmatte_model=vitmatte_model,
            mask_threshold=mask_threshold,
            trimap_params=trimap_params,
            vitmatte_size=vitmatte_size,
            pass_name="Cutie[bwd]"
        )
        # alphas_bwd is in reverse processing order: [orig B-1, ..., orig 0]

        if blend_mode != "none":
            alphas_blended = self._blend_passes(alphas_fwd, alphas_bwd, blend_mode)
            del alphas_fwd, alphas_bwd
            alpha_tensor = torch.from_numpy(np.stack(alphas_blended, axis=0)).float()
        else:
            del alphas_bwd
            alpha_tensor = torch.from_numpy(np.stack(alphas_fwd, axis=0)).float()

        mask_tensor = torch.from_numpy(np.stack(masks_fwd, axis=0)).float()
        trimap_tensor = torch.from_numpy(np.stack(trimaps_fwd, axis=0)).float()
        return (mask_tensor, alpha_tensor, trimap_tensor)

    def _run_pass(self, processor, device, images,
                  init_mask, kf_indices, kf_masks_np, correction_np,
                  vitmatte_model, mask_threshold, trimap_params, vitmatte_size,
                  reverse=False, alpha_only=False, pass_name="Cutie"):
        """Run a single propagation pass with per-frame image conversion.

        Args:
            images: [B, H, W, C] torch tensor (float32, NOT pre-converted numpy)
            reverse: iterate frames B-1 → 0 instead of 0 → B-1
            alpha_only: skip collecting masks/trimaps (for backward pass blending)
        """
        b = images.shape[0]
        output_masks = []
        output_alphas = []
        output_trimaps = []

        pbar = comfy.utils.ProgressBar(b)

        with torch.no_grad():
            for step_idx in tqdm(range(b), desc=pass_name):
                frame_idx = (b - 1 - step_idx) if reverse else step_idx

                # Per-frame conversion: avoids allocating full [B,H,W,C] uint8 array
                frame_rgb = (images[frame_idx].cpu().numpy() * 255).astype(np.uint8)
                image_tensor = to_tensor(frame_rgb).to(device).float()

                # Keyframe handling
                if frame_idx in kf_indices:
                    if step_idx == 0 and init_mask is not None:
                        kf_mask = init_mask
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
                        output_prob = processor.step(image_tensor)
                        propagated = processor.output_prob_to_mask(output_prob)
                        cutie_mask = propagated.cpu().numpy().astype(np.float32)
                else:
                    output_prob = processor.step(image_tensor)
                    propagated = processor.output_prob_to_mask(output_prob)
                    cutie_mask = propagated.cpu().numpy().astype(np.float32)

                if not alpha_only:
                    output_masks.append(cutie_mask)

                # Generate trimap (needed for ViTMatte even in alpha_only mode)
                trimap = None
                if correction_np is not None:
                    trimap = self._generate_hybrid_trimap(
                        cutie_mask, correction_np[frame_idx],
                        trimap_params["erode_size"], trimap_params["erode_iter"],
                        trimap_params["dilate_size"], trimap_params["dilate_iter"],
                        trimap_params["fg_thresh"], trimap_params["conflict_thresh"]
                    )
                elif vitmatte_model is not None or not alpha_only:
                    trimap = self._generate_trimap_from_mask(
                        cutie_mask,
                        trimap_params["erode_size"], trimap_params["erode_iter"],
                        trimap_params["dilate_size"], trimap_params["dilate_iter"]
                    )

                if not alpha_only and trimap is not None:
                    output_trimaps.append(trimap)

                # Optionally refine with ViTMatte
                if vitmatte_model is not None and trimap is not None:
                    refined_alpha = self._run_vitmatte(
                        vitmatte_model, frame_rgb, trimap, vitmatte_size
                    )
                    output_alphas.append(refined_alpha)

                    refined_mask = (refined_alpha > mask_threshold).astype(np.uint8)
                    mask_tensor = torch.from_numpy(refined_mask).to(device)
                    _ = processor.step(image_tensor, mask_tensor, objects=[1])
                else:
                    output_alphas.append(cutie_mask)

                pbar.update(1)

        return output_masks, output_alphas, output_trimaps

    def _blend_passes(self, alphas_fwd, alphas_bwd, blend_mode):
        """Blend forward and backward alpha lists.

        alphas_fwd: list in forward order [0→B-1]
        alphas_bwd: list in reverse processing order [B-1→0]
        """
        B = len(alphas_fwd)
        result = []
        for i in range(B):
            a_fwd = alphas_fwd[i]
            a_bwd = alphas_bwd[B - 1 - i]

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
        return result

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
