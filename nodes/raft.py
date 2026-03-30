"""
RAFT Optical Flow + Alpha Deflicker Nodes

- RAFTLoader: Load TorchVision RAFT model
- AlphaDeflicker: Temporal smoothing of alpha using optical flow warping
"""

import os
import importlib.util

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import comfy.utils

# Lazy-load ProPainter flow utilities
_flow_utils = None


def _load_flow_utils():
    global _flow_utils
    if _flow_utils is not None:
        return _flow_utils
    path = os.path.expanduser('~/workspace/nomadop/ProPainter/model/modules/flow_loss_utils.py')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ProPainter flow utilities not found at {path}. "
            "Please install ProPainter or update the path."
        )
    spec = importlib.util.spec_from_file_location("flow_loss_utils", path)
    _flow_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_flow_utils)
    return _flow_utils


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class RAFTLoader:
    """Load TorchVision RAFT optical flow model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "use_fp16": ("BOOLEAN", {"default": False, "tooltip": "Use FP16 for RAFT inference (faster, slightly less accurate)"}),
            }
        }

    RETURN_TYPES = ("RAFT_MODEL",)
    RETURN_NAMES = ("raft_model",)
    FUNCTION = "load"
    CATEGORY = "Video Matting"

    def load(self, use_fp16=False):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        device = _get_device()
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        model = model.to(device).eval()

        print(f"RAFT loaded on {device}" + (" (FP16)" if use_fp16 else ""))

        return ({"model": model, "device": device, "use_fp16": use_fp16},)


class AlphaDeflicker:
    """Temporal alpha smoothing using RAFT optical flow.

    Warps neighboring frames' alpha to the current frame, computes weighted
    average, and soft-blends with current alpha. Weights are based on:
    - Forward-backward flow consistency (occlusion detection)
    - Photometric similarity (warped image vs current)
    - Temporal Gaussian decay (closer frames weighted more)
    - Motion magnitude (less smoothing in high-motion areas)
    - Edge proximity (more smoothing at alpha edges where flicker is visible)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raft_model": ("RAFT_MODEL",),
                "images": ("IMAGE",),
                "alpha": ("MASK",),
            },
            "optional": {
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Overall smoothing strength (0=off, 1=full replacement with neighbor avg)"}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 15, "step": 1,
                                        "tooltip": "Number of neighbor frames to consider in each direction"}),
                "motion_scale": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0,
                                           "tooltip": "Motion decay scale. Higher = smoother in high-motion areas"}),
                "edge_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.5,
                                         "tooltip": "Edge weighting power. Higher = only smooth edges, 0 = smooth everywhere"}),
                "fb_power": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.5,
                                       "tooltip": "FB consistency weight exponent. Higher = stricter occlusion penalty"}),
                "raft_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                                      "tooltip": "Max resolution for RAFT (longer edge). 4K images are downscaled to this before flow computation, then flow is upscaled back."}),
                "smooth_flow_kernel": (["0", "3", "5", "7"],
                                       {"default": "0",
                                        "tooltip": "Median filter kernel size for flow smoothing (0=off, 3 or 5 recommended)"}),
                "n_iterations": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1,
                                         "tooltip": "Number of smoothing iterations"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "deflicker"
    CATEGORY = "Video Matting"

    def deflicker(self, raft_model, images, alpha,
                  blend_strength=0.5, window_size=5, motion_scale=20.0,
                  edge_power=1.0, fb_power=1.0, raft_size=512,
                  smooth_flow_kernel="0", n_iterations=1):
        smooth_flow_kernel = int(smooth_flow_kernel)

        if blend_strength == 0:
            return (alpha,)

        flow_utils = _load_flow_utils()
        flow_warp = flow_utils.flow_warp
        fb_check = flow_utils.fbConsistencyCheck

        raft = raft_model["model"]
        device = raft_model["device"]
        use_fp16 = raft_model["use_fp16"]

        b = images.shape[0]
        if b < 3:
            return (alpha,)

        # Convert images: [B, 3, H, W] float32 [0, 255] for RAFT, [0, 1] for photometric
        frames = (images.permute(0, 3, 1, 2) * 255.0).to(device)
        frames_norm = frames / 255.0
        _, _, orig_h, orig_w = frames.shape

        # Downscale frames for RAFT (correlation volume is O(H*W*H*W))
        if max(orig_h, orig_w) > raft_size:
            scale = raft_size / max(orig_h, orig_w)
            new_h = int(orig_h * scale) // 8 * 8  # RAFT requires multiples of 8
            new_w = int(orig_w * scale) // 8 * 8
            raft_frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)
            flow_scale_x = orig_w / new_w
            flow_scale_y = orig_h / new_h
        else:
            raft_frames = frames
            new_h, new_w = orig_h, orig_w
            flow_scale_x = 1.0
            flow_scale_y = 1.0

        current_alpha = alpha.clone()

        pbar = comfy.utils.ProgressBar(b * n_iterations + b)

        for iteration in range(n_iterations):
            alpha_tensors = current_alpha.unsqueeze(1).float().to(device)  # [B, 1, H, W]

            fixed_alphas = self._process_frames(
                raft, flow_warp, fb_check, device, use_fp16,
                raft_frames, frames_norm, alpha_tensors,
                b, window_size, blend_strength, motion_scale,
                edge_power, fb_power, smooth_flow_kernel, pbar,
                flow_scale_x, flow_scale_y, orig_h, orig_w
            )

            current_alpha = torch.stack(fixed_alphas, dim=0)  # [B, H, W]

        return (current_alpha.cpu(),)

    def _process_frames(self, raft, flow_warp, fb_check, device, use_fp16,
                        raft_frames, frames_norm, alpha_tensors,
                        b, window_size, blend_strength, motion_scale,
                        edge_power, fb_power, smooth_flow_kernel, pbar,
                        flow_scale_x=1.0, flow_scale_y=1.0,
                        orig_h=None, orig_w=None):
        """Process all frames with sliding window flow cache.

        raft_frames: downscaled frames for RAFT computation
        frames_norm: original resolution normalized frames for photometric comparison
        flow_scale_x/y: scale factors to upscale flow from RAFT resolution to original
        """
        # Flow cache: (i, j) -> flow at ORIGINAL resolution
        flow_cache = {}
        fixed_alphas = []
        need_upscale = flow_scale_x != 1.0 or flow_scale_y != 1.0

        with torch.no_grad():
            for t in tqdm(range(b), desc="Deflicker"):
                # Determine which flows we need for this frame
                needed_flows = set()
                for offset in range(1, window_size + 1):
                    if t - offset >= 0:
                        # Chain: t-offset → t (forward flows)
                        for k in range(t - offset, t):
                            needed_flows.add((k, k + 1))  # forward
                            needed_flows.add((k + 1, k))  # backward (for FB check)
                    if t + offset < b:
                        # Chain: t+offset → t (backward flows)
                        for k in range(t, t + offset):
                            needed_flows.add((k, k + 1))
                            needed_flows.add((k + 1, k))

                # Compute missing flows (at RAFT resolution, then upscale)
                missing = [(i, j) for (i, j) in needed_flows if (i, j) not in flow_cache]
                if missing:
                    self._compute_flows_batch(
                        raft, raft_frames, device, use_fp16,
                        missing, flow_cache, smooth_flow_kernel,
                        need_upscale, flow_scale_x, flow_scale_y, orig_h, orig_w
                    )

                # Evict flows outside sliding window
                min_frame = max(0, t - window_size)
                max_frame = min(b - 1, t + window_size)
                evict_keys = [k for k in flow_cache
                              if k[0] < min_frame or k[0] > max_frame
                              or k[1] < min_frame or k[1] > max_frame]
                for k in evict_keys:
                    del flow_cache[k]

                # Gather warped references
                alpha_curr = alpha_tensors[t:t+1]  # [1, 1, H, W]
                frame_curr = frames_norm[t:t+1]    # [1, 3, H, W]
                refs = []
                weights = []

                # From previous frames
                for offset in range(1, window_size + 1):
                    if t - offset >= 0:
                        ref, weight = self._warp_reference(
                            alpha_tensors[t-offset:t-offset+1],
                            frames_norm[t-offset:t-offset+1],
                            frame_curr, flow_warp, fb_check, flow_cache,
                            t - offset, t, direction="forward",
                            offset=offset, window_size=window_size, fb_power=fb_power
                        )
                        if ref is not None:
                            refs.append(ref)
                            weights.append(weight)

                # From next frames
                for offset in range(1, window_size + 1):
                    if t + offset < b:
                        ref, weight = self._warp_reference(
                            alpha_tensors[t+offset:t+offset+1],
                            frames_norm[t+offset:t+offset+1],
                            frame_curr, flow_warp, fb_check, flow_cache,
                            t + offset, t, direction="backward",
                            offset=offset, window_size=window_size, fb_power=fb_power
                        )
                        if ref is not None:
                            refs.append(ref)
                            weights.append(weight)

                if len(refs) == 0:
                    fixed_alphas.append(alpha_curr[0, 0])
                    pbar.update(1)
                    continue

                # Weighted average of references
                refs_stack = torch.cat(refs, dim=0)      # [N, 1, H, W]
                weights_stack = torch.cat(weights, dim=0)  # [N, 1, H, W]
                weight_sum = weights_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)
                weights_norm = weights_stack / weight_sum
                ref_avg = (refs_stack * weights_norm).sum(dim=0, keepdim=True)  # [1, 1, H, W]

                # Weight confidence: continuous [0, 1]
                # Normalize weight_sum to roughly [0, 1] range
                weight_confidence = (weight_sum[0] / (window_size * 0.5)).clamp(max=1.0)

                # Motion factor (normalize flow magnitude back to RAFT resolution
                # so motion_scale is resolution-independent)
                motion_factor = self._compute_motion_factor(
                    flow_cache, t, b, motion_scale, alpha_curr,
                    flow_scale_x, flow_scale_y
                )

                # Edge weight: more smoothing at alpha edges
                if edge_power > 0:
                    edge_weight = 1.0 - (2.0 * torch.abs(alpha_curr - 0.5)) ** edge_power
                    edge_weight = edge_weight.clamp(min=0.0)
                else:
                    edge_weight = torch.ones_like(alpha_curr)

                # Soft blend
                blend = blend_strength * weight_confidence * motion_factor * edge_weight
                blend = blend.clamp(0.0, 1.0)
                alpha_out = (1.0 - blend) * alpha_curr + blend * ref_avg
                alpha_out = alpha_out.clamp(0.0, 1.0)

                fixed_alphas.append(alpha_out[0, 0])
                pbar.update(1)

        return fixed_alphas

    def _warp_reference(self, ref_alpha, ref_img, curr_img,
                        flow_warp, fb_check, flow_cache,
                        src_t, dst_t, direction, offset, window_size, fb_power):
        """Warp a reference alpha from src_t to dst_t using chained flows.

        Returns (warped_ref, weight) or (None, None) if flow is missing.
        """
        ref = ref_alpha
        img_warped = ref_img
        valid_mask = torch.ones_like(ref_alpha)

        if direction == "forward":
            # Chain: src_t → src_t+1 → ... → dst_t
            for k in range(src_t, dst_t):
                fwd_key = (k, k + 1)
                bwd_key = (k + 1, k)
                if fwd_key not in flow_cache or bwd_key not in flow_cache:
                    return None, None
                flow_fwd = flow_cache[fwd_key]
                flow_bwd = flow_cache[bwd_key]
                ref = flow_warp(ref, flow_fwd.permute(0, 2, 3, 1), padding_mode='zeros')
                img_warped = flow_warp(img_warped, flow_fwd.permute(0, 2, 3, 1), padding_mode='zeros')
                occ, _ = fb_check(flow_fwd, flow_bwd)
                valid_mask = valid_mask * (1 - occ)
        else:
            # Chain: src_t → src_t-1 → ... → dst_t
            for k in range(src_t - 1, dst_t - 1, -1):
                fwd_key = (k, k + 1)
                bwd_key = (k + 1, k)
                if fwd_key not in flow_cache or bwd_key not in flow_cache:
                    return None, None
                flow_fwd = flow_cache[fwd_key]
                flow_bwd = flow_cache[bwd_key]
                ref = flow_warp(ref, flow_bwd.permute(0, 2, 3, 1), padding_mode='zeros')
                img_warped = flow_warp(img_warped, flow_bwd.permute(0, 2, 3, 1), padding_mode='zeros')
                occ, _ = fb_check(flow_bwd, flow_fwd)
                valid_mask = valid_mask * (1 - occ)

        # Photometric similarity
        photo_diff = torch.abs(curr_img - img_warped).mean(dim=1, keepdim=True)
        photo_weight = torch.exp(-photo_diff * 10)

        # Temporal Gaussian decay
        sigma = window_size / 2.0
        temporal_weight = np.exp(-offset ** 2 / (2 * sigma ** 2))

        # Combined weight
        weight = (valid_mask ** fb_power) * photo_weight * temporal_weight

        return ref, weight

    def _compute_flows_batch(self, raft, raft_frames, device, use_fp16,
                             pairs, flow_cache, smooth_flow_kernel,
                             need_upscale=False, flow_scale_x=1.0, flow_scale_y=1.0,
                             orig_h=None, orig_w=None):
        """Compute optical flows at RAFT resolution, optionally upscale to original."""
        if not pairs:
            return

        img1_batch = torch.stack([raft_frames[i] for i, j in pairs], dim=0)
        img2_batch = torch.stack([raft_frames[j] for i, j in pairs], dim=0)

        with torch.no_grad():
            if use_fp16 and device.type in ('cuda', 'mps'):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    flow_list = raft(img1_batch, img2_batch, num_flow_updates=12)
                flow_batch = flow_list[-1].float()
            else:
                flow_list = raft(img1_batch, img2_batch, num_flow_updates=12)
                flow_batch = flow_list[-1]

        for idx, (i, j) in enumerate(pairs):
            flow = flow_batch[idx:idx+1]  # [1, 2, raft_H, raft_W]
            if smooth_flow_kernel > 0:
                flow = self._smooth_flow(flow, smooth_flow_kernel)
            if need_upscale:
                # Upscale flow to original resolution and scale displacement values
                flow = F.interpolate(flow, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                flow[:, 0] *= flow_scale_x  # horizontal displacement
                flow[:, 1] *= flow_scale_y  # vertical displacement
            flow_cache[(i, j)] = flow

    def _compute_motion_factor(self, flow_cache, t, b, motion_scale, alpha_like,
                               flow_scale_x=1.0, flow_scale_y=1.0):
        """Compute motion-adaptive factor: high motion → less smoothing.

        Flow magnitudes are normalized back to RAFT resolution so that
        motion_scale is resolution-independent (calibrated for ~512px).
        """
        flow_mags = []
        if t > 0 and (t - 1, t) in flow_cache:
            flow = flow_cache[(t - 1, t)]
            # Normalize flow components back to RAFT resolution
            mag = torch.sqrt((flow[:, 0:1] / flow_scale_x) ** 2 +
                             (flow[:, 1:2] / flow_scale_y) ** 2)
            flow_mags.append(mag)
        if t < b - 1 and (t, t + 1) in flow_cache:
            flow = flow_cache[(t, t + 1)]
            mag = torch.sqrt((flow[:, 0:1] / flow_scale_x) ** 2 +
                             (flow[:, 1:2] / flow_scale_y) ** 2)
            flow_mags.append(mag)

        if flow_mags:
            avg_mag = torch.stack(flow_mags).mean(dim=0)
            return torch.exp(-avg_mag / motion_scale)
        return torch.ones_like(alpha_like)

    @staticmethod
    def _smooth_flow(flow, kernel_size):
        """Smooth optical flow using median filter."""
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        flow_u = cv2.medianBlur(flow_np[..., 0].astype(np.float32), kernel_size)
        flow_v = cv2.medianBlur(flow_np[..., 1].astype(np.float32), kernel_size)
        flow_smooth = np.stack([flow_u, flow_v], axis=-1)
        return torch.from_numpy(flow_smooth).permute(2, 0, 1).unsqueeze(0).to(flow.device)
