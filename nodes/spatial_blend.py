"""
Spatial Alpha Blend Node

Per-block spatial adaptive blending of multiple alpha sources.
Each spatial block independently solves a QP to find optimal
blend weights that minimize temporal flicker.
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import comfy.utils
import folder_paths

from .alpha_ops import _input_fingerprint


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class SpatialAlphaBlend:
    """Blend multiple alpha sources with per-block spatial adaptive weights.

    Automatically computes optimal blend weights for each spatial block
    by minimizing temporal flicker via constrained QP. Supports optical
    flow for motion-compensated comparison.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "_spblend_" + ''.join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_a": ("MASK",),
            },
            "optional": {
                "alpha_b": ("MASK",),
                "alpha_c": ("MASK",),
                "optical_flow": ("OPTICAL_FLOW",),
                "grid_size": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "temporal_sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "spatial_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "blend"
    OUTPUT_NODE = True
    CATEGORY = "Video Matting/Alpha"

    def blend(self, alpha_a, alpha_b=None, alpha_c=None, optical_flow=None,
              grid_size=8, temporal_sigma=3.0, spatial_sigma=1.0):
        sources = {"a": alpha_a}
        if alpha_b is not None:
            sources["b"] = alpha_b
        if alpha_c is not None:
            sources["c"] = alpha_c

        source_ids = sorted(sources.keys())
        num_src = len(source_ids)
        B, H, W = alpha_a.shape

        # Validate shapes
        for sid in source_ids:
            t = sources[sid]
            if t.shape != (B, H, W):
                raise ValueError(
                    f"alpha_{sid} shape {t.shape} does not match alpha_a {(B, H, W)}"
                )

        # Single source: passthrough
        if num_src == 1:
            return self._build_output(alpha_a, sources, source_ids, B, H, W, grid_size, None)

        device = _get_device()
        GH, GW = grid_size, grid_size

        # Pad H, W to be divisible by grid_size
        pad_h = (GH - H % GH) % GH
        pad_w = (GW - W % GW) % GW
        pH, pW = H + pad_h, W + pad_w
        BH, BW = pH // GH, pW // GW

        # Stack sources on CPU: (num_src, B, H, W)
        src_stack = torch.stack([sources[sid] for sid in source_ids])
        if pad_h > 0 or pad_w > 0:
            src_stack = F.pad(src_stack, (0, pad_w, 0, pad_h), mode='replicate')

        # Prepare optical flow on device
        flow_bwd_dev = None
        if optical_flow is not None:
            flow_bwd = optical_flow["bwd"]  # (N, 2, fH, fW)
            fH, fW = flow_bwd.shape[2], flow_bwd.shape[3]
            if fH != pH or fW != pW:
                # Resize and rescale flow vectors
                flow_bwd = F.interpolate(
                    flow_bwd.float(), size=(pH, pW), mode='bilinear', align_corners=False
                )
                flow_bwd[:, 0] *= pW / fW  # scale dx
                flow_bwd[:, 1] *= pH / fH  # scale dy
            elif pad_h > 0 or pad_w > 0:
                flow_bwd = F.pad(flow_bwd, (0, pad_w, 0, pad_h), mode='replicate')
            flow_bwd_dev = flow_bwd.to(device)

        # Precompute base grid for warp (on device)
        yy, xx = torch.meshgrid(
            torch.arange(pH, device=device, dtype=torch.float32),
            torch.arange(pW, device=device, dtype=torch.float32),
            indexing='ij',
        )

        # Step 1: Per-block QP — vectorized on GPU
        block_weights = np.zeros((B, GH, GW, num_src), dtype=np.float32)
        block_weights[0] = 1.0 / num_src

        pbar = comfy.utils.ProgressBar(B - 1)
        for i in range(1, B):
            # Move only current + previous frame to device
            curr = src_stack[:, i].to(device)       # (num_src, pH, pW)
            prev = src_stack[:, i - 1].to(device)   # (num_src, pH, pW)

            # Warp previous frame
            if flow_bwd_dev is not None and i - 1 < flow_bwd_dev.shape[0]:
                flow = flow_bwd_dev[i - 1]  # (2, pH, pW)
                # Build normalized grid for grid_sample: [-1, 1]
                grid_x = (xx + flow[0]) / (pW - 1) * 2 - 1
                grid_y = (yy + flow[1]) / (pH - 1) * 2 - 1
                grid = torch.stack([grid_x, grid_y], dim=-1)  # (pH, pW, 2)
                grid = grid.unsqueeze(0).expand(num_src, -1, -1, -1)  # (num_src, pH, pW, 2)
                prev_warped = F.grid_sample(
                    prev.unsqueeze(1),  # (num_src, 1, pH, pW)
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                ).squeeze(1)  # (num_src, pH, pW)
            else:
                prev_warped = prev

            # Delta: (num_src, pH, pW)
            delta = curr - prev_warped

            # Reshape to blocks: (num_src, GH, BH, GW, BW) → (num_src, GH, GW, BH*BW)
            delta_blocks = (
                delta.reshape(num_src, GH, BH, GW, BW)
                .permute(0, 1, 3, 2, 4)
                .reshape(num_src, GH, GW, BH * BW)
            )

            # M matrices for all blocks: (GH, GW, num_src, num_src)
            # Move to CPU for float64 einsum (MPS doesn't support float64)
            db_cpu = delta_blocks.cpu().double()
            M = torch.einsum('sghp,tghp->ghst', db_cpu, db_cpu)

            # Batch QP solve (on CPU, small matrices)
            w = self._batch_solve_qp(M, num_src)  # (GH, GW, num_src)
            block_weights[i] = w.cpu().numpy()

            pbar.update(1)

        # Step 2: Temporal smoothing (on CPU, small array)
        if temporal_sigma > 0 and B > 1:
            for by in range(GH):
                for bx in range(GW):
                    for s in range(num_src):
                        block_weights[:, by, bx, s] = gaussian_filter(
                            block_weights[:, by, bx, s].astype(np.float64),
                            sigma=temporal_sigma,
                            mode='nearest',
                        ).astype(np.float32)
            wsum = block_weights.sum(axis=-1, keepdims=True)
            block_weights /= np.maximum(wsum, 1e-8)

        # Step 3: Spatial smoothing (on CPU, small array)
        if spatial_sigma > 0:
            for i in range(B):
                for s in range(num_src):
                    block_weights[i, :, :, s] = gaussian_filter(
                        block_weights[i, :, :, s].astype(np.float64),
                        sigma=spatial_sigma,
                        mode='nearest',
                    ).astype(np.float32)
            wsum = block_weights.sum(axis=-1, keepdims=True)
            block_weights /= np.maximum(wsum, 1e-8)

        # Step 4: Upsample + blend (on GPU, per-frame)
        bw_tensor = torch.from_numpy(block_weights)  # (B, GH, GW, num_src)
        blended = self._blend_with_weight_maps_gpu(
            sources, source_ids, bw_tensor, B, H, W, GH, GW, device
        )

        return self._build_output(blended, sources, source_ids, B, H, W, grid_size, block_weights)

    # ── Batch QP solver (vectorized across all blocks) ───────────

    @staticmethod
    def _batch_solve_qp(M, num_src):
        """Solve min w^T M w, s.t. sum(w)=1, w>=0 for all blocks.

        M: (GH, GW, n, n) double tensor on device.
        Returns: (GH, GW, n) float32 tensor.
        """
        GH, GW = M.shape[:2]
        device = M.device

        if num_src == 1:
            return torch.ones(GH, GW, 1, dtype=torch.float32, device=device)

        if num_src == 2:
            return SpatialAlphaBlend._batch_solve_qp_2(M, device)

        if num_src == 3:
            return SpatialAlphaBlend._batch_solve_qp_3(M, device)

        # Fallback: equal weights
        return torch.full((GH, GW, num_src), 1.0 / num_src, dtype=torch.float32, device=device)

    @staticmethod
    def _batch_solve_qp_2(M, device):
        """Closed-form for 2 sources, vectorized."""
        M00, M01, M11 = M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]
        denom = M00 - 2 * M01 + M11
        wa = torch.where(
            denom.abs() < 1e-12,
            torch.tensor(0.5, dtype=torch.float64, device=device),
            (M11 - M01) / denom,
        ).clamp(0, 1).float()
        return torch.stack([wa, 1 - wa], dim=-1)

    @staticmethod
    def _batch_solve_qp_3(M, device):
        """Enumerate all 7 simplex faces for 3 sources, vectorized."""
        GH, GW = M.shape[:2]
        best_w = torch.full((GH, GW, 3), 1.0 / 3, dtype=torch.float32, device=device)
        best_cost = torch.full((GH, GW), float('inf'), dtype=torch.float64, device=device)

        # --- 1-source faces ---
        for s in range(3):
            cost = M[..., s, s]
            w = torch.zeros(GH, GW, 3, dtype=torch.float32, device=device)
            w[..., s] = 1.0
            update = cost < best_cost
            best_cost = torch.where(update, cost, best_cost)
            best_w = torch.where(update.unsqueeze(-1), w, best_w)

        # --- 2-source faces ---
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            Maa, Mab, Mbb = M[..., a, a], M[..., a, b], M[..., b, b]
            denom = Maa - 2 * Mab + Mbb
            wa = torch.where(
                denom.abs() < 1e-12,
                torch.tensor(0.5, dtype=torch.float64, device=device),
                (Mbb - Mab) / denom,
            ).clamp(0, 1)
            wb = 1 - wa
            cost = wa * wa * Maa + 2 * wa * wb * Mab + wb * wb * Mbb
            w = torch.zeros(GH, GW, 3, dtype=torch.float32, device=device)
            w[..., a] = wa.float()
            w[..., b] = wb.float()
            update = cost < best_cost
            best_cost = torch.where(update, cost, best_cost)
            best_w = torch.where(update.unsqueeze(-1), w, best_w)

        # --- 3-source face: solve M @ v = 1, normalize ---
        # Add small regularization to avoid singular matrices
        M_reg = M + torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0) * 1e-10
        ones = torch.ones(GH, GW, 3, 1, dtype=torch.float64, device=device)
        try:
            v = torch.linalg.solve(M_reg, ones).squeeze(-1)
        except RuntimeError:
            try:
                v = torch.linalg.solve(M_reg.cpu(), ones.cpu()).squeeze(-1).to(device)
            except RuntimeError:
                v = None
        if v is not None:
            v_sum = v.sum(dim=-1, keepdim=True)
            v_norm = torch.where(
                v_sum.abs() < 1e-12,
                torch.full_like(v, 1.0 / 3),
                v / v_sum,
            )
            feasible = (v_norm >= -1e-6).all(dim=-1)  # (GH, GW)
            v_norm = v_norm.clamp(min=0)
            cost = torch.einsum('ghi,ghij,ghj->gh', v_norm, M, v_norm)
            update = feasible & (cost < best_cost)
            best_cost = torch.where(update, cost, best_cost)
            best_w = torch.where(update.unsqueeze(-1), v_norm.float(), best_w)

        return best_w

    # ── GPU blend with weight maps ───────────────────────────────

    @staticmethod
    def _blend_with_weight_maps_gpu(sources, source_ids, block_weights, B, H, W, GH, GW, device):
        """Upsample block weights and blend, per-frame on GPU."""
        num_src = len(source_ids)
        blended = torch.zeros(B, H, W)

        for i in range(B):
            # Upsample weight map: (num_src, GH, GW) → (num_src, H, W)
            w_block = block_weights[i].permute(2, 0, 1).unsqueeze(0).float().to(device)
            # (1, num_src, GH, GW) → (1, num_src, H, W)
            w_up = F.interpolate(w_block, size=(H, W), mode='bilinear', align_corners=False)
            w_up = w_up.squeeze(0)  # (num_src, H, W)

            # Normalize
            w_up = w_up / w_up.sum(dim=0, keepdim=True).clamp(min=1e-8)

            # Blend
            frame = torch.zeros(H, W, device=device)
            for s, sid in enumerate(source_ids):
                frame += w_up[s] * sources[sid][i].to(device)
            blended[i] = frame.cpu()

        return blended

    # ── Output helpers ───────────────────────────────────────────

    def _build_output(self, result, sources, source_ids, B, H, W, grid_size, block_weights):
        """Build output dict with UI data."""
        fp = _input_fingerprint(**{sid: sources[sid] for sid in source_ids})

        blended_frames = self._save_frames(result, "blend")

        ui_data = {
            "blended_frames": [blended_frames],
            "total_frames": [B],
            "grid_size": [grid_size],
            "source_ids": [source_ids],
            "input_fingerprint": [fp],
        }

        if block_weights is not None:
            weight_frames = self._encode_weights(block_weights, source_ids)
            ui_data["weight_frames"] = [weight_frames]

        return {
            "ui": ui_data,
            "result": (result,),
        }

    def _save_frames(self, tensor, label):
        """Save mask frames as grayscale temp PNGs."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        B = tensor.shape[0]
        frames = []
        for i in range(B):
            arr = (tensor[i] * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            filename = f"{self.prefix}_{label}_{i:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath, compress_level=self.compress_level)
            frames.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type,
            })
        return frames

    def _encode_weights(self, block_weights, source_ids):
        """Encode block weights as RGB PNG (R=src0, G=src1, B=src2)."""
        B, GH, GW, num_src = block_weights.shape
        frames = []
        for i in range(B):
            rgb = np.zeros((GH, GW, 3), dtype=np.uint8)
            for s in range(min(num_src, 3)):
                rgb[:, :, s] = (block_weights[i, :, :, s] * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode='RGB')
            filename = f"{self.prefix}_wmap_{i:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath, compress_level=self.compress_level)
            frames.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type,
            })
        return frames
