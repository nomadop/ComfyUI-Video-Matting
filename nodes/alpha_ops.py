"""
Alpha Operation Nodes

- AlphaCombine: Combine multiple alpha channels
- TwoPassBlend: Blend two alpha sequences from 2-pass propagation
- MaskToGrayscaleImage: Convert mask to grayscale IMAGE for preview/export
- AlphaCurveBlend: Blend multiple alpha sources using Bezier curve weights over time
"""

import os
import json
import math
import random
import torch
import numpy as np
from PIL import Image
import comfy.utils
import folder_paths


def _input_fingerprint(**tensors):
    """Multi-input fingerprint: name + shape + first-frame partial hash."""
    import hashlib
    parts = []
    for name in sorted(tensors):
        t = tensors[name]
        if t is None:
            continue
        if isinstance(t, dict):  # OPTICAL_FLOW
            fwd = t.get("fwd")
            if fwd is not None:
                parts.append(f"{name}:{fwd.shape[0]}")
            continue
        b = t[0].cpu().contiguous().numpy().tobytes()[:256]
        h = hashlib.md5(b).hexdigest()[:12]
        parts.append(f"{name}:{t.shape[0]}x{t.shape[1]}x{t.shape[2]}_{h}")
    return "_".join(parts) or "empty"


class MaskToGrayscaleImage:
    """Convert MASK [B,H,W] to grayscale IMAGE [B,H,W,3]

    Useful for previewing masks, exporting as grayscale PNG sequences,
    or any workflow that needs mask data in IMAGE format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "Video Matting/Alpha"

    def convert(self, mask):
        """Convert mask to grayscale image

        Args:
            mask: [B, H, W] float32 tensor [0,1]

        Returns:
            image: [B, H, W, 3] float32 tensor (grayscale replicated to RGB)
        """
        # [B, H, W] -> [B, H, W, 1] -> [B, H, W, 3]
        gray = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        return (gray,)


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

        pbar = comfy.utils.ProgressBar(B)
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
            pbar.update(1)

        alpha_tensor = torch.from_numpy(np.stack(result, axis=0)).float()
        return (alpha_tensor,)


class AlphaCurveBlend:
    """Blend multiple alpha sources using curve weights over time.

    Supports multiple interpolation modes: catmull-rom, linear, monotone, step, gaussian.
    Features a frontend curve editor with real-time pixel-level preview.
    """

    INTERP_MODES = ["catmull-rom", "linear", "monotone", "step", "gaussian"]

    DEFAULT_CURVE_DATA = json.dumps({
        "mode": "catmull-rom",
        "curves": [{
            "id": "a",
            "anchors": [[0, 1], [1, 1]],
            "color": "#4CAF50",
        }]
    })

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "_curve_" + ''.join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_a": ("MASK",),
                "curve_data": ("STRING", {
                    "default": cls.DEFAULT_CURVE_DATA,
                    "multiline": True,
                }),
            },
            "optional": {
                "alpha_b": ("MASK",),
                "alpha_c": ("MASK",),
                "optical_flow": ("OPTICAL_FLOW", {"tooltip": "Bidirectional optical flow for motion-compensated attention waveform"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "blend"
    OUTPUT_NODE = True
    CATEGORY = "Video Matting/Alpha"

    def blend(self, alpha_a, curve_data, alpha_b=None, alpha_c=None, optical_flow=None):
        sources = {"a": alpha_a}
        if alpha_b is not None:
            sources["b"] = alpha_b
        if alpha_c is not None:
            sources["c"] = alpha_c

        # Validate all sources have matching shapes
        B, H, W = alpha_a.shape
        for src_id, tensor in sources.items():
            if src_id == "a":
                continue
            if tensor.shape[1:] != (H, W):
                raise ValueError(
                    f"alpha_{src_id} spatial size {tensor.shape[1:]} "
                    f"does not match alpha_a ({H}, {W})"
                )
            if tensor.shape[0] != B:
                raise ValueError(
                    f"alpha_{src_id} has {tensor.shape[0]} frames, "
                    f"expected {B} (same as alpha_a)"
                )

        # Detect input source change — reset stale curves to flat defaults
        fp = _input_fingerprint(a=alpha_a, b=alpha_b, c=alpha_c, flow=optical_flow)
        if fp != getattr(self, '_last_input_fp', None):
            curve_data = self.DEFAULT_CURVE_DATA
        self._last_input_fp = fp

        curves, mode = self._parse_curves(curve_data, sources)

        # Save source frames as temp PNGs for frontend preview
        source_frames = {}
        for src_id, tensor in sources.items():
            source_frames[src_id] = self._save_frames(tensor, src_id)

        # Blend
        pbar = comfy.utils.ProgressBar(B)
        result = []
        for i in range(B):
            t = i / (B - 1) if B > 1 else 0.0
            weights = {}
            weight_sum = 0.0
            for curve in curves:
                sid = curve["id"]
                if sid not in sources:
                    continue
                w = max(0.0, self._eval_anchors_at_x(curve["anchors"], t, mode))
                weights[sid] = w
                weight_sum += w

            if weight_sum == 0:
                weight_sum = 1.0

            frame = torch.zeros_like(alpha_a[0])
            for sid, w in weights.items():
                frame += sources[sid][i] * (w / weight_sum)
            result.append(frame)
            pbar.update(1)

        blended = torch.stack(result, dim=0)

        ui_data = {
            "source_frames": [source_frames],
            "total_frames": [B],
            "input_fingerprint": [fp],
        }

        # Encode optical flow to PNG temp files for frontend
        if optical_flow is not None:
            flow_frames = self._encode_flow(optical_flow)
            ui_data["optical_flow"] = [flow_frames]

        return {
            "ui": ui_data,
            "result": (blended,),
        }

    def _save_frames(self, tensor, src_id):
        """Save all frames of a MASK tensor as grayscale temp PNGs."""
        B = tensor.shape[0]
        frames = []
        for i in range(B):
            arr = (tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            filename = f"{self.prefix}_{src_id}_{i:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath, compress_level=self.compress_level)
            frames.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type,
            })
        return frames

    def _encode_flow(self, optical_flow):
        """Encode optical flow tensors to uint8 RGB PNGs for frontend.

        Flow vectors are stored as: R = dx + 128, G = dy + 128, B = 0.
        Values are clamped to [-127, 127] pixel range.

        Returns dict with fwd/bwd frame lists and metadata.
        """
        import torch.nn.functional as F

        fwd = optical_flow["fwd"]  # [N, 2, H, W]
        bwd = optical_flow["bwd"]  # [N, 2, H, W]
        flow_h = optical_flow["flow_h"]
        flow_w = optical_flow["flow_w"]
        N = fwd.shape[0]

        fwd_frames = []
        bwd_frames = []

        for i in range(N):
            for flow, direction, frame_list in [
                (fwd[i], "fwd", fwd_frames),
                (bwd[i], "bwd", bwd_frames),
            ]:
                # flow: [2, H, W] float32 in pixel units
                dx = flow[0].numpy()  # [H, W]
                dy = flow[1].numpy()  # [H, W]

                # Clamp to [-127, 127] and encode as uint8 with 128 offset
                dx_u8 = np.clip(dx + 128, 0, 255).astype(np.uint8)
                dy_u8 = np.clip(dy + 128, 0, 255).astype(np.uint8)
                zero = np.zeros_like(dx_u8)

                rgb = np.stack([dx_u8, dy_u8, zero], axis=-1)  # [H, W, 3]
                img = Image.fromarray(rgb, mode='RGB')

                filename = f"{self.prefix}_flow_{direction}_{i:05d}.png"
                filepath = os.path.join(self.output_dir, filename)
                img.save(filepath, compress_level=self.compress_level)

                frame_list.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type,
                })

        return {
            "fwd": fwd_frames,
            "bwd": bwd_frames,
            "flow_h": flow_h,
            "flow_w": flow_w,
        }

    def _parse_curves(self, curve_data, sources):
        """Parse curve_data JSON. Returns (curves_list, mode)."""
        mode = "catmull-rom"
        try:
            data = json.loads(curve_data)
            curves = data.get("curves", [])
            mode = data.get("mode", "catmull-rom")
        except (json.JSONDecodeError, TypeError):
            curves = []

        if mode not in self.INTERP_MODES:
            mode = "catmull-rom"

        # Ensure anchors exist on each curve
        for c in curves:
            if "anchors" not in c:
                c["anchors"] = [[0, 1], [1, 1]]

        # Ensure all sources have a curve
        valid_ids = {c["id"] for c in curves if c.get("id") in sources}
        for sid in sources:
            if sid not in valid_ids:
                curves.append({
                    "id": sid,
                    "anchors": [[0, 1], [1, 1]],
                    "color": "#888",
                })

        return [c for c in curves if c.get("id") in sources], mode

    # ── Interpolation dispatcher ─────────────────────────────────

    @staticmethod
    def _eval_anchors_at_x(anchors, x, mode):
        """Evaluate curve at x using the specified interpolation mode."""
        if mode == "linear":
            return AlphaCurveBlend._eval_linear(anchors, x)
        elif mode == "step":
            return AlphaCurveBlend._eval_step(anchors, x)
        elif mode == "monotone":
            return AlphaCurveBlend._eval_monotone(anchors, x)
        elif mode == "gaussian":
            return AlphaCurveBlend._eval_gaussian(anchors, x)
        else:  # catmull-rom
            return AlphaCurveBlend._eval_catmull_rom(anchors, x)

    # ── Linear ───────────────────────────────────────────────────

    @staticmethod
    def _eval_linear(anchors, x):
        n = len(anchors)
        if n == 0:
            return 1.0
        if n == 1 or x <= anchors[0][0]:
            return max(0.0, min(1.0, anchors[0][1]))
        if x >= anchors[-1][0]:
            return max(0.0, min(1.0, anchors[-1][1]))
        for i in range(n - 1):
            if x <= anchors[i + 1][0]:
                dx = anchors[i + 1][0] - anchors[i][0]
                if dx < 1e-9:
                    return max(0.0, min(1.0, anchors[i][1]))
                t = (x - anchors[i][0]) / dx
                y = anchors[i][1] + t * (anchors[i + 1][1] - anchors[i][1])
                return max(0.0, min(1.0, y))
        return max(0.0, min(1.0, anchors[-1][1]))

    # ── Step ─────────────────────────────────────────────────────

    @staticmethod
    def _eval_step(anchors, x):
        if not anchors:
            return 1.0
        for i in range(len(anchors) - 1, -1, -1):
            if x >= anchors[i][0]:
                return max(0.0, min(1.0, anchors[i][1]))
        return max(0.0, min(1.0, anchors[0][1]))

    # ── Monotone cubic (Fritsch-Carlson) ─────────────────────────

    @staticmethod
    def _eval_monotone(anchors, x):
        n = len(anchors)
        if n == 0:
            return 1.0
        if n == 1 or x <= anchors[0][0]:
            return max(0.0, min(1.0, anchors[0][1]))
        if x >= anchors[-1][0]:
            return max(0.0, min(1.0, anchors[-1][1]))

        dx = [anchors[i + 1][0] - anchors[i][0] for i in range(n - 1)]
        dy = [anchors[i + 1][1] - anchors[i][1] for i in range(n - 1)]
        delta = [dy[i] / dx[i] if dx[i] > 1e-9 else 0.0 for i in range(n - 1)]

        # Initial tangents
        m = [0.0] * n
        m[0] = delta[0]
        for i in range(1, n - 1):
            m[i] = (delta[i - 1] + delta[i]) / 2
        m[-1] = delta[-1]

        # Fritsch-Carlson modification
        for i in range(n - 1):
            if abs(delta[i]) < 1e-9:
                m[i] = 0.0
                m[i + 1] = 0.0
            else:
                alpha = m[i] / delta[i]
                beta = m[i + 1] / delta[i]
                s = alpha * alpha + beta * beta
                if s > 9:
                    tau = 3.0 / math.sqrt(s)
                    m[i] = tau * alpha * delta[i]
                    m[i + 1] = tau * beta * delta[i]

        # Find segment and evaluate cubic Hermite
        for i in range(n - 1):
            if x <= anchors[i + 1][0]:
                h = dx[i]
                if h < 1e-9:
                    return max(0.0, min(1.0, anchors[i][1]))
                t = (x - anchors[i][0]) / h
                t2, t3 = t * t, t * t * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                y = h00 * anchors[i][1] + h10 * h * m[i] + h01 * anchors[i + 1][1] + h11 * h * m[i + 1]
                return max(0.0, min(1.0, y))

        return max(0.0, min(1.0, anchors[-1][1]))

    # ── Gaussian RBF ─────────────────────────────────────────────

    @staticmethod
    def _eval_gaussian(anchors, x):
        n = len(anchors)
        if n == 0:
            return 1.0
        if n == 1:
            return max(0.0, min(1.0, anchors[0][1]))

        num, den = 0.0, 0.0
        for i in range(n):
            if i == 0:
                sigma = (anchors[1][0] - anchors[0][0]) * 0.6
            elif i == n - 1:
                sigma = (anchors[-1][0] - anchors[-2][0]) * 0.6
            else:
                sigma = min(anchors[i][0] - anchors[i - 1][0],
                            anchors[i + 1][0] - anchors[i][0]) * 0.6
            sigma = max(sigma, 0.01)
            d = x - anchors[i][0]
            g = math.exp(-(d * d) / (2 * sigma * sigma))
            num += anchors[i][1] * g
            den += g

        if den <= 0:
            return 1.0
        return max(0.0, min(1.0, num / den))

    # ── Catmull-Rom (via cubic Bezier) ───────────────────────────

    @staticmethod
    def _eval_catmull_rom(anchors, x):
        """Convert anchors to Bezier points, then evaluate."""
        points = AlphaCurveBlend._anchors_to_points(anchors)
        return AlphaCurveBlend._eval_bezier_at_x(points, x)

    @staticmethod
    def _anchors_to_points(anchors):
        """Convert anchors to cubic Bezier control points (Catmull-Rom)."""
        n = len(anchors)
        if n < 2:
            return list(anchors)
        if n == 2:
            x0, y0 = anchors[0]
            x1, y1 = anchors[1]
            return [
                [x0, y0],
                [x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3],
                [x1 - (x1 - x0) / 3, y1 - (y1 - y0) / 3],
                [x1, y1],
            ]
        points = []
        for i in range(n - 1):
            p0, p1 = anchors[i], anchors[i + 1]
            if i == 0:
                t0 = [p1[0] - p0[0], p1[1] - p0[1]]
            else:
                t0 = [(anchors[i + 1][0] - anchors[i - 1][0]) / 2,
                       (anchors[i + 1][1] - anchors[i - 1][1]) / 2]
            if i == n - 2:
                t1 = [p1[0] - p0[0], p1[1] - p0[1]]
            else:
                t1 = [(anchors[i + 2][0] - anchors[i][0]) / 2,
                       (anchors[i + 2][1] - anchors[i][1]) / 2]
            cp1 = [p0[0] + t0[0] / 3, p0[1] + t0[1] / 3]
            cp2 = [p1[0] - t1[0] / 3, p1[1] - t1[1] / 3]
            if i == 0:
                points.append([p0[0], p0[1]])
            points.extend([cp1, cp2, [p1[0], p1[1]]])
        return points

    @staticmethod
    def _cubic_bezier(p0, p1, p2, p3, t):
        u = 1 - t
        x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
        y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        return x, y

    @staticmethod
    def _eval_bezier_at_x(points, x):
        n = len(points)
        if n < 2:
            return points[0][1] if n == 1 else 1.0
        if n < 4:
            x0, y0 = points[0]
            x1, y1 = points[-1]
            if abs(x1 - x0) < 1e-9:
                return y0
            frac = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
            return max(0.0, min(1.0, y0 + (y1 - y0) * frac))

        num_segments = (n - 1) // 3
        for seg in range(num_segments):
            idx = seg * 3
            p0, p1, p2, p3 = points[idx], points[idx+1], points[idx+2], points[idx+3]
            if x < p0[0] and seg == 0:
                return max(0.0, min(1.0, p0[1]))
            if x > p3[0] and seg == num_segments - 1:
                return max(0.0, min(1.0, p3[1]))
            if p0[0] <= x <= p3[0] or seg == num_segments - 1:
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    bx, _ = AlphaCurveBlend._cubic_bezier(p0, p1, p2, p3, mid)
                    if bx < x:
                        lo = mid
                    else:
                        hi = mid
                t = (lo + hi) / 2
                _, y = AlphaCurveBlend._cubic_bezier(p0, p1, p2, p3, t)
                return max(0.0, min(1.0, y))
        return max(0.0, min(1.0, points[-1][1]))
