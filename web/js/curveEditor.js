import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── Interpolation modes ─────────────────────────────────────────────────────

const INTERP_MODES = ["catmull-rom", "linear", "monotone", "step", "gaussian"];

// ── Bezier math (used by catmull-rom mode) ──────────────────────────────────

function cubicBezier(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return [
        u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
        u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1],
    ];
}

function evalBezierAtX(points, x) {
    const n = points.length;
    if (n < 2) return n === 1 ? Math.max(0, Math.min(1, points[0][1])) : 1.0;
    if (n < 4) {
        const [x0, y0] = points[0];
        const [x1, y1] = points[n - 1];
        if (Math.abs(x1 - x0) < 1e-9) return y0;
        const f = Math.max(0, Math.min(1, (x - x0) / (x1 - x0)));
        return Math.max(0, Math.min(1, y0 + (y1 - y0) * f));
    }
    const numSeg = Math.floor((n - 1) / 3);
    for (let seg = 0; seg < numSeg; seg++) {
        const idx = seg * 3;
        const p0 = points[idx], p1 = points[idx+1], p2 = points[idx+2], p3 = points[idx+3];
        if (x < p0[0] && seg === 0) return Math.max(0, Math.min(1, p0[1]));
        if (x > p3[0] && seg === numSeg - 1) return Math.max(0, Math.min(1, p3[1]));
        if (p0[0] <= x && x <= p3[0] || seg === numSeg - 1) {
            let lo = 0, hi = 1;
            for (let i = 0; i < 20; i++) {
                const mid = (lo + hi) / 2;
                const bx = cubicBezier(p0, p1, p2, p3, mid)[0];
                if (bx < x) lo = mid; else hi = mid;
            }
            const y = cubicBezier(p0, p1, p2, p3, (lo + hi) / 2)[1];
            return Math.max(0, Math.min(1, y));
        }
    }
    return Math.max(0, Math.min(1, points[n-1][1]));
}

function anchorsToPoints(anchors) {
    const n = anchors.length;
    if (n < 2) return anchors.slice();
    if (n === 2) {
        const [x0, y0] = anchors[0];
        const [x1, y1] = anchors[1];
        return [
            [x0, y0],
            [x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3],
            [x1 - (x1 - x0) / 3, y1 - (y1 - y0) / 3],
            [x1, y1],
        ];
    }
    const points = [];
    for (let i = 0; i < n - 1; i++) {
        const p0 = anchors[i];
        const p1 = anchors[i + 1];
        let t0x, t0y;
        if (i === 0) { t0x = p1[0] - p0[0]; t0y = p1[1] - p0[1]; }
        else { t0x = (anchors[i+1][0] - anchors[i-1][0]) / 2; t0y = (anchors[i+1][1] - anchors[i-1][1]) / 2; }
        let t1x, t1y;
        if (i === n - 2) { t1x = p1[0] - p0[0]; t1y = p1[1] - p0[1]; }
        else { t1x = (anchors[i+2][0] - anchors[i][0]) / 2; t1y = (anchors[i+2][1] - anchors[i][1]) / 2; }
        const cp1 = [p0[0] + t0x / 3, p0[1] + t0y / 3];
        const cp2 = [p1[0] - t1x / 3, p1[1] - t1y / 3];
        if (i === 0) points.push([p0[0], p0[1]]);
        points.push(cp1, cp2, [p1[0], p1[1]]);
    }
    return points;
}

// ── Interpolation functions ─────────────────────────────────────────────────

function evalLinear(anchors, x) {
    const n = anchors.length;
    if (n === 0) return 1.0;
    if (n === 1 || x <= anchors[0][0]) return Math.max(0, Math.min(1, anchors[0][1]));
    if (x >= anchors[n-1][0]) return Math.max(0, Math.min(1, anchors[n-1][1]));
    for (let i = 0; i < n - 1; i++) {
        if (x <= anchors[i+1][0]) {
            const dx = anchors[i+1][0] - anchors[i][0];
            if (dx < 1e-9) return Math.max(0, Math.min(1, anchors[i][1]));
            const t = (x - anchors[i][0]) / dx;
            const y = anchors[i][1] + t * (anchors[i+1][1] - anchors[i][1]);
            return Math.max(0, Math.min(1, y));
        }
    }
    return Math.max(0, Math.min(1, anchors[n-1][1]));
}

function evalStep(anchors, x) {
    if (anchors.length === 0) return 1.0;
    for (let i = anchors.length - 1; i >= 0; i--) {
        if (x >= anchors[i][0]) return Math.max(0, Math.min(1, anchors[i][1]));
    }
    return Math.max(0, Math.min(1, anchors[0][1]));
}

function evalMonotone(anchors, x) {
    const n = anchors.length;
    if (n === 0) return 1.0;
    if (n === 1 || x <= anchors[0][0]) return Math.max(0, Math.min(1, anchors[0][1]));
    if (x >= anchors[n-1][0]) return Math.max(0, Math.min(1, anchors[n-1][1]));

    // Fritsch-Carlson monotone cubic
    const dx = [], dy = [], delta = [], m = [];
    for (let i = 0; i < n - 1; i++) {
        dx.push(anchors[i+1][0] - anchors[i][0]);
        dy.push(anchors[i+1][1] - anchors[i][1]);
        delta.push(dx[i] < 1e-9 ? 0 : dy[i] / dx[i]);
    }
    // Initial tangents
    m.push(delta[0]);
    for (let i = 1; i < n - 1; i++) {
        m.push((delta[i-1] + delta[i]) / 2);
    }
    m.push(delta[n-2]);
    // Fritsch-Carlson modification
    for (let i = 0; i < n - 1; i++) {
        if (Math.abs(delta[i]) < 1e-9) {
            m[i] = 0;
            m[i+1] = 0;
        } else {
            const alpha = m[i] / delta[i];
            const beta = m[i+1] / delta[i];
            const s = alpha * alpha + beta * beta;
            if (s > 9) {
                const tau = 3 / Math.sqrt(s);
                m[i] = tau * alpha * delta[i];
                m[i+1] = tau * beta * delta[i];
            }
        }
    }
    // Find segment and evaluate cubic Hermite
    for (let i = 0; i < n - 1; i++) {
        if (x <= anchors[i+1][0]) {
            const h = dx[i];
            if (h < 1e-9) return Math.max(0, Math.min(1, anchors[i][1]));
            const t = (x - anchors[i][0]) / h;
            const t2 = t * t, t3 = t2 * t;
            const h00 = 2*t3 - 3*t2 + 1;
            const h10 = t3 - 2*t2 + t;
            const h01 = -2*t3 + 3*t2;
            const h11 = t3 - t2;
            return Math.max(0, Math.min(1,
                h00 * anchors[i][1] + h10 * h * m[i] +
                h01 * anchors[i+1][1] + h11 * h * m[i+1]
            ));
        }
    }
    return Math.max(0, Math.min(1, anchors[n-1][1]));
}

function evalGaussian(anchors, x) {
    const n = anchors.length;
    if (n === 0) return 1.0;
    if (n === 1) return anchors[0][1];
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
        let sigma;
        if (i === 0) sigma = (anchors[1][0] - anchors[0][0]) * 0.6;
        else if (i === n-1) sigma = (anchors[n-1][0] - anchors[n-2][0]) * 0.6;
        else sigma = Math.min(anchors[i][0] - anchors[i-1][0], anchors[i+1][0] - anchors[i][0]) * 0.6;
        sigma = Math.max(sigma, 0.01);
        const d = x - anchors[i][0];
        const g = Math.exp(-(d * d) / (2 * sigma * sigma));
        num += anchors[i][1] * g;
        den += g;
    }
    return den > 0 ? Math.max(0, Math.min(1, num / den)) : 1.0;
}

function evalCatmullRom(anchors, x) {
    return evalBezierAtX(anchorsToPoints(anchors), x);
}

function evalAnchorsAtX(anchors, x, mode) {
    switch (mode) {
        case "linear": return evalLinear(anchors, x);
        case "step": return evalStep(anchors, x);
        case "monotone": return evalMonotone(anchors, x);
        case "gaussian": return evalGaussian(anchors, x);
        case "catmull-rom":
        default: return evalCatmullRom(anchors, x);
    }
}

// ── Color helper ────────────────────────────────────────────────────────────

function parseHexColor(hex) {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return { r, g, b };
}

// ── Constants ────────────────────────────────────────────────────────────────

const PREVIEW_MODES = ["colorize", "blend", "source"];
const COLORS = {
    a: "#4CAF50",
    b: "#2196F3",
    c: "#FF9800",
};
const GRID_COLOR = "#333";
const CURVE_WIDTH = 2;
const POINT_RADIUS = 5;
const HIT_RADIUS = 8;
const CURVE_AREA_HEIGHT = 180;
const PADDING = { top: 15, right: 15, bottom: 25, left: 35 };
const SAMPLE_COUNT = 200;

// ── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "VideoMatting.CurveEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AlphaCurveBlend") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            // ── State ───────────────────────────────────────────
            this.curves = [{
                id: "a",
                anchors: [[0, 1], [1, 1]],
                color: COLORS.a,
            }];
            this.activeCurveId = "a";
            this.interpMode = "catmull-rom";
            this.previewMode = "colorize";
            this.frameCache = {};
            this.totalFrames = 0;
            this.currentFrame = 0;
            this.dragging = null;
            this.previewSize = { w: 0, h: 0 };
            this.anchorOffset = 5;

            // ── DOM structure ────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex; flex-direction: column; width: 100%;
                padding: 4px; box-sizing: border-box; gap: 4px;
            `;

            // Preview mode toggle row
            const previewModeRow = document.createElement("div");
            previewModeRow.style.cssText = `
                display: flex; align-items: center; gap: 4px; width: 100%; flex-shrink: 0;
            `;
            this.previewModeRow = previewModeRow;
            container.appendChild(previewModeRow);
            this._rebuildPreviewModeButtons();

            // Preview canvas
            const PREVIEW_HEIGHT = 120;
            const previewCanvas = document.createElement("canvas");
            previewCanvas.style.cssText = `
                width: 100%; height: ${PREVIEW_HEIGHT}px; flex-shrink: 0;
                background: #1a1a1a; border-radius: 4px;
            `;
            previewCanvas.height = PREVIEW_HEIGHT;
            container.appendChild(previewCanvas);
            this.previewCanvas = previewCanvas;
            this.previewCtx = previewCanvas.getContext("2d");

            // Frame slider row
            const sliderRow = document.createElement("div");
            sliderRow.style.cssText = `
                display: flex; align-items: center; gap: 6px; width: 100%; flex-shrink: 0;
            `;
            const frameInput = document.createElement("input");
            frameInput.type = "number";
            frameInput.min = 0;
            frameInput.value = 0;
            frameInput.style.cssText = `
                width: 50px; padding: 2px 4px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #fff; font-size: 11px;
            `;
            frameInput.addEventListener("change", () => {
                const idx = Math.max(0, Math.min(this.totalFrames - 1, parseInt(frameInput.value) || 0));
                this.currentFrame = idx;
                frameSlider.value = idx;
                this._renderBlendedFrame();
                this._drawCurves();
            });
            sliderRow.appendChild(frameInput);
            this.frameInput = frameInput;

            const frameSlider = document.createElement("input");
            frameSlider.type = "range";
            frameSlider.min = 0;
            frameSlider.max = 0;
            frameSlider.value = 0;
            frameSlider.style.cssText = "flex: 1; cursor: pointer;";
            frameSlider.addEventListener("input", () => {
                this.currentFrame = parseInt(frameSlider.value);
                frameInput.value = this.currentFrame;
                this._renderBlendedFrame();
                this._drawCurves();
            });
            sliderRow.appendChild(frameSlider);
            this.frameSlider = frameSlider;

            const frameLabel = document.createElement("span");
            frameLabel.style.cssText = "font-size: 11px; color: #aaa; min-width: 40px; text-align: right;";
            frameLabel.textContent = "/ -";
            sliderRow.appendChild(frameLabel);
            this.frameLabel = frameLabel;

            // Add / Remove anchor buttons
            const anchorBtnStyle = `
                padding: 2px 8px; border: 1px solid #555; border-radius: 3px;
                background: #2a2a2a; color: #ccc; font-size: 12px; cursor: pointer;
                font-weight: bold; min-width: 28px; text-align: center;
            `;
            const addAnchorBtn = document.createElement("button");
            addAnchorBtn.textContent = "+";
            addAnchorBtn.title = "Add anchor at current frame (all curves)";
            addAnchorBtn.style.cssText = anchorBtnStyle;
            addAnchorBtn.addEventListener("click", () => this._addAnchorAtCurrentFrame());
            sliderRow.appendChild(addAnchorBtn);

            const removeAnchorBtn = document.createElement("button");
            removeAnchorBtn.textContent = "−";
            removeAnchorBtn.title = "Remove anchor at current frame (all curves)";
            removeAnchorBtn.style.cssText = anchorBtnStyle;
            removeAnchorBtn.addEventListener("click", () => this._removeAnchorAtCurrentFrame());
            sliderRow.appendChild(removeAnchorBtn);

            // Offset input
            const offsetLabel = document.createElement("span");
            offsetLabel.style.cssText = "font-size: 11px; color: #888;";
            offsetLabel.textContent = "±";
            sliderRow.appendChild(offsetLabel);

            const offsetInput = document.createElement("input");
            offsetInput.type = "number";
            offsetInput.min = 0;
            offsetInput.value = this.anchorOffset;
            offsetInput.style.cssText = `
                width: 36px; padding: 2px 4px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #fff; font-size: 11px;
            `;
            offsetInput.addEventListener("change", () => {
                this.anchorOffset = Math.max(0, parseInt(offsetInput.value) || 0);
                offsetInput.value = this.anchorOffset;
            });
            sliderRow.appendChild(offsetInput);
            this.offsetInput = offsetInput;

            container.appendChild(sliderRow);

            // Curve editor canvas
            const CURVE_CANVAS_HEIGHT = CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom;
            const curveCanvas = document.createElement("canvas");
            curveCanvas.style.cssText = `
                width: 100%; height: ${CURVE_CANVAS_HEIGHT}px; flex-shrink: 0;
                cursor: default; border-radius: 4px; background: #1a1a1a;
            `;
            curveCanvas.height = CURVE_CANVAS_HEIGHT;
            container.appendChild(curveCanvas);
            this.curveCanvas = curveCanvas;
            this.curveCtx = curveCanvas.getContext("2d");

            // Legend row (curve buttons + mode selector + reset)
            const legendRow = document.createElement("div");
            legendRow.style.cssText = `
                display: flex; align-items: center; gap: 8px; width: 100%;
                padding: 2px 0; flex-shrink: 0;
            `;
            this.legendRow = legendRow;
            container.appendChild(legendRow);

            this._rebuildLegend();

            // Hidden canvas for pixel reads
            this.tmpCanvas = document.createElement("canvas");
            this.tmpCtx = this.tmpCanvas.getContext("2d", { willReadFrequently: true });

            // ── Widget ───────────────────────────────────────────
            const widget = this.addDOMWidget("curve_editor", "custom", container, {
                serialize: false,
                hideOnZoom: false,
            });
            const CONTENT_HEIGHT = 24 + PREVIEW_HEIGHT + 28 + CURVE_CANVAS_HEIGHT + 28 + 12 + 8;
            widget.computeSize = (width) => {
                return [width, CONTENT_HEIGHT];
            };
            this.curveWidget = widget;
            this.setSize([340, CONTENT_HEIGHT + 100]);

            // ── Mouse events for curve canvas ────────────────────
            curveCanvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
            curveCanvas.addEventListener("mousemove", (e) => this._onMouseMove(e));
            curveCanvas.addEventListener("mouseup", () => this._onMouseUp());
            curveCanvas.addEventListener("mouseleave", () => this._onMouseUp());

            // Initial draw
            requestAnimationFrame(() => this._drawCurves());
        };

        // ── onConnectionsChange ─────────────────────────────────
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side, slotIdx, connected, link) {
            onConnectionsChange?.apply(this, arguments);
            if (side !== 1) return;

            const inputNames = ["alpha_a", "alpha_b", "alpha_c"];
            const ids = ["a", "b", "c"];

            for (let i = 0; i < inputNames.length; i++) {
                const input = this.inputs?.find(inp => inp.name === inputNames[i]);
                if (!input) continue;
                const isConnected = input.link != null;
                const curveExists = this.curves.some(c => c.id === ids[i]);

                if (isConnected && !curveExists) {
                    this.curves.push({
                        id: ids[i],
                        anchors: [[0, 1], [1, 1]],
                        color: COLORS[ids[i]] || "#888",
                    });
                } else if (!isConnected && curveExists && ids[i] !== "a") {
                    this.curves = this.curves.filter(c => c.id !== ids[i]);
                    if (this.activeCurveId === ids[i]) {
                        this.activeCurveId = "a";
                    }
                }
            }
            this._rebuildLegend();
            this._syncCurveData();
            this._drawCurves();
        };

        // ── onExecuted ──────────────────────────────────────────
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const sourceFrames = message?.source_frames?.[0];
            const totalFrames = message?.total_frames?.[0];
            if (sourceFrames) {
                this.totalFrames = totalFrames || 0;
                this.currentFrame = 0;
                this.frameSlider.max = Math.max(0, this.totalFrames - 1);
                this.frameSlider.value = 0;
                this.frameInput.max = Math.max(0, this.totalFrames - 1);
                this.frameInput.value = 0;
                this.frameLabel.textContent = `/ ${Math.max(0, this.totalFrames - 1)}`;
                this._preloadFrames(sourceFrames);
            }
        };

        // ── onConfigure ─────────────────────────────────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            const cdWidget = this.widgets?.find(w => w.name === "curve_data");
            if (cdWidget?.value) {
                try {
                    const data = JSON.parse(cdWidget.value);
                    if (data.curves?.length > 0) {
                        this.curves = data.curves.map(c => ({
                            id: c.id,
                            anchors: c.anchors || [[0, 1], [1, 1]],
                            color: c.color || COLORS[c.id] || "#888",
                        }));
                        this.activeCurveId = this.curves[0].id;
                    }
                    if (data.mode && INTERP_MODES.includes(data.mode)) {
                        this.interpMode = data.mode;
                    }
                    this._rebuildLegend();
                    requestAnimationFrame(() => this._drawCurves());
                } catch (e) { /* keep defaults */ }
            }
        };

        // ── Methods ─────────────────────────────────────────────

        nodeType.prototype._preloadFrames = function (sourceFrames) {
            this.frameCache = {};
            let firstLoaded = false;
            for (const [id, frames] of Object.entries(sourceFrames)) {
                this.frameCache[id] = frames.map((f, idx) => {
                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    img.src = api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}&type=${f.type || "temp"}`);
                    if (idx === 0 && !firstLoaded) {
                        firstLoaded = true;
                        img.onload = () => {
                            this.previewSize = { w: img.naturalWidth, h: img.naturalHeight };
                            this.tmpCanvas.width = img.naturalWidth;
                            this.tmpCanvas.height = img.naturalHeight;
                            this._renderBlendedFrame();
                            this._drawCurves();
                        };
                    }
                    return img;
                });
            }
        };

        nodeType.prototype._renderBlendedFrame = function () {
            const { w, h } = this.previewSize;
            if (w === 0 || h === 0 || this.totalFrames === 0) return;

            const mode = this.previewMode;

            // Source mode: just draw the active source frame directly
            if (mode === "source") {
                const cache = this.frameCache[this.activeCurveId];
                const img = cache?.[this.currentFrame];
                if (!img || !img.complete || img.naturalWidth === 0) return;
                this._drawToPreview(img);
                return;
            }

            // Blend / Colorize modes need weights + pixel data
            const t = this.totalFrames > 1
                ? this.currentFrame / (this.totalFrames - 1)
                : 0;

            const weights = {};
            let weightSum = 0;
            for (const curve of this.curves) {
                const cw = Math.max(0, evalAnchorsAtX(curve.anchors, t, this.interpMode));
                weights[curve.id] = cw;
                weightSum += cw;
            }
            if (weightSum === 0) weightSum = 1;

            const pixelArrays = {};
            for (const [id, cache] of Object.entries(this.frameCache)) {
                if (weights[id] === undefined || weights[id] === 0) continue;
                const img = cache[this.currentFrame];
                if (!img || !img.complete || img.naturalWidth === 0) continue;
                this.tmpCtx.drawImage(img, 0, 0, w, h);
                pixelArrays[id] = this.tmpCtx.getImageData(0, 0, w, h).data;
            }

            const ids = Object.keys(pixelArrays);
            if (ids.length === 0) return;

            const out = this.previewCtx.createImageData(w, h);
            const len = out.data.length;

            if (mode === "colorize") {
                // Active source tinted with curve color, others grayscale
                const activeId = this.activeCurveId;
                const activeCurve = this.curves.find(c => c.id === activeId);
                const tint = activeCurve ? parseHexColor(activeCurve.color) : { r: 1, g: 1, b: 1 };

                for (let i = 0; i < len; i += 4) {
                    let activeWeighted = 0;
                    let otherWeighted = 0;
                    for (const id of ids) {
                        const contrib = (pixelArrays[id][i] / 255) * weights[id];
                        if (id === activeId) activeWeighted = contrib;
                        else otherWeighted += contrib;
                    }
                    const aw = activeWeighted / weightSum;
                    const ow = otherWeighted / weightSum;
                    out.data[i]     = Math.round(Math.min(1, aw * tint.r + ow) * 255);
                    out.data[i + 1] = Math.round(Math.min(1, aw * tint.g + ow) * 255);
                    out.data[i + 2] = Math.round(Math.min(1, aw * tint.b + ow) * 255);
                    out.data[i + 3] = 255;
                }
            } else {
                // Blend: pure grayscale
                for (let i = 0; i < len; i += 4) {
                    let val = 0;
                    for (const id of ids) {
                        val += pixelArrays[id][i] * weights[id];
                    }
                    val = Math.round(val / weightSum);
                    out.data[i] = out.data[i+1] = out.data[i+2] = val;
                    out.data[i+3] = 255;
                }
            }

            // Put image data to temp canvas, then scale to preview
            const tmpOut = document.createElement("canvas");
            tmpOut.width = w;
            tmpOut.height = h;
            tmpOut.getContext("2d").putImageData(out, 0, 0);
            this._drawToPreview(tmpOut);
        };

        nodeType.prototype._drawToPreview = function (source) {
            const { w, h } = this.previewSize;
            const dpr = window.devicePixelRatio || 1;
            const pNewW = Math.round(this.previewCanvas.clientWidth * dpr);
            const pNewH = Math.round(120 * dpr);
            if (this.previewCanvas.width !== pNewW || this.previewCanvas.height !== pNewH) {
                this.previewCanvas.width = pNewW;
                this.previewCanvas.height = pNewH;
            }
            const pCtx = this.previewCtx;
            pCtx.save();
            pCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            const dispW = this.previewCanvas.clientWidth;
            const dispH = 120;
            const scale = Math.min(dispW / w, dispH / h);
            const dx = (dispW - w * scale) / 2;
            const dy = (dispH - h * scale) / 2;
            pCtx.fillStyle = "#1a1a1a";
            pCtx.fillRect(0, 0, dispW, dispH);
            pCtx.drawImage(source, dx, dy, w * scale, h * scale);
            pCtx.restore();
        };

        // ── Curve drawing ───────────────────────────────────────

        nodeType.prototype._drawCurves = function () {
            const canvas = this.curveCanvas;
            const dpr = window.devicePixelRatio || 1;
            const dispW = canvas.clientWidth;
            const dispH = CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom;
            this._drawW = dispW;
            this._drawH = dispH;
            const newW = Math.round(dispW * dpr);
            const newH = Math.round(dispH * dpr);
            if (canvas.width !== newW || canvas.height !== newH) {
                canvas.width = newW;
                canvas.height = newH;
            }
            const ctx = this.curveCtx;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const areaW = dispW - PADDING.left - PADDING.right;
            const areaH = CURVE_AREA_HEIGHT;

            // Background
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, dispW, dispH);

            // Grid
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {
                const y = PADDING.top + (areaH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(PADDING.left, y);
                ctx.lineTo(PADDING.left + areaW, y);
                ctx.stroke();
            }
            for (let i = 0; i <= 4; i++) {
                const x = PADDING.left + (areaW / 4) * i;
                ctx.beginPath();
                ctx.moveTo(x, PADDING.top);
                ctx.lineTo(x, PADDING.top + areaH);
                ctx.stroke();
            }

            // Axis labels
            ctx.fillStyle = "#666";
            ctx.font = "10px monospace";
            ctx.textAlign = "right";
            for (let i = 0; i <= 4; i++) {
                const val = (1 - i / 4).toFixed(1);
                const y = PADDING.top + (areaH / 4) * i;
                ctx.fillText(val, PADDING.left - 4, y + 3);
            }
            ctx.textAlign = "center";
            for (let i = 0; i <= 4; i++) {
                const val = (i / 4).toFixed(1);
                const x = PADDING.left + (areaW / 4) * i;
                ctx.fillText(val, x, PADDING.top + areaH + 14);
            }

            // Playhead
            if (this.totalFrames > 1) {
                const t = this.currentFrame / (this.totalFrames - 1);
                const px = PADDING.left + t * areaW;
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(px, PADDING.top);
                ctx.lineTo(px, PADDING.top + areaH);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            const toCanvas = (nx, ny) => [
                PADDING.left + nx * areaW,
                PADDING.top + (1 - ny) * areaH,
            ];

            // Current frame x for highlight
            const currentX = this.totalFrames > 1
                ? this.currentFrame / (this.totalFrames - 1)
                : -1;
            const HIGHLIGHT_EPS = 0.001;
            const mode = this.interpMode;

            // Draw curves (inactive first, active last = on top)
            const sortedCurves = [...this.curves].sort((a, b) =>
                (a.id === this.activeCurveId ? 1 : 0) - (b.id === this.activeCurveId ? 1 : 0)
            );
            for (const curve of sortedCurves) {
                const isActive = curve.id === this.activeCurveId;
                const alpha = isActive ? 1.0 : 0.35;
                const color = curve.color;

                ctx.strokeStyle = color;
                ctx.globalAlpha = alpha;
                ctx.lineWidth = isActive ? CURVE_WIDTH + 0.5 : CURVE_WIDTH;
                ctx.beginPath();

                if (mode === "catmull-rom") {
                    // Native bezierCurveTo for smooth rendering
                    const pts = anchorsToPoints(curve.anchors);
                    if (pts.length >= 4) {
                        const numSeg = Math.floor((pts.length - 1) / 3);
                        const [sx, sy] = toCanvas(pts[0][0], pts[0][1]);
                        ctx.moveTo(sx, sy);
                        for (let seg = 0; seg < numSeg; seg++) {
                            const idx = seg * 3;
                            const [c1x, c1y] = toCanvas(pts[idx+1][0], pts[idx+1][1]);
                            const [c2x, c2y] = toCanvas(pts[idx+2][0], pts[idx+2][1]);
                            const [ex, ey] = toCanvas(pts[idx+3][0], pts[idx+3][1]);
                            ctx.bezierCurveTo(c1x, c1y, c2x, c2y, ex, ey);
                        }
                    } else if (pts.length >= 2) {
                        const [sx, sy] = toCanvas(pts[0][0], pts[0][1]);
                        const [ex, ey] = toCanvas(pts[pts.length-1][0], pts[pts.length-1][1]);
                        ctx.moveTo(sx, sy);
                        ctx.lineTo(ex, ey);
                    }
                } else {
                    // Sample-based polyline for all other modes
                    for (let s = 0; s <= SAMPLE_COUNT; s++) {
                        const nx = s / SAMPLE_COUNT;
                        const ny = evalAnchorsAtX(curve.anchors, nx, mode);
                        const [cx, cy] = toCanvas(nx, ny);
                        if (s === 0) ctx.moveTo(cx, cy);
                        else ctx.lineTo(cx, cy);
                    }
                }
                ctx.stroke();

                // Draw anchor points
                for (const anchor of curve.anchors) {
                    const [cx, cy] = toCanvas(anchor[0], anchor[1]);
                    const isHighlight = isActive && currentX >= 0 && Math.abs(anchor[0] - currentX) < HIGHLIGHT_EPS;

                    ctx.globalAlpha = isActive ? 1.0 : 0.5;
                    ctx.beginPath();
                    ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
                    ctx.fillStyle = isHighlight ? "#fff" : color;
                    ctx.fill();
                }

                ctx.globalAlpha = 1.0;
            }

            ctx.restore();
        };

        // ── Mouse interaction ───────────────────────────────────

        nodeType.prototype._canvasToNorm = function (e) {
            const rect = this.curveCanvas.getBoundingClientRect();
            const drawW = this._drawW || (PADDING.left + PADDING.right + 100);
            const drawH = this._drawH || (CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom);
            const mx = (e.clientX - rect.left) * (drawW / rect.width);
            const my = (e.clientY - rect.top) * (drawH / rect.height);
            const areaW = drawW - PADDING.left - PADDING.right;
            const areaH = CURVE_AREA_HEIGHT;
            const nx = (mx - PADDING.left) / areaW;
            const ny = 1 - (my - PADDING.top) / areaH;
            return [nx, ny];
        };

        nodeType.prototype._hitTest = function (e) {
            const [nx, ny] = this._canvasToNorm(e);
            const rect = this.curveCanvas.getBoundingClientRect();
            const drawW = this._drawW || (PADDING.left + PADDING.right + 100);
            const drawH = this._drawH || (CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom);
            const areaW = drawW - PADDING.left - PADDING.right;
            const areaH = CURVE_AREA_HEIGHT;
            const hitNx = (HIT_RADIUS / rect.width) * (drawW / areaW);
            const hitNy = (HIT_RADIUS / rect.height) * (drawH / areaH);

            const sorted = [...this.curves].sort((a, b) =>
                (b.id === this.activeCurveId ? 1 : 0) - (a.id === this.activeCurveId ? 1 : 0)
            );

            for (const curve of sorted) {
                for (let i = 0; i < curve.anchors.length; i++) {
                    const [px, py] = curve.anchors[i];
                    if (Math.abs(nx - px) < hitNx && Math.abs(ny - py) < hitNy) {
                        const curveIdx = this.curves.indexOf(curve);
                        return { curveIdx, anchorIdx: i };
                    }
                }
            }
            return null;
        };

        nodeType.prototype._onMouseDown = function (e) {
            if (e.button !== 0) return;
            const hit = this._hitTest(e);
            if (hit) {
                this.dragging = hit;
                this.activeCurveId = this.curves[hit.curveIdx].id;
                // Snapshot initial anchors for proportional editing
                const curve = this.curves[hit.curveIdx];
                this._dragInitialAnchors = curve.anchors.map(a => [a[0], a[1]]);
                this._rebuildLegend();
                this.curveCanvas.style.cursor = "grabbing";
            }
        };

        nodeType.prototype._onMouseMove = function (e) {
            if (!this.dragging) {
                const hit = this._hitTest(e);
                this.curveCanvas.style.cursor = hit ? "ns-resize" : "default";
                return;
            }
            const [, ny] = this._canvasToNorm(e);
            const curve = this.curves[this.dragging.curveIdx];
            const idx = this.dragging.anchorIdx;
            const initAnchors = this._dragInitialAnchors;
            const clampedY = Math.max(0, Math.min(1, ny));
            const deltaY = clampedY - initAnchors[idx][1];

            // Update dragged anchor
            curve.anchors[idx] = [curve.anchors[idx][0], clampedY];

            // Proportional editing: move neighbors with Gaussian falloff
            const offset = this.anchorOffset;
            if (offset > 0 && this.totalFrames > 1 && Math.abs(deltaY) > 1e-6) {
                const sigma = offset / (this.totalFrames - 1);
                const dragX = curve.anchors[idx][0];
                for (let i = 0; i < curve.anchors.length; i++) {
                    if (i === idx) continue;
                    // Don't move first/last anchors
                    if (i === 0 || i === curve.anchors.length - 1) continue;
                    const dist = Math.abs(curve.anchors[i][0] - dragX);
                    if (dist > sigma * 3) continue; // cutoff
                    const falloff = Math.exp(-(dist * dist) / (2 * sigma * sigma));
                    const newY = initAnchors[i][1] + deltaY * falloff;
                    curve.anchors[i] = [curve.anchors[i][0], Math.max(0, Math.min(1, newY))];
                }
            }

            if (this.previewMode !== "source") {
                this._renderBlendedFrame();
            }
            this._drawCurves();
        };

        nodeType.prototype._onMouseUp = function () {
            if (this.dragging) {
                this.dragging = null;
                this._dragInitialAnchors = null;
                this.curveCanvas.style.cursor = "default";
                this._syncCurveData();
                this._drawCurves();
            }
        };

        // ── Add/Remove anchors at current frame ─────────────────

        nodeType.prototype._currentFrameX = function () {
            if (this.totalFrames <= 1) return 0;
            return this.currentFrame / (this.totalFrames - 1);
        };

        nodeType.prototype._insertAnchor = function (curve, x) {
            const EPS = 0.001;
            if (curve.anchors.some(a => Math.abs(a[0] - x) < EPS)) return false;
            const y = Math.max(0, Math.min(1, evalAnchorsAtX(curve.anchors, x, this.interpMode)));
            let insertIdx = curve.anchors.length;
            for (let i = 0; i < curve.anchors.length; i++) {
                if (x < curve.anchors[i][0]) { insertIdx = i; break; }
            }
            curve.anchors.splice(insertIdx, 0, [x, y]);
            return true;
        };

        nodeType.prototype._addAnchorAtCurrentFrame = function () {
            if (this.totalFrames === 0) return;
            const curve = this.curves.find(c => c.id === this.activeCurveId);
            if (!curve) return;
            const x = this._currentFrameX();
            const EPS = 0.001;
            if (curve.anchors.some(a => Math.abs(a[0] - x) < EPS)) return;

            // Insert guards first (so center eval uses original curve)
            const offset = this.anchorOffset;
            if (offset > 0 && this.totalFrames > 1) {
                const offsetX = offset / (this.totalFrames - 1);
                const leftX = x - offsetX;
                const rightX = x + offsetX;
                if (leftX > EPS) this._insertAnchor(curve, leftX);
                if (rightX < 1 - EPS) this._insertAnchor(curve, rightX);
            }

            // Insert center
            this._insertAnchor(curve, x);

            this._syncCurveData();
            this._drawCurves();
            this._renderBlendedFrame();
        };

        nodeType.prototype._removeAnchorAtCurrentFrame = function () {
            if (this.totalFrames === 0) return;
            const curve = this.curves.find(c => c.id === this.activeCurveId);
            if (!curve) return;
            const x = this._currentFrameX();
            const EPS = 0.001;
            if (x < EPS || x > 1 - EPS) return;
            const idx = curve.anchors.findIndex(a => Math.abs(a[0] - x) < EPS);
            if (idx > 0 && idx < curve.anchors.length - 1 && curve.anchors.length > 2) {
                curve.anchors.splice(idx, 1);
                this._syncCurveData();
                this._drawCurves();
                this._renderBlendedFrame();
            }
        };

        // ── Preview mode buttons ────────────────────────────────

        nodeType.prototype._rebuildPreviewModeButtons = function () {
            const row = this.previewModeRow;
            row.innerHTML = "";
            for (const m of PREVIEW_MODES) {
                const btn = document.createElement("button");
                const isActive = m === this.previewMode;
                btn.textContent = m.charAt(0).toUpperCase() + m.slice(1);
                btn.style.cssText = `
                    padding: 2px 10px; border: 1px solid ${isActive ? "#aaa" : "#555"};
                    border-radius: 3px; background: ${isActive ? "#444" : "#2a2a2a"};
                    color: ${isActive ? "#fff" : "#888"}; font-size: 11px; cursor: pointer;
                    font-weight: ${isActive ? "bold" : "normal"};
                `;
                btn.addEventListener("click", () => {
                    this.previewMode = m;
                    this._rebuildPreviewModeButtons();
                    this._renderBlendedFrame();
                });
                row.appendChild(btn);
            }
        };

        // ── Legend ───────────────────────────────────────────────

        nodeType.prototype._rebuildLegend = function () {
            const row = this.legendRow;
            row.innerHTML = "";

            // Curve buttons
            for (const curve of this.curves) {
                const btn = document.createElement("button");
                const isActive = curve.id === this.activeCurveId;
                btn.style.cssText = `
                    display: flex; align-items: center; gap: 4px;
                    padding: 2px 8px; border: 1px solid ${isActive ? curve.color : "#555"};
                    border-radius: 3px; background: ${isActive ? curve.color + "22" : "#2a2a2a"};
                    color: ${curve.color}; font-size: 11px; cursor: pointer;
                    font-weight: ${isActive ? "bold" : "normal"};
                `;
                const dot = document.createElement("span");
                dot.style.cssText = `
                    width: 8px; height: 8px; border-radius: 50%;
                    background: ${curve.color}; display: inline-block;
                `;
                btn.appendChild(dot);
                btn.appendChild(document.createTextNode(curve.id.toUpperCase()));
                btn.addEventListener("click", () => {
                    this.activeCurveId = curve.id;
                    this._rebuildLegend();
                    this._drawCurves();
                    if (this.previewMode !== "blend") {
                        this._renderBlendedFrame();
                    }
                });
                row.appendChild(btn);
            }

            // Spacer
            const spacer = document.createElement("div");
            spacer.style.cssText = "flex: 1;";
            row.appendChild(spacer);

            // Mode selector
            const modeSelect = document.createElement("select");
            modeSelect.style.cssText = `
                padding: 2px 4px; border: 1px solid #555; border-radius: 3px;
                background: #2a2a2a; color: #ccc; font-size: 11px; cursor: pointer;
            `;
            for (const m of INTERP_MODES) {
                const opt = document.createElement("option");
                opt.value = m;
                opt.textContent = m;
                if (m === this.interpMode) opt.selected = true;
                modeSelect.appendChild(opt);
            }
            modeSelect.addEventListener("change", () => {
                this.interpMode = modeSelect.value;
                this._syncCurveData();
                this._drawCurves();
                this._renderBlendedFrame();
            });
            row.appendChild(modeSelect);

            // Reset button
            const resetBtn = document.createElement("button");
            resetBtn.style.cssText = `
                padding: 2px 8px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #aaa;
                font-size: 11px; cursor: pointer;
            `;
            resetBtn.textContent = "Reset";
            resetBtn.addEventListener("click", () => {
                for (const curve of this.curves) {
                    curve.anchors = [[0, 1], [1, 1]];
                }
                this._syncCurveData();
                this._drawCurves();
                this._renderBlendedFrame();
            });
            row.appendChild(resetBtn);
        };

        // ── Sync curve data to hidden widget ────────────────────

        nodeType.prototype._syncCurveData = function () {
            const cdWidget = this.widgets?.find(w => w.name === "curve_data");
            if (cdWidget) {
                const data = {
                    mode: this.interpMode,
                    curves: this.curves.map(c => ({
                        id: c.id,
                        anchors: c.anchors,
                        color: c.color,
                    })),
                };
                cdWidget.value = JSON.stringify(data);
            }
        };
    },
});
