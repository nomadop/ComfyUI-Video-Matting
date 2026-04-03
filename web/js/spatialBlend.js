import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── Constants ────────────────────────────────────────────────────────────────

const COLORS = { a: "#4CAF50", b: "#2196F3", c: "#FF9800" };
const GRID_COLOR = "rgba(255,255,255,0.25)";
const GRID_HIGHLIGHT_COLOR = "rgba(255,255,255,0.85)";
const CURVE_AREA_HEIGHT = 150;
const PREVIEW_HEIGHT = 200;
const PADDING = { top: 15, right: 15, bottom: 25, left: 35 };
const GRID_LINE_COLOR = "#333";

// ── Helpers ──────────────────────────────────────────────────────────────────

function viewURL(frame, cacheTs) {
    const params = new URLSearchParams({
        filename: frame.filename,
        subfolder: frame.subfolder || "",
        type: frame.type || "temp",
    });
    if (cacheTs) params.set("_t", cacheTs);
    return api.apiURL(`/view?${params.toString()}`);
}

// ── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "VideoMatting.SpatialBlend",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SpatialAlphaBlend") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            // ── State ────────────────────────────────────────────
            this.totalFrames = 0;
            this.currentFrame = 0;
            this.gridSize = 8;
            this.sourceIds = ["a"];
            this.selectedBlock = null; // { bx, by } or null for global
            this.previewSize = { w: 0, h: 0 };

            // Caches
            this._blendedImages = [];    // Image objects for blended frames
            this._weightData = null;     // Float32Array[B][GH * GW * numSrc]
            this._generation = 0;
            this._cacheTs = "0";
            this._lastFingerprint = null;

            // Prevent LiteGraph image preview
            const origDrawBg = this.onDrawBackground;
            this.onDrawBackground = function (ctx) {
                const saved = this.imgs;
                this.imgs = null;
                origDrawBg?.call(this, ctx);
                this.imgs = saved;
            };

            // ── DOM ──────────────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex; flex-direction: column; width: 100%;
                padding: 4px; box-sizing: border-box; gap: 4px;
            `;

            // Preview canvas
            const previewCanvas = document.createElement("canvas");
            previewCanvas.style.cssText = `
                width: 100%; height: ${PREVIEW_HEIGHT}px; flex-shrink: 0;
                background: #1a1a1a; border-radius: 4px; cursor: crosshair;
            `;
            previewCanvas.height = PREVIEW_HEIGHT;
            previewCanvas.addEventListener("click", (e) => this._onPreviewClick(e));
            container.appendChild(previewCanvas);
            this.previewCanvas = previewCanvas;
            this.previewCtx = previewCanvas.getContext("2d");

            // Slider row
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
                this.frameSlider.value = idx;
                this._renderPreview();
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
                this._renderPreview();
                this._drawCurves();
            });
            sliderRow.appendChild(frameSlider);
            this.frameSlider = frameSlider;

            const frameLabel = document.createElement("span");
            frameLabel.style.cssText = "font-size: 11px; color: #aaa; min-width: 40px; text-align: right;";
            frameLabel.textContent = "/ -";
            sliderRow.appendChild(frameLabel);
            this.frameLabel = frameLabel;

            container.appendChild(sliderRow);

            // Curve canvas
            const CURVE_CANVAS_HEIGHT = CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom;
            const curveCanvas = document.createElement("canvas");
            curveCanvas.style.cssText = `
                width: 100%; height: ${CURVE_CANVAS_HEIGHT}px; flex-shrink: 0;
                border-radius: 4px; background: #1a1a1a;
            `;
            curveCanvas.height = CURVE_CANVAS_HEIGHT;
            container.appendChild(curveCanvas);
            this.curveCanvas = curveCanvas;
            this.curveCtx = curveCanvas.getContext("2d");

            // Legend row
            const legendRow = document.createElement("div");
            legendRow.style.cssText = `
                display: flex; align-items: center; gap: 8px; width: 100%;
                padding: 2px 0; flex-shrink: 0; font-size: 11px; color: #aaa;
            `;
            container.appendChild(legendRow);
            this.legendRow = legendRow;

            // Widget
            const CONTENT_HEIGHT = PREVIEW_HEIGHT + 28 + CURVE_CANVAS_HEIGHT + 24 + 16;
            const widget = this.addDOMWidget("spatial_blend_view", "custom", container, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = (width) => [width, CONTENT_HEIGHT];
            this.setSize([380, CONTENT_HEIGHT + 100]);
        };

        // ── onExecuted ───────────────────────────────────────────
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const blendedFrames = message?.blended_frames?.[0];
            const totalFrames = message?.total_frames?.[0];
            const gridSize = message?.grid_size?.[0];
            const sourceIds = message?.source_ids?.[0];
            const weightFrames = message?.weight_frames?.[0];

            if (!blendedFrames || !totalFrames) return;

            this._generation++;
            const gen = this._generation;

            // Fingerprint change detection
            const fp = message.input_fingerprint?.[0];
            if (fp) {
                this._cacheTs = fp;
                if (this._lastFingerprint !== null && this._lastFingerprint !== fp) {
                    this.selectedBlock = null;
                }
                this._lastFingerprint = fp;
            }

            this.totalFrames = totalFrames;
            this.gridSize = gridSize || 8;
            this.sourceIds = sourceIds || ["a"];
            this.currentFrame = 0;
            this.frameSlider.max = Math.max(0, totalFrames - 1);
            this.frameSlider.value = 0;
            this.frameInput.max = Math.max(0, totalFrames - 1);
            this.frameInput.value = 0;
            this.frameLabel.textContent = `/ ${Math.max(0, totalFrames - 1)}`;

            // Load blended frames
            this._blendedImages = blendedFrames.map((f, idx) => {
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.src = viewURL(f, this._cacheTs);
                if (idx === 0) {
                    img.onload = () => {
                        if (this._generation !== gen) return;
                        this.previewSize = { w: img.naturalWidth, h: img.naturalHeight };
                        this._renderPreview();
                        this._drawCurves();
                    };
                }
                return img;
            });

            // Load weight maps
            this._weightData = null;
            if (weightFrames && weightFrames.length > 0) {
                this._loadWeightMaps(weightFrames, gen);
            }

            this._rebuildLegend();
        };

        // ── Weight map loading ───────────────────────────────────
        nodeType.prototype._loadWeightMaps = function (weightFrames, gen) {
            const GH = this.gridSize;
            const GW = this.gridSize;
            const numSrc = this.sourceIds.length;
            const B = weightFrames.length;

            const canvas = document.createElement("canvas");
            canvas.width = GW;
            canvas.height = GH;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });

            const data = new Array(B);
            let loaded = 0;

            weightFrames.forEach((f, idx) => {
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => {
                    if (this._generation !== gen) return;
                    ctx.clearRect(0, 0, GW, GH);
                    ctx.drawImage(img, 0, 0, GW, GH);
                    const pixels = ctx.getImageData(0, 0, GW, GH).data;

                    // Decode: R=src0, G=src1, B=src2
                    const weights = new Float32Array(GH * GW * numSrc);
                    for (let p = 0; p < GH * GW; p++) {
                        for (let s = 0; s < numSrc; s++) {
                            weights[p * numSrc + s] = pixels[p * 4 + s] / 255.0;
                        }
                    }
                    data[idx] = weights;

                    loaded++;
                    if (loaded === B) {
                        this._weightData = data;
                        this._drawCurves();
                    }
                };
                img.src = viewURL(f, this._cacheTs);
            });
        };

        // ── Preview rendering ────────────────────────────────────
        nodeType.prototype._renderPreview = function () {
            const img = this._blendedImages[this.currentFrame];
            if (!img || !img.complete || img.naturalWidth === 0) return;
            this._drawToPreview(img);
            this._drawGrid();
        };

        nodeType.prototype._drawToPreview = function (source) {
            const { w, h } = this.previewSize;
            if (!w || !h) return;
            const dpr = window.devicePixelRatio || 1;
            const dispW = this.previewCanvas.clientWidth;
            const dispH = PREVIEW_HEIGHT;
            const pW = Math.round(dispW * dpr);
            const pH = Math.round(dispH * dpr);
            if (this.previewCanvas.width !== pW || this.previewCanvas.height !== pH) {
                this.previewCanvas.width = pW;
                this.previewCanvas.height = pH;
            }
            const ctx = this.previewCtx;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const scale = Math.min(dispW / w, dispH / h);
            this._imgScale = scale;
            this._imgOffX = (dispW - w * scale) / 2;
            this._imgOffY = (dispH - h * scale) / 2;

            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, dispW, dispH);
            ctx.drawImage(source, this._imgOffX, this._imgOffY, w * scale, h * scale);
            ctx.restore();
        };

        nodeType.prototype._drawGrid = function () {
            const { w, h } = this.previewSize;
            if (!w || !h) return;
            const dpr = window.devicePixelRatio || 1;
            const ctx = this.previewCtx;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const scale = this._imgScale || 1;
            const ox = this._imgOffX || 0;
            const oy = this._imgOffY || 0;
            const GH = this.gridSize;
            const GW = this.gridSize;
            const blockW = (w * scale) / GW;
            const blockH = (h * scale) / GH;

            // Draw grid lines
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 0.5;
            for (let i = 1; i < GW; i++) {
                const x = ox + i * blockW;
                ctx.beginPath();
                ctx.moveTo(x, oy);
                ctx.lineTo(x, oy + h * scale);
                ctx.stroke();
            }
            for (let i = 1; i < GH; i++) {
                const y = oy + i * blockH;
                ctx.beginPath();
                ctx.moveTo(ox, y);
                ctx.lineTo(ox + w * scale, y);
                ctx.stroke();
            }

            // Highlight selected block
            if (this.selectedBlock) {
                const { bx, by } = this.selectedBlock;
                ctx.strokeStyle = GRID_HIGHLIGHT_COLOR;
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    ox + bx * blockW,
                    oy + by * blockH,
                    blockW,
                    blockH,
                );
            }

            ctx.restore();
        };

        // ── Preview click → select block ─────────────────────────
        nodeType.prototype._onPreviewClick = function (e) {
            const { w, h } = this.previewSize;
            if (!w || !h) return;

            const rect = this.previewCanvas.getBoundingClientRect();
            // Convert screen coords to logical coords (matching setTransform(dpr,...) used in drawing)
            const dispW = this.previewCanvas.clientWidth;
            const dispH = PREVIEW_HEIGHT;
            const mx = (e.clientX - rect.left) * (dispW / rect.width);
            const my = (e.clientY - rect.top) * (dispH / rect.height);

            const scale = this._imgScale || 1;
            const ox = this._imgOffX || 0;
            const oy = this._imgOffY || 0;

            // Convert to image coords
            const ix = (mx - ox) / scale;
            const iy = (my - oy) / scale;

            if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
                // Click outside image → deselect (global view)
                this.selectedBlock = null;
            } else {
                const bx = Math.min(Math.floor(ix / w * this.gridSize), this.gridSize - 1);
                const by = Math.min(Math.floor(iy / h * this.gridSize), this.gridSize - 1);

                // Toggle: click same block → deselect
                if (this.selectedBlock && this.selectedBlock.bx === bx && this.selectedBlock.by === by) {
                    this.selectedBlock = null;
                } else {
                    this.selectedBlock = { bx, by };
                }
            }

            this._renderPreview();
            this._drawCurves();
            this._rebuildLegend();
        };

        // ── Curve drawing (read-only) ────────────────────────────
        nodeType.prototype._drawCurves = function () {
            const canvas = this.curveCanvas;
            const dpr = window.devicePixelRatio || 1;
            const dispW = canvas.clientWidth;
            const dispH = CURVE_AREA_HEIGHT + PADDING.top + PADDING.bottom;
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
            ctx.strokeStyle = GRID_LINE_COLOR;
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

            // Draw weight curves
            if (this._weightData && this.totalFrames > 0) {
                const GH = this.gridSize;
                const GW = this.gridSize;
                const numSrc = this.sourceIds.length;

                for (let s = 0; s < numSrc; s++) {
                    const sid = this.sourceIds[s];
                    const color = COLORS[sid] || "#888";
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    for (let i = 0; i < this.totalFrames; i++) {
                        const weights = this._weightData[i];
                        if (!weights) continue;

                        let w;
                        if (this.selectedBlock) {
                            // Single block
                            const { bx, by } = this.selectedBlock;
                            const p = (by * GW + bx) * numSrc + s;
                            w = weights[p];
                        } else {
                            // Global average
                            let sum = 0;
                            for (let p = 0; p < GH * GW; p++) {
                                sum += weights[p * numSrc + s];
                            }
                            w = sum / (GH * GW);
                        }

                        const x = PADDING.left + (i / Math.max(1, this.totalFrames - 1)) * areaW;
                        const y = PADDING.top + (1 - w) * areaH;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                }
            }

            ctx.restore();
        };

        // ── Legend ────────────────────────────────────────────────
        nodeType.prototype._rebuildLegend = function () {
            const row = this.legendRow;
            row.innerHTML = "";

            // Source dots
            for (const sid of this.sourceIds) {
                const color = COLORS[sid] || "#888";
                const span = document.createElement("span");
                span.style.cssText = `display: flex; align-items: center; gap: 3px;`;
                const dot = document.createElement("span");
                dot.style.cssText = `
                    width: 8px; height: 8px; border-radius: 50%;
                    background: ${color}; display: inline-block;
                `;
                span.appendChild(dot);
                span.appendChild(document.createTextNode(sid.toUpperCase()));
                row.appendChild(span);
            }

            // Spacer
            const spacer = document.createElement("div");
            spacer.style.cssText = "flex: 1;";
            row.appendChild(spacer);

            // Block info
            const blockInfo = document.createElement("span");
            blockInfo.style.cssText = "color: #888;";
            if (this.selectedBlock) {
                blockInfo.textContent = `Block (${this.selectedBlock.bx}, ${this.selectedBlock.by})`;
            } else {
                blockInfo.textContent = `Global avg (${this.gridSize}×${this.gridSize})`;
            }
            row.appendChild(blockInfo);
        };
    },
});
