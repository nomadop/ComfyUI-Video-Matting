import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── Constants ─────────────────────────────────────────────────────────────────

const PROPAGATE_DIFF_THRESHOLD = 0.03; // ignore diff below this (re-encoding noise)

// ── Helpers ────────────────────────────────────────────────────────────────────

function viewURL(frame, channel, cacheTs) {
    const params = new URLSearchParams({
        filename: frame.filename,
        subfolder: frame.subfolder || "",
        type: frame.type || "temp",
    });
    if (channel) params.set("channel", channel);
    if (cacheTs) params.set("_t", cacheTs);
    return api.apiURL(`/view?${params.toString()}`);
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// ── Extension ──────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "VideoMatting.PreviewSlider",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PreviewSlider") return;

        // ── onNodeCreated ──────────────────────────────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.frames = [];
            this.totalFrames = 0;
            this.currentFrame = 0;
            this.hasImages = false;
            this.hasMask = false;
            this.editedMasks = new Map(); // frameIndex → {filename, subfolder, type}
            this._renderGen = 0; // generation counter to discard stale renders
            this._showMaskBW = false; // toggle: overlay vs B&W mask
            this._cacheTs = "0";
            this._lastFingerprint = null;
            this._flowCache = null; // { fwd: [{dx,dy},...], bwd: [{dx,dy},...] }
            this._flowMeta = null;  // { flow_h, flow_w }
            this._propagating = false;
            this._propagateGen = 0;
            this.propagateWindow = 3;
            this._undoStack = []; // [Map snapshots of editedMasks]
            this._allKnownFiles = new Set(); // all filenames ever seen (never shrinks)

            // Prevent LiteGraph from rendering built-in image preview
            // (we have our own canvas preview). This stops node.imgs from
            // causing height explosion when mask editor sets it.
            const origDrawBg = this.onDrawBackground;
            this.onDrawBackground = function (ctx) {
                const savedImgs = this.imgs;
                this.imgs = null;
                origDrawBg?.call(this, ctx);
                this.imgs = savedImgs;
            };

            // ── Main container ─────────────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex; flex-direction: column; align-items: center;
                width: 100%; height: 100%; padding: 8px;
                box-sizing: border-box; overflow: hidden;
            `;

            // ── Preview area (canvas) ──────────────────────────────────────
            const imgContainer = document.createElement("div");
            imgContainer.style.cssText = `
                flex: 1; width: 100%; display: flex;
                align-items: center; justify-content: center;
                overflow: hidden; min-height: 50px;
            `;

            const canvas = document.createElement("canvas");
            canvas.style.cssText = `
                max-width: 100%; max-height: 100%;
                border-radius: 4px; background: #1a1a1a;
                object-fit: contain;
            `;
            canvas.addEventListener("click", () => {
                this._showMaskBW = !this._showMaskBW;
                this.updatePreview(this.currentFrame);
            });
            canvas.style.cursor = "pointer";
            imgContainer.appendChild(canvas);
            container.appendChild(imgContainer);
            this.previewCanvas = canvas;
            this.previewCtx = canvas.getContext("2d");
            this.imgContainer = imgContainer;

            // ── Slider row ─────────────────────────────────────────────────
            const sliderRow = document.createElement("div");
            sliderRow.style.cssText = `
                display: flex; align-items: center; width: 100%;
                margin-top: 8px; gap: 8px; flex-shrink: 0;
            `;

            // Frame number input
            const frameInput = document.createElement("input");
            frameInput.type = "number";
            frameInput.min = 0;
            frameInput.value = 0;
            frameInput.style.cssText = `
                width: 55px; padding: 4px 6px;
                border: 1px solid #555; border-radius: 3px;
                background: #2a2a2a; color: #fff; font-size: 12px;
            `;
            frameInput.addEventListener("change", () => {
                if (this.totalFrames > 0) {
                    const idx = Math.max(0, Math.min(this.totalFrames - 1, parseInt(frameInput.value) || 0));
                    this.updatePreview(idx);
                }
            });
            sliderRow.appendChild(frameInput);
            this.frameInput = frameInput;

            // Slider
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            slider.style.cssText = "flex: 1; cursor: pointer;";
            slider.addEventListener("input", () => {
                const idx = parseInt(slider.value);
                this.currentFrame = idx;
                this.frameInput.value = idx;
                this.updatePreview(idx);
            });
            sliderRow.appendChild(slider);
            this.slider = slider;

            // Total frames label
            const totalLabel = document.createElement("span");
            totalLabel.style.cssText = `
                font-size: 12px; color: #aaa;
                min-width: 40px; text-align: right;
            `;
            totalLabel.textContent = "/ -";
            sliderRow.appendChild(totalLabel);
            this.totalLabel = totalLabel;

            container.appendChild(sliderRow);

            // ── Controls row (edit, undo, reset, propagation window) ──────
            const controlsRow = document.createElement("div");
            controlsRow.style.cssText = `
                display: flex; align-items: center; width: 100%;
                margin-top: 4px; gap: 6px; flex-shrink: 0;
            `;
            const btnStyle = `
                padding: 3px 8px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #aaa;
                font-size: 12px; cursor: pointer; white-space: nowrap;
            `;

            // Edit Mask button
            const editBtn = document.createElement("button");
            editBtn.style.cssText = btnStyle;
            editBtn.textContent = "\u270E Edit";
            editBtn.title = "Edit mask for current frame";
            editBtn.addEventListener("click", () => this.openMaskEditor());
            controlsRow.appendChild(editBtn);
            this.editBtn = editBtn;

            // Propagation window (next to Edit)
            const propLabel = document.createElement("span");
            propLabel.style.cssText = "font-size: 11px; color: #888;";
            propLabel.textContent = "±";
            controlsRow.appendChild(propLabel);

            const propInput = document.createElement("input");
            propInput.type = "number";
            propInput.min = 1;
            propInput.max = 30;
            propInput.value = this.propagateWindow;
            propInput.title = "Propagation window (±N frames)";
            propInput.style.cssText = `
                width: 36px; padding: 2px 4px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #fff; font-size: 11px;
            `;
            propInput.addEventListener("input", () => {
                const val = parseInt(propInput.value);
                if (val >= 1 && val <= 30) this.propagateWindow = val;
            });
            propInput.addEventListener("blur", () => {
                this.propagateWindow = Math.max(1, Math.min(30, parseInt(propInput.value) || 3));
                propInput.value = this.propagateWindow;
            });
            controlsRow.appendChild(propInput);

            // Undo button
            const undoBtn = document.createElement("button");
            undoBtn.style.cssText = btnStyle;
            undoBtn.textContent = "\u21B6 Undo";
            undoBtn.title = "Undo last edit + propagation";
            undoBtn.addEventListener("click", () => this.undoEdit());
            controlsRow.appendChild(undoBtn);

            // Spacer
            const spacer = document.createElement("div");
            spacer.style.cssText = "flex: 1;";
            controlsRow.appendChild(spacer);

            // Reset button
            const resetBtn = document.createElement("button");
            resetBtn.style.cssText = btnStyle;
            resetBtn.textContent = "\u21BA Reset";
            resetBtn.title = "Reset all mask edits";
            resetBtn.addEventListener("click", () => this.resetEdits());
            controlsRow.appendChild(resetBtn);

            container.appendChild(controlsRow);

            // ── Progress bar + edited frames indicator ─────────────────────
            const statusRow = document.createElement("div");
            statusRow.style.cssText = `
                width: 100%; margin-top: 4px; min-height: 16px; flex-shrink: 0;
            `;

            const progressBar = document.createElement("div");
            progressBar.style.cssText = `
                width: 100%; height: 3px; background: #333;
                border-radius: 2px; overflow: hidden; display: none;
            `;
            const progressFill = document.createElement("div");
            progressFill.style.cssText = `
                height: 100%; width: 0%; background: #4CAF50;
                transition: width 0.15s ease;
            `;
            progressBar.appendChild(progressFill);
            statusRow.appendChild(progressBar);
            this.progressBar = progressBar;
            this.progressFill = progressFill;

            const editedRow = document.createElement("div");
            editedRow.style.cssText = `
                width: 100%; font-size: 11px; color: #888;
                overflow: hidden; white-space: nowrap;
                text-overflow: ellipsis;
            `;
            statusRow.appendChild(editedRow);
            this.editedRow = editedRow;

            container.appendChild(statusRow);

            this.container = container;

            // ── Register DOM widget ────────────────────────────────────────
            const widget = this.addDOMWidget("preview_slider", "preview", container, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = (width) => {
                const nodeHeight = this.size?.[1] || 340;
                return [width, Math.max(nodeHeight - 100, 100)];
            };
            this.previewWidget = widget;
            this.setSize([300, 340]);

            // Hide the edited_masks widget (data-only, synced by _syncEditedMasks)
            // Protect against external overwrites (e.g., mask editor's updateNodeWithServerReferences)
            const emWidget = this.widgets?.find(w => w.name === "edited_masks");
            if (emWidget) {
                emWidget.type = "converted-widget";
                emWidget.computeSize = () => [0, -4];
                emWidget.hidden = true;

                const self2 = this;
                let _emValue = emWidget.value;
                Object.defineProperty(emWidget, "value", {
                    get() { return _emValue; },
                    set(val) {
                        if (val === "{}" && self2.editedMasks.size > 0) {
                            return; // block external overwrite
                        }
                        _emValue = val;
                    },
                    configurable: true,
                    enumerable: true,
                });
            }

            // Restore editedMasks from widget
            this._restoreEditedMasks();
        };

        // ── onResize ───────────────────────────────────────────────────────
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            if (this.previewWidget) {
                this.previewWidget.computeSize(size[0]);
            }
        };

        // ── onExecuted ─────────────────────────────────────────────────────
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.frames && message.frames.length > 0) {
                // Fingerprint-based input change detection
                const fp = message.input_fingerprint?.[0];
                if (fp) {
                    this._cacheTs = fp;
                    if (this._lastFingerprint !== null && this._lastFingerprint !== fp) {
                        // Input changed — clear stale edits and caches
                        this._previewCache = null;
                        if (this.editedMasks.size > 0) {
                            this.editedMasks.clear();
                            this._syncEditedMasks();
                            this._updateEditedIndicator();
                        }
                    }
                    this._lastFingerprint = fp;
                }

                this.frames = message.frames;
                this.totalFrames = message.frames.length;
                this.hasImages = message.has_images?.[0] ?? false;
                this.hasMask = message.has_mask?.[0] ?? false;
                this.updateSlider();
                this.updatePreview(Math.min(this.currentFrame, this.totalFrames - 1));
                this._buildPreviewCache();

                // Load optical flow if present
                const flowData = message.optical_flow?.[0];
                if (flowData) {
                    this._flowMeta = { flow_h: flowData.flow_h, flow_w: flowData.flow_w };
                    this._preloadFlow(flowData);
                } else {
                    this._flowCache = null;
                    this._flowMeta = null;
                }
            }
        };

        // ── updateSlider ───────────────────────────────────────────────────
        nodeType.prototype.updateSlider = function () {
            const max = Math.max(0, this.totalFrames - 1);
            this.slider.max = max;
            this.frameInput.max = max;
            this.totalLabel.textContent = `/ ${max}`;
        };

        // ── updatePreview ──────────────────────────────────────────────────
        nodeType.prototype.updatePreview = function (frameIndex) {
            if (!this.frames || this.frames.length === 0) return;
            frameIndex = Math.max(0, Math.min(this.frames.length - 1, frameIndex));
            this.currentFrame = frameIndex;
            this.frameInput.value = frameIndex;
            this.slider.value = frameIndex;

            // Try cache first
            const cacheKey = this._showMaskBW ? "bw" : "main";
            const cache = this._previewCache?.[cacheKey];
            if (cache?.[frameIndex]) {
                this._drawToCanvas(cache[frameIndex]);
                return;
            }

            // Bump generation to discard any in-flight renders
            const gen = ++this._renderGen;

            // Determine which frame source to display
            const edited = this.editedMasks.has(frameIndex);
            const frame = edited ? this.editedMasks.get(frameIndex) : this.frames[frameIndex];
            const needsOverlay = this.hasMask || edited;

            if (this._showMaskBW && needsOverlay) {
                this._renderMaskOnly(frame, gen);
            } else if (needsOverlay) {
                this._renderOverlay(frame, frame, gen);
            } else if (this.hasImages) {
                this._renderSimple(frame, "rgb", gen);
            } else {
                this._renderMaskOnly(frame, gen);
            }
        };

        // ── _renderSimple: draw single channel to canvas ───────────────────
        nodeType.prototype._renderSimple = function (frame, channel, gen) {
            const url = viewURL(frame, channel, this._cacheTs);
            loadImage(url).then(img => {
                if (gen === this._renderGen) this._drawToCanvas(img);
            }).catch(() => {});
        };

        // ── _renderMaskOnly: show alpha channel as grayscale ───────────────
        nodeType.prototype._renderMaskOnly = function (frame, gen) {
            const url = viewURL(frame, "a", this._cacheTs);
            loadImage(url).then(img => {
                if (gen !== this._renderGen) return;
                const tmp = document.createElement("canvas");
                tmp.width = img.naturalWidth;
                tmp.height = img.naturalHeight;
                const ctx = tmp.getContext("2d");
                ctx.drawImage(img, 0, 0);
                const data = ctx.getImageData(0, 0, tmp.width, tmp.height);
                // Temp PNGs use mask editor convention (alpha=0 → masked/foreground).
                // Invert so foreground displays as bright in grayscale.
                for (let i = 0; i < data.data.length; i += 4) {
                    const a = 255 - data.data[i + 3];
                    data.data[i] = a;
                    data.data[i + 1] = a;
                    data.data[i + 2] = a;
                    data.data[i + 3] = 255;
                }
                ctx.putImageData(data, 0, 0);
                this._drawToCanvas(tmp);
            }).catch(() => {});
        };

        // ── _renderOverlay: RGB base + dark mask overlay ───────────────────
        nodeType.prototype._renderOverlay = function (rgbFrame, alphaFrame, gen) {
            const rgbURL = viewURL(rgbFrame, "rgb", this._cacheTs);
            const alphaURL = viewURL(alphaFrame, "a", this._cacheTs);

            Promise.all([loadImage(rgbURL), loadImage(alphaURL)]).then(([rgbImg, aImg]) => {
                if (gen !== this._renderGen) return;
                const w = rgbImg.naturalWidth;
                const h = rgbImg.naturalHeight;
                const tmp = document.createElement("canvas");
                tmp.width = w;
                tmp.height = h;
                const ctx = tmp.getContext("2d");

                // Draw RGB base
                ctx.drawImage(rgbImg, 0, 0);

                // Extract alpha data
                const aTmp = document.createElement("canvas");
                aTmp.width = w;
                aTmp.height = h;
                const aCtx = aTmp.getContext("2d");
                aCtx.drawImage(aImg, 0, 0);
                const aData = aCtx.getImageData(0, 0, w, h).data;

                // Create dark overlay where mask is present, matching
                // the mask editor's visual style (semi-transparent black).
                const overlay = ctx.createImageData(w, h);
                for (let i = 0; i < aData.length; i += 4) {
                    const a = aData[i + 3];
                    if (a < 255) {
                        overlay.data[i] = 0;
                        overlay.data[i + 1] = 0;
                        overlay.data[i + 2] = 0;
                        overlay.data[i + 3] = Math.round((255 - a) * 0.75);
                    }
                }

                // Composite overlay onto base
                const oTmp = document.createElement("canvas");
                oTmp.width = w;
                oTmp.height = h;
                const oCtx = oTmp.getContext("2d");
                oCtx.putImageData(overlay, 0, 0);
                ctx.drawImage(oTmp, 0, 0);

                this._drawToCanvas(tmp);
            }).catch(() => {});
        };

        // ── _drawToCanvas: scale source to preview canvas ──────────────────
        nodeType.prototype._drawToCanvas = function (source) {
            const sw = source.naturalWidth || source.width;
            const sh = source.naturalHeight || source.height;
            if (!sw || !sh) return;

            // Fit to container, accounting for device pixel ratio
            const container = this.imgContainer;
            const maxW = container.clientWidth || 280;
            const maxH = container.clientHeight || 200;
            const scale = Math.min(maxW / sw, maxH / sh, 1);
            const dw = Math.round(sw * scale);
            const dh = Math.round(sh * scale);
            const dpr = window.devicePixelRatio || 1;

            this.previewCanvas.width = Math.round(dw * dpr);
            this.previewCanvas.height = Math.round(dh * dpr);
            this.previewCanvas.style.width = dw + "px";
            this.previewCanvas.style.height = dh + "px";
            this.previewCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this.previewCtx.drawImage(source, 0, 0, dw, dh);
            this.previewCtx.setTransform(1, 0, 0, 1, 0, 0);
        };

        // ── Preview cache ─────────────────────────────────────────────────
        const MAX_CONCURRENT = 8;

        nodeType.prototype._buildPreviewCache = function () {
            this._previewCache = { main: new Array(this.totalFrames), bw: new Array(this.totalFrames) };
            this._cacheQueue = [];
            this._cacheLoading = 0;
            this._cacheTotal = this.totalFrames;
            this._cacheDone = 0;

            // Show progress bar
            if (this.progressBar) {
                this.progressBar.style.display = "block";
                this.progressFill.style.width = "0%";
            }

            // Current frame first, then by distance
            const order = [this.currentFrame];
            for (let d = 1; d < this.totalFrames; d++) {
                if (this.currentFrame + d < this.totalFrames) order.push(this.currentFrame + d);
                if (this.currentFrame - d >= 0) order.push(this.currentFrame - d);
            }
            this._cacheQueue = order;
            this._drainCacheQueue();
        };

        nodeType.prototype._drainCacheQueue = function () {
            const cache = this._previewCache;
            while (this._cacheLoading < MAX_CONCURRENT && this._cacheQueue.length > 0) {
                const idx = this._cacheQueue.shift();
                if (cache?.main[idx]) {
                    this._cacheDone++;
                    continue;
                }
                this._cacheLoading++;
                this._cacheFrame(idx).then(() => {
                    this._cacheLoading--;
                    this._cacheDone++;
                    // Update progress
                    if (this.progressFill && this._cacheTotal > 0) {
                        const pct = Math.round(this._cacheDone / this._cacheTotal * 100);
                        this.progressFill.style.width = pct + "%";
                        if (pct >= 100) {
                            setTimeout(() => { this.progressBar.style.display = "none"; }, 500);
                        }
                    }
                    // Yield to main thread so other nodes' onExecuted can run
                    setTimeout(() => this._drainCacheQueue(), 0);
                });
            }
        };

        nodeType.prototype._cacheFrame = function (frameIndex) {
            const cache = this._previewCache;
            if (!cache) return Promise.resolve();

            const edited = this.editedMasks.has(frameIndex);
            const frame = edited ? this.editedMasks.get(frameIndex) : this.frames[frameIndex];
            const needsOverlay = this.hasMask || edited;

            if (needsOverlay) {
                // Cache both modes: overlay (main) and B&W mask (bw)
                const rgbURL = viewURL(frame, "rgb", this._cacheTs);
                const alphaURL = viewURL(frame, "a", this._cacheTs);

                return Promise.all([loadImage(rgbURL), loadImage(alphaURL)]).then(([rgbImg, aImg]) => {
                    const w = rgbImg.naturalWidth;
                    const h = rgbImg.naturalHeight;

                    // Extract alpha data once
                    const aTmp = document.createElement("canvas");
                    aTmp.width = w;
                    aTmp.height = h;
                    const aCtx = aTmp.getContext("2d");
                    aCtx.drawImage(aImg, 0, 0);
                    const aData = aCtx.getImageData(0, 0, w, h).data;

                    // Main mode: RGB + dark overlay
                    const mainCanvas = document.createElement("canvas");
                    mainCanvas.width = w;
                    mainCanvas.height = h;
                    const mainCtx = mainCanvas.getContext("2d");
                    mainCtx.drawImage(rgbImg, 0, 0);
                    const overlay = mainCtx.createImageData(w, h);
                    for (let i = 0; i < aData.length; i += 4) {
                        const a = aData[i + 3];
                        if (a < 255) {
                            overlay.data[i + 3] = Math.round((255 - a) * 0.75);
                        }
                    }
                    const oTmp = document.createElement("canvas");
                    oTmp.width = w;
                    oTmp.height = h;
                    oTmp.getContext("2d").putImageData(overlay, 0, 0);
                    mainCtx.drawImage(oTmp, 0, 0);
                    if (cache.main) cache.main[frameIndex] = mainCanvas;

                    // BW mode: inverted alpha as grayscale
                    const bwCanvas = document.createElement("canvas");
                    bwCanvas.width = w;
                    bwCanvas.height = h;
                    const bwCtx = bwCanvas.getContext("2d");
                    const bwData = bwCtx.createImageData(w, h);
                    for (let i = 0; i < aData.length; i += 4) {
                        const v = 255 - aData[i + 3];
                        bwData.data[i] = v;
                        bwData.data[i + 1] = v;
                        bwData.data[i + 2] = v;
                        bwData.data[i + 3] = 255;
                    }
                    bwCtx.putImageData(bwData, 0, 0);
                    if (cache.bw) cache.bw[frameIndex] = bwCanvas;

                    // Refresh display if this is the current frame
                    if (frameIndex === this.currentFrame) this.updatePreview(frameIndex);
                }).catch(() => {});
            } else if (this.hasImages) {
                // Simple RGB
                return loadImage(viewURL(frame, "rgb", this._cacheTs)).then(img => {
                    const c = document.createElement("canvas");
                    c.width = img.naturalWidth;
                    c.height = img.naturalHeight;
                    c.getContext("2d").drawImage(img, 0, 0);
                    if (cache.main) cache.main[frameIndex] = c;
                    if (cache.bw) cache.bw[frameIndex] = c;
                    if (frameIndex === this.currentFrame) this.updatePreview(frameIndex);
                }).catch(() => {});
            } else {
                // Mask only (no RGB)
                return loadImage(viewURL(frame, "a", this._cacheTs)).then(img => {
                    const w = img.naturalWidth;
                    const h = img.naturalHeight;
                    const c = document.createElement("canvas");
                    c.width = w;
                    c.height = h;
                    const ctx = c.getContext("2d");
                    ctx.drawImage(img, 0, 0);
                    const data = ctx.getImageData(0, 0, w, h);
                    for (let i = 0; i < data.data.length; i += 4) {
                        const v = 255 - data.data[i + 3];
                        data.data[i] = v;
                        data.data[i + 1] = v;
                        data.data[i + 2] = v;
                        data.data[i + 3] = 255;
                    }
                    ctx.putImageData(data, 0, 0);
                    if (cache.main) cache.main[frameIndex] = c;
                    if (cache.bw) cache.bw[frameIndex] = c;
                    if (frameIndex === this.currentFrame) this.updatePreview(frameIndex);
                }).catch(() => {});
            }
        };

        // ── openMaskEditor ─────────────────────────────────────────────────
        nodeType.prototype.openMaskEditor = async function () {
            if (!this.frames || this.frames.length === 0) return;

            const editFrame = this.currentFrame;
            const self = this;

            // Load current state of the frame (could be original or edited)
            const source = this.editedMasks.has(editFrame)
                ? this.editedMasks.get(editFrame)
                : this.frames[editFrame];

            loadImage(viewURL(source, null, this._cacheTs)).then(img => {
                // Clean up any existing property trap from a previous edit
                if (Object.getOwnPropertyDescriptor(this, "images")?.set) {
                    delete this.images;
                }

                // Set node.imgs/images so isImageNode(node) returns true
                // onDrawBackground override prevents LiteGraph from resizing
                this.imgs = [img];
                this.images = [source];
                this.imageIndex = 0;

                // Trap on images: mask editor sets node.images = [ref] on save.
                // Build a set of all known filenames — only a genuinely NEW file
                // (not seen before) counts as a save. This prevents false triggers
                // when the editor writes stale clipspace data on cancel/init.
                // The trap stays active (never deleted) to absorb all writes safely.
                const knownFiles = new Set(this._allKnownFiles);
                for (const f of this.frames) knownFiles.add(f.filename);
                for (const v of this.editedMasks.values()) knownFiles.add(v.filename);

                let _images = this.images;
                let _handled = false;
                Object.defineProperty(this, "images", {
                    get() { return _images; },
                    set(val) {
                        _images = val;
                        const isNew = val?.[0]?.filename && !knownFiles.has(val[0].filename);
                        if (!_handled && isNew) {
                            _handled = true;
                            knownFiles.add(val[0].filename);
                            self._handleMaskSave(editFrame, val[0]);
                        }
                    },
                    configurable: true,
                    enumerable: true,
                });

                // Temporarily block ComfyUI's undo/redo while mask editor is open.
                // The changeTracker's keydown listener fires before any stopPropagation
                // can catch it, so we intercept at the loadGraphData level instead.
                const origLoadGraphData = app.loadGraphData?.bind(app);
                if (origLoadGraphData) {
                    app.loadGraphData = function (...args) {
                        // Allow normal loads (e.g. from file), block undo-triggered loads
                        // Undo calls come through changeTracker → loadGraphData synchronously
                        // after the keydown event. We block ALL calls while editor is open.
                        console.warn("[PreviewSlider] loadGraphData blocked while mask editor open");
                    };
                }

                // Watch for editor close to restore loadGraphData
                const restoreOnClose = () => {
                    const check = () => {
                        const editorOpen = document.querySelector(
                            ".comfy-modal-content canvas, .mask-editor, [class*='mask-editor']"
                        );
                        if (!editorOpen) {
                            if (origLoadGraphData) app.loadGraphData = origLoadGraphData;
                            return;
                        }
                        requestAnimationFrame(check);
                    };
                    // Start checking after a short delay (editor needs time to mount)
                    setTimeout(check, 500);
                };
                restoreOnClose();

                // Open mask editor via context menu callback
                const options = [];
                this.getExtraMenuOptions?.(app.canvas, options);
                const opt = options.find(o => o?.content?.includes("MaskEditor"));
                if (opt) {
                    opt.callback();
                } else {
                    if (origLoadGraphData) app.loadGraphData = origLoadGraphData;
                    console.warn("[PreviewSlider] MaskEditor not found in context menu");
                }
            }).catch(err => {
                console.error("[PreviewSlider] Failed to load frame:", err);
            });
        };

        // ── _handleMaskSave ────────────────────────────────────────────────
        nodeType.prototype._handleMaskSave = function (frameIndex, ref) {
            // ref = {filename, subfolder, type} from mask editor save
            if (!ref || !ref.filename) return;

            // Snapshot entire editedMasks for undo (deep copy values)
            const snapshot = new Map();
            for (const [k, v] of this.editedMasks) snapshot.set(k, { ...v });
            this._undoStack.push(snapshot);

            const newRef = {
                filename: ref.filename,
                subfolder: ref.subfolder || "",
                type: ref.type || "input",
            };
            this._allKnownFiles.add(ref.filename);
            this.editedMasks.set(frameIndex, newRef);
            this._syncEditedMasks();
            this._updateEditedIndicator();

            // Invalidate + rebuild cache for this frame
            if (this._previewCache) {
                if (this._previewCache.main) this._previewCache.main[frameIndex] = null;
                if (this._previewCache.bw) this._previewCache.bw[frameIndex] = null;
                this._cacheFrame(frameIndex);
            }
            this.updatePreview(frameIndex);

            // Propagate edit to neighboring frames (appends to the same undo entry)
            if (this._flowCache) {
                this._propagateEdit(frameIndex, ref);
            }
        };

        // ── _syncEditedMasks: serialize to hidden widget ───────────────────
        nodeType.prototype._syncEditedMasks = function () {
            const obj = Object.fromEntries(this.editedMasks);
            const jsonStr = JSON.stringify(obj);
            const widget = this.widgets?.find(w => w.name === "edited_masks");
            if (widget) {
                widget.value = jsonStr;
            }
        };

        // ── _restoreEditedMasks: load from hidden widget on node creation ──
        nodeType.prototype._restoreEditedMasks = function () {
            const widget = this.widgets?.find(w => w.name === "edited_masks");
            if (widget && widget.value && widget.value !== "{}") {
                try {
                    const obj = JSON.parse(widget.value);
                    for (const [k, v] of Object.entries(obj)) {
                        this.editedMasks.set(parseInt(k), v);
                        if (v.filename) this._allKnownFiles.add(v.filename);
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }
        };

        // ── _updateEditedIndicator ─────────────────────────────────────────
        nodeType.prototype._updateEditedIndicator = function () {
            if (!this.editedRow) return;
            if (this.editedMasks.size === 0) {
                this.editedRow.textContent = "";
                return;
            }
            const indices = [...this.editedMasks.keys()].sort((a, b) => a - b);
            this.editedRow.textContent = `Modified: ${indices.join(", ")} (${indices.length})`;
        };

        // ── resetEdits ────────────────────────────────────────────────────
        nodeType.prototype.resetEdits = function () {
            if (this.editedMasks.size === 0) return;
            this.editedMasks.clear();
            this._undoStack = [];
            this._syncEditedMasks();
            this._updateEditedIndicator();
            this._buildPreviewCache();
            this.updatePreview(this.currentFrame);
        };

        // ── undoEdit ──────────────────────────────────────────────────────
        nodeType.prototype.undoEdit = function () {
            if (this._undoStack.length === 0) return;

            // Cancel any in-progress propagation
            this._propagateGen++;
            this._propagating = false;

            const prev = this._undoStack.pop();

            // Find frames that changed between current and previous
            const allKeys = new Set([...this.editedMasks.keys(), ...prev.keys()]);
            const changed = [];
            for (const k of allKeys) {
                const cur = this.editedMasks.get(k);
                const old = prev.get(k);
                if (cur?.filename !== old?.filename) changed.push(k);
            }

            // Restore
            this.editedMasks = prev;

            // Invalidate cache for changed frames
            for (const idx of changed) {
                if (this._previewCache) {
                    if (this._previewCache.main) this._previewCache.main[idx] = null;
                    if (this._previewCache.bw) this._previewCache.bw[idx] = null;
                    this._cacheFrame(idx);
                }
            }

            this._syncEditedMasks();
            this._updateEditedIndicator();
            this.updatePreview(this.currentFrame);
        };

        // ── Optical flow loading ──────────────────────────────────────────
        nodeType.prototype._preloadFlow = function (flowData) {
            this._flowCache = null;
            const fwdList = flowData.fwd || [];
            const bwdList = flowData.bwd || [];
            const N = fwdList.length;
            if (N === 0) return;

            const flowH = flowData.flow_h;
            const flowW = flowData.flow_w;
            const cache = { fwd: new Array(N), bwd: new Array(N) };
            let pending = 0;
            const total = N * 2;

            const canvas = document.createElement("canvas");
            canvas.width = flowW;
            canvas.height = flowH;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });

            const processImg = (img, direction, idx) => {
                ctx.clearRect(0, 0, flowW, flowH);
                ctx.drawImage(img, 0, 0, flowW, flowH);
                const data = ctx.getImageData(0, 0, flowW, flowH).data;
                const dx = new Float32Array(flowH * flowW);
                const dy = new Float32Array(flowH * flowW);
                for (let p = 0; p < dx.length; p++) {
                    dx[p] = data[p * 4] - 128;
                    dy[p] = data[p * 4 + 1] - 128;
                }
                cache[direction][idx] = { dx, dy };
                pending++;
                if (pending === total) {
                    this._flowCache = cache;
                }
            };

            for (let i = 0; i < N; i++) {
                for (const [direction, list] of [["fwd", fwdList], ["bwd", bwdList]]) {
                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    const dir = direction, idx = i;
                    img.onload = () => processImg(img, dir, idx);
                    img.src = viewURL(list[i], null, this._cacheTs);
                }
            }
        };

        // ── Edit propagation via optical flow ─────────────────────────────
        //
        // Propagates the edited mask to ±WINDOW neighboring frames using absolute
        // values. A region mask gates which pixels are affected. Target frames
        // receive the warped edited mask values directly in the edit region.

        nodeType.prototype._propagateEdit = async function (editFrame, editedRef) {
            if (this._propagating) return;
            this._propagating = true;
            const gen = ++this._propagateGen;

            try {
                // Determine target frames in window
                const targets = [];
                for (let d = 1; d <= this.propagateWindow; d++) {
                    if (editFrame + d < this.totalFrames) targets.push(editFrame + d);
                    if (editFrame - d >= 0) targets.push(editFrame - d);
                }
                if (targets.length === 0) { this._propagating = false; return; }

                // Progress: 1 (load edit data) + targets (warp/upload) + cache rebuilds
                let totalSteps = 1 + targets.length;
                let pendingCacheRebuilds = 0;
                let doneSteps = 0;
                const updateProgress = () => {
                    if (this.progressBar && this.progressFill) {
                        this.progressBar.style.display = "block";
                        this.progressFill.style.width = Math.round(doneSteps / totalSteps * 100) + "%";
                    }
                };
                updateProgress();

                // Load edited mask and original mask at edit frame
                const [editedPx, originalEditPx] = await Promise.all([
                    this._loadMaskPixels(editedRef),
                    this._loadMaskPixels(this.frames[editFrame]),
                ]);
                if (!editedPx || !originalEditPx) { this._propagating = false; return; }

                const maskW = editedPx.w;
                const maskH = editedPx.h;

                // Region mask: which pixels were edited (boolean, used as gate)
                const region = new Float32Array(maskW * maskH);
                for (let i = 0; i < region.length; i++) {
                    region[i] = Math.abs(editedPx[i] - originalEditPx[i]) > PROPAGATE_DIFF_THRESHOLD ? 1 : 0;
                }
                doneSteps++;
                updateProgress();

                // Propagate to each target frame
                for (const targetFrame of targets) {
                    if (gen !== this._propagateGen) break;
                    // Warp both region mask and absolute edited mask
                    const warpedRegion = this._warpDiffChain(region, maskW, maskH, editFrame, targetFrame);
                    const warpedMask = this._warpDiffChain(editedPx, maskW, maskH, editFrame, targetFrame);
                    if (!warpedRegion || !warpedMask) { doneSteps++; updateProgress(); continue; }

                    // Load target's current state
                    const currentRef = this.editedMasks.has(targetFrame)
                        ? this.editedMasks.get(targetFrame)
                        : this.frames[targetFrame];
                    const currentPx = await this._loadMaskPixels(currentRef);
                    if (gen !== this._propagateGen) break;
                    if (!currentPx) { doneSteps++; updateProgress(); continue; }

                    // Apply: in warped region, directly use warped absolute value
                    const finalMask = new Float32Array(maskW * maskH);
                    let changedCount = 0;
                    for (let i = 0; i < finalMask.length; i++) {
                        if (warpedRegion[i] > 0.5) {
                            finalMask[i] = warpedMask[i];
                        } else {
                            finalMask[i] = currentPx[i];
                        }
                        if (Math.abs(finalMask[i] - currentPx[i]) > 0.01) changedCount++;
                    }
                    if (changedCount === 0) { doneSteps++; updateProgress(); continue; }

                    // Upload
                    const ref = await this._uploadPropagatedMask(finalMask, maskW, maskH, targetFrame);
                    if (gen !== this._propagateGen) break;
                    if (ref) {
                        this._allKnownFiles.add(ref.filename);
                        this.editedMasks.set(targetFrame, ref);
                        if (this._previewCache) {
                            if (this._previewCache.main) this._previewCache.main[targetFrame] = null;
                            if (this._previewCache.bw) this._previewCache.bw[targetFrame] = null;
                            totalSteps++;
                            pendingCacheRebuilds++;
                            this._cacheFrame(targetFrame).then(() => {
                                doneSteps++;
                                pendingCacheRebuilds--;
                                updateProgress();
                                if (pendingCacheRebuilds === 0 && this.progressBar) {
                                    setTimeout(() => { this.progressBar.style.display = "none"; }, 500);
                                }
                            });
                        }
                    }
                    doneSteps++;
                    updateProgress();
                }

                this._syncEditedMasks();
                this._updateEditedIndicator();
                this.updatePreview(this.currentFrame);

                if (pendingCacheRebuilds === 0 && this.progressBar) {
                    setTimeout(() => { this.progressBar.style.display = "none"; }, 500);
                }
            } finally {
                this._propagating = false;
            }
        };

        // Load mask pixels from a frame ref as Float32Array [0,1] (foreground=1)
        // Uses non-premultiplied alpha to avoid precision loss on transparent pixels.
        nodeType.prototype._loadMaskPixels = async function (frameRef) {
            try {
                // Load full RGBA (not channel=a) to get original alpha without server re-encoding
                const url = viewURL(frameRef, null, this._cacheTs);
                const resp = await fetch(url);
                const blob = await resp.blob();
                // createImageBitmap with premultiplyAlpha:"none" preserves exact alpha values
                const bmp = await createImageBitmap(blob, { premultiplyAlpha: "none" });
                const w = bmp.width;
                const h = bmp.height;
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                const ctx = c.getContext("2d", { willReadFrequently: true });
                ctx.drawImage(bmp, 0, 0);
                const data = ctx.getImageData(0, 0, w, h).data;
                bmp.close();
                // Mask editor convention: alpha=0 → foreground, alpha=255 → background
                const pixels = new Float32Array(w * h);
                for (let i = 0; i < pixels.length; i++) {
                    pixels[i] = 1.0 - data[i * 4 + 3] / 255.0;
                }
                pixels.w = w;
                pixels.h = h;
                return pixels;
            } catch {
                return null;
            }
        };

        // Warp diff from editFrame to targetFrame by chaining per-frame flows.
        // Returns warped diff Float32Array or null.
        nodeType.prototype._warpDiffChain = function (diff, w, h, editFrame, targetFrame) {
            if (!this._flowCache || !this._flowMeta) return null;

            const flowH = this._flowMeta.flow_h;
            const flowW = this._flowMeta.flow_w;
            let current = diff;

            const step = targetFrame > editFrame ? 1 : -1;
            // Warp does backward lookup (dst→src), so directions are inverted:
            // propagate forward (5→6): use bwd flow (frame 6 looks up frame 5)
            // propagate backward (10→9): use fwd flow (frame 9 looks up frame 10)
            const warpDir = step > 0 ? "bwd" : "fwd";

            for (let f = editFrame; f !== targetFrame; f += step) {
                const flowIdx = step > 0 ? f : f - 1;
                const flow = this._flowCache[warpDir]?.[flowIdx];
                if (!flow) return null;
                current = this._warpDiff(current, w, h, flow, flowH, flowW);
            }
            return current;
        };

        // Warp a diff array using a single flow field {dx, dy}
        nodeType.prototype._warpDiff = function (diff, w, h, flow, flowH, flowW) {
            const result = new Float32Array(w * h);
            const scaleX = flowW / w;
            const scaleY = flowH / h;

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    // Map to flow coordinates
                    const fx = x * scaleX;
                    const fy = y * scaleY;
                    const fi = Math.min(Math.floor(fy), flowH - 1) * flowW + Math.min(Math.floor(fx), flowW - 1);

                    // Source position
                    const sx = x + flow.dx[fi] / scaleX;
                    const sy = y + flow.dy[fi] / scaleY;

                    // Bilinear interpolation
                    const sx0 = Math.floor(sx), sy0 = Math.floor(sy);
                    const sx1 = sx0 + 1, sy1 = sy0 + 1;
                    if (sx0 >= 0 && sx1 < w && sy0 >= 0 && sy1 < h) {
                        const fx2 = sx - sx0, fy2 = sy - sy0;
                        result[y * w + x] =
                            diff[sy0 * w + sx0] * (1 - fx2) * (1 - fy2) +
                            diff[sy0 * w + sx1] * fx2 * (1 - fy2) +
                            diff[sy1 * w + sx0] * (1 - fx2) * fy2 +
                            diff[sy1 * w + sx1] * fx2 * fy2;
                    }
                }
            }
            return result;
        };

        // Upload propagated mask as RGBA PNG (alpha only, RGB=0) to ComfyUI
        nodeType.prototype._uploadPropagatedMask = async function (maskData, w, h, frameIndex) {
            try {
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                const ctx = c.getContext("2d");
                const imgData = ctx.createImageData(w, h);
                for (let i = 0; i < maskData.length; i++) {
                    imgData.data[i * 4 + 3] = Math.round((1 - maskData[i]) * 255);
                }
                ctx.putImageData(imgData, 0, 0);

                const pngBlob = await new Promise(resolve => c.toBlob(resolve, "image/png"));
                const filename = `propagated-mask-${frameIndex}-${performance.now()}.png`;
                // Include original_ref so server combines original RGB + propagated alpha
                const originalFrame = this.frames[frameIndex];
                const originalRef = {
                    filename: originalFrame.filename,
                    subfolder: originalFrame.subfolder || "",
                    type: originalFrame.type || "temp",
                };
                const formData = new FormData();
                formData.append("image", pngBlob, filename);
                formData.append("original_ref", JSON.stringify(originalRef));
                formData.append("type", "input");
                formData.append("subfolder", "clipspace");
                await api.fetchApi("/upload/mask", { method: "POST", body: formData });
                return { filename, subfolder: "clipspace", type: "input" };
            } catch (err) {
                console.error("[PreviewSlider] Failed to upload propagated mask:", err);
                return null;
            }
        };
    },
});
