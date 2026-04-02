import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── Helpers ────────────────────────────────────────────────────────────────────

function viewURL(frame, channel) {
    const params = new URLSearchParams({
        filename: frame.filename,
        subfolder: frame.subfolder || "",
        type: frame.type || "temp",
    });
    if (channel) params.set("channel", channel);
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

            // Edit Mask button
            const editBtn = document.createElement("button");
            editBtn.style.cssText = `
                padding: 3px 8px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #aaa;
                font-size: 12px; cursor: pointer; white-space: nowrap;
            `;
            editBtn.textContent = "\u270E";
            editBtn.title = "Edit mask for current frame";
            editBtn.addEventListener("click", () => this.openMaskEditor());
            sliderRow.appendChild(editBtn);
            this.editBtn = editBtn;

            // Reset button
            const resetBtn = document.createElement("button");
            resetBtn.style.cssText = `
                padding: 3px 8px; border: 1px solid #555;
                border-radius: 3px; background: #2a2a2a; color: #aaa;
                font-size: 12px; cursor: pointer; white-space: nowrap;
            `;
            resetBtn.textContent = "\u21BA";
            resetBtn.title = "Reset all mask edits";
            resetBtn.addEventListener("click", () => this.resetEdits());
            sliderRow.appendChild(resetBtn);

            container.appendChild(sliderRow);

            // ── Edited frames indicator ────────────────────────────────────
            const editedRow = document.createElement("div");
            editedRow.style.cssText = `
                width: 100%; font-size: 11px; color: #888;
                margin-top: 4px; min-height: 16px; flex-shrink: 0;
                overflow: hidden; white-space: nowrap;
                text-overflow: ellipsis;
            `;
            container.appendChild(editedRow);
            this.editedRow = editedRow;

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
            const emWidget = this.widgets?.find(w => w.name === "edited_masks");
            if (emWidget) {
                emWidget.type = "converted-widget";
                emWidget.computeSize = () => [0, -4];
                emWidget.hidden = true;
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
                this.frames = message.frames;
                this.totalFrames = message.frames.length;
                this.hasImages = message.has_images?.[0] ?? false;
                this.hasMask = message.has_mask?.[0] ?? false;
                this.updateSlider();
                this.updatePreview(Math.min(this.currentFrame, this.totalFrames - 1));
                this._buildPreviewCache();
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
                // B&W mask view (click canvas to toggle)
                this._renderMaskOnly(frame, gen);
            } else if (needsOverlay) {
                this._renderOverlay(frame, edited, gen);
            } else if (this.hasImages) {
                this._renderSimple(frame, "rgb", gen);
            } else {
                this._renderMaskOnly(frame, gen);
            }
        };

        // ── _renderSimple: draw single channel to canvas ───────────────────
        nodeType.prototype._renderSimple = function (frame, channel, gen) {
            const url = viewURL(frame, channel);
            loadImage(url).then(img => {
                if (gen === this._renderGen) this._drawToCanvas(img);
            }).catch(() => {});
        };

        // ── _renderMaskOnly: show alpha channel as grayscale ───────────────
        nodeType.prototype._renderMaskOnly = function (frame, gen) {
            const url = viewURL(frame, "a");
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
        nodeType.prototype._renderOverlay = function (frame, isEdited, gen) {
            // For edited frames, the file itself is RGBA (from mask editor save)
            // For original frames with mask, the backend saved RGBA
            const rgbURL = viewURL(frame, "rgb");
            const alphaURL = viewURL(frame, "a");

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
        nodeType.prototype._buildPreviewCache = function () {
            this._previewCache = { main: new Array(this.totalFrames), bw: new Array(this.totalFrames) };
            for (let i = 0; i < this.totalFrames; i++) {
                this._cacheFrame(i);
            }
        };

        nodeType.prototype._cacheFrame = function (frameIndex) {
            const cache = this._previewCache;
            if (!cache) return;

            const edited = this.editedMasks.has(frameIndex);
            const frame = edited ? this.editedMasks.get(frameIndex) : this.frames[frameIndex];
            const needsOverlay = this.hasMask || edited;

            if (needsOverlay) {
                // Cache both modes: overlay (main) and B&W mask (bw)
                const rgbURL = viewURL(frame, "rgb");
                const alphaURL = viewURL(frame, "a");

                Promise.all([loadImage(rgbURL), loadImage(alphaURL)]).then(([rgbImg, aImg]) => {
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
                loadImage(viewURL(frame, "rgb")).then(img => {
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
                loadImage(viewURL(frame, "a")).then(img => {
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
        nodeType.prototype.openMaskEditor = function () {
            if (!this.frames || this.frames.length === 0) return;

            const editFrame = this.currentFrame;
            const self = this;

            // If edited before, load edited version (continues editing)
            const source = this.editedMasks.has(editFrame)
                ? this.editedMasks.get(editFrame)
                : this.frames[editFrame];

            loadImage(viewURL(source)).then(img => {
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
                // The editor may also write to node.images during init
                // (e.g. normalizing type), so only treat writes with a NEW
                // filename as actual saves.
                let _images = this.images;
                const sourceFilename = source.filename;
                Object.defineProperty(this, "images", {
                    get() { return _images; },
                    set(val) {
                        _images = val;
                        if (val?.[0]?.filename && val[0].filename !== sourceFilename) {
                            self._handleMaskSave(editFrame, val[0]);
                            delete self.images;
                            self.images = val;
                        }
                    },
                    configurable: true,
                    enumerable: true,
                });

                // Open mask editor via context menu callback
                const options = [];
                this.getExtraMenuOptions?.(app.canvas, options);
                const opt = options.find(o => o?.content?.includes("MaskEditor"));
                if (opt) {
                    opt.callback();
                } else {
                    delete this.images;
                    this.images = null;
                    this.imgs = null;
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

            this.editedMasks.set(frameIndex, {
                filename: ref.filename,
                subfolder: ref.subfolder || "",
                type: ref.type || "input",
            });
            this._syncEditedMasks();
            this._updateEditedIndicator();

            // Invalidate + rebuild cache for this frame
            if (this._previewCache) {
                if (this._previewCache.main) this._previewCache.main[frameIndex] = null;
                if (this._previewCache.bw) this._previewCache.bw[frameIndex] = null;
                this._cacheFrame(frameIndex);
            }
            this.updatePreview(frameIndex);
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
            this.editedRow.textContent = `Edited: ${indices.join(", ")}`;
        };

        // ── resetEdits ────────────────────────────────────────────────────
        nodeType.prototype.resetEdits = function () {
            if (this.editedMasks.size === 0) return;
            this.editedMasks.clear();
            this._syncEditedMasks();
            this._updateEditedIndicator();
            this._buildPreviewCache();
            this.updatePreview(this.currentFrame);
        };
    },
});
