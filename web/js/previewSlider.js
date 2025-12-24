import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Register extension for PreviewSlider node
app.registerExtension({
    name: "VideoMatting.PreviewSlider",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PreviewSlider") {
            return;
        }

        // Override onNodeCreated to add custom widgets
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            this.frames = [];
            this.totalFrames = 0;
            this.currentFrame = 0;

            // Create main container
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
                height: 100%;
                padding: 8px;
                box-sizing: border-box;
                overflow: hidden;
            `;

            // Create image container (flexible)
            const imgContainer = document.createElement("div");
            imgContainer.style.cssText = `
                flex: 1;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                min-height: 50px;
            `;

            // Create image element
            const img = document.createElement("img");
            img.style.cssText = `
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                border-radius: 4px;
                background: #1a1a1a;
            `;
            img.src = "";
            imgContainer.appendChild(img);
            container.appendChild(imgContainer);
            this.previewImg = img;
            this.imgContainer = imgContainer;

            // Create slider container (fixed height)
            const sliderContainer = document.createElement("div");
            sliderContainer.style.cssText = `
                display: flex;
                align-items: center;
                width: 100%;
                margin-top: 8px;
                gap: 8px;
                flex-shrink: 0;
            `;

            // Frame number input
            const frameInput = document.createElement("input");
            frameInput.type = "number";
            frameInput.min = 0;
            frameInput.value = 0;
            frameInput.style.cssText = `
                width: 55px;
                padding: 4px 6px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #2a2a2a;
                color: #fff;
                font-size: 12px;
            `;
            frameInput.addEventListener("change", () => {
                if (this.totalFrames > 0) {
                    const idx = Math.max(0, Math.min(this.totalFrames - 1, parseInt(frameInput.value) || 0));
                    this.updatePreview(idx);
                }
            });
            sliderContainer.appendChild(frameInput);
            this.frameInput = frameInput;

            // Slider
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            slider.style.cssText = "flex: 1; cursor: pointer;";
            slider.addEventListener("input", () => {
                this.updatePreview(parseInt(slider.value));
            });
            sliderContainer.appendChild(slider);
            this.slider = slider;

            // Total frames label
            const totalLabel = document.createElement("span");
            totalLabel.style.cssText = `
                font-size: 12px;
                color: #aaa;
                min-width: 50px;
                text-align: right;
            `;
            totalLabel.textContent = "/ -";
            sliderContainer.appendChild(totalLabel);
            this.totalLabel = totalLabel;

            container.appendChild(sliderContainer);
            this.container = container;

            // Add as DOM widget
            const widget = this.addDOMWidget("preview_slider", "preview", container, {
                serialize: false,
                hideOnZoom: false,
            });

            // Compute size based on node width
            widget.computeSize = (width) => {
                const nodeHeight = this.size?.[1] || 340;
                // Reserve space for title, inputs, and slider controls
                const reservedHeight = 100;
                return [width, Math.max(nodeHeight - reservedHeight, 100)];
            };

            this.previewWidget = widget;
            this.setSize([300, 340]);
        };

        // Handle resize
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            // Force widget to recalculate
            if (this.previewWidget) {
                this.previewWidget.computeSize(size[0]);
            }
        };

        // Override onExecuted to handle frame data
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            // Get frames from message
            if (message?.frames && message.frames.length > 0) {
                this.frames = message.frames;
                this.totalFrames = message.frames.length;
                this.updateSlider();
                this.updatePreview(0);
            }
        };

        // Add method to update preview image
        nodeType.prototype.updatePreview = function (frameIndex) {
            if (!this.frames || this.frames.length === 0) {
                this.previewImg.src = "";
                return;
            }

            frameIndex = Math.max(0, Math.min(this.frames.length - 1, frameIndex));
            this.currentFrame = frameIndex;

            const frame = this.frames[frameIndex];
            const params = new URLSearchParams({
                filename: frame.filename,
                subfolder: frame.subfolder || "",
                type: frame.type || "temp"
            });
            const url = api.apiURL(`/view?${params.toString()}`);
            this.previewImg.src = url;

            // Update controls
            this.frameInput.value = frameIndex;
            this.slider.value = frameIndex;
        };

        // Add method to update slider range
        nodeType.prototype.updateSlider = function () {
            const max = Math.max(0, this.totalFrames - 1);
            this.slider.max = max;
            this.frameInput.max = max;
            this.totalLabel.textContent = `/ ${max}`;
        };
    },
});
