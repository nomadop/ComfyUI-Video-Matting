import { app } from "../../../scripts/app.js";

// Register extension for ImageSequencePackager node
app.registerExtension({
    name: "VideoMatting.ImageSequencePackager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageSequencePackager") {
            return;
        }

        // Override onNodeCreated to add custom widgets
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            this.downloadUrl = null;
            this.zipFilename = null;

            // Create container for download button
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
                gap: 8px;
            `;

            // Status text
            const statusText = document.createElement("div");
            statusText.style.cssText = `
                font-size: 12px;
                color: #aaa;
                text-align: center;
                word-break: break-all;
            `;
            statusText.textContent = "Run node to pack images";
            container.appendChild(statusText);
            this.statusText = statusText;

            // Download button
            const downloadBtn = document.createElement("button");
            downloadBtn.textContent = "📦 Download ZIP";
            downloadBtn.style.cssText = `
                padding: 8px 16px;
                background: #4a9eff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                display: none;
                transition: background 0.2s;
            `;
            downloadBtn.addEventListener("mouseenter", () => {
                downloadBtn.style.background = "#3a8eef";
            });
            downloadBtn.addEventListener("mouseleave", () => {
                downloadBtn.style.background = "#4a9eff";
            });
            downloadBtn.addEventListener("click", () => {
                if (this.downloadUrl) {
                    // Create temporary link and trigger download
                    const a = document.createElement("a");
                    a.href = this.downloadUrl;
                    a.download = this.zipFilename || "download.zip";
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            });
            container.appendChild(downloadBtn);
            this.downloadBtn = downloadBtn;

            // Add as DOM widget
            const widget = this.addDOMWidget("download_panel", "download", container, {
                serialize: false,
                hideOnZoom: false,
            });

            widget.computeSize = (width) => {
                return [width, 70];
            };

            this.downloadWidget = widget;
        };

        // Override onExecuted to handle download URL
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.text && message.text.length >= 2) {
                // Extract info from text array
                const infoText = message.text[0];
                const urlText = message.text[1];

                // Parse download URL
                if (urlText && urlText.startsWith("Download: ")) {
                    this.downloadUrl = urlText.replace("Download: ", "");

                    // Extract filename from URL
                    const match = this.downloadUrl.match(/filename=([^&]+)/);
                    if (match) {
                        this.zipFilename = decodeURIComponent(match[1]);
                    }

                    // Update UI
                    this.statusText.textContent = `✅ ${infoText}`;
                    this.statusText.style.color = "#4a9eff";
                    this.downloadBtn.style.display = "block";
                    this.downloadBtn.textContent = `📦 Download ${this.zipFilename || "ZIP"}`;
                }
            }
        };
    },
});
