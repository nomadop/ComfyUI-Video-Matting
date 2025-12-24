# ComfyUI Video Matting

Modular video matting nodes for ComfyUI supporting multiple matting models and video object segmentation.

## Features

- **Multiple matting models**: RVM, MODNet, U2Net, RMBG-2.0
- **Video object segmentation**: Cutie with temporal propagation
- **Edge refinement**: ViTMatte for high-quality alpha edges
- **Flexible pipeline**: Mix and match models for different use cases

## Nodes

### Loaders
| Node | Description |
|------|-------------|
| RVM Loader | Load RobustVideoMatting model |
| MODNet Loader | Load MODNet model |
| U2Net Loader | Load U-2-Net model |
| RMBG Loader | Load RMBG-2.0 from HuggingFace (no repo clone needed) |
| Cutie Loader | Load Cutie video segmentation model |
| ViTMatte Loader | Load ViTMatte refinement model |

### Inference
| Node | Description |
|------|-------------|
| RVM Inference | Process frames through RVM (maintains temporal state) |
| MODNet Inference | Process frames through MODNet |
| U2Net Inference | Process frames through U2Net |
| RMBG Inference | Process frames through RMBG-2.0 |

### Processing
| Node | Description |
|------|-------------|
| Cutie Process | Video propagation with optional ViTMatte refinement |
| ViTMatte Refine | Standalone edge refinement with built-in trimap generation |

### Utilities
| Node | Description |
|------|-------------|
| Alpha Combine | Combine multiple alpha channels (avg/max/min/multiply) |
| Apply Alpha | Output RGBA or composite over background |
| Frame Selector | Select single frame from batch |
| Preview Slider | Preview sequence with slider (no re-run needed) |
| Trimap Visualize | Debug visualization of trimap |

## Installation

### 1. Clone this repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xxx/comfyui_video_matting.git
```

### 2. Install dependencies
```bash
cd comfyui_video_matting
pip install -r requirements.txt
```

For RMBG-2.0 support (optional, no repo clone needed):
```bash
pip install transformers kornia
```
Note: RMBG-2.0 is CC BY-NC 4.0 licensed (non-commercial use only).

For ViTMatte support:
```bash
pip install fairscale timm
pip install wheel ninja
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

**Important**: After installing detectron2, remove its `tools` package to avoid conflicts:
```bash
rm -rf $COMFYUI_VENV/lib/python3.10/site-packages/tools
```

### 3. Clone model repositories
```bash
# Choose location for model repos
cd /path/to/repos

git clone https://github.com/PeterL1n/RobustVideoMatting
git clone https://github.com/ZHKKKe/MODNet
git clone https://github.com/xuebinqin/U-2-Net
git clone https://github.com/hkchengrex/Cutie
git clone https://github.com/hustvl/ViTMatte
```

### 4. Configure paths
```bash
cd ComfyUI/custom_nodes/comfyui_video_matting
cp config.yaml.example config.yaml
# Edit config.yaml with your repo paths
```

### 5. Download model weights

Place weights in `ComfyUI/models/video_matting/<model>/`:

```
models/video_matting/
├── rvm/
│   └── rvm_mobilenetv3.pth
├── modnet/
│   └── modnet_photographic_portrait_matting.ckpt
├── u2net/
│   └── u2net.pth
├── cutie/
│   └── cutie-base-mega.pth
└── vitmatte/
    └── ViTMatte_B_DIS.pth
```

**Weight downloads**:
- RVM: [GitHub Releases](https://github.com/PeterL1n/RobustVideoMatting/releases)
- MODNet: [Google Drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR)
- U2Net: [Google Drive](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ)
- Cutie: [GitHub Releases](https://github.com/hkchengrex/Cutie/releases)
- ViTMatte: [Hugging Face](https://huggingface.co/hustvl/vitmatte-base-distinctions-646)

## Usage

### Basic matting (single model)
```
Load Video → RVM Loader → RVM Inference → Apply Alpha → Save Video
```

### Video segmentation with refinement
```
Load Video ─┬→ Cutie Loader ─┐
            │                │
            └→ ViTMatte Loader → Cutie Process → Apply Alpha → Save Video
                               ↑
            Initial Mask ──────┘
```

### Standalone refinement
```
Load Video → Any Matting Model → ViTMatte Loader → ViTMatte Refine → Apply Alpha
```

## Known Issues

- **detectron2 tools conflict**: detectron2 installs a top-level `tools` package that conflicts with other nodes (e.g., GSTTS). Delete `site-packages/tools/` after installing detectron2.

## License

MIT
