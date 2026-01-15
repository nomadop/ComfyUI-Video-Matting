"""
ComfyUI Video Matting Nodes

Modular video matting pipeline supporting:
- RVM / MODNet / U2Net / RMBG for initial matting
- Cutie for video object segmentation
- ViTMatte for edge refinement
"""

# Note: Model repo paths are added to sys.path lazily in loaders.py
# via ensure_path_first() to avoid polluting sys.path and breaking other nodes

# Import loaders
from .nodes.loaders import (
    RVMLoader,
    MODNetLoader,
    U2NetLoader,
    CutieLoader,
    ViTMatteLoader,
)

# RMBG (optional - requires transformers)
from .nodes.rmbg import RMBGLoader, RMBGInference

# Import inference nodes
from .nodes.inference import (
    RVMInference,
    MODNetInference,
    U2NetInference,
)

# Import other nodes
from .nodes.cutie import CutieProcess
from .nodes.vitmatte import ViTMatteRefine, TrimapVisualize
from .nodes.alpha_ops import AlphaCombine
from .nodes.output import ApplyAlpha, FrameSelector, PreviewSlider, ImageSequencePackager

NODE_CLASS_MAPPINGS = {
    # Loaders
    "RVMLoader": RVMLoader,
    "MODNetLoader": MODNetLoader,
    "U2NetLoader": U2NetLoader,
    "CutieLoader": CutieLoader,
    "ViTMatteLoader": ViTMatteLoader,
    "RMBGLoader": RMBGLoader,
    # Inference
    "RVMInference": RVMInference,
    "MODNetInference": MODNetInference,
    "U2NetInference": U2NetInference,
    "RMBGInference": RMBGInference,
    # Processing
    "CutieProcess": CutieProcess,
    "ViTMatteRefine": ViTMatteRefine,
    # Alpha operations
    "AlphaCombine": AlphaCombine,
    # Debug
    "TrimapVisualize": TrimapVisualize,
    "FrameSelector": FrameSelector,
    "PreviewSlider": PreviewSlider,
    # Output
    "ApplyAlpha": ApplyAlpha,
    "ImageSequencePackager": ImageSequencePackager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "RVMLoader": "RVM Loader",
    "MODNetLoader": "MODNet Loader",
    "U2NetLoader": "U2Net Loader",
    "CutieLoader": "Cutie Loader",
    "ViTMatteLoader": "ViTMatte Loader",
    "RMBGLoader": "RMBG Loader",
    # Inference
    "RVMInference": "RVM Inference",
    "MODNetInference": "MODNet Inference",
    "U2NetInference": "U2Net Inference",
    "RMBGInference": "RMBG Inference",
    # Processing
    "CutieProcess": "Cutie Process",
    "ViTMatteRefine": "ViTMatte Refine",
    # Alpha operations
    "AlphaCombine": "Alpha Combine",
    # Debug
    "TrimapVisualize": "Trimap Visualize",
    "FrameSelector": "Frame Selector",
    "PreviewSlider": "Preview Slider",
    # Output
    "ApplyAlpha": "Apply Alpha",
    "ImageSequencePackager": "Image Sequence Packager",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
