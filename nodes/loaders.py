"""
Model Loader Nodes (Split Version)

- RVMLoader: RobustVideoMatting model
- MODNetLoader: MODNet model
- U2NetLoader: U-2-Net model
- CutieLoader: Cutie video object segmentation model
- ViTMatteLoader: ViTMatte edge refinement model
"""

import os
import sys
import torch
import yaml
import folder_paths

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Model weights directory (ComfyUI standard)
MODELS_DIR = os.path.join(folder_paths.models_dir, "video_matting")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def scan_checkpoints(model_type):
    """Scan available checkpoints for a model type from ComfyUI models directory"""
    type_dir = os.path.join(MODELS_DIR, model_type.lower())

    checkpoints = []

    if os.path.exists(type_dir):
        for f in os.listdir(type_dir):
            if f.endswith(('.pth', '.ckpt', '.pt')):
                checkpoints.append(f)

    return sorted(checkpoints) if checkpoints else ['default']


def ensure_path_first(repo_key):
    """Ensure the repo path is first in sys.path and clear cached modules"""
    repo_path = os.path.expanduser(CONFIG['model_repos'][repo_key]['path'])
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)

    # Clear potentially conflicting cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('model', 'modeling') or mod_name.startswith(('model.', 'modeling.')):
            del sys.modules[mod_name]

    return repo_path


# =============================================================================
# RVM Loader
# =============================================================================

class RVMLoader:
    """RobustVideoMatting model loader"""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = scan_checkpoints("rvm")
        return {
            "required": {
                "checkpoint": (ckpts, {"default": ckpts[0] if ckpts else "default"}),
                "downsample_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Downsample ratio for processing (lower = faster)"
                }),
            }
        }

    RETURN_TYPES = ("RVM_MODEL",)
    RETURN_NAMES = ("rvm_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, checkpoint, downsample_ratio):
        device = get_device()

        # Ensure RVM path is first
        ensure_path_first('rvm')

        from model import MattingNetwork as RVMMattingNetwork

        # Resolve checkpoint
        models_dir = MODELS_DIR
        ckpt_path = os.path.join(models_dir, 'rvm', checkpoint)

        model = RVMMattingNetwork('resnet50').eval().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        return ({
            "model": model,
            "device": device,
            "downsample_ratio": downsample_ratio,
        },)


# =============================================================================
# MODNet Loader
# =============================================================================

class MODNetLoader:
    """MODNet model loader"""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = scan_checkpoints("modnet")
        return {
            "required": {
                "checkpoint": (ckpts, {"default": ckpts[0] if ckpts else "default"}),
                "input_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Input image size for processing"
                }),
            }
        }

    RETURN_TYPES = ("MODNET_MODEL",)
    RETURN_NAMES = ("modnet_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, checkpoint, input_size):
        import torch.nn as nn
        device = get_device()

        # Ensure MODNet path is first
        ensure_path_first('modnet')

        from src.models.modnet import MODNet

        # Resolve checkpoint
        models_dir = MODELS_DIR
        ckpt_path = os.path.join(models_dir, 'modnet', checkpoint)

        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)
        modnet.load_state_dict(torch.load(ckpt_path, map_location=device))
        modnet = modnet.to(device).eval()

        return ({
            "model": modnet,
            "device": device,
            "input_size": input_size,
        },)


# =============================================================================
# U2Net Loader
# =============================================================================

class U2NetLoader:
    """U-2-Net model loader"""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = scan_checkpoints("u2net")
        return {
            "required": {
                "checkpoint": (ckpts, {"default": ckpts[0] if ckpts else "default"}),
                "input_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Input image size for processing"
                }),
            }
        }

    RETURN_TYPES = ("U2NET_MODEL",)
    RETURN_NAMES = ("u2net_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, checkpoint, input_size):
        device = get_device()

        # Ensure U2Net path is first
        ensure_path_first('u2net')

        from model import U2NET

        # Resolve checkpoint
        models_dir = MODELS_DIR
        ckpt_path = os.path.join(models_dir, 'u2net', checkpoint)

        u2net = U2NET(3, 1)
        weights = torch.load(ckpt_path, map_location=device)
        u2net.load_state_dict(weights)
        u2net.to(device).eval()

        return ({
            "model": u2net,
            "device": device,
            "input_size": input_size,
        },)


# =============================================================================
# Cutie Loader
# =============================================================================

class CutieLoader:
    """Cutie video object segmentation model loader"""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = scan_checkpoints("cutie")
        return {
            "required": {
                "checkpoint": (ckpts, {"default": ckpts[0] if ckpts else "default"}),
                "max_internal_size": ("INT", {
                    "default": 480,
                    "min": 256,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Internal processing size, smaller = faster"
                }),
            }
        }

    RETURN_TYPES = ("CUTIE_MODEL",)
    RETURN_NAMES = ("cutie_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, checkpoint, max_internal_size):
        # Ensure Cutie path is first
        ensure_path_first('cutie')

        from omegaconf import open_dict
        from hydra import compose, initialize_config_dir
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg

        device = get_device()

        # Resolve checkpoint path
        models_dir = MODELS_DIR
        ckpt_path = os.path.join(models_dir, 'cutie', checkpoint)

        # Load config from Cutie repo
        repo_path = os.path.expanduser(CONFIG['model_repos']['cutie']['path'])
        config_dir = os.path.join(repo_path, 'cutie', 'config')

        initialize_config_dir(version_base='1.3.2', config_dir=config_dir)
        cfg = compose(config_name="eval_config")

        with open_dict(cfg):
            cfg['weights'] = ckpt_path
        get_dataset_cfg(cfg)

        # Load model
        cutie = CUTIE(cfg).to(device).eval()
        model_weights = torch.load(ckpt_path, map_location=device)
        cutie.load_weights(model_weights)

        processor = InferenceCore(cutie, cfg=cutie.cfg)
        processor.max_internal_size = max_internal_size

        return ({
            "processor": processor,
            "device": device,
            "max_internal_size": max_internal_size,
        },)


# =============================================================================
# ViTMatte Loader
# =============================================================================

class ViTMatteLoader:
    """ViTMatte edge refinement model loader"""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = scan_checkpoints("vitmatte")
        return {
            "required": {
                "checkpoint": (ckpts, {"default": ckpts[0] if ckpts else "default"}),
                "use_fp16": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use FP16 for faster inference (CUDA/MPS only)"
                }),
            }
        }

    RETURN_TYPES = ("VITMATTE_MODEL",)
    RETURN_NAMES = ("vitmatte_model",)
    FUNCTION = "load_model"
    CATEGORY = "Video Matting/Loaders"

    def load_model(self, checkpoint, use_fp16):
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        device = get_device()
        repo_path = ensure_path_first('vitmatte')

        # Resolve checkpoint
        models_dir = MODELS_DIR
        ckpt_path = os.path.join(models_dir, 'vitmatte', checkpoint)

        # ViTMatte needs to load config from its directory
        original_dir = os.getcwd()
        os.chdir(repo_path)

        try:
            config = 'configs/common/model.py'
            cfg = LazyConfig.load(config)
            cfg.model.backbone.embed_dim = 768
            cfg.model.backbone.num_heads = 12
            cfg.model.decoder.in_chans = 768
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()

            DetectionCheckpointer(model).load(ckpt_path)
        finally:
            os.chdir(original_dir)

        return ({
            "model": model,
            "device": device,
            "use_fp16": use_fp16,
        },)
