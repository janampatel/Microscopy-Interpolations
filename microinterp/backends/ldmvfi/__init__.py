"""LDMVFI backend (danier97/LDMVFI, AAAI 2024). Needs an NVIDIA GPU and the engine
installed by `microinterp setup`. The config is bundled; the checkpoint is
downloaded on first use (override with LDMVFI_CKPT / LDMVFI_CONFIG).
"""
import os
import sys
import torch
import torch.nn.functional as F

from ... import weights as _weights

LDMVFI_DIR = os.path.dirname(__file__)
ENGINE_REPOS = ("LDMVFI", "taming-transformers", "CLIP")
_CKPT_NAME = "ldmvfi-vqflow-f32-c256-concat_max.ckpt"
_CONFIG_NAME = "ldmvfi-vqflow-f32-c256-concat_max.yaml"

# Config is bundled in the package (tiny). Env override wins.
_BUNDLED_CONFIG = os.path.join(LDMVFI_DIR, _CONFIG_NAME)
CONFIG_PATH = os.environ.get("LDMVFI_CONFIG") or _BUNDLED_CONFIG

# LDMVFI_CKPT wins, else the per-user cache (downloaded on first use)
CKPT_PATH = os.environ.get("LDMVFI_CKPT") or str(_weights.cache_dir() / _CKPT_NAME)

DDIM_STEPS = int(os.environ.get("LDMVFI_DDIM_STEPS", "200"))
DDIM_ETA = float(os.environ.get("LDMVFI_DDIM_ETA", "1.0"))

# VQ autoencoder downsampling factor is 32 -> pad H,W to a multiple of 32.
_ALIGN = 32


def ldmvfi_weights_loaded() -> bool:
    """True if the LDMVFI checkpoint and config are present."""
    return os.path.exists(CKPT_PATH) and os.path.exists(CONFIG_PATH)


def add_engine_paths():
    """Put the engine fetched by `microinterp setup` on sys.path (no PYTHONPATH needed)."""
    for name in ENGINE_REPOS:
        p = str(_weights.engine_dir() / name)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _resolve_ckpt() -> str:
    """Return a local LDMVFI checkpoint path, downloading to cache if needed."""
    if os.path.exists(CKPT_PATH):
        return CKPT_PATH
    return str(_weights.ensure("ldmvfi"))


class LDMVFIModel:
    """LDMVFI wrapper with the shared `.inference` interface."""

    def __init__(self, config_path=CONFIG_PATH, ckpt_path=CKPT_PATH,
                 ddim_steps=DDIM_STEPS, eta=DDIM_ETA, device=None):
        # lazy: the ldm engine exists only on GPU machines
        add_engine_paths()
        from .._cuda import _preload_cuda_libs
        _preload_cuda_libs()
        try:
            from omegaconf import OmegaConf
            from ldm.util import instantiate_from_config
            from ldm.models.diffusion.ddim import DDIMSampler
        except Exception as e:
            raise ImportError(
                f"LDMVFI engine not available ({e}). Run `microinterp setup` on an NVIDIA GPU machine."
            ) from e

        if not os.path.exists(ckpt_path):
            ckpt_path = _resolve_ckpt()  # download to cache on first use
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LDMVFI config not found: {config_path}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        # strict=False tolerates EMA / auxiliary keys that aren't part of forward.
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model = model

        self.sampler = DDIMSampler(model)
        self.ddim_steps = ddim_steps
        self.eta = eta

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def inference(self, img0, img1, scale=1, timestep=0.5):
        """Midpoint frame between img0 and img1 (LDMVFI is trained for t=0.5)."""
        h, w = img0.shape[2], img0.shape[3]
        ph = ((h - 1) // _ALIGN + 1) * _ALIGN
        pw = ((w - 1) // _ALIGN + 1) * _ALIGN
        pad = (0, pw - w, 0, ph - h)

        # microinterp feeds [0,1]; LDMVFI expects [-1,1].
        x0 = F.pad(img0 * 2.0 - 1.0, pad).to(self.device)
        x1 = F.pad(img1 * 2.0 - 1.0, pad).to(self.device)

        with self.model.ema_scope():
            xc = {"prev_frame": x0, "next_frame": x1}
            c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
            shape = (self.model.channels, c.shape[2], c.shape[3])
            samples = self.sampler.sample(
                S=self.ddim_steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False,
                eta=self.eta,
                x_T=None,
            )
            if isinstance(samples, (tuple, list)):
                samples = samples[0]  # DDIMSampler returns (samples, intermediates)
            out = self.model.decode_first_stage(samples, xc, phi_prev_list, phi_next_list)
            out = torch.clamp(out, min=-1.0, max=1.0)

        out01 = (out + 1.0) / 2.0                     # back to [0,1]
        return out01[:, :, :h, :w].clamp(0.0, 1.0)    # crop away padding
