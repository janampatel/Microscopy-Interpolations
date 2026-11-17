"""FILM backend (TorchScript port dajes/frame-interpolation-pytorch). The weights
are downloaded to the per-user cache on first use.
"""
import os
import torch
import torch.nn.functional as F

from ... import weights as _weights

FILM_WEIGHTS_NAME = "film_net_fp32.pt"

# FILM's film_net pyramid requires height and width divisible by 64.
_ALIGN = 64


def _cached_film_path():
    """Resolved local path candidates for the FILM weight (env, then cache)."""
    env = os.environ.get("FILM_WEIGHTS")
    if env:
        return env
    return str(_weights.cache_dir() / FILM_WEIGHTS_NAME)


def film_weights_loaded() -> bool:
    """True if the FILM weight is present (env override or cache)."""
    return os.path.exists(_cached_film_path())


def _resolve_film_weights() -> str:
    """Return a local FILM weight path, downloading to the cache if needed."""
    env = os.environ.get("FILM_WEIGHTS")
    if env and os.path.exists(env):
        return env
    return str(_weights.ensure("film"))


class FILMModel:
    """FILM wrapper with the shared `.inference` interface."""

    def __init__(self, weights_path: str = None, device=None):
        self.dev = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        weights_path = weights_path or _resolve_film_weights()
        self.model = torch.jit.load(weights_path, map_location=self.dev)
        self.model.eval()
        self.model.float()
        self.model.to(self.dev)

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def inference(self, img0, img1, scale=1, timestep=0.5):
        """Frame at `timestep` between img0 and img1, same H, W as the inputs."""
        h, w = img0.shape[2], img0.shape[3]
        ph = ((h - 1) // _ALIGN + 1) * _ALIGN
        pw = ((w - 1) // _ALIGN + 1) * _ALIGN
        pad = (0, pw - w, 0, ph - h)
        x0 = F.pad(img0, pad)
        x1 = F.pad(img1, pad)
        dt = x0.new_full((1, 1), float(timestep))
        out = self.model(x0, x1, dt)
        return out[:, :, :h, :w].clamp(0.0, 1.0)
