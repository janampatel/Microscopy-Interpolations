"""CDFI backend (tding1/CDFI, CVPR 2021). Needs an NVIDIA GPU with cupy.

The checkpoint and the AdaCoF op are downloaded from tding1/CDFI on first use;
override the checkpoint with `CDFI_CKPT`.
"""
import os
from types import SimpleNamespace

import torch

from ... import weights as _weights

_CKPT_NAME = "CDFI_adacof.pth"
CKPT_PATH = os.environ.get("CDFI_CKPT") or str(_weights.cache_dir() / _CKPT_NAME)
KERNEL_SIZE = 11  # must match CDFI_adacof.pth
DILATION = 2


def cdfi_weights_loaded() -> bool:
    """True if the CDFI checkpoint is present (no heavy imports)."""
    return os.path.exists(CKPT_PATH)


def load_op():
    """Import the downloaded upstream AdaCoF op, adapted to current CuPy."""
    import importlib.util
    import types
    import warnings

    from .._cuda import _preload_cuda_libs
    _preload_cuda_libs()
    import cupy
    if not hasattr(cupy, "util"):  # upstream decorates with cupy.util.memoize, gone in newer CuPy
        cupy.util = types.SimpleNamespace(memoize=lambda **_: (lambda f: f))
    spec = importlib.util.spec_from_file_location("microinterp_cdfi_op", _weights.ensure("cdfi_op"))
    op = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # upstream regex strings
        spec.loader.exec_module(op)
    cache = {}

    def launch(name, code):  # replaces cupy.cuda.compile_with_cache, gone in newer CuPy
        if (name, code) not in cache:
            cache[name, code] = cupy.RawModule(code=code).get_function(name)
        return cache[name, code]

    op.cupy_launch = launch
    return op


class CDFIModel:
    """CDFI wrapper with the shared `.inference` interface."""

    def __init__(self, ckpt_path: str = CKPT_PATH, kernel_size: int = KERNEL_SIZE,
                 dilation: int = DILATION, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type != "cuda":
            raise RuntimeError(
                "The CDFI backend requires an NVIDIA GPU (it uses a CUDA-only "
                "cupy custom op) and cannot run on CPU."
            )
        if not os.path.exists(ckpt_path):
            ckpt_path = str(_weights.ensure("cdfi"))

        # Lazy import: pulls in cupy (GPU only) via cdfi_adacof -> load_op.
        from .cdfi_adacof import CDFI_adacof

        args = SimpleNamespace(kernel_size=kernel_size, dilation=dilation)
        model = CDFI_adacof(args)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def inference(self, img0, img1, scale=1, timestep=0.5):
        """Midpoint frame between img0 and img1; `scale` and `timestep` are ignored (CDFI
        is trained for t=0.5).
        """
        x0 = img0.to(self.device).float()
        x1 = img1.to(self.device).float()
        out = self.model(x0, x1)  # CDFI pads to /32 internally and crops back
        return out.clamp(0.0, 1.0)
