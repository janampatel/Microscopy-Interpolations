"""Backend registry. Every model exposes `.inference(img0, img1, scale=1,
timestep=0.5)` on (1, 3, H, W) float tensors in [0, 1]. GPU-only dependencies
are imported lazily, so `import microinterp` works on CPU-only machines.
"""
import os

RIFE_DIR = os.path.join(os.path.dirname(__file__), "rife")
RIFE_WEIGHTS_PATH = os.path.join(RIFE_DIR, "flownet.pkl")

# name -> human-facing metadata (for CLI/GUI listing and docs)
BACKENDS = {
    "rife": {
        "name": "RIFE",
        "type": "optical flow (IFNet)",
        "device": "CPU or GPU",
        "notes": "fast, general default; supports arbitrary timestep",
    },
    "film": {
        "name": "FILM",
        "type": "feature pyramid + flow (TorchScript)",
        "device": "GPU (runs on CPU, slowly)",
        "notes": "strong on large motion",
    },
    "cdfi": {
        "name": "CDFI",
        "type": "AdaCoF kernel prediction + GridNet synthesis",
        "device": "GPU only (cupy custom op)",
        "notes": "compact, sharp; fixed midpoint (t=0.5)",
    },
    "ldmvfi": {
        "name": "LDMVFI",
        "type": "latent diffusion (VQ + DDIM)",
        "device": "GPU only",
        "notes": "highest perceptual quality, slowest; needs external engine",
    },
}


def list_backends():
    """All backend names known to microinterp, in a sensible display order."""
    return ["rife", "film", "cdfi", "ldmvfi"]


def weights_loaded(backend: str) -> bool:
    """True if the backend's weights are present (no heavy imports)."""
    backend = backend.lower()
    if backend == "rife":
        return os.path.exists(RIFE_WEIGHTS_PATH)
    if backend == "film":
        from .film import film_weights_loaded
        return film_weights_loaded()
    if backend == "ldmvfi":
        from .ldmvfi import ldmvfi_weights_loaded
        return ldmvfi_weights_loaded()
    if backend == "cdfi":
        from .cdfi import cdfi_weights_loaded
        return cdfi_weights_loaded()
    raise ValueError(f"Unknown backend: {backend!r}. Choose from {list_backends()}.")


def available_backends():
    """Backends whose weights/checkpoints are currently present on disk."""
    return [b for b in list_backends() if weights_loaded(b)]


def check(backend: str):
    """Return (ok, reason): can `backend` run here? Weights downloaded on first use count
    as available.
    """
    backend = backend.lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from {list_backends()}.")
    if backend == "rife":
        return (True, "") if os.path.exists(RIFE_WEIGHTS_PATH) else (False, "flownet.pkl missing")
    if backend == "film":
        return True, ""

    import importlib.util
    import torch
    if not torch.cuda.is_available():
        return False, "needs an NVIDIA GPU with CUDA torch (run: microinterp setup)"
    need = ["cupy"]
    if backend == "ldmvfi":
        from .ldmvfi import add_engine_paths
        add_engine_paths()
        need += ["omegaconf", "pytorch_lightning", "ldm", "taming", "clip"]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"missing {', '.join(missing)} (run: microinterp setup)"
    return True, ""


_cached = {}  # holds only the last model, so switching backends frees GPU memory


def load_model(backend: str = "rife", device=None):
    """Return (model, device) for `backend`, reusing the last loaded model."""
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    backend = backend.lower()
    key = (backend, str(device))
    if key not in _cached:
        _cached.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cached[key] = _build_model(backend, device)
    return _cached[key], device


def _build_model(backend, device):
    if backend == "film":
        from .film import FILMModel
        return FILMModel(device=device).eval()

    if backend == "ldmvfi":
        from .ldmvfi import LDMVFIModel
        return LDMVFIModel(device=device).eval()

    if backend == "cdfi":
        from .cdfi import CDFIModel
        return CDFIModel(device=device).eval()

    if backend == "rife":
        from .rife.RIFE import Model
        model = Model()
        model.load_model(RIFE_DIR, -1)
        model.eval()
        model.device()
        return model

    raise ValueError(f"Unknown backend: {backend!r}. Choose from {list_backends()}.")
