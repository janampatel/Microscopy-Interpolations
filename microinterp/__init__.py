"""microinterp -- microscopy frame interpolation with selectable backends.

Four interchangeable frame-interpolation models (RIFE, FILM, CDFI, LDMVFI)
behind one interface. Real frames pass through unchanged and the input dtype is
preserved (16-bit TIFF in -> 16-bit TIFF out); only the in-between frames are
generated.
"""
import os

from .backends import (
    BACKENDS,
    list_backends,
    available_backends,
    weights_loaded,
    check,
    load_model,
)

DEMO_DATA = os.path.join(os.path.dirname(__file__), "data", "toy_dataset")
_CORE = ("interpolate_intervals", "interpolate", "describe_layout", "get_interpolated_length")


def __getattr__(name):
    # torch loads on first use only, so `microinterp setup` can replace it safely
    if name in _CORE:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module 'microinterp' has no attribute {name!r}")

__all__ = [
    "BACKENDS",
    "DEMO_DATA",
    "list_backends",
    "available_backends",
    "weights_loaded",
    "check",
    "load_model",
    "interpolate_intervals",
    "interpolate",
    "describe_layout",
    "get_interpolated_length",
]
