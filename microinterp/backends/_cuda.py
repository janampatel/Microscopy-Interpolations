_cuda_preloaded = False


def _preload_cuda_libs():
    """Best-effort preload of the CUDA libraries from pip's nvidia-* wheels, so cupy
    finds NVRTC in a venv.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True
    import ctypes
    import glob
    import os
    try:
        import nvidia
    except Exception:
        return
    # nvidia is a namespace package (no __file__) in the CUDA 13 wheels, so use __path__
    for base in list(nvidia.__path__):
        # nvrtc first (the usual culprit), then everything else the wheels provide.
        patterns = [
            os.path.join(base, "cuda_nvrtc", "lib", "libnvrtc.so*"),
            os.path.join(base, "*", "lib", "*.so*"),
        ]
        for pat in patterns:
            for so in sorted(glob.glob(pat)):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
