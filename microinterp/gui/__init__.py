"""microinterp GUI (PySide6, imported lazily)."""


def launch_gui(*args, **kwargs):
    from .app import launch_gui as _launch
    return _launch(*args, **kwargs)
