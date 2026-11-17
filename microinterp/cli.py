"""Command-line interface for microinterp.

Subcommands: setup, list, interpolate, download-weights, compare, gui.
Input is a folder of frames (.tif/.png/.jpg/.bmp) loaded in sorted order; the
data type and value range are preserved on output.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _read_image(path):
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(path)
    from PIL import Image
    return np.array(Image.open(path))


def _write_image(path, arr):
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        tifffile.imwrite(path, arr)
    else:
        from PIL import Image
        Image.fromarray(arr).save(path)


def _resolve_input(folder):
    """`demo` means the sample frames bundled with the package."""
    if folder == "demo":
        from . import DEMO_DATA
        return DEMO_DATA
    return folder


def load_stack(folder):
    """Load a folder of consecutive frames -> (N, H, W) stack + file list."""
    folder = _resolve_input(folder)
    if not os.path.isdir(folder):
        sys.exit(f"Input is not a folder: {folder}")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(_EXTS))
    if not files:
        sys.exit(f"No images found in {folder}")
    arrs = []
    for f in files:
        a = _read_image(os.path.join(folder, f))
        if a.ndim == 3:  # RGB -> grayscale, keep dtype
            a = a[..., :3].mean(axis=-1).astype(a.dtype)
        arrs.append(a)
    return np.stack(arrs), files


def _default_ext(files):
    ext = os.path.splitext(files[0])[1].lower()
    return ext if ext in _EXTS else ".tif"


def cmd_list(args):
    import microinterp
    print(f"{'backend':8s} {'available':10s} {'device':26s} notes")
    print("-" * 80)
    for b in microinterp.list_backends():
        meta = microinterp.BACKENDS[b]
        ok, reason = microinterp.check(b)
        print(f"{b:8s} {'yes' if ok else 'no':10s} {meta['device']:26s} {reason or meta['notes']}")


def _factor_to_interp(factor):
    if factor < 2 or (factor & (factor - 1)) != 0:
        sys.exit(f"--factor must be a power of two >= 2 (got {factor}).")
    return int(round(math.log2(factor)))


def cmd_interpolate(args):
    import microinterp

    ok, reason = microinterp.check(args.backend)
    if not ok:
        sys.exit(f"[{args.backend}] {reason}")

    input_dir = os.path.abspath(_resolve_input(args.input))
    stack, files = load_stack(input_dir)
    n = len(stack)
    interp = _factor_to_interp(args.factor)
    start = args.start if args.start is not None else 0
    stop = args.stop if args.stop is not None else n - 1
    if not (0 <= start < stop <= n - 1):
        sys.exit(f"Invalid interval {start}-{stop} for a {n}-frame stack.")
    intervals = [{"start": start, "stop": stop, "interp": interp}]

    print(f"Loaded {n} frames ({stack.dtype}, {stack.shape[1]}x{stack.shape[2]}); "
          f"backend={args.backend}, {args.factor}x over frames {start}-{stop}")
    try:
        out, new_iv = microinterp.interpolate_intervals(stack, intervals, backend=args.backend)
    except Exception as e:
        if os.environ.get("MICROINTERP_DEBUG"):
            raise
        sys.exit(f"[{args.backend}] {type(e).__name__}: {e}")
    layout = microinterp.describe_layout(n, intervals)

    ext = args.ext or _default_ext(files)
    os.makedirs(args.out, exist_ok=True)
    for i, fr in enumerate(out):
        _write_image(os.path.join(args.out, f"frame_{i:05d}{ext}"), fr)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"backend": args.backend, "dtype": str(stack.dtype),
                   "intervals": [dict(iv, factor=2 ** iv["interp"]) for iv in intervals],
                   "input_dir": input_dir, "input_files": files,
                   "output_dir": os.path.abspath(args.out),
                   "num_input": n, "num_output": len(out), "layout": layout, "roles": layout},
                  fh, indent=2)
    print(f"Wrote {len(out)} frames + manifest.json to {args.out}")


def cmd_download_weights(args):
    """Prefetch backend weights into the per-user cache (env MICROINTERP_HOME
    or ~/.cache/microinterp/weights). The RIFE weight is bundled."""
    from . import weights

    for b in args.backend:
        if b == "rife":
            from . import core
            print(f"[rife] bundled weight {'present' if core.rife_weights_loaded() else 'MISSING'}")
        else:
            print(f"[{b}] ready: {weights.ensure(b)}")
            if b == "cdfi":
                weights.ensure("cdfi_op")


def cmd_setup(args):
    from .setup_env import run
    run(yes=args.yes)


def cmd_compare(args):
    from .compare import run_compare
    run_compare(args.input, args.backends, args.out, args.max_show)


def cmd_gui(args):
    from .gui.app import launch_gui
    launch_gui()


def build_parser():
    p = argparse.ArgumentParser(prog="microinterp",
                                description="Microscopy frame interpolation (RIFE / FILM / CDFI / LDMVFI).")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("setup", help="detect this machine and install what the GPU backends need")
    sp.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("list", help="list backends and availability")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("interpolate", help="interpolate a folder of frames with one backend")
    sp.add_argument("input", help="folder of consecutive frames, or `demo` for the bundled sample")
    sp.add_argument("--backend", default="rife", choices=["rife", "film", "cdfi", "ldmvfi"])
    sp.add_argument("--factor", type=int, default=2, help="interpolation factor, power of 2 (2,4,8,16)")
    sp.add_argument("--start", type=int, default=None, help="interval start frame (default 0)")
    sp.add_argument("--stop", type=int, default=None, help="interval stop frame (default last)")
    sp.add_argument("--out", default="./interp_out")
    sp.add_argument("--ext", default=None, help="output extension (default: same as input)")
    sp.set_defaults(func=cmd_interpolate)

    sp = sub.add_parser("download-weights", help="prefetch backend weights into the per-user cache")
    sp.add_argument("--backend", nargs="+", default=["rife", "film", "cdfi", "ldmvfi"],
                    choices=["rife", "film", "cdfi", "ldmvfi"])
    sp.set_defaults(func=cmd_download_weights)

    sp = sub.add_parser("compare", help="compare backends (PSNR/SSIM + figure) via i-1,i+1 -> i check")
    sp.add_argument("input", help="folder of consecutive frames (odd count recommended), or `demo`")
    sp.add_argument("--backends", nargs="+", default=["rife", "film", "cdfi", "ldmvfi"])
    sp.add_argument("--out", default="./comparison_out")
    sp.add_argument("--max-show", dest="max_show", type=int, default=6)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("gui", help="launch the interpolation GUI (needs PySide6)")
    sp.set_defaults(func=cmd_gui)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
