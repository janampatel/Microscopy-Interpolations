"""Headless backend comparison: feed every other real frame, regenerate the
skipped ones with 2x interpolation and score them (PSNR/SSIM). Writes
per-backend .tif folders, scores.json and comparison_overview.png.
"""
import json
import os

import numpy as np


def _metrics(pred, gt):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    lo, hi = float(gt.min()), float(gt.max())
    rng = (hi - lo) if hi > lo else 1.0
    g = np.clip((gt.astype(np.float32) - lo) / rng, 0, 1)
    p = np.clip((pred.astype(np.float32) - lo) / rng, 0, 1)
    return float(psnr(g, p, data_range=1.0)), float(ssim(g, p, data_range=1.0))


def _disp(a):
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _run_backend(microinterp, backend, input_stack, out_dir):
    ok, reason = microinterp.check(backend)
    if not ok:
        print(f"  [skip] {backend}: {reason}")
        return None
    n = len(input_stack)
    intervals = [{"start": 0, "stop": n - 1, "interp": 1}]  # 2x over whole series
    try:
        out, _ = microinterp.interpolate_intervals(input_stack, intervals, backend=backend)
    except Exception as e:
        print(f"  [skip] {backend}: run failed -> {type(e).__name__}: {str(e)[:140]}")
        return None
    roles = microinterp.describe_layout(n, intervals)
    import tifffile
    bdir = os.path.join(out_dir, backend)
    os.makedirs(bdir, exist_ok=True)
    for i, fr in enumerate(out):
        tifffile.imwrite(os.path.join(bdir, f"frame_{i + 1:04d}.tif"), fr)
    return out, roles


def run_compare(input_dir, backends, out, max_show=6):
    import microinterp
    from .cli import load_stack

    frames, _ = load_stack(input_dir)
    if len(frames) < 3:
        raise SystemExit("Need at least 3 consecutive frames.")
    gt = frames
    inp = frames[::2]
    print(f"Loaded {len(frames)} frames ({frames.dtype}); input={len(inp)}, "
          f"expected output={2 * len(inp) - 1}, gt={len(gt)}")
    if 2 * len(inp) - 1 != len(gt):
        print("NOTE: output length won't equal GT length (need an ODD number of frames). "
              "Interpolated frames are still scored against the nearest real frame.")

    os.makedirs(out, exist_ok=True)
    results, scores = {}, {}
    for b in backends:
        print(f"[{b}] running...")
        r = _run_backend(microinterp, b, inp, out)
        if r is None:
            continue
        out_stack, roles = r
        results[b] = (out_stack, roles)
        s = []
        for i, role in enumerate(roles):
            if role["role"] == "interpolated" and i < len(gt):
                ps, ss = _metrics(out_stack[i], gt[i])
                s.append((i, ps, ss))
        scores[b] = s
        if s:
            mp = np.mean([x[1] for x in s]); ms = np.mean([x[2] for x in s])
            print(f"  {b}: mean PSNR={mp:.2f} dB  mean SSIM={ms:.4f}  over {len(s)} frames")

    print("\n==================== SUMMARY ====================")
    print(f"{'backend':10s} {'mean PSNR (dB)':>16s} {'mean SSIM':>12s}")
    for b in backends:
        if scores.get(b):
            mp = np.mean([x[1] for x in scores[b]]); ms = np.mean([x[2] for x in scores[b]])
            print(f"{b:10s} {mp:16.2f} {ms:12.4f}")
        else:
            print(f"{b:10s} {'(unavailable)':>16s}")
    with open(os.path.join(out, "scores.json"), "w") as f:
        json.dump({b: scores.get(b, []) for b in backends}, f, indent=2)

    _make_figure(gt, results, scores, os.path.join(out, "comparison_overview.png"), max_show)
    print(f"\nSaved: {os.path.join(out, 'comparison_overview.png')} and per-backend folders in {out}")


def _make_figure(gt, results, scores, out_path, max_show):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    backends = list(results.keys())
    if not backends:
        print("No backend produced output; skipping figure.")
        return
    roles = results[backends[0]][1]
    interp_idx = [i for i, r in enumerate(roles) if r["role"] == "interpolated"][:max_show]
    ncols = 1 + len(backends)
    nrows = max(1, len(interp_idx))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for row, idx in enumerate(interp_idx):
        axes[row][0].imshow(_disp(gt[idx]), cmap="gray")
        axes[row][0].set_title(f"GT frame {idx}", fontsize=9)
        for c, b in enumerate(backends, start=1):
            out_stack = results[b][0]
            axes[row][c].imshow(_disp(out_stack[idx]), cmap="gray")
            sc = {i: (p, s) for i, p, s in scores.get(b, [])}
            if idx in sc:
                axes[row][c].set_title(f"{b.upper()}\nPSNR {sc[idx][0]:.2f}  SSIM {sc[idx][1]:.3f}",
                                       fontsize=9, color="#e67e22")
            else:
                axes[row][c].set_title(b.upper(), fontsize=9)
        for c in range(ncols):
            axes[row][c].set_xticks([]); axes[row][c].set_yticks([])
    fig.suptitle("Interpolation comparison: Ground Truth vs " + " vs ".join(b.upper() for b in backends),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
