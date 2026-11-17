import argparse
import math
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import microinterp  # noqa: E402
from microinterp.cli import load_stack  # noqa: E402


class VideoUtils:
    @staticmethod
    def to_uint8(arr, lo, hi):
        rng = (hi - lo) if hi > lo else 1.0
        a = np.clip((arr.astype(np.float32) - lo) / rng, 0, 1)
        return (a * 255).astype(np.uint8)

    @staticmethod
    def metrics(pred, gt):
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        lo, hi = float(gt.min()), float(gt.max())
        rng = (hi - lo) if hi > lo else 1.0
        g = np.clip((gt.astype(np.float32) - lo) / rng, 0, 1)
        p = np.clip((pred.astype(np.float32) - lo) / rng, 0, 1)
        try:
            return float(psnr(g, p, data_range=1.0)), float(ssim(g, p, data_range=1.0))
        except Exception:
            return float("nan"), float("nan")

    @staticmethod
    def center_crop(stack, size):
        if not size:
            return stack
        _, h, w = stack.shape
        s = min(size, h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        return stack[:, top:top + s, left:left + s]


class MediaWriter:
    def __init__(self, fps):
        self.fps = fps

    def save_gif(self, frames, path):
        try:
            import imageio.v2 as imageio
            imageio.mimsave(path, frames, duration=1000.0 / self.fps, loop=0)
            return True
        except Exception:
            try:
                from PIL import Image
                imgs = [Image.fromarray(f) for f in frames]
                imgs[0].save(path, save_all=True, append_images=imgs[1:],
                             duration=int(1000 / self.fps), loop=0, disposal=2)
                return True
            except Exception as e:
                print(f"  [warn] could not write {path}: {e}")
                return False

    def save_mp4(self, frames, path):
        try:
            import imageio.v2 as imageio
            imageio.mimsave(path, frames, fps=self.fps, codec="libx264",
                            macro_block_size=None, quality=8)
            return True
        except Exception as e:
            print(f"  [skip] mp4 not written ({e}); install imageio-ffmpeg for video output.")
            return False

    def save_gif_and_mp4(self, frames, out_dir, name):
        self.save_gif(frames, os.path.join(out_dir, f"{name}.gif"))
        self.save_mp4(frames, os.path.join(out_dir, f"{name}.mp4"))


class DemoGenerator:
    def __init__(self, out_dir, fps):
        self.out_dir = out_dir
        self.fps = fps
        self.writer = MediaWriter(fps)

    def run_backends_lofo(self, stack, backends):
        inp = stack[::2]
        if len(inp) < 2:
            sys.exit(f"Need at least 3 frames for a GT comparison; got {len(stack)}.")
        intervals = [{"start": 0, "stop": len(inp) - 1, "interp": 1}]
        roles = microinterp.describe_layout(len(inp), intervals)
        expected = 2 * len(inp) - 1
        if expected != len(stack):
            print(f"  [note] {len(stack)} frames -> output length {expected} (use an ODD frame "
                  "count for exact GT alignment; comparison is truncated to the common length).")

        results = {}
        for b in backends:
            ok, reason = microinterp.check(b)
            if not ok:
                print(f"  [skip] {b}: {reason}")
                continue
            print(f"  [{b}] running 2x reconstruction...")
            try:
                out, _ = microinterp.interpolate_intervals(inp, intervals, backend=b)
            except Exception as e:
                print(f"  [skip] {b}: run failed -> {type(e).__name__}: {str(e)[:120]}")
                continue
            L = min(len(out), len(stack))
            scores = {i: VideoUtils.metrics(out[i], stack[i]) for i in range(L)
                      if i < len(roles) and roles[i]["role"] == "interpolated"}
            results[b] = (out[:L], scores)

        if not results:
            sys.exit("No backend produced output; nothing to compare.")
        common = min(min(len(v[0]) for v in results.values()), len(stack))
        for b in list(results):
            out, sc = results[b]
            results[b] = (out[:common], {i: s for i, s in sc.items() if i < common})
        return stack[:common], results, roles

    def run_single(self, stack, backend, factor, per_frame_norm):
        ok, reason = microinterp.check(backend)
        if not ok:
            sys.exit(f"[{backend}] {reason}")
        n = len(stack)
        interp = int(round(math.log2(factor)))
        intervals = [{"start": 0, "stop": n - 1, "interp": interp}]

        print(f"Backend: {backend}   Factor: {factor}x")
        out, _ = microinterp.interpolate_intervals(stack, intervals, backend=backend)
        print(f"Output:  {len(out)} frames")

        if per_frame_norm:
            out_disp = np.stack([VideoUtils.to_uint8(f, f.min(), f.max()) for f in out])
        else:
            lo = float(min(stack.min(), out.min()))
            hi = float(max(stack.max(), out.max()))
            out_disp = np.stack([VideoUtils.to_uint8(f, lo, hi) for f in out])

        print("Rendering smooth loop...")
        smooth = [np.stack([f, f, f], axis=-1) for f in out_disp]
        self.writer.save_gif(smooth, os.path.join(self.out_dir, "smooth.gif"))

        print(f"\nDone. Assets written to: {self.out_dir}")
        print("  smooth.gif                  -> the interpolated series loop (hero clip)")

    def run_algos(self, stack, backends):
        print(f"Backends: {', '.join(backends)}   Protocol: leave-one-out 2x vs ground truth")
        gt, results, roles = self.run_backends_lofo(stack, backends)
        backends = [b for b in backends if b in results]  # keep requested order

        all_arrays = [gt] + [results[b][0] for b in backends]
        lo = float(min(a.min() for a in all_arrays))
        hi = float(max(a.max() for a in all_arrays))
        gt_disp = np.stack([VideoUtils.to_uint8(f, lo, hi) for f in gt])
        results_disp = {b: (np.stack([VideoUtils.to_uint8(f, lo, hi) for f in results[b][0]]), results[b][1])
                        for b in backends}

        print("\n  mean over reconstructed frames:")
        for b in backends:
            sc = list(results_disp[b][1].values())
            if sc:
                mp = float(np.mean([s[0] for s in sc])); ms = float(np.mean([s[1] for s in sc]))
                print(f"    {b:8s} PSNR {mp:6.2f} dB   SSIM {ms:.4f}")

        print("Rendering algos comparison frames...")
        gt_rgb = [np.stack([f, f, f], axis=-1) for f in gt_disp]
        self.writer.save_gif_and_mp4(gt_rgb, self.out_dir, "gt")

        for b in backends:
            out_disp, _ = results_disp[b]
            out_rgb = [np.stack([f, f, f], axis=-1) for f in out_disp]
            self.writer.save_gif_and_mp4(out_rgb, self.out_dir, b)

        print(f"\nDone. Assets written to: {self.out_dir}")
        print("  gt.gif / .mp4              -> ground truth")
        for b in backends:
            print(f"  {b}.gif / .mp4{(14 - len(b)) * ' '}-> {b.upper()} reconstruction")


def build_arg_parser(default_input, default_out):
    ap = argparse.ArgumentParser(description="Generate README demo media for microinterp.")
    ap.add_argument("--input", default=default_input,
                    help="folder of consecutive frames (default: bundled toy dataset)")
    ap.add_argument("--mode", default="single", choices=["single", "algos"],
                    help="single: interpolated loop (one backend). "
                         "algos: ground truth vs several backends.")
    ap.add_argument("--backend", default="rife", choices=["rife", "film", "cdfi", "ldmvfi"],
                    help="[single mode] which backend to run")
    ap.add_argument("--backends", nargs="+", default=["rife", "film", "cdfi", "ldmvfi"],
                    choices=["rife", "film", "cdfi", "ldmvfi"],
                    help="[algos mode] backends to compare against ground truth")
    ap.add_argument("--factor", type=int, default=4,
                    help="[single mode] interpolation factor, power of 2 (2,4,8,16)")
    ap.add_argument("--out", default=default_out,
                    help="output folder for the assets (default: ./demo_out)")
    ap.add_argument("--fps", type=int, default=2, help="playback frames per second for the clips")
    ap.add_argument("--crop", type=int, default=0,
                    help="center-crop each frame to CROPxCROP px (0 = full frame)")
    ap.add_argument("--per-frame-norm", action="store_true",
                    help="normalize each frame independently (default: one shared window, less flicker)")
    return ap


def main():
    default_input = os.path.join(_REPO_ROOT, "microinterp", "data", "toy_dataset")
    default_out = os.path.join(os.getcwd(), "demo_out")
    args = build_arg_parser(default_input, default_out).parse_args()

    if args.mode == "single" and (args.factor < 2 or (args.factor & (args.factor - 1)) != 0):
        sys.exit(f"--factor must be a power of two >= 2 (got {args.factor}).")

    stack, _ = load_stack(args.input)
    stack = VideoUtils.center_crop(stack, args.crop)
    n = len(stack)
    min_needed = 3 if args.mode == "algos" else 2
    if n < min_needed:
        sys.exit(f"Need at least {min_needed} frames for --mode {args.mode}; found {n} in {args.input}.")

    print(f"Input:   {args.input}  ({n} frames, {stack.dtype}, {stack.shape[1]}x{stack.shape[2]})")
    os.makedirs(args.out, exist_ok=True)

    demo = DemoGenerator(args.out, args.fps)
    if args.mode == "algos":
        demo.run_algos(stack, args.backends)
    else:
        demo.run_single(stack, args.backend, args.factor, args.per_frame_norm)


if __name__ == "__main__":
    main()
