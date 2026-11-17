"""Backend-agnostic interpolation driver. Real frames pass through unchanged and
the input dtype and value range are preserved.
"""
from typing import List, Tuple
import copy
from queue import Queue, Empty

import numpy as np
import torch
from torch.nn import functional as F

from .backends import (
    load_model as _load_backend_model,
    weights_loaded as _weights_loaded,
)
from .backends.rife.pytorch_msssim import ssim_matlab


def get_interpolated_length(t: int, intervals: List[dict]) -> int:
    """Number of output frames for `t` input frames and the given intervals."""
    if not intervals:
        return t

    sorted_intervals = sorted(intervals, key=lambda x: x['start'])
    total_frames = 0
    last_pos = 0
    for interval in sorted_intervals:
        start = interval['start']
        stop = interval['stop']
        interp_factor_log2 = int(interval['interp'])
        # Gap frames before this interval.
        total_frames += (start - last_pos)
        # Interpolated frames within the interval (both endpoints included).
        num_original_frames = stop - start + 1
        num_interpolated_frames = num_original_frames + (num_original_frames - 1) * (2**interp_factor_log2 - 1)
        total_frames += num_interpolated_frames
        last_pos = stop + 1
    # Remaining frames after the last interval.
    total_frames += (t - last_pos)
    return int(total_frames)


def describe_layout(num_input_frames, intervals):
    """Describe every output frame of an interpolation run WITHOUT running a model.

    Returns a list (parallel to the output series) of dicts:
      {'role': 'original',     'src': <input frame index>}
      {'role': 'interpolated', 'between': [a, b], 't': <fraction in (0,1)>}
    Mirrors the frame placement of interpolate_intervals().
    """
    ivs = sorted(intervals, key=lambda x: x['start']) if intervals else []
    layout = []
    original_pos = 0
    interval_index = 0
    while original_pos < num_input_frames:
        if interval_index < len(ivs) and original_pos == ivs[interval_index]['start']:
            start = int(ivs[interval_index]['start'])
            stop = int(ivs[interval_index]['stop'])
            interp = int(ivs[interval_index]['interp'])
            step = 2 ** interp
            n_added = (stop - start) * step + 1
            for j in range(n_added):
                if j % step == 0:
                    layout.append({'role': 'original', 'src': start + j // step})
                else:
                    a = start + j // step
                    layout.append({'role': 'interpolated', 'between': [a, a + 1], 't': round((j % step) / step, 4)})
            original_pos = stop + 1
            interval_index += 1
        else:
            layout.append({'role': 'original', 'src': original_pos})
            original_pos += 1
    return layout


def rife_weights_loaded():
    """True if the RIFE weight file is present."""
    return _weights_loaded("rife")


def film_weights_loaded():
    """True if the FILM (TorchScript) backend weight file is present."""
    return _weights_loaded("film")


def ldmvfi_weights_loaded():
    """True if the LDMVFI (diffusion) checkpoint + config are present."""
    return _weights_loaded("ldmvfi")


def cdfi_weights_loaded():
    """True if the CDFI checkpoint is present."""
    return _weights_loaded("cdfi")


def _denormalize(arr01, lo, hi, out_dtype):
    """Map floats in [0, 1] back to the input value range and dtype (exact for real frames)."""
    rng = float(hi - lo)
    if rng == 0:
        rng = 1.0
    arr = np.clip(arr01, 0.0, 1.0).astype(np.float32) * rng + float(lo)
    out_dtype = np.dtype(out_dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        arr = np.clip(np.rint(arr), info.min, info.max)
    return arr.astype(out_dtype)


def load_model(backend: str = 'rife') -> Tuple[object, str]:
    """Return (model, device) for `backend`; see `microinterp.backends.load_model`."""
    return _load_backend_model(backend)


def interpolate_intervals(ims: np.ndarray, intervals: List[dict], backend: str = 'rife', progress_callback=None) -> Tuple[np.ndarray, List[dict]]:
    """Interpolate `ims` over an arbitrary number of intervals.

    Parameters
    ims (np.ndarray): an (n x h x w) stack of frames to interpolate.
    intervals (list of dict): each with keys `start`/`stop`/`interp` where
        `interp` is a log2 factor in [1,4]. If empty, no interpolation is done.
    backend (str): 'rife' (default), 'film', 'cdfi', or 'ldmvfi'.

    Returns
    interpolated (np.ndarray): result stack, same dtype/H/W as input; real
        frames are preserved bit-exactly.
    new_intervals (List[dict]): where interpolated frames live in the output.
    """
    # Global min/max for consistent normalization across ALL frames.
    global_min = ims.min()
    global_max = ims.max()

    model, device = load_model(backend)
    single_im_shape = ims[0].shape
    new_length = get_interpolated_length(len(ims), intervals)
    template = np.zeros((new_length, single_im_shape[0], single_im_shape[1]), dtype=ims.dtype)

    template_pos = 0  # position in the output template
    original_pos = 0
    interval_index = 0
    original_len = len(ims)
    new_intervals = []

    while original_pos < original_len:
        if interval_index < len(intervals) and original_pos == intervals[interval_index]['start']:
            start, stop, interp = intervals[interval_index]["start"], intervals[interval_index]["stop"], int(intervals[interval_index]["interp"])
            interval_frames = ims[start:stop+1]

            # Pass global min/max so normalization is consistent across intervals.
            interval_interp = interpolate(interval_frames, interp, device, model, global_min, global_max,
                                          include_last=True, progress_callback=progress_callback, out_dtype=ims.dtype)

            num_frames_added = interval_interp.shape[0]
            template[template_pos:template_pos+num_frames_added] = interval_interp

            # put the originals back bit-exactly
            step = 2 ** int(interp)
            for k in range(stop - start + 1):
                pos = template_pos + k * step
                if pos < template_pos + num_frames_added:
                    template[pos] = ims[start + k]

            new_intervals.append({"start": template_pos, "stop": template_pos+num_frames_added,
                            "interp": int(interp)})

            template_pos += (num_frames_added)
            original_pos = stop + 1  # skip past the interval's last frame
            interval_index += 1
        else:
            # Preserve non-interpolated (real) frames EXACTLY.
            template[template_pos] = ims[original_pos]
            original_pos += 1
            template_pos += 1

    del model
    template = template[:template_pos]
    return template, new_intervals


def interpolate(ims: np.ndarray, exp: int, device, model, global_min, global_max,
                include_last=True, progress_callback=None, out_dtype=None) -> np.ndarray:
    """Interpolate all of `ims` at scale `exp` (2^exp frames per pair).

    Parameters
    ims (np.ndarray): images of shape (n x r x c).
    exp (int): interpolation level.
    device, model: from `load_model`.
    global_min/global_max: value range for consistent reversible normalization.

    Returns the interpolated stack (including real frames) in the caller's
    original dtype/range.
    """
    scale = 1

    def make_inference(model, I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(model, I0, middle, n=n//2)
        second_half = make_inference(model, middle, I1, n=n//2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(img):
        tmp = 32
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return F.pad(img, padding)

    # reversible normalization; guard against a constant image
    global_min = ims.min()
    global_max = ims.max()
    _range = float(global_max - global_min)
    if _range == 0:
        _range = 1.0
    if out_dtype is None:
        out_dtype = ims.dtype

    def reformat(im):
        # float32 in [0, 1], reversible
        clipped = np.clip(im.astype(np.float32), global_min, global_max)
        im_normalized = (clipped - global_min) / _range
        return np.stack((im_normalized, im_normalized, im_normalized), axis=-1)

    def make_read_buffer(ims):
        read_buffer = Queue(maxsize=500)
        for im in ims:
            read_buffer.put(im)
        return read_buffer

    # Save the last image (normalized) to append after the loop.
    if include_last:
        final_im = copy.deepcopy(ims[-1])
        clipped = np.clip(final_im.astype(np.float32), global_min, global_max)
        final_im = (clipped - global_min) / _range

    ims = [reformat(im) for im in ims]

    lastframe = ims[0]
    h, w, _ = lastframe.shape
    ims = ims[1:]

    write_buffer = []
    read_buffer = make_read_buffer(ims)

    I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
    I1 = pad_image(I1)

    temp = None  # holds lastframe when processing a static frame
    frame_count = 0

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            try:
                frame = read_buffer.get(False)
            except Empty:
                break

        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            # Static frame: pull the next frame instead of interpolating motion.
            try:
                frame = read_buffer.get(False)
                temp = frame
            except Empty:
                break_flag = True
                frame = lastframe

            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = I1[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]  # float32 [0,1]

        if ssim < 0.2:
            # Wildly different frames: duplicate rather than hallucinate motion.
            output = []
            for _ in range((2 ** exp) - 1):
                output.append(I0)
        else:
            output = make_inference(model, I0, I1, 2**exp-1) if exp else []

        write_buffer.append(lastframe)

        for mid in output:
            mid = mid[0].cpu().numpy().transpose(1, 2, 0)  # float32 [0,1]
            cropped = mid[:h, :w]
            write_buffer.append(cropped)
            if progress_callback is not None:
                progress_callback()

        lastframe = frame
        if break_flag:
            break
        frame_count += 1

    result_list = [im[:, :, 0] for im in write_buffer]

    if include_last:
        result01 = np.array(result_list + [final_im], dtype=np.float32)
    else:
        result01 = np.array(result_list, dtype=np.float32)

    # back to the caller's dtype and value range
    return _denormalize(result01, global_min, global_max, out_dtype)
