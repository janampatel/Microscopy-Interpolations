"""Download and cache the files not shipped with the package, under
$MICROINTERP_HOME/weights (default ~/.cache/microinterp/weights).
"""
import os
import urllib.request
from pathlib import Path

FILM_URL = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt"
CDFI_SHA = "537291c7adc78117e50a31628f143aee9263ce20"
CDFI_URL = f"https://github.com/tding1/CDFI/raw/{CDFI_SHA}/checkpoints/CDFI_adacof.pth"
# one file instead of the whole CDFI repo (~270 MB)
CDFI_OP_URL = f"https://raw.githubusercontent.com/tding1/CDFI/{CDFI_SHA}/cupy_module/adacof.py"
LDMVFI_GDRIVE_ID = "1_Xx2fBYQT9O-6O3zjzX76O9XduGnCh_7"

# name -> (filename, direct url, google-drive id)
WEIGHTS = {
    "film": ("film_net_fp32.pt", FILM_URL, None),
    "cdfi": ("CDFI_adacof.pth", CDFI_URL, None),
    "cdfi_op": ("CDFI_adacof_op.py", CDFI_OP_URL, None),
    "ldmvfi": ("ldmvfi-vqflow-f32-c256-concat_max.ckpt", None, LDMVFI_GDRIVE_ID),
}


def cache_dir():
    """Directory where downloaded weights are stored."""
    root = os.environ.get("MICROINTERP_HOME")
    base = Path(root) if root else Path.home() / ".cache" / "microinterp"
    d = base / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def engine_dir():
    """Directory where `microinterp setup` puts the LDMVFI engine code."""
    return cache_dir().parent / "engine"


def _fetch(url, dest, desc):
    from tqdm import tqdm
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f, \
            tqdm(total=int(r.headers.get("Content-Length") or 0) or None,
                 unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            bar.update(len(chunk))


def ensure(name):
    """Return the local path to a weight, downloading it first if needed."""
    filename, url, gdrive_id = WEIGHTS[name]
    path = cache_dir() / filename
    if path.exists() and path.stat().st_size > 0:
        return path

    # write to .part and rename, so an interrupted download is never mistaken for a valid file
    part = path.with_name(filename + ".part")
    source = url or f"https://drive.google.com/file/d/{gdrive_id}"
    print(f"Downloading {filename} ...")
    try:
        if url:
            _fetch(url, part, filename)
        else:
            import gdown
            gdown.download(id=gdrive_id, output=str(part), quiet=False)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {filename} ({e}).\n"
                           f"Download it manually from {source}\nand save it as {path}") from e

    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {filename}.\n"
                           f"Download it manually from {source}\nand save it as {path}")
    part.replace(path)
    return path
