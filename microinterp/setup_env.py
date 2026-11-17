"""`microinterp setup`: detect this machine and install what each backend needs.

Runs pip in the current environment. torch is only ever inspected in a
subprocess, so it can be replaced safely (Windows locks loaded DLLs).
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from importlib import metadata, util

from . import weights

# LDMVFI engine: repo, pinned commit (tested), folders to keep
ENGINE = {
    "LDMVFI": ("danier97/LDMVFI", "eee2dc3566f21f22787d66b0f0337e48318403c6", ("ldm", "cupy_module")),
    "taming-transformers": ("CompVis/taming-transformers", "3ba01b241669f5ade541ce990f7650a3b8f65318", ("taming",)),
    "CLIP": ("openai/CLIP", "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", ("clip",)),
}
# newest PyTorch CUDA wheel index the NVIDIA driver can run
TORCH_INDEX = [((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print("  $ " + " ".join(cmd))
    subprocess.check_call(cmd)


def _driver_cuda():
    """Highest CUDA version the NVIDIA driver supports, or None without an NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _torch_info():
    """(version, cuda) of the installed torch; cuda is '' for a CPU build."""
    code = "import torch; print(torch.__version__); print(torch.version.cuda or '')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split("\n")
    return out[0].strip(), out[1].strip() if len(out) > 1 else ""


def _torch_sees_gpu():
    out = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                         capture_output=True, text=True).stdout.split()
    return bool(out) and out[-1] == "True"


def _gpu_requirements():
    reqs = metadata.requires("microinterp") or []
    return [r.split(";")[0].strip() for r in reqs if 'extra == "gpu"' in r]


def _fetch_engine():
    for name, (repo, sha, keep) in ENGINE.items():
        dest = weights.engine_dir() / name
        if all((dest / k).is_dir() for k in keep):
            continue
        print(f"  fetching {repo} @ {sha[:7]}")
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zip")
            urllib.request.urlretrieve(f"https://codeload.github.com/{repo}/zip/{sha}", src)
            with zipfile.ZipFile(src) as z:
                top = z.namelist()[0].split("/")[0]
                wanted = keep + ("LICENSE", "License.txt")
                z.extractall(tmp, [m for m in z.namelist() if m.split("/")[1:2] and m.split("/")[1] in wanted])
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(os.path.join(tmp, top), dest)


def run(yes=False):
    driver = _driver_cuda()
    tver, tcuda = _torch_info()
    via_torch = not driver and bool(tcuda) and _torch_sees_gpu()
    if via_torch:  # nvidia-smi is broken or missing, but torch already runs on the GPU
        driver = tuple(int(x) for x in tcuda.split(".")[:2])
    note = " (nvidia-smi unavailable, using PyTorch)" if via_torch else ""
    print(f"System : {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print("GPU    : " + (f"NVIDIA, driver supports CUDA {driver[0]}.{driver[1]}{note}" if driver else "no NVIDIA GPU found"))
    print(f"torch  : {tver} ({'CUDA ' + tcuda if tcuda else 'CPU build'})")

    if not driver:
        print("\nRIFE and FILM run on this machine (CPU). CDFI and LDMVFI need an NVIDIA GPU.")
        weights.ensure("film")
        print("\nSetup complete. Check with: microinterp list")
        return

    # pip's default Linux torch can be newer than the driver supports
    torch_ok = bool(tcuda) and tuple(map(int, tcuda.split(".")[:2])) <= driver
    index = f"cu{tcuda.replace('.', '')}" if torch_ok else next((i for v, i in TORCH_INDEX if driver >= v), None)
    if index is None:
        sys.exit("The NVIDIA driver is too old (needs CUDA 11.8 or newer). Update it and rerun `microinterp setup`.")
    url = f"https://download.pytorch.org/whl/{index}"
    plan = [f"install CUDA torch + torchvision from {url}"] if not torch_ok else []
    if torch_ok and not util.find_spec("torchvision"):
        plan.append(f"install torchvision matching torch {tver} from {url}")
    plan += ["install cupy and the LDMVFI Python dependencies",
             f"download the LDMVFI engine code into {weights.engine_dir()}",
             f"download FILM, CDFI and LDMVFI weights and the CDFI AdaCoF op into {weights.cache_dir()}"]
    print("\nPlan:\n" + "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(plan)))
    if not yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("Aborted, nothing changed.")

    if not torch_ok:
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])
        _pip("torch", "torchvision", "--index-url", url)
        tver, tcuda = _torch_info()
    elif not util.find_spec("torchvision"):
        _pip(f"torch=={tver.split('+')[0]}", "torchvision", "--index-url", url)
    major = tcuda.split(".")[0]
    if major not in ("11", "12", "13"):
        sys.exit(f"torch uses CUDA {tcuda}; CDFI/LDMVFI need a torch build for CUDA 11.x, 12.x or 13.x.")
    # CuPy compiles kernels against the CUDA headers; torch ships them on Linux, not Windows
    rt = "nvidia-cuda-runtime" if major == "13" else f"nvidia-cuda-runtime-cu{major}"
    # pinning torch keeps pip from swapping in a CPU build from PyPI
    _pip(f"torch=={tver.split('+')[0]}", "numpy<2", f"cupy-cuda{major}x<14",
         f"{rt}=={tcuda}.*", *_gpu_requirements())
    _fetch_engine()
    for b in ("film", "cdfi", "cdfi_op", "ldmvfi"):
        weights.ensure(b)
    print("\nSetup complete. Check with: microinterp list")
