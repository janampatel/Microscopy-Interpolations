# Notice

microinterp is released under the MIT License (see [`LICENSE`](LICENSE)).
It wraps four published frame-interpolation methods. Their code and weights
remain under their original licenses, listed below. Full license texts
are in [`licenses/`](licenses/). Please also cite the original papers
(see the README) when you use a backend.

## filai

- Source: https://github.com/vatsal-dp/filogui3 (PyPI: `filai`)
- Copyright: 2026 North Carolina State University
- License: BSD-3-Clause ([text](licenses/FILAI_LICENSE.txt))
- Files: `microinterp/gui/`, `microinterp/core.py`, `microinterp/backends/rife/` (adapted)

## RIFE

- Source: https://github.com/megvii-research/ECCV2022-RIFE
- Copyright: Megvii Inc.
- License: MIT ([text](licenses/RIFE_LICENSE.txt))
- Files: `microinterp/backends/rife/` (code, adapted); `microinterp/backends/rife/flownet.pkl`
  (the official RIFE model weights, unchanged)

## pytorch-msssim

- Source: https://github.com/jorge-pessoa/pytorch-msssim
- License: MIT ([text](licenses/PYTORCH_MSSSIM_LICENSE.txt))
- Files: `microinterp/backends/rife/pytorch_msssim/` (adapted)

## FILM

- Source: https://github.com/google-research/frame-interpolation; TorchScript port:
  https://github.com/dajes/frame-interpolation-pytorch
- License: Apache-2.0 ([text](licenses/APACHE-2.0_FILM.txt))
- Not redistributed: the weights (`film_net_fp32.pt`) are downloaded at first use from the
  GitHub release of the port.

## CDFI

- Source: https://github.com/tding1/CDFI
- Copyright: 2021 Tianyu Ding, Luming Liang, Zhihui Zhu, Ilya Zharkov
- License: MIT ([text](licenses/CDFI_LICENSE.txt)), granted in writing by the CDFI authors
  to the microinterp maintainers (the repository itself has no license file)
- Files: `microinterp/backends/cdfi/` (code, adapted)
- Not redistributed: the weights (`CDFI_adacof.pth`) and the AdaCoF CUDA op are
  downloaded at first use from the CDFI repository.

## LDMVFI

- Source: https://github.com/danier97/LDMVFI
- Copyright: 2023 danielism97
- License: MIT ([text](licenses/LDMVFI_LICENSE.txt))
- Files: `microinterp/backends/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.yaml` (configuration)
- Not redistributed: the checkpoint is downloaded at first use from the authors' Google Drive
  link. `microinterp setup` also downloads the LDMVFI engine code and its dependencies
  taming-transformers and CLIP (both MIT).

## Sample data

The frames in `microinterp/data/toy_dataset/` were acquired by the Miranda Lab, North Carolina
State University.