# Third-party notices

The code written for PACE is released under the MIT licence in `LICENSE`.
This file lists the third-party code, models and tools that PACE uses, and the
licence of each. Some of them are not MIT licensed; their terms apply to the
parts that use or derive from them, and the MIT licence of this repository does
not change those terms.

## Repositories required at runtime and not included

These repositories are not vendored. Clone them next to this repository or
point the environment variables below at them.

| Repository | Licence | Used for | How PACE finds it |
|---|---|---|---|
| [NVlabs/edm](https://github.com/NVlabs/edm) | CC BY-NC-SA 4.0 | unpickling the EDM teacher checkpoints (`torch_utils`, `dnnlib`); the layers of every U-Net student (`training.networks`); `fid.py` for the FFHQ NVLabs FID protocol | `$EDM_REPO`, a sibling `../edm` checkout, or `PYTHONPATH` (`pace/external_repos.py`) |
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | CC BY-NC 4.0 | the DiT building blocks reused by `NarrowDiT` (`models.py`), the DDPM sampler (`diffusion/`), and the DiT-XL/2 checkpoint | `$DIT_REPO` or the `--dit_repo` flag of the DiT scripts (`pace/external_repos.py`) |

The pretrained EDM checkpoints published by NVIDIA are also licensed under
CC BY-NC-SA 4.0, and the released DiT-XL/2 checkpoint is licensed under
CC BY-NC 4.0. Both licences restrict use to non-commercial purposes.

## Code in this repository adapted from those repositories

The following parts of this repository are adapted from, or are structural
re-implementations of, code in the repositories above. The licence of the
original applies to them.

| Location | Origin | Licence |
|---|---|---|
| `pace/edm_augment.py` | port of `training/augment.py` from NVlabs/edm | CC BY-NC-SA 4.0 |
| `edm_sampler` in `scripts/sample_edm_distilled.py` | adapted from the EDM Heun sampler in `generate.py` of NVlabs/edm | CC BY-NC-SA 4.0 |
| `NarrowSongUNet` and `NarrowEDMPrecond` in `pace/edm_distillation.py` | variable-width re-implementation of `SongUNet` and the EDM preconditioning, built from `training.networks` layers of NVlabs/edm | CC BY-NC-SA 4.0 |
| `NarrowDiT` and `NarrowDiTBlock` in `pace/dit_arch_alloc.py` | variable-width re-implementation of the DiT block and model, reusing components of `models.py` of facebookresearch/DiT | CC BY-NC 4.0 |

## Vendored code

These files are copied into `pace/vendor/` with small compatibility changes
that each licence file describes. They keep their original licences.

| File | Origin | Licence |
|---|---|---|
| `pace/vendor/openai_consistency_unet.py` | [openai/consistency_models](https://github.com/openai/consistency_models) at commit `e32b69ee436d518377db86fb2127a3972d0d8716` | MIT, see `pace/vendor/OPENAI_CONSISTENCY_MODELS_LICENSE.md` |
| `pace/vendor/diffwave_legacy.py` | [albertfgu/diffwave-sashimi](https://github.com/albertfgu/diffwave-sashimi) at commit `9bd78f8c894cad0952a5692450f2145e24466b29` | MIT, see `pace/vendor/DIFFWAVE_SASHIMI_LICENSE.md` |

## External tools

| Tool | Licence | Used for |
|---|---|---|
| [openai/guided-diffusion](https://github.com/openai/guided-diffusion) `evaluations/evaluator.py`, commit `22e0df8183507e13a7813f8d38d51b072ca1e67c` | MIT | FID, sFID, precision, recall and Inception Score of the ADM protocols (LSUN Bedroom, FFHQ custom suite, DiT evaluations). It is called as a separate process through `--adm-evaluator`/`--adm_dir` and is not included. Its reference batches and Inception graph keep their upstream terms. |
| [clean-fid](https://github.com/GaParmar/clean-fid) | MIT | Clean-FID for CIFAR-10 and ImageNet-64 |

## Checkpoints and datasets

Checkpoints and datasets are downloaded by the user and are not part of this
repository. The teacher checkpoints are the NVLabs EDM releases (CIFAR-10,
ImageNet-64, FFHQ-64; CC BY-NC-SA 4.0), the OpenAI consistency-models LSUN
Bedroom EDM checkpoint, the DiffWave SC09 checkpoint of albertfgu/diffwave-sashimi,
the facebookresearch/DiT DiT-XL/2 checkpoint (CC BY-NC 4.0) and the
`normalcomputing/dit-cifar10-32x32-class` checkpoint on Hugging Face (its
licence is not stated by its publisher). Dataset terms: CIFAR-10 (as published
by its authors), ImageNet (research terms of the ImageNet project), FFHQ
(CC BY-NC-SA 4.0 at the dataset level, with per-image licences), LSUN (no
redistribution grant), Speech Commands SC09 (CC BY 4.0).

This summary is not legal advice. Review the upstream licences before any use
beyond non-commercial research.
