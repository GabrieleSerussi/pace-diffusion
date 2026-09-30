# Reproducing the paper

This document maps each result of "Different Noise Levels, Different Needs:
Discovering Diffusion Phases for Efficient Distillation" to the commands that
produce it, lists the inputs that you provide, and records where the runs
behind the paper differ from its text. All commands run from the repository
root.

The scripts in `reproduce/` print every command before they run it.
`DRY_RUN=1 bash reproduce/<script>.sh` prints the whole command list without
running anything, and `STAGES="group allocate"` runs a subset of stages. The
header of each script lists its stages, marks the ones that need a GPU, and
names the inputs it reads. Paths default to directories inside the repository
(`data/`, `checkpoints/`, `references/`, `external/`, `outputs/`); the
variables `DATA_ROOT`, `MODEL_CACHE`, `REFERENCE_DIR`, `ADM_DIR` and
`OUTPUT_ROOT` move them (see `reproduce/common.sh`).

1. [Installation](#1-installation)
2. [CPU check of the phases and budgets](#2-cpu-check-of-the-phases-and-budgets)
3. [Paper results and commands](#3-paper-results-and-commands)
4. [Pipeline scripts](#4-pipeline-scripts)
5. [Inputs that you provide](#5-inputs-that-you-provide)
6. [What the release does not contain](#6-what-the-release-does-not-contain)

## 1. Installation

Python 3.11 or 3.12. Install a CUDA build of PyTorch first when you need one
(https://pytorch.org/get-started/locally/), then:

```bash
pip install -e ".[all]"
```

The extras are `dit` (timm, diffusers), `eval` (clean-fid), `audio`
(torchaudio, torchcodec, which need the FFmpeg shared libraries), `viz`
(plotly), `train` (TensorBoard) and `dev` (pytest).

Two repositories are needed at run time and are not included (their licences
are in `THIRD_PARTY_NOTICES.md`):

```bash
git clone https://github.com/NVlabs/edm ../edm
git -C ../edm checkout 008a4e5316c8e3bfe61a62f874bddba254295afb
git clone https://github.com/facebookresearch/DiT ../DiT
git -C ../DiT checkout ed81ce2229091fd4ecc9a223645f95cf379d582b
export EDM_REPO="$PWD/../edm" DIT_REPO="$PWD/../DiT"
```

NVlabs/edm unpickles the EDM teachers and provides the layers of every U-Net
student; the code finds it through `EDM_REPO`, a sibling `../edm` checkout or
`PYTHONPATH`. facebookresearch/DiT provides the DiT blocks, the DDPM sampler
and the DiT-XL/2 checkpoint; the DiT scripts find it through `DIT_REPO` or
`--dit_repo`. Without either repository, `import pace`, the CPU check below
and most of the test suite still work.

The ADM evaluator of openai/guided-diffusion computes FID, sFID, precision,
recall and Inception Score for FFHQ-64 (ADM metrics), LSUN Bedroom and all DiT
experiments. It runs on the CPU in a separate TensorFlow environment:

```bash
git clone https://github.com/openai/guided-diffusion external/guided-diffusion
git -C external/guided-diffusion checkout 22e0df8183507e13a7813f8d38d51b072ca1e67c
python3 -m venv external/adm-env
external/adm-env/bin/pip install -r reproduce/requirements-adm.txt
export ADM_PYTHON="$PWD/external/adm-env/bin/python"
```

`external/guided-diffusion/evaluations` is the default `ADM_DIR`. The U-Net
evaluations also read the Inception graph at
`references/classify_image_graph_def.pb` (`ADM_DETECTOR`), from
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb.

The tests run on a CPU:

```bash
pytest                  # tests that need DiT, EDM, bash or torchcodec skip with a reason
pytest -m "not slow"    # a faster subset
```

## 2. CPU check of the phases and budgets

```bash
python reproduce/cpu_phases_and_budgets.py
```

The script runs the phase discovery (`scripts/optimize_timestep_grouping.py`)
and the U-Net allocator (`scripts/dry_run_capacity_allocation.py`) of this
repository on the released profiles in `artifacts/`, and checks 38 values
against the released groupings and allocations and against the paper. It needs
no GPU and no external repository, and runs in under a minute:

| Result | Values checked |
|---|---|
| Section 3.3, Appendix C | automatic K selects `[0,8,20]` (CIFAR-10, J = 15.149346), `[0,16,20]` (ImageNet-64, J = 15.308086), `[0,16,20]` (LSUN Bedroom, J = 15.666175) and `[0,4,8,16,20]` (DiT-Micro, J = 13.345745); on FFHQ-64 it selects one phase, and the released phases `[0,3,16,20]` are the K = 3 optimum of `--builtin_cost matrix_correlation` (cost 0.207837) and of the objective with lambda_sep = 0 |
| Table 3, Appendix C | within-phase correlations 0.912 / 0.923 and cross 0.363 (CIFAR-10); 0.904 / 0.951 and 0.429 (ImageNet-64); 0.917 / 0.984 / 0.982, middle-late 0.884 and early-late 0.554 (FFHQ-64); 0.950 / 0.940 and 0.547 (LSUN Bedroom); 0.956 / 0.832 / 0.914 / 0.928, early-third -0.146 and early-fourth -0.189 (DiT-Micro) |
| Section 3.4 | phase budgets of the Phase-aware students: CIFAR-10 5,406,655.84 / 45,619,011.16 of P_tot = 51,025,667; ImageNet-64 183,586,906.62 / 83,231,272.38 of 266,818,179; FFHQ-64 5,795,995.43 / 39,836,608.68 / 10,669,350.89 of 56,301,955; LSUN Bedroom 317,097,723.42 / 181,240,327.58 of 498,338,051 (Horvitz-Thompson weighted); the Global and Uniform budgets equal the released ones, and every layer budget of the layerwise students agrees with the released one to 1e-6 parameters |
| Appendix B | phases `[0,13,17,20]`, similarities 0.9915 / 0.9434 / 0.9118 within and 0.8153 / 0.2683 / 0.6926 across, capacity shares 0.2339 / 0.3042 / 0.4619 |

Each experiment script also runs these CPU stages on the released profile, for
example:

```bash
STAGES="group allocate" PROFILE_JSON=artifacts/profiles/cifar10_ddpmpp_random_same_norm.json.gz \
  bash reproduce/cifar10_unet.sh
```

## 3. Paper results and commands

The U-Net scripts train from the released plans of the trained students
(`artifacts/plans/<dataset>/`); set `PLAN_ROOT` to train from plans that you
rebuild with the `plans` stage.

| Paper result | Command | Notes |
|---|---|---|
| Figure 1 | `bash reproduce/figure1.sh` | The paper figure is `outputs/figures/figure1/seed_candidates_8_17/figure1_empirical_clean_labels_seed11.pdf`. The correlation matrix and the phases come from the released FFHQ-64 profile and grouping. The `trajectory` stage uses a GPU by default; the other stages run on the CPU. |
| Figure 3 | no single command | The paper figure was assembled outside this repository. `scripts/paper/plot_ffhq_benchmark_pareto.py` and `scripts/paper/plot_lsun_benchmark_pareto.py` (the `summarize` stages of the FFHQ-64 and LSUN Bedroom U-Net scripts) plot the measured metrics against throughput. |
| Tables 1 and 2, ConvNet, CIFAR-10 | `bash reproduce/cifar10_unet.sh` | The `report` stage prints the lowest Clean-FID-5k of each student's training monitor (5,000 samples, 18 Heun steps) and its step; in the recorded runs these minima are at steps 64,000 to 69,000. There is no separate 50k evaluation and no throughput measurement for CIFAR-10. |
| Tables 1 and 2, ConvNet, ImageNet-64 | `bash reproduce/imagenet64_unet.sh` | The `evaluate` stage computes Clean-FID-50k of the best-FID checkpoint and, in a second run of 5,000 samples, the throughput. |
| Tables 1 and 2, ConvNet, FFHQ-64 | `bash reproduce/ffhq64_unet.sh` | NVLabs FID-50k with three seeds (the recorded tables use the minimum), ADM precision, recall and Inception Score on the images of the first seed, and throughput. `summarize` writes `outputs/ffhq64_unet/pareto/pareto_metrics.csv`. |
| Tables 1 and 2, ConvNet, LSUN Bedroom | `bash reproduce/lsun_bedroom256_unet.sh` | ADM FID-50k and the other ADM metrics with a mixed FP16 teacher and students (40 Heun steps, seed 2100000), and throughput. `summarize` writes `outputs/lsun_bedroom256_unet/pareto/pareto_metrics.csv`. |
| Tables 1 and 2, DiT, ImageNet | `bash reproduce/dit_imagenet256.sh` | DiT-XL/2 teacher, 2 phases, ADM FID-10k (DDPM with 250 steps, guidance 1.5). The paper's Global student is step 800,000. |
| Tables 1 and 2, DiT, FFHQ | `bash reproduce/dit_ffhq256.sh` | DiT-B/2 teacher trained from scratch, 3 phases, ADM FID-10k (DDPM with 250 steps). The paper uses step 150,000. |
| Tables 1 and 2, DiT, LSUN Bedroom | `bash reproduce/dit_lsun256.sh` | DiT-B/2 teacher trained from scratch, 3 phases, ADM FID-10k (DDPM with 1,000 steps). The paper uses steps 235,000 to 245,000. |
| Tables 1 and 2, DiT, CIFAR-10 | `bash reproduce/dit_cifar10.sh` | Pixel-space DiT teacher trained from scratch with the EDM formulation, 2 phases, ADM FID-10k (18 Heun steps, guidance 1.25). The paper uses step 306,000. |
| Table 1, DyDiT, TinyFusion and ALTER rows | none | The baselines are not part of this repository. |
| Figure 4, Appendix B | `bash reproduce/sc09_diffwave.sh` | `STAGES=figure` renders Figure 4 on the CPU from `artifacts/audio/audio_appendix_metrics.json`. The full pipeline verifies the teacher against the upstream sampler (1 GPU), profiles all 37,377 filters with PFI and refreshes the metrics before plotting. |
| Figure 5, Table 3, Appendix C.1 | `python reproduce/cpu_phases_and_budgets.py` | From the released profiles. The `group` stage of each U-Net script writes the correlation matrix with the phase boundaries (`timestep_grouping_matrix.png`); the script that drew the paper figure is not part of the release. A new profile comes from the `profile` stage. |
| Figure 6, Appendix C.2 | `bash reproduce/dit_micro_cifar10.sh` | Profile (2 GPUs) and phase discovery of the archived DiT-Micro teacher; the statistics are part of the CPU check. |
| Section 3.4 budgets | `python reproduce/cpu_phases_and_budgets.py` | The `allocate` stage of each U-Net script writes the same allocations. |
| Appendix D, Figures 7 to 14 | see notes | The grids were assembled outside this repository. `scripts/sample_edm_distilled.py --snapshot <checkpoint> --output-dir <directory> --seeds 0-9` samples a U-Net student, and `scripts/evaluate_students.py` samples a DiT teacher or student. |

The recorded U-Net throughputs were measured on NVIDIA RTX PRO 6000 Blackwell
GPUs: FFHQ-64 with 4 GPUs and 32 images per GPU, LSUN Bedroom with 2 GPUs and
32 images per GPU, and ImageNet-64 with 1 GPU and 16 images per batch. The
DiT-XL/2 students were trained on 8 NVIDIA B200 GPUs, and the DiffWave profile
ran on 8 NVIDIA L40S GPUs. Throughput depends on the GPU and the software
versions.

## 4. Pipeline scripts

In pipeline order, with the stage of the paper each one implements:

| Step | Script | Paper |
|---|---|---|
| Data | `scripts/data/prepare_ffhq_dataset.py`, `scripts/data/preflight_edm_dataset.py`, `scripts/data/convert_imagenet_parquet_to_webdataset.py`, `scripts/data/prepare_sc09_dataset.py`, `scripts/data/precompute_latent_cache.py` | Appendix A.1 |
| References | `scripts/data/build_ffhq_adm_reference.py`, `scripts/data/preflight_adm_evaluator.py`, `scripts/data/extract_fid_reference_pngs.py`, `scripts/data/pack_pngs_to_npz.py` | Appendix A.2 |
| Profile | `scripts/evaluate_parameters_edm.py` (U-Nets), `scripts/evaluate_parameters_dit.py` (latent DiTs), `scripts/evaluate_parameters_dit_micro.py` (pixel DiTs), `scripts/paper/evaluate_parameters_diffwave.py` and `scripts/paper/postprocess_diffwave_residual.py` (DiffWave) | Section 3.2 |
| Discover phases | `scripts/optimize_timestep_grouping.py` | Section 3.3 |
| Allocate | `scripts/dry_run_capacity_allocation.py` (U-Nets), `scripts/dit_arch_to_plans.py` (DiTs) | Section 3.4 |
| Build students | `scripts/prepare_edm_distillation.py` | Appendix A.4 |
| Train | `scripts/train_edm_distillation.py` (U-Nets), `scripts/train_phase_students.py` (DiT students and the DiT-B/2 and CIFAR-10 DiT teachers) | Section 3.5, Appendix A.5 |
| Sample | `scripts/sample_edm_distilled.py` (U-Nets), `scripts/evaluate_students.py` (DiTs) | Appendix A.5 |
| Evaluate | `scripts/evaluate_edm_checkpoint.py`, `scripts/paper/summarize_edm_benchmark.py`, `scripts/paper/finalize_edm_benchmark_artifacts.py` (U-Nets), `scripts/eval_composite_curve.py` (DiTs) | Appendix A.2 |
| Throughput | measured by `scripts/evaluate_edm_checkpoint.py` (U-Nets), `scripts/bench_throughput.py` (DiTs) | Appendix A.2.3 |
| Plot | `scripts/paper/plot_ffhq_benchmark_pareto.py`, `scripts/paper/plot_lsun_benchmark_pareto.py`, `scripts/paper/plot_audio_appendix.py`, `scripts/paper/figures/` (Figure 1) | Figures 1, 3 and 4 |
| Audio checks | `scripts/paper/verify_diffwave_teacher.py`, `scripts/paper/compare_diffwave_filter_stability.py`, `scripts/paper/report_diffwave_per_filter_decision.py` | Appendix B |
| Utilities | `scripts/paper/slim_profile.py` (released profile format); `scripts/paper/run_edm_benchmark_pipeline.py` (a resumable driver with preflight checks for the FFHQ-64 and LSUN Bedroom pipelines; its training settings are not those of the recorded runs, which the `reproduce/` scripts use); `scripts/paper/check_attn_impl_gpu_equivalence.py` (GPU check of the two attention implementations of NarrowDiT) | none |

Every script prints its options with `--help`. The defaults of the phase
discovery, the U-Net profiler and the U-Net allocator are the paper settings
(PFI with 20 bins, the Section 3.3 objective with automatic K, and the Section
3.4 rule with the four paper variants). The CIFAR-10 U-Net pipeline, for
example:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 8 scripts/evaluate_parameters_edm.py \
  --dataset cifar10 --cifar_split validation --data_root data --download --model_family vp \
  --network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
  --model-cache-dir checkpoints --trust-local-pickle --output_dir outputs/cifar10_unet/profile \
  --device cuda --batch_size 2000 --grouping per_filter --num_bins 20 --num_workers 10 \
  --max_images 100 --ablation_mode random_same_norm --corruption_order image_major
python scripts/optimize_timestep_grouping.py --matrix outputs/cifar10_unet/profile/results.json \
  --matrix_key relative_delta_stack --num_blocks auto --output outputs/cifar10_unet/grouping/timestep_grouping.json
python scripts/dry_run_capacity_allocation.py --results-json outputs/cifar10_unet/profile/results.json \
  --timestep-grouping outputs/cifar10_unet/grouping/timestep_grouping.json \
  --allocation-results-dir outputs/cifar10_unet/allocations
python scripts/prepare_edm_distillation.py --results-json outputs/cifar10_unet/profile/results.json \
  --allocation-results-dir outputs/cifar10_unet/allocations --output-dir outputs/cifar10_unet/plans
python scripts/train_edm_distillation.py \
  --architecture-plan artifacts/plans/cifar10/combined_layerwise/architecture_plan.json \
  --output-dir outputs/cifar10_unet/training/combined_layerwise/seed0 --dataset cifar10 --data-root data \
  --download --model-cache-dir checkpoints --trust-local-pickle --device cuda --steps 70000 \
  --batch-size 512 --microbatch 128 --fid-every 1000 --fid-num-samples 5000
```

`reproduce/cifar10_unet.sh` runs the same pipeline with every recorded flag.

## 5. Inputs that you provide

| Input | Used by | Source |
|---|---|---|
| CIFAR-10 | CIFAR-10 U-Net and DiT scripts | downloaded by torchvision into `data/` |
| ImageNet-1k in the Hugging Face parquet layout | ImageNet-64 U-Net and ImageNet DiT scripts | `data/imagenet-1k/data/{train,validation}-*.parquet` (`IMAGENET_PARQUET`) |
| FFHQ at 256x256 (70,000 images, a flat directory or ZIP) | FFHQ-64 U-Net and FFHQ DiT scripts | `data/ffhq256` (`FFHQ256_SOURCE`, `FFHQ256_ROOT`) |
| LSUN Bedroom: the first 1,000,000 images of `bedroom_train_lmdb`, written as raw JPEG bytes to `0000000.jpg` ... `0999999.jpg` | LSUN Bedroom U-Net and DiT scripts | `data/lsun_bedroom256/img256` (`LSUN_ROOT`) |
| Speech Commands SC09 | audio script | downloaded and verified by `scripts/data/prepare_sc09_dataset.py` |
| EDM teachers (CIFAR-10, ImageNet-64, FFHQ-64) and the LSUN Bedroom teacher of openai/consistency_models | U-Net scripts | downloaded into `checkpoints/` from the URLs in `pace/teacher_models.py`; the NVLabs pickles of CIFAR-10 and ImageNet-64 need `--trust-local-pickle` |
| `DiT-XL-2-256x256.pt` | ImageNet DiT script | facebookresearch/DiT (`download.py`) or the Hugging Face repository facebook/DiT-XL-2-256 |
| `dit-micro-cifar10-class-ema-e5000.pt` | DiT-Micro script | Hugging Face repository normalcomputing/dit-cifar10-32x32-class |
| DiffWave SC09 checkpoint `1000000.pkl` | audio script | downloaded and verified by the `teacher` stage |
| `ffhq-64x64.npz` | FFHQ-64 NVLabs FID | https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz |
| `VIRTUAL_lsun_bedroom256.npz` | LSUN Bedroom U-Net and DiT evaluations | https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz |
| `VIRTUAL_imagenet256_labeled.npz` | ImageNet DiT evaluation | listed in `guided-diffusion/evaluations/README.md` |
| `ffhq256_ref_50k.npz` | FFHQ DiT evaluation | not recorded; the `references` stage of `reproduce/dit_ffhq256.sh` packs the first 50,000 training images |
| `cifar_test_ref_32.npz` | CIFAR-10 DiT evaluation | the `references` stage of `reproduce/dit_cifar10.sh` packs the 10,000 test images |
| `VIRTUAL_ffhq64_first50k_adm.npz` | FFHQ-64 ADM metrics | the `references` stage of `reproduce/ffhq64_unet.sh` |

## 6. What the release does not contain

Checkpoints, datasets, reference batches, training logs and evaluation outputs
are not included. The released intermediate results in `artifacts/` are the
four U-Net profiles, their groupings and allocations, the plans of the trained
U-Net students, the archived DiT-Micro profile and grouping, and the audio
metrics snapshot; `artifacts/README.md` describes each file. The profiles and
groupings of the DiT experiments and the DiffWave profile are not released.
