# Released artifacts

These files are intermediate results of the runs behind the paper. With them,
the phases, the Table 3 statistics and the U-Net phase budgets reproduce on a
CPU (`python reproduce/cpu_phases_and_budgets.py`), and the U-Net students
train from the exact plans of the paper. Checkpoints, datasets and evaluation
outputs are not included.

Every loader in this repository reads these files directly, including the
gzip-compressed profiles (`pace.jsonio.load_json`). Format identifiers inside
the files, such as `diffdist_edm_ablation_protocol_v1`, keep their original
values because the loaders validate them.

## Changes made for the release

Absolute paths of the original runs are rewritten to repository-relative paths
or to placeholders: `$PACE_SHARED_ROOT` (the storage root of the original
runs), `$DATA_ROOT` (the dataset root), `$RUN_ROOT` (a training run root) and
`$EDM_REPO`. The placeholders are documentation; no code expands them. Apart
from the `cross_block_lambda` record of the FFHQ-64 grouping described below,
no numerical content changes.

Groupings, allocations and plans carry a `pace_release_copy` record (format
`pace_release_copy_v1`) with the relative path of the original file, the
SHA-256 of the original file and the list of changes. Besides the path
rewrites, these changes are:

* `source_profile.profile_fingerprint` is replaced by `null`; its digest stays
  in `source_profile.profile_fingerprint_sha256`, which is what the provenance
  checks compare (FFHQ-64 and LSUN Bedroom);
* the per-file listings `dataset_info.manifest_metadata.entries` and `.splits`
  are dropped from the FFHQ-64 and LSUN Bedroom plans (their digests stay);
* `timestep_grouping_path` of each allocation points to the released grouping;
* the FFHQ-64 grouping records `cross_block_lambda` as 0.0. Its cost,
  `matrix_correlation`, has no separation term; at a fixed number of phases its
  optimum is that of the Section 3.3 objective with lambda_sep = 0, which gives
  the same phases on the released profile.

The profiles are slim copies (format `pace_slim_profile_v1`, see
`pace/slim_profile.py`). A slim profile keeps every key of the full
`results.json` except the matrices that no later stage reads (`weights`,
`row_normalized_relative_delta_stack`, `signed_delta_stack`,
`sampling_adjusted_delta_stack`), the embedded profile fingerprint and the
per-file dataset listings. `delta_stack` is stored as a compressed float64
block and `relative_delta_stack` is recomputed on load as
`delta_stack / (baseline_mean + 1e-12)`, the formula of the profiler. Both are
bit-exact for all four profiles. The `pace_slim_profile` record holds the
SHA-256, size and relative path of the full profile, so groupings,
allocations and plans computed from the full profile validate against the slim
copy. `python scripts/paper/slim_profile.py --results <results.json> --output
<file>.json.gz` writes a slim copy and checks the round trip.

## Files

Sizes are in bytes. The source paths are relative paths in the repository of
the original experiments.

### `profiles/`

Group-by-bin sensitivity profiles of the U-Net teachers: 20 noise bins, one
group per convolution output channel, 100 images.

| File | Size | Teacher | Protocol | Source (SHA-256 prefix) |
|---|---|---|---|---|
| `cifar10_ddpmpp_random_same_norm.json.gz` | 1,636,335 | NVLabs EDM CIFAR-10 DDPM++ (VP), class-conditional | `random_same_norm`, 28,291 filters, 256 noise levels, CIFAR-10 test images | `out_eval_edm_cifar10/results.json` (`19935730a1f6fc85`) |
| `imagenet64_adm_random_same_norm.json.gz` | 5,135,923 | NVLabs EDM ImageNet-64 ADM, class-conditional | `random_same_norm`, 96,771 filters, 256 noise levels, ImageNet validation images | `out_eval_edm_imagenet/results.json` (`8733cae074609c35`) |
| `ffhq64_ddpmpp_pfi.json.gz` | 2,134,099 | NVLabs EDM FFHQ-64 DDPM++ (VP), unconditional | PFI `batch_local_exact_sigma_pfi_v1`, 32,387 filters, 256 noise levels, FFHQ monitor images | `out_eval_edm_ffhq64/pfi_batch_local_exact_sigma_v1/profile_per_filter_n100_s256_b256/results.json` (`89559aded9b95cb3`) |
| `lsun256_adm_pfi_stratified.json.gz` | 329,517 | openai/consistency_models EDM LSUN Bedroom 256 | PFI `batch_local_exact_sigma_pfi_v1`, 4,387 of 119,555 filters (32 per module, Horvitz-Thompson weights), 64 noise levels, LSUN monitor images, mixed FP16 | `out_eval_edm_lsun_bedroom256/pfi_batch_local_exact_sigma_v1/profile_per_filter_stratified_k32_n100_s64_b100/results.json` (`ae5fd91aa4c487eb`) |

Regenerate with the `profile` stage of `reproduce/cifar10_unet.sh`,
`reproduce/imagenet64_unet.sh`, `reproduce/ffhq64_unet.sh` or
`reproduce/lsun_bedroom256_unet.sh` (GPU), then `scripts/paper/slim_profile.py`. A
new profile is not bit-identical to the released one (GPU arithmetic, sampling
of the replacement noise).

### `groupings/`

Phase boundaries over the 20 bins, written by `scripts/optimize_timestep_grouping.py`.

| File | Size | Phases | Objective | Source (SHA-256 prefix) |
|---|---|---|---|---|
| `cifar10_ddpmpp_random_same_norm.json` | 2,821 | `[0,8,20]` | Section 3.3, lambda_sep = 0.02, K = 2 | `out_eval_edm_cifar10/grouping/timestep_grouping.json` (`8eb1f709004efbc4`) |
| `imagenet64_adm_random_same_norm.json` | 2,840 | `[0,16,20]` | Section 3.3, lambda_sep = 0.02, K = 2 | `out_eval_edm_imagenet/grouping/timestep_grouping.json` (`2da9433274ca23a2`) |
| `ffhq64_ddpmpp_pfi.json` | 9,523 | `[0,3,16,20]` | `matrix_correlation`, K = 3 (recorded with `cross_block_lambda` 0.0, see above) | `out_eval_edm_ffhq64/pfi_batch_local_exact_sigma_v1/grouping/timestep_grouping.json` (`9504a900cfbe9df3`) |
| `lsun256_adm_pfi_stratified.json` | 833,969 | `[0,16,20]` | Section 3.3, lambda_sep = 0.02, K = 2, Horvitz-Thompson weighted | `out_eval_edm_lsun_bedroom256/pfi_batch_local_exact_sigma_v1/profile_per_filter_stratified_k32_n100_s64_b100/grouping/timestep_grouping.json` (`610d3efa121255bc`) |

Every grouping records `builtin_cost`, `cross_block_lambda`, `num_blocks` and
`boundaries`. Regenerate with the `group` stage of the U-Net scripts, which runs
on a CPU; with `PROFILE_JSON` set to a released profile it gives the same
boundaries and objective values (`reproduce/cpu_phases_and_budgets.py` checks
this). The CIFAR-10, ImageNet-64 and LSUN Bedroom runs passed K = 2; automatic
selection chooses the same K.

### `allocations/<dataset>/<variant>.json`

Files above 100 KB are stored gzip-compressed as `<variant>.json.gz`; every loader reads both forms
(`pace.jsonio.load_json`, `pace.jsonio.find_json_file`). Sizes below are in bytes, compressed where the name ends in `.gz`.

Phase budgets and, for `combined_layerwise`, per-filter layer budgets of the
four students, written by `scripts/dry_run_capacity_allocation.py` with
`--allocation-metric delta_p_eff_geomean --score-reduction sum`. The variants
are `global` (Global), `uniform_blockwise` (Uniform blockwise),
`combined_blockwise` (Phase-aware blockwise) and `combined_layerwise`
(Phase-aware layerwise).

| Dataset | Files and sizes | Source directory |
|---|---|---|
| `cifar10` | `global.json` 2,464; `uniform_blockwise.json` 2,526; `combined_blockwise.json` 2,769; `combined_layerwise.json.gz` 486,629 (1,548,750 uncompressed) | `out_eval_edm_cifar10/allocation_results/` (Global, Uniform) and `out_eval_edm_cifar10/allocation_results_delta_peff/` (Phase-aware) |
| `imagenet64` | `global.json` 2,487; `uniform_blockwise.json` 2,550; `combined_blockwise.json` 2,793; `combined_layerwise.json.gz` 1,803,664 (5,144,235 uncompressed) | `out_eval_edm_imagenet/allocation_results/` (Global, Uniform) and `out_eval_edm_imagenet/allocation_results_delta_peff/` (Phase-aware) |
| `ffhq64` | `global.json` 9,517; `uniform_blockwise.json` 9,632; `combined_blockwise.json` 9,901; `combined_layerwise.json.gz` 849,655 (2,499,164 uncompressed) | `out_eval_edm_ffhq64/pfi_batch_local_exact_sigma_v1/allocation_results/` |
| `lsun256` | `global.json.gz` 46,986 (834,039 uncompressed); `uniform_blockwise.json.gz` 47,017 (834,113 uncompressed); `combined_blockwise.json.gz` 47,213 (834,927 uncompressed); `combined_layerwise.json.gz` 130,686 (1,067,523 uncompressed) | `out_eval_edm_lsun_bedroom256/pfi_batch_local_exact_sigma_v1/profile_per_filter_stratified_k32_n100_s64_b100/allocation_results/` |

The CIFAR-10 and ImageNet-64 Global and Uniform files come from an earlier run
that recorded a different score metric; the Global and Uniform budgets do not
depend on the metric, and the files record the same budgets as a new run.
Regenerate with the `allocate` stage of the U-Net scripts (CPU). From the
released profiles and groupings, the phase budgets are identical and the layer
budgets agree to within 1e-10 parameters (the order of floating-point sums can
differ).

### `plans/<dataset>/<variant>/architecture_plan.json`

Architectures of the trained U-Net students, as read by
`scripts/train_edm_distillation.py`: the width of every layer of every phase
specialist, the phase boundaries and the teacher reference.

| Dataset | Sizes (global, uniform_blockwise, combined_blockwise, combined_layerwise) | Source |
|---|---|---|
| `cifar10` | 8,457; 9,747; 9,630; 14,071 | `out_distill_edm_cifar10/delta_peff/<variant>/architecture_plan.json` for the Phase-aware students; the Global and Uniform plans are the `plans.global` and `plans.uniform_blockwise` entries of `out_distill_edm_cifar10/delta_peff/architecture_summary.json`, written with the serialization of `scripts/prepare_edm_distillation.py` |
| `imagenet64` | 8,328; 9,613; 11,819; 14,149 | `out_distill_edm_imagenet/delta_peff/<variant>/architecture_plan.json` |
| `ffhq64` | 24,613; 28,140; 27,563; 36,617 | `out_eval_edm_ffhq64/pfi_batch_local_exact_sigma_v1/plans/<variant>/architecture_plan.json` |
| `lsun256` | 877,700; 879,775; 879,450; 885,249 | `out_eval_edm_lsun_bedroom256/pfi_batch_local_exact_sigma_v1/profile_per_filter_stratified_k32_n100_s64_b100/plans/<variant>/architecture_plan.json` |

These are the plans that were trained; they are not regenerated. The student
`model_kwargs` of the CIFAR-10 and ImageNet-64 plans do not record
`preconditioning`, `num_blocks`, `attn_resolutions` or `augment_dim`, so the
students are built with EDM preconditioning and the `NarrowSongUNet` defaults,
as they were for training. The `plans` stage of the U-Net scripts rebuilds
plans with `scripts/prepare_edm_distillation.py` (CPU, needs NVlabs/edm); the
rebuilt CIFAR-10 plans differ from the trained ones (`REPRODUCING.md`,
section 6).

### `dit_micro/`

The archived DiT-Micro profile of Appendix C.2 and Figure 6: 24 attention
heads of the class-conditional DiT-Micro EDM teacher
(`dit-micro-cifar10-class-ema-e5000.pt`), legacy `permutation` protocol, 1,000
CIFAR-10 training images, 64 EDM noise levels in 20 bins, 2 processes.

| File | Size | Content | Source |
|---|---|---|---|
| `dit_micro_perm_results.json` | 80,744 | profile | `handoff/groupings/dit_micro_perm_results.json`, with the paths rewritten |
| `dit_micro_perm_4phase.json` | 2,800 | phases `[0,4,8,16,20]`, Section 3.3 objective with automatic K | `handoff/groupings/dit_micro_perm_4phase.json`, with the paths rewritten |

Regenerate with `reproduce/dit_micro_cifar10.sh` (the profile needs 2 GPUs;
the grouping runs on a CPU and gives the same phases and objective value).

### `audio/`

| File | Size | Content | Source |
|---|---|---|---|
| `audio_appendix_metrics.json` | 16,167 | the values of Appendix B and Figure 4 (format `diffdist_audio_appendix_metrics_v1`): the 20 by 20 correlation matrix of the residual-only DiffWave profile, the phases `[0,13,17,20]`, the phase similarities, the capacity shares and the stability summary, with the SHA-256 of the residual-only profile, of the profile fingerprint and of the teacher checkpoint | `analysis/audio_appendix_metrics.json`, unchanged |

`python scripts/paper/plot_audio_appendix.py` renders Figure 4 from this file on a
CPU. `reproduce/sc09_diffwave.sh` regenerates it from a new profile (8 GPUs)
with `scripts/paper/plot_audio_appendix.py --refresh-from-profile`.
