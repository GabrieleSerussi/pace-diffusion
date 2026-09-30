#!/usr/bin/env bash
# Archived DiT-Micro profile on CIFAR-10 (Appendix C.2, Figure 6).
#
# The released profile artifacts/dit_micro/dit_micro_perm_results.json and its
# grouping artifacts/dit_micro/dit_micro_perm_4phase.json come from this
# pipeline. The profile uses the legacy head permutation of the DiT scripts
# (torch.randperm with a seed per head), not PFI. The recorded code ordered the
# samples image-major and permuted each head's output across the whole batch,
# so a permutation could exchange activations between noise levels;
# --corruption_order image_major --permutation_group_by_level false restore
# that behaviour.
#
# Stages, in order:
#   profile  GPU  head-level permutation profile with the recorded settings (2 processes)
#   group    CPU  phase discovery with the Section 3.3 objective and automatic K
#
# Inputs you provide:
#   DIT_MICRO_CHECKPOINT  dit-micro-cifar10-class-ema-e5000.pt from the Hugging Face
#                         repository normalcomputing/dit-cifar10-32x32-class
#                         (default: $MODEL_CACHE/dit-micro-cifar10-class-ema-e5000.pt)
#   facebookresearch/DiT  cloned at DIT_REPO (default ../DiT); pip install "pace-diffusion[dit]"
# CIFAR-10 is downloaded into DATA_ROOT.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/dit_micro_cifar10)
#   PROFILE_NPROC  processes for the profile (default: 2, the recorded run)
#   PROFILE_JSON   profile read by group (default: $OUT/profile/results.json;
#                  artifacts/dit_micro/dit_micro_perm_results.json is the released one)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/dit_micro_cifar10}"
DIT_MICRO_CHECKPOINT="${DIT_MICRO_CHECKPOINT:-$MODEL_CACHE/dit-micro-cifar10-class-ema-e5000.pt}"
PROFILE_NPROC="${PROFILE_NPROC:-2}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"

if stage profile; then
  announce "profile (GPU, $PROFILE_NPROC processes)"
  require_input "$DIT_MICRO_CHECKPOINT" "DiT-Micro checkpoint"
  launch "$PROFILE_NPROC" scripts/evaluate_parameters_dit_micro.py \
    --checkpoint "$DIT_MICRO_CHECKPOINT" --num_heads 3 --diffusion edm \
    --grouping attention_heads --ablation_mode permutation \
    --corruption_order image_major --permutation_group_by_level false \
    --data_root "$DATA_ROOT" --download --cifar_split train --max_images 1000 \
    --num_timestep_levels 64 --num_bins 20 --batch_size 256 --num_workers 4 \
    --dtype bf16 --seed 0 --output_dir "$OUT/profile"
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # Automatic K selects four phases, [0, 4, 8, 16, 20], on the released profile.
  run "$PYTHON" scripts/optimize_timestep_grouping.py \
    --matrix "$PROFILE_JSON" --matrix_key relative_delta_stack \
    --builtin_cost matrix_correlation_cross_penalty --cross_block_lambda 0.02 \
    --cross_block_reward_normalization size --num_blocks auto \
    --output "$OUT/grouping/timestep_grouping.json" \
    --plot_output "$OUT/grouping/grouping_matrix.png" --plot_matrix_source recomputed_similarity \
    --cost_curve_output "$OUT/grouping/cost_curve.png"
fi
