#!/usr/bin/env bash
# CIFAR-10, class-conditional DDPM++ (VP) teacher of NVlabs/edm.
#
# Paper results: the CIFAR-10 ConvNet rows of Tables 1 and 2, the CIFAR-10 row
# of Table 3 and Figure 5(a).
#
# Stages, in order (select a subset with STAGES="group allocate"):
#   profile   GPU  noise-level profile with the random_same_norm protocol (Section 3.2, Appendix C.1)
#   group     CPU  phase discovery with the Section 3.3 objective and automatic K
#   allocate  CPU  Section 3.4 capacity allocation for the four students
#   plans     CPU  student architectures from the allocations (Appendix A.4)
#   train     GPU  hybrid distillation of each student (Eq. 2), with Clean-FID-5k monitoring
#   report    CPU  lowest Clean-FID-5k of each student's training monitor, and its step
#
# Inputs: CIFAR-10 and the teacher checkpoint are downloaded; NVlabs/edm must be
# cloned (EDM_REPO, default ../edm).
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/cifar10_unet)
#   PROFILE_NPROC  processes for the profile (default: 8, the recorded run)
#   PROFILE_JSON   profile read by group, allocate and plans
#                  (default: $OUT/profile/results.json). Set it to
#                  artifacts/profiles/cifar10_ddpmpp_random_same_norm.json.gz
#                  to start from the released profile without a GPU.
#   GROUPING_JSON  grouping read by allocate and plans (default: $OUT/grouping/timestep_grouping.json)
#   PLAN_ROOT      plans read by train (default: artifacts/plans/cifar10, the plans
#                  of the trained students; $OUT/plans holds the plans rebuilt by
#                  the plans stage)
#   VARIANTS       students to train (default: the four paper variants)
#   TRAIN_NPROC    processes per student (default: 1)
#   STEPS          optimizer steps per student (default: 70000; the recorded runs
#                  reached at least 69,000 steps, and their exact length is not recorded)
#   FID_EVERY      Clean-FID-5k monitoring interval in steps (default: 1000)
#   RESUME=1       continue interrupted training runs

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/cifar10_unet}"
PROFILE_NPROC="${PROFILE_NPROC:-8}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLAN_ROOT="${PLAN_ROOT:-artifacts/plans/cifar10}"
VARIANTS="${VARIANTS:-$PAPER_VARIANTS}"
TRAIN_NPROC="${TRAIN_NPROC:-1}"
STEPS="${STEPS:-70000}"
FID_EVERY="${FID_EVERY:-1000}"
TEACHER_URL="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"

if stage profile; then
  announce "profile (GPU, $PROFILE_NPROC processes)"
  # Recorded invocation of the released profile. The profile predates the
  # level-major sample order, so --corruption_order image_major restores the
  # batch composition of that run. --trust-local-pickle is needed because the
  # NVlabs checkpoints are Python pickles.
  launch "$PROFILE_NPROC" scripts/evaluate_parameters_edm.py \
    --dataset cifar10 --cifar_split validation --data_root "$DATA_ROOT" --download \
    --model_family vp --network_pkl "$TEACHER_URL" \
    --model-cache-dir "$MODEL_CACHE" --trust-local-pickle \
    --output_dir "$OUT/profile" --device cuda --dtype fp32 --seed 0 \
    --batch_size 2000 --grouping per_filter --num_bins 20 --num_sigma_levels 256 --sigma_stride 1 \
    --num_workers 10 --max_images 100 --ablation_mode random_same_norm \
    --corruption_order image_major
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # The recorded run passed --num_blocks 2; automatic selection chooses the same K.
  run "$PYTHON" scripts/optimize_timestep_grouping.py \
    --matrix "$PROFILE_JSON" --matrix_key relative_delta_stack \
    --builtin_cost matrix_correlation_cross_penalty --cross_block_lambda 0.02 \
    --cross_block_reward_normalization size --num_blocks auto \
    --output "$GROUPING_JSON" \
    --plot_output "$(dirname "$GROUPING_JSON")/timestep_grouping_matrix.png" \
    --plot_matrix_source recomputed_similarity \
    --cost_curve_output "$(dirname "$GROUPING_JSON")/timestep_grouping_cost_curve.png"
fi

if stage allocate; then
  announce "allocate (CPU)"
  require_input "$PROFILE_JSON" "profile"
  require_input "$GROUPING_JSON" "grouping"
  run "$PYTHON" scripts/dry_run_capacity_allocation.py \
    --results-json "$PROFILE_JSON" --timestep-grouping "$GROUPING_JSON" \
    --allocation-results-dir "$OUT/allocations" \
    --allocation-metric delta_p_eff_geomean --score-reduction sum \
    --layer-score-source delta_p_eff_geomean \
    --student-variant global --student-variant uniform_blockwise \
    --student-variant combined_blockwise --student-variant combined_layerwise
fi

if stage plans; then
  announce "plans (CPU, needs NVlabs/edm)"
  require_input "$PROFILE_JSON" "profile"
  # The trained combined_blockwise plan of CIFAR-10 differs from a rebuild with
  # this code (REPRODUCING.md); train reads artifacts/plans/cifar10 by default.
  run "$PYTHON" scripts/prepare_edm_distillation.py \
    --results-json "$PROFILE_JSON" --allocation-results-dir "$OUT/allocations" \
    --output-dir "$OUT/plans" \
    --variant global --variant uniform_blockwise \
    --variant combined_blockwise --variant combined_layerwise
fi

if stage train; then
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC process(es))"
    plan="$PLAN_ROOT/$variant/architecture_plan.json"
    require_input "$plan" "architecture plan"
    launch "$TRAIN_NPROC" scripts/train_edm_distillation.py \
      --architecture-plan "$plan" --output-dir "$OUT/training/$variant/seed0" \
      --dataset cifar10 --data-root "$DATA_ROOT" --download \
      --model-cache-dir "$MODEL_CACHE" --trust-local-pickle \
      --device cuda --dtype fp32 --steps "$STEPS" \
      --batch-size 512 --microbatch 128 --lr 2e-4 --seed 0 --num-workers 20 \
      --kd-weight 1.0 --data-weight 0.25 --ema-beta 0.999 --snapshot-every 5000 \
      --val-every 5000 --val-max-images 2048 --val-batch-size 256 --val-microbatch 64 --val-seed 12345 \
      --early-stop-patience 4 --early-stop-min-steps 10000 --early-stop-min-delta 1e-4 \
      --fid-every "$FID_EVERY" --fid-num-samples 5000 --fid-batch-size 64 --fid-num-steps 18 \
      --fid-ref-stats-cache-dir "$OUT/fid_reference_stats" --fid-feature-batch-size 32 \
      --fid-ref-split train --fid-ref-dataset-name cifar10 --fid-ref-dataset-res 32 \
      --fid-mode clean --fid-seed 0 $(resume_flag)
  done
fi

if stage report; then
  announce "report (CPU): lowest Clean-FID-5k of each training monitor"
  # CIFAR-10 has no separate 50k evaluation. The recorded results are the lowest
  # values of the training monitor (5,000 samples, 18 Heun steps, Clean-FID
  # against the CIFAR-10 training statistics), at steps 64,000 to 69,000.
  run "$PYTHON" - "$OUT/training" $VARIANTS <<'PY'
import json
import sys
from pathlib import Path

root, variants = Path(sys.argv[1]), sys.argv[2:]
for variant in variants:
    path = root / variant / "seed0" / "fid_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("fid") is not None]
    if not rows:
        print(f"{variant:>20s}  no FID rows in {path}")
        continue
    best = min(rows, key=lambda row: row["fid"])
    print(f"{variant:>20s}  lowest Clean-FID-5k {best['fid']:.3f} at step {best['step']} "
          f"({len(rows)} monitored steps)")
PY
fi
