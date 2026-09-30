#!/usr/bin/env bash
# LSUN Bedroom 256, unconditional EDM teacher of openai/consistency_models, PFI
# profile on a stratified sample of filters.
#
# Paper results: the LSUN Bedroom ConvNet rows of Tables 1 and 2, the LSUN row of
# Table 3, Figure 5(d) and Appendix C.1.
#
# Stages, in order (select a subset with STAGES="group allocate"):
#   data       CPU  dataset manifest and full decode check of the 1,000,000 training images
#   profile    GPU  PFI profile of 32 filters per module with Horvitz-Thompson weights (Section 3.2)
#   group      CPU  phase discovery with the Section 3.3 objective and automatic K
#   allocate   CPU  Section 3.4 capacity allocation for the four students
#   plans      CPU  student architectures from the allocations (Appendix A.4)
#   train      GPU  hybrid distillation of each student (Eq. 2), mixed FP16 teacher
#   evaluate   GPU  ADM FID-50k, sFID, precision, recall and Inception Score, mixed FP16
#   summarize  CPU  metric/throughput table of the teacher and the four students
#
# Inputs you provide:
#   LSUN_ROOT       the first 1,000,000 images of the LSUN bedroom_train_lmdb, written as
#                   raw JPEG bytes to 0000000.jpg ... 0999999.jpg without cropping
#                   (default: $DATA_ROOT/lsun_bedroom256/img256)
#   $REFERENCE_DIR/VIRTUAL_lsun_bedroom256.npz
#                   OpenAI reference batch listed in guided-diffusion/evaluations/README.md
#   $ADM_DIR, ADM_PYTHON, ADM_DETECTOR
#                   guided-diffusion evaluator, its TensorFlow environment and
#                   classify_image_graph_def.pb
#   NVlabs/edm      cloned at EDM_REPO (default ../edm)
# The teacher checkpoint (edm_bedroom256_ema.pt, 2.1 GB) is downloaded into MODEL_CACHE.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/lsun_bedroom256_unet)
#   LSUN_MANIFEST  dataset manifest (default: $OUT/dataset/manifest.json)
#   PROFILE_NPROC  processes for the profile (default: 8, the recorded run)
#   PROFILE_JSON   profile read by group, allocate and plans (default: $OUT/profile/results.json;
#                  artifacts/profiles/lsun256_adm_pfi_stratified.json.gz is the released one)
#   GROUPING_JSON  grouping read by allocate and plans (default: $OUT/grouping/timestep_grouping.json)
#   PLAN_ROOT      plans read by train (default: artifacts/plans/lsun256, the trained plans)
#   VARIANTS       students to train and evaluate (default: the four paper variants)
#   TRAIN_NPROC    processes per student (default: 2, the recorded runs)
#   STEPS          optimizer steps per student (default: 100000; the evaluated
#                  checkpoints are from steps 90,000 to 100,000)
#   EVAL_NPROC     processes per evaluation (default: 2; summarize requires 2)
#   RESUME=1       continue interrupted training runs

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/lsun_bedroom256_unet}"
LSUN_ROOT="${LSUN_ROOT:-$DATA_ROOT/lsun_bedroom256/img256}"
LSUN_MANIFEST="${LSUN_MANIFEST:-$OUT/dataset/manifest.json}"
PROFILE_NPROC="${PROFILE_NPROC:-8}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLAN_ROOT="${PLAN_ROOT:-artifacts/plans/lsun256}"
VARIANTS="${VARIANTS:-$PAPER_VARIANTS}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
STEPS="${STEPS:-100000}"
EVAL_NPROC="${EVAL_NPROC:-2}"
LSUN_REF="$REFERENCE_DIR/VIRTUAL_lsun_bedroom256.npz"
TEACHER_PROTOCOL="lsun_bedroom256_openai_adm_v1"
STUDENT_PROTOCOL="lsun_bedroom256_openai_adm_mixed_fp16_v1"

if stage data; then
  announce "data (CPU)"
  require_input "$LSUN_ROOT" "LSUN Bedroom images"
  run "$PYTHON" scripts/data/preflight_edm_dataset.py \
    --dataset lsun_bedroom --data-root "$LSUN_ROOT" --dataset-manifest "$LSUN_MANIFEST" \
    --dataset-split train --image-size 256 --lsun-monitor-size 10000 --lsun-monitor-seed 12345 \
    --report "$OUT/dataset/preflight.json"
fi

if stage profile; then
  announce "profile (GPU, $PROFILE_NPROC processes)"
  require_input "$LSUN_MANIFEST" "LSUN dataset manifest (data stage)"
  # Recorded invocation of the released profile: 4,387 of 119,555 filters,
  # 64 sigma levels (stride 4), 100 monitor images, mixed FP16 teacher.
  launch "$PROFILE_NPROC" scripts/evaluate_parameters_edm.py \
    --distributed-timeout-seconds 86400 \
    --dataset lsun_bedroom --data_root "$LSUN_ROOT" --dataset-manifest "$LSUN_MANIFEST" \
    --dataset-split monitor --dataset-preflight --lsun-monitor-size 10000 --lsun-monitor-seed 12345 \
    --network-preset lsun_bedroom_256 --model-cache-dir "$MODEL_CACHE" \
    --output_dir "$OUT/profile" --grouping per_filter \
    --filter_sampling stratified_module --filters_per_module 32 --filter_sampling_seed 0 \
    --ablation_mode pfi --pfi_seed 0 --seed 0 --max_images 100 \
    --num_sigma_levels 256 --sigma_stride 4 --num_bins 20 --batch_size 100 \
    --num_workers 4 --image_size 256 --device cuda --dtype fp16
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # The Horvitz-Thompson filter weights of the profile are applied automatically.
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
  run "$PYTHON" scripts/prepare_edm_distillation.py \
    --results-json "$PROFILE_JSON" --allocation-results-dir "$OUT/allocations" \
    --output-dir "$OUT/plans" \
    --variant global --variant uniform_blockwise \
    --variant combined_blockwise --variant combined_layerwise
fi

if stage train; then
  require_input "$LSUN_MANIFEST" "LSUN dataset manifest (data stage)"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC processes)"
    plan="$PLAN_ROOT/$variant/architecture_plan.json"
    require_input "$plan" "architecture plan"
    # 64 images per step (32 per process, microbatch 12); the published
    # checkpoint is the one with the lowest EMA validation loss.
    launch "$TRAIN_NPROC" scripts/train_edm_distillation.py \
      --architecture-plan "$plan" --output-dir "$OUT/training/$variant/seed0" \
      --dataset lsun_bedroom --data-root "$LSUN_ROOT" --dataset-manifest "$LSUN_MANIFEST" \
      --dataset-split train --model-cache-dir "$MODEL_CACHE" \
      --device cuda --dtype fp16 --steps "$STEPS" \
      --batch-size 64 --microbatch 12 --lr 2e-4 --seed 0 --num-workers 8 \
      --kd-weight 1.0 --data-weight 0.25 --ema-beta 0.999 --snapshot-every 5000 \
      --val-every 5000 --val-max-images 10000 --val-batch-size 64 --val-microbatch 32 --val-seed 12345 \
      --early-stop-patience 4 --early-stop-min-steps 10000 --early-stop-min-delta 1e-4 \
      --fid-every 0 --ddp-timeout-minutes 180 \
      --keep-last-snapshots 0 --checkpoint-selection best_val $(resume_flag)
  done
fi

common_eval_args=(
  --reference-stats "$LSUN_REF" --num-samples 50000 --seed 2100000 --num-steps 40
  --batch-size-per-rank 32 --warmup-samples-per-rank 32 --metric-backend openai_adm
  --model-cache-dir "$MODEL_CACHE" --dataset-manifest "$LSUN_MANIFEST"
  --adm-evaluator "$ADM_EVALUATOR" --adm-python "$ADM_PYTHON" --adm-detector "$ADM_DETECTOR"
  --artifact-retention keep_preview --preview-count 64 --distributed-timeout-minutes 1440
)

if stage evaluate; then
  require_input "$LSUN_REF" "LSUN Bedroom ADM reference batch"
  announce "evaluate teacher (GPU, $EVAL_NPROC processes)"
  launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py \
    --network-preset lsun_bedroom_256 --teacher-dtype fp16 \
    --output-dir "$OUT/evaluation/teacher/$TEACHER_PROTOCOL" --protocol-id "$TEACHER_PROTOCOL" \
    "${common_eval_args[@]}"
  for variant in $VARIANTS; do
    announce "evaluate $variant (GPU, $EVAL_NPROC processes)"
    checkpoint="$OUT/training/$variant/seed0/student-best-val.pt"
    require_input "$checkpoint" "best-validation checkpoint"
    launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py \
      --checkpoint "$checkpoint" --weights ema --teacher-dtype fp16 --student-dtype fp16 \
      --output-dir "$OUT/evaluation/$variant/$STUDENT_PROTOCOL" --protocol-id "$STUDENT_PROTOCOL" \
      "${common_eval_args[@]}"
  done
fi

if stage summarize; then
  announce "summarize (CPU)"
  selection="$OUT/evaluation/selected_results.json"
  # The comparison pairs the mixed-FP16 teacher evaluation with the mixed-FP16
  # students; the selection file records each source result and its SHA-256.
  run "$PYTHON" - "$OUT/evaluation" "$selection" "$TEACHER_PROTOCOL" "$STUDENT_PROTOCOL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, selection_path, teacher_protocol, student_protocol = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
models = {}
for variant in ("teacher", "global", "uniform_blockwise", "combined_blockwise", "combined_layerwise"):
    protocol = teacher_protocol if variant == "teacher" else student_protocol
    path = root / variant / protocol / "evaluation_result.json"
    models[variant] = {
        "source_result": str(path.resolve().relative_to(selection_path.parent.resolve())),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_protocol_id": protocol,
        "inference_precision": "mixed_fp16",
        "reused_completed_evaluation": False,
    }
selection = {
    "format": "diffdist_lsun_mixed_fp16_comparison_selection_v1",
    "selection_status": "complete",
    "models": models,
}
selection_path.write_text(json.dumps(selection, indent=2) + "\n")
print(f"Wrote {selection_path}")
PY
  run "$PYTHON" scripts/paper/plot_lsun_benchmark_pareto.py --selection "$selection" --output-dir "$OUT/pareto"
fi
