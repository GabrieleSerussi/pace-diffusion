#!/usr/bin/env bash
# FFHQ-64, unconditional DDPM++ (VP) teacher of NVlabs/edm, PFI profile.
#
# Paper results: the FFHQ-64 ConvNet rows of Tables 1 and 2, the FFHQ-64 row of
# Table 3, Figure 5(c) and Appendix C.1.
#
# Stages, in order (select a subset with STAGES="group allocate"):
#   data        CPU  resize FFHQ-256 to 64x64 with LANCZOS and write the dataset manifest
#   references  CPU  check the ADM evaluator and build the FFHQ-64 ADM reference batch
#   profile     GPU  PFI noise-level profile of every convolution filter (Section 3.2)
#   group       CPU  phase discovery (see the note in this stage)
#   allocate    CPU  Section 3.4 capacity allocation for the four students
#   plans       CPU  student architectures from the allocations (Appendix A.4)
#   train       GPU  hybrid distillation of each student (Eq. 2)
#   evaluate    GPU  NVLabs FID-50k (three seeds) and ADM metrics of the teacher and students
#   summarize   CPU  aggregate the three NVLabs FIDs and write the metric/throughput table
#
# Inputs you provide:
#   FFHQ256_SOURCE    canonical FFHQ at 256x256, a flat directory or ZIP of 70,000 images
#                     (default: $DATA_ROOT/ffhq256)
#   $REFERENCE_DIR/ffhq-64x64.npz
#                     NVLabs FID reference, https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
#   $ADM_DIR, ADM_PYTHON, ADM_DETECTOR
#                     guided-diffusion evaluator (commit 22e0df8183507e13a7813f8d38d51b072ca1e67c),
#                     its TensorFlow environment (reproduce/requirements-adm.txt) and
#                     classify_image_graph_def.pb
#   NVlabs/edm        cloned at EDM_REPO (default ../edm); its fid.py computes the NVLabs FID
# The teacher checkpoint is downloaded into MODEL_CACHE.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/ffhq64_unet)
#   FFHQ64         prepared 64x64 dataset (default: $DATA_ROOT/ffhq256_to64_lanczos_v1)
#   FFHQ_MANIFEST  dataset manifest (default: $OUT/dataset/manifest.json)
#   PROFILE_NPROC  processes for the profile (default: 8; the recorded run used 7)
#   PROFILE_JSON   profile read by group, allocate and plans (default: $OUT/profile/results.json;
#                  artifacts/profiles/ffhq64_ddpmpp_pfi.json.gz is the released one)
#   GROUPING_JSON  grouping read by allocate and plans (default: $OUT/grouping/timestep_grouping.json)
#   PLAN_ROOT      plans read by train (default: artifacts/plans/ffhq64, the trained plans)
#   VARIANTS       students to train and evaluate (default: the four paper variants)
#   TRAIN_NPROC    processes per student (default: 2, the recorded runs)
#   EVAL_NPROC     processes per evaluation (default: 4, the recorded runs)
#   RESUME=1       continue interrupted training runs

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/ffhq64_unet}"
FFHQ256_SOURCE="${FFHQ256_SOURCE:-$DATA_ROOT/ffhq256}"
FFHQ64="${FFHQ64:-$DATA_ROOT/ffhq256_to64_lanczos_v1}"
FFHQ_MANIFEST="${FFHQ_MANIFEST:-$OUT/dataset/manifest.json}"
PROFILE_NPROC="${PROFILE_NPROC:-8}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLAN_ROOT="${PLAN_ROOT:-artifacts/plans/ffhq64}"
VARIANTS="${VARIANTS:-$PAPER_VARIANTS}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
EVAL_NPROC="${EVAL_NPROC:-4}"
BENCH="$OUT/benchmark"
NVLABS_REF="$REFERENCE_DIR/ffhq-64x64.npz"
ADM_REF="$REFERENCE_DIR/VIRTUAL_ffhq64_first50k_adm.npz"

if stage data; then
  announce "data (CPU)"
  require_input "$FFHQ256_SOURCE" "FFHQ-256 source images"
  run "$PYTHON" scripts/data/prepare_ffhq_dataset.py \
    --source "$FFHQ256_SOURCE" --output-dir "$FFHQ64" --resolution 64 --workers 16
  run "$PYTHON" scripts/data/preflight_edm_dataset.py \
    --dataset ffhq --data-root "$FFHQ64" --dataset-manifest "$FFHQ_MANIFEST" \
    --dataset-split train --image-size 64 --ffhq-protocol ffhq256_numeric_v1
fi

if stage references; then
  announce "references (CPU)"
  require_input "$ADM_EVALUATOR" "guided-diffusion evaluator.py"
  run "$ADM_PYTHON" scripts/data/preflight_adm_evaluator.py --evaluator "$ADM_EVALUATOR"
  # Custom ADM reference: prepared FFHQ-64 images 00000 to 49999.
  run "$PYTHON" scripts/data/build_ffhq_adm_reference.py \
    --prepared-root "$FFHQ64" --output "$ADM_REF" \
    --adm-evaluator "$ADM_EVALUATOR" --adm-python "$ADM_PYTHON" --adm-detector "$ADM_DETECTOR"
fi

if stage profile; then
  announce "profile (GPU, $PROFILE_NPROC processes)"
  require_input "$FFHQ_MANIFEST" "FFHQ-64 dataset manifest (data stage)"
  # Recorded invocation of the released profile: 100 monitor images, all 256
  # sigma levels, one PFI batch per level.
  launch "$PROFILE_NPROC" scripts/evaluate_parameters_edm.py \
    --distributed-timeout-seconds 86400 \
    --dataset ffhq --data_root "$FFHQ64" --dataset-manifest "$FFHQ_MANIFEST" \
    --dataset-split monitor --ffhq-protocol ffhq256_numeric_v1 \
    --network-preset ffhq_64_vp --model-cache-dir "$MODEL_CACHE" \
    --output_dir "$OUT/profile" --grouping per_filter --confirm-full-profile \
    --ablation_mode pfi --pfi_seed 0 --seed 0 --max_images 100 \
    --num_sigma_levels 256 --sigma_stride 1 --num_bins 20 --batch_size 256 \
    --num_workers 4 --image_size 64 --device cuda --dtype fp32
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # The published FFHQ-64 phases [0, 3, 16, 20] are the K = 3 optimum of the
  # within-phase correlation cost (matrix_correlation), which is the recorded
  # setting and the grouping that the released allocations and plans use.
  run "$PYTHON" scripts/optimize_timestep_grouping.py \
    --matrix "$PROFILE_JSON" --matrix_key relative_delta_stack \
    --builtin_cost matrix_correlation --num_blocks 3 \
    --output "$GROUPING_JSON" \
    --plot_output "$(dirname "$GROUPING_JSON")/timestep_grouping_matrix.png" \
    --plot_matrix_source recomputed_similarity \
    --cost_curve_output "$(dirname "$GROUPING_JSON")/timestep_grouping_cost_curve.png"
  # For comparison only: the Section 3.3 objective with automatic K selects a
  # single phase on this profile. Nothing downstream reads this file.
  run "$PYTHON" scripts/optimize_timestep_grouping.py \
    --matrix "$PROFILE_JSON" --matrix_key relative_delta_stack \
    --builtin_cost matrix_correlation_cross_penalty --cross_block_lambda 0.02 \
    --cross_block_reward_normalization size --num_blocks auto \
    --output "$OUT/grouping_section3_objective/timestep_grouping.json"
fi

if stage allocate; then
  announce "allocate (CPU)"
  require_input "$PROFILE_JSON" "profile"
  require_input "$GROUPING_JSON" "grouping"
  run "$PYTHON" scripts/dry_run_capacity_allocation.py \
    --results-json "$PROFILE_JSON" --timestep-grouping "$GROUPING_JSON" \
    --allocation-results-dir "$OUT/allocations" \
    --allocation-metric delta_p_eff_geomean --score-reduction sum \
    --layer-score-source delta_p_eff_geomean --shuffle-seed 3 \
    --student-variant global --student-variant uniform_blockwise \
    --student-variant combined_blockwise --student-variant combined_layerwise
fi

if stage plans; then
  announce "plans (CPU, needs NVlabs/edm)"
  require_input "$PROFILE_JSON" "profile"
  run "$PYTHON" scripts/prepare_edm_distillation.py \
    --results-json "$PROFILE_JSON" --allocation-results-dir "$OUT/allocations" \
    --output-dir "$OUT/plans" --shuffle-seed 3 \
    --variant global --variant uniform_blockwise \
    --variant combined_blockwise --variant combined_layerwise
fi

if stage train; then
  require_input "$FFHQ_MANIFEST" "FFHQ-64 dataset manifest (data stage)"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC processes)"
    plan="$PLAN_ROOT/$variant/architecture_plan.json"
    require_input "$plan" "architecture plan"
    # 50,000 steps at 512 images per step (256 per process); the published
    # checkpoint is the one with the lowest EMA validation loss.
    launch "$TRAIN_NPROC" scripts/train_edm_distillation.py \
      --architecture-plan "$plan" --output-dir "$OUT/training/$variant/seed0" \
      --dataset ffhq --data-root "$FFHQ64" --dataset-manifest "$FFHQ_MANIFEST" \
      --dataset-split train --ffhq-protocol ffhq256_numeric_v1 \
      --model-cache-dir "$MODEL_CACHE" --device cuda --dtype fp32 --steps 50000 \
      --batch-size 512 --microbatch 256 --lr 2e-4 --seed 0 --num-workers 16 \
      --kd-weight 1.0 --data-weight 0.25 --ema-beta 0.999 --snapshot-every 5000 \
      --val-every 5000 --val-max-images 10000 --val-batch-size 256 --val-microbatch 64 --val-seed 12345 \
      --early-stop-patience 4 --early-stop-min-steps 10000 --early-stop-min-delta 1e-4 \
      --fid-every 0 --ddp-timeout-minutes 180 \
      --keep-last-snapshots 0 --checkpoint-selection best_val $(resume_flag)
  done
fi

model_args() {
  # Arguments that select the evaluated network: the teacher preset or a student checkpoint.
  if [[ "$1" == "teacher" ]]; then
    printf '%s\n' --network-preset ffhq_64_vp
  else
    printf '%s\n' --checkpoint "$OUT/training/$1/seed0/student-best-val.pt"
  fi
}

if stage evaluate; then
  require_input "$NVLABS_REF" "NVLabs FFHQ-64 FID reference"
  require_input "$ADM_REF" "FFHQ-64 ADM reference (references stage)"
  seeds=(0 50000 100000)
  for model in teacher $VARIANTS; do
    # Portable replacement for mapfile (bash 3.2).
    margs=()
    while IFS= read -r line; do margs+=("$line"); done < <(model_args "$model")
    nvlabs_dir="$BENCH/$model/ffhq64_nvlabs_edm_fid_v1"
    shared_dir="$BENCH/$model/shared/seed0_openai_truncate"
    for k in 0 1 2; do
      announce "evaluate $model, NVLabs FID run $k (GPU, $EVAL_NPROC processes)"
      secondary=()
      if [[ "$k" == "0" ]]; then
        # Run 0 also writes OpenAI-quantized copies of its images for the ADM metrics.
        secondary=(--secondary-samples-dir "$shared_dir" --secondary-quantizer openai_x_plus_1_x127_5)
      fi
      launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py "${margs[@]}" \
        --output-dir "$nvlabs_dir/run$k" --reference-stats "$NVLABS_REF" \
        --num-samples 50000 --seed "${seeds[$k]}" --num-steps 40 --batch-size-per-rank 32 \
        --metric-backend nvlabs_edm_fid --protocol-id ffhq64_nvlabs_edm_fid_v1 \
        --teacher-dtype fp32 --model-cache-dir "$MODEL_CACHE" --dataset-manifest "$FFHQ_MANIFEST" \
        --artifact-retention keep --preview-count 64 --nvlabs-edm-root "$EDM_REPO" \
        ${secondary[@]+"${secondary[@]}"}
    done
    announce "evaluate $model, ADM metrics on the run 0 images (GPU, $EVAL_NPROC processes)"
    launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py "${margs[@]}" \
      --reuse-samples --reuse-provenance "$nvlabs_dir/run0/evaluation_result.json" \
      --samples-dir "$shared_dir" --output-dir "$BENCH/$model/ffhq64_openai_adm_custom_first50k_v1" \
      --reference-stats "$ADM_REF" --num-samples 50000 --seed 0 --num-steps 40 --batch-size-per-rank 32 \
      --metric-backend openai_adm --protocol-id ffhq64_openai_adm_custom_first50k_v1 \
      --teacher-dtype fp32 --model-cache-dir "$MODEL_CACHE" --dataset-manifest "$FFHQ_MANIFEST" \
      --artifact-retention keep --preview-count 64 \
      --adm-evaluator "$ADM_EVALUATOR" --adm-python "$ADM_PYTHON" --adm-detector "$ADM_DETECTOR"
  done
fi

if stage summarize; then
  announce "summarize (CPU)"
  for model in teacher $VARIANTS; do
    nvlabs_dir="$BENCH/$model/ffhq64_nvlabs_edm_fid_v1"
    # The recorded tables report the minimum of the three NVLabs FIDs
    # (Appendix A.2 says mean; --aggregation mean gives the mean).
    run "$PYTHON" scripts/paper/summarize_edm_benchmark.py \
      --protocol-id ffhq64_nvlabs_edm_fid_v1 --metric fid_nvlabs_legacy --aggregation minimum \
      --output "$nvlabs_dir/evaluation_summary.json" \
      --result "$nvlabs_dir/run0/evaluation_result.json" \
      --result "$nvlabs_dir/run1/evaluation_result.json" \
      --result "$nvlabs_dir/run2/evaluation_result.json"
  done
  # Needs all five models (teacher and the four paper variants).
  run "$PYTHON" scripts/paper/plot_ffhq_benchmark_pareto.py --benchmark-root "$BENCH" --output-dir "$OUT/pareto"
fi
