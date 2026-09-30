#!/usr/bin/env bash
# ImageNet-64, class-conditional ADM teacher of NVlabs/edm.
#
# Paper results: the ImageNet-64 ConvNet rows of Tables 1 and 2, the ImageNet-64
# row of Table 3 and Figure 5(b).
#
# Stages, in order (select a subset with STAGES="group allocate"):
#   data        CPU  convert the ImageNet parquet shards to WebDataset shards for training
#   profile     GPU  noise-level profile with the random_same_norm protocol (Section 3.2, Appendix C.1)
#   group       CPU  phase discovery with the Section 3.3 objective and automatic K
#   allocate    CPU  Section 3.4 capacity allocation for the four students
#   plans       CPU  student architectures from the allocations (Appendix A.4)
#   train       GPU  hybrid distillation of each student (Eq. 2), with Clean-FID-5k monitoring
#   evaluate    GPU  Clean-FID-50k of the best-FID checkpoint, then a 5k run for throughput
#
# Inputs you provide:
#   IMAGENET_PARQUET  ImageNet-1k in the Hugging Face parquet layout (train-*.parquet,
#                     validation-*.parquet; default: $DATA_ROOT/imagenet-1k/data)
#   NVlabs/edm        cloned at EDM_REPO (default ../edm)
# The teacher checkpoint is downloaded into MODEL_CACHE.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/imagenet64_unet)
#   IMAGENET_WDS   WebDataset shards written by the data stage (default: $DATA_ROOT/imagenet-1k/webdataset)
#   PROFILE_NPROC  processes for the profile (default: 8, the recorded run)
#   PROFILE_JSON   profile read by group, allocate and plans (default: $OUT/profile/results.json;
#                  artifacts/profiles/imagenet64_adm_random_same_norm.json.gz is the released one)
#   GROUPING_JSON  grouping read by allocate and plans (default: $OUT/grouping/timestep_grouping.json)
#   PLAN_ROOT      plans read by train (default: artifacts/plans/imagenet64, the trained plans)
#   VARIANTS       students to train and evaluate (default: the four paper variants)
#   TRAIN_NPROC    processes per student (default: 1, the recorded runs)
#   STEPS          optimizer steps per student (default: 120000, the recorded length)
#   RESUME=1       continue interrupted training runs
#   EVAL_NPROC     processes per evaluation (default: 1, the recorded runs)
#   REF_STATS      Clean-FID reference statistics of the ImageNet-64 validation split.
#                  The first FID call of training writes them to
#                  $OUT/fid_reference_stats/imagenet_clean_validation_64_<hash>.npz.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/imagenet64_unet}"
IMAGENET_PARQUET="${IMAGENET_PARQUET:-$DATA_ROOT/imagenet-1k/data}"
IMAGENET_WDS="${IMAGENET_WDS:-$DATA_ROOT/imagenet-1k/webdataset}"
PROFILE_NPROC="${PROFILE_NPROC:-8}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLAN_ROOT="${PLAN_ROOT:-artifacts/plans/imagenet64}"
VARIANTS="${VARIANTS:-$PAPER_VARIANTS}"
TRAIN_NPROC="${TRAIN_NPROC:-1}"
STEPS="${STEPS:-120000}"
EVAL_NPROC="${EVAL_NPROC:-1}"
TEACHER_URL="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"

if stage data; then
  announce "data (CPU)"
  require_input "$IMAGENET_PARQUET" "ImageNet-1k parquet shards"
  run "$PYTHON" scripts/data/convert_imagenet_parquet_to_webdataset.py \
    --input-root "$IMAGENET_PARQUET" --output-root "$IMAGENET_WDS" \
    --splits train validation --samples-per-shard 4096 --parquet-batch-size 1024 --jpeg-quality 95
fi

if stage profile; then
  announce "profile (GPU, $PROFILE_NPROC processes)"
  require_input "$IMAGENET_PARQUET" "ImageNet-1k parquet shards"
  # Recorded invocation of the released profile (see cifar10_unet.sh for the
  # sample order and pickle flags).
  launch "$PROFILE_NPROC" scripts/evaluate_parameters_edm.py \
    --dataset imagenet1k_parquet --parquet_split validation --data_root "$IMAGENET_PARQUET" \
    --network_pkl "$TEACHER_URL" --model-cache-dir "$MODEL_CACHE" --trust-local-pickle \
    --output_dir "$OUT/profile" --device cuda --dtype fp32 --seed 0 \
    --batch_size 1000 --grouping per_filter --num_bins 20 --num_sigma_levels 256 --sigma_stride 1 \
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
  run "$PYTHON" scripts/prepare_edm_distillation.py \
    --results-json "$PROFILE_JSON" --allocation-results-dir "$OUT/allocations" \
    --output-dir "$OUT/plans" \
    --variant global --variant uniform_blockwise \
    --variant combined_blockwise --variant combined_layerwise
fi

if stage train; then
  require_input "$IMAGENET_WDS" "ImageNet WebDataset shards (data stage)"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC process(es))"
    plan="$PLAN_ROOT/$variant/architecture_plan.json"
    require_input "$plan" "architecture plan"
    # The recorded runs trained 50,000 steps and were resumed to 120,000 steps;
    # the learning rate is constant, so one run of 120,000 steps has the same
    # schedule. RESUME=1 continues an interrupted run.
    launch "$TRAIN_NPROC" scripts/train_edm_distillation.py \
      --architecture-plan "$plan" --output-dir "$OUT/training/$variant/seed0" \
      --dataset imagenet1k_webdataset --data-root "$IMAGENET_WDS" \
      --parquet-split train --val-parquet-split validation --webdataset-shuffle-buffer 2048 \
      --model-cache-dir "$MODEL_CACHE" --trust-local-pickle \
      --device cuda --dtype fp32 --steps "$STEPS" \
      --batch-size 256 --microbatch 24 --lr 2e-4 --seed 0 --num-workers 20 \
      --kd-weight 1.0 --data-weight 0.25 --ema-beta 0.999 --snapshot-every 5000 \
      --val-every 5000 --val-max-images 2048 --val-batch-size 64 --val-microbatch 16 --val-seed 12345 \
      --early-stop-patience 4 --early-stop-min-steps 10000 --early-stop-min-delta 1e-4 \
      --fid-every 1000 --fid-num-samples 5000 --fid-batch-size 16 --fid-num-steps 18 \
      --fid-ref-stats-cache-dir "$OUT/fid_reference_stats" --fid-feature-batch-size 32 \
      --fid-ref-split validation --fid-ref-dataset-name imagenet --fid-ref-dataset-res 64 \
      --fid-mode clean --fid-seed 0 $(resume_flag)
  done
fi

if stage evaluate; then
  ref_stats="${REF_STATS:-}"
  if [[ -z "$ref_stats" ]]; then
    for candidate in "$OUT"/fid_reference_stats/imagenet_clean_validation_64_*.npz; do
      ref_stats="$candidate"
      break
    done
  fi
  require_input "$ref_stats" "ImageNet-64 validation reference statistics (set REF_STATS)"
  for variant in $VARIANTS; do
    announce "evaluate $variant (GPU, $EVAL_NPROC process(es))"
    checkpoint="$OUT/training/$variant/seed0/student-best-fid.pt"
    require_input "$checkpoint" "best-FID checkpoint"
    # Clean-FID-50k, 18 Heun steps, labels = seed modulo 1000 (Table 2).
    launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py \
      --checkpoint "$checkpoint" --weights ema \
      --output-dir "$OUT/evaluation/$variant/best_fid_50k_validation" \
      --samples-dir "$OUT/evaluation_samples/$variant/best_fid_50k_validation" \
      --reference-stats "$ref_stats" --num-samples 50000 --seed 0 --num-steps 18 \
      --batch-size-per-rank 16 --feature-batch-size 32 --num-workers 12 --label-mode seed_modulo \
      --discard-samples
    # Throughput (Table 1): the recorded values come from a separate fresh 5k run.
    launch "$EVAL_NPROC" scripts/evaluate_edm_checkpoint.py \
      --checkpoint "$checkpoint" --weights ema \
      --output-dir "$OUT/evaluation/$variant/best_fid_5k_validation" \
      --samples-dir "$OUT/evaluation_samples/$variant/best_fid_5k_validation" \
      --reference-stats "$ref_stats" --num-samples 5000 --seed 0 --num-steps 18 \
      --batch-size-per-rank 16 --feature-batch-size 32 --num-workers 12 --label-mode seed_modulo \
      --discard-samples
  done
fi
