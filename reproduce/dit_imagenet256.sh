#!/usr/bin/env bash
# ImageNet 256, class-conditional DiT-XL/2 teacher of facebookresearch/DiT
# (latent space, DDPM with 1,000 steps).
#
# Paper results: the ImageNet DiT rows of Tables 1 and 2. The settings below are
# the recorded ones where a record exists. The planner settings of the split students of the
# paper (Uniform, Phase-aware blockwise and layerwise) are not recorded, so the
# plans stage builds them with the default settings of dit_arch_to_plans.py and
# its split students differ from the paper's.
#
# Stages, in order (select a subset with STAGES="group plans"):
#   profile      GPU  head-level permutation profile of the teacher (legacy protocol, 1,000 images)
#   group        CPU  phase discovery with the Section 3.3 objective and automatic K
#   plans        CPU  NarrowDiT student plans for the four DiT variants
#   train        GPU  distillation of each variant: WSD schedule, 160 epochs, 8 processes
#   teacher_fid  GPU  10,000 teacher samples (DDPM, 250 steps, guidance 1.5) and their ADM metrics
#   evaluate     GPU  ADM FID-10k of the saved checkpoints of each student (eval_composite_curve.py)
#   throughput   GPU  forward steps per second of each plan (bench_throughput.py)
#
# Inputs you provide:
#   DIT_CHECKPOINT    DiT-XL-2-256x256.pt (python $DIT_REPO/download.py, or the Hugging Face
#                     repository facebook/DiT-XL-2-256; default: $MODEL_CACHE/DiT-XL-2-256x256.pt)
#   IMAGENET_PARQUET  ImageNet-1k parquet shards train-*.parquet (default: $DATA_ROOT/imagenet-1k/data)
#   $REFERENCE_DIR/VIRTUAL_imagenet256_labeled.npz, $ADM_DIR, ADM_PYTHON
#                     guided-diffusion reference batch, evaluator and its TensorFlow environment
#   facebookresearch/DiT at DIT_REPO (default ../DiT); pip install "pace-diffusion[dit]"
# The Stable Diffusion VAE (stabilityai/sd-vae-ft-mse) is downloaded by diffusers.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/dit_imagenet256)
#   NPROC          processes for profile, train, teacher_fid and evaluate (default: 8)
#   PROFILE_JSON   profile read by group and plans (default: $OUT/profile/results.json)
#   GROUPING_JSON  grouping read by plans (default: $OUT/grouping/timestep_grouping.json)
#   PLANS_JSON     plans read by train, evaluate and throughput (default: $OUT/plans/plans.json)
#   VARIANTS       DiT variants (default: global uniform_blockwise blockwise_capacity layerwise_capacity)
#   NUM_PHASES     phases of the split variants (default: 2, the phases of the paper's students)
#   EVAL_STEPS     comma-separated checkpoint steps for evaluate (default: every saved step;
#                  the paper's Global student is step 800000)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/dit_imagenet256}"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-$MODEL_CACHE/DiT-XL-2-256x256.pt}"
IMAGENET_PARQUET="${IMAGENET_PARQUET:-$DATA_ROOT/imagenet-1k/data}"
NPROC="${NPROC:-8}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLANS_JSON="${PLANS_JSON:-$OUT/plans/plans.json}"
VARIANTS="${VARIANTS:-global uniform_blockwise blockwise_capacity layerwise_capacity}"
NUM_PHASES="${NUM_PHASES:-2}"
EVAL_STEPS="${EVAL_STEPS:-}"
REF_NPZ="$REFERENCE_DIR/VIRTUAL_imagenet256_labeled.npz"
TORCHRUN="$PYTHON -m torch.distributed.run --standalone --nproc-per-node $NPROC"

if stage profile; then
  announce "profile (GPU, $NPROC processes)"
  require_input "$DIT_CHECKPOINT" "DiT-XL/2 checkpoint"
  # Recorded settings of the legacy DiT-XL/2 profile: image-major sample order
  # and a batch permutation across noise levels (see dit_micro_cifar10.sh), 64
  # timestep levels, batch 16; its grouping is [0, 14, 20]. A later re-profile
  # used --corruption_order level_major --permutation_group_by_level true
  # --batch_size 256 --vae_batch_size 32. Which of the two produced the plans of
  # the paper is not recorded.
  launch "$NPROC" scripts/evaluate_parameters_dit.py \
    --grouping attention_heads --ablation_mode permutation \
    --corruption_order image_major --permutation_group_by_level false \
    --dataset imagenet1k_parquet --parquet_split_prefix train- --data_root "$IMAGENET_PARQUET" \
    --checkpoint "$DIT_CHECKPOINT" --num_timestep_levels 64 --num_bins 20 --max_images 1000 \
    --batch_size 16 --dtype bf16 --seed 0 --output_dir "$OUT/profile"
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  run "$PYTHON" scripts/optimize_timestep_grouping.py \
    --matrix "$PROFILE_JSON" --matrix_key relative_delta_stack \
    --builtin_cost matrix_correlation_cross_penalty --cross_block_lambda 0.02 \
    --cross_block_reward_normalization size --num_blocks auto \
    --output "$GROUPING_JSON" \
    --plot_output "$(dirname "$GROUPING_JSON")/grouping_matrix.png" --plot_matrix_source recomputed_similarity \
    --cost_curve_output "$(dirname "$GROUPING_JSON")/cost_curve.png"
fi

if stage plans; then
  announce "plans (CPU, needs facebookresearch/DiT)"
  require_input "$PROFILE_JSON" "profile"
  require_input "$GROUPING_JSON" "grouping"
  # dit_arch_to_plans.py sets phase budgets from n_eff and layer widths from
  # relative deltas; it is not the Section 3.4 allocator (REPRODUCING.md).
  run "$PYTHON" scripts/dit_arch_to_plans.py \
    --results_json "$PROFILE_JSON" --grouping_json "$GROUPING_JSON" --model dit_xl \
    --variants global,uniform_blockwise,blockwise_capacity,layerwise_capacity --out "$PLANS_JSON"
fi

if stage train; then
  require_input "$PLANS_JSON" "plans"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $NPROC processes)"
    # 160 epochs of 5,004 steps at a global batch of 256: constant learning rate
    # after 1,000 warm-up steps, then a cooldown over the last 20 percent. The
    # recorded Global student was trained in legs resumed from pre-cooldown
    # checkpoints that end with this configuration; the paper uses its step
    # 800,000. --cfg_label_drop 0.1 is not recorded; guidance at sampling time
    # needs a trained null class. Each phase specialist of a split variant is
    # trained for the full schedule.
    launch "$NPROC" scripts/train_phase_students.py \
      --model_type dit_xl --diffusion ddpm \
      --teacher_checkpoint "$DIT_CHECKPOINT" --arch_plan "$PLANS_JSON" --variant "$variant" \
      --dataset imagenet1k_parquet --data_root "$IMAGENET_PARQUET" --parquet_split_prefix train- \
      --image_size 256 --num_timesteps 1000 --vae_batch_size 128 --num_workers 8 \
      --num_epochs 160 --batch_size 32 --lr 2e-4 --dtype bf16 --gt_weight 0.25 --kd_weight 1.0 \
      --ema_beta 0.999 --optimizer adam --weight_decay 0 --lr_schedule wsd --warmup_steps 1000 \
      --cooldown_frac 0.2 --cfg_label_drop 0.1 --full_ckpt --ckpt_every 10000 --curve_keep 32 \
      --output_dir "$OUT/students/$variant"
  done
fi

if stage teacher_fid; then
  announce "teacher_fid (GPU, $NPROC processes; the ADM evaluator runs on the CPU)"
  require_input "$DIT_CHECKPOINT" "DiT-XL/2 checkpoint"
  require_input "$REF_NPZ" "ImageNet 256 reference batch"
  launch "$NPROC" scripts/evaluate_students.py \
    --model_type dit_xl --mode teacher --teacher_checkpoint "$DIT_CHECKPOINT" \
    --num_samples 10000 --batch_size 32 --num_steps 250 --num_timesteps 1000 \
    --sampler ddpm --diffusion ddpm --cfg_scale 1.5 --dtype bf16 --seed 0 --skip_fid \
    --output_dir "$OUT/teacher_samples"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py \
    --png_dir "$OUT/teacher_samples/samples" --out "$OUT/teacher_samples/samples.npz" --size 256
  run env CUDA_VISIBLE_DEVICES= "$ADM_PYTHON" "$ADM_DIR/evaluator.py" "$REF_NPZ" "$OUT/teacher_samples/samples.npz"
fi

if stage evaluate; then
  require_input "$REF_NPZ" "ImageNet 256 reference batch"
  steps_args=()
  if [[ -n "$EVAL_STEPS" ]]; then
    steps_args=(--steps "$EVAL_STEPS")
  fi
  for variant in $VARIANTS; do
    announce "evaluate $variant (GPU, $NPROC processes; the ADM evaluator runs on the CPU)"
    phases="$NUM_PHASES"
    if [[ "$variant" == "global" ]]; then
      phases=1
    fi
    run "$PYTHON" scripts/eval_composite_curve.py \
      --student_dir "$OUT/students/$variant" --model_type dit_xl --diffusion ddpm --num_heads 16 \
      --num_samples 10000 --num_steps 250 --cfg_scale 1.5 --sampler ddpm --dtype bf16 \
      --ref_npz "$REF_NPZ" --pack_size 256 --adm_python "$ADM_PYTHON" --adm_dir "$ADM_DIR" \
      --num_phases "$phases" --out_json "$OUT/students/$variant/curve.json" \
      --torchrun "$TORCHRUN" --repo "$PACE_ROOT" ${steps_args[@]+"${steps_args[@]}"}
  done
fi

if stage throughput; then
  announce "throughput (1 GPU)"
  require_input "$PLANS_JSON" "plans"
  run "$PYTHON" scripts/bench_throughput.py \
    --arch_plan "$PLANS_JSON" --variants global,uniform_blockwise,blockwise_capacity,layerwise_capacity \
    --batch 32 --dtype bf16 --attn_impl sdpa --mode infer --num_bins 20 \
    --out "$OUT/throughput.json"
fi
