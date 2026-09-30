#!/usr/bin/env bash
# LSUN Bedroom 256, unconditional DiT-B/2 teacher trained from scratch in the
# latent space of the Stable Diffusion VAE (DDPM with 1,000 steps), 3 phases.
#
# Paper results: the LSUN Bedroom DiT rows of Tables 1 and 2. Most flag values
# below are recorded. The planner
# settings of the paper's split students are not recorded, so the plans stage
# uses the default settings of dit_arch_to_plans.py and its split students
# differ from the paper's.
#
# Stages, in order (select a subset with STAGES="group plans"):
#   latents         GPU  encode the 1,000,000 training images once with the VAE
#   teacher_plan    CPU  architecture of the DiT-B/2 teacher (129,548,576 parameters)
#   teacher         GPU  teacher training from scratch, 105 epochs, 2 processes
#   teacher_extend  GPU  continue the teacher from its pre-cooldown checkpoint to 258 epochs, 8 processes
#   pin_teacher     CPU  use the extension checkpoint of step 725,000 as the teacher
#   profile         GPU  head-level permutation profile of the extension checkpoint of step 500,000
#   group           CPU  phase discovery with the Section 3.3 objective and automatic K
#   plans           CPU  NarrowDiT student plans for the four DiT variants
#   train           GPU  distillation of each variant, 64 epochs, 4 processes
#   teacher_fid     GPU  10,000 teacher samples (DDPM, 1,000 steps, no guidance) and their ADM metrics
#   evaluate        GPU  ADM FID-10k of the saved checkpoints of each student
#   throughput      GPU  forward steps per second of each plan
#
# Inputs you provide:
#   LSUN_ROOT   the first 1,000,000 images of the LSUN bedroom_train_lmdb, written as raw
#               JPEG bytes to 0000000.jpg ... 0999999.jpg (default: $DATA_ROOT/lsun_bedroom256/img256);
#               the trainer crops and resizes them
#   $REFERENCE_DIR/VIRTUAL_lsun_bedroom256.npz, $ADM_DIR, ADM_PYTHON
#               guided-diffusion reference batch, evaluator and its TensorFlow environment
#   facebookresearch/DiT at DIT_REPO (default ../DiT); pip install "pace-diffusion[dit]"
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT                 output directory (default: outputs/dit_lsun256)
#   LATENT_CACHE        VAE latent cache (default: $OUT/latent_cache)
#   TEACHER_NPROC       processes for the base teacher run (default: 2)
#   NPROC               processes for latents, teacher_extend, profile, teacher_fid and evaluate (default: 8)
#   TRAIN_NPROC         processes per student (default: 4, the recorded runs)
#   PROFILE_CHECKPOINT  profiled checkpoint (default: $OUT/teacher_ext/phase_0/curve/step_500000.pt)
#   PROFILE_JSON, GROUPING_JSON, PLANS_JSON  inputs of the later stages
#   VARIANTS            DiT variants (default: global uniform_blockwise blockwise_capacity layerwise_capacity)
#   NUM_PHASES          phases of the split variants (default: 3)
#   EVAL_STEPS          comma-separated checkpoint steps for evaluate (default: every saved step;
#                       the paper uses steps 235000 to 245000)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/dit_lsun256}"
LSUN_ROOT="${LSUN_ROOT:-$DATA_ROOT/lsun_bedroom256/img256}"
LATENT_CACHE="${LATENT_CACHE:-$OUT/latent_cache}"
FID_REF="$REFERENCE_DIR/VIRTUAL_lsun_bedroom256.npz"
TEACHER_NPROC="${TEACHER_NPROC:-2}"
NPROC="${NPROC:-8}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
TEACHER_BASE="$OUT/teacher_base"
TEACHER_EXT="$OUT/teacher_ext"
TEACHER_DIR="$OUT/teacher"
PROFILE_CHECKPOINT="${PROFILE_CHECKPOINT:-$TEACHER_EXT/phase_0/curve/step_500000.pt}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLANS_JSON="${PLANS_JSON:-$OUT/plans/plans.json}"
VARIANTS="${VARIANTS:-global uniform_blockwise blockwise_capacity layerwise_capacity}"
NUM_PHASES="${NUM_PHASES:-3}"
EVAL_STEPS="${EVAL_STEPS:-}"
TORCHRUN="$PYTHON -m torch.distributed.run --standalone --nproc-per-node $NPROC"

teacher_args=(
  --model_type dit_xl --diffusion ddpm --arch_plan "$OUT/teacher_plan/plan.json" --variant global
  --kd_weight 0 --gt_weight 1.0 --unconditional --teacher_checkpoint none
  --dataset image_folder --image_root "$LSUN_ROOT" --max_images 1000000 --latent_cache_dir "$LATENT_CACHE"
  --image_size 256 --num_timesteps 1000
  --lr 1e-4 --optimizer adamw --weight_decay 0 --grad_clip 1.0 --ema_beta 0.9999
  --lr_schedule wsd --warmup_steps 1000 --cooldown_frac 0.1 --dtype bf16
  --full_ckpt --ckpt_every 25000 --curve_keep 20
)

if stage latents; then
  announce "latents (GPU, $NPROC processes)"
  require_input "$LSUN_ROOT" "LSUN Bedroom images"
  # --max_images is part of the cache fingerprint and must match training.
  launch "$NPROC" scripts/data/precompute_latent_cache.py \
    --image_root "$LSUN_ROOT" --image_size 256 --max_images 1000000 --latent_cache_dir "$LATENT_CACHE"
fi

if stage teacher_plan; then
  announce "teacher_plan (CPU, needs facebookresearch/DiT)"
  write_text "$OUT/teacher_plan/stub_results.json" \
    '{"n_eff": [1.0], "group_names": ["blocks.0.attn.head_0"], "relative_delta_stack": [[0.0]]}'
  write_text "$OUT/teacher_plan/stub_grouping.json" '{"boundaries": [0, 1]}'
  run "$PYTHON" scripts/dit_arch_to_plans.py \
    --results_json "$OUT/teacher_plan/stub_results.json" \
    --grouping_json "$OUT/teacher_plan/stub_grouping.json" \
    --model dit_b --variants global --out "$OUT/teacher_plan/plan.json"
fi

if stage teacher; then
  announce "teacher (GPU, $TEACHER_NPROC processes)"
  # Global batch 256 (2 processes x 32 x 4), 3,906 steps per epoch; the
  # pre-cooldown checkpoint is written at step 369,117.
  launch "$TEACHER_NPROC" scripts/train_phase_students.py "${teacher_args[@]}" \
    --num_epochs 105 --batch_size 32 --grad_accum 4 --output_dir "$TEACHER_BASE"
fi

if stage teacher_extend; then
  announce "teacher_extend (GPU, $NPROC processes)"
  require_input "$TEACHER_BASE/phase_0/pre_cooldown.pt" "pre-cooldown teacher checkpoint"
  # The extension resumes the full pre-cooldown state with a longer schedule
  # (258 epochs, global batch 256 as 8 processes x 32); the learning rate stays
  # constant past step 725,000.
  make_dirs "$TEACHER_EXT/phase_0"
  run cp "$TEACHER_BASE/phase_0/pre_cooldown.pt" "$TEACHER_EXT/phase_0/last.pt"
  run cp "$TEACHER_BASE/phase_0/arch_cfg.json" "$TEACHER_EXT/phase_0/arch_cfg.json"
  launch "$NPROC" scripts/train_phase_students.py "${teacher_args[@]}" \
    --num_epochs 258 --batch_size 32 --grad_accum 1 --num_workers 8 --output_dir "$TEACHER_EXT"
fi

if stage pin_teacher; then
  announce "pin_teacher (CPU)"
  require_input "$TEACHER_EXT/phase_0/curve/step_725000.pt" "teacher extension checkpoint of step 725,000"
  make_dirs "$TEACHER_DIR/phase_0"
  run cp "$TEACHER_EXT/phase_0/curve/step_725000.pt" "$TEACHER_DIR/phase_0/student.pt"
  run cp "$TEACHER_EXT/phase_0/arch_cfg.json" "$TEACHER_DIR/phase_0/arch_cfg.json"
fi

if stage profile; then
  announce "profile (GPU, $NPROC processes)"
  require_input "$PROFILE_CHECKPOINT" "profiled teacher checkpoint"
  launch "$NPROC" scripts/evaluate_parameters_dit.py \
    --arch_cfg "$TEACHER_EXT/phase_0/arch_cfg.json" --checkpoint "$PROFILE_CHECKPOINT" \
    --dataset image_folder --image_root "$LSUN_ROOT" --image_size 256 \
    --grouping attention_heads --ablation_mode permutation \
    --corruption_order level_major --permutation_group_by_level true \
    --num_timestep_levels 64 --num_bins 20 --max_images 1000 --batch_size 256 --vae_batch_size 32 \
    --dtype bf16 --seed 0 --class_idx_override 0 --importance_clip_mode post_agg \
    --output_dir "$OUT/profile"
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # The paper's LSUN DiT students use the phases [0, 4, 7, 20].
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
  run "$PYTHON" scripts/dit_arch_to_plans.py \
    --results_json "$PROFILE_JSON" --grouping_json "$GROUPING_JSON" --model dit_b \
    --variants global,uniform_blockwise,blockwise_capacity,layerwise_capacity --out "$PLANS_JSON"
fi

if stage train; then
  require_input "$PLANS_JSON" "plans"
  require_input "$TEACHER_DIR/phase_0/student.pt" "pinned teacher (pin_teacher stage)"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC processes)"
    # Global batch 256. The recorded students ran 38 epochs with a cooldown and
    # were then resumed from their pre-cooldown checkpoints to 64 epochs
    # without a cooldown; one run of 64 epochs with --cooldown_frac 0 has the
    # same learning-rate schedule. The split of 256 into processes, batch and
    # accumulation is not recorded; 4 x 64 x 1 is used here.
    launch "$TRAIN_NPROC" scripts/train_phase_students.py \
      --model_type dit_xl --diffusion ddpm --unconditional \
      --teacher_checkpoint "$TEACHER_DIR/phase_0/student.pt" \
      --teacher_arch_cfg "$TEACHER_DIR/phase_0/arch_cfg.json" \
      --arch_plan "$PLANS_JSON" --variant "$variant" \
      --dataset image_folder --image_root "$LSUN_ROOT" --max_images 1000000 \
      --latent_cache_dir "$LATENT_CACHE" --image_size 256 --num_timesteps 1000 \
      --num_epochs 64 --batch_size 64 --grad_accum 1 --lr 2e-4 --gt_weight 0.25 --kd_weight 1.0 \
      --ema_beta 0.999 --optimizer adam --weight_decay 0 --lr_schedule wsd --warmup_steps 1000 \
      --cooldown_frac 0 --dtype bf16 --full_ckpt --ckpt_every 5000 --curve_keep 32 \
      --output_dir "$OUT/students/$variant"
  done
fi

if stage teacher_fid; then
  announce "teacher_fid (GPU, $NPROC processes; the ADM evaluator runs on the CPU)"
  require_input "$TEACHER_DIR/phase_0/student.pt" "pinned teacher (pin_teacher stage)"
  require_input "$FID_REF" "LSUN Bedroom reference batch"
  launch "$NPROC" scripts/evaluate_students.py \
    --model_type dit_xl --mode composite --student_dir "$TEACHER_DIR" \
    --num_samples 10000 --batch_size 32 --num_steps 1000 --num_timesteps 1000 \
    --sampler ddpm --diffusion ddpm --cfg_scale 1.0 --class_idx_override 0 --dtype bf16 --seed 0 \
    --skip_fid --output_dir "$OUT/teacher_samples"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py \
    --png_dir "$OUT/teacher_samples/samples" --out "$OUT/teacher_samples/samples.npz" --size 256
  run env CUDA_VISIBLE_DEVICES= "$ADM_PYTHON" "$ADM_DIR/evaluator.py" "$FID_REF" "$OUT/teacher_samples/samples.npz"
fi

if stage evaluate; then
  require_input "$FID_REF" "LSUN Bedroom reference batch"
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
      --student_dir "$OUT/students/$variant" --model_type dit_xl --diffusion ddpm \
      --num_samples 10000 --num_steps 1000 --cfg_scale 1.0 --sampler ddpm --dtype bf16 \
      --class_idx_override 0 --ref_npz "$FID_REF" --pack_size 256 \
      --adm_python "$ADM_PYTHON" --adm_dir "$ADM_DIR" \
      --num_phases "$phases" --out_json "$OUT/students/$variant/curve.json" \
      --torchrun "$TORCHRUN" --repo "$PACE_ROOT" ${steps_args[@]+"${steps_args[@]}"}
  done
fi

if stage throughput; then
  announce "throughput (1 GPU)"
  require_input "$PLANS_JSON" "plans"
  run "$PYTHON" scripts/bench_throughput.py \
    --arch_plan "$PLANS_JSON" --variants global,uniform_blockwise,blockwise_capacity,layerwise_capacity \
    --batch 32 --dtype bf16 --attn_impl sdpa --mode infer --num_bins 20 --out "$OUT/throughput.json"
fi
