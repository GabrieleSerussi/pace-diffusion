#!/usr/bin/env bash
# FFHQ 256, unconditional DiT-B/2 teacher trained from scratch in the latent
# space of the Stable Diffusion VAE (DDPM with 1,000 steps), 3 phases.
#
# Paper results: the FFHQ DiT rows of Tables 1 and 2. Most flag values below are
# recorded. The planner settings of
# the paper's split students are not recorded, so the plans stage uses the
# default settings of dit_arch_to_plans.py and its split students differ from
# the paper's.
#
# Stages, in order (select a subset with STAGES="group plans"):
#   teacher_plan  CPU  architecture of the DiT-B/2 teacher (129,548,576 parameters)
#   teacher       GPU  teacher training from scratch, 1,470 epochs, 2 processes
#   profile       GPU  head-level permutation profile of the teacher, 1,000 images
#   group         CPU  phase discovery with the Section 3.3 objective and automatic K
#   plans         CPU  NarrowDiT student plans for the four DiT variants
#   train         GPU  distillation of each variant, 550 epochs, 2 processes
#   references    CPU  pack an ADM reference batch from the training images (see FID_REF)
#   teacher_fid   GPU  10,000 teacher samples (DDPM, 250 steps, no guidance) and their ADM metrics
#   evaluate      GPU  ADM FID-10k of the saved checkpoints of each student
#   throughput    GPU  forward steps per second of each plan
#
# Inputs you provide:
#   FFHQ256_ROOT  flat directory of the 70,000 FFHQ images at 256x256 (default: $DATA_ROOT/ffhq256)
#   FID_REF       ADM reference batch of FFHQ 256 (default: $REFERENCE_DIR/ffhq256_ref_50k.npz).
#                 Its construction is not recorded; the references stage packs the
#                 first 50,000 images of FFHQ256_ROOT, which can differ from the
#                 reference behind the paper's numbers.
#   $ADM_DIR, ADM_PYTHON  guided-diffusion evaluator and its TensorFlow environment
#   facebookresearch/DiT at DIT_REPO (default ../DiT); pip install "pace-diffusion[dit]"
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT                 output directory (default: outputs/dit_ffhq256)
#   TEACHER_NPROC       processes for teacher training (default: 2)
#   NPROC               processes for profile, teacher_fid and evaluate (default: 8)
#   TRAIN_NPROC         processes per student (default: 2, the recorded runs)
#   PROFILE_CHECKPOINT  teacher checkpoint that is profiled (default: the final teacher,
#                       $OUT/teacher/phase_0/student.pt; the step behind the paper's
#                       FFHQ plans is not recorded)
#   PROFILE_JSON, GROUPING_JSON, PLANS_JSON  inputs of the later stages
#   VARIANTS            DiT variants (default: global uniform_blockwise blockwise_capacity layerwise_capacity)
#   NUM_PHASES          phases of the split variants (default: 3)
#   EVAL_STEPS          comma-separated checkpoint steps for evaluate (default: every saved
#                       step; the paper uses step 150000)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/dit_ffhq256}"
FFHQ256_ROOT="${FFHQ256_ROOT:-$DATA_ROOT/ffhq256}"
FID_REF="${FID_REF:-$REFERENCE_DIR/ffhq256_ref_50k.npz}"
TEACHER_NPROC="${TEACHER_NPROC:-2}"
NPROC="${NPROC:-8}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
TEACHER_DIR="$OUT/teacher"
PROFILE_CHECKPOINT="${PROFILE_CHECKPOINT:-$TEACHER_DIR/phase_0/student.pt}"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLANS_JSON="${PLANS_JSON:-$OUT/plans/plans.json}"
VARIANTS="${VARIANTS:-global uniform_blockwise blockwise_capacity layerwise_capacity}"
NUM_PHASES="${NUM_PHASES:-3}"
EVAL_STEPS="${EVAL_STEPS:-}"
TORCHRUN="$PYTHON -m torch.distributed.run --standalone --nproc-per-node $NPROC"

if stage teacher_plan; then
  announce "teacher_plan (CPU, needs facebookresearch/DiT)"
  # A one-group stub profile makes dit_arch_to_plans.py emit the full DiT-B/2 width.
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
  require_input "$FFHQ256_ROOT" "FFHQ 256 images"
  # Global batch 256 (2 processes x 32 x 4 accumulation steps), 273 steps per
  # epoch. The teacher encodes images on the fly because --hflip is not
  # combined with a latent cache. --teacher_checkpoint is required by the CLI
  # and not read with --kd_weight 0. The teacher is phase_0/student.pt.
  launch "$TEACHER_NPROC" scripts/train_phase_students.py \
    --model_type dit_xl --diffusion ddpm --arch_plan "$OUT/teacher_plan/plan.json" --variant global \
    --kd_weight 0 --gt_weight 1.0 --unconditional --teacher_checkpoint none \
    --dataset image_folder --image_root "$FFHQ256_ROOT" --image_size 256 --num_timesteps 1000 \
    --hflip --net_dropout 0.1 \
    --num_epochs 1470 --batch_size 32 --grad_accum 4 --lr 1e-4 --optimizer adamw --weight_decay 0 \
    --grad_clip 1.0 --ema_beta 0.9999 --lr_schedule wsd --warmup_steps 1000 --cooldown_frac 0.1 \
    --dtype bf16 --full_ckpt --ckpt_every 25000 --curve_keep 20 --output_dir "$TEACHER_DIR"
fi

if stage profile; then
  announce "profile (GPU, $NPROC processes)"
  require_input "$PROFILE_CHECKPOINT" "teacher checkpoint"
  launch "$NPROC" scripts/evaluate_parameters_dit.py \
    --arch_cfg "$TEACHER_DIR/phase_0/arch_cfg.json" --checkpoint "$PROFILE_CHECKPOINT" \
    --dataset image_folder --image_root "$FFHQ256_ROOT" --image_size 256 \
    --grouping attention_heads --ablation_mode permutation \
    --corruption_order level_major --permutation_group_by_level true \
    --num_timestep_levels 64 --num_bins 20 --max_images 1000 --batch_size 256 --vae_batch_size 32 \
    --dtype bf16 --seed 0 --class_idx_override 0 --importance_clip_mode post_agg \
    --output_dir "$OUT/profile"
fi

if stage group; then
  announce "group (CPU)"
  require_input "$PROFILE_JSON" "profile"
  # The paper's FFHQ DiT students use the phases [0, 4, 14, 20].
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
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC processes)"
    # Global batch 256 (2 processes x 32 x 4), 550 epochs (150,150 steps).
    launch "$TRAIN_NPROC" scripts/train_phase_students.py \
      --model_type dit_xl --diffusion ddpm --unconditional \
      --teacher_checkpoint "$TEACHER_DIR/phase_0/student.pt" \
      --teacher_arch_cfg "$TEACHER_DIR/phase_0/arch_cfg.json" \
      --arch_plan "$PLANS_JSON" --variant "$variant" \
      --dataset image_folder --image_root "$FFHQ256_ROOT" --image_size 256 --num_timesteps 1000 \
      --num_epochs 550 --batch_size 32 --grad_accum 4 --lr 2e-4 --gt_weight 0.25 --kd_weight 1.0 \
      --ema_beta 0.999 --optimizer adam --weight_decay 0 --lr_schedule wsd --warmup_steps 1000 \
      --cooldown_frac 0.2 --dtype bf16 --full_ckpt --ckpt_every 5000 --curve_keep 32 \
      --output_dir "$OUT/students/$variant"
  done
fi

if stage references; then
  announce "references (CPU)"
  require_input "$FFHQ256_ROOT" "FFHQ 256 images"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py --png_dir "$FFHQ256_ROOT" --out "$FID_REF" --size 256 --max_images 50000
fi

if stage teacher_fid; then
  announce "teacher_fid (GPU, $NPROC processes; the ADM evaluator runs on the CPU)"
  require_input "$TEACHER_DIR/phase_0/student.pt" "trained teacher"
  require_input "$FID_REF" "FFHQ 256 reference batch"
  launch "$NPROC" scripts/evaluate_students.py \
    --model_type dit_xl --mode composite --student_dir "$TEACHER_DIR" \
    --num_samples 10000 --batch_size 32 --num_steps 250 --num_timesteps 1000 \
    --sampler ddpm --diffusion ddpm --cfg_scale 1.0 --class_idx_override 0 --dtype bf16 --seed 0 \
    --skip_fid --output_dir "$OUT/teacher_samples"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py \
    --png_dir "$OUT/teacher_samples/samples" --out "$OUT/teacher_samples/samples.npz" --size 256
  run env CUDA_VISIBLE_DEVICES= "$ADM_PYTHON" "$ADM_DIR/evaluator.py" "$FID_REF" "$OUT/teacher_samples/samples.npz"
fi

if stage evaluate; then
  require_input "$FID_REF" "FFHQ 256 reference batch"
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
      --num_samples 10000 --num_steps 250 --cfg_scale 1.0 --sampler ddpm --dtype bf16 \
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
