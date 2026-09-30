#!/usr/bin/env bash
# CIFAR-10, class-conditional pixel-space DiT teacher (DiT-S width: hidden size
# 384, depth 12, 6 heads) trained from scratch with the EDM formulation, 2 phases.
#
# Paper results: the CIFAR-10 DiT rows of Tables 1 and 2. This experiment has the
# fewest records: the teacher recipe is recorded, the profile, grouping and
# student settings are partly inferred, and the
# planner settings of the paper's split students are not recorded.
#
# Stages, in order (select a subset with STAGES="group plans"):
#   teacher_plan  CPU  architecture of the teacher (32,475,660 parameters)
#   teacher       GPU  teacher training from scratch, 4,000 epochs, 8 processes
#   references    CPU  ADM reference batch from the 10,000 CIFAR-10 test images
#   teacher_fid   GPU  10,000 teacher samples (18 Heun steps, guidance 1.25) and their ADM metrics
#   profile       GPU  head-level permutation profile of the teacher, 1,000 training images
#   group         CPU  phase discovery with the Section 3.3 objective and automatic K
#   plans         CPU  NarrowDiT student plans for the four DiT variants
#   train         GPU  distillation of each variant, 3,200 epochs, 2 processes
#   evaluate      GPU  ADM FID-10k of the saved checkpoints of each student
#
# Inputs: facebookresearch/DiT at DIT_REPO (default ../DiT), pip install
# "pace-diffusion[dit]", and the guided-diffusion evaluator ($ADM_DIR, ADM_PYTHON).
# CIFAR-10 is downloaded into DATA_ROOT.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT            output directory (default: outputs/dit_cifar10)
#   NPROC          processes for teacher, teacher_fid, profile and evaluate (default: 8)
#   TRAIN_NPROC    processes per student (default: 2, the recorded runs)
#   PROFILE_JSON, GROUPING_JSON, PLANS_JSON  inputs of the later stages
#   VARIANTS       DiT variants (default: global uniform_blockwise blockwise_capacity layerwise_capacity)
#   NUM_PHASES     phases of the split variants (default: 2)
#   EVAL_STEPS     comma-separated checkpoint steps for evaluate (default: every saved step;
#                  the paper uses step 306000)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="${OUT:-$OUTPUT_ROOT/dit_cifar10}"
NPROC="${NPROC:-8}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
TEACHER_DIR="$OUT/teacher"
FID_REF="$REFERENCE_DIR/cifar_test_ref_32.npz"
PROFILE_JSON="${PROFILE_JSON:-$OUT/profile/results.json}"
GROUPING_JSON="${GROUPING_JSON:-$OUT/grouping/timestep_grouping.json}"
PLANS_JSON="${PLANS_JSON:-$OUT/plans/plans.json}"
VARIANTS="${VARIANTS:-global uniform_blockwise blockwise_capacity layerwise_capacity}"
NUM_PHASES="${NUM_PHASES:-2}"
EVAL_STEPS="${EVAL_STEPS:-}"
TORCHRUN="$PYTHON -m torch.distributed.run --standalone --nproc-per-node $NPROC"

if stage teacher_plan; then
  announce "teacher_plan (CPU, needs facebookresearch/DiT)"
  # A one-group stub profile makes dit_arch_to_plans.py emit the full teacher
  # width. The recorded teacher used an equivalent hand-written plan.
  write_text "$OUT/teacher_plan/stub_results.json" \
    '{"n_eff": [1.0], "group_names": ["blocks.0.attn.head_0"], "relative_delta_stack": [[0.0]]}'
  write_text "$OUT/teacher_plan/stub_grouping.json" '{"boundaries": [0, 1]}'
  run "$PYTHON" scripts/dit_arch_to_plans.py \
    --results_json "$OUT/teacher_plan/stub_results.json" \
    --grouping_json "$OUT/teacher_plan/stub_grouping.json" \
    --model dit_smicro --variants global --out "$OUT/teacher_plan/plan.json"
fi

if stage teacher; then
  announce "teacher (GPU, $NPROC processes)"
  # Global batch 512 (8 processes x 64), 388,000 steps. EDM loss in F space,
  # lognormal noise levels, non-leaky augmentation with probability 0.12,
  # dropout 0.13 and 10 percent label dropout for guidance. --teacher_checkpoint
  # is required by the CLI and not read with --kd_weight 0.
  launch "$NPROC" scripts/train_phase_students.py \
    --model_type dit_micro --arch_plan "$OUT/teacher_plan/plan.json" --variant global \
    --teacher_checkpoint none --kd_weight 0.0 --gt_weight 1.0 --diffusion edm --edm_loss_space f \
    --dataset cifar10 --data_root "$DATA_ROOT" --download --cifar_split train --num_workers 4 \
    --augment_prob 0.12 --net_dropout 0.13 --cfg_label_drop 0.1 \
    --sigma_sampling lognormal --p_mean -1.2 --p_std 1.2 \
    --num_epochs 4000 --batch_size 64 --lr 1e-3 --ema_beta 0.9993 --lr_schedule wsd \
    --warmup_steps 1000 --cooldown_frac 0.1 --optimizer adam --weight_decay 0 --dtype fp32 \
    --full_ckpt --ckpt_every 3000 --val_every 5000 --output_dir "$TEACHER_DIR"
fi

if stage references; then
  announce "references (CPU)"
  run "$PYTHON" scripts/data/extract_fid_reference_pngs.py \
    --dataset cifar10 --data_root "$DATA_ROOT" --cifar_split test --num_images 10000 --image_size 32 \
    --output_dir "$OUT/cifar_test_pngs"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py --png_dir "$OUT/cifar_test_pngs" --out "$FID_REF" --size 32
fi

if stage teacher_fid; then
  announce "teacher_fid (GPU, $NPROC processes; the ADM evaluator runs on the CPU)"
  require_input "$TEACHER_DIR/phase_0/student.pt" "trained teacher"
  require_input "$FID_REF" "CIFAR-10 test reference batch"
  launch "$NPROC" scripts/evaluate_students.py \
    --model_type dit_micro --mode composite --diffusion edm --student_dir "$TEACHER_DIR" \
    --num_samples 10000 --num_steps 18 --cfg_scale 1.25 --dtype fp32 --seed 0 --skip_fid \
    --output_dir "$OUT/teacher_samples"
  run "$PYTHON" scripts/data/pack_pngs_to_npz.py \
    --png_dir "$OUT/teacher_samples/samples" --out "$OUT/teacher_samples/samples.npz" --size 32
  run env CUDA_VISIBLE_DEVICES= "$ADM_PYTHON" "$ADM_DIR/evaluator.py" "$FID_REF" "$OUT/teacher_samples/samples.npz"
fi

if stage profile; then
  announce "profile (GPU, $NPROC processes)"
  require_input "$TEACHER_DIR/phase_0/student.pt" "trained teacher"
  # Inferred settings: the launcher of this profile is not recorded.
  launch "$NPROC" scripts/evaluate_parameters_dit_micro.py \
    --arch_cfg "$TEACHER_DIR/phase_0/arch_cfg.json" --checkpoint "$TEACHER_DIR/phase_0/student.pt" \
    --diffusion edm --grouping attention_heads --ablation_mode permutation \
    --corruption_order level_major --permutation_group_by_level true \
    --data_root "$DATA_ROOT" --cifar_split train --max_images 1000 \
    --num_timestep_levels 64 --num_bins 20 --batch_size 256 --seed 0 --output_dir "$OUT/profile"
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
  # --uniform_budget_match keeps the Uniform student within a few percent of
  # its budget at this small width, as for the recorded plans of this teacher.
  run "$PYTHON" scripts/dit_arch_to_plans.py \
    --results_json "$PROFILE_JSON" --grouping_json "$GROUPING_JSON" --model dit_smicro \
    --variants global,uniform_blockwise,blockwise_capacity,layerwise_capacity \
    --uniform_budget_match --blockwise_budget_g_max 50 --out "$PLANS_JSON"
fi

if stage train; then
  require_input "$PLANS_JSON" "plans"
  for variant in $VARIANTS; do
    announce "train $variant (GPU, $TRAIN_NPROC processes)"
    # Recorded: 3,200 epochs at 2 processes x 256 (310,400 steps), cooldown over
    # the last 20 percent, EDM loss in F space, FP32. The remaining values follow
    # the other DiT students and are not recorded for this run.
    launch "$TRAIN_NPROC" scripts/train_phase_students.py \
      --model_type dit_micro --diffusion edm --edm_loss_space f \
      --teacher_checkpoint "$TEACHER_DIR/phase_0/student.pt" \
      --teacher_arch_cfg "$TEACHER_DIR/phase_0/arch_cfg.json" \
      --arch_plan "$PLANS_JSON" --variant "$variant" \
      --dataset cifar10 --data_root "$DATA_ROOT" --cifar_split train --num_workers 4 \
      --num_epochs 3200 --batch_size 256 --lr 2e-4 --gt_weight 0.25 --kd_weight 1.0 \
      --ema_beta 0.999 --optimizer adam --weight_decay 0 --lr_schedule wsd --warmup_steps 1000 \
      --cooldown_frac 0.2 --cfg_label_drop 0.1 --dtype fp32 --full_ckpt --ckpt_every 3000 \
      --output_dir "$OUT/students/$variant"
  done
fi

if stage evaluate; then
  require_input "$FID_REF" "CIFAR-10 test reference batch"
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
      --student_dir "$OUT/students/$variant" --model_type dit_micro --diffusion edm \
      --num_samples 10000 --num_steps 18 --cfg_scale 1.25 --dtype fp32 \
      --ref_npz "$FID_REF" --pack_size 32 --adm_python "$ADM_PYTHON" --adm_dir "$ADM_DIR" \
      --num_phases "$phases" --out_json "$OUT/students/$variant/curve.json" \
      --torchrun "$TORCHRUN" --repo "$PACE_ROOT" ${steps_args[@]+"${steps_args[@]}"}
  done
fi
