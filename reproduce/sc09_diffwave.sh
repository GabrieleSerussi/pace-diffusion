#!/usr/bin/env bash
# Speech Commands SC09, unconditional DiffWave teacher of albertfgu/diffwave-sashimi.
#
# Paper results: Appendix B and Figure 4 (audio phases and capacity shares).
#
# Stages, in order (select a subset with STAGES="figure"):
#   data       CPU  download, extract and validate SC09 (pinned archive)
#   teacher    CPU  download the pinned 1M-step checkpoint and clone the upstream repository
#   verify     GPU  1 GPU: run the upstream sampler and check that the vendored teacher matches it
#   smoke      GPU  16-filter PFI run that checks the setup
#   stability  GPU  three 256-filter PFI runs with different seeds, then their rank correlation
#   profile    GPU  full per-filter PFI profile (37,377 filters); it also writes the
#                   residual-only view, the phase groupings, the allocations and the decision report
#   figure     CPU  Figure 4 from the new profile, and Figure 4 from the released metrics snapshot
#
# Inputs you provide: FFmpeg shared libraries for torchaudio and torchcodec (the
# recorded runs used FFmpeg 6.1.1), git and curl. The dataset and checkpoint are
# downloaded and verified against pinned hashes.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   OUT         output directory (default: outputs/sc09_diffwave)
#   SC09_ROOT   extracted dataset (default: $DATA_ROOT/sc09_v0.02)
#   TEACHER_DIR checkpoint directory (default: $MODEL_CACHE/diffwave_sc09_1m_legacy)
#   UPSTREAM    diffwave-sashimi clone for the parity check (default: external/diffwave-sashimi)
#   REFENV      virtual environment of the upstream sampler (default: external/diffwave-reference-env)
#   NPROC       processes for the PFI stages (default: 8, the recorded runs; any count works)
#
# The figure stage alone needs only the CPU: with STAGES=figure and no profile it
# still renders Figure 4 from artifacts/audio/audio_appendix_metrics.json.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

abspath() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PACE_ROOT" "$1" ;;
  esac
}

OUT="${OUT:-$OUTPUT_ROOT/sc09_diffwave}"
SC09_ROOT="${SC09_ROOT:-$DATA_ROOT/sc09_v0.02}"
TEACHER_DIR="${TEACHER_DIR:-$MODEL_CACHE/diffwave_sc09_1m_legacy}"
UPSTREAM="$(abspath "${UPSTREAM:-external/diffwave-sashimi}")"
REFENV="$(abspath "${REFENV:-external/diffwave-reference-env}")"
NPROC="${NPROC:-8}"
UPSTREAM_COMMIT="9bd78f8c894cad0952a5692450f2145e24466b29"
CHECKPOINT_PATH="exp/wnet_h256_d36_T200_betaT0.02_uncond/checkpoint/1000000.pkl"
CHECKPOINT_URL="https://media.githubusercontent.com/media/albertfgu/diffwave-sashimi/$UPSTREAM_COMMIT/$CHECKPOINT_PATH"
CHECKPOINT_SHA256="f34b9bcca4970572775fff15dbccda09f7ec8d56befe113620e1ad25eee416e1"
CHECKPOINT_SIZE=96451975
REPORT="$OUT/teacher_reproduction/reproduction_report.json"
PROFILE_DIR="$OUT/profile_pool100_n4_t200_bins20_seed0"

profile_args=(
  --data-root "$SC09_ROOT" --model-cache-dir "$TEACHER_DIR" --teacher-verification-report "$REPORT"
)

if stage data; then
  announce "data (CPU, downloads 894 MB)"
  run "$PYTHON" scripts/data/prepare_sc09_dataset.py --output-dir "$SC09_ROOT"
fi

if stage teacher; then
  announce "teacher (CPU, downloads 96 MB)"
  make_dirs "$TEACHER_DIR" "$(dirname "$UPSTREAM")"
  if [[ ! -f "$TEACHER_DIR/1000000.pkl" ]]; then
    run curl -fL "$CHECKPOINT_URL" -o "$TEACHER_DIR/.1000000.pkl.part"
    run mv "$TEACHER_DIR/.1000000.pkl.part" "$TEACHER_DIR/1000000.pkl"
  fi
  run "$PYTHON" - "$TEACHER_DIR/1000000.pkl" "$CHECKPOINT_SIZE" "$CHECKPOINT_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

path, size, sha256 = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if path.stat().st_size != size or digest != sha256:
    raise SystemExit(f"checkpoint mismatch: {path.stat().st_size} bytes, sha256 {digest}")
print("checkpoint size and SHA-256 verified")
PY
  if [[ ! -d "$UPSTREAM/.git" ]]; then
    run env GIT_LFS_SKIP_SMUDGE=1 git clone --branch checkpoints --single-branch \
      https://github.com/albertfgu/diffwave-sashimi.git "$UPSTREAM"
  fi
  run git -C "$UPSTREAM" checkout --detach "$UPSTREAM_COMMIT"
  run cp "$TEACHER_DIR/1000000.pkl" "$UPSTREAM/$CHECKPOINT_PATH"
fi

if stage verify; then
  announce "verify (1 GPU)"
  require_input "$UPSTREAM/$CHECKPOINT_PATH" "upstream checkpoint (teacher stage)"
  # The upstream sampler runs untouched in its own environment, which inherits
  # torch, numpy, scipy and tqdm from the main environment.
  run "$PYTHON" -m venv --system-site-packages "$REFENV"
  run "$REFENV/bin/python" -m pip install -r reproduce/requirements-diffwave-reference.txt
  printf '+ cd %q\n' "$UPSTREAM"
  (
    if [[ "$DRY_RUN" != "1" ]]; then
      cd "$UPSTREAM"
    fi
    run env CUDA_VISIBLE_DEVICES=0 "$REFENV/bin/python" generate.py experiment=sc09 model=wavenet \
      generate.ckpt_iter=1000000 generate.n_samples=1 generate.batch_size=1
  )
  make_dirs "$OUT/teacher_reproduction"
  run cp "$UPSTREAM/exp/wnet_h256_d36_T200_betaT0.02_uncond/waveforms/1000000/1000k_0.wav" \
    "$OUT/teacher_reproduction/upstream_generate_original.wav"
  # The report binds the SHA-256 of pace/teacher_models.py and
  # pace/vendor/diffwave_legacy.py; the profile stages refuse a report that was
  # written for other versions of these files.
  run "$PYTHON" scripts/paper/verify_diffwave_teacher.py \
    --upstream-repo "$UPSTREAM" --checkpoint "$TEACHER_DIR/1000000.pkl" \
    --reference-python "$REFENV/bin/python" --output "$REPORT" \
    --audio-output "$OUT/teacher_reproduction/verified_sample.wav" \
    --reference-audio "$OUT/teacher_reproduction/upstream_generate_original.wav" \
    --device cuda:0
fi

if stage smoke; then
  announce "smoke (GPU, $NPROC processes)"
  require_input "$REPORT" "teacher verification report (verify stage)"
  launch "$NPROC" scripts/paper/evaluate_parameters_diffwave.py \
    --config reproduce/configs/diffwave_sc09_analysis_per_filter_smoke.json "${profile_args[@]}" \
    --output-dir "$OUT/smoke_seed0"
fi

if stage stability; then
  require_input "$REPORT" "teacher verification report (verify stage)"
  for seed in 0 1 2; do
    announce "stability seed $seed (GPU, $NPROC processes)"
    launch "$NPROC" scripts/paper/evaluate_parameters_diffwave.py \
      --config reproduce/configs/diffwave_sc09_analysis_per_filter_stability.json "${profile_args[@]}" \
      --output-dir "$OUT/stability/seed$seed" --seed "$seed" --pfi-seed "$seed"
  done
  run "$PYTHON" scripts/paper/compare_diffwave_filter_stability.py \
    --results "$OUT/stability/seed0/results.json" \
    --results "$OUT/stability/seed1/results.json" \
    --results "$OUT/stability/seed2/results.json" \
    --output "$OUT/stability/stability_report.json" \
    --plot-output "$OUT/stability/stability_spearman.png"
fi

if stage profile; then
  announce "profile (GPU, $NPROC processes)"
  require_input "$REPORT" "teacher verification report (verify stage)"
  # The research configuration sets --run-postprocessing: after the profile,
  # rank 0 writes residual_only/ with four groupings, their allocations and
  # decision_report.json. The published phases [0, 13, 17, 20] are the
  # grouping_pearson_k3_min2 result: the Section 3.3 objective without the
  # separation term (lambda_sep = 0), K = 3 and phases of at least two bins.
  # The published capacity shares use the mean over bins, not the sum.
  launch "$NPROC" scripts/paper/evaluate_parameters_diffwave.py \
    --config reproduce/configs/diffwave_sc09_analysis_per_filter_research.json "${profile_args[@]}" \
    --output-dir "$PROFILE_DIR"
fi

if stage figure; then
  announce "figure (CPU)"
  if [[ -e "$PROFILE_DIR/residual_only/results.json" || "$DRY_RUN" == "1" ]]; then
    run "$PYTHON" scripts/paper/plot_audio_appendix.py --refresh-from-profile \
      --residual-dir "$PROFILE_DIR/residual_only" \
      --stability-report "$OUT/stability/stability_report.json" \
      --metrics "$OUT/audio_appendix_metrics.json" --output-dir "$OUT/figure"
  fi
  # Figure 4 from the released metrics snapshot (no profile needed).
  run "$PYTHON" scripts/paper/plot_audio_appendix.py \
    --metrics artifacts/audio/audio_appendix_metrics.json --output-dir "$OUT/figure_released"
fi
