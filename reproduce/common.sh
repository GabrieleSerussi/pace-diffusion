# Shared settings and helpers for the scripts in reproduce/.
#
# Every experiment script sources this file. It is not meant to be run on its
# own. All paths are relative to the repository root unless you set them to
# absolute paths.
#
# Environment variables read here (all optional):
#   PYTHON         Python interpreter with pace-diffusion installed (default: python)
#   DRY_RUN=1      Print every command without running it
#   STAGES         Space-separated stages to run, for example "group allocate".
#                  Empty (the default) runs every stage of the script in order.
#   DATA_ROOT      Root of the datasets (default: data)
#   MODEL_CACHE    Download cache for teacher checkpoints (default: checkpoints)
#   OUTPUT_ROOT    Root of everything the scripts write (default: outputs)
#   REFERENCE_DIR  FID reference batches and the Inception graph (default: references)
#   ADM_DIR        guided-diffusion evaluations/ directory (default: external/guided-diffusion/evaluations)
#   ADM_PYTHON     Python of the separate TensorFlow environment of the ADM evaluator
#                  (default: python)
#   EDM_REPO       NVlabs/edm checkout (default: ../edm next to this repository)
#   DIT_REPO       facebookresearch/DiT checkout (default: ../DiT next to this repository)

set -euo pipefail

PACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACE_ROOT"

PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
STAGES="${STAGES:-}"
DATA_ROOT="${DATA_ROOT:-data}"
MODEL_CACHE="${MODEL_CACHE:-checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
REFERENCE_DIR="${REFERENCE_DIR:-references}"
ADM_DIR="${ADM_DIR:-external/guided-diffusion/evaluations}"
ADM_PYTHON="${ADM_PYTHON:-python}"
ADM_EVALUATOR="${ADM_EVALUATOR:-$ADM_DIR/evaluator.py}"
ADM_DETECTOR="${ADM_DETECTOR:-$REFERENCE_DIR/classify_image_graph_def.pb}"

export EDM_REPO="${EDM_REPO:-$PACE_ROOT/../edm}"
export DIT_REPO="${DIT_REPO:-$PACE_ROOT/../DiT}"
export PACE_MODEL_CACHE="${PACE_MODEL_CACHE:-$MODEL_CACHE}"
export PYTHONUNBUFFERED=1

# The four students of Tables 1 and 2, in the order the paper lists them.
PAPER_VARIANTS="global uniform_blockwise combined_blockwise combined_layerwise"

# RESUME=1 continues interrupted U-Net training runs from their newest snapshot.
RESUME="${RESUME:-0}"

run() {
  # Print a command, then run it unless DRY_RUN=1.
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

launch() {
  # launch NPROC SCRIPT ARGS...: one process with plain Python, several with torchrun.
  local nproc="$1"
  shift
  if [[ "$nproc" == "1" ]]; then
    run "$PYTHON" "$@"
  else
    run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$nproc" "$@"
  fi
}

stage() {
  # True when stage "$1" is selected by STAGES (every stage when STAGES is empty).
  [[ -z "$STAGES" || " $STAGES " == *" $1 "* ]]
}

resume_flag() {
  # Prints --resume when RESUME=1. Use it unquoted: $(resume_flag).
  if [[ "$RESUME" == "1" ]]; then
    printf '%s' "--resume"
  fi
}

announce() {
  printf '\n== %s\n' "$*"
}

require_input() {
  # Stop with a clear message when an input the user provides is missing.
  # The check is skipped in DRY_RUN mode, so the command list can be read first.
  local path="$1"
  local description="$2"
  if [[ "$DRY_RUN" != "1" && ! -e "$path" ]]; then
    printf 'error: %s is missing: %s\n' "$description" "$path" >&2
    exit 1
  fi
}

make_dirs() {
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$@"
  fi
}

write_text() {
  # write_text PATH TEXT: write a small input file (printed only in DRY_RUN mode).
  printf '+ write %q\n' "$1"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$(dirname "$1")"
    printf '%s\n' "$2" > "$1"
  fi
}
