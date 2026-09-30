#!/usr/bin/env bash
# Figure 1: measured FFHQ-64 timestep correlation, the three discovered phases,
# and teacher states at the phase boundaries.
#
# The correlation matrix and the phase boundaries come from the released FFHQ-64
# profile and grouping (artifacts/). The paper figure is the seed 11 candidate
# written by the render stage.
#
# Stages, in order:
#   trajectory  GPU  sample seeds 0 to 7 with the teacher (40 Heun steps) and record the solver states
#   template    CPU  predict the teacher's clean images at the two boundaries and write the
#                    figure template (figure1_empirical_clean_labels.json)
#   candidates  CPU  sample seeds 8 to 17 and record states and clean-image predictions
#   render      CPU  render the ten candidates; the paper uses seed 11
#
# Inputs: NVlabs/edm at EDM_REPO (default ../edm). The FFHQ-64 teacher is
# downloaded into MODEL_CACHE.
#
# Environment variables (reproduce/common.sh lists the shared ones):
#   DEVICE   device of the trajectory stage (default: cuda, as in the recorded
#            run). The random latents depend on the device and the PyTorch build.
#            The paper figure uses seeds 8 to 17, which the candidates stage
#            samples on the CPU.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DEVICE="${DEVICE:-cuda}"
FIGURE_DIR="$OUTPUT_ROOT/figures/figure1"
SAMPLING_DIR="$OUTPUT_ROOT/figure1_sampling"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/.matplotlib}"

if stage trajectory; then
  announce "trajectory ($DEVICE)"
  run "$PYTHON" scripts/paper/figures/sample_figure1_trajectory.py \
    --device "$DEVICE" --cache-dir "$MODEL_CACHE" --output "$SAMPLING_DIR"
fi

if stage template; then
  announce "template (CPU)"
  # 1. Raw solver states of seed 6 at the frames nearest to the two boundaries.
  run "$PYTHON" scripts/paper/figures/plot_empirical_figure1.py --raw-states \
    --trajectory "$SAMPLING_DIR/trajectories.npz" --output "$FIGURE_DIR/figure1_empirical"
  # 2. Teacher clean-image predictions at those two states.
  run "$PYTHON" scripts/paper/figures/predict_figure1_boundaries.py --device cpu --cache-dir "$MODEL_CACHE" \
    --states "$FIGURE_DIR/figure1_empirical_states.npz" --data "$FIGURE_DIR/figure1_empirical.json" \
    --output "$SAMPLING_DIR/boundary_predictions.npz"
  # 3. The template with minimal labels that the candidate stages reuse.
  run "$PYTHON" scripts/paper/figures/plot_empirical_figure1.py --minimal-labels \
    --trajectory "$SAMPLING_DIR/trajectories.npz" --predictions "$SAMPLING_DIR/boundary_predictions.npz" \
    --output "$FIGURE_DIR/figure1_empirical_clean_labels"
fi

if stage candidates; then
  announce "candidates (CPU)"
  run "$PYTHON" scripts/paper/figures/sample_figure1_candidates.py --seeds 8-17 --device cpu --threads 8 \
    --cache-dir "$MODEL_CACHE" --data "$FIGURE_DIR/figure1_empirical_clean_labels.json" \
    --output "$SAMPLING_DIR/seeds8_17"
fi

if stage render; then
  announce "render (CPU)"
  run "$PYTHON" scripts/paper/figures/render_figure1_candidates.py \
    --trajectory "$SAMPLING_DIR/seeds8_17/trajectories.npz" \
    --data "$FIGURE_DIR/figure1_empirical_clean_labels.json" \
    --output-dir "$FIGURE_DIR/seed_candidates_8_17"
  printf 'Paper figure: %s\n' "$FIGURE_DIR/seed_candidates_8_17/figure1_empirical_clean_labels_seed11.pdf"
fi
