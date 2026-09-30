#!/usr/bin/env python3
"""Render ten FFHQ Figure 1 candidates from captured states and predictions.

The trajectory NPZ must contain states[41,10,3,64,64],
denoised[40,10,3,64,64], sigmas[41], and seeds[10] for seeds 8 through 17.
Its sibling manifest.json identifies the teacher and sampling configuration.
This script does not load a teacher or perform model inference.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.paper.figures.plot_empirical_figure1 import FIGURE_DIR, plot
from scripts.paper.figures.sample_figure1_trajectory import boundary_step_indices, display_rgb


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def validate_inputs(data, manifest, states, denoised, sigmas, seeds):
    expected_seeds = np.arange(8, 18)
    if seeds.shape != (10,) or not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("Expected ten integer seed identifiers")
    if not np.array_equal(np.sort(seeds), expected_seeds):
        raise ValueError("Expected each of seeds 8 through 17 exactly once")
    if states.shape != (41, 10, 3, 64, 64):
        raise ValueError(f"Unexpected solver-state shape: {states.shape}")
    if denoised.shape != (40, 10, 3, 64, 64):
        raise ValueError(f"Unexpected captured-prediction shape: {denoised.shape}")
    if sigmas.shape != (41,) or not np.isfinite(sigmas).all():
        raise ValueError("Expected 41 finite sigma values")
    if not (sigmas[-1] == 0 and np.all(sigmas[:-1] > 0)
            and np.all(np.diff(sigmas) < 0)):
        raise ValueError("Sigma schedule must strictly decrease to zero")
    if not np.isfinite(states).all() or not np.isfinite(denoised).all():
        raise ValueError("States and predictions must be finite")
    if not (np.issubdtype(states.dtype, np.floating)
            and np.issubdtype(denoised.dtype, np.floating)):
        raise ValueError("Expected floating-point states and predictions")
    if data.get("dataset") != "FFHQ-64":
        raise ValueError("The reference figure must describe FFHQ-64")
    if data.get("boundaries") != [0, 3, 16, 20]:
        raise ValueError("Expected the original three-phase partition")
    if data.get("figure_style") != {"minimal_labels": True}:
        raise ValueError("Expected the canonical clean-label figure style")
    correlation = np.asarray(data["correlation"])
    if correlation.shape != (20, 20) or not np.isfinite(correlation).all():
        raise ValueError("Expected the complete measured 20-by-20 matrix")
    if manifest.get("teacher_sha256") != data["teacher_sha256"]:
        raise ValueError("Trajectory teacher differs from the profiling teacher")
    if "seeds" in manifest and manifest["seeds"] != seeds.tolist():
        raise ValueError("Manifest seed order differs from the trajectory archive")
    for key, expected in (("steps", 40), ("num_steps", 40), ("nfe", 79),
                          ("rho", 7), ("churn", 0), ("s_churn", 0),
                          ("clip_denoised", False)):
        if key in manifest and manifest[key] != expected:
            raise ValueError(f"Unexpected sampling configuration: {key}={manifest[key]}")
    if "sampler" in manifest and "heun" not in str(manifest["sampler"]).lower():
        raise ValueError("The figure labels require the EDM Heun sampler")
    reference_steps = np.arange(40, dtype=np.float64)
    expected_sigmas = (80 ** (1 / 7) + reference_steps / 39
                      * (.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
    np.testing.assert_allclose(sigmas[:-1], expected_sigmas, rtol=1e-6, atol=1e-9)
    steps = np.asarray(boundary_step_indices(sigmas, data["phase_boundary_sigmas"]))
    if steps.tolist() != [0, 10, 35, 40]:
        raise ValueError(f"Unexpected boundary-adjacent frame indices: {steps.tolist()}")
    return steps


def write_caption(output, seed, steps, sigmas, boundary_sigmas):
    try:
        pdf_path = output.with_suffix(".pdf").resolve().relative_to(FIGURE_DIR.parent)
    except ValueError:
        pdf_path = output.with_suffix(".pdf").name
    caption = rf"""% Requires graphicx and amsmath.
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{{pdf_path}}}
  \caption{{\textbf{{Measured diffusion phases and denoising predictions.}}
  \textbf{{Top:}} Initial noise, two clean-image predictions
  $\hat{{x}}_0=D_\theta(x_\sigma,\sigma)$, and the final sample from one
  FFHQ-64 EDM teacher trajectory (seed {seed}; deterministic 40-step Heun).
  Intermediate predictions are captured directly from the teacher's main-step
  forward calls at the states nearest the two phase boundaries.
  The boundaries are at $\sigma\approx{boundary_sigmas[0]:.4g}$ and
  ${boundary_sigmas[1]:.4g}$; the nearest states in log noise are steps
  {steps[1]} and {steps[2]}, at $\sigma\approx{sigmas[1]:.4g}$ and
  ${sigmas[2]:.4g}$. Predictions and the final sample are mapped directly to RGB;
  only initial noise is scaled by $1/\sqrt{{1+\sigma^2}}$ for display.
  Image spacing is for readability, not proportional to denoising time.
  \textbf{{Bottom:}} The full $20\times20$ matrix of Pearson correlations between
  relative-sensitivity fingerprints of the same teacher, measured by
  exact-noise-level permutation importance over 32,387 activation groups
  using 100 calibration images at 256 noise levels aggregated into 20 bins.
  The phase strip shows the original three-phase partition: 3, 13, and 4 bins.
  Matrix cells are rectangular to fit the layout.}}
  \label{{fig:empirical-phases-clean-labels-seed{seed}}}
\end{{figure}}
"""
    output.with_suffix(".tex").write_text(caption)


def contact_sheet(rows, steps):
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 14)
    bold = ImageFont.truetype(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")), 22)
    tile, gap, margin, label_width, header, row_height = 192, 12, 20, 94, 106, 220
    width = margin * 2 + label_width + 4 * tile + 3 * gap
    sheet = Image.new("RGB", (width, header + len(rows) * row_height + 12), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), f"FFHQ candidates: seeds {rows[0][0]}–{rows[-1][0]}",
              fill="#25364A", font=bold)
    labels = [("Initial noise", f"step {steps[0]}"),
              ("Near P1 → P2", f"denoised estimate · step {steps[1]}"),
              ("Near P2 → P3", f"denoised estimate · step {steps[2]}"),
              ("Final sample", f"step {steps[3]}")]
    for column, (title, detail) in enumerate(labels):
        center = margin + label_width + column * (tile + gap) + tile / 2
        draw.text((center, 52), title, anchor="mt", fill="#25364A", font=font)
        draw.text((center, 77), detail, anchor="mt", fill="#536277", font=small)
    for row_index, (seed, frames) in enumerate(rows):
        top = header + row_index * row_height
        draw.text((margin, top + tile / 2 - 14), f"Seed {seed}",
                  fill="#25364A", font=font)
        for column, frame in enumerate(frames):
            x = margin + label_width + column * (tile + gap)
            image = Image.fromarray(frame).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(image, (x, top))
            draw.rectangle((x, top, x + tile - 1, top + tile - 1), outline="#BCC5CD")
        if row_index < len(rows) - 1:
            draw.line((margin, top + tile + 14, width - margin, top + tile + 14),
                      fill="#E1E7ED")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=FIGURE_DIR / "figure1_empirical_clean_labels.json")
    parser.add_argument("--output-dir", type=Path,
                        default=FIGURE_DIR / "seed_candidates_8_17")
    args = parser.parse_args()
    template = json.loads(args.data.read_text())
    manifest_path = args.trajectory.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        states = trajectory["states"]
        denoised = trajectory["denoised"]
        sigmas = trajectory["sigmas"]
        seeds = trajectory["seeds"]
    steps = validate_inputs(template, manifest, states, denoised, sigmas, seeds)
    trajectory_hash = file_sha256(args.trajectory)
    manifest_hash = file_sha256(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, candidates = [], []
    for seed in range(8, 18):
        position = int(np.flatnonzero(seeds == seed)[0])
        selected_states = states[steps, position]
        selected_sigmas = sigmas[steps]
        predictions = denoised[steps[1:3], position]
        output = args.output_dir / f"figure1_empirical_clean_labels_seed{seed}"
        states_file = output.with_name(output.name + "_states.npz")
        prediction_metadata = dict(
            teacher_sha256=template["teacher_sha256"], seed=seed,
            source_trajectory=source_path(args.trajectory),
            source_trajectory_sha256=trajectory_hash,
            source_manifest=source_path(manifest_path),
            source_manifest_sha256=manifest_hash,
            sampler_metadata=copy.deepcopy(manifest),
            device=manifest.get("device", "unspecified in source manifest"),
            rng=manifest.get("rng", "unspecified in source manifest"),
            dtype=f"{denoised.dtype} captured predictions; {states.dtype} solver inputs",
            step_indices=steps[1:3].tolist(), sigmas=selected_sigmas[1:3].tolist(),
            kind="D(x_sigma, sigma): teacher prediction of clean x0; not a noisy solver state",
            computation="Actual teacher main-step forward outputs captured during this sampling run; "
                        "correction calls excluded; no re-evaluation",
            display_rule="uint8(clip(128 + 127.5*prediction, 0, 255)); no sigma scaling",
            mae_to_final=[float(np.abs(pred - selected_states[-1]).mean())
                          for pred in predictions])
        np.savez_compressed(states_file, states=selected_states, sigmas=selected_sigmas,
                            step_indices=steps, seed=seed, predictions=predictions,
                            prediction_metadata=json.dumps(prediction_metadata))
        data = copy.deepcopy(template)
        data["sampling"] = dict(
            seed=seed, step_indices=steps.tolist(), sigmas=selected_sigmas.tolist(),
            states_file=states_file.name, intermediate_display="denoised_estimate",
            prediction_metadata=prediction_metadata,
            display_rule="Intermediate estimates: uint8(clip(128+127.5*D(x_sigma,sigma),0,255)); "
                         "endpoints: uint8(clip(128+127.5*x_sigma/sqrt(1+sigma^2),0,255))",
            state_kind="initial noisy state; two captured teacher clean-image predictions; final sample",
            sampler="deterministic EDM Heun, 40 steps, rho=7, 79 NFE, no denoised clipping",
            sampler_metadata=copy.deepcopy(manifest), candidate_seeds=list(range(8, 18)),
            frame_selection="nearest solver state in log sigma to each routing boundary",
            phase_boundary_sigmas=data["phase_boundary_sigmas"],
            selection_reason="One of ten consecutive new seeds requested for visual comparison; "
                             "all seeds 8 through 17 are included")
        frames = display_rgb(selected_states, selected_sigmas)
        frames[1:3] = display_rgb(predictions, np.zeros(2))
        plot_data = copy.deepcopy(data)
        plot_data["sampling"]["frames_rgb"] = frames
        plot(plot_data, output)
        output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
        write_caption(output, seed, steps, selected_sigmas, data["phase_boundary_sigmas"])
        rows.append((seed, frames))
        candidates.append(dict(seed=seed, figure=output.name, states_file=states_file.name))
        print(f"Rendered seed {seed}: {output}", flush=True)
    sheets = [contact_sheet(rows[:5], steps), contact_sheet(rows[5:], steps)]
    sheet_names = ["contact_sheet_seeds_8_12.png", "contact_sheet_seeds_13_17.png"]
    for sheet, name in zip(sheets, sheet_names):
        sheet.save(args.output_dir / name)
    all_sheet = Image.new("RGB", (sheets[0].width * 2 + 20, sheets[0].height), "white")
    all_sheet.paste(sheets[0], (0, 0))
    all_sheet.paste(sheets[1], (sheets[0].width + 20, 0))
    all_sheet.save(args.output_dir / "contact_sheet_all_10.png")
    summary = dict(teacher_sha256=template["teacher_sha256"],
                   source_trajectory=source_path(args.trajectory),
                   source_trajectory_sha256=trajectory_hash,
                   source_manifest_sha256=manifest_hash, seeds=list(range(8, 18)),
                   step_indices=steps.tolist(), phase_boundaries=template["boundaries"],
                   candidates=candidates,
                   contact_sheets=[*sheet_names, "contact_sheet_all_10.png"])
    (args.output_dir / "candidates.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote ten candidates and three comparison sheets to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
