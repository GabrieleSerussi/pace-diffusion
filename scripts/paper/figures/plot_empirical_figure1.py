#!/usr/bin/env python3
"""Build Figure 1 from the completed FFHQ PFI profile (no synthetic data).

Run from the repository root. The correlation matrix and phase boundaries come
from the released FFHQ-64 profile and grouping (``artifacts/profiles/`` and
``artifacts/groupings/``); the sampler states come from
``scripts/paper/figures/sample_figure1_trajectory.py``. The compact exported JSON can
subsequently be plotted without the profile:
--data outputs/figures/figure1/figure1_empirical.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pace.jsonio import load_json  # noqa: E402
from pace.slim_profile import slim_profile_record  # noqa: E402

PROFILE = ROOT / "artifacts/profiles/ffhq64_ddpmpp_pfi.json.gz"
GROUPING = ROOT / "artifacts/groupings/ffhq64_ddpmpp_pfi.json"
FIGURE_DIR = ROOT / "outputs/figures/figure1"
SAMPLING_DIR = ROOT / "outputs/figure1_sampling"


def extract():
    profile = load_json(PROFILE)
    grouping = load_json(GROUPING)
    # The released profile is a slim copy; its record carries the SHA-256 of
    # the full profile that the grouping was computed from.
    slim = slim_profile_record(profile)
    digest = (slim["source_results_sha256"] if slim is not None
              else hashlib.sha256(PROFILE.read_bytes()).hexdigest())
    assert digest == grouping["source_profile"]["results_sha256"]
    assert profile["ablation_protocol"]["protocol_id"] == "batch_local_exact_sigma_pfi_v1"
    assert grouping["array_key"] == "relative_delta_stack"
    features = np.asarray(profile["relative_delta_stack"])
    corr = np.corrcoef(features.T)
    delta = np.asarray(profile["delta_stack"])
    weights = delta / (delta.sum(axis=0, keepdims=True) + 1e-12)
    neff = 1 / ((weights ** 2).sum(axis=0) + 1e-12)
    np.testing.assert_allclose(neff, profile["n_eff"], rtol=1e-9)
    assert np.isfinite(corr).all() and features.shape == (32387, 20)
    bounds = grouping["boundaries"]
    schedule = np.asarray(profile["sigma_values"])
    # Match RoutedEDMStudent: floor(index * num_bins / num_levels),
    # with nearest-log-sigma assignment between profiled noise levels.
    cuts = np.ceil(np.asarray(bounds[1:-1]) * len(schedule) / 20).astype(int)
    boundary_sigmas = np.sqrt(schedule[cuts-1] * schedule[cuts])
    cost = sum(((b-a)**2 - corr[a:b, a:b].sum()) / (2 * (b-a))
               for a, b in zip(bounds[:-1], bounds[1:]))
    np.testing.assert_allclose(cost, grouping["total_cost"], atol=1e-10)
    return dict(
        dataset="FFHQ-64", protocol=profile["ablation_protocol"]["protocol_id"],
        source_profile=str(PROFILE.relative_to(ROOT)), profile_sha256=digest,
        source_grouping=str(GROUPING.relative_to(ROOT)),
        teacher_sha256=profile["teacher"]["checkpoint_sha256"],
        monitor_images=100, exact_sigma_levels=256, actual_pfi_batch_size=100,
        filter_count=features.shape[0], sigma_bin_labels=profile["sigma_bin_labels"],
        boundaries=bounds, phase_boundary_sigmas=boundary_sigmas.tolist(),
        correlation=corr.tolist(), n_eff=neff.tolist(),
        grouping_cost=cost, grouping_cost_type=grouping["builtin_cost"],
        note="Existing fixed-K=3 partition; not a new automatic K selection. "
             "Correlation uses relative_delta_stack, not stored C_noise_levels. "
             "n_eff is inverse-Simpson count of positive PFI loss-change shares."
    )


def plot(data, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = ["#537FB3", "#38868B", "#B87842"]
    minimal_labels = data.get("figure_style", {}).get("minimal_labels", False)
    fills = ["#DCE9F8", "#DFEFE7", "#F8E8D9"]
    fig = plt.figure(figsize=(8.2, 4.0))
    left, width = .14, .71
    ax = fig.add_axes([left, .195, width, .315])
    bounds = data["boundaries"]
    assert len(bounds) == 4, "Figure requires exactly three phases"
    matrix = np.asarray(data["correlation"])
    assert matrix.shape == (20, 20)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   origin="upper", interpolation="nearest", extent=[0, 20, 20, 0])
    ax.set(xticks=[], yticks=[] if minimal_labels else [.5, 5.5, 10.5, 15.5, 19.5],
           yticklabels=[] if minimal_labels else ["1", "6", "11", "16", "20"],
           ylabel="Noise bin")
    ax.tick_params(length=2, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#BCC5CD")
        spine.set_linewidth(.5)
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        ax.add_patch(Rectangle((a, a), b-a, b-a, fill=False,
                               edgecolor=colors[i], linewidth=1.4))
    cax = fig.add_axes([.879, .22, .012, .245])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_edgecolor("#9AABBC")
    fig.text(.879, .485, "Correlation" if minimal_labels else "Pearson\ncorrelation", fontsize=7)
    fig.text(left, .535, "Measured timestep correlation", fontsize=8, weight="bold")
    sampling = data["sampling"]
    fig.text(left+width, .535, f"FFHQ-64 · seed {sampling['seed']} · 40-step Heun",
             fontsize=7, color="#536277", ha="right")
    strip = fig.add_axes([left, .118, width, .06])
    strip.set(xlim=(0, 20), ylim=(0, 1))
    strip.axis("off")
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        strip.add_patch(Rectangle((a, 0), b-a, 1, facecolor=fills[i], edgecolor="none"))
        strip.text((a+b)/2, .5, f"Phase {i+1}", ha="center", va="center",
                   fontsize=8, weight="bold", color=colors[i])
    for boundary in bounds[1:-1]:
        ax.axvline(boundary, color="#91A7BC", linestyle=(0, (3, 3)), lw=.7)
        ax.annotate("", xy=(left+width*boundary/20, .195),
                    xytext=(left+width*boundary/20, .118), xycoords=fig.transFigure,
                    arrowprops=dict(arrowstyle="-", color="#91A7BC", lw=.7,
                                    linestyle=(0, (3, 3))))
    fig.text(left+width/2, .064, "Noise bin · high noise → low noise", ha="center",
             color="#536277", fontsize=7)
    frames = sampling.pop("frames_rgb")
    assert len(frames) == 4, "Figure requires exactly four sampling frames"
    frame_width = .145
    starts = np.linspace(left, left+width-frame_width, 4)
    for i, (start, rgb, sigma, step) in enumerate(zip(
            starts, frames, sampling["sigmas"], sampling["step_indices"])):
        frame_ax = fig.add_axes([start, .645, frame_width, frame_width*8.2/4.0])
        frame_ax.imshow(rgb, interpolation="nearest")
        frame_ax.set(xticks=[], yticks=[])
        for spine in frame_ax.spines.values():
            spine.set_color("#BCC5CD")
            spine.set_linewidth(.6)
        center = start+frame_width/2
        label = ("Initial noise" if i == 0 else "Final sample" if i == 3 else
                 rf"Near $P_{i}\!\rightarrow\!P_{i+1}$")
        fig.text(center, .609, label, ha="center", fontsize=8)
        detail = (f"Denoised estimate · step {step}" if i in (1, 2)
                  and sampling.get("intermediate_display") == "denoised_estimate"
                  else rf"Step {step} · $\sigma={sigma:.3g}$")
        fig.text(center, .572, detail, ha="center",
                 fontsize=7, color="#536277")
        if i < 3:
            ax.annotate("", xy=(starts[i+1]-.005, .794),
                        xytext=(start+frame_width+.005, .794), xycoords=fig.transFigure,
                        arrowprops=dict(arrowstyle="->", color="#536277", lw=.9))
    ax.annotate("", xy=(left+width, .967), xytext=(left, .967),
                xycoords=fig.transFigure,
                arrowprops=dict(arrowstyle="->", color="#536277", lw=.85))
    fig.text(left, .985, "High noise", fontsize=8)
    fig.text(left+width/2, .985, "Denoising progress", ha="center", fontsize=8)
    fig.text(left+width, .985, "Low noise", ha="right", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output.with_suffix("."+suffix), dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Replot exported compact data")
    parser.add_argument("--output", type=Path, default=FIGURE_DIR / "figure1_empirical")
    parser.add_argument("--trajectory", type=Path,
                        default=SAMPLING_DIR / "trajectories.npz")
    parser.add_argument("--seed", type=int, default=6,
                        help="Candidate seed to select from the full trajectory archive")
    parser.add_argument("--predictions", type=Path,
                        default=SAMPLING_DIR / "boundary_predictions.npz",
                        help="Teacher clean-image estimates at the two boundary states")
    parser.add_argument("--raw-states", action="store_true",
                        help="Display the noisy intermediate states instead of clean-image estimates")
    parser.add_argument("--minimal-labels", action="store_true",
                        help="Title the colorbar Correlation and hide numeric noise-bin ticks")
    args = parser.parse_args()
    data = json.loads(args.data.read_text()) if args.data else extract()
    if args.minimal_labels:
        data["figure_style"] = {"minimal_labels": True}
    trajectory_path = (args.data.parent / data["sampling"]["states_file"]
                       if args.data else args.trajectory)
    predictions, prediction_metadata = None, None
    with np.load(trajectory_path) as trajectory:
        states = trajectory["states"]
        sigmas = trajectory["sigmas"]
        if states.ndim == 5:
            from scripts.paper.figures.sample_figure1_trajectory import boundary_step_indices
            steps = np.asarray(boundary_step_indices(sigmas, data["phase_boundary_sigmas"]))
            seed = args.seed
            matches = np.flatnonzero(trajectory["seeds"] == seed)
            if len(matches) != 1:
                raise ValueError(f"Seed {seed} is not in the trajectory archive")
            states = states[steps, int(matches[0])]
            sigmas = sigmas[steps]
        else:
            steps = trajectory["step_indices"]
            seed = int(trajectory["seed"])
        if not args.raw_states and "predictions" in trajectory:
            predictions = trajectory["predictions"]
            prediction_metadata = json.loads(str(trajectory["prediction_metadata"]))
    if predictions is None and not args.raw_states:
        with np.load(args.predictions) as pack:
            assert int(pack["seed"]) == seed, "Recompute predictions for the selected seed"
            np.testing.assert_array_equal(pack["states"], states[1:3])
            np.testing.assert_array_equal(pack["step_indices"], steps[1:3])
            np.testing.assert_array_equal(pack["sigmas"], sigmas[1:3])
            predictions = pack["predictions"]
        prediction_metadata = json.loads(args.predictions.with_suffix(".json").read_text())
    if predictions is not None:
        assert prediction_metadata["teacher_sha256"] == data["teacher_sha256"]
        assert prediction_metadata["seed"] == seed
        assert prediction_metadata["step_indices"] == steps[1:3].tolist()
        assert predictions.shape == states[1:3].shape and np.isfinite(predictions).all()
    from scripts.paper.figures.sample_figure1_trajectory import display_rgb
    args.output.parent.mkdir(parents=True, exist_ok=True)
    states_file = args.output.with_name(args.output.name + "_states.npz")
    extra = (dict(predictions=predictions, prediction_metadata=json.dumps(prediction_metadata))
             if predictions is not None else {})
    np.savez_compressed(states_file, states=states, sigmas=sigmas, step_indices=steps, seed=seed, **extra)
    frames_rgb = display_rgb(states, sigmas)
    if predictions is not None:
        frames_rgb[1:3] = display_rgb(predictions, np.zeros(2))
    data["sampling"] = dict(seed=seed, step_indices=steps.tolist(), sigmas=sigmas.tolist(),
                            states_file=states_file.name,
                            frames_rgb=frames_rgb,
                            intermediate_display="denoised_estimate" if predictions is not None else "raw_state",
                            prediction_metadata=prediction_metadata,
                            display_rule=("Intermediate estimates: uint8(clip(128+127.5*D(x_sigma,sigma),0,255)); "
                                          "endpoints: uint8(clip(128+127.5*x_sigma/sqrt(1+sigma^2),0,255))"
                                          if predictions is not None else
                                          "uint8(clip(128+127.5*x_sigma/sqrt(1+sigma^2),0,255))"),
                            state_kind=("initial noisy state; two teacher clean-image predictions; final sample"
                                        if predictions is not None else "actual solver states; final frame is x_0"),
                            sampler="deterministic EDM Heun, 40 steps, rho=7, 79 NFE, no denoised clipping",
                            candidate_seeds=list(range(8)),
                            frame_selection="nearest solver state in log sigma to each routing boundary",
                            phase_boundary_sigmas=data["phase_boundary_sigmas"],
                            selection_reason="visually selected for clarity of facial features at the phase boundaries")
    plot(data, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2)+"\n")
    print(f"Wrote {args.output}.{{pdf,svg,png,json}}")
