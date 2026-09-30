#!/usr/bin/env python3
"""Record genuine sampler states using the repository's EDM Heun sampler."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pace.teacher_models import load_teacher_network
from scripts.sample_edm_distilled import edm_sampler, StackedRandomGenerator


def display_rgb(states, sigmas):
    # One fixed analytic scale rule, not a per-frame contrast adjustment.
    scaled = states / np.sqrt(1 + np.asarray(sigmas) ** 2)[..., None, None, None]
    return np.clip(scaled * 127.5 + 128, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


def boundary_step_indices(sigmas, boundary_sigmas):
    sigmas = np.asarray(sigmas)
    positive = np.flatnonzero(sigmas > 0)
    middle = [int(positive[np.argmin(abs(np.log(sigmas[positive])-np.log(target)))])
              for target in boundary_sigmas]
    return [0, *middle, len(sigmas)-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-dir', default=os.environ.get('PACE_MODEL_CACHE', str(ROOT/'checkpoints')),
                        help='Teacher checkpoint cache (default: $PACE_MODEL_CACHE or checkpoints/).')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/figure1_sampling')
    parser.add_argument('--selected-seed', type=int, choices=range(8), default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    from scripts.paper.figures.plot_empirical_figure1 import extract
    data = extract()
    net = load_teacher_network(
        'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl',
        device=device, dtype=torch.float32, cache_dir=args.cache_dir, preset='ffhq_64_vp')
    assert net.teacher_spec['checkpoint_sha256'] == data['teacher_sha256']
    seeds = list(range(8))
    rng = StackedRandomGenerator(device, seeds)
    latents = rng.randn([len(seeds), 3, 64, 64], device=device)
    states, sigmas = [], []
    calls = 0

    def record(module, inputs):
        nonlocal calls
        # Calls alternate main-step evaluation and Heun correction.
        if calls % 2 == 0:
            states.append(inputs[0].detach().cpu().numpy())
            sigmas.append(float(inputs[1][0]))
            if calls % 20 == 0:
                print(f'Recorded solver step {calls//2}, sigma={sigmas[-1]:.4g}', flush=True)
        calls += 1

    hook = net.register_forward_pre_hook(record)
    with torch.inference_mode():
        final = edm_sampler(net, latents, randn_like=rng.randn_like, num_steps=40,
                            S_churn=0, clip_denoised=False)
    hook.remove()
    assert calls == 79 and len(states) == 40
    states.append(final.cpu().numpy())
    sigmas.append(0.)
    states = np.stack(states)
    assert np.isfinite(states).all()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'trajectories.npz', states=states,
                        sigmas=sigmas, seeds=seeds)
    indices = boundary_step_indices(sigmas, data['phase_boundary_sigmas'])
    np.savez_compressed(args.output/'selected_trajectory.npz',
                        states=states[indices, args.selected_seed], sigmas=np.array(sigmas)[indices],
                        step_indices=indices, seed=args.selected_seed)
    manifest = dict(teacher_sha256=data['teacher_sha256'], seeds=seeds,
                    sampler='repository edm_sampler, deterministic Heun', steps=40,
                    nfe=79, rho=7, sigma_min=.002, sigma_max=80, churn=0,
                    clip_denoised=False, network_dtype='float32', solver_dtype='float64',
                    device=str(device), torch_version=torch.__version__,
                    display_rule='uint8(clip(128 + 127.5*x_sigma/sqrt(1+sigma^2), 0, 255))',
                    selected_step_indices=indices, selected_sigmas=[sigmas[i] for i in indices],
                    selected_seed=args.selected_seed,
                    phase_boundary_sigmas=data['phase_boundary_sigmas'],
                    frame_selection='four frames: initial noise, nearest states to two phase boundaries, final sample',
                    selection_reason='visually selected for clarity of facial features at the phase boundaries',
                    state_kind='Actual noisy solver states x_sigma, followed by final x_0; not denoised predictions')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sheet = Image.new('RGB', (len(indices)*150+70, len(seeds)*150+30), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(indices):
        draw.text((75+j*150, 5), f'step {i}, sigma {sigmas[i]:.3g}', fill='black')
    for k, seed in enumerate(seeds):
        draw.text((5, 80+k*150), f'seed {seed}', fill='black')
        for j, rgb in enumerate(display_rgb(states[indices, k], np.array(sigmas)[indices])):
            sheet.paste(Image.fromarray(rgb).resize((144, 144), Image.Resampling.NEAREST),
                        (70+j*150, 30+k*150))
    sheet.save(args.output/'candidate_contact_sheet.png')
    print(f'Wrote trajectories and contact sheet to {args.output}', flush=True)


if __name__ == '__main__':
    main()
