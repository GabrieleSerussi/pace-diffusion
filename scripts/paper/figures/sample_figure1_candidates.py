#!/usr/bin/env python3
"""Sample new FFHQ seeds, recording noisy states and actual clean predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pace.teacher_models import load_teacher_network
from scripts.sample_edm_distilled import StackedRandomGenerator, edm_sampler, parse_int_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='8-17')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cache-dir', default=os.environ.get('PACE_MODEL_CACHE', str(ROOT/'checkpoints')),
                        help='Teacher checkpoint cache (default: $PACE_MODEL_CACHE or checkpoints/).')
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/figures/figure1/figure1_empirical_clean_labels.json',
                        help='Compact Figure 1 data written by plot_empirical_figure1.py --minimal-labels.')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/figure1_sampling/seeds8_17')
    args = parser.parse_args()
    seeds = parse_int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Provide a nonempty list of distinct seeds')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    data = json.loads(args.data.read_text())
    net = load_teacher_network(
        'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl',
        device=device, dtype=torch.float32, cache_dir=args.cache_dir, preset='ffhq_64_vp')
    assert net.teacher_spec['checkpoint_sha256'] == data['teacher_sha256']
    rng = StackedRandomGenerator(device, seeds)
    latents = rng.randn([len(seeds), 3, 64, 64], device=device)
    states = np.empty((41, len(seeds), 3, 64, 64), dtype=np.float64)
    denoised = np.empty((40, len(seeds), 3, 64, 64), dtype=np.float32)
    sigmas = np.zeros(41, dtype=np.float64)
    calls = 0
    started = time.monotonic()

    def record(module, inputs, prediction):
        nonlocal calls
        if calls % 2 == 0:
            step = calls // 2
            states[step] = inputs[0].detach().cpu().numpy()
            denoised[step] = prediction.detach().cpu().numpy()
            sigmas[step] = float(inputs[1][0])
            if step % 5 == 0 or step == 39:
                print(f'Seeds {seeds[0]}–{seeds[-1]}: step {step}/40, '
                      f'sigma={sigmas[step]:.5g}, elapsed={time.monotonic()-started:.1f}s', flush=True)
        calls += 1

    hook = net.register_forward_hook(record)
    try:
        with torch.inference_mode():
            final = edm_sampler(net, latents, randn_like=rng.randn_like,
                                num_steps=40, S_churn=0, clip_denoised=False)
    finally:
        hook.remove()
    assert calls == 79
    states[-1] = final.cpu().numpy()
    assert np.isfinite(states).all() and np.isfinite(denoised).all()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'trajectories.npz', states=states, denoised=denoised,
                        sigmas=sigmas, seeds=seeds)
    manifest = dict(teacher_sha256=data['teacher_sha256'], seeds=seeds,
                    sampler='repository EDM deterministic Heun', steps=40, nfe=79,
                    rho=7, sigma_min=.002, sigma_max=80, churn=0, clip_denoised=False,
                    network_dtype='float32', solver_dtype='float64', device=str(device),
                    rng=f'stacked per-image torch.Generator on {device}',
                    torch_version=torch.__version__, threads=args.threads,
                    denoised_source='actual teacher outputs captured during each main sampler call',
                    denoised_shape=list(denoised.shape), states_shape=list(states.shape),
                    elapsed_seconds=time.monotonic()-started,
                    note='New CPU seeds are device-specific; no previous trajectory or boundary is altered.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {len(seeds)} complete trajectories and their denoised predictions to {args.output}', flush=True)


if __name__ == '__main__':
    main()
