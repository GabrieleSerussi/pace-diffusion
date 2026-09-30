#!/usr/bin/env python3
"""Re-evaluate the FFHQ teacher's clean-image estimates at saved boundary states.

Does not resample, blur, enhance, or change the phase boundaries.
"""
import argparse
import hashlib
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
from scripts.paper.figures.sample_figure1_trajectory import display_rgb

FIGURE_DIR = ROOT / 'outputs/figures/figure1'
SAMPLING_DIR = ROOT / 'outputs/figure1_sampling'


def _display_path(path):
    """Repository-relative path when possible, so outputs carry no local paths."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cache-dir', default=os.environ.get('PACE_MODEL_CACHE', str(ROOT/'checkpoints')),
                        help='Teacher checkpoint cache (default: $PACE_MODEL_CACHE or checkpoints/).')
    parser.add_argument('--states', type=Path, default=FIGURE_DIR/'figure1_empirical_states.npz')
    parser.add_argument('--data', type=Path, default=FIGURE_DIR/'figure1_empirical.json',
                        help='Compact Figure 1 data written by plot_empirical_figure1.py.')
    parser.add_argument('--output', type=Path, default=SAMPLING_DIR/'boundary_predictions.npz')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = json.loads(args.data.read_text())
    with np.load(args.states) as pack:
        states, sigmas, steps = pack['states'], pack['sigmas'], pack['step_indices']
        seed = int(pack['seed'])
    assert len(states) == 4 and steps.tolist() == [0, 10, 35, 40]
    net = load_teacher_network(
        'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl',
        device=torch.device(args.device), dtype=torch.float32,
        cache_dir=args.cache_dir, preset='ffhq_64_vp')
    assert net.teacher_spec['checkpoint_sha256'] == data['teacher_sha256']
    predictions = []
    with torch.inference_mode():
        for i in [1, 2]:
            x = torch.from_numpy(states[i:i+1]).to(args.device)
            sigma = torch.tensor([sigmas[i]], dtype=x.dtype, device=args.device)
            prediction = net(x, sigma, None).cpu().numpy()[0]
            assert np.isfinite(prediction).all()
            predictions.append(prediction)
            print(f'Predicted clean image at step {steps[i]}, sigma={sigmas[i]:.6g}', flush=True)
    predictions = np.stack(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predictions=predictions, states=states[1:3],
                        step_indices=steps[1:3], sigmas=sigmas[1:3], seed=seed)
    metadata = dict(teacher_sha256=data['teacher_sha256'], seed=seed,
                    source_states=_display_path(args.states),
                    source_states_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
                    device=args.device, dtype='float32 teacher; saved float64 solver inputs',
                    step_indices=steps[1:3].tolist(), sigmas=sigmas[1:3].tolist(),
                    kind='D(x_sigma, sigma): teacher prediction of clean x0; not a noisy solver state',
                    display_rule='uint8(clip(128 + 127.5*prediction, 0, 255)); no sigma scaling',
                    computation='Re-evaluation on saved states with same teacher; CPU may differ slightly from original GPU forward',
                    mae_to_final=[float(np.abs(pred-states[-1]).mean()) for pred in predictions])
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rgb = display_rgb(states, sigmas)
    rgb[1:3] = display_rgb(predictions, np.zeros(2))
    sheet = Image.new('RGB', (4*190, 215), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (frame, label) in enumerate(zip(rgb, ['Initial noise', 'Boundary 1: clean estimate',
                                               'Boundary 2: clean estimate', 'Final sample'])):
        draw.text((i*190+4, 3), label, fill='black')
        sheet.paste(Image.fromarray(frame).resize((180,180), Image.Resampling.NEAREST), (i*190+4, 25))
    sheet.save(args.output.with_suffix('.png'))
    print(json.dumps(metadata['mae_to_final']), flush=True)


if __name__ == '__main__':
    main()
