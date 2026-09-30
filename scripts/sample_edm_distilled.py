#!/usr/bin/env python3
"""Generate a small fixed-seed image set from a distilled EDM student snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.edm_distillation import construct_student_from_plan


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


class StackedRandomGenerator:
    def __init__(self, device: torch.device, seeds: list[int]):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size: Sequence[int], **kwargs):
        if size[0] != len(self.generators):
            raise ValueError("batch dimension must match number of seeds")
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, tensor: torch.Tensor):
        return self.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype)


def edm_sampler(
    net,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    randn_like=torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    clip_denoised: bool = False,
):
    """Deterministic (``S_churn=0``) or stochastic EDM Heun sampler.

    Adapted from ``edm_sampler`` in ``generate.py`` of NVlabs/edm (CC BY-NC-SA
    4.0; see ``THIRD_PARTY_NOTICES.md``), with optional clipping of every
    denoised prediction to ``[-1, 1]`` (the LSUN Bedroom protocol).
    """
    sigma_min = max(float(sigma_min), float(getattr(net, "sigma_min", 0.0)))
    sigma_max = min(float(sigma_max), float(getattr(net, "sigma_max", float("inf"))))
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        denoised = net(x_hat, t_hat.repeat(x_hat.shape[0]), class_labels).to(torch.float64)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next.repeat(x_next.shape[0]), class_labels).to(torch.float64)
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def make_labels(seeds: list[int], label_dim: int, device: torch.device) -> torch.Tensor | None:
    if label_dim <= 0:
        return None
    labels = torch.zeros([len(seeds), label_dim], device=device)
    class_indices = torch.as_tensor([seed % label_dim for seed in seeds], device=device, dtype=torch.long)
    labels.scatter_(1, class_indices.reshape(-1, 1), 1.0)
    return labels


def load_snapshot_network(snapshot_path: str | Path, *, device: torch.device) -> tuple[torch.nn.Module, dict]:
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    plan = snapshot.get("architecture_plan", {})
    if "ema_state_dict" in snapshot or "student_state_dict" in snapshot:
        if not plan:
            raise ValueError("state-dict snapshot is missing architecture_plan")
        net = construct_student_from_plan(plan)
        state_dict = snapshot.get("ema_state_dict", snapshot.get("student_state_dict"))
        net.load_state_dict(state_dict)
    else:
        net = snapshot.get("ema", snapshot.get("student"))
        if net is None:
            raise ValueError("snapshot must contain ema_state_dict/student_state_dict or legacy 'ema'/'student'")
    return net.eval().requires_grad_(False).to(device=device), plan


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0-15")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    device = torch.device(args.device)
    net, plan = load_snapshot_network(args.snapshot, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    saved: list[str] = []
    for start in tqdm(range(0, len(seeds), args.batch_size), desc="sampling", dynamic_ncols=True):
        batch_seeds = seeds[start : start + args.batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), int(net.img_channels), int(net.img_resolution), int(net.img_resolution)],
            device=device,
            dtype=torch.float32,
        )
        labels = make_labels(batch_seeds, int(getattr(net, "label_dim", 0)), device)
        images = edm_sampler(net, latents, class_labels=labels, randn_like=rnd.randn_like, num_steps=args.num_steps)
        images = (images * 0.5 + 0.5).clamp(0, 1)
        for seed, image in zip(batch_seeds, images):
            path = output_dir / f"seed{seed:06d}.png"
            save_image(image, path)
            saved.append(str(path))

    manifest = {"snapshot": args.snapshot, "architecture_variant": plan.get("variant"), "images": saved}
    (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
