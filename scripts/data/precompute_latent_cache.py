#!/usr/bin/env python3
"""
Tiny standalone driver: build a --latent_cache_dir persistent VAE latent
cache and exit. No training loop -- just the one-shot VAE encode that
train_phase_students.py would otherwise repeat at every process start.

Meant to run ONCE per dataset (for example as a short GPU job) ahead of the
actual training jobs, so the very first training
launch already finds a complete cache instead of paying the encode tax
itself. Idempotent: if a complete cache already exists for the given
fingerprint (dataset path + image count + VAE id + resolution), this exits
immediately without touching the VAE.

Always builds the CANONICAL (unflipped) cache regardless of whether a
downstream training job passes --hflip -- see
train_phase_students.PersistentLatentCache's docstring for how --hflip is
applied at LOAD time instead (flipping the sampled latent, not the raw
pixel), so this driver deliberately ignores --hflip and never applies it
here.

Usage (mirrors train_phase_students.py's --dataset image_folder wiring):
  torchrun --nproc_per_node=N scripts/data/precompute_latent_cache.py \\
      --image_root $DATA_ROOT/lsun_bedroom256/img256 --image_size 256 \\
      --latent_cache_dir $LATENT_CACHE/lsun_bedroom256
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# The training helpers live in scripts/ (one level up).
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from evaluate_parameters_edm import (  # noqa: E402
    ImageFolderFlat,
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_main_process,
)
from train_phase_students import (  # noqa: E402
    _VAE_PRETRAINED_ID,
    build_persistent_latent_cache,
    latent_cache_fingerprint,
    latent_cache_is_complete,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="image_folder", choices=["image_folder"],
                    help="Only image_folder (FFHQ256/Bedroom256) is wired -- this driver "
                         "exists specifically for the two datasets every latent-space "
                         "training job would otherwise re-encode on every restart.")
    p.add_argument("--image_root", required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--latent_cache_dir", required=True)
    p.add_argument("--vae_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    t0 = time.perf_counter()
    args = parse_args()
    device, rank, world_size = init_distributed(args.device)

    if is_main_process():
        print(f"[precompute_latent_cache] image_root={args.image_root} "
              f"cache_dir={args.latent_cache_dir} world_size={world_size} "
              f"{time.strftime('%F %T')}")

    # hflip is deliberately NOT threaded through here -- the persistent cache
    # always stores the CANONICAL (unflipped) encode; --hflip is applied at
    # PersistentLatentCache LOAD time instead (see its docstring).
    image_dataset = ImageFolderFlat(root=args.image_root, image_size=args.image_size,
                                     max_images=args.max_images, hflip=False)
    fingerprint = latent_cache_fingerprint(
        argparse.Namespace(dataset=args.dataset, image_root=args.image_root,
                            data_root=args.image_root),
        image_dataset, args.image_size,
    )

    if latent_cache_is_complete(args.latent_cache_dir, fingerprint):
        if is_main_process():
            print(f"[precompute_latent_cache] cache already complete at "
                  f"{args.latent_cache_dir} -> nothing to do "
                  f"({time.perf_counter() - t0:.1f}s)")
        cleanup_distributed()
        return

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(_VAE_PRETRAINED_ID).to(device)
    vae.eval().requires_grad_(False)

    build_persistent_latent_cache(
        image_dataset=image_dataset, vae=vae, cache_dir=args.latent_cache_dir,
        fingerprint=fingerprint, rank=rank, world_size=world_size, device=device,
        encode_batch_size=args.vae_batch_size, num_workers=args.num_workers,
    )

    if is_distributed():
        dist.barrier()
    if is_main_process():
        ok = latent_cache_is_complete(args.latent_cache_dir, fingerprint)
        print(f"[precompute_latent_cache] DONE cache_dir={args.latent_cache_dir} "
              f"complete={ok} elapsed={time.perf_counter() - t0:.1f}s")
        if not ok:
            raise SystemExit("build_persistent_latent_cache returned but the cache is "
                              "still not complete -- see build_persistent_latent_cache logs above")
    cleanup_distributed()


if __name__ == "__main__":
    main()
