# PACE in five minutes (Colab notebook)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrieleSerussi/pace-diffusion/blob/main/colab/pace_playground.ipynb)

`pace_playground.ipynb` is a short tour of PACE (*Different Noise Levels, Different Needs: Discovering Diffusion Phases for Efficient Distillation*, Eitan Kosman, Gabriele Serussi, Chaim Baskin). It runs on the free CPU runtime in about a minute, from the released profile of one of the paper's four U-Net teachers (FFHQ-64 by default), and every result comes from the repository's own scripts.

1. **Setup.** Clones the repository; Colab already has the packages that these steps need.
2. **Where the teacher works.** The loss increase of every channel at every noise level, summed by U-Net level.
3. **Phases.** `scripts/optimize_timestep_grouping.py` cuts the noise range into phases; the similarity matrix shows them.
4. **Budgets.** `scripts/dry_run_capacity_allocation.py` splits the teacher's parameter count across the phases by measured demand, shown next to equal budgets.
5. **Routing.** An animation of the EDM sampler in which each denoising call runs the student of its phase, with the router's rule of `BlockwiseEDMStudent`.
6. **Your own teacher.** The four commands from a pretrained EDM teacher to student architectures. Setting `my_profile` to the `results.json` of your own teacher shows its phases, budgets and routing in the notebook.

The notebook was executed end to end for all four teachers; on a laptop CPU each run takes 4 to 8 seconds after the clone.
