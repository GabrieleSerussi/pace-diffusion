#!/usr/bin/env python3
"""For each saved training step, assemble a composite from every phase's step_<k>.pt,
sample, ADM-evaluate, and record (step, FID, IS, sFID, Prec, Rec). Emits curve.json.

Sampling needs a GPU (runs via torchrun + evaluate_students.py). The ADM evaluator
scoring runs CPU-only (CUDA_VISIBLE_DEVICES="" set for that subprocess).

The sampled PNGs are packed into the evaluator's NPZ format with
``scripts/data/pack_pngs_to_npz.py`` (or ``<adm_dir>/pack_pngs_to_npz.py`` when that
file exists), and scored with ``<adm_dir>/evaluator.py`` from
openai/guided-diffusion."""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _phase_curve_dir(student_dir, p):
    """Where phase ``p``'s FID-curve checkpoints live.

    --full_ckpt runs put them under ``phase_<p>/curve/step_<k>.pt``; legacy
    (non-full) runs keep them at the top level ``phase_<p>/step_<k>.pt``. Prefer the
    ``curve/`` subdir when it exists, else fall back to the legacy top-level layout."""
    pdir = os.path.join(student_dir, f"phase_{p}")
    curve = os.path.join(pdir, "curve")
    return curve if os.path.isdir(curve) else pdir


def discover_steps(student_dir, num_phases):
    """Steps for which ALL phases have a checkpoint (so a composite can be built).

    Each phase reads from whichever layout it uses (``curve/`` subdir for --full_ckpt
    runs, else legacy top-level); the returned steps are common to all phases."""
    per = []
    for p in range(num_phases):
        steps = set()
        for f in glob.glob(os.path.join(_phase_curve_dir(student_dir, p), "step_*.pt")):
            steps.add(int(re.search(r"step_(\d+)\.pt", f).group(1)))
        per.append(steps)
    if not per or any(len(s) == 0 for s in per):
        return []
    return sorted(set.intersection(*per))


def _parse_error_marker_path(out_json):
    return out_json + ".PARSE_ERROR"


def _write_parse_error_marker(out_json, msg):
    """Guard against a silent empty extraction: an EXISTING ``curve.json`` that fails to parse
    (truncated by a preemption mid-write, disk corruption, ...) must not look
    identical to "no rows yet" to a caller -- that is exactly how a watcher
    silently stalls forever with no diagnostic. Write a sentinel file a
    watcher or a human can check for instead. Best-effort: never lets a
    write failure here mask the original parse error for the caller."""
    try:
        with open(_parse_error_marker_path(out_json), "w") as f:
            f.write(f"{msg}\n")
    except OSError:
        pass


def _clear_parse_error_marker(out_json):
    """Remove a stale marker once the file parses cleanly again (e.g. after a
    human repairs it, or a fresh write replaces the corrupt one)."""
    try:
        os.remove(_parse_error_marker_path(out_json))
    except OSError:
        pass


def load_scored_rows(out_json):
    """Resume support: load already-scored rows from a partial ``out_json``.

    Returns ``(rows, done_steps, eval_config)`` where ``done_steps`` is the set of
    steps that already carry a parsed ``FID`` (and so can be skipped on a requeue),
    and ``eval_config`` is the eval-hyperparameter dict previously recorded by
    ``write_curve`` (see ``current_eval_config``), or ``None`` for a legacy
    ``curve.json`` written before that field existed (or a missing/fresh/corrupt
    file) -- ``None`` is always treated as compatible so old files keep loading.
    Rows lacking a ``FID`` (written before scoring finished) are dropped so they
    get recomputed, and a truncated/corrupt file (killed mid-write) starts clean.

    A MISSING file is ordinary "no rows yet" -- no marker written. An EXISTING
    file that fails to parse (or an unexpected top-level shape) additionally
    writes ``<out_json>.PARSE_ERROR`` (see ``_write_parse_error_marker``) so
    callers that want to distinguish "not ready yet" from "corrupted, needs a
    human" can check for it; the return value itself is unchanged (still the
    safe empty fallback) so every existing caller keeps working byte-identically."""
    if not os.path.exists(out_json):
        return [], set(), None
    try:
        data = json.load(open(out_json))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        _write_parse_error_marker(out_json, f"failed to parse {out_json}: {e}")
        return [], set(), None
    if not isinstance(data, dict) or "curve" not in data:
        got = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        _write_parse_error_marker(
            out_json, f"{out_json} has unexpected shape (expected a dict with a "
            f"'curve' key, got {got})")
        return [], set(), None
    _clear_parse_error_marker(out_json)
    rows = data.get("curve", []) if isinstance(data, dict) else []
    rows = [r for r in rows if isinstance(r, dict) and "step" in r and "FID" in r]
    eval_config = data.get("eval_config") if isinstance(data, dict) else None
    return rows, {r["step"] for r in rows}, eval_config


def current_eval_config(args):
    """The eval hyperparameters that determine what a scored FID row actually
    means. Recorded into curve.json so a later resume can detect a changed
    invocation silently mixing FID
    points computed under different configs into one curve."""
    return {
        "cfg_scale": args.cfg_scale,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "dtype": args.dtype,
        "ref_npz": os.path.basename(args.ref_npz),
        # getattr-guarded: pre-existing callers building a bare argparse.Namespace/
        # SimpleNamespace stand-in without this attribute (it postdates every other
        # field here) keep working unchanged, exactly like args.class_idx_override
        # in build_evaluate_students_cmd below.
        "class_idx_override": getattr(args, "class_idx_override", None),
    }


def check_eval_config(stored, current, out_json):
    """Refuse to resume if ``out_json`` already carries a recorded eval_config
    that disagrees with the current invocation.

    Policy: REFUSE (hard error), not warn-and-continue. A warning that still let
    the run proceed would recompute nothing and just append new rows next to old
    ones scored under a different config -- i.e. it would not actually prevent
    the silent-mixing failure mode this is meant to close. Refusing
    is safe for the normal automated case: a Slurm preemption/requeue re-invokes
    the identical launcher with identical --export values, so this only fires
    when a human has actually changed the eval config for an existing out_json
    (bump SAMP, change CFG, ...) -- exactly the case that used to corrupt the
    curve silently. ``stored=None`` (legacy curve.json predating this field, or
    no/fresh file) is always compatible -- nothing to compare against."""
    if stored is None:
        return
    mismatched = {k: (stored.get(k), current[k]) for k in current if stored.get(k) != current[k]}
    if mismatched:
        detail = "\n".join(f"  {k}: stored={old!r} != current={new!r}"
                            for k, (old, new) in mismatched.items())
        raise SystemExit(
            f"[curve] REFUSING to resume {out_json}: its recorded eval_config "
            f"differs from this invocation's, which would silently mix FID points "
            f"measured under different configs into one curve:\n{detail}\n"
            f"[curve] Fix: re-run with the prior config's flags, or point --out_json "
            f"at a new file to start a fresh curve under the new config."
        )


def write_curve(out_json, student_dir, rows, eval_config=None):
    """Atomically write the curve (sorted by step). Writing to a temp file then
    ``os.replace`` means a preemption mid-write cannot truncate ``out_json`` and
    lose already-scored points -- the old file stays intact until the rename.

    ``eval_config`` (see ``current_eval_config``) is stored alongside the rows so
    a later resume can detect a changed invocation."""
    rows = sorted(rows, key=lambda r: r["step"])
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"student_dir": student_dir, "curve": rows, "eval_config": eval_config},
                   f, indent=2)
    os.replace(tmp, out_json)


def merge_and_write_curve(out_json, student_dir, rows, eval_config=None):
    """Like ``write_curve``, but first re-reads whatever is CURRENTLY on disk at
    ``out_json`` and unions it with ``rows`` (keyed by step; ``rows`` -- this
    process's own view -- wins on overlap), instead of blindly overwriting.

    Without this, two ``eval_composite_curve.py`` invocations racing against the
    SAME ``out_json`` (e.g. two Slurm jobs deliberately queued at once, each given
    a disjoint ``--steps`` batch so whichever gets a node first can start) would
    silently clobber each other: each holds its own ``rows`` snapshot loaded once
    at startup (see ``main``), so a plain ``write_curve`` after the other process
    already landed its own new point would overwrite the file and drop that point.
    Re-reading immediately before every write closes this race in practice --
    writes are infrequent (about one per ~30-minute checkpoint), so the remaining
    TOCTOU window is negligible; a fully airtight fix would need a file lock this
    codebase doesn't otherwise use, which is out of scope here."""
    on_disk_rows, _, _ = load_scored_rows(out_json)
    merged = {r["step"]: r for r in on_disk_rows}
    merged.update({r["step"]: r for r in rows})
    write_curve(out_json, student_dir, list(merged.values()), eval_config)


def build_evaluate_students_cmd(args, stage, fdir):
    """Assemble the ``evaluate_students.py`` subprocess command for one staged
    composite (``stage``, sampled into ``fdir``). Pulled out of ``main`` as a pure
    function so the exact argv it builds -- notably that ``--dtype`` is forwarded
    -- is unit-testable without
    standing up real checkpoints/torchrun/ADM."""
    cmd = args.torchrun.split() + [
        f"{args.repo}/scripts/evaluate_students.py",
        "--model_type", args.model_type, "--mode", "composite",
        "--diffusion", args.diffusion, "--num_heads", str(args.num_heads),
        "--student_dir", stage,
        "--num_samples", str(args.num_samples), "--batch_size", "256",
        "--num_steps", str(args.num_steps), "--cfg_scale", str(args.cfg_scale),
        "--sampler", args.sampler, "--dtype", args.dtype,
        "--skip_fid", "--output_dir", fdir]
    if args.grouping_json is not None:
        cmd += ["--grouping_json", args.grouping_json]
    class_idx_override = getattr(args, "class_idx_override", None)
    if class_idx_override is not None:
        cmd += ["--class_idx_override", str(class_idx_override)]
    dit_repo = getattr(args, "dit_repo", None)
    if dit_repo is not None:
        cmd += ["--dit_repo", dit_repo]
    return cmd


def pack_script(args):
    """The PNG-to-NPZ packer: ``<adm_dir>/pack_pngs_to_npz.py`` when present,
    else this repository's ``scripts/data/pack_pngs_to_npz.py``."""
    candidate = os.path.join(args.adm_dir, "pack_pngs_to_npz.py")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(args.repo, "scripts", "data", "pack_pngs_to_npz.py")


def log_row_to_wandb(wandb, row):
    """Log one scored curve ``row`` (as produced in ``main``) into the active wandb
    run under ``eval/*`` keys, indexed by training step. ``wandb`` is the module (or a
    stand-in with a ``log`` method); pass ``None`` to make this a no-op so callers can
    invoke it unconditionally. Missing metric keys are skipped (guard partial rows).

    Wrapped in try/except so a wandb failure NEVER aborts the FID curve: curve.json
    stays the source of truth. Returns True if a log was attempted, False otherwise."""
    if wandb is None:
        return False
    k = row.get("step")
    payload = {}
    if k is not None:
        payload["eval/step"] = k
    for src, dst in (("FID", "eval/FID"), ("Inception Score", "eval/InceptionScore"),
                     ("sFID", "eval/sFID"), ("Precision", "eval/Precision"),
                     ("Recall", "eval/Recall")):
        if src in row:
            payload[dst] = row[src]
    try:
        if k is not None:
            wandb.log(payload, step=k)
        else:
            wandb.log(payload)
        return True
    except Exception as e:  # never let a wandb hiccup abort the curve
        print(f"[curve] WARNING: wandb.log failed for step {k}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", required=True)
    # Teacher-architecture composite runs pass a timestep_grouping.json for phase
    # routing. --arch_plan (width-allocation) runs may omit it: evaluate_students
    # falls back to the per-phase bins in each staged arch_cfg.json.
    ap.add_argument("--grouping_json", default=None)
    ap.add_argument("--model_type", required=True, choices=["dit_micro", "dit_xl"])
    ap.add_argument("--dit_repo", default=None,
                    help="facebookresearch/DiT checkout, forwarded to evaluate_students.py "
                         "(default: $DIT_REPO).")
    ap.add_argument("--diffusion", default="edm", choices=["edm", "ddpm"])
    ap.add_argument("--num_heads", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=5000)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"],
                    help="Autocast dtype forwarded to evaluate_students.py's own --dtype "
                         "(same flag/choices/semantics). Default 'bf16' preserves this "
                         "script's historical behavior: it never forwarded --dtype before "
                         "this flag existed, so evaluate_students.py's own bf16 default "
                         "silently applied to every caller (this is the bf16-eval side "
                         "of the dit_micro train-fp32/eval-bf16 mismatch). Passing nothing "
                         "keeps existing pipelines (e.g. dit_xl, which trains bf16) "
                         "byte-for-byte unchanged; a run whose training dtype differs (dit_micro, "
                         "trained fp32) should pass --dtype fp32 explicitly to match.")
    ap.add_argument("--class_idx_override", type=int, default=None,
                    help="Forwarded verbatim to evaluate_students.py's own "
                         "--class_idx_override (dit_xl only): force every sampled "
                         "image to this fixed label instead of a uniform-random real "
                         "class -- required for a checkpoint trained via a single "
                         "forced label (--force_label / --unconditional), e.g. "
                         "students distilled from the unconditional DiT-B/2 teachers. "
                         "Default None omits the flag entirely, byte-identical to "
                         "every invocation before this flag existed.")
    ap.add_argument("--ref_npz", required=True)
    ap.add_argument("--pack_size", type=int, required=True)   # 32 or 256
    ap.add_argument("--adm_python", required=True)
    ap.add_argument("--adm_dir", required=True)               # guided-diffusion evaluations/ (evaluator.py)
    ap.add_argument("--num_phases", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--torchrun", required=True)              # e.g. "<py> -m torch.distributed.run --nproc_per_node=N ..."
    ap.add_argument("--repo", required=True)
    # Optional: log the scored FID/IS/sFID/P/R into the SAME wandb run as training
    # (matches the trainer's wandb.init with a stable id + resume="allow"). Both must
    # be set to enable; otherwise this stays a pure no-op (no import, no logging).
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_run_id", default=None)
    # Optional cap on how many not-yet-scored steps (ascending) to evaluate in THIS
    # invocation. Default (None) preserves existing behavior byte-for-byte: process
    # every missing step. Set this to bound wall-clock under a hard time cap (e.g. a
    # Slurm partition with a 1h limit) -- leftover steps simply stay unscored and get
    # picked up by a later resume, same as if the process had been preempted, except
    # this way the job exits cleanly instead of risking a checkpoint being killed
    # mid-sample/mid-eval.
    ap.add_argument("--max_new_steps", type=int, default=None)
    # Optional explicit allowlist of steps (comma-separated) to restrict this
    # invocation's candidate set to, BEFORE --max_new_steps is applied. Default
    # (None) preserves existing behavior: every missing step is a candidate.
    # Lets two invocations be pointed at disjoint step batches of the SAME
    # student_dir/out_json (e.g. two Slurm jobs queued at once so whichever
    # gets a node first starts sooner) without racing over which checkpoint
    # each one claims -- see merge_and_write_curve() for the other half of
    # making that safe (the write side).
    ap.add_argument("--steps", default=None,
                    help="Comma-separated list of specific steps to restrict this "
                         "invocation's candidates to (intersected with steps still "
                         "missing; ascending order preserved).")
    args = ap.parse_args()

    # Lazily bring up wandb only when BOTH flags are given. A failure here must never
    # abort the curve: we fall back to wandb=None and keep writing curve.json.
    wandb = None
    if args.wandb_project and args.wandb_run_id:
        try:
            import wandb as _wandb
            _wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow")
            wandb = _wandb
            print(f"[curve] wandb: logging eval/* into run {args.wandb_run_id} "
                  f"(project {args.wandb_project})", flush=True)
        except Exception as e:
            wandb = None
            print(f"[curve] WARNING: wandb init failed ({e}); continuing without wandb", flush=True)

    steps = discover_steps(args.student_dir, args.num_phases)
    # Resume after preemption: keep rows already scored in a prior run; only
    # evaluate the steps that are still missing (avoids redoing the whole curve).
    rows, done, stored_config = load_scored_rows(args.out_json)
    eval_config = current_eval_config(args)
    check_eval_config(stored_config, eval_config, args.out_json)
    todo_all = [k for k in steps if k not in done]
    print(f"[curve] {len(steps)} common checkpoint step(s): {steps}", flush=True)
    print(f"[curve] resume: {len(done)} already scored {sorted(done)}; "
          f"{len(todo_all)} to evaluate {todo_all}", flush=True)
    if args.steps is not None:
        wanted = {int(s) for s in args.steps.split(",") if s.strip()}
        restricted = [k for k in todo_all if k in wanted]
        print(f"[curve] --steps restricts candidates to {sorted(wanted)}: "
              f"{restricted} of the missing set are in that set", flush=True)
        todo_all = restricted
    todo = todo_all[:args.max_new_steps] if args.max_new_steps is not None else todo_all
    if args.max_new_steps is not None and len(todo) < len(todo_all):
        print(f"[curve] --max_new_steps={args.max_new_steps}: this invocation will "
              f"evaluate only {todo}; {todo_all[len(todo):]} left for a later resume",
              flush=True)
    for k in todo:
        stage = tempfile.mkdtemp(prefix=f"curve_{k}_")
        try:
            for p in range(args.num_phases):
                src_pdir = os.path.join(args.student_dir, f"phase_{p}")
                pdir = os.path.join(stage, f"phase_{p}")
                os.makedirs(pdir, exist_ok=True)
                # Source the step ckpt from phase_<p>/curve/ for --full_ckpt runs, else
                # from the legacy top-level phase_<p>/ (matches discover_steps).
                shutil.copy(os.path.join(_phase_curve_dir(args.student_dir, p), f"step_{k}.pt"),
                            os.path.join(pdir, "student.pt"))
                # Stage the per-phase reconstruction config. Width-allocation
                # (--arch_plan) students carry arch_cfg.json (rebuilt as a NarrowDiT);
                # teacher-architecture students carry none.
                arch_cfg = os.path.join(src_pdir, "arch_cfg.json")
                if os.path.exists(arch_cfg):
                    shutil.copy(arch_cfg, os.path.join(pdir, "arch_cfg.json"))
            fdir = os.path.join(stage, "fid")
            cmd = build_evaluate_students_cmd(args, stage, fdir)
            subprocess.run(cmd, check=True)
            npz = os.path.join(stage, "samples.npz")
            subprocess.run([sys.executable, pack_script(args),
                            "--png_dir", f"{fdir}/samples", "--out", npz,
                            "--size", str(args.pack_size)], check=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            out = subprocess.run([args.adm_python, f"{args.adm_dir}/evaluator.py", args.ref_npz, npz],
                                 capture_output=True, text=True, env=env, cwd=args.adm_dir).stdout
            row = {"step": k}
            for key in ("FID", "Inception Score", "sFID", "Precision", "Recall"):
                m = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", out, re.M)
                if m:
                    row[key] = float(m.group(1))
            if "FID" not in row:
                print(f"[curve] WARNING: no FID parsed for step {k}; will retry on resume", flush=True)
                continue
            rows.append(row)
            print("[curve]", row, flush=True)
            # write incrementally + atomically (and merged with whatever's on disk,
            # so a concurrent invocation on a disjoint --steps batch can't clobber
            # this point) so a requeue resumes from here
            merge_and_write_curve(args.out_json, args.student_dir, rows, eval_config)
            # ...then mirror the same point into wandb (no-op / never fatal if unset)
            log_row_to_wandb(wandb, row)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    merge_and_write_curve(args.out_json, args.student_dir, rows, eval_config)
    print("wrote", args.out_json)
    if wandb is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"[curve] WARNING: wandb.finish failed: {e}", flush=True)


if __name__ == "__main__":
    main()
