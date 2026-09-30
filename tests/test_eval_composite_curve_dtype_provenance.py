"""Tests for two properties of ``scripts/eval_composite_curve.py``:

1. ``--dtype`` is now accepted and forwarded into the ``evaluate_students.py``
   subprocess command it builds (previously it was silently dropped, so every
   caller got evaluate_students.py's own bf16 default regardless of what dtype
   the student was actually trained with, for example an FP32-trained
   dit_micro student evaluated in bf16).
2. ``curve.json`` now records the eval hyperparameters it was produced under
   (cfg_scale, num_samples, num_steps, sampler, dtype, ref_npz basename), and a
   resume REFUSES (raises SystemExit) if the stored config disagrees with the
   current invocation, instead of silently mixing FID points measured under
   different configs into one curve.

Pure / mocked -- no GPU, no real torchrun/ADM subprocess is ever executed.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_composite_curve as ecc  # noqa: E402


# --------------------------------------------------------------------------
# 1a. --dtype forwarding into the built evaluate_students.py command
# --------------------------------------------------------------------------

def _args(**overrides):
    """A stand-in for the argparse.Namespace main() builds, populated with just
    the fields build_evaluate_students_cmd/current_eval_config read."""
    base = dict(
        torchrun="python3 -m torch.distributed.run --nproc_per_node=2",
        repo="/repo", model_type="dit_micro", diffusion="edm", num_heads=3,
        num_samples=5000, num_steps=50, cfg_scale=2.0, sampler="ddim",
        dtype="bf16", grouping_json=None, ref_npz="/x/cifar_test_ref_32.npz",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dtype_default_bf16_is_forwarded():
    # Even the untouched default must be explicitly forwarded now (this is the
    # actual fix: previously --dtype was never emitted at all).
    cmd = ecc.build_evaluate_students_cmd(_args(), "/stage", "/stage/fid")
    assert "--dtype" in cmd
    assert cmd[cmd.index("--dtype") + 1] == "bf16"


def test_dtype_fp32_override_is_forwarded():
    cmd = ecc.build_evaluate_students_cmd(_args(dtype="fp32"), "/stage", "/stage/fid")
    assert cmd[cmd.index("--dtype") + 1] == "fp32"


def test_dtype_fp16_override_is_forwarded():
    cmd = ecc.build_evaluate_students_cmd(_args(dtype="fp16"), "/stage", "/stage/fid")
    assert cmd[cmd.index("--dtype") + 1] == "fp16"


def test_dtype_argparse_default_and_choices():
    # main()'s own argparse: no --dtype given -> "bf16" (preserves pre-fix
    # behavior byte-for-byte for every caller that doesn't opt in to fp32/fp16).
    ap = ecc.argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    assert ap.parse_args([]).dtype == "bf16"
    assert ap.parse_args(["--dtype", "fp32"]).dtype == "fp32"
    with pytest.raises(SystemExit):
        ap.parse_args(["--dtype", "int8"])


def test_grouping_json_appended_only_when_set():
    cmd_none = ecc.build_evaluate_students_cmd(_args(grouping_json=None), "/s", "/s/fid")
    assert "--grouping_json" not in cmd_none
    cmd_set = ecc.build_evaluate_students_cmd(_args(grouping_json="/g.json"), "/s", "/s/fid")
    assert cmd_set[cmd_set.index("--grouping_json") + 1] == "/g.json"


# --------------------------------------------------------------------------
# 1c. curve.json eval-config provenance: round trip + backward compat
# --------------------------------------------------------------------------

def test_current_eval_config_extracts_ref_npz_basename():
    cfg = ecc.current_eval_config(_args(ref_npz="/refs/adm_eval/cifar_test_ref_32.npz"))
    assert cfg == {
        "cfg_scale": 2.0, "num_samples": 5000, "num_steps": 50,
        "sampler": "ddim", "dtype": "bf16", "ref_npz": "cifar_test_ref_32.npz",
        "class_idx_override": None,
    }


def test_eval_config_round_trips_through_write_and_load(tmp_path):
    f = str(tmp_path / "curve.json")
    cfg = ecc.current_eval_config(_args(dtype="fp32", num_samples=10000))
    ecc.write_curve(f, "sd", [{"step": 1000, "FID": 12.3}], cfg)
    data = json.load(open(f))
    assert data["eval_config"] == cfg
    rows, done, stored = ecc.load_scored_rows(f)
    assert done == {1000}
    assert stored == cfg


def test_legacy_curve_json_without_eval_config_still_loads(tmp_path):
    # Backward compat: a curve.json written before this field existed (plain
    # {"student_dir":..., "curve":[...]}) must still load, with stored=None.
    f = str(tmp_path / "curve.json")
    json.dump({"student_dir": "sd", "curve": [{"step": 500, "FID": 20.0}]}, open(f, "w"))
    rows, done, stored = ecc.load_scored_rows(f)
    assert done == {500}
    assert stored is None


# --------------------------------------------------------------------------
# 1c. mismatched-config resume detection
# --------------------------------------------------------------------------

def test_check_eval_config_stored_none_is_always_compatible():
    ecc.check_eval_config(None, ecc.current_eval_config(_args()), "curve.json")  # no raise


def test_check_eval_config_identical_config_is_compatible():
    cfg = ecc.current_eval_config(_args())
    ecc.check_eval_config(dict(cfg), cfg, "curve.json")  # no raise


def test_check_eval_config_refuses_on_dtype_mismatch():
    stored = ecc.current_eval_config(_args(dtype="bf16"))
    current = ecc.current_eval_config(_args(dtype="fp32"))
    with pytest.raises(SystemExit, match="dtype"):
        ecc.check_eval_config(stored, current, "curve.json")


def test_check_eval_config_refuses_on_num_samples_mismatch():
    # The scenario the check guards against: the sample count raised from 5000 to
    # 10000 on a resubmit against an existing curve.json.
    stored = ecc.current_eval_config(_args(num_samples=5000))
    current = ecc.current_eval_config(_args(num_samples=10000))
    with pytest.raises(SystemExit, match="num_samples"):
        ecc.check_eval_config(stored, current, "curve.json")


# --------------------------------------------------------------------------
# End-to-end (mocked subprocess): main() forwards --dtype, records eval_config,
# and refuses a resume whose config disagrees with what's already on disk.
# --------------------------------------------------------------------------

FAKE_ADM_STDOUT = ("Inception Score: 9.0\nFID: 11.5\nsFID: 20.0\n"
                   "Precision: 0.6\nRecall: 0.5\n")


class _FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout


def _stage_one_phase_step(student_dir, step=1000):
    pdir = os.path.join(student_dir, "phase_0", "curve")
    os.makedirs(pdir, exist_ok=True)
    open(os.path.join(pdir, f"step_{step}.pt"), "w").close()


def _run_main(monkeypatch, argv, captured_cmds):
    def fake_run(cmd, *a, **k):
        captured_cmds.append(cmd)
        return _FakeCompleted(stdout=FAKE_ADM_STDOUT)

    monkeypatch.setattr(ecc.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["eval_composite_curve.py"] + argv)
    ecc.main()


def _base_argv(student_dir, out_json, extra=()):
    return [
        "--student_dir", student_dir, "--model_type", "dit_micro",
        "--ref_npz", "/refs/cifar_test_ref_32.npz", "--pack_size", "32",
        "--adm_python", "python3", "--adm_dir", "/adm", "--num_phases", "1",
        "--out_json", out_json, "--torchrun", "python3", "--repo", "/repo",
    ] + list(extra)


def test_main_forwards_dtype_and_records_eval_config(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_one_phase_step(student_dir, step=1000)
    out_json = str(tmp_path / "curve.json")
    cmds = []
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--dtype", "fp32"]), cmds)

    # The evaluate_students.py subprocess call forwarded --dtype fp32.
    eval_cmd = next(c for c in cmds if any("evaluate_students.py" in str(a) for a in c))
    assert eval_cmd[eval_cmd.index("--dtype") + 1] == "fp32"

    data = json.load(open(out_json))
    assert data["eval_config"]["dtype"] == "fp32"
    assert data["eval_config"]["ref_npz"] == "cifar_test_ref_32.npz"
    assert [r["step"] for r in data["curve"]] == [1000]
    assert data["curve"][0]["FID"] == 11.5


def test_main_resume_with_same_config_does_not_raise(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_one_phase_step(student_dir, step=1000)
    out_json = str(tmp_path / "curve.json")
    argv = _base_argv(student_dir, out_json, ["--dtype", "fp32"])
    _run_main(monkeypatch, argv, [])
    # Re-invoking with the IDENTICAL config (as an automatic Slurm requeue would)
    # must not raise, and (nothing new to score) must not shell out again.
    cmds2 = []
    _run_main(monkeypatch, argv, cmds2)
    assert cmds2 == []


def test_main_resume_with_changed_dtype_refuses(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_one_phase_step(student_dir, step=1000)
    out_json = str(tmp_path / "curve.json")
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--dtype", "fp32"]), [])
    # A second invocation against the SAME out_json with a different --dtype
    # (e.g. someone forgot to pass --dtype fp32 on a later resubmit) must refuse
    # rather than silently mixing a bf16 FID point into an fp32 curve.
    with pytest.raises(SystemExit, match="dtype"):
        _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--dtype", "bf16"]), [])


def test_main_resume_with_changed_num_samples_refuses(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_one_phase_step(student_dir, step=1000)
    out_json = str(tmp_path / "curve.json")
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--num_samples", "5000"]), [])
    with pytest.raises(SystemExit, match="num_samples"):
        _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--num_samples", "10000"]), [])


# --------------------------------------------------------------------------
# --max_new_steps: bound wall-clock per invocation under a hard time cap
# (e.g. a one-hour job limit) without changing what gets scored overall.
# --------------------------------------------------------------------------

def _stage_phase_steps(student_dir, steps, phase=0):
    pdir = os.path.join(student_dir, f"phase_{phase}", "curve")
    os.makedirs(pdir, exist_ok=True)
    for step in steps:
        open(os.path.join(pdir, f"step_{step}.pt"), "w").close()


def test_max_new_steps_caps_this_invocation_ascending_and_resume_continues(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_phase_steps(student_dir, [3000, 1000, 2000])  # unordered on disk
    out_json = str(tmp_path / "curve.json")

    # First invocation, capped at 1: only the SMALLEST step is scored (ascending).
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--max_new_steps", "1"]), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000]

    # Resume with the same cap: picks up exactly the next one, leaves the rest.
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--max_new_steps", "1"]), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000, 2000]

    # A final resume WITHOUT the cap processes everything still missing.
    _run_main(monkeypatch, _base_argv(student_dir, out_json), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000, 2000, 3000]


def test_max_new_steps_none_matches_uncapped_behavior(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_phase_steps(student_dir, [1000, 2000])
    out_json = str(tmp_path / "curve.json")
    # Not passing --max_new_steps at all must behave exactly as before the flag existed.
    _run_main(monkeypatch, _base_argv(student_dir, out_json), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000, 2000]


# --------------------------------------------------------------------------
# --steps + merge_and_write_curve: two invocations safely covering DISJOINT
# step batches of the SAME out_json (e.g. two Slurm jobs queued in parallel
# against one curve.json) without racing over which checkpoint gets scored,
# and without one invocation's write clobbering the other's.
# --------------------------------------------------------------------------

def test_merge_and_write_curve_prevents_concurrent_clobber(tmp_path):
    f = str(tmp_path / "curve.json")
    ecc.write_curve(f, "sd", [{"step": 0, "FID": 100.0}])  # baseline both "processes" see

    # "Process A" loaded rows=[step 0] at startup, computed step 10000, writes.
    rows_a = [{"step": 0, "FID": 100.0}, {"step": 10000, "FID": 20.0}]
    ecc.merge_and_write_curve(f, "sd", rows_a)

    # "Process B" ALSO loaded rows=[step 0] at startup (before A's write landed),
    # computed a DIFFERENT step, and writes AFTER A. A naive write_curve() here
    # would overwrite A's step-10000 point; merge_and_write_curve must not.
    rows_b = [{"step": 0, "FID": 100.0}, {"step": 15000, "FID": 18.0}]
    ecc.merge_and_write_curve(f, "sd", rows_b)

    data = json.load(open(f))
    assert sorted(r["step"] for r in data["curve"]) == [0, 10000, 15000]


def test_steps_flag_restricts_candidates_for_disjoint_parallel_batches(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_phase_steps(student_dir, [1000, 2000, 3000])
    out_json = str(tmp_path / "curve.json")

    # "Job A" is only allowed to touch step 2000.
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--steps", "2000"]), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [2000]

    # "Job B" (simulating a separate, concurrently-queued invocation) covers the
    # other two. Its own --steps allowlist keeps it off step 2000 even though
    # that step is (by now) already scored -- and the merge-on-write means its
    # write does not need to, and does not, disturb job A's point.
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--steps", "1000,3000"]), [])
    assert sorted(r["step"] for r in json.load(open(out_json))["curve"]) == [1000, 2000, 3000]


def test_steps_flag_is_intersected_with_still_missing_steps(tmp_path, monkeypatch):
    # A step already scored (present in --steps by mistake, e.g. a stale batch
    # list) must not be recomputed -- --steps only ever narrows the missing set.
    student_dir = str(tmp_path / "student")
    _stage_phase_steps(student_dir, [1000, 2000])
    out_json = str(tmp_path / "curve.json")
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--steps", "1000"]), [])
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000]

    cmds = []
    _run_main(monkeypatch, _base_argv(student_dir, out_json, ["--steps", "1000,2000"]), cmds)
    # Only step 2000 actually needed a subprocess call (1000 already scored).
    eval_cmds = [c for c in cmds if any("evaluate_students.py" in str(a) for a in c)]
    assert len(eval_cmds) == 1
    assert [r["step"] for r in json.load(open(out_json))["curve"]] == [1000, 2000]
