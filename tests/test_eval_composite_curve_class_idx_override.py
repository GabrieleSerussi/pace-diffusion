"""``--class_idx_override`` (opt-in, forwarded to evaluate_students.py's own flag
of the same name): needed to curve-evaluate the structured-teacher KD students
(distilled from the unconditional finetune_xl_*/pretrain_ditb_* teachers, trained
via --force_label/--unconditional) -- without it, eval_composite_curve.py's
composite sampling would silently draw uniform-random REAL classes the student
never learned to condition on. Default None omits the flag entirely, byte-
identical to every invocation before this flag existed.

Pure / mocked -- no GPU, no real torchrun/ADM subprocess is ever executed.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_composite_curve as ecc  # noqa: E402


def _args(**overrides):
    base = dict(
        torchrun="python3 -m torch.distributed.run --nproc_per_node=2",
        repo="/repo", model_type="dit_xl", diffusion="ddpm", num_heads=3,
        num_samples=10000, num_steps=250, cfg_scale=1.0, sampler="ddpm",
        dtype="bf16", grouping_json=None, ref_npz="/x/VIRTUAL_lsun_bedroom256.npz",
        class_idx_override=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_class_idx_override_omitted_when_none():
    cmd = ecc.build_evaluate_students_cmd(_args(class_idx_override=None), "/s", "/s/fid")
    assert "--class_idx_override" not in cmd


def test_class_idx_override_forwarded_when_set():
    cmd = ecc.build_evaluate_students_cmd(_args(class_idx_override=1000), "/s", "/s/fid")
    assert cmd[cmd.index("--class_idx_override") + 1] == "1000"


def test_class_idx_override_zero_is_forwarded_not_treated_as_falsy():
    """0 is a valid, meaningful index (the DiT-B pretrains' unconditional class) --
    must not be dropped by an `if class_idx_override:` truthiness bug."""
    cmd = ecc.build_evaluate_students_cmd(_args(class_idx_override=0), "/s", "/s/fid")
    assert cmd[cmd.index("--class_idx_override") + 1] == "0"


def test_build_cmd_missing_attribute_defaults_to_omitted():
    """A pre-existing caller building a bare stand-in without this attribute at
    all (it postdates every other field build_evaluate_students_cmd reads) must
    keep working exactly as before this flag existed."""
    args = _args()
    del args.class_idx_override
    cmd = ecc.build_evaluate_students_cmd(args, "/s", "/s/fid")
    assert "--class_idx_override" not in cmd


def test_current_eval_config_includes_class_idx_override():
    cfg = ecc.current_eval_config(_args(class_idx_override=1000))
    assert cfg["class_idx_override"] == 1000


def test_current_eval_config_missing_attribute_defaults_to_none():
    args = _args()
    del args.class_idx_override
    cfg = ecc.current_eval_config(args)
    assert cfg["class_idx_override"] is None


class _FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout


FAKE_ADM_STDOUT = ("Inception Score: 9.0\nFID: 11.5\nsFID: 20.0\n"
                   "Precision: 0.6\nRecall: 0.5\n")


def _stage_one_phase_step(student_dir, step=1000):
    pdir = os.path.join(student_dir, "phase_0", "curve")
    os.makedirs(pdir, exist_ok=True)
    open(os.path.join(pdir, f"step_{step}.pt"), "w").close()


def test_main_forwards_class_idx_override_end_to_end(tmp_path, monkeypatch):
    student_dir = str(tmp_path / "student")
    _stage_one_phase_step(student_dir, step=1000)
    out_json = str(tmp_path / "curve.json")
    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(cmd)
        return _FakeCompleted(stdout=FAKE_ADM_STDOUT)

    monkeypatch.setattr(ecc.subprocess, "run", fake_run)
    argv = [
        "eval_composite_curve.py",
        "--student_dir", student_dir, "--model_type", "dit_xl",
        "--ref_npz", "/refs/VIRTUAL_lsun_bedroom256.npz", "--pack_size", "256",
        "--adm_python", "python3", "--adm_dir", "/adm", "--num_phases", "1",
        "--out_json", out_json, "--torchrun", "python3", "--repo", "/repo",
        "--class_idx_override", "1000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ecc.main()

    eval_cmd = next(c for c in cmds if any("evaluate_students.py" in str(a) for a in c))
    assert eval_cmd[eval_cmd.index("--class_idx_override") + 1] == "1000"
    data = json.load(open(out_json))
    assert data["eval_config"]["class_idx_override"] == 1000
