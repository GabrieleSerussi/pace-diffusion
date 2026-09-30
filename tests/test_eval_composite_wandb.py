import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eval_composite_curve import log_row_to_wandb


class FakeWandb:
    """Records wandb.log/finish calls so we can assert on them without a real run."""

    def __init__(self):
        self.logged = []   # list of (payload, kwargs)
        self.finished = 0
        self.inited = None

    def init(self, **kwargs):
        self.inited = kwargs
        return self

    def log(self, payload, **kwargs):
        self.logged.append((payload, kwargs))

    def finish(self):
        self.finished += 1


ROW = {"step": 3000, "FID": 12.5, "Inception Score": 9.0,
       "sFID": 7.1, "Precision": 0.6, "Recall": 0.5}


def test_log_row_maps_all_metrics_to_eval_keys():
    w = FakeWandb()
    assert log_row_to_wandb(w, ROW) is True
    assert len(w.logged) == 1
    payload, kwargs = w.logged[0]
    assert payload == {
        "eval/step": 3000,
        "eval/FID": 12.5,
        "eval/InceptionScore": 9.0,
        "eval/sFID": 7.1,
        "eval/Precision": 0.6,
        "eval/Recall": 0.5,
    }
    assert kwargs == {"step": 3000}  # logged at the training step index


def test_log_row_none_wandb_is_noop():
    # No flags -> wandb=None -> nothing logged, no exception.
    assert log_row_to_wandb(None, ROW) is False


def test_log_row_guards_missing_metric_keys():
    w = FakeWandb()
    partial = {"step": 100, "FID": 8.0}  # only FID parsed
    assert log_row_to_wandb(w, partial) is True
    payload, _ = w.logged[0]
    assert payload == {"eval/step": 100, "eval/FID": 8.0}
    assert "eval/Recall" not in payload


def test_log_row_never_raises_on_wandb_failure():
    class Boom:
        def log(self, *a, **k):
            raise RuntimeError("wandb down")
    # A failing wandb.log must be swallowed (curve.json stays source of truth).
    assert log_row_to_wandb(Boom(), ROW) is False


def _build_parser():
    """Mirror the arg surface added to eval_composite_curve.main so we can assert
    the flags parse and default to None (the no-wandb path)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_run_id", default=None)
    return ap


def test_flags_default_to_none_so_wandb_is_skipped():
    args = _build_parser().parse_args([])
    assert args.wandb_project is None and args.wandb_run_id is None
    # The main() gate is `if args.wandb_project and args.wandb_run_id` -> False here.
    assert not (args.wandb_project and args.wandb_run_id)


def test_flags_parse_when_both_given():
    args = _build_parser().parse_args(
        ["--wandb_project", "pace-test", "--wandb_run_id", "micro_global"])
    assert args.wandb_project == "pace-test"
    assert args.wandb_run_id == "micro_global"
    assert bool(args.wandb_project and args.wandb_run_id)
