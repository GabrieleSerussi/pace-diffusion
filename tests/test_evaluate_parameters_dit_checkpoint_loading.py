"""``load_dit_network``'s checkpoint loading, re-derived without ``DiT/download.py``'s
``find_model()`` (graduation-and-launch task, finetune-structured-teachers final
STRUCTURE GATE probe).

``find_model()`` calls ``torch.load(path, map_location=...)`` with no
``weights_only=False`` -- under torch>=2.6 (default flipped to
``weights_only=True``) this crashes loading our own ``--full_ckpt`` payloads,
which carry non-tensor Python/numpy RNG state the restrictive unpickler
rejects (observed in vivo: ``_pickle.UnpicklingError: ... numpy.core.multiarray
._reconstruct``, re-running the finetune-structured-teachers final-checkpoint
probe against the workspace's now-2.9.1 torch). ``find_model()``'s own
``if "ema" in checkpoint`` extraction also doesn't guard against an
explicitly-``None`` "ema" (written whenever a run trains without EMA, see
``train_phase_students.save_full_ckpt``).

``load_dit_network`` now uses a local ``_load_dit_checkpoint_prefer_ema``
instead, fixing both: ``weights_only=False`` (we only ever load our own
trusted local checkpoints here, never something downloaded from the
internet at this call site), and ``ema``-present-and-non-None -> ``ema``,
else ``model``, else the raw payload unchanged (bare state_dict, e.g. the
officially-released DiT-XL/2 weights, or any legacy non-full-ckpt run).
Mirrors ``evaluate_students._student_state_from_payload``'s extraction logic.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from evaluate_parameters_dit import _load_dit_checkpoint_prefer_ema  # noqa: E402


def test_prefers_ema_when_present_and_non_none(tmp_path):
    model_sd = {"a": torch.zeros(3), "b": torch.ones(2)}
    ema_sd = {"a": torch.full((3,), 7.0), "b": torch.full((2,), 9.0)}
    path = os.path.join(str(tmp_path), "student.pt")
    torch.save({"model": model_sd, "ema": ema_sd, "optimizer": {}, "rng": {}}, path)

    loaded = _load_dit_checkpoint_prefer_ema(path)
    assert torch.equal(loaded["a"], ema_sd["a"])
    assert torch.equal(loaded["b"], ema_sd["b"])


def test_falls_back_to_model_when_ema_is_none(tmp_path):
    model_sd = {"a": torch.zeros(3)}
    path = os.path.join(str(tmp_path), "student.pt")
    torch.save({"model": model_sd, "ema": None, "optimizer": {}, "rng": {}}, path)

    loaded = _load_dit_checkpoint_prefer_ema(path)
    assert torch.equal(loaded["a"], model_sd["a"])


def test_falls_back_to_model_when_ema_key_absent(tmp_path):
    model_sd = {"a": torch.zeros(3)}
    path = os.path.join(str(tmp_path), "student.pt")
    torch.save({"model": model_sd, "optimizer": {}}, path)

    loaded = _load_dit_checkpoint_prefer_ema(path)
    assert torch.equal(loaded["a"], model_sd["a"])


def test_bare_state_dict_returned_unchanged(tmp_path):
    """No {"model", "ema"} wrapping (e.g. the officially-released DiT-XL/2
    weights, or any legacy non-full-ckpt run) -- returned unchanged, matching
    find_model()'s own historical behavior for this shape exactly."""
    bare_sd = {"x_embedder.proj.weight": torch.randn(4, 4)}
    path = os.path.join(str(tmp_path), "bare.pt")
    torch.save(bare_sd, path)

    loaded = _load_dit_checkpoint_prefer_ema(path)
    assert torch.equal(loaded["x_embedder.proj.weight"], bare_sd["x_embedder.proj.weight"])


def test_loads_under_torch_weights_only_default_true_env():
    """Regression guard for the actual bug: a payload containing a non-tensor,
    non-allowlisted-by-default global (numpy RNG state, as save_full_ckpt's own
    "rng" field carries) must still load -- this is exactly what crashed with
    find_model()'s missing weights_only=False under torch>=2.6."""
    import numpy as np
    import tempfile
    payload = {
        "model": {"a": torch.zeros(2)},
        "ema": {"a": torch.ones(2)},
        "rng": {"numpy": np.random.get_state()},
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "student.pt")
        torch.save(payload, path)
        # Sanity: confirm this payload shape actually trips the strict default
        # unpickler (i.e. this test would have caught the original bug).
        try:
            torch.load(path, map_location="cpu", weights_only=True)
            trips_strict_default = False
        except Exception:
            trips_strict_default = True
        assert trips_strict_default, (
            "test payload no longer exercises the weights_only=True failure mode "
            "-- update it so this regression test still means something"
        )
        loaded = _load_dit_checkpoint_prefer_ema(path)
        assert torch.equal(loaded["a"], torch.ones(2))
