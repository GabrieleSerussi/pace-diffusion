import os, sys, glob, warnings, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import (save_full_ckpt, load_full_ckpt, restore_rng, prune_curve_ckpts,
                                   create_ema, _as_cpu_byte, _restore_cuda_rng_states,
                                   validate_full_ckpt_payload)
from evaluate_parameters_edm import atomic_torch_save


def test_restore_rng_coerces_and_roundtrips(tmp_path):
    # _as_cpu_byte must coerce any tensor (e.g. a state loaded via map_location=cuda, or a
    # non-uint8) to a CPU uint8 ByteTensor -- torch.set_rng_state requires exactly that.
    # This is the regression for the resume crash: full ckpts load with map_location=device,
    # so rng["torch"] came back on CUDA and set_rng_state raised "must be a torch.ByteTensor".
    b = _as_cpu_byte(torch.arange(5, dtype=torch.int64))
    assert b.device.type == "cpu" and b.dtype == torch.uint8
    # real save -> load -> restore cycle reproduces the torch RNG stream
    m = torch.nn.Linear(2, 2); ema = create_ema(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    p = tmp_path / "last.pt"
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=1, epoch=0, cfg={}, kind="last")
    ck = load_full_ckpt(p)
    restore_rng(ck["rng"]); a = torch.randn(3)
    restore_rng(ck["rng"]); a2 = torch.randn(3)
    assert torch.equal(a, a2)   # deterministic stream after restore (no crash)

def test_full_ckpt_roundtrip_exact_resume(tmp_path):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4); ema = create_ema(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    # take 2 steps
    for _ in range(2):
        opt.zero_grad(); (m(torch.randn(8,4)).sum()).backward(); opt.step(); sch.step()
    p = tmp_path/"last.pt"
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=2, epoch=1, cfg={"x":1}, kind="last")
    ck = load_full_ckpt(p)
    assert ck["step"] == 2 and ck["epoch"] == 1 and ck["kind"] == "last" and ck["cfg"] == {"x":1}
    # restore into fresh objects and confirm optimizer state (exp_avg) matches
    m2 = torch.nn.Linear(4,4); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    m2.load_state_dict(ck["model"]); opt2.load_state_dict(ck["optimizer"])
    assert torch.allclose(list(m.parameters())[0], list(m2.parameters())[0])
    s1 = opt.state[list(opt.state)[0]]["exp_avg"]; s2 = opt2.state[list(opt2.state)[0]]["exp_avg"]
    assert torch.allclose(s1, s2)
    assert ck["ema"] is not None   # ema saved

def test_validate_full_ckpt_payload_accepts_a_real_full_ckpt(tmp_path):
    # Hardening item 4: a genuine save_full_ckpt() payload must pass through
    # validate_full_ckpt_payload completely unaffected (opt-in-safe).
    m = torch.nn.Linear(2, 2); ema = create_ema(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    p = tmp_path / "last.pt"
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=1, epoch=0, cfg={}, kind="last")
    ck = load_full_ckpt(p)
    validated = validate_full_ckpt_payload(ck, str(p))
    assert validated is ck


def test_validate_full_ckpt_payload_rejects_bare_state_dict(tmp_path):
    # bugs #15a/b: a plain model.state_dict() (e.g. a curve/step_<k>.pt or a
    # teacher checkpoint) accidentally placed at last.pt must fail LOUDLY and
    # NAME THE FILE + its actual keys, not raise a bare KeyError deep inside
    # training once resume_ckpt["optimizer"] is accessed.
    m = torch.nn.Linear(2, 2)
    p = tmp_path / "last.pt"
    torch.save(m.state_dict(), p)
    payload = torch.load(p, weights_only=False)
    with pytest.raises(ValueError) as exc:
        validate_full_ckpt_payload(payload, str(p))
    msg = str(exc.value)
    assert str(p) in msg
    assert "optimizer" in msg  # names at least one of the actually-missing keys


def test_validate_full_ckpt_payload_rejects_curve_snapshot_seeded_as_full_ckpt(tmp_path):
    # bugs #15a/b exactly: a {"model": ..., "ema": ...}-only curve snapshot
    # (this codebase's curve/step_<k>.pt shape) seeded into the resume slot
    # in place of a real --full_ckpt payload.
    m = torch.nn.Linear(2, 2); ema = create_ema(m)
    p = tmp_path / "last.pt"
    torch.save({"model": m.state_dict(), "ema": ema.state_dict()}, p)
    payload = torch.load(p, weights_only=False)
    with pytest.raises(ValueError) as exc:
        validate_full_ckpt_payload(payload, str(p))
    msg = str(exc.value)
    assert str(p) in msg
    assert "['model', 'ema']" in msg or "'model'" in msg  # actual keys reported
    for k in ("optimizer", "scheduler", "step", "epoch", "rng"):
        assert k in msg  # every missing required key is named


def test_validate_full_ckpt_payload_rejects_non_dict_payload(tmp_path):
    p = tmp_path / "last.pt"
    with pytest.raises(ValueError, match=str(p).replace("(", r"\(").replace(")", r"\)")):
        validate_full_ckpt_payload([1, 2, 3], str(p))


# ---------------------------------------------------------------------------
# Hardening item 5: atomic_torch_save (tmp+os.replace) + opt-in keep_prev
# rotation, so a crash mid-write can never leave a truncated file at the
# real path, and (opt-in) a filesystem-level-corrupt file has a fallback.
# ---------------------------------------------------------------------------

def test_atomic_torch_save_leaves_no_tmp_file_and_round_trips(tmp_path):
    p = tmp_path / "x.pt"
    atomic_torch_save({"a": 1}, p)
    assert os.path.exists(p)
    assert not os.path.exists(str(p) + ".tmp")
    assert torch.load(p, weights_only=False) == {"a": 1}


def test_atomic_torch_save_default_keep_prev_false_never_creates_prev_file(tmp_path):
    p = tmp_path / "x.pt"
    atomic_torch_save({"a": 1}, p)
    atomic_torch_save({"a": 2}, p)  # overwrite
    assert not os.path.exists(str(p) + ".prev")
    assert torch.load(p, weights_only=False) == {"a": 2}


def test_atomic_torch_save_keep_prev_true_rotates_existing_file(tmp_path):
    p = tmp_path / "x.pt"
    atomic_torch_save({"a": 1}, p, keep_prev=True)
    assert not os.path.exists(str(p) + ".prev")  # nothing to rotate on first write
    atomic_torch_save({"a": 2}, p, keep_prev=True)
    assert os.path.exists(str(p) + ".prev")
    assert torch.load(p, weights_only=False) == {"a": 2}
    assert torch.load(str(p) + ".prev", weights_only=False) == {"a": 1}
    # A third write rotates again -- .prev always holds the IMMEDIATELY prior
    # generation, not a deep history.
    atomic_torch_save({"a": 3}, p, keep_prev=True)
    assert torch.load(p, weights_only=False) == {"a": 3}
    assert torch.load(str(p) + ".prev", weights_only=False) == {"a": 2}


def test_save_full_ckpt_default_keep_prev_false_is_byte_identical_footprint(tmp_path):
    # Regression guard: default behavior must not create any .prev file --
    # opt-in-safe per the hardening item 5 requirement.
    m = torch.nn.Linear(2, 2); ema = create_ema(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    p = tmp_path / "last.pt"
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=1, epoch=0, cfg={}, kind="last")
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=2, epoch=1, cfg={}, kind="last")
    assert not os.path.exists(str(p) + ".prev")


def test_save_full_ckpt_keep_prev_true_gives_one_generation_fallback(tmp_path):
    m = torch.nn.Linear(2, 2); ema = create_ema(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    p = tmp_path / "last.pt"
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=1, epoch=0, cfg={}, kind="last",
                    keep_prev=True)
    save_full_ckpt(p, model=m, ema=ema, optimizer=opt, scheduler=sch, step=2, epoch=1, cfg={}, kind="last",
                    keep_prev=True)
    assert os.path.exists(str(p) + ".prev")
    prev = load_full_ckpt(str(p) + ".prev")
    cur = load_full_ckpt(p)
    assert prev["step"] == 1 and cur["step"] == 2
    # The fallback is itself a valid, resumable full checkpoint.
    validate_full_ckpt_payload(prev, str(p) + ".prev")


def test_prune_curve_keeps_newest(tmp_path):
    d = tmp_path/"curve"; d.mkdir()
    for k in [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000]:  # 12
        torch.save({"model":{}}, d/f"step_{k}.pt")
    prune_curve_ckpts(str(d), keep=10)
    left = sorted(int(f.split("step_")[1].split(".pt")[0]) for f in glob.glob(str(d/"step_*.pt")))
    assert left == [2000,3000,4000,5000,6000,7000,8000,9000,10000,11000]   # 10 newest


# ---------------------------------------------------------------------------
# _restore_cuda_rng_states: resume-across-world-size CUDA RNG fix.
#
# Bug this covers (found by a resume smoke test on a GPU cluster):
# a --full_ckpt written by an 8-GPU job stores torch.cuda.get_rng_state_all()
# -- one entry per device VISIBLE AT SAVE TIME (8). The historical
# restore_rng replayed all 8 via set_rng_state_all() unconditionally on every
# rank; resuming at a smaller world_size (fewer visible devices, e.g. 2 under
# --grad_accum) crashed with "IndexError: tuple index out of range" the
# instant torch.cuda.set_rng_state_all indexed device 2 against only 2
# torch.cuda.default_generators. All CUDA calls are dependency-injected so
# this is fully exercised CPU-only (no real CUDA needed), the same style
# tests/test_grad_accum.py uses to monkeypatch trainer-level functions.
# ---------------------------------------------------------------------------

class _FakeCuda:
    """Records set_rng_state/set_rng_state_all calls instead of touching real CUDA."""
    def __init__(self, n_devices, current):
        self._n = n_devices
        self._current = current
        self.set_all_calls = []
        self.set_one_calls = []

    def device_count(self):
        return self._n

    def current_device(self):
        return self._current

    def set_rng_state_all(self, states):
        self.set_all_calls.append(states)

    def set_rng_state(self, state, device):
        self.set_one_calls.append((state, device))


def _states(n):
    return [torch.full((4,), i, dtype=torch.uint8) for i in range(n)]


def test_restore_cuda_rng_matched_device_count_is_byte_identical_to_historical():
    # Matched save/restore device count (the ONLY case that existed before this
    # fix) -> historical set_rng_state_all(ALL states) path, unchanged.
    fake = _FakeCuda(n_devices=3, current=1)
    states = _states(3)
    _restore_cuda_rng_states(
        states, device_count=fake.device_count, current_device=fake.current_device,
        set_rng_state_all=fake.set_rng_state_all, set_rng_state=fake.set_rng_state,
    )
    assert len(fake.set_all_calls) == 1
    assert all(torch.equal(a, b) for a, b in zip(fake.set_all_calls[0], states))
    assert fake.set_one_calls == []   # single-device path never touched


def test_restore_cuda_rng_mismatch_restores_only_calling_ranks_own_device():
    # World-size shrink: checkpoint has 8 saved states (the wsd_bedroom repro's
    # actual shape), only 2 devices visible now, this rank is device 0.
    fake = _FakeCuda(n_devices=2, current=0)
    states = _states(8)
    _restore_cuda_rng_states(
        states, device_count=fake.device_count, current_device=fake.current_device,
        set_rng_state_all=fake.set_rng_state_all, set_rng_state=fake.set_rng_state,
    )
    assert fake.set_all_calls == []   # never replays the full (now out-of-range) list
    assert len(fake.set_one_calls) == 1
    got_state, got_device = fake.set_one_calls[0]
    assert got_device == 0
    assert torch.equal(got_state, states[0])   # THIS rank's own saved entry, not another's


def test_restore_cuda_rng_mismatch_rank1_restores_its_own_index():
    # Same checkpoint, a different rank (current_device=1) -- confirms it's
    # indexed by the CALLING rank's own device, not always device 0.
    fake = _FakeCuda(n_devices=2, current=1)
    states = _states(8)
    _restore_cuda_rng_states(
        states, device_count=fake.device_count, current_device=fake.current_device,
        set_rng_state_all=fake.set_rng_state_all, set_rng_state=fake.set_rng_state,
    )
    got_state, got_device = fake.set_one_calls[0]
    assert got_device == 1
    assert torch.equal(got_state, states[1])


def test_restore_cuda_rng_own_device_out_of_range_warns_and_skips_not_raises():
    # Calling rank's device index (5) exceeds what the checkpoint saved (2
    # states) -- must warn, must NOT raise, must NOT call either setter.
    fake = _FakeCuda(n_devices=6, current=5)
    states = _states(2)
    warned = []
    _restore_cuda_rng_states(
        states, device_count=fake.device_count, current_device=fake.current_device,
        set_rng_state_all=fake.set_rng_state_all, set_rng_state=fake.set_rng_state,
        warn=lambda msg: warned.append(msg),
    )
    assert fake.set_all_calls == [] and fake.set_one_calls == []
    assert len(warned) == 1 and "skipping CUDA RNG restore" in warned[0]


def test_restore_rng_wires_real_torch_cuda_functions_on_mismatch(monkeypatch):
    # End-to-end through the public restore_rng() entry point: patch
    # torch.cuda.* (is_available/device_count/current_device/set_rng_state*)
    # so the mismatch branch is reached and wired correctly, without any real
    # CUDA device. Regression target: this exact call used to be
    # `torch.cuda.set_rng_state_all(all_8_states)` unconditionally and crashed.
    fake = _FakeCuda(n_devices=2, current=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", fake.device_count)
    monkeypatch.setattr(torch.cuda, "current_device", fake.current_device)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake.set_rng_state_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state", fake.set_rng_state)

    rng = {
        "torch": torch.get_rng_state(),
        "cuda": _states(8),   # as an 8-GPU job would have saved
        "numpy": __import__("numpy").random.get_state(),
        "python": __import__("random").getstate(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # would turn an unexpected warning into a failure
        restore_rng(rng)   # must not raise (historical code: IndexError here)
    assert fake.set_all_calls == []
    assert len(fake.set_one_calls) == 1
    got_state, got_device = fake.set_one_calls[0]
    assert got_device == 0 and torch.equal(got_state, rng["cuda"][0])
