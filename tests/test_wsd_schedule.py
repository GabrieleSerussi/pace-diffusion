import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import wsd_lr_lambda

def test_wsd_shape():
    T, W, C = 1000, 100, 0.2          # decay_start = 800
    assert wsd_lr_lambda(0, total_steps=T, warmup_steps=W, cooldown_frac=C) == 0.0
    assert abs(wsd_lr_lambda(50, total_steps=T, warmup_steps=W, cooldown_frac=C) - 0.5) < 1e-6
    assert wsd_lr_lambda(100, total_steps=T, warmup_steps=W, cooldown_frac=C) == 1.0
    assert wsd_lr_lambda(799, total_steps=T, warmup_steps=W, cooldown_frac=C) == 1.0
    assert abs(wsd_lr_lambda(800, total_steps=T, warmup_steps=W, cooldown_frac=C) - 1.0) < 1e-6
    assert wsd_lr_lambda(1000, total_steps=T, warmup_steps=W, cooldown_frac=C) == 0.0
    xs = [wsd_lr_lambda(s, total_steps=T, warmup_steps=W, cooldown_frac=C) for s in range(800, 1001)]
    assert all(xs[i] >= xs[i+1] - 1e-12 for i in range(len(xs)-1))   # non-increasing through cooldown
