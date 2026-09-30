import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import create_ema, ema_update

def test_ema_lerp_and_buffers():
    m = torch.nn.Linear(4, 4); m.register_buffer("b", torch.zeros(4))
    ema = create_ema(m)
    for p in ema.parameters(): assert not p.requires_grad
    with torch.no_grad():
        for p in m.parameters(): p.add_(1.0)
        m.b.add_(5.0)
    before = ema.weight.detach().clone()
    ema_update(ema, m, beta=0.9)
    assert torch.allclose(ema.weight, before*0.9 + m.weight*0.1, atol=1e-6)  # 0.9*ema + 0.1*model
    assert torch.equal(ema.b, m.b)   # buffers copied verbatim
