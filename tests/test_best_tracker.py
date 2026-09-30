import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import BestTracker

def test_best_tracker():
    t = BestTracker(min_delta=0.01)
    assert t.update(1.0, 10) is True and t.best_step == 10
    assert t.update(0.995, 20) is False        # improvement < min_delta
    assert t.update(0.98, 30) is True and t.best_value == 0.98 and t.best_step == 30
    assert t.update(1.5, 40) is False and t.best_step == 30
