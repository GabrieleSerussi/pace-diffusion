import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import phase_resume_action

def test_phase_resume_action(tmp_path):
    d = str(tmp_path)
    assert phase_resume_action(d) == "fresh"
    open(os.path.join(d, "resume.pt"), "w").close()
    assert phase_resume_action(d) == "resume"
    open(os.path.join(d, "student.pt"), "w").close()
    assert phase_resume_action(d) == "skip"   # completed phase takes precedence
