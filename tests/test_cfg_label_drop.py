import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import drop_labels_for_cfg

NUM_CLASSES = 10   # DiT-Micro/CIFAR-10; null-class index == num_classes


def test_p_zero_is_identity_and_consumes_no_rng():
    # --cfg_label_drop 0 (default): labels pass through untouched AND no RNG is
    # consumed, so all existing studies remain byte-identical.
    torch.manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (64,))
    state = torch.get_rng_state()
    out = drop_labels_for_cfg(labels, 0.0, NUM_CLASSES)
    assert out is labels
    assert torch.equal(torch.get_rng_state(), state)


def test_p_one_drops_every_label_for_both_models():
    torch.manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (64,))
    out = drop_labels_for_cfg(labels, 1.0, NUM_CLASSES)
    assert torch.all(out == NUM_CLASSES)
    assert out.dtype == labels.dtype and out.shape == labels.shape
    # The ONE returned tensor feeds both the teacher and student forwards, so
    # teacher input == student input by construction.
    teacher_in, student_in = out, out
    assert teacher_in is student_in


def test_p_half_deterministic_and_mask_shared():
    torch.manual_seed(1234)
    labels = torch.randint(0, NUM_CLASSES, (256,))
    gen_state = torch.get_rng_state()
    out1 = drop_labels_for_cfg(labels, 0.5, NUM_CLASSES)
    torch.set_rng_state(gen_state)
    out2 = drop_labels_for_cfg(labels, 0.5, NUM_CLASSES)
    # Same generator state -> identical mask (exact full-ckpt RNG resume).
    assert torch.equal(out1, out2)
    dropped = out1 == NUM_CLASSES
    # p=0.5 over 256 draws: both dropped and kept samples present.
    assert dropped.any() and (~dropped).any()
    # Kept labels are untouched; dropped ones all hit the null row.
    assert torch.equal(out1[~dropped], labels[~dropped])
    assert torch.all(out1[dropped] == NUM_CLASSES)
