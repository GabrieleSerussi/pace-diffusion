"""Focused offline tests for the FFHQ LANCZOS preparation command."""

from __future__ import annotations

import os
import sys

from PIL import Image
import pytest

from scripts.data.prepare_ffhq_dataset import build_arg_parser, resize_ffhq_image


def test_resize_ffhq_image_uses_lanczos_exactly():
    image = Image.new("RGB", (256, 256))
    pixels = image.load()
    for y in range(256):
        for x in range(256):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    actual = resize_ffhq_image(image, 64)
    expected = image.resize((64, 64), resample=Image.Resampling.LANCZOS)
    assert actual.mode == "RGB"
    assert actual.size == (64, 64)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("mode", "size", "message"),
    [("L", (256, 256), "must be RGB"), ("RGB", (255, 256), "exactly 256x256")],
)
def test_resize_ffhq_image_rejects_noncanonical_source(mode, size, message):
    with pytest.raises(ValueError, match=message):
        resize_ffhq_image(Image.new(mode, size), 64)


def test_prepare_cli_defaults_to_teacher_64_lanczos_protocol():
    args = build_arg_parser().parse_args(["--source", "/source", "--output-dir", "/output"])
    assert args.resolution == 64
    assert args.workers == 4
    assert not args.overwrite

