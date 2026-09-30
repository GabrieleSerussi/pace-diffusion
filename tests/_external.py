"""Availability checks for the external repositories and optional packages.

The test suite runs with only the pip dependencies installed.  Tests that need
facebookresearch/DiT, NVlabs/edm or an optional package carry a marker
(``external_dit``, ``external_edm``) or call one of the ``require_*`` helpers,
and are skipped with a clear reason when the dependency is missing.
"""

from __future__ import annotations

import importlib.util
import shutil
from functools import lru_cache

import pytest


@lru_cache(maxsize=None)
def dit_status() -> tuple[bool, str]:
    """Return ``(available, detail)`` for facebookresearch/DiT plus timm."""

    from pace.external_repos import ExternalRepositoryError, ensure_dit_importable

    try:
        root = ensure_dit_importable()
    except ExternalRepositoryError as exc:
        return False, str(exc)
    if importlib.util.find_spec("timm") is None:
        return False, "timm is not installed (pip install 'pace-diffusion[dit]')"
    return True, str(root)


@lru_cache(maxsize=None)
def edm_status() -> tuple[bool, str]:
    """Return ``(available, detail)`` for NVlabs/edm."""

    from pace.external_repos import ExternalRepositoryError, ensure_edm_importable

    try:
        root = ensure_edm_importable()
    except ExternalRepositoryError as exc:
        return False, str(exc)
    return True, "PYTHONPATH" if root is None else str(root)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=None)
def torchcodec_usable() -> bool:
    """torchaudio >= 2.9 decodes audio through torchcodec, which needs FFmpeg."""

    try:
        import torchcodec.decoders  # noqa: F401
    except Exception:  # ImportError, or missing FFmpeg shared libraries
        return False
    return True


requires_torchcodec = pytest.mark.skipif(
    not torchcodec_usable(),
    reason="torchaudio.load needs torchcodec and FFmpeg (pip install 'pace-diffusion[audio]')",
)


def bash_available() -> bool:
    return shutil.which("bash") is not None


def require_dit() -> None:
    """Skip the calling module when DiT or timm is unavailable."""

    available, detail = dit_status()
    if not available:
        pytest.skip(f"facebookresearch/DiT unavailable: {detail}", allow_module_level=True)


def require_edm() -> None:
    """Skip the calling module when NVlabs/edm is unavailable."""

    available, detail = edm_status()
    if not available:
        pytest.skip(f"NVlabs/edm unavailable: {detail}", allow_module_level=True)


def require_modules(*names: str) -> None:
    """Skip the calling module when any optional package is missing."""

    missing = [name for name in names if not module_available(name)]
    if missing:
        pytest.skip(f"optional package(s) not installed: {', '.join(missing)}", allow_module_level=True)
