"""Shared pytest configuration for the PACE test suite.

Markers (also registered in ``pyproject.toml``):

``external_dit``
    needs facebookresearch/DiT (``DIT_REPO=/path/to/DiT``) and timm;
``external_edm``
    needs NVlabs/edm (``EDM_REPO=/path/to/edm``, a sibling ``../edm`` checkout,
    or ``PYTHONPATH``);
``launcher``
    runs or parses the shell scripts under ``reproduce/`` (needs bash);
``slow``
    takes tens of seconds on a laptop CPU (deselect with ``-m "not slow"``).

Tests marked ``external_dit``, ``external_edm`` or ``launcher`` are skipped
with an explicit reason when the dependency is missing, so ``pytest`` passes
on a machine that has only the pip dependencies installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "paper"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tests._external import bash_available, dit_status, edm_status  # noqa: E402

MARKERS = {
    "external_dit": "needs facebookresearch/DiT (set DIT_REPO) and timm",
    "external_edm": "needs NVlabs/edm (set EDM_REPO, or use ../edm or PYTHONPATH)",
    "launcher": "runs or parses the shell scripts under reproduce/ (needs bash)",
    "slow": "takes tens of seconds on a laptop CPU",
}


def pytest_configure(config: pytest.Config) -> None:
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_report_header(config: pytest.Config) -> list[str]:
    dit_ok, dit_detail = dit_status()
    edm_ok, edm_detail = edm_status()
    return [
        f"facebookresearch/DiT: {'available at ' + dit_detail if dit_ok else 'unavailable (external_dit tests skip)'}",
        f"NVlabs/edm: {'available via ' + edm_detail if edm_ok else 'unavailable (external_edm tests skip)'}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    dit_ok, dit_detail = dit_status()
    edm_ok, edm_detail = edm_status()
    have_bash = bash_available()
    for item in items:
        if "external_dit" in item.keywords and not dit_ok:
            item.add_marker(pytest.mark.skip(reason=f"facebookresearch/DiT unavailable: {dit_detail}"))
        if "external_edm" in item.keywords and not edm_ok:
            item.add_marker(pytest.mark.skip(reason=f"NVlabs/edm unavailable: {edm_detail}"))
        if "launcher" in item.keywords and not have_bash:
            item.add_marker(pytest.mark.skip(reason="bash is not available"))
