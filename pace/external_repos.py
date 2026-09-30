"""Locate the external research repositories that PACE imports at runtime.

PACE builds on two public repositories that are not vendored here because of
their licences (see ``THIRD_PARTY_NOTICES.md``):

* ``NVlabs/edm`` (CC BY-NC-SA 4.0) provides the EDM teacher pickles
  (``torch_utils``, ``dnnlib``) and the layers of every U-Net student
  (``training.networks``).
* ``facebookresearch/DiT`` (CC BY-NC 4.0) provides the DiT building blocks
  (``models``) and the DDPM sampler (``diffusion``).  Its ``models.py`` needs
  ``timm``.

Importing :mod:`pace` never imports either repository.  Code that needs one of
them calls :func:`ensure_edm_importable` or :func:`ensure_dit_importable` first.

Resolution order for DiT: an explicit path (the ``--dit_repo`` flag of the DiT
scripts), then ``$DIT_REPO``.  There is no implicit fallback, because DiT's
generic top-level module names (``models``, ``diffusion``) can collide with
unrelated packages.

Resolution order for EDM: an explicit path, ``$EDM_REPO``, a sibling ``edm``
folder next to this checkout, ``../edm`` relative to the working directory,
and finally a checkout that is already importable (for example through
``PYTHONPATH``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

DIT_REPO_ENV = "DIT_REPO"
EDM_REPO_ENV = "EDM_REPO"

DIT_REPO_URL = "https://github.com/facebookresearch/DiT"
EDM_REPO_URL = "https://github.com/NVlabs/edm"

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ExternalRepositoryError(ModuleNotFoundError):
    """Raised when a required external repository cannot be located."""


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


# ---------------------------------------------------------------------------
# NVlabs/edm
# ---------------------------------------------------------------------------


def is_edm_checkout(path: str | Path) -> bool:
    """Return whether ``path`` looks like an NVlabs/edm checkout."""

    return (Path(path) / "training" / "networks.py").is_file()


def edm_repo_candidates(edm_root: str | Path | None = None) -> list[Path]:
    """Return the EDM checkout locations searched, in priority order."""

    candidates: list[Path] = []
    if edm_root is not None:
        candidates.append(Path(edm_root).expanduser())
    env_value = os.environ.get(EDM_REPO_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(_REPO_ROOT.parent / "edm")
    candidates.append(Path.cwd().parent / "edm")
    return candidates


def default_edm_root() -> Path:
    """Return the EDM checkout used for NVLabs ``fid.py`` when none is given."""

    env_value = os.environ.get(EDM_REPO_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _REPO_ROOT.parent / "edm"


def _edm_already_importable() -> bool:
    try:
        return importlib.util.find_spec("training.networks") is not None
    except (ImportError, ValueError):
        return False


def ensure_edm_importable(edm_root: str | Path | None = None) -> Path | None:
    """Put an NVlabs/edm checkout on ``sys.path``.

    Returns the checkout that was added, or ``None`` when EDM was already
    importable through ``PYTHONPATH``.  Raises :class:`ExternalRepositoryError`
    with setup instructions when no checkout can be found.
    """

    for candidate in edm_repo_candidates(edm_root):
        if is_edm_checkout(candidate):
            _prepend_sys_path(candidate)
            return candidate.resolve()
    if _edm_already_importable():
        return None
    searched = ", ".join(str(path) for path in edm_repo_candidates(edm_root))
    raise ExternalRepositoryError(
        "This operation needs the NVlabs/edm repository (training.networks, torch_utils, dnnlib), "
        f"which is not vendored. Clone {EDM_REPO_URL} and set {EDM_REPO_ENV}=/path/to/edm "
        "(or place the checkout next to this repository as ../edm, or add it to PYTHONPATH). "
        f"Searched: {searched}."
    )


# ---------------------------------------------------------------------------
# facebookresearch/DiT
# ---------------------------------------------------------------------------


def is_dit_checkout(path: str | Path) -> bool:
    """Return whether ``path`` looks like a facebookresearch/DiT checkout."""

    root = Path(path)
    return (root / "models.py").is_file() and (root / "diffusion" / "__init__.py").is_file()


def configure_dit_repo(dit_repo: str | Path | None) -> Path | None:
    """Record an explicit DiT checkout (the ``--dit_repo`` flag).

    The location is exported as ``$DIT_REPO`` so that subprocesses launched by
    the evaluation scripts resolve the same checkout.  ``None`` leaves the
    environment unchanged.
    """

    if dit_repo is None or str(dit_repo) == "":
        return None
    root = Path(dit_repo).expanduser().resolve()
    if not is_dit_checkout(root):
        raise ExternalRepositoryError(
            f"--dit_repo {root} is not a facebookresearch/DiT checkout "
            "(expected models.py and diffusion/__init__.py)."
        )
    os.environ[DIT_REPO_ENV] = str(root)
    return root


def resolve_dit_repo(dit_repo: str | Path | None = None) -> Path:
    """Return the DiT checkout from ``dit_repo`` or ``$DIT_REPO``."""

    raw = dit_repo if dit_repo is not None and str(dit_repo) != "" else os.environ.get(DIT_REPO_ENV)
    if not raw:
        raise ExternalRepositoryError(
            "This operation needs the facebookresearch/DiT repository (models.py, diffusion/), "
            f"which is not vendored. Clone {DIT_REPO_URL} and set {DIT_REPO_ENV}=/path/to/DiT "
            "or pass --dit_repo /path/to/DiT. DiT's models.py also needs timm "
            "(pip install 'pace-diffusion[dit]')."
        )
    root = Path(raw).expanduser().resolve()
    if not is_dit_checkout(root):
        raise ExternalRepositoryError(
            f"{DIT_REPO_ENV}={root} is not a facebookresearch/DiT checkout "
            "(expected models.py and diffusion/__init__.py)."
        )
    return root


def ensure_dit_importable(dit_repo: str | Path | None = None) -> Path:
    """Put the DiT checkout on ``sys.path`` and return its location."""

    root = resolve_dit_repo(dit_repo)
    _prepend_sys_path(root)
    return root


def import_dit_module(name: str, dit_repo: str | Path | None = None) -> ModuleType:
    """Import a top-level DiT module (``models``, ``diffusion``, ``download``)."""

    root = ensure_dit_importable(dit_repo)
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == "timm":
            raise ExternalRepositoryError(
                "facebookresearch/DiT's models.py needs timm: pip install 'pace-diffusion[dit]'"
            ) from exc
        raise
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            Path(module_file).resolve().relative_to(root)
        except ValueError as exc:
            raise ExternalRepositoryError(
                f"Imported {name!r} from {module_file}, not from the DiT checkout {root}. "
                "Another installed package shadows DiT's top-level module name."
            ) from exc
    return module
