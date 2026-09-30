"""Checks of the reproduction scripts in reproduce/ that need no GPU and no data.

Every script supports ``DRY_RUN=1``, which prints each command instead of
running it. The tests check the shell syntax, that each dry run succeeds, that
every flag passed to a repository script is accepted by that script's argument
parser, that the released inputs the scripts read by default exist, and that
the CPU stages of the CIFAR-10 script run on the released profile.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "reproduce"
SCRIPTS = sorted(path for path in REPRODUCE.glob("*.sh") if path.name != "common.sh")

pytestmark = pytest.mark.launcher


def _environment(**overrides: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key not in {"STAGES", "OUT"}}
    environment.update({"DRY_RUN": "1", "PYTHON": sys.executable})
    environment.update(overrides)
    return environment


def dry_run(script: Path, **overrides: str) -> list[str]:
    completed = subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=_environment(**overrides),
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [line[2:] for line in completed.stdout.splitlines() if line.startswith("+ ")]


def repository_invocations(commands: list[str]):
    """Yield (script, arguments) for every command that runs a script of this repository."""

    for command in commands:
        tokens = shlex.split(command)
        for index, token in enumerate(tokens):
            if token.startswith("scripts/") and token.endswith(".py"):
                yield token, tokens[index + 1:]
                break


def accepted_options(script: str) -> frozenset[str]:
    """Long options listed by ``script --help``."""

    completed = subprocess.run(
        [sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "COLUMNS": "200"}, check=False,
    )
    assert completed.returncode == 0, f"{script} --help failed:\n{completed.stderr}"
    return frozenset(re.findall(r"(?<![\w-])--[A-Za-z0-9][\w-]*", completed.stdout))


def _ids(path: Path) -> str:
    return path.name


def test_every_experiment_has_a_script() -> None:
    names = {path.name for path in SCRIPTS}
    assert {
        "cifar10_unet.sh", "imagenet64_unet.sh", "ffhq64_unet.sh", "lsun_bedroom256_unet.sh",
        "sc09_diffwave.sh", "figure1.sh", "dit_imagenet256.sh", "dit_ffhq256.sh", "dit_lsun256.sh",
        "dit_cifar10.sh", "dit_micro_cifar10.sh",
    } <= names
    assert (REPRODUCE / "cpu_phases_and_budgets.py").is_file()


@pytest.mark.parametrize("script", [*SCRIPTS, REPRODUCE / "common.sh"], ids=_ids)
def test_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=_ids)
def test_dry_run_prints_commands(script: Path) -> None:
    commands = dry_run(script)
    assert commands
    assert list(repository_invocations(commands)), "no repository script is invoked"


@pytest.mark.slow
def test_dry_run_flags_are_accepted() -> None:
    used: dict[str, set[str]] = {}
    for script in SCRIPTS:
        for path, arguments in repository_invocations(dry_run(script)):
            assert (ROOT / path).is_file(), f"{script.name}: {path}"
            flags = {argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")}
            used.setdefault(path, set()).update(flags)
    # One --help per repository script, in parallel (each one imports torch).
    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        options = dict(zip(used, pool.map(accepted_options, used)))
    rejected = {path: sorted(flags - options[path]) for path, flags in used.items() if flags - options[path]}
    assert not rejected, f"flags not accepted: {rejected}"


def test_stage_selection_limits_the_commands() -> None:
    commands = dry_run(REPRODUCE / "cifar10_unet.sh", STAGES="group allocate")
    scripts = [path for path, _ in repository_invocations(commands)]
    assert scripts == ["scripts/optimize_timestep_grouping.py", "scripts/dry_run_capacity_allocation.py"]


@pytest.mark.parametrize(
    "script, dataset",
    [
        ("cifar10_unet.sh", "cifar10"),
        ("imagenet64_unet.sh", "imagenet64"),
        ("ffhq64_unet.sh", "ffhq64"),
        ("lsun_bedroom256_unet.sh", "lsun256"),
    ],
)
def test_unet_training_reads_the_released_plans(script: str, dataset: str) -> None:
    commands = dry_run(REPRODUCE / script, STAGES="train")
    plans = []
    for path, arguments in repository_invocations(commands):
        assert path == "scripts/train_edm_distillation.py"
        plans.append(arguments[arguments.index("--architecture-plan") + 1])
    assert len(plans) == 4
    for plan in plans:
        assert plan.startswith(f"artifacts/plans/{dataset}/")
        assert (ROOT / plan).is_file(), plan


def test_released_inputs_named_in_the_scripts_exist() -> None:
    text = "\n".join(path.read_text() for path in SCRIPTS)
    referenced = set(re.findall(r"artifacts/[\w./-]+\.json(?:\.gz)?", text))
    referenced |= {f"reproduce/configs/{name}" for name in re.findall(r"reproduce/configs/([\w.-]+\.json)", text)}
    assert referenced
    missing = sorted(path for path in referenced if not (ROOT / path).is_file())
    assert not missing, missing


@pytest.mark.slow
def test_cifar10_cpu_stages_run_on_the_released_profile(tmp_path: Path) -> None:
    out = tmp_path / "cifar10_unet"
    environment = _environment(
        DRY_RUN="0", STAGES="group allocate", OUT=str(out),
        PROFILE_JSON="artifacts/profiles/cifar10_ddpmpp_random_same_norm.json.gz",
    )
    subprocess.run(["bash", str(REPRODUCE / "cifar10_unet.sh")], cwd=ROOT, env=environment,
                   check=True, capture_output=True, text=True)
    grouping = json.loads((out / "grouping" / "timestep_grouping.json").read_text())
    assert grouping["boundaries"] == [0, 8, 20]
    released = json.loads((ROOT / "artifacts/allocations/cifar10/combined_blockwise.json").read_text())
    allocation = json.loads((out / "allocations" / "combined_blockwise.json").read_text())
    assert allocation["target_budget_plan"]["block_budgets"] == released["target_budget_plan"]["block_budgets"]
