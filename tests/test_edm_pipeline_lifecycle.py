"""Bounded, offline coverage for the EDM profile-to-metric lifecycle."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pace.capacity_allocation import (
    CapacityAllocationConfig,
    ModelBudget,
    discover_evaluation_output,
)
from pace.dataset_specs import dataset_spec
import pace.edm_distillation as distillation
from pace.edm_distillation import (
    construct_student_from_kwargs,
    construct_student_from_plan,
    count_full_parameters,
    count_grouped_parameters,
    create_ema,
    default_model_kwargs,
    hybrid_distillation_loss,
    update_ema,
)
from pace.evaluation_protocols import (
    BENCHMARK_MANIFEST_FORMAT,
    LSUN_BEDROOM256_ADM_PROTOCOL,
    SampleQuantizer,
    benchmark_protocol_records,
    build_benchmark_manifest,
    quantize_samples,
    samples_to_layout,
    validate_adm_sample_npz,
)
from pace.image_datasets import discover_numeric_image_records
from pace.teacher_models import OPENAI_CONSISTENCY_STATE_DICT


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dry_run_capacity_allocation as allocation_cli
from evaluate_parameters_edm import build_profile_cost_estimate
from train_edm_distillation import (
    BestMetricTracker,
    build_checkpoint_selection,
    save_snapshot,
)


class _ZeroTeacher(torch.nn.Module):
    def forward(self, images, sigmas, class_labels=None, **_kwargs):
        del sigmas, class_labels
        return torch.zeros_like(images)


@pytest.mark.external_edm
def test_offline_unconditional_profile_to_quantized_metric_lifecycle(monkeypatch, tmp_path):
    """Exercise the persisted handoffs without a checkpoint download, GPU, or real dataset."""

    torch.manual_seed(7)
    topology = {
        "channel_mult": [1],
        "num_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
        "augment_dim": 0,
    }
    fixture_kwargs = default_model_kwargs(
        model_channels=8,
        label_dim=0,
        img_resolution=4,
        channel_mult=topology["channel_mult"],
        num_blocks=topology["num_blocks"],
        attn_resolutions=topology["attn_resolutions"],
        dropout=topology["dropout"],
        augment_dim=topology["augment_dim"],
    )
    count_model = construct_student_from_kwargs(fixture_kwargs)
    grouped_budget = count_grouped_parameters(count_model)
    full_budget = count_full_parameters(count_model)
    del count_model

    profile_dir = tmp_path / "profile"
    (profile_dir / "grouping").mkdir(parents=True)
    teacher_source = tmp_path / "fixture-teacher.pt"
    teacher_model_config = {
        "architecture": "offline_fixture",
        "model_family": "edm",
        "image_size": 4,
        "in_channels": 3,
        "out_channels": 3,
        "label_dim": 0,
        "model_channels": 8,
        "num_res_blocks": 1,
        "channel_mult": [1],
        "attention_resolutions": [],
        "dropout": 0.0,
        "augment_dim": 0,
        "sigma_data": 0.5,
        "student_topology": topology,
    }
    teacher = {
        "source": str(teacher_source),
        "format": OPENAI_CONSISTENCY_STATE_DICT,
        "model_config": teacher_model_config,
        "sampling": {"sampler": "heun", "num_steps": 2},
    }
    dataset_metadata = {
        "dataset": "lsun_bedroom",
        "protocol": dataset_spec("lsun_bedroom").protocol,
        "split": "monitor",
        "is_heldout": False,
        "count": 1,
        "entries_sha256": "a" * 64,
        "source_listing_sha256": "b" * 64,
        "source_records_sha256": "c" * 64,
        "dataset_spec": asdict(dataset_spec("lsun_bedroom")),
        "manifest_metadata": {
            "format": "diffdist_image_manifest_v1",
            "entries_sha256": "a" * 64,
            "source_records_sha256": "c" * 64,
        },
    }
    protocol_records = list(benchmark_protocol_records("lsun_bedroom"))
    profile = {
        "profile_format": "diffdist_edm_parameter_profile_fixture_v1",
        "profile_cost_estimate": build_profile_cost_estimate(
            grouping="per_filter",
            full_group_count=1,
            selected_group_count=1,
            bounded_by_max_groups=True,
            dataset_images=1,
            sigma_levels_per_image=2,
            world_size=1,
        ),
        "config": {
            "dataset": "lsun_bedroom",
            "network_pkl": str(teacher_source),
            "network_format": OPENAI_CONSISTENCY_STATE_DICT,
            "model_family": "edm",
            "image_size": 4,
        },
        "teacher": teacher,
        "dataset_info": dataset_metadata,
        "benchmark_protocol_ids": [record["identity"] for record in protocol_records],
        "benchmark_protocols": protocol_records,
        "model_info": {
            "label_dim": 0,
            "img_resolution": 4,
            "img_channels": 3,
            "model_family": "edm",
            "model_config": teacher_model_config,
            "teacher": teacher,
        },
        "group_names": ["fixture.group"],
        "group_param_counts": {"fixture.group": grouped_budget},
        "sigma_values": [0.1, 1.0],
        "sigma_bin_labels": ["low", "high"],
        "n_eff": [0.5, 1.0],
        "p_eff": [0.75, 1.0],
        "relative_delta_stack": [[0.25, 0.75]],
    }
    (profile_dir / "results.json").write_text(json.dumps(profile, indent=2) + "\n")
    (profile_dir / "grouping" / "timestep_grouping.json").write_text(
        json.dumps({"boundaries": [0, 2]}, indent=2) + "\n"
    )

    layout = discover_evaluation_output(profile_dir)
    allocation_config = CapacityAllocationConfig.from_mapping(
        {
            "student_variant": "blockwise_capacity",
            "eval_output_dir": str(profile_dir),
            "match_original_total_budget": True,
            "allocation_metric": "n_eff",
            "score_reduction": "mean",
        }
    )
    timestep_blocks, grouping_source = allocation_cli.resolve_timestep_blocks(
        allocation_config, tmp_path, layout
    )
    allocation_payload = allocation_cli.build_variant_payload(
        config=allocation_config,
        config_path=None,
        base_dir=tmp_path,
        layout=layout,
        timestep_blocks=timestep_blocks,
        grouping_source=grouping_source,
        original_model_budget=ModelBudget(parameters=grouped_budget),
        total_student_system_budget=float(grouped_budget),
    )
    allocation_cli.write_allocation_results(
        layout=layout,
        payloads={"blockwise_capacity": allocation_payload},
    )
    assert allocation_payload["score_sources"]["block_capacity_scores"]["metric"] == "n_eff"
    assert allocation_payload["target_budget_plan"]["block_budgets"] == [pytest.approx(grouped_budget)]

    monkeypatch.setattr(
        distillation,
        "build_uniform_candidate_table",
        lambda **_kwargs: [(8, grouped_budget, full_budget)],
    )
    architecture_summary = distillation.prepare_distillation_architectures(
        eval_output_dir=profile_dir,
        output_dir=tmp_path / "plans",
        variants=["blockwise_capacity"],
        budget_tolerance=0.0,
    )
    plan = architecture_summary["plans"]["blockwise_capacity"]
    assert plan["students"][0]["model_kwargs"]["label_dim"] == 0
    assert plan["students"][0]["model_kwargs"]["channel_mult"] == [1]
    assert plan["dataset_info"] == dataset_metadata
    assert plan["benchmark_protocol_ids"] == [LSUN_BEDROOM256_ADM_PROTOCOL.identity]

    student = construct_student_from_plan(plan)
    assert next(student.parameters()).device.type == "cpu"
    ema = create_ema(student)
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
    images = torch.linspace(-1.0, 1.0, 3 * 4 * 4).reshape(1, 3, 4, 4)
    sigmas = torch.tensor([0.5])
    loss, metrics = hybrid_distillation_loss(
        student=student,
        teacher=_ZeroTeacher(),
        images=images,
        labels=None,
        sigmas=sigmas,
        noise=torch.zeros_like(images),
        model_family="edm",
        sigma_data=0.5,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    update_ema(ema, student, beta=0.5)
    assert np.isfinite(metrics["loss"])

    training_dir = tmp_path / "training"
    row = {"step": 1, "val_loss": metrics["loss"], **metrics}
    final_snapshot = training_dir / "student-final.pt"
    save_snapshot(
        final_snapshot,
        step=1,
        student=student,
        ema=ema,
        plan=plan,
        stats=[row],
        optimizer=optimizer,
    )
    best_val = BestMetricTracker(metric_name="val_loss", snapshot_name="student-best-val.pt")
    assert best_val.update(row, output_dir=training_dir)
    best_snapshot = training_dir / best_val.snapshot_name
    save_snapshot(best_snapshot, step=1, student=student, ema=ema, plan=plan, stats=[row])
    selection = build_checkpoint_selection(
        stop_reason="bounded_fixture_complete",
        final_step=1,
        final_snapshot=final_snapshot,
        best_val=best_val,
        best_fid=BestMetricTracker(metric_name="fid", snapshot_name="student-best-fid.pt"),
        selection_policy="best_val",
    )
    assert selection["selected"] == {
        "kind": "best_val",
        "step": 1,
        "snapshot_path": str(best_snapshot),
    }
    assert selection["validation_split"]["is_heldout"] is False
    snapshot_payload = torch.load(best_snapshot, map_location="cpu", weights_only=False)
    assert snapshot_payload["snapshot_format"] == "diffdist_edm_state_dict_v2"
    assert snapshot_payload["architecture_plan"]["dataset_info"]["dataset_spec"] == dataset_metadata["dataset_spec"]

    with torch.inference_mode():
        generated = ema(images, sigmas, None)
    quantized_nchw = quantize_samples(generated, SampleQuantizer.OPENAI_TRUNCATE)
    quantized_nhwc = samples_to_layout(quantized_nchw, source="NCHW", destination="NHWC")
    sample_artifact = tmp_path / "samples.npz"
    np.savez(sample_artifact, arr_0=quantized_nhwc.cpu().numpy())
    tiny_adm_protocol = replace(LSUN_BEDROOM256_ADM_PROTOCOL, resolution=4, sample_count=1)
    artifact_schema = validate_adm_sample_npz(sample_artifact, tiny_adm_protocol)
    assert artifact_schema["count"] == 1
    assert artifact_schema["layout"] == "NHWC_uint8"

    benchmark_manifest = build_benchmark_manifest(
        tiny_adm_protocol,
        teacher=plan["teacher"],
        artifacts={
            "sample_batch": artifact_schema,
            "quantizer": SampleQuantizer.OPENAI_TRUNCATE.value,
            "checkpoint_selection": selection["selected"],
            "dataset_info": plan["dataset_info"],
        },
        runtime={"device": "cpu", "bounded_fixture": True},
        versions={"fixture_schema": "v1"},
    )
    assert benchmark_manifest["manifest_format"] == BENCHMARK_MANIFEST_FORMAT
    assert benchmark_manifest["artifacts"]["checkpoint_selection"]["snapshot_path"] == str(best_snapshot)
    assert benchmark_manifest["artifacts"]["dataset_info"]["dataset_spec"]["conditional"] is False
    assert len(benchmark_manifest["manifest_sha256"]) == 64
    assert profile["profile_cost_estimate"]["estimate_format"].endswith("_v1")


def test_numeric_zip_rejects_names_the_lazy_sequence_cannot_reopen(tmp_path):
    archive_path = tmp_path / "uppercase-extension.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("00000.PNG", b"fixture")

    with pytest.raises(ValueError, match="invalid/duplicate"):
        discover_numeric_image_records(
            archive_path,
            count=1,
            digits=5,
            suffix=".png",
            display_name="fixture",
        )
