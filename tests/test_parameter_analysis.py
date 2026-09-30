import pytest
import torch

from pace.parameter_analysis import (
    AnalysisAxis,
    BinnedEvaluationBatch,
    DistributedAnalysisContext,
    ExactLevelPFIBatchSampler,
    FingerprintedBinnedCache,
    IndexedOutputTarget,
    PFIOutputHook,
    assert_distributed_hash_invariant,
    canonicalize_complete_group_results,
    canonical_json_sha256,
    compute_usage_metrics,
    dataset_population_fingerprint,
    discover_group_tensor_cache_shards,
    group_catalog_sha256,
    infer_bin_axis_metadata,
    init_distributed_analysis,
    load_binned_statistics_cache,
    load_group_tensor_cache_shards,
    make_output_hook,
    ranked_group_cache_metadata,
    run_binned_ablation_profile,
    run_rank_zero_analysis_operation,
    save_binned_statistics_cache,
    save_group_tensor_cache,
    shard_missing_group_items,
    singleton_analysis_context,
)


class _ExactLevelFixture:
    num_examples = 5
    level_indices = [9, 5, 0]

    def sample_index(self, example_index, level_position):
        return example_index * len(self.level_indices) + level_position


class _DistributedRunnerFixtureAdapter:
    def __init__(self):
        self.forward_calls = 0
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        self.model = torch.nn.Sequential(*self.layers)
        self.axis = AnalysisAxis(
            kind="fixture",
            ordering="descending",
            native_coordinate="level",
            normalized_coordinate="normalized_level",
            native_values=(1, 0),
            normalized_values=(1.0, 0.0),
            bin_labels=("1", "0"),
            bin_members=((1,), (0,)),
            metadata={},
        )

    def named_analysis_groups(self):
        return tuple((f"g{index}", layer) for index, layer in enumerate(self.layers))

    def predict_noise(self, x_t, levels):
        del levels
        return self.model(x_t)

    def unpack_analysis_batch(self, batch):
        values, positions = batch
        return BinnedEvaluationBatch(loss_inputs=(values,), level_positions=positions)

    def forward_losses_from_fixed_corruption(self, values):
        self.forward_calls += 1
        return self.model(values).reshape(-1)


def _distributed_runner_worker(rank, world_size, rendezvous_path, output_dir):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    context = DistributedAnalysisContext(
        torch.device("cpu"),
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        backend="gloo",
    )
    try:
        fingerprint = {"format": "distributed_fixture_v1", "seed": 0}
        cache = FingerprintedBinnedCache(
            baseline_path=output_dir / "baseline_pfi.pt",
            baseline_format="distributed_fixture_baseline_v1",
            groups_path=output_dir / f"checkpoint_pfi_rank{rank}.pt",
            groups_format="distributed_fixture_groups_v1",
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=canonical_json_sha256(fingerprint),
            group_shard_pattern=output_dir / "checkpoint_pfi_rank*.pt",
            require_rank_metadata=True,
        )
        profile = run_binned_ablation_profile(
            _DistributedRunnerFixtureAdapter(),
            [(torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1]))],
            num_bins=2,
            ablation_mode="zero",
            ablation_random_seed=0,
            cache=cache,
            distributed_context=context,
        )
        assert list(profile.ablated_means) == ["g0", "g1", "g2", "g3"]
        with pytest.raises(RuntimeError, match="fixture failure.*builtins.ValueError: broken"):
            run_rank_zero_analysis_operation(
                context,
                "fixture failure",
                lambda: (_ for _ in ()).throw(ValueError("broken")),
            )
    finally:
        torch.distributed.destroy_process_group()


def test_exact_level_pfi_is_reproducible_and_exact_level_conditioned():
    dataset = _ExactLevelFixture()
    first = ExactLevelPFIBatchSampler(
        dataset,
        batch_size=3,
        pfi_seed=17,
        population_fingerprint="fixture",
        level_kind="timestep",
    )
    second = ExactLevelPFIBatchSampler(
        dataset,
        batch_size=3,
        pfi_seed=17,
        population_fingerprint="fixture",
        level_kind="timestep",
    )

    assert first.artifact == second.artifact
    assert first.artifact["level_indices"] == [9, 5, 0]
    assert len(first) == 6
    for batch in first:
        assert len({reference.level_index for reference in batch}) == 1
        donors = [reference.donor_position for reference in batch]
        assert sorted(donors) == list(range(len(batch)))
        assert all(index != donor for index, donor in enumerate(donors))


def test_pfi_output_hook_exchanges_every_tensor_leaf_with_one_mapping():
    class Structured(torch.nn.Module):
        def forward(self, value):
            return value, {"skip": value + 10}

    module = Structured()
    values = torch.tensor([[1.0], [2.0], [3.0]])
    with PFIOutputHook(module) as hook:
        hook.set_pfi_permutation(torch.tensor([1, 2, 0]))
        residual, payload = module(values)

    assert residual.flatten().tolist() == [2.0, 3.0, 1.0]
    assert payload["skip"].flatten().tolist() == [12.0, 13.0, 11.0]


def test_tuple_pfi_hook_is_removed_and_normal_forward_is_restored():
    class TupleOutput(torch.nn.Module):
        def forward(self, value):
            return value + 1, value * 2

    module = TupleOutput()
    values = torch.tensor([[1.0], [2.0], [3.0]])
    expected = module(values)

    with PFIOutputHook(module) as hook:
        hook.set_pfi_permutation(torch.tensor([1, 2, 0]))
        ablated = module(values)

    assert not module._forward_hooks
    restored = module(values)
    assert not torch.equal(ablated[0], expected[0])
    for actual, reference in zip(restored, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_indexed_pfi_exchanges_only_selected_channel_and_restores_hook():
    module = torch.nn.Identity()
    values = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    original = values.clone()
    permutation = torch.tensor([1, 2, 0])
    target = IndexedOutputTarget(module, channel_index=2)

    with make_output_hook(target, "pfi") as hook:
        hook.set_pfi_permutation(permutation)
        replaced = module(values)

    torch.testing.assert_close(replaced[:, 2], values[permutation, 2], rtol=0, atol=0)
    torch.testing.assert_close(replaced[:, :2], values[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(replaced[:, 3:], values[:, 3:], rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert not module._forward_hooks
    assert module(values) is values


def test_indexed_zero_supports_explicit_negative_channel_dimension():
    module = torch.nn.Identity()
    values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    target = IndexedOutputTarget(module, channel_index=1, channel_dim=-1)

    with make_output_hook(target, "zero"):
        replaced = module(values)

    torch.testing.assert_close(replaced[..., 0], values[..., 0], rtol=0, atol=0)
    assert torch.count_nonzero(replaced[..., 1]) == 0
    torch.testing.assert_close(replaced[..., 2], values[..., 2], rtol=0, atol=0)


def test_indexed_random_replacement_preserves_selected_per_example_norm():
    module = torch.nn.Identity()
    values = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
    target = IndexedOutputTarget(module, channel_index=1)

    with make_output_hook(target, "random_same_norm", random_seed=17):
        first = module(values)
    with make_output_hook(target, "random_same_norm", random_seed=17):
        second = module(values)

    torch.testing.assert_close(first[:, 0], values[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(first[:, 2], values[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first[:, 1], dim=1),
        torch.linalg.vector_norm(values[:, 1], dim=1),
    )
    assert not torch.equal(first[:, 1], values[:, 1])


def test_indexed_hook_rejects_structured_output_and_is_removed():
    class TupleOutput(torch.nn.Module):
        def forward(self, value):
            return value, value + 1

    module = TupleOutput()
    target = IndexedOutputTarget(module, channel_index=0)
    with pytest.raises(TypeError, match="require a single tensor output"):
        with make_output_hook(target, "pfi") as hook:
            hook.set_pfi_permutation(torch.tensor([1, 0]))
            module(torch.ones(2, 2, 3))
    assert not module._forward_hooks


def test_indexed_target_and_runtime_bounds_are_validated():
    module = torch.nn.Identity()
    with pytest.raises(ValueError, match="batch dimension"):
        IndexedOutputTarget(module, channel_index=0, channel_dim=0)
    with pytest.raises(ValueError, match="non-negative"):
        IndexedOutputTarget(module, channel_index=-1)

    target = IndexedOutputTarget(module, channel_index=3)
    with pytest.raises(ValueError, match="out of range"):
        with make_output_hook(target, "zero"):
            module(torch.ones(2, 3, 4))
    assert not module._forward_hooks


def test_neff_is_canonical_and_positive_mass_gate_rejects_empty_bins():
    delta = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    relaxed = compute_usage_metrics(
        delta,
        baseline_mean=torch.ones(2),
        group_param_counts={"g0": 3, "g1": 5},
        group_names=["g0", "g1"],
    )
    assert relaxed["n_eff"][0].item() == pytest.approx(2.0)
    assert relaxed["n_eff"][1].item() == pytest.approx(1e12)

    with pytest.raises(ValueError, match="Positive ablation-delta mass is zero"):
        compute_usage_metrics(
            delta,
            baseline_mean=torch.ones(2),
            group_param_counts={"g0": 3, "g1": 5},
            group_names=["g0", "g1"],
            require_positive_delta_mass=True,
        )


def test_timestep_axis_selects_timestep_correlation_artifacts():
    metadata = infer_bin_axis_metadata(
        {"timestep_bin_labels": ["199-100", "99-0"]},
        num_bins=2,
    )
    assert metadata["correlation_key"] == "C_timesteps"
    assert metadata["correlation_plot_name"] == "timestep_correlation_heatmap.png"
    assert metadata["axis_title"] == "Timestep bin"


def test_population_fingerprint_includes_selected_content_hash():
    class Population:
        def __init__(self, content_hash):
            self.metadata = {
                "split": "validation",
                "selected_entries_sha256": "entries",
                "selected_content_sha256": content_hash,
            }

        def __len__(self):
            return 2

    assert dataset_population_fingerprint(Population("a" * 64)) != (
        dataset_population_fingerprint(Population("b" * 64))
    )


def test_shared_binned_runner_and_runtime_bound_cache(tmp_path):
    class Adapter:
        def __init__(self):
            self.model = torch.nn.Identity()
            self.axis = AnalysisAxis(
                kind="fixture",
                ordering="descending",
                native_coordinate="level",
                normalized_coordinate="normalized_level",
                native_values=(1, 0),
                normalized_values=(1.0, 0.0),
                bin_labels=("1", "0"),
                bin_members=((1,), (0,)),
                metadata={},
            )

        def named_analysis_groups(self):
            return (("identity", self.model),)

        def predict_noise(self, x_t, levels):
            del levels
            return self.model(x_t)

        def unpack_analysis_batch(self, batch):
            values, positions = batch
            return BinnedEvaluationBatch(
                loss_inputs=(values,),
                level_positions=positions,
            )

        def forward_losses_from_fixed_corruption(self, values):
            return self.model(values).reshape(-1)

    runtime_a = {"python": "3.11", "torch": "2.10", "device_type": "cpu"}
    fingerprint_a = {"format": "fixture", "runtime": {"identity": runtime_a}}
    fingerprint_a_sha = canonical_json_sha256(fingerprint_a)
    cache = FingerprintedBinnedCache(
        baseline_path=tmp_path / "baseline.pt",
        baseline_format="fixture_baseline_v1",
        groups_path=tmp_path / "groups.pt",
        groups_format="fixture_groups_v1",
        profile_fingerprint=fingerprint_a,
        profile_fingerprint_sha256=fingerprint_a_sha,
    )
    batches = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([0, 0])),
        (torch.tensor([[3.0], [4.0]]), torch.tensor([1, 1])),
    ]
    profile = run_binned_ablation_profile(
        Adapter(),
        batches,
        num_bins=2,
        ablation_mode="zero",
        ablation_random_seed=0,
        cache=cache,
    )
    assert profile.baseline_mean.tolist() == [1.5, 3.5]
    assert profile.baseline_count.tolist() == [2, 2]
    assert profile.ablated_means["identity"].tolist() == [0.0, 0.0]

    resumed_adapter = Adapter()
    resumed_adapter.forward_losses_from_fixed_corruption = lambda *args: pytest.fail(
        "a complete fingerprint-matched cache should skip model execution"
    )
    resumed = run_binned_ablation_profile(
        resumed_adapter,
        batches,
        num_bins=2,
        ablation_mode="zero",
        ablation_random_seed=0,
        cache=cache,
    )
    torch.testing.assert_close(resumed.baseline_mean, profile.baseline_mean)
    torch.testing.assert_close(
        resumed.ablated_means["identity"],
        profile.ablated_means["identity"],
    )

    fingerprint_b = {
        "format": "fixture",
        "runtime": {"identity": {**runtime_a, "torch": "2.11"}},
    }
    with pytest.raises(ValueError, match="different profile fingerprint"):
        load_binned_statistics_cache(
            cache.baseline_path,
            cache_format=cache.baseline_format,
            num_bins=2,
            profile_fingerprint=fingerprint_b,
            profile_fingerprint_sha256=canonical_json_sha256(fingerprint_b),
        )


def test_missing_group_sharding_uses_global_catalog_indices():
    group_items = tuple((f"g{index}", object()) for index in range(7))
    completed = {"g1", "g4"}

    rank_zero = shard_missing_group_items(group_items, completed, rank=0, world_size=2)
    rank_one = shard_missing_group_items(group_items, completed, rank=1, world_size=2)

    assert [(index, item[0]) for index, item in rank_zero] == [(0, "g0"), (3, "g3"), (6, "g6")]
    assert [(index, item[0]) for index, item in rank_one] == [(2, "g2"), (5, "g5")]
    assert {item[0] for _, item in (*rank_zero, *rank_one)} == {
        "g0",
        "g2",
        "g3",
        "g5",
        "g6",
    }
    with pytest.raises(ValueError, match="unknown names"):
        shard_missing_group_items(group_items, {"not-a-group"}, rank=0, world_size=1)


def test_ranked_group_cache_merge_is_canonical_and_rejects_conflicts(tmp_path):
    names = ["g0", "g1", "g2"]
    fingerprint = {"format": "fixture_profile_v1", "seed": 0}
    fingerprint_sha = canonical_json_sha256(fingerprint)
    catalog_sha = group_catalog_sha256(names)
    paths = [tmp_path / "checkpoint_pfi_rank0.pt", tmp_path / "checkpoint_pfi_rank1.pt"]
    contexts = [
        DistributedAnalysisContext(torch.device("cpu"), rank=0, local_rank=0, world_size=2),
        DistributedAnalysisContext(torch.device("cpu"), rank=1, local_rank=1, world_size=2),
    ]
    shard_values = [
        {"g0": torch.tensor([1.0, 2.0]), "g1": torch.tensor([3.0, 4.0])},
        {"g1": torch.tensor([3.0, 4.0]), "g2": torch.tensor([5.0, 6.0])},
    ]
    for path, context, values in zip(paths, contexts, shard_values, strict=True):
        save_group_tensor_cache(
            path,
            cache_format="fixture_group_shard_v1",
            tensor_key="groups",
            completed_key="completed_groups",
            values=values,
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=fingerprint_sha,
            extra_metadata=ranked_group_cache_metadata(
                context,
                group_catalog_digest=catalog_sha,
            ),
        )

    discovered = discover_group_tensor_cache_shards(tmp_path / "checkpoint_pfi_rank*.pt")
    merged = load_group_tensor_cache_shards(
        discovered,
        cache_format="fixture_group_shard_v1",
        tensor_key="groups",
        completed_key="completed_groups",
        group_names=names,
        num_bins=2,
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=fingerprint_sha,
        group_catalog_digest=catalog_sha,
    )
    assert list(merged.values) == names
    assert merged.duplicate_groups == ("g1",)
    assert merged.missing_groups == ()
    assert set(merged.shard_ranks.values()) == {0, 1}

    save_group_tensor_cache(
        paths[1],
        cache_format="fixture_group_shard_v1",
        tensor_key="groups",
        completed_key="completed_groups",
        values={"g1": torch.tensor([30.0, 40.0]), "g2": torch.tensor([5.0, 6.0])},
        num_bins=2,
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=fingerprint_sha,
        extra_metadata=ranked_group_cache_metadata(contexts[1], group_catalog_digest=catalog_sha),
    )
    with pytest.raises(ValueError, match="Conflicting duplicate cached group 'g1'"):
        load_group_tensor_cache_shards(
            discovered,
            cache_format="fixture_group_shard_v1",
            tensor_key="groups",
            completed_key="completed_groups",
            group_names=names,
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=fingerprint_sha,
            group_catalog_digest=catalog_sha,
        )


def test_ranked_group_cache_rejects_corrupt_shape_finiteness_catalog_and_rank(tmp_path):
    names = ["g0"]
    fingerprint = {"format": "strict_shard_fixture_v1"}
    fingerprint_sha = canonical_json_sha256(fingerprint)
    catalog_sha = group_catalog_sha256(names)
    path = tmp_path / "checkpoint_pfi_rank0.pt"
    save_group_tensor_cache(
        path,
        cache_format="strict_shard_groups_v1",
        tensor_key="groups",
        completed_key="completed_groups",
        values={"g0": torch.ones(2)},
        num_bins=2,
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=fingerprint_sha,
        extra_metadata=ranked_group_cache_metadata(
            singleton_analysis_context("cpu"),
            group_catalog_digest=catalog_sha,
        ),
    )
    valid = torch.load(path, map_location="cpu", weights_only=True)

    corruptions = [
        (lambda payload: payload["groups"].__setitem__("g0", torch.ones(3)), "Invalid cached"),
        (
            lambda payload: payload["groups"].__setitem__("g0", torch.tensor([1.0, float("nan")])),
            "Invalid cached",
        ),
        (lambda payload: payload.__setitem__("group_catalog_sha256", "f" * 64), "different group catalog"),
        (lambda payload: payload.__setitem__("writer_world_size", 0), "invalid rank metadata"),
    ]
    for mutate, expected_error in corruptions:
        payload = dict(valid)
        payload["groups"] = dict(valid["groups"])
        mutate(payload)
        torch.save(payload, path)
        with pytest.raises(ValueError, match=expected_error):
            load_group_tensor_cache_shards(
                [path],
                cache_format="strict_shard_groups_v1",
                tensor_key="groups",
                completed_key="completed_groups",
                group_names=names,
                num_bins=2,
                profile_fingerprint=fingerprint,
                profile_fingerprint_sha256=fingerprint_sha,
                group_catalog_digest=catalog_sha,
            )

    torch.save(valid, path)
    second_path = tmp_path / "checkpoint_pfi_duplicate_rank.pt"
    torch.save(valid, second_path)
    with pytest.raises(ValueError, match="Multiple group tensor cache shards claim rank 0"):
        load_group_tensor_cache_shards(
            [path, second_path],
            cache_format="strict_shard_groups_v1",
            tensor_key="groups",
            completed_key="completed_groups",
            group_names=names,
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=fingerprint_sha,
            group_catalog_digest=catalog_sha,
        )


def test_complete_group_results_and_singleton_hash_invariants_are_strict():
    context = singleton_analysis_context("cpu")
    digest = canonical_json_sha256({"teacher": "fixture"})
    assert assert_distributed_hash_invariant(context, "teacher", digest) == digest
    with pytest.raises(ValueError, match="not a SHA-256"):
        assert_distributed_hash_invariant(context, "teacher", "bad")
    with pytest.raises(RuntimeError, match="Missing ablation results"):
        canonicalize_complete_group_results({"g0": torch.ones(2)}, ["g0", "g1"], num_bins=2)
    with pytest.raises(ValueError, match="unknown groups"):
        canonicalize_complete_group_results({"g0": torch.ones(2)}, [], num_bins=2)


def test_distributed_initialization_validates_singleton_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    context = init_distributed_analysis("cpu")
    assert context == singleton_analysis_context("cpu")
    with pytest.raises(ValueError, match="timeout must be positive"):
        init_distributed_analysis("cpu", timeout_seconds=0)
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(ValueError, match="WORLD_SIZE must be positive"):
        init_distributed_analysis("cpu")


def test_shared_binned_runner_supports_strict_rank_shard_resume(tmp_path):
    class Adapter:
        def __init__(self):
            self.model = torch.nn.Identity()
            self.axis = AnalysisAxis(
                kind="fixture",
                ordering="descending",
                native_coordinate="level",
                normalized_coordinate="normalized_level",
                native_values=(1, 0),
                normalized_values=(1.0, 0.0),
                bin_labels=("1", "0"),
                bin_members=((1,), (0,)),
                metadata={},
            )

        def named_analysis_groups(self):
            return (("identity", self.model),)

        def predict_noise(self, x_t, levels):
            del levels
            return self.model(x_t)

        def unpack_analysis_batch(self, batch):
            values, positions = batch
            return BinnedEvaluationBatch(loss_inputs=(values,), level_positions=positions)

        def forward_losses_from_fixed_corruption(self, values):
            return self.model(values).reshape(-1)

    fingerprint = {"format": "ranked_fixture_v1", "seed": 0}
    cache = FingerprintedBinnedCache(
        baseline_path=tmp_path / "baseline_pfi.pt",
        baseline_format="ranked_fixture_baseline_v1",
        groups_path=tmp_path / "checkpoint_pfi_rank0.pt",
        groups_format="ranked_fixture_groups_v1",
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=canonical_json_sha256(fingerprint),
        group_shard_pattern=tmp_path / "checkpoint_pfi_rank*.pt",
        require_rank_metadata=True,
    )
    batches = [(torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1]))]
    profile = run_binned_ablation_profile(
        Adapter(),
        batches,
        num_bins=2,
        ablation_mode="zero",
        ablation_random_seed=0,
        cache=cache,
        distributed_context=singleton_analysis_context("cpu"),
    )
    assert profile.ablated_means["identity"].tolist() == [0.0, 0.0]
    shard = torch.load(cache.groups_path, map_location="cpu", weights_only=True)
    assert shard["rank"] == 0
    assert shard["writer_world_size"] == 1
    assert shard["group_catalog_sha256"] == group_catalog_sha256(["identity"])

    resumed = Adapter()
    resumed.forward_losses_from_fixed_corruption = lambda *args: pytest.fail(
        "a complete all-shard resume should skip model execution"
    )
    loaded = run_binned_ablation_profile(
        resumed,
        batches,
        num_bins=2,
        ablation_mode="zero",
        ablation_random_seed=0,
        cache=cache,
        distributed_context=singleton_analysis_context("cpu"),
    )
    torch.testing.assert_close(loaded.ablated_means["identity"], profile.ablated_means["identity"])


def test_shared_binned_runner_executes_two_rank_gloo_profile(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    fingerprint = {"format": "distributed_fixture_v1", "seed": 0}
    save_group_tensor_cache(
        tmp_path / "checkpoint_pfi_rank0.pt",
        cache_format="distributed_fixture_groups_v1",
        tensor_key="ablated_means",
        completed_key="completed_groups",
        values={"g0": torch.zeros(2)},
        num_bins=2,
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=canonical_json_sha256(fingerprint),
        extra_metadata=ranked_group_cache_metadata(
            singleton_analysis_context("cpu"),
            group_catalog_digest=group_catalog_sha256(["g0", "g1", "g2", "g3"]),
        ),
    )
    rendezvous_path = tmp_path / "gloo_init"
    torch.multiprocessing.spawn(
        _distributed_runner_worker,
        args=(2, rendezvous_path, tmp_path),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(
        tmp_path / "checkpoint_pfi_rank0.pt",
        map_location="cpu",
        weights_only=True,
    )
    rank_one = torch.load(
        tmp_path / "checkpoint_pfi_rank1.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert rank_zero["completed_groups"] == ["g0", "g1", "g3"]
    assert rank_one["completed_groups"] == ["g2"]
    assert rank_zero["writer_world_size"] == rank_one["writer_world_size"] == 2


def test_rank_shard_resume_can_change_from_two_workers_to_one(tmp_path):
    names = ["g0", "g1", "g2", "g3"]
    fingerprint = {"format": "world_size_resume_fixture_v1", "seed": 0}
    fingerprint_sha = canonical_json_sha256(fingerprint)
    catalog_sha = group_catalog_sha256(names)
    contexts = [
        DistributedAnalysisContext(torch.device("cpu"), rank=0, local_rank=0, world_size=2),
        DistributedAnalysisContext(torch.device("cpu"), rank=1, local_rank=1, world_size=2),
    ]
    for rank, name in enumerate(("g0", "g1")):
        save_group_tensor_cache(
            tmp_path / f"checkpoint_pfi_rank{rank}.pt",
            cache_format="world_size_resume_groups_v1",
            tensor_key="ablated_means",
            completed_key="completed_groups",
            values={name: torch.zeros(2)},
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=fingerprint_sha,
            extra_metadata=ranked_group_cache_metadata(
                contexts[rank],
                group_catalog_digest=catalog_sha,
            ),
        )
    save_binned_statistics_cache(
        tmp_path / "baseline_pfi.pt",
        cache_format="world_size_resume_baseline_v1",
        mean=torch.tensor([1.0, 2.0], dtype=torch.float64),
        stderr=torch.zeros(2, dtype=torch.float64),
        count=torch.ones(2, dtype=torch.long),
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=fingerprint_sha,
    )
    cache = FingerprintedBinnedCache(
        baseline_path=tmp_path / "baseline_pfi.pt",
        baseline_format="world_size_resume_baseline_v1",
        groups_path=tmp_path / "checkpoint_pfi_rank0.pt",
        groups_format="world_size_resume_groups_v1",
        profile_fingerprint=fingerprint,
        profile_fingerprint_sha256=fingerprint_sha,
        group_shard_pattern=tmp_path / "checkpoint_pfi_rank*.pt",
        require_rank_metadata=True,
    )
    adapter = _DistributedRunnerFixtureAdapter()
    profile = run_binned_ablation_profile(
        adapter,
        [(torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1]))],
        num_bins=2,
        ablation_mode="zero",
        ablation_random_seed=0,
        cache=cache,
        distributed_context=singleton_analysis_context("cpu"),
    )
    assert list(profile.ablated_means) == names
    assert adapter.forward_calls == 2
    rank_zero = torch.load(cache.groups_path, map_location="cpu", weights_only=True)
    rank_one = torch.load(
        tmp_path / "checkpoint_pfi_rank1.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert rank_zero["completed_groups"] == ["g0", "g2", "g3"]
    assert rank_zero["writer_world_size"] == 1
    assert rank_one["completed_groups"] == ["g1"]
    assert rank_one["writer_world_size"] == 2
