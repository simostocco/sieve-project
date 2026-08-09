from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explain.gradients import MAX_TORCH_SEED, IntegratedGradientsExplainer


class RecordingModel(nn.Module):
    def forward(
        self,
        variant_features,
        positions,
        gene_ids,
        mask,
        covariates=None,
        return_attention=False,
        return_intermediate=False,
        chrom_ids=None,
        content_features=None,
        absolute_position_features=None,
    ):
        source = content_features if content_features is not None else variant_features
        return source.sum(dim=(1, 2), keepdim=False).unsqueeze(-1), None


class FakeLoader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = [None] * sum(batch["positions"].shape[0] for batch in batches)

    def __iter__(self):
        yield from self.batches


def _rows(values, width):
    base = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
    offsets = torch.arange(width, dtype=torch.float32).reshape(1, width) / 10.0
    return base + offsets


def _make_batch(
    *,
    rows=8,
    batch_size=1,
    mask_values=None,
    include_features=True,
    include_splits=False,
):
    if mask_values is None:
        mask_values = [True] * rows
    mask = torch.tensor(mask_values, dtype=torch.bool).repeat(batch_size, 1)
    positions = torch.arange(1000, 1000 + rows, dtype=torch.long).repeat(batch_size, 1)
    gene_ids = torch.arange(10, 10 + rows, dtype=torch.long).repeat(batch_size, 1)
    chrom_ids = (torch.arange(rows, dtype=torch.long) % 3).repeat(batch_size, 1)
    batch = {
        "positions": positions,
        "gene_ids": gene_ids,
        "mask": mask,
        "chrom_ids": chrom_ids,
    }
    if include_features:
        batch["features"] = _rows(range(rows), 3).repeat(batch_size, 1, 1)
    if include_splits:
        batch["content_features"] = _rows(range(100, 100 + rows), 2).repeat(batch_size, 1, 1)
        batch["absolute_position_features"] = _rows(
            range(200, 200 + rows),
            1,
        ).repeat(batch_size, 1, 1)
    return batch


def _explainer(*, max_variants=3, sampling_seed=0, ig_mode="legacy"):
    explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=2,
        max_variants=max_variants,
        sampling_seed=sampling_seed,
        ig_mode=ig_mode,
    )

    def fake_attribute(
        variant_features,
        positions,
        gene_ids,
        mask,
        target=None,
        baseline=None,
        covariates=None,
        chrom_ids=None,
        *,
        content_features=None,
        absolute_position_features=None,
    ):
        if content_features is not None:
            return content_features.clone()
        return variant_features.clone()

    explainer.attribute = fake_attribute
    return explainer


def _run_batch(
    batch,
    *,
    max_variants=3,
    sampling_seed=0,
    ig_mode="legacy",
    aggregate="l2",
):
    explainer = _explainer(
        max_variants=max_variants,
        sampling_seed=sampling_seed,
        ig_mode=ig_mode,
    )
    return explainer.attribute_batch(FakeLoader([batch]), aggregate=aggregate)


def _selected(metadata):
    return metadata[0]["selected_variant_indices"]


def test_constructor_default_sampling_seed_is_zero():
    explainer = IntegratedGradientsExplainer(RecordingModel(), device="cpu", n_steps=2)

    assert explainer.sampling_seed == 0


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_constructor_accepts_supported_sampling_seeds(seed):
    explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=2,
        sampling_seed=seed,
    )

    assert explainer.sampling_seed == seed


@pytest.mark.parametrize("seed", [True, False, -1, 1.5, "1", MAX_TORCH_SEED + 1])
def test_constructor_rejects_invalid_sampling_seeds(seed):
    with pytest.raises(ValueError, match="sampling_seed"):
        IntegratedGradientsExplainer(
            RecordingModel(),
            device="cpu",
            n_steps=2,
            sampling_seed=seed,
        )


def test_same_seed_repeats_selected_indices_and_global_rng_calls_do_not_matter():
    batch = _make_batch(rows=20)

    torch.manual_seed(123)
    first = _run_batch(batch, sampling_seed=42)[2]
    _ = torch.rand(50)
    second = _run_batch(batch, sampling_seed=42)[2]

    np.testing.assert_array_equal(_selected(first), _selected(second))


def test_deterministic_sampling_does_not_advance_global_torch_rng_state():
    batch = _make_batch(rows=20)
    torch.manual_seed(999)
    before = torch.random.get_rng_state()

    _run_batch(batch, sampling_seed=42)

    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_two_fixed_seeds_are_repeatable_and_select_different_subsets():
    batch = _make_batch(rows=40)

    seed_a_first = _run_batch(batch, sampling_seed=2)[2]
    seed_a_second = _run_batch(batch, sampling_seed=2)[2]
    seed_b_first = _run_batch(batch, sampling_seed=33)[2]
    seed_b_second = _run_batch(batch, sampling_seed=33)[2]

    np.testing.assert_array_equal(_selected(seed_a_first), _selected(seed_a_second))
    np.testing.assert_array_equal(_selected(seed_b_first), _selected(seed_b_second))
    assert not np.array_equal(_selected(seed_a_first), _selected(seed_b_first))


def test_selected_indices_exclude_padding_are_sorted_int64_and_counted():
    batch = _make_batch(
        rows=8,
        mask_values=[False, True, True, False, True, True, False, True],
    )

    _, _, metadata = _run_batch(batch, max_variants=3, sampling_seed=5)
    indices = metadata[0]["selected_variant_indices"]

    assert indices.dtype == np.int64
    assert indices.tolist() == sorted(indices.tolist())
    assert all(batch["mask"][0, int(idx)].item() for idx in indices)
    assert len(indices) == metadata[0]["num_variants_analyzed"]


def test_no_truncation_persists_indices_and_metadata_without_sampling():
    mask_values = [False, True, True, False, True]
    batch = _make_batch(rows=5, mask_values=mask_values)

    _, scores, metadata = _run_batch(batch, max_variants=10, sampling_seed=9)

    assert metadata[0]["selected_variant_indices"].tolist() == [1, 2, 4]
    assert metadata[0]["num_variants_analyzed"] == 3
    assert metadata[0]["sampling_applied"] is False
    assert metadata[0]["truncated"] is False
    assert metadata[0]["effective_sampling_seed"] is None
    assert scores[0].shape == (3,)


def test_no_truncation_preserves_padded_tensor_execution_and_filters_outputs():
    batch = _make_batch(
        rows=5,
        mask_values=[False, True, True, False, True],
    )
    captured = {}
    explainer = _explainer(max_variants=10, sampling_seed=9)

    def fake_attribute(
        variant_features,
        positions,
        gene_ids,
        mask,
        target=None,
        baseline=None,
        covariates=None,
        chrom_ids=None,
        *,
        content_features=None,
        absolute_position_features=None,
    ):
        captured["features_shape"] = tuple(variant_features.shape)
        captured["mask"] = mask.clone()
        return variant_features.clone()

    explainer.attribute = fake_attribute
    attrs, scores, metadata = explainer.attribute_batch(FakeLoader([batch]))

    assert captured["features_shape"] == (1, 5, 3)
    assert torch.equal(
        captured["mask"],
        torch.tensor([[False, True, True, False, True]]),
    )
    assert metadata[0]["selected_variant_indices"].tolist() == [1, 2, 4]
    np.testing.assert_allclose(attrs[0], batch["features"][0, [1, 2, 4], :].numpy())
    np.testing.assert_allclose(scores[0], np.linalg.norm(attrs[0], ord=2, axis=1))
    assert metadata[0]["positions"].tolist() == batch["positions"][0, [1, 2, 4]].tolist()
    assert metadata[0]["gene_ids"].tolist() == batch["gene_ids"][0, [1, 2, 4]].tolist()


def test_deterministic_truncation_metadata_records_effective_seed():
    batch = _make_batch(rows=12)

    _, _, metadata = _run_batch(batch, max_variants=4, sampling_seed=17)

    assert metadata[0]["sampling_applied"] is True
    assert metadata[0]["truncated"] is True
    assert metadata[0]["sampling_seed"] == 17
    assert metadata[0]["effective_sampling_seed"] == 17
    assert metadata[0]["num_variants_original"] == 12
    assert metadata[0]["num_variants_analyzed"] == 4


def test_deterministic_sampling_requires_randperm_generator(monkeypatch):
    batch = _make_batch(rows=12)
    observed = {}
    original_randperm = torch.randperm

    def generator_required_randperm(*args, **kwargs):
        observed["generator"] = kwargs.get("generator")
        if observed["generator"] is None:
            raise AssertionError("deterministic sampling must pass a generator")
        return original_randperm(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", generator_required_randperm)

    _, _, metadata = _run_batch(batch, max_variants=4, sampling_seed=17)

    assert isinstance(observed["generator"], torch.Generator)
    assert metadata[0]["effective_sampling_seed"] == 17


def test_effective_seed_wraps_at_max_torch_seed():
    batches = [
        _make_batch(rows=2, mask_values=[True, True]),
        _make_batch(rows=2, mask_values=[True, True]),
        _make_batch(rows=8),
    ]
    explainer = _explainer(max_variants=3, sampling_seed=MAX_TORCH_SEED)

    _, _, metadata = explainer.attribute_batch(FakeLoader(batches))

    assert metadata[0]["effective_sampling_seed"] is None
    assert metadata[1]["effective_sampling_seed"] is None
    assert metadata[2]["effective_sampling_seed"] == 1


def test_dataloader_batch_size_changes_keep_global_sample_seed_assignment():
    two_at_once = _make_batch(rows=16, batch_size=2)
    one_at_a_time = [
        _make_batch(rows=16, batch_size=1),
        _make_batch(rows=16, batch_size=1),
    ]

    explainer_a = _explainer(max_variants=4, sampling_seed=12)
    _, _, metadata_a = explainer_a.attribute_batch(FakeLoader([two_at_once]))
    explainer_b = _explainer(max_variants=4, sampling_seed=12)
    _, _, metadata_b = explainer_b.attribute_batch(FakeLoader(one_at_a_time))

    for idx in range(2):
        np.testing.assert_array_equal(
            metadata_a[idx]["selected_variant_indices"],
            metadata_b[idx]["selected_variant_indices"],
        )
        assert metadata_a[idx]["effective_sampling_seed"] == 12 + idx
        assert metadata_b[idx]["effective_sampling_seed"] == 12 + idx


def test_legacy_and_content_modes_share_same_seed_selected_indices():
    legacy_batch = _make_batch(rows=18, include_features=True)
    content_batch = _make_batch(rows=18, include_features=False, include_splits=True)

    _, _, legacy_metadata = _run_batch(
        legacy_batch,
        max_variants=5,
        sampling_seed=21,
        ig_mode="legacy",
    )
    _, _, content_metadata = _run_batch(
        content_batch,
        max_variants=5,
        sampling_seed=21,
        ig_mode="content",
    )

    np.testing.assert_array_equal(
        legacy_metadata[0]["selected_variant_indices"],
        content_metadata[0]["selected_variant_indices"],
    )


@pytest.mark.parametrize(
    "ig_mode",
    ["legacy", "content"],
)
def test_selected_indices_apply_to_all_variant_tensors_and_returned_rows(ig_mode):
    batch = _make_batch(
        rows=10,
        include_features=ig_mode == "legacy",
        include_splits=ig_mode == "content",
    )

    attrs, scores, metadata = _run_batch(
        batch,
        max_variants=4,
        sampling_seed=3,
        ig_mode=ig_mode,
    )
    indices = metadata[0]["selected_variant_indices"]

    source = batch["features"] if ig_mode == "legacy" else batch["content_features"]
    np.testing.assert_allclose(attrs[0], source[0, indices, :].numpy())
    np.testing.assert_allclose(scores[0], np.linalg.norm(attrs[0], ord=2, axis=1))
    assert metadata[0]["positions"].tolist() == batch["positions"][0, indices].tolist()
    assert metadata[0]["gene_ids"].tolist() == batch["gene_ids"][0, indices].tolist()
    assert batch["mask"][0, indices].all()


def test_content_mode_selected_indices_apply_to_absolute_position_and_chromosomes():
    batch = _make_batch(rows=10, include_features=False, include_splits=True)
    captured = {}
    explainer = _explainer(max_variants=4, sampling_seed=3, ig_mode="content")

    def fake_attribute(
        variant_features,
        positions,
        gene_ids,
        mask,
        target=None,
        baseline=None,
        covariates=None,
        chrom_ids=None,
        *,
        content_features=None,
        absolute_position_features=None,
    ):
        captured["content_features"] = content_features.clone()
        captured["absolute_position_features"] = absolute_position_features.clone()
        captured["positions"] = positions.clone()
        captured["gene_ids"] = gene_ids.clone()
        captured["mask"] = mask.clone()
        captured["chrom_ids"] = chrom_ids.clone()
        return content_features.clone()

    explainer.attribute = fake_attribute
    _, _, metadata = explainer.attribute_batch(FakeLoader([batch]))
    indices = torch.tensor(metadata[0]["selected_variant_indices"], dtype=torch.long)

    assert torch.equal(captured["content_features"], batch["content_features"][:, indices, :])
    assert torch.equal(
        captured["absolute_position_features"],
        batch["absolute_position_features"][:, indices, :],
    )
    assert torch.equal(captured["positions"], batch["positions"][:, indices])
    assert torch.equal(captured["gene_ids"], batch["gene_ids"][:, indices])
    assert torch.equal(captured["mask"], batch["mask"][:, indices])
    assert torch.equal(captured["chrom_ids"], batch["chrom_ids"][:, indices])


@pytest.mark.parametrize("aggregate", ["l2", "l1", "sum", "mean"])
def test_aggregation_definitions_remain_unchanged(aggregate):
    batch = _make_batch(rows=3)
    attrs, scores, _ = _run_batch(batch, max_variants=10, aggregate=aggregate)

    if aggregate == "l2":
        expected = np.linalg.norm(attrs[0], ord=2, axis=1)
    elif aggregate == "l1":
        expected = np.linalg.norm(attrs[0], ord=1, axis=1)
    elif aggregate == "sum":
        expected = np.sum(attrs[0], axis=1)
    else:
        expected = np.mean(attrs[0], axis=1)
    np.testing.assert_allclose(scores[0], expected)


def test_attribute_batch_return_tuple_shape_remains_three_elements():
    result = _run_batch(_make_batch(rows=3), max_variants=10)

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_constructor_usage_without_sampling_seed_remains_valid():
    explainer = IntegratedGradientsExplainer(RecordingModel(), device="cpu", n_steps=2)

    assert explainer.sampling_seed == 0


def test_model_state_dict_is_unchanged_by_sampling_configuration():
    model = RecordingModel()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    IntegratedGradientsExplainer(model, device="cpu", n_steps=2, sampling_seed=4)

    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, value in before.items():
        torch.testing.assert_close(value, after[key], rtol=0, atol=0)


def test_scripts_explain_remains_free_of_attribute_batch_production_call():
    source = Path("scripts/explain.py").read_text()

    assert "attribute_batch(" not in source


def test_sampling_seed_none_is_valid_and_persists_selected_indices_metadata():
    batch = _make_batch(rows=10)

    _, _, metadata = _run_batch(batch, max_variants=4, sampling_seed=None)

    assert metadata[0]["sampling_seed"] is None
    assert metadata[0]["effective_sampling_seed"] is None
    assert metadata[0]["selected_variant_indices"].dtype == np.int64
    assert len(metadata[0]["selected_variant_indices"]) == 4
