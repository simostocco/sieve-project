import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.encoding.position_config import ResolvedIGMode
from src.explain.gradients import (
    ContentSIEVEWrapper,
    IntegratedGradientsExplainer,
    SIEVEWrapper,
)
from src.explain.ig_mode import (
    IGModeCompatibilityWarning,
    RequestedIGMode,
    resolve_ig_mode,
)
from src.models.sieve import SIEVE

RTOL = 1e-6
ATOL = 1e-7


def _new_config(default: str = "content"):
    return {"position_encoding": {"attribution": {"default_ig_mode": default}}}


def _make_sieve(
    input_dim: int,
    *,
    num_chromosomes: int = 0,
    num_covariates: int = 0,
) -> SIEVE:
    torch.manual_seed(1000 + input_dim + num_chromosomes + num_covariates)
    model = SIEVE(
        input_dim=input_dim,
        num_genes=4,
        latent_dim=8,
        hidden_dim=12,
        num_heads=2,
        num_attention_layers=1,
        classifier_hidden_dim=10,
        dropout=0.0,
        num_chromosomes=num_chromosomes,
        num_covariates=num_covariates,
    )
    model.eval()
    return model


def _split_from_historical(features: torch.Tensor, content_dim: int):
    if features.shape[-1] == content_dim:
        return features.clone(), features[..., :0].clone()
    return (
        torch.cat([features[..., :1], features[..., 65:]], dim=-1),
        features[..., 1:65].clone(),
    )


def _make_inputs(input_dim: int, content_dim: int, *, variants: int = 4):
    torch.manual_seed(input_dim + content_dim + variants)
    features = torch.randn(1, variants, input_dim)
    content, absolute = _split_from_historical(features, content_dim)
    positions = torch.arange(100, 100 + variants).unsqueeze(0)
    gene_ids = (torch.arange(variants) % 4).unsqueeze(0)
    mask = torch.ones(1, variants, dtype=torch.bool)
    chrom_ids = (torch.arange(variants) % 2).unsqueeze(0)
    return features, content, absolute, positions, gene_ids, mask, chrom_ids


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

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
        self.calls.append(
            {
                "variant_features": variant_features,
                "content_features": content_features,
                "absolute_position_features": absolute_position_features,
                "positions": positions,
                "gene_ids": gene_ids,
                "mask": mask,
                "covariates": covariates,
                "chrom_ids": chrom_ids,
            }
        )
        source = content_features if content_features is not None else variant_features
        logits = source.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        if covariates is not None:
            logits = logits + covariates.sum(dim=1, keepdim=True)
        return logits, None


def _fake_loader(batch):
    class FakeLoader:
        dataset = [None] * batch["positions"].shape[0]

        def __iter__(self):
            yield batch

    return FakeLoader()


@pytest.mark.parametrize(
    ("requested", "saved", "expected"),
    [
        (RequestedIGMode.AUTO, "content", ResolvedIGMode.CONTENT),
        ("auto", "legacy", ResolvedIGMode.LEGACY),
        ("content", "legacy", ResolvedIGMode.CONTENT),
        (RequestedIGMode.LEGACY, "content", ResolvedIGMode.LEGACY),
    ],
)
def test_resolve_ig_mode_new_schema(requested, saved, expected):
    assert resolve_ig_mode(requested, config=_new_config(saved)) is expected


def test_resolve_ig_mode_old_config_warnings():
    with pytest.warns(IGModeCompatibilityWarning, match="historical legacy"):
        assert resolve_ig_mode("auto", config={}) is ResolvedIGMode.LEGACY

    with pytest.warns(IGModeCompatibilityWarning, match="content-only attribution override"):
        assert resolve_ig_mode("content", config={}) is ResolvedIGMode.CONTENT

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_ig_mode("legacy", config={}) is ResolvedIGMode.LEGACY
    assert not caught


@pytest.mark.parametrize("requested", ["bogus", object()])
def test_resolve_ig_mode_rejects_unsupported_request(requested):
    with pytest.raises(ValueError, match="requested_mode"):
        resolve_ig_mode(requested, config={})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"position_encoding": "bad"}, "position_encoding must be a mapping"),
        ({"position_encoding": {}}, "attribution is required"),
        ({"position_encoding": {"attribution": "bad"}}, "attribution must be a mapping"),
        (
            {"position_encoding": {"attribution": {}}},
            "default_ig_mode is required",
        ),
        (
            {"position_encoding": {"attribution": {"default_ig_mode": "auto"}}},
            "must be 'content' or 'legacy'",
        ),
    ],
)
def test_resolve_ig_mode_validates_new_schema_metadata(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_ig_mode("content", config=config)


def test_constructor_default_and_resolved_modes_select_wrappers():
    model = RecordingModel()

    default_explainer = IntegratedGradientsExplainer(model, device="cpu", n_steps=2)
    assert default_explainer.ig_mode is ResolvedIGMode.LEGACY
    assert isinstance(default_explainer.model_wrapper, SIEVEWrapper)

    legacy_explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=2,
        ig_mode="legacy",
    )
    assert isinstance(legacy_explainer.model_wrapper, SIEVEWrapper)

    content_explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=2,
        ig_mode=ResolvedIGMode.CONTENT,
    )
    assert isinstance(content_explainer.model_wrapper, ContentSIEVEWrapper)


@pytest.mark.parametrize("mode", ["auto", "bogus"])
def test_constructor_rejects_unresolved_or_unsupported_modes(mode):
    with pytest.raises(ValueError, match="ig_mode"):
        IntegratedGradientsExplainer(RecordingModel(), device="cpu", n_steps=2, ig_mode=mode)


def test_legacy_attribute_call_shape_and_split_rejection():
    model = RecordingModel()
    features, content, absolute, positions, gene_ids, mask, _ = _make_inputs(65, 1)
    explainer = IntegratedGradientsExplainer(model, device="cpu", n_steps=3)

    attrs = explainer.attribute(features, positions, gene_ids, mask)

    assert attrs.shape == features.shape
    with pytest.raises(ValueError, match="only valid in content IG mode"):
        explainer.attribute(
            features,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute,
        )
    with pytest.raises(ValueError, match="variant_features is required"):
        explainer.attribute(None, positions, gene_ids, mask)


def test_legacy_covariates_and_chromosomes_remain_valid():
    model = RecordingModel()
    features, _, _, positions, gene_ids, mask, chrom_ids = _make_inputs(69, 5)
    covariates = torch.tensor([[1.0]])
    explainer = IntegratedGradientsExplainer(model, device="cpu", n_steps=3)

    attrs = explainer.attribute(
        features,
        positions,
        gene_ids,
        mask,
        covariates=covariates,
        chrom_ids=chrom_ids,
    )

    assert attrs.shape == features.shape


def test_content_wrapper_matches_direct_split_primary_call_and_fixed_args():
    model = _make_sieve(69, num_chromosomes=2, num_covariates=1)
    features, content, absolute, positions, gene_ids, mask, chrom_ids = _make_inputs(69, 5)
    covariates = torch.tensor([[0.5]])
    wrapper = ContentSIEVEWrapper(model)

    wrapped = wrapper(
        content,
        absolute,
        positions,
        gene_ids,
        mask,
        covariates=covariates,
        chrom_ids=chrom_ids,
    )
    direct, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        covariates=covariates,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )

    torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)
    historical, _ = model(
        features, positions, gene_ids, mask, covariates=covariates, chrom_ids=chrom_ids
    )
    torch.testing.assert_close(wrapped, historical, rtol=RTOL, atol=ATOL)


def test_content_wrapper_passes_none_variant_features_and_observed_absolute_position():
    model = RecordingModel()
    content = torch.tensor([[[1.0, 2.0]]])
    absolute = torch.tensor([[[3.0, 4.0]]])
    positions = torch.tensor([[10]])
    gene_ids = torch.tensor([[1]])
    mask = torch.tensor([[True]])
    covariates = torch.tensor([[0.5]])
    chrom_ids = torch.tensor([[2]])

    ContentSIEVEWrapper(model)(
        content,
        absolute,
        positions,
        gene_ids,
        mask,
        covariates=covariates,
        chrom_ids=chrom_ids,
    )

    call = model.calls[-1]
    assert call["variant_features"] is None
    assert torch.equal(call["absolute_position_features"], absolute)
    assert torch.equal(call["positions"], positions)
    assert torch.equal(call["gene_ids"], gene_ids)
    assert torch.equal(call["mask"], mask)
    assert torch.equal(call["covariates"], covariates)
    assert torch.equal(call["chrom_ids"], chrom_ids)


def test_content_mode_requires_content_boundary_arguments():
    model = RecordingModel()
    features, content, absolute, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=3,
        ig_mode=ResolvedIGMode.CONTENT,
    )

    with pytest.raises(ValueError, match="variant_features must be None"):
        explainer.attribute(
            features,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute,
        )
    with pytest.raises(ValueError, match="required"):
        explainer.attribute(None, positions, gene_ids, mask)
    with pytest.raises(ValueError, match="supplied together"):
        explainer.attribute(None, positions, gene_ids, mask, content_features=content)


@pytest.mark.parametrize(
    ("input_dim", "content_dim"),
    [(1, 1), (65, 1), (69, 5), (71, 7), (71, 7)],
)
def test_content_attribution_widths_cover_l0_to_l4(input_dim, content_dim):
    model = RecordingModel()
    _, content, absolute, positions, gene_ids, mask, _ = _make_inputs(input_dim, content_dim)
    absolute = absolute.clone().detach().requires_grad_(True)
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=3,
        ig_mode="content",
    )

    attrs = explainer.attribute(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=absolute,
    )

    assert attrs.shape == content.shape
    assert absolute.grad is None


def test_content_default_baseline_and_completeness_keep_absolute_position_fixed():
    model = RecordingModel()
    _, content, absolute, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=8,
        ig_mode=ResolvedIGMode.CONTENT,
    )
    wrapper = ContentSIEVEWrapper(model)

    attrs = explainer.attribute(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=absolute,
    )

    with torch.no_grad():
        observed = wrapper(content, absolute, positions, gene_ids, mask)
        baseline = wrapper(torch.zeros_like(content), absolute, positions, gene_ids, mask)

    torch.testing.assert_close(
        attrs.sum(),
        (observed - baseline).sum(),
        rtol=5e-3,
        atol=5e-3,
    )


def test_content_mode_captum_calls_keep_absolute_position_observed():
    model = RecordingModel()
    content = torch.tensor([[[1.0], [2.0]]])
    absolute = torch.tensor([[[5.0], [6.0]]], requires_grad=True)
    positions = torch.tensor([[10, 20]])
    gene_ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=4,
        ig_mode="content",
    )

    explainer.attribute(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=absolute,
    )

    assert model.calls
    for call in model.calls:
        expected_absolute = absolute.detach().expand_as(call["absolute_position_features"])
        expected_positions = positions.expand_as(call["positions"])
        expected_gene_ids = gene_ids.expand_as(call["gene_ids"])
        expected_mask = mask.expand_as(call["mask"])
        assert torch.equal(call["absolute_position_features"], expected_absolute)
        assert call["absolute_position_features"].requires_grad is False
        assert torch.equal(call["positions"], expected_positions)
        assert torch.equal(call["gene_ids"], expected_gene_ids)
        assert torch.equal(call["mask"], expected_mask)


def test_l0_content_and_legacy_attributions_match():
    model = RecordingModel()
    features, content, absolute, positions, gene_ids, mask, _ = _make_inputs(1, 1)
    legacy = IntegratedGradientsExplainer(model, device="cpu", n_steps=16)
    content_explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=16,
        ig_mode="content",
    )

    legacy_attr = legacy.attribute(features, positions, gene_ids, mask)
    content_attr = content_explainer.attribute(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=absolute,
    )

    torch.testing.assert_close(content_attr, legacy_attr, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mode", ["legacy", "content"])
def test_user_supplied_baseline_validation(mode):
    model = RecordingModel()
    features, content, absolute, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    explainer = IntegratedGradientsExplainer(model, device="cpu", n_steps=3, ig_mode=mode)
    valid_input = features if mode == "legacy" else content
    valid_baseline = torch.zeros_like(valid_input)

    kwargs = {}
    variant_features = features
    if mode == "content":
        variant_features = None
        kwargs = {
            "content_features": content,
            "absolute_position_features": absolute,
        }

    attrs = explainer.attribute(
        variant_features,
        positions,
        gene_ids,
        mask,
        baseline=valid_baseline,
        **kwargs,
    )
    assert attrs.shape == valid_input.shape

    with pytest.raises(ValueError, match="torch.Tensor"):
        explainer.attribute(
            variant_features,
            positions,
            gene_ids,
            mask,
            baseline=object(),
            **kwargs,
        )
    with pytest.raises(ValueError, match="shape"):
        explainer.attribute(
            variant_features,
            positions,
            gene_ids,
            mask,
            baseline=torch.zeros_like(valid_input[..., :1]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="dtype"):
        explainer.attribute(
            variant_features,
            positions,
            gene_ids,
            mask,
            baseline=valid_baseline.double(),
            **kwargs,
        )


@pytest.mark.parametrize("aggregate", ["l2", "l1", "sum", "mean"])
def test_attribute_batch_content_mode_without_historical_features(aggregate):
    model = RecordingModel()
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=3,
        ig_mode="content",
    )
    batch = {
        "content_features": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        "absolute_position_features": torch.tensor([[[5.0], [6.0]]]),
        "positions": torch.tensor([[10, 20]]),
        "gene_ids": torch.tensor([[0, 1]]),
        "mask": torch.tensor([[True, True]]),
        "labels": torch.tensor([1]),
    }

    attrs, scores, metadata = explainer.attribute_batch(_fake_loader(batch), aggregate=aggregate)

    assert len(attrs) == 1
    assert attrs[0].shape == (2, 2)
    assert scores[0].shape == (2,)
    assert metadata[0]["positions"].tolist() == [10, 20]
    assert not np.isnan(attrs[0]).any()
    assert not np.isnan(scores[0]).any()


def test_attribute_batch_content_mode_requires_split_keys():
    explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=2,
        ig_mode="content",
    )
    batch = {
        "positions": torch.tensor([[10]]),
        "gene_ids": torch.tensor([[0]]),
        "mask": torch.tensor([[True]]),
    }

    with pytest.raises(ValueError, match="content_features"):
        explainer.attribute_batch(_fake_loader(batch))


def test_attribute_batch_content_mode_excludes_padded_variants():
    explainer = IntegratedGradientsExplainer(
        RecordingModel(),
        device="cpu",
        n_steps=3,
        ig_mode="content",
    )
    batch = {
        "content_features": torch.tensor([[[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]]]),
        "absolute_position_features": torch.tensor([[[5.0], [99.0], [6.0]]]),
        "positions": torch.tensor([[10, 999, 30]]),
        "gene_ids": torch.tensor([[0, 99, 1]]),
        "mask": torch.tensor([[True, False, True]]),
    }

    attrs, scores, metadata = explainer.attribute_batch(_fake_loader(batch))

    assert attrs[0].shape == (2, 2)
    assert scores[0].shape == (2,)
    assert metadata[0]["positions"].tolist() == [10, 30]


def test_attribute_batch_content_mode_truncates_all_variant_tensors(monkeypatch):
    model = RecordingModel()
    explainer = IntegratedGradientsExplainer(
        model,
        device="cpu",
        n_steps=2,
        max_variants=2,
        ig_mode="content",
    )
    captured = {}

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
        captured["variant_features"] = variant_features
        captured["content_features"] = content_features.clone()
        captured["absolute_position_features"] = absolute_position_features.clone()
        captured["positions"] = positions.clone()
        captured["gene_ids"] = gene_ids.clone()
        captured["mask"] = mask.clone()
        captured["chrom_ids"] = chrom_ids.clone()
        return torch.ones_like(content_features)

    monkeypatch.setattr(explainer, "attribute", fake_attribute)
    monkeypatch.setattr(torch, "randperm", lambda n: torch.tensor([3, 1, 0, 2]))
    batch = {
        "content_features": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        "absolute_position_features": torch.arange(4, dtype=torch.float32).reshape(1, 4, 1),
        "positions": torch.tensor([[10, 20, 30, 40]]),
        "gene_ids": torch.tensor([[0, 1, 2, 3]]),
        "mask": torch.tensor([[True, True, True, True]]),
        "chrom_ids": torch.tensor([[0, 1, 0, 1]]),
    }

    attrs, _, metadata = explainer.attribute_batch(_fake_loader(batch))

    assert captured["variant_features"] is None
    assert torch.equal(captured["content_features"], batch["content_features"][:, [1, 3], :])
    assert torch.equal(
        captured["absolute_position_features"],
        batch["absolute_position_features"][:, [1, 3], :],
    )
    assert torch.equal(captured["positions"], batch["positions"][:, [1, 3]])
    assert torch.equal(captured["gene_ids"], batch["gene_ids"][:, [1, 3]])
    assert torch.equal(captured["mask"], batch["mask"][:, [1, 3]])
    assert torch.equal(captured["chrom_ids"], batch["chrom_ids"][:, [1, 3]])
    assert attrs[0].shape == (2, 2)
    assert metadata[0]["positions"].tolist() == [20, 40]
    assert metadata[0]["truncated"] is True


def test_attribute_batch_legacy_mode_preserves_historical_behavior():
    explainer = IntegratedGradientsExplainer(RecordingModel(), device="cpu", n_steps=3)
    batch = {
        "features": torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        "positions": torch.tensor([[10, 20]]),
        "gene_ids": torch.tensor([[0, 1]]),
        "mask": torch.tensor([[True, False]]),
        "labels": torch.tensor([1]),
    }

    attrs, scores, metadata = explainer.attribute_batch(_fake_loader(batch))

    assert attrs[0].shape == (1, 3)
    assert scores[0].shape == (1,)
    assert metadata[0]["positions"].tolist() == [10]


def test_state_dict_surface_is_unchanged_by_explainers():
    model = _make_sieve(69)
    before = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    IntegratedGradientsExplainer(model, device="cpu", n_steps=2)
    IntegratedGradientsExplainer(model, device="cpu", n_steps=2, ig_mode="content")

    after = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert before == after
