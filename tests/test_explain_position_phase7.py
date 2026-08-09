"""Phase 7B4B explanation integration tests for positional reconstruction."""

from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts import explain
from src.encoding import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedIGMode,
    resolve_position_encoding_config,
)
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.ig_mode import IGModeCompatibilityWarning, resolve_ig_mode
from src.models.reconstruction import ReconstructedSIEVEModel
from src.models.sieve import SIEVE

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 2,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 12,
    "dropout": 0.0,
    "aggregation": "max",
    "aggregation_method": "mean",
    "num_covariates": 0,
    "classifier_type": "flatten",
}


def _resolve(
    *,
    preset=PositionPreset.CUSTOM,
    absolute=AbsolutePositionEncoding.NONE,
    relative=RelativePositionEncoding.NONE,
    chromosome=ChromosomeEncoding.NONE,
    cross_policy=CrossChromosomePolicy.SEPARATE,
    num_chromosomes=0,
    **kwargs,
):
    request = PositionEncodingRequest(
        preset=preset,
        absolute_position_encoding=absolute if preset is PositionPreset.CUSTOM else None,
        relative_position_encoding=relative if preset is PositionPreset.CUSTOM else None,
        chromosome_encoding=chromosome if preset is PositionPreset.CUSTOM else None,
        cross_chromosome_policy=cross_policy if preset is PositionPreset.CUSTOM else None,
        **kwargs,
    )
    return resolve_position_encoding_config(
        request,
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=num_chromosomes,
    )


def _position_dict(config):
    data = copy.deepcopy(config.to_dict())
    data["chromosome"]["mapping"] = {
        str(idx): name
        for idx, name in enumerate(["1", "2", "X", "Y"][: config.chromosome.num_chromosomes])
    }
    return data


def _case_a_config(config, *, num_genes=5):
    return {
        **MODEL_KWARGS,
        "config_schema_version": 2,
        "metadata_schema_version": 1,
        "position_encoding_schema_version": config.schema_version,
        "input_dim": config.input_dim,
        "content_dim": config.content_dim,
        "num_genes": num_genes,
        "num_chromosomes": config.chromosome.num_chromosomes,
        "position_encoding": _position_dict(config),
        "position_encoding_execution": {
            "schema_version": 2,
            "resolved_config_applied_to_model": True,
        },
    }


def _transitional_config(intended):
    return {
        **MODEL_KWARGS,
        "input_dim": 71,
        "position_encoding": _position_dict(intended),
        "position_encoding_execution": {"resolved_config_applied_to_model": False},
    }


def _model(config=None, *, input_dim=None, num_chromosomes=0, num_genes=5):
    if config is not None:
        input_dim = config.input_dim
        num_chromosomes = config.chromosome.num_chromosomes
    model = SIEVE(
        input_dim=input_dim,
        num_genes=num_genes,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=num_chromosomes,
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
    )
    model.eval()
    return model


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def _dataset(*, num_genes=5, num_chromosomes=0):
    return SimpleNamespace(num_genes=num_genes, num_chromosomes=num_chromosomes)


def _fake_reconstruction(*, config=None, is_new_schema=True, effective_config=None):
    if effective_config is None:
        effective_config = {} if config is None else _case_a_config(config)
    return ReconstructedSIEVEModel(
        model=nn.Identity(),
        base_model=SimpleNamespace(input_dim=71),
        resolved_position_encoding=config,
        is_new_schema=is_new_schema,
        is_chunked_checkpoint=False,
        effective_config=effective_config,
    )


def test_explain_reconstruction_helper_delegates_without_mutating_config(monkeypatch):
    config = {"level": "L3", "input_dim": 69}
    checkpoint = {"model_state_dict": {}}
    dataset = _dataset(num_genes=7, num_chromosomes=3)
    original = copy.deepcopy(config)
    captured = {}
    expected = object()

    def fake_reconstruct(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(explain, "reconstruct_sieve_from_checkpoint", fake_reconstruct)

    result = explain._reconstruct_model_for_explanation(config, checkpoint, dataset)

    assert result is expected
    assert captured == {
        "args": (config, checkpoint),
        "kwargs": {"num_genes": 7, "dataset_num_chromosomes": 3},
    }
    assert config == original


@pytest.mark.parametrize(
    "config",
    [
        _resolve(num_chromosomes=0),
        _resolve(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            position_dim=8,
            num_position_buckets=8,
            max_position_distance=1000,
            num_chromosomes=3,
        ),
    ],
)
def test_schema_v2_custom_models_reconstruct_for_explanation(config):
    source = _model(config)

    reconstruction = explain._reconstruct_model_for_explanation(
        _case_a_config(config),
        _checkpoint(source),
        _dataset(num_chromosomes=config.chromosome.num_chromosomes),
    )

    assert reconstruction.is_new_schema is True
    assert reconstruction.resolved_position_encoding == config
    assert reconstruction.base_model.input_dim == config.input_dim


def test_case_b_and_case_c_reconstruct_historically_for_explanation():
    source = _model(input_dim=69, num_chromosomes=0)
    old_config = {**MODEL_KWARGS}
    case_b = explain._reconstruct_model_for_explanation(
        old_config,
        _checkpoint(source),
        _dataset(num_chromosomes=24),
    )
    assert case_b.is_new_schema is False
    assert case_b.base_model.input_dim == 69

    intended = _resolve(preset=PositionPreset.LEGACY, num_chromosomes=3)
    source_c = _model(input_dim=71, num_chromosomes=0)
    case_c = explain._reconstruct_model_for_explanation(
        _transitional_config(intended),
        _checkpoint(source_c),
        _dataset(num_chromosomes=24),
    )
    assert case_c.is_new_schema is False
    assert case_c.resolved_position_encoding is None
    assert case_c.base_model.input_dim == 71


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("auto", ResolvedIGMode.CONTENT), ("content", ResolvedIGMode.CONTENT)],
)
def test_resolve_ig_mode_new_schema_authority(requested, expected):
    config = _case_a_config(_resolve())

    assert resolve_ig_mode(requested, config=config, is_new_schema=True) is expected
    assert resolve_ig_mode("legacy", config=config, is_new_schema=True) is ResolvedIGMode.LEGACY


def test_resolve_ig_mode_false_ignores_malformed_transitional_metadata():
    config = {
        "position_encoding": {"attribution": {"default_ig_mode": "not-valid"}},
    }

    with pytest.warns(IGModeCompatibilityWarning, match="historical/transitional"):
        assert resolve_ig_mode("auto", config=config, is_new_schema=False) is ResolvedIGMode.LEGACY
    with pytest.warns(IGModeCompatibilityWarning, match="historical/transitional"):
        assert (
            resolve_ig_mode("content", config=config, is_new_schema=False) is ResolvedIGMode.CONTENT
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            resolve_ig_mode("legacy", config=config, is_new_schema=False) is ResolvedIGMode.LEGACY
        )
    assert not caught


@pytest.mark.parametrize("is_new_schema", [1, "false"])
def test_resolve_ig_mode_rejects_non_bool_execution_authority(is_new_schema):
    with pytest.raises(ValueError, match="is_new_schema"):
        resolve_ig_mode(
            "auto",
            config={},
            is_new_schema=is_new_schema,
        )


def test_custom_legacy_ig_policy_uses_reconstruction_preset():
    custom = _fake_reconstruction(config=_resolve())
    legacy = _fake_reconstruction(config=_resolve(preset=PositionPreset.LEGACY, num_chromosomes=3))
    historical = _fake_reconstruction(is_new_schema=False)
    transitional = _fake_reconstruction(
        is_new_schema=False,
        effective_config={"position_encoding": {"attribution": {"default_ig_mode": "content"}}},
    )

    with pytest.raises(ValueError, match="custom positional execution"):
        explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, custom)
    explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.CONTENT, custom)
    explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, legacy)
    explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, historical)
    explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, transitional)


def test_content_dim_and_strategy_metadata_follow_reconstruction_authority():
    custom_config = _resolve(
        absolute=AbsolutePositionEncoding.SINUSOIDAL,
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
        num_chromosomes=3,
    )
    case_a = _fake_reconstruction(config=custom_config)
    case_b = _fake_reconstruction(is_new_schema=False, effective_config={})
    case_c = _fake_reconstruction(
        is_new_schema=False,
        effective_config={"position_encoding": _position_dict(custom_config)},
    )

    assert explain._content_dim_for_reconstruction(case_a, AnnotationLevel.L3) == 7
    assert explain._content_dim_for_reconstruction(case_b, AnnotationLevel.L2) == 5
    assert explain._read_position_strategy_metadata(case_a) == {
        "absolute_position_encoding": "sinusoidal",
        "relative_position_encoding": "t5_bucket",
        "chromosome_encoding": "learned",
        "position_encoding_metadata_source": "reconstructed_resolved_config",
    }
    assert explain._read_position_strategy_metadata(case_b) == {
        "absolute_position_encoding": None,
        "relative_position_encoding": None,
        "chromosome_encoding": None,
        "position_encoding_metadata_source": "unavailable_old_config",
    }
    assert explain._read_position_strategy_metadata(case_c) == {
        "absolute_position_encoding": None,
        "relative_position_encoding": None,
        "chromosome_encoding": None,
        "position_encoding_metadata_source": "transitional_historical_execution",
    }


def test_build_ig_metadata_uses_reconstruction_input_dim_and_provenance():
    config = _resolve()
    reconstruction = _fake_reconstruction(config=config)
    reconstruction.base_model.input_dim = 123

    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="content",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        reconstruction=reconstruction,
        content_dim=config.content_dim,
        n_steps=3,
        max_variants=4,
    )

    assert metadata["input_dim"] == 123
    assert metadata["attribution_width"] == config.content_dim
    assert metadata["position_encoding_metadata_source"] == "reconstructed_resolved_config"

    legacy = explain._build_ig_run_metadata(
        requested_ig_mode="legacy",
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        reconstruction=reconstruction,
        content_dim=config.content_dim,
        n_steps=3,
        max_variants=4,
    )

    assert legacy["input_dim"] == 123
    assert legacy["attribution_width"] == 123


class AttentionSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_attention_patterns(
        self,
        variant_features,
        positions,
        gene_ids,
        mask,
        chrom_ids=None,
        *,
        content_features=None,
        absolute_position_features=None,
    ):
        self.calls.append(
            {
                "variant_features": variant_features,
                "content_features": content_features,
                "absolute_position_features": absolute_position_features,
                "chrom_ids": chrom_ids,
            }
        )
        return [torch.ones(positions.shape[0], 1, positions.shape[1], positions.shape[1])]


def _attention_inputs():
    return {
        "features": torch.randn(1, 2, 3),
        "content": torch.randn(1, 2, 2),
        "absolute": torch.randn(1, 2, 1),
        "positions": torch.tensor([[10, 20]]),
        "gene_ids": torch.tensor([[0, 1]]),
        "mask": torch.tensor([[True, True]]),
        "chrom_ids": torch.tensor([[0, 1]]),
    }


def test_attention_analyzer_historical_and_split_modes_propagate_chrom_ids():
    spy = AttentionSpy()
    analyzer = AttentionAnalyzer(spy, device="cpu")
    tensors = _attention_inputs()

    analyzer.extract_attention_weights(
        tensors["features"],
        tensors["positions"],
        tensors["gene_ids"],
        tensors["mask"],
        chrom_ids=tensors["chrom_ids"],
    )
    analyzer.extract_attention_weights(
        None,
        tensors["positions"],
        tensors["gene_ids"],
        tensors["mask"],
        chrom_ids=tensors["chrom_ids"],
        content_features=tensors["content"],
        absolute_position_features=tensors["absolute"],
    )

    assert spy.calls[0]["variant_features"] is not None
    assert spy.calls[0]["content_features"] is None
    assert torch.equal(spy.calls[0]["chrom_ids"], tensors["chrom_ids"])
    assert spy.calls[1]["variant_features"] is None
    assert torch.equal(spy.calls[1]["content_features"], tensors["content"])
    assert torch.equal(spy.calls[1]["absolute_position_features"], tensors["absolute"])
    assert torch.equal(spy.calls[1]["chrom_ids"], tensors["chrom_ids"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"content_features": torch.randn(1, 2, 2)},
        {"absolute_position_features": torch.randn(1, 2, 1)},
        {
            "variant_features": torch.randn(1, 2, 3),
            "content_features": torch.randn(1, 2, 2),
        },
    ],
)
def test_attention_analyzer_rejects_ambiguous_feature_modes(kwargs):
    analyzer = AttentionAnalyzer(AttentionSpy(), device="cpu")
    variant_features = kwargs.pop("variant_features", None)

    with pytest.raises(ValueError):
        analyzer.extract_attention_weights(
            variant_features,
            torch.tensor([[10, 20]]),
            torch.tensor([[0, 1]]),
            torch.tensor([[True, True]]),
            **kwargs,
        )


def test_attention_routing_rule_is_custom_only():
    custom = _fake_reconstruction(config=_resolve())
    legacy = _fake_reconstruction(config=_resolve(preset=PositionPreset.LEGACY, num_chromosomes=3))
    case_b = _fake_reconstruction(is_new_schema=False, effective_config={})
    case_c = _fake_reconstruction(
        is_new_schema=False,
        effective_config={"position_encoding": {"attribution": {}}},
    )

    assert explain._attention_uses_split_inputs(custom) is True
    assert explain._attention_uses_split_inputs(legacy) is False
    assert explain._attention_uses_split_inputs(case_b) is False
    assert explain._attention_uses_split_inputs(case_c) is False


def test_skip_ig_metadata_does_not_resolve_or_reject_custom_legacy(monkeypatch):
    calls = []

    monkeypatch.setattr(
        explain,
        "resolve_ig_mode",
        lambda *args, **kwargs: calls.append("resolve"),
    )
    monkeypatch.setattr(
        explain,
        "_validate_ig_mode_for_reconstruction",
        lambda *args, **kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        explain,
        "_create_integrated_gradients_explainer",
        lambda *args, **kwargs: calls.append("explainer"),
    )

    metadata = explain._build_skipped_ig_metadata("legacy")

    assert metadata == {
        "executed": False,
        "requested_ig_mode": "legacy",
        "resolved_ig_mode": None,
    }
    assert calls == []
