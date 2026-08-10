"""Tests for Phase 7B4A config and checkpoint reconstruction foundations."""

from __future__ import annotations

import copy

import pytest
import torch

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
    resolved_position_encoding_from_dict,
)
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.reconstruction import reconstruct_sieve_from_checkpoint
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
OLD_CONFIG = {
    "latent_dim": MODEL_KWARGS["latent_dim"],
    "hidden_dim": MODEL_KWARGS["hidden_dim"],
    "num_heads": MODEL_KWARGS["num_heads"],
    "num_attention_layers": MODEL_KWARGS["num_attention_layers"],
    "classifier_hidden_dim": MODEL_KWARGS["classifier_hidden_dim"],
    "dropout": MODEL_KWARGS["dropout"],
    "aggregation": MODEL_KWARGS["aggregation"],
    "num_covariates": MODEL_KWARGS["num_covariates"],
    "classifier_type": MODEL_KWARGS["classifier_type"],
}


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    level: AnnotationLevel = AnnotationLevel.L3,
    num_chromosomes: int = 3,
    **kwargs,
):
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            **kwargs,
        ),
        level,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=num_chromosomes,
    )


def _resolve_legacy(level: AnnotationLevel = AnnotationLevel.L3, *, num_chromosomes: int = 3):
    return resolve_position_encoding_config(
        PositionEncodingRequest(),
        level,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=num_chromosomes,
    )


def _position_dict(config, *, include_mapping: bool = True):
    data = copy.deepcopy(config.to_dict())
    if include_mapping:
        data["chromosome"]["mapping"] = (
            {} if config.chromosome.num_chromosomes == 0 else {"0": "1", "1": "2", "2": "X"}
        )
    return data


def _case_a_config(config, *, num_genes: int = 5):
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
        "dataset_identity": {"mappings_artifact": "dataset_mappings.json"},
        "position_encoding_execution": {
            "schema_version": 2,
            "resolved_config_applied_to_model": True,
        },
    }


def _base_model(config, *, num_genes: int = 5) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=num_genes,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=config.chromosome.num_chromosomes,
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
    )
    model.eval()
    return model


def _old_base_model(*, input_dim: int = 71, num_chromosomes: int = 0, num_genes: int = 5):
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
    )
    model.eval()
    return model


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def _chunked_checkpoint(base_model):
    chunked = ChunkedSIEVEModel(base_model, aggregation_method=MODEL_KWARGS["aggregation_method"])
    return {"model_state_dict": copy.deepcopy(chunked.state_dict())}


def _assert_state_exact(source_state, target_state):
    assert set(source_state) == set(target_state)
    forbidden = (
        "position_encoding",
        "position_runtime",
        "_absolute_position_runtime",
        "_relative_position_runtime",
    )
    for key, tensor in source_state.items():
        assert not any(token in key for token in forbidden)
        assert target_state[key].shape == tensor.shape
        assert torch.equal(target_state[key], tensor)


@pytest.mark.parametrize(
    "absolute", [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL]
)
@pytest.mark.parametrize(
    "relative", [RelativePositionEncoding.NONE, RelativePositionEncoding.T5_BUCKET]
)
@pytest.mark.parametrize("chromosome", [ChromosomeEncoding.NONE, ChromosomeEncoding.LEARNED])
@pytest.mark.parametrize(
    "cross_policy", [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK]
)
def test_all_supported_custom_configs_round_trip_through_deserializer(
    absolute,
    relative,
    chromosome,
    cross_policy,
):
    config = _resolve_custom(
        absolute=absolute,
        relative=relative,
        chromosome=chromosome,
        cross_policy=cross_policy,
        num_chromosomes=3,
    )

    parsed = resolved_position_encoding_from_dict(
        config.to_dict(),
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config


@pytest.mark.parametrize("level", list(AnnotationLevel))
def test_legacy_levels_round_trip_through_deserializer(level):
    config = _resolve_legacy(level)

    parsed = resolved_position_encoding_from_dict(
        config.to_dict(),
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config


def test_training_chromosome_mapping_extension_is_validated_but_not_retained():
    config = _resolve_custom(chromosome=ChromosomeEncoding.LEARNED)
    data = _position_dict(config)

    parsed = resolved_position_encoding_from_dict(
        data,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config
    assert "mapping" not in parsed.to_dict()["chromosome"]


def test_training_chromosome_mapping_extension_may_be_absent():
    config = _resolve_custom(chromosome=ChromosomeEncoding.LEARNED)

    parsed = resolved_position_encoding_from_dict(
        _position_dict(config, include_mapping=False),
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config


def test_training_chromosome_mapping_extension_accepts_empty_mapping_for_zero_chromosomes():
    config = _resolve_custom(num_chromosomes=0)
    data = _position_dict(config)

    parsed = resolved_position_encoding_from_dict(
        data,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.update(schema_version=99), "schema_version"),
        (lambda data: data.update(schema_version=True), "schema_version"),
        (lambda data: data.update(preset="bad"), "preset"),
        (lambda data: data.update(annotation_level="LX"), "annotation_level"),
        (lambda data: data.pop("absolute"), "absolute"),
        (lambda data: data.pop("relative"), "relative"),
        (lambda data: data.pop("chromosome"), "chromosome"),
        (lambda data: data.update(attribution="bad"), "attribution"),
        (lambda data: data.update(input_dim=data["input_dim"] + 1), "input_dim"),
        (lambda data: data.update(content_dim=data["content_dim"] + 1), "content_dim"),
        (lambda data: data.update(content_dim=float(data["content_dim"])), "content_dim"),
        (lambda data: data["absolute"].update(dim=66), "input_dim"),
        (lambda data: data["relative"].update(total_bias_rows=99), "total_bias_rows"),
        (
            lambda data: data["relative"].update(
                total_bias_rows=float(data["relative"]["total_bias_rows"])
            ),
            "total_bias_rows",
        ),
        (lambda data: data["chromosome"].update(requires_chrom_ids=False), "requires_chrom_ids"),
        (
            lambda data: data["chromosome"].update(cross_chromosome_parameter=None),
            "cross_chromosome_parameter",
        ),
        (lambda data: data["attribution"].update(default_ig_mode="legacy"), "default_ig_mode"),
        (lambda data: data["chromosome"].update(mapping={"0": "1", "1": "2"}), "mapping"),
        (lambda data: data["chromosome"].update(mapping=None), "mapping must be a mapping"),
        (
            lambda data: data["chromosome"].update(
                mapping={"0": "1", "1": "2", "2": "X", "3": "Y"}
            ),
            "mapping",
        ),
        (lambda data: data["chromosome"].update(mapping={"0": "1", "1": "2", "2": 3}), "values"),
        (lambda data: data["absolute"].update(extra="bad"), "unknown"),
    ],
)
def test_deserializer_rejects_corrupt_serialized_config(mutator, message):
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.SINUSOIDAL,
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
    )
    data = _position_dict(config)
    mutator(data)

    with pytest.raises(ValueError, match=message):
        resolved_position_encoding_from_dict(
            data,
            latent_dim=MODEL_KWARGS["latent_dim"],
            num_heads=MODEL_KWARGS["num_heads"],
        )


def test_deserializer_rejects_bool_input_dim_that_python_would_compare_equal_to_one():
    config = _resolve_custom(level=AnnotationLevel.L0, num_chromosomes=0)
    data = _position_dict(config)
    data["input_dim"] = True

    with pytest.raises(ValueError, match="position_encoding.input_dim"):
        resolved_position_encoding_from_dict(
            data,
            latent_dim=MODEL_KWARGS["latent_dim"],
            num_heads=MODEL_KWARGS["num_heads"],
        )


def test_future_strategy_deserializes_but_reconstruction_runtime_gate_rejects():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_FIXED,
        alibi_distance_scale=10000.0,
    )
    serialized = _case_a_config(config)
    parsed = resolved_position_encoding_from_dict(
        serialized["position_encoding"],
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config
    with pytest.raises(NotImplementedError, match="alibi_fixed"):
        reconstruct_sieve_from_checkpoint(
            serialized,
            _checkpoint(_old_base_model(input_dim=config.input_dim)),
            num_genes=5,
        )


@pytest.mark.parametrize(
    "config",
    [
        _resolve_custom(
            absolute=AbsolutePositionEncoding.NONE,
            relative=RelativePositionEncoding.NONE,
            num_chromosomes=0,
        ),
        _resolve_custom(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            position_dim=8,
            num_position_buckets=8,
            max_position_distance=1000,
        ),
        _resolve_custom(
            relative=RelativePositionEncoding.T5_BUCKET,
            cross_policy=CrossChromosomePolicy.MASK,
            num_position_buckets=8,
            max_position_distance=1000,
        ),
        _resolve_legacy(),
    ],
)
def test_case_a_reconstructs_exact_new_schema_state(config):
    source = _base_model(config)
    checkpoint = _checkpoint(source)
    result = reconstruct_sieve_from_checkpoint(_case_a_config(config), checkpoint, num_genes=5)

    assert result.is_new_schema is True
    assert result.is_chunked_checkpoint is False
    assert result.resolved_position_encoding == config
    assert result.base_model.position_encoding == config
    _assert_state_exact(checkpoint["model_state_dict"], result.model.state_dict())


def test_case_a_chunked_topology_reconstructs_wrapper_and_exact_state():
    config = _resolve_custom(relative=RelativePositionEncoding.T5_BUCKET)
    source = _base_model(config)
    checkpoint = _chunked_checkpoint(source)

    result = reconstruct_sieve_from_checkpoint(_case_a_config(config), checkpoint, num_genes=5)

    assert result.is_chunked_checkpoint is True
    assert isinstance(result.model, ChunkedSIEVEModel)
    assert result.model.base_model is result.base_model
    _assert_state_exact(checkpoint["model_state_dict"], result.model.state_dict())


@pytest.mark.parametrize(
    "config",
    [
        _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0),
        _resolve_custom(relative=RelativePositionEncoding.T5_BUCKET),
    ],
)
def test_case_a_accepts_raw_nullable_num_position_buckets_from_training_config(config):
    source = _base_model(config)
    serialized = _case_a_config(config)
    serialized["num_position_buckets"] = None

    result = reconstruct_sieve_from_checkpoint(serialized, _checkpoint(source), num_genes=5)

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config
    _assert_state_exact(source.state_dict(), result.model.state_dict())


@pytest.mark.parametrize("schema_version", [2.0, True])
def test_case_a_rejects_malformed_config_schema_version(schema_version):
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    source = _base_model(config)
    serialized = _case_a_config(config)
    serialized["config_schema_version"] = schema_version

    with pytest.raises(ValueError, match="config_schema_version"):
        reconstruct_sieve_from_checkpoint(serialized, _checkpoint(source), num_genes=5)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("input_dim", True),
        ("content_dim", 7.0),
    ],
)
def test_case_a_rejects_malformed_top_level_architecture_integer(key, value):
    config = _resolve_custom(level=AnnotationLevel.L0, num_chromosomes=0)
    source = _base_model(config)
    serialized = _case_a_config(config)
    serialized[key] = float(config.content_dim) if key == "content_dim" else value

    with pytest.raises(ValueError, match=key):
        reconstruct_sieve_from_checkpoint(serialized, _checkpoint(source), num_genes=5)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("input_dim", True),
        ("position_encoding_schema_version", True),
    ],
)
def test_case_a_rejects_checkpoint_metadata_scalar_type_conflicts(key, value):
    config = _resolve_custom(level=AnnotationLevel.L0, num_chromosomes=0)
    source = _base_model(config)
    serialized = _case_a_config(config)
    metadata = copy.deepcopy(serialized)
    metadata[key] = value

    with pytest.raises(ValueError, match=key):
        reconstruct_sieve_from_checkpoint(
            serialized,
            {"model_state_dict": source.state_dict(), "metadata": metadata},
            num_genes=5,
        )


def test_case_a_rejects_bool_dataset_num_chromosomes():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    source = _base_model(config)

    with pytest.raises(ValueError, match="dataset_num_chromosomes"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=True,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.pop("variant_encoder.encoder.0.weight"),
        lambda state: state.update({"unexpected.weight": torch.zeros(1)}),
        lambda state: state.update(
            {
                "attention.attention_layers.0.position_bias.weight": state[
                    "attention.attention_layers.0.position_bias.weight"
                ][:32].clone()
            }
        ),
        lambda state: state.pop("attention.attention_layers.0.chrom_embedding.weight"),
        lambda state: state.update({"variant_encoder.encoder.0.bias": torch.zeros(11)}),
    ],
)
def test_case_a_strict_loading_rejects_corruption(mutator):
    config = _resolve_custom(
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
    )
    source = _base_model(config)
    checkpoint = _checkpoint(source)
    mutator(checkpoint["model_state_dict"])

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(_case_a_config(config), checkpoint, num_genes=5)


def test_case_a_strict_loading_rejects_unexpected_positional_state_for_none_strategies():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    source = _base_model(config)
    checkpoint = _checkpoint(source)
    checkpoint["model_state_dict"]["attention.attention_layers.0.position_bias.weight"] = (
        torch.zeros(8, 2)
    )
    checkpoint["model_state_dict"]["attention.attention_layers.0.chrom_embedding.weight"] = (
        torch.zeros(4, 8)
    )

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(_case_a_config(config), checkpoint, num_genes=5)


def test_case_a_metadata_reconciliation_fills_missing_fields_and_rejects_conflicts():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    source = _base_model(config)
    full_config = _case_a_config(config)
    missing_content = copy.deepcopy(full_config)
    metadata = copy.deepcopy(full_config)
    missing_content.pop("content_dim")
    result = reconstruct_sieve_from_checkpoint(
        missing_content,
        {"model_state_dict": source.state_dict(), "metadata": metadata},
        num_genes=5,
    )
    assert result.effective_config["content_dim"] == config.content_dim

    conflicting = copy.deepcopy(metadata)
    conflicting["input_dim"] += 1
    with pytest.raises(ValueError, match="input_dim"):
        reconstruct_sieve_from_checkpoint(
            full_config,
            {"model_state_dict": source.state_dict(), "metadata": conflicting},
            num_genes=5,
        )

    nested_conflict = copy.deepcopy(metadata)
    nested_conflict["position_encoding"]["relative"]["type"] = "t5_bucket"
    with pytest.raises(ValueError, match="position_encoding.relative.type"):
        reconstruct_sieve_from_checkpoint(
            full_config,
            {"model_state_dict": source.state_dict(), "metadata": nested_conflict},
            num_genes=5,
        )


def test_case_a_metadata_reconciliation_does_not_mutate_inputs():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    source = _base_model(config)
    original_config = _case_a_config(config)
    metadata = copy.deepcopy(original_config)
    config_copy = copy.deepcopy(original_config)
    metadata_copy = copy.deepcopy(metadata)

    reconstruct_sieve_from_checkpoint(
        original_config,
        {"model_state_dict": source.state_dict(), "metadata": metadata},
        num_genes=5,
    )

    assert original_config == config_copy
    assert metadata == metadata_copy


def test_case_a_chromosome_mapping_conflict_in_metadata_raises():
    config = _resolve_custom(chromosome=ChromosomeEncoding.LEARNED)
    source = _base_model(config)
    full_config = _case_a_config(config)
    metadata = copy.deepcopy(full_config)
    metadata["position_encoding"]["chromosome"]["mapping"]["2"] = "Y"

    with pytest.raises(ValueError, match="mapping"):
        reconstruct_sieve_from_checkpoint(
            full_config,
            {"model_state_dict": source.state_dict(), "metadata": metadata},
            num_genes=5,
        )


def test_case_b_old_config_infers_input_dim_and_no_chromosome_embedding():
    source = _old_base_model(input_dim=69, num_chromosomes=0)
    result = reconstruct_sieve_from_checkpoint(
        OLD_CONFIG,
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=24,
    )

    assert result.is_new_schema is False
    assert result.resolved_position_encoding is None
    assert result.base_model.input_dim == 69
    assert result.base_model.num_chromosomes == 0
    assert result.base_model.attention.attention_layers[0].chrom_embedding is None


def test_case_b_old_config_infers_chromosome_embedding_rows():
    source = _old_base_model(input_dim=69, num_chromosomes=2)
    result = reconstruct_sieve_from_checkpoint(
        OLD_CONFIG,
        _checkpoint(source),
        num_genes=5,
    )

    assert result.base_model.num_chromosomes == 2
    assert result.base_model.attention.attention_layers[0].chrom_embedding.weight.shape == (3, 8)


def test_case_b_old_config_rejects_input_dim_conflict():
    source = _old_base_model(input_dim=69)
    with pytest.raises(ValueError, match="input_dim"):
        reconstruct_sieve_from_checkpoint(
            {**OLD_CONFIG, "input_dim": 71},
            _checkpoint(source),
            num_genes=5,
        )


def test_case_b_chunked_topology_reconstructs_wrapper():
    source = _old_base_model(input_dim=69)
    result = reconstruct_sieve_from_checkpoint(
        OLD_CONFIG,
        _chunked_checkpoint(source),
        num_genes=5,
    )

    assert result.is_chunked_checkpoint is True
    assert isinstance(result.model, ChunkedSIEVEModel)


def test_case_b_old_32_row_t5_migrates_and_31_row_t5_fails():
    source = _old_base_model(input_dim=69)
    old_state = copy.deepcopy(source.state_dict())
    key = "attention.attention_layers.0.position_bias.weight"
    old_state[key] = old_state[key][:32].clone()
    result = reconstruct_sieve_from_checkpoint(
        OLD_CONFIG,
        {"model_state_dict": old_state},
        num_genes=5,
    )
    assert torch.equal(result.model.state_dict()[key][:32], old_state[key])

    bad_state = copy.deepcopy(source.state_dict())
    bad_state[key] = bad_state[key][:31].clone()
    with pytest.raises(ValueError, match="position_bias"):
        reconstruct_sieve_from_checkpoint(
            OLD_CONFIG,
            {"model_state_dict": bad_state},
            num_genes=5,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.pop("variant_encoder.encoder.0.weight"),
        lambda state: state.update({"bad.weight": torch.zeros(1)}),
        lambda state: state.update({"variant_encoder.encoder.0.bias": torch.zeros(11)}),
    ],
)
def test_case_b_preflight_rejects_arbitrary_corruption(mutator):
    source = _old_base_model(input_dim=69)
    checkpoint = _checkpoint(source)
    mutator(checkpoint["model_state_dict"])

    with pytest.raises(ValueError):
        reconstruct_sieve_from_checkpoint(
            OLD_CONFIG,
            checkpoint,
            num_genes=5,
        )


def test_checkpoint_metadata_does_not_promote_old_config_to_new_schema():
    source = _old_base_model(input_dim=69)
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    checkpoint = _checkpoint(source)
    checkpoint["metadata"] = _case_a_config(config)

    result = reconstruct_sieve_from_checkpoint(
        OLD_CONFIG,
        checkpoint,
        num_genes=5,
    )

    assert result.is_new_schema is False
    assert result.resolved_position_encoding is None


def _transitional_config(position_config, *, applied: object = False):
    return {
        **MODEL_KWARGS,
        "input_dim": 71,
        "position_encoding": _position_dict(position_config),
        "position_encoding_execution": {"resolved_config_applied_to_model": applied},
    }


def test_case_c_transitional_single_split_uses_state_driven_no_chromosome_architecture():
    intended = _resolve_legacy()
    source = _old_base_model(input_dim=71, num_chromosomes=0)
    result = reconstruct_sieve_from_checkpoint(
        _transitional_config(intended),
        _checkpoint(source),
        num_genes=5,
    )

    assert result.is_new_schema is False
    assert result.resolved_position_encoding is None
    assert result.base_model.attention.attention_layers[0].chrom_embedding is None


def test_case_c_transitional_cv_uses_state_driven_chromosome_architecture():
    intended = _resolve_legacy()
    source = _old_base_model(input_dim=71, num_chromosomes=2)
    result = reconstruct_sieve_from_checkpoint(
        _transitional_config(intended),
        _checkpoint(source),
        num_genes=5,
    )

    assert result.is_new_schema is False
    assert result.base_model.num_chromosomes == 2


def test_case_c_transitional_accepts_raw_nullable_num_position_buckets():
    intended = _resolve_legacy()
    source = _old_base_model(input_dim=71, num_chromosomes=0)
    config = _transitional_config(intended)
    config["num_position_buckets"] = None

    result = reconstruct_sieve_from_checkpoint(
        config,
        _checkpoint(source),
        num_genes=5,
    )

    assert result.is_new_schema is False
    assert result.base_model.attention.attention_layers[0].num_position_buckets == 32


@pytest.mark.parametrize(
    "config",
    [
        lambda intended: {**MODEL_KWARGS, "position_encoding": _position_dict(intended)},
        lambda intended: {
            **MODEL_KWARGS,
            "position_encoding": _position_dict(intended),
            "position_encoding_execution": "bad",
        },
        lambda intended: _transitional_config(intended, applied=True),
    ],
)
def test_case_c_ambiguous_transitional_configs_raise(config):
    intended = _resolve_legacy()
    source = _old_base_model(input_dim=71)

    with pytest.raises(ValueError):
        reconstruct_sieve_from_checkpoint(
            config(intended),
            _checkpoint(source),
            num_genes=5,
        )


def test_case_c_checkpoint_metadata_schema_v2_does_not_silently_promote():
    intended = _resolve_legacy()
    source = _old_base_model(input_dim=71)
    checkpoint = _checkpoint(source)
    checkpoint["metadata"] = {"config_schema_version": 2}

    with pytest.raises(ValueError, match="promoted|promote"):
        reconstruct_sieve_from_checkpoint(
            _transitional_config(intended),
            checkpoint,
            num_genes=5,
        )
