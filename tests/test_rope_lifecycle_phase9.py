"""Phase 9C RoPE lifecycle diagnostics and integration tests."""

from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import explain, train
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedIGMode,
    resolve_position_encoding_config,
    resolved_position_encoding_from_dict,
)
from src.encoding.position_layout import learned_binned_layout_from_position_encoding_dict
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.ig_mode import resolve_ig_mode
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
CHROM_INDEX = {"1": 0, "2": 1, "X": 2}
SWAPPED_CHROM_INDEX = {"1": 1, "2": 0, "X": 2}
RENAMED_CHROM_INDEX = {"1": 0, "2": 1, "Y": 2}
DIFFERENT_CHROM_INDEX = {"A": 0, "B": 1, "C": 2}


def _args(*extra: str, num_attention_layers: int | None = None) -> argparse.Namespace:
    return train.parse_args(
        [
            "--level",
            "L3",
            "--latent-dim",
            str(MODEL_KWARGS["latent_dim"]),
            "--hidden-dim",
            str(MODEL_KWARGS["hidden_dim"]),
            "--num-heads",
            str(MODEL_KWARGS["num_heads"]),
            "--num-attention-layers",
            str(
                MODEL_KWARGS["num_attention_layers"]
                if num_attention_layers is None
                else num_attention_layers
            ),
            *extra,
        ]
    )


def _rope_args(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    rope_coordinate_scale: float = 12345.0,
    rope_base: float = 4321.0,
    num_attention_layers: int | None = None,
) -> argparse.Namespace:
    args = [
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        absolute.value,
        "--relative-position-encoding",
        "rope",
        "--chromosome-encoding",
        chromosome.value,
        "--cross-chromosome-policy",
        cross_policy.value,
        "--rope-coordinate-scale",
        str(rope_coordinate_scale),
        "--rope-base",
        str(rope_base),
    ]
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        args.extend(["--position-dim", "4"])
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        args.extend(["--position-dim", "8", "--position-bin-size", "100000000"])
    return _args(*args, num_attention_layers=num_attention_layers)


def _resolve_custom(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.ROPE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
):
    kwargs = {}
    if relative is RelativePositionEncoding.ROPE:
        kwargs = {"rope_coordinate_scale": 12345.0, "rope_base": 4321.0}
    elif relative is RelativePositionEncoding.T5_BUCKET:
        kwargs = {"num_position_buckets": 8, "max_position_distance": 1000}
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        kwargs.update({"position_dim": 8, "position_bin_size": 100000000})
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            **kwargs,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=3,
    )


def _resolve_rope(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_attention_layers: int | None = None,
):
    return train.prepare_training_position_encoding(
        _rope_args(
            chromosome=chromosome,
            absolute=absolute,
            cross_policy=cross_policy,
            num_attention_layers=num_attention_layers,
        ),
        AnnotationLevel.L3,
        num_chromosomes=3,
    )


def _run_metadata(config, *, chrom_index=CHROM_INDEX):
    return train.build_training_run_metadata(
        input_dim=config.input_dim,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        genome_build="GRCh37",
        resolved_position_encoding=config,
        chrom_index=chrom_index,
        gene_mapping_sha256="genehash",
        chromosome_mapping_sha256="chromhash",
        training_mode="cv",
    )


def _case_a_config(config, metadata, **overrides):
    return {
        **MODEL_KWARGS,
        "level": "L3",
        **copy.deepcopy(metadata),
        **overrides,
    }


def _layout_from_metadata(metadata):
    serialized = metadata["position_encoding"]
    return learned_binned_layout_from_position_encoding_dict(
        serialized,
        chromosome_mapping=serialized["chromosome"]["mapping"],
    )


def _source_model(config, layout=None, *, num_attention_layers: int | None = None) -> SIEVE:
    torch.manual_seed(1234)
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=(
            MODEL_KWARGS["num_attention_layers"]
            if num_attention_layers is None
            else num_attention_layers
        ),
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=config.chromosome.num_chromosomes,
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
        learned_binned_position_layout=layout,
    )
    model.eval()
    return model


def _training_model(config, *, layout=None, num_attention_layers: int | None = None):
    model = train.create_training_model(
        args=_rope_args(
            absolute=config.absolute.encoding,
            chromosome=config.chromosome.encoding,
            cross_policy=config.chromosome.cross_chromosome_policy,
            rope_coordinate_scale=config.relative.rope_coordinate_scale or 12345.0,
            rope_base=config.relative.rope_base or 4321.0,
            num_attention_layers=num_attention_layers,
        ),
        resolved_position_encoding=config,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        num_covariates=0,
        learned_binned_position_layout=layout,
    )
    model.eval()
    return model


def _old_source_model() -> SIEVE:
    model = SIEVE(
        input_dim=71,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=0,
        classifier_type=MODEL_KWARGS["classifier_type"],
    )
    model.eval()
    return model


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def _dataset():
    return SimpleNamespace(
        num_genes=5,
        num_chromosomes=3,
        chrom_index=CHROM_INDEX,
    )


def _split_batch(config, *, positions=None, chrom_ids=None):
    variants = 3
    content = torch.tensor(
        [
            [
                [0.2, 1.0, 0.0, 0.0, 1.0, 0.3, 0.7],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.6, 0.1],
                [0.5, 0.0, 0.0, 1.0, 0.0, 0.4, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    absolute_width = config.absolute.position_dim or 0
    absolute = torch.zeros(1, variants, absolute_width)
    return {
        "content_features": content,
        "absolute_position_features": absolute,
        "positions": (
            torch.tensor([[10, 20, 35]], dtype=torch.long) if positions is None else positions
        ),
        "gene_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "mask": torch.tensor([[True, True, True]]),
        "chrom_ids": (
            torch.tensor([[0, 1, 0]], dtype=torch.long) if chrom_ids is None else chrom_ids
        ),
    }


def _chunk(config, *, positions=None, chrom_ids=None):
    batch = _split_batch(
        config,
        positions=(
            torch.tensor([[10, 20, 0]], dtype=torch.long) if positions is None else positions
        ),
        chrom_ids=(torch.tensor([[0, 0, 0]], dtype=torch.long) if chrom_ids is None else chrom_ids),
    )
    batch["mask"] = torch.tensor([[True, True, False]])
    return {key: value.squeeze(0) for key, value in batch.items()}


def _run_split_forward(model, config, *, batch=None):
    batch = _split_batch(config) if batch is None else batch
    return model(
        None,
        batch["positions"],
        batch["gene_ids"],
        batch["mask"],
        chrom_ids=batch["chrom_ids"],
        content_features=batch["content_features"],
        absolute_position_features=batch["absolute_position_features"],
    )


def _rope_state_keys(model) -> set[str]:
    return {
        key
        for key in model.state_dict()
        if "cross_chromosome_bias" in key or "position_bias.weight" in key
    }


def _assert_state_exact(source_state, target_state):
    assert set(target_state) == set(source_state)
    for key, tensor in source_state.items():
        assert torch.equal(target_state[key], tensor), key


def _set_rope_biases(model, values_by_layer=None):
    layers = (
        model.base_model.attention.attention_layers
        if isinstance(model, ChunkedSIEVEModel)
        else (model.attention.attention_layers)
    )
    if values_by_layer is None:
        values_by_layer = [
            torch.tensor([0.25 + idx, -0.50 - idx], dtype=torch.float32)
            for idx in range(len(layers))
        ]
    with torch.no_grad():
        for layer, values in zip(layers, values_by_layer, strict=True):
            layer.cross_chromosome_bias.copy_(values)


def _explain_reconstruction(config, metadata, source):
    return explain._reconstruct_model_for_explanation(
        _case_a_config(config, metadata),
        _checkpoint(source),
        _dataset(),
    )


def test_rope_only_reconstruction_does_not_require_exact_chromosome_name_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.NONE)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=SWAPPED_CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_rope_with_learned_chromosome_embedding_requires_exact_chromosome_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)
    serialized = _case_a_config(config, metadata)
    checkpoint = _checkpoint(source)

    matching = reconstruct_sieve_from_checkpoint(
        serialized,
        checkpoint,
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    assert matching.is_new_schema is True
    assert matching.resolved_position_encoding == config

    with pytest.raises(ValueError, match="chromosome|mapping|chrom_index"):
        reconstruct_sieve_from_checkpoint(
            serialized,
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_learned_chromosome_embedding_rejects_wrong_chromosome_name_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=RENAMED_CHROM_INDEX,
        )


def test_t5_with_learned_chromosome_embedding_uses_same_identity_guard():
    config = _resolve_custom(
        chromosome=ChromosomeEncoding.LEARNED,
        relative=RelativePositionEncoding.T5_BUCKET,
    )
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_rope_only_reconstruction_allows_different_valid_chromosome_names():
    config = _resolve_rope(chromosome=ChromosomeEncoding.NONE)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=DIFFERENT_CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_learned_binned_still_requires_exact_mapping_after_identity_refactor():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        relative=RelativePositionEncoding.ROPE,
        chromosome=ChromosomeEncoding.NONE,
    )
    metadata = _run_metadata(config)
    source = _source_model(config, _layout_from_metadata(metadata))

    matching = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    assert matching.is_new_schema is True

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_learned_chromosome_embedding_rejects_dataset_count_without_mapping():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="dataset_chrom_index"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
        )


def test_learned_chromosome_embedding_pure_checkpoint_reconstruction_succeeds():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_learned_chromosome_embedding_requires_saved_mapping():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    metadata["position_encoding"]["chromosome"].pop("mapping")
    source = _source_model(config)

    with pytest.raises(ValueError, match="position_encoding.chromosome.mapping"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
        )


def test_case_b_and_case_c_remain_state_driven_compatibility_paths():
    intended = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    old_source = _old_source_model()
    transitional = {
        **MODEL_KWARGS,
        "level": "L3",
        "input_dim": 71,
        "position_encoding": intended.to_dict(),
        "position_encoding_execution": {"resolved_config_applied_to_model": False},
    }

    case_b = reconstruct_sieve_from_checkpoint(
        {**MODEL_KWARGS, "level": "L3"},
        _checkpoint(old_source),
        num_genes=5,
        dataset_num_chromosomes=3,
    )
    case_c = reconstruct_sieve_from_checkpoint(
        transitional,
        _checkpoint(old_source),
        num_genes=5,
        dataset_num_chromosomes=3,
    )

    assert case_b.is_new_schema is False
    assert case_c.is_new_schema is False
    assert not any("cross_chromosome_bias" in key for key in case_b.model.state_dict())
    assert not any("cross_chromosome_bias" in key for key in case_c.model.state_dict())


@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_training_serializes_rope_strategy_and_execution_metadata(cross_policy):
    config = _resolve_rope(cross_policy=cross_policy)
    metadata = _run_metadata(config)

    relative = metadata["position_encoding"]["relative"]
    execution = metadata["position_encoding_execution"]
    assert relative["type"] == "rope"
    assert relative["rope_coordinate_scale"] == 12345.0
    assert relative["rope_base"] == 4321.0
    assert relative["total_bias_rows"] is None
    assert execution["relative_position_encoding"] == "rope"
    assert execution["position_bias_rows"] is None
    assert execution["cross_chromosome_policy"] == cross_policy.value


@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_serialized_training_position_config_round_trips_exactly(cross_policy):
    config = _resolve_rope(cross_policy=cross_policy)
    metadata = _run_metadata(config)

    parsed = resolved_position_encoding_from_dict(
        metadata["position_encoding"],
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert parsed == config
    assert parsed.relative.rope_coordinate_scale == 12345.0
    assert parsed.relative.rope_base == 4321.0


def test_training_created_rope_separate_model_has_exact_relative_state_surface():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _training_model(config)
    key = "base_model.attention.attention_layers.0.cross_chromosome_bias"
    layer = model.base_model.attention.attention_layers[0]

    assert _rope_state_keys(model) == {key}
    assert layer.position_bias is None
    assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
    assert torch.equal(
        layer.cross_chromosome_bias,
        torch.zeros_like(layer.cross_chromosome_bias),
    )


def test_training_created_rope_mask_model_has_no_relative_state_surface():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.MASK)
    model = _training_model(config)
    layer = model.base_model.attention.attention_layers[0]

    assert _rope_state_keys(model) == set()
    assert layer.position_bias is None
    assert layer.cross_chromosome_bias is None


def test_training_created_rope_separate_forward_backward_succeeds():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _training_model(config)
    batch = _split_batch(config)
    batch["content_features"] = batch["content_features"].clone().requires_grad_(True)

    logits, _ = _run_split_forward(model, config, batch=batch)
    logits.sum().backward()

    grad = model.base_model.attention.attention_layers[0].cross_chromosome_bias.grad
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(batch["content_features"].grad).all()
    assert grad is not None
    assert torch.any(grad != 0)


def test_training_created_rope_mask_forward_succeeds():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.MASK)
    model = _training_model(config)

    logits, _ = _run_split_forward(model, config)

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()


def test_base_case_a_rope_separate_strict_round_trip_restores_nonzero_bias():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    _set_rope_biases(source)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.is_chunked_checkpoint is False
    assert result.resolved_position_encoding.relative.encoding is RelativePositionEncoding.ROPE
    assert result.resolved_position_encoding.relative.rope_coordinate_scale == 12345.0
    assert result.resolved_position_encoding.relative.rope_base == 4321.0
    assert result.resolved_position_encoding.chromosome.cross_chromosome_policy is (
        CrossChromosomePolicy.SEPARATE
    )
    assert torch.equal(
        result.base_model.attention.attention_layers[0].cross_chromosome_bias,
        source.attention.attention_layers[0].cross_chromosome_bias,
    )
    _assert_state_exact(source.state_dict(), result.model.state_dict())


def test_chunked_case_a_rope_separate_strict_round_trip_restores_nonzero_bias():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _training_model(config)
    _set_rope_biases(source)
    key = "base_model.attention.attention_layers.0.cross_chromosome_bias"

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata, classifier_hidden_dim=256),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert key in source.state_dict()
    assert result.is_chunked_checkpoint is True
    assert isinstance(result.model, ChunkedSIEVEModel)
    assert torch.equal(
        result.base_model.attention.attention_layers[0].cross_chromosome_bias,
        source.base_model.attention.attention_layers[0].cross_chromosome_bias,
    )
    _assert_state_exact(source.state_dict(), result.model.state_dict())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state, key: state.pop(key),
        lambda state, key: state.update({key: torch.zeros(MODEL_KWARGS["num_heads"] + 1)}),
        lambda state, key: state.update(
            {
                "attention.attention_layers.0.position_bias.weight": torch.zeros(
                    8,
                    MODEL_KWARGS["num_heads"],
                )
            }
        ),
    ],
)
def test_rope_separate_strict_loading_rejects_corrupt_relative_state(mutator):
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    checkpoint = _checkpoint(source)
    key = next(key for key in checkpoint["model_state_dict"] if "cross_chromosome_bias" in key)
    mutator(checkpoint["model_state_dict"], key)

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


def test_multilayer_rope_separate_requires_each_layer_cross_bias_key():
    config = _resolve_rope(
        cross_policy=CrossChromosomePolicy.SEPARATE,
        num_attention_layers=2,
    )
    metadata = _run_metadata(config)
    source = _source_model(config, num_attention_layers=2)
    checkpoint = _checkpoint(source)

    assert {
        "attention.attention_layers.0.cross_chromosome_bias",
        "attention.attention_layers.1.cross_chromosome_bias",
    }.issubset(checkpoint["model_state_dict"])
    checkpoint["model_state_dict"].pop("attention.attention_layers.1.cross_chromosome_bias")

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata, num_attention_layers=2),
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


@pytest.mark.parametrize(
    "unexpected_key",
    [
        "attention.attention_layers.0.cross_chromosome_bias",
        "attention.attention_layers.0.position_bias.weight",
    ],
)
def test_rope_mask_strict_loading_rejects_unexpected_relative_state(unexpected_key):
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.MASK)
    metadata = _run_metadata(config)
    source = _source_model(config)
    checkpoint = _checkpoint(source)

    assert not any("cross_chromosome_bias" in key for key in checkpoint["model_state_dict"])
    assert not any("position_bias.weight" in key for key in checkpoint["model_state_dict"])
    checkpoint["model_state_dict"][unexpected_key] = torch.zeros(MODEL_KWARGS["num_heads"])

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


@pytest.mark.parametrize(
    "field",
    ["rope_coordinate_scale", "rope_base"],
)
def test_rope_checkpoint_metadata_conflicts_reject_before_state_authority(field):
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    checkpoint = _checkpoint(source)
    checkpoint["metadata"] = copy.deepcopy(_case_a_config(config, metadata))
    checkpoint["metadata"]["position_encoding"]["relative"][field] += 1.0

    with pytest.raises(ValueError, match=f"position_encoding.relative.{field}"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


def test_explanation_reconstructs_schema_v2_rope_and_restores_bias():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    _set_rope_biases(source)

    reconstruction = _explain_reconstruction(config, metadata, source)

    assert reconstruction.is_new_schema is True
    assert reconstruction.resolved_position_encoding.relative.encoding is (
        RelativePositionEncoding.ROPE
    )
    assert reconstruction.resolved_position_encoding.relative.rope_coordinate_scale == 12345.0
    assert reconstruction.resolved_position_encoding.relative.rope_base == 4321.0
    assert torch.equal(
        reconstruction.base_model.attention.attention_layers[0].cross_chromosome_bias,
        source.attention.attention_layers[0].cross_chromosome_bias,
    )


def test_schema_v2_rope_ig_policy_and_provenance_are_content_only():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    reconstruction = _explain_reconstruction(config, metadata, source)
    run_config = reconstruction.effective_config

    assert resolve_ig_mode("auto", config=run_config, is_new_schema=True) is (
        ResolvedIGMode.CONTENT
    )
    assert resolve_ig_mode("content", config=run_config, is_new_schema=True) is (
        ResolvedIGMode.CONTENT
    )
    with pytest.raises(ValueError, match="custom positional execution"):
        explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, reconstruction)

    ig_metadata = explain._build_ig_run_metadata(
        requested_ig_mode="auto",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        reconstruction=reconstruction,
        content_dim=config.content_dim,
        n_steps=4,
        max_variants=3,
    )
    assert ig_metadata["absolute_position_encoding"] == "none"
    assert ig_metadata["relative_position_encoding"] == "rope"
    assert ig_metadata["position_encoding_metadata_source"] == "reconstructed_resolved_config"
    assert ig_metadata["resolved_ig_mode"] == "content"
    assert ig_metadata["attribution_feature_space"] == "content"
    assert ig_metadata["attribution_width"] == config.content_dim


def test_content_ig_for_schema_v2_rope_is_content_width_and_keeps_rope_state_fixed():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    _set_rope_biases(source)
    reconstruction = _explain_reconstruction(config, metadata, source)
    before = [
        layer.cross_chromosome_bias.detach().clone()
        for layer in reconstruction.base_model.attention.attention_layers
    ]
    chunk = _chunk(config)
    explainer = IntegratedGradientsExplainer(
        reconstruction.base_model,
        device="cpu",
        n_steps=4,
        ig_mode=ResolvedIGMode.CONTENT,
    )

    attributions, positions, _gene_ids, mask, chrom_ids = explain._attribute_chunk_for_ig(
        explainer=explainer,
        chunk=chunk,
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        device="cpu",
    )
    after = [
        layer.cross_chromosome_bias.detach().clone()
        for layer in reconstruction.base_model.attention.attention_layers
    ]

    assert attributions.shape == (1, 3, config.content_dim)
    assert torch.isfinite(attributions).all()
    assert torch.equal(positions, chunk["positions"].unsqueeze(0))
    assert torch.equal(chrom_ids, chunk["chrom_ids"].unsqueeze(0))
    assert torch.equal(mask, chunk["mask"].unsqueeze(0))
    for before_layer, after_layer in zip(before, after, strict=True):
        assert torch.equal(before_layer, after_layer)


def test_rope_position_sensitivity_changes_same_chromosome_attention():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    reconstruction = _explain_reconstruction(config, metadata, source)
    analyzer = AttentionAnalyzer(reconstruction.model, device="cpu")
    batch_a = _split_batch(
        config,
        positions=torch.tensor([[10, 20, 30]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0, 0]], dtype=torch.long),
    )
    batch_b = {
        **batch_a,
        "positions": torch.tensor([[10, 35, 30]], dtype=torch.long),
    }

    attention_a = analyzer.extract_attention_weights(
        None,
        batch_a["positions"],
        batch_a["gene_ids"],
        batch_a["mask"],
        chrom_ids=batch_a["chrom_ids"],
        content_features=batch_a["content_features"],
        absolute_position_features=batch_a["absolute_position_features"],
    )[0]
    attention_b = analyzer.extract_attention_weights(
        None,
        batch_b["positions"],
        batch_b["gene_ids"],
        batch_b["mask"],
        chrom_ids=batch_b["chrom_ids"],
        content_features=batch_b["content_features"],
        absolute_position_features=batch_b["absolute_position_features"],
    )[0]

    assert not torch.equal(attention_a, attention_b)


def test_attention_analyzer_extracts_schema_v2_rope_split_attention():
    config = _resolve_rope(cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _source_model(config)
    reconstruction = _explain_reconstruction(config, metadata, source)
    analyzer = AttentionAnalyzer(reconstruction.model, device="cpu")
    batch = _split_batch(config)

    attention = analyzer.extract_attention_weights(
        None,
        batch["positions"],
        batch["gene_ids"],
        batch["mask"],
        chrom_ids=batch["chrom_ids"],
        content_features=batch["content_features"],
        absolute_position_features=batch["absolute_position_features"],
    )

    assert len(attention) == MODEL_KWARGS["num_attention_layers"]
    assert attention[0].shape == (1, MODEL_KWARGS["num_heads"], 3, 3)
    assert torch.isfinite(attention[0]).all()


def test_learned_binned_rope_strict_lifecycle_restores_both_state_surfaces():
    config = _resolve_rope(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    metadata = _run_metadata(config)
    layout = _layout_from_metadata(metadata)
    source = _source_model(config, layout)
    _set_rope_biases(source)
    with torch.no_grad():
        values = torch.arange(
            source.absolute_position_embedding.weight.numel(),
            dtype=torch.float32,
        ).reshape_as(source.absolute_position_embedding.weight)
        source.absolute_position_embedding.weight.copy_(values / 10.0)
    serialized = _case_a_config(config, metadata)

    result = reconstruct_sieve_from_checkpoint(
        serialized,
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert "mapping" in metadata["position_encoding"]["chromosome"]
    assert "binning" in metadata["position_encoding"]["absolute"]
    assert metadata["position_encoding"]["relative"]["rope_coordinate_scale"] == 12345.0
    assert metadata["position_encoding"]["relative"]["rope_base"] == 4321.0
    assert torch.equal(
        result.base_model.absolute_position_embedding.weight,
        source.absolute_position_embedding.weight,
    )
    assert torch.equal(
        result.base_model.attention.attention_layers[0].cross_chromosome_bias,
        source.attention.attention_layers[0].cross_chromosome_bias,
    )

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            serialized,
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_direct_resolver_fixture_keeps_rope_non_default_values():
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.ROPE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            rope_coordinate_scale=12345.0,
            rope_base=4321.0,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=3,
    )

    assert config.relative.rope_coordinate_scale == 12345.0
    assert config.relative.rope_base == 4321.0
