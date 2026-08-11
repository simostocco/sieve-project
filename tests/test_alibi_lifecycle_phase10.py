"""Phase 10C ALiBi lifecycle validation without production changes."""

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
    RelativePositionEncoding,
    ResolvedIGMode,
    resolved_position_encoding_from_dict,
)
from src.encoding.position_layout import learned_binned_layout_from_position_encoding_dict
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.ig_mode import resolve_ig_mode
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.reconstruction import reconstruct_sieve_from_checkpoint

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 2,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 256,
    "dropout": 0.0,
    "aggregation": "max",
    "aggregation_method": "mean",
    "num_covariates": 0,
    "classifier_type": "flatten",
}
CHROM_INDEX = {"1": 0, "2": 1, "X": 2}
SWAPPED_CHROM_INDEX = {"1": 1, "2": 0, "X": 2}
DIFFERENT_CHROM_INDEX = {"A": 0, "B": 1, "C": 2}
ALIBI_DISTANCE_SCALE = 23456.0


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


def _alibi_args(
    *,
    relative: RelativePositionEncoding,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_attention_layers: int | None = None,
) -> argparse.Namespace:
    args = [
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        absolute.value,
        "--relative-position-encoding",
        relative.value,
        "--chromosome-encoding",
        chromosome.value,
        "--cross-chromosome-policy",
        cross_policy.value,
        "--alibi-distance-function",
        "linear",
        "--alibi-distance-scale",
        str(ALIBI_DISTANCE_SCALE),
    ]
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        args.extend(["--position-dim", "4"])
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        args.extend(["--position-dim", "8", "--position-bin-size", "100000000"])
    return _args(*args, num_attention_layers=num_attention_layers)


def _resolve_alibi(
    *,
    relative: RelativePositionEncoding,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_attention_layers: int | None = None,
):
    return train.prepare_training_position_encoding(
        _alibi_args(
            relative=relative,
            absolute=absolute,
            chromosome=chromosome,
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


def _training_model(config, *, layout=None, num_attention_layers: int | None = None):
    model = train.create_training_model(
        args=_alibi_args(
            relative=config.relative.encoding,
            absolute=config.absolute.encoding,
            chromosome=config.chromosome.encoding,
            cross_policy=config.chromosome.cross_chromosome_policy,
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


def _checkpoint(model, *, metadata=None):
    checkpoint = {"model_state_dict": copy.deepcopy(model.state_dict())}
    if metadata is not None:
        checkpoint["metadata"] = copy.deepcopy(metadata)
    return checkpoint


def _dataset(chrom_index=CHROM_INDEX):
    return SimpleNamespace(
        num_genes=5,
        num_chromosomes=3,
        chrom_index=chrom_index,
    )


def _split_batch(config, *, positions=None, chrom_ids=None):
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
    absolute = torch.zeros(1, 3, config.absolute.position_dim or 0)
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


def _chunk(config):
    batch = _split_batch(
        config,
        positions=torch.tensor([[10, 20, 0]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0, 0]], dtype=torch.long),
    )
    batch["mask"] = torch.tensor([[True, True, False]])
    return {key: value.squeeze(0) for key, value in batch.items()}


def _run_split_forward(model, config, *, batch=None, return_attention=False):
    batch = _split_batch(config) if batch is None else batch
    return model(
        None,
        batch["positions"],
        batch["gene_ids"],
        batch["mask"],
        chrom_ids=batch["chrom_ids"],
        return_attention=return_attention,
        content_features=batch["content_features"],
        absolute_position_features=batch["absolute_position_features"],
    )


def _relative_state_keys(model) -> set[str]:
    return {
        key
        for key in model.state_dict()
        if (
            "position_bias.weight" in key
            or "cross_chromosome_bias" in key
            or "alibi_slope_logits" in key
        )
    }


def _assert_state_exact(source_state, target_state):
    assert set(target_state) == set(source_state)
    for key, tensor in source_state.items():
        assert torch.equal(target_state[key], tensor), key


def _layers(model):
    base_model = model.base_model if isinstance(model, ChunkedSIEVEModel) else model
    return base_model.attention.attention_layers


def _set_alibi_state(model, *, learned: bool, values_by_layer=None):
    layers = _layers(model)
    if values_by_layer is None:
        values_by_layer = [
            {
                "logits": torch.tensor([-3.0 - idx, -1.5 - idx], dtype=torch.float32),
                "cross": torch.tensor([0.25 + idx, -0.50 - idx], dtype=torch.float32),
            }
            for idx in range(len(layers))
        ]
    with torch.no_grad():
        for layer, values in zip(layers, values_by_layer, strict=True):
            if learned:
                layer.alibi_slope_logits.copy_(values["logits"])
            if layer.cross_chromosome_bias is not None:
                layer.cross_chromosome_bias.copy_(values["cross"])


def _explain_reconstruction(config, metadata, source, *, chrom_index=CHROM_INDEX):
    return explain._reconstruct_model_for_explanation(
        _case_a_config(config, metadata),
        _checkpoint(source),
        _dataset(chrom_index),
    )


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_alibi_training_resolution_metadata_and_round_trip(relative, cross_policy):
    config = _resolve_alibi(relative=relative, cross_policy=cross_policy)
    metadata = _run_metadata(config)
    serialized = metadata["position_encoding"]
    execution = metadata["position_encoding_execution"]
    parsed = resolved_position_encoding_from_dict(
        serialized,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
    )

    assert config.relative.encoding is relative
    assert config.relative.alibi_distance_function.value == "linear"
    assert config.relative.alibi_distance_scale == ALIBI_DISTANCE_SCALE
    assert config.chromosome.cross_chromosome_policy is cross_policy
    assert config.input_dim == config.content_dim
    assert serialized["relative"]["type"] == relative.value
    assert serialized["relative"]["alibi_distance_function"] == "linear"
    assert serialized["relative"]["alibi_distance_scale"] == ALIBI_DISTANCE_SCALE
    assert serialized["relative"]["total_bias_rows"] is None
    assert execution["relative_position_encoding"] == relative.value
    assert execution["position_bias_rows"] is None
    assert execution["cross_chromosome_policy"] == cross_policy.value
    assert parsed == config


@pytest.mark.parametrize(
    ("relative", "cross_policy", "expected_keys"),
    [
        (RelativePositionEncoding.ALIBI_FIXED, CrossChromosomePolicy.MASK, set()),
        (
            RelativePositionEncoding.ALIBI_FIXED,
            CrossChromosomePolicy.SEPARATE,
            {"base_model.attention.attention_layers.0.cross_chromosome_bias"},
        ),
        (
            RelativePositionEncoding.ALIBI_LEARNED,
            CrossChromosomePolicy.MASK,
            {"base_model.attention.attention_layers.0.alibi_slope_logits"},
        ),
        (
            RelativePositionEncoding.ALIBI_LEARNED,
            CrossChromosomePolicy.SEPARATE,
            {
                "base_model.attention.attention_layers.0.alibi_slope_logits",
                "base_model.attention.attention_layers.0.cross_chromosome_bias",
            },
        ),
    ],
)
def test_training_created_alibi_state_surfaces_are_exact(relative, cross_policy, expected_keys):
    config = _resolve_alibi(relative=relative, cross_policy=cross_policy)
    model = _training_model(config)
    layer = model.base_model.attention.attention_layers[0]

    assert _relative_state_keys(model) == expected_keys
    assert layer.position_bias is None
    if relative is RelativePositionEncoding.ALIBI_LEARNED:
        assert layer.alibi_slope_logits.shape == (MODEL_KWARGS["num_heads"],)
        assert layer.alibi_slope_logits.requires_grad is True
        assert torch.isfinite(layer.alibi_slope_logits).all()
        torch.testing.assert_close(
            torch.nn.functional.softplus(layer.alibi_slope_logits.detach()),
            torch.tensor((0.0625, 0.00390625), dtype=layer.alibi_slope_logits.dtype),
        )
    else:
        assert layer.alibi_slope_logits is None


@pytest.mark.parametrize(
    ("relative", "cross_policy"),
    [
        (RelativePositionEncoding.ALIBI_FIXED, CrossChromosomePolicy.SEPARATE),
        (RelativePositionEncoding.ALIBI_FIXED, CrossChromosomePolicy.MASK),
        (RelativePositionEncoding.ALIBI_LEARNED, CrossChromosomePolicy.SEPARATE),
        (RelativePositionEncoding.ALIBI_LEARNED, CrossChromosomePolicy.MASK),
    ],
)
def test_training_created_alibi_forward_backward(relative, cross_policy):
    config = _resolve_alibi(relative=relative, cross_policy=cross_policy)
    model = _training_model(config)
    batch = _split_batch(
        config,
        chrom_ids=torch.tensor([[0, 0, 1]], dtype=torch.long),
    )
    batch["content_features"] = batch["content_features"].clone().requires_grad_(True)

    logits, _ = _run_split_forward(model, config, batch=batch)
    logits.sum().backward()

    layer = model.base_model.attention.attention_layers[0]
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(batch["content_features"].grad).all()
    if relative is RelativePositionEncoding.ALIBI_LEARNED:
        assert layer.alibi_slope_logits.grad is not None
        assert torch.isfinite(layer.alibi_slope_logits.grad).all()
        assert torch.any(layer.alibi_slope_logits.grad != 0)
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.grad is not None
        assert torch.isfinite(layer.cross_chromosome_bias.grad).all()
        assert torch.any(layer.cross_chromosome_bias.grad != 0)
    else:
        assert layer.cross_chromosome_bias is None


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_base_schema_v2_alibi_separate_strict_round_trip(relative):
    learned = relative is RelativePositionEncoding.ALIBI_LEARNED
    config = _resolve_alibi(relative=relative, cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    _set_alibi_state(source, learned=learned)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.is_chunked_checkpoint is False
    assert result.resolved_position_encoding.relative.encoding is relative
    assert result.resolved_position_encoding.relative.alibi_distance_function.value == "linear"
    assert result.resolved_position_encoding.relative.alibi_distance_scale == ALIBI_DISTANCE_SCALE
    _assert_state_exact(source.state_dict(), result.model.state_dict())
    layer = result.model.attention.attention_layers[0]
    source_layer = source.attention.attention_layers[0]
    if learned:
        assert torch.equal(layer.alibi_slope_logits, source_layer.alibi_slope_logits)
    assert torch.equal(layer.cross_chromosome_bias, source_layer.cross_chromosome_bias)


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_chunked_schema_v2_alibi_separate_strict_round_trip(relative):
    learned = relative is RelativePositionEncoding.ALIBI_LEARNED
    config = _resolve_alibi(relative=relative, cross_policy=CrossChromosomePolicy.SEPARATE)
    metadata = _run_metadata(config)
    source = _training_model(config)
    _set_alibi_state(source, learned=learned)
    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert result.is_chunked_checkpoint is True
    assert isinstance(result.model, ChunkedSIEVEModel)
    _assert_state_exact(source.state_dict(), result.model.state_dict())
    keys = set(result.model.state_dict())
    assert "base_model.attention.attention_layers.0.cross_chromosome_bias" in keys
    if learned:
        assert "base_model.attention.attention_layers.0.alibi_slope_logits" in keys


@pytest.mark.parametrize(
    ("cross_policy", "mutator"),
    [
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.pop("attention.attention_layers.0.cross_chromosome_bias"),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.cross_chromosome_bias",
                torch.zeros(3),
            ),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.alibi_slope_logits",
                torch.zeros(2),
            ),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.position_bias.weight",
                torch.zeros(8, 2),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.alibi_slope_logits",
                torch.zeros(2),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.cross_chromosome_bias",
                torch.zeros(2),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.position_bias.weight",
                torch.zeros(8, 2),
            ),
        ),
    ],
)
def test_fixed_alibi_strict_corrupt_state_rejects(cross_policy, mutator):
    config = _resolve_alibi(
        relative=RelativePositionEncoding.ALIBI_FIXED,
        cross_policy=cross_policy,
    )
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    state = copy.deepcopy(source.state_dict())
    if cross_policy is CrossChromosomePolicy.MASK:
        assert not any(
            token in key
            for token in ("alibi_slope_logits", "cross_chromosome_bias", "position_bias.weight")
            for key in state
        )
    mutator(state)

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            {"model_state_dict": state},
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


@pytest.mark.parametrize(
    ("cross_policy", "mutator"),
    [
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.pop("attention.attention_layers.0.alibi_slope_logits"),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.alibi_slope_logits",
                torch.zeros(3),
            ),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.pop("attention.attention_layers.0.cross_chromosome_bias"),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.cross_chromosome_bias",
                torch.zeros(3),
            ),
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.position_bias.weight",
                torch.zeros(8, 2),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.pop("attention.attention_layers.0.alibi_slope_logits"),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.alibi_slope_logits",
                torch.zeros(3),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.cross_chromosome_bias",
                torch.zeros(2),
            ),
        ),
        (
            CrossChromosomePolicy.MASK,
            lambda state: state.__setitem__(
                "attention.attention_layers.0.position_bias.weight",
                torch.zeros(8, 2),
            ),
        ),
    ],
)
def test_learned_alibi_strict_corrupt_state_rejects(cross_policy, mutator):
    config = _resolve_alibi(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=cross_policy,
    )
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    _set_alibi_state(source, learned=True)
    state = copy.deepcopy(source.state_dict())
    if cross_policy is CrossChromosomePolicy.MASK:
        assert "attention.attention_layers.0.alibi_slope_logits" in state
        assert not any("cross_chromosome_bias" in key for key in state)
        assert not any("position_bias.weight" in key for key in state)
    mutator(state)

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            {"model_state_dict": state},
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


def test_multilayer_learned_alibi_state_is_independent_and_required():
    config = _resolve_alibi(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
        num_attention_layers=2,
    )
    metadata = _run_metadata(config)
    source = _training_model(config, num_attention_layers=2).base_model
    _set_alibi_state(source, learned=True)
    state = source.state_dict()
    required = {
        "attention.attention_layers.0.alibi_slope_logits",
        "attention.attention_layers.1.alibi_slope_logits",
        "attention.attention_layers.0.cross_chromosome_bias",
        "attention.attention_layers.1.cross_chromosome_bias",
    }

    assert required <= set(state)
    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata, num_attention_layers=2),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    _assert_state_exact(source.state_dict(), result.model.state_dict())

    corrupt = copy.deepcopy(state)
    corrupt.pop("attention.attention_layers.1.alibi_slope_logits")
    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata, num_attention_layers=2),
            {"model_state_dict": corrupt},
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


@pytest.mark.parametrize(
    ("relative", "field", "value"),
    [
        (RelativePositionEncoding.ALIBI_FIXED, "alibi_distance_function", "log1p"),
        (RelativePositionEncoding.ALIBI_FIXED, "alibi_distance_scale", 123.0),
        (RelativePositionEncoding.ALIBI_LEARNED, "alibi_distance_function", "log1p"),
        (RelativePositionEncoding.ALIBI_LEARNED, "alibi_distance_scale", 123.0),
    ],
)
def test_alibi_config_checkpoint_metadata_conflicts_reject_before_state_authority(
    relative,
    field,
    value,
):
    config = _resolve_alibi(relative=relative)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    checkpoint_metadata = copy.deepcopy(metadata)
    checkpoint_metadata["position_encoding"]["relative"][field] = value

    with pytest.raises(ValueError, match="checkpoint metadata conflict"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source, metadata=checkpoint_metadata),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=CHROM_INDEX,
        )


def test_historical_and_transitional_compatibility_do_not_infer_alibi_state():
    config = _resolve_alibi(relative=RelativePositionEncoding.ALIBI_LEARNED)
    metadata = _run_metadata(config)
    old_model = train.create_model(
        input_dim=71,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
    ).base_model
    case_b = reconstruct_sieve_from_checkpoint(
        {**MODEL_KWARGS, "input_dim": 71},
        _checkpoint(old_model),
        num_genes=5,
    )
    case_c = reconstruct_sieve_from_checkpoint(
        {
            **MODEL_KWARGS,
            "input_dim": 71,
            "position_encoding": metadata["position_encoding"],
            "position_encoding_execution": {"resolved_config_applied_to_model": False},
        },
        _checkpoint(old_model),
        num_genes=5,
    )

    assert case_b.is_new_schema is False
    assert case_c.is_new_schema is False
    assert not any("alibi_slope_logits" in key for key in case_b.model.state_dict())
    assert not any("cross_chromosome_bias" in key for key in case_c.model.state_dict())


def test_alibi_only_reconstruction_does_not_require_exact_chromosome_name_identity():
    config = _resolve_alibi(relative=RelativePositionEncoding.ALIBI_LEARNED)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=DIFFERENT_CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


@pytest.mark.parametrize(
    ("chromosome", "absolute"),
    [
        (ChromosomeEncoding.LEARNED, AbsolutePositionEncoding.NONE),
        (ChromosomeEncoding.NONE, AbsolutePositionEncoding.LEARNED_BINNED),
    ],
)
def test_existing_row_identity_guards_remain_orthogonal_to_alibi(chromosome, absolute):
    config = _resolve_alibi(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        chromosome=chromosome,
        absolute=absolute,
    )
    metadata = _run_metadata(config)
    layout = (
        _layout_from_metadata(metadata)
        if absolute is AbsolutePositionEncoding.LEARNED_BINNED
        else None
    )
    source = _training_model(config, layout=layout).base_model

    matching = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    assert matching.is_new_schema is True

    with pytest.raises(ValueError, match="chromosome|mapping|chrom_index|exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_explanation_reconstruction_and_ig_policy_for_alibi(relative):
    config = _resolve_alibi(relative=relative)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    _set_alibi_state(source, learned=relative is RelativePositionEncoding.ALIBI_LEARNED)
    reconstruction = _explain_reconstruction(config, metadata, source)

    assert reconstruction.resolved_position_encoding.relative.encoding is relative
    assert (
        reconstruction.resolved_position_encoding.relative.alibi_distance_scale
        == ALIBI_DISTANCE_SCALE
    )
    if relative is RelativePositionEncoding.ALIBI_LEARNED:
        assert torch.equal(
            reconstruction.base_model.attention.attention_layers[0].alibi_slope_logits,
            source.attention.attention_layers[0].alibi_slope_logits,
        )
    assert torch.equal(
        reconstruction.base_model.attention.attention_layers[0].cross_chromosome_bias,
        source.attention.attention_layers[0].cross_chromosome_bias,
    )
    assert resolve_ig_mode("auto", config=reconstruction.effective_config, is_new_schema=True) is (
        ResolvedIGMode.CONTENT
    )
    assert resolve_ig_mode(
        "content", config=reconstruction.effective_config, is_new_schema=True
    ) is (ResolvedIGMode.CONTENT)
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
    assert ig_metadata["relative_position_encoding"] == relative.value
    assert ig_metadata["position_encoding_metadata_source"] == "reconstructed_resolved_config"
    assert ig_metadata["resolved_ig_mode"] == "content"
    assert ig_metadata["attribution_feature_space"] == "content"
    assert ig_metadata["attribution_width"] == config.content_dim
    assert ig_metadata["absolute_position_encoding"] == "none"
    assert "alibi_slope_logits" not in ig_metadata


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_content_ig_for_alibi_is_content_width_and_keeps_parameters_fixed(relative):
    config = _resolve_alibi(relative=relative)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
    _set_alibi_state(source, learned=relative is RelativePositionEncoding.ALIBI_LEARNED)
    reconstruction = _explain_reconstruction(config, metadata, source)
    before = {
        key: value.detach().clone()
        for key, value in reconstruction.base_model.state_dict().items()
        if "alibi_slope_logits" in key or "cross_chromosome_bias" in key
    }
    explainer = IntegratedGradientsExplainer(
        reconstruction.base_model,
        device="cpu",
        n_steps=4,
        ig_mode=ResolvedIGMode.CONTENT,
    )
    chunk = _chunk(config)

    attributions, positions, _gene_ids, mask, chrom_ids = explain._attribute_chunk_for_ig(
        explainer=explainer,
        chunk=chunk,
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        device="cpu",
    )
    after = reconstruction.base_model.state_dict()

    assert attributions.shape == (1, 3, config.content_dim)
    assert torch.isfinite(attributions).all()
    assert torch.equal(positions, chunk["positions"].unsqueeze(0))
    assert torch.equal(chrom_ids, chunk["chrom_ids"].unsqueeze(0))
    assert torch.equal(mask, chunk["mask"].unsqueeze(0))
    for key, value in before.items():
        assert torch.equal(value, after[key])


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_reconstructed_alibi_attention_changes_with_same_chromosome_distance(relative):
    config = _resolve_alibi(relative=relative)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
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


@pytest.mark.parametrize(
    ("relative", "cross_policy"),
    [
        (RelativePositionEncoding.ALIBI_FIXED, CrossChromosomePolicy.SEPARATE),
        (RelativePositionEncoding.ALIBI_LEARNED, CrossChromosomePolicy.SEPARATE),
        (RelativePositionEncoding.ALIBI_FIXED, CrossChromosomePolicy.MASK),
    ],
)
def test_attention_analyzer_extracts_alibi_attention(relative, cross_policy):
    config = _resolve_alibi(relative=relative, cross_policy=cross_policy)
    metadata = _run_metadata(config)
    source = _training_model(config).base_model
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
    if cross_policy is CrossChromosomePolicy.MASK:
        assert torch.all(attention[0][0, :, 0, 1] == 0)


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_learned_binned_alibi_strict_lifecycle_restores_all_state(relative):
    learned = relative is RelativePositionEncoding.ALIBI_LEARNED
    config = _resolve_alibi(
        relative=relative,
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
    )
    metadata = _run_metadata(config)
    layout = _layout_from_metadata(metadata)
    source = _training_model(config, layout=layout).base_model
    with torch.no_grad():
        source.absolute_position_embedding.weight.copy_(
            torch.arange(
                source.absolute_position_embedding.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(source.absolute_position_embedding.weight)
            / 10.0
        )
    _set_alibi_state(source, learned=learned)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    _assert_state_exact(source.state_dict(), result.model.state_dict())
    assert torch.equal(
        result.model.absolute_position_embedding.weight,
        source.absolute_position_embedding.weight,
    )
    with pytest.raises(ValueError, match="chromosome|mapping|chrom_index|exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )
