"""Tests for Phase 9B RoPE relative-position runtime execution."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import LearnedBinnedAbsolutePositionLayout
from src.models.position_runtime import (
    NoRelativePositionRuntime,
    RopeRelativePositionRuntime,
    T5RelativePositionRuntime,
    build_relative_position_runtime,
)
from src.models.sieve import SIEVE

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 2,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 12,
    "dropout": 0.0,
    "num_covariates": 0,
    "classifier_type": "flatten",
}


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.ROPE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_chromosomes: int = 3,
    position_dim: int = 4,
    position_bin_size: int = 10,
    **kwargs,
):
    request_kwargs = {}
    if relative is RelativePositionEncoding.ROPE:
        request_kwargs.update(
            {
                "rope_coordinate_scale": kwargs.pop("rope_coordinate_scale", 10.0),
                "rope_base": kwargs.pop("rope_base", 100.0),
            }
        )
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        request_kwargs["position_dim"] = position_dim
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        request_kwargs["position_dim"] = position_dim
        request_kwargs["position_bin_size"] = position_bin_size
    request_kwargs.update(kwargs)
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            **request_kwargs,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=num_chromosomes,
    )


def _layout() -> LearnedBinnedAbsolutePositionLayout:
    return LearnedBinnedAbsolutePositionLayout(
        schema_version=1,
        coordinate_origin=1,
        layout="chromosome_local_contiguous",
        chromosome_lengths_bp=(20, 15, 12),
        bins_per_chromosome=(2, 2, 2),
        num_embeddings=6,
    )


def _model(config, *, layout=None, layers: int = 1) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=layers,
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        num_chromosomes=config.chromosome.num_chromosomes,
        num_covariates=MODEL_KWARGS["num_covariates"],
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
        learned_binned_position_layout=layout,
    )
    model.eval()
    return model


def _batch(config):
    content = torch.tensor(
        [
            [
                [0.2, 1.0, 0.0, 0.0, 1.0, 0.3, 0.7],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.6, 0.1],
                [0.5, 0.0, 0.0, 1.0, 0.0, 0.4, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    absolute_width = config.absolute.position_dim or 0
    absolute = torch.zeros(1, 4, absolute_width)
    positions = torch.tensor([[1, 6, 11, 0]], dtype=torch.long)
    gene_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])
    chrom_ids = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    return content, absolute, positions, gene_ids, mask, chrom_ids


def _base_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)


def _positional_state_keys(model: nn.Module) -> set[str]:
    tokens = (
        "position_bias.weight",
        "cross_chromosome_bias",
        "chrom_embedding.weight",
        "absolute_position_embedding.weight",
    )
    return {key for key in model.state_dict() if any(token in key for token in tokens)}


def _assert_parameterless_plain_runtime(runtime) -> None:
    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert is_dataclass(runtime)
    for field in fields(runtime):
        assert not isinstance(getattr(runtime, field.name), torch.Tensor)


def _runtime(
    *,
    head_dim: int = 4,
    coordinate_scale: float = 1.0,
    rope_base: float = 100.0,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
) -> RopeRelativePositionRuntime:
    return RopeRelativePositionRuntime(
        head_dim=head_dim,
        coordinate_scale=coordinate_scale,
        rope_base=rope_base,
        cross_chromosome_policy=cross_policy,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_dim": True}, "head_dim"),
        ({"head_dim": 0}, "head_dim"),
        ({"head_dim": 3}, "even"),
        ({"coordinate_scale": 0.0}, "coordinate_scale"),
        ({"coordinate_scale": math.inf}, "coordinate_scale"),
        ({"rope_base": 0.0}, "rope_base"),
        ({"rope_base": math.nan}, "rope_base"),
        ({"cross_policy": "separate"}, "CrossChromosomePolicy"),
    ],
)
def test_rope_runtime_rejects_invalid_direct_settings(kwargs, message):
    settings = {
        "head_dim": 4,
        "coordinate_scale": 10.0,
        "rope_base": 100.0,
        "cross_policy": CrossChromosomePolicy.SEPARATE,
    }
    settings.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _runtime(**settings)


def test_rope_factory_preserves_resolved_settings_and_requires_head_dim():
    config = _resolve_custom(rope_coordinate_scale=25.0, rope_base=50000.0)

    with pytest.raises(ValueError, match="head_dim"):
        build_relative_position_runtime(config)

    runtime = build_relative_position_runtime(config, head_dim=4)

    assert isinstance(runtime, RopeRelativePositionRuntime)
    assert runtime.head_dim == 4
    assert runtime.coordinate_scale == 25.0
    assert runtime.rope_base == 50000.0
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE
    _assert_parameterless_plain_runtime(runtime)


def test_rope_adjacent_pair_rotation_and_raw_position_convention():
    runtime = _runtime(head_dim=4, coordinate_scale=2.0, rope_base=16.0)
    values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = runtime.rotate(values, positions)
    first_angle = 0.5
    second_angle = 0.125
    expected = torch.tensor(
        [
            [
                [
                    [
                        1.0 * math.cos(first_angle) - 2.0 * math.sin(first_angle),
                        1.0 * math.sin(first_angle) + 2.0 * math.cos(first_angle),
                        3.0 * math.cos(second_angle) - 4.0 * math.sin(second_angle),
                        3.0 * math.sin(second_angle) + 4.0 * math.cos(second_angle),
                    ]
                ]
            ]
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(rotated, expected, rtol=0, atol=1e-12)


def test_rope_zero_position_helper_rotation_is_identity_reference_only():
    runtime = _runtime()
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    positions = torch.zeros(1, 3, dtype=torch.long)

    torch.testing.assert_close(runtime.rotate(values, positions), values, rtol=0, atol=0)


def test_equal_positions_preserve_query_key_dot_product():
    runtime = _runtime(coordinate_scale=3.0, rope_base=1000.0)
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.5, -1.0, 2.5, 0.25]]]], dtype=torch.float64)
    positions = torch.tensor([[17]], dtype=torch.long)

    rotated_query = runtime.rotate(query, positions)
    rotated_key = runtime.rotate(key, positions)

    torch.testing.assert_close(
        torch.sum(rotated_query * rotated_key, dim=-1),
        torch.sum(query * key, dim=-1),
        rtol=0,
        atol=1e-12,
    )


def test_unequal_positions_lock_relative_rotation_direction_and_score():
    runtime = _runtime(head_dim=2, coordinate_scale=1.0, rope_base=100.0)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float64),
    )
    expected_01 = math.cos(1.0 - 3.0) / math.sqrt(2.0)
    expected_10 = math.cos(3.0 - 1.0) / math.sqrt(2.0)

    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(expected_01, dtype=torch.float64),
    )
    torch.testing.assert_close(
        adjusted[0, 0, 1, 0],
        torch.tensor(expected_10, dtype=torch.float64),
    )


def test_coordinate_scale_affects_angle_exactly_once():
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[4]], dtype=torch.long)

    rotated_scale_1 = _runtime(head_dim=2, coordinate_scale=1.0).rotate(values, positions)
    rotated_scale_2 = _runtime(head_dim=2, coordinate_scale=2.0).rotate(values, positions)

    torch.testing.assert_close(
        rotated_scale_1[0, 0, 0],
        torch.tensor([math.cos(4.0), math.sin(4.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        rotated_scale_2[0, 0, 0],
        torch.tensor([math.cos(2.0), math.sin(2.0)], dtype=torch.float64),
    )


def test_rope_base_controls_second_adjacent_pair_frequency():
    values = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[4]], dtype=torch.long)

    base_16 = _runtime(head_dim=4, rope_base=16.0).rotate(values, positions)
    base_10000 = _runtime(head_dim=4, rope_base=10000.0).rotate(values, positions)

    torch.testing.assert_close(
        base_16[0, 0, 0, 2:],
        torch.tensor([math.cos(1.0), math.sin(1.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        base_10000[0, 0, 0, 2:],
        torch.tensor([math.cos(0.04), math.sin(0.04)], dtype=torch.float64),
    )


def test_rope_separate_cross_chromosome_uses_unrotated_base_plus_bias():
    runtime = _runtime(head_dim=2, coordinate_scale=1.0)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)
    bias = torch.tensor([0.25], dtype=torch.float64)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=bias,
    )

    rotated_cross = _base_scores(runtime.rotate(query, positions), runtime.rotate(key, positions))
    assert not torch.isclose(rotated_cross[0, 0, 0, 1], base_scores[0, 0, 0, 1])
    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 0, 1, 0], base_scores[0, 0, 1, 0] + 0.25)


def test_zero_cross_bias_makes_cross_score_equal_unrotated_base_score():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.5, 2.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.25, 1.0], [2.0, -0.5]]]], dtype=torch.float64)
    positions = torch.tensor([[5, 9]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])


def test_nonzero_cross_bias_broadcasts_independently_by_head():
    runtime = _runtime(head_dim=2)
    query = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    key = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float64)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=bias,
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 1, 0, 1], base_scores[0, 1, 0, 1] - 0.5)


def test_same_chromosome_score_ignores_cross_bias():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted_a = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.tensor([0.0], dtype=torch.float64),
    )
    adjusted_b = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.tensor([100.0], dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted_a, adjusted_b)


def test_rope_mask_routes_same_chromosome_scores_without_cross_bias():
    runtime = _runtime(head_dim=2, cross_policy=CrossChromosomePolicy.MASK)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
    )

    rotated_scores = _base_scores(runtime.rotate(query, positions), runtime.rotate(key, positions))
    torch.testing.assert_close(adjusted[0, 0, 0, 0], rotated_scores[0, 0, 0, 0])
    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])
    with pytest.raises(ValueError, match="must be None"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(1),
        )


def test_rope_runtime_rejects_missing_chrom_ids_bad_bias_and_position_float_dtype():
    runtime = _runtime()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    base_scores = _base_scores(query, key)
    positions = torch.tensor([[1, 2, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 999]], dtype=torch.long)

    with pytest.raises(ValueError, match="chrom_ids"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=None,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="shape"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(3),
        )
    with pytest.raises(ValueError, match="integer dtype"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions.float(),
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )


def test_rope_runtime_rejects_integer_base_scores_before_score_routing():
    runtime = _runtime()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    positions = torch.tensor([[1, 2, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    base_scores = torch.zeros(1, 2, 3, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="base_scores must use a floating dtype"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )


def test_rope_runtime_rejects_position_bias_and_nonfloating_cross_bias():
    runtime = _runtime()
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    base_scores = _base_scores(query, key)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="position_bias"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=nn.Embedding(2, 2),
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="floating"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2, dtype=torch.long),
        )


def test_non_rope_runtimes_reject_accidental_cross_bias_wiring():
    base_scores = torch.randn(1, 2, 2, 2)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="cross_chromosome_bias"):
        NoRelativePositionRuntime().adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="cross_chromosome_bias"):
        T5RelativePositionRuntime(8, 1000, CrossChromosomePolicy.MASK).adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=nn.Embedding(8, 2),
            cross_chromosome_bias=torch.zeros(2),
        )


def test_float64_input_uses_float64_rotation_and_matching_reference():
    runtime = _runtime(head_dim=2, coordinate_scale=2.0)
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = runtime.rotate(values, positions)

    assert rotated.dtype is torch.float64
    torch.testing.assert_close(
        rotated[0, 0, 0],
        torch.tensor([math.cos(0.5), math.sin(0.5)], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_promote_rope_rotation_to_float32(dtype):
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=dtype)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = _runtime(head_dim=2).rotate(values, positions)

    assert rotated.dtype is torch.float32


def test_low_precision_adjusted_score_dtype_matches_base_scores_when_supported():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float16)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float16)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    try:
        base_scores = _base_scores(query, key)
    except RuntimeError as exc:
        pytest.skip(f"CPU float16 matmul unsupported: {exc}")

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float16),
    )

    assert adjusted.dtype is base_scores.dtype


@pytest.mark.parametrize(
    ("relative", "expected_keys"),
    [
        (RelativePositionEncoding.NONE, set()),
        (
            RelativePositionEncoding.T5_BUCKET,
            {"attention.attention_layers.0.position_bias.weight"},
        ),
    ],
)
def test_representative_non_rope_state_key_sets_are_exact(relative, expected_keys):
    config = _resolve_custom(
        relative=relative,
        num_chromosomes=0 if relative is RelativePositionEncoding.NONE else 3,
        num_position_buckets=8 if relative is RelativePositionEncoding.T5_BUCKET else None,
        max_position_distance=1000 if relative is RelativePositionEncoding.T5_BUCKET else None,
    )
    model = _model(config)

    assert _positional_state_keys(model) == expected_keys


@pytest.mark.parametrize(
    ("cross_policy", "expected_keys"),
    [
        (CrossChromosomePolicy.MASK, set()),
        (
            CrossChromosomePolicy.SEPARATE,
            {"attention.attention_layers.0.cross_chromosome_bias"},
        ),
    ],
)
def test_rope_state_key_sets_are_exact_for_mask_and_separate(cross_policy, expected_keys):
    config = _resolve_custom(cross_policy=cross_policy)
    model = _model(config)
    layer = model.attention.attention_layers[0]

    assert layer.position_bias is None
    assert _positional_state_keys(model) == expected_keys
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
        assert torch.equal(
            layer.cross_chromosome_bias, torch.zeros_like(layer.cross_chromosome_bias)
        )
    else:
        assert layer.cross_chromosome_bias is None


def test_rope_chromosome_embedding_and_learned_binned_state_surfaces_are_independent():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        chromosome=ChromosomeEncoding.LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())

    assert _positional_state_keys(model) == {
        "absolute_position_embedding.weight",
        "attention.attention_layers.0.cross_chromosome_bias",
        "attention.attention_layers.0.chrom_embedding.weight",
    }


def test_learned_binned_absolute_and_rope_relative_compose_in_model_forward_backward():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)
    layer = model.attention.attention_layers[0]

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert model.absolute_position_embedding is not None
    assert layer.cross_chromosome_bias is not None
    assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
    assert torch.isfinite(content.grad).all()
    assert torch.isfinite(layer.cross_chromosome_bias.grad).all()


def test_chunked_rope_state_keys_use_base_model_prefix_naturally():
    from src.models.chunked_sieve import ChunkedSIEVEModel

    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    chunked = ChunkedSIEVEModel(_model(config))

    assert _positional_state_keys(chunked) == {
        "base_model.attention.attention_layers.0.cross_chromosome_bias"
    }


@pytest.mark.parametrize(
    ("chromosome", "cross_policy"),
    [
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.LEARNED, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.MASK),
    ],
)
def test_rope_model_forward_backward_return_attention_and_padding(chromosome, cross_policy):
    config = _resolve_custom(chromosome=chromosome, cross_policy=cross_policy)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)

    logits, intermediates = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    attention = intermediates["attention_weights"][0]
    assert logits.shape == (1, 1)
    assert attention.shape == (1, MODEL_KWARGS["num_heads"], 4, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(content.grad).all()
    if cross_policy is CrossChromosomePolicy.MASK:
        assert torch.all(attention[0, :, 0, 1] == 0)
        assert torch.all(attention[0, :, 0, 0] > 0)


def test_rope_attention_rejects_real_position_zero_but_allows_padded_zero():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)

    model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    bad_positions = positions.clone()
    bad_positions[0, 1] = 0
    with pytest.raises(ValueError, match="real RoPE positions"):
        model(
            None,
            bad_positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_rope_requires_chrom_ids_in_model_even_without_chromosome_embedding():
    config = _resolve_custom(chromosome=ChromosomeEncoding.NONE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)

    with pytest.raises(ValueError, match="chrom_ids"):
        model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_rope_cross_bias_receives_gradient_with_cross_chromosome_pairs():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    grad = model.attention.attention_layers[0].cross_chromosome_bias.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_rope_cross_bias_has_no_nonzero_gradient_for_same_chromosome_batch():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)
    same_chrom_ids = torch.tensor([[0, 0, 0, 999]], dtype=torch.long)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    grad = model.attention.attention_layers[0].cross_chromosome_bias.grad
    if grad is not None:
        assert torch.all(grad == 0)


def test_rope_runtime_api_has_no_value_tensor_argument():
    parameter_names = RopeRelativePositionRuntime.adjust_attention_scores.__code__.co_varnames

    assert "value" not in parameter_names
