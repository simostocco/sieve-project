"""Tests for Phase 10B1 fixed ALiBi relative-position execution."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import LearnedBinnedAbsolutePositionLayout
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.position_runtime import (
    FixedAlibiRelativePositionRuntime,
    build_alibi_fixed_slopes,
    build_relative_position_runtime,
    validate_attention_runtime_support,
    validate_phase7_runtime_support,
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
    relative: RelativePositionEncoding = RelativePositionEncoding.ALIBI_FIXED,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_chromosomes: int = 3,
    position_dim: int = 4,
    position_bin_size: int = 10,
    alibi_distance_function: AlibiDistanceFunction = AlibiDistanceFunction.LOG1P,
    alibi_distance_scale: float = 10.0,
):
    request_kwargs = {}
    if relative is RelativePositionEncoding.ALIBI_FIXED:
        request_kwargs.update(
            {
                "alibi_distance_function": alibi_distance_function,
                "alibi_distance_scale": alibi_distance_scale,
            }
        )
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        request_kwargs["position_dim"] = position_dim
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        request_kwargs["position_dim"] = position_dim
        request_kwargs["position_bin_size"] = position_bin_size
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


def _model(config, *, layout=None) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
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


def _runtime(
    *,
    num_heads: int = 2,
    distance_function: AlibiDistanceFunction = AlibiDistanceFunction.LINEAR,
    distance_scale: float = 10.0,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
) -> FixedAlibiRelativePositionRuntime:
    return FixedAlibiRelativePositionRuntime(
        num_heads=num_heads,
        distance_function=distance_function,
        distance_scale=distance_scale,
        cross_chromosome_policy=cross_policy,
    )


def _base_scores(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
            ]
        ],
        dtype=dtype,
    )


def _unused_query_key(dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 3, 4, dtype=dtype)
    key = torch.randn(1, 2, 3, 4, dtype=dtype)
    return query, key


def _adjust(
    runtime: FixedAlibiRelativePositionRuntime,
    base_scores: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    chrom_ids: torch.Tensor | None = None,
    cross_chromosome_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    query_key_dtype = base_scores.dtype if base_scores.dtype.is_floating_point else torch.float32
    query, key = _unused_query_key(query_key_dtype)
    return runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=(
            torch.tensor([[10, 30, 50]], dtype=torch.long) if positions is None else positions
        ),
        chrom_ids=(torch.tensor([[0, 0, 0]], dtype=torch.long) if chrom_ids is None else chrom_ids),
        position_bias=None,
        cross_chromosome_bias=cross_chromosome_bias,
    )


def _positional_state_keys(model: nn.Module) -> set[str]:
    tokens = (
        "position_bias.weight",
        "cross_chromosome_bias",
        "chrom_embedding.weight",
        "absolute_position_embedding.weight",
        "alibi_slope",
    )
    return {key for key in model.state_dict() if any(token in key for token in tokens)}


def _assert_parameterless_plain_runtime(runtime) -> None:
    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert is_dataclass(runtime)
    for field in fields(runtime):
        assert not isinstance(getattr(runtime, field.name), torch.Tensor)


@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [
        (1, (0.00390625,)),
        (2, (0.0625, 0.00390625)),
        (4, (0.25, 0.0625, 0.015625, 0.00390625)),
        (6, (0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125)),
    ],
)
def test_fixed_alibi_slope_schedule_is_exact(num_heads, expected):
    assert build_alibi_fixed_slopes(num_heads) == expected


@pytest.mark.parametrize("num_heads", [True, 0, -1])
def test_fixed_alibi_slope_schedule_rejects_invalid_head_counts(num_heads):
    with pytest.raises(ValueError, match="num_heads"):
        build_alibi_fixed_slopes(num_heads)


def test_fixed_alibi_runtime_is_plain_and_stores_only_python_slope_tuple():
    runtime = _runtime(num_heads=6)

    _assert_parameterless_plain_runtime(runtime)
    assert runtime.fixed_slopes == (
        0.25,
        0.0625,
        0.015625,
        0.00390625,
        0.5,
        0.125,
    )
    assert len(runtime.fixed_slopes) == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_heads": True}, "num_heads"),
        ({"num_heads": 0}, "num_heads"),
        ({"distance_function": "linear"}, "distance_function"),
        ({"distance_scale": 0.0}, "distance_scale"),
        ({"distance_scale": math.inf}, "distance_scale"),
        ({"cross_policy": "separate"}, "cross_chromosome_policy"),
    ],
)
def test_fixed_alibi_runtime_rejects_invalid_direct_construction(kwargs, message):
    valid = {
        "num_heads": 2,
        "distance_function": AlibiDistanceFunction.LINEAR,
        "distance_scale": 10.0,
        "cross_policy": CrossChromosomePolicy.SEPARATE,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _runtime(**valid)


def test_factory_builds_fixed_alibi_runtime_with_resolved_settings():
    config = _resolve_custom(
        alibi_distance_function=AlibiDistanceFunction.LOG1P,
        alibi_distance_scale=25.0,
    )

    with pytest.raises(ValueError, match="num_heads"):
        build_relative_position_runtime(config)
    runtime = build_relative_position_runtime(config, num_heads=2)

    assert isinstance(runtime, FixedAlibiRelativePositionRuntime)
    assert runtime.distance_function is AlibiDistanceFunction.LOG1P
    assert runtime.distance_scale == 25.0
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE


def test_fixed_alibi_linear_zero_distance_and_symmetry_are_exact():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(2))

    torch.testing.assert_close(adjusted[0, 0, 0, 0], _base_scores()[0, 0, 0, 0])
    torch.testing.assert_close(adjusted[0, 1, 1, 1], _base_scores()[0, 1, 1, 1])
    torch.testing.assert_close(
        _base_scores()[0, 0, 0, 1] - adjusted[0, 0, 0, 1],
        _base_scores()[0, 0, 1, 0] - adjusted[0, 0, 1, 0],
    )


def test_fixed_alibi_linear_transform_is_hand_computable():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(2))

    # positions 10 and 30 have distance 20, scale 10, transformed distance 2.
    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(2.0 - 2 * 0.0625, dtype=torch.float64),
    )
    torch.testing.assert_close(
        adjusted[0, 1, 0, 1],
        torch.tensor(-2.0 - 2 * 0.00390625, dtype=torch.float64),
    )


def test_fixed_alibi_log1p_transform_and_scale_placement_are_hand_computable():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LOG1P)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(2))
    expected_transform = math.log1p(20.0 / 10.0)
    wrong_transform = math.log1p(20.0) / 10.0

    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(2.0 - 0.0625 * expected_transform, dtype=torch.float64),
    )
    assert not math.isclose(expected_transform, wrong_transform)


def test_larger_same_chromosome_distance_has_stronger_negative_penalty():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(2))
    penalty_near = _base_scores()[0, 0, 0, 1] - adjusted[0, 0, 0, 1]
    penalty_far = _base_scores()[0, 0, 0, 2] - adjusted[0, 0, 0, 2]

    assert penalty_far > penalty_near > 0


def test_fixed_alibi_multiple_heads_broadcast_distinct_slopes():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(2))

    torch.testing.assert_close(adjusted[0, 0, 0, 1], torch.tensor(1.875, dtype=torch.float64))
    torch.testing.assert_close(
        adjusted[0, 1, 0, 1],
        torch.tensor(-2.0078125, dtype=torch.float64),
    )


def test_fixed_alibi_separate_cross_pairs_use_bias_without_distance_penalty():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.tensor([0.25, -0.5], dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 1, 0, 1], base_scores[0, 1, 0, 1] - 0.5)
    assert adjusted[0, 0, 0, 2] != base_scores[0, 0, 0, 2]


def test_fixed_alibi_separate_zero_cross_bias_leaves_cross_score_as_base():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])


def test_fixed_alibi_mask_runtime_leaves_cross_pairs_for_outer_attention_mask():
    runtime = _runtime(cross_policy=CrossChromosomePolicy.MASK)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])
    assert adjusted[0, 0, 0, 2] != base_scores[0, 0, 0, 2]


@pytest.mark.parametrize(
    ("cross_policy", "bias", "message"),
    [
        (CrossChromosomePolicy.SEPARATE, None, "cross_chromosome_bias"),
        (CrossChromosomePolicy.SEPARATE, torch.zeros(3), "shape"),
        (CrossChromosomePolicy.SEPARATE, torch.zeros(2, dtype=torch.long), "floating"),
        (CrossChromosomePolicy.MASK, torch.zeros(2), "must be None"),
    ],
)
def test_fixed_alibi_cross_bias_contract_rejects_invalid_inputs(cross_policy, bias, message):
    runtime = _runtime(cross_policy=cross_policy)

    with pytest.raises(ValueError, match=message):
        _adjust(
            runtime,
            _base_scores(),
            chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
            cross_chromosome_bias=bias,
        )


@pytest.mark.parametrize(
    ("base_scores", "message"),
    [
        (torch.zeros(1, 2, 3, 3, dtype=torch.long), "floating"),
        (torch.zeros(1, 2, 3), "shape"),
        (torch.zeros(1, 3, 3, 3), "head"),
        (torch.zeros(1, 2, 3, 4), "square"),
    ],
)
def test_fixed_alibi_rejects_invalid_base_scores(base_scores, message):
    with pytest.raises(ValueError, match=message):
        _adjust(_runtime(), base_scores, cross_chromosome_bias=torch.zeros(2))


def test_fixed_alibi_rejects_float_positions_and_missing_chrom_ids():
    runtime = _runtime()
    base_scores = _base_scores()

    with pytest.raises(ValueError, match="integer"):
        _adjust(
            runtime,
            base_scores,
            positions=torch.tensor([[10.0, 30.0, 50.0]]),
            cross_chromosome_bias=torch.zeros(2),
        )
    query, key = _unused_query_key()
    with pytest.raises(ValueError, match="chrom_ids"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=torch.tensor([[10, 30, 50]], dtype=torch.long),
            chrom_ids=None,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )


def test_fixed_alibi_accepts_narrow_integer_positions_without_overflowing_subtraction():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    adjusted = _adjust(
        runtime,
        base_scores,
        positions=torch.tensor([[1, 100]], dtype=torch.int8),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], torch.tensor(-0.61875, dtype=torch.float64))


def test_fixed_alibi_preserves_one_bp_distance_for_large_float32_coordinates():
    runtime = _runtime(
        num_heads=2,
        distance_function=AlibiDistanceFunction.LINEAR,
        distance_scale=1.0,
    )
    adjusted = _adjust(
        runtime,
        torch.zeros(1, 2, 2, 2, dtype=torch.float32),
        positions=torch.tensor([[250_000_001, 250_000_002]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float32),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], torch.tensor(-0.0625))
    torch.testing.assert_close(adjusted[0, 0, 1, 0], torch.tensor(-0.0625))
    torch.testing.assert_close(adjusted[0, 1, 0, 1], torch.tensor(-0.00390625))
    torch.testing.assert_close(adjusted[0, 1, 1, 0], torch.tensor(-0.00390625))


def test_fixed_alibi_runtime_allows_padded_zero_because_it_has_no_mask():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)

    adjusted = _adjust(
        runtime,
        torch.zeros(1, 2, 2, 2, dtype=torch.float64),
        positions=torch.tensor([[0, 10]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float64),
    )

    assert torch.isfinite(adjusted).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_fixed_alibi_returned_score_dtype_matches_base_scores(dtype):
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LOG1P)
    base_scores = _base_scores(dtype=dtype)

    adjusted = _adjust(
        runtime,
        base_scores,
        cross_chromosome_bias=torch.zeros(2, dtype=dtype),
    )

    assert adjusted.dtype is dtype


def test_fixed_alibi_output_does_not_depend_on_query_or_key_when_base_scores_are_fixed():
    runtime = _runtime(num_heads=2, distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    positions = torch.tensor([[10, 30, 50]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
    query_a, key_a = _unused_query_key()
    query_b = query_a + 1000.0
    key_b = key_a - 1000.0

    adjusted_a = runtime.adjust_attention_scores(
        base_scores,
        query=query_a,
        key=key_a,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float64),
    )
    adjusted_b = runtime.adjust_attention_scores(
        base_scores,
        query=query_b,
        key=key_b,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(2, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted_a, adjusted_b)


def test_fixed_alibi_rejects_position_bias_embedding():
    with pytest.raises(ValueError, match="position_bias"):
        _runtime().adjust_attention_scores(
            _base_scores(),
            query=_unused_query_key()[0],
            key=_unused_query_key()[1],
            positions=torch.tensor([[10, 30, 50]], dtype=torch.long),
            chrom_ids=torch.tensor([[0, 0, 0]], dtype=torch.long),
            position_bias=nn.Embedding(2, 2),
            cross_chromosome_bias=torch.zeros(2),
        )


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
def test_fixed_alibi_state_key_sets_are_exact_for_mask_and_separate(
    cross_policy,
    expected_keys,
):
    config = _resolve_custom(cross_policy=cross_policy)
    model = _model(config)
    layer = model.attention.attention_layers[0]

    assert layer.position_bias is None
    assert _positional_state_keys(model) == expected_keys
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
        assert torch.equal(
            layer.cross_chromosome_bias,
            torch.zeros_like(layer.cross_chromosome_bias),
        )
    else:
        assert layer.cross_chromosome_bias is None


def test_fixed_alibi_chromosome_embedding_and_learned_binned_state_surfaces_are_independent():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        chromosome=ChromosomeEncoding.LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())

    assert _positional_state_keys(model) == {
        "absolute_position_embedding.weight",
        "attention.attention_layers.0.chrom_embedding.weight",
        "attention.attention_layers.0.cross_chromosome_bias",
    }


def test_chunked_fixed_alibi_state_keys_use_base_model_prefix_naturally():
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
def test_fixed_alibi_model_forward_backward_return_attention_and_padding(
    chromosome,
    cross_policy,
):
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
    assert not any("alibi_slope" in key for key in model.state_dict())
    if cross_policy is CrossChromosomePolicy.MASK:
        assert torch.all(attention[0, :, 0, 1] == 0)
        assert torch.all(attention[0, :, 0, 0] > 0)


def test_fixed_alibi_cross_bias_receives_gradient_with_cross_chromosome_pairs():
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
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)


def test_fixed_alibi_rejects_real_position_zero_but_allows_masked_zero():
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
    with pytest.raises(ValueError, match="ALiBi positions"):
        model(
            None,
            bad_positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_fixed_alibi_requires_chrom_ids_in_model_even_without_chromosome_embedding():
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


def test_fixed_alibi_position_sensitivity_changes_same_chromosome_attention():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)
    same_chrom_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    positions_b = positions.clone()
    positions_b[0, 1] = 30

    _, intermediates_a = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    _, intermediates_b = model(
        None,
        positions_b,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )

    assert not torch.equal(
        intermediates_a["attention_weights"][0],
        intermediates_b["attention_weights"][0],
    )


def test_fixed_alibi_with_learned_binned_absolute_forward_backward_succeeds():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)

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
    assert torch.isfinite(content.grad).all()
    assert torch.isfinite(model.attention.attention_layers[0].cross_chromosome_bias.grad).all()


def test_attention_runtime_support_allows_fixed_alibi_but_not_learned_alibi():
    fixed = _resolve_custom(relative=RelativePositionEncoding.ALIBI_FIXED)
    learned = _resolve_custom(relative=RelativePositionEncoding.ALIBI_LEARNED)

    validate_attention_runtime_support(fixed)
    with pytest.raises(NotImplementedError, match="alibi_learned"):
        validate_attention_runtime_support(learned)


def test_phase7_gate_still_rejects_fixed_alibi_after_attention_support_exists():
    config = _resolve_custom(relative=RelativePositionEncoding.ALIBI_FIXED)

    with pytest.raises(NotImplementedError, match="alibi_fixed"):
        validate_phase7_runtime_support(config)
