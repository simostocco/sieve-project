"""Tests for Phase 7 baseline positional runtime implementations."""

import math
from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from src.encoding import relative_position_bucket
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
from src.models.position_runtime import (
    LegacyT5RelativePositionRuntime,
    NoAbsolutePositionRuntime,
    NoRelativePositionRuntime,
    ObservedAbsolutePositionRuntime,
    SinusoidalAbsolutePositionRuntime,
    T5RelativePositionRuntime,
    build_absolute_position_runtime,
    build_relative_position_runtime,
    build_same_chromosome_pair_mask,
    validate_phase7_runtime_support,
)


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    level: AnnotationLevel = AnnotationLevel.L3,
    latent_dim: int = 16,
    num_heads: int = 2,
    num_chromosomes: int = 24,
    **kwargs,
):
    request = PositionEncodingRequest(
        preset=PositionPreset.CUSTOM,
        absolute_position_encoding=absolute,
        relative_position_encoding=relative,
        chromosome_encoding=chromosome,
        cross_chromosome_policy=cross_policy,
        **kwargs,
    )
    return resolve_position_encoding_config(
        request,
        level,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )


def _resolve_legacy(level: AnnotationLevel, *, num_chromosomes: int = 24):
    return resolve_position_encoding_config(
        PositionEncodingRequest(),
        level,
        latent_dim=16,
        num_heads=2,
        num_chromosomes=num_chromosomes,
    )


def _manual_sinusoidal(
    positions: torch.Tensor,
    *,
    position_dim: int,
    coordinate_scale: float,
    max_wavelength: float,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scaled_positions = positions.to(device=reference.device, dtype=reference.dtype)
    scaled_positions = scaled_positions / coordinate_scale
    div = torch.exp(
        torch.arange(0, position_dim, 2, device=reference.device, dtype=reference.dtype)
        * -(math.log(max_wavelength) / position_dim)
    )
    expected = torch.empty(
        (*positions.shape, position_dim),
        device=reference.device,
        dtype=reference.dtype,
    )
    angles = scaled_positions[..., None] * div
    expected[..., 0::2] = torch.sin(angles)
    expected[..., 1::2] = torch.cos(angles)
    if mask is not None:
        expected = expected.masked_fill(~mask.to(device=reference.device).unsqueeze(-1), 0)
    return expected


def _assert_parameterless_plain_runtime(runtime) -> None:
    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert is_dataclass(runtime)
    for field in fields(runtime):
        assert not isinstance(getattr(runtime, field.name), torch.Tensor)


def _make_position_bias(rows: int, heads: int) -> nn.Embedding:
    position_bias = nn.Embedding(rows, heads)
    with torch.no_grad():
        values = torch.arange(rows * heads, dtype=torch.float32).reshape(rows, heads)
        position_bias.weight.copy_(values / 100.0)
    return position_bias


@pytest.mark.parametrize(
    "absolute",
    [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL],
)
@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.NONE, RelativePositionEncoding.T5_BUCKET],
)
@pytest.mark.parametrize(
    "chromosome",
    [ChromosomeEncoding.NONE, ChromosomeEncoding.LEARNED],
)
@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_all_phase7_baseline_combinations_resolve_and_build_runtimes(
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
        num_chromosomes=24,
    )

    validate_phase7_runtime_support(config)
    absolute_runtime = build_absolute_position_runtime(config)
    relative_runtime = build_relative_position_runtime(config)

    expected_absolute_type = (
        NoAbsolutePositionRuntime
        if absolute is AbsolutePositionEncoding.NONE
        else SinusoidalAbsolutePositionRuntime
    )
    expected_relative_type = (
        NoRelativePositionRuntime
        if relative is RelativePositionEncoding.NONE
        else T5RelativePositionRuntime
    )
    assert isinstance(absolute_runtime, expected_absolute_type)
    assert isinstance(relative_runtime, expected_relative_type)

    reference = torch.ones(2, 3, config.content_dim)
    positions = torch.tensor([[0, 100, 200], [300, 400, 500]], dtype=torch.long)
    observed = torch.full((2, 3, 64), 9.0)
    mask = torch.ones(2, 3, dtype=torch.bool)
    resolved_absolute = absolute_runtime.resolve(
        observed,
        positions,
        chrom_ids=torch.zeros(2, 3, dtype=torch.long),
        mask=mask,
        reference=reference,
    )

    assert resolved_absolute.shape[-1] == (config.absolute.position_dim or 0)
    _assert_parameterless_plain_runtime(absolute_runtime)
    _assert_parameterless_plain_runtime(relative_runtime)


def test_relative_none_chromosome_none_separate_supports_zero_chromosomes():
    config = _resolve_custom(num_chromosomes=0)

    validate_phase7_runtime_support(config)

    assert isinstance(build_absolute_position_runtime(config), NoAbsolutePositionRuntime)
    assert isinstance(build_relative_position_runtime(config), NoRelativePositionRuntime)


@pytest.mark.parametrize(
    ("relative", "cross_policy"),
    [
        (RelativePositionEncoding.NONE, CrossChromosomePolicy.MASK),
        (RelativePositionEncoding.T5_BUCKET, CrossChromosomePolicy.SEPARATE),
    ],
)
def test_resolver_rejects_chromosome_aware_routing_with_zero_chromosomes(
    relative,
    cross_policy,
):
    with pytest.raises(ValueError, match="num_chromosomes"):
        _resolve_custom(
            relative=relative,
            cross_policy=cross_policy,
            num_chromosomes=0,
        )


@pytest.mark.parametrize(
    "level",
    [
        AnnotationLevel.L0,
        AnnotationLevel.L1,
        AnnotationLevel.L2,
        AnnotationLevel.L3,
        AnnotationLevel.L4,
    ],
)
def test_legacy_factories_preserve_phase6_runtime_classes_for_all_levels(level):
    config = _resolve_legacy(level)

    assert isinstance(build_absolute_position_runtime(config), ObservedAbsolutePositionRuntime)
    assert isinstance(build_relative_position_runtime(config), LegacyT5RelativePositionRuntime)
    assert not isinstance(build_absolute_position_runtime(config), NoAbsolutePositionRuntime)
    assert not isinstance(
        build_absolute_position_runtime(config),
        SinusoidalAbsolutePositionRuntime,
    )
    assert not isinstance(build_relative_position_runtime(config), T5RelativePositionRuntime)


def test_absolute_factory_preserves_resolved_non_default_sinusoidal_settings():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.SINUSOIDAL,
        position_dim=8,
        sinusoidal_coordinate_scale=25.0,
        sinusoidal_max_wavelength=50000.0,
    )

    runtime = build_absolute_position_runtime(config)

    assert isinstance(runtime, SinusoidalAbsolutePositionRuntime)
    assert runtime.position_dim == 8
    assert runtime.coordinate_scale == 25.0
    assert runtime.max_wavelength == 50000.0


def test_relative_factory_preserves_resolved_non_default_t5_settings():
    config = _resolve_custom(
        relative=RelativePositionEncoding.T5_BUCKET,
        cross_policy=CrossChromosomePolicy.MASK,
        num_position_buckets=8,
        max_position_distance=1000,
    )

    runtime = build_relative_position_runtime(config)

    assert isinstance(runtime, T5RelativePositionRuntime)
    assert runtime.num_position_buckets == 8
    assert runtime.max_distance == 1000
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.MASK


@pytest.mark.parametrize(
    ("position_dim", "coordinate_scale", "max_wavelength"),
    [
        (64, 1.0, 10000.0),
        (8, 1.0, 10000.0),
        (64, 100.0, 10000.0),
        (64, 1.0, 50000.0),
    ],
)
def test_sinusoidal_runtime_matches_independent_torch_formula(
    position_dim,
    coordinate_scale,
    max_wavelength,
):
    runtime = SinusoidalAbsolutePositionRuntime(
        position_dim=position_dim,
        coordinate_scale=coordinate_scale,
        max_wavelength=max_wavelength,
    )
    reference = torch.zeros(2, 3, 5, dtype=torch.float64)
    positions = torch.tensor([[0, 100, 200], [300, 400, 500]], dtype=torch.long)
    observed = torch.randn(2, 3, 64, dtype=torch.float64)

    actual = runtime.resolve(
        observed,
        positions,
        chrom_ids=torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long),
        mask=torch.ones(2, 3, dtype=torch.bool),
        reference=reference,
    )
    expected = _manual_sinusoidal(
        positions,
        position_dim=position_dim,
        coordinate_scale=coordinate_scale,
        max_wavelength=max_wavelength,
        reference=reference,
        mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert actual.shape == (2, 3, position_dim)
    assert actual.dtype == reference.dtype
    assert actual.device == reference.device
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual[..., 0::2], expected[..., 0::2])
    assert torch.allclose(actual[..., 1::2], expected[..., 1::2])


def test_sinusoidal_runtime_scale_wavelength_mask_and_observed_behavior():
    positions = torch.tensor([[0, 100, 200]], dtype=torch.long)
    mask = torch.tensor([[True, False, True]])
    reference = torch.zeros(1, 3, 5)
    observed_a = torch.randn(1, 3, 64)
    observed_b = observed_a + 1000.0

    base = SinusoidalAbsolutePositionRuntime(8, 1.0, 10000.0)
    changed_scale = SinusoidalAbsolutePositionRuntime(8, 10.0, 10000.0)
    changed_wavelength = SinusoidalAbsolutePositionRuntime(8, 1.0, 50000.0)

    base_features = base.resolve(observed_a, positions, None, mask, reference)
    base_features_from_other_observed = base.resolve(observed_b, positions, None, mask, reference)
    scaled_features = changed_scale.resolve(observed_a, positions, None, mask, reference)
    wavelength_features = changed_wavelength.resolve(observed_a, positions, None, mask, reference)

    assert torch.equal(base_features, base_features_from_other_observed)
    assert not torch.allclose(base_features[:, 2], scaled_features[:, 2])
    assert not torch.allclose(base_features[:, 2], wavelength_features[:, 2])
    assert torch.equal(base_features[0, 1], torch.zeros(8))
    assert torch.equal(base_features[0, 0, 0::2], torch.zeros(4))
    assert torch.equal(base_features[0, 0, 1::2], torch.ones(4))
    _assert_parameterless_plain_runtime(base)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"position_dim": 0}, "position_dim"),
        ({"position_dim": 7}, "even"),
        ({"position_dim": True}, "position_dim"),
        ({"coordinate_scale": 0.0}, "coordinate_scale"),
        ({"coordinate_scale": float("nan")}, "coordinate_scale"),
        ({"coordinate_scale": float("inf")}, "coordinate_scale"),
        ({"max_wavelength": 0.0}, "max_wavelength"),
        ({"max_wavelength": float("-inf")}, "max_wavelength"),
    ],
)
def test_sinusoidal_runtime_rejects_invalid_direct_settings(kwargs, message):
    settings = {"position_dim": 8, "coordinate_scale": 1.0, "max_wavelength": 10000.0}
    settings.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SinusoidalAbsolutePositionRuntime(**settings)


def test_sinusoidal_runtime_validates_inputs():
    runtime = SinusoidalAbsolutePositionRuntime(8, 1.0, 10000.0)
    reference = torch.zeros(1, 3, 5)
    positions = torch.zeros(1, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="reference"):
        runtime.resolve(torch.zeros(1, 3, 8), positions, None, None, reference="bad")
    with pytest.raises(ValueError, match="positions"):
        runtime.resolve(torch.zeros(1, 3, 8), "bad", None, None, reference)
    with pytest.raises(ValueError, match="rank"):
        runtime.resolve(
            torch.zeros(1, 8), torch.zeros(1, dtype=torch.long), None, None, torch.zeros(5)
        )
    with pytest.raises(ValueError, match="positions shape"):
        runtime.resolve(
            torch.zeros(1, 3, 8), torch.zeros(1, 2, dtype=torch.long), None, None, reference
        )
    with pytest.raises(ValueError, match="mask shape"):
        runtime.resolve(
            torch.zeros(1, 3, 8), positions, None, torch.ones(1, 2, dtype=torch.bool), reference
        )
    with pytest.raises(ValueError, match="boolean"):
        runtime.resolve(torch.zeros(1, 3, 8), positions, None, torch.ones(1, 3), reference)


def test_no_absolute_runtime_returns_zero_width_reference_view():
    runtime = NoAbsolutePositionRuntime()
    reference = torch.randn(2, 3, 5, dtype=torch.float64)
    observed = torch.full((2, 3, 64), 99.0)

    resolved = runtime.resolve(
        observed,
        positions=torch.full((2, 3), 100, dtype=torch.long),
        chrom_ids=torch.ones(2, 3, dtype=torch.long),
        mask=torch.zeros(2, 3, dtype=torch.bool),
        reference=reference,
    )

    assert resolved.shape == (2, 3, 0)
    assert resolved.dtype == reference.dtype
    assert resolved.device == reference.device
    assert resolved._base is not None
    _assert_parameterless_plain_runtime(runtime)


def test_no_absolute_runtime_validates_reference_only():
    runtime = NoAbsolutePositionRuntime()

    with pytest.raises(ValueError, match="reference"):
        runtime.resolve(None, None, None, None, reference="bad")
    with pytest.raises(ValueError, match="rank"):
        runtime.resolve(None, None, None, None, reference=torch.ones(3))


def test_no_relative_runtime_returns_exact_base_score_object():
    runtime = NoRelativePositionRuntime()
    base_scores = torch.randn(2, 2, 3, 3)

    resolved = runtime.adjust_attention_scores(
        base_scores,
        query=torch.randn(2, 2, 3, 4),
        key=torch.randn(2, 2, 3, 4),
        positions=torch.ones(2, 3, dtype=torch.long),
        chrom_ids=torch.zeros(2, 3, dtype=torch.long),
        position_bias=_make_position_bias(8, 2),
    )

    assert resolved is base_scores
    _assert_parameterless_plain_runtime(runtime)


def test_custom_t5_separate_uses_dedicated_cross_chromosome_bucket():
    runtime = T5RelativePositionRuntime(
        num_position_buckets=8,
        max_distance=1000,
        cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
    )
    positions = torch.tensor([[100, 130, 500]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    position_bias = _make_position_bias(9, 2)
    before = position_bias.weight.detach().clone()

    bias = runtime.compute_bias(
        positions,
        positions,
        position_bias,
        query_chroms=chrom_ids,
        key_chroms=chrom_ids,
    )
    expected_buckets = relative_position_bucket(
        positions[0],
        positions[0],
        num_buckets=8,
        max_distance=1000,
        query_chroms=chrom_ids[0],
        key_chroms=chrom_ids[0],
    )
    expected_bias = position_bias(expected_buckets.unsqueeze(0)).permute(0, 3, 1, 2)
    base_scores = torch.randn(1, 2, 3, 3)

    assert position_bias.num_embeddings == 9
    assert expected_buckets[0, 1].item() == 8
    assert bias.shape == (1, 2, 3, 3)
    assert torch.equal(bias, expected_bias)
    assert torch.equal(
        runtime.adjust_attention_scores(
            base_scores,
            query=torch.randn(1, 2, 3, 4),
            key=torch.randn(1, 2, 3, 4),
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=position_bias,
        ),
        base_scores + bias,
    )
    assert torch.equal(position_bias.weight, before)
    _assert_parameterless_plain_runtime(runtime)

    with pytest.raises(ValueError, match="query_chroms and key_chroms"):
        runtime.compute_bias(positions, positions, position_bias)


def test_custom_t5_mask_uses_only_ordinary_position_buckets():
    runtime = T5RelativePositionRuntime(
        num_position_buckets=8,
        max_distance=1000,
        cross_chromosome_policy=CrossChromosomePolicy.MASK,
    )
    positions = torch.tensor([[100, 130, 500]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    position_bias = _make_position_bias(8, 2)
    before = position_bias.weight.detach().clone()

    bias = runtime.compute_bias(
        positions,
        positions,
        position_bias,
        query_chroms=chrom_ids,
        key_chroms=chrom_ids,
    )
    ordinary_buckets = relative_position_bucket(
        positions[0],
        positions[0],
        num_buckets=8,
        max_distance=1000,
    )
    expected_bias = position_bias(ordinary_buckets.unsqueeze(0)).permute(0, 3, 1, 2)
    base_scores = torch.randn(1, 2, 3, 3)

    assert position_bias.num_embeddings == 8
    assert torch.all(ordinary_buckets < 8)
    assert bias.shape == (1, 2, 3, 3)
    assert torch.equal(bias, expected_bias)
    assert torch.equal(
        runtime.adjust_attention_scores(
            base_scores,
            query=torch.randn(1, 2, 3, 4),
            key=torch.randn(1, 2, 3, 4),
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=position_bias,
        ),
        base_scores + bias,
    )
    assert torch.equal(position_bias.weight, before)
    _assert_parameterless_plain_runtime(runtime)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_position_buckets": True}, "num_position_buckets"),
        ({"num_position_buckets": 2}, "at least 4"),
        ({"num_position_buckets": 7}, "even"),
        ({"max_distance": True}, "max_distance"),
        ({"max_distance": 0}, "max_distance"),
        ({"max_distance": 2}, "greater"),
        ({"cross_chromosome_policy": "separate"}, "CrossChromosomePolicy"),
    ],
)
def test_custom_t5_runtime_rejects_invalid_direct_settings(kwargs, message):
    settings = {
        "num_position_buckets": 8,
        "max_distance": 1000,
        "cross_chromosome_policy": CrossChromosomePolicy.SEPARATE,
    }
    settings.update(kwargs)

    with pytest.raises(ValueError, match=message):
        T5RelativePositionRuntime(**settings)


def test_custom_t5_runtime_validates_embedding_row_count():
    positions = torch.tensor([[100, 200]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    separate = T5RelativePositionRuntime(8, 1000, CrossChromosomePolicy.SEPARATE)
    mask = T5RelativePositionRuntime(8, 1000, CrossChromosomePolicy.MASK)

    with pytest.raises(ValueError, match="position_bias"):
        separate.compute_bias(
            positions, positions, None, query_chroms=chrom_ids, key_chroms=chrom_ids
        )
    with pytest.raises(ValueError, match="num_embeddings"):
        separate.compute_bias(
            positions,
            positions,
            _make_position_bias(8, 2),
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )
    with pytest.raises(ValueError, match="num_embeddings"):
        mask.compute_bias(
            positions,
            positions,
            _make_position_bias(9, 2),
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )


def test_same_chromosome_pair_mask_treats_zero_as_real_chromosome_id():
    chrom_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)

    same_chromosome = build_same_chromosome_pair_mask(chrom_ids)

    assert torch.equal(
        same_chromosome,
        torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, True],
                ]
            ]
        ),
    )


def test_same_chromosome_pair_mask_validates_input_shape():
    with pytest.raises(ValueError, match="chrom_ids"):
        build_same_chromosome_pair_mask("bad")
    with pytest.raises(ValueError, match="shape"):
        build_same_chromosome_pair_mask(torch.ones(3, dtype=torch.long))


@pytest.mark.parametrize(
    "runtime",
    [
        NoAbsolutePositionRuntime(),
        SinusoidalAbsolutePositionRuntime(8, 1.0, 10000.0),
        NoRelativePositionRuntime(),
        T5RelativePositionRuntime(8, 1000, CrossChromosomePolicy.SEPARATE),
    ],
)
def test_phase7_runtime_objects_have_no_registered_or_tensor_state(runtime):
    _assert_parameterless_plain_runtime(runtime)


@pytest.mark.parametrize(
    "config",
    [
        _resolve_custom(
            absolute=AbsolutePositionEncoding.LEARNED_BINNED,
            relative=RelativePositionEncoding.NONE,
            position_dim=8,
            position_bin_size=1000,
            num_chromosomes=2,
        ),
        _resolve_custom(
            relative=RelativePositionEncoding.ALIBI_FIXED,
            alibi_distance_scale=10000.0,
        ),
        _resolve_custom(
            relative=RelativePositionEncoding.ALIBI_LEARNED,
            alibi_distance_scale=10000.0,
        ),
    ],
)
def test_phase7_runtime_support_rejects_unsupported_strategies(config):
    with pytest.raises(NotImplementedError, match="not implemented"):
        validate_phase7_runtime_support(config)


def test_phase7_runtime_support_explicitly_rejects_rope():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ROPE,
        rope_coordinate_scale=10000.0,
        rope_base=10000.0,
    )

    with pytest.raises(NotImplementedError, match="Phase 7"):
        validate_phase7_runtime_support(config)


def test_phase7_runtime_support_requires_resolved_config():
    with pytest.raises(ValueError, match="ResolvedPositionEncodingConfig"):
        validate_phase7_runtime_support(PositionEncodingRequest())
