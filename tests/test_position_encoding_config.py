"""Tests for the pure position-encoding configuration resolver."""

import math
from dataclasses import FrozenInstanceError

import pytest

from src.encoding.levels import (
    AnnotationLevel,
    get_content_feature_dimension,
    get_feature_dimension,
)
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedIGMode,
    resolve_position_encoding_config,
)


def resolve(
    request: PositionEncodingRequest,
    level: AnnotationLevel = AnnotationLevel.L3,
    *,
    latent_dim: int = 16,
    num_heads: int = 2,
    num_chromosomes: int = 24,
):
    return resolve_position_encoding_config(
        request,
        level,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )


def custom_request(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy | None = CrossChromosomePolicy.SEPARATE,
    **kwargs,
) -> PositionEncodingRequest:
    return PositionEncodingRequest(
        preset=PositionPreset.CUSTOM,
        absolute_position_encoding=absolute,
        relative_position_encoding=relative,
        chromosome_encoding=chromosome,
        cross_chromosome_policy=cross_policy,
        **kwargs,
    )


def test_legacy_l0_resolves_without_absolute_position():
    config = resolve(PositionEncodingRequest(), AnnotationLevel.L0)

    assert config.content_dim == 1
    assert config.input_dim == 1
    assert config.absolute.encoding is AbsolutePositionEncoding.NONE
    assert config.relative.encoding is RelativePositionEncoding.T5_BUCKET
    assert config.relative.num_buckets == 32
    assert config.relative.total_bias_rows == 33
    assert config.relative.max_distance_bp == 100000
    assert config.chromosome.encoding is ChromosomeEncoding.LEARNED
    assert config.chromosome.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE
    assert config.chromosome.requires_chrom_ids is True
    assert config.chromosome.cross_chromosome_parameter == "dedicated_bucket"
    assert config.attribution.default_ig_mode is ResolvedIGMode.CONTENT


@pytest.mark.parametrize(
    ("level", "content_dim", "input_dim"),
    [
        (AnnotationLevel.L1, 1, 65),
        (AnnotationLevel.L2, 5, 69),
        (AnnotationLevel.L3, 7, 71),
        (AnnotationLevel.L4, 7, 71),
    ],
)
def test_legacy_l1_to_l4_resolve_sinusoidal_input_concat(level, content_dim, input_dim):
    config = resolve(PositionEncodingRequest(), level)

    assert config.content_dim == content_dim
    assert config.input_dim == input_dim
    assert config.absolute.encoding is AbsolutePositionEncoding.SINUSOIDAL
    assert config.absolute.fusion.value == "input_concat"
    assert config.absolute.position_dim == 64
    assert config.absolute.coordinate_scale == 1.0
    assert config.absolute.max_wavelength == 10000.0
    assert config.relative.encoding is RelativePositionEncoding.T5_BUCKET
    assert config.relative.num_buckets == 32
    assert config.relative.total_bias_rows == 33
    assert config.chromosome.encoding is ChromosomeEncoding.LEARNED
    assert config.attribution.default_ig_mode is ResolvedIGMode.CONTENT


def test_feature_dimension_legacy_values_are_unchanged():
    assert get_feature_dimension(AnnotationLevel.L0) == 1
    assert get_feature_dimension(AnnotationLevel.L1) == 65
    assert get_feature_dimension(AnnotationLevel.L2) == 69
    assert get_feature_dimension(AnnotationLevel.L3) == 71
    assert get_feature_dimension(AnnotationLevel.L4) == 71


def test_content_feature_dimensions_are_non_positional_values():
    assert get_content_feature_dimension(AnnotationLevel.L0) == 1
    assert get_content_feature_dimension(AnnotationLevel.L1) == 1
    assert get_content_feature_dimension(AnnotationLevel.L2) == 5
    assert get_content_feature_dimension(AnnotationLevel.L3) == 7
    assert get_content_feature_dimension(AnnotationLevel.L4) == 7


def test_to_dict_uses_nested_plain_values():
    config = resolve(PositionEncodingRequest(), AnnotationLevel.L3)
    data = config.to_dict()

    assert data["schema_version"] == 1
    assert data["preset"] == "legacy"
    assert data["annotation_level"] == "L3"
    assert data["absolute"]["type"] == "sinusoidal"
    assert data["absolute"]["fusion"] == "input_concat"
    assert data["relative"]["type"] == "t5_bucket"
    assert data["chromosome"]["encoding"] == "learned"
    assert data["attribution"] == {"default_ig_mode": "content"}
    assert data["content_dim"] == 7
    assert data["input_dim"] == 71


def test_request_and_resolved_configs_are_immutable():
    request = PositionEncodingRequest()
    with pytest.raises(FrozenInstanceError):
        request.preset = PositionPreset.CUSTOM

    config = resolve(request, AnnotationLevel.L0)
    with pytest.raises(FrozenInstanceError):
        config.input_dim = 99


@pytest.mark.parametrize(
    "field_name",
    [
        "absolute_position_encoding",
        "relative_position_encoding",
        "chromosome_encoding",
    ],
)
def test_custom_requires_explicit_core_strategies(field_name):
    kwargs = {
        "preset": PositionPreset.CUSTOM,
        "absolute_position_encoding": AbsolutePositionEncoding.NONE,
        "relative_position_encoding": RelativePositionEncoding.NONE,
        "chromosome_encoding": ChromosomeEncoding.NONE,
    }
    kwargs[field_name] = None

    with pytest.raises(ValueError, match=field_name):
        resolve(PositionEncodingRequest(**kwargs), num_chromosomes=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absolute_position_encoding": AbsolutePositionEncoding.NONE},
        {"relative_position_encoding": RelativePositionEncoding.NONE},
        {"chromosome_encoding": ChromosomeEncoding.NONE},
        {"cross_chromosome_policy": CrossChromosomePolicy.MASK},
        {"position_dim": 64},
        {"sinusoidal_coordinate_scale": 1.0},
        {"sinusoidal_max_wavelength": 10000.0},
        {"position_bin_size": 10000},
        {"num_position_buckets": 32},
        {"max_position_distance": 100000},
        {"rope_coordinate_scale": 10000.0},
        {"rope_base": 10000.0},
        {"alibi_distance_function": AlibiDistanceFunction.LOG1P},
        {"alibi_distance_scale": 10000.0},
    ],
)
def test_legacy_rejects_explicit_positional_overrides(kwargs):
    with pytest.raises(ValueError, match="position_preset=legacy"):
        resolve(PositionEncodingRequest(**kwargs))


def test_custom_none_none_none_separate_with_zero_chromosomes_is_valid():
    config = resolve(custom_request(), num_chromosomes=0)

    assert config.absolute.encoding is AbsolutePositionEncoding.NONE
    assert config.relative.encoding is RelativePositionEncoding.NONE
    assert config.chromosome.encoding is ChromosomeEncoding.NONE
    assert config.chromosome.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE
    assert config.chromosome.requires_chrom_ids is False
    assert config.chromosome.cross_chromosome_parameter is None
    assert config.input_dim == config.content_dim


def test_relative_none_mask_chromosome_none_with_zero_chromosomes_is_invalid():
    with pytest.raises(ValueError, match="num_chromosomes"):
        resolve(
            custom_request(cross_policy=CrossChromosomePolicy.MASK),
            num_chromosomes=0,
        )


def test_t5_separate_chromosome_none_with_zero_chromosomes_is_invalid():
    with pytest.raises(ValueError, match="num_chromosomes"):
        resolve(
            custom_request(relative=RelativePositionEncoding.T5_BUCKET),
            num_chromosomes=0,
        )


def test_t5_separate_chromosome_none_with_positive_chromosomes_is_valid():
    config = resolve(
        custom_request(relative=RelativePositionEncoding.T5_BUCKET),
        num_chromosomes=24,
    )

    assert config.chromosome.encoding is ChromosomeEncoding.NONE
    assert config.chromosome.requires_chrom_ids is True
    assert config.chromosome.cross_chromosome_parameter == "dedicated_bucket"
    assert config.relative.total_bias_rows == 33


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "legacy"},
        {
            "preset": PositionPreset.CUSTOM,
            "absolute_position_encoding": "none",
            "relative_position_encoding": RelativePositionEncoding.NONE,
            "chromosome_encoding": ChromosomeEncoding.NONE,
        },
        {
            "preset": PositionPreset.CUSTOM,
            "absolute_position_encoding": AbsolutePositionEncoding.NONE,
            "relative_position_encoding": "none",
            "chromosome_encoding": ChromosomeEncoding.NONE,
        },
        {
            "preset": PositionPreset.CUSTOM,
            "absolute_position_encoding": AbsolutePositionEncoding.NONE,
            "relative_position_encoding": RelativePositionEncoding.NONE,
            "chromosome_encoding": "none",
        },
        {
            "preset": PositionPreset.CUSTOM,
            "absolute_position_encoding": AbsolutePositionEncoding.NONE,
            "relative_position_encoding": RelativePositionEncoding.NONE,
            "chromosome_encoding": ChromosomeEncoding.NONE,
            "cross_chromosome_policy": "separate",
        },
    ],
)
def test_raw_string_enum_values_do_not_silently_resolve(kwargs):
    with pytest.raises(ValueError, match="enum instance"):
        resolve(PositionEncodingRequest(**kwargs), num_chromosomes=0)


def test_raw_string_annotation_level_does_not_silently_resolve():
    with pytest.raises(ValueError, match="annotation_level"):
        resolve_position_encoding_config(
            PositionEncodingRequest(),
            "L3",
            latent_dim=16,
            num_heads=2,
            num_chromosomes=24,
        )


def test_sinusoidal_absolute_resolves_custom_defaults_and_input_dimension():
    config = resolve(
        custom_request(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            sinusoidal_coordinate_scale=2.0,
            sinusoidal_max_wavelength=20000.0,
        )
    )

    assert config.absolute.position_dim == 64
    assert config.absolute.coordinate_scale == 2.0
    assert config.absolute.max_wavelength == 20000.0
    assert config.input_dim == 71


def test_sinusoidal_absolute_rejects_odd_position_dim():
    with pytest.raises(ValueError, match="position_dim"):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.SINUSOIDAL,
                position_dim=63,
            )
        )


def test_sinusoidal_absolute_rejects_zero_position_dim():
    with pytest.raises(ValueError, match="position_dim"):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.SINUSOIDAL,
                position_dim=0,
            )
        )


@pytest.mark.parametrize(
    "field_name",
    ["sinusoidal_coordinate_scale", "sinusoidal_max_wavelength"],
)
def test_sinusoidal_absolute_rejects_zero_scale_values(field_name):
    with pytest.raises(ValueError, match=field_name):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.SINUSOIDAL,
                **{field_name: 0},
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sinusoidal_coordinate_scale", math.nan),
        ("sinusoidal_coordinate_scale", math.inf),
        ("sinusoidal_coordinate_scale", -math.inf),
        ("sinusoidal_max_wavelength", math.nan),
        ("sinusoidal_max_wavelength", math.inf),
        ("sinusoidal_max_wavelength", -math.inf),
    ],
)
def test_sinusoidal_absolute_rejects_non_finite_scale_values(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.SINUSOIDAL,
                **{field_name: value},
            )
        )


def test_absolute_none_rejects_active_absolute_fields():
    with pytest.raises(ValueError, match="position_dim"):
        resolve(custom_request(position_dim=64), num_chromosomes=0)


def test_sinusoidal_rejects_learned_binned_fields():
    with pytest.raises(ValueError, match="position_bin_size"):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.SINUSOIDAL,
                position_bin_size=10000,
            )
        )


def test_learned_binned_absolute_resolves_defaults():
    config = resolve(custom_request(absolute=AbsolutePositionEncoding.LEARNED_BINNED))

    assert config.absolute.position_dim == 64
    assert config.absolute.bin_size_bp == 10000
    assert config.absolute.fusion.value == "input_concat"
    assert config.input_dim == 71


def test_learned_binned_absolute_rejects_zero_position_bin_size():
    with pytest.raises(ValueError, match="position_bin_size"):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.LEARNED_BINNED,
                position_bin_size=0,
            )
        )


def test_learned_binned_rejects_sinusoidal_fields():
    with pytest.raises(ValueError, match="sinusoidal_coordinate_scale"):
        resolve(
            custom_request(
                absolute=AbsolutePositionEncoding.LEARNED_BINNED,
                sinusoidal_coordinate_scale=1.0,
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_position_buckets": 3},
        {"num_position_buckets": 5},
        {"num_position_buckets": 32, "max_position_distance": 8},
    ],
)
def test_t5_bucket_validation(kwargs):
    with pytest.raises(ValueError):
        resolve(custom_request(relative=RelativePositionEncoding.T5_BUCKET, **kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_position_buckets": 0},
        {"max_position_distance": 0},
    ],
)
def test_t5_bucket_rejects_zero_numeric_values(kwargs):
    with pytest.raises(ValueError):
        resolve(custom_request(relative=RelativePositionEncoding.T5_BUCKET, **kwargs))


def test_t5_mask_requires_chromosomes_but_does_not_add_dedicated_bucket_row():
    config = resolve(
        custom_request(
            relative=RelativePositionEncoding.T5_BUCKET,
            cross_policy=CrossChromosomePolicy.MASK,
        ),
        num_chromosomes=24,
    )

    assert config.relative.num_buckets == 32
    assert config.relative.total_bias_rows == 32
    assert config.chromosome.requires_chrom_ids is True
    assert config.chromosome.cross_chromosome_parameter == "mask"


def test_relative_none_rejects_relative_method_fields():
    with pytest.raises(ValueError, match="num_position_buckets"):
        resolve(custom_request(num_position_buckets=32), num_chromosomes=0)


def test_rope_validates_latent_head_geometry():
    config = resolve(
        custom_request(relative=RelativePositionEncoding.ROPE), latent_dim=16, num_heads=2
    )

    assert config.relative.encoding is RelativePositionEncoding.ROPE
    assert config.relative.rope_coordinate_scale == 10000.0
    assert config.relative.rope_base == 10000.0
    assert config.chromosome.cross_chromosome_parameter == "learned_bias"


@pytest.mark.parametrize("field_name", ["rope_coordinate_scale", "rope_base"])
def test_rope_rejects_zero_numeric_values(field_name):
    with pytest.raises(ValueError, match=field_name):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ROPE,
                **{field_name: 0},
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rope_coordinate_scale", math.nan),
        ("rope_coordinate_scale", math.inf),
        ("rope_coordinate_scale", -math.inf),
        ("rope_base", math.nan),
        ("rope_base", math.inf),
        ("rope_base", -math.inf),
    ],
)
def test_rope_rejects_non_finite_numeric_values(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ROPE,
                **{field_name: value},
            )
        )


def test_rope_rejects_non_divisible_latent_dim():
    with pytest.raises(ValueError, match="divisible"):
        resolve(custom_request(relative=RelativePositionEncoding.ROPE), latent_dim=18, num_heads=4)


def test_rope_rejects_odd_head_dimension():
    with pytest.raises(ValueError, match="even"):
        resolve(custom_request(relative=RelativePositionEncoding.ROPE), latent_dim=18, num_heads=2)


def test_rope_rejects_inactive_t5_fields():
    with pytest.raises(ValueError, match="num_position_buckets"):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ROPE,
                num_position_buckets=32,
            )
        )


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_alibi_resolves_defaults(relative):
    config = resolve(custom_request(relative=relative))

    assert config.relative.encoding is relative
    assert config.relative.alibi_distance_function is AlibiDistanceFunction.LOG1P
    assert config.relative.alibi_distance_scale == 10000.0
    assert config.chromosome.cross_chromosome_parameter == "learned_bias"


def test_alibi_rejects_zero_distance_scale():
    with pytest.raises(ValueError, match="alibi_distance_scale"):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ALIBI_FIXED,
                alibi_distance_scale=0,
            )
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_alibi_rejects_non_finite_distance_scale(value):
    with pytest.raises(ValueError, match="alibi_distance_scale"):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ALIBI_FIXED,
                alibi_distance_scale=value,
            )
        )


def test_alibi_rejects_inactive_rope_fields():
    with pytest.raises(ValueError, match="rope_base"):
        resolve(
            custom_request(
                relative=RelativePositionEncoding.ALIBI_FIXED,
                rope_base=10000.0,
            )
        )


def test_learned_chromosome_encoding_requires_positive_chromosome_count():
    with pytest.raises(ValueError, match="num_chromosomes"):
        resolve(
            custom_request(chromosome=ChromosomeEncoding.LEARNED),
            num_chromosomes=0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "absolute": AbsolutePositionEncoding.SINUSOIDAL,
            "position_dim": False,
        },
        {
            "absolute": AbsolutePositionEncoding.SINUSOIDAL,
            "sinusoidal_coordinate_scale": False,
        },
        {
            "absolute": AbsolutePositionEncoding.LEARNED_BINNED,
            "position_bin_size": False,
        },
        {
            "relative": RelativePositionEncoding.T5_BUCKET,
            "num_position_buckets": False,
        },
        {
            "relative": RelativePositionEncoding.ROPE,
            "rope_base": False,
        },
        {
            "relative": RelativePositionEncoding.ALIBI_FIXED,
            "alibi_distance_scale": False,
        },
    ],
)
def test_representative_false_numeric_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve(custom_request(**kwargs))


@pytest.mark.parametrize("num_chromosomes", [-1, 1.5, True, "24"])
def test_num_chromosomes_must_be_a_non_negative_integer(num_chromosomes):
    with pytest.raises(ValueError, match="num_chromosomes"):
        resolve(custom_request(), num_chromosomes=num_chromosomes)


def test_zero_chromosomes_remains_valid_when_no_chromosome_information_is_needed():
    config = resolve(
        custom_request(
            absolute=AbsolutePositionEncoding.NONE,
            relative=RelativePositionEncoding.NONE,
            chromosome=ChromosomeEncoding.NONE,
            cross_policy=CrossChromosomePolicy.SEPARATE,
        ),
        num_chromosomes=0,
    )

    assert config.chromosome.requires_chrom_ids is False
    assert config.input_dim == config.content_dim
