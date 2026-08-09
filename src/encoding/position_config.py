"""
Pure position-encoding configuration definitions for SIEVE.

This module resolves requested position-encoding options into immutable nested
configuration objects. It intentionally does not construct models, import
PyTorch, parse CLI arguments, write files, or change preprocessing behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .levels import AnnotationLevel, get_content_feature_dimension
from .position_layout import learned_binned_layout_from_absolute_dict

DEFAULT_POSITION_DIM = 64
DEFAULT_SINUSOIDAL_COORDINATE_SCALE = 1.0
DEFAULT_SINUSOIDAL_MAX_WAVELENGTH = 10000.0
DEFAULT_POSITION_BIN_SIZE = 10000
DEFAULT_NUM_POSITION_BUCKETS = 32
DEFAULT_MAX_POSITION_DISTANCE = 100000
DEFAULT_ROPE_COORDINATE_SCALE = 10000.0
DEFAULT_ROPE_BASE = 10000.0
DEFAULT_ALIBI_DISTANCE_FUNCTION = "log1p"
DEFAULT_ALIBI_DISTANCE_SCALE = 10000.0


class PositionPreset(str, Enum):
    """High-level position-encoding preset."""

    LEGACY = "legacy"
    CUSTOM = "custom"


class AbsolutePositionEncoding(str, Enum):
    """Absolute position encoding strategy."""

    NONE = "none"
    SINUSOIDAL = "sinusoidal"
    LEARNED_BINNED = "learned_binned"


class RelativePositionEncoding(str, Enum):
    """Relative position encoding strategy."""

    NONE = "none"
    T5_BUCKET = "t5_bucket"
    ROPE = "rope"
    ALIBI_FIXED = "alibi_fixed"
    ALIBI_LEARNED = "alibi_learned"


class ChromosomeEncoding(str, Enum):
    """Chromosome-level encoding strategy."""

    NONE = "none"
    LEARNED = "learned"


class CrossChromosomePolicy(str, Enum):
    """Policy for variant pairs on different chromosomes."""

    SEPARATE = "separate"
    MASK = "mask"


class AbsolutePositionFusion(str, Enum):
    """Location where absolute position features enter the model."""

    INPUT_CONCAT = "input_concat"


class AlibiDistanceFunction(str, Enum):
    """Distance transform for ALiBi relative scoring."""

    LINEAR = "linear"
    LOG1P = "log1p"


class ResolvedIGMode(str, Enum):
    """Resolved attribution mode for positional inputs."""

    CONTENT = "content"
    LEGACY = "legacy"


@dataclass(frozen=True)
class PositionEncodingRequest:
    """Unresolved position-encoding request.

    Phase 4A accepts enum instances only. CLI string-to-enum conversion belongs
    to later integration work.
    """

    preset: PositionPreset = PositionPreset.LEGACY
    absolute_position_encoding: AbsolutePositionEncoding | None = None
    relative_position_encoding: RelativePositionEncoding | None = None
    chromosome_encoding: ChromosomeEncoding | None = None
    cross_chromosome_policy: CrossChromosomePolicy | None = None
    position_dim: int | None = None
    sinusoidal_coordinate_scale: float | None = None
    sinusoidal_max_wavelength: float | None = None
    position_bin_size: int | None = None
    num_position_buckets: int | None = None
    max_position_distance: int | None = None
    rope_coordinate_scale: float | None = None
    rope_base: float | None = None
    alibi_distance_function: AlibiDistanceFunction | None = None
    alibi_distance_scale: float | None = None


@dataclass(frozen=True)
class ResolvedAbsolutePositionConfig:
    """Resolved absolute-position configuration."""

    encoding: AbsolutePositionEncoding
    fusion: AbsolutePositionFusion | None = None
    position_dim: int | None = None
    coordinate_scale: float | None = None
    max_wavelength: float | None = None
    bin_size_bp: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.encoding.value,
            "fusion": _enum_value(self.fusion),
            "dim": self.position_dim,
            "coordinate_scale": self.coordinate_scale,
            "max_wavelength": self.max_wavelength,
            "bin_size_bp": self.bin_size_bp,
        }


@dataclass(frozen=True)
class ResolvedRelativePositionConfig:
    """Resolved relative-position configuration."""

    encoding: RelativePositionEncoding
    num_buckets: int | None = None
    total_bias_rows: int | None = None
    max_distance_bp: int | None = None
    rope_coordinate_scale: float | None = None
    rope_base: float | None = None
    alibi_distance_function: AlibiDistanceFunction | None = None
    alibi_distance_scale: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.encoding.value,
            "num_buckets": self.num_buckets,
            "total_bias_rows": self.total_bias_rows,
            "max_distance_bp": self.max_distance_bp,
            "rope_coordinate_scale": self.rope_coordinate_scale,
            "rope_base": self.rope_base,
            "alibi_distance_function": _enum_value(self.alibi_distance_function),
            "alibi_distance_scale": self.alibi_distance_scale,
        }


@dataclass(frozen=True)
class ResolvedChromosomeConfig:
    """Resolved chromosome configuration."""

    encoding: ChromosomeEncoding
    cross_chromosome_policy: CrossChromosomePolicy
    num_chromosomes: int
    requires_chrom_ids: bool
    cross_chromosome_parameter: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding.value,
            "cross_chromosome_policy": self.cross_chromosome_policy.value,
            "num_chromosomes": self.num_chromosomes,
            "requires_chrom_ids": self.requires_chrom_ids,
            "cross_chromosome_parameter": self.cross_chromosome_parameter,
        }


@dataclass(frozen=True)
class ResolvedAttributionConfig:
    """Resolved attribution configuration."""

    default_ig_mode: ResolvedIGMode

    def to_dict(self) -> dict[str, object]:
        return {"default_ig_mode": self.default_ig_mode.value}


@dataclass(frozen=True)
class ResolvedPositionEncodingConfig:
    """Resolved nested position-encoding configuration."""

    schema_version: int
    preset: PositionPreset
    annotation_level: AnnotationLevel
    absolute: ResolvedAbsolutePositionConfig
    relative: ResolvedRelativePositionConfig
    chromosome: ResolvedChromosomeConfig
    attribution: ResolvedAttributionConfig
    content_dim: int
    input_dim: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "preset": self.preset.value,
            "annotation_level": self.annotation_level.value,
            "absolute": self.absolute.to_dict(),
            "relative": self.relative.to_dict(),
            "chromosome": self.chromosome.to_dict(),
            "attribution": self.attribution.to_dict(),
            "content_dim": self.content_dim,
            "input_dim": self.input_dim,
        }


def resolve_position_encoding_config(
    request: PositionEncodingRequest,
    annotation_level: AnnotationLevel,
    *,
    latent_dim: int,
    num_heads: int,
    num_chromosomes: int,
) -> ResolvedPositionEncodingConfig:
    """Resolve and validate a pure position-encoding configuration."""
    _validate_request_type(request)
    _validate_enum("preset", request.preset, PositionPreset)
    _validate_enum("annotation_level", annotation_level, AnnotationLevel)
    _validate_positive_int("latent_dim", latent_dim)
    _validate_positive_int("num_heads", num_heads)
    _validate_non_negative_int("num_chromosomes", num_chromosomes)

    if request.preset is PositionPreset.LEGACY:
        _reject_legacy_overrides(request)
        absolute_encoding = (
            AbsolutePositionEncoding.NONE
            if annotation_level is AnnotationLevel.L0
            else AbsolutePositionEncoding.SINUSOIDAL
        )
        relative_encoding = RelativePositionEncoding.T5_BUCKET
        chromosome_encoding = ChromosomeEncoding.LEARNED
        cross_policy = CrossChromosomePolicy.SEPARATE
        normalized = PositionEncodingRequest(
            preset=PositionPreset.LEGACY,
            absolute_position_encoding=absolute_encoding,
            relative_position_encoding=relative_encoding,
            chromosome_encoding=chromosome_encoding,
            cross_chromosome_policy=cross_policy,
        )
    else:
        _require_custom_strategies(request)
        absolute_encoding = request.absolute_position_encoding
        relative_encoding = request.relative_position_encoding
        chromosome_encoding = request.chromosome_encoding
        cross_policy = (
            CrossChromosomePolicy.SEPARATE
            if request.cross_chromosome_policy is None
            else request.cross_chromosome_policy
        )
        _validate_enum("cross_chromosome_policy", cross_policy, CrossChromosomePolicy)
        normalized = request

    absolute = _resolve_absolute(normalized, absolute_encoding)
    relative = _resolve_relative(
        normalized,
        relative_encoding,
        cross_policy,
        latent_dim=latent_dim,
        num_heads=num_heads,
    )
    chromosome = _resolve_chromosome(
        chromosome_encoding,
        cross_policy,
        relative_encoding,
        absolute_encoding,
        num_chromosomes=num_chromosomes,
    )
    content_dim = get_content_feature_dimension(annotation_level)
    position_dim = absolute.position_dim or 0

    return ResolvedPositionEncodingConfig(
        schema_version=1,
        preset=request.preset,
        annotation_level=annotation_level,
        absolute=absolute,
        relative=relative,
        chromosome=chromosome,
        attribution=ResolvedAttributionConfig(default_ig_mode=ResolvedIGMode.CONTENT),
        content_dim=content_dim,
        input_dim=content_dim + position_dim,
    )


def resolved_position_encoding_from_dict(
    data: Mapping[str, object],
    *,
    latent_dim: int,
    num_heads: int,
) -> ResolvedPositionEncodingConfig:
    """Deserialize and canonical-validate a resolved position config.

    Serialized resolved configs are treated as claims about what the resolver
    produced during training. This helper reconstructs the unresolved request,
    calls the resolver again, and then compares every canonical field. That
    keeps the resolver as the single source of configuration math.

    Training-only ``chromosome.mapping`` and learned-binned
    ``absolute.binning`` extensions are validated but not copied into the
    returned dataclass.
    """
    if not isinstance(data, Mapping):
        raise ValueError("position_encoding must be a mapping")

    required = {
        "schema_version",
        "preset",
        "annotation_level",
        "absolute",
        "relative",
        "chromosome",
        "attribution",
        "content_dim",
        "input_dim",
    }
    _require_keys(data, required, "position_encoding")
    _reject_unknown_keys(data, required, "position_encoding")

    schema_version = _serialized_int(
        data["schema_version"],
        "position_encoding.schema_version",
    )
    if schema_version != 1:
        raise ValueError("position_encoding.schema_version must be 1")

    preset = _enum_from_serialized(
        data["preset"],
        PositionPreset,
        "position_encoding.preset",
    )
    annotation_level = _enum_from_serialized(
        data["annotation_level"],
        AnnotationLevel,
        "position_encoding.annotation_level",
    )
    absolute = _required_mapping(data, "absolute", "position_encoding.absolute")
    relative = _required_mapping(data, "relative", "position_encoding.relative")
    chromosome = _required_mapping(data, "chromosome", "position_encoding.chromosome")
    attribution = _required_mapping(data, "attribution", "position_encoding.attribution")

    _validate_serialized_absolute_section(absolute)
    _validate_serialized_relative_section(relative)
    _validate_serialized_chromosome_section(chromosome)
    _validate_serialized_attribution_section(attribution)

    absolute_encoding = _enum_from_serialized(
        absolute["type"],
        AbsolutePositionEncoding,
        "position_encoding.absolute.type",
    )
    relative_encoding = _enum_from_serialized(
        relative["type"],
        RelativePositionEncoding,
        "position_encoding.relative.type",
    )
    chromosome_encoding = _enum_from_serialized(
        chromosome["encoding"],
        ChromosomeEncoding,
        "position_encoding.chromosome.encoding",
    )
    cross_policy = _enum_from_serialized(
        chromosome["cross_chromosome_policy"],
        CrossChromosomePolicy,
        "position_encoding.chromosome.cross_chromosome_policy",
    )
    num_chromosomes = _serialized_int(
        chromosome["num_chromosomes"],
        "position_encoding.chromosome.num_chromosomes",
    )
    if "mapping" in chromosome:
        _validate_chromosome_mapping_extension(
            chromosome["mapping"],
            num_chromosomes=num_chromosomes,
        )
    if "binning" in absolute:
        if absolute_encoding is not AbsolutePositionEncoding.LEARNED_BINNED:
            raise ValueError(
                "position_encoding.absolute.binning is only valid for learned_binned"
            )
        if "mapping" not in chromosome:
            raise ValueError(
                "position_encoding.absolute.binning requires "
                "position_encoding.chromosome.mapping"
            )
        learned_binned_layout_from_absolute_dict(
            absolute,
            chromosome_mapping=chromosome["mapping"],
        )

    if preset is PositionPreset.LEGACY:
        request = PositionEncodingRequest(preset=PositionPreset.LEGACY)
    else:
        request = PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute_encoding,
            relative_position_encoding=relative_encoding,
            chromosome_encoding=chromosome_encoding,
            cross_chromosome_policy=cross_policy,
            **_absolute_request_kwargs(absolute_encoding, absolute),
            **_relative_request_kwargs(relative_encoding, relative),
        )

    resolved = resolve_position_encoding_config(
        request,
        annotation_level,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )
    canonical = resolved.to_dict()
    serialized_canonical = _strip_training_extensions(data)
    _assert_canonical_equal(
        serialized_canonical,
        canonical,
        "position_encoding",
    )
    return resolved


def _absolute_request_kwargs(
    encoding: AbsolutePositionEncoding,
    absolute: Mapping[str, object],
) -> dict[str, object]:
    if encoding is AbsolutePositionEncoding.NONE:
        return {}
    if encoding is AbsolutePositionEncoding.SINUSOIDAL:
        return {
            "position_dim": absolute["dim"],
            "sinusoidal_coordinate_scale": absolute["coordinate_scale"],
            "sinusoidal_max_wavelength": absolute["max_wavelength"],
        }
    if encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        return {
            "position_dim": absolute["dim"],
            "position_bin_size": absolute["bin_size_bp"],
        }
    raise ValueError(f"unsupported position_encoding.absolute.type: {encoding.value}")


def _relative_request_kwargs(
    encoding: RelativePositionEncoding,
    relative: Mapping[str, object],
) -> dict[str, object]:
    if encoding is RelativePositionEncoding.NONE:
        return {}
    if encoding is RelativePositionEncoding.T5_BUCKET:
        return {
            "num_position_buckets": relative["num_buckets"],
            "max_position_distance": relative["max_distance_bp"],
        }
    if encoding is RelativePositionEncoding.ROPE:
        return {
            "rope_coordinate_scale": relative["rope_coordinate_scale"],
            "rope_base": relative["rope_base"],
        }
    if encoding in {
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }:
        return {
            "alibi_distance_function": _enum_from_serialized(
                relative["alibi_distance_function"],
                AlibiDistanceFunction,
                "position_encoding.relative.alibi_distance_function",
            ),
            "alibi_distance_scale": relative["alibi_distance_scale"],
        }
    raise ValueError(f"unsupported position_encoding.relative.type: {encoding.value}")


def _resolve_absolute(
    request: PositionEncodingRequest,
    encoding: AbsolutePositionEncoding,
) -> ResolvedAbsolutePositionConfig:
    if encoding is AbsolutePositionEncoding.NONE:
        _reject_present(
            request,
            [
                "position_dim",
                "sinusoidal_coordinate_scale",
                "sinusoidal_max_wavelength",
                "position_bin_size",
            ],
            "absolute_position_encoding=none",
        )
        return ResolvedAbsolutePositionConfig(encoding=encoding)

    position_dim = DEFAULT_POSITION_DIM if request.position_dim is None else request.position_dim
    _validate_positive_int("position_dim", position_dim)

    if encoding is AbsolutePositionEncoding.SINUSOIDAL:
        _reject_present(request, ["position_bin_size"], "absolute_position_encoding=sinusoidal")
        if position_dim % 2 != 0:
            raise ValueError("position_dim must be even for absolute_position_encoding=sinusoidal")
        coordinate_scale = (
            DEFAULT_SINUSOIDAL_COORDINATE_SCALE
            if request.sinusoidal_coordinate_scale is None
            else request.sinusoidal_coordinate_scale
        )
        max_wavelength = (
            DEFAULT_SINUSOIDAL_MAX_WAVELENGTH
            if request.sinusoidal_max_wavelength is None
            else request.sinusoidal_max_wavelength
        )
        _validate_positive_number("sinusoidal_coordinate_scale", coordinate_scale)
        _validate_positive_number("sinusoidal_max_wavelength", max_wavelength)
        return ResolvedAbsolutePositionConfig(
            encoding=encoding,
            fusion=AbsolutePositionFusion.INPUT_CONCAT,
            position_dim=position_dim,
            coordinate_scale=coordinate_scale,
            max_wavelength=max_wavelength,
        )

    if encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        _reject_present(
            request,
            ["sinusoidal_coordinate_scale", "sinusoidal_max_wavelength"],
            "absolute_position_encoding=learned_binned",
        )
        bin_size = (
            DEFAULT_POSITION_BIN_SIZE
            if request.position_bin_size is None
            else request.position_bin_size
        )
        _validate_positive_int("position_bin_size", bin_size)
        return ResolvedAbsolutePositionConfig(
            encoding=encoding,
            fusion=AbsolutePositionFusion.INPUT_CONCAT,
            position_dim=position_dim,
            bin_size_bp=bin_size,
        )

    raise ValueError(f"unsupported absolute_position_encoding: {encoding!r}")


def _resolve_relative(
    request: PositionEncodingRequest,
    encoding: RelativePositionEncoding,
    cross_policy: CrossChromosomePolicy,
    *,
    latent_dim: int,
    num_heads: int,
) -> ResolvedRelativePositionConfig:
    if encoding is RelativePositionEncoding.NONE:
        _reject_present(
            request,
            [
                "num_position_buckets",
                "max_position_distance",
                "rope_coordinate_scale",
                "rope_base",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            "relative_position_encoding=none",
        )
        return ResolvedRelativePositionConfig(encoding=encoding)

    if encoding is RelativePositionEncoding.T5_BUCKET:
        _reject_present(
            request,
            [
                "rope_coordinate_scale",
                "rope_base",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            "relative_position_encoding=t5_bucket",
        )
        buckets = (
            DEFAULT_NUM_POSITION_BUCKETS
            if request.num_position_buckets is None
            else request.num_position_buckets
        )
        max_distance = (
            DEFAULT_MAX_POSITION_DISTANCE
            if request.max_position_distance is None
            else request.max_position_distance
        )
        _validate_t5_bucket_settings(buckets, max_distance)
        return ResolvedRelativePositionConfig(
            encoding=encoding,
            num_buckets=buckets,
            total_bias_rows=buckets + (1 if cross_policy is CrossChromosomePolicy.SEPARATE else 0),
            max_distance_bp=max_distance,
        )

    if encoding is RelativePositionEncoding.ROPE:
        _reject_present(
            request,
            [
                "num_position_buckets",
                "max_position_distance",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            "relative_position_encoding=rope",
        )
        if latent_dim % num_heads != 0:
            raise ValueError("rope requires latent_dim divisible by num_heads")
        head_dim = latent_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("rope requires an even per-head dimension")
        coordinate_scale = (
            DEFAULT_ROPE_COORDINATE_SCALE
            if request.rope_coordinate_scale is None
            else request.rope_coordinate_scale
        )
        rope_base = DEFAULT_ROPE_BASE if request.rope_base is None else request.rope_base
        _validate_positive_number("rope_coordinate_scale", coordinate_scale)
        _validate_positive_number("rope_base", rope_base)
        return ResolvedRelativePositionConfig(
            encoding=encoding,
            rope_coordinate_scale=coordinate_scale,
            rope_base=rope_base,
        )

    if encoding in {RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED}:
        _reject_present(
            request,
            [
                "num_position_buckets",
                "max_position_distance",
                "rope_coordinate_scale",
                "rope_base",
            ],
            f"relative_position_encoding={encoding.value}",
        )
        distance_function = (
            AlibiDistanceFunction(DEFAULT_ALIBI_DISTANCE_FUNCTION)
            if request.alibi_distance_function is None
            else request.alibi_distance_function
        )
        _validate_enum("alibi_distance_function", distance_function, AlibiDistanceFunction)
        distance_scale = (
            DEFAULT_ALIBI_DISTANCE_SCALE
            if request.alibi_distance_scale is None
            else request.alibi_distance_scale
        )
        _validate_positive_number("alibi_distance_scale", distance_scale)
        return ResolvedRelativePositionConfig(
            encoding=encoding,
            alibi_distance_function=distance_function,
            alibi_distance_scale=distance_scale,
        )

    raise ValueError(f"unsupported relative_position_encoding: {encoding!r}")


def _resolve_chromosome(
    encoding: ChromosomeEncoding,
    cross_policy: CrossChromosomePolicy,
    relative_encoding: RelativePositionEncoding,
    absolute_encoding: AbsolutePositionEncoding,
    *,
    num_chromosomes: int,
) -> ResolvedChromosomeConfig:
    learned_binned_absolute = absolute_encoding is AbsolutePositionEncoding.LEARNED_BINNED
    chromosome_aware_relative = relative_encoding in {
        RelativePositionEncoding.T5_BUCKET,
        RelativePositionEncoding.ROPE,
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }
    requires_chrom_ids = (
        encoding is ChromosomeEncoding.LEARNED
        or learned_binned_absolute
        or cross_policy is CrossChromosomePolicy.MASK
        or (cross_policy is CrossChromosomePolicy.SEPARATE and chromosome_aware_relative)
    )
    if requires_chrom_ids and num_chromosomes <= 0:
        raise ValueError(
            "num_chromosomes must be positive when chromosome-aware encoding, "
            "routing, or learned-binned absolute position is required"
        )

    if cross_policy is CrossChromosomePolicy.MASK:
        cross_parameter = "mask"
    elif (
        cross_policy is CrossChromosomePolicy.SEPARATE
        and relative_encoding is RelativePositionEncoding.T5_BUCKET
    ):
        cross_parameter = "dedicated_bucket"
    elif cross_policy is CrossChromosomePolicy.SEPARATE and relative_encoding in {
        RelativePositionEncoding.ROPE,
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }:
        cross_parameter = "learned_bias"
    else:
        cross_parameter = None

    return ResolvedChromosomeConfig(
        encoding=encoding,
        cross_chromosome_policy=cross_policy,
        num_chromosomes=num_chromosomes,
        requires_chrom_ids=requires_chrom_ids,
        cross_chromosome_parameter=cross_parameter,
    )


def _validate_request_type(request: PositionEncodingRequest) -> None:
    if not isinstance(request, PositionEncodingRequest):
        raise ValueError("request must be a PositionEncodingRequest")


def _validate_enum(field_name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a {enum_type.__name__} enum instance")


def _require_custom_strategies(request: PositionEncodingRequest) -> None:
    required_fields = [
        (
            "absolute_position_encoding",
            request.absolute_position_encoding,
            AbsolutePositionEncoding,
        ),
        (
            "relative_position_encoding",
            request.relative_position_encoding,
            RelativePositionEncoding,
        ),
        ("chromosome_encoding", request.chromosome_encoding, ChromosomeEncoding),
    ]
    for field_name, value, enum_type in required_fields:
        if value is None:
            raise ValueError(f"custom preset requires {field_name}")
        _validate_enum(field_name, value, enum_type)
    if request.cross_chromosome_policy is not None:
        _validate_enum(
            "cross_chromosome_policy",
            request.cross_chromosome_policy,
            CrossChromosomePolicy,
        )
    if request.alibi_distance_function is not None:
        _validate_enum(
            "alibi_distance_function",
            request.alibi_distance_function,
            AlibiDistanceFunction,
        )


def _reject_legacy_overrides(request: PositionEncodingRequest) -> None:
    fields = [
        "absolute_position_encoding",
        "relative_position_encoding",
        "chromosome_encoding",
        "cross_chromosome_policy",
        "position_dim",
        "sinusoidal_coordinate_scale",
        "sinusoidal_max_wavelength",
        "position_bin_size",
        "num_position_buckets",
        "max_position_distance",
        "rope_coordinate_scale",
        "rope_base",
        "alibi_distance_function",
        "alibi_distance_scale",
    ]
    _reject_present(request, fields, "position_preset=legacy")


def _reject_present(
    request: PositionEncodingRequest,
    field_names: list[str],
    context: str,
) -> None:
    present = [field_name for field_name in field_names if getattr(request, field_name) is not None]
    if present:
        raise ValueError(f"{', '.join(present)} inactive for {context}")


def _validate_positive_int(field_name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(field_name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_positive_number(field_name: str, value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be positive and finite")


def _validate_t5_bucket_settings(num_buckets: int, max_distance: int) -> None:
    _validate_positive_int("num_position_buckets", num_buckets)
    _validate_positive_int("max_position_distance", max_distance)
    if num_buckets < 4:
        raise ValueError("num_position_buckets must be at least 4")
    if num_buckets % 2 != 0:
        raise ValueError("num_position_buckets must be even")
    if max_distance <= num_buckets // 4:
        raise ValueError("max_position_distance must be greater than num_position_buckets // 4")


def _enum_value(value: Enum | None) -> str | None:
    if value is None:
        return None
    return value.value


def _required_mapping(
    data: Mapping[str, object],
    key: str,
    dotted_name: str,
) -> Mapping[str, object]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{dotted_name} must be a mapping")
    return value


def _require_keys(
    data: Mapping[str, object],
    required: set[str],
    dotted_name: str,
) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{dotted_name} missing required field: {missing[0]}")


def _reject_unknown_keys(
    data: Mapping[str, object],
    allowed: set[str],
    dotted_name: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{dotted_name} contains unknown field: {unknown[0]}")


def _enum_from_serialized(
    value: object,
    enum_type: type[Enum],
    dotted_name: str,
) -> Enum:
    if not isinstance(value, str):
        raise ValueError(f"{dotted_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{dotted_name} has invalid value: {value!r}") from exc


def _serialized_int(value: object, dotted_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be an integer")
    return value


def _validate_serialized_absolute_section(data: Mapping[str, object]) -> None:
    canonical = {
        "type",
        "fusion",
        "dim",
        "coordinate_scale",
        "max_wavelength",
        "bin_size_bp",
    }
    allowed = canonical | {"binning"}
    _require_keys(data, canonical, "position_encoding.absolute")
    _reject_unknown_keys(data, allowed, "position_encoding.absolute")
    if "binning" in data and not isinstance(data["binning"], Mapping):
        raise ValueError("position_encoding.absolute.binning must be a mapping")


def _validate_serialized_relative_section(data: Mapping[str, object]) -> None:
    allowed = {
        "type",
        "num_buckets",
        "total_bias_rows",
        "max_distance_bp",
        "rope_coordinate_scale",
        "rope_base",
        "alibi_distance_function",
        "alibi_distance_scale",
    }
    _require_keys(data, allowed, "position_encoding.relative")
    _reject_unknown_keys(data, allowed, "position_encoding.relative")


def _validate_serialized_chromosome_section(data: Mapping[str, object]) -> None:
    canonical = {
        "encoding",
        "cross_chromosome_policy",
        "num_chromosomes",
        "requires_chrom_ids",
        "cross_chromosome_parameter",
    }
    allowed = canonical | {"mapping"}
    _require_keys(data, canonical, "position_encoding.chromosome")
    _reject_unknown_keys(data, allowed, "position_encoding.chromosome")


def _validate_serialized_attribution_section(data: Mapping[str, object]) -> None:
    allowed = {"default_ig_mode"}
    _require_keys(data, allowed, "position_encoding.attribution")
    _reject_unknown_keys(data, allowed, "position_encoding.attribution")


def _validate_chromosome_mapping_extension(
    mapping: object,
    *,
    num_chromosomes: int,
) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError("position_encoding.chromosome.mapping must be a mapping")
    expected_keys = {str(idx) for idx in range(num_chromosomes)}
    actual_keys = set(mapping)
    if actual_keys != expected_keys:
        raise ValueError(
            "position_encoding.chromosome.mapping keys must exactly cover "
            "0..num_chromosomes-1"
        )
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError("position_encoding.chromosome.mapping keys must be strings")
        if not isinstance(value, str):
            raise ValueError("position_encoding.chromosome.mapping values must be strings")


def _strip_training_extensions(data: Mapping[str, object]) -> dict[str, object]:
    stripped = {}
    for key, value in data.items():
        if key == "chromosome" and isinstance(value, Mapping):
            stripped[key] = {
                sub_key: sub_value
                for sub_key, sub_value in value.items()
                if sub_key != "mapping"
            }
        elif key == "absolute" and isinstance(value, Mapping):
            stripped[key] = {
                sub_key: sub_value
                for sub_key, sub_value in value.items()
                if sub_key != "binning"
            }
        else:
            stripped[key] = value
    return stripped


def _assert_canonical_equal(
    serialized: object,
    canonical: object,
    dotted_name: str,
) -> None:
    if isinstance(serialized, Mapping) and isinstance(canonical, Mapping):
        if set(serialized) != set(canonical):
            missing = sorted(set(canonical) - set(serialized))
            extra = sorted(set(serialized) - set(canonical))
            if missing:
                raise ValueError(f"{dotted_name} missing canonical field: {missing[0]}")
            raise ValueError(f"{dotted_name} contains unknown field: {extra[0]}")
        for key in sorted(canonical):
            _assert_canonical_equal(
                serialized[key],
                canonical[key],
                f"{dotted_name}.{key}",
            )
        return
    if type(serialized) is not type(canonical):
        raise ValueError(
            f"{dotted_name} differs from resolver canonical type: "
            f"{type(serialized).__name__} != {type(canonical).__name__}"
        )
    if serialized != canonical:
        raise ValueError(
            f"{dotted_name} differs from resolver canonical value: "
            f"{serialized!r} != {canonical!r}"
        )
