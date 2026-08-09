"""Focused tests for train.py positional CLI plumbing."""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train
from src.encoding.levels import AnnotationLevel, get_feature_dimension
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionPreset,
    RelativePositionEncoding,
)


@pytest.fixture
def base_argv() -> list[str]:
    """Minimum argv accepted by the existing train parser."""
    return ["--level", "L3"]


def parse_with(base_argv: list[str], *extra: str):
    return train.parse_args([*base_argv, *extra])


def test_parser_default_position_preset_is_legacy(base_argv):
    args = parse_with(base_argv)
    assert args.position_preset == "legacy"


def test_all_optional_positional_parser_defaults_are_none(base_argv):
    args = parse_with(base_argv)

    positional_fields = [
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
    assert {field: getattr(args, field) for field in positional_fields} == dict.fromkeys(
        positional_fields
    )


@pytest.mark.parametrize("choice", [member.value for member in PositionPreset])
def test_position_preset_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--position-preset", choice)
    request = train.build_position_encoding_request(args)
    assert request.preset is PositionPreset(choice)


@pytest.mark.parametrize("choice", [member.value for member in AbsolutePositionEncoding])
def test_absolute_position_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--absolute-position-encoding", choice)
    request = train.build_position_encoding_request(args)
    assert request.absolute_position_encoding is AbsolutePositionEncoding(choice)


@pytest.mark.parametrize("choice", [member.value for member in RelativePositionEncoding])
def test_relative_position_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--relative-position-encoding", choice)
    request = train.build_position_encoding_request(args)
    assert request.relative_position_encoding is RelativePositionEncoding(choice)


@pytest.mark.parametrize("choice", [member.value for member in ChromosomeEncoding])
def test_chromosome_encoding_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--chromosome-encoding", choice)
    request = train.build_position_encoding_request(args)
    assert request.chromosome_encoding is ChromosomeEncoding(choice)


@pytest.mark.parametrize("choice", [member.value for member in CrossChromosomePolicy])
def test_cross_chromosome_policy_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--cross-chromosome-policy", choice)
    request = train.build_position_encoding_request(args)
    assert request.cross_chromosome_policy is CrossChromosomePolicy(choice)


@pytest.mark.parametrize("choice", [member.value for member in AlibiDistanceFunction])
def test_alibi_distance_function_choices_parse_and_convert(base_argv, choice):
    args = parse_with(base_argv, "--alibi-distance-function", choice)
    request = train.build_position_encoding_request(args)
    assert request.alibi_distance_function is AlibiDistanceFunction(choice)


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--position-preset", "compat"),
        ("--absolute-position-encoding", "fourier"),
        ("--relative-position-encoding", "relative_bias"),
        ("--chromosome-encoding", "one_hot"),
        ("--cross-chromosome-policy", "ignore"),
        ("--alibi-distance-function", "sqrt"),
    ],
)
def test_invalid_position_choices_raise_system_exit(base_argv, flag, bad_value):
    with pytest.raises(SystemExit):
        parse_with(base_argv, flag, bad_value)


def test_parser_help_generation_requires_no_data_or_training():
    parser = train.build_arg_parser()
    help_text = parser.format_help()
    assert "--position-preset" in help_text
    assert "--alibi-distance-scale" in help_text


def test_explicit_numeric_zero_values_survive_parsing(base_argv):
    args = parse_with(
        base_argv,
        "--position-dim",
        "0",
        "--sinusoidal-coordinate-scale",
        "0",
        "--sinusoidal-max-wavelength",
        "0",
        "--position-bin-size",
        "0",
        "--num-position-buckets",
        "0",
        "--max-position-distance",
        "0",
        "--rope-coordinate-scale",
        "0",
        "--rope-base",
        "0",
        "--alibi-distance-scale",
        "0",
    )

    assert args.position_dim == 0
    assert args.sinusoidal_coordinate_scale == 0.0
    assert args.sinusoidal_max_wavelength == 0.0
    assert args.position_bin_size == 0
    assert args.num_position_buckets == 0
    assert args.max_position_distance == 0
    assert args.rope_coordinate_scale == 0.0
    assert args.rope_base == 0.0
    assert args.alibi_distance_scale == 0.0


def test_default_legacy_request_contains_only_preset_and_none_optional_fields(base_argv):
    request = train.build_position_encoding_request(parse_with(base_argv))

    assert request.preset is PositionPreset.LEGACY
    assert request.absolute_position_encoding is None
    assert request.relative_position_encoding is None
    assert request.chromosome_encoding is None
    assert request.cross_chromosome_policy is None
    assert request.position_dim is None
    assert request.sinusoidal_coordinate_scale is None
    assert request.sinusoidal_max_wavelength is None
    assert request.position_bin_size is None
    assert request.num_position_buckets is None
    assert request.max_position_distance is None
    assert request.rope_coordinate_scale is None
    assert request.rope_base is None
    assert request.alibi_distance_function is None
    assert request.alibi_distance_scale is None


def test_valid_custom_request_conversion_produces_enum_instances(base_argv):
    args = parse_with(
        base_argv,
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        "sinusoidal",
        "--relative-position-encoding",
        "alibi_fixed",
        "--chromosome-encoding",
        "learned",
        "--cross-chromosome-policy",
        "mask",
        "--alibi-distance-function",
        "linear",
        "--position-dim",
        "64",
        "--alibi-distance-scale",
        "1000",
    )

    request = train.build_position_encoding_request(args)

    assert request.preset is PositionPreset.CUSTOM
    assert request.absolute_position_encoding is AbsolutePositionEncoding.SINUSOIDAL
    assert request.relative_position_encoding is RelativePositionEncoding.ALIBI_FIXED
    assert request.chromosome_encoding is ChromosomeEncoding.LEARNED
    assert request.cross_chromosome_policy is CrossChromosomePolicy.MASK
    assert request.alibi_distance_function is AlibiDistanceFunction.LINEAR
    assert request.position_dim == 64
    assert request.alibi_distance_scale == 1000.0


@pytest.mark.parametrize("level", list(AnnotationLevel))
def test_legacy_resolved_input_dim_matches_historical_dimensions(base_argv, level):
    args = parse_with(base_argv, "--level", level.value)

    resolved = train.prepare_training_position_encoding(
        args,
        level,
        num_chromosomes=24,
    )

    assert resolved.input_dim == get_feature_dimension(level)


def test_valid_supported_custom_configuration_resolves_for_training(base_argv):
    args = parse_with(
        base_argv,
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        "none",
        "--relative-position-encoding",
        "none",
        "--chromosome-encoding",
        "none",
    )

    resolved = train.prepare_training_position_encoding(
        args,
        AnnotationLevel.L3,
        num_chromosomes=0,
    )

    assert resolved.preset is PositionPreset.CUSTOM
    assert resolved.absolute.encoding is AbsolutePositionEncoding.NONE
    assert resolved.relative.encoding is RelativePositionEncoding.NONE
    assert resolved.chromosome.encoding is ChromosomeEncoding.NONE


def test_invalid_custom_configuration_raises_value_error_before_custom_guard(base_argv):
    args = parse_with(
        base_argv,
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        "sinusoidal",
        "--relative-position-encoding",
        "none",
        "--chromosome-encoding",
        "none",
        "--position-dim",
        "0",
    )

    with pytest.raises(ValueError, match="position_dim"):
        train.prepare_training_position_encoding(
            args,
            AnnotationLevel.L3,
            num_chromosomes=0,
        )


@pytest.mark.parametrize(
    "extra",
    [
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "sinusoidal",
            "--relative-position-encoding",
            "none",
            "--chromosome-encoding",
            "none",
            "--position-dim",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "sinusoidal",
            "--relative-position-encoding",
            "none",
            "--chromosome-encoding",
            "none",
            "--sinusoidal-coordinate-scale",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "sinusoidal",
            "--relative-position-encoding",
            "none",
            "--chromosome-encoding",
            "none",
            "--sinusoidal-max-wavelength",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "learned_binned",
            "--relative-position-encoding",
            "none",
            "--chromosome-encoding",
            "none",
            "--position-bin-size",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "t5_bucket",
            "--chromosome-encoding",
            "none",
            "--num-position-buckets",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "t5_bucket",
            "--chromosome-encoding",
            "none",
            "--max-position-distance",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "rope",
            "--chromosome-encoding",
            "none",
            "--rope-coordinate-scale",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "rope",
            "--chromosome-encoding",
            "none",
            "--rope-base",
            "0",
        ],
        [
            "--position-preset",
            "custom",
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "alibi_fixed",
            "--chromosome-encoding",
            "none",
            "--alibi-distance-scale",
            "0",
        ],
    ],
)
def test_zero_values_reach_resolver_validation(base_argv, extra):
    args = parse_with(base_argv, *extra)

    with pytest.raises(ValueError):
        train.prepare_training_position_encoding(
            args,
            AnnotationLevel.L3,
            num_chromosomes=24,
        )


def test_main_uses_one_shared_position_resolution_call_before_training_branch():
    source = inspect.getsource(train.main)
    call = "prepare_training_position_encoding("
    branch = "if args.cv is not None:"

    assert source.count(call) == 1
    assert source.index(call) < source.index(branch)
