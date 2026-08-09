"""Tests for learned-binned absolute-position metadata."""

import pytest

from scripts import train
from src.data.genome import (
    get_chromosome_length,
    get_chromosome_lengths,
)
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
from src.encoding.position_layout import (
    LEARNED_BINNED_COORDINATE_ORIGIN,
    LEARNED_BINNED_LAYOUT,
    build_learned_binned_absolute_position_layout,
    learned_binned_layout_from_position_encoding_dict,
    validate_saved_chromosome_mapping_matches_chrom_index,
)


def _learned_binned_config(num_chromosomes: int = 3):
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.LEARNED_BINNED,
            relative_position_encoding=RelativePositionEncoding.NONE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            position_dim=8,
            position_bin_size=100000000,
        ),
        AnnotationLevel.L3,
        latent_dim=16,
        num_heads=2,
        num_chromosomes=num_chromosomes,
    )


def test_genome_lengths_are_copy_safe_and_alias_normalized():
    lengths = get_chromosome_lengths("GRCh37")

    assert lengths["1"] == 249250621
    assert lengths["X"] == 155270560
    assert lengths["MT"] == 16569
    assert get_chromosome_length("hg38", "chrX") == 156040895
    assert get_chromosome_length("GRCh38", "23") == 156040895
    lengths["1"] = 1
    assert get_chromosome_lengths("GRCh37")["1"] == 249250621


def test_genome_length_rejects_unsupported_build_and_contig():
    with pytest.raises(ValueError, match="Unsupported genome build"):
        get_chromosome_lengths("T2T")
    with pytest.raises(ValueError, match="Unsupported chromosome"):
        get_chromosome_length("GRCh37", "GL000207.1")


def test_layout_uses_coordinate_origin_one_and_chromosome_id_order():
    layout = build_learned_binned_absolute_position_layout(
        genome_build="GRCh37",
        chromosome_mapping={"2": "X", "0": "1", "1": "2"},
        bin_size_bp=100000000,
    )

    assert layout.coordinate_origin == LEARNED_BINNED_COORDINATE_ORIGIN
    assert layout.layout == LEARNED_BINNED_LAYOUT
    assert layout.chromosome_lengths_bp == (249250621, 243199373, 155270560)
    assert layout.bins_per_chromosome == (3, 3, 2)
    assert layout.chromosome_offsets == (0, 3, 6)
    assert layout.num_embeddings == 8
    assert layout.to_dict()["chromosome_lengths_bp"] == [249250621, 243199373, 155270560]


def test_layout_accepts_canonical_zero_based_string_chromosome_ids():
    layout = build_learned_binned_absolute_position_layout(
        genome_build="GRCh37",
        chromosome_mapping={"0": "1"},
        bin_size_bp=100000000,
    )

    assert layout.chromosome_lengths_bp == (249250621,)


@pytest.mark.parametrize(
    "chromosome_mapping",
    [
        {"00": "1"},
        {"0": "1", "01": "2"},
    ],
)
def test_layout_rejects_noncanonical_decimal_chromosome_id_spellings(
    chromosome_mapping,
):
    with pytest.raises(ValueError, match="canonical decimal chromosome IDs"):
        build_learned_binned_absolute_position_layout(
            genome_build="GRCh37",
            chromosome_mapping=chromosome_mapping,
            bin_size_bp=100000000,
        )


def test_layout_rejects_bad_mapping_and_bin_size_values():
    with pytest.raises(ValueError, match="positive integer"):
        build_learned_binned_absolute_position_layout(
            genome_build="GRCh37",
            chromosome_mapping={"0": "1"},
            bin_size_bp=0,
        )
    with pytest.raises(ValueError, match="exactly cover"):
        build_learned_binned_absolute_position_layout(
            genome_build="GRCh37",
            chromosome_mapping={"0": "1", "2": "2"},
            bin_size_bp=1000,
        )
    with pytest.raises(ValueError, match="values must be strings"):
        build_learned_binned_absolute_position_layout(
            genome_build="GRCh37",
            chromosome_mapping={"0": 1},
            bin_size_bp=1000,
        )
    with pytest.raises(ValueError, match="chromosome names must be unique"):
        build_learned_binned_absolute_position_layout(
            genome_build="GRCh37",
            chromosome_mapping={"0": "1", "1": "1"},
            bin_size_bp=1000,
        )


def test_saved_mapping_must_match_current_chrom_index_exactly():
    validate_saved_chromosome_mapping_matches_chrom_index(
        {"0": "1", "1": "2", "2": "X"},
        {"1": 0, "2": 1, "X": 2},
    )

    with pytest.raises(ValueError, match="exactly match"):
        validate_saved_chromosome_mapping_matches_chrom_index(
            {"0": "1", "1": "2", "2": "X"},
            {"1": 0, "2": 2, "X": 1},
        )
    with pytest.raises(ValueError, match="exactly match"):
        validate_saved_chromosome_mapping_matches_chrom_index(
            {"0": "1", "1": "2", "2": "X"},
            {"1": 0, "3": 1, "X": 2},
        )


def test_saved_mapping_validation_rejects_falsey_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        validate_saved_chromosome_mapping_matches_chrom_index([], {})


def test_training_serialization_attaches_learned_binned_layout_extension():
    config = _learned_binned_config()

    serialized = train.serialize_position_encoding_for_training(
        config,
        {"1": 0, "2": 1, "X": 2},
        genome_build="GRCh37",
    )

    assert serialized["absolute"]["binning"] == {
        "schema_version": 1,
        "coordinate_origin": 1,
        "layout": "chromosome_local_contiguous",
        "chromosome_lengths_bp": [249250621, 243199373, 155270560],
        "bins_per_chromosome": [3, 3, 2],
        "num_embeddings": 8,
    }
    parsed_layout = learned_binned_layout_from_position_encoding_dict(
        serialized,
        chromosome_mapping=serialized["chromosome"]["mapping"],
    )
    assert parsed_layout.num_embeddings == 8
    parsed_config = resolved_position_encoding_from_dict(
        serialized,
        latent_dim=16,
        num_heads=2,
    )
    assert parsed_config == config


@pytest.mark.parametrize(
    "absolute",
    [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL],
)
def test_training_serialization_omits_binning_for_non_learned_binned(absolute):
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=RelativePositionEncoding.NONE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            position_dim=8 if absolute is AbsolutePositionEncoding.SINUSOIDAL else None,
        ),
        AnnotationLevel.L3,
        latent_dim=16,
        num_heads=2,
        num_chromosomes=0,
    )

    serialized = train.serialize_position_encoding_for_training(
        config,
        {},
        genome_build="GRCh37",
    )

    assert "binning" not in serialized["absolute"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda data: data["absolute"]["binning"].update(schema_version=True),
            "schema_version",
        ),
        (
            lambda data: data["absolute"]["binning"].update(chromosome_lengths_bp=[1.0, 2, 3]),
            "chromosome_lengths_bp",
        ),
        (
            lambda data: data["absolute"]["binning"].update(bins_per_chromosome=[1, 1, 1]),
            "bins_per_chromosome",
        ),
        (
            lambda data: data["absolute"]["binning"].update(num_embeddings=1.0),
            "num_embeddings",
        ),
        (
            lambda data: data["absolute"]["binning"].update(num_embeddings=99),
            "num_embeddings",
        ),
    ],
)
def test_learned_binned_layout_parser_rejects_malformed_extension(mutator, message):
    serialized = train.serialize_position_encoding_for_training(
        _learned_binned_config(),
        {"1": 0, "2": 1, "X": 2},
        genome_build="GRCh37",
    )
    mutator(serialized)

    with pytest.raises(ValueError, match=message):
        resolved_position_encoding_from_dict(serialized, latent_dim=16, num_heads=2)


def test_binning_extension_on_non_learned_absolute_config_is_rejected():
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.NONE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
        ),
        AnnotationLevel.L3,
        latent_dim=16,
        num_heads=2,
        num_chromosomes=0,
    )
    data = config.to_dict()
    data["absolute"]["binning"] = {"schema_version": 1}

    with pytest.raises(ValueError, match="only valid for learned_binned"):
        resolved_position_encoding_from_dict(data, latent_dim=16, num_heads=2)


def test_training_serialization_rejects_non_standard_contigs_for_learned_binned():
    with pytest.raises(ValueError, match="Unsupported chromosome"):
        train.serialize_position_encoding_for_training(
            _learned_binned_config(num_chromosomes=1),
            {"GL000207.1": 0},
            genome_build="GRCh37",
        )
