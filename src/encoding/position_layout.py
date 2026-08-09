"""Pure learned-binned absolute-position layout metadata.

The Phase 8B1 layout is a serialized training/reconstruction extension. It
does not allocate Torch parameters or change model execution; it records the
deterministic chromosome-local bin table a future runtime will use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.data.genome import get_chromosome_length

LEARNED_BINNED_LAYOUT_SCHEMA_VERSION = 1
LEARNED_BINNED_COORDINATE_ORIGIN = 1
LEARNED_BINNED_LAYOUT = "chromosome_local_contiguous"


@dataclass(frozen=True)
class LearnedBinnedAbsolutePositionLayout:
    """Resolved learned-bin table metadata ordered by chromosome ID."""

    schema_version: int
    coordinate_origin: int
    layout: str
    chromosome_lengths_bp: tuple[int, ...]
    bins_per_chromosome: tuple[int, ...]
    num_embeddings: int

    @property
    def chromosome_offsets(self) -> tuple[int, ...]:
        """Derived prefix sums for chromosome-local contiguous table offsets."""
        offsets: list[int] = []
        running = 0
        for bin_count in self.bins_per_chromosome:
            offsets.append(running)
            running += bin_count
        return tuple(offsets)

    def to_dict(self) -> dict[str, object]:
        """Serialize as the ``position_encoding.absolute.binning`` extension."""
        return {
            "schema_version": self.schema_version,
            "coordinate_origin": self.coordinate_origin,
            "layout": self.layout,
            "chromosome_lengths_bp": list(self.chromosome_lengths_bp),
            "bins_per_chromosome": list(self.bins_per_chromosome),
            "num_embeddings": self.num_embeddings,
        }


def build_learned_binned_absolute_position_layout(
    *,
    genome_build: str,
    chromosome_mapping: Mapping[str, str],
    bin_size_bp: int,
) -> LearnedBinnedAbsolutePositionLayout:
    """Build deterministic chromosome-local learned-bin metadata."""
    _strict_positive_int(bin_size_bp, "position_encoding.absolute.bin_size_bp")
    chromosome_names = _chromosome_names_ordered_by_id(
        chromosome_mapping,
        dotted_name="position_encoding.chromosome.mapping",
    )
    lengths = tuple(
        get_chromosome_length(genome_build, chromosome_name) for chromosome_name in chromosome_names
    )
    bins = tuple(_ceil_div(length, bin_size_bp) for length in lengths)
    return LearnedBinnedAbsolutePositionLayout(
        schema_version=LEARNED_BINNED_LAYOUT_SCHEMA_VERSION,
        coordinate_origin=LEARNED_BINNED_COORDINATE_ORIGIN,
        layout=LEARNED_BINNED_LAYOUT,
        chromosome_lengths_bp=lengths,
        bins_per_chromosome=bins,
        num_embeddings=sum(bins),
    )


def learned_binned_layout_from_position_encoding_dict(
    data: Mapping[str, object],
    *,
    chromosome_mapping: Mapping[str, str],
) -> LearnedBinnedAbsolutePositionLayout:
    """Parse and validate ``position_encoding.absolute.binning`` metadata."""
    if not isinstance(data, Mapping):
        raise ValueError("position_encoding must be a mapping")
    absolute = data.get("absolute")
    if not isinstance(absolute, Mapping):
        raise ValueError("position_encoding.absolute must be a mapping")
    return learned_binned_layout_from_absolute_dict(
        absolute,
        chromosome_mapping=chromosome_mapping,
    )


def learned_binned_layout_from_absolute_dict(
    absolute: Mapping[str, object],
    *,
    chromosome_mapping: Mapping[str, str],
) -> LearnedBinnedAbsolutePositionLayout:
    """Parse and validate an absolute-section learned-bin extension."""
    if not isinstance(absolute, Mapping):
        raise ValueError("position_encoding.absolute must be a mapping")
    binning = absolute.get("binning")
    if not isinstance(binning, Mapping):
        raise ValueError("position_encoding.absolute.binning must be a mapping")
    bin_size_bp = _strict_positive_int(
        absolute.get("bin_size_bp"),
        "position_encoding.absolute.bin_size_bp",
    )
    chromosome_names = _chromosome_names_ordered_by_id(
        chromosome_mapping,
        dotted_name="position_encoding.chromosome.mapping",
    )

    allowed = {
        "schema_version",
        "coordinate_origin",
        "layout",
        "chromosome_lengths_bp",
        "bins_per_chromosome",
        "num_embeddings",
    }
    unknown = sorted(set(binning) - allowed)
    if unknown:
        raise ValueError(
            "position_encoding.absolute.binning contains unknown field: " f"{unknown[0]}"
        )
    missing = sorted(allowed - set(binning))
    if missing:
        raise ValueError(
            "position_encoding.absolute.binning missing required field: " f"{missing[0]}"
        )

    schema_version = _strict_positive_int(
        binning["schema_version"],
        "position_encoding.absolute.binning.schema_version",
    )
    if schema_version != LEARNED_BINNED_LAYOUT_SCHEMA_VERSION:
        raise ValueError("position_encoding.absolute.binning.schema_version must be 1")
    coordinate_origin = _strict_positive_int(
        binning["coordinate_origin"],
        "position_encoding.absolute.binning.coordinate_origin",
    )
    if coordinate_origin != LEARNED_BINNED_COORDINATE_ORIGIN:
        raise ValueError("position_encoding.absolute.binning.coordinate_origin must be 1")
    layout = binning["layout"]
    if layout != LEARNED_BINNED_LAYOUT:
        raise ValueError(
            "position_encoding.absolute.binning.layout must be " f"{LEARNED_BINNED_LAYOUT!r}"
        )

    lengths = _strict_positive_int_sequence(
        binning["chromosome_lengths_bp"],
        "position_encoding.absolute.binning.chromosome_lengths_bp",
    )
    bins = _strict_positive_int_sequence(
        binning["bins_per_chromosome"],
        "position_encoding.absolute.binning.bins_per_chromosome",
    )
    if len(lengths) != len(chromosome_names):
        raise ValueError(
            "position_encoding.absolute.binning chromosome_lengths_bp length "
            "must match chromosome.mapping"
        )
    if len(bins) != len(chromosome_names):
        raise ValueError(
            "position_encoding.absolute.binning bins_per_chromosome length "
            "must match chromosome.mapping"
        )

    expected_bins = tuple(_ceil_div(length, bin_size_bp) for length in lengths)
    if bins != expected_bins:
        raise ValueError(
            "position_encoding.absolute.binning.bins_per_chromosome "
            "must match chromosome_lengths_bp and bin_size_bp"
        )
    num_embeddings = _strict_positive_int(
        binning["num_embeddings"],
        "position_encoding.absolute.binning.num_embeddings",
    )
    if num_embeddings != sum(bins):
        raise ValueError(
            "position_encoding.absolute.binning.num_embeddings must equal "
            "sum(bins_per_chromosome)"
        )

    return LearnedBinnedAbsolutePositionLayout(
        schema_version=schema_version,
        coordinate_origin=coordinate_origin,
        layout=layout,
        chromosome_lengths_bp=lengths,
        bins_per_chromosome=bins,
        num_embeddings=num_embeddings,
    )


def validate_saved_chromosome_mapping_matches_chrom_index(
    saved_mapping: Mapping[str, str],
    chrom_index: Mapping[str, int],
) -> None:
    """Require serialized ID->name mapping to be the exact inverse of chrom_index."""
    if not isinstance(saved_mapping, Mapping):
        raise ValueError("position_encoding.chromosome.mapping must be a mapping")
    expected = _chromosome_id_to_name_from_chrom_index(chrom_index)
    if not saved_mapping and not expected:
        return
    names_by_id = _chromosome_names_ordered_by_id(
        saved_mapping,
        dotted_name="position_encoding.chromosome.mapping",
    )
    actual = {str(idx): name for idx, name in enumerate(names_by_id)}
    if actual != expected:
        raise ValueError(
            "position_encoding.chromosome.mapping must exactly match chrom_index inverse"
        )


def _chromosome_id_to_name_from_chrom_index(
    chrom_index: Mapping[str, int],
) -> dict[str, str]:
    if not isinstance(chrom_index, Mapping):
        raise ValueError("chrom_index must be a mapping")
    if not chrom_index:
        return {}
    seen_ids = set()
    by_id: dict[int, str] = {}
    for name, idx in chrom_index.items():
        if not isinstance(name, str):
            raise ValueError("chrom_index keys must be strings")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError("chrom_index values must be non-negative integers")
        if idx in seen_ids:
            raise ValueError("chrom_index values must be unique")
        seen_ids.add(idx)
        by_id[idx] = name
    expected_ids = set(range(len(chrom_index)))
    if set(by_id) != expected_ids:
        raise ValueError("chrom_index values must be contiguous from 0")
    return {str(idx): by_id[idx] for idx in range(len(chrom_index))}


def _chromosome_names_ordered_by_id(
    chromosome_mapping: Mapping[str, str],
    *,
    dotted_name: str,
) -> tuple[str, ...]:
    if not isinstance(chromosome_mapping, Mapping):
        raise ValueError(f"{dotted_name} must be a mapping")
    if not chromosome_mapping:
        raise ValueError(f"{dotted_name} must not be empty")
    by_id: dict[int, str] = {}
    for raw_id, name in chromosome_mapping.items():
        if not isinstance(raw_id, str):
            raise ValueError(f"{dotted_name} keys must be strings")
        if not raw_id.isdecimal():
            raise ValueError(f"{dotted_name} keys must be decimal chromosome IDs")
        chrom_id = int(raw_id)
        if raw_id != str(chrom_id):
            raise ValueError(f"{dotted_name} keys must use canonical decimal chromosome IDs")
        if not isinstance(name, str):
            raise ValueError(f"{dotted_name} values must be strings")
        if chrom_id in by_id:
            raise ValueError(f"{dotted_name} chromosome IDs must be unique")
        by_id[chrom_id] = name
    expected_ids = set(range(len(chromosome_mapping)))
    if set(by_id) != expected_ids:
        raise ValueError(f"{dotted_name} keys must exactly cover 0..num_chromosomes-1")
    if len(set(by_id.values())) != len(by_id):
        raise ValueError(f"{dotted_name} chromosome names must be unique")
    return tuple(by_id[idx] for idx in range(len(by_id)))


def _strict_positive_int(value: object, dotted_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{dotted_name} must be a positive integer")
    return value


def _strict_positive_int_sequence(value: object, dotted_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{dotted_name} must be a list or tuple")
    return tuple(
        _strict_positive_int(item, f"{dotted_name}[{idx}]") for idx, item in enumerate(value)
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
