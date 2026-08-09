"""
Genome build definitions for SIEVE.

This module centralises all reference-genome-dependent constants:
- Pseudoautosomal region (PAR) coordinates
- Sex chromosome contig identifiers
- Autosomal chromosome lists
- Contig harmonisation rules

All build-specific logic throughout the pipeline must import from this
module rather than hardcoding coordinates.

Supported builds: GRCh37, GRCh38
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Tuple

SUPPORTED_BUILDS = ('GRCh37', 'GRCh38')


@dataclass(frozen=True)
class GenomeBuild:
    """Immutable container for genome build parameters."""

    name: str
    par_regions: Dict[str, List[Tuple[int, int]]]
    sex_chroms: Tuple[str, ...]
    autosomal_chroms: Tuple[str, ...]
    x_contig_aliases: Tuple[str, ...]
    y_contig_aliases: Tuple[str, ...]
    chromosome_lengths: Mapping[str, int]


# Canonical chromosome lengths for the GRCh37 and GRCh38 primary assemblies.
# Autosomal and sex-chromosome lengths are the GRC primary assembled sequence
# lengths; MT uses the 16,569 bp rCRS mitochondrial sequence.
GRCH37_CHROMOSOME_LENGTHS = MappingProxyType(
    {
        "1": 249250621,
        "2": 243199373,
        "3": 198022430,
        "4": 191154276,
        "5": 180915260,
        "6": 171115067,
        "7": 159138663,
        "8": 146364022,
        "9": 141213431,
        "10": 135534747,
        "11": 135006516,
        "12": 133851895,
        "13": 115169878,
        "14": 107349540,
        "15": 102531392,
        "16": 90354753,
        "17": 81195210,
        "18": 78077248,
        "19": 59128983,
        "20": 63025520,
        "21": 48129895,
        "22": 51304566,
        "X": 155270560,
        "Y": 59373566,
        "MT": 16569,
    }
)

GRCH38_CHROMOSOME_LENGTHS = MappingProxyType(
    {
        "1": 248956422,
        "2": 242193529,
        "3": 198295559,
        "4": 190214555,
        "5": 181538259,
        "6": 170805979,
        "7": 159345973,
        "8": 145138636,
        "9": 138394717,
        "10": 133797422,
        "11": 135086622,
        "12": 133275309,
        "13": 114364328,
        "14": 107043718,
        "15": 101991189,
        "16": 90338345,
        "17": 83257441,
        "18": 80373285,
        "19": 58617616,
        "20": 64444167,
        "21": 46709983,
        "22": 50818468,
        "X": 156040895,
        "Y": 57227415,
        "MT": 16569,
    }
)


GRCH37 = GenomeBuild(
    name='GRCh37',
    par_regions={
        'X': [(60001, 2699520), (154931044, 155260560)],
        'Y': [(10001, 2649520), (59034050, 59363566)],
    },
    sex_chroms=('X', 'Y'),
    autosomal_chroms=tuple(str(c) for c in range(1, 23)),
    x_contig_aliases=('X', 'chrX', '23', 'chr23'),
    y_contig_aliases=('Y', 'chrY', '24', 'chr24'),
    chromosome_lengths=GRCH37_CHROMOSOME_LENGTHS,
)

GRCH38 = GenomeBuild(
    name='GRCh38',
    par_regions={
        'X': [(10001, 2781479), (155701383, 156030895)],
        'Y': [(10001, 2781479), (56887903, 57217415)],
    },
    sex_chroms=('X', 'Y'),
    autosomal_chroms=tuple(str(c) for c in range(1, 23)),
    x_contig_aliases=('X', 'chrX', '23', 'chr23'),
    y_contig_aliases=('Y', 'chrY', '24', 'chr24'),
    chromosome_lengths=GRCH38_CHROMOSOME_LENGTHS,
)

BUILDS = {
    'GRCh37': GRCH37,
    'GRCh38': GRCH38,
}


def get_genome_build(name: str) -> GenomeBuild:
    """
    Retrieve GenomeBuild by name.

    Accepts case-insensitive input and common aliases
    (hg19 -> GRCh37, hg38 -> GRCh38).

    Parameters
    ----------
    name : str
        Build name or alias.

    Returns
    -------
    GenomeBuild
        The corresponding build object.

    Raises
    ------
    ValueError
        If the build name is not recognised.
    """
    aliases = {
        'hg19': 'GRCh37', 'grch37': 'GRCh37', 'b37': 'GRCh37',
        'hg38': 'GRCh38', 'grch38': 'GRCh38', 'b38': 'GRCh38',
    }
    normalised = aliases.get(name.lower(), name)
    if normalised not in BUILDS:
        raise ValueError(
            f"Unsupported genome build '{name}'. "
            f"Supported: {', '.join(SUPPORTED_BUILDS)} "
            f"(aliases: hg19, hg38, b37, b38)"
        )
    return BUILDS[normalised]


def get_chromosome_lengths(genome_build: str | GenomeBuild) -> dict[str, int]:
    """Return copy-safe canonical chromosome lengths for a genome build."""
    build = get_genome_build(genome_build) if isinstance(genome_build, str) else genome_build
    if not isinstance(build, GenomeBuild):
        raise ValueError("genome_build must be a build name or GenomeBuild")
    return dict(build.chromosome_lengths)


def get_chromosome_length(
    genome_build: str | GenomeBuild,
    chromosome_name: str,
) -> int:
    """Return the canonical length for a supported normalized chromosome.

    Chromosome aliases are resolved through :func:`normalise_chrom`; unsupported
    alternate contigs are rejected so learned-bin tables are deterministic.
    """
    build = get_genome_build(genome_build) if isinstance(genome_build, str) else genome_build
    if not isinstance(build, GenomeBuild):
        raise ValueError("genome_build must be a build name or GenomeBuild")
    normalized = normalise_chrom(chromosome_name, build)
    try:
        return build.chromosome_lengths[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported chromosome '{chromosome_name}' for genome build {build.name}"
        ) from exc


def is_in_par(pos: int, chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a position falls within a pseudoautosomal region.

    Parameters
    ----------
    pos : int
        Genomic position (1-based).
    chrom : str
        Harmonised chromosome name (e.g. 'X', 'Y').
    build : GenomeBuild
        Genome build to use for PAR coordinates.

    Returns
    -------
    bool
        True if the position is inside a PAR.
    """
    regions = build.par_regions.get(chrom, [])
    return any(start <= pos <= end for start, end in regions)


def is_sex_chrom(chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a harmonised contig name is a sex chromosome.

    Parameters
    ----------
    chrom : str
        Harmonised chromosome name.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    bool
        True if the chromosome is X or Y.
    """
    return chrom in build.sex_chroms


def is_autosomal(chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a harmonised contig name is autosomal.

    Parameters
    ----------
    chrom : str
        Harmonised chromosome name.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    bool
        True if the chromosome is autosomal (1-22).
    """
    return chrom in build.autosomal_chroms


def normalise_chrom(contig: str, build: GenomeBuild) -> str:
    """
    Normalise a contig name to the canonical form for this build.

    Extends the existing ``harmonize_contig()`` logic to also handle
    numeric sex chromosome aliases (23 -> X, 24 -> Y).

    Parameters
    ----------
    contig : str
        Raw contig name from VCF (e.g. 'chr1', 'chrX', '23').
    build : GenomeBuild
        Genome build (used for future build-specific contig rules).

    Returns
    -------
    str
        Canonical contig name (e.g. '1', 'X').
    """
    # Strip chr prefix first (existing logic)
    stripped = contig[3:] if contig.startswith('chr') else contig
    # Handle numeric aliases
    if stripped == '23':
        return 'X'
    if stripped == '24':
        return 'Y'
    return stripped
