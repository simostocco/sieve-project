#!/usr/bin/env python3
"""
Compare null-contrasted variant rankings across annotation levels L0-L3.

After running the per-level null baseline workflow, this script compares the
resulting significance-annotated variant rankings to quantify how much the
discovered variants depend on the annotation information provided. Key
analyses:

1. Jaccard similarity matrices at multiple top-k thresholds
2. Level-specific variant discovery (high rank at one level, low at others)

Note on score column choice
---------------------------
``delta_rank`` is the recommended column for cross-level comparison. It is
scale-free, stable across annotation levels, and is the primary variant
ranking metric.

``z_attribution`` is a per-chromosome z-score. Z-scoring within each
chromosome removes the between-chromosome component of the signal, which
flattens genome-wide differences and makes the column unsuitable for
cross-level ranking comparison. It is retained as a visualisation score for
Manhattan plots and for continuity with earlier runs.

``empirical_p_variant`` is bounded below by 1/(N_null + 1), which pins most
real variants at the floor when the model is informative, making top-K
selection a random draw from a tied set.

Usage:
    # From a directory with L{0..3}_sieve_variant_rankings.csv files
    python scripts/compare_ablation_rankings.py \\
        --ranking-dir results/ablation \\
        --score-column delta_rank \\
        --out-comparison ablation_ranking_comparison.yaml

    # With explicit per-level paths (using chrX-corrected files which contain z_attribution)
    python scripts/compare_ablation_rankings.py \\
        --rankings L0:results/null_baseline_L0/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
                   L1:results/null_baseline_L1/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
                   L2:results/null_baseline_L2/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
        --score-column z_attribution \\
        --out-comparison ablation_ranking_comparison.yaml

Author: Francesco Lescai
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

if __package__ in {None, ""}:
    from position_benchmark_metadata import (
        ComparisonContext,
        ExplanationContext,
        PositionStrategyIdentity,
        extract_comparison_context,
        extract_explanation_context,
        position_strategy_identity,
        require_compatible_contexts,
        require_compatible_explanation_contexts,
    )
else:
    from .position_benchmark_metadata import (
        ComparisonContext,
        ExplanationContext,
        PositionStrategyIdentity,
        extract_comparison_context,
        extract_explanation_context,
        position_strategy_identity,
        require_compatible_contexts,
        require_compatible_explanation_contexts,
    )


LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
POSITION_SCORE_COLUMNS = {
    "rank": "ascending",
    "mean_attribution": "descending",
    "max_attribution": "descending",
}
POSITION_RANKING_PROVENANCE_COLUMNS = (
    "resolved_ig_mode",
    "attribution_feature_space",
    "variant_score_aggregation",
)
DEFERRED_CALIBRATED_SCORE_COLUMNS = {
    "delta_rank",
    "z_attribution",
    "p_rank_boot",
    "rank_real",
    "median_rank_null_boot",
    "corrected_rank",
}

# ---------------------------------------------------------------------------
# YAML output helper
# ---------------------------------------------------------------------------


def dump_yaml(value: Any, path: pathlib.Path) -> None:
    """
    Write a Python object as YAML to *path*.

    Parameters
    ----------
    value : Any
        Data structure to serialise.
    path : pathlib.Path
        Destination file.
    """
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


# ---------------------------------------------------------------------------
# CSV loading, flexible column matching
# ---------------------------------------------------------------------------

# Primary SIEVE columns produced by src/explain/variant_ranking.py and the
# null-contrast workflow.
VARIANT_ID_COLUMNS = [
    "variant_id",
    "feature",
    "feature_id",
]
SCORE_COLUMNS = [
    "empirical_p_variant",
    "fdr_variant",
    "z_attribution",
    "corrected_rank",
    "mean_attribution",
    "score",
    "attribution",
    "max_attribution",
    "mean_score",
    "importance",
]
GENE_COLUMNS = ["gene_name", "gene", "gene_symbol"]
GENE_ID_COLUMNS = ["gene_id"]
CHROM_COLUMNS = ["chromosome", "chrom", "chr"]
POS_COLUMNS = ["position", "pos", "start"]
_DESCENDING_RANK_COLUMNS = frozenset({"delta_rank"})
_ASCENDING_SCORE_COLUMNS = frozenset(
    {
        "p_rank_boot",
        "rank_real",
        "median_rank_null_boot",
    }
)


@dataclass(frozen=True)
class PositionRunSpec:
    """CLI-supplied files for one positional ranking run."""

    run_id: str
    config_path: pathlib.Path
    ranking_path: pathlib.Path
    analysis_metadata_path: pathlib.Path


@dataclass(frozen=True)
class PositionRankingRun:
    """Loaded metadata and rankings for one positional ranking run."""

    spec: PositionRunSpec
    config: dict[str, Any]
    analysis_metadata: dict[str, Any]
    identity: PositionStrategyIdentity
    training_context: ComparisonContext
    explanation_context: ExplanationContext
    rankings: list[dict[str, Any]]


def _find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    lower_headers = {h.lower(): h for h in headers}
    for candidate in candidates:
        if candidate.lower() in lower_headers:
            return lower_headers[candidate.lower()]
    return None


def _build_variant_id(row: Dict[str, str], headers: List[str]) -> Optional[str]:
    """
    Build a unique variant identifier from a ranking CSV row.

    Preferred format is ``{chrom}:{pos}_{gene_id}`` which matches SIEVE's
    chromosome-aware variant keying and prevents position collisions across
    chromosomes.
    """
    # Try explicit variant_id column first
    vid_col = _find_column(headers, VARIANT_ID_COLUMNS)
    if vid_col and row.get(vid_col, "").strip():
        return row[vid_col].strip()

    # Build from components (chrom + pos + gene_id)
    chrom_col = _find_column(headers, CHROM_COLUMNS)
    pos_col = _find_column(headers, POS_COLUMNS)
    gene_id_col = _find_column(headers, GENE_ID_COLUMNS)

    if chrom_col and pos_col:
        chrom = row.get(chrom_col, "").strip()
        pos = row.get(pos_col, "").strip()
        gene_id = row.get(gene_id_col, "").strip() if gene_id_col else ""
        if chrom and pos:
            base = f"{chrom}:{pos}"
            if gene_id:
                return f"{base}_{gene_id}"
            return base

    return None


def _resolve_score_column(
    headers: List[str],
    score_column: Optional[str] = None,
) -> Tuple[str, bool]:
    """Resolve which score column to use for a given set of headers.

    Parameters
    ----------
    headers : List[str]
        Column headers from the CSV.
    score_column : str, optional
        Explicit column name override.

    Returns
    -------
    Tuple[str, bool]
        (resolved_column_name, was_explicit). *was_explicit* is True when
        the returned column came from *score_column* rather than auto-detection.
    """
    if score_column is not None:
        lower_headers = {h.lower(): h for h in headers}
        if score_column.lower() in lower_headers:
            return lower_headers[score_column.lower()], True
        raise ValueError(
            f"--score-column '{score_column}' not found in headers: {headers}"
        )

    # Auto-detect from SCORE_COLUMNS list
    col = _find_column(headers, SCORE_COLUMNS)
    return (col or ""), False


def _score_column_is_ascending(score_column: str) -> bool:
    """Return True when lower values indicate stronger ranking."""
    lowered = score_column.lower()
    if lowered in _DESCENDING_RANK_COLUMNS:
        return False
    if lowered in _ASCENDING_SCORE_COLUMNS:
        return True
    return (
        lowered.startswith("empirical_p")
        or lowered.startswith("fdr")
        or lowered.endswith("_rank")
        or lowered == "rank"
    )


def _get_score(
    row: Dict[str, str],
    resolved_col: str,
    missing_value: float,
) -> float:
    """Extract attribution score from a row using a pre-resolved column.

    Parameters
    ----------
    row : Dict[str, str]
        A single CSV row.
    resolved_col : str
        Column name (already resolved by :func:`_resolve_score_column`).

    Returns
    -------
    float
        The numeric score value, or *missing_value* on failure.
    """
    if resolved_col:
        try:
            return float(row[resolved_col])
        except (ValueError, TypeError, KeyError):
            return missing_value
    return missing_value


def _get_field(
    row: Dict[str, str], headers: List[str], candidates: List[str]
) -> str:
    """Get a field value from a row, trying multiple column names."""
    col = _find_column(headers, candidates)
    if col:
        return row.get(col, "").strip()
    return ""


def load_rankings(
    csv_path: pathlib.Path,
    score_column: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Load a variant ranking CSV and return a list of dicts sorted by score.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to a SIEVE variant ranking CSV.
    score_column : str, optional
        Explicit column name to use for ranking. When *None*, auto-detects
        from :data:`SCORE_COLUMNS`.

    Returns
    -------
    Tuple[List[Dict[str, Any]], str, bool]
        (records, resolved_column_name, was_explicit). Records have keys
        ``variant_id``, ``score``, ``gene``, ``chrom``, ``pos``, ``rank``.
    """
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        resolved_col, was_explicit = _resolve_score_column(headers, score_column)
        ascending = _score_column_is_ascending(resolved_col)
        missing_value = float("inf") if ascending else float("-inf")
        rows: List[Dict[str, Any]] = []
        for row in reader:
            vid = _build_variant_id(row, headers)
            if vid is None:
                continue
            rows.append(
                {
                    "variant_id": vid,
                    "score": _get_score(row, resolved_col, missing_value),
                    "gene": _get_field(row, headers, GENE_COLUMNS),
                    "chrom": _get_field(row, headers, CHROM_COLUMNS),
                    "pos": _get_field(row, headers, POS_COLUMNS),
                }
            )

    rows.sort(key=lambda r: r["score"], reverse=not ascending)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows, resolved_col, was_explicit


def _resolve_position_score_column(headers: list[str], score_column: str | None) -> str:
    """Resolve and validate an explicitly requested raw explanation score column."""
    if score_column is None:
        raise ValueError("position comparison requires explicit --score-column")

    requested = score_column.lower()
    if (
        requested in DEFERRED_CALIBRATED_SCORE_COLUMNS
        or requested.startswith("empirical_p")
        or requested.startswith("fdr")
    ):
        raise ValueError(
            "calibrated/null-derived ranking comparison is deferred until "
            "provenance can be validated in Phase 12C"
        )
    if requested not in POSITION_SCORE_COLUMNS:
        allowed = ", ".join(sorted(POSITION_SCORE_COLUMNS))
        raise ValueError(
            f"position comparison --score-column must be one of: {allowed}"
        )

    lower_headers = {h.lower(): h for h in headers}
    if requested not in lower_headers:
        raise ValueError(f"--score-column '{score_column}' not found in headers: {headers}")
    return lower_headers[requested]


def _build_position_variant_id(
    row: dict[str, str],
    headers: list[str],
    *,
    run_id: str,
    row_number: int,
) -> str:
    """Build the strict position-mode variant key from one CSV row."""
    vid_col = _find_column(headers, VARIANT_ID_COLUMNS)
    if vid_col and row.get(vid_col, "").strip():
        return row[vid_col].strip()

    chrom_col = _find_column(headers, CHROM_COLUMNS)
    pos_col = _find_column(headers, POS_COLUMNS)
    gene_id_col = _find_column(headers, GENE_ID_COLUMNS)
    components = {
        "chromosome": row.get(chrom_col, "").strip() if chrom_col else "",
        "position": row.get(pos_col, "").strip() if pos_col else "",
        "gene_id": row.get(gene_id_col, "").strip() if gene_id_col else "",
    }
    missing = [name for name, value in components.items() if not value]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"run {run_id!r} row {row_number} cannot build variant_id; "
            f"missing components: {missing_text}"
        )
    return f"{components['chromosome']}:{components['position']}_{components['gene_id']}"


def _parse_position_score(
    value: str,
    *,
    run_id: str,
    variant_id: str,
    score_column: str,
) -> float:
    """Parse a strict finite score for position-mode ranking."""
    if value is None or not str(value).strip():
        raise ValueError(
            f"run {run_id!r} variant {variant_id!r} has empty {score_column} score"
        )
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"run {run_id!r} variant {variant_id!r} has non-numeric "
            f"{score_column} score: {value!r}"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            f"run {run_id!r} variant {variant_id!r} has non-finite "
            f"{score_column} score: {value!r}"
        )
    return score


def load_position_rankings(
    csv_path: pathlib.Path,
    *,
    run_id: str,
    score_column: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Load a strict raw explanation ranking CSV for position-mode comparison."""
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        resolved_col = _resolve_position_score_column(headers, score_column)
        sort_order = POSITION_SCORE_COLUMNS[resolved_col.lower()]
        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            vid = _build_position_variant_id(
                row,
                headers,
                run_id=run_id,
                row_number=row_number,
            )
            score = _parse_position_score(
                row.get(resolved_col, ""),
                run_id=run_id,
                variant_id=vid,
                score_column=resolved_col,
            )
            rows.append(
                {
                    "variant_id": vid,
                    "score": score,
                    "gene": _get_field(row, headers, GENE_COLUMNS),
                    "chrom": _get_field(row, headers, CHROM_COLUMNS),
                    "pos": _get_field(row, headers, POS_COLUMNS),
                }
            )

    counts = Counter(row["variant_id"] for row in rows)
    duplicates = sorted(variant_id for variant_id, count in counts.items() if count > 1)
    if duplicates:
        examples = ", ".join(duplicates[:5])
        raise ValueError(
            f"run {run_id!r} contains {len(duplicates)} duplicate variant_id "
            f"values; examples: {examples}"
        )

    if sort_order == "ascending":
        rows.sort(key=lambda row: (row["score"], row["variant_id"]))
    else:
        rows.sort(key=lambda row: (-row["score"], row["variant_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, resolved_col, sort_order


def load_position_ranking_provenance(
    csv_path: pathlib.Path,
    *,
    run_id: str,
) -> dict[str, str]:
    """Load constant ranking provenance fields from a position-mode CSV."""
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        lower_headers = {header.lower(): header for header in headers}
        resolved_columns = {}
        for field in POSITION_RANKING_PROVENANCE_COLUMNS:
            if field not in lower_headers:
                raise ValueError(
                    f"run {run_id!r} ranking CSV is missing provenance column {field!r}"
                )
            resolved_columns[field] = lower_headers[field]

        values_by_field = {field: set() for field in POSITION_RANKING_PROVENANCE_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            for field, column in resolved_columns.items():
                value = row.get(column, "").strip()
                if not value:
                    raise ValueError(
                        f"run {run_id!r} ranking CSV row {row_number} has empty "
                        f"provenance field {field!r}"
                    )
                values_by_field[field].add(value)

    provenance = {}
    for field, values in values_by_field.items():
        if len(values) != 1:
            examples = sorted(values)
            raise ValueError(
                f"run {run_id!r} ranking CSV provenance field {field!r} has "
                f"inconsistent row values: {examples}"
            )
        provenance[field] = next(iter(values))
    return provenance


def find_ranking_files(ranking_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """
    Discover variant ranking CSVs in *ranking_dir*.

    Looks for filenames matching ``L{0,1,2,3}_sieve_variant_rankings.csv``
    first, then falls back to more flexible globbing.

    Parameters
    ----------
    ranking_dir : pathlib.Path
        Directory to scan.

    Returns
    -------
    Dict[str, pathlib.Path]
        Mapping of level label to file path.
    """
    level_files: Dict[str, pathlib.Path] = {}
    for level in LEVEL_ORDER:
        # Exact match first
        pattern = f"{level}_sieve_variant_rankings.csv"
        candidates = list(ranking_dir.glob(pattern))
        if candidates:
            level_files[level] = candidates[0]
            continue
        # Flexible match
        for f in ranking_dir.glob(f"{level}_*variant*rank*.csv"):
            level_files[level] = f
            break
    return level_files


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_jaccard(set_a: Set[str], set_b: Set[str]) -> Tuple[float, int, int]:
    """
    Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set_a, set_b : Set[str]
        Sets of variant IDs.

    Returns
    -------
    Tuple[float, int, int]
        (jaccard_index, intersection_size, union_size)
    """
    if not set_a and not set_b:
        return 0.0, 0, 0
    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard, len(intersection), len(union)


def compute_jaccard_matrices(
    level_rankings: Dict[str, List[Dict[str, Any]]],
    top_k_values: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Compute pairwise Jaccard similarity matrices for each top-k threshold.

    Parameters
    ----------
    level_rankings : Dict[str, List[Dict[str, Any]]]
        Rankings per annotation level.
    top_k_values : List[int]
        Top-k thresholds to evaluate.

    Returns
    -------
    Dict[int, List[Dict[str, Any]]]
        Mapping of top_k to list of pairwise comparison records.
    """
    matrices: Dict[int, List[Dict[str, Any]]] = {}
    levels = sorted(
        level_rankings.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )

    for top_k in top_k_values:
        top_sets: Dict[str, Set[str]] = {}
        for level in levels:
            ranked = level_rankings[level]
            top_sets[level] = {r["variant_id"] for r in ranked[:top_k]}

        rows: List[Dict[str, Any]] = []
        for la, lb in itertools.combinations(levels, 2):
            jaccard, overlap, union_size = compute_jaccard(
                top_sets[la], top_sets[lb]
            )
            rows.append(
                {
                    "top_k": top_k,
                    "level_a": la,
                    "level_b": lb,
                    "jaccard": round(jaccard, 4),
                    "overlap": overlap,
                    "size_a": len(top_sets[la]),
                    "size_b": len(top_sets[lb]),
                    "union": union_size,
                }
            )
        matrices[top_k] = rows

    return matrices


def find_level_specific_variants(
    level_rankings: Dict[str, List[Dict[str, Any]]],
    high_rank_threshold: int,
    low_rank_threshold: int,
) -> List[Dict[str, Any]]:
    """
    Find variants ranked highly at one level but poorly at all others.

    A variant is *level-specific* if it appears in the top
    ``high_rank_threshold`` at exactly one level and is outside the top
    ``low_rank_threshold`` at every other level.

    Parameters
    ----------
    level_rankings : Dict[str, List[Dict[str, Any]]]
        Rankings per annotation level.
    high_rank_threshold : int
        Must be within this rank at the specific level.
    low_rank_threshold : int
        Must be outside this rank at all other levels.

    Returns
    -------
    List[Dict[str, Any]]
        Level-specific variant records.
    """
    levels = sorted(
        level_rankings.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )

    # Build rank lookup: level -> variant_id -> rank
    rank_lookup: Dict[str, Dict[str, int]] = {}
    info_lookup: Dict[str, Dict[str, Any]] = {}
    total_variants: Dict[str, int] = {}

    for level in levels:
        ranked = level_rankings[level]
        total_variants[level] = len(ranked)
        rank_lookup[level] = {}
        for r in ranked:
            rank_lookup[level][r["variant_id"]] = r["rank"]
            if r["variant_id"] not in info_lookup:
                info_lookup[r["variant_id"]] = {
                    "gene": r.get("gene", ""),
                    "chrom": r.get("chrom", ""),
                    "pos": r.get("pos", ""),
                }

    results: List[Dict[str, Any]] = []
    for level in levels:
        other_levels = [l for l in levels if l != level]
        ranked = level_rankings[level]
        top_at_level = [r for r in ranked if r["rank"] <= high_rank_threshold]

        for variant_row in top_at_level:
            vid = variant_row["variant_id"]
            is_specific = True
            for other in other_levels:
                other_rank = rank_lookup[other].get(
                    vid, total_variants[other] + 1
                )
                if other_rank <= low_rank_threshold:
                    is_specific = False
                    break

            if is_specific:
                info = info_lookup.get(vid, {})
                entry: Dict[str, Any] = {
                    "variant_id": vid,
                    "gene": info.get("gene", ""),
                    "chrom": info.get("chrom", ""),
                    "pos": info.get("pos", ""),
                    "specific_to_level": level,
                    "rank_at_specific_level": variant_row["rank"],
                    "score_at_specific_level": variant_row["score"],
                }
                # Cross-level ranks
                for l in LEVEL_ORDER:
                    if l in rank_lookup:
                        entry[f"rank_at_{l}"] = rank_lookup[l].get(
                            vid, total_variants.get(l, 0) + 1
                        )
                results.append(entry)

    return results


def _parse_top_k_values(raw_top_k: str, *, require_positive: bool = False) -> list[int]:
    """Parse comma-separated top-k values."""
    values = [int(k.strip()) for k in raw_top_k.split(",")]
    if require_positive and any(value <= 0 for value in values):
        raise ValueError("--top-k values must be positive integers")
    return values


def _validate_position_universe(runs: list[PositionRankingRun]) -> set[str]:
    """Require every position run to contain the exact same variant IDs."""
    ordered_runs = sorted(runs, key=lambda run: run.spec.run_id)
    reference = ordered_runs[0]
    reference_ids = {row["variant_id"] for row in reference.rankings}
    for run in ordered_runs[1:]:
        run_ids = {row["variant_id"] for row in run.rankings}
        if run_ids != reference_ids:
            missing = sorted(reference_ids - run_ids)
            extra = sorted(run_ids - reference_ids)
            raise ValueError(
                f"variant universe mismatch: reference run {reference.spec.run_id!r} "
                f"has {len(reference_ids)} variants but run {run.spec.run_id!r} "
                f"has {len(run_ids)} variants; missing count {len(missing)}, "
                f"extra count {len(extra)}; missing examples {missing[:5]}, "
                f"extra examples {extra[:5]}"
            )
    return reference_ids


def compute_position_jaccard_matrices(
    runs: list[PositionRankingRun],
    top_k_values: list[int],
    *,
    score_column: str,
    score_sort_order: str,
) -> dict[int, list[dict[str, Any]]]:
    """Compute pairwise top-k Jaccard rows for position runs."""
    matrices: dict[int, list[dict[str, Any]]] = {}
    ordered_runs = sorted(runs, key=lambda run: run.spec.run_id)

    for top_k in top_k_values:
        top_sets: dict[str, set[str]] = {}
        for run in ordered_runs:
            top_sets[run.spec.run_id] = {
                row["variant_id"] for row in run.rankings[:top_k]
            }

        rows: list[dict[str, Any]] = []
        for run_a, run_b in itertools.combinations(ordered_runs, 2):
            run_id_a = run_a.spec.run_id
            run_id_b = run_b.spec.run_id
            set_a = top_sets[run_id_a]
            set_b = top_sets[run_id_b]
            jaccard, overlap, union_size = compute_jaccard(set_a, set_b)
            rows.append(
                {
                    "top_k": top_k,
                    "run_id_a": run_id_a,
                    "run_id_b": run_id_b,
                    "position_strategy_id_a": run_a.identity.strategy_id,
                    "position_strategy_id_b": run_b.identity.strategy_id,
                    "jaccard": round(jaccard, 4),
                    "overlap": overlap,
                    "size_a": len(set_a),
                    "size_b": len(set_b),
                    "union": union_size,
                    "score_column": score_column,
                    "score_sort_order": score_sort_order,
                }
            )
        matrices[top_k] = rows

    return matrices


def find_strategy_specific_variants(
    runs: list[PositionRankingRun],
    high_rank_threshold: int,
    low_rank_threshold: int,
) -> list[dict[str, Any]]:
    """Find variants high-ranked in one positional strategy and low-ranked elsewhere."""
    ordered_runs = sorted(runs, key=lambda run: run.spec.run_id)
    rank_lookup = {
        run.spec.run_id: {row["variant_id"]: row["rank"] for row in run.rankings}
        for run in ordered_runs
    }

    rows: list[dict[str, Any]] = []
    for run in ordered_runs:
        other_runs = [other for other in ordered_runs if other.spec.run_id != run.spec.run_id]
        high_rows = [row for row in run.rankings if row["rank"] <= high_rank_threshold]
        for variant_row in high_rows:
            variant_id = variant_row["variant_id"]
            if all(
                rank_lookup[other.spec.run_id][variant_id] > low_rank_threshold
                for other in other_runs
            ):
                for other in other_runs:
                    rows.append(
                        {
                            "variant_id": variant_id,
                            "gene": variant_row.get("gene", ""),
                            "chrom": variant_row.get("chrom", ""),
                            "pos": variant_row.get("pos", ""),
                            "specific_to_run_id": run.spec.run_id,
                            "specific_to_position_strategy_id": run.identity.strategy_id,
                            "rank_at_specific_strategy": variant_row["rank"],
                            "score_at_specific_strategy": variant_row["score"],
                            "other_run_id": other.spec.run_id,
                            "other_position_strategy_id": other.identity.strategy_id,
                            "rank_at_other_strategy": rank_lookup[other.spec.run_id][
                                variant_id
                            ],
                        }
                    )

    rows.sort(
        key=lambda row: (
            row["specific_to_run_id"],
            row["rank_at_specific_strategy"],
            row["variant_id"],
            row["other_run_id"],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--comparison-axis",
        choices=("level", "position"),
        default="level",
        help="Comparison axis to evaluate (default: level)",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--ranking-dir",
        type=str,
        help=(
            "Directory containing variant ranking CSVs named as "
            "L{0,1,2,3}_sieve_variant_rankings.csv"
        ),
    )
    group.add_argument(
        "--rankings",
        nargs="+",
        metavar="LEVEL:PATH",
        help=(
            "Explicit per-level ranking files, e.g. "
            "L0:results/L0/sieve_variant_rankings.csv "
            "L1:results/L1/sieve_variant_rankings.csv"
        ),
    )
    parser.add_argument(
        "--position-run",
        action="append",
        nargs=4,
        metavar=("RUN_ID", "CONFIG_YAML", "RANKING_CSV", "ANALYSIS_METADATA_YAML"),
        help=(
            "Position-mode run specification. Repeat once per strategy. "
            "Strategy identity is read from CONFIG_YAML."
        ),
    )
    parser.add_argument(
        "--top-k",
        default="50,100,200,500",
        help="Comma-separated top-k values for Jaccard computation (default: 50,100,200,500)",
    )
    parser.add_argument(
        "--high-rank-threshold",
        type=int,
        default=100,
        help="Threshold for high-ranking variants (default: 100)",
    )
    parser.add_argument(
        "--low-rank-threshold",
        type=int,
        default=500,
        help="Threshold for low-ranking variants at other levels (default: 500)",
    )
    parser.add_argument(
        "--out-comparison",
        default="ablation_ranking_comparison.yaml",
        help="Output YAML summary path",
    )
    parser.add_argument(
        "--out-jaccard",
        default="ablation_jaccard_matrix.tsv",
        help="Output Jaccard matrix TSV path",
    )
    parser.add_argument(
        "--out-level-specific",
        default="level_specific_variants.tsv",
        help="Output level-specific variants TSV path",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default=None,
        help=(
            "Column name to use for ranking variants. delta_rank is the "
            "recommended choice: it is scale-free, stable across annotation "
            "levels, and is the primary ranking metric. z_attribution is a "
            "per-chromosome z-score, which flattens genome-wide signal and is "
            "retained as a visualisation score for Manhattan plots and for "
            "continuity with earlier runs. "
            "Columns such as empirical_p_variant, fdr_variant, and corrected_rank "
            "are ranked ascending automatically; attribution-like scores are "
            "ranked descending."
        ),
    )
    return parser.parse_args()


def _parse_rankings_arg(rankings: List[str]) -> Dict[str, pathlib.Path]:
    """Parse ``LEVEL:PATH`` arguments into a dict."""
    level_files: Dict[str, pathlib.Path] = {}
    for item in rankings:
        if ":" not in item:
            raise ValueError(
                f"Invalid --rankings format '{item}'. Expected LEVEL:PATH, "
                f"e.g. L0:results/L0/sieve_variant_rankings.csv"
            )
        level, path_str = item.split(":", 1)
        level = level.upper()
        if level not in LEVEL_ORDER:
            print(
                f"WARNING: Level '{level}' not in expected order {LEVEL_ORDER}",
                file=sys.stderr,
            )
        path = pathlib.Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Ranking file not found: {path}")
        level_files[level] = path
    return level_files


def _parse_position_run_specs(
    position_runs: list[list[str]] | None,
) -> list[PositionRunSpec]:
    """Parse and validate repeated --position-run arguments."""
    if position_runs is None or len(position_runs) < 2:
        raise ValueError("position comparison requires at least two --position-run entries")

    specs: list[PositionRunSpec] = []
    seen = set()
    duplicates = set()
    for run_id, config_path, ranking_path, analysis_path in position_runs:
        if run_id in seen:
            duplicates.add(run_id)
        seen.add(run_id)
        spec = PositionRunSpec(
            run_id=run_id,
            config_path=pathlib.Path(config_path),
            ranking_path=pathlib.Path(ranking_path),
            analysis_metadata_path=pathlib.Path(analysis_path),
        )
        for path in (spec.config_path, spec.ranking_path, spec.analysis_metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"position run {run_id!r} path not found: {path}")
        specs.append(spec)

    if duplicates:
        duplicate_list = ", ".join(repr(run_id) for run_id in sorted(duplicates))
        raise ValueError(f"duplicate position run IDs are not allowed: {duplicate_list}")
    return sorted(specs, key=lambda spec: spec.run_id)


def _required_mapping(
    data: dict[str, Any],
    key: str,
    *,
    run_id: str,
) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"run {run_id!r} requires analysis_metadata[{key!r}] mapping")
    return value


def _require_run_equal(
    *,
    run_id: str,
    field: str,
    config_value: Any,
    analysis_value: Any,
) -> None:
    if config_value != analysis_value:
        raise ValueError(
            f"run {run_id!r} {field} mismatch: config value {config_value!r}, "
            f"analysis metadata value {analysis_value!r}"
        )


def _require_ranking_provenance_equal(
    *,
    run_id: str,
    field: str,
    ranking_value: Any,
    analysis_value: Any,
) -> None:
    if ranking_value != analysis_value:
        raise ValueError(
            f"run {run_id!r} {field} mismatch: ranking CSV value "
            f"{ranking_value!r}, analysis metadata value {analysis_value!r}"
        )


def _validate_position_analysis_metadata(
    *,
    spec: PositionRunSpec,
    config: dict[str, Any],
    analysis_metadata: dict[str, Any],
    identity: PositionStrategyIdentity,
) -> None:
    """Validate analysis metadata required for raw content-IG ranking comparison."""
    run_id = spec.run_id
    ig = _required_mapping(analysis_metadata, "integrated_gradients", run_id=run_id)
    if ig.get("executed") is not True:
        raise ValueError(f"run {run_id!r} integrated_gradients.executed must be True")
    if analysis_metadata.get("is_null_baseline") is not False:
        raise ValueError(f"run {run_id!r} analysis_metadata.is_null_baseline must be False")

    required_ig_values = {
        "resolved_ig_mode": "content",
        "attribution_feature_space": "content",
        "comparability_warning": None,
        "baseline_policy": "zero_content_observed_absolute_position",
        "position_encoding_metadata_source": "reconstructed_resolved_config",
    }
    for field, expected in required_ig_values.items():
        value = ig.get(field)
        if value != expected:
            raise ValueError(
                f"run {run_id!r} integrated_gradients.{field} must be "
                f"{expected!r}, got {value!r}"
            )

    dataset_identity = config.get("dataset_identity")
    if not isinstance(dataset_identity, dict):
        raise ValueError(f"run {run_id!r} config.dataset_identity must be a mapping")

    _require_run_equal(
        run_id=run_id,
        field="annotation_level",
        config_value=config.get("level"),
        analysis_value=analysis_metadata.get("annotation_level"),
    )
    _require_run_equal(
        run_id=run_id,
        field="genome_build",
        config_value=dataset_identity.get("genome_build"),
        analysis_value=analysis_metadata.get("genome_build"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.content_dim",
        config_value=config.get("content_dim"),
        analysis_value=ig.get("content_dim"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.attribution_width",
        config_value=config.get("content_dim"),
        analysis_value=ig.get("attribution_width"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.attribution_width",
        config_value=ig.get("content_dim"),
        analysis_value=ig.get("attribution_width"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.input_dim",
        config_value=config.get("input_dim"),
        analysis_value=ig.get("input_dim"),
    )

    absolute = identity.payload["absolute"]
    relative = identity.payload["relative"]
    chromosome = identity.payload["chromosome"]
    if not isinstance(absolute, dict) or not isinstance(relative, dict) or not isinstance(
        chromosome,
        dict,
    ):
        raise ValueError(f"run {run_id!r} position strategy payload is malformed")

    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.absolute_position_encoding",
        config_value=absolute.get("type"),
        analysis_value=ig.get("absolute_position_encoding"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.relative_position_encoding",
        config_value=relative.get("type"),
        analysis_value=ig.get("relative_position_encoding"),
    )
    _require_run_equal(
        run_id=run_id,
        field="integrated_gradients.chromosome_encoding",
        config_value=chromosome.get("encoding"),
        analysis_value=ig.get("chromosome_encoding"),
    )


def _validate_position_ranking_provenance(
    *,
    spec: PositionRunSpec,
    analysis_metadata: dict[str, Any],
) -> None:
    """Require ranking CSV provenance to match IG analysis metadata."""
    run_id = spec.run_id
    ig = _required_mapping(analysis_metadata, "integrated_gradients", run_id=run_id)
    provenance = load_position_ranking_provenance(
        spec.ranking_path,
        run_id=run_id,
    )
    for field, ranking_value in provenance.items():
        _require_ranking_provenance_equal(
            run_id=run_id,
            field=field,
            ranking_value=ranking_value,
            analysis_value=ig.get(field),
        )


def _load_position_run(spec: PositionRunSpec, *, score_column: str) -> PositionRankingRun:
    """Load and validate one position-mode run."""
    config = load_yaml(spec.config_path)
    analysis_metadata = load_yaml(spec.analysis_metadata_path)
    identity = position_strategy_identity(config)
    _validate_position_analysis_metadata(
        spec=spec,
        config=config,
        analysis_metadata=analysis_metadata,
        identity=identity,
    )
    _validate_position_ranking_provenance(
        spec=spec,
        analysis_metadata=analysis_metadata,
    )
    rankings, _, _ = load_position_rankings(
        spec.ranking_path,
        run_id=spec.run_id,
        score_column=score_column,
    )
    return PositionRankingRun(
        spec=spec,
        config=config,
        analysis_metadata=analysis_metadata,
        identity=identity,
        training_context=extract_comparison_context(config, run_id=spec.run_id),
        explanation_context=extract_explanation_context(
            analysis_metadata,
            run_id=spec.run_id,
        ),
        rankings=rankings,
    )


def _write_position_jaccard_tsv(
    path: pathlib.Path,
    matrices: dict[int, list[dict[str, Any]]],
    top_k_values: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "top_k",
        "run_id_a",
        "run_id_b",
        "position_strategy_id_a",
        "position_strategy_id_b",
        "jaccard",
        "overlap",
        "size_a",
        "size_b",
        "union",
        "score_column",
        "score_sort_order",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for top_k in top_k_values:
            for row in matrices.get(top_k, []):
                writer.writerow(row)


def _write_strategy_specific_tsv(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_id",
        "gene",
        "chrom",
        "pos",
        "specific_to_run_id",
        "specific_to_position_strategy_id",
        "rank_at_specific_strategy",
        "score_at_specific_strategy",
        "other_run_id",
        "other_position_strategy_id",
        "rank_at_other_strategy",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_position_comparison(args: argparse.Namespace) -> int:
    """Run strategy-aware position ranking comparison."""
    try:
        if args.ranking_dir or args.rankings:
            raise ValueError("position comparison rejects --ranking-dir and --rankings")
        top_k_values = _parse_top_k_values(args.top_k, require_positive=True)
        specs = _parse_position_run_specs(args.position_run)
        score_column = args.score_column
        if score_column is None:
            raise ValueError("position comparison requires explicit --score-column")
        runs = [_load_position_run(spec, score_column=score_column) for spec in specs]
        training_report = require_compatible_contexts(
            [run.training_context for run in runs]
        )
        explanation_report = require_compatible_explanation_contexts(
            [run.explanation_context for run in runs]
        )
        variant_universe = _validate_position_universe(runs)
        _, resolved_score_column, score_sort_order = load_position_rankings(
            specs[0].ranking_path,
            run_id=specs[0].run_id,
            score_column=score_column,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    jaccard_matrices = compute_position_jaccard_matrices(
        runs,
        top_k_values,
        score_column=resolved_score_column,
        score_sort_order=score_sort_order,
    )
    strategy_specific = find_strategy_specific_variants(
        runs,
        args.high_rank_threshold,
        args.low_rank_threshold,
    )

    _write_position_jaccard_tsv(
        pathlib.Path(args.out_jaccard),
        jaccard_matrices,
        top_k_values,
    )
    _write_strategy_specific_tsv(
        pathlib.Path(args.out_level_specific),
        strategy_specific,
    )

    strategy_specific_counts = []
    for run in sorted(runs, key=lambda item: item.spec.run_id):
        count = len(
            {
                row["variant_id"]
                for row in strategy_specific
                if row["specific_to_run_id"] == run.spec.run_id
            }
        )
        strategy_specific_counts.append(
            {
                "run_id": run.spec.run_id,
                "position_strategy_id": run.identity.strategy_id,
                "count": count,
            }
        )

    yaml_summary: dict[str, Any] = {
        "comparison_axis": "position",
        "score": {
            "column": resolved_score_column,
            "sort_order": score_sort_order,
        },
        "compatibility": {
            "training_context": training_report.to_dict(),
            "explanation_context": explanation_report.to_dict(),
        },
        "runs": [
            {
                "run_id": run.spec.run_id,
                "position_strategy_id": run.identity.strategy_id,
                "position_strategy_name": run.identity.name,
                "position_strategy_hash": run.identity.hash,
                "position_strategy": run.identity.payload,
                "config_path": str(run.spec.config_path),
                "ranking_path": str(run.spec.ranking_path),
                "analysis_metadata_path": str(run.spec.analysis_metadata_path),
                "n_variants": len(run.rankings),
            }
            for run in sorted(runs, key=lambda item: item.spec.run_id)
        ],
        "top_k_values": top_k_values,
        "variant_universe": {
            "n_variants": len(variant_universe),
            "key_rule": "explicit_variant_id_else_chromosome_position_gene_id",
        },
        "jaccard_matrices": {
            f"top_{top_k}": jaccard_matrices.get(top_k, [])
            for top_k in top_k_values
        },
        "strategy_specific_variant_counts": strategy_specific_counts,
        "thresholds": {
            "high_rank_threshold": args.high_rank_threshold,
            "low_rank_threshold": args.low_rank_threshold,
        },
    }
    comparison_path = pathlib.Path(args.out_comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(yaml_summary, comparison_path)

    print(f"Jaccard matrix written to {args.out_jaccard}", file=sys.stderr)
    print(
        f"Strategy-specific variants written to {args.out_level_specific} "
        f"({len(strategy_specific)} rows)",
        file=sys.stderr,
    )
    print(f"Comparison summary written to {args.out_comparison}", file=sys.stderr)
    return 0


def _run_level_comparison(args: argparse.Namespace) -> int:
    """Run the historical annotation-level ranking comparison."""
    if args.position_run:
        print("ERROR: level comparison rejects --position-run", file=sys.stderr)
        return 1

    score_column = args.score_column or "z_attribution"

    if score_column == "empirical_p_variant":
        print(
            "Warning: empirical_p_variant may be at the resolution floor for "
            "high-information annotation levels (median p pinned to 1/(N+1)), "
            "making top-K selection a draw from a tied set. "
            "Consider delta_rank for cross-level comparison.",
            file=sys.stderr,
        )

    top_k_values = _parse_top_k_values(args.top_k)

    # Discover ranking files
    if args.ranking_dir:
        ranking_dir = pathlib.Path(args.ranking_dir)
        level_files = find_ranking_files(ranking_dir)
    elif args.rankings:
        level_files = _parse_rankings_arg(args.rankings)
    else:
        print(
            "ERROR: No ranking files found. Check --ranking-dir or --rankings.",
            file=sys.stderr,
        )
        return 1

    if not level_files:
        print(
            "ERROR: No ranking files found. Check --ranking-dir or --rankings.",
            file=sys.stderr,
        )
        return 1

    levels_found = sorted(
        level_files.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )
    print(
        f"Found ranking files for levels: {', '.join(levels_found)}",
        file=sys.stderr,
    )

    # Load all rankings
    level_rankings: Dict[str, List[Dict[str, Any]]] = {}
    resolved_score_col = ""
    score_was_explicit = False
    for level, fpath in level_files.items():
        try:
            rankings, col_name, was_explicit = load_rankings(
                fpath, score_column=score_column
            )
        except ValueError as exc:
            print(
                f"ERROR: Failed to load rankings for {level} from {fpath}: {exc}",
                file=sys.stderr,
            )
            return 1
        level_rankings[level] = rankings
        if not resolved_score_col and col_name:
            resolved_score_col = col_name
            score_was_explicit = was_explicit
        print(
            f"  {level}: {len(level_rankings[level])} variants from {fpath.name}",
            file=sys.stderr,
        )

    # Log which score column is in use
    if resolved_score_col:
        source = "from --score-column" if score_was_explicit else "auto-detected"
        print(
            f"Score column: {resolved_score_col} ({source}, "
            f"{'ascending' if _score_column_is_ascending(resolved_score_col) else 'descending'})",
            file=sys.stderr,
        )

    if len(level_rankings) < 2:
        print(
            f"WARNING: Need at least 2 levels for comparison, found {len(level_rankings)}",
            file=sys.stderr,
        )

    # Compute Jaccard matrices
    jaccard_matrices = compute_jaccard_matrices(level_rankings, top_k_values)

    # Find level-specific variants
    level_specific = find_level_specific_variants(
        level_rankings,
        args.high_rank_threshold,
        args.low_rank_threshold,
    )

    # Write Jaccard TSV
    jaccard_path = pathlib.Path(args.out_jaccard)
    jaccard_path.parent.mkdir(parents=True, exist_ok=True)
    with jaccard_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["top_k", "level_a", "level_b", "jaccard", "overlap", "size_a", "size_b", "union"]
        )
        for top_k in top_k_values:
            for row in jaccard_matrices.get(top_k, []):
                writer.writerow(
                    [
                        row["top_k"],
                        row["level_a"],
                        row["level_b"],
                        row["jaccard"],
                        row["overlap"],
                        row["size_a"],
                        row["size_b"],
                        row["union"],
                    ]
                )

    # Write level-specific variants TSV
    level_specific_path = pathlib.Path(args.out_level_specific)
    level_specific_path.parent.mkdir(parents=True, exist_ok=True)
    rank_cols = [f"rank_at_{l}" for l in LEVEL_ORDER if l in level_rankings]
    fieldnames = [
        "variant_id",
        "gene",
        "chrom",
        "pos",
        "specific_to_level",
        "rank_at_specific_level",
    ] + rank_cols + ["score_at_specific_level"]

    with level_specific_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in level_specific:
            writer.writerow(row)

    # Write YAML summary
    yaml_summary: Dict[str, Any] = {
        "levels_analysed": levels_found,
        "top_k_values": top_k_values,
        "high_rank_threshold": args.high_rank_threshold,
        "low_rank_threshold": args.low_rank_threshold,
        "score_column": resolved_score_col,
        "score_sort_order": (
            "ascending" if resolved_score_col and _score_column_is_ascending(resolved_score_col)
            else "descending"
        ),
        "variants_per_level": {
            level: len(rankings) for level, rankings in level_rankings.items()
        },
        "jaccard_matrices": {},
        "level_specific_variant_counts": {},
    }

    for top_k in top_k_values:
        key = f"top_{top_k}"
        yaml_summary["jaccard_matrices"][key] = {}
        for row in jaccard_matrices.get(top_k, []):
            pair_key = f"{row['level_a']}_vs_{row['level_b']}"
            yaml_summary["jaccard_matrices"][key][pair_key] = {
                "jaccard": row["jaccard"],
                "overlap": row["overlap"],
                "union": row["union"],
            }

    for level in levels_found:
        count = sum(1 for v in level_specific if v["specific_to_level"] == level)
        yaml_summary["level_specific_variant_counts"][level] = count

    yaml_summary["total_level_specific_variants"] = len(level_specific)

    comparison_path = pathlib.Path(args.out_comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(yaml_summary, comparison_path)

    print(f"Jaccard matrix written to {args.out_jaccard}", file=sys.stderr)
    print(
        f"Level-specific variants written to {args.out_level_specific} "
        f"({len(level_specific)} variants)",
        file=sys.stderr,
    )
    print(f"Comparison summary written to {args.out_comparison}", file=sys.stderr)
    return 0


def main() -> int:
    """Entry point for ranking comparison."""
    args = parse_args()
    if args.comparison_axis == "position":
        return _run_position_comparison(args)
    return _run_level_comparison(args)


if __name__ == "__main__":
    sys.exit(main())
