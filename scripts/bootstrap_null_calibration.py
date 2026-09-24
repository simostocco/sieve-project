#!/usr/bin/env python3
"""
Bootstrap rank-based null calibration for SIEVE variant rankings.

This script complements ``compare_attributions.py`` by comparing the real
variant ranking against an ensemble of bootstrap-resampled null rankings built
from the null run's per-sample ``attributions.npz`` file. It preserves every
input column from the real rankings CSV and appends rank-based calibration
statistics, including ``delta_rank`` for downstream ablation comparison.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import hypergeom, ks_2samp, mannwhitneyu, rankdata
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compare_attributions import _bh_fdr
from src.data.genome import get_genome_build, is_sex_chrom, normalise_chrom


LOGGER = logging.getLogger("bootstrap_null_calibration")
DEFAULT_TOP_K = "50,100,200,500,1000"
DEFAULT_BOOTSTRAP = 1000
BOOTSTRAP_BATCH_SIZE = 100
MEMMAP_THRESHOLD_BYTES = 4 * 1024**3
DEFAULT_GENOME_BUILD = "GRCh37"
MAX_EXACT_HL_PAIRWISE = 1_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap rank-based null calibration for a real variant ranking CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--real-rankings",
        type=Path,
        required=True,
        help=(
            "Path to the real variant rankings CSV. Must include chromosome, "
            "position, gene_name, and mean_attribution."
        ),
    )
    parser.add_argument(
        "--null-attributions",
        type=Path,
        required=True,
        help="Path to the null attributions.npz file produced by explain.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the bootstrap-calibrated variant rankings.",
    )
    parser.add_argument(
        "--output-gene-stats",
        type=Path,
        default=None,
        help="Optional output CSV path for gene-level Wilcoxon statistics.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional output YAML path for the global bootstrap summary.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sample-index draws.",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default=DEFAULT_TOP_K,
        help="Comma-separated top-k thresholds for overlap and KS summaries.",
    )
    parser.add_argument(
        "--exclude-sex-chroms",
        action="store_true",
        default=False,
        help="Drop sex-chromosome variants from both real and null inputs.",
    )
    parser.add_argument(
        "--min-variants-per-gene",
        type=int,
        default=10,
        help="Minimum number of variants required to test a gene.",
    )
    parser.add_argument(
        "--gene-delta-rank-aggregation",
        type=str,
        choices=("max", "mean"),
        default="max",
        help=(
            "How to aggregate variant-level delta_rank values to the gene level. "
            "'max' (default) mirrors the gene_z_score=max(z_attribution) convention "
            "used by the chrX-corrected workflow. 'mean' is the alternative mirroring "
            "mean_z_score=mean(z_attribution). Higher gene_delta_rank values indicate "
            "stronger promotion of the gene's variants by the real model relative to "
            "the bootstrap null."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for bootstrap replicates.",
    )
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for the memmap-backed bootstrap rank matrix. "
            "The rank matrix is always backed by a memmap file; this flag "
            "controls where that file is created."
        ),
    )
    parser.add_argument(
        "--genome-build",
        type=str,
        default=None,
        help=(
            "Genome build for chromosome normalisation and sex-chromosome "
            "detection. Defaults to nearby analysis metadata when available, "
            f"otherwise {DEFAULT_GENOME_BUILD}."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Configure root logging for the script."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_top_k(value: str) -> list[int]:
    """Parse a comma-separated top-k string into positive unique integers."""
    top_k_values: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            raise ValueError("--top-k must contain only positive integers")
        if parsed not in top_k_values:
            top_k_values.append(parsed)
    if not top_k_values:
        raise ValueError("--top-k did not contain any valid integer threshold")
    return top_k_values


def _default_gene_stats_path(output_path: Path) -> Path:
    """Return the default gene-stats path derived from the main CSV output."""
    return output_path.with_name(f"{output_path.stem}_gene_stats.csv")


def _default_summary_path(output_path: Path) -> Path:
    """Return the default summary path derived from the main CSV output."""
    return output_path.with_name(f"{output_path.stem}_summary.yaml")


def _unwrap_object(value: Any) -> Any:
    """Return the underlying Python object from NumPy object containers."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _normalise_real_rankings(
    real_df: pd.DataFrame,
    chrom_build: Any,
) -> tuple[pd.DataFrame, bool]:
    """Validate and normalise the real rankings dataframe."""
    required = {"chromosome", "position", "gene_name", "mean_attribution"}
    missing = sorted(required - set(real_df.columns))
    if missing:
        raise ValueError(
            "Real rankings CSV is missing required columns: "
            + ", ".join(missing)
        )

    real_df = real_df.copy()
    real_df["chromosome"] = real_df["chromosome"].astype(str).map(
        lambda chrom: normalise_chrom(chrom, chrom_build)
    )
    real_df["position"] = pd.to_numeric(real_df["position"], errors="raise").astype(int)
    real_df["mean_attribution"] = pd.to_numeric(
        real_df["mean_attribution"], errors="raise"
    )

    use_gene_id = False
    if "gene_id" in real_df.columns:
        gene_ids = pd.to_numeric(real_df["gene_id"], errors="coerce")
        if gene_ids.notna().all():
            real_df["gene_id"] = gene_ids.astype(int)
            use_gene_id = True
        else:
            LOGGER.warning(
                "gene_id column contains missing or non-integer values; "
                "joining real and null variants by chromosome + position only."
            )

    return real_df, use_gene_id


def _build_variant_keys(
    chromosomes: Sequence[str],
    positions: Sequence[int],
    gene_ids: Sequence[int] | None,
) -> list[tuple[Any, ...]]:
    """Build canonical variant keys matching the real/null join granularity."""
    if gene_ids is None:
        return list(zip(chromosomes, positions))
    return list(zip(chromosomes, positions, gene_ids))


def _maybe_filter_real_df(
    real_df: pd.DataFrame,
    exclude_sex_chroms: bool,
    chrom_build: Any,
) -> tuple[pd.DataFrame, int]:
    """Optionally remove sex chromosomes from the real dataframe.

    The filtered frame is re-indexed to a contiguous ``0..n-1`` index because
    downstream code (``_compute_gene_statistics``) uses pandas index labels as
    positional indices into NumPy arrays (``rank_real``, ``real_to_null_index``)
    built from the filtered rows. Without the reset, sex-chromosome rows that
    precede or sit between autosomal rows leave gaps in the labels, which
    selects the wrong ranks or raises ``IndexError``. Row order is unchanged.
    """
    if not exclude_sex_chroms:
        return real_df, 0
    mask = ~real_df["chromosome"].map(
        lambda chrom: is_sex_chrom(str(chrom), chrom_build)
    )
    removed = int((~mask).sum())
    return real_df.loc[mask].copy().reset_index(drop=True), removed


def _load_nearby_analysis_metadata(reference_path: Path) -> dict[str, Any]:
    """Load nearby analysis metadata when available."""
    candidate_dirs = [reference_path.parent, reference_path.parent.parent]
    for candidate_dir in candidate_dirs:
        metadata_path = candidate_dir / "analysis_metadata.yaml"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _load_real_sample_count(real_rankings_path: Path) -> int | None:
    """Load n_samples from nearby analysis_metadata.yaml when available."""
    metadata = _load_nearby_analysis_metadata(real_rankings_path)
    n_samples = metadata.get("n_samples")
    if n_samples is None:
        return None
    try:
        return int(n_samples)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Ignoring non-integer n_samples=%r in nearby analysis metadata.",
            n_samples,
        )
        return None


def _resolve_genome_build_name(
    requested_build: str | None,
    real_rankings_path: Path,
) -> str:
    """Resolve the genome build from CLI or nearby analysis metadata."""
    if requested_build:
        return get_genome_build(requested_build).name

    metadata = _load_nearby_analysis_metadata(real_rankings_path)
    metadata_build = metadata.get("genome_build")
    if metadata_build:
        try:
            return get_genome_build(str(metadata_build)).name
        except ValueError:
            LOGGER.warning(
                "Ignoring unsupported genome_build=%r in nearby analysis metadata.",
                metadata_build,
            )
    return DEFAULT_GENOME_BUILD


def _load_null_attributions(
    null_path: Path,
    use_gene_id: bool,
    exclude_sex_chroms: bool,
    chrom_build: Any,
) -> dict[str, Any]:
    """Load null attributions.npz into a flat bootstrap substrate."""
    with np.load(null_path, allow_pickle=True) as data:
        if "variant_scores" not in data or "metadata" not in data:
            raise ValueError(
                "Null attributions.npz must contain 'variant_scores' and 'metadata'."
            )
        variant_scores = data["variant_scores"]
        metadata = data["metadata"]

    if len(variant_scores) != len(metadata):
        raise ValueError(
            "Null attributions.npz has inconsistent lengths for "
            "variant_scores and metadata."
        )

    n_samples = len(variant_scores)
    if n_samples == 0:
        raise ValueError("Null attributions.npz does not contain any samples.")
    if n_samples < 100:
        LOGGER.warning(
            "Null attributions contain only %d samples; bootstrap calibration "
            "will run, but the null ensemble may be underpowered.",
            n_samples,
        )

    key_to_index: dict[tuple[Any, ...], int] = {}
    sample_starts = np.empty(n_samples, dtype=np.int64)
    sample_lengths = np.empty(n_samples, dtype=np.int32)
    key_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    cursor = 0
    total_rows = 0
    total_rows_removed = 0

    for sample_idx, (scores_obj, meta_obj) in enumerate(zip(variant_scores, metadata)):
        sample_scores = np.asarray(_unwrap_object(scores_obj), dtype=float)
        sample_meta = _unwrap_object(meta_obj)
        positions = np.asarray(sample_meta["positions"])
        chromosomes = np.asarray(sample_meta["chromosomes"]).astype(str)
        gene_ids = np.asarray(sample_meta["gene_ids"])

        if not (
            len(sample_scores)
            == len(positions)
            == len(chromosomes)
            == len(gene_ids)
        ):
            raise ValueError(
                "Null attributions sample "
                f"{sample_idx} has mismatched lengths between scores and metadata."
            )

        total_rows += len(sample_scores)
        norm_chromosomes = np.array(
            [normalise_chrom(chrom, chrom_build) for chrom in chromosomes],
            dtype=object,
        )
        if exclude_sex_chroms:
            mask = np.array(
                [
                    not is_sex_chrom(str(chrom), chrom_build)
                    for chrom in norm_chromosomes
                ],
                dtype=bool,
            )
        else:
            mask = np.ones(len(sample_scores), dtype=bool)
        total_rows_removed += int((~mask).sum())

        sample_scores = sample_scores[mask]
        positions = positions[mask]
        norm_chromosomes = norm_chromosomes[mask]
        gene_ids = gene_ids[mask]

        key_indices = np.empty(len(sample_scores), dtype=np.int32)
        if use_gene_id:
            gene_ids = pd.to_numeric(gene_ids, errors="raise").astype(int)
            key_iter = zip(
                norm_chromosomes.tolist(),
                positions.astype(int).tolist(),
                gene_ids.tolist(),
            )
        else:
            key_iter = zip(
                norm_chromosomes.tolist(),
                positions.astype(int).tolist(),
            )

        for idx, key in enumerate(key_iter):
            key_index = key_to_index.get(key)
            if key_index is None:
                key_index = len(key_to_index)
                key_to_index[key] = key_index
            key_indices[idx] = key_index

        sample_starts[sample_idx] = cursor
        sample_lengths[sample_idx] = len(sample_scores)
        cursor += len(sample_scores)
        key_parts.append(key_indices)
        score_parts.append(sample_scores.astype(np.float64, copy=False))

    flat_key_indices = (
        np.concatenate(key_parts).astype(np.int32, copy=False)
        if key_parts
        else np.empty(0, dtype=np.int32)
    )
    flat_scores = (
        np.concatenate(score_parts).astype(np.float64, copy=False)
        if score_parts
        else np.empty(0, dtype=np.float64)
    )

    return {
        "n_samples": n_samples,
        "n_unique_variants": len(key_to_index),
        "key_to_index": key_to_index,
        "flat_key_indices": flat_key_indices,
        "flat_scores": flat_scores,
        "sample_starts": sample_starts,
        "sample_lengths": sample_lengths,
        "n_rows_total": total_rows,
        "n_rows_removed": total_rows_removed,
    }


def _compute_rank_vector(
    flat_key_indices: np.ndarray,
    flat_scores: np.ndarray,
    sample_starts: np.ndarray,
    sample_lengths: np.ndarray,
    selected_sample_indices: np.ndarray,
    n_unique_variants: int,
) -> tuple[np.ndarray, int]:
    """Aggregate a sample draw and return per-variant null ranks."""
    key_views: list[np.ndarray] = []
    score_views: list[np.ndarray] = []

    for sample_idx in selected_sample_indices:
        length = int(sample_lengths[sample_idx])
        if length == 0:
            continue
        start = int(sample_starts[sample_idx])
        end = start + length
        key_views.append(flat_key_indices[start:end])
        score_views.append(flat_scores[start:end])

    if not key_views:
        return np.ones(n_unique_variants, dtype=np.float32), 0

    selected_keys = np.concatenate(key_views)
    selected_scores = np.concatenate(score_views)
    sums = np.bincount(
        selected_keys,
        weights=selected_scores,
        minlength=n_unique_variants,
    )
    counts = np.bincount(selected_keys, minlength=n_unique_variants)
    present = counts > 0
    present_count = int(present.sum())
    ranks = np.full(n_unique_variants, float(n_unique_variants + 1), dtype=np.float32)
    if present_count == 0:
        return ranks, 0

    mean_scores = sums[present] / counts[present]
    ranks[present] = rankdata(-mean_scores, method="average").astype(np.float32)
    return ranks, present_count


def _write_rank_column(
    matrix_path: Path,
    shape: tuple[int, int],
    column_index: int,
    values: np.ndarray,
) -> None:
    """Write one bootstrap replicate into the rank memmap."""
    matrix = np.memmap(matrix_path, dtype=np.float32, mode="r+", shape=shape)
    matrix[:, column_index] = values.astype(np.float32, copy=False)
    matrix.flush()


def _bootstrap_worker(
    replicate_index: int,
    seed: int,
    flat_key_indices: np.ndarray,
    flat_scores: np.ndarray,
    sample_starts: np.ndarray,
    sample_lengths: np.ndarray,
    n_unique_variants: int,
    n_samples: int,
    real_to_null_index: np.ndarray,
    top_k_values: Sequence[int],
    real_top_canonical: Sequence[np.ndarray],
    real_top_extended: Sequence[np.ndarray],
    real_rank_extended: np.ndarray,
    n_real_variants: int,
    matrix_path: Path,
    matrix_shape: tuple[int, int],
) -> dict[str, Any]:
    """Run one bootstrap replicate and return small summary statistics."""
    rng = np.random.default_rng(seed)
    selected_sample_indices = rng.integers(0, n_samples, size=n_samples)
    null_ranks, _ = _compute_rank_vector(
        flat_key_indices=flat_key_indices,
        flat_scores=flat_scores,
        sample_starts=sample_starts,
        sample_lengths=sample_lengths,
        selected_sample_indices=selected_sample_indices,
        n_unique_variants=n_unique_variants,
    )

    worst_rank = float(n_unique_variants + 1)
    real_ranks = np.full(n_real_variants, worst_rank, dtype=np.float32)
    present_in_null = real_to_null_index >= 0
    real_ranks[present_in_null] = null_ranks[real_to_null_index[present_in_null]]
    _write_rank_column(matrix_path, matrix_shape, replicate_index, real_ranks)

    overlaps = np.zeros(len(top_k_values), dtype=np.int32)
    ks_statistics = np.zeros(len(top_k_values), dtype=np.float64)

    for idx, _ in enumerate(top_k_values):
        null_top_ids = np.flatnonzero(null_ranks <= float(top_k_values[idx])).astype(
            np.int64,
            copy=False,
        )
        overlaps[idx] = np.intersect1d(
            real_top_canonical[idx],
            null_top_ids,
            assume_unique=True,
        ).size

        pool_ids = np.union1d(real_top_extended[idx], null_top_ids)
        if pool_ids.size == 0:
            ks_statistics[idx] = 0.0
            continue

        real_pool = real_rank_extended[pool_ids]
        null_pool = np.full(pool_ids.size, worst_rank, dtype=np.float32)
        canonical_mask = pool_ids < n_unique_variants
        if canonical_mask.any():
            null_pool[canonical_mask] = null_ranks[pool_ids[canonical_mask]]
        ks_statistics[idx] = float(ks_2samp(real_pool, null_pool).statistic)

    return {
        "replicate_index": replicate_index,
        "overlaps": overlaps,
        "ks_statistics": ks_statistics,
    }


def _estimate_hodges_lehmann_shift(
    real_ranks: np.ndarray,
    null_ranks: np.ndarray,
) -> float:
    """Estimate the Hodges-Lehmann shift as median(null - real).

    For small inputs, use the exact pairwise-difference estimator. For larger
    inputs, fall back to a linear-memory approximation based on the difference
    in medians to avoid materializing an O(n_real * n_null) matrix.
    """
    if real_ranks.size == 0 or null_ranks.size == 0:
        return float("nan")

    pair_count = int(real_ranks.size) * int(null_ranks.size)
    if pair_count > MAX_EXACT_HL_PAIRWISE:
        return float(np.median(null_ranks) - np.median(real_ranks))

    differences = null_ranks[:, None] - real_ranks[None, :]
    return float(np.median(differences))


def _gene_key_column(real_df: pd.DataFrame) -> str:
    """Return the gene grouping column for gene-level output."""
    if "gene_name" in real_df.columns:
        return "gene_name"
    if "gene_id" in real_df.columns:
        return "gene_id"
    raise ValueError("Real rankings CSV must contain gene_name or gene_id.")


def _compute_gene_statistics(
    real_df: pd.DataFrame,
    rank_real: np.ndarray,
    rank_null_full: np.ndarray,
    real_to_null_index: np.ndarray,
    min_variants_per_gene: int,
    *,
    delta_rank_aggregation: str,
) -> pd.DataFrame:
    """Compute per-gene Mann-Whitney tests on real versus null ranks."""
    gene_column = _gene_key_column(real_df)
    gene_records: list[dict[str, Any]] = []
    gene_labels = real_df[gene_column].astype(str)

    for gene_name, group in real_df.groupby(gene_labels, sort=True):
        row_indices = group.index.to_numpy(dtype=int)
        gene_real_ranks = rank_real[row_indices]
        gene_null_indices = real_to_null_index[row_indices]
        gene_null_ranks = rank_null_full[gene_null_indices[gene_null_indices >= 0]]
        underpowered = (
            len(gene_real_ranks) < min_variants_per_gene
            or len(gene_null_ranks) < min_variants_per_gene
        )

        gene_delta_vals = real_df.loc[row_indices, "delta_rank"].to_numpy(dtype=float)
        if len(gene_delta_vals) == 0:
            agg_delta = float("nan")
        elif delta_rank_aggregation == "max":
            agg_delta = float(np.max(gene_delta_vals))
        elif delta_rank_aggregation == "mean":
            agg_delta = float(np.mean(gene_delta_vals))
        else:
            raise ValueError(
                "Unsupported delta_rank_aggregation: "
                f"{delta_rank_aggregation!r}. Expected one of: 'max', 'mean'."
            )

        record: dict[str, Any] = {
            "gene_name": gene_name,
            "n_variants_real": int(len(gene_real_ranks)),
            "n_variants_null": int(len(gene_null_ranks)),
            "median_rank_real": float(np.median(gene_real_ranks))
            if len(gene_real_ranks)
            else np.nan,
            "median_rank_null": float(np.median(gene_null_ranks))
            if len(gene_null_ranks)
            else np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p": np.nan,
            "effect_size_hodges_lehmann": np.nan,
            "underpowered": bool(underpowered),
            "gene_delta_rank": agg_delta,
            "gene_delta_rank_aggregation": delta_rank_aggregation,
        }

        if not underpowered:
            result = mannwhitneyu(
                gene_real_ranks,
                gene_null_ranks,
                alternative="less",
            )
            record["wilcoxon_statistic"] = float(result.statistic)
            record["wilcoxon_p"] = float(result.pvalue)
            record["effect_size_hodges_lehmann"] = _estimate_hodges_lehmann_shift(
                gene_real_ranks,
                gene_null_ranks,
            )
        else:
            LOGGER.warning(
                "Gene %s is underpowered for the Wilcoxon test "
                "(real=%d, null=%d, threshold=%d).",
                gene_name,
                len(gene_real_ranks),
                len(gene_null_ranks),
                min_variants_per_gene,
            )

        gene_records.append(record)

    gene_df = pd.DataFrame(gene_records)
    valid_mask = gene_df["wilcoxon_p"].notna()
    gene_df["fdr_gene_wilcoxon"] = np.nan
    if valid_mask.any():
        gene_df.loc[valid_mask, "fdr_gene_wilcoxon"] = _bh_fdr(
            gene_df.loc[valid_mask, "wilcoxon_p"].to_numpy(dtype=float)
        )
    return gene_df


def _to_float_ci(values: np.ndarray) -> list[float]:
    """Return a [low, high] list from bootstrap percentiles."""
    if values.size == 0:
        return [float("nan"), float("nan")]
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _prepare_storage(
    n_real_variants: int,
    n_bootstrap: int,
    memmap_dir: Path | None,
    n_jobs: int,
) -> tuple[Path, Path | None]:
    """Prepare rank-matrix storage and return its file path plus temp dir."""
    total_bytes = n_real_variants * n_bootstrap * np.dtype(np.float32).itemsize
    auto_temp_dir: Path | None = None
    target_dir = memmap_dir

    if target_dir is None and (total_bytes > MEMMAP_THRESHOLD_BYTES or n_jobs != 1):
        auto_temp_dir = Path(tempfile.mkdtemp(prefix="sieve_bootstrap_"))
        target_dir = auto_temp_dir
    elif target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    if target_dir is None:
        auto_temp_dir = Path(tempfile.mkdtemp(prefix="sieve_bootstrap_"))
        target_dir = auto_temp_dir

    matrix_path = target_dir / "null_rank_matrix.float32.dat"
    np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_real_variants, n_bootstrap),
    ).flush()
    return matrix_path, auto_temp_dir


def main(argv: list[str] | None = None) -> int:
    """Run bootstrap null calibration."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    top_k_values = _parse_top_k(args.top_k)
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be a positive integer.")
    if args.n_bootstrap < 100:
        LOGGER.warning(
            "n-bootstrap=%d is below 100; p-value resolution and confidence "
            "interval stability will be coarse.",
            args.n_bootstrap,
        )
    if args.min_variants_per_gene <= 0:
        raise ValueError("--min-variants-per-gene must be positive.")

    output_path = args.output
    gene_stats_path = args.output_gene_stats or _default_gene_stats_path(output_path)
    summary_path = args.output_summary or _default_summary_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_stats_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Real rankings: %s", args.real_rankings)
    LOGGER.info("Null attributions: %s", args.null_attributions)
    LOGGER.info("Output CSV: %s", output_path)
    LOGGER.info("Bootstrap replicates: %d", args.n_bootstrap)
    LOGGER.info("exclude_sex_chroms=%s", args.exclude_sex_chroms)

    overall_start = time.monotonic()
    genome_build_name = _resolve_genome_build_name(args.genome_build, args.real_rankings)
    chrom_build = get_genome_build(genome_build_name)
    LOGGER.info("genome_build=%s", chrom_build.name)

    load_start = time.monotonic()
    real_df = pd.read_csv(args.real_rankings)
    real_df, use_gene_id = _normalise_real_rankings(real_df, chrom_build)
    real_df, n_real_sex_removed = _maybe_filter_real_df(
        real_df,
        args.exclude_sex_chroms,
        chrom_build,
    )
    real_gene_ids = (
        real_df["gene_id"].astype(int).tolist() if use_gene_id else None
    )
    real_keys = _build_variant_keys(
        chromosomes=real_df["chromosome"].tolist(),
        positions=real_df["position"].tolist(),
        gene_ids=real_gene_ids,
    )
    if len(set(real_keys)) != len(real_keys):
        raise ValueError(
            "Real rankings contain duplicate variant keys at the join granularity. "
            "Use unique (chromosome, position, gene_id) rows before bootstrap calibration."
        )

    null_payload = _load_null_attributions(
        args.null_attributions,
        use_gene_id=use_gene_id,
        exclude_sex_chroms=args.exclude_sex_chroms,
        chrom_build=chrom_build,
    )
    LOGGER.info(
        "Loaded %d real variants and %d null samples (%d unique null variants).",
        len(real_df),
        null_payload["n_samples"],
        null_payload["n_unique_variants"],
    )
    if args.exclude_sex_chroms:
        LOGGER.info(
            "Removed %d real variants and %d null per-sample rows on sex chromosomes.",
            n_real_sex_removed,
            null_payload["n_rows_removed"],
        )

    real_sample_count = _load_real_sample_count(args.real_rankings)
    if real_sample_count is not None and real_sample_count != null_payload["n_samples"]:
        raise ValueError(
            "Sample-count mismatch between the real analysis metadata "
            f"({real_sample_count}) and the null attributions "
            f"({null_payload['n_samples']})."
        )
    if real_sample_count is None:
        LOGGER.info(
            "No nearby analysis_metadata.yaml with n_samples was found; "
            "real/null sample-count consistency could not be checked."
        )
    LOGGER.info("Load/validation completed in %.2fs", time.monotonic() - load_start)

    rank_start = time.monotonic()
    rank_real = rankdata(-real_df["mean_attribution"].to_numpy(dtype=float), method="average")
    real_to_null_index = np.array(
        [null_payload["key_to_index"].get(key, -1) for key in real_keys],
        dtype=np.int64,
    )
    n_real_missing_from_null = int((real_to_null_index < 0).sum())
    if n_real_missing_from_null > 0:
        LOGGER.warning(
            "%d real variants are absent from the null universe and will "
            "receive the worst available null rank in each replicate.",
            n_real_missing_from_null,
        )
    if len(real_df) > 0 and (n_real_missing_from_null / len(real_df)) > 0.10:
        LOGGER.warning(
            "%.1f%% of real variants are absent from the null universe "
            "(%d / %d).",
            100.0 * n_real_missing_from_null / len(real_df),
            n_real_missing_from_null,
            len(real_df),
        )

    synthetic_ids = np.arange(
        null_payload["n_unique_variants"],
        null_payload["n_unique_variants"] + n_real_missing_from_null,
        dtype=np.int64,
    )
    extended_ids = real_to_null_index.copy()
    missing_mask = real_to_null_index < 0
    extended_ids[missing_mask] = synthetic_ids
    real_rank_extended = np.full(
        null_payload["n_unique_variants"] + n_real_missing_from_null,
        float(len(real_df) + 1),
        dtype=np.float32,
    )
    real_rank_extended[extended_ids] = rank_real.astype(np.float32, copy=False)

    real_top_canonical = [
        np.unique(real_to_null_index[(rank_real <= top_k) & (real_to_null_index >= 0)])
        for top_k in top_k_values
    ]
    real_top_extended = [
        np.unique(extended_ids[rank_real <= top_k]) for top_k in top_k_values
    ]
    LOGGER.info("Real-ranking preparation completed in %.2fs", time.monotonic() - rank_start)

    matrix_path, auto_temp_dir = _prepare_storage(
        n_real_variants=len(real_df),
        n_bootstrap=args.n_bootstrap,
        memmap_dir=args.memmap_dir,
        n_jobs=args.n_jobs,
    )
    matrix_shape = (len(real_df), args.n_bootstrap)
    overlap_matrix = np.zeros((args.n_bootstrap, len(top_k_values)), dtype=np.int32)
    ks_matrix = np.zeros((args.n_bootstrap, len(top_k_values)), dtype=np.float64)

    bootstrap_start = time.monotonic()
    batch_starts = range(0, args.n_bootstrap, BOOTSTRAP_BATCH_SIZE)
    progress = tqdm(total=args.n_bootstrap, desc="bootstrap", unit="replicate")
    completed = 0

    try:
        for batch_start in batch_starts:
            batch_stop = min(batch_start + BOOTSTRAP_BATCH_SIZE, args.n_bootstrap)
            results = Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(_bootstrap_worker)(
                    replicate_index=replicate_index,
                    seed=args.seed * 1_000_003 + replicate_index,
                    flat_key_indices=null_payload["flat_key_indices"],
                    flat_scores=null_payload["flat_scores"],
                    sample_starts=null_payload["sample_starts"],
                    sample_lengths=null_payload["sample_lengths"],
                    n_unique_variants=null_payload["n_unique_variants"],
                    n_samples=null_payload["n_samples"],
                    real_to_null_index=real_to_null_index,
                    top_k_values=top_k_values,
                    real_top_canonical=real_top_canonical,
                    real_top_extended=real_top_extended,
                    real_rank_extended=real_rank_extended,
                    n_real_variants=len(real_df),
                    matrix_path=matrix_path,
                    matrix_shape=matrix_shape,
                )
                for replicate_index in range(batch_start, batch_stop)
            )
            for result in results:
                overlap_matrix[result["replicate_index"], :] = result["overlaps"]
                ks_matrix[result["replicate_index"], :] = result["ks_statistics"]

            completed += len(results)
            progress.update(len(results))
            if completed % 100 == 0 or completed == args.n_bootstrap:
                print(
                    f"Completed {completed}/{args.n_bootstrap} bootstrap replicates",
                    flush=True,
                )
    finally:
        progress.close()

    LOGGER.info(
        "Bootstrap replicates completed in %.2fs",
        time.monotonic() - bootstrap_start,
    )

    summary_start = time.monotonic()
    null_rank_matrix = np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="r",
        shape=matrix_shape,
    )
    p_rank_boot = (
        1.0
        + (null_rank_matrix <= rank_real[:, None]).sum(axis=1)
    ) / (args.n_bootstrap + 1.0)
    fdr_rank_boot = _bh_fdr(p_rank_boot)
    q25 = np.percentile(null_rank_matrix, 25, axis=1)
    q50 = np.percentile(null_rank_matrix, 50, axis=1)
    q75 = np.percentile(null_rank_matrix, 75, axis=1)
    resolution_floor = 1.0 / (args.n_bootstrap + 1.0)

    real_df_out = real_df.copy()
    real_df_out["rank_real"] = rank_real
    real_df_out["median_rank_null_boot"] = q50
    real_df_out["iqr_rank_null_boot"] = q75 - q25
    real_df_out["delta_rank"] = q50 - rank_real
    real_df_out["p_rank_boot"] = p_rank_boot
    real_df_out["fdr_rank_boot"] = fdr_rank_boot
    real_df_out["at_resolution_floor"] = np.isclose(p_rank_boot, resolution_floor)

    selected_sample_indices = np.arange(null_payload["n_samples"], dtype=np.int64)
    rank_null_full, _ = _compute_rank_vector(
        flat_key_indices=null_payload["flat_key_indices"],
        flat_scores=null_payload["flat_scores"],
        sample_starts=null_payload["sample_starts"],
        sample_lengths=null_payload["sample_lengths"],
        selected_sample_indices=selected_sample_indices,
        n_unique_variants=null_payload["n_unique_variants"],
    )
    gene_stats_df = _compute_gene_statistics(
        real_df=real_df_out,
        rank_real=rank_real,
        rank_null_full=rank_null_full,
        real_to_null_index=real_to_null_index,
        min_variants_per_gene=args.min_variants_per_gene,
        delta_rank_aggregation=args.gene_delta_rank_aggregation,
    )

    top_k_analysis: dict[str, Any] = {}
    for idx, top_k in enumerate(top_k_values):
        point_null_ids = np.flatnonzero(rank_null_full <= float(top_k)).astype(
            np.int64,
            copy=False,
        )
        overlap_point = int(
            np.intersect1d(
                real_top_canonical[idx],
                point_null_ids,
                assume_unique=True,
            ).size
        )
        pool_ids = np.union1d(real_top_extended[idx], point_null_ids)
        if pool_ids.size == 0:
            ks_point = 0.0
            ks_p = 1.0
        else:
            real_pool = real_rank_extended[pool_ids]
            null_pool = np.full(
                pool_ids.size,
                float(null_payload["n_unique_variants"] + 1),
                dtype=np.float32,
            )
            canonical_mask = pool_ids < null_payload["n_unique_variants"]
            if canonical_mask.any():
                null_pool[canonical_mask] = rank_null_full[pool_ids[canonical_mask]]
            ks_result = ks_2samp(real_pool, null_pool)
            ks_point = float(ks_result.statistic)
            ks_p = float(ks_result.pvalue)

        real_top_size = int(len(real_top_canonical[idx]))
        null_top_size = int(len(point_null_ids))
        if (
            null_payload["n_unique_variants"] == 0
            or real_top_size == 0
            or null_top_size == 0
        ):
            hypergeom_p = 1.0
        else:
            hypergeom_p = float(
                hypergeom.sf(
                    overlap_point - 1,
                    null_payload["n_unique_variants"],
                    real_top_size,
                    null_top_size,
                )
            )

        top_k_analysis[f"k_{top_k}"] = {
            "overlap_observed_pointest": overlap_point,
            "overlap_mean_bootstrap": float(overlap_matrix[:, idx].mean()),
            "overlap_ci95": _to_float_ci(overlap_matrix[:, idx].astype(float)),
            "hypergeometric_p_pointest": hypergeom_p,
            "ks_statistic_pointest": ks_point,
            "ks_p_analytical_pointest": ks_p,
            "ks_statistic_ci95": _to_float_ci(ks_matrix[:, idx]),
        }

    valid_gene_p = gene_stats_df["wilcoxon_p"].notna()
    summary = {
        "n_bootstrap": int(args.n_bootstrap),
        "genome_build": chrom_build.name,
        "n_real_variants": int(len(real_df_out)),
        "n_null_samples": int(null_payload["n_samples"]),
        "n_unique_null_variants": int(null_payload["n_unique_variants"]),
        "excluded_sex_chroms": bool(args.exclude_sex_chroms),
        "n_real_variants_removed_sex_chroms": int(n_real_sex_removed),
        "n_null_rows_removed_sex_chroms": int(null_payload["n_rows_removed"]),
        "n_real_variants_missing_from_null": int(n_real_missing_from_null),
        "top_k_analysis": top_k_analysis,
        "per_variant": {
            "n_tested": int(len(real_df_out)),
            "n_fdr_rank_boot_005": int((fdr_rank_boot < 0.05).sum()),
            "n_fdr_rank_boot_001": int((fdr_rank_boot < 0.01).sum()),
            "p_rank_boot_resolution_floor": float(resolution_floor),
            "n_at_resolution_floor": int(real_df_out["at_resolution_floor"].sum()),
        },
        "per_gene": {
            "n_tested": int(valid_gene_p.sum()),
            "n_underpowered": int(gene_stats_df["underpowered"].sum()),
            "n_fdr_gene_wilcoxon_005": int(
                (gene_stats_df["fdr_gene_wilcoxon"] < 0.05).fillna(False).sum()
            ),
            "n_fdr_gene_wilcoxon_001": int(
                (gene_stats_df["fdr_gene_wilcoxon"] < 0.01).fillna(False).sum()
            ),
            "gene_delta_rank_aggregation": args.gene_delta_rank_aggregation,
            "gene_delta_rank_median": float(
                gene_stats_df["gene_delta_rank"].median(skipna=True)
            ),
        },
    }
    LOGGER.info(
        "Summary statistics completed in %.2fs",
        time.monotonic() - summary_start,
    )

    write_start = time.monotonic()
    real_df_out.to_csv(output_path, index=False)
    gene_stats_df.to_csv(gene_stats_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Wrote %s", gene_stats_path)
    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Output writing completed in %.2fs", time.monotonic() - write_start)
    LOGGER.info("Total runtime: %.2fs", time.monotonic() - overall_start)

    if auto_temp_dir is not None:
        shutil.rmtree(auto_temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
