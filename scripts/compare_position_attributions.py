#!/usr/bin/env python3
"""Compare raw content-IG attributions across positional strategies.

This downstream comparator reads completed SIEVE explanation artifacts only.
It does not import torch, load checkpoints, construct models, infer strategy
identity from directory names, or traverse experiment directories.

Aggregate ``attributions.npz`` files are loaded with ``allow_pickle=True``
because the historical artifact stores ``variant_scores`` and ``metadata`` as
object arrays. Treat those aggregate files as trusted local SIEVE explanation
artifacts. Per-sample attribution matrices are loaded with
``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
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


SAMPLE_TSV_COLUMNS = [
    "run_id_a",
    "run_id_b",
    "position_strategy_id_a",
    "position_strategy_id_b",
    "sample_id",
    "label",
    "n_variants",
    "content_dim",
    "signed_cosine",
    "signed_cosine_reason",
    "signed_pearson",
    "signed_pearson_valid",
    "signed_pearson_reason",
    "score_cosine",
    "score_cosine_reason",
    "score_normalized_l2",
]
FEATURE_TSV_COLUMNS = [
    "run_id_a",
    "run_id_b",
    "position_strategy_id_a",
    "position_strategy_id_b",
    "feature_index",
    "feature_name",
    "n_values",
    "cosine",
    "cosine_reason",
    "pearson",
    "pearson_valid",
    "pearson_reason",
    "mean_abs_a",
    "mean_abs_b",
]
SUMMARY_TSV_COLUMNS = [
    "run_id_a",
    "run_id_b",
    "position_strategy_id_a",
    "position_strategy_id_b",
    "n_samples",
    "n_aligned_variant_instances",
    "content_dim",
    "mean_signed_cosine",
    "median_signed_cosine",
    "std_signed_cosine",
    "valid_signed_pearson_samples",
    "mean_signed_pearson",
    "median_signed_pearson",
    "mean_score_cosine",
    "median_score_cosine",
    "mean_score_normalized_l2",
    "median_score_normalized_l2",
]
AGGREGATE_SCALAR_KEYS = (
    "attribution_schema_version",
    "requested_ig_mode",
    "resolved_ig_mode",
    "attribution_feature_space",
    "attribution_width",
    "content_dim",
    "input_dim",
    "absolute_position_encoding",
    "relative_position_encoding",
    "chromosome_encoding",
    "position_encoding_metadata_source",
    "variant_score_aggregation",
    "baseline_policy",
    "n_steps",
    "max_variants",
    "sampling_policy",
    "sampling_seed",
    "comparability_warning",
)
PER_SAMPLE_SCALAR_KEYS = (
    "attribution_schema_version",
    "requested_ig_mode",
    "resolved_ig_mode",
    "attribution_feature_space",
    "attribution_width",
    "content_dim",
    "input_dim",
    "variant_score_aggregation",
    "baseline_policy",
)
MODEL_PROVENANCE_KEYS = {
    "schema_version",
    "checkpoint_selection_mode",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_path",
    "selected_fold",
    "selected_fold_auc",
    "cv_results_path",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_FILE_RE = re.compile(r"^sample_([0-9]+)\.npz$")
FEATURE_NAMES_BY_LEVEL = {
    "L0": ["dosage"],
    "L1": ["dosage"],
    "L2": [
        "dosage",
        "consequence_modifier",
        "consequence_low",
        "consequence_moderate",
        "consequence_high",
    ],
    "L3": [
        "dosage",
        "consequence_modifier",
        "consequence_low",
        "consequence_moderate",
        "consequence_high",
        "sift",
        "polyphen",
    ],
    "L4": [
        "dosage",
        "consequence_modifier",
        "consequence_low",
        "consequence_moderate",
        "consequence_high",
        "sift",
        "polyphen",
    ],
}
SCORE_RTOL = 1.0e-6
SCORE_ATOL = 1.0e-8


@dataclass(frozen=True)
class PositionAttributionRunSpec:
    """CLI-supplied artifacts for one completed explanation run."""

    run_id: str
    config_path: pathlib.Path
    analysis_metadata_path: pathlib.Path
    attributions_path: pathlib.Path
    per_sample_dir: pathlib.Path


@dataclass(frozen=True)
class SampleMetadata:
    """Aggregate metadata for one explained sample."""

    sample_idx: int
    sample_id: str
    label: object
    positions: np.ndarray
    gene_ids: np.ndarray
    chromosomes: np.ndarray
    variant_keys: tuple[str, ...]


@dataclass(frozen=True)
class LoadedPositionAttributionRun:
    """Validated metadata for one attribution-stability input run."""

    spec: PositionAttributionRunSpec
    config: dict[str, Any]
    analysis_metadata: dict[str, Any]
    identity: PositionStrategyIdentity
    training_context: ComparisonContext
    explanation_context: ExplanationContext
    model_provenance: dict[str, Any]
    samples_by_id: dict[str, SampleMetadata]
    aggregate_scores_by_id: dict[str, np.ndarray]
    feature_names: list[str]


@dataclass(frozen=True)
class MetricResult:
    """One scalar metric plus its validity reason."""

    value: float | None
    valid: bool
    reason: str


@dataclass
class FeatureStats:
    """Sufficient statistics for one content feature across samples."""

    n: int = 0
    sum_a: float = 0.0
    sum_b: float = 0.0
    sum_a2: float = 0.0
    sum_b2: float = 0.0
    sum_ab: float = 0.0
    sum_abs_a: float = 0.0
    sum_abs_b: float = 0.0

    def update(self, values_a: np.ndarray, values_b: np.ndarray) -> None:
        """Accumulate one feature column in float64."""
        a = np.asarray(values_a, dtype=np.float64)
        b = np.asarray(values_b, dtype=np.float64)
        self.n += int(a.size)
        self.sum_a += float(np.sum(a, dtype=np.float64))
        self.sum_b += float(np.sum(b, dtype=np.float64))
        self.sum_a2 += float(np.sum(a * a, dtype=np.float64))
        self.sum_b2 += float(np.sum(b * b, dtype=np.float64))
        self.sum_ab += float(np.sum(a * b, dtype=np.float64))
        self.sum_abs_a += float(np.sum(np.abs(a), dtype=np.float64))
        self.sum_abs_b += float(np.sum(np.abs(b), dtype=np.float64))


@dataclass(frozen=True)
class LoadedSample:
    """One full per-sample attribution matrix while it is being compared."""

    attributions: np.ndarray
    scores: np.ndarray
    order: np.ndarray
    metadata: SampleMetadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the attribution-stability CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw content-IG attribution stability across completed "
            "positional-strategy explanation artifacts."
        )
    )
    parser.add_argument(
        "--position-run",
        nargs=5,
        action="append",
        metavar=(
            "RUN_ID",
            "CONFIG_YAML",
            "ANALYSIS_METADATA_YAML",
            "ATTRIBUTIONS_NPZ",
            "ATTRIBUTIONS_PER_SAMPLE_DIR",
        ),
        help=(
            "Add one positional explanation run as five tokens: RUN_ID "
            "CONFIG_YAML ANALYSIS_METADATA_YAML ATTRIBUTIONS_NPZ "
            "ATTRIBUTIONS_PER_SAMPLE_DIR."
        ),
    )
    parser.add_argument(
        "--out-summary-tsv",
        type=pathlib.Path,
        default=pathlib.Path("position_attribution_summary.tsv"),
    )
    parser.add_argument(
        "--out-sample-tsv",
        type=pathlib.Path,
        default=pathlib.Path("position_attribution_per_sample.tsv"),
    )
    parser.add_argument(
        "--out-feature-tsv",
        type=pathlib.Path,
        default=pathlib.Path("position_attribution_features.tsv"),
    )
    parser.add_argument(
        "--out-comparison-yaml",
        type=pathlib.Path,
        default=pathlib.Path("position_attribution_comparison.yaml"),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_arg_parser().parse_args(argv)


def parse_position_run_specs(
    position_runs: Sequence[Sequence[str]] | None,
) -> list[PositionAttributionRunSpec]:
    """Validate and normalize CLI run specifications."""
    if position_runs is None or len(position_runs) < 2:
        raise ValueError("at least two --position-run entries are required")

    specs = []
    run_ids = set()
    for entry in position_runs:
        run_id, config, analysis, aggregate, per_sample = entry
        if not run_id:
            raise ValueError("RUN_ID must be non-empty")
        if run_id in run_ids:
            raise ValueError(f"duplicate run ID {run_id!r} is not allowed")
        run_ids.add(run_id)
        config_path = pathlib.Path(config)
        analysis_path = pathlib.Path(analysis)
        aggregate_path = pathlib.Path(aggregate)
        per_sample_dir = pathlib.Path(per_sample)
        _require_file(config_path, run_id, "config")
        _require_file(analysis_path, run_id, "analysis metadata")
        _require_file(aggregate_path, run_id, "aggregate attributions")
        if not per_sample_dir.exists() or not per_sample_dir.is_dir():
            raise ValueError(
                f"run {run_id!r} per-sample directory does not exist: {per_sample_dir}"
            )
        specs.append(
            PositionAttributionRunSpec(
                run_id=run_id,
                config_path=config_path,
                analysis_metadata_path=analysis_path,
                attributions_path=aggregate_path,
                per_sample_dir=per_sample_dir,
            )
        )
    return specs


def compare_position_attributions(
    specs: Sequence[PositionAttributionRunSpec],
    *,
    out_summary_tsv: pathlib.Path,
    out_sample_tsv: pathlib.Path,
    out_feature_tsv: pathlib.Path,
    out_comparison_yaml: pathlib.Path,
) -> dict[str, Any]:
    """Validate artifacts, compute stability metrics, and write outputs."""
    runs = [load_position_attribution_run(spec) for spec in specs]
    training_report = require_compatible_contexts([run.training_context for run in runs])
    explanation_report = require_compatible_explanation_contexts(
        [run.explanation_context for run in runs]
    )
    checkpoint_policy = _validate_checkpoint_policy(runs)
    _validate_sample_universe(runs)

    sample_rows, summary_rows, feature_rows = _compute_pairwise_outputs(runs)

    _write_tsv(out_sample_tsv, SAMPLE_TSV_COLUMNS, sample_rows)
    _write_tsv(out_summary_tsv, SUMMARY_TSV_COLUMNS, summary_rows)
    _write_tsv(out_feature_tsv, FEATURE_TSV_COLUMNS, feature_rows)
    comparison = _build_yaml_output(
        runs,
        training_report,
        explanation_report,
        checkpoint_policy,
        summary_rows,
        out_summary_tsv,
        out_sample_tsv,
        out_feature_tsv,
        out_comparison_yaml,
    )
    _dump_yaml(comparison, out_comparison_yaml)
    return comparison


def load_position_attribution_run(
    spec: PositionAttributionRunSpec,
) -> LoadedPositionAttributionRun:
    """Load and validate one completed content-IG explanation run."""
    config = _load_yaml(spec.config_path)
    analysis_metadata = _load_yaml(spec.analysis_metadata_path)
    identity = position_strategy_identity(config)
    training_context = extract_comparison_context(config, run_id=spec.run_id)
    explanation_context = extract_explanation_context(
        analysis_metadata,
        run_id=spec.run_id,
    )
    feature_names = _feature_names(config, spec.run_id)
    _validate_content_ig_metadata(spec.run_id, config, analysis_metadata, identity)
    provenance = _validate_model_provenance(spec, analysis_metadata)
    samples, aggregate_scores = _load_aggregate_attributions(
        spec,
        config,
        analysis_metadata,
        identity,
    )
    _validate_per_sample_file_set(spec, samples)
    return LoadedPositionAttributionRun(
        spec=spec,
        config=config,
        analysis_metadata=analysis_metadata,
        identity=identity,
        training_context=training_context,
        explanation_context=explanation_context,
        model_provenance=provenance,
        samples_by_id=samples,
        aggregate_scores_by_id=aggregate_scores,
        feature_names=feature_names,
    )


def signed_cosine(values_a: np.ndarray, values_b: np.ndarray) -> MetricResult:
    """Return signed cosine with the documented zero-vector policy."""
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return MetricResult(1.0, True, "both_zero")
    if norm_a == 0.0 or norm_b == 0.0:
        return MetricResult(0.0, True, "one_zero")
    value = float(np.dot(a, b) / (norm_a * norm_b))
    return MetricResult(_clamp_correlation(value), True, "ok")


def pearson(values_a: np.ndarray, values_b: np.ndarray) -> MetricResult:
    """Return Pearson correlation and the constant-vector diagnostic reason."""
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if a.size < 2:
        return MetricResult(None, False, "insufficient_values")
    constant_a = bool(np.all(a == a[0]))
    constant_b = bool(np.all(b == b[0]))
    if constant_a and constant_b:
        return MetricResult(None, False, "both_constant")
    if constant_a or constant_b:
        return MetricResult(None, False, "one_constant")
    value = float(np.corrcoef(a, b)[0, 1])
    return MetricResult(_clamp_correlation(value), True, "ok")


def normalized_l2(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return ||a-b||/(||a||+||b||), with zero distance for two zero vectors."""
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) + np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    value = float(np.linalg.norm(a - b) / denominator)
    return max(0.0, min(value, 1.0))


def feature_metrics_from_stats(stats: FeatureStats) -> dict[str, object]:
    """Derive feature metrics from sufficient statistics only."""
    cosine = _cosine_from_stats(stats)
    pearson_result = _pearson_from_stats(stats)
    return {
        "n_values": stats.n,
        "cosine": cosine.value,
        "cosine_reason": cosine.reason,
        "pearson": pearson_result.value,
        "pearson_valid": pearson_result.valid,
        "pearson_reason": pearson_result.reason,
        "mean_abs_a": (stats.sum_abs_a / stats.n if stats.n else None),
        "mean_abs_b": (stats.sum_abs_b / stats.n if stats.n else None),
    }


def _compute_pairwise_outputs(
    runs: Sequence[LoadedPositionAttributionRun],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sample_ids = sorted(runs[0].samples_by_id)
    run_pairs = list(itertools.combinations(sorted(runs, key=lambda run: run.spec.run_id), 2))
    sample_rows: list[dict[str, object]] = []
    per_pair_metrics: dict[tuple[str, str], list[dict[str, object]]] = {
        (run_a.spec.run_id, run_b.spec.run_id): [] for run_a, run_b in run_pairs
    }
    feature_stats: dict[tuple[str, str], list[FeatureStats]] = {
        (run_a.spec.run_id, run_b.spec.run_id): [
            FeatureStats() for _ in range(len(run_a.feature_names))
        ]
        for run_a, run_b in run_pairs
    }

    for sample_id in sample_ids:
        for run_a, run_b in run_pairs:
            with _open_ordered_sample(run_a, sample_id) as sample_a:
                with _open_ordered_sample(run_b, sample_id) as sample_b:
                    _validate_variant_universe(run_a, run_b, sample_id, sample_a, sample_b)
                    row = _sample_metric_row(run_a, run_b, sample_id, sample_a, sample_b)
                    sample_rows.append(row)
                    per_pair_metrics[(run_a.spec.run_id, run_b.spec.run_id)].append(row)
                    for feature_index, stats in enumerate(
                        feature_stats[(run_a.spec.run_id, run_b.spec.run_id)]
                    ):
                        stats.update(
                            _ordered_feature_values(sample_a, feature_index),
                            _ordered_feature_values(sample_b, feature_index),
                        )

    summary_rows = [
        _summary_metric_row(run_a, run_b, per_pair_metrics[(run_a.spec.run_id, run_b.spec.run_id)])
        for run_a, run_b in run_pairs
    ]
    feature_rows = []
    for run_a, run_b in run_pairs:
        pair_key = (run_a.spec.run_id, run_b.spec.run_id)
        for feature_index, stats in enumerate(feature_stats[pair_key]):
            metrics = feature_metrics_from_stats(stats)
            feature_rows.append(
                {
                    "run_id_a": run_a.spec.run_id,
                    "run_id_b": run_b.spec.run_id,
                    "position_strategy_id_a": run_a.identity.strategy_id,
                    "position_strategy_id_b": run_b.identity.strategy_id,
                    "feature_index": feature_index,
                    "feature_name": run_a.feature_names[feature_index],
                    **metrics,
                }
            )
    sample_rows.sort(
        key=lambda row: (str(row["sample_id"]), str(row["run_id_a"]), str(row["run_id_b"]))
    )
    summary_rows.sort(key=lambda row: (str(row["run_id_a"]), str(row["run_id_b"])))
    feature_rows.sort(
        key=lambda row: (
            str(row["run_id_a"]),
            str(row["run_id_b"]),
            int(row["feature_index"]),
        )
    )
    return sample_rows, summary_rows, feature_rows


def _sample_metric_row(
    run_a: LoadedPositionAttributionRun,
    run_b: LoadedPositionAttributionRun,
    sample_id: str,
    sample_a: LoadedSample,
    sample_b: LoadedSample,
) -> dict[str, object]:
    cosine = _signed_cosine_for_aligned_samples(sample_a, sample_b)
    pearson_result = _pearson_for_aligned_samples(sample_a, sample_b)
    score_cosine = signed_cosine(_ordered_scores(sample_a), _ordered_scores(sample_b))
    return {
        "run_id_a": run_a.spec.run_id,
        "run_id_b": run_b.spec.run_id,
        "position_strategy_id_a": run_a.identity.strategy_id,
        "position_strategy_id_b": run_b.identity.strategy_id,
        "sample_id": sample_id,
        "label": sample_a.metadata.label,
        "n_variants": sample_a.attributions.shape[0],
        "content_dim": sample_a.attributions.shape[1],
        "signed_cosine": cosine.value,
        "signed_cosine_reason": cosine.reason,
        "signed_pearson": pearson_result.value,
        "signed_pearson_valid": pearson_result.valid,
        "signed_pearson_reason": pearson_result.reason,
        "score_cosine": score_cosine.value,
        "score_cosine_reason": score_cosine.reason,
        "score_normalized_l2": normalized_l2(_ordered_scores(sample_a), _ordered_scores(sample_b)),
    }


def _summary_metric_row(
    run_a: LoadedPositionAttributionRun,
    run_b: LoadedPositionAttributionRun,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    signed_cosines = [float(row["signed_cosine"]) for row in rows]
    valid_pearsons = [
        float(row["signed_pearson"]) for row in rows if row["signed_pearson_valid"] is True
    ]
    score_cosines = [float(row["score_cosine"]) for row in rows]
    score_l2 = [float(row["score_normalized_l2"]) for row in rows]
    return {
        "run_id_a": run_a.spec.run_id,
        "run_id_b": run_b.spec.run_id,
        "position_strategy_id_a": run_a.identity.strategy_id,
        "position_strategy_id_b": run_b.identity.strategy_id,
        "n_samples": len(rows),
        "n_aligned_variant_instances": sum(int(row["n_variants"]) for row in rows),
        "content_dim": len(run_a.feature_names),
        "mean_signed_cosine": _mean(signed_cosines),
        "median_signed_cosine": _median(signed_cosines),
        "std_signed_cosine": _std_population(signed_cosines),
        "valid_signed_pearson_samples": len(valid_pearsons),
        "mean_signed_pearson": _mean(valid_pearsons) if valid_pearsons else None,
        "median_signed_pearson": _median(valid_pearsons) if valid_pearsons else None,
        "mean_score_cosine": _mean(score_cosines),
        "median_score_cosine": _median(score_cosines),
        "mean_score_normalized_l2": _mean(score_l2),
        "median_score_normalized_l2": _median(score_l2),
    }


@contextmanager
def _open_ordered_sample(
    run: LoadedPositionAttributionRun,
    sample_id: str,
) -> Any:
    sample = _load_per_sample_attributions(run, sample_id)
    try:
        yield sample
    finally:
        del sample


def _load_per_sample_attributions(
    run: LoadedPositionAttributionRun,
    sample_id: str,
) -> LoadedSample:
    metadata = run.samples_by_id[sample_id]
    path = run.spec.per_sample_dir / f"sample_{metadata.sample_idx}.npz"
    with np.load(path, allow_pickle=False) as data:
        for key in ("attributions", "variant_scores", *PER_SAMPLE_SCALAR_KEYS):
            if key not in data:
                raise ValueError(f"run {run.spec.run_id!r} sample {sample_id!r} missing {key!r}")
        _validate_per_sample_scalars(run, sample_id, data)
        attributions = np.asarray(data["attributions"])
        scores = np.asarray(data["variant_scores"])

    content_dim = _config_content_dim(run.config, run.spec.run_id)
    row_count = len(metadata.variant_keys)
    _validate_numeric_matrix(
        attributions,
        run.spec.run_id,
        sample_id,
        row_count,
        content_dim,
    )
    _validate_score_vector(scores, run.spec.run_id, sample_id, row_count)
    aggregate_scores = run.aggregate_scores_by_id[sample_id]
    if not np.allclose(scores, aggregate_scores, rtol=SCORE_RTOL, atol=SCORE_ATOL):
        raise ValueError(
            f"run {run.spec.run_id!r} sample {sample_id!r} "
            "aggregate/per-sample score mismatch: expected aggregate scores "
            "to match per-sample scores"
        )
    recomputed = _row_l2_norms(attributions)
    if not np.allclose(recomputed, scores, rtol=SCORE_RTOL, atol=SCORE_ATOL):
        raise ValueError(
            f"run {run.spec.run_id!r} sample {sample_id!r} variant score L2 "
            "self-consistency check failed"
        )
    order = np.argsort(np.asarray(metadata.variant_keys, dtype=str), kind="stable")
    return LoadedSample(
        attributions=attributions,
        scores=scores,
        order=order,
        metadata=metadata,
    )


def _row_l2_norms(attributions: np.ndarray) -> np.ndarray:
    """Compute row L2 norms without materializing a full float64 matrix copy."""
    norms = np.empty(attributions.shape[0], dtype=np.float64)
    for row_index in range(attributions.shape[0]):
        row = np.asarray(attributions[row_index], dtype=np.float64)
        norms[row_index] = float(np.linalg.norm(row))
    return norms


def _ordered_feature_values(sample: LoadedSample, feature_index: int) -> np.ndarray:
    """Return one ordered feature column as the largest metric temporary."""
    return np.asarray(sample.attributions[sample.order, feature_index], dtype=np.float64)


def _ordered_scores(sample: LoadedSample) -> np.ndarray:
    """Return ordered score vector for pairwise score metrics."""
    return np.asarray(sample.scores[sample.order], dtype=np.float64)


def _signed_cosine_for_aligned_samples(
    sample_a: LoadedSample,
    sample_b: LoadedSample,
) -> MetricResult:
    dot = 0.0
    sum_a2 = 0.0
    sum_b2 = 0.0
    for feature_index in range(sample_a.attributions.shape[1]):
        values_a = _ordered_feature_values(sample_a, feature_index)
        values_b = _ordered_feature_values(sample_b, feature_index)
        dot += float(np.dot(values_a, values_b))
        sum_a2 += float(np.dot(values_a, values_a))
        sum_b2 += float(np.dot(values_b, values_b))
    norm_a = math.sqrt(sum_a2)
    norm_b = math.sqrt(sum_b2)
    if norm_a == 0.0 and norm_b == 0.0:
        return MetricResult(1.0, True, "both_zero")
    if norm_a == 0.0 or norm_b == 0.0:
        return MetricResult(0.0, True, "one_zero")
    return MetricResult(_clamp_correlation(dot / (norm_a * norm_b)), True, "ok")


def _pearson_for_aligned_samples(
    sample_a: LoadedSample,
    sample_b: LoadedSample,
) -> MetricResult:
    stats = FeatureStats()
    for feature_index in range(sample_a.attributions.shape[1]):
        stats.update(
            _ordered_feature_values(sample_a, feature_index),
            _ordered_feature_values(sample_b, feature_index),
        )
    return _pearson_from_stats(stats)


def _validate_per_sample_scalars(
    run: LoadedPositionAttributionRun,
    sample_id: str,
    data: Mapping[str, Any],
) -> None:
    ig_metadata = _ig_metadata(run.analysis_metadata, run.spec.run_id)
    for key in PER_SAMPLE_SCALAR_KEYS:
        actual = _npz_scalar(data, key, run.spec.run_id, sample_id=sample_id)
        expected = ig_metadata[key]
        if actual != expected:
            raise ValueError(
                f"run {run.spec.run_id!r} sample {sample_id!r} field {key!r} "
                f"expected {expected!r}, artifact value {actual!r}"
            )


def _validate_variant_universe(
    run_a: LoadedPositionAttributionRun,
    run_b: LoadedPositionAttributionRun,
    sample_id: str,
    sample_a: LoadedSample,
    sample_b: LoadedSample,
) -> None:
    keys_a = set(sample_a.metadata.variant_keys)
    keys_b = set(sample_b.metadata.variant_keys)
    if keys_a != keys_b:
        missing_keys = sorted(keys_a - keys_b)
        extra_keys = sorted(keys_b - keys_a)
        raise ValueError(
            f"variant universe mismatch for sample {sample_id!r}: reference run "
            f"{run_a.spec.run_id!r}, other run {run_b.spec.run_id!r}, reference "
            f"variant count {len(keys_a)}, other variant count {len(keys_b)}, "
            f"missing count {len(missing_keys)}, extra count {len(extra_keys)}, "
            f"missing examples {missing_keys[:5]}, extra examples {extra_keys[:5]}"
        )
    if sample_a.metadata.label != sample_b.metadata.label:
        raise ValueError(
            f"label mismatch for sample {sample_id!r}: run {run_a.spec.run_id!r} "
            f"has {sample_a.metadata.label!r}, run {run_b.spec.run_id!r} has "
            f"{sample_b.metadata.label!r}"
        )


def _load_aggregate_attributions(
    spec: PositionAttributionRunSpec,
    config: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    identity: PositionStrategyIdentity,
) -> tuple[dict[str, SampleMetadata], dict[str, np.ndarray]]:
    with np.load(spec.attributions_path, allow_pickle=True) as data:
        for key in ("variant_scores", "metadata", *AGGREGATE_SCALAR_KEYS):
            if key not in data:
                raise ValueError(f"run {spec.run_id!r} aggregate NPZ missing {key!r}")
        _validate_aggregate_scalars(spec.run_id, config, analysis_metadata, identity, data)
        variant_scores = np.asarray(data["variant_scores"], dtype=object)
        metadata_entries = np.asarray(data["metadata"], dtype=object)

    n_samples = _required_positive_int(
        analysis_metadata.get("n_samples"),
        f"run {spec.run_id!r} analysis_metadata.n_samples",
    )
    if variant_scores.shape[0] != n_samples or metadata_entries.shape[0] != n_samples:
        raise ValueError(
            f"run {spec.run_id!r} aggregate sample count mismatch: expected "
            f"{n_samples}, variant_scores has {variant_scores.shape[0]}, metadata "
            f"has {metadata_entries.shape[0]}"
        )
    samples_by_id: dict[str, SampleMetadata] = {}
    aggregate_scores_by_id: dict[str, np.ndarray] = {}
    seen_indices: set[int] = set()
    for index, raw_metadata in enumerate(metadata_entries):
        metadata = _parse_sample_metadata(spec.run_id, _as_mapping(raw_metadata), index)
        if metadata.sample_idx in seen_indices:
            raise ValueError(f"run {spec.run_id!r} duplicate sample_idx {metadata.sample_idx}")
        if metadata.sample_id in samples_by_id:
            raise ValueError(f"run {spec.run_id!r} duplicate sample_id {metadata.sample_id!r}")
        seen_indices.add(metadata.sample_idx)
        scores = _aggregate_score_vector(
            variant_scores[index],
            spec.run_id,
            metadata.sample_id,
            len(metadata.variant_keys),
        )
        samples_by_id[metadata.sample_id] = metadata
        aggregate_scores_by_id[metadata.sample_id] = scores
    return samples_by_id, aggregate_scores_by_id


def _aggregate_score_vector(
    raw_scores: object,
    run_id: str,
    sample_id: str,
    expected_rows: int,
) -> np.ndarray:
    """Normalize one trusted aggregate score vector to finite float64."""
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} aggregate variant_scores "
            "must contain numeric values"
        ) from error
    _validate_score_vector(scores, run_id, sample_id, expected_rows)
    return scores


def _validate_aggregate_scalars(
    run_id: str,
    config: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    identity: PositionStrategyIdentity,
    data: Mapping[str, Any],
) -> None:
    ig_metadata = _ig_metadata(analysis_metadata, run_id)
    for key in AGGREGATE_SCALAR_KEYS:
        actual = _npz_scalar(data, key, run_id)
        expected = _expected_aggregate_scalar(key, ig_metadata, identity)
        if actual != expected:
            raise ValueError(
                f"run {run_id!r} aggregate field {key!r} expected {expected!r}, "
                f"artifact value {actual!r}"
            )
    if ig_metadata["content_dim"] != _config_content_dim(config, run_id):
        raise ValueError(f"run {run_id!r} content_dim mismatch")
    if ig_metadata["input_dim"] != _config_input_dim(config, run_id):
        raise ValueError(f"run {run_id!r} input_dim mismatch")


def _expected_aggregate_scalar(
    key: str,
    ig_metadata: Mapping[str, Any],
    identity: PositionStrategyIdentity,
) -> object:
    if key == "absolute_position_encoding":
        return identity.payload["absolute"]["type"]
    if key == "relative_position_encoding":
        return identity.payload["relative"]["type"]
    if key == "chromosome_encoding":
        return identity.payload["chromosome"]["encoding"]
    return ig_metadata[key]


def _parse_sample_metadata(
    run_id: str,
    raw_metadata: Mapping[str, object],
    aggregate_index: int,
) -> SampleMetadata:
    sample_idx = _required_non_negative_int(
        raw_metadata.get("sample_idx"),
        f"run {run_id!r} metadata[{aggregate_index}].sample_idx",
    )
    sample_id = raw_metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(
            f"run {run_id!r} metadata[{aggregate_index}].sample_id must be a non-empty string"
        )
    label = raw_metadata.get("label")
    positions = _one_dimensional_object_array(
        raw_metadata.get("positions"),
        run_id,
        sample_id,
        "positions",
    )
    gene_ids = _one_dimensional_object_array(
        raw_metadata.get("gene_ids"),
        run_id,
        sample_id,
        "gene_ids",
    )
    chromosomes = _one_dimensional_object_array(
        raw_metadata.get("chromosomes"),
        run_id,
        sample_id,
        "chromosomes",
    )
    if not (len(positions) == len(gene_ids) == len(chromosomes)):
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} metadata positions, gene_ids, "
            "and chromosomes must have equal length"
        )
    variant_keys = tuple(
        _variant_key(run_id, sample_id, chrom, pos, gene)
        for chrom, pos, gene in zip(chromosomes, positions, gene_ids, strict=True)
    )
    if len(set(variant_keys)) != len(variant_keys):
        raise ValueError(f"run {run_id!r} sample {sample_id!r} has duplicate variant key")
    return SampleMetadata(
        sample_idx=sample_idx,
        sample_id=sample_id,
        label=label,
        positions=positions,
        gene_ids=gene_ids,
        chromosomes=chromosomes,
        variant_keys=variant_keys,
    )


def _variant_key(
    run_id: str,
    sample_id: str,
    chromosome: object,
    position: object,
    gene_id: object,
) -> str:
    if not isinstance(chromosome, str) or not chromosome:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} chromosome must be a non-empty string"
        )
    pos = _integer_like(position, f"run {run_id!r} sample {sample_id!r} position")
    gene = _integer_like(gene_id, f"run {run_id!r} sample {sample_id!r} gene_id")
    return f"{chromosome}:{pos}_{gene}"


def _integer_like(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be integer-like and not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, np.floating) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{path} must be integer-like")


def _validate_per_sample_file_set(
    spec: PositionAttributionRunSpec,
    samples: Mapping[str, SampleMetadata],
) -> None:
    expected = {metadata.sample_idx for metadata in samples.values()}
    discovered = {}
    for path in spec.per_sample_dir.iterdir():
        match = SAMPLE_FILE_RE.match(path.name)
        if match is not None:
            discovered[int(match.group(1))] = path
    missing = sorted(expected - set(discovered))
    extra = sorted(set(discovered) - expected)
    if missing:
        raise ValueError(
            f"run {spec.run_id!r} missing per-sample files for sample_idx values {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"run {spec.run_id!r} has extra sample_N.npz files for sample_idx values {extra[:5]}"
        )


def _validate_content_ig_metadata(
    run_id: str,
    config: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    identity: PositionStrategyIdentity,
) -> None:
    if analysis_metadata.get("is_null_baseline") is not False:
        raise ValueError(f"run {run_id!r} analysis_metadata.is_null_baseline must be False")
    if analysis_metadata.get("annotation_level") != config.get("level"):
        raise ValueError(
            f"run {run_id!r} annotation_level mismatch: expected {config.get('level')!r}, "
            f"artifact value {analysis_metadata.get('annotation_level')!r}"
        )
    genome_build = _mapping(config, "dataset_identity", run_id).get("genome_build")
    if analysis_metadata.get("genome_build") != genome_build:
        raise ValueError(
            f"run {run_id!r} genome_build mismatch: expected {genome_build!r}, "
            f"artifact value {analysis_metadata.get('genome_build')!r}"
        )
    ig_metadata = _ig_metadata(analysis_metadata, run_id)
    required_values = {
        "executed": True,
        "resolved_ig_mode": "content",
        "attribution_feature_space": "content",
        "baseline_policy": "zero_content_observed_absolute_position",
        "comparability_warning": None,
        "position_encoding_metadata_source": "reconstructed_resolved_config",
    }
    for field, expected in required_values.items():
        if ig_metadata.get(field) != expected:
            raise ValueError(
                f"run {run_id!r} field integrated_gradients.{field!r} expected "
                f"{expected!r}, artifact value {ig_metadata.get(field)!r}"
            )
    content_dim = _config_content_dim(config, run_id)
    if ig_metadata.get("content_dim") != content_dim:
        raise ValueError(
            f"run {run_id!r} field integrated_gradients.content_dim expected "
            f"{content_dim!r}, artifact value {ig_metadata.get('content_dim')!r}"
        )
    if ig_metadata.get("attribution_width") != content_dim:
        raise ValueError(
            f"run {run_id!r} field integrated_gradients.attribution_width expected "
            f"{content_dim!r}, artifact value {ig_metadata.get('attribution_width')!r}"
        )
    input_dim = _config_input_dim(config, run_id)
    if ig_metadata.get("input_dim") != input_dim:
        raise ValueError(
            f"run {run_id!r} field integrated_gradients.input_dim expected "
            f"{input_dim!r}, artifact value {ig_metadata.get('input_dim')!r}"
        )
    strategy_fields = {
        "absolute_position_encoding": identity.payload["absolute"]["type"],
        "relative_position_encoding": identity.payload["relative"]["type"],
        "chromosome_encoding": identity.payload["chromosome"]["encoding"],
    }
    for field, expected in strategy_fields.items():
        if ig_metadata.get(field) != expected:
            raise ValueError(
                f"run {run_id!r} field integrated_gradients.{field!r} expected "
                f"{expected!r}, artifact value {ig_metadata.get(field)!r}"
            )


def _validate_model_provenance(
    spec: PositionAttributionRunSpec,
    analysis_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = analysis_metadata.get("model_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"run {spec.run_id!r} requires model_provenance")
    if set(provenance) != MODEL_PROVENANCE_KEYS:
        raise ValueError(f"run {spec.run_id!r} model_provenance schema keys are invalid")
    if provenance.get("schema_version") != 1 or isinstance(provenance.get("schema_version"), bool):
        raise ValueError(f"run {spec.run_id!r} model_provenance.schema_version must be 1")
    mode = provenance.get("checkpoint_selection_mode")
    allowed = {
        "cv_explicit_fold",
        "cv_best_fold",
        "single_run_best_model",
        "explicit_checkpoint",
    }
    if not isinstance(mode, str) or mode not in allowed:
        raise ValueError(
            f"run {spec.run_id!r} model_provenance.checkpoint_selection_mode is invalid"
        )
    checkpoint_path = _absolute_path(
        provenance.get("checkpoint_path"), spec.run_id, "checkpoint_path"
    )
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise ValueError(f"run {spec.run_id!r} checkpoint file is missing: {checkpoint_path}")
    saved_hash = provenance.get("checkpoint_sha256")
    if not isinstance(saved_hash, str) or SHA256_RE.fullmatch(saved_hash) is None:
        raise ValueError(f"run {spec.run_id!r} checkpoint_sha256 must be lowercase SHA-256")
    actual_hash = _sha256_file(checkpoint_path)
    if actual_hash != saved_hash:
        raise ValueError(
            f"run {spec.run_id!r} checkpoint hash mismatch for {checkpoint_path}: "
            f"saved hash {saved_hash}, recomputed hash {actual_hash}"
        )
    config_path = _absolute_path(provenance.get("config_path"), spec.run_id, "config_path")
    if config_path.resolve() != spec.config_path.resolve():
        raise ValueError(
            f"run {spec.run_id!r} config_path mismatch: expected "
            f"{spec.config_path.resolve()}, artifact value {config_path}"
        )
    _validate_nullable_fold_fields(spec.run_id, provenance)
    return dict(provenance)


def _validate_nullable_fold_fields(run_id: str, provenance: Mapping[str, Any]) -> None:
    selected_fold = provenance.get("selected_fold")
    selected_fold_auc = provenance.get("selected_fold_auc")
    cv_results_path = provenance.get("cv_results_path")
    mode = provenance["checkpoint_selection_mode"]
    if mode in {"cv_explicit_fold", "cv_best_fold"}:
        _required_non_negative_int(selected_fold, f"run {run_id!r} selected_fold")
        _required_finite_number(selected_fold_auc, f"run {run_id!r} selected_fold_auc")
        _absolute_path(cv_results_path, run_id, "cv_results_path")
    else:
        if (
            selected_fold is not None
            or selected_fold_auc is not None
            or cv_results_path is not None
        ):
            raise ValueError(
                f"run {run_id!r} single-run checkpoint provenance must not include "
                "selected_fold, selected_fold_auc, or cv_results_path"
            )


def _validate_checkpoint_policy(
    runs: Sequence[LoadedPositionAttributionRun],
) -> dict[str, object]:
    training_mode = runs[0].training_context.fields["position_encoding_execution.training_mode"]
    if training_mode == "cv":
        modes = {run.model_provenance["checkpoint_selection_mode"] for run in runs}
        if modes != {"cv_explicit_fold"}:
            raise ValueError(
                "cv training_mode requires checkpoint_selection_mode cv_explicit_fold "
                f"for every run; observed {sorted(modes)}"
            )
        folds = {run.model_provenance["selected_fold"] for run in runs}
        if len(folds) != 1:
            raise ValueError(f"cv selected_fold mismatch across runs: {sorted(folds)}")
        return {
            "training_mode": "cv",
            "checkpoint_selection_mode": "cv_explicit_fold",
            "selected_fold": next(iter(folds)),
            "matched_fold_required": True,
        }
    if training_mode == "single_split":
        modes = {run.model_provenance["checkpoint_selection_mode"] for run in runs}
        allowed = {"single_run_best_model", "explicit_checkpoint"}
        if not modes <= allowed:
            raise ValueError(
                "single_split training_mode requires single_run_best_model or "
                f"explicit_checkpoint; observed {sorted(modes)}"
            )
        if len(modes) != 1:
            raise ValueError(
                f"single_split checkpoint_selection_mode mismatch across runs: {sorted(modes)}"
            )
        for run in runs:
            if (
                run.model_provenance["selected_fold"] is not None
                or run.model_provenance["selected_fold_auc"] is not None
                or run.model_provenance["cv_results_path"] is not None
            ):
                raise ValueError(
                    f"run {run.spec.run_id!r} single_split provenance must have null CV fields"
                )
        return {
            "training_mode": "single_split",
            "checkpoint_selection_mode": next(iter(modes)),
            "selected_fold": None,
            "matched_fold_required": False,
        }
    raise ValueError(f"unknown training_mode for attribution comparison: {training_mode!r}")


def _validate_sample_universe(runs: Sequence[LoadedPositionAttributionRun]) -> None:
    reference = runs[0]
    reference_ids = set(reference.samples_by_id)
    for run in runs[1:]:
        sample_ids = set(run.samples_by_id)
        if sample_ids != reference_ids:
            missing_ids = sorted(reference_ids - sample_ids)
            extra_ids = sorted(sample_ids - reference_ids)
            raise ValueError(
                f"sample universe mismatch: reference run {reference.spec.run_id!r} "
                f"other run {run.spec.run_id!r}, reference sample count "
                f"{len(reference_ids)}, other sample count {len(sample_ids)}, "
                f"missing count {len(missing_ids)}, extra count {len(extra_ids)}, "
                f"missing examples {missing_ids[:5]}, extra examples {extra_ids[:5]}"
            )
        for sample_id in sorted(reference_ids):
            if run.samples_by_id[sample_id].label != reference.samples_by_id[sample_id].label:
                raise ValueError(
                    f"label mismatch for sample {sample_id!r}: reference run "
                    f"{reference.spec.run_id!r} has "
                    f"{reference.samples_by_id[sample_id].label!r}, run "
                    f"{run.spec.run_id!r} has {run.samples_by_id[sample_id].label!r}"
                )


def _feature_names(config: Mapping[str, Any], run_id: str) -> list[str]:
    level = config.get("level")
    if level not in FEATURE_NAMES_BY_LEVEL:
        raise ValueError(f"run {run_id!r} unsupported annotation level {level!r}")
    names = FEATURE_NAMES_BY_LEVEL[str(level)]
    content_dim = _config_content_dim(config, run_id)
    if len(names) != content_dim:
        raise ValueError(
            f"run {run_id!r} content_dim {content_dim} does not match "
            f"{len(names)} feature names for level {level}"
        )
    return list(names)


def _build_yaml_output(
    runs: Sequence[LoadedPositionAttributionRun],
    training_report: Any,
    explanation_report: Any,
    checkpoint_policy: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, object]],
    out_summary_tsv: pathlib.Path,
    out_sample_tsv: pathlib.Path,
    out_feature_tsv: pathlib.Path,
    out_comparison_yaml: pathlib.Path,
) -> dict[str, object]:
    reference = runs[0]
    return {
        "schema_version": 1,
        "comparison_axis": "position",
        "analysis_type": "attribution_stability",
        "metrics": {
            "primary": "signed_cosine",
            "definitions": {
                "signed_cosine": "cosine similarity on flattened signed raw content attributions",
                "signed_pearson": "Pearson correlation on flattened signed raw content attributions",
                "score_cosine": "cosine similarity on per-variant L2 attribution scores",
                "score_normalized_l2": "||scores_a-scores_b||/(||scores_a||+||scores_b||)",
            },
            "summary_weighting": "equal_per_sample",
        },
        "zero_constant_policy": {
            "cosine": {
                "both_zero": 1.0,
                "one_zero": 0.0,
            },
            "pearson": {
                "insufficient_values": "invalid",
                "both_constant": "invalid",
                "one_constant": "invalid",
            },
        },
        "compatibility": {
            "training_context": training_report.to_dict(),
            "explanation_context": explanation_report.to_dict(),
            "checkpoint_policy": dict(checkpoint_policy),
        },
        "runs": [
            {
                "run_id": run.spec.run_id,
                "position_strategy_id": run.identity.strategy_id,
                "position_strategy_name": run.identity.name,
                "position_strategy_hash": run.identity.hash,
                "position_strategy": run.identity.payload,
                "config_path": str(run.spec.config_path.resolve()),
                "analysis_metadata_path": str(run.spec.analysis_metadata_path.resolve()),
                "attributions_path": str(run.spec.attributions_path.resolve()),
                "per_sample_dir": str(run.spec.per_sample_dir.resolve()),
                "checkpoint_path": run.model_provenance["checkpoint_path"],
                "checkpoint_sha256": run.model_provenance["checkpoint_sha256"],
                "checkpoint_selection_mode": run.model_provenance["checkpoint_selection_mode"],
                "selected_fold": run.model_provenance["selected_fold"],
                "selected_fold_auc": run.model_provenance["selected_fold_auc"],
                "sample_count": len(run.samples_by_id),
            }
            for run in runs
        ],
        "alignment": {
            "sample_key": "sample_id",
            "variant_key": "chromosome_position_gene_id",
            "exact_sample_universe_required": True,
            "exact_per_sample_variant_universe_required": True,
            "labels_must_match": True,
        },
        "artifact_integrity": {
            "aggregate_pickle_required": True,
            "aggregate_pickle_trust_boundary": "trusted_local_sieve_explanation_artifact",
            "variant_score_formula": "l2_norm_of_raw_attribution_row",
            "variant_score_rtol": SCORE_RTOL,
            "variant_score_atol": SCORE_ATOL,
        },
        "content_features": {
            "annotation_level": reference.config["level"],
            "content_dim": _config_content_dim(reference.config, reference.spec.run_id),
            "names": reference.feature_names,
        },
        "outputs": {
            "summary_tsv": str(out_summary_tsv.resolve()),
            "sample_tsv": str(out_sample_tsv.resolve()),
            "feature_tsv": str(out_feature_tsv.resolve()),
            "comparison_yaml": str(out_comparison_yaml.resolve()),
        },
        "pairwise_summary": [dict(row) for row in summary_rows],
        "warnings": [
            "Aggregate attributions.npz files require allow_pickle=True because "
            "SIEVE stores variant_scores and metadata as object arrays; compare "
            "only trusted local SIEVE explanation artifacts."
        ],
    }


def _write_tsv(
    path: pathlib.Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _tsv_value(row.get(field)) for field in fieldnames})


def _dump_yaml(value: Mapping[str, object], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _require_file(path: pathlib.Path, run_id: str, artifact: str) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"run {run_id!r} {artifact} file does not exist: {path}")


def _ig_metadata(analysis_metadata: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    ig_metadata = analysis_metadata.get("integrated_gradients")
    if not isinstance(ig_metadata, Mapping):
        raise ValueError(f"run {run_id!r} requires integrated_gradients metadata")
    return ig_metadata


def _config_content_dim(config: Mapping[str, Any], run_id: str) -> int:
    value = config.get("content_dim")
    return _required_positive_int(value, f"run {run_id!r} config.content_dim")


def _config_input_dim(config: Mapping[str, Any], run_id: str) -> int:
    value = config.get("input_dim")
    return _required_positive_int(value, f"run {run_id!r} config.input_dim")


def _mapping(data: Mapping[str, Any], key: str, run_id: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"run {run_id!r} {key} must be a mapping")
    return value


def _npz_scalar(
    data: Mapping[str, Any],
    key: str,
    run_id: str,
    *,
    sample_id: str | None = None,
) -> object:
    value = np.asarray(data[key])
    location = f"run {run_id!r}"
    if sample_id is not None:
        location += f" sample {sample_id!r}"
    if value.shape != ():
        raise ValueError(f"{location} field {key!r} must be a scalar NPZ value")
    scalar = value.item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if key == "sampling_seed" and scalar == -1:
        return None
    if key == "comparability_warning" and scalar == "":
        return None
    if (
        key
        in {
            "absolute_position_encoding",
            "relative_position_encoding",
            "chromosome_encoding",
            "position_encoding_metadata_source",
        }
        and scalar == "unavailable"
    ):
        raise ValueError(f"{location} field {key!r} is unavailable")
    return scalar


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, Mapping):
        return value
    raise ValueError("aggregate metadata entries must be mappings")


def _one_dimensional_object_array(
    value: object,
    run_id: str,
    sample_id: str,
    field: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} metadata field {field!r} "
            "must be one-dimensional"
        )
    return array


def _validate_numeric_matrix(
    attributions: np.ndarray,
    run_id: str,
    sample_id: str,
    expected_rows: int,
    expected_width: int,
) -> None:
    if attributions.ndim != 2:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} attributions must be two-dimensional"
        )
    if attributions.shape != (expected_rows, expected_width):
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} attribution shape mismatch: "
            f"expected {(expected_rows, expected_width)}, actual {attributions.shape}"
        )
    if not np.issubdtype(attributions.dtype, np.number):
        raise ValueError(f"run {run_id!r} sample {sample_id!r} attributions must be numeric")
    if not np.all(np.isfinite(attributions)):
        raise ValueError(f"run {run_id!r} sample {sample_id!r} attributions must be finite")


def _validate_score_vector(
    scores: np.ndarray,
    run_id: str,
    sample_id: str,
    expected_rows: int,
) -> None:
    if scores.ndim != 1:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} variant_scores must be one-dimensional"
        )
    if scores.shape[0] != expected_rows:
        raise ValueError(
            f"run {run_id!r} sample {sample_id!r} variant_scores length mismatch: "
            f"expected {expected_rows}, actual {scores.shape[0]}"
        )
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError(f"run {run_id!r} sample {sample_id!r} variant_scores must be numeric")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"run {run_id!r} sample {sample_id!r} variant_scores must be finite")


def _absolute_path(value: object, run_id: str, field: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"run {run_id!r} {field} must be a non-empty absolute path")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise ValueError(f"run {run_id!r} {field} must be an absolute path")
    return path


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _required_non_negative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _required_finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _cosine_from_stats(stats: FeatureStats) -> MetricResult:
    norm_a = math.sqrt(stats.sum_a2)
    norm_b = math.sqrt(stats.sum_b2)
    if norm_a == 0.0 and norm_b == 0.0:
        return MetricResult(1.0, True, "both_zero")
    if norm_a == 0.0 or norm_b == 0.0:
        return MetricResult(0.0, True, "one_zero")
    return MetricResult(_clamp_correlation(stats.sum_ab / (norm_a * norm_b)), True, "ok")


def _pearson_from_stats(stats: FeatureStats) -> MetricResult:
    if stats.n < 2:
        return MetricResult(None, False, "insufficient_values")
    numerator = stats.n * stats.sum_ab - stats.sum_a * stats.sum_b
    variance_a = stats.n * stats.sum_a2 - stats.sum_a * stats.sum_a
    variance_b = stats.n * stats.sum_b2 - stats.sum_b * stats.sum_b
    constant_a = variance_a <= 0.0
    constant_b = variance_b <= 0.0
    if constant_a and constant_b:
        return MetricResult(None, False, "both_constant")
    if constant_a or constant_b:
        return MetricResult(None, False, "one_constant")
    value = numerator / math.sqrt(variance_a * variance_b)
    return MetricResult(_clamp_correlation(value), True, "ok")


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _std_population(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _clamp_correlation(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        specs = parse_position_run_specs(args.position_run)
        compare_position_attributions(
            specs,
            out_summary_tsv=args.out_summary_tsv,
            out_sample_tsv=args.out_sample_tsv,
            out_feature_tsv=args.out_feature_tsv,
            out_comparison_yaml=args.out_comparison_yaml,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
