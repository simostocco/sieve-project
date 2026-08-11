#!/usr/bin/env python3
"""
Summarise ablation performance across annotation levels.

Collects model metrics (AUC, accuracy, loss) from training run directories
and ranks annotation levels by predictive performance. Supports both the
standard SIEVE experiment layout (``experiments/ablation_L{0..3}/``) and
explicit ``--run-dir`` arguments.

Usage:
    # Standard layout
    python scripts/ablation_compare.py \\
        --results-dir experiments \\
        --out-summary-tsv ablation_summary.tsv \\
        --out-summary-yaml ablation_summary.yaml

    # Explicit run directories
    python scripts/ablation_compare.py \\
        --run-dir experiments/ablation_L0 \\
        --run-dir experiments/ablation_L1 \\
        --run-dir experiments/ablation_L2 \\
        --run-dir experiments/ablation_L3

Author: Francesco Lescai
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List

import yaml

if __package__ in (None, ""):
    from position_benchmark_metadata import (
        extract_comparison_context,
        position_strategy_identity,
        require_compatible_contexts,
    )
else:
    from .position_benchmark_metadata import (
        extract_comparison_context,
        position_strategy_identity,
        require_compatible_contexts,
    )


AUC_KEYS = (
    "auc",
    "roc_auc",
    "mean_auc",
    "metrics.auc",
    "metrics.roc_auc",
    "validation.auc",
    "validation.roc_auc",
    "val.auc",
    "val.roc_auc",
    "val_auc",
)

ACC_KEYS = (
    "accuracy",
    "acc",
    "mean_accuracy",
    "metrics.accuracy",
    "metrics.acc",
    "validation.accuracy",
    "validation.acc",
    "val.accuracy",
    "val.acc",
    "val_accuracy",
)

LOSS_KEYS = (
    "loss",
    "mean_loss",
    "metrics.loss",
    "validation.loss",
    "val.loss",
    "val_loss",
)

STD_AUC_KEYS = (
    "std_auc",
    "auc_std",
    "std.auc",
    "validation.std_auc",
)

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        help=(
            "Ablation run directory containing results.yaml / cv_results.yaml "
            "and config.yaml (repeatable)"
        ),
    )
    group.add_argument(
        "--results-dir",
        type=str,
        help=(
            "Parent directory with standard layout "
            "ablation_L{0..3}/ sub-directories"
        ),
    )
    parser.add_argument(
        "--comparison-axis",
        choices=["level", "position"],
        default="level",
        help=(
            "Comparison axis. 'level' preserves historical annotation-ablation "
            "behavior; 'position' compares explicit runs by positional strategy."
        ),
    )
    parser.add_argument(
        "--out-summary-tsv",
        default="ablation_summary.tsv",
        help="Output TSV path (default: ablation_summary.tsv)",
    )
    parser.add_argument(
        "--out-summary-yaml",
        default="ablation_summary.yaml",
        help="Output YAML path (default: ablation_summary.yaml)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dict.

    Parameters
    ----------
    path : pathlib.Path
        YAML file to read.

    Returns
    -------
    Dict[str, Any]
        Parsed contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at: {path}")
    return data


def dump_yaml(value: Any, path: pathlib.Path) -> None:
    """
    Write a Python object as YAML.

    Parameters
    ----------
    value : Any
        Data structure to serialise.
    path : pathlib.Path
        Destination file.
    """
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------


def flatten_dict(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively flatten a nested dict using dot notation.

    Parameters
    ----------
    obj : dict
        Input dictionary (possibly nested).
    prefix : str
        Key prefix for recursion.

    Returns
    -------
    Dict[str, Any]
        Flattened dictionary with lower-cased dotted keys.
    """
    flat: Dict[str, Any] = {}
    for key, value in obj.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        flat[dotted.lower()] = value
        if isinstance(value, dict):
            flat.update(flatten_dict(value, dotted))
    return flat


def as_float(value: Any) -> float:
    """Convert a value to float, returning NaN on failure."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pick_metric(flat: Dict[str, Any], keys: Iterable[str]) -> float:
    """
    Pick the first available metric from a flattened dict.

    Parameters
    ----------
    flat : Dict[str, Any]
        Flattened results dictionary.
    keys : Iterable[str]
        Candidate key names in priority order.

    Returns
    -------
    float
        Metric value, or NaN if not found.
    """
    for key in keys:
        if key.lower() in flat:
            return as_float(flat[key.lower()])
    return math.nan


def metric_rank_value(value: float, maximize: bool) -> float:
    """Return a sort key for ranking (NaN → worst)."""
    if math.isnan(value):
        return math.inf
    return -value if maximize else value


def resolve_level(run_id: str, config_data: Dict[str, Any]) -> str:
    """
    Determine the annotation level from config data or directory name.

    Parameters
    ----------
    run_id : str
        Directory name of the run.
    config_data : Dict[str, Any]
        Parsed config.yaml contents.

    Returns
    -------
    str
        Level label (e.g. ``'L2'``), or ``'UNKNOWN'``.
    """
    level_candidates = [
        config_data.get("annotation_level"),
        config_data.get("level"),
    ]
    # Check nested structures
    for nested_key in ("train", "data", "encoding"):
        nested = config_data.get(nested_key)
        if isinstance(nested, dict):
            level_candidates.append(nested.get("annotation_level"))
            level_candidates.append(nested.get("level"))

    for candidate in level_candidates:
        if isinstance(candidate, str) and candidate in LEVEL_ORDER:
            return candidate

    # Fall back to directory name
    match = re.search(r"(L[0-3])", run_id)
    if match:
        return match.group(1)
    return "UNKNOWN"


def _discover_run_dirs(results_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Find ablation run directories under *results_dir*.

    Looks for ``ablation_L{0..3}`` sub-directories, then falls back to
    any directory whose name matches ``*L[0-3]*``.
    """
    found: List[pathlib.Path] = []
    for level in ("L0", "L1", "L2", "L3"):
        candidate = results_dir / f"ablation_{level}"
        if candidate.is_dir():
            found.append(candidate)

    if found:
        return found

    # Flexible fallback
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and re.search(r"L[0-3]", d.name):
            found.append(d)
    return found


def _results_yaml_path(run_dir: pathlib.Path) -> pathlib.Path:
    """Return the results file path using the historical preference order."""
    results_yaml = run_dir / "results.yaml"
    if not results_yaml.exists():
        results_yaml = run_dir / "cv_results.yaml"
    return results_yaml


def _extract_metrics(results_data: dict[str, Any]) -> dict[str, float]:
    """Extract predictive metrics using the existing flexible key aliases."""
    flat_results = flatten_dict(results_data)
    return {
        "auc": pick_metric(flat_results, AUC_KEYS),
        "std_auc": pick_metric(flat_results, STD_AUC_KEYS),
        "accuracy": pick_metric(flat_results, ACC_KEYS),
        "loss": pick_metric(flat_results, LOSS_KEYS),
    }


def _rank_performance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by the historical performance priority."""
    return sorted(
        rows,
        key=lambda row: (
            metric_rank_value(row["auc"], maximize=True),
            metric_rank_value(row["accuracy"], maximize=True),
            metric_rank_value(row["loss"], maximize=False),
            row["run_id"],
        ),
    )


def _run_level_comparison(args: argparse.Namespace) -> int:
    """Run the historical annotation-level ablation comparison."""
    # Resolve run directories
    if args.results_dir:
        results_dir = pathlib.Path(args.results_dir).resolve()
        run_dirs = _discover_run_dirs(results_dir)
        if not run_dirs:
            print(
                f"ERROR: No ablation run directories found in {results_dir}",
                file=sys.stderr,
            )
            return 1
    else:
        run_dirs = [pathlib.Path(d).resolve() for d in args.run_dirs]

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            print(
                f"WARNING: Skipping non-existent directory: {run_dir}",
                file=sys.stderr,
            )
            continue

        run_id = run_dir.name
        results_yaml = _results_yaml_path(run_dir)
        config_yaml = run_dir / "config.yaml"

        try:
            results_data = load_yaml(results_yaml)
        except FileNotFoundError:
            print(
                f"WARNING: No results.yaml or cv_results.yaml in {run_dir}",
                file=sys.stderr,
            )
            continue

        try:
            config_data = load_yaml(config_yaml)
        except FileNotFoundError:
            # Config is optional, infer level from directory name
            config_data = {}

        metrics = _extract_metrics(results_data)
        rows.append(
            {
                "run_id": run_id,
                "level": resolve_level(run_id, config_data),
                **metrics,
                "results_yaml": str(results_yaml),
            }
        )

    if not rows:
        print("ERROR: No ablation runs could be loaded", file=sys.stderr)
        return 1

    # Sort by level order
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            LEVEL_ORDER.get(row["level"], math.inf),
            row["run_id"],
        ),
    )

    # Rank by performance (best first)
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            metric_rank_value(row["auc"], maximize=True),
            metric_rank_value(row["accuracy"], maximize=True),
            metric_rank_value(row["loss"], maximize=False),
            LEVEL_ORDER.get(row["level"], math.inf),
            row["run_id"],
        ),
    )
    best = ranked_rows[0]

    # Write TSV
    tsv_path = pathlib.Path(args.out_summary_tsv)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["level", "run_id", "auc", "std_auc", "accuracy", "loss", "results_yaml"]
        )
        for row in rows_sorted:
            writer.writerow(
                [
                    row["level"],
                    row["run_id"],
                    "" if math.isnan(row["auc"]) else row["auc"],
                    "" if math.isnan(row["std_auc"]) else row["std_auc"],
                    "" if math.isnan(row["accuracy"]) else row["accuracy"],
                    "" if math.isnan(row["loss"]) else row["loss"],
                    row["results_yaml"],
                ]
            )

    # Write YAML summary
    summary_yaml: Dict[str, Any] = {
        "best_level": best["level"],
        "best_run_id": best["run_id"],
        "ranking_metric_priority": ["auc", "accuracy", "loss"],
        "levels": [
            {
                "level": row["level"],
                "run_id": row["run_id"],
                "auc": None if math.isnan(row["auc"]) else row["auc"],
                "std_auc": None if math.isnan(row["std_auc"]) else row["std_auc"],
                "accuracy": None
                if math.isnan(row["accuracy"])
                else row["accuracy"],
                "loss": None if math.isnan(row["loss"]) else row["loss"],
            }
            for row in rows_sorted
        ],
    }

    yaml_path = pathlib.Path(args.out_summary_yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(summary_yaml, yaml_path)

    # Print summary
    print(f"Ablation Performance Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for row in rows_sorted:
        auc_str = f"{row['auc']:.4f}" if not math.isnan(row["auc"]) else "N/A"
        std_str = f" +/- {row['std_auc']:.4f}" if not math.isnan(row["std_auc"]) else ""
        acc_str = f"{row['accuracy']:.4f}" if not math.isnan(row["accuracy"]) else "N/A"
        loss_str = f"{row['loss']:.4f}" if not math.isnan(row["loss"]) else "N/A"
        print(
            f"  {row['level']:>4s}  AUC={auc_str}{std_str}  Acc={acc_str}  Loss={loss_str}",
            file=sys.stderr,
        )
    print(f"\nBest level: {best['level']} ({best['run_id']})", file=sys.stderr)
    print(f"Summary TSV: {args.out_summary_tsv}", file=sys.stderr)
    print(f"Summary YAML: {args.out_summary_yaml}", file=sys.stderr)
    return 0


def _strategy_column(identity, section: str, key: str) -> object:
    return identity.payload[section][key]


def _run_position_comparison(args: argparse.Namespace) -> int:
    """Run positional-strategy predictive performance comparison."""
    if args.results_dir is not None:
        print(
            "ERROR: --comparison-axis position requires explicit --run-dir inputs; "
            "--results-dir discovery is level-mode only.",
            file=sys.stderr,
        )
        return 1
    if not args.run_dirs:
        print("ERROR: --comparison-axis position requires at least one --run-dir", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    contexts = []

    for run_dir_arg in args.run_dirs:
        run_dir = pathlib.Path(run_dir_arg).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
            return 1

        run_id = run_dir.name
        config_path = run_dir / "config.yaml"
        results_yaml = _results_yaml_path(run_dir)

        try:
            config_data = load_yaml(config_path)
            results_data = load_yaml(results_yaml)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: Failed to load {run_dir}: {exc}", file=sys.stderr)
            return 1

        try:
            identity = position_strategy_identity(config_data)
            context = extract_comparison_context(config_data, run_id=run_id)
        except ValueError as exc:
            print(
                f"ERROR: Invalid position benchmark metadata in {run_dir}: {exc}",
                file=sys.stderr,
            )
            return 1

        metrics = _extract_metrics(results_data)
        rows.append(
            {
                "position_strategy_id": identity.strategy_id,
                "position_strategy_name": identity.name,
                "position_strategy_hash": identity.hash,
                "run_id": run_id,
                "level": config_data["level"],
                "preset": identity.payload["preset"],
                "absolute_position_encoding": _strategy_column(identity, "absolute", "type"),
                "relative_position_encoding": _strategy_column(identity, "relative", "type"),
                "chromosome_encoding": _strategy_column(identity, "chromosome", "encoding"),
                "cross_chromosome_policy": _strategy_column(
                    identity,
                    "chromosome",
                    "cross_chromosome_policy",
                ),
                **metrics,
                "config_path": str(config_path),
                "results_yaml": str(results_yaml),
                "position_strategy": identity.payload,
            }
        )
        contexts.append(context)

    try:
        compatibility = require_compatible_contexts(contexts)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows_sorted = sorted(rows, key=lambda row: (row["position_strategy_id"], row["run_id"]))
    best = _rank_performance_rows(rows)[0]

    tsv_columns = [
        "position_strategy_id",
        "position_strategy_name",
        "position_strategy_hash",
        "run_id",
        "level",
        "preset",
        "absolute_position_encoding",
        "relative_position_encoding",
        "chromosome_encoding",
        "cross_chromosome_policy",
        "auc",
        "std_auc",
        "accuracy",
        "loss",
        "config_path",
        "results_yaml",
    ]
    tsv_path = pathlib.Path(args.out_summary_tsv)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_columns, delimiter="\t")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    column: (
                        ""
                        if isinstance(row[column], float) and math.isnan(row[column])
                        else row[column]
                    )
                    for column in tsv_columns
                }
            )

    yaml_summary: dict[str, Any] = {
        "comparison_axis": "position",
        "ranking_metric_priority": ["auc", "accuracy", "loss"],
        "best_position_strategy_id": best["position_strategy_id"],
        "best_run_id": best["run_id"],
        "compatibility": compatibility.to_dict(),
        "runs": [
            {
                "run_id": row["run_id"],
                "position_strategy_id": row["position_strategy_id"],
                "position_strategy_name": row["position_strategy_name"],
                "position_strategy_hash": row["position_strategy_hash"],
                "position_strategy": row["position_strategy"],
                "metrics": {
                    "auc": None if math.isnan(row["auc"]) else row["auc"],
                    "std_auc": None if math.isnan(row["std_auc"]) else row["std_auc"],
                    "accuracy": None if math.isnan(row["accuracy"]) else row["accuracy"],
                    "loss": None if math.isnan(row["loss"]) else row["loss"],
                },
                "paths": {
                    "config": row["config_path"],
                    "results": row["results_yaml"],
                },
            }
            for row in rows_sorted
        ],
    }
    yaml_path = pathlib.Path(args.out_summary_yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(yaml_summary, yaml_path)

    print("Position Strategy Performance Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for row in rows_sorted:
        auc_str = f"{row['auc']:.4f}" if not math.isnan(row["auc"]) else "N/A"
        std_str = f" +/- {row['std_auc']:.4f}" if not math.isnan(row["std_auc"]) else ""
        acc_str = f"{row['accuracy']:.4f}" if not math.isnan(row["accuracy"]) else "N/A"
        loss_str = f"{row['loss']:.4f}" if not math.isnan(row["loss"]) else "N/A"
        print(
            f"  {row['position_strategy_id']}  AUC={auc_str}{std_str}  "
            f"Acc={acc_str}  Loss={loss_str}",
            file=sys.stderr,
        )
    print(
        f"\nBest position strategy: {best['position_strategy_id']} ({best['run_id']})",
        file=sys.stderr,
    )
    print(f"Summary TSV: {args.out_summary_tsv}", file=sys.stderr)
    print(f"Summary YAML: {args.out_summary_yaml}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the ablation performance comparison."""
    args = parse_args()
    if args.comparison_axis == "position":
        return _run_position_comparison(args)
    return _run_level_comparison(args)


if __name__ == "__main__":
    sys.exit(main())
