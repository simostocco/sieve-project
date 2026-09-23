"""Manifest validation and dry-run planning for positional benchmarks.

This module builds deterministic command plans only. It does not execute
benchmark stages, load preprocessed cohorts, create null data, or inspect
completed training/explanation artifacts. Existing training, explanation, and
comparison scripts remain the execution and validation authorities.

Manifest ``schema_version: 1`` plans real-only positional benchmarks (Phase
12C2B) and is unchanged. ``schema_version: 2`` (Phase 12C3B1) additionally
plans, for every positional strategy, one null-trained model on ONE shared,
benchmark-level null artifact plus the per-strategy bootstrap calibration
command. v2 dry-run planning hashes the real/null artifact bytes and parses the
12C3A lineage sidecar, but never unpickles either cohort; passing this
lightweight binding does NOT authorize execution. Full ``validate_null_pair``
is the Phase 12C3B2 execution-preflight authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.data.genome import SUPPORTED_BUILDS
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionPreset,
    RelativePositionEncoding,
)
from src.training.split_plan import split_plan_sha256

DEFERRED_STRATEGY_IDENTITY = "deferred_until_saved_config"
DEFERRED_SAMPLE_BINDING = "deferred_to_train_runtime"
POSITION_SCORE_COLUMN = "mean_attribution"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_KEYS = {
    "schema_version",
    "benchmark_id",
    "annotation_level",
    "benchmark_role",
    "paths",
    "dataset",
    "training",
    "explanation",
    "runs",
    "runtime",
}
POSITION_KEYS = {
    "position_preset",
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
}
CUSTOM_REQUIRED_KEYS = {
    "absolute_position_encoding",
    "relative_position_encoding",
    "chromosome_encoding",
    "cross_chromosome_policy",
}
TRAIN_COMMON_KEYS = [
    "seed",
    "val_split",
    "latent_dim",
    "hidden_dim",
    "num_heads",
    "num_attention_layers",
    "aggregation_method",
    "classifier_type",
    "batch_size",
    "epochs",
    "lr",
    "lambda_attr",
    "early_stopping",
    "gradient_accumulation_steps",
    "class_weighting",
    "chunk_size",
    "chunk_overlap",
]
TRAIN_CHOICES = {
    "aggregation_method": {"mean", "max"},
    "classifier_type": {"flatten", "attention_pool"},
    "class_weighting": {"auto", "on", "off"},
}
EXPLAIN_AGGREGATION_CHOICES = {"mean", "max", "rank_average"}
RUN_LEAF_NAMES = ("training", "explanation")
COMPARISON_NAMES = ("performance", "raw_rankings", "raw_attributions")
PAIRED_SCHEMA_VERSION = 2
NULL_BASELINE_KEY = "null_baseline"
PAIRED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS | {NULL_BASELINE_KEY, "calibration"}
PAIRED_RUN_LEAF_NAMES = (
    "training",
    "explanation",
    "null_training",
    "null_explanation",
    "calibration",
)
PAIRED_CV_FOLD_INDEX = 0
PAIRED_CLASS_WEIGHTING = "off"
NULL_KEYS = {"artifact"}
NULL_SIDECAR_DERIVED_KEYS = (
    "lineage_sidecar",
    "source_dataset",
    "permutation_seed",
    "lineage_sha256",
    "source_artifact_sha256",
    "sample_ids_sha256",
    "null_artifact_sha256",
    "reuse",
)
CALIBRATION_KEYS = {
    "n_bootstrap",
    "seed",
    "top_k",
    "exclude_sex_chroms",
    "min_variants_per_gene",
    "gene_delta_rank_aggregation",
}
GENE_DELTA_RANK_AGGREGATION_CHOICES = {"max", "mean"}
CALIBRATION_RANKINGS_NAME = "bootstrap_calibrated_variant_rankings.csv"
CALIBRATION_GENE_STATS_NAME = "bootstrap_calibrated_variant_rankings_gene_stats.csv"
CALIBRATION_SUMMARY_NAME = "bootstrap_calibrated_variant_rankings_summary.yaml"
PAIRED_COMPATIBILITY_NAME = "paired_compatibility.yaml"
RESERVED_CALIBRATED_COMPARISON = "calibrated_rankings"
NULL_BINDING_VALIDATION_LEVEL = "sidecar_schema_and_file_sha256_only"
NULL_FULL_PAIR_VALIDATION = "deferred_to_phase_12c3b2_preflight_validate_null_pair"
SIDECAR_SAMPLE_BINDING = "bound_via_null_lineage_sidecar_sample_ids_sha256"


class BenchmarkManifestError(ValueError):
    """Raised for ordinary manifest and dry-run validation errors."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a benchmark manifest YAML mapping from *path*."""
    manifest_path = Path(path)
    data = _load_yaml_mapping(manifest_path, "manifest")
    if not isinstance(data, Mapping):
        raise BenchmarkManifestError("manifest must contain a YAML mapping")
    return dict(data)


def build_resolved_plan(
    manifest_path: str | Path,
    *,
    python_override: str | None = None,
    device_override: str | None = None,
    allow_existing_outputs: bool = False,
) -> dict[str, Any]:
    """Validate *manifest_path* and return a deterministic dry-run plan."""
    manifest_path = _resolve_input_file(Path(manifest_path), "manifest")
    manifest = load_manifest(manifest_path)
    manifest_file_sha256 = _sha256_file(manifest_path)
    validated = _validate_manifest(manifest, manifest_path)
    runtime = _resolve_runtime(
        validated.get("runtime", {}),
        manifest_path=manifest_path,
        python_override=python_override,
        device_override=device_override,
    )
    repo_root = Path(__file__).resolve().parent.parent
    output_root = _resolve_output_path(validated["paths"]["output_root"], manifest_path)
    _validate_output_root(output_root, validated)
    benchmark_root = (
        output_root / validated["benchmark_id"] / validated["annotation_level"]
    ).resolve(strict=False)
    split_plan = _load_structural_split_plan(
        validated["training"]["split_plan_path"],
        training=validated["training"],
    )
    paired = validated["schema_version"] == PAIRED_SCHEMA_VERSION
    null_binding = _build_null_binding(validated, split_plan) if paired else None
    if paired:
        split_plan["dataset_sample_binding_validation"] = SIDECAR_SAMPLE_BINDING
    runs = _build_runs(
        validated,
        runtime=runtime,
        repo_root=repo_root,
        benchmark_root=benchmark_root,
    )
    comparisons = _build_comparisons(
        runs, runtime=runtime, repo_root=repo_root, benchmark_root=benchmark_root
    )
    warnings = _inspect_existing_outputs(
        runs,
        comparisons,
        allow_existing_outputs=allow_existing_outputs,
    )
    _validate_output_collisions(
        runs,
        comparisons,
        dataset_path=validated["dataset"]["preprocessed_data_path"],
        split_plan_path=validated["training"]["split_plan_path"],
        sex_map_path=validated["training"]["sex_map_path"],
        pc_map_path=validated["training"]["pc_map_path"],
        null_input_paths=(
            [
                (Path(null_binding["null_artifact_path"]), "null_baseline.artifact"),
                (Path(null_binding["sidecar_path"]), "null lineage sidecar"),
            ]
            if paired
            else None
        ),
    )

    if paired:
        return {
            "schema_version": PAIRED_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_file_sha256,
            "repository_root": str(repo_root),
            "repository_revision": _read_repository_revision(repo_root),
            "benchmark": {
                "benchmark_id": validated["benchmark_id"],
                "annotation_level": validated["annotation_level"],
                "benchmark_role": validated["benchmark_role"],
            },
            "dataset": {
                "preprocessed_data": str(validated["dataset"]["preprocessed_data_path"]),
                "genome_build": validated["dataset"]["genome_build"],
            },
            "split_plan": split_plan,
            "null_binding": null_binding,
            "calibration": _plan_calibration_settings(validated["calibration"]),
            "paired_policy": _paired_policy(validated),
            "runtime": runtime,
            "runs": runs,
            "comparisons": comparisons,
            "reserved_comparisons": {
                RESERVED_CALIBRATED_COMPARISON: {
                    "directory": str(
                        benchmark_root / "comparisons" / RESERVED_CALIBRATED_COMPARISON
                    ),
                    "status": "reserved_for_phase_12c3c",
                }
            },
            "warnings": warnings,
            "null_execution": {
                "status": "planned_not_executed",
                "execution_authorized": False,
                "required_preflight": NULL_FULL_PAIR_VALIDATION,
            },
        }

    return {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha256,
        "repository_root": str(repo_root),
        "repository_revision": _read_repository_revision(repo_root),
        "benchmark": {
            "benchmark_id": validated["benchmark_id"],
            "annotation_level": validated["annotation_level"],
            "benchmark_role": validated["benchmark_role"],
        },
        "dataset": {
            "preprocessed_data": str(validated["dataset"]["preprocessed_data_path"]),
            "genome_build": validated["dataset"]["genome_build"],
        },
        "split_plan": split_plan,
        "runtime": runtime,
        "runs": runs,
        "comparisons": comparisons,
        "warnings": warnings,
        "null_execution": "deferred_to_phase_12c3",
    }


def build_human_summary(plan: Mapping[str, Any]) -> str:
    """Return a deterministic human-readable dry-run summary."""
    lines = [
        "POSITION BENCHMARK DRY RUN",
        f"Benchmark: {plan['benchmark']['benchmark_id']}",
        f"Level / role: {plan['benchmark']['annotation_level']} / {plan['benchmark']['benchmark_role']}",
        f"Manifest SHA256: {plan['manifest_file_sha256']}",
        f"Dataset: {plan['dataset']['preprocessed_data']}",
        (
            "Split plan: "
            f"{plan['split_plan']['input_path']} "
            f"({plan['split_plan']['membership_sha256']})"
        ),
        f"Runtime: {plan['runtime']['python']} on {plan['runtime']['device']}",
        "Run order:",
    ]
    paired = plan["schema_version"] == PAIRED_SCHEMA_VERSION
    if paired:
        binding = plan["null_binding"]
        lines[5:5] = [
            f"Null artifact: {binding['null_artifact_path']}",
            f"Null lineage sidecar: {binding['sidecar_path']}",
            f"Null lineage SHA256: {binding['lineage_sha256']}",
            f"Null source SHA256: {binding['source_artifact_sha256']}",
            f"Null artifact SHA256: {binding['null_artifact_sha256']}",
        ]
    for run in plan["runs"]:
        lines.extend(
            [
                f"  - {run['run_id']}",
                f"    position: {json.dumps(run['position_intent'], sort_keys=True)}",
                f"    training: {run['directories']['training']}",
                f"    explanation: {run['directories']['explanation']}",
                f"    train: {_display_argv(run['train_argv'])}",
                f"    explain: {_display_argv(run['explain_argv'])}",
            ]
        )
        if paired:
            lines.extend(
                [
                    f"    null training: {run['directories']['null_training']}",
                    f"    null explanation: {run['directories']['null_explanation']}",
                    f"    null train: {_display_argv(run['null_train_argv'])}",
                    f"    null explain: {_display_argv(run['null_explain_argv'])}",
                    f"    calibration: {_display_argv(run['calibration']['argv'])}",
                ]
            )
    lines.extend(
        [
            "Comparison commands:",
            f"  performance: {_display_argv(plan['comparisons']['performance']['argv'])}",
            f"  raw_rankings: {_display_argv(plan['comparisons']['raw_rankings']['argv'])}",
            f"  raw_attributions: {_display_argv(plan['comparisons']['raw_attributions']['argv'])}",
            (
                "NULL EXECUTION: planned only; not authorized until the Phase 12C3B2 "
                "validate_null_pair preflight"
                if paired
                else "NULL EXECUTION: deferred to Phase 12C3"
            ),
        ]
    )
    if plan["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in plan["warnings"])
    return "\n".join(lines)


def write_resolved_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    """Write one explicit resolved-plan YAML file, refusing overwrite."""
    output_path = Path(path)
    if output_path.exists():
        raise BenchmarkManifestError(f"--out-plan already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(plan), handle, sort_keys=False)


def _validate_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    raw_version = manifest.get("schema_version")
    paired = raw_version == PAIRED_SCHEMA_VERSION and not isinstance(raw_version, bool)
    top_level_keys = PAIRED_TOP_LEVEL_KEYS if paired else TOP_LEVEL_KEYS
    unknown = set(manifest) - top_level_keys
    if unknown:
        # key=str keeps historical ordering for string keys and tolerates a
        # YAML-null key (e.g. a bare ``null:``) without a TypeError.
        raise BenchmarkManifestError(
            f"manifest has unknown top-level keys: {sorted(unknown, key=str)}"
        )
    for key in sorted(top_level_keys - TOP_LEVEL_KEYS):
        if key not in manifest:
            raise BenchmarkManifestError(
                f"manifest.{key} is required for schema_version {PAIRED_SCHEMA_VERSION}"
            )
    for key in TOP_LEVEL_KEYS - {"runtime"}:
        if key not in manifest:
            raise BenchmarkManifestError(f"manifest.{key} is required")
    if not paired and (
        manifest["schema_version"] != 1 or isinstance(manifest["schema_version"], bool)
    ):
        raise BenchmarkManifestError("manifest.schema_version must be 1 or 2")

    benchmark_id = _validate_id(manifest["benchmark_id"], "manifest.benchmark_id")
    level = _required_choice(
        manifest["annotation_level"], {"L0", "L1", "L2", "L3"}, "manifest.annotation_level"
    )
    role = _required_choice(
        manifest["benchmark_role"], {"primary", "sensitivity"}, "manifest.benchmark_role"
    )
    if role == "primary" and level != "L3":
        raise BenchmarkManifestError("manifest.benchmark_role primary requires annotation_level L3")
    if role == "sensitivity" and level != "L0":
        raise BenchmarkManifestError(
            "manifest.benchmark_role sensitivity requires annotation_level L0"
        )

    paths = _validate_paths(manifest["paths"])
    dataset = _validate_dataset(manifest["dataset"], manifest_path)
    if (
        paired
        and isinstance(manifest["training"], Mapping)
        and manifest["training"].get("class_weighting") is False
    ):
        raise BenchmarkManifestError(
            "manifest.training.class_weighting parsed as boolean false; write the string "
            '"off" (quoted) because bare off is YAML boolean false'
        )
    training = _validate_training(manifest["training"], manifest_path)
    explanation = _validate_explanation(manifest["explanation"], training=training)
    runtime = _validate_runtime(manifest.get("runtime", {}), paired=paired)
    runs = _validate_runs(manifest["runs"], paired=paired)

    validated = {
        "schema_version": PAIRED_SCHEMA_VERSION if paired else 1,
        "benchmark_id": benchmark_id,
        "annotation_level": level,
        "benchmark_role": role,
        "paths": paths,
        "dataset": dataset,
        "training": training,
        "explanation": explanation,
        "runtime": runtime,
        "runs": runs,
    }
    if paired:
        _validate_paired_protocol(training=training, explanation=explanation)
        validated[NULL_BASELINE_KEY] = _validate_null_baseline(
            manifest[NULL_BASELINE_KEY], manifest_path
        )
        validated["calibration"] = _validate_calibration(manifest["calibration"])
    return validated


def _validate_paired_protocol(
    *, training: Mapping[str, Any], explanation: Mapping[str, Any]
) -> None:
    """Enforce the controlled real/null protocol restrictions of schema v2.

    ``class_weighting`` must be ``off``: ``auto`` switches on the training
    fold's case fraction and ``on`` computes a label-dependent ``pos_weight``,
    so either would make the loss definition differ between real and null
    after exact split replay. CV explanation must use fold 0: ``train.py``
    seeds once before the CV loop, so only fold 0 is guaranteed to start from
    the same RNG state on both sides (later folds inherit RNG consumption from
    label-dependent early stopping in earlier folds).
    """
    if training["class_weighting"] != PAIRED_CLASS_WEIGHTING:
        raise BenchmarkManifestError(
            "manifest.training.class_weighting must be 'off' for schema_version 2 "
            "paired benchmarks ('auto' and 'on' are label-dependent)"
        )
    if training["mode"] == "cv" and explanation["fold_index"] != PAIRED_CV_FOLD_INDEX:
        raise BenchmarkManifestError(
            "manifest.explanation.fold_index must be 0 for schema_version 2 paired CV "
            "benchmarks (only fold 0 shares the real/null post-seed RNG state)"
        )


def _validate_null_baseline(value: Any, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.null_baseline must be a mapping")
    restated = sorted(key for key in value if key in NULL_SIDECAR_DERIVED_KEYS)
    if restated:
        raise BenchmarkManifestError(
            f"manifest.null_baseline must not restate {restated}; lineage identity and the "
            "sidecar path are derived from the 12C3A lineage sidecar"
        )
    if set(value) != NULL_KEYS:
        raise BenchmarkManifestError("manifest.null_baseline must contain exactly artifact")
    artifact_path = _resolve_existing_path(
        value["artifact"], manifest_path, "manifest.null_baseline.artifact"
    )
    return {"artifact": value["artifact"], "artifact_path": artifact_path}


def _validate_calibration(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.calibration must be a mapping")
    unknown = set(value) - CALIBRATION_KEYS
    if unknown:
        raise BenchmarkManifestError(f"manifest.calibration has unknown keys: {sorted(unknown)}")
    missing = CALIBRATION_KEYS - set(value)
    if missing:
        raise BenchmarkManifestError(
            f"manifest.calibration missing required keys: {sorted(missing)}"
        )
    top_k = value["top_k"]
    if not isinstance(top_k, Sequence) or isinstance(top_k, (str, bytes)) or not top_k:
        raise BenchmarkManifestError(
            "manifest.calibration.top_k must be a non-empty list of positive integers"
        )
    top_k_values = [
        _required_positive_int(item, f"manifest.calibration.top_k[{index}]")
        for index, item in enumerate(top_k)
    ]
    if len(set(top_k_values)) != len(top_k_values):
        raise BenchmarkManifestError("manifest.calibration.top_k must not contain duplicates")
    if not isinstance(value["exclude_sex_chroms"], bool):
        raise BenchmarkManifestError("manifest.calibration.exclude_sex_chroms must be a bool")
    return {
        "n_bootstrap": _required_positive_int(
            value["n_bootstrap"], "manifest.calibration.n_bootstrap"
        ),
        "seed": _required_non_negative_int(value["seed"], "manifest.calibration.seed"),
        "top_k": top_k_values,
        "exclude_sex_chroms": value["exclude_sex_chroms"],
        "min_variants_per_gene": _required_positive_int(
            value["min_variants_per_gene"], "manifest.calibration.min_variants_per_gene"
        ),
        "gene_delta_rank_aggregation": _required_choice(
            value["gene_delta_rank_aggregation"],
            GENE_DELTA_RANK_AGGREGATION_CHOICES,
            "manifest.calibration.gene_delta_rank_aggregation",
        ),
    }


def _validate_paths(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.paths must be a mapping")
    if set(value) != {"output_root"}:
        raise BenchmarkManifestError("manifest.paths must contain exactly output_root")
    if not isinstance(value["output_root"], str) or not value["output_root"]:
        raise BenchmarkManifestError("manifest.paths.output_root must be a non-empty string")
    return {"output_root": value["output_root"]}


def _validate_dataset(value: Any, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.dataset must be a mapping")
    if set(value) != {"preprocessed_data", "genome_build"}:
        raise BenchmarkManifestError(
            "manifest.dataset must contain exactly preprocessed_data and genome_build"
        )
    preprocessed_data = _resolve_existing_path(
        value["preprocessed_data"],
        manifest_path,
        "manifest.dataset.preprocessed_data",
    )
    genome_build = _required_choice(
        value["genome_build"], set(SUPPORTED_BUILDS), "manifest.dataset.genome_build"
    )
    return {
        "preprocessed_data": value["preprocessed_data"],
        "preprocessed_data_path": preprocessed_data,
        "genome_build": genome_build,
    }


def _validate_training(value: Any, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.training must be a mapping")
    required = {
        "mode",
        "seed",
        "val_split",
        "split_plan",
        "latent_dim",
        "hidden_dim",
        "num_heads",
        "num_attention_layers",
        "aggregation_method",
        "classifier_type",
        "batch_size",
        "epochs",
        "lr",
        "lambda_attr",
        "early_stopping",
        "gradient_clip",
        "gradient_accumulation_steps",
        "class_weighting",
        "chunk_size",
        "chunk_overlap",
        "sex_map",
        "pc_map",
        "num_pcs",
    }
    allowed = required | {"cv_folds"}
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkManifestError(f"manifest.training has unknown keys: {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise BenchmarkManifestError(f"manifest.training missing required keys: {sorted(missing)}")

    mode = _required_choice(value["mode"], {"cv", "single_split"}, "manifest.training.mode")
    cv_folds = value.get("cv_folds")
    if mode == "cv":
        cv_folds = _required_positive_int(cv_folds, "manifest.training.cv_folds")
    elif cv_folds is not None:
        raise BenchmarkManifestError(
            "manifest.training.cv_folds must be null or absent for single_split"
        )

    validated = {
        "mode": mode,
        "cv_folds": cv_folds,
        "seed": _required_int(value["seed"], "manifest.training.seed"),
        "val_split": _required_float_range(
            value["val_split"], "manifest.training.val_split", low=0.0, high=1.0
        ),
        "split_plan": value["split_plan"],
        "split_plan_path": _resolve_existing_path(
            value["split_plan"], manifest_path, "manifest.training.split_plan"
        ),
        "latent_dim": _required_positive_int(value["latent_dim"], "manifest.training.latent_dim"),
        "hidden_dim": _required_positive_int(value["hidden_dim"], "manifest.training.hidden_dim"),
        "num_heads": _required_positive_int(value["num_heads"], "manifest.training.num_heads"),
        "num_attention_layers": _required_positive_int(
            value["num_attention_layers"],
            "manifest.training.num_attention_layers",
        ),
        "batch_size": _required_positive_int(value["batch_size"], "manifest.training.batch_size"),
        "epochs": _required_positive_int(value["epochs"], "manifest.training.epochs"),
        "lr": _required_positive_number(value["lr"], "manifest.training.lr"),
        "lambda_attr": _required_non_negative_number(
            value["lambda_attr"], "manifest.training.lambda_attr"
        ),
        "early_stopping": _required_positive_int(
            value["early_stopping"], "manifest.training.early_stopping"
        ),
        "gradient_accumulation_steps": _required_positive_int(
            value["gradient_accumulation_steps"],
            "manifest.training.gradient_accumulation_steps",
        ),
        "chunk_size": _required_positive_int(value["chunk_size"], "manifest.training.chunk_size"),
        "chunk_overlap": _required_non_negative_int(
            value["chunk_overlap"], "manifest.training.chunk_overlap"
        ),
        "num_pcs": _required_non_negative_int(value["num_pcs"], "manifest.training.num_pcs"),
    }
    for key, choices in TRAIN_CHOICES.items():
        validated[key] = _required_choice(value[key], choices, f"manifest.training.{key}")
    gradient_clip = value["gradient_clip"]
    validated["gradient_clip"] = (
        None
        if gradient_clip is None
        else _required_positive_number(gradient_clip, "manifest.training.gradient_clip")
    )
    validated["sex_map"] = value["sex_map"]
    validated["sex_map_path"] = (
        None
        if value["sex_map"] is None
        else _resolve_existing_path(value["sex_map"], manifest_path, "manifest.training.sex_map")
    )
    validated["pc_map"] = value["pc_map"]
    validated["pc_map_path"] = (
        None
        if value["pc_map"] is None
        else _resolve_existing_path(value["pc_map"], manifest_path, "manifest.training.pc_map")
    )
    if validated["num_pcs"] > 0 and validated["pc_map_path"] is None:
        raise BenchmarkManifestError(
            "manifest.training.num_pcs > 0 requires manifest.training.pc_map"
        )
    if validated["pc_map_path"] is not None and validated["num_pcs"] <= 0:
        raise BenchmarkManifestError(
            "manifest.training.pc_map requires manifest.training.num_pcs > 0"
        )
    return validated


def _validate_explanation(value: Any, *, training: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.explanation must be a mapping")
    required = {"ig_mode", "n_steps", "max_variants", "aggregation_method"}
    allowed = required | {"fold_index"}
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkManifestError(f"manifest.explanation has unknown keys: {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise BenchmarkManifestError(
            f"manifest.explanation missing required keys: {sorted(missing)}"
        )
    if value["ig_mode"] != "content":
        raise BenchmarkManifestError("manifest.explanation.ig_mode must be 'content'")
    fold_index = value.get("fold_index")
    if training["mode"] == "cv":
        fold_index = _required_non_negative_int(fold_index, "manifest.explanation.fold_index")
        if fold_index >= training["cv_folds"]:
            raise BenchmarkManifestError(
                "manifest.explanation.fold_index must be less than training.cv_folds"
            )
    elif fold_index is not None:
        raise BenchmarkManifestError(
            "manifest.explanation.fold_index must be null or absent for single_split"
        )
    return {
        "ig_mode": "content",
        "fold_index": fold_index,
        "n_steps": _required_positive_int(value["n_steps"], "manifest.explanation.n_steps"),
        "max_variants": _required_positive_int(
            value["max_variants"], "manifest.explanation.max_variants"
        ),
        "aggregation_method": _required_choice(
            value["aggregation_method"],
            EXPLAIN_AGGREGATION_CHOICES,
            "manifest.explanation.aggregation_method",
        ),
    }


def _validate_runtime(value: Any, *, paired: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError("manifest.runtime must be a mapping")
    allowed = {"python", "device", "train_num_workers", "explain_batch_size"}
    if paired:
        allowed = allowed | {"calibration_n_jobs"}
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkManifestError(f"manifest.runtime has unknown keys: {sorted(unknown)}")
    device = value.get("device", "cuda")
    if device not in {"cuda", "cpu"}:
        raise BenchmarkManifestError("manifest.runtime.device must be 'cuda' or 'cpu'")
    if paired:
        n_jobs = value.get("calibration_n_jobs", 1)
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise BenchmarkManifestError(
                "manifest.runtime.calibration_n_jobs must be a positive integer or -1"
            )
    return {
        "python": value.get("python"),
        "device": device,
        "train_num_workers": _required_non_negative_int(
            value.get("train_num_workers", 0),
            "manifest.runtime.train_num_workers",
        ),
        "explain_batch_size": _required_positive_int(
            value.get("explain_batch_size", 4),
            "manifest.runtime.explain_batch_size",
        ),
        **({"calibration_n_jobs": value.get("calibration_n_jobs", 1)} if paired else {}),
    }


def _validate_runs(value: Any, *, paired: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkManifestError("manifest.runs must be a list")
    if len(value) < 2:
        raise BenchmarkManifestError("manifest.runs must contain at least two runs")
    seen_ids = set()
    seen_position_intents = set()
    runs = []
    for index, raw_run in enumerate(value):
        path = f"manifest.runs[{index}]"
        if not isinstance(raw_run, Mapping):
            raise BenchmarkManifestError(f"{path} must be a mapping")
        strategy_null = sorted(key for key in (NULL_BASELINE_KEY, "null") if key in raw_run)
        if paired and strategy_null:
            raise BenchmarkManifestError(
                f"{path}.{strategy_null[0]} is not allowed: schema_version 2 uses exactly one "
                "benchmark-level manifest.null_baseline artifact shared by every strategy"
            )
        if set(raw_run) != {"run_id", "position"}:
            raise BenchmarkManifestError(f"{path} must contain exactly run_id and position")
        run_id = _validate_id(raw_run["run_id"], f"{path}.run_id")
        if run_id in seen_ids:
            raise BenchmarkManifestError(f"duplicate run_id: {run_id}")
        seen_ids.add(run_id)
        position = _validate_position_intent(raw_run["position"], f"{path}.position")
        position_key = json.dumps(position, sort_keys=True, separators=(",", ":"))
        if position_key in seen_position_intents:
            raise BenchmarkManifestError(
                f"{path}.position duplicates an existing explicit position intent"
            )
        seen_position_intents.add(position_key)
        runs.append({"run_id": run_id, "position": position})
    return runs


def _validate_position_intent(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestError(f"{path} must be a mapping")
    unknown = set(value) - POSITION_KEYS
    if unknown:
        raise BenchmarkManifestError(f"{path} has unknown keys: {sorted(unknown)}")
    if "position_preset" not in value:
        raise BenchmarkManifestError(f"{path}.position_preset is required")
    preset = _enum_value(value["position_preset"], PositionPreset, f"{path}.position_preset")
    if preset == PositionPreset.LEGACY.value:
        extras = sorted(key for key in value if key != "position_preset")
        if extras:
            raise BenchmarkManifestError(f"{path} legacy preset rejects custom fields: {extras}")
        return {"position_preset": preset}

    required_missing = CUSTOM_REQUIRED_KEYS - set(value)
    if required_missing:
        raise BenchmarkManifestError(
            f"{path} custom preset missing required keys: {sorted(required_missing)}"
        )
    position = {"position_preset": preset}
    position["absolute_position_encoding"] = _enum_value(
        value["absolute_position_encoding"],
        AbsolutePositionEncoding,
        f"{path}.absolute_position_encoding",
    )
    position["relative_position_encoding"] = _enum_value(
        value["relative_position_encoding"],
        RelativePositionEncoding,
        f"{path}.relative_position_encoding",
    )
    position["chromosome_encoding"] = _enum_value(
        value["chromosome_encoding"],
        ChromosomeEncoding,
        f"{path}.chromosome_encoding",
    )
    position["cross_chromosome_policy"] = _enum_value(
        value["cross_chromosome_policy"],
        CrossChromosomePolicy,
        f"{path}.cross_chromosome_policy",
    )

    _validate_absolute_position_fields(value, position, path)
    _validate_relative_position_fields(value, position, path)
    return position


def _validate_absolute_position_fields(
    raw: Mapping[str, Any], position: dict[str, Any], path: str
) -> None:
    absolute = position["absolute_position_encoding"]
    if absolute == AbsolutePositionEncoding.NONE.value:
        _reject_present(
            raw,
            [
                "position_dim",
                "sinusoidal_coordinate_scale",
                "sinusoidal_max_wavelength",
                "position_bin_size",
            ],
            path,
        )
    elif absolute == AbsolutePositionEncoding.SINUSOIDAL.value:
        position["position_dim"] = _required_positive_int(
            raw.get("position_dim"), f"{path}.position_dim"
        )
        position["sinusoidal_coordinate_scale"] = _required_positive_number(
            raw.get("sinusoidal_coordinate_scale"),
            f"{path}.sinusoidal_coordinate_scale",
        )
        position["sinusoidal_max_wavelength"] = _required_positive_number(
            raw.get("sinusoidal_max_wavelength"),
            f"{path}.sinusoidal_max_wavelength",
        )
        _reject_present(raw, ["position_bin_size"], path)
    elif absolute == AbsolutePositionEncoding.LEARNED_BINNED.value:
        position["position_dim"] = _required_positive_int(
            raw.get("position_dim"), f"{path}.position_dim"
        )
        position["position_bin_size"] = _required_positive_int(
            raw.get("position_bin_size"), f"{path}.position_bin_size"
        )
        _reject_present(raw, ["sinusoidal_coordinate_scale", "sinusoidal_max_wavelength"], path)


def _validate_relative_position_fields(
    raw: Mapping[str, Any], position: dict[str, Any], path: str
) -> None:
    relative = position["relative_position_encoding"]
    if relative == RelativePositionEncoding.NONE.value:
        _reject_present(
            raw,
            [
                "num_position_buckets",
                "max_position_distance",
                "rope_coordinate_scale",
                "rope_base",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            path,
        )
    elif relative == RelativePositionEncoding.T5_BUCKET.value:
        position["num_position_buckets"] = _required_positive_int(
            raw.get("num_position_buckets"),
            f"{path}.num_position_buckets",
        )
        position["max_position_distance"] = _required_positive_int(
            raw.get("max_position_distance"),
            f"{path}.max_position_distance",
        )
        _reject_present(
            raw,
            [
                "rope_coordinate_scale",
                "rope_base",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            path,
        )
    elif relative == RelativePositionEncoding.ROPE.value:
        position["rope_coordinate_scale"] = _required_positive_number(
            raw.get("rope_coordinate_scale"),
            f"{path}.rope_coordinate_scale",
        )
        position["rope_base"] = _required_positive_number(raw.get("rope_base"), f"{path}.rope_base")
        _reject_present(
            raw,
            [
                "num_position_buckets",
                "max_position_distance",
                "alibi_distance_function",
                "alibi_distance_scale",
            ],
            path,
        )
    elif relative in {
        RelativePositionEncoding.ALIBI_FIXED.value,
        RelativePositionEncoding.ALIBI_LEARNED.value,
    }:
        position["alibi_distance_function"] = _enum_value(
            raw.get("alibi_distance_function"),
            AlibiDistanceFunction,
            f"{path}.alibi_distance_function",
        )
        position["alibi_distance_scale"] = _required_positive_number(
            raw.get("alibi_distance_scale"),
            f"{path}.alibi_distance_scale",
        )
        _reject_present(
            raw,
            ["num_position_buckets", "max_position_distance", "rope_coordinate_scale", "rope_base"],
            path,
        )


def _load_structural_split_plan(
    split_plan_path: Path, *, training: Mapping[str, Any]
) -> dict[str, Any]:
    raw_plan = _load_yaml_mapping(split_plan_path, "split_plan")
    if not isinstance(raw_plan, Mapping):
        raise BenchmarkManifestError("split_plan must contain a YAML mapping")
    plan = dict(raw_plan)
    if plan.get("schema_version") != 1 or isinstance(plan.get("schema_version"), bool):
        raise BenchmarkManifestError("split_plan.schema_version must be 1")
    mode = plan.get("mode")
    if mode != training["mode"]:
        raise BenchmarkManifestError("split_plan.mode must match manifest.training.mode")
    n_samples = _required_positive_int(plan.get("n_samples"), "split_plan.n_samples")
    sample_hash = _required_sha256(plan.get("sample_ids_sha256"), "split_plan.sample_ids_sha256")
    seed = _required_int(plan.get("seed"), "split_plan.seed")
    split_source = _required_choice(
        plan.get("split_source"),
        {"generated", "replayed"},
        "split_plan.split_source",
    )
    if mode == "cv":
        n_folds = _required_positive_int(plan.get("n_folds"), "split_plan.n_folds")
        if n_folds != training["cv_folds"]:
            raise BenchmarkManifestError("split_plan.n_folds must match manifest.training.cv_folds")
        folds = _validate_cv_split_structure(
            plan.get("folds"),
            n_folds=n_folds,
            n_samples=n_samples,
        )
        normalized_plan = {
            "schema_version": 1,
            "mode": "cv",
            "n_samples": n_samples,
            "sample_ids_sha256": sample_hash,
            "seed": seed,
            "split_source": split_source,
            "n_folds": n_folds,
            "folds": folds,
        }
        normalized: dict[str, Any] = {
            "input_path": str(split_plan_path),
            "membership_sha256": split_plan_sha256(normalized_plan),
            "sample_ids_sha256": sample_hash,
            "mode": "cv",
            "n_samples": n_samples,
            "n_folds": n_folds,
            "folds": folds,
            "dataset_sample_binding_validation": DEFERRED_SAMPLE_BINDING,
        }
    else:
        train_indices = _validate_index_list(
            plan.get("train_indices"),
            "split_plan.train_indices",
            n_samples=n_samples,
        )
        val_indices = _validate_index_list(
            plan.get("val_indices"),
            "split_plan.val_indices",
            n_samples=n_samples,
        )
        _validate_train_val_partition(
            train_indices,
            val_indices,
            n_samples=n_samples,
            path="split_plan",
        )
        train_hash = _required_sha256(
            plan.get("train_sample_ids_sha256"),
            "split_plan.train_sample_ids_sha256",
        )
        val_hash = _required_sha256(
            plan.get("val_sample_ids_sha256"),
            "split_plan.val_sample_ids_sha256",
        )
        normalized_plan = {
            "schema_version": 1,
            "mode": "single_split",
            "n_samples": n_samples,
            "sample_ids_sha256": sample_hash,
            "seed": seed,
            "split_source": split_source,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_sample_ids_sha256": train_hash,
            "val_sample_ids_sha256": val_hash,
        }
        normalized = {
            "input_path": str(split_plan_path),
            "membership_sha256": split_plan_sha256(normalized_plan),
            "sample_ids_sha256": sample_hash,
            "mode": "single_split",
            "n_samples": n_samples,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_sample_ids_sha256": train_hash,
            "val_sample_ids_sha256": val_hash,
            "dataset_sample_binding_validation": DEFERRED_SAMPLE_BINDING,
        }
    return normalized


def _validate_cv_split_structure(
    value: Any,
    *,
    n_folds: int,
    n_samples: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkManifestError("split_plan.folds must be a list")
    if len(value) != n_folds:
        raise BenchmarkManifestError("split_plan.folds length must match split_plan.n_folds")
    seen = set()
    all_val_indices = []
    folds = []
    for pos, fold in enumerate(value):
        path = f"split_plan.folds[{pos}]"
        if not isinstance(fold, Mapping):
            raise BenchmarkManifestError(f"{path} must be a mapping")
        fold_index = _required_non_negative_int(fold.get("fold_index"), f"{path}.fold_index")
        if fold_index in seen:
            raise BenchmarkManifestError(f"duplicate split_plan fold_index: {fold_index}")
        seen.add(fold_index)
        train_indices = _validate_index_list(
            fold.get("train_indices"),
            f"{path}.train_indices",
            n_samples=n_samples,
        )
        val_indices = _validate_index_list(
            fold.get("val_indices"),
            f"{path}.val_indices",
            n_samples=n_samples,
        )
        _validate_train_val_partition(
            train_indices,
            val_indices,
            n_samples=n_samples,
            path=path,
        )
        all_val_indices.extend(val_indices)
        folds.append(
            {
                "fold_index": fold_index,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_sample_ids_sha256": _required_sha256(
                    fold.get("train_sample_ids_sha256"),
                    f"{path}.train_sample_ids_sha256",
                ),
                "val_sample_ids_sha256": _required_sha256(
                    fold.get("val_sample_ids_sha256"),
                    f"{path}.val_sample_ids_sha256",
                ),
            }
        )
    if seen != set(range(n_folds)):
        raise BenchmarkManifestError("split_plan fold indices must be exactly 0..n_folds-1")
    if sorted(all_val_indices) != list(range(n_samples)):
        raise BenchmarkManifestError(
            "split_plan CV validation indices must contain each sample exactly once"
        )
    return sorted(folds, key=lambda item: item["fold_index"])


def _build_runs(
    validated: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    repo_root: Path,
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    paired = validated["schema_version"] == PAIRED_SCHEMA_VERSION
    runs = []
    for run in validated["runs"]:
        run_root = (benchmark_root / "runs" / run["run_id"]).resolve(strict=False)
        directories = {
            "run_root": str(run_root),
            "training": str(run_root / "real" / "training"),
            "explanation": str(run_root / "real" / "explanation"),
        }
        train_argv = _build_train_argv(
            run,
            validated=validated,
            runtime=runtime,
            repo_root=repo_root,
            run_root=run_root,
        )
        explain_argv = _build_explain_argv(
            validated=validated,
            runtime=runtime,
            repo_root=repo_root,
            run_root=run_root,
        )
        mode = validated["training"]["mode"]
        fold_index = validated["explanation"]["fold_index"]
        planned = {
            "run_id": run["run_id"],
            "position_intent": run["position"],
            "canonical_position_strategy_identity": DEFERRED_STRATEGY_IDENTITY,
            "directories": directories,
            "train_argv": train_argv,
            "explain_argv": explain_argv,
            "expected_artifacts": _expected_artifacts(run_root, mode, fold_index),
        }
        if paired:
            directories["null_training"] = str(run_root / "null" / "training")
            directories["null_explanation"] = str(run_root / "null" / "explanation")
            directories["calibration"] = str(run_root / "calibration")
            planned["null_train_argv"] = _build_train_argv(
                run,
                validated=validated,
                runtime=runtime,
                repo_root=repo_root,
                run_root=run_root,
                side="null",
            )
            planned["null_explain_argv"] = _build_explain_argv(
                validated=validated,
                runtime=runtime,
                repo_root=repo_root,
                run_root=run_root,
                side="null",
            )
            planned["calibration"] = _build_calibration(
                validated=validated,
                runtime=runtime,
                repo_root=repo_root,
                run_root=run_root,
            )
            null_artifacts = _expected_artifacts(run_root, mode, fold_index, side="null")
            planned["expected_artifacts"]["null_training"] = null_artifacts["training"]
            planned["expected_artifacts"]["null_explanation"] = null_artifacts["explanation"]
            planned["expected_artifacts"]["calibration"] = planned["calibration"][
                "expected_outputs"
            ]
        runs.append(planned)
    return runs


def _side_dataset_path(validated: Mapping[str, Any], side: str) -> Path:
    """Return the dataset for *side*; the only scientific input that differs."""
    if side == "real":
        return validated["dataset"]["preprocessed_data_path"]
    if side == "null":
        return validated[NULL_BASELINE_KEY]["artifact_path"]
    raise BenchmarkManifestError(f"unknown benchmark side: {side!r}")


def _build_train_argv(
    run: Mapping[str, Any],
    *,
    validated: Mapping[str, Any],
    runtime: Mapping[str, Any],
    repo_root: Path,
    run_root: Path,
    side: str = "real",
) -> list[str]:
    """Build real or null training argv from one shared definition.

    Real and null differ only in ``--preprocessed-data`` and ``--output-dir``;
    split plan, training seed, positional flags, architecture, optimizer, and
    class weighting are emitted from the same manifest values for both sides.
    """
    training = validated["training"]
    argv = [
        runtime["python"],
        str(repo_root / "scripts" / "train.py"),
        "--preprocessed-data",
        str(_side_dataset_path(validated, side)),
        "--level",
        validated["annotation_level"],
        "--seed",
        str(training["seed"]),
        "--val-split",
        _scalar(training["val_split"]),
        "--split-plan",
        str(training["split_plan_path"]),
    ]
    if training["mode"] == "cv":
        argv.extend(["--cv", str(training["cv_folds"])])
    for key in TRAIN_COMMON_KEYS[2:]:
        argv.extend([f"--{key.replace('_', '-')}", _scalar(training[key])])
    if training["gradient_clip"] is not None:
        argv.extend(["--gradient-clip", _scalar(training["gradient_clip"])])
    if training["sex_map_path"] is not None:
        argv.extend(["--sex-map", str(training["sex_map_path"])])
    if training["pc_map_path"] is not None:
        argv.extend(
            ["--pc-map", str(training["pc_map_path"]), "--num-pcs", str(training["num_pcs"])]
        )
    else:
        argv.extend(["--num-pcs", str(training["num_pcs"])])
    argv.extend(
        [
            "--genome-build",
            validated["dataset"]["genome_build"],
            "--device",
            runtime["device"],
            "--num-workers",
            str(runtime["train_num_workers"]),
        ]
    )
    argv.extend(_position_to_train_argv(run["position"]))
    argv.extend(["--output-dir", str(run_root / side), "--experiment-name", "training"])
    return argv


def _build_explain_argv(
    *,
    validated: Mapping[str, Any],
    runtime: Mapping[str, Any],
    repo_root: Path,
    run_root: Path,
    side: str = "real",
) -> list[str]:
    """Build real or null explanation argv from one shared definition.

    Both sides always use ``--experiment-dir`` with an explicit ``--fold-index``
    in CV (never best-fold selection or ``--checkpoint``) and identical IG
    settings; the null side additionally declares ``--is-null-baseline``.
    """
    explanation = validated["explanation"]
    training = validated["training"]
    argv = [
        runtime["python"],
        str(repo_root / "scripts" / "explain.py"),
        "--experiment-dir",
        str(run_root / side / "training"),
    ]
    if training["mode"] == "cv":
        argv.extend(["--fold-index", str(explanation["fold_index"])])
    argv.extend(
        [
            "--preprocessed-data",
            str(_side_dataset_path(validated, side)),
            "--output-dir",
            str(run_root / side / "explanation"),
            "--genome-build",
            validated["dataset"]["genome_build"],
            "--device",
            runtime["device"],
            "--batch-size",
            str(runtime["explain_batch_size"]),
            "--ig-mode",
            "content",
            "--n-steps",
            str(explanation["n_steps"]),
            "--max-variants",
            str(explanation["max_variants"]),
            "--aggregation-method",
            explanation["aggregation_method"],
        ]
    )
    if training["pc_map_path"] is not None:
        argv.extend(
            ["--pc-map", str(training["pc_map_path"]), "--num-pcs", str(training["num_pcs"])]
        )
    if side == "null":
        argv.append("--is-null-baseline")
    return argv


def _build_calibration(
    *,
    validated: Mapping[str, Any],
    runtime: Mapping[str, Any],
    repo_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Plan (never execute) unchanged bootstrap calibration for one strategy pair.

    ``bootstrap_null_calibration.py`` resamples NULL SAMPLES with replacement
    from this strategy's single null explanation. It does not bootstrap
    phenotype permutations, variants, model initializations, or independently
    trained null models.
    """
    calibration = validated["calibration"]
    calibration_dir = run_root / "calibration"
    outputs = {
        "rankings": calibration_dir / CALIBRATION_RANKINGS_NAME,
        "gene_stats": calibration_dir / CALIBRATION_GENE_STATS_NAME,
        "summary": calibration_dir / CALIBRATION_SUMMARY_NAME,
    }
    argv = [
        runtime["python"],
        str(repo_root / "scripts" / "bootstrap_null_calibration.py"),
        "--real-rankings",
        str(run_root / "real" / "explanation" / "sieve_variant_rankings.csv"),
        "--null-attributions",
        str(run_root / "null" / "explanation" / "attributions.npz"),
        "--output",
        str(outputs["rankings"]),
        "--output-gene-stats",
        str(outputs["gene_stats"]),
        "--output-summary",
        str(outputs["summary"]),
        "--n-bootstrap",
        str(calibration["n_bootstrap"]),
        "--seed",
        str(calibration["seed"]),
        "--top-k",
        ",".join(str(value) for value in calibration["top_k"]),
        "--min-variants-per-gene",
        str(calibration["min_variants_per_gene"]),
        "--gene-delta-rank-aggregation",
        calibration["gene_delta_rank_aggregation"],
        "--genome-build",
        validated["dataset"]["genome_build"],
        "--n-jobs",
        str(runtime["calibration_n_jobs"]),
    ]
    if calibration["exclude_sex_chroms"]:
        argv.append("--exclude-sex-chroms")
    return {
        "directory": str(calibration_dir),
        "status": "planned_not_executed",
        "requires": "passing paired compatibility report and Phase 12C3B2 preflight",
        "paired_compatibility": str(calibration_dir / PAIRED_COMPATIBILITY_NAME),
        "argv": argv,
        "expected_outputs": [str(path) for path in outputs.values()],
    }


def _plan_calibration_settings(calibration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "script": "scripts/bootstrap_null_calibration.py",
        "n_bootstrap": calibration["n_bootstrap"],
        "seed": calibration["seed"],
        "top_k": list(calibration["top_k"]),
        "exclude_sex_chroms": calibration["exclude_sex_chroms"],
        "min_variants_per_gene": calibration["min_variants_per_gene"],
        "gene_delta_rank_aggregation": calibration["gene_delta_rank_aggregation"],
        "resampling_unit": "null_samples_with_replacement",
        "not_bootstrapped": [
            "phenotype_permutations",
            "variants",
            "model_initializations",
            "independently_trained_null_models",
        ],
        "rank_ties": "rankdata(method='average') for rank_real and null bootstrap ranks",
        "delta_rank": "median_rank_null_boot - rank_real",
    }


def _paired_policy(validated: Mapping[str, Any]) -> dict[str, Any]:
    training = validated["training"]
    return {
        "shared_null_artifacts_per_benchmark": 1,
        "null_models_per_strategy": 1,
        "split_replay": "same --split-plan for real and null",
        "training_seed": training["seed"],
        "training_seed_shared_by_real_and_null": True,
        "permutation_seed_role": "provenance_only_permutation_vector_is_authoritative",
        "class_weighting": training["class_weighting"],
        "explanation_fold_index": validated["explanation"]["fold_index"],
        "checkpoint_selection": (
            "cv_explicit_fold" if training["mode"] == "cv" else "single_run_best_model"
        ),
        "raw_comparisons": "real_only",
    }


def _build_null_binding(
    validated: Mapping[str, Any], split_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the one benchmark null artifact to the real dataset and split plan.

    Lightweight by design: hashes raw file bytes and parses the 12C3A sidecar,
    but never unpickles either cohort. Passing this check does NOT authorize
    execution; full ``validate_null_pair`` runs in the Phase 12C3B2 preflight.
    """
    from src.data import null_lineage

    dataset_path = validated["dataset"]["preprocessed_data_path"]
    null_path = validated[NULL_BASELINE_KEY]["artifact_path"]
    if null_path == dataset_path:
        raise BenchmarkManifestError(
            "manifest.null_baseline.artifact must differ from dataset.preprocessed_data"
        )
    sidecar_path = null_lineage.sidecar_path_for(null_path)
    if not sidecar_path.is_file():
        raise BenchmarkManifestError(
            f"null lineage sidecar not found at its deterministic path: {sidecar_path}"
        )
    sidecar = _load_yaml_mapping(sidecar_path, "null lineage sidecar")
    try:
        null_lineage.validate_sidecar_schema(sidecar)
    except ValueError as error:
        raise BenchmarkManifestError(f"null lineage sidecar schema is invalid: {error}") from error

    real_sha = _sha256_file(dataset_path)
    null_sha = _sha256_file(null_path)
    if real_sha == null_sha:
        raise BenchmarkManifestError("null artifact bytes are identical to the real dataset bytes")
    if sidecar["source"]["sha256"] != real_sha:
        raise BenchmarkManifestError(
            "null lineage sidecar source.sha256 does not match dataset.preprocessed_data bytes"
        )
    if sidecar["null"]["sha256"] != null_sha:
        raise BenchmarkManifestError(
            "null lineage sidecar null.sha256 does not match manifest.null_baseline.artifact bytes"
        )
    if sidecar["samples"]["sample_ids_sha256"] != split_plan["sample_ids_sha256"]:
        raise BenchmarkManifestError(
            "null lineage sidecar samples.sample_ids_sha256 does not match "
            "split_plan.sample_ids_sha256"
        )
    if sidecar["samples"]["n_samples"] != split_plan["n_samples"]:
        raise BenchmarkManifestError(
            "null lineage sidecar samples.n_samples does not match split_plan.n_samples"
        )
    return {
        "schema_version": 1,
        "lineage_sha256": sidecar["lineage_sha256"],
        "source_artifact_sha256": real_sha,
        "null_artifact_sha256": null_sha,
        "sample_ids_sha256": sidecar["samples"]["sample_ids_sha256"],
        "n_samples": sidecar["samples"]["n_samples"],
        "null_artifact_path": str(null_path),
        "sidecar_path": str(sidecar_path),
        "validation_level": NULL_BINDING_VALIDATION_LEVEL,
        "full_pair_validation": NULL_FULL_PAIR_VALIDATION,
        "execution_authorized": False,
    }


def _position_to_train_argv(position: Mapping[str, Any]) -> list[str]:
    argv = ["--position-preset", position["position_preset"]]
    ordered_keys = [
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
    for key in ordered_keys:
        if key in position:
            argv.extend([f"--{key.replace('_', '-')}", _scalar(position[key])])
    return argv


def _build_comparisons(
    runs: Sequence[Mapping[str, Any]],
    *,
    runtime: Mapping[str, Any],
    repo_root: Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    performance_dir = benchmark_root / "comparisons" / "performance"
    rankings_dir = benchmark_root / "comparisons" / "raw_rankings"
    attributions_dir = benchmark_root / "comparisons" / "raw_attributions"
    performance_argv = [
        runtime["python"],
        str(repo_root / "scripts" / "ablation_compare.py"),
        "--comparison-axis",
        "position",
    ]
    ranking_argv = [
        runtime["python"],
        str(repo_root / "scripts" / "compare_ablation_rankings.py"),
        "--comparison-axis",
        "position",
    ]
    attribution_argv = [
        runtime["python"],
        str(repo_root / "scripts" / "compare_position_attributions.py"),
    ]
    for run in runs:
        training_dir = Path(run["directories"]["training"])
        explanation_dir = Path(run["directories"]["explanation"])
        performance_argv.extend(["--run-dir", str(training_dir)])
        ranking_argv.extend(
            [
                "--position-run",
                run["run_id"],
                str(training_dir / "config.yaml"),
                str(explanation_dir / "sieve_variant_rankings.csv"),
                str(explanation_dir / "analysis_metadata.yaml"),
            ]
        )
        attribution_argv.extend(
            [
                "--position-run",
                run["run_id"],
                str(training_dir / "config.yaml"),
                str(explanation_dir / "analysis_metadata.yaml"),
                str(explanation_dir / "attributions.npz"),
                str(explanation_dir / "attributions_per_sample"),
            ]
        )
    performance_argv.extend(
        [
            "--out-summary-tsv",
            str(performance_dir / "position_performance_summary.tsv"),
            "--out-summary-yaml",
            str(performance_dir / "position_performance_summary.yaml"),
        ]
    )
    ranking_argv.extend(
        [
            "--score-column",
            POSITION_SCORE_COLUMN,
            "--out-comparison",
            str(rankings_dir / "position_ranking_comparison.yaml"),
            "--out-jaccard",
            str(rankings_dir / "position_ranking_jaccard.tsv"),
            "--out-level-specific",
            str(rankings_dir / "position_strategy_specific_variants.tsv"),
        ]
    )
    attribution_argv.extend(
        [
            "--out-summary-tsv",
            str(attributions_dir / "position_attribution_summary.tsv"),
            "--out-sample-tsv",
            str(attributions_dir / "position_attribution_per_sample.tsv"),
            "--out-feature-tsv",
            str(attributions_dir / "position_attribution_features.tsv"),
            "--out-comparison-yaml",
            str(attributions_dir / "position_attribution_comparison.yaml"),
        ]
    )
    return {
        "performance": {
            "directory": str(performance_dir),
            "argv": performance_argv,
            "expected_outputs": [
                str(performance_dir / "position_performance_summary.tsv"),
                str(performance_dir / "position_performance_summary.yaml"),
            ],
        },
        "raw_rankings": {
            "directory": str(rankings_dir),
            "score_column": POSITION_SCORE_COLUMN,
            "argv": ranking_argv,
            "expected_outputs": [
                str(rankings_dir / "position_ranking_comparison.yaml"),
                str(rankings_dir / "position_ranking_jaccard.tsv"),
                str(rankings_dir / "position_strategy_specific_variants.tsv"),
            ],
        },
        "raw_attributions": {
            "directory": str(attributions_dir),
            "argv": attribution_argv,
            "expected_outputs": [
                str(attributions_dir / "position_attribution_summary.tsv"),
                str(attributions_dir / "position_attribution_per_sample.tsv"),
                str(attributions_dir / "position_attribution_features.tsv"),
                str(attributions_dir / "position_attribution_comparison.yaml"),
            ],
        },
    }


def _expected_artifacts(
    run_root: Path, mode: str, fold_index: int | None, *, side: str = "real"
) -> dict[str, Any]:
    training_dir = run_root / side / "training"
    explanation_dir = run_root / side / "explanation"
    model_artifact = (
        training_dir / f"fold_{fold_index}" / "best_model.pt"
        if mode == "cv"
        else training_dir / "best_model.pt"
    )
    return {
        "status": "expected_not_yet_validated",
        "training": [
            str(training_dir / "config.yaml"),
            str(training_dir / "split_plan.yaml"),
            str(training_dir / ("cv_results.yaml" if mode == "cv" else "results.yaml")),
            str(model_artifact),
        ],
        "explanation": [
            str(explanation_dir / "analysis_metadata.yaml"),
            str(explanation_dir / "attributions.npz"),
            str(explanation_dir / "attributions_per_sample"),
            str(explanation_dir / "sieve_variant_rankings.csv"),
            str(explanation_dir / "sieve_gene_rankings.csv"),
        ],
    }


def _inspect_existing_outputs(
    runs: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    *,
    allow_existing_outputs: bool,
) -> list[str]:
    warnings = []
    leaf_paths = []
    for run in runs:
        leaf_paths.extend(Path(run["directories"][name]) for name in _run_leaf_names(run))
    leaf_paths.extend(Path(comparison["directory"]) for comparison in comparisons.values())
    for path in leaf_paths:
        if path.is_file():
            raise BenchmarkManifestError(f"planned output directory is an existing file: {path}")
        if path.is_dir() and any(path.iterdir()):
            message = f"planned output directory is non-empty: {path}"
            if allow_existing_outputs:
                warnings.append(message)
            else:
                raise BenchmarkManifestError(message)
    return warnings


def _run_leaf_names(run: Mapping[str, Any]) -> tuple[str, ...]:
    return PAIRED_RUN_LEAF_NAMES if "null_training" in run["directories"] else RUN_LEAF_NAMES


def _validate_output_collisions(
    runs: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    *,
    dataset_path: Path,
    split_plan_path: Path,
    sex_map_path: Path | None,
    pc_map_path: Path | None,
    null_input_paths: Sequence[tuple[Path, str]] | None = None,
) -> None:
    run_leafs = []
    for run in runs:
        training = Path(run["directories"]["training"])
        explanation = Path(run["directories"]["explanation"])
        if training == explanation:
            raise BenchmarkManifestError(
                f"training and explanation paths collide for {run['run_id']}"
            )
        run_leafs.extend([training, explanation])
        run_leafs.extend(
            Path(run["directories"][name])
            for name in _run_leaf_names(run)
            if name not in RUN_LEAF_NAMES
        )
    comparison_leafs = [Path(comparison["directory"]) for comparison in comparisons.values()]
    _reject_duplicate_paths(run_leafs, "run output")
    _reject_duplicate_paths(comparison_leafs, "comparison output")
    sources = [
        (dataset_path, "dataset"),
        (split_plan_path, "split_plan"),
    ]
    if sex_map_path is not None:
        sources.append((sex_map_path, "sex_map"))
    if pc_map_path is not None:
        sources.append((pc_map_path, "pc_map"))
    for source_path, name in sources:
        if source_path in run_leafs or source_path in comparison_leafs:
            raise BenchmarkManifestError(
                f"{name} path collides with a planned output path: {source_path}"
            )
    for path in comparison_leafs:
        if path in run_leafs:
            raise BenchmarkManifestError(f"comparison output collides with run output: {path}")
    for source_path, name in null_input_paths or ():
        for leaf in [*run_leafs, *comparison_leafs]:
            if source_path == leaf or _is_nested(source_path, leaf):
                raise BenchmarkManifestError(
                    f"{name} path collides with a planned output path: {source_path}"
                )
    for index, left in enumerate(run_leafs):
        for right in run_leafs[index + 1 :]:
            if _is_nested(left, right) or _is_nested(right, left):
                raise BenchmarkManifestError(
                    f"planned run output paths are unexpectedly nested: {left} and {right}"
                )


def _resolve_runtime(
    runtime: Mapping[str, Any],
    *,
    manifest_path: Path,
    python_override: str | None,
    device_override: str | None,
) -> dict[str, Any]:
    if python_override is not None:
        python_value = python_override
        python_source = "cli"
    elif runtime.get("python") is not None:
        python_value = runtime["python"]
        python_source = "manifest"
    else:
        python_value = sys.executable
        python_source = "default"
    device = device_override or runtime.get("device", "cuda")
    if device not in {"cuda", "cpu"}:
        raise BenchmarkManifestError("--device must be 'cuda' or 'cpu'")
    return {
        "python": str(
            _resolve_python(
                python_value,
                manifest_path=manifest_path,
                source=python_source,
            )
        ),
        "device": device,
        "train_num_workers": runtime.get("train_num_workers", 0),
        "explain_batch_size": runtime.get("explain_batch_size", 4),
        **(
            {"calibration_n_jobs": runtime["calibration_n_jobs"]}
            if "calibration_n_jobs" in runtime
            else {}
        ),
    }


def _resolve_python(value: Any, *, manifest_path: Path, source: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkManifestError("runtime.python must be a non-empty string")
    if any(separator in value for separator in ("/", "\\")):
        path = Path(value).expanduser()
        if not path.is_absolute() and source == "manifest":
            path = manifest_path.parent / path
        elif not path.is_absolute():
            path = path.resolve(strict=False)
        if not path.is_file():
            raise BenchmarkManifestError(f"Python executable does not exist: {path}")
        return path.resolve(strict=True)
    resolved = shutil.which(value)
    if resolved is None:
        raise BenchmarkManifestError(f"Python executable could not be resolved: {value}")
    path = Path(resolved).resolve(strict=True)
    if not path.is_file():
        raise BenchmarkManifestError(f"Python executable is not a file: {path}")
    return path


def _read_repository_revision(repo_root: Path) -> str:
    """Read the Git revision from files only; never spawn git in dry-run."""
    git_dir = repo_root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if _is_sha256ish(head, length=40) else "unknown"
        ref = head.removeprefix("ref: ").strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            value = ref_path.read_text(encoding="utf-8").strip()
            return value if _is_sha256ish(value, length=40) else "unknown"
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                revision, packed_ref = line.split(" ", 1)
                if packed_ref == ref and _is_sha256ish(revision, length=40):
                    return revision
    except OSError:
        return "unknown"
    return "unknown"


def _resolve_existing_path(value: Any, manifest_path: Path, field_path: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkManifestError(f"{field_path} must be a non-empty path string")
    path = _resolve_manifest_path(value, manifest_path, strict=False)
    if not path.is_file():
        raise BenchmarkManifestError(f"{field_path} must be an existing file: {path}")
    return path.resolve(strict=True)


def _resolve_output_path(value: str, manifest_path: Path) -> Path:
    return _resolve_manifest_path(value, manifest_path, strict=False)


def _resolve_manifest_path(value: str, manifest_path: Path, *, strict: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=strict)


def _validate_output_root(output_root: Path, validated: Mapping[str, Any]) -> None:
    if output_root.is_file():
        raise BenchmarkManifestError(
            f"manifest.paths.output_root is an existing file: {output_root}"
        )
    sources = [
        (validated["dataset"]["preprocessed_data_path"], "dataset.preprocessed_data"),
        (validated["training"]["split_plan_path"], "training.split_plan"),
    ]
    if validated["training"]["sex_map_path"] is not None:
        sources.append((validated["training"]["sex_map_path"], "training.sex_map"))
    if validated["training"]["pc_map_path"] is not None:
        sources.append((validated["training"]["pc_map_path"], "training.pc_map"))
    if NULL_BASELINE_KEY in validated:
        from src.data.null_lineage import sidecar_path_for

        null_path = validated[NULL_BASELINE_KEY]["artifact_path"]
        sources.append((null_path, "null_baseline.artifact"))
        sources.append((sidecar_path_for(null_path), "null lineage sidecar"))
    for source_path, source_name in sources:
        if output_root == source_path:
            raise BenchmarkManifestError(
                f"manifest.paths.output_root must not equal manifest.{source_name}"
            )


def _reject_present(raw: Mapping[str, Any], keys: Sequence[str], path: str) -> None:
    present = [key for key in keys if key in raw]
    if present:
        raise BenchmarkManifestError(f"{path} rejects irrelevant position fields: {present}")


def _required_choice(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise BenchmarkManifestError(f"{path} must be one of {sorted(choices)}")
    return value


def _enum_value(value: Any, enum_type: type, path: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkManifestError(f"{path} must be a string")
    choices = {item.value for item in enum_type}
    if value not in choices:
        raise BenchmarkManifestError(f"{path} must be one of {sorted(choices)}")
    return value


def _validate_id(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkManifestError(f"{path} must be a string")
    if value in {".", ".."} or not RUN_ID_RE.fullmatch(value):
        raise BenchmarkManifestError(f"{path} contains an unsafe identifier: {value!r}")
    if any(char.isspace() for char in value) or "/" in value or "\\" in value:
        raise BenchmarkManifestError(f"{path} contains an unsafe identifier: {value!r}")
    return value


def _required_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkManifestError(f"{path} must be an integer")
    return value


def _required_positive_int(value: Any, path: str) -> int:
    value = _required_int(value, path)
    if value <= 0:
        raise BenchmarkManifestError(f"{path} must be positive")
    return value


def _required_non_negative_int(value: Any, path: str) -> int:
    value = _required_int(value, path)
    if value < 0:
        raise BenchmarkManifestError(f"{path} must be non-negative")
    return value


def _required_positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkManifestError(f"{path} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkManifestError(f"{path} must be positive")
    return value


def _required_non_negative_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkManifestError(f"{path} must be a non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise BenchmarkManifestError(f"{path} must be non-negative")
    return value


def _required_float_range(value: Any, path: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkManifestError(f"{path} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low < value < high:
        raise BenchmarkManifestError(f"{path} must be > {low} and < {high}")
    return value


def _required_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BenchmarkManifestError(f"{path} must be a lowercase SHA256 string")
    return value


def _validate_index_list(value: Any, path: str, *, n_samples: int | None = None) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkManifestError(f"{path} must be a list of non-negative integers")
    indices = [
        _required_non_negative_int(item, f"{path}[{index}]") for index, item in enumerate(value)
    ]
    if len(indices) != len(set(indices)):
        raise BenchmarkManifestError(f"{path} must not contain duplicate indices")
    if n_samples is not None:
        out_of_range = [index for index in indices if index >= n_samples]
        if out_of_range:
            raise BenchmarkManifestError(f"{path} contains an out-of-range sample index")
    return indices


def _validate_train_val_partition(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    *,
    n_samples: int,
    path: str,
) -> None:
    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        raise BenchmarkManifestError(f"{path} train/validation indices must be disjoint")
    if train_set | val_set != set(range(n_samples)):
        raise BenchmarkManifestError(
            f"{path} train/validation indices must cover exactly range(n_samples)"
        )


def _resolve_input_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise BenchmarkManifestError(f"{label} file cannot be read: {path}") from exc
    if not resolved.is_file():
        raise BenchmarkManifestError(f"{label} path must be an existing file: {resolved}")
    return resolved


def _load_yaml_mapping(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except OSError as exc:
        raise BenchmarkManifestError(f"{label} file cannot be read: {path}") from exc
    except yaml.YAMLError as exc:
        raise BenchmarkManifestError(f"{label} YAML is malformed: {path}") from exc


def _reject_duplicate_paths(paths: Sequence[Path], label: str) -> None:
    seen = set()
    for path in paths:
        if path in seen:
            raise BenchmarkManifestError(f"duplicate {label} path: {path}")
        seen.add(path)


def _is_nested(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return child != parent


def _is_sha256ish(value: str, *, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any) -> str:
    return str(value)


def _display_argv(argv: Sequence[str]) -> str:
    import shlex

    return shlex.join(argv)
