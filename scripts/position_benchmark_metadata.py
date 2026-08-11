"""Position-benchmark metadata helpers for downstream comparison scripts.

This module reads already-saved training configuration dictionaries. It does
not load checkpoints, construct models, import torch, or infer positional
strategy from paths. The helpers keep two concepts deliberately separate:

* position strategy identity: the configured positional architecture only;
* comparison context: the non-positional training and dataset conditions that
  must match before predictive metrics can be interpreted as a positional
  benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

STRATEGY_SCHEMA_VERSION = 1
POSITION_METADATA_SOURCE = "resolved_position_encoding_applied_to_model"


@dataclass(frozen=True)
class PositionStrategyIdentity:
    """Stable identity for a positional architecture."""

    name: str
    hash: str
    strategy_id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class ComparisonContext:
    """Predictive comparison context for one run."""

    run_id: str
    fields: dict[str, object]


@dataclass(frozen=True)
class CompatibilityMismatch:
    """One context field whose values differ across compared runs."""

    field: str
    values_by_run: dict[str, object]


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of pairwise or multi-run predictive context validation."""

    compatible: bool
    compared_fields: list[str]
    mismatches: list[CompatibilityMismatch]

    def to_dict(self) -> dict[str, object]:
        """Return a YAML-friendly representation."""
        return {
            "compatible": self.compatible,
            "compared_fields": self.compared_fields,
            "mismatches": [
                {
                    "field": mismatch.field,
                    "values_by_run": mismatch.values_by_run,
                }
                for mismatch in self.mismatches
            ],
        }


ABSOLUTE_STRATEGY_FIELDS = {
    "none": ("type",),
    "sinusoidal": ("type", "dim", "coordinate_scale", "max_wavelength"),
    "learned_binned": ("type", "dim", "bin_size_bp"),
}

RELATIVE_STRATEGY_FIELDS = {
    "none": ("type",),
    "t5_bucket": ("type", "num_buckets", "max_distance_bp"),
    "rope": ("type", "rope_coordinate_scale", "rope_base"),
    "alibi_fixed": ("type", "alibi_distance_function", "alibi_distance_scale"),
    "alibi_learned": ("type", "alibi_distance_function", "alibi_distance_scale"),
}

CHROMOSOME_STRATEGY_FIELDS = ("encoding", "cross_chromosome_policy")

VALID_PRESETS = {"legacy", "custom"}
VALID_CHROMOSOME_ENCODINGS = {"none", "learned"}
VALID_CROSS_CHROMOSOME_POLICIES = {"separate", "mask"}
VALID_ALIBI_DISTANCE_FUNCTIONS = {"linear", "log1p"}

REQUIRED_CONTEXT_FIELDS = (
    "level",
    "content_dim",
    "num_genes",
    "num_chromosomes",
    "dataset_identity.genome_build",
    "dataset_identity.gene_mapping_sha256",
    "dataset_identity.chromosome_mapping_sha256",
    "seed",
    "position_encoding_execution.training_mode",
    "cv",
    "val_split",
    "latent_dim",
    "hidden_dim",
    "num_heads",
    "num_attention_layers",
    "aggregation_method",
    "classifier_type",
    "num_covariates",
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
    "preprocessed_data",
    "vcf",
    "phenotypes",
    "sex_map",
    "pc_map",
    "pc_map_sha256",
    "num_pcs",
)


def require_authoritative_position_metadata(config: Mapping[str, object]) -> None:
    """Require normalized position metadata for position-benchmark mode."""
    position_encoding = config.get("position_encoding")
    if not isinstance(position_encoding, Mapping):
        raise ValueError("position benchmark mode requires config['position_encoding']")

    execution = config.get("position_encoding_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("position benchmark mode requires config['position_encoding_execution']")
    if execution.get("resolved_config_applied_to_model") is not True:
        raise ValueError(
            "position_encoding_execution.resolved_config_applied_to_model must be True"
        )
    if execution.get("source") != POSITION_METADATA_SOURCE:
        raise ValueError(
            "position_encoding_execution.source must be " f"{POSITION_METADATA_SOURCE!r}"
        )
    if not isinstance(execution.get("schema_version"), int) or isinstance(
        execution.get("schema_version"), bool
    ):
        raise ValueError("position_encoding_execution.schema_version must be an integer")


def canonical_position_strategy_payload(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical positional-architecture payload for *config*."""
    require_authoritative_position_metadata(config)
    position_encoding = _mapping(config, "position_encoding")
    absolute = _mapping(position_encoding, "absolute", path="position_encoding")
    relative = _mapping(position_encoding, "relative", path="position_encoding")
    chromosome = _mapping(position_encoding, "chromosome", path="position_encoding")

    preset = _required_string(position_encoding, "preset", path="position_encoding")
    _validate_string_choice(
        preset,
        VALID_PRESETS,
        "position_encoding.preset",
    )

    absolute_type = _required_string(
        absolute,
        "type",
        path="position_encoding.absolute",
    )
    relative_type = _required_string(
        relative,
        "type",
        path="position_encoding.relative",
    )
    _validate_known_strategy(
        absolute_type,
        ABSOLUTE_STRATEGY_FIELDS,
        "position_encoding.absolute.type",
    )
    _validate_known_strategy(
        relative_type,
        RELATIVE_STRATEGY_FIELDS,
        "position_encoding.relative.type",
    )

    absolute_payload = _whitelisted_section(
        absolute,
        ABSOLUTE_STRATEGY_FIELDS[absolute_type],
        "position_encoding.absolute",
    )
    relative_payload = _whitelisted_section(
        relative,
        RELATIVE_STRATEGY_FIELDS[relative_type],
        "position_encoding.relative",
    )
    chromosome_payload = _whitelisted_section(
        chromosome,
        CHROMOSOME_STRATEGY_FIELDS,
        "position_encoding.chromosome",
    )
    _validate_absolute_payload(absolute_payload, absolute_type)
    _validate_relative_payload(relative_payload, relative_type)
    _validate_chromosome_payload(chromosome_payload)

    return {
        "strategy_schema_version": STRATEGY_SCHEMA_VERSION,
        "preset": preset,
        "absolute": absolute_payload,
        "relative": relative_payload,
        "chromosome": chromosome_payload,
    }


def canonical_strategy_json(payload: Mapping[str, object]) -> str:
    """Serialize a strategy payload into deterministic canonical JSON."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def position_strategy_hash(payload: Mapping[str, object]) -> str:
    """Return the full SHA-256 hash for a canonical strategy payload."""
    encoded = canonical_strategy_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def position_strategy_name(payload: Mapping[str, object]) -> str:
    """Return a readable strategy name from strategy types only."""
    absolute = _mapping(payload, "absolute")
    relative = _mapping(payload, "relative")
    chromosome = _mapping(payload, "chromosome")
    preset = _required_string(payload, "preset")
    abs_type = _required_string(absolute, "type", path="absolute")
    rel_type = _required_string(relative, "type", path="relative")
    chrom_encoding = _required_string(chromosome, "encoding", path="chromosome")
    cross_policy = _required_string(
        chromosome,
        "cross_chromosome_policy",
        path="chromosome",
    )
    return (
        f"preset-{preset}__abs-{abs_type}__rel-{rel_type}__"
        f"chr-{chrom_encoding}__cross-{cross_policy}"
    )


def position_strategy_identity(config: Mapping[str, object]) -> PositionStrategyIdentity:
    """Build the full position strategy identity for a saved config."""
    payload = canonical_position_strategy_payload(config)
    full_hash = position_strategy_hash(payload)
    name = position_strategy_name(payload)
    return PositionStrategyIdentity(
        name=name,
        hash=full_hash,
        strategy_id=f"{name}__{full_hash[:12]}",
        payload=payload,
    )


def extract_comparison_context(
    config: Mapping[str, object],
    *,
    run_id: str,
) -> ComparisonContext:
    """Extract required predictive comparison fields from a saved config."""
    fields: dict[str, object] = {}
    missing_fields = []
    for field in REQUIRED_CONTEXT_FIELDS:
        present, value = _lookup_dotted(config, field)
        if not present:
            missing_fields.append(field)
        else:
            fields[field] = value
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"run {run_id!r} is missing required context fields: {missing}")
    return ComparisonContext(run_id=run_id, fields=fields)


def compare_contexts(contexts: Sequence[ComparisonContext]) -> CompatibilityReport:
    """Compare predictive contexts and return all field mismatches."""
    if not contexts:
        raise ValueError("at least one comparison context is required")

    _validate_unique_run_ids(contexts)
    compared_fields = list(REQUIRED_CONTEXT_FIELDS)
    mismatches: list[CompatibilityMismatch] = []
    for field in compared_fields:
        values_by_run = {context.run_id: context.fields[field] for context in contexts}
        if len({_canonical_compare_value(value) for value in values_by_run.values()}) != 1:
            mismatches.append(
                CompatibilityMismatch(
                    field=field,
                    values_by_run=values_by_run,
                )
            )

    return CompatibilityReport(
        compatible=not mismatches,
        compared_fields=compared_fields,
        mismatches=mismatches,
    )


def require_compatible_contexts(
    contexts: Sequence[ComparisonContext],
) -> CompatibilityReport:
    """Return compatibility report or raise with a mismatch summary."""
    report = compare_contexts(contexts)
    if report.compatible:
        return report

    details = "; ".join(
        f"{mismatch.field}: {mismatch.values_by_run}" for mismatch in report.mismatches
    )
    raise ValueError(f"position comparison context mismatch: {details}")


def _mapping(
    data: Mapping[str, object],
    key: str,
    *,
    path: str = "",
) -> Mapping[str, object]:
    value = data.get(key)
    dotted = f"{path}.{key}" if path else key
    if not isinstance(value, Mapping):
        raise ValueError(f"{dotted} must be a mapping")
    return value


def _required_string(
    data: Mapping[str, object],
    key: str,
    *,
    path: str = "",
) -> str:
    value = data.get(key)
    dotted = f"{path}.{key}" if path else key
    if not isinstance(value, str):
        raise ValueError(f"{dotted} must be a string")
    return value


def _validate_known_strategy(
    strategy_type: str,
    strategy_fields: Mapping[str, tuple[str, ...]],
    path: str,
) -> None:
    if strategy_type not in strategy_fields:
        allowed = ", ".join(sorted(strategy_fields))
        raise ValueError(f"{path} must be one of: {allowed}")


def _whitelisted_section(
    section: Mapping[str, object],
    fields: Sequence[str],
    path: str,
) -> dict[str, object]:
    canonical = {}
    for field in fields:
        if field not in section:
            raise ValueError(f"{path}.{field} is required for strategy identity")
        canonical[field] = section[field]
    return canonical


def _lookup_dotted(config: Mapping[str, object], dotted_field: str) -> tuple[bool, object]:
    current: object = config
    for part in dotted_field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _canonical_compare_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def _validate_unique_run_ids(contexts: Sequence[ComparisonContext]) -> None:
    seen = set()
    duplicates = set()
    for context in contexts:
        if context.run_id in seen:
            duplicates.add(context.run_id)
        seen.add(context.run_id)
    if duplicates:
        duplicate_list = ", ".join(repr(run_id) for run_id in sorted(duplicates))
        raise ValueError(f"duplicate comparison run IDs are not allowed: {duplicate_list}")


def _validate_string_choice(value: object, choices: set[str], path: str) -> None:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{path} must be one of: {allowed}")


def _validate_positive_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")


def _validate_positive_finite_number(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} must be a positive finite number")


def _validate_absolute_payload(
    payload: Mapping[str, object],
    absolute_type: str,
) -> None:
    if absolute_type == "sinusoidal":
        _validate_positive_int(payload["dim"], "position_encoding.absolute.dim")
        _validate_positive_finite_number(
            payload["coordinate_scale"],
            "position_encoding.absolute.coordinate_scale",
        )
        _validate_positive_finite_number(
            payload["max_wavelength"],
            "position_encoding.absolute.max_wavelength",
        )
    elif absolute_type == "learned_binned":
        _validate_positive_int(payload["dim"], "position_encoding.absolute.dim")
        _validate_positive_int(
            payload["bin_size_bp"],
            "position_encoding.absolute.bin_size_bp",
        )


def _validate_relative_payload(
    payload: Mapping[str, object],
    relative_type: str,
) -> None:
    if relative_type == "t5_bucket":
        _validate_positive_int(
            payload["num_buckets"],
            "position_encoding.relative.num_buckets",
        )
        _validate_positive_int(
            payload["max_distance_bp"],
            "position_encoding.relative.max_distance_bp",
        )
    elif relative_type == "rope":
        _validate_positive_finite_number(
            payload["rope_coordinate_scale"],
            "position_encoding.relative.rope_coordinate_scale",
        )
        _validate_positive_finite_number(
            payload["rope_base"],
            "position_encoding.relative.rope_base",
        )
    elif relative_type in {"alibi_fixed", "alibi_learned"}:
        _validate_string_choice(
            payload["alibi_distance_function"],
            VALID_ALIBI_DISTANCE_FUNCTIONS,
            "position_encoding.relative.alibi_distance_function",
        )
        _validate_positive_finite_number(
            payload["alibi_distance_scale"],
            "position_encoding.relative.alibi_distance_scale",
        )


def _validate_chromosome_payload(payload: Mapping[str, object]) -> None:
    _validate_string_choice(
        payload["encoding"],
        VALID_CHROMOSOME_ENCODINGS,
        "position_encoding.chromosome.encoding",
    )
    _validate_string_choice(
        payload["cross_chromosome_policy"],
        VALID_CROSS_CHROMOSOME_POLICIES,
        "position_encoding.chromosome.cross_chromosome_policy",
    )
