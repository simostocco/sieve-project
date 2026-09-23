"""Null-dataset lineage and provenance for phenotype-permuted baselines.

Phase 12C3A makes one phenotype-permuted null dataset a machine-verifiable
scientific artifact. The null transformation changes phenotype-label
assignment only, holding everything else fixed::

    X_null = X_real
    y_null = y_real[permutation_indices]

This module implements the pure hashing, validation, and non-mutating
construction logic behind that guarantee:

- ``source_artifact_sha256`` answers "what exact source .pt bytes produced
  this null artifact?"
- ``lineage_sha256`` answers "what scientific label-permutation
  transformation is this?" (source identity + sample order + original
  labels + permutation + resulting labels -- independent of paths, seeds,
  split plans, and downstream training/strategy choices).
- ``null_artifact_sha256`` answers "are these still the exact saved bytes
  that were originally validated?"

This module concerns dataset lineage only. It does not train models, run
explanations, or compute delta_rank; those remain later Phase 12C3B
concerns.

Layering note: this module has a narrow, accepted dependency on
``src.training.split_plan`` for ordered sample-ID extraction and hashing
(``ordered_sample_ids``, ``sample_ids_sha256``). ``split_plan`` is otherwise a
training-split concern; the dependency exists so null-dataset lineage and
split-plan sample membership share exactly one sample-ID hashing convention
instead of a second, competing one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.covariates import compute_file_sha256
from src.data.vcf_parser import SampleVariants, VariantRecord
from src.training.split_plan import ordered_sample_ids, sample_ids_sha256

__all__ = [
    "SCHEMA_VERSION",
    "NULL_LINEAGE_SIDECAR_SUFFIX",
    "compute_file_sha256",
    "validate_label",
    "extract_ordered_labels",
    "labels_sha256",
    "validate_permutation_indices",
    "permutation_indices_sha256",
    "apply_permutation_gather",
    "compute_lineage_sha256",
    "variant_record_equal",
    "sample_non_label_mismatch",
    "assert_samples_non_label_equal",
    "build_null_samples",
    "class_counts",
    "count_same_position",
    "compute_source_lineage_facts",
    "build_null_lineage",
    "build_embedded_metadata",
    "build_sidecar_payload",
    "sidecar_path_for",
    "write_sidecar",
    "load_sidecar",
    "extract_samples",
    "validate_embedded_metadata_schema",
    "validate_sidecar_schema",
    "validate_null_pair",
]

SCHEMA_VERSION = 1

# Deterministic strict sidecar filename suffix: "<null-artifact-name>" + this.
NULL_LINEAGE_SIDECAR_SUFFIX = ".null-lineage.yaml"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def validate_label(value: Any, *, index: int) -> int:
    """Validate a single phenotype label: plain 0/1 integer, never bool."""
    if isinstance(value, torch.Tensor):
        if value.dim() != 0:
            raise ValueError(f"label at sample {index} must be a scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"label at sample {index} must be a plain integer, not bool/other")
    plain_value = int(value)
    if plain_value not in (0, 1):
        raise ValueError(f"label at sample {index} must be 0 or 1, got {plain_value}")
    return plain_value


def extract_ordered_labels(samples: Sequence[SampleVariants]) -> list[int]:
    """Return validated 0/1 labels for *samples* in sample-index order."""
    return [validate_label(sample.label, index=index) for index, sample in enumerate(samples)]


def labels_sha256(labels: Sequence[int]) -> str:
    """Return the canonical SHA-256 hash over an ordered 0/1 label list."""
    validated = [validate_label(value, index=index) for index, value in enumerate(labels)]
    return _sha256_json(validated)


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


def validate_permutation_indices(indices: Any, *, n_samples: int) -> list[int]:
    """Validate *indices* as a true permutation of ``range(n_samples)``.

    The stored permutation vector -- not the RNG seed -- is the authoritative
    scientific transformation. Every element must be a plain, in-range,
    non-negative integer, and the full index set must cover
    ``{0, ..., n_samples - 1}`` exactly once.
    """
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("permutation_indices must be a list of integers")
    if len(indices) != n_samples:
        raise ValueError(
            f"permutation_indices length ({len(indices)}) must equal n_samples ({n_samples})"
        )
    validated: list[int] = []
    for position, value in enumerate(indices):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"permutation_indices[{position}] must be a plain integer, not bool/other"
            )
        plain_value = int(value)
        if plain_value < 0 or plain_value >= n_samples:
            raise ValueError(f"permutation_indices[{position}] is out of range: {plain_value}")
        validated.append(plain_value)
    if len(set(validated)) != n_samples:
        raise ValueError("permutation_indices must contain each sample index exactly once")
    return validated


def permutation_indices_sha256(indices: Sequence[int]) -> str:
    """Return the canonical SHA-256 hash over an already-validated permutation vector."""
    return _sha256_json([int(index) for index in indices])


def apply_permutation_gather(labels: Sequence[int], indices: Sequence[int]) -> list[int]:
    """Return null labels via gather: ``null_label[i] = original_label[indices[i]]``.

    This is a gather, not a scatter. Do not invoke the inverse/scatter
    interpretation.
    """
    return [labels[index] for index in indices]


# ---------------------------------------------------------------------------
# Semantic lineage identity
# ---------------------------------------------------------------------------


def compute_lineage_sha256(
    *,
    source_artifact_sha256: str,
    sample_ids_sha256: str,
    original_labels_sha256: str,
    permutation_indices: Sequence[int],
    permuted_labels_sha256: str,
) -> str:
    """Return ``lineage_sha256``: the scientific null-transformation identity.

    Deliberately excludes paths, ``permutation_seed``,
    ``permutation_indices_sha256``, generator argv, Git revision,
    ``null_artifact_sha256``, split plan, split/training seeds, annotation
    level, positional strategy, fold, and checkpoint -- none of those define
    what scientific label-permutation transformation this is.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_artifact_sha256": source_artifact_sha256,
        "sample_ids_sha256": sample_ids_sha256,
        "original_labels_sha256": original_labels_sha256,
        "permutation_indices": [int(index) for index in permutation_indices],
        "permuted_labels_sha256": permuted_labels_sha256,
    }
    return _sha256_json(payload)


# ---------------------------------------------------------------------------
# Non-label content equality
# ---------------------------------------------------------------------------


def _deep_equal(a: Any, b: Any) -> bool:
    """Recursively compare annotation-style values.

    Supports the value types actually emitted by preprocessing (strings,
    numbers, bool-like values, ``None``, and flat dict/list containers), plus
    a narrow tensor/array fallback in case non-scalar annotation values ever
    occur. This is intentionally not a general canonical serializer.
    """
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        return torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return bool(np.array_equal(a, b))
    return a == b


def variant_record_equal(a: VariantRecord, b: VariantRecord) -> bool:
    """Return True if two variant records are identical in all non-label content."""
    return (
        a.chrom == b.chrom
        and a.pos == b.pos
        and a.ref == b.ref
        and a.alt == b.alt
        and a.gene == b.gene
        and a.consequence == b.consequence
        and a.genotype == b.genotype
        and _deep_equal(a.annotations, b.annotations)
    )


def sample_non_label_mismatch(a: SampleVariants, b: SampleVariants) -> str | None:
    """Return a description of the first non-label mismatch between two samples, or None."""
    if a.sample_id != b.sample_id:
        return f"sample_id differs: {a.sample_id!r} != {b.sample_id!r}"
    if a.sex != b.sex:
        return f"sample {a.sample_id!r}: sex differs: {a.sex!r} != {b.sex!r}"
    if len(a.variants) != len(b.variants):
        return (
            f"sample {a.sample_id!r}: variant count differs: "
            f"{len(a.variants)} != {len(b.variants)}"
        )
    for index, (variant_a, variant_b) in enumerate(zip(a.variants, b.variants, strict=True)):
        if not variant_record_equal(variant_a, variant_b):
            return f"sample {a.sample_id!r}: variant[{index}] differs"
    return None


def assert_samples_non_label_equal(
    source_samples: Sequence[SampleVariants],
    null_samples: Sequence[SampleVariants],
) -> None:
    """Raise ValueError if any non-label content differs between paired samples."""
    if len(source_samples) != len(null_samples):
        raise ValueError(
            f"sample count mismatch: source has {len(source_samples)}, "
            f"null has {len(null_samples)}"
        )
    for index, (source_sample, null_sample) in enumerate(
        zip(source_samples, null_samples, strict=True)
    ):
        mismatch = sample_non_label_mismatch(source_sample, null_sample)
        if mismatch is not None:
            raise ValueError(f"sample {index}: non-label content mismatch: {mismatch}")


# ---------------------------------------------------------------------------
# Non-mutating null construction
# ---------------------------------------------------------------------------


def build_null_samples(
    samples: Sequence[SampleVariants],
    permutation_indices: Sequence[int],
) -> list[SampleVariants]:
    """Return new ``SampleVariants`` with permuted labels, without mutating *samples*.

    Uses ``dataclasses.replace`` so a new ``SampleVariants`` instance is
    produced per sample; source objects are never mutated in place. This
    intentionally avoids the historical non-strict generator's aliasing bug
    (``sample_copy = sample; sample_copy.label = ...``), which mutates the
    original object rather than copying it.

    ``permutation_indices`` must already be validated as a true permutation.
    """
    original_labels = extract_ordered_labels(samples)
    permuted_labels = apply_permutation_gather(original_labels, permutation_indices)
    return [
        dataclasses.replace(sample, label=permuted_labels[index])
        for index, sample in enumerate(samples)
    ]


# ---------------------------------------------------------------------------
# Descriptive / validation statistics
# ---------------------------------------------------------------------------


def class_counts(labels: Sequence[int]) -> tuple[int, int]:
    """Return ``(n_cases, n_controls)`` for already-validated 0/1 labels."""
    n_cases = sum(1 for value in labels if value == 1)
    n_controls = sum(1 for value in labels if value == 0)
    return n_cases, n_controls


def count_same_position(original_labels: Sequence[int], permuted_labels: Sequence[int]) -> int:
    """Return how many labels stayed in the same sample position after permutation."""
    return sum(
        1
        for original, permuted in zip(original_labels, permuted_labels, strict=True)
        if original == permuted
    )


# ---------------------------------------------------------------------------
# Lineage orchestration
# ---------------------------------------------------------------------------


def compute_source_lineage_facts(samples: Sequence[SampleVariants]) -> dict[str, Any]:
    """Compute ordered sample-ID and label facts needed for null lineage identity."""
    ids = ordered_sample_ids(samples)
    original_labels = extract_ordered_labels(samples)
    n_cases, n_controls = class_counts(original_labels)
    return {
        "sample_ids": ids,
        "sample_ids_sha256": sample_ids_sha256(ids),
        "original_labels": original_labels,
        "original_labels_sha256": labels_sha256(original_labels),
        "n_samples": len(ids),
        "n_cases": n_cases,
        "n_controls": n_controls,
    }


def build_null_lineage(
    *,
    samples: Sequence[SampleVariants],
    source_artifact_sha256: str,
    permutation_indices: Sequence[int],
) -> dict[str, Any]:
    """Compute the full null-dataset scientific lineage and construct null samples.

    Returns a dict of scientific-identity/validation fields plus
    ``null_samples`` (new ``SampleVariants`` instances with permuted labels;
    *samples* is never mutated).
    """
    facts = compute_source_lineage_facts(samples)
    n_samples = facts["n_samples"]
    validated_indices = validate_permutation_indices(permutation_indices, n_samples=n_samples)

    null_samples = build_null_samples(samples, validated_indices)
    permuted_labels = apply_permutation_gather(facts["original_labels"], validated_indices)
    permuted_labels_hash = labels_sha256(permuted_labels)
    indices_hash = permutation_indices_sha256(validated_indices)
    lineage_hash = compute_lineage_sha256(
        source_artifact_sha256=source_artifact_sha256,
        sample_ids_sha256=facts["sample_ids_sha256"],
        original_labels_sha256=facts["original_labels_sha256"],
        permutation_indices=validated_indices,
        permuted_labels_sha256=permuted_labels_hash,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_artifact_sha256": source_artifact_sha256,
        "sample_ids": facts["sample_ids"],
        "sample_ids_sha256": facts["sample_ids_sha256"],
        "original_labels_sha256": facts["original_labels_sha256"],
        "permuted_labels_sha256": permuted_labels_hash,
        "permutation_indices": validated_indices,
        "permutation_indices_sha256": indices_hash,
        "lineage_sha256": lineage_hash,
        "n_samples": n_samples,
        "n_cases": facts["n_cases"],
        "n_controls": facts["n_controls"],
        "same_position_count": count_same_position(facts["original_labels"], permuted_labels),
        "null_samples": null_samples,
    }


# ---------------------------------------------------------------------------
# Strict embedded metadata / sidecar payload builders
# ---------------------------------------------------------------------------


def build_embedded_metadata(
    *,
    lineage: Mapping[str, Any],
    source_artifact_path: str,
    original_path: str,
    permutation_seed: int,
    generator_script: str,
    repository_revision: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    """Build the strict ``_null_baseline_metadata`` payload embedded in the null ``.pt``.

    Deliberately excludes ``null_artifact_sha256``: a file cannot
    authoritatively contain the hash of its own final serialized bytes.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "is_null_baseline": True,
        # Operational / reproduction provenance.
        "source_artifact_path": source_artifact_path,
        "original_path": original_path,
        "permutation_seed": permutation_seed,
        "generator": {
            "script": generator_script,
            "repository_revision": repository_revision,
            "argv": list(argv),
        },
        # Scientific identity.
        "source_artifact_sha256": lineage["source_artifact_sha256"],
        "sample_ids_sha256": lineage["sample_ids_sha256"],
        "original_labels_sha256": lineage["original_labels_sha256"],
        "permuted_labels_sha256": lineage["permuted_labels_sha256"],
        "permutation_indices": list(lineage["permutation_indices"]),
        "permutation_indices_sha256": lineage["permutation_indices_sha256"],
        "lineage_sha256": lineage["lineage_sha256"],
        # Scientific validation / descriptive.
        "n_samples": lineage["n_samples"],
        "n_cases": lineage["n_cases"],
        "n_controls": lineage["n_controls"],
        "same_position_count": lineage["same_position_count"],
    }


def build_sidecar_payload(
    *,
    lineage: Mapping[str, Any],
    source_path: str,
    source_sha256: str,
    null_path: str,
    null_sha256: str,
    permutation_seed: int,
    generator_script: str,
    repository_revision: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    """Build the strict ``<null-artifact>.null-lineage.yaml`` sidecar payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_sha256": lineage["lineage_sha256"],
        "source": {"path": source_path, "sha256": source_sha256},
        "null": {"path": null_path, "sha256": null_sha256},
        "samples": {
            "n_samples": lineage["n_samples"],
            "sample_ids_sha256": lineage["sample_ids_sha256"],
            "n_cases": lineage["n_cases"],
            "n_controls": lineage["n_controls"],
            "same_position_count": lineage["same_position_count"],
        },
        "labels": {
            "original_sha256": lineage["original_labels_sha256"],
            "permuted_sha256": lineage["permuted_labels_sha256"],
        },
        "permutation": {
            "seed": permutation_seed,
            "indices": list(lineage["permutation_indices"]),
            "indices_sha256": lineage["permutation_indices_sha256"],
        },
        "generator": {
            "repository_revision": repository_revision,
            "argv": list(argv),
            "script": generator_script,
        },
    }


def sidecar_path_for(null_path: Path) -> Path:
    """Return the deterministic strict sidecar path for a null artifact path."""
    null_path = Path(null_path)
    return null_path.parent / f"{null_path.name}{NULL_LINEAGE_SIDECAR_SUFFIX}"


def write_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a lineage sidecar as YAML with stable, human-readable key order."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def load_sidecar(path: Path) -> dict[str, Any]:
    """Load a lineage sidecar YAML mapping from *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            payload = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"null-lineage sidecar at {path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"null-lineage sidecar at {path} must contain a YAML mapping")
    return dict(payload)


# ---------------------------------------------------------------------------
# Pair validator
# ---------------------------------------------------------------------------


def extract_samples(data: Any, *, label: str) -> list[SampleVariants]:
    """Return the validated ``samples`` list from a loaded preprocessed/null artifact.

    Requires the real preprocessing artifact shape (a top-level mapping with a
    non-empty ``'samples'`` list of ``SampleVariants`` instances); *label* is
    used only for error messages (e.g. ``"source"`` or ``"null"``).
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} artifact must contain a top-level mapping")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label} artifact must contain a non-empty 'samples' list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, SampleVariants):
            raise ValueError(
                f"{label} artifact samples[{index}] must be a SampleVariants instance "
                "for strict lineage validation"
            )
    return samples


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_key(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_sha256_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX_DIGITS for c in value):
        raise ValueError(f"{name} must be a lowercase 64-character hexadecimal string")
    return value


def _validate_plain_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain integer, not bool/other")
    return value


def _validate_non_negative_int(value: Any, name: str) -> int:
    value = _validate_plain_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _validate_non_empty_string(value: Any, name: str) -> str:
    _validate_string(value, name)
    if value == "":
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{index}] must be a string")
    return list(value)


def _validate_index_sequence_shape(value: Any, name: str) -> None:
    """Check *value* is list-like without yet enforcing the true-permutation invariant.

    ``validate_permutation_indices`` remains the sole authority for the
    permutation invariant itself (true bijection over ``range(n_samples)``);
    this only guards the container shape so schema validation can run before
    ``n_samples`` is known.
    """
    if isinstance(value, np.ndarray):
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers")


def _validate_embedded_metadata_schema(embedded: Mapping[str, Any]) -> None:
    """Validate presence and basic types of every strict embedded provenance field.

    Scientific-identity fields (the various SHA-256 hashes, the permutation
    vector, and the descriptive counts) are independently recomputed and
    compared elsewhere in ``validate_null_pair``; this only guards that the
    strict schema is complete and well-typed so a partially-populated
    artifact fails closed instead of silently passing via ``.get()``
    returning ``None`` on both sides of a comparison.
    """
    path = "embedded"
    _validate_plain_int(_require_key(embedded, "schema_version", path), f"{path}.schema_version")
    if embedded["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}.schema_version must be {SCHEMA_VERSION}")
    if _require_key(embedded, "is_null_baseline", path) is not True:
        raise ValueError(f"{path}.is_null_baseline must be True")

    # Operational / reproduction provenance. Paths are required to be present
    # (a strict artifact must record where it came from) but are never
    # compared to the validator's own current paths: relocation is allowed.
    _validate_string(
        _require_key(embedded, "source_artifact_path", path), f"{path}.source_artifact_path"
    )
    _validate_string(_require_key(embedded, "original_path", path), f"{path}.original_path")
    _validate_plain_int(
        _require_key(embedded, "permutation_seed", path), f"{path}.permutation_seed"
    )

    generator = _require_mapping(_require_key(embedded, "generator", path), f"{path}.generator")
    _validate_non_empty_string(
        _require_key(generator, "script", f"{path}.generator"), f"{path}.generator.script"
    )
    _validate_string(
        _require_key(generator, "repository_revision", f"{path}.generator"),
        f"{path}.generator.repository_revision",
    )
    _validate_string_list(
        _require_key(generator, "argv", f"{path}.generator"), f"{path}.generator.argv"
    )

    for key in (
        "source_artifact_sha256",
        "sample_ids_sha256",
        "original_labels_sha256",
        "permuted_labels_sha256",
        "permutation_indices_sha256",
        "lineage_sha256",
    ):
        _validate_sha256_hex(_require_key(embedded, key, path), f"{path}.{key}")

    _validate_index_sequence_shape(
        _require_key(embedded, "permutation_indices", path), f"{path}.permutation_indices"
    )

    for key in ("n_samples", "n_cases", "n_controls", "same_position_count"):
        _validate_non_negative_int(_require_key(embedded, key, path), f"{path}.{key}")


def _validate_sidecar_schema(sidecar: Mapping[str, Any]) -> None:
    """Validate presence and basic types of every strict sidecar provenance field.

    Mirrors ``_validate_embedded_metadata_schema``; see its docstring for why
    this is a separate, upfront pass rather than relying on ``.get()``
    defaults during scientific-identity comparison.
    """
    path = "sidecar"
    _validate_plain_int(_require_key(sidecar, "schema_version", path), f"{path}.schema_version")
    if sidecar["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}.schema_version must be {SCHEMA_VERSION}")
    _validate_sha256_hex(_require_key(sidecar, "lineage_sha256", path), f"{path}.lineage_sha256")

    source = _require_mapping(_require_key(sidecar, "source", path), f"{path}.source")
    _validate_string(_require_key(source, "path", f"{path}.source"), f"{path}.source.path")
    _validate_sha256_hex(_require_key(source, "sha256", f"{path}.source"), f"{path}.source.sha256")

    null_section = _require_mapping(_require_key(sidecar, "null", path), f"{path}.null")
    _validate_string(_require_key(null_section, "path", f"{path}.null"), f"{path}.null.path")
    _validate_sha256_hex(
        _require_key(null_section, "sha256", f"{path}.null"), f"{path}.null.sha256"
    )

    samples = _require_mapping(_require_key(sidecar, "samples", path), f"{path}.samples")
    _validate_non_negative_int(
        _require_key(samples, "n_samples", f"{path}.samples"), f"{path}.samples.n_samples"
    )
    _validate_sha256_hex(
        _require_key(samples, "sample_ids_sha256", f"{path}.samples"),
        f"{path}.samples.sample_ids_sha256",
    )
    _validate_non_negative_int(
        _require_key(samples, "n_cases", f"{path}.samples"), f"{path}.samples.n_cases"
    )
    _validate_non_negative_int(
        _require_key(samples, "n_controls", f"{path}.samples"), f"{path}.samples.n_controls"
    )
    _validate_non_negative_int(
        _require_key(samples, "same_position_count", f"{path}.samples"),
        f"{path}.samples.same_position_count",
    )

    labels = _require_mapping(_require_key(sidecar, "labels", path), f"{path}.labels")
    _validate_sha256_hex(
        _require_key(labels, "original_sha256", f"{path}.labels"), f"{path}.labels.original_sha256"
    )
    _validate_sha256_hex(
        _require_key(labels, "permuted_sha256", f"{path}.labels"), f"{path}.labels.permuted_sha256"
    )

    permutation = _require_mapping(
        _require_key(sidecar, "permutation", path), f"{path}.permutation"
    )
    _validate_plain_int(
        _require_key(permutation, "seed", f"{path}.permutation"), f"{path}.permutation.seed"
    )
    _validate_index_sequence_shape(
        _require_key(permutation, "indices", f"{path}.permutation"), f"{path}.permutation.indices"
    )
    _validate_sha256_hex(
        _require_key(permutation, "indices_sha256", f"{path}.permutation"),
        f"{path}.permutation.indices_sha256",
    )

    generator = _require_mapping(_require_key(sidecar, "generator", path), f"{path}.generator")
    _validate_string(
        _require_key(generator, "repository_revision", f"{path}.generator"),
        f"{path}.generator.repository_revision",
    )
    _validate_string_list(
        _require_key(generator, "argv", f"{path}.generator"), f"{path}.generator.argv"
    )
    _validate_non_empty_string(
        _require_key(generator, "script", f"{path}.generator"), f"{path}.generator.script"
    )


def validate_embedded_metadata_schema(embedded: Mapping[str, Any]) -> None:
    """Public strict-schema check for embedded ``_null_baseline_metadata``.

    Phase 12C3B1 additive export: a thin alias over the 12C3A schema pass so
    training/explanation provenance can fail closed on a structurally invalid
    strict artifact without reaching into a private helper. Semantics are
    identical to the check ``validate_null_pair`` already performs.
    """
    _validate_embedded_metadata_schema(_require_mapping(embedded, "embedded"))


def validate_sidecar_schema(sidecar: Mapping[str, Any]) -> None:
    """Public strict-schema check for a loaded ``.null-lineage.yaml`` sidecar.

    Phase 12C3B1 additive export used by the dry-run benchmark planner, which
    must validate sidecar structure without loading either cohort artifact.
    Semantics are identical to the check ``validate_null_pair`` performs.
    """
    _validate_sidecar_schema(_require_mapping(sidecar, "sidecar"))


def validate_null_pair(
    source_path: Path,
    null_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    """Validate a strict null artifact against its source and lineage sidecar.

    Recomputes every scientific-identity value from raw file bytes and the
    loaded dataset objects; it never trusts an embedded or sidecar claim
    without independently recomputing and comparing it. Path relocation alone
    does not invalidate the pair as long as byte identities still match.

    Raises ``ValueError`` naming the failed invariant. Returns a compact
    report dict of recomputed identity/validation values on success.
    """
    source_path = Path(source_path)
    null_path = Path(null_path)
    sidecar_path = Path(sidecar_path)

    for label, path in (("source", source_path), ("null", null_path), ("sidecar", sidecar_path)):
        if not path.exists():
            raise ValueError(f"{label} path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label} path is not a file: {path}")

    sidecar = load_sidecar(sidecar_path)
    _validate_sidecar_schema(sidecar)

    source_sha_actual = compute_file_sha256(source_path)
    sidecar_source = _require_mapping(sidecar.get("source"), "sidecar.source")
    _require_equal(sidecar_source.get("sha256"), source_sha_actual, "sidecar.source.sha256")

    null_sha_actual = compute_file_sha256(null_path)
    sidecar_null = _require_mapping(sidecar.get("null"), "sidecar.null")
    _require_equal(sidecar_null.get("sha256"), null_sha_actual, "sidecar.null.sha256")

    source_data = torch.load(source_path, weights_only=False)
    null_data = torch.load(null_path, weights_only=False)
    source_samples = extract_samples(source_data, label="source")
    null_samples = extract_samples(null_data, label="null")

    embedded = _require_mapping(
        null_data.get("_null_baseline_metadata"), "null._null_baseline_metadata"
    )
    _validate_embedded_metadata_schema(embedded)

    # Non-label biological/preprocessing invariant: every top-level artifact
    # key other than 'samples' (label-bearing) and the additive strict
    # '_null_baseline_metadata' must be byte-for-byte unchanged.
    source_extra_keys = set(source_data.keys()) - {"samples"}
    null_extra_keys = set(null_data.keys()) - {"samples", "_null_baseline_metadata"}
    if source_extra_keys != null_extra_keys:
        raise ValueError(
            "top-level preprocessing metadata keys differ between source and null: "
            f"source={sorted(source_extra_keys)}, null={sorted(null_extra_keys)}"
        )
    for key in source_extra_keys:
        if not _deep_equal(source_data[key], null_data[key]):
            raise ValueError(
                f"top-level preprocessing metadata key {key!r} differs between source and null"
            )

    source_sample_ids = ordered_sample_ids(source_samples)
    null_sample_ids = ordered_sample_ids(null_samples)
    source_ids_hash = sample_ids_sha256(source_sample_ids)
    null_ids_hash = sample_ids_sha256(null_sample_ids)
    if source_ids_hash != null_ids_hash:
        raise ValueError(
            "source and null sample ordering/identity differ (sample_ids_sha256 mismatch)"
        )

    original_labels = extract_ordered_labels(source_samples)
    null_labels = extract_ordered_labels(null_samples)
    original_labels_hash = labels_sha256(original_labels)
    null_labels_hash = labels_sha256(null_labels)

    validated_indices = validate_permutation_indices(
        embedded.get("permutation_indices"), n_samples=len(source_sample_ids)
    )
    indices_hash = permutation_indices_sha256(validated_indices)

    expected_null_labels = apply_permutation_gather(original_labels, validated_indices)
    if expected_null_labels != null_labels:
        raise ValueError(
            "null labels do not match original_labels[permutation_indices]; "
            "gather reconstruction failed"
        )

    lineage_hash = compute_lineage_sha256(
        source_artifact_sha256=source_sha_actual,
        sample_ids_sha256=source_ids_hash,
        original_labels_sha256=original_labels_hash,
        permutation_indices=validated_indices,
        permuted_labels_sha256=null_labels_hash,
    )

    _require_equal(
        embedded.get("source_artifact_sha256"), source_sha_actual, "embedded.source_artifact_sha256"
    )
    _require_equal(embedded.get("sample_ids_sha256"), source_ids_hash, "embedded.sample_ids_sha256")
    _require_equal(
        embedded.get("original_labels_sha256"),
        original_labels_hash,
        "embedded.original_labels_sha256",
    )
    _require_equal(
        embedded.get("permuted_labels_sha256"), null_labels_hash, "embedded.permuted_labels_sha256"
    )
    _require_equal(
        embedded.get("permutation_indices_sha256"),
        indices_hash,
        "embedded.permutation_indices_sha256",
    )
    _require_equal(embedded.get("lineage_sha256"), lineage_hash, "embedded.lineage_sha256")

    n_cases, n_controls = class_counts(original_labels)
    same_position = count_same_position(original_labels, null_labels)
    _require_equal(embedded.get("n_samples"), len(source_sample_ids), "embedded.n_samples")
    _require_equal(embedded.get("n_cases"), n_cases, "embedded.n_cases")
    _require_equal(embedded.get("n_controls"), n_controls, "embedded.n_controls")
    _require_equal(
        embedded.get("same_position_count"), same_position, "embedded.same_position_count"
    )

    assert_samples_non_label_equal(source_samples, null_samples)

    _require_equal(sidecar.get("lineage_sha256"), lineage_hash, "sidecar.lineage_sha256")

    sidecar_samples = _require_mapping(sidecar.get("samples"), "sidecar.samples")
    _require_equal(
        sidecar_samples.get("n_samples"), len(source_sample_ids), "sidecar.samples.n_samples"
    )
    _require_equal(
        sidecar_samples.get("sample_ids_sha256"),
        source_ids_hash,
        "sidecar.samples.sample_ids_sha256",
    )
    _require_equal(sidecar_samples.get("n_cases"), n_cases, "sidecar.samples.n_cases")
    _require_equal(sidecar_samples.get("n_controls"), n_controls, "sidecar.samples.n_controls")
    _require_equal(
        sidecar_samples.get("same_position_count"),
        same_position,
        "sidecar.samples.same_position_count",
    )

    sidecar_labels = _require_mapping(sidecar.get("labels"), "sidecar.labels")
    _require_equal(
        sidecar_labels.get("original_sha256"),
        original_labels_hash,
        "sidecar.labels.original_sha256",
    )
    _require_equal(
        sidecar_labels.get("permuted_sha256"), null_labels_hash, "sidecar.labels.permuted_sha256"
    )

    sidecar_permutation = _require_mapping(sidecar.get("permutation"), "sidecar.permutation")
    sidecar_indices = validate_permutation_indices(
        sidecar_permutation.get("indices"), n_samples=len(source_sample_ids)
    )
    if sidecar_indices != validated_indices:
        raise ValueError("sidecar.permutation.indices does not match embedded permutation_indices")
    _require_equal(
        sidecar_permutation.get("indices_sha256"),
        indices_hash,
        "sidecar.permutation.indices_sha256",
    )
    _require_equal(
        sidecar_permutation.get("seed"),
        embedded.get("permutation_seed"),
        "sidecar.permutation.seed",
    )

    embedded_generator = _require_mapping(embedded.get("generator"), "embedded.generator")
    sidecar_generator = _require_mapping(sidecar.get("generator"), "sidecar.generator")
    _require_equal(
        sidecar_generator.get("script"),
        embedded_generator.get("script"),
        "sidecar.generator.script",
    )
    _require_equal(
        sidecar_generator.get("repository_revision"),
        embedded_generator.get("repository_revision"),
        "sidecar.generator.repository_revision",
    )
    _require_equal(
        list(sidecar_generator.get("argv") or []),
        list(embedded_generator.get("argv") or []),
        "sidecar.generator.argv",
    )

    return {
        "lineage_sha256": lineage_hash,
        "source_artifact_sha256": source_sha_actual,
        "null_artifact_sha256": null_sha_actual,
        "sample_ids_sha256": source_ids_hash,
        "original_labels_sha256": original_labels_hash,
        "permuted_labels_sha256": null_labels_hash,
        "permutation_indices_sha256": indices_hash,
        "n_samples": len(source_sample_ids),
        "n_cases": n_cases,
        "n_controls": n_controls,
        "same_position_count": same_position,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_json(value: Any) -> str:
    """Return SHA-256 over canonical compact JSON, matching split_plan's convention."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
