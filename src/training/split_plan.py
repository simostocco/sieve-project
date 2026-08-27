"""Versioned training split-plan helpers.

Split plans record sample membership only. They intentionally do not record or
validate labels so a real-run plan can be replayed against a null dataset with
permuted phenotypes while preserving the exact same people in each subset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

SplitMode = Literal["cv", "single_split"]
SplitSource = Literal["generated", "replayed"]


def ordered_sample_ids(samples: Sequence[object]) -> list[str]:
    """Return validated sample IDs in the sample-index order used by training."""
    sample_ids = []
    seen = set()
    for sample_index, sample in enumerate(samples):
        sample_id = getattr(sample, "sample_id", None)
        if not isinstance(sample_id, str):
            raise ValueError(f"sample {sample_index} sample_id must be a string")
        if sample_id.strip() == "":
            raise ValueError(f"sample {sample_index} sample_id must be non-empty")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in training data: {sample_id!r}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Return SHA-256 over the exact ordered sample-ID strings."""
    _validate_sample_id_list(sample_ids)
    return _sha256_json(list(sample_ids))


def subset_sample_ids_sha256(sample_ids: Sequence[str], indices: Sequence[int]) -> str:
    """Return SHA-256 over exact ordered sample IDs selected by *indices*."""
    _validate_sample_id_list(sample_ids)
    selected = [sample_ids[index] for index in indices]
    return _sha256_json(selected)


def build_cv_split_plan(
    *,
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    sample_ids: Sequence[str],
    seed: int,
    split_source: SplitSource,
    n_folds: int,
) -> dict[str, object]:
    """Build a normalized CV split plan from sample-level fold indices."""
    plan = {
        "schema_version": 1,
        "mode": "cv",
        "n_samples": len(sample_ids),
        "sample_ids_sha256": sample_ids_sha256(sample_ids),
        "seed": _plain_int(seed, "seed"),
        "split_source": split_source,
        "n_folds": _plain_int(n_folds, "n_folds"),
        "folds": [
            {
                "fold_index": fold_index,
                "train_indices": [_plain_int(index, "train_indices") for index in train],
                "val_indices": [_plain_int(index, "val_indices") for index in val],
            }
            for fold_index, (train, val) in enumerate(folds)
        ],
    }
    return validate_split_plan(
        plan,
        sample_ids=sample_ids,
        expected_mode="cv",
        expected_n_folds=n_folds,
    )


def build_single_split_plan(
    *,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    sample_ids: Sequence[str],
    seed: int,
    split_source: SplitSource,
) -> dict[str, object]:
    """Build a normalized single train/validation split plan."""
    plan = {
        "schema_version": 1,
        "mode": "single_split",
        "n_samples": len(sample_ids),
        "sample_ids_sha256": sample_ids_sha256(sample_ids),
        "seed": _plain_int(seed, "seed"),
        "split_source": split_source,
        "train_indices": [_plain_int(index, "train_indices") for index in train_indices],
        "val_indices": [_plain_int(index, "val_indices") for index in val_indices],
    }
    return validate_split_plan(
        plan,
        sample_ids=sample_ids,
        expected_mode="single_split",
    )


def validate_split_plan(
    plan: Mapping[str, object],
    *,
    sample_ids: Sequence[str],
    expected_mode: SplitMode,
    expected_n_folds: int | None = None,
) -> dict[str, object]:
    """Validate *plan* against the current ordered sample IDs and normalize it."""
    if not isinstance(plan, Mapping):
        raise ValueError("split plan must be a mapping")
    _validate_sample_id_list(sample_ids)

    schema_version = _required_int(plan, "schema_version", path="split_plan")
    if schema_version != 1:
        raise ValueError("split_plan.schema_version must be 1")
    mode = plan.get("mode")
    if mode != expected_mode:
        raise ValueError(f"split_plan.mode must be {expected_mode!r}; got {mode!r}")
    n_samples = _required_int(plan, "n_samples", path="split_plan")
    if n_samples != len(sample_ids):
        raise ValueError(
            "split_plan.n_samples does not match current dataset sample count "
            f"({n_samples} != {len(sample_ids)})"
        )
    full_hash = plan.get("sample_ids_sha256")
    expected_full_hash = sample_ids_sha256(sample_ids)
    if full_hash != expected_full_hash:
        raise ValueError("split_plan.sample_ids_sha256 does not match current sample IDs")

    seed = _required_int(plan, "seed", path="split_plan")
    split_source = plan.get("split_source")
    if split_source not in {"generated", "replayed"}:
        raise ValueError("split_plan.split_source must be 'generated' or 'replayed'")

    normalized: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "n_samples": n_samples,
        "sample_ids_sha256": expected_full_hash,
        "seed": seed,
        "split_source": split_source,
    }

    if mode == "cv":
        if expected_n_folds is None:
            raise ValueError("expected_n_folds is required for CV split-plan validation")
        normalized.update(
            _validate_cv_plan(
                plan,
                sample_ids=sample_ids,
                expected_n_folds=expected_n_folds,
            )
        )
    else:
        normalized.update(_validate_single_plan(plan, sample_ids=sample_ids))

    return normalized


def split_plan_membership_payload(plan: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical sample-membership payload used for split-plan SHA-256."""
    mode = plan.get("mode")
    payload: dict[str, object] = {
        "schema_version": plan.get("schema_version"),
        "mode": mode,
        "n_samples": plan.get("n_samples"),
        "sample_ids_sha256": plan.get("sample_ids_sha256"),
    }
    if mode == "cv":
        folds = plan.get("folds")
        if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
            raise ValueError("split_plan.folds must be a list")
        payload["n_folds"] = plan.get("n_folds")
        payload["folds"] = [
            {
                "fold_index": fold.get("fold_index"),
                "train_indices": list(fold.get("train_indices", [])),
                "val_indices": list(fold.get("val_indices", [])),
                "train_sample_ids_sha256": fold.get("train_sample_ids_sha256"),
                "val_sample_ids_sha256": fold.get("val_sample_ids_sha256"),
            }
            for fold in folds
            if isinstance(fold, Mapping)
        ]
    elif mode == "single_split":
        payload.update(
            {
                "train_indices": list(plan.get("train_indices", [])),
                "val_indices": list(plan.get("val_indices", [])),
                "train_sample_ids_sha256": plan.get("train_sample_ids_sha256"),
                "val_sample_ids_sha256": plan.get("val_sample_ids_sha256"),
            }
        )
    else:
        raise ValueError("split_plan.mode must be 'cv' or 'single_split'")
    return payload


def split_plan_sha256(plan: Mapping[str, object]) -> str:
    """Return the canonical sample-membership identity hash for *plan*."""
    return _sha256_json(split_plan_membership_payload(plan))


def load_split_plan(path: Path) -> dict[str, object]:
    """Load a YAML split plan mapping from *path*."""
    with path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if not isinstance(plan, Mapping):
        raise ValueError(f"split plan at {path} must contain a YAML mapping")
    return dict(plan)


def write_split_plan(path: Path, plan: Mapping[str, object]) -> None:
    """Write *plan* as YAML with stable key order preserved by construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(plan), handle, sort_keys=False)


def write_or_validate_existing_split_plan(
    path: Path,
    requested_plan: Mapping[str, object],
    *,
    sample_ids: Sequence[str],
    expected_mode: SplitMode,
    expected_n_folds: int | None = None,
) -> dict[str, object]:
    """Write a new plan or reuse an existing file with identical membership.

    Existing ``split_plan.yaml`` files are immutable when their sample
    membership matches. The current invocation source is recorded separately in
    training config/checkpoint metadata.
    """
    requested_normalized = validate_split_plan(
        requested_plan,
        sample_ids=sample_ids,
        expected_mode=expected_mode,
        expected_n_folds=expected_n_folds,
    )
    requested_sha = split_plan_sha256(requested_normalized)
    if not path.exists():
        write_split_plan(path, requested_normalized)
        return requested_normalized

    existing = validate_split_plan(
        load_split_plan(path),
        sample_ids=sample_ids,
        expected_mode=expected_mode,
        expected_n_folds=expected_n_folds,
    )
    existing_sha = split_plan_sha256(existing)
    if existing_sha != requested_sha:
        raise ValueError(f"existing split_plan.yaml at {path} has different sample membership")
    return existing


def cv_folds_from_plan(plan: Mapping[str, object]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return CV fold arrays from a validated split plan."""
    folds = plan.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise ValueError("split_plan.folds must be a list")
    return [
        (
            np.asarray(fold["train_indices"], dtype=np.int64),
            np.asarray(fold["val_indices"], dtype=np.int64),
        )
        for fold in folds
        if isinstance(fold, Mapping)
    ]


def single_split_from_plan(plan: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return single-split train/validation arrays from a validated split plan."""
    return (
        np.asarray(plan["train_indices"], dtype=np.int64),
        np.asarray(plan["val_indices"], dtype=np.int64),
    )


def build_split_plan_metadata(
    *,
    source: SplitSource,
    experiment_plan_path: Path,
    plan: Mapping[str, object],
    input_plan_path: Path | None = None,
    input_plan: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build config/checkpoint metadata for the current split-plan invocation."""
    return {
        "schema_version": 1,
        "source": source,
        "path": str(experiment_plan_path.resolve()),
        "sha256": split_plan_sha256(plan),
        "sample_ids_sha256": plan["sample_ids_sha256"],
        "input_path": str(input_plan_path.resolve()) if input_plan_path is not None else None,
        "input_sha256": split_plan_sha256(input_plan) if input_plan is not None else None,
    }


def _validate_cv_plan(
    plan: Mapping[str, object],
    *,
    sample_ids: Sequence[str],
    expected_n_folds: int,
) -> dict[str, object]:
    n_folds = _required_int(plan, "n_folds", path="split_plan")
    if n_folds != expected_n_folds:
        raise ValueError(
            f"split_plan.n_folds must match requested CV folds ({n_folds} != {expected_n_folds})"
        )
    folds = plan.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise ValueError("split_plan.folds must be a list")
    if len(folds) != n_folds:
        raise ValueError("split_plan.folds length must match split_plan.n_folds")

    normalized_folds = []
    seen_fold_indices = set()
    all_val_indices = []
    for fold_pos, raw_fold in enumerate(folds):
        if not isinstance(raw_fold, Mapping):
            raise ValueError(f"split_plan.folds[{fold_pos}] must be a mapping")
        fold_index = _required_int(raw_fold, "fold_index", path=f"split_plan.folds[{fold_pos}]")
        if fold_index in seen_fold_indices:
            raise ValueError(f"duplicate CV fold_index in split plan: {fold_index}")
        seen_fold_indices.add(fold_index)
        train_indices = _validate_index_list(
            raw_fold.get("train_indices"),
            path=f"split_plan.folds[{fold_pos}].train_indices",
            n_samples=len(sample_ids),
        )
        val_indices = _validate_index_list(
            raw_fold.get("val_indices"),
            path=f"split_plan.folds[{fold_pos}].val_indices",
            n_samples=len(sample_ids),
        )
        _validate_partition(
            train_indices,
            val_indices,
            n_samples=len(sample_ids),
            path=f"split_plan.folds[{fold_pos}]",
        )
        train_hash = _validate_subset_hash(
            raw_fold,
            "train_sample_ids_sha256",
            sample_ids,
            train_indices,
            path=f"split_plan.folds[{fold_pos}]",
        )
        val_hash = _validate_subset_hash(
            raw_fold,
            "val_sample_ids_sha256",
            sample_ids,
            val_indices,
            path=f"split_plan.folds[{fold_pos}]",
        )
        all_val_indices.extend(val_indices)
        normalized_folds.append(
            {
                "fold_index": fold_index,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_sample_ids_sha256": train_hash,
                "val_sample_ids_sha256": val_hash,
            }
        )

    expected_fold_indices = set(range(n_folds))
    if seen_fold_indices != expected_fold_indices:
        raise ValueError("CV split plan must contain exactly fold_index values 0..n_folds-1")
    if sorted(all_val_indices) != list(range(len(sample_ids))):
        raise ValueError("each sample must appear exactly once across CV validation folds")

    normalized_folds.sort(key=lambda fold: int(fold["fold_index"]))
    return {"n_folds": n_folds, "folds": normalized_folds}


def _validate_single_plan(
    plan: Mapping[str, object],
    *,
    sample_ids: Sequence[str],
) -> dict[str, object]:
    train_indices = _validate_index_list(
        plan.get("train_indices"),
        path="split_plan.train_indices",
        n_samples=len(sample_ids),
    )
    val_indices = _validate_index_list(
        plan.get("val_indices"),
        path="split_plan.val_indices",
        n_samples=len(sample_ids),
    )
    _validate_partition(
        train_indices,
        val_indices,
        n_samples=len(sample_ids),
        path="split_plan",
    )
    train_hash = _validate_subset_hash(
        plan,
        "train_sample_ids_sha256",
        sample_ids,
        train_indices,
        path="split_plan",
    )
    val_hash = _validate_subset_hash(
        plan,
        "val_sample_ids_sha256",
        sample_ids,
        val_indices,
        path="split_plan",
    )
    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_ids_sha256": train_hash,
        "val_sample_ids_sha256": val_hash,
    }


def _validate_partition(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    *,
    n_samples: int,
    path: str,
) -> None:
    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        raise ValueError(f"{path} train and validation indices must be disjoint")
    if train_set | val_set != set(range(n_samples)):
        raise ValueError(f"{path} train and validation indices must cover all samples")


def _validate_subset_hash(
    data: Mapping[str, object],
    key: str,
    sample_ids: Sequence[str],
    indices: Sequence[int],
    *,
    path: str,
) -> str:
    expected = subset_sample_ids_sha256(sample_ids, indices)
    observed = data.get(key)
    if observed is None:
        return expected
    if observed != expected:
        raise ValueError(f"{path}.{key} does not match selected sample IDs")
    return expected


def _validate_index_list(value: object, *, path: str, n_samples: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a list of integer sample indices")
    indices = [_plain_int(index, path) for index in value]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path} must not contain duplicate indices")
    for index in indices:
        if index < 0 or index >= n_samples:
            raise ValueError(f"{path} contains out-of-range sample index {index}")
    return indices


def _validate_sample_id_list(sample_ids: Sequence[str]) -> None:
    seen = set()
    for sample_index, sample_id in enumerate(sample_ids):
        if not isinstance(sample_id, str):
            raise ValueError(f"sample_ids[{sample_index}] must be a string")
        if sample_id.strip() == "":
            raise ValueError(f"sample_ids[{sample_index}] must be non-empty")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in training data: {sample_id!r}")
        seen.add(sample_id)


def _required_int(data: Mapping[str, object], key: str, *, path: str) -> int:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    return _plain_int(data[key], f"{path}.{key}")


def _plain_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{path} must be an integer, not bool")
    return int(value)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
