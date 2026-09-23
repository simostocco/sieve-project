"""Dataset byte/sample provenance for training and explanation runs.

Phase 12C3B1 binds every preprocessed-data training run and explanation run to
the exact dataset bytes it consumed, so a paired real/null benchmark can prove
that a null checkpoint was trained (and explained) on the exact validated null
artifact rather than trusting a filesystem path.

The block this module builds is recorded as ``dataset_provenance``::

    schema_version: 1
    preprocessed_data_path: <resolved operational path; provenance only>
    preprocessed_data_sha256: <raw file-byte SHA-256; scientific identity>
    sample_ids_sha256: <ordered sample-ID SHA-256>
    is_null_baseline: <bool>
    null_metadata_kind: none | strict_v1 | legacy_unversioned
    null_lineage: null | {lineage_sha256, source_artifact_sha256,
                          permutation_indices_sha256, original_labels_sha256,
                          permuted_labels_sha256}

``null_lineage`` values are copied from the strict embedded 12C3A metadata
after a strict schema check and a cheap self-consistency check against the
loaded samples (ordered sample IDs, sample count, and permuted-label hash).
This is *not* real/null pair validation: ``validate_null_pair`` (which loads the
source artifact too) remains the Phase 12C3B2 execution-preflight authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data import null_lineage
from src.data.covariates import compute_file_sha256
from src.training.split_plan import ordered_sample_ids, sample_ids_sha256

DATASET_PROVENANCE_SCHEMA_VERSION = 1
NULL_METADATA_KEY = "_null_baseline_metadata"
NULL_METADATA_KIND_NONE = "none"
NULL_METADATA_KIND_STRICT = "strict_v1"
NULL_METADATA_KIND_LEGACY = "legacy_unversioned"
NULL_LINEAGE_FIELDS = (
    "lineage_sha256",
    "source_artifact_sha256",
    "permutation_indices_sha256",
    "original_labels_sha256",
    "permuted_labels_sha256",
)


def build_dataset_provenance(
    preprocessed: Mapping[str, Any],
    *,
    path: str | Path,
    preprocessed_data_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the ``dataset_provenance`` block for one loaded preprocessed artifact.

    *preprocessed* is the mapping returned by ``torch.load`` on *path*. The raw
    byte SHA-256 is computed from *path* unless the caller already hashed it.
    Raises ``ValueError`` when ``_null_baseline_metadata`` is present but is not
    a well-formed strict (or recognisably legacy) null declaration, so a broken
    null artifact never silently passes as ordinary real data.
    """
    if not isinstance(preprocessed, Mapping):
        raise ValueError("preprocessed data must be a mapping with a 'samples' list")
    samples = preprocessed.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("preprocessed data must contain a 'samples' list")

    resolved_path = Path(path).resolve()
    data_sha = (
        compute_file_sha256(resolved_path)
        if preprocessed_data_sha256 is None
        else preprocessed_data_sha256
    )
    sample_ids = ordered_sample_ids(samples)
    ids_sha = sample_ids_sha256(sample_ids)

    kind, lineage = _classify_null_metadata(preprocessed, samples=samples, ids_sha=ids_sha)
    return {
        "schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
        "preprocessed_data_path": str(resolved_path),
        "preprocessed_data_sha256": data_sha,
        "sample_ids_sha256": ids_sha,
        "is_null_baseline": kind != NULL_METADATA_KIND_NONE,
        "null_metadata_kind": kind,
        "null_lineage": lineage,
    }


def _classify_null_metadata(
    preprocessed: Mapping[str, Any],
    *,
    samples: Sequence[Any],
    ids_sha: str,
) -> tuple[str, dict[str, str] | None]:
    if NULL_METADATA_KEY not in preprocessed:
        return NULL_METADATA_KIND_NONE, None

    embedded = preprocessed[NULL_METADATA_KEY]
    if not isinstance(embedded, Mapping):
        raise ValueError(f"{NULL_METADATA_KEY} must be a mapping when present")

    if "schema_version" not in embedded:
        # Historical non-strict generator output: declares itself null but has
        # no lineage. Recorded as null (never as real) without lineage, so the
        # paired benchmark validator rejects it while legacy workflows run.
        if embedded.get("is_null_baseline") is not True:
            raise ValueError(
                f"{NULL_METADATA_KEY} without schema_version must declare is_null_baseline: true"
            )
        return NULL_METADATA_KIND_LEGACY, None

    try:
        null_lineage.validate_embedded_metadata_schema(embedded)
    except ValueError as error:
        raise ValueError(f"invalid strict {NULL_METADATA_KEY}: {error}") from error

    if embedded["sample_ids_sha256"] != ids_sha:
        raise ValueError(f"{NULL_METADATA_KEY}.sample_ids_sha256 does not match the loaded samples")
    if embedded["n_samples"] != len(samples):
        raise ValueError(f"{NULL_METADATA_KEY}.n_samples does not match the loaded samples")
    permuted_labels = null_lineage.labels_sha256(null_lineage.extract_ordered_labels(samples))
    if embedded["permuted_labels_sha256"] != permuted_labels:
        raise ValueError(
            f"{NULL_METADATA_KEY}.permuted_labels_sha256 does not match the loaded labels"
        )
    return NULL_METADATA_KIND_STRICT, {field: embedded[field] for field in NULL_LINEAGE_FIELDS}


def training_has_dataset_provenance(training_config: Mapping[str, Any]) -> bool:
    """Return whether a training config uses the Phase 12C3B1 provenance contract."""
    return training_config.get("dataset_provenance") is not None


def resolve_explanation_dataset_provenance(
    preprocessed: Mapping[str, Any],
    *,
    path: str | Path,
    training_config: Mapping[str, Any],
    is_null_baseline_flag: bool,
) -> dict[str, Any] | None:
    """Build and validate explanation ``dataset_provenance`` at the contract boundary.

    New provenance-aware training configs (with ``dataset_provenance``) get
    strict fail-closed validation via
    :func:`require_explanation_dataset_matches_training`. Historical configs
    that predate the contract keep historical ``explain.py`` behaviour:
    provenance is recorded best-effort (``None`` if the dataset cannot be
    classified) and the ``--is-null-baseline`` flag is not validated. Paired
    benchmark validation still rejects such runs.
    """
    if not training_has_dataset_provenance(training_config):
        try:
            return build_dataset_provenance(preprocessed, path=path)
        except ValueError:
            return None
    provenance = build_dataset_provenance(preprocessed, path=path)
    require_explanation_dataset_matches_training(
        training_config=training_config,
        explanation_provenance=provenance,
        is_null_baseline_flag=is_null_baseline_flag,
    )
    return provenance


def require_explanation_dataset_matches_training(
    *,
    training_config: Mapping[str, Any],
    explanation_provenance: Mapping[str, Any],
    is_null_baseline_flag: bool,
) -> None:
    """Fail closed when a provenance-aware explanation contradicts its training run.

    Applies only when the saved training config records ``dataset_provenance``
    (Phase 12C3B1+ runs): the training run's null status must equal the
    ``--is-null-baseline`` flag and the explanation dataset bytes must be the
    exact training dataset bytes. Historical configs without
    ``dataset_provenance`` are not validated here, preserving historical
    ``explain.py`` behaviour.
    """
    training = training_config.get("dataset_provenance")
    if training is None:
        return
    if not isinstance(training, Mapping):
        raise ValueError("training config dataset_provenance must be a mapping")
    training_is_null = training.get("is_null_baseline")
    if not isinstance(training_is_null, bool):
        raise ValueError("training config dataset_provenance.is_null_baseline must be a bool")
    flag = bool(is_null_baseline_flag)
    if training_is_null != flag:
        if training_is_null:
            raise ValueError(
                "checkpoint was trained on a null baseline dataset; pass --is-null-baseline"
            )
        raise ValueError("--is-null-baseline was given but the checkpoint was trained on real data")
    if (
        training.get("preprocessed_data_sha256")
        != explanation_provenance["preprocessed_data_sha256"]
    ):
        raise ValueError(
            "explanation --preprocessed-data bytes do not match the training dataset "
            "(dataset_provenance.preprocessed_data_sha256 mismatch)"
        )
    if bool(explanation_provenance["is_null_baseline"]) != flag:
        raise ValueError(
            "explanation dataset null status contradicts --is-null-baseline "
            "for a provenance-aware training run"
        )
