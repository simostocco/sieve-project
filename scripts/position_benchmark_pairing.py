"""Real/null pair compatibility validation for paired positional benchmarks.

Phase 12C3B1. For each positional strategy ``m`` a schema-v2 benchmark trains

    REAL: Train(X, y,      S, strategy_m, protocol)
    NULL: Train(X, y_perm, S, strategy_m, protocol)

on one shared, benchmark-level null artifact. Before calibration may consume a
completed pair, this module proves from saved artifacts that phenotype
assignment is the only intended difference: same positional strategy, same
architecture and training protocol, same training seed, same replayed split
membership, same explicit fold, same content-only IG protocol, same variant
universe, same code revision, and a dataset relation bound to the benchmark
``null_binding`` (real bytes == lineage source bytes; null bytes == validated
null artifact bytes).

This module validates only. It never trains, explains, or runs
``bootstrap_null_calibration.py`` (whose mathematics stay unchanged); Phase
12C3B2 serializes the report as ``paired_compatibility.yaml`` and executes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in {None, ""}:
    from position_benchmark_metadata import (
        REQUIRED_CONTEXT_FIELDS,
        REQUIRED_EXPLANATION_CONTEXT_FIELDS,
        position_strategy_identity,
    )
else:
    from .position_benchmark_metadata import (
        REQUIRED_CONTEXT_FIELDS,
        REQUIRED_EXPLANATION_CONTEXT_FIELDS,
        position_strategy_identity,
    )

PAIRING_SCHEMA_VERSION = 1
VARIANT_UNIVERSE_SCHEMA = "sieve.variant_universe.v1"
PAIRED_CV_FOLD_INDEX = 0
PAIRED_CLASS_WEIGHTING = "off"
CONTENT_BASELINE_POLICY = "zero_content_observed_absolute_position"
NULL_METADATA_KIND_STRICT = "strict_v1"
NULL_METADATA_KIND_NONE = "none"
MISSING = "<missing>"

# Training-config fields that must be exactly equal between the real and null
# runs of one strategy. Dataset *paths* are excluded: real and null datasets are
# different files by design and are bound by byte hashes instead.
_DATASET_PATH_FIELDS = {"preprocessed_data", "vcf", "phenotypes"}
CONFIG_MUST_EQUAL_FIELDS = tuple(
    field for field in REQUIRED_CONTEXT_FIELDS if field not in _DATASET_PATH_FIELDS
) + (
    "input_dim",
    "position_encoding",
    "position_encoding_execution",
    "split_plan.schema_version",
    "split_plan.source",
    "split_plan.sha256",
    "split_plan.sample_ids_sha256",
    "split_plan.input_sha256",
    "dataset_provenance.schema_version",
    "dataset_provenance.sample_ids_sha256",
)

# Explanation-metadata fields that must be exactly equal. The whole
# integrated_gradients block is deterministic protocol metadata, so it is
# compared in full in addition to the named explanation-context fields.
EXPLANATION_MUST_EQUAL_FIELDS = tuple(REQUIRED_EXPLANATION_CONTEXT_FIELDS) + (
    "integrated_gradients",
    "genome_build",
    "max_variants_per_sample",
    "skip_ig",
    "skip_attention",
    "attention_threshold_mode",
    "attention_threshold",
    "attention_percentile",
    "model_provenance.checkpoint_selection_mode",
    "model_provenance.selected_fold",
    "dataset_provenance.schema_version",
    "dataset_provenance.sample_ids_sha256",
)


class PairCompatibilityError(ValueError):
    """Raised when a real/null pair (or pair set) is not scientifically compatible."""


@dataclass(frozen=True)
class PairSide:
    """Saved artifacts of one completed side (real or null) of a strategy pair.

    ``config`` is the root training ``config.yaml``; ``analysis_metadata`` is
    the explanation ``analysis_metadata.yaml``; ``variant_universe_sha256`` is
    :func:`variant_universe_sha256` of the explanation ``attributions.npz``;
    ``repository_revision`` is the code revision that produced the side (Phase
    12C3B2 stage records supply it; training/explanation do not record it).
    """

    config: Mapping[str, Any]
    analysis_metadata: Mapping[str, Any]
    variant_universe_sha256: str
    repository_revision: str


# ---------------------------------------------------------------------------
# Variant-universe identity
# ---------------------------------------------------------------------------


def variant_universe_sha256(attributions_path: str | Path) -> str:
    """Return a sample-boundary-preserving fingerprint of an ``attributions.npz``.

    The canonical payload is one header record followed by one record per
    sample, each serialized as canonical JSON on its own line::

        {"n_samples": N, "schema": "sieve.variant_universe.v1"}
        {"chromosomes": [...], "gene_ids": [...], "n_variants": k,
         "positions": [...], "sample_id": "...", "sample_index": i}

    Labels and attribution scores are deliberately excluded: they differ
    between real and null by design. Two explanations share a fingerprint only
    if every sample carries the exact same ordered (chromosome, position,
    gene_id) variant sequence, which is what bootstrap calibration joins on.
    """
    with np.load(Path(attributions_path), allow_pickle=True) as data:
        if "variant_scores" not in data or "metadata" not in data:
            raise PairCompatibilityError(
                f"{attributions_path} must contain 'variant_scores' and 'metadata'"
            )
        variant_scores = data["variant_scores"]
        metadata = data["metadata"]
    if len(variant_scores) != len(metadata):
        raise PairCompatibilityError(
            f"{attributions_path} has inconsistent variant_scores/metadata lengths"
        )

    digest = hashlib.sha256()
    digest.update(_canonical_json({"schema": VARIANT_UNIVERSE_SCHEMA, "n_samples": len(metadata)}))
    for sample_index, (scores_obj, meta_obj) in enumerate(
        zip(variant_scores, metadata, strict=True)
    ):
        meta = _unwrap_object(meta_obj)
        if not isinstance(meta, Mapping):
            raise PairCompatibilityError(f"sample {sample_index} metadata must be a mapping")
        if int(meta.get("sample_idx", -1)) != sample_index:
            raise PairCompatibilityError(
                f"sample {sample_index} metadata sample_idx does not match its position"
            )
        chromosomes = [str(value) for value in np.asarray(meta["chromosomes"]).tolist()]
        positions = [int(value) for value in np.asarray(meta["positions"]).tolist()]
        gene_ids = [int(value) for value in np.asarray(meta["gene_ids"]).tolist()]
        n_scores = len(np.asarray(_unwrap_object(scores_obj)))
        if not len(chromosomes) == len(positions) == len(gene_ids) == n_scores:
            raise PairCompatibilityError(
                f"sample {sample_index} has mismatched variant/metadata lengths"
            )
        digest.update(
            _canonical_json(
                {
                    "sample_index": sample_index,
                    "sample_id": str(meta["sample_id"]),
                    "n_variants": len(positions),
                    "chromosomes": chromosomes,
                    "positions": positions,
                    "gene_ids": gene_ids,
                }
            )
        )
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Loading completed sides
# ---------------------------------------------------------------------------


def load_pair_side(
    training_dir: str | Path,
    explanation_dir: str | Path,
    *,
    repository_revision: str,
) -> PairSide:
    """Load one completed side and bind its explanation to the supplied training_dir.

    Proves the explanation was produced from *this* training directory: the
    recorded ``config_path`` must resolve to ``training_dir/config.yaml`` and
    its recorded ``config_sha256`` must equal the current config bytes; the
    recorded ``checkpoint_path`` must resolve to the checkpoint location that
    the recorded selection mode implies inside ``training_dir``, and its
    recorded ``checkpoint_sha256`` must equal the current checkpoint bytes.
    Anything else fails closed, so an explanation from run B can never be
    paired with the training config of run A.
    """
    training_dir = Path(training_dir).resolve()
    explanation_dir = Path(explanation_dir)
    expected_config_path = training_dir / "config.yaml"
    config = _load_yaml_mapping(expected_config_path)
    analysis_metadata = _load_yaml_mapping(explanation_dir / "analysis_metadata.yaml")
    provenance = analysis_metadata.get("model_provenance")
    if not isinstance(provenance, Mapping):
        raise PairCompatibilityError(
            f"{explanation_dir / 'analysis_metadata.yaml'} is missing model_provenance"
        )

    recorded_config = provenance.get("config_path")
    if not isinstance(recorded_config, str) or not recorded_config:
        raise PairCompatibilityError("model_provenance.config_path must be a non-empty path")
    if Path(recorded_config).resolve() != expected_config_path:
        raise PairCompatibilityError(
            "explanation model_provenance.config_path does not belong to the supplied "
            f"training_dir: expected {expected_config_path}, recorded {recorded_config}"
        )
    recorded_config_sha = provenance.get("config_sha256")
    if not isinstance(recorded_config_sha, str) or not recorded_config_sha:
        raise PairCompatibilityError(
            "model_provenance.config_sha256 is required for paired benchmark provenance"
        )
    if _sha256_file(expected_config_path) != recorded_config_sha:
        raise PairCompatibilityError(
            f"training config bytes changed since explanation: {expected_config_path}"
        )

    expected_checkpoint = _expected_checkpoint_path(training_dir, provenance)
    recorded_checkpoint = provenance.get("checkpoint_path")
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint:
        raise PairCompatibilityError("model_provenance.checkpoint_path must be a non-empty path")
    if Path(recorded_checkpoint).resolve() != expected_checkpoint:
        raise PairCompatibilityError(
            "explanation model_provenance.checkpoint_path does not match the checkpoint "
            f"implied by its selection mode in training_dir: expected {expected_checkpoint}, "
            f"recorded {recorded_checkpoint}"
        )
    if not expected_checkpoint.is_file():
        raise PairCompatibilityError(f"explained checkpoint does not exist: {expected_checkpoint}")
    if _sha256_file(expected_checkpoint) != provenance.get("checkpoint_sha256"):
        raise PairCompatibilityError(
            f"explained checkpoint bytes changed since explanation: {expected_checkpoint}"
        )
    return PairSide(
        config=config,
        analysis_metadata=analysis_metadata,
        variant_universe_sha256=variant_universe_sha256(explanation_dir / "attributions.npz"),
        repository_revision=repository_revision,
    )


def _expected_checkpoint_path(training_dir: Path, provenance: Mapping[str, Any]) -> Path:
    """Return the checkpoint location inside *training_dir* implied by the selection mode.

    Only the two modes the paired benchmark can use are supported; best-fold
    and explicit-checkpoint selection fail closed here (and the pair rules
    reject them independently).
    """
    mode = provenance.get("checkpoint_selection_mode")
    selected_fold = provenance.get("selected_fold")
    if mode == "cv_explicit_fold":
        if (
            isinstance(selected_fold, bool)
            or not isinstance(selected_fold, int)
            or (selected_fold < 0)
        ):
            raise PairCompatibilityError(
                "cv_explicit_fold model_provenance.selected_fold must be a plain "
                f"non-negative integer, got {selected_fold!r}"
            )
        return (training_dir / f"fold_{selected_fold}" / "best_model.pt").resolve()
    if mode == "single_run_best_model":
        if selected_fold is not None:
            raise PairCompatibilityError(
                "single_run_best_model model_provenance.selected_fold must be null"
            )
        return (training_dir / "best_model.pt").resolve()
    raise PairCompatibilityError(
        f"checkpoint_selection_mode {mode!r} cannot be bound to a paired training_dir; "
        "expected cv_explicit_fold or single_run_best_model"
    )


# ---------------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------------


def compare_real_null_pair(
    *,
    run_id: str,
    real: PairSide,
    null: PairSide,
    null_binding: Mapping[str, Any],
    expected_fold_index: int | None,
) -> dict[str, Any]:
    """Return a deterministic compatibility report for one strategy's real/null pair.

    *null_binding* is the benchmark-level binding from the resolved plan.
    *expected_fold_index* is ``0`` for schema-v2 CV and ``None`` for single
    split. The report lists every violated rule; it never raises for ordinary
    incompatibility (see :func:`require_compatible_real_null_pair`).
    """
    binding = _binding_identity(null_binding)
    mismatches: list[dict[str, Any]] = []
    checked: list[str] = []

    def must_equal(field: str, real_value: Any, null_value: Any) -> None:
        checked.append(field)
        if (
            real_value is MISSING
            or null_value is MISSING
            or _canon(real_value) != _canon(null_value)
        ):
            mismatches.append(_mismatch(field, "must_equal", real_value, null_value))

    def must_be(field: str, side: str, actual: Any, expected: Any) -> None:
        checked.append(f"{side}:{field}")
        if actual is MISSING or _canon(actual) != _canon(expected):
            mismatches.append(
                {
                    "field": field,
                    "rule": "must_be",
                    "side": side,
                    "expected": expected,
                    "actual": actual,
                }
            )

    for field in CONFIG_MUST_EQUAL_FIELDS:
        must_equal(f"config.{field}", _lookup(real.config, field), _lookup(null.config, field))
    for field in EXPLANATION_MUST_EQUAL_FIELDS:
        must_equal(
            f"analysis.{field}",
            _lookup(real.analysis_metadata, field),
            _lookup(null.analysis_metadata, field),
        )

    real_strategy = _strategy_identity(real.config)
    null_strategy = _strategy_identity(null.config)
    must_equal("position_strategy_identity.hash", real_strategy, null_strategy)
    must_equal(
        "variant_universe_sha256", real.variant_universe_sha256, null.variant_universe_sha256
    )
    must_equal("repository_revision", real.repository_revision, null.repository_revision)

    for side_name, side in (("real", real), ("null", null)):
        _check_side_protocol(side_name, side, binding, expected_fold_index, must_be)

    _check_real_dataset(real, binding, must_be)
    _check_null_dataset(null, binding, must_be)

    checked.append("checkpoint_sha256")
    real_checkpoint = _lookup(real.analysis_metadata, "model_provenance.checkpoint_sha256")
    null_checkpoint = _lookup(null.analysis_metadata, "model_provenance.checkpoint_sha256")
    if real_checkpoint is MISSING or null_checkpoint is MISSING:
        mismatches.append(
            _mismatch("checkpoint_sha256", "required", real_checkpoint, null_checkpoint)
        )
    elif real_checkpoint == null_checkpoint:
        mismatches.append(
            _mismatch("checkpoint_sha256", "sanity_must_differ", real_checkpoint, null_checkpoint)
        )

    return {
        "schema_version": PAIRING_SCHEMA_VERSION,
        "run_id": run_id,
        "compatible": not mismatches,
        "null_binding": binding,
        "shared": {
            "position_strategy_hash": real_strategy,
            "split_plan_sha256": _lookup(real.config, "split_plan.sha256"),
            "sample_ids_sha256": _lookup(real.config, "split_plan.sample_ids_sha256"),
            "training_seed": _lookup(real.config, "seed"),
            "fold_index": expected_fold_index,
            "checkpoint_selection_mode": _lookup(
                real.analysis_metadata, "model_provenance.checkpoint_selection_mode"
            ),
            "resolved_ig_mode": _lookup(
                real.analysis_metadata, "integrated_gradients.resolved_ig_mode"
            ),
            "n_steps": _lookup(real.analysis_metadata, "integrated_gradients.n_steps"),
            "max_variants": _lookup(real.analysis_metadata, "integrated_gradients.max_variants"),
            "variant_universe_sha256": real.variant_universe_sha256,
            "repository_revision": real.repository_revision,
        },
        "real": _side_summary(real),
        "null": _side_summary(null),
        "checked_rules": sorted(set(checked)),
        "mismatches": sorted(mismatches, key=_mismatch_sort_key),
    }


def require_compatible_real_null_pair(
    *,
    run_id: str,
    real: PairSide,
    null: PairSide,
    null_binding: Mapping[str, Any],
    expected_fold_index: int | None,
) -> dict[str, Any]:
    """Return the compatibility report or raise :class:`PairCompatibilityError`."""
    report = compare_real_null_pair(
        run_id=run_id,
        real=real,
        null=null,
        null_binding=null_binding,
        expected_fold_index=expected_fold_index,
    )
    if report["compatible"]:
        return report
    for mismatch in report["mismatches"]:
        if mismatch["rule"] == "sanity_must_differ":
            raise PairCompatibilityError(
                f"run {run_id!r}: real and null checkpoints are byte-identical "
                f"(checkpoint_sha256 {mismatch['real']}); a completed real/null training "
                "pair is not expected to produce identical weights, so the pair is refused"
            )
    details = "; ".join(_format_mismatch(mismatch) for mismatch in report["mismatches"])
    raise PairCompatibilityError(f"run {run_id!r} real/null pair is incompatible: {details}")


def require_shared_null_across_pairs(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require every compatible strategy pair to share one null binding.

    Different strategies are expected to have different checkpoints and
    strategies; they must nevertheless reference exactly one null lineage,
    null artifact, source artifact, and sample identity.
    """
    if not reports:
        raise PairCompatibilityError("at least one pair report is required")
    run_ids = [report["run_id"] for report in reports]
    if len(set(run_ids)) != len(run_ids):
        raise PairCompatibilityError(f"duplicate run_id in pair reports: {run_ids}")
    incompatible = [report["run_id"] for report in reports if not report["compatible"]]
    if incompatible:
        raise PairCompatibilityError(f"incompatible pair reports: {incompatible}")
    bindings = {_canon(report["null_binding"]) for report in reports}
    if len(bindings) != 1:
        raise PairCompatibilityError(
            "strategy pairs reference different null bindings; a paired benchmark must "
            "use exactly one shared null artifact"
        )
    return {
        "schema_version": PAIRING_SCHEMA_VERSION,
        "run_ids": run_ids,
        "null_binding": dict(reports[0]["null_binding"]),
    }


# ---------------------------------------------------------------------------
# Rule groups
# ---------------------------------------------------------------------------


def _check_side_protocol(
    side_name: str,
    side: PairSide,
    binding: Mapping[str, Any],
    expected_fold_index: int | None,
    must_be,
) -> None:
    config = side.config
    analysis = side.analysis_metadata
    must_be("config.class_weighting", side_name, _lookup(config, "class_weighting"), "off")
    must_be("config.split_plan.source", side_name, _lookup(config, "split_plan.source"), "replayed")
    must_be(
        "config.split_plan.sample_ids_sha256",
        side_name,
        _lookup(config, "split_plan.sample_ids_sha256"),
        binding["sample_ids_sha256"],
    )
    must_be(
        "config.dataset_provenance.sample_ids_sha256",
        side_name,
        _lookup(config, "dataset_provenance.sample_ids_sha256"),
        binding["sample_ids_sha256"],
    )
    must_be(
        "analysis.annotation_level",
        side_name,
        _lookup(analysis, "annotation_level"),
        _lookup(config, "level"),
    )
    must_be(
        "analysis.genome_build",
        side_name,
        _lookup(analysis, "genome_build"),
        _lookup(config, "dataset_identity.genome_build"),
    )
    must_be("analysis.n_samples", side_name, _lookup(analysis, "n_samples"), binding["n_samples"])
    for field, expected in (
        ("integrated_gradients.executed", True),
        ("integrated_gradients.resolved_ig_mode", "content"),
        ("integrated_gradients.attribution_feature_space", "content"),
        ("integrated_gradients.baseline_policy", CONTENT_BASELINE_POLICY),
        ("integrated_gradients.comparability_warning", None),
    ):
        must_be(f"analysis.{field}", side_name, _lookup(analysis, field), expected)

    training_mode = _lookup(config, "position_encoding_execution.training_mode")
    if training_mode == "cv":
        must_be(
            "analysis.model_provenance.checkpoint_selection_mode",
            side_name,
            _lookup(analysis, "model_provenance.checkpoint_selection_mode"),
            "cv_explicit_fold",
        )
        must_be(
            "analysis.model_provenance.selected_fold",
            side_name,
            _lookup(analysis, "model_provenance.selected_fold"),
            PAIRED_CV_FOLD_INDEX,
        )
        must_be("expected_fold_index", side_name, expected_fold_index, PAIRED_CV_FOLD_INDEX)
    else:
        must_be(
            "analysis.model_provenance.checkpoint_selection_mode",
            side_name,
            _lookup(analysis, "model_provenance.checkpoint_selection_mode"),
            "single_run_best_model",
        )
        must_be("expected_fold_index", side_name, expected_fold_index, None)

    # The explanation must have consumed the exact training dataset bytes.
    for field in ("preprocessed_data_sha256", "is_null_baseline", "null_lineage"):
        must_be(
            f"analysis.dataset_provenance.{field}",
            side_name,
            _lookup(analysis, f"dataset_provenance.{field}"),
            _lookup(config, f"dataset_provenance.{field}"),
        )
    revision = side.repository_revision
    if not isinstance(revision, str) or revision in {"", "unknown"}:
        must_be("repository_revision", side_name, revision, "<known revision>")


def _check_real_dataset(real: PairSide, binding: Mapping[str, Any], must_be) -> None:
    config = real.config
    must_be(
        "config.dataset_provenance.is_null_baseline",
        "real",
        _lookup(config, "dataset_provenance.is_null_baseline"),
        False,
    )
    must_be(
        "config.dataset_provenance.null_metadata_kind",
        "real",
        _lookup(config, "dataset_provenance.null_metadata_kind"),
        NULL_METADATA_KIND_NONE,
    )
    must_be(
        "config.dataset_provenance.null_lineage",
        "real",
        _lookup(config, "dataset_provenance.null_lineage"),
        None,
    )
    must_be(
        "config.dataset_provenance.preprocessed_data_sha256",
        "real",
        _lookup(config, "dataset_provenance.preprocessed_data_sha256"),
        binding["source_artifact_sha256"],
    )
    must_be(
        "analysis.is_null_baseline",
        "real",
        _lookup(real.analysis_metadata, "is_null_baseline"),
        False,
    )


def _check_null_dataset(null: PairSide, binding: Mapping[str, Any], must_be) -> None:
    config = null.config
    must_be(
        "config.dataset_provenance.is_null_baseline",
        "null",
        _lookup(config, "dataset_provenance.is_null_baseline"),
        True,
    )
    must_be(
        "config.dataset_provenance.null_metadata_kind",
        "null",
        _lookup(config, "dataset_provenance.null_metadata_kind"),
        NULL_METADATA_KIND_STRICT,
    )
    must_be(
        "config.dataset_provenance.preprocessed_data_sha256",
        "null",
        _lookup(config, "dataset_provenance.preprocessed_data_sha256"),
        binding["null_artifact_sha256"],
    )
    must_be(
        "config.dataset_provenance.null_lineage.lineage_sha256",
        "null",
        _lookup(config, "dataset_provenance.null_lineage.lineage_sha256"),
        binding["lineage_sha256"],
    )
    must_be(
        "config.dataset_provenance.null_lineage.source_artifact_sha256",
        "null",
        _lookup(config, "dataset_provenance.null_lineage.source_artifact_sha256"),
        binding["source_artifact_sha256"],
    )
    must_be(
        "analysis.is_null_baseline",
        "null",
        _lookup(null.analysis_metadata, "is_null_baseline"),
        True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binding_identity(null_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the scientific identity of a plan ``null_binding`` (no paths)."""
    keys = (
        "lineage_sha256",
        "source_artifact_sha256",
        "null_artifact_sha256",
        "sample_ids_sha256",
        "n_samples",
    )
    missing = [key for key in keys if key not in null_binding]
    if missing:
        raise PairCompatibilityError(f"null_binding is missing required fields: {missing}")
    return {key: null_binding[key] for key in keys}


def _strategy_identity(config: Mapping[str, Any]) -> Any:
    try:
        return position_strategy_identity(config).hash
    except (KeyError, TypeError, ValueError):
        return MISSING


def _side_summary(side: PairSide) -> dict[str, Any]:
    return {
        "preprocessed_data_sha256": _lookup(
            side.config, "dataset_provenance.preprocessed_data_sha256"
        ),
        "is_null_baseline": _lookup(side.config, "dataset_provenance.is_null_baseline"),
        "null_lineage_sha256": _lookup(
            side.config, "dataset_provenance.null_lineage.lineage_sha256"
        ),
        "checkpoint_sha256": _lookup(side.analysis_metadata, "model_provenance.checkpoint_sha256"),
        "selected_fold_auc": _lookup(side.analysis_metadata, "model_provenance.selected_fold_auc"),
    }


def _lookup(data: Mapping[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def _mismatch(field: str, rule: str, real_value: Any, null_value: Any) -> dict[str, Any]:
    return {"field": field, "rule": rule, "real": real_value, "null": null_value}


def _mismatch_sort_key(mismatch: Mapping[str, Any]) -> tuple[str, str, str]:
    return (mismatch["field"], mismatch.get("side", ""), mismatch["rule"])


def _format_mismatch(mismatch: Mapping[str, Any]) -> str:
    if mismatch["rule"] == "must_be":
        return (
            f"{mismatch['side']} {mismatch['field']} must be {mismatch['expected']!r}, "
            f"got {mismatch['actual']!r}"
        )
    return (
        f"{mismatch['field']} ({mismatch['rule']}): real={mismatch['real']!r}, "
        f"null={mismatch['null']!r}"
    )


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PairCompatibilityError(f"required pair artifact does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise PairCompatibilityError(f"{path} must contain a YAML mapping")
    return dict(data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
