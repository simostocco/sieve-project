"""Read-only provenance gate for calibrated cross-strategy ranking comparison.

Phase 12C3C. A completed schema-v2 paired benchmark (Phase 12C3B2B) holds,
for every positional strategy, a bootstrap-calibrated variant ranking whose
``delta_rank`` column (``median_rank_null_boot - rank_real``) is the primary
calibrated prioritisation score. This module decides whether those calibrated
rankings may be compared ACROSS strategies. It never trusts a file because of
its name or location; it proves, from the benchmark's own execution records,
that each calibrated ranking CSV is exactly the output of the planned
calibration command run on exactly the artifacts that passed real/null pair
validation, and that every strategy shares one validated null binding.

Authority chain (all read-only)::

    execution/resolved_plan.yaml   exact bound plan bytes   (plan authority)
    execution/plan_binding.yaml    binds those bytes' SHA-256, revision, manifest
    plan["runs"]                   the authoritative run set and order
    runs/<id>/pair_validation      completed record -> paired_compatibility.yaml
    runs/<id>/calibration          completed record, planned argv, inputs equal
                                   the pair-validated fingerprints, outputs equal
                                   the current calibrated CSV / summary / gene stats
    benchmark/shared_null          completed record over exactly every pair record
                                   -> shared_null_validation.yaml

Every record must be a schema-valid ``completed`` record at its canonical
path with no sibling ``running``/``failed`` record, and must carry the bound
plan SHA-256, repository revision, and manifest SHA-256. Dependency record
hashes must equal the current record files, so any edit to an upstream record
breaks the chain.

What this proves and what it does not
-------------------------------------
Standalone use (``compare_ablation_rankings.py --position-calibrated-benchmark``
outside the executor) proves CONSISTENCY with the saved execution provenance:
the files read now are byte-identical to the ones the saved records describe,
and those records form the planned DAG. It does not cryptographically
authenticate who wrote the records (they are unsigned YAML) and it does not
require a live git checkout or take the execution lock. The authoritative
production result is the executor-run ``comparisons/calibrated_rankings``
stage, which additionally holds the lock, runs the repository gate, and
post-validates this module's reported record hashes against the records the
executor itself accepted.

Deliberately NOT re-hashed here: checkpoints and ``attributions.npz``. Their
identities are bound transitively through the unchanged pair-validation and
calibration record bytes (whose input fingerprints name them); full output
revalidation is the executor ``--resume`` path's job.

This module imports only the standard library, PyYAML, and the lightweight
benchmark record/pairing helpers. It deliberately does not import the
executor or planner modules, whose imports pull in the training stack; the
few stage-layout constants it needs are mirrored below and locked equal to
the executor's by tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    from position_benchmark_pairing import (
        PairCompatibilityError,
        require_shared_null_across_pairs,
    )
    from position_benchmark_records import (
        StageRecordError,
        existing_stage_records,
        file_fingerprint,
        require_completed_record,
        sha256_file,
        stage_record_path,
    )
else:
    from .position_benchmark_pairing import (
        PairCompatibilityError,
        require_shared_null_across_pairs,
    )
    from .position_benchmark_records import (
        StageRecordError,
        existing_stage_records,
        file_fingerprint,
        require_completed_record,
        sha256_file,
        stage_record_path,
    )

# Mirrors of executor/planner layout constants (see module docstring). Tests
# assert these equal position_benchmark_execution / position_benchmark_manifest.
PAIRED_SCHEMA_VERSION = 2
PLAN_BINDING_SCHEMA_VERSION = 1
EXECUTION_DIRNAME = "execution"
BOUND_PLAN_NAME = "resolved_plan.yaml"
PLAN_BINDING_NAME = "plan_binding.yaml"
NULL_BINDING_DIRNAME = "null_binding"
SHARED_NULL_NAME = "shared_null_validation.yaml"
NULL_VALIDATION_STAGE_ID = "benchmark/null_validation"
SHARED_NULL_STAGE_ID = "benchmark/shared_null"
NULL_VALIDATOR_NAME = "src.data.null_lineage.validate_null_pair"
PAIR_VALIDATOR_NAME = "scripts.position_benchmark_pairing.require_compatible_real_null_pair"
SHARED_NULL_VALIDATOR_NAME = "scripts.position_benchmark_pairing.require_shared_null_across_pairs"
NULL_BINDING_IDENTITY_FIELDS = (
    "lineage_sha256",
    "source_artifact_sha256",
    "null_artifact_sha256",
    "sample_ids_sha256",
    "n_samples",
)
CALIBRATION_RANKINGS_NAME = "bootstrap_calibrated_variant_rankings.csv"
CALIBRATION_GENE_STATS_NAME = "bootstrap_calibrated_variant_rankings_gene_stats.csv"
CALIBRATION_SUMMARY_NAME = "bootstrap_calibrated_variant_rankings_summary.yaml"
PAIRED_COMPATIBILITY_NAME = "paired_compatibility.yaml"
CALIBRATION_OUTPUT_NAMES = (
    CALIBRATION_RANKINGS_NAME,
    CALIBRATION_GENE_STATS_NAME,
    CALIBRATION_SUMMARY_NAME,
)
# Calibration inputs that must be byte-identical to what pair validation saw.
PAIR_BOUND_CALIBRATION_INPUTS = (
    "real/sieve_variant_rankings.csv",
    "real/analysis_metadata.yaml",
    "null/attributions.npz",
)


class CalibratedProvenanceError(ValueError):
    """The benchmark's saved execution provenance does not authorize calibrated comparison."""


@dataclass(frozen=True)
class AcceptedRecord:
    """A completed stage record accepted by this gate, with the hash it was accepted at."""

    stage_id: str
    path: Path
    sha256: str
    record: dict[str, Any]


@dataclass(frozen=True)
class CalibratedRunProvenance:
    """Validated calibrated-ranking inputs and provenance for one planned strategy run.

    ``config_path`` / ``analysis_metadata_path`` are the REAL model's training
    config and explanation metadata: the positional-strategy authority for the
    comparison. ``calibrated_ranking_path`` is the planned bootstrap output.
    """

    run_id: str
    config_path: Path
    config_sha256: str
    analysis_metadata_path: Path
    analysis_metadata_sha256: str
    calibrated_ranking_path: Path
    calibrated_ranking_sha256: str
    calibration_summary_path: Path
    calibration_summary_sha256: str
    calibration_gene_stats_path: Path
    calibration_gene_stats_sha256: str
    calibration_record_path: Path
    calibration_record_sha256: str
    pair_validation_record_path: Path
    pair_validation_record_sha256: str
    paired_compatibility_path: Path
    paired_compatibility_sha256: str
    n_calibrated_variants: int

    def provenance_dict(self) -> dict[str, Any]:
        """Return the per-run provenance fields written to the comparison YAML."""
        return {
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "analysis_metadata_path": str(self.analysis_metadata_path),
            "analysis_metadata_sha256": self.analysis_metadata_sha256,
            "calibrated_ranking_path": str(self.calibrated_ranking_path),
            "calibrated_ranking_sha256": self.calibrated_ranking_sha256,
            "calibration_summary_path": str(self.calibration_summary_path),
            "calibration_summary_sha256": self.calibration_summary_sha256,
            "calibration_record_path": str(self.calibration_record_path),
            "calibration_record_sha256": self.calibration_record_sha256,
            "pair_validation_record_path": str(self.pair_validation_record_path),
            "pair_validation_record_sha256": self.pair_validation_record_sha256,
            "paired_compatibility_path": str(self.paired_compatibility_path),
            "paired_compatibility_sha256": self.paired_compatibility_sha256,
        }


@dataclass(frozen=True)
class CalibratedBenchmark:
    """A completed benchmark whose calibrated rankings passed the provenance gate.

    ``runs`` follows plan order. ``consumed`` maps every file this gate (and
    the comparator after it) reads to the SHA-256 it was accepted at, so
    :func:`require_unchanged` can prove nothing changed while comparing.
    """

    benchmark_root: Path
    resolved_plan_path: Path
    resolved_plan_sha256: str
    plan_binding_path: Path
    plan_binding_sha256: str
    repository_revision: str
    manifest_file_sha256: str
    shared_null_record_path: Path
    shared_null_record_sha256: str
    shared_null_report_path: Path
    shared_null_report_sha256: str
    null_binding: dict[str, Any]
    runs: tuple[CalibratedRunProvenance, ...]
    consumed: tuple[tuple[str, str], ...]

    def execution_provenance(self) -> dict[str, Any]:
        """Return the ``execution_provenance`` block of the calibrated comparison YAML."""
        return {
            "benchmark_root": str(self.benchmark_root),
            "resolved_plan_path": str(self.resolved_plan_path),
            "resolved_plan_sha256": self.resolved_plan_sha256,
            "plan_binding_path": str(self.plan_binding_path),
            "plan_binding_sha256": self.plan_binding_sha256,
            "repository_revision": self.repository_revision,
            "manifest_file_sha256": self.manifest_file_sha256,
            "shared_null_record_path": str(self.shared_null_record_path),
            "shared_null_record_sha256": self.shared_null_record_sha256,
            "shared_null_report_path": str(self.shared_null_report_path),
            "shared_null_report_sha256": self.shared_null_report_sha256,
            "null_binding": dict(self.null_binding),
        }


# ---------------------------------------------------------------------------
# Planned stage layout (mirrors position_benchmark_execution.build_benchmark_stages)
# ---------------------------------------------------------------------------


def run_stage_id(run_id: str, name: str) -> str:
    """Return the executor stage ID ``runs/<run_id>/<name>``."""
    return f"runs/{run_id}/{name}"


def expected_pair_dependency_ids(run_ids: Sequence[str], index: int) -> tuple[str, ...]:
    """Return the B2B dependency IDs of ``runs/<run_ids[index]>/pair_validation``.

    Earlier strategies' pair validations are included so the cumulative
    shared-null check is bound to their exact reports.
    """
    run_id = run_ids[index]
    return (
        run_stage_id(run_id, "real_training"),
        run_stage_id(run_id, "real_explanation"),
        run_stage_id(run_id, "null_training"),
        run_stage_id(run_id, "null_explanation"),
        NULL_VALIDATION_STAGE_ID,
        *(run_stage_id(prior, "pair_validation") for prior in run_ids[:index]),
    )


def expected_calibration_dependency_ids(run_id: str) -> tuple[str, ...]:
    """Return the B2B dependency IDs of ``runs/<run_id>/calibration``."""
    return (run_stage_id(run_id, "pair_validation"), NULL_VALIDATION_STAGE_ID)


def expected_shared_null_dependency_ids(run_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the B2B dependency IDs of ``benchmark/shared_null``.

    Exactly every planned pair validation in plan order, followed by the null
    validation stage (the executor's DAG binds both).
    """
    return (
        *(run_stage_id(run_id, "pair_validation") for run_id in run_ids),
        NULL_VALIDATION_STAGE_ID,
    )


def planned_benchmark_root(plan: Mapping[str, Any]) -> Path:
    """Derive the benchmark root from planned directories, rejecting inconsistency.

    Same rule as the executor's ``benchmark_root_from_plan``: every run root
    is ``<root>/runs/<run_id>`` and every comparison directory is
    ``<root>/comparisons/<name>``.
    """
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise CalibratedProvenanceError("plan.runs must be a non-empty list")
    roots = set()
    for run in runs:
        run_root = Path(_plan_str(run, ("directories", "run_root")))
        if run_root.parent.name != "runs" or run_root.name != run.get("run_id"):
            raise CalibratedProvenanceError(f"unexpected run_root layout: {run_root}")
        roots.add(run_root.parent.parent)
    comparisons = plan.get("comparisons")
    if isinstance(comparisons, Mapping):
        for comparison in comparisons.values():
            directory = Path(_plan_str(comparison, ("directory",)))
            if directory.parent.name != "comparisons":
                raise CalibratedProvenanceError(f"unexpected comparison layout: {directory}")
            roots.add(directory.parent.parent)
    if len(roots) != 1:
        raise CalibratedProvenanceError(
            f"planned directories disagree on benchmark root: {sorted(roots)}"
        )
    root = roots.pop()
    if not root.is_absolute():
        raise CalibratedProvenanceError(f"benchmark root must be absolute: {root}")
    return root


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_calibrated_position_benchmark(benchmark_root: str | Path) -> CalibratedBenchmark:
    """Validate a completed paired benchmark for calibrated comparison and return its provenance.

    *benchmark_root* is ``<output_root>/<benchmark_id>/<annotation_level>``
    (not its ``execution/`` subdirectory) and must resolve exactly to the root
    implied by the bound plan. Raises :class:`CalibratedProvenanceError` (a
    ``ValueError``) naming the first broken link.
    """
    try:
        return _load(Path(benchmark_root))
    except CalibratedProvenanceError:
        raise
    except (StageRecordError, PairCompatibilityError, OSError, yaml.YAMLError) as error:
        raise CalibratedProvenanceError(f"calibrated benchmark provenance: {error}") from error


def require_unchanged(benchmark: CalibratedBenchmark) -> None:
    """Re-hash every consumed file and require the bytes accepted by the gate.

    The comparator calls this after it has parsed the rankings, so a file
    changed between validation and use is detected before outputs are written.
    """
    for path, expected in benchmark.consumed:
        try:
            current = sha256_file(path)
        except StageRecordError as error:
            raise CalibratedProvenanceError(f"consumed file disappeared: {error}") from error
        if current != expected:
            raise CalibratedProvenanceError(
                f"consumed file changed during calibrated comparison: {path}"
            )


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class _Gate:
    """Accumulates accepted records and consumed-file hashes for one validation pass."""

    def __init__(self, root: Path, plan: Mapping[str, Any], plan_sha256: str) -> None:
        self.root = root
        self.execution_dir = root / EXECUTION_DIRNAME
        self.plan = plan
        self.identity = {
            "resolved_plan_sha256": plan_sha256,
            "repository_revision": plan["repository_revision"],
            "manifest_file_sha256": plan["manifest_file_sha256"],
        }
        self.consumed: dict[str, str] = {}

    def consume(self, path: Path, sha256: str) -> None:
        self.consumed[str(path)] = sha256

    def accept(
        self, stage_id: str, *, stage_type: str, side: str, run_id: str | None
    ) -> AcceptedRecord:
        """Load one completed record and require its identity; no running/failed sibling."""
        found = existing_stage_records(self.execution_dir, stage_id)
        for kind in ("running", "failed"):
            if kind in found:
                raise CalibratedProvenanceError(
                    f"stage {stage_id} has a {kind} record ({found[kind]}); a calibrated "
                    "comparison requires a cleanly completed benchmark"
                )
        path = stage_record_path(self.execution_dir, stage_id, "completed")
        if "completed" not in found:
            raise CalibratedProvenanceError(f"stage {stage_id} has no completed record: {path}")
        record = require_completed_record(path)
        expected = {
            "stage_id": stage_id,
            "stage_type": stage_type,
            "side": side,
            "run_id": run_id,
            **self.identity,
        }
        for key, value in expected.items():
            if record[key] != value:
                raise CalibratedProvenanceError(
                    f"stage {stage_id} record {key} is {record[key]!r}, expected {value!r}"
                )
        sha256 = sha256_file(path)
        self.consume(path, sha256)
        return AcceptedRecord(stage_id, path, sha256, record)

    def require_in_process(self, accepted: AcceptedRecord, callable_name: str) -> None:
        execution = accepted.record["execution"]
        if execution != {"kind": "in_process", "callable": callable_name}:
            raise CalibratedProvenanceError(
                f"stage {accepted.stage_id} was not produced by {callable_name}: {execution!r}"
            )

    def require_dependencies(
        self,
        accepted: AcceptedRecord,
        expected_ids: Sequence[str],
        known: Mapping[str, AcceptedRecord],
    ) -> None:
        """Require exact dependency IDs/order and each dependency hash to equal the current record.

        Dependencies in *known* must additionally equal the hash this gate
        accepted them at; others are required to be the current bytes of the
        canonical record file.
        """
        dependencies = accepted.record["dependencies"]
        recorded_ids = [dependency["stage_id"] for dependency in dependencies]
        if recorded_ids != list(expected_ids):
            raise CalibratedProvenanceError(
                f"stage {accepted.stage_id} dependencies {recorded_ids} differ from the planned "
                f"dependencies {list(expected_ids)}"
            )
        for dependency in dependencies:
            dependency_id = dependency["stage_id"]
            canonical = stage_record_path(self.execution_dir, dependency_id, "completed")
            if dependency["record_path"] != str(canonical):
                raise CalibratedProvenanceError(
                    f"stage {accepted.stage_id} dependency {dependency_id} names a "
                    f"non-canonical record path: {dependency['record_path']}"
                )
            if dependency_id in known:
                current = known[dependency_id].sha256
            else:
                current = sha256_file(canonical)
            if dependency["record_sha256"] != current:
                raise CalibratedProvenanceError(
                    f"stage {accepted.stage_id} dependency record {dependency_id} changed after "
                    "it was bound"
                )

    def require_current_file(self, fingerprint: Mapping[str, Any], label: str) -> str:
        """Require a recorded file fingerprint to equal the current bytes; return its SHA-256."""
        try:
            current = file_fingerprint(fingerprint["path"])
        except StageRecordError as error:
            raise CalibratedProvenanceError(f"{label}: {error}") from error
        if current != dict(fingerprint):
            raise CalibratedProvenanceError(
                f"{label} changed since its stage completed: {fingerprint['path']} "
                f"(recorded {fingerprint.get('sha256')}, current {current['sha256']})"
            )
        self.consume(Path(fingerprint["path"]), current["sha256"])
        return current["sha256"]


def _load(supplied_root: Path) -> CalibratedBenchmark:
    execution_dir = supplied_root / EXECUTION_DIRNAME
    plan_path = execution_dir / BOUND_PLAN_NAME
    binding_path = execution_dir / PLAN_BINDING_NAME

    # 1. Bound plan bytes are the only plan authority.
    plan_fingerprint = file_fingerprint(plan_path)
    plan_bytes = Path(plan_path).read_bytes()
    plan = yaml.safe_load(plan_bytes)
    if not isinstance(plan, dict):
        raise CalibratedProvenanceError(f"bound plan must be a YAML mapping: {plan_path}")
    if _sha256_bytes(plan_bytes) != plan_fingerprint["sha256"]:
        raise CalibratedProvenanceError(f"bound plan changed while it was read: {plan_path}")
    plan_sha256 = plan_fingerprint["sha256"]
    version = plan.get("schema_version")
    if isinstance(version, bool) or version != PAIRED_SCHEMA_VERSION:
        raise CalibratedProvenanceError(
            "calibrated comparison requires a schema_version 2 (paired) bound plan"
        )
    for key in ("repository_revision", "manifest_file_sha256", "repository_root"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise CalibratedProvenanceError(f"bound plan {key} must be a non-empty string")

    # 2. The supplied root must be exactly the plan's root; all paths below
    #    are built from the plan root so they match the executor's records.
    root = planned_benchmark_root(plan)
    if supplied_root.resolve() != root:
        raise CalibratedProvenanceError(
            f"supplied benchmark root {supplied_root.resolve()} is not the bound plan's "
            f"benchmark root {root}"
        )
    gate = _Gate(root, plan, plan_sha256)
    bound_plan_path = gate.execution_dir / BOUND_PLAN_NAME
    gate.consume(bound_plan_path, plan_sha256)

    # 3. plan_binding.yaml must bind exactly these bytes.
    binding_sha256 = sha256_file(binding_path)
    binding = _load_mapping(binding_path, "plan binding")
    expected_binding = {
        "schema_version": PLAN_BINDING_SCHEMA_VERSION,
        "resolved_plan_path": str(bound_plan_path),
        "resolved_plan_sha256": plan_sha256,
        "repository_revision": plan["repository_revision"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
    }
    for key, value in expected_binding.items():
        if not _same(binding.get(key), value):
            raise CalibratedProvenanceError(
                f"plan binding {key} is {binding.get(key)!r}, expected {value!r}"
            )
    plan_binding_path = gate.execution_dir / PLAN_BINDING_NAME
    gate.consume(plan_binding_path, binding_sha256)

    null_binding = _plan_null_binding(plan)
    runs = plan["runs"]
    run_ids = [_plan_str(run, ("run_id",)) for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise CalibratedProvenanceError(f"bound plan has duplicate run IDs: {run_ids}")

    known: dict[str, AcceptedRecord] = {}
    null_validation = gate.accept(
        NULL_VALIDATION_STAGE_ID, stage_type="null_validation", side="benchmark", run_id=None
    )
    gate.require_in_process(null_validation, NULL_VALIDATOR_NAME)
    known[NULL_VALIDATION_STAGE_ID] = null_validation

    run_provenance = []
    pair_reports = []
    for index, run in enumerate(runs):
        provenance, report = _validate_run(gate, run, run_ids, index, null_binding, known)
        run_provenance.append(provenance)
        pair_reports.append(report)

    shared = _validate_shared_null(gate, run_ids, null_binding, pair_reports, known)
    return CalibratedBenchmark(
        benchmark_root=root,
        resolved_plan_path=bound_plan_path,
        resolved_plan_sha256=plan_sha256,
        plan_binding_path=plan_binding_path,
        plan_binding_sha256=binding_sha256,
        repository_revision=plan["repository_revision"],
        manifest_file_sha256=plan["manifest_file_sha256"],
        shared_null_record_path=shared["record"].path,
        shared_null_record_sha256=shared["record"].sha256,
        shared_null_report_path=shared["report_path"],
        shared_null_report_sha256=shared["report_sha256"],
        null_binding=null_binding,
        runs=tuple(run_provenance),
        consumed=tuple(sorted(gate.consumed.items())),
    )


def _validate_run(
    gate: _Gate,
    run: Mapping[str, Any],
    run_ids: Sequence[str],
    index: int,
    null_binding: Mapping[str, Any],
    known: dict[str, AcceptedRecord],
) -> tuple[CalibratedRunProvenance, dict[str, Any]]:
    """Validate one run's pair-validation -> calibration chain and return its provenance."""
    plan = gate.plan
    run_id = run_ids[index]
    real_training = Path(_plan_str(run, ("directories", "training")))
    real_explanation = Path(_plan_str(run, ("directories", "explanation")))
    calibration_plan = run.get("calibration")
    if not isinstance(calibration_plan, Mapping):
        raise CalibratedProvenanceError(f"plan run {run_id!r} has no calibration block")
    paired_path = Path(_plan_str(calibration_plan, ("paired_compatibility",)))

    # --- pair validation --------------------------------------------------
    pair_id = run_stage_id(run_id, "pair_validation")
    pair = gate.accept(pair_id, stage_type="pair_validation", side="paired", run_id=run_id)
    gate.require_in_process(pair, PAIR_VALIDATOR_NAME)
    gate.require_dependencies(pair, expected_pair_dependency_ids(run_ids, index), known)
    known[pair_id] = pair
    pair_outputs = pair.record["outputs"]
    if list(pair_outputs) != [PAIRED_COMPATIBILITY_NAME]:
        raise CalibratedProvenanceError(
            f"stage {pair_id} outputs {list(pair_outputs)} are not [{PAIRED_COMPATIBILITY_NAME}]"
        )
    paired_fingerprint = pair_outputs[PAIRED_COMPATIBILITY_NAME]
    if paired_fingerprint["path"] != str(paired_path):
        raise CalibratedProvenanceError(
            f"stage {pair_id} report path {paired_fingerprint['path']} is not the planned "
            f"{paired_path}"
        )
    paired_sha256 = gate.require_current_file(paired_fingerprint, f"{run_id} {paired_path.name}")
    report = _load_mapping(paired_path, "paired compatibility report")
    if report.get("compatible") is not True:
        raise CalibratedProvenanceError(
            f"run {run_id!r} paired compatibility report is not passing"
        )
    if report.get("run_id") != run_id:
        raise CalibratedProvenanceError(
            f"paired compatibility report run_id {report.get('run_id')!r} is not {run_id!r}"
        )
    if not _same(report.get("null_binding"), null_binding):
        raise CalibratedProvenanceError(
            f"run {run_id!r} paired compatibility null binding differs from plan null_binding"
        )

    # The comparator re-reads the real config and analysis metadata: they must
    # be the exact bytes pair validation approved.
    pair_inputs = pair.record["inputs"]
    authority = {}
    for name, planned in (
        ("real/config.yaml", real_training / "config.yaml"),
        ("real/analysis_metadata.yaml", real_explanation / "analysis_metadata.yaml"),
    ):
        fingerprint = pair_inputs.get(name)
        if not isinstance(fingerprint, Mapping) or fingerprint.get("path") != str(planned):
            raise CalibratedProvenanceError(
                f"stage {pair_id} input {name} is not the planned file {planned}"
            )
        authority[name] = gate.require_current_file(fingerprint, f"{run_id} {name}")

    # --- calibration ------------------------------------------------------
    calibration_id = run_stage_id(run_id, "calibration")
    calibration = gate.accept(
        calibration_id, stage_type="calibration", side="paired", run_id=run_id
    )
    execution = calibration.record["execution"]
    if execution["kind"] != "subprocess":
        raise CalibratedProvenanceError(f"stage {calibration_id} is not a subprocess stage")
    if execution["argv"] != list(calibration_plan.get("argv") or []):
        raise CalibratedProvenanceError(
            f"stage {calibration_id} recorded argv differs from the planned calibration argv"
        )
    if execution["cwd"] != plan["repository_root"]:
        raise CalibratedProvenanceError(
            f"stage {calibration_id} cwd {execution['cwd']!r} is not the plan repository_root"
        )
    if execution["exit_code"] != 0:
        raise CalibratedProvenanceError(f"stage {calibration_id} did not exit 0")
    gate.require_dependencies(calibration, expected_calibration_dependency_ids(run_id), known)

    calibration_inputs = calibration.record["inputs"]
    for name in PAIR_BOUND_CALIBRATION_INPUTS:
        if not _same(calibration_inputs.get(name), pair_inputs.get(name)):
            raise CalibratedProvenanceError(
                f"stage {calibration_id} input {name} is not the pair-validated file"
            )
    if not _same(calibration_inputs.get(PAIRED_COMPATIBILITY_NAME), paired_fingerprint):
        raise CalibratedProvenanceError(
            f"stage {calibration_id} input {PAIRED_COMPATIBILITY_NAME} is not the pair "
            "validation output"
        )

    # Output path authority comes from the plan's expected_outputs mapping.
    planned_outputs = [Path(path) for path in calibration_plan.get("expected_outputs") or []]
    if [path.name for path in planned_outputs] != list(CALIBRATION_OUTPUT_NAMES):
        raise CalibratedProvenanceError(
            f"plan run {run_id!r} calibration expected_outputs are not the bootstrap outputs"
        )
    recorded_outputs = calibration.record["outputs"]
    if list(recorded_outputs) != list(CALIBRATION_OUTPUT_NAMES):
        raise CalibratedProvenanceError(
            f"stage {calibration_id} outputs {list(recorded_outputs)} differ from the planned "
            "bootstrap outputs"
        )
    output_sha256 = {}
    for planned in planned_outputs:
        fingerprint = recorded_outputs[planned.name]
        if fingerprint["kind"] != "file" or fingerprint["path"] != str(planned):
            raise CalibratedProvenanceError(
                f"stage {calibration_id} output {planned.name} is not the planned path {planned}"
            )
        output_sha256[planned.name] = gate.require_current_file(
            fingerprint, f"{run_id} {planned.name}"
        )
    rankings_path, gene_stats_path, summary_path = planned_outputs
    known[calibration_id] = calibration

    n_rows = _count_csv_rows(rankings_path)
    _recheck_calibration_summary(gate, run_id, summary_path, n_rows, null_binding)

    provenance = CalibratedRunProvenance(
        run_id=run_id,
        config_path=real_training / "config.yaml",
        config_sha256=authority["real/config.yaml"],
        analysis_metadata_path=real_explanation / "analysis_metadata.yaml",
        analysis_metadata_sha256=authority["real/analysis_metadata.yaml"],
        calibrated_ranking_path=rankings_path,
        calibrated_ranking_sha256=output_sha256[CALIBRATION_RANKINGS_NAME],
        calibration_summary_path=summary_path,
        calibration_summary_sha256=output_sha256[CALIBRATION_SUMMARY_NAME],
        calibration_gene_stats_path=gene_stats_path,
        calibration_gene_stats_sha256=output_sha256[CALIBRATION_GENE_STATS_NAME],
        calibration_record_path=calibration.path,
        calibration_record_sha256=calibration.sha256,
        pair_validation_record_path=pair.path,
        pair_validation_record_sha256=pair.sha256,
        paired_compatibility_path=paired_path,
        paired_compatibility_sha256=paired_sha256,
        n_calibrated_variants=n_rows,
    )
    return provenance, report


def _recheck_calibration_summary(
    gate: _Gate,
    run_id: str,
    summary_path: Path,
    n_rows: int,
    null_binding: Mapping[str, Any],
) -> None:
    """Lightweight consistency check of the bootstrap summary (its actual key names).

    This never recomputes bootstrap statistics; it only confirms the summary
    describes the planned calibration on the plan's null and the CSV it sits
    beside.
    """
    plan = gate.plan
    settings = plan.get("calibration")
    if not isinstance(settings, Mapping):
        raise CalibratedProvenanceError("bound plan has no calibration settings block")
    summary = _load_mapping(summary_path, "calibration summary")
    per_gene = summary.get("per_gene")
    checks = (
        ("n_bootstrap", summary.get("n_bootstrap"), settings.get("n_bootstrap")),
        ("n_null_samples", summary.get("n_null_samples"), null_binding["n_samples"]),
        ("genome_build", summary.get("genome_build"), _plan_str(plan, ("dataset", "genome_build"))),
        (
            "excluded_sex_chroms",
            summary.get("excluded_sex_chroms"),
            settings.get("exclude_sex_chroms"),
        ),
        (
            "per_gene.gene_delta_rank_aggregation",
            per_gene.get("gene_delta_rank_aggregation") if isinstance(per_gene, Mapping) else None,
            settings.get("gene_delta_rank_aggregation"),
        ),
        (
            "n_real_variants_missing_from_null",
            summary.get("n_real_variants_missing_from_null"),
            0,
        ),
        ("n_real_variants", summary.get("n_real_variants"), n_rows),
    )
    for label, actual, expected in checks:
        if not _same(actual, expected):
            raise CalibratedProvenanceError(
                f"run {run_id!r} calibration summary {label} is {actual!r}, expected {expected!r}"
            )


def _validate_shared_null(
    gate: _Gate,
    run_ids: Sequence[str],
    null_binding: Mapping[str, Any],
    pair_reports: Sequence[Mapping[str, Any]],
    known: Mapping[str, AcceptedRecord],
) -> dict[str, Any]:
    """Validate the final shared-null record over exactly every planned pair record."""
    shared = gate.accept(
        SHARED_NULL_STAGE_ID, stage_type="shared_null", side="benchmark", run_id=None
    )
    gate.require_in_process(shared, SHARED_NULL_VALIDATOR_NAME)
    gate.require_dependencies(shared, expected_shared_null_dependency_ids(run_ids), known)

    # Each pair report the shared-null stage read must be the pair's output.
    inputs = shared.record["inputs"]
    for run_id in run_ids:
        pair_output = known[run_stage_id(run_id, "pair_validation")].record["outputs"][
            PAIRED_COMPATIBILITY_NAME
        ]
        if not _same(inputs.get(f"{run_id}/{PAIRED_COMPATIBILITY_NAME}"), pair_output):
            raise CalibratedProvenanceError(
                f"stage {SHARED_NULL_STAGE_ID} did not read run {run_id!r}'s pair-validated report"
            )

    outputs = shared.record["outputs"]
    report_path = gate.root / NULL_BINDING_DIRNAME / SHARED_NULL_NAME
    if list(outputs) != [SHARED_NULL_NAME] or outputs[SHARED_NULL_NAME]["path"] != str(report_path):
        raise CalibratedProvenanceError(
            f"stage {SHARED_NULL_STAGE_ID} outputs are not the planned {report_path}"
        )
    report_sha256 = gate.require_current_file(outputs[SHARED_NULL_NAME], SHARED_NULL_NAME)
    saved = _load_mapping(report_path, "shared null report")
    recomputed = yaml.safe_load(
        yaml.safe_dump(require_shared_null_across_pairs(list(pair_reports)), sort_keys=False)
    )
    if not _same(saved, recomputed):
        raise CalibratedProvenanceError(
            f"{report_path} does not equal require_shared_null_across_pairs over the current "
            "pair reports"
        )
    if not _same(recomputed.get("null_binding"), null_binding):
        raise CalibratedProvenanceError("shared null binding differs from the plan null_binding")
    if recomputed.get("run_ids") != list(run_ids):
        raise CalibratedProvenanceError(
            f"shared null run_ids {recomputed.get('run_ids')} differ from the planned runs"
        )
    return {"record": shared, "report_path": report_path, "report_sha256": report_sha256}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _plan_null_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    binding = plan.get("null_binding")
    if not isinstance(binding, Mapping):
        raise CalibratedProvenanceError("bound plan has no null_binding mapping")
    missing = [key for key in NULL_BINDING_IDENTITY_FIELDS if key not in binding]
    if missing:
        raise CalibratedProvenanceError(f"bound plan null_binding is missing {missing}")
    return {key: binding[key] for key in NULL_BINDING_IDENTITY_FIELDS}


def _plan_str(data: Any, keys: Sequence[str]) -> str:
    current = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise CalibratedProvenanceError(f"bound plan field {'.'.join(keys)} is missing")
        current = current[key]
    if not isinstance(current, str) or not current:
        raise CalibratedProvenanceError(f"bound plan field {'.'.join(keys)} must be a string")
    return current


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise CalibratedProvenanceError(f"{label} {path} must contain a YAML mapping")
    return data


def _count_csv_rows(path: Path) -> int:
    """Return the number of data rows (header excluded) in a CSV file."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) is None:
            raise CalibratedProvenanceError(f"calibrated ranking CSV has no header: {path}")
        return sum(1 for _ in reader)


def _same(left: Any, right: Any) -> bool:
    """Type-strict equality for YAML-loaded values (``True`` never equals ``1``)."""
    return _canonical(left) == _canonical(right)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
