"""Execution foundation and stage DAG for paired (schema-v2) positional benchmarks.

Phase 12C3B2A built the safe substrate (plan authority, repository/output
gates, lock, environment probe, full null preflight, subprocess primitive,
stage records). Phase 12C3B2B (the second half of this module) wires it into
the paired benchmark stage DAG behind the public interface::

    python scripts/run_position_benchmark.py MANIFEST --execute-plan PLAN [--resume]

See the "Phase 12C3B2B" section below for the DAG, post-validation, the
calibration input hash gate, and the resume rules. Phase 12C3C appends one
final stage, ``comparisons/calibrated_rankings``: the provenance-gated
cross-strategy comparison of calibrated ``delta_rank`` rankings (see
"Phase 12C3C" below).

Execution authority
-------------------
A benchmark executes exactly one reviewed resolved plan, identified by the
SHA-256 of its persisted YAML bytes (``resolved_plan_file_sha256``). Before
execution the plan is strictly loaded (schema v2 only; v1 is never
executable), rebuilt from its recorded manifest with its recorded Python and
device, and required to deep-equal the persisted plan except for ``warnings``
(which legitimately change once outputs exist). This detects any change to
the manifest, split membership, input-file bytes, real/null artifacts, null
sidecar, repository revision, Python path, planned argv, or output paths
between review and execution. On first execution the exact plan bytes are
copied to ``<benchmark_root>/execution/resolved_plan.yaml`` and never
rewritten; every later invocation must present identical bytes.

Production plans must therefore be generated AFTER the final Phase 12C3B2
executor commit: ``repository_revision`` is execution-bound, so a plan built
at an earlier commit is a review fixture only and can never execute.

Repository and output gates
---------------------------
Execution requires the repository root and ``HEAD`` to equal the plan, and a
clean worktree, index, and untracked state (``git status --porcelain=v1
--untracked-files=all``). There is no dirty-worktree override. Benchmark
outputs and the persisted plan must lie outside the repository or be ignored
by git, otherwise the executor's own outputs would dirty the scientific
worktree. All git calls use argv lists with ``shell=False``.

Null preflight
--------------
Before any benchmark stage, the full 12C3A ``validate_null_pair`` (which
loads both cohorts) must pass and its recomputed identity must equal the
plan's ``null_binding`` exactly, and every ``input_files`` byte hash must still
match. The null artifact is never regenerated. The result is recorded in
``<benchmark_root>/null_binding/null_validation.yaml``, an execution-stage
record rather than scientific identity.

Split identity note: planner ``split_plan.membership_sha256`` and training
config ``split_plan.sha256`` (and ``input_sha256``) are all
``src.training.split_plan.split_plan_sha256()`` over the canonical membership
payload, which excludes ``seed`` and ``split_source``; 12C3B2B training
post-validation requires their equality. ``input_files.split_plan.sha256`` is separately the raw
file-byte hash of the reviewed YAML.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    from position_benchmark_manifest import (
        CALIBRATED_COMPARISON_OUTPUT_NAMES,
        CALIBRATED_RANKINGS_COMPARISON,
        CALIBRATED_SCORE_COLUMN,
        CALIBRATION_RANKINGS_NAME,
        PAIRED_SCHEMA_VERSION,
        BenchmarkManifestError,
        build_resolved_plan,
    )
    from position_benchmark_metadata import (
        position_strategy_identity,
        require_authoritative_position_metadata,
    )
    from position_benchmark_pairing import (
        PairCompatibilityError,
        load_pair_side,
        require_compatible_real_null_pair,
        require_shared_null_across_pairs,
    )
    from position_benchmark_records import (
        GIT_REVISION_RE,
        SHA256_RE,
        StageRecordError,
        atomic_write_bytes,
        atomic_write_yaml,
        build_stage_record,
        dependency_reference,
        directory_fingerprint,
        directory_manifest,
        dump_yaml_bytes,
        existing_stage_records,
        file_fingerprint,
        load_stage_record,
        require_completed_record,
        sha256_file,
        stage_record_path,
        utc_now_iso,
        validate_stage_id,
        write_stage_record,
    )
else:
    from .position_benchmark_manifest import (
        CALIBRATED_COMPARISON_OUTPUT_NAMES,
        CALIBRATED_RANKINGS_COMPARISON,
        CALIBRATED_SCORE_COLUMN,
        CALIBRATION_RANKINGS_NAME,
        PAIRED_SCHEMA_VERSION,
        BenchmarkManifestError,
        build_resolved_plan,
    )
    from .position_benchmark_metadata import (
        position_strategy_identity,
        require_authoritative_position_metadata,
    )
    from .position_benchmark_pairing import (
        PairCompatibilityError,
        load_pair_side,
        require_compatible_real_null_pair,
        require_shared_null_across_pairs,
    )
    from .position_benchmark_records import (
        GIT_REVISION_RE,
        SHA256_RE,
        StageRecordError,
        atomic_write_bytes,
        atomic_write_yaml,
        build_stage_record,
        dependency_reference,
        directory_fingerprint,
        directory_manifest,
        dump_yaml_bytes,
        existing_stage_records,
        file_fingerprint,
        load_stage_record,
        require_completed_record,
        sha256_file,
        stage_record_path,
        utc_now_iso,
        validate_stage_id,
        write_stage_record,
    )

EXECUTION_DIRNAME = "execution"
BOUND_PLAN_NAME = "resolved_plan.yaml"
PLAN_BINDING_NAME = "plan_binding.yaml"
LOCK_NAME = "execution.lock"
NULL_BINDING_DIRNAME = "null_binding"
NULL_VALIDATION_NAME = "null_validation.yaml"
NULL_VALIDATION_SCHEMA_VERSION = 1
PLAN_BINDING_SCHEMA_VERSION = 1
NULL_VALIDATOR_NAME = "src.data.null_lineage.validate_null_pair"
NULL_BINDING_IDENTITY_FIELDS = (
    "lineage_sha256",
    "source_artifact_sha256",
    "null_artifact_sha256",
    "sample_ids_sha256",
    "n_samples",
)
INPUT_FILE_KEYS = (
    "preprocessed_data",
    "null_artifact",
    "null_lineage_sidecar",
    "split_plan",
    "sex_map",
    "pc_map",
)
OPTIONAL_INPUT_FILE_KEYS = ("sex_map", "pc_map")
RUN_ARGV_KEYS = ("train_argv", "explain_argv", "null_train_argv", "null_explain_argv")
# Raw (real-only) comparisons. Kept under its historical name; the Phase
# 12C3C calibrated comparison is added separately so raw bookkeeping (for
# example the summary's raw_comparisons_completed) is unchanged.
COMPARISON_KEYS = ("performance", "raw_rankings", "raw_attributions")
CALIBRATED_COMPARISON_KEY = CALIBRATED_RANKINGS_COMPARISON
PLAN_COMPARISON_KEYS = (*COMPARISON_KEYS, CALIBRATED_COMPARISON_KEY)
INTERRUPT_GRACE_SECONDS = 30.0
TERMINATE_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 10.0
GROUP_POLL_SECONDS = 0.1
PROBE_TIMEOUT_SECONDS = 300.0

# Executed in the plan's Python interpreter to capture execution provenance.
# It prints exactly one JSON object and never touches benchmark data.
ENVIRONMENT_PROBE = """
import json, os, platform, socket, sys
info = {
    "executable": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
try:
    import torch
except Exception as error:
    info["torch_import_error"] = repr(error)
else:
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    available = bool(torch.cuda.is_available())
    info["cuda_available"] = available
    info["cuda_device_names"] = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if available
        else []
    )
print(json.dumps(info, sort_keys=True))
"""


class ExecutionFoundationError(RuntimeError):
    """Base class for execution-gate, plan-authority, and preflight failures."""


class PlanAuthorityError(ExecutionFoundationError):
    """The persisted plan is invalid, changed, or no longer matches its inputs."""


class RepositoryGateError(ExecutionFoundationError):
    """The repository is not in the exact clean state the plan was reviewed at."""


class OutputLocationError(ExecutionFoundationError):
    """A benchmark output location would dirty the scientific worktree."""


class ExecutionLockError(ExecutionFoundationError):
    """Another executor holds the benchmark lock."""


class EnvironmentProbeError(ExecutionFoundationError):
    """The execution environment cannot be probed or cannot satisfy the plan."""


class NullPreflightError(ExecutionFoundationError):
    """Full real/null validation failed or disagrees with the plan binding."""


class StageStateError(ExecutionFoundationError):
    """A stage cannot start because records or outputs already exist."""


# ---------------------------------------------------------------------------
# Persisted plan loading and rebuild authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedPlan:
    """A persisted plan whose bytes, structure, and rebuild equality were verified."""

    path: Path
    plan: dict[str, Any]
    plan_bytes: bytes
    resolved_plan_sha256: str


def load_persisted_execution_plan(path: str | Path) -> tuple[dict[str, Any], bytes, str]:
    """Strictly load a persisted plan for future execution; executes nothing.

    Returns ``(plan, exact_bytes, sha256_of_exact_bytes)``. Validation is
    structural: schema v2 only, a valid ``null_binding`` whose
    ``execution_authorized`` is still false (planning never authorizes),
    empty ``warnings``, valid ``input_files``, a full 40-hex repository
    revision, a resolved absolute Python path, and argv lists of strings.
    """
    plan_path = Path(path)
    try:
        plan_bytes = plan_path.read_bytes()
    except OSError as error:
        raise PlanAuthorityError(f"cannot read resolved plan {plan_path}: {error}") from error
    try:
        plan = yaml.safe_load(plan_bytes)
    except yaml.YAMLError as error:
        raise PlanAuthorityError(f"resolved plan is not valid YAML: {error}") from error
    if not isinstance(plan, dict):
        raise PlanAuthorityError("resolved plan must contain a YAML mapping")
    version = plan.get("schema_version")
    if isinstance(version, bool) or version != PAIRED_SCHEMA_VERSION:
        raise PlanAuthorityError(
            "only schema_version 2 (paired) plans can be executed; schema v1 remains "
            "real-only dry-run planning"
        )
    _validate_null_binding(plan.get("null_binding"))
    null_execution = plan.get("null_execution")
    if (
        not isinstance(null_execution, Mapping)
        or null_execution.get("execution_authorized") is not False
    ):
        raise PlanAuthorityError("null_execution.execution_authorized must be false in a plan")
    if plan.get("warnings") != []:
        raise PlanAuthorityError(
            "resolved plan has warnings (it was reviewed against existing outputs); "
            "regenerate it without --allow-existing-outputs"
        )
    _validate_input_files(plan)
    revision = plan.get("repository_revision")
    if not isinstance(revision, str) or not GIT_REVISION_RE.match(revision):
        raise PlanAuthorityError(f"repository_revision must be a full Git SHA, got {revision!r}")
    for key in ("repository_root", "manifest_path"):
        _require_absolute_path_string(plan.get(key), key)
    if not isinstance(plan.get("manifest_file_sha256"), str) or not SHA256_RE.match(
        plan["manifest_file_sha256"]
    ):
        raise PlanAuthorityError("manifest_file_sha256 must be a lowercase SHA-256")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise PlanAuthorityError("runtime must be a mapping")
    _require_absolute_path_string(runtime.get("python"), "runtime.python")
    if runtime.get("device") not in {"cuda", "cpu"}:
        raise PlanAuthorityError("runtime.device must be 'cuda' or 'cpu'")
    _validate_plan_argv(plan)
    benchmark_root_from_plan(plan)
    # Hash the bytes that were parsed, not a second read, so the returned
    # identity always describes exactly the validated content.
    return plan, plan_bytes, hashlib.sha256(plan_bytes).hexdigest()


def verify_plan_rebuild(
    plan_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    rebuild: Callable[..., dict[str, Any]] = build_resolved_plan,
) -> VerifiedPlan:
    """Load *plan_path* and prove it still equals a fresh rebuild of its manifest.

    The rebuild uses the plan's own ``manifest_path``, ``runtime.python``, and
    ``runtime.device`` with ``allow_existing_outputs=True`` (outputs of an
    in-progress benchmark may exist). If *manifest_path* is supplied it must
    resolve to the plan's manifest. The rebuilt plan is YAML round-tripped so
    both sides are compared in their persisted form; every difference except
    ``warnings`` rejects.
    """
    plan, plan_bytes, plan_sha = load_persisted_execution_plan(plan_path)
    recorded_manifest = Path(plan["manifest_path"])
    if manifest_path is not None and Path(manifest_path).resolve() != recorded_manifest:
        raise PlanAuthorityError(
            f"supplied manifest {Path(manifest_path).resolve()} is not the plan's manifest "
            f"{recorded_manifest}"
        )
    try:
        rebuilt = rebuild(
            recorded_manifest,
            python_override=plan["runtime"]["python"],
            device_override=plan["runtime"]["device"],
            allow_existing_outputs=True,
        )
    except BenchmarkManifestError as error:
        raise PlanAuthorityError(
            f"plan can no longer be rebuilt from its manifest: {error}"
        ) from error
    rebuilt = yaml.safe_load(yaml.safe_dump(rebuilt, sort_keys=False))
    persisted = copy.deepcopy(plan)
    persisted.pop("warnings", None)
    rebuilt.pop("warnings", None)
    differences = _deep_differences(persisted, rebuilt)
    if differences:
        shown = "; ".join(differences[:10])
        more = f" (+{len(differences) - 10} more)" if len(differences) > 10 else ""
        raise PlanAuthorityError(
            f"persisted plan differs from a fresh rebuild of its manifest: {shown}{more}"
        )
    return VerifiedPlan(
        path=Path(plan_path).resolve(),
        plan=plan,
        plan_bytes=plan_bytes,
        resolved_plan_sha256=plan_sha,
    )


def benchmark_root_from_plan(plan: Mapping[str, Any]) -> Path:
    """Derive ``<output_root>/<benchmark_id>/<annotation_level>`` from planned directories.

    Every run root must be ``<benchmark_root>/runs/<run_id>`` and every
    comparison directory ``<benchmark_root>/comparisons/<name>``; anything
    inconsistent rejects rather than guessing.
    """
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise PlanAuthorityError("plan.runs must be a non-empty list")
    roots = set()
    for run in runs:
        run_root = Path(
            _lookup_str(run, ("directories", "run_root"), "runs[].directories.run_root")
        )
        if run_root.parent.name != "runs" or run_root.name != run.get("run_id"):
            raise PlanAuthorityError(f"unexpected run_root layout: {run_root}")
        roots.add(run_root.parent.parent)
    comparisons = plan.get("comparisons")
    if isinstance(comparisons, Mapping):
        for name, comparison in comparisons.items():
            directory = Path(_lookup_str(comparison, ("directory",), f"comparisons.{name}"))
            if directory.parent.name != "comparisons":
                raise PlanAuthorityError(f"unexpected comparison layout: {directory}")
            roots.add(directory.parent.parent)
    if len(roots) != 1:
        raise PlanAuthorityError(f"planned directories disagree on benchmark root: {sorted(roots)}")
    root = roots.pop()
    if not root.is_absolute():
        raise PlanAuthorityError(f"benchmark root must be absolute: {root}")
    return root


def bind_benchmark_plan(benchmark_root: str | Path, verified: VerifiedPlan) -> dict[str, Any]:
    """Permanently bind *benchmark_root* to the verified plan's exact bytes.

    First binding creates ``execution/``, atomically copies the exact plan
    bytes to ``execution/resolved_plan.yaml``, then writes
    ``execution/plan_binding.yaml`` recording ``resolved_plan_sha256``. Neither
    file is ever rewritten. Later invocations must present byte-identical plan
    bytes. Never called during ``--dry-run``.
    """
    execution_dir = Path(benchmark_root) / EXECUTION_DIRNAME
    bound_path = execution_dir / BOUND_PLAN_NAME
    binding_path = execution_dir / PLAN_BINDING_NAME
    status = "verified"
    if os.path.lexists(bound_path):
        if bound_path.is_symlink() or not bound_path.is_file():
            raise PlanAuthorityError(f"bound plan is not a regular file: {bound_path}")
        if bound_path.read_bytes() != verified.plan_bytes:
            raise PlanAuthorityError(
                f"benchmark root is bound to a different plan: {bound_path} "
                f"(sha256 {sha256_file(bound_path)}), supplied {verified.resolved_plan_sha256}"
            )
    else:
        execution_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(bound_path, verified.plan_bytes, overwrite=False)
        status = "bound"
    if sha256_file(bound_path) != verified.resolved_plan_sha256:
        raise PlanAuthorityError(f"bound plan bytes do not hash to the verified plan: {bound_path}")

    expected_binding = {
        "schema_version": PLAN_BINDING_SCHEMA_VERSION,
        "resolved_plan_path": str(bound_path),
        "resolved_plan_sha256": verified.resolved_plan_sha256,
        "repository_revision": verified.plan["repository_revision"],
        "manifest_file_sha256": verified.plan["manifest_file_sha256"],
    }
    if os.path.lexists(binding_path):
        existing = _load_yaml_mapping(binding_path, "plan binding")
        comparable = {key: existing.get(key) for key in expected_binding}
        if comparable != expected_binding:
            raise PlanAuthorityError(
                f"plan binding record disagrees with bound plan: {binding_path}"
            )
    else:
        # A crash between the two writes leaves a bound plan without its
        # binding record; the plan bytes were just verified, so completing the
        # record here is safe and never rewrites anything.
        atomic_write_yaml(binding_path, {**expected_binding, "bound_at": utc_now_iso()})
    return {
        "status": status,
        "resolved_plan_path": str(bound_path),
        "plan_binding_path": str(binding_path),
        "resolved_plan_sha256": verified.resolved_plan_sha256,
    }


# ---------------------------------------------------------------------------
# Repository gate and output locations
# ---------------------------------------------------------------------------


GitRunner = Callable[..., subprocess.CompletedProcess]


def run_git(args: Sequence[str], *, cwd: str | Path, runner: GitRunner = subprocess.run):
    """Run ``git <args>`` as an argv list with ``shell=False`` and return the result."""
    try:
        return runner(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
    except OSError as error:
        raise RepositoryGateError(f"cannot run git: {error}") from error


def _git_output(args: Sequence[str], *, cwd: str | Path, runner: GitRunner) -> str:
    result = run_git(args, cwd=cwd, runner=runner)
    if result.returncode != 0:
        raise RepositoryGateError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def require_repository_gate(
    plan: Mapping[str, Any], *, runner: GitRunner = subprocess.run
) -> dict[str, str]:
    """Require the repository to be exactly the clean checkout the plan names.

    Checks ``git rev-parse --show-toplevel`` against ``plan.repository_root``,
    ``git rev-parse HEAD`` against ``plan.repository_revision`` (never
    ``unknown``), and an empty ``git status --porcelain=v1
    --untracked-files=all``. There is no dirty-worktree override.
    """
    revision = plan.get("repository_revision")
    if not isinstance(revision, str) or not GIT_REVISION_RE.match(revision):
        raise RepositoryGateError(f"plan repository_revision is not a full Git SHA: {revision!r}")
    repo_root = Path(str(plan.get("repository_root")))
    toplevel = Path(
        _git_output(["rev-parse", "--show-toplevel"], cwd=repo_root, runner=runner).strip()
    )
    if toplevel.resolve() != repo_root.resolve():
        raise RepositoryGateError(
            f"repository root {toplevel} does not match plan repository_root {repo_root}"
        )
    head = _git_output(["rev-parse", "HEAD"], cwd=repo_root, runner=runner).strip()
    if head != revision:
        raise RepositoryGateError(f"HEAD {head} does not match plan repository_revision {revision}")
    status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"], cwd=repo_root, runner=runner
    )
    dirty = [line for line in status.splitlines() if line.strip()]
    if dirty:
        preview = "; ".join(dirty[:10])
        raise RepositoryGateError(
            f"worktree, index, or untracked state is not clean ({len(dirty)} entries): {preview}"
        )
    return {"repository_root": str(toplevel.resolve()), "repository_revision": head}


def require_output_location_safe(
    path: str | Path, *, repository_root: str | Path, runner: GitRunner = subprocess.run
) -> str:
    """Require *path* to be outside the repository or ignored by git.

    Returns ``"outside_repository"`` or ``"git_ignored"``. A non-ignored path
    inside the repository is rejected because executor outputs there would
    make the scientific worktree dirty and fail every later repository gate.
    ``git check-ignore`` exit 1 (not ignored) and any error both reject.
    """
    target = Path(path).resolve()
    repo_root = Path(repository_root).resolve()
    if target != repo_root and repo_root not in target.parents:
        return "outside_repository"
    if target == repo_root:
        raise OutputLocationError(f"output location is the repository root: {target}")
    result = run_git(["check-ignore", "-q", str(target)], cwd=repo_root, runner=runner)
    if result.returncode == 0:
        return "git_ignored"
    if result.returncode == 1:
        raise OutputLocationError(
            f"output location is inside the repository and not ignored by git: {target}"
        )
    raise OutputLocationError(
        f"git check-ignore failed ({result.returncode}) for {target}: {result.stderr.strip()}"
    )


def require_execution_locations_safe(
    plan: Mapping[str, Any], plan_path: str | Path, *, runner: GitRunner = subprocess.run
) -> dict[str, str]:
    """Apply :func:`require_output_location_safe` to the benchmark root and the plan file."""
    repository_root = plan["repository_root"]
    return {
        "benchmark_root": require_output_location_safe(
            benchmark_root_from_plan(plan), repository_root=repository_root, runner=runner
        ),
        "resolved_plan": require_output_location_safe(
            plan_path, repository_root=repository_root, runner=runner
        ),
    }


# ---------------------------------------------------------------------------
# Execution lock
# ---------------------------------------------------------------------------


class ExecutionLock:
    """Exclusive ``fcntl.flock`` lock on ``<benchmark_root>/execution/execution.lock``.

    The kernel lock is the authority; the holder information written into the
    file (pid, hostname, start time, plan SHA) is diagnostic only. The lock is
    released when the descriptor closes, including when the process dies, so
    there is no stale-lock deletion heuristic and the file is never deleted.
    ``flock`` may be unreliable on some network filesystems (for example
    older NFS setups); run benchmarks from local storage.
    """

    def __init__(self, benchmark_root: str | Path, *, resolved_plan_sha256: str) -> None:
        self.path = Path(benchmark_root) / EXECUTION_DIRNAME / LOCK_NAME
        self.resolved_plan_sha256 = resolved_plan_sha256
        self._fd: int | None = None

    def acquire(self) -> ExecutionLock:
        """Acquire the lock without blocking or raise :class:`ExecutionLockError`."""
        if self._fd is not None:
            raise ExecutionLockError("execution lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise ExecutionLockError(
                f"another executor holds the benchmark lock {self.path}; "
                "wait for it to finish (the lock is released automatically if it dies)"
            ) from error
        except BaseException:
            os.close(fd)
            raise
        holder = {
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "started_at": utc_now_iso(),
            "resolved_plan_sha256": self.resolved_plan_sha256,
        }
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(holder, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        """Release the lock and close the descriptor (the file is left in place)."""
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        """Whether this object currently holds the lock."""
        return self._fd is not None

    def __enter__(self) -> ExecutionLock:
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Environment probe (execution provenance only)
# ---------------------------------------------------------------------------


def probe_execution_environment(
    plan: Mapping[str, Any], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> dict[str, Any]:
    """Probe the plan's Python interpreter and return execution provenance.

    Runs ``[plan.runtime.python, "-c", ENVIRONMENT_PROBE]`` with
    ``shell=False``. The result is execution provenance only: it never enters
    positional strategy identity or any scientific hash, and the interpreter
    binary is not hashed. A plan requesting ``cuda`` on a host where CUDA is
    unavailable fails here, before any benchmark stage.
    """
    python = str(plan["runtime"]["python"])
    try:
        result = runner(
            [python, "-c", ENVIRONMENT_PROBE],
            capture_output=True,
            text=True,
            shell=False,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EnvironmentProbeError(f"environment probe could not run {python}: {error}") from error
    if result.returncode != 0:
        raise EnvironmentProbeError(
            f"environment probe failed ({result.returncode}): {result.stderr.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        info = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise EnvironmentProbeError("environment probe did not print a JSON object") from error
    if not isinstance(info, dict):
        raise EnvironmentProbeError("environment probe output must be a JSON object")
    if "torch_import_error" in info:
        raise EnvironmentProbeError(
            f"plan Python cannot import torch: {info['torch_import_error']}"
        )
    required = (
        "executable",
        "python_version",
        "platform",
        "hostname",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "cuda_device_names",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise EnvironmentProbeError(f"environment probe output is missing {missing}")
    environment = {
        "plan_python": python,
        "probe_executable": info["executable"],
        "python_version": info["python_version"],
        "torch_version": info["torch_version"],
        "torch_cuda_version": info["torch_cuda_version"],
        "cuda_available": bool(info["cuda_available"]),
        "cuda_device_names": list(info["cuda_device_names"]),
        "platform": info["platform"],
        "hostname": info["hostname"],
        "cuda_visible_devices": info.get("cuda_visible_devices"),
    }
    if plan["runtime"]["device"] == "cuda" and not environment["cuda_available"]:
        raise EnvironmentProbeError(
            "plan runtime.device is 'cuda' but CUDA is unavailable to the plan Python"
        )
    return environment


# ---------------------------------------------------------------------------
# Full null preflight and null_validation.yaml
# ---------------------------------------------------------------------------


def verify_input_files(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every ``input_files`` raw-byte SHA-256 and require exact equality."""
    current: dict[str, Any] = {}
    for key in INPUT_FILE_KEYS:
        entry = plan["input_files"][key]
        if entry is None:
            current[key] = None
            continue
        try:
            fingerprint = file_fingerprint(entry["path"])
        except StageRecordError as error:
            raise NullPreflightError(f"input_files.{key}: {error}") from error
        if fingerprint["sha256"] != entry["sha256"]:
            raise NullPreflightError(
                f"input_files.{key} bytes changed since planning: {entry['path']} "
                f"(planned {entry['sha256']}, current {fingerprint['sha256']})"
            )
        current[key] = fingerprint
    return current


def run_null_preflight(
    plan: Mapping[str, Any], *, validator: Callable[..., Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Run the full 12C3A real/null validation and bind it to the plan.

    Requires every ``input_files`` hash to still match, the sidecar to sit at
    its deterministic ``sidecar_path_for(null)`` location, and
    ``validate_null_pair(real, null, sidecar)`` (which loads both artifacts)
    to succeed with ``lineage_sha256``, ``source_artifact_sha256``,
    ``null_artifact_sha256``, ``sample_ids_sha256``, and ``n_samples`` exactly
    equal to ``plan.null_binding``. It only reads; the null artifact is never
    regenerated or rewritten.
    """
    from src.data import null_lineage

    input_files = verify_input_files(plan)
    real_path = Path(plan["input_files"]["preprocessed_data"]["path"])
    null_path = Path(plan["input_files"]["null_artifact"]["path"])
    sidecar_path = null_lineage.sidecar_path_for(null_path)
    if str(sidecar_path) != plan["input_files"]["null_lineage_sidecar"]["path"]:
        raise NullPreflightError(
            f"null lineage sidecar {plan['input_files']['null_lineage_sidecar']['path']} is not "
            f"the deterministic sidecar path {sidecar_path}"
        )
    validate = validator or null_lineage.validate_null_pair
    try:
        report = dict(validate(real_path, null_path, sidecar_path))
    except ValueError as error:
        raise NullPreflightError(f"validate_null_pair failed: {error}") from error
    binding = plan["null_binding"]
    mismatches = [
        f"{key}: validator={report.get(key)!r}, plan={binding.get(key)!r}"
        for key in NULL_BINDING_IDENTITY_FIELDS
        if key not in report or report[key] != binding[key]
    ]
    if mismatches:
        raise NullPreflightError(
            "validate_null_pair report does not match plan null_binding: " + "; ".join(mismatches)
        )
    return {
        "validator": NULL_VALIDATOR_NAME,
        "validator_report": report,
        "input_files": input_files,
        "sidecar_path": str(sidecar_path),
    }


def build_null_validation_report(
    plan: Mapping[str, Any],
    *,
    resolved_plan_sha256: str,
    repository_revision: str,
    preflight: Mapping[str, Any],
    validated_at: str | None = None,
) -> dict[str, Any]:
    """Build the ``null_validation.yaml`` execution-stage report.

    This records that the full null preflight passed for one bound plan. It is
    execution provenance; no scientific identity is derived from it. The
    timestamp is the only non-deterministic field.
    """
    files = preflight["input_files"]
    report = preflight["validator_report"]
    split_plan = plan["split_plan"]
    return {
        "schema_version": NULL_VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "validated_at": validated_at or utc_now_iso(),
        "repository_revision": repository_revision,
        "resolved_plan_sha256": resolved_plan_sha256,
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "source": _path_sha(files["preprocessed_data"]),
        "null": _path_sha(files["null_artifact"]),
        "lineage_sidecar": _path_sha(files["null_lineage_sidecar"]),
        "lineage_sha256": report["lineage_sha256"],
        "sample_ids_sha256": report["sample_ids_sha256"],
        "n_samples": report["n_samples"],
        "split_plan": {
            "path": files["split_plan"]["path"],
            "file_sha256": files["split_plan"]["sha256"],
            "membership_sha256": split_plan["membership_sha256"],
            "sample_ids_sha256": split_plan["sample_ids_sha256"],
        },
        "validator": preflight["validator"],
        "validator_report": dict(report),
        "plan_binding_match": True,
    }


def null_validation_path(benchmark_root: str | Path) -> Path:
    """Return ``<benchmark_root>/null_binding/null_validation.yaml``."""
    return Path(benchmark_root) / NULL_BINDING_DIRNAME / NULL_VALIDATION_NAME


def write_null_validation_report(benchmark_root: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write ``null_validation.yaml`` once; an existing file is refused."""
    path = null_validation_path(benchmark_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_yaml(path, report)


# ---------------------------------------------------------------------------
# Generic subprocess stage primitive
# ---------------------------------------------------------------------------


@dataclass
class SubprocessStageResult:
    """Structured outcome of one subprocess stage invocation."""

    argv: list[str]
    cwd: str
    started_at: str
    completed_at: str
    exit_code: int | None
    stdout_log: dict[str, Any] | None
    stderr_log: dict[str, Any] | None
    interrupted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def execution_block(self) -> dict[str, Any]:
        """Return the ``execution`` mapping used by stage records."""
        logs = (
            None
            if self.stdout_log is None or self.stderr_log is None
            else {"stdout": self.stdout_log, "stderr": self.stderr_log}
        )
        return {
            "kind": "subprocess",
            "argv": list(self.argv),
            "cwd": self.cwd,
            "logs": logs,
            "exit_code": self.exit_code,
        }


class StageFailure(ExecutionFoundationError):
    """A subprocess stage exited non-zero (or failed to start)."""

    def __init__(self, message: str, result: SubprocessStageResult | None) -> None:
        super().__init__(message)
        self.result = result


class StageInterrupted(KeyboardInterrupt):
    """A subprocess stage was interrupted; propagates like ``KeyboardInterrupt``."""

    def __init__(self, result: SubprocessStageResult) -> None:
        super().__init__("stage interrupted")
        self.result = result


@dataclass(frozen=True)
class SubprocessLaunch:
    """Validated local launch conditions for one subprocess stage."""

    argv: list[str]
    cwd: Path
    log_dir: Path
    stdout_path: Path
    stderr_path: Path


def validate_launch_preconditions(
    argv: Sequence[str], *, cwd: str | Path, log_dir: str | Path, log_stem: str
) -> SubprocessLaunch:
    """Check every deterministic local launch condition without touching the filesystem.

    Validates that *argv* is a non-empty list of strings, *cwd* is an existing
    absolute directory, the log layout is sane (absolute *log_dir* that is, or
    would be created under, a real directory; a single-segment *log_stem*),
    and neither ``<log_stem>.stdout.log`` nor ``.stderr.log`` exists.
    :func:`execute_subprocess_stage` calls this before publishing a
    ``running`` record, so a launch that can never start leaves no record, no
    log, and no output behind. Raises :class:`StageStateError`.
    """
    argv_list = _require_argv(argv)
    cwd_path = Path(cwd)
    if not cwd_path.is_absolute():
        raise StageStateError(f"stage cwd must be an absolute path: {cwd_path}")
    if not cwd_path.is_dir():
        raise StageStateError(f"stage cwd must be an existing directory: {cwd_path}")
    if (
        not isinstance(log_stem, str)
        or not log_stem
        or "/" in log_stem
        or "\\" in log_stem
        or log_stem in {".", ".."}
    ):
        raise StageStateError(f"stage log stem must be a single path segment: {log_stem!r}")
    log_directory = Path(log_dir)
    if not log_directory.is_absolute():
        raise StageStateError(f"stage log directory must be absolute: {log_directory}")
    # The nearest existing ancestor must be a real directory, otherwise the
    # later mkdir would fail after the running record was published.
    ancestor = log_directory
    while not os.path.lexists(ancestor):
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise StageStateError(f"stage log directory path is blocked by a non-directory: {ancestor}")
    stdout_path = log_directory / f"{log_stem}.stdout.log"
    stderr_path = log_directory / f"{log_stem}.stderr.log"
    for path in (stdout_path, stderr_path):
        if os.path.lexists(path):
            raise StageStateError(f"refusing to overwrite existing stage log: {path}")
    return SubprocessLaunch(
        argv=argv_list,
        cwd=cwd_path,
        log_dir=log_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def run_subprocess_stage(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    log_dir: str | Path,
    log_stem: str,
    env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> SubprocessStageResult:
    """Run one planned command exactly as given and persist its logs.

    *argv* is passed through unchanged as a list with ``shell=False``; this
    primitive never rebuilds, reorders, or quotes commands. stdout and stderr
    are written directly to ``<log_dir>/<log_stem>.stdout.log`` and
    ``.stderr.log`` (refused if they already exist) so logs survive a crash of
    the executor; both are hashed afterwards. A non-zero exit raises
    :class:`StageFailure`.

    Each stage runs in its own POSIX session and process group
    (``start_new_session=True``) so worker processes it spawns (for example
    bootstrap calibration workers) can be stopped together. Because the
    stage is no longer in the terminal's foreground group, a terminal Ctrl-C
    reaches only the executor; on ``KeyboardInterrupt`` the executor signals
    the stage group with SIGINT, then SIGTERM, then SIGKILL, each after a
    grace period, and raises :class:`StageInterrupted`. It never signals its
    own process group. The primitive never deletes or modifies any output the
    child produced.
    """
    launch = validate_launch_preconditions(argv, cwd=cwd, log_dir=log_dir, log_stem=log_stem)
    argv_list, cwd_path = launch.argv, launch.cwd
    stdout_path, stderr_path = launch.stdout_path, launch.stderr_path
    launch.log_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    exit_code: int | None = None
    interrupted = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            process = popen(
                list(argv_list),
                cwd=str(cwd_path),
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
                env=None if env is None else dict(env),
            )
        except OSError as error:
            result = _stage_result(
                argv_list, cwd_path, started_at, None, stdout_path, stderr_path, False
            )
            raise StageFailure(f"stage failed to start: {error}", result) from error
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            exit_code = _stop_process_group(process)
    result = _stage_result(
        argv_list, cwd_path, started_at, exit_code, stdout_path, stderr_path, interrupted
    )
    if interrupted:
        raise StageInterrupted(result)
    if exit_code != 0:
        raise StageFailure(f"stage exited with code {exit_code}", result)
    return result


def ensure_stage_can_start(execution_dir: str | Path, stage_id: str, owned_paths: Sequence[Path]):
    """Fail closed unless the stage has no records and its owned outputs are absent or empty.

    Any existing ``running``, ``failed``, or ``completed`` record blocks a
    fresh start: running/failed require manual recovery, and reuse of a
    completed stage is a Phase 12C3B2B resume decision, not an implicit skip.
    """
    records = existing_stage_records(execution_dir, stage_id)
    if records:
        raise StageStateError(
            f"stage {stage_id} already has records {sorted(records)}; manual review required"
        )
    for path in owned_paths:
        if _has_partial_output(Path(path)):
            raise StageStateError(
                f"stage {stage_id} output already exists without a completed record: {path}"
            )


def execute_subprocess_stage(
    *,
    execution_dir: str | Path,
    stage_id: str,
    stage_type: str,
    run_id: str | None,
    side: str,
    argv: Sequence[str],
    cwd: str | Path,
    owned_paths: Sequence[Path],
    common: Mapping[str, Any],
    inputs: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> tuple[SubprocessStageResult, Path]:
    """Run one subprocess stage under running/failed record discipline.

    The sequence is fixed: (1) stage record/output preconditions, (2) local
    launch preconditions (:func:`validate_launch_preconditions`), (3) publish
    the ``running`` record, (4) launch. A deterministic precondition failure
    therefore leaves no record, no log, no subprocess, and no output change.
    After the running record exists, ``StageFailure`` and
    ``StageInterrupted`` publish a ``failed`` record and remove the running
    marker. Any other ordinary ``Exception`` is recorded best-effort as a
    ``failed`` record (reason ``unexpected_error``) and re-raised as
    :class:`StageFailure` chained to the original; if that failed record
    cannot be published, the running marker is left in place (fail closed)
    and :class:`StageRecordError` is raised. ``KeyboardInterrupt``,
    ``SystemExit``, and other ``BaseException`` are never converted. On success it returns ``(result, running_record_path)`` and
    leaves the running marker in place: the caller must post-validate and
    hash outputs, then call :func:`publish_completed_record`. This primitive
    therefore can never publish a ``completed`` record itself, and it never
    deletes partial outputs. *common* supplies ``repository_revision``,
    ``resolved_plan_sha256``, and ``manifest_file_sha256``.
    """
    ensure_stage_can_start(execution_dir, stage_id, owned_paths)
    log_dir = Path(execution_dir).resolve() / "logs" / Path(stage_id).parent
    log_stem = Path(stage_id).name
    launch = validate_launch_preconditions(argv, cwd=cwd, log_dir=log_dir, log_stem=log_stem)
    started_at = utc_now_iso()
    record_fields = {
        "stage_id": stage_id,
        "stage_type": stage_type,
        "run_id": run_id,
        "side": side,
        "repository_revision": common["repository_revision"],
        "resolved_plan_sha256": common["resolved_plan_sha256"],
        "manifest_file_sha256": common["manifest_file_sha256"],
        "inputs": inputs,
        "dependencies": dependencies,
        "environment": environment,
        "started_at": started_at,
    }
    running_path = write_stage_record(
        execution_dir,
        build_stage_record(
            record_kind="running",
            execution={
                "kind": "subprocess",
                "argv": list(launch.argv),
                "cwd": str(launch.cwd),
                "logs": None,
                "exit_code": None,
            },
            **record_fields,
        ),
    )
    try:
        result = run_subprocess_stage(
            launch.argv, cwd=launch.cwd, log_dir=log_dir, log_stem=log_stem, popen=popen
        )
    except StageInterrupted as interrupt:
        _publish_failure(
            execution_dir, record_fields, interrupt.result, "interrupted", owned_paths, None
        )
        running_path.unlink()
        raise
    except StageFailure as failure:
        _publish_failure(
            execution_dir,
            record_fields,
            failure.result,
            "subprocess_failed",
            owned_paths,
            str(failure),
        )
        running_path.unlink()
        raise
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        try:
            _publish_failure(
                execution_dir,
                record_fields,
                None,
                "unexpected_error",
                owned_paths,
                detail,
                launch=launch,
            )
        except Exception as publish_error:
            raise StageRecordError(
                f"stage {stage_id} failed unexpectedly ({detail}) and its failed record could "
                f"not be published ({type(publish_error).__name__}: {publish_error}); the "
                f"running marker is left at {running_path} for manual review"
            ) from error
        running_path.unlink()
        raise StageFailure(f"stage {stage_id} failed unexpectedly: {detail}", None) from error
    return result, running_path


def publish_completed_record(
    execution_dir: str | Path, running_path: str | Path, completed_record: Mapping[str, Any]
) -> Path:
    """Publish a validated ``completed`` record, then remove the stage's running marker.

    Callers must build *completed_record* only after post-validation and
    output hashing succeeded. The completed record is written atomically
    before the running marker is removed, so a crash in between leaves both
    files and the stage fails closed on the next invocation.
    """
    if completed_record.get("record_kind") != "completed":
        raise StageRecordError("publish_completed_record requires a completed record")
    expected_running = stage_record_path(execution_dir, completed_record["stage_id"], "running")
    if Path(running_path) != expected_running:
        raise StageRecordError(f"running marker {running_path} does not belong to this stage")
    path = write_stage_record(execution_dir, completed_record)
    Path(running_path).unlink()
    return path


def record_stage_failure(
    *,
    execution_dir: str | Path,
    running_path: str | Path,
    result: SubprocessStageResult,
    reason: str,
    owned_paths: Sequence[Path],
    exception: str | None,
) -> Path:
    """Publish a ``failed`` record for a stage that ran but failed post-validation.

    Used by callers when a zero-exit stage fails output validation. The running
    marker is replaced by the failed record; outputs are left untouched.
    """
    running = load_stage_record(running_path)
    if running["record_kind"] != "running":
        raise StageRecordError(f"{running_path} is not a running record")
    if Path(running_path) != stage_record_path(execution_dir, running["stage_id"], "running"):
        raise StageRecordError(f"running marker {running_path} is not at its canonical path")
    record_fields = {
        key: running[key]
        for key in (
            "stage_id",
            "stage_type",
            "run_id",
            "side",
            "repository_revision",
            "resolved_plan_sha256",
            "manifest_file_sha256",
            "inputs",
            "dependencies",
            "environment",
            "started_at",
        )
    }
    path = _publish_failure(execution_dir, record_fields, result, reason, owned_paths, exception)
    Path(running_path).unlink()
    return path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _publish_failure(
    execution_dir: str | Path,
    record_fields: Mapping[str, Any],
    result: SubprocessStageResult | None,
    reason: str,
    owned_paths: Sequence[Path],
    exception: str | None,
    *,
    launch: SubprocessLaunch | None = None,
) -> Path:
    """Write a ``failed`` record; never touches the stage's outputs.

    Without a subprocess result (an unexpected error before or during launch)
    the execution block is built from the validated *launch*, with log
    fingerprints only if both log files were already created.
    """
    if result is None:
        if launch is None:
            raise StageRecordError("a failed record needs a subprocess result or launch")
        logs = None
        if launch.stdout_path.is_file() and launch.stderr_path.is_file():
            logs = {
                "stdout": file_fingerprint(launch.stdout_path),
                "stderr": file_fingerprint(launch.stderr_path),
            }
        execution = {
            "kind": "subprocess",
            "argv": list(launch.argv),
            "cwd": str(launch.cwd),
            "logs": logs,
            "exit_code": None,
        }
        exit_code = None
    else:
        execution = result.execution_block()
        exit_code = result.exit_code
    failed = build_stage_record(
        record_kind="failed",
        execution=execution,
        completed_at=utc_now_iso(),
        failure={
            "reason": reason,
            "exit_code": exit_code,
            "exception": exception,
            "partial_outputs_present": any(_has_partial_output(Path(p)) for p in owned_paths),
        },
        **record_fields,
    )
    return write_stage_record(execution_dir, failed)


def _stage_result(
    argv: list[str],
    cwd: Path,
    started_at: str,
    exit_code: int | None,
    stdout_path: Path,
    stderr_path: Path,
    interrupted: bool,
) -> SubprocessStageResult:
    return SubprocessStageResult(
        argv=list(argv),
        cwd=str(cwd),
        started_at=started_at,
        completed_at=utc_now_iso(),
        exit_code=exit_code,
        stdout_log=file_fingerprint(stdout_path),
        stderr_log=file_fingerprint(stderr_path),
        interrupted=interrupted,
    )


def _stop_process_group(process: subprocess.Popen) -> int | None:
    """Stop an interrupted stage's whole process group and reap the leader.

    Sends SIGINT to the group, waits ``INTERRUPT_GRACE_SECONDS``; sends SIGTERM,
    waits ``TERMINATE_GRACE_SECONDS``; sends SIGKILL, waits
    ``KILL_GRACE_SECONDS``. Escalation stops as soon as no group member is
    left, including workers that outlive the leader. Further Ctrl-C presses
    during shutdown are ignored (main thread only) so the group is always
    handled. Returns the leader's exit code, or ``None`` if it cannot be
    reaped.
    """
    swap_handler = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if swap_handler else None
    try:
        for sig, grace in (
            (signal.SIGINT, INTERRUPT_GRACE_SECONDS),
            (signal.SIGTERM, TERMINATE_GRACE_SECONDS),
            (signal.SIGKILL, KILL_GRACE_SECONDS),
        ):
            if not _group_alive(process):
                break
            _signal_group(process, sig)
            if _wait_for_group_exit(process, grace):
                break
        try:
            return process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return process.poll()
    finally:
        if swap_handler:
            signal.signal(signal.SIGINT, previous)


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the stage's process group; never the executor's own group.

    With ``start_new_session=True`` the group ID equals the leader's PID. If
    that would be the executor's own group (which only a misconfigured launch
    could cause), only the direct child is signalled. A group or child that
    has already exited is not an error.
    """
    try:
        if process.pid == os.getpgrp():
            process.send_signal(sig)
        else:
            os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _group_alive(process: subprocess.Popen) -> bool:
    """Whether any member of the stage's process group is still alive.

    The leader is reaped first (``poll``) so an exited-but-unreaped leader
    (a zombie) does not count as alive. A group ID cannot be reused while any
    member exists, so probing it with signal 0 is safe.
    """
    leader_running = process.poll() is None
    if process.pid == os.getpgrp():
        return leader_running
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_exit(process: subprocess.Popen, timeout: float) -> bool:
    """Poll until the stage group is empty or *timeout* elapses; return whether it emptied."""
    deadline = time.monotonic() + timeout
    while True:
        if not _group_alive(process):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(GROUP_POLL_SECONDS, remaining))


def _has_partial_output(path: Path) -> bool:
    """Whether *path* holds anything: a file/symlink, or a non-empty directory."""
    if not os.path.lexists(path):
        return False
    if path.is_dir() and not path.is_symlink():
        return any(path.iterdir())
    return True


def _require_argv(argv: Any) -> list[str]:
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        raise StageStateError("argv must be a non-empty list of strings")
    return list(argv)


def _validate_null_binding(binding: Any) -> None:
    if not isinstance(binding, Mapping):
        raise PlanAuthorityError("null_binding must be a mapping")
    for key in NULL_BINDING_IDENTITY_FIELDS[:-1]:
        value = binding.get(key)
        if not isinstance(value, str) or not SHA256_RE.match(value):
            raise PlanAuthorityError(f"null_binding.{key} must be a lowercase SHA-256")
    n_samples = binding.get("n_samples")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise PlanAuthorityError("null_binding.n_samples must be a positive integer")
    for key in ("null_artifact_path", "sidecar_path"):
        _require_absolute_path_string(binding.get(key), f"null_binding.{key}")
    if binding.get("execution_authorized") is not False:
        raise PlanAuthorityError("null_binding.execution_authorized must be false in a plan")


def _validate_input_files(plan: Mapping[str, Any]) -> None:
    input_files = plan.get("input_files")
    if not isinstance(input_files, Mapping) or set(input_files) != set(INPUT_FILE_KEYS):
        raise PlanAuthorityError(f"input_files must have exactly the keys {list(INPUT_FILE_KEYS)}")
    for key in INPUT_FILE_KEYS:
        entry = input_files[key]
        if entry is None and key in OPTIONAL_INPUT_FILE_KEYS:
            continue
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise PlanAuthorityError(f"input_files.{key} must be {{path, sha256}}")
        _require_absolute_path_string(entry["path"], f"input_files.{key}.path")
        if not isinstance(entry["sha256"], str) or not SHA256_RE.match(entry["sha256"]):
            raise PlanAuthorityError(f"input_files.{key}.sha256 must be a lowercase SHA-256")
    binding = plan["null_binding"]
    expected = {
        ("preprocessed_data", "sha256"): binding["source_artifact_sha256"],
        ("null_artifact", "sha256"): binding["null_artifact_sha256"],
        ("null_artifact", "path"): binding["null_artifact_path"],
        ("null_lineage_sidecar", "path"): binding["sidecar_path"],
        ("preprocessed_data", "path"): _lookup_str(
            plan, ("dataset", "preprocessed_data"), "dataset.preprocessed_data"
        ),
        ("split_plan", "path"): _lookup_str(plan, ("split_plan", "input_path"), "split_plan"),
    }
    for (key, attribute), value in expected.items():
        if input_files[key][attribute] != value:
            raise PlanAuthorityError(
                f"input_files.{key}.{attribute} disagrees with the rest of the plan"
            )


def _validate_plan_argv(plan: Mapping[str, Any]) -> None:
    python = plan["runtime"]["python"]
    argvs: list[tuple[str, Any]] = []
    for index, run in enumerate(plan["runs"]):
        if not isinstance(run, Mapping):
            raise PlanAuthorityError(f"runs[{index}] must be a mapping")
        for key in RUN_ARGV_KEYS:
            argvs.append((f"runs[{index}].{key}", run.get(key)))
        calibration = run.get("calibration")
        argvs.append(
            (
                f"runs[{index}].calibration.argv",
                calibration.get("argv") if isinstance(calibration, Mapping) else None,
            )
        )
    comparisons = plan.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(PLAN_COMPARISON_KEYS):
        raise PlanAuthorityError(f"comparisons must have exactly {list(PLAN_COMPARISON_KEYS)}")
    for key in PLAN_COMPARISON_KEYS:
        argvs.append((f"comparisons.{key}.argv", comparisons[key].get("argv")))
    for name, argv in argvs:
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise PlanAuthorityError(f"{name} must be a non-empty list of strings")
        if argv[0] != python:
            raise PlanAuthorityError(f"{name}[0] must be runtime.python")


def _require_absolute_path_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise PlanAuthorityError(f"{name} must be an absolute path string")


def _lookup_str(data: Any, keys: Sequence[str], name: str) -> str:
    current = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise PlanAuthorityError(f"{name} is missing")
        current = current[key]
    if not isinstance(current, str) or not current:
        raise PlanAuthorityError(f"{name} must be a non-empty string")
    return current


def _deep_differences(left: Any, right: Any, path: str = "plan") -> list[str]:
    """Return dotted paths where two YAML-loaded structures differ (type-strict)."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}"
            if key not in left:
                differences.append(f"{child} added by rebuild")
            elif key not in right:
                differences.append(f"{child} missing from rebuild")
            else:
                differences.extend(_deep_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path} length {len(left)} != {len(right)}"]
        differences = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            differences.extend(_deep_differences(a, b, f"{path}[{index}]"))
        return differences
    if type(left) is not type(right) or left != right:
        return [f"{path}: persisted={left!r}, rebuilt={right!r}"]
    return []


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise PlanAuthorityError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(data, dict):
        raise PlanAuthorityError(f"{label} {path} must contain a YAML mapping")
    return data


def _path_sha(fingerprint: Mapping[str, Any]) -> dict[str, str]:
    return {"path": fingerprint["path"], "sha256": fingerprint["sha256"]}


# ===========================================================================
# Phase 12C3B2B: paired benchmark stage DAG, post-validation, and resume
# ===========================================================================
#
# Scientific invariant. For every positional strategy m the DAG runs
#
#     REAL: Train(X, y,      S, strategy_m, protocol)  -> explain
#     NULL: Train(X, y_perm, S, strategy_m, protocol)  -> explain
#
# on one shared null artifact, proves from saved artifacts that phenotype
# assignment is the only difference (pair validation), and only then runs the
# unchanged bootstrap calibration on the exact bytes that passed validation.
# The executor never rebuilds a command: every subprocess stage runs the argv
# list stored in the reviewed plan, byte for byte.
#
# Stage order is deterministic and strategy-major (manifest run order):
#
#     benchmark/null_validation
#     runs/<id>/real_training, real_explanation, null_training,
#               null_explanation, pair_validation, calibration   (per run)
#     benchmark/shared_null
#     comparisons/performance, comparisons/raw_rankings, comparisons/raw_attributions
#     comparisons/calibrated_rankings                             (Phase 12C3C)
#
# Execution stops at the first failure; there is no keep-going and no
# retry-failed mode. B1/B2/B3 stay real-only raw comparisons. The final
# calibrated comparison is the only stage that reads calibrated rankings
# across strategies, and only through the provenance gate.
#
# Execution-provenance layout under ``<benchmark_root>/execution/``:
#
#     stages/<stage_id>.{running,completed,failed}.yaml   stage records
#     logs/<stage_id>.{stdout,stderr}.log                  subprocess logs
#     manifests/<stage_id>.outputs.jsonl                   output-tree manifests
#
# Logs and manifests live outside every scientific output directory, so the
# fingerprint of a training/explanation/comparison tree never includes the
# executor's own bookkeeping (and writing a manifest cannot change the tree it
# describes).
#
# Manual recovery. A ``running`` or ``failed`` record always blocks execution
# (also under ``--resume``). To recover, inspect the stage's logs and outputs,
# then deliberately archive or remove that stage's record, its logs, its
# output manifest (if written), and its partial outputs, and start again.
# Nothing here ever deletes scientific outputs automatically.

NULL_VALIDATION_STAGE_ID = "benchmark/null_validation"
SHARED_NULL_STAGE_ID = "benchmark/shared_null"
SHARED_NULL_NAME = "shared_null_validation.yaml"
EXECUTION_SUMMARY_NAME = "benchmark_execution_summary.yaml"
# Version 2 (Phase 12C3C): the calibrated comparison became the final stage.
EXECUTION_SUMMARY_SCHEMA_VERSION = 2
MANIFESTS_DIRNAME = "manifests"
LOGS_DIRNAME = "logs"
STAGES_STATE_DIRNAMES = ("stages", LOGS_DIRNAME, MANIFESTS_DIRNAME)
PAIR_VALIDATOR_NAME = "scripts.position_benchmark_pairing.require_compatible_real_null_pair"
SHARED_NULL_VALIDATOR_NAME = "scripts.position_benchmark_pairing.require_shared_null_across_pairs"
RUN_STAGE_NAMES = (
    "real_training",
    "real_explanation",
    "null_training",
    "null_explanation",
    "pair_validation",
    "calibration",
)
OUTPUT_TREE = "output_tree"
RESOLVED_PLAN_INPUT = "resolved_plan"
# train.py dumps vars(args) to config.yaml and then overwrites these CLI keys
# with richer run metadata (``split_plan`` becomes the split-plan metadata
# block), so they are validated through that metadata instead.
TRAIN_CONFIG_OVERRIDDEN_ARGS = ("split_plan",)
PAIRED_CLASS_WEIGHTING_OFF = "off"
# Phase 12C3C. ``comparisons/calibrated_rankings`` runs
# ``compare_ablation_rankings.py --position-calibrated-benchmark <root>
# --score-column delta_rank``. It depends on ``benchmark/shared_null`` and every
# ``runs/<id>/calibration`` record; its argv is held to the exact planned shape
# (require_calibrated_comparison_argv); its inputs are named files only (never
# the benchmark root or execution/ tree, which would be self-referential); and
# post-validation requires the comparison YAML to report exactly the record
# hashes this invocation accepted. Resume reuses the normal completed-stage
# revalidation, so any calibration, pair, or shared-null record change
# invalidates it through the dependency hashes.
CALIBRATED_COMPARISON_STAGE_ID = f"comparisons/{CALIBRATED_COMPARISON_KEY}"
EXECUTION_SUMMARY_STATUS = (
    "paired_benchmark_complete_with_provenance_gated_calibrated_position_ranking_comparison"
)
_MISSING = "<missing>"


class StagePostValidationError(ExecutionFoundationError):
    """A stage ran but its outputs failed validation; its failed record was published."""


class CompletedStageMismatchError(StageStateError):
    """A completed stage record no longer matches its inputs, outputs, or dependencies."""


class CalibrationInputGateError(CompletedStageMismatchError):
    """Calibration inputs differ from the exact bytes that passed pair validation."""


# Errors that describe bad or changed stage artifacts rather than executor
# bugs. They produce a failed record and a concise CLI error; any other
# exception still produces a failed record but propagates unchanged.
_ORDINARY_STAGE_ERRORS = (ExecutionFoundationError, ValueError, OSError, yaml.YAMLError)


@dataclass(frozen=True)
class BenchmarkStage:
    """One node of the paired benchmark DAG.

    Subprocess stages carry the exact planned ``argv``; in-process validator
    stages carry a stable ``callable_name`` instead (never a fake argv).
    ``owned_paths`` are the outputs this stage alone may create; sibling
    stages' files in a shared directory are not contamination.
    ``dependency_ids`` are upstream stages whose completed-record hashes this
    stage binds.
    """

    stage_id: str
    stage_type: str
    side: str
    run_id: str | None
    argv: tuple[str, ...] | None
    callable_name: str | None
    owned_paths: tuple[Path, ...]
    dependency_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompletedStage:
    """A completed record accepted in this invocation, with the hash it was accepted at."""

    record_path: Path
    record_sha256: str
    record: dict[str, Any]


@dataclass
class _ExecutionContext:
    verified: VerifiedPlan
    benchmark_root: Path
    execution_dir: Path
    environment: dict[str, Any]
    preflight: dict[str, Any]
    git_runner: GitRunner
    popen: Callable[..., subprocess.Popen]
    completed: dict[str, CompletedStage] = field(default_factory=dict)

    @property
    def plan(self) -> dict[str, Any]:
        return self.verified.plan

    @property
    def common(self) -> dict[str, str]:
        return {
            "repository_revision": self.plan["repository_revision"],
            "resolved_plan_sha256": self.verified.resolved_plan_sha256,
            "manifest_file_sha256": self.plan["manifest_file_sha256"],
        }

    @property
    def bound_plan_path(self) -> Path:
        return self.execution_dir / BOUND_PLAN_NAME

    @property
    def repository_root(self) -> str:
        return self.plan["repository_root"]


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------


def build_benchmark_stages(plan: Mapping[str, Any]) -> list[BenchmarkStage]:
    """Return the deterministic, strategy-major stage DAG for a schema-v2 plan.

    Commands are taken verbatim from the plan (``train_argv``,
    ``explain_argv``, ``null_train_argv``, ``null_explain_argv``,
    ``calibration.argv``, ``comparisons.<name>.argv``); nothing is rebuilt.
    """
    root = benchmark_root_from_plan(plan)
    nv = NULL_VALIDATION_STAGE_ID
    stages = [
        BenchmarkStage(
            nv,
            "null_validation",
            "benchmark",
            None,
            None,
            NULL_VALIDATOR_NAME,
            (null_validation_path(root),),
            (),
        )
    ]
    pair_ids: list[str] = []
    calibration_ids: list[str] = []
    real_training_ids: list[str] = []
    real_ids: list[str] = []
    for run in plan["runs"]:
        run_id = run["run_id"]
        directories = run["directories"]
        ids = {name: f"runs/{run_id}/{name}" for name in RUN_STAGE_NAMES}
        calibration = run["calibration"]
        stages.extend(
            [
                BenchmarkStage(
                    ids["real_training"],
                    "training",
                    "real",
                    run_id,
                    tuple(run["train_argv"]),
                    None,
                    (Path(directories["training"]),),
                    (nv,),
                ),
                BenchmarkStage(
                    ids["real_explanation"],
                    "explanation",
                    "real",
                    run_id,
                    tuple(run["explain_argv"]),
                    None,
                    (Path(directories["explanation"]),),
                    (ids["real_training"], nv),
                ),
                BenchmarkStage(
                    ids["null_training"],
                    "training",
                    "null",
                    run_id,
                    tuple(run["null_train_argv"]),
                    None,
                    (Path(directories["null_training"]),),
                    (nv,),
                ),
                BenchmarkStage(
                    ids["null_explanation"],
                    "explanation",
                    "null",
                    run_id,
                    tuple(run["null_explain_argv"]),
                    None,
                    (Path(directories["null_explanation"]),),
                    (ids["null_training"], nv),
                ),
                # Earlier strategies' pair validations are dependencies so the
                # cumulative shared-null check is bound to their exact reports.
                BenchmarkStage(
                    ids["pair_validation"],
                    "pair_validation",
                    "paired",
                    run_id,
                    None,
                    PAIR_VALIDATOR_NAME,
                    (Path(calibration["paired_compatibility"]),),
                    (
                        ids["real_training"],
                        ids["real_explanation"],
                        ids["null_training"],
                        ids["null_explanation"],
                        nv,
                        *pair_ids,
                    ),
                ),
                BenchmarkStage(
                    ids["calibration"],
                    "calibration",
                    "paired",
                    run_id,
                    tuple(calibration["argv"]),
                    None,
                    tuple(Path(path) for path in calibration["expected_outputs"]),
                    (ids["pair_validation"], nv),
                ),
            ]
        )
        pair_ids.append(ids["pair_validation"])
        calibration_ids.append(ids["calibration"])
        real_training_ids.append(ids["real_training"])
        real_ids.extend([ids["real_training"], ids["real_explanation"]])
    stages.append(
        BenchmarkStage(
            SHARED_NULL_STAGE_ID,
            "shared_null",
            "benchmark",
            None,
            None,
            SHARED_NULL_VALIDATOR_NAME,
            (shared_null_validation_path(root),),
            (*pair_ids, nv),
        )
    )
    comparison_dependencies = {
        "performance": tuple(real_training_ids),
        "raw_rankings": tuple(real_ids),
        "raw_attributions": tuple(real_ids),
        # The calibrated comparison is the final scientific stage: it binds the
        # shared-null record (which binds every pair record) and every
        # calibration record, in plan order.
        CALIBRATED_COMPARISON_KEY: (SHARED_NULL_STAGE_ID, *calibration_ids),
    }
    for name in PLAN_COMPARISON_KEYS:
        comparison = plan["comparisons"][name]
        stages.append(
            BenchmarkStage(
                f"comparisons/{name}",
                "comparison",
                "comparison",
                None,
                tuple(comparison["argv"]),
                None,
                (Path(comparison["directory"]),),
                comparison_dependencies[name],
            )
        )
    for stage in stages:
        validate_stage_id(stage.stage_id)
    return stages


def shared_null_validation_path(benchmark_root: str | Path) -> Path:
    """Return ``<benchmark_root>/null_binding/shared_null_validation.yaml``."""
    return Path(benchmark_root) / NULL_BINDING_DIRNAME / SHARED_NULL_NAME


def output_manifest_path(execution_dir: str | Path, stage_id: str) -> Path:
    """Return ``<execution_dir>/manifests/<stage_id>.outputs.jsonl``."""
    validate_stage_id(stage_id)
    return Path(execution_dir) / MANIFESTS_DIRNAME / f"{stage_id}.outputs.jsonl"


# ---------------------------------------------------------------------------
# Plan navigation helpers
# ---------------------------------------------------------------------------


def _run_entry(plan: Mapping[str, Any], run_id: str | None) -> Mapping[str, Any]:
    for run in plan["runs"]:
        if run["run_id"] == run_id:
            return run
    raise PlanAuthorityError(f"plan has no run {run_id!r}")


def _side_directories(run: Mapping[str, Any], side: str) -> tuple[Path, Path]:
    directories = run["directories"]
    prefix = "" if side == "real" else "null_"
    return Path(directories[f"{prefix}training"]), Path(directories[f"{prefix}explanation"])


def _is_cv(plan: Mapping[str, Any]) -> bool:
    return plan["split_plan"]["mode"] == "cv"


def _training_mode(plan: Mapping[str, Any]) -> str:
    return "cv" if _is_cv(plan) else "single_split"


def _expected_fold_index(plan: Mapping[str, Any]) -> int | None:
    """Return the paired explanation fold (0 for CV, ``None`` for single split)."""
    if not _is_cv(plan):
        return None
    fold = plan["paired_policy"]["explanation_fold_index"]
    if fold != 0 or isinstance(fold, bool):
        raise PlanAuthorityError("paired CV plans must explain fold 0")
    return 0


def _selected_checkpoint(plan: Mapping[str, Any], training_dir: Path) -> Path:
    fold = _expected_fold_index(plan)
    if fold is None:
        return training_dir / "best_model.pt"
    return training_dir / f"fold_{fold}" / "best_model.pt"


def _side_dataset_key(side: str) -> str:
    return "preprocessed_data" if side == "real" else "null_artifact"


def _binding_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: plan["null_binding"][key] for key in NULL_BINDING_IDENTITY_FIELDS}


def _stage_by_id(stages: Sequence[BenchmarkStage], stage_id: str) -> BenchmarkStage:
    for stage in stages:
        if stage.stage_id == stage_id:
            return stage
    raise KeyError(stage_id)


# ---------------------------------------------------------------------------
# Input and output specifications
# ---------------------------------------------------------------------------

IOSpec = tuple[str, Path, str]


def _input_specs(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[IOSpec]:
    """Return ``(logical_name, path, kind)`` for every input a stage binds by bytes."""
    plan = ctx.plan
    files = plan["input_files"]
    resolved_plan = (RESOLVED_PLAN_INPUT, ctx.bound_plan_path, "file")
    if stage.stage_type == "null_validation":
        return [
            ("real_dataset", Path(files["preprocessed_data"]["path"]), "file"),
            ("null_artifact", Path(files["null_artifact"]["path"]), "file"),
            ("null_lineage_sidecar", Path(files["null_lineage_sidecar"]["path"]), "file"),
            ("split_plan", Path(files["split_plan"]["path"]), "file"),
            resolved_plan,
        ]
    if stage.stage_type == "training":
        specs = [
            ("dataset", Path(files[_side_dataset_key(stage.side)]["path"]), "file"),
            ("split_plan", Path(files["split_plan"]["path"]), "file"),
        ]
        for key in OPTIONAL_INPUT_FILE_KEYS:
            if files[key] is not None:
                specs.append((key, Path(files[key]["path"]), "file"))
        return [*specs, resolved_plan]
    if stage.stage_type == "explanation":
        run = _run_entry(plan, stage.run_id)
        training_dir, _ = _side_directories(run, stage.side)
        specs = [
            ("training/config.yaml", training_dir / "config.yaml", "file"),
            ("training/selected_checkpoint", _selected_checkpoint(plan, training_dir), "file"),
            ("dataset", Path(files[_side_dataset_key(stage.side)]["path"]), "file"),
        ]
        if files["pc_map"] is not None:
            specs.append(("pc_map", Path(files["pc_map"]["path"]), "file"))
        return [*specs, resolved_plan]
    if stage.stage_type == "pair_validation":
        run = _run_entry(plan, stage.run_id)
        real_training, real_explanation = _side_directories(run, "real")
        null_training, null_explanation = _side_directories(run, "null")
        return [
            ("real/config.yaml", real_training / "config.yaml", "file"),
            ("null/config.yaml", null_training / "config.yaml", "file"),
            ("real/analysis_metadata.yaml", real_explanation / "analysis_metadata.yaml", "file"),
            ("null/analysis_metadata.yaml", null_explanation / "analysis_metadata.yaml", "file"),
            ("real/attributions.npz", real_explanation / "attributions.npz", "file"),
            ("null/attributions.npz", null_explanation / "attributions.npz", "file"),
            (
                "real/sieve_variant_rankings.csv",
                real_explanation / "sieve_variant_rankings.csv",
                "file",
            ),
            ("real/checkpoint", _selected_checkpoint(plan, real_training), "file"),
            ("null/checkpoint", _selected_checkpoint(plan, null_training), "file"),
            ("null_validation.yaml", null_validation_path(ctx.benchmark_root), "file"),
            resolved_plan,
        ]
    if stage.stage_type == "calibration":
        run = _run_entry(plan, stage.run_id)
        _, real_explanation = _side_directories(run, "real")
        _, null_explanation = _side_directories(run, "null")
        return [
            (
                "real/sieve_variant_rankings.csv",
                real_explanation / "sieve_variant_rankings.csv",
                "file",
            ),
            ("real/analysis_metadata.yaml", real_explanation / "analysis_metadata.yaml", "file"),
            ("null/attributions.npz", null_explanation / "attributions.npz", "file"),
            (
                "paired_compatibility.yaml",
                Path(run["calibration"]["paired_compatibility"]),
                "file",
            ),
            resolved_plan,
        ]
    if stage.stage_type == "shared_null":
        specs = [
            (
                f"{run['run_id']}/paired_compatibility.yaml",
                Path(run["calibration"]["paired_compatibility"]),
                "file",
            )
            for run in plan["runs"]
        ]
        specs.append(("null_validation.yaml", null_validation_path(ctx.benchmark_root), "file"))
        return [*specs, resolved_plan]
    if stage.stage_type == "comparison":
        if stage.stage_id == CALIBRATED_COMPARISON_STAGE_ID:
            return [*_calibrated_comparison_input_specs(ctx), resolved_plan]
        return [*_comparison_input_specs(ctx, stage), resolved_plan]
    raise PlanAuthorityError(f"unknown stage type {stage.stage_type!r}")


def _comparison_input_specs(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[IOSpec]:
    """Bind every benchmark path named by a comparison argv, except its own outputs.

    The names are ``argv[<index>]`` so the record states exactly which planned
    token each hash belongs to. Directory tokens (B1 ``--run-dir`` training
    directories, B3 ``attributions_per_sample``) are fingerprinted as full
    directory manifests; a missing referenced path fails.
    """
    comparison_dir = stage.owned_paths[0]
    specs = []
    for index, token in enumerate(stage.argv or ()):
        path = Path(token)
        if not path.is_absolute() or not _is_within(path, ctx.benchmark_root):
            continue
        if _is_within(path, comparison_dir):
            continue
        specs.append((f"argv[{index}]", path, "directory" if path.is_dir() else "file"))
    return specs


def _calibrated_comparison_input_specs(ctx: _ExecutionContext) -> list[IOSpec]:
    """Bind the named files the calibrated comparator reads, never a directory.

    The calibrated argv names the benchmark root, which contains this stage's
    own records, logs, and outputs, so the generic argv-token discovery would
    fingerprint the executor's bookkeeping (self-reference). Instead every
    consumed file is listed explicitly; the upstream completed records are
    bound through dependency record hashes, not as inputs.
    """
    specs: list[IOSpec] = [
        (PLAN_BINDING_NAME, ctx.execution_dir / PLAN_BINDING_NAME, "file"),
        (SHARED_NULL_NAME, shared_null_validation_path(ctx.benchmark_root), "file"),
    ]
    for run in ctx.plan["runs"]:
        run_id = run["run_id"]
        real_training, real_explanation = _side_directories(run, "real")
        specs.extend(
            (f"{run_id}/{Path(path).name}", Path(path), "file")
            for path in run["calibration"]["expected_outputs"]
        )
        specs.extend(
            [
                (f"{run_id}/real/config.yaml", real_training / "config.yaml", "file"),
                (
                    f"{run_id}/real/analysis_metadata.yaml",
                    real_explanation / "analysis_metadata.yaml",
                    "file",
                ),
                (
                    f"{run_id}/{Path(run['calibration']['paired_compatibility']).name}",
                    Path(run["calibration"]["paired_compatibility"]),
                    "file",
                ),
            ]
        )
    return specs


def _planned_input_hashes(plan: Mapping[str, Any], stage: BenchmarkStage) -> dict[str, str]:
    """Return the reviewed ``input_files`` hash each plan-bound input must still have."""
    files = plan["input_files"]
    if stage.stage_type == "null_validation":
        mapping = {
            "real_dataset": "preprocessed_data",
            "null_artifact": "null_artifact",
            "null_lineage_sidecar": "null_lineage_sidecar",
            "split_plan": "split_plan",
        }
    elif stage.stage_type in {"training", "explanation"}:
        mapping = {"dataset": _side_dataset_key(stage.side), "pc_map": "pc_map"}
        if stage.stage_type == "training":
            mapping.update({"split_plan": "split_plan", "sex_map": "sex_map"})
    else:
        return {}
    return {name: files[key]["sha256"] for name, key in mapping.items() if files[key] is not None}


def _output_specs(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[IOSpec]:
    """Return ``(logical_name, path, kind)`` for every output a completed stage hashes."""
    plan = ctx.plan
    if stage.stage_type == "null_validation":
        return [(NULL_VALIDATION_NAME, stage.owned_paths[0], "file")]
    if stage.stage_type == "training":
        training_dir = stage.owned_paths[0]
        specs = [
            ("config.yaml", training_dir / "config.yaml", "file"),
            ("dataset_mappings.json", training_dir / "dataset_mappings.json", "file"),
            ("split_plan.yaml", training_dir / "split_plan.yaml", "file"),
        ]
        if _is_cv(plan):
            specs.append(("cv_results.yaml", training_dir / "cv_results.yaml", "file"))
            for fold in range(plan["split_plan"]["n_folds"]):
                fold_dir = training_dir / f"fold_{fold}"
                for name in ("config.yaml", "fold_info.yaml", "best_model.pt"):
                    specs.append((f"fold_{fold}/{name}", fold_dir / name, "file"))
        else:
            specs.append(("results.yaml", training_dir / "results.yaml", "file"))
            specs.append(("best_model.pt", training_dir / "best_model.pt", "file"))
        specs.append(("selected_checkpoint", _selected_checkpoint(plan, training_dir), "file"))
        specs.append((OUTPUT_TREE, training_dir, "directory"))
        return specs
    if stage.stage_type == "explanation":
        explanation_dir = stage.owned_paths[0]
        specs = [
            (name, explanation_dir / name, "file")
            for name in (
                "analysis_metadata.yaml",
                "attributions.npz",
                "sieve_variant_rankings.csv",
                "sieve_gene_rankings.csv",
            )
        ]
        specs.append((OUTPUT_TREE, explanation_dir, "directory"))
        return specs
    if stage.stage_type == "pair_validation":
        return [(Path(stage.owned_paths[0]).name, stage.owned_paths[0], "file")]
    if stage.stage_type == "calibration":
        return [(Path(path).name, Path(path), "file") for path in stage.owned_paths]
    if stage.stage_type == "shared_null":
        return [(SHARED_NULL_NAME, stage.owned_paths[0], "file")]
    if stage.stage_type == "comparison":
        name = stage.stage_id.split("/", 1)[1]
        specs = [
            (Path(path).name, Path(path), "file")
            for path in plan["comparisons"][name]["expected_outputs"]
        ]
        specs.append((OUTPUT_TREE, stage.owned_paths[0], "directory"))
        return specs
    raise PlanAuthorityError(f"unknown stage type {stage.stage_type!r}")


def _fingerprint(path: Path, kind: str, *, manifest_path: Path | None = None) -> dict[str, Any]:
    if kind == "file":
        return file_fingerprint(path)
    return directory_fingerprint(path, manifest_path=manifest_path)


def _compute_inputs(ctx: _ExecutionContext, stage: BenchmarkStage) -> dict[str, Any]:
    """Fingerprint a stage's inputs now and require plan-bound inputs to be unchanged."""
    inputs = {name: _fingerprint(path, kind) for name, path, kind in _input_specs(ctx, stage)}
    for name, planned in _planned_input_hashes(ctx.plan, stage).items():
        if inputs[name]["sha256"] != planned:
            raise StageStateError(
                f"stage {stage.stage_id} input {name} bytes changed since planning: "
                f"{inputs[name]['path']} (planned {planned}, current {inputs[name]['sha256']})"
            )
    if inputs[RESOLVED_PLAN_INPUT]["sha256"] != ctx.verified.resolved_plan_sha256:
        raise PlanAuthorityError(f"bound plan bytes changed: {ctx.bound_plan_path}")
    return inputs


def _compute_outputs(ctx: _ExecutionContext, stage: BenchmarkStage) -> dict[str, Any]:
    """Fingerprint a stage's outputs, persisting any output-tree manifest outside the tree."""
    outputs = {}
    for name, path, kind in _output_specs(ctx, stage):
        manifest_path = None
        if kind == "directory":
            manifest_path = output_manifest_path(ctx.execution_dir, stage.stage_id)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
        outputs[name] = _fingerprint(path, kind, manifest_path=manifest_path)
    return outputs


def _current_dependencies(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[dict[str, Any]]:
    """Return dependency references, requiring each record to be unchanged since acceptance."""
    references = []
    for dependency_id in stage.dependency_ids:
        accepted = ctx.completed.get(dependency_id)
        if accepted is None:
            raise StageStateError(
                f"stage {stage.stage_id} dependency {dependency_id} has not completed"
            )
        reference = dependency_reference(accepted.record_path)
        if reference["record_sha256"] != accepted.record_sha256:
            raise CompletedStageMismatchError(
                f"stage {stage.stage_id} dependency record {accepted.record_path} changed "
                "after it was accepted"
            )
        references.append(reference)
    return references


# ---------------------------------------------------------------------------
# Post-validation helpers
# ---------------------------------------------------------------------------


def _lookup(data: Any, dotted: str) -> Any:
    current = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _check(stage: BenchmarkStage, label: str, actual: Any, expected: Any, checks: list[str]):
    """Require *actual* to equal *expected* type-strictly, recording the check name."""
    if actual is _MISSING or _deep_differences(actual, expected):
        raise StagePostValidationError(
            f"{stage.stage_id}: {label} must be {expected!r}, got {actual!r}"
        )
    checks.append(label)


def _require_file(stage: BenchmarkStage, path: Path, checks: list[str]) -> None:
    if path.is_symlink() or not path.is_file():
        raise StagePostValidationError(f"{stage.stage_id}: required output file is missing: {path}")
    checks.append(f"exists:{path.name}")


def _load_stage_yaml(stage: BenchmarkStage, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StagePostValidationError(f"{stage.stage_id}: required output file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise StagePostValidationError(f"{stage.stage_id}: cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise StagePostValidationError(f"{stage.stage_id}: {path} must contain a YAML mapping")
    return data


def _parse_planned_argv(stage: BenchmarkStage, parser) -> Any:
    """Parse the planned argv with the stage script's own argument parser.

    Reusing ``train.py``/``explain.py``'s parser means post-validation compares
    saved metadata against exactly what the script received, without a second
    hand-written flag normaliser.
    """
    try:
        return parser.parse_args(list(stage.argv[2:]))
    except SystemExit as error:
        raise PlanAuthorityError(
            f"{stage.stage_id}: planned argv is not accepted by its script parser"
        ) from error


def _yaml_round_trip(value: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(dump_yaml_bytes(value))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


# ---------------------------------------------------------------------------
# Stage post-validators
# ---------------------------------------------------------------------------


def _validate_null_validation_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    """Require ``null_validation.yaml`` to equal the report of this invocation's full preflight."""
    checks: list[str] = []
    path = stage.owned_paths[0]
    saved = _load_stage_yaml(stage, path)
    expected = build_null_validation_report(
        ctx.plan,
        resolved_plan_sha256=ctx.verified.resolved_plan_sha256,
        repository_revision=ctx.plan["repository_revision"],
        preflight=ctx.preflight,
        validated_at=saved.get("validated_at"),
    )
    _check(stage, NULL_VALIDATION_NAME, saved, _yaml_round_trip(expected), checks)
    checks.append("validate_null_pair:full_preflight_passed")
    return checks


def _validate_training_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    """Validate a completed ``train.py`` run against the reviewed plan (exit 0 is not enough)."""
    from scripts import train as train_script
    from src.data.dataset_provenance import DATASET_PROVENANCE_SCHEMA_VERSION
    from src.training.split_plan import split_plan_sha256

    plan = ctx.plan
    binding = plan["null_binding"]
    split = plan["split_plan"]
    training_dir = stage.owned_paths[0]
    is_null = stage.side == "null"
    checks: list[str] = []
    for _, path, kind in _output_specs(ctx, stage):
        if kind == "file":
            _require_file(stage, path, checks)
    config = _load_stage_yaml(stage, training_dir / "config.yaml")
    namespace = _parse_planned_argv(stage, train_script.build_arg_parser())

    # Every CLI value train.py received must be what it saved.
    for key, value in sorted(vars(namespace).items()):
        if key not in TRAIN_CONFIG_OVERRIDDEN_ARGS:
            _check(stage, f"config.{key}", config.get(key, _MISSING), value, checks)
    _check(
        stage, "config.level", config.get("level"), plan["benchmark"]["annotation_level"], checks
    )
    _check(stage, "config.seed", config.get("seed"), plan["paired_policy"]["training_seed"], checks)
    _check(
        stage,
        "config.class_weighting",
        config.get("class_weighting"),
        PAIRED_CLASS_WEIGHTING_OFF,
        checks,
    )
    # train.py writes class_weighting_applied/pos_weight to the root config in
    # both modes: null/null for CV (fold configs carry the per-fold values) and
    # false/null for single split. With class_weighting "off" no pos_weight is
    # ever computed.
    _check(
        stage,
        "config.class_weighting_applied",
        config.get("class_weighting_applied", _MISSING),
        None if _is_cv(plan) else False,
        checks,
    )
    _check(
        stage,
        "config.class_weighting_pos_weight",
        config.get("class_weighting_pos_weight", _MISSING),
        None,
        checks,
    )

    # Dataset relation to the one shared null binding.
    provenance = config.get("dataset_provenance")
    _check(
        stage,
        "dataset_provenance.schema_version",
        _lookup(provenance, "schema_version"),
        DATASET_PROVENANCE_SCHEMA_VERSION,
        checks,
    )
    _check(
        stage,
        "dataset_provenance.is_null_baseline",
        _lookup(provenance, "is_null_baseline"),
        is_null,
        checks,
    )
    _check(
        stage,
        "dataset_provenance.preprocessed_data_sha256",
        _lookup(provenance, "preprocessed_data_sha256"),
        binding["null_artifact_sha256" if is_null else "source_artifact_sha256"],
        checks,
    )
    _check(
        stage,
        "dataset_provenance.sample_ids_sha256",
        _lookup(provenance, "sample_ids_sha256"),
        binding["sample_ids_sha256"],
        checks,
    )
    if is_null:
        _check(
            stage,
            "dataset_provenance.null_lineage.lineage_sha256",
            _lookup(provenance, "null_lineage.lineage_sha256"),
            binding["lineage_sha256"],
            checks,
        )
        _check(
            stage,
            "dataset_provenance.null_lineage.source_artifact_sha256",
            _lookup(provenance, "null_lineage.source_artifact_sha256"),
            binding["source_artifact_sha256"],
            checks,
        )
    else:
        _check(
            stage,
            "dataset_provenance.null_lineage",
            _lookup(provenance, "null_lineage"),
            None,
            checks,
        )

    # Exact split replay. membership_sha256, sha256 and input_sha256 are all
    # split_plan_sha256() over the canonical membership payload.
    membership = split["membership_sha256"]
    for field_name, expected in (
        ("source", "replayed"),
        ("sha256", membership),
        ("sample_ids_sha256", split["sample_ids_sha256"]),
        ("input_sha256", membership),
        ("input_path", plan["input_files"]["split_plan"]["path"]),
        ("path", str((training_dir / "split_plan.yaml").resolve())),
    ):
        _check(
            stage,
            f"config.split_plan.{field_name}",
            _lookup(config, f"split_plan.{field_name}"),
            expected,
            checks,
        )
    saved_split = _load_stage_yaml(stage, training_dir / "split_plan.yaml")
    try:
        saved_membership = split_plan_sha256(saved_split)
    except (KeyError, TypeError, ValueError) as error:
        raise StagePostValidationError(
            f"{stage.stage_id}: saved split_plan.yaml is invalid: {error}"
        ) from error
    _check(stage, "split_plan.yaml.membership_sha256", saved_membership, membership, checks)
    _check(
        stage, "split_plan.yaml.split_source", saved_split.get("split_source"), "replayed", checks
    )

    _validate_training_position(stage, config, namespace, _training_mode(plan), checks)
    _validate_training_results(stage, plan, training_dir, config, checks)
    try:
        mappings = json.loads((training_dir / "dataset_mappings.json").read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise StagePostValidationError(
            f"{stage.stage_id}: dataset_mappings.json is not valid JSON: {error}"
        ) from error
    if not isinstance(mappings, dict):
        raise StagePostValidationError(f"{stage.stage_id}: dataset_mappings.json must be a mapping")
    checks.append("dataset_mappings.json:parsed")
    return checks


def _validate_training_position(
    stage: BenchmarkStage,
    config: Mapping[str, Any],
    namespace: Any,
    training_mode: str,
    checks: list[str],
) -> None:
    """Require saved position metadata to be authoritative and to equal the planned strategy.

    The resolver is the single source of positional configuration: the saved
    ``position_encoding`` is re-resolved by ``resolved_position_encoding_from_dict``
    and the planned CLI intent is resolved by ``train.py``'s own
    ``prepare_training_position_encoding``; both must be identical, and the
    execution metadata must be exactly what ``train.py`` derives from it.
    """
    from scripts import train as train_script
    from src.encoding import AnnotationLevel
    from src.encoding.position_config import resolved_position_encoding_from_dict

    num_chromosomes = config.get("num_chromosomes")
    try:
        require_authoritative_position_metadata(config)
        if isinstance(num_chromosomes, bool) or not isinstance(num_chromosomes, int):
            raise ValueError("config.num_chromosomes must be an integer")
        saved = resolved_position_encoding_from_dict(
            config["position_encoding"],
            latent_dim=namespace.latent_dim,
            num_heads=namespace.num_heads,
        )
        intended = train_script.prepare_training_position_encoding(
            namespace, AnnotationLevel[namespace.level], num_chromosomes=num_chromosomes
        )
        identity = position_strategy_identity(config)
    except (KeyError, TypeError, ValueError) as error:
        raise StagePostValidationError(
            f"{stage.stage_id}: position metadata is not authoritative: {error}"
        ) from error
    if saved != intended:
        raise StagePostValidationError(
            f"{stage.stage_id}: saved position_encoding does not match the planned strategy "
            f"(saved {saved.to_dict()!r}, planned {intended.to_dict()!r})"
        )
    checks.append("position_encoding:matches_planned_strategy")
    _check(
        stage,
        "config.position_encoding_execution",
        config.get("position_encoding_execution", _MISSING),
        _yaml_round_trip(
            train_script.build_position_encoding_execution_metadata(
                resolved_position_encoding=intended, training_mode=training_mode
            )
        ),
        checks,
    )
    _check(stage, "config.input_dim", config.get("input_dim"), intended.input_dim, checks)
    _check(stage, "config.content_dim", config.get("content_dim"), intended.content_dim, checks)
    checks.append(f"position_strategy_identity:{identity.hash}")


def _validate_training_results(
    stage: BenchmarkStage,
    plan: Mapping[str, Any],
    training_dir: Path,
    config: Mapping[str, Any],
    checks: list[str],
) -> None:
    """Validate CV fold metadata (every planned fold) or single-split results."""
    seed = plan["paired_policy"]["training_seed"]
    level = plan["benchmark"]["annotation_level"]
    split = plan["split_plan"]
    if not _is_cv(plan):
        results = _load_stage_yaml(stage, training_dir / "results.yaml")
        _check(
            stage,
            "results.class_weighting_applied",
            results.get("class_weighting_applied", _MISSING),
            False,
            checks,
        )
        _check(
            stage,
            "results.class_weighting_pos_weight",
            results.get("class_weighting_pos_weight", _MISSING),
            None,
            checks,
        )
        return
    results = _load_stage_yaml(stage, training_dir / "cv_results.yaml")
    fold_results = results.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != split["n_folds"]:
        raise StagePostValidationError(
            f"{stage.stage_id}: cv_results.yaml must have {split['n_folds']} fold_results"
        )
    checks.append("cv_results.yaml:fold_results")
    for fold in split["folds"]:
        index = fold["fold_index"]
        fold_dir = training_dir / f"fold_{index}"
        fold_config = _load_stage_yaml(stage, fold_dir / "config.yaml")
        for key, expected in (
            ("fold_index", index),
            ("seed", seed),
            ("level", level),
            ("class_weighting_applied", False),
            ("class_weighting_pos_weight", None),
            ("parent_config", "../config.yaml"),
            ("position_encoding", config.get("position_encoding")),
            ("dataset_provenance", config.get("dataset_provenance")),
        ):
            _check(
                stage,
                f"fold_{index}/config.yaml.{key}",
                fold_config.get(key, _MISSING),
                expected,
                checks,
            )
        info = _load_stage_yaml(stage, fold_dir / "fold_info.yaml")
        for key, expected in (
            ("fold_index", index),
            ("n_folds", split["n_folds"]),
            ("random_seed", seed),
        ):
            _check(
                stage,
                f"fold_{index}/fold_info.yaml.{key}",
                info.get(key, _MISSING),
                expected,
                checks,
            )
        for key, planned in (
            ("train_sample_indices", fold["train_indices"]),
            ("val_sample_indices", fold["val_indices"]),
        ):
            recorded = info.get(key)
            if not isinstance(recorded, list) or sorted(recorded) != sorted(planned):
                raise StagePostValidationError(
                    f"{stage.stage_id}: fold_{index}/fold_info.yaml {key} does not match the "
                    "replayed split plan"
                )
            checks.append(f"fold_{index}/fold_info.yaml.{key}")


def _validate_explanation_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    """Validate a completed ``explain.py`` run against its training authority and the plan."""
    from scripts import explain as explain_script

    plan = ctx.plan
    run = _run_entry(plan, stage.run_id)
    training_dir, explanation_dir = _side_directories(run, stage.side)
    training = ctx.completed[f"runs/{stage.run_id}/{stage.side}_training"].record
    checks: list[str] = []
    for _, path, kind in _output_specs(ctx, stage):
        if kind == "file":
            _require_file(stage, path, checks)
    per_sample = explanation_dir / "attributions_per_sample"
    if per_sample.is_symlink() or not per_sample.is_dir() or not any(per_sample.iterdir()):
        raise StagePostValidationError(
            f"{stage.stage_id}: attributions_per_sample/ is missing or empty: {per_sample}"
        )
    checks.append("attributions_per_sample:non_empty")
    if os.path.lexists(explanation_dir / "_tmp_attributions"):
        raise StagePostValidationError(
            f"{stage.stage_id}: lingering _tmp_attributions/ (explanation did not finish)"
        )
    checks.append("no_tmp_attributions")

    metadata = _load_stage_yaml(stage, explanation_dir / "analysis_metadata.yaml")
    namespace = _parse_planned_argv(stage, explain_script.build_arg_parser())
    try:
        side = load_pair_side(
            training_dir, explanation_dir, repository_revision=training["repository_revision"]
        )
    except PairCompatibilityError as error:
        raise StagePostValidationError(f"{stage.stage_id}: {error}") from error
    checks.append("load_pair_side:config_and_checkpoint_bound")

    selected = _selected_checkpoint(plan, training_dir)
    expected_fold = _expected_fold_index(plan)
    for label, actual, expected in (
        (
            "model_provenance.config_path",
            _lookup(metadata, "model_provenance.config_path"),
            str((training_dir / "config.yaml").resolve()),
        ),
        (
            "model_provenance.config_sha256",
            _lookup(metadata, "model_provenance.config_sha256"),
            training["outputs"]["config.yaml"]["sha256"],
        ),
        (
            "model_provenance.checkpoint_path",
            _lookup(metadata, "model_provenance.checkpoint_path"),
            str(selected.resolve()),
        ),
        (
            "model_provenance.checkpoint_sha256",
            _lookup(metadata, "model_provenance.checkpoint_sha256"),
            training["outputs"]["selected_checkpoint"]["sha256"],
        ),
        (
            "model_provenance.checkpoint_selection_mode",
            _lookup(metadata, "model_provenance.checkpoint_selection_mode"),
            "cv_explicit_fold" if expected_fold is not None else "single_run_best_model",
        ),
        (
            "model_provenance.selected_fold",
            _lookup(metadata, "model_provenance.selected_fold"),
            expected_fold,
        ),
        ("argv.fold_index", namespace.fold_index, expected_fold),
        ("is_null_baseline", metadata.get("is_null_baseline", _MISSING), stage.side == "null"),
        ("argv.is_null_baseline", namespace.is_null_baseline, stage.side == "null"),
        ("n_samples", metadata.get("n_samples", _MISSING), plan["null_binding"]["n_samples"]),
        (
            "annotation_level",
            metadata.get("annotation_level", _MISSING),
            plan["benchmark"]["annotation_level"],
        ),
        ("genome_build", metadata.get("genome_build", _MISSING), plan["dataset"]["genome_build"]),
        ("experiment_dir", metadata.get("experiment_dir", _MISSING), namespace.experiment_dir),
        ("skip_ig", metadata.get("skip_ig", _MISSING), False),
        ("skip_attention", metadata.get("skip_attention", _MISSING), namespace.skip_attention),
        (
            "aggregation_method",
            metadata.get("aggregation_method", _MISSING),
            namespace.aggregation_method,
        ),
        (
            "max_variants_per_sample",
            metadata.get("max_variants_per_sample", _MISSING),
            namespace.max_variants,
        ),
        ("n_integration_steps", metadata.get("n_integration_steps", _MISSING), namespace.n_steps),
        ("argv.ig_mode", namespace.ig_mode, "content"),
        ("integrated_gradients.executed", _lookup(metadata, "integrated_gradients.executed"), True),
        (
            "integrated_gradients.resolved_ig_mode",
            _lookup(metadata, "integrated_gradients.resolved_ig_mode"),
            "content",
        ),
        (
            "integrated_gradients.n_steps",
            _lookup(metadata, "integrated_gradients.n_steps"),
            namespace.n_steps,
        ),
        # integrated_gradients.max_variants is explain.py's IG chunk width
        # (min(--max-variants, 2000)), not the planned value, so the planned
        # --max-variants is checked via max_variants_per_sample above; real
        # and null IG blocks are compared in full by pair validation.
        (
            "dataset_provenance",
            metadata.get("dataset_provenance", _MISSING),
            side.config.get("dataset_provenance", _MISSING),
        ),
    ):
        _check(stage, label, actual, expected, checks)
    return checks


def _compute_pair_report(ctx: _ExecutionContext, stage: BenchmarkStage) -> dict[str, Any]:
    """Run the 12C3B1 pair rules and the cumulative shared-null check; return the report."""
    plan = ctx.plan
    run = _run_entry(plan, stage.run_id)
    sides = {}
    for side_name in ("real", "null"):
        training_dir, explanation_dir = _side_directories(run, side_name)
        # Repository revision comes from the authoritative explanation record.
        explanation = ctx.completed[f"runs/{stage.run_id}/{side_name}_explanation"].record
        sides[side_name] = load_pair_side(
            training_dir, explanation_dir, repository_revision=explanation["repository_revision"]
        )
    report = require_compatible_real_null_pair(
        run_id=stage.run_id,
        real=sides["real"],
        null=sides["null"],
        null_binding=plan["null_binding"],
        expected_fold_index=_expected_fold_index(plan),
    )
    prior = [
        _load_pair_report(ctx, dependency.split("/")[1])
        for dependency in stage.dependency_ids
        if dependency.endswith("/pair_validation")
    ]
    # Catch a divergent null binding as early as possible, before calibration.
    require_shared_null_across_pairs([*prior, report])
    return _yaml_round_trip(report)


def _load_pair_report(ctx: _ExecutionContext, run_id: str) -> dict[str, Any]:
    path = Path(_run_entry(ctx.plan, run_id)["calibration"]["paired_compatibility"])
    with path.open("r", encoding="utf-8") as handle:
        report = yaml.safe_load(handle)
    if not isinstance(report, dict):
        raise PairCompatibilityError(f"{path} must contain a pair report mapping")
    return report


def _validate_pair_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    checks: list[str] = []
    expected = _compute_pair_report(ctx, stage)
    checks.extend(
        ["require_compatible_real_null_pair", "require_shared_null_across_pairs:cumulative"]
    )
    saved = _load_stage_yaml(stage, stage.owned_paths[0])
    _check(stage, "paired_compatibility.yaml", saved, expected, checks)
    return checks


def _compute_shared_null_report(ctx: _ExecutionContext) -> dict[str, Any]:
    reports = [_load_pair_report(ctx, run["run_id"]) for run in ctx.plan["runs"]]
    result = require_shared_null_across_pairs(reports)
    if result["null_binding"] != _binding_identity(ctx.plan):
        raise PairCompatibilityError("shared null binding differs from the plan null_binding")
    return _yaml_round_trip(result)


def _validate_shared_null_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    checks = ["require_shared_null_across_pairs:all_runs"]
    saved = _load_stage_yaml(stage, stage.owned_paths[0])
    _check(stage, SHARED_NULL_NAME, saved, _compute_shared_null_report(ctx), checks)
    return checks


def _validate_calibration_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    """Validate the unchanged bootstrap outputs using the summary's actual key names."""
    plan = ctx.plan
    settings = plan["calibration"]
    checks: list[str] = []
    for path in stage.owned_paths:
        _require_file(stage, path, checks)
    summary_path = next(path for path in stage.owned_paths if path.suffix == ".yaml")
    summary = _load_stage_yaml(stage, summary_path)
    for label, actual, expected in (
        ("summary.n_bootstrap", summary.get("n_bootstrap", _MISSING), settings["n_bootstrap"]),
        (
            "summary.n_null_samples",
            summary.get("n_null_samples", _MISSING),
            plan["null_binding"]["n_samples"],
        ),
        (
            "summary.excluded_sex_chroms",
            summary.get("excluded_sex_chroms", _MISSING),
            settings["exclude_sex_chroms"],
        ),
        (
            "summary.per_gene.gene_delta_rank_aggregation",
            _lookup(summary, "per_gene.gene_delta_rank_aggregation"),
            settings["gene_delta_rank_aggregation"],
        ),
        (
            "summary.genome_build",
            summary.get("genome_build", _MISSING),
            plan["dataset"]["genome_build"],
        ),
        (
            "summary.n_real_variants_missing_from_null",
            summary.get("n_real_variants_missing_from_null", _MISSING),
            0,
        ),
    ):
        _check(stage, label, actual, expected, checks)
    n_real = summary.get("n_real_variants")
    if isinstance(n_real, bool) or not isinstance(n_real, int) or n_real <= 0:
        raise StagePostValidationError(
            f"{stage.stage_id}: summary.n_real_variants must be a positive integer, got {n_real!r}"
        )
    checks.append("summary.n_real_variants>0")
    return checks


def _validate_comparison_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    checks: list[str] = []
    for _, path, kind in _output_specs(ctx, stage):
        if kind == "file":
            _require_file(stage, path, checks)
    if stage.stage_id == CALIBRATED_COMPARISON_STAGE_ID:
        checks.extend(_validate_calibrated_comparison_outputs(ctx, stage))
    return checks


def _validate_calibrated_comparison_outputs(
    ctx: _ExecutionContext, stage: BenchmarkStage
) -> list[str]:
    """Require the comparison YAML to report exactly the identities this executor accepted.

    The comparator validated provenance on its own (it is also usable
    standalone); this cross-check proves it validated the SAME plan binding,
    shared-null record, calibration records, and pair-validation records that
    this invocation accepted, so no substitute benchmark state can pass.
    """
    plan = ctx.plan
    checks: list[str] = []
    comparison_yaml = Path(plan["comparisons"][CALIBRATED_COMPARISON_KEY]["expected_outputs"][0])
    summary = _load_stage_yaml(stage, comparison_yaml)
    _check(stage, "comparison_mode", summary.get("comparison_mode", _MISSING), "calibrated", checks)
    _check(stage, "score.column", _lookup(summary, "score.column"), CALIBRATED_SCORE_COLUMN, checks)
    _check(stage, "score.sort_order", _lookup(summary, "score.sort_order"), "descending", checks)

    shared = ctx.completed[SHARED_NULL_STAGE_ID]
    plan_binding_path = ctx.execution_dir / PLAN_BINDING_NAME
    expected_provenance = {
        "benchmark_root": str(ctx.benchmark_root),
        "resolved_plan_path": str(ctx.bound_plan_path),
        "resolved_plan_sha256": ctx.verified.resolved_plan_sha256,
        "plan_binding_path": str(plan_binding_path),
        "plan_binding_sha256": sha256_file(plan_binding_path),
        "repository_revision": plan["repository_revision"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "shared_null_record_path": str(shared.record_path),
        "shared_null_record_sha256": shared.record_sha256,
        "shared_null_report_path": str(shared_null_validation_path(ctx.benchmark_root)),
        "shared_null_report_sha256": shared.record["outputs"][SHARED_NULL_NAME]["sha256"],
        "null_binding": _binding_identity(plan),
    }
    _check(
        stage,
        "execution_provenance",
        summary.get("execution_provenance", _MISSING),
        expected_provenance,
        checks,
    )

    reported_runs = summary.get("runs")
    planned_ids = sorted(run["run_id"] for run in plan["runs"])
    if (
        not isinstance(reported_runs, list)
        or [run.get("run_id") if isinstance(run, Mapping) else None for run in reported_runs]
        != planned_ids
    ):
        raise StagePostValidationError(
            f"{stage.stage_id}: comparison runs do not equal the planned runs {planned_ids}"
        )
    for reported in reported_runs:
        run_id = reported["run_id"]
        calibration = ctx.completed[f"runs/{run_id}/calibration"]
        pair = ctx.completed[f"runs/{run_id}/pair_validation"]
        if sha256_file(pair.record_path) != pair.record_sha256:
            raise StagePostValidationError(
                f"{stage.stage_id}: pair-validation record changed after acceptance: "
                f"{pair.record_path}"
            )
        rankings = calibration.record["outputs"][CALIBRATION_RANKINGS_NAME]
        for label, expected in (
            ("calibration_record_path", str(calibration.record_path)),
            ("calibration_record_sha256", calibration.record_sha256),
            ("pair_validation_record_path", str(pair.record_path)),
            ("pair_validation_record_sha256", pair.record_sha256),
            ("calibrated_ranking_path", rankings["path"]),
            ("calibrated_ranking_sha256", rankings["sha256"]),
        ):
            _check(stage, f"runs.{run_id}.{label}", reported.get(label, _MISSING), expected, checks)
    return checks


_POST_VALIDATORS = {
    "null_validation": _validate_null_validation_stage,
    "training": _validate_training_stage,
    "explanation": _validate_explanation_stage,
    "pair_validation": _validate_pair_stage,
    "calibration": _validate_calibration_stage,
    "shared_null": _validate_shared_null_stage,
    "comparison": _validate_comparison_stage,
}


# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------


def calibration_input_gate(ctx: _ExecutionContext, stage: BenchmarkStage) -> list[str]:
    """Require calibration to read exactly the bytes that passed pair validation.

    Recomputes the SHA-256 of the real ranking CSV, real analysis metadata,
    null ``attributions.npz``, and ``paired_compatibility.yaml`` and requires
    them to equal the pair-validation completed record, which itself must be
    unchanged since it was accepted. The planned calibration argv must name
    exactly those files.
    """
    pair_id = f"runs/{stage.run_id}/pair_validation"
    accepted = ctx.completed[pair_id]
    if sha256_file(accepted.record_path) != accepted.record_sha256:
        raise CalibrationInputGateError(
            f"{stage.stage_id}: pair-validation record changed after acceptance: "
            f"{accepted.record_path}"
        )
    record = load_stage_record(accepted.record_path)
    bound = {
        "real/sieve_variant_rankings.csv": record["inputs"]["real/sieve_variant_rankings.csv"],
        "real/analysis_metadata.yaml": record["inputs"]["real/analysis_metadata.yaml"],
        "null/attributions.npz": record["inputs"]["null/attributions.npz"],
        "paired_compatibility.yaml": record["outputs"]["paired_compatibility.yaml"],
    }
    checks = []
    for name, fingerprint in bound.items():
        try:
            current = file_fingerprint(fingerprint["path"])
        except StageRecordError as error:
            raise CalibrationInputGateError(f"{stage.stage_id}: {name}: {error}") from error
        if current != fingerprint:
            raise CalibrationInputGateError(
                f"{stage.stage_id}: {name} changed after pair validation: {fingerprint['path']} "
                f"(validated {fingerprint['sha256']}, current {current['sha256']})"
            )
        checks.append(f"calibration_input_gate:{name}")
    argv = list(stage.argv or ())
    for flag, name in (
        ("--real-rankings", "real/sieve_variant_rankings.csv"),
        ("--null-attributions", "null/attributions.npz"),
    ):
        if flag not in argv or argv[argv.index(flag) + 1] != bound[name]["path"]:
            raise CalibrationInputGateError(
                f"{stage.stage_id}: planned {flag} does not name the pair-validated {name}"
            )
    report = yaml.safe_load(Path(bound["paired_compatibility.yaml"]["path"]).read_bytes())
    if not isinstance(report, Mapping) or report.get("compatible") is not True:
        raise CalibrationInputGateError(
            f"{stage.stage_id}: paired compatibility report is not passing"
        )
    return checks


def require_calibrated_comparison_argv(plan: Mapping[str, Any], stage: BenchmarkStage) -> None:
    """Require the calibrated comparison argv to be exactly its planned, narrow shape.

    The only accepted command is ``compare_ablation_rankings.py
    --comparison-axis position --position-calibrated-benchmark <this benchmark
    root> --score-column delta_rank`` writing this stage's own three planned
    outputs. Any ``--position-run``, null or raw-comparison path, other score
    column, extra flag, or external benchmark root is refused.
    """
    root = benchmark_root_from_plan(plan)
    comparison = plan["comparisons"][CALIBRATED_COMPARISON_KEY]
    directory = Path(comparison["directory"])
    outputs = [Path(path) for path in comparison["expected_outputs"]]
    if (
        directory != root / "comparisons" / CALIBRATED_COMPARISON_KEY
        or [path.name for path in outputs] != list(CALIBRATED_COMPARISON_OUTPUT_NAMES)
        or any(path.parent != directory for path in outputs)
    ):
        raise StageStateError(
            f"{stage.stage_id}: planned calibrated comparison outputs are not its own directory"
        )
    argv = list(stage.argv or ())
    if "--position-run" in argv:
        raise StageStateError(f"{stage.stage_id}: calibrated comparison argv names --position-run")
    expected = [
        plan["runtime"]["python"],
        str(Path(plan["repository_root"]) / "scripts" / "compare_ablation_rankings.py"),
        "--comparison-axis",
        "position",
        "--position-calibrated-benchmark",
        str(root),
        "--score-column",
        CALIBRATED_SCORE_COLUMN,
        "--out-comparison",
        str(outputs[0]),
        "--out-jaccard",
        str(outputs[1]),
        "--out-level-specific",
        str(outputs[2]),
    ]
    if argv != expected:
        raise StageStateError(
            f"{stage.stage_id}: calibrated comparison argv is not the exact planned shape "
            f"(benchmark root {root}, score column {CALIBRATED_SCORE_COLUMN}, own outputs)"
        )


def _require_comparison_argv(plan: Mapping[str, Any], stage: BenchmarkStage) -> None:
    """Apply the calibrated or the real-only argv guard to a comparison stage."""
    if stage.stage_id == CALIBRATED_COMPARISON_STAGE_ID:
        require_calibrated_comparison_argv(plan, stage)
    else:
        require_real_only_comparison(plan, stage)


def require_real_only_comparison(plan: Mapping[str, Any], stage: BenchmarkStage) -> None:
    """Refuse a raw B1/B2/B3 argv that names null, calibration, or non-real benchmark paths."""
    root = benchmark_root_from_plan(plan)
    comparison_dir = stage.owned_paths[0]
    allowed = [
        Path(run["directories"][key]) for run in plan["runs"] for key in ("training", "explanation")
    ]
    for token in stage.argv or ():
        path = Path(token)
        if path.name.startswith("bootstrap_calibrated_") or path.name == CALIBRATION_RANKINGS_NAME:
            raise StageStateError(f"{stage.stage_id}: raw comparison argv names calibrated output")
        if not path.is_absolute() or not _is_within(path, root) or _is_within(path, comparison_dir):
            continue
        if not any(_is_within(path, real_root) for real_root in allowed):
            raise StageStateError(
                f"{stage.stage_id}: raw comparison argv references a non-real path: {path}"
            )


def _require_stage_can_start(ctx: _ExecutionContext, stage: BenchmarkStage) -> None:
    ensure_stage_can_start(ctx.execution_dir, stage.stage_id, list(stage.owned_paths))
    manifest = output_manifest_path(ctx.execution_dir, stage.stage_id)
    if os.path.lexists(manifest):
        raise StageStateError(
            f"stage {stage.stage_id} output manifest already exists without a completed "
            f"record: {manifest}"
        )


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _produce_in_process(ctx: _ExecutionContext, stage: BenchmarkStage) -> None:
    """Run an in-process validator stage and atomically write its deterministic report."""
    path = stage.owned_paths[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage.stage_type == "null_validation":
        write_null_validation_report(
            ctx.benchmark_root,
            build_null_validation_report(
                ctx.plan,
                resolved_plan_sha256=ctx.verified.resolved_plan_sha256,
                repository_revision=ctx.plan["repository_revision"],
                preflight=ctx.preflight,
            ),
        )
    elif stage.stage_type == "pair_validation":
        # Pure scientific compatibility report: no timestamps, hashes of
        # files, or environment (those live in the stage record).
        atomic_write_bytes(path, dump_yaml_bytes(_compute_pair_report(ctx, stage)))
    elif stage.stage_type == "shared_null":
        atomic_write_bytes(path, dump_yaml_bytes(_compute_shared_null_report(ctx)))
    else:
        raise PlanAuthorityError(f"{stage.stage_type} is not an in-process stage")


def _stage_record_fields(ctx: _ExecutionContext, stage: BenchmarkStage) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "stage_type": stage.stage_type,
        "run_id": stage.run_id,
        "side": stage.side,
        **ctx.common,
    }


def _publish_in_process_failure(
    ctx: _ExecutionContext,
    stage: BenchmarkStage,
    running_path: Path,
    reason: str,
    exception: str | None,
) -> Path:
    running = load_stage_record(running_path)
    failed = build_stage_record(
        record_kind="failed",
        execution=running["execution"],
        inputs=running["inputs"],
        dependencies=running["dependencies"],
        environment=running["environment"],
        started_at=running["started_at"],
        completed_at=utc_now_iso(),
        failure={
            "reason": reason,
            "exit_code": None,
            "exception": exception,
            "partial_outputs_present": any(_has_partial_output(p) for p in stage.owned_paths),
        },
        **_stage_record_fields(ctx, stage),
    )
    path = write_stage_record(ctx.execution_dir, failed)
    running_path.unlink()
    return path


def _fail_started_stage(
    ctx: _ExecutionContext,
    stage: BenchmarkStage,
    running_path: Path,
    result: SubprocessStageResult | None,
    reason: str,
    exception: str | None,
) -> None:
    if result is None:
        _publish_in_process_failure(ctx, stage, running_path, reason, exception)
    else:
        record_stage_failure(
            execution_dir=ctx.execution_dir,
            running_path=running_path,
            result=result,
            reason=reason,
            owned_paths=list(stage.owned_paths),
            exception=exception,
        )


def _execute_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> None:
    """Run one stage fresh: gates, launch, post-validation, hashing, completed record.

    A completed record is published only after post-validation, output
    hashing, and a second repository gate all pass. Any failure after the
    running record exists publishes a failed record and keeps partial outputs.
    """
    require_repository_gate(ctx.plan, runner=ctx.git_runner)
    _require_stage_can_start(ctx, stage)
    pre_checks: list[str] = []
    if stage.stage_type == "calibration":
        pre_checks = calibration_input_gate(ctx, stage)
    if stage.stage_type == "comparison":
        _require_comparison_argv(ctx.plan, stage)
    dependencies = _current_dependencies(ctx, stage)
    inputs = _compute_inputs(ctx, stage)

    result: SubprocessStageResult | None = None
    if stage.argv is not None:
        try:
            result, running_path = execute_subprocess_stage(
                execution_dir=ctx.execution_dir,
                stage_id=stage.stage_id,
                stage_type=stage.stage_type,
                run_id=stage.run_id,
                side=stage.side,
                argv=list(stage.argv),
                cwd=ctx.repository_root,
                owned_paths=list(stage.owned_paths),
                common=ctx.common,
                inputs=inputs,
                dependencies=dependencies,
                environment=ctx.environment,
                popen=ctx.popen,
            )
        except StageFailure as failure:
            logs = ctx.execution_dir / LOGS_DIRNAME / f"{stage.stage_id}.stderr.log"
            raise StageFailure(
                f"stage {stage.stage_id} failed: {failure} (see {logs})", failure.result
            ) from failure
        execution = result.execution_block()
    else:
        execution = {"kind": "in_process", "callable": stage.callable_name}
        running_path = write_stage_record(
            ctx.execution_dir,
            build_stage_record(
                record_kind="running",
                execution=execution,
                inputs=inputs,
                dependencies=dependencies,
                environment=ctx.environment,
                started_at=utc_now_iso(),
                **_stage_record_fields(ctx, stage),
            ),
        )
    started_at = load_stage_record(running_path)["started_at"]

    phase = "validator_failed" if result is None else "post_validation_failed"
    try:
        if result is None:
            _produce_in_process(ctx, stage)
        phase = "post_validation_failed"
        checks = pre_checks + _POST_VALIDATORS[stage.stage_type](ctx, stage)
        outputs = _compute_outputs(ctx, stage)
        phase = "repository_gate_failed"
        require_repository_gate(ctx.plan, runner=ctx.git_runner)
        completed = build_stage_record(
            record_kind="completed",
            execution=execution,
            inputs=inputs,
            dependencies=dependencies,
            environment=ctx.environment,
            started_at=started_at,
            completed_at=utc_now_iso(),
            outputs=outputs,
            post_validation={"status": "passed", "checks": checks},
            **_stage_record_fields(ctx, stage),
        )
        path = publish_completed_record(ctx.execution_dir, running_path, completed)
    except KeyboardInterrupt:
        _fail_started_stage(ctx, stage, running_path, result, "interrupted", None)
        raise
    except _ORDINARY_STAGE_ERRORS as error:
        detail = f"{type(error).__name__}: {error}"
        _fail_started_stage(ctx, stage, running_path, result, phase, detail)
        raise StagePostValidationError(
            f"stage {stage.stage_id} failed ({phase}): {error}"
        ) from error
    except Exception as error:
        _fail_started_stage(
            ctx, stage, running_path, result, "unexpected_error", f"{type(error).__name__}: {error}"
        )
        raise
    ctx.completed[stage.stage_id] = CompletedStage(path, sha256_file(path), completed)


# ---------------------------------------------------------------------------
# Resume: completed-record revalidation
# ---------------------------------------------------------------------------


def _revalidate_completed_stage(ctx: _ExecutionContext, stage: BenchmarkStage) -> None:
    """Accept a completed record only if every recorded invariant still holds.

    Checks record schema and location; stage identity; plan, repository, and
    manifest binding; the exact argv/cwd (or in-process callable); log
    hashes; dependency record hashes; recomputed input and output
    fingerprints (directory trees must have no added, removed, or changed
    files); a re-run of the stage post-validation; and the repository gate.
    Any mismatch aborts and names it. The record is never rewritten and an
    invalid stage is never re-run automatically.
    """
    path = stage_record_path(ctx.execution_dir, stage.stage_id, "completed")

    def mismatch(message: str) -> CompletedStageMismatchError:
        return CompletedStageMismatchError(f"stage {stage.stage_id} cannot be resumed: {message}")

    try:
        record = require_completed_record(path)
    except StageRecordError as error:
        raise mismatch(f"completed record is invalid: {error}") from error
    expected_fields = {**_stage_record_fields(ctx, stage)}
    for key, expected in expected_fields.items():
        if record[key] != expected:
            raise mismatch(f"record {key} is {record[key]!r}, expected {expected!r}")

    execution = record["execution"]
    if stage.argv is not None:
        if execution["kind"] != "subprocess" or execution["argv"] != list(stage.argv):
            raise mismatch("recorded argv differs from the planned argv")
        if execution["cwd"] != ctx.repository_root or execution["exit_code"] != 0:
            raise mismatch("recorded cwd or exit code is not the planned successful launch")
        if execution["logs"] is None:
            raise mismatch("recorded execution has no log fingerprints")
        for stream, fingerprint in execution["logs"].items():
            try:
                current = file_fingerprint(fingerprint["path"])
            except StageRecordError as error:
                raise mismatch(f"{stream} log: {error}") from error
            if current != fingerprint:
                raise mismatch(f"{stream} log changed: {fingerprint['path']}")
    elif execution != {"kind": "in_process", "callable": stage.callable_name}:
        raise mismatch(f"recorded in-process callable differs: {execution!r}")

    recorded_ids = [dependency["stage_id"] for dependency in record["dependencies"]]
    if recorded_ids != list(stage.dependency_ids):
        raise mismatch(f"dependencies {recorded_ids} differ from {list(stage.dependency_ids)}")
    for dependency in record["dependencies"]:
        accepted = ctx.completed[dependency["stage_id"]]
        if (
            dependency["record_path"] != str(accepted.record_path)
            or dependency["record_sha256"] != accepted.record_sha256
        ):
            raise mismatch(f"dependency record {dependency['stage_id']} changed")

    try:
        inputs = _compute_inputs(ctx, stage)
    except (StageRecordError, StageStateError, PlanAuthorityError) as error:
        raise mismatch(f"input no longer matches: {error}") from error
    _require_same_fingerprints(record["inputs"], inputs, "input", mismatch)
    _verify_recorded_outputs(ctx, stage, record["outputs"], mismatch)

    try:
        if stage.stage_type == "calibration":
            calibration_input_gate(ctx, stage)
        if stage.stage_type == "comparison":
            _require_comparison_argv(ctx.plan, stage)
        _POST_VALIDATORS[stage.stage_type](ctx, stage)
    except _ORDINARY_STAGE_ERRORS as error:
        raise mismatch(f"post-validation no longer passes: {error}") from error
    require_repository_gate(ctx.plan, runner=ctx.git_runner)
    ctx.completed[stage.stage_id] = CompletedStage(path, sha256_file(path), record)


def _require_same_fingerprints(
    recorded: Mapping[str, Any], current: Mapping[str, Any], label: str, mismatch
) -> None:
    if list(recorded) != list(current):
        raise mismatch(f"{label} names {list(recorded)} differ from expected {list(current)}")
    for name, fingerprint in current.items():
        if recorded[name] != fingerprint:
            raise mismatch(f"{label} {name} changed: {fingerprint['path']}")


def _verify_recorded_outputs(
    ctx: _ExecutionContext, stage: BenchmarkStage, recorded: Mapping[str, Any], mismatch
) -> None:
    specs = _output_specs(ctx, stage)
    if list(recorded) != [name for name, _, _ in specs]:
        raise mismatch(f"output names {list(recorded)} differ from the expected outputs")
    for name, path, kind in specs:
        fingerprint = recorded[name]
        if fingerprint["path"] != str(path) or fingerprint["kind"] != kind:
            raise mismatch(f"output {name} location or kind differs")
        if kind == "file":
            try:
                current = file_fingerprint(path)
            except StageRecordError as error:
                raise mismatch(f"output {name} removed or unreadable: {error}") from error
            if current != fingerprint:
                raise mismatch(f"output {name} changed: {path}")
        else:
            _verify_directory_output(ctx, stage, name, fingerprint, path, mismatch)


def _verify_directory_output(
    ctx: _ExecutionContext,
    stage: BenchmarkStage,
    name: str,
    fingerprint: Mapping[str, Any],
    path: Path,
    mismatch,
) -> None:
    """Recompute an output tree and name any added, removed, or changed file."""
    manifest_path = output_manifest_path(ctx.execution_dir, stage.stage_id)
    if fingerprint["manifest_path"] != str(manifest_path):
        raise mismatch(f"output {name} manifest location differs")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise mismatch(f"output {name} manifest is unreadable: {error}") from error
    if hashlib.sha256(manifest_bytes).hexdigest() != fingerprint["manifest_sha256"]:
        raise mismatch(f"persisted output manifest changed: {manifest_path}")
    persisted = {}
    for line in manifest_bytes.decode("utf-8").splitlines()[1:]:
        entry = json.loads(line)
        persisted[entry["path"]] = entry
    try:
        current_manifest = directory_manifest(path)
    except StageRecordError as error:
        raise mismatch(f"output {name}: {error}") from error
    current = {entry["path"]: entry for entry in current_manifest["entries"]}
    added = sorted(set(current) - set(persisted))
    removed = sorted(set(persisted) - set(current))
    changed = sorted(key for key in set(current) & set(persisted) if current[key] != persisted[key])
    if added or removed or changed:
        raise mismatch(
            f"output tree {path} changed: added={added[:5]}, removed={removed[:5]}, "
            f"changed={changed[:5]}"
        )
    for key in ("n_files", "total_size"):
        if current_manifest[key] != fingerprint[key]:
            raise mismatch(f"output {name} {key} changed")
    if current_manifest["sha256"] != fingerprint["manifest_sha256"]:
        raise mismatch(f"output {name} manifest hash changed")


# ---------------------------------------------------------------------------
# Benchmark state scan and top-level executor
# ---------------------------------------------------------------------------


def scan_benchmark_state(
    execution_dir: str | Path, stages: Sequence[BenchmarkStage], *, resume: bool
) -> int:
    """Return how many leading stages hold completed records that resume must revalidate.

    Without *resume*, any stage record, any non-empty owned output, or any
    existing stage record/log/manifest bookkeeping fails closed: completed
    work is never silently accepted. With *resume*, ``running`` and ``failed``
    records always fail, completed records must form a prefix of the DAG, and
    no stage after that prefix may have any record.
    """
    execution_dir = Path(execution_dir)
    records = {
        stage.stage_id: existing_stage_records(execution_dir, stage.stage_id) for stage in stages
    }
    if not resume:
        present = [stage_id for stage_id, found in records.items() if found]
        if present:
            raise StageStateError(
                f"benchmark already has stage records ({present[:5]}); rerun with --resume "
                "to revalidate and reuse completed stages"
            )
        for stage in stages:
            for path in stage.owned_paths:
                if _has_partial_output(Path(path)):
                    raise StageStateError(
                        f"stage {stage.stage_id} output already exists without a completed "
                        f"record: {path}"
                    )
        for name in STAGES_STATE_DIRNAMES:
            if _has_partial_output(execution_dir / name):
                raise StageStateError(
                    f"execution bookkeeping already exists: {execution_dir / name}; rerun "
                    "with --resume or recover manually"
                )
        return 0
    for stage in stages:
        for kind in ("running", "failed"):
            if kind in records[stage.stage_id]:
                raise StageStateError(
                    f"stage {stage.stage_id} has a {kind} record "
                    f"({records[stage.stage_id][kind]}); inspect its logs and outputs, archive "
                    "or remove the stage deliberately, then start again (no retry-failed mode)"
                )
    prefix = 0
    while prefix < len(stages) and "completed" in records[stages[prefix].stage_id]:
        prefix += 1
    for stage in stages[prefix:]:
        if records[stage.stage_id]:
            raise StageStateError(
                f"stage {stage.stage_id} has records although an earlier stage "
                f"({stages[prefix].stage_id}) has not completed"
            )
    return prefix


def build_execution_summary(
    ctx_plan: Mapping[str, Any],
    resolved_plan_sha256: str,
    stages: Sequence[BenchmarkStage],
    completed_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the deterministic benchmark execution summary (no timestamps).

    It claims no more than the executed DAG: a paired benchmark whose final
    stage was the provenance-gated calibrated position ranking comparison.
    """
    done = set(completed_ids)

    def count(stage_type: str, side: str | None = None) -> int:
        return sum(
            1
            for stage in stages
            if stage.stage_id in done
            and stage.stage_type == stage_type
            and (side is None or stage.side == side)
        )

    return {
        "schema_version": EXECUTION_SUMMARY_SCHEMA_VERSION,
        "status": EXECUTION_SUMMARY_STATUS,
        "resolved_plan_sha256": resolved_plan_sha256,
        "repository_revision": ctx_plan["repository_revision"],
        "manifest_file_sha256": ctx_plan["manifest_file_sha256"],
        "n_runs": len(ctx_plan["runs"]),
        "n_stages": len(stages),
        "completed_stages": len(done),
        "real_training_completed": count("training", "real"),
        "null_training_completed": count("training", "null"),
        "real_explanations_completed": count("explanation", "real"),
        "null_explanations_completed": count("explanation", "null"),
        "pair_validations_completed": count("pair_validation"),
        "calibrations_completed": count("calibration"),
        "shared_null_validation": "passed" if SHARED_NULL_STAGE_ID in done else "not_run",
        "raw_comparisons_completed": [
            name for name in COMPARISON_KEYS if f"comparisons/{name}" in done
        ],
        "calibrated_position_ranking_comparison": (
            "completed" if CALIBRATED_COMPARISON_STAGE_ID in done else "not_run"
        ),
        "calibrated_score_column": CALIBRATED_SCORE_COLUMN,
    }


def _publish_execution_summary(execution_dir: Path, summary: Mapping[str, Any]) -> Path:
    path = execution_dir / EXECUTION_SUMMARY_NAME
    data = dump_yaml_bytes(summary)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise StageStateError(f"existing execution summary disagrees: {path}")
        return path
    return atomic_write_bytes(path, data)


def execute_benchmark_plan(
    plan_path: str | Path,
    *,
    manifest_path: str | Path,
    resume: bool = False,
    git_runner: GitRunner = subprocess.run,
    probe_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    null_validator: Callable[..., Mapping[str, Any]] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    rebuild: Callable[..., dict[str, Any]] = build_resolved_plan,
) -> dict[str, Any]:
    """Execute (or resume) one reviewed schema-v2 plan through the full paired DAG.

    Authority sequence: ``verify_plan_rebuild`` -> repository gate -> output
    locations -> ``ExecutionLock`` (held throughout) -> ``bind_benchmark_plan``
    -> environment probe -> full ``run_null_preflight`` -> state scan -> DAG.
    No benchmark subprocess starts before the null preflight passed. Returns
    ``{"summary", "summary_path", "executed", "reused"}``.
    """
    verified = verify_plan_rebuild(plan_path, manifest_path=manifest_path, rebuild=rebuild)
    plan = verified.plan
    require_repository_gate(plan, runner=git_runner)
    require_execution_locations_safe(plan, plan_path, runner=git_runner)
    benchmark_root = benchmark_root_from_plan(plan)
    stages = build_benchmark_stages(plan)
    execution_dir = benchmark_root / EXECUTION_DIRNAME
    with ExecutionLock(benchmark_root, resolved_plan_sha256=verified.resolved_plan_sha256):
        bind_benchmark_plan(benchmark_root, verified)
        environment = probe_execution_environment(plan, runner=probe_runner)
        preflight = run_null_preflight(plan, validator=null_validator)
        ctx = _ExecutionContext(
            verified=verified,
            benchmark_root=benchmark_root,
            execution_dir=execution_dir,
            environment=environment,
            preflight=preflight,
            git_runner=git_runner,
            popen=popen,
        )
        prefix = scan_benchmark_state(execution_dir, stages, resume=resume)
        executed, reused = [], []
        for index, stage in enumerate(stages):
            if index < prefix:
                _revalidate_completed_stage(ctx, stage)
                reused.append(stage.stage_id)
            else:
                _execute_stage(ctx, stage)
                executed.append(stage.stage_id)
        summary = build_execution_summary(
            plan, verified.resolved_plan_sha256, stages, list(ctx.completed)
        )
        summary_path = _publish_execution_summary(execution_dir, summary)
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "executed": executed,
        "reused": reused,
    }
