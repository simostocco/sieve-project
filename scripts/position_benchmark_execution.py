"""Execution foundation for paired (schema-v2) positional benchmarks.

Phase 12C3B2A. This module builds the safe substrate that Phase 12C3B2B will
wire into the benchmark stage DAG. It does NOT launch ``train.py``,
``explain.py``, pair calibration, ``bootstrap_null_calibration.py``, or the
B1/B2/B3 comparisons, and ``run_position_benchmark.py`` stays dry-run only.
The approved future interface (not exposed yet) is::

    python scripts/run_position_benchmark.py MANIFEST --execute-plan PLAN [--resume]

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
Before any future benchmark stage, the full 12C3A ``validate_null_pair`` (which
loads both cohorts) must pass and its recomputed identity must equal the
plan's ``null_binding`` exactly, and every ``input_files`` byte hash must still
match. The null artifact is never regenerated. The result is recorded in
``<benchmark_root>/null_binding/null_validation.yaml``, an execution-stage
record rather than scientific identity.

Split identity note: planner ``split_plan.membership_sha256`` and training
config ``split_plan.sha256`` (and ``input_sha256``) are all
``src.training.split_plan.split_plan_sha256()`` over the canonical membership
payload, which excludes ``seed`` and ``split_source``; 12C3B2B will require
their equality. ``input_files.split_plan.sha256`` is separately the raw
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
        PAIRED_SCHEMA_VERSION,
        BenchmarkManifestError,
        build_resolved_plan,
    )
    from position_benchmark_records import (
        GIT_REVISION_RE,
        SHA256_RE,
        StageRecordError,
        atomic_write_bytes,
        atomic_write_yaml,
        build_stage_record,
        existing_stage_records,
        file_fingerprint,
        load_stage_record,
        sha256_file,
        stage_record_path,
        utc_now_iso,
        write_stage_record,
    )
else:
    from .position_benchmark_manifest import (
        PAIRED_SCHEMA_VERSION,
        BenchmarkManifestError,
        build_resolved_plan,
    )
    from .position_benchmark_records import (
        GIT_REVISION_RE,
        SHA256_RE,
        StageRecordError,
        atomic_write_bytes,
        atomic_write_yaml,
        build_stage_record,
        existing_stage_records,
        file_fingerprint,
        load_stage_record,
        sha256_file,
        stage_record_path,
        utc_now_iso,
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
COMPARISON_KEYS = ("performance", "raw_rankings", "raw_attributions")
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
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(COMPARISON_KEYS):
        raise PlanAuthorityError(f"comparisons must have exactly {list(COMPARISON_KEYS)}")
    for key in COMPARISON_KEYS:
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
