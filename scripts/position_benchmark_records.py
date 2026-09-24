"""Execution-stage records, byte fingerprints, and atomic writes for benchmarks.

Phase 12C3B2A. This module is the execution-provenance substrate for the
paired positional benchmark executor (wired into a stage DAG in Phase
12C3B2B). It deliberately has no scientific content: it never trains,
explains, calibrates, or interprets model outputs, and nothing here enters
positional strategy identity or any scientific hash.

It provides:

- streaming SHA-256 file fingerprints (``path``, ``sha256``, ``size``);
- deterministic ``sieve.directory_manifest.v1`` directory fingerprints that
  hash relative paths, sizes, and file bytes only (never mtimes, inodes, or
  absolute paths) and reject symlinks and special files;
- atomic publication (same-directory temporary file, ``fsync``,
  ``os.replace``, parent-directory ``fsync``) that refuses to overwrite by
  default;
- versioned ``running`` / ``completed`` / ``failed`` stage records with an
  explicit schema validator. Only a schema-valid ``completed`` record can
  ever authorize reuse of a stage; ``running`` and ``failed`` records never
  can, and mere file existence is never treated as validity.

The module imports only the standard library and PyYAML so the executor can
fingerprint and validate records without importing torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RECORD_SCHEMA_VERSION = 1
DIRECTORY_MANIFEST_SCHEMA = "sieve.directory_manifest.v1"
RECORD_KINDS = ("running", "completed", "failed")
STAGE_TYPES = (
    "null_validation",
    "training",
    "explanation",
    "pair_validation",
    "calibration",
    "shared_null",
    "comparison",
)
STAGE_SIDES = ("real", "null", "paired", "benchmark", "comparison")
EXECUTION_KINDS = ("subprocess", "in_process")
FINGERPRINT_KINDS = ("file", "directory")
STAGES_DIRNAME = "stages"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# Stage IDs are path-like (``runs/<run_id>/real_training``) because benchmark
# run IDs may contain ``.``; a dotted stage ID would therefore be ambiguous.
STAGE_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_COMMON_RECORD_KEYS = (
    "schema_version",
    "record_kind",
    "stage_id",
    "stage_type",
    "run_id",
    "side",
    "repository_revision",
    "resolved_plan_sha256",
    "manifest_file_sha256",
    "execution",
    "inputs",
    "dependencies",
    "environment",
    "started_at",
)
_KIND_SPECIFIC_KEYS = {
    "running": (),
    "completed": ("completed_at", "outputs", "post_validation"),
    "failed": ("completed_at", "failure"),
}
_HASH_CHUNK_BYTES = 1024 * 1024


class StageRecordError(ValueError):
    """Raised when a fingerprint, directory manifest, or stage record is invalid."""


# ---------------------------------------------------------------------------
# Time and canonical serialization
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (execution provenance only)."""
    return datetime.now(timezone.utc).isoformat()


def canonical_json_line(value: Any) -> bytes:
    """Return *value* as one canonical JSON line (sorted keys, no whitespace, ``\\n``)."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def dump_yaml_bytes(mapping: Mapping[str, Any]) -> bytes:
    """Serialize *mapping* exactly as the benchmark YAML writers do (insertion order)."""
    return yaml.safe_dump(dict(mapping), sort_keys=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------


def fsync_directory(path: Path) -> None:
    """Flush a directory entry to disk where the platform supports it.

    POSIX needs a directory ``fsync`` for a completed ``os.replace`` to be
    durable across power loss. Platforms or filesystems that cannot open a
    directory for reading (for example Windows) skip this step silently; the
    rename itself is still atomic.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> Path:
    """Atomically publish *data* at *path* and return the path.

    Writes to a temporary file in the same directory, flushes and ``fsync``s
    it, then ``os.replace``s it into place and ``fsync``s the parent. Readers
    therefore see either no file or the complete bytes, never a partial file.

    With ``overwrite=False`` (the default) an existing *path* is refused. The
    existence check and the rename are not one atomic operation; concurrent
    writers to one benchmark root are excluded by the execution lock instead.
    The parent directory must already exist.
    """
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {parent}")
    if not overwrite and os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    fd, temp_name = tempfile.mkstemp(dir=parent, prefix=f".{target.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite existing file: {target}")
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    fsync_directory(parent)
    return target


def atomic_write_yaml(path: str | Path, mapping: Mapping[str, Any]) -> Path:
    """Atomically publish *mapping* as YAML at *path*, refusing overwrite."""
    return atomic_write_bytes(path, dump_yaml_bytes(mapping), overwrite=False)


# ---------------------------------------------------------------------------
# File fingerprints
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 of a regular file's raw bytes."""
    return file_fingerprint(path)["sha256"]


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return ``{kind, path, sha256, size}`` for one regular file.

    Symlinks, directories, missing paths, and special files are rejected so a
    fingerprint always names exactly the bytes that were hashed. The size is
    the number of bytes actually hashed and must agree with ``fstat`` taken
    before reading, which catches a file being rewritten mid-hash.
    """
    file_path = Path(path)
    try:
        link_stat = os.lstat(file_path)
    except FileNotFoundError as error:
        raise StageRecordError(f"required regular file does not exist: {file_path}") from error
    if stat.S_ISLNK(link_stat.st_mode):
        raise StageRecordError(f"symlinks are not accepted as fingerprinted files: {file_path}")
    if not stat.S_ISREG(link_stat.st_mode):
        raise StageRecordError(f"path is not a regular file: {file_path}")
    digest = hashlib.sha256()
    hashed_bytes = 0
    with file_path.open("rb") as handle:
        expected_size = os.fstat(handle.fileno()).st_size
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            hashed_bytes += len(chunk)
    if hashed_bytes != expected_size:
        raise StageRecordError(f"file changed while it was being hashed: {file_path}")
    return {
        "kind": "file",
        "path": str(file_path),
        "sha256": digest.hexdigest(),
        "size": hashed_bytes,
    }


# ---------------------------------------------------------------------------
# Directory manifests
# ---------------------------------------------------------------------------


def directory_manifest(root: str | Path) -> dict[str, Any]:
    """Return a deterministic ``sieve.directory_manifest.v1`` for *root*.

    Every regular file below *root* contributes ``{path, size, sha256}``,
    where ``path`` is the POSIX path relative to *root*. Entries are sorted by
    that relative path. Symlinks (to files or directories), sockets, FIFOs,
    devices, and any other non-regular entries are rejected rather than
    skipped, so nothing can hide outside the manifest. Empty directories
    contain no files and therefore do not contribute entries.

    The manifest hash is SHA-256 over one canonical JSON header line
    (``schema``, ``n_files``) followed by one canonical JSON line per entry.
    It excludes mtimes, inodes, permissions, and the absolute root path, so it
    identifies content, not filesystem metadata or location.
    """
    root_path = Path(root)
    try:
        root_stat = os.lstat(root_path)
    except FileNotFoundError as error:
        raise StageRecordError(f"directory does not exist: {root_path}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise StageRecordError(f"path is not a real directory: {root_path}")

    entries: list[dict[str, Any]] = []
    pending = [root_path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            children = list(iterator)
        for child in children:
            child_path = Path(child.path)
            relative = child_path.relative_to(root_path).as_posix()
            if child.is_symlink():
                raise StageRecordError(f"symlink found in directory manifest: {relative}")
            if child.is_dir(follow_symlinks=False):
                pending.append(child_path)
            elif child.is_file(follow_symlinks=False):
                fingerprint = file_fingerprint(child_path)
                entries.append(
                    {"path": relative, "size": fingerprint["size"], "sha256": fingerprint["sha256"]}
                )
            else:
                raise StageRecordError(f"special file found in directory manifest: {relative}")
    entries.sort(key=lambda entry: entry["path"])
    lines = directory_manifest_lines(entries)
    return {
        "schema": DIRECTORY_MANIFEST_SCHEMA,
        "n_files": len(entries),
        "total_size": sum(entry["size"] for entry in entries),
        "sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
        "entries": entries,
    }


def directory_manifest_lines(entries: Sequence[Mapping[str, Any]]) -> list[bytes]:
    """Return the canonical header and entry lines whose bytes define the manifest hash."""
    header = canonical_json_line({"schema": DIRECTORY_MANIFEST_SCHEMA, "n_files": len(entries)})
    return [
        header,
        *(
            canonical_json_line(
                {"path": entry["path"], "sha256": entry["sha256"], "size": entry["size"]}
            )
            for entry in entries
        ),
    ]


def write_directory_manifest_jsonl(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically write *manifest* as JSONL whose file SHA-256 equals ``manifest['sha256']``."""
    data = b"".join(directory_manifest_lines(manifest["entries"]))
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise StageRecordError("directory manifest entries do not match its sha256")
    return atomic_write_bytes(path, data, overwrite=False)


def directory_fingerprint(root: str | Path, *, manifest_path: str | Path | None = None) -> dict:
    """Return a record fingerprint for a directory, optionally persisting its JSONL manifest."""
    manifest = directory_manifest(root)
    if manifest_path is not None:
        write_directory_manifest_jsonl(manifest_path, manifest)
    return {
        "kind": "directory",
        "path": str(Path(root)),
        "manifest_schema": DIRECTORY_MANIFEST_SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "n_files": manifest["n_files"],
        "total_size": manifest["total_size"],
        "manifest_path": None if manifest_path is None else str(Path(manifest_path)),
    }


# ---------------------------------------------------------------------------
# Stage records
# ---------------------------------------------------------------------------


def validate_stage_id(stage_id: Any) -> str:
    """Validate a path-like stage ID such as ``runs/L3_rope/real_training``."""
    if not isinstance(stage_id, str) or not stage_id:
        raise StageRecordError("stage_id must be a non-empty string")
    for segment in stage_id.split("/"):
        if not STAGE_ID_SEGMENT_RE.match(segment) or segment in {".", ".."}:
            raise StageRecordError(f"stage_id has an invalid segment: {stage_id!r}")
    return stage_id


def stage_record_path(execution_dir: str | Path, stage_id: str, record_kind: str) -> Path:
    """Return ``<execution_dir>/stages/<stage_id>.<record_kind>.yaml``."""
    validate_stage_id(stage_id)
    if record_kind not in RECORD_KINDS:
        raise StageRecordError(f"record_kind must be one of {RECORD_KINDS}")
    return Path(execution_dir) / STAGES_DIRNAME / f"{stage_id}.{record_kind}.yaml"


def build_stage_record(
    *,
    record_kind: str,
    stage_id: str,
    stage_type: str,
    run_id: str | None,
    side: str,
    repository_revision: str,
    resolved_plan_sha256: str,
    manifest_file_sha256: str,
    execution: Mapping[str, Any],
    inputs: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    started_at: str,
    completed_at: str | None = None,
    outputs: Mapping[str, Any] | None = None,
    post_validation: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one stage record of *record_kind*.

    ``running`` records carry only the common fields. ``completed`` records
    additionally require ``completed_at``, ``outputs``, and a passed
    ``post_validation``; callers must build them only after output validation
    and hashing succeeded. ``failed`` records require ``completed_at`` and a
    ``failure`` block and never carry ``outputs``, so they cannot be mistaken
    for reusable work.
    """
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_kind": record_kind,
        "stage_id": stage_id,
        "stage_type": stage_type,
        "run_id": run_id,
        "side": side,
        "repository_revision": repository_revision,
        "resolved_plan_sha256": resolved_plan_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "execution": _plain(execution),
        "inputs": _plain(inputs),
        "dependencies": _plain(list(dependencies)),
        "environment": _plain(environment),
        "started_at": started_at,
    }
    if record_kind in {"completed", "failed"}:
        record["completed_at"] = completed_at
    if record_kind == "completed":
        record["outputs"] = _plain(outputs)
        record["post_validation"] = _plain(post_validation)
    if record_kind == "failed":
        record["failure"] = _plain(failure)
    validate_stage_record(record)
    return record


def validate_stage_record(record: Any) -> dict[str, Any]:
    """Strictly validate a stage record mapping and return it as a plain dict."""
    if not isinstance(record, Mapping):
        raise StageRecordError("stage record must be a mapping")
    record = dict(record)
    kind = record.get("record_kind")
    if kind not in RECORD_KINDS:
        raise StageRecordError(f"record_kind must be one of {RECORD_KINDS}, got {kind!r}")
    expected_keys = set(_COMMON_RECORD_KEYS) | set(_KIND_SPECIFIC_KEYS[kind])
    actual_keys = set(record)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys, key=str)
        raise StageRecordError(
            f"{kind} stage record keys are invalid: missing={missing}, unexpected={extra}"
        )
    version = record["schema_version"]
    if isinstance(version, bool) or version != RECORD_SCHEMA_VERSION:
        raise StageRecordError(f"stage record schema_version must be {RECORD_SCHEMA_VERSION}")
    validate_stage_id(record["stage_id"])
    if record["stage_type"] not in STAGE_TYPES:
        raise StageRecordError(f"stage_type must be one of {STAGE_TYPES}")
    if record["run_id"] is not None and (
        not isinstance(record["run_id"], str) or not STAGE_ID_SEGMENT_RE.match(record["run_id"])
    ):
        raise StageRecordError("run_id must be null or a valid run identifier")
    if record["side"] not in STAGE_SIDES:
        raise StageRecordError(f"side must be one of {STAGE_SIDES}")
    _require_pattern(record["repository_revision"], GIT_REVISION_RE, "repository_revision")
    _require_pattern(record["resolved_plan_sha256"], SHA256_RE, "resolved_plan_sha256")
    _require_pattern(record["manifest_file_sha256"], SHA256_RE, "manifest_file_sha256")
    _validate_execution(record["execution"], kind)
    _validate_fingerprint_map(record["inputs"], "inputs")
    _validate_dependencies(record["dependencies"])
    if not isinstance(record["environment"], Mapping):
        raise StageRecordError("environment must be a mapping")
    _require_timestamp(record["started_at"], "started_at")
    if kind in {"completed", "failed"}:
        _require_timestamp(record["completed_at"], "completed_at")
    if kind == "completed":
        _validate_fingerprint_map(record["outputs"], "outputs")
        if not record["outputs"]:
            raise StageRecordError("completed stage record must hash at least one output")
        _validate_post_validation(record["post_validation"])
        if record["execution"]["kind"] == "subprocess" and record["execution"]["exit_code"] != 0:
            raise StageRecordError("completed subprocess stage must have exit_code 0")
    if kind == "failed":
        _validate_failure(record["failure"])
    return record


def write_stage_record(execution_dir: str | Path, record: Mapping[str, Any]) -> Path:
    """Validate *record* and atomically publish it at its canonical path (no overwrite)."""
    validated = validate_stage_record(record)
    path = stage_record_path(execution_dir, validated["stage_id"], validated["record_kind"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_yaml(path, validated)


def load_stage_record(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate one stage record file."""
    record_path = Path(path)
    try:
        with record_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise StageRecordError(f"cannot read stage record {record_path}: {error}") from error
    return validate_stage_record(data)


def require_completed_record(path: str | Path) -> dict[str, Any]:
    """Return a schema-valid ``completed`` record; ``running``/``failed`` never qualify."""
    record = load_stage_record(path)
    if record["record_kind"] != "completed":
        raise StageRecordError(
            f"stage record {path} is {record['record_kind']!r}; only a completed record "
            "can authorize reuse"
        )
    # A record must sit at its canonical location so a copied or renamed file
    # cannot vouch for a different stage.
    expected_suffix = f"/{STAGES_DIRNAME}/{record['stage_id']}.completed.yaml"
    if not Path(path).as_posix().endswith(expected_suffix):
        raise StageRecordError(f"stage record location does not match its stage_id: {path}")
    return record


def existing_stage_records(execution_dir: str | Path, stage_id: str) -> dict[str, Path]:
    """Return the record files that currently exist for *stage_id*, keyed by kind."""
    found = {}
    for kind in RECORD_KINDS:
        path = stage_record_path(execution_dir, stage_id, kind)
        if os.path.lexists(path):
            found[kind] = path
    return found


def dependency_reference(record_path: str | Path) -> dict[str, Any]:
    """Return ``{stage_id, record_path, record_sha256}`` for a completed dependency record."""
    record = require_completed_record(record_path)
    return {
        "stage_id": record["stage_id"],
        "record_path": str(Path(record_path)),
        "record_sha256": sha256_file(record_path),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Return a YAML-safe deep copy with mappings as dicts and sequences as lists."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _require_pattern(value: Any, pattern: re.Pattern[str], name: str) -> None:
    if not isinstance(value, str) or not pattern.match(value):
        raise StageRecordError(f"{name} is not valid: {value!r}")


def _require_timestamp(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise StageRecordError(f"{name} must be a non-empty ISO-8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise StageRecordError(f"{name} is not an ISO-8601 timestamp: {value!r}") from error


def _require_int(value: Any, name: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageRecordError(f"{name} must be an integer")


def _validate_execution(execution: Any, kind: str) -> None:
    if not isinstance(execution, Mapping):
        raise StageRecordError("execution must be a mapping")
    execution_kind = execution.get("kind")
    if execution_kind == "subprocess":
        expected = {"kind", "argv", "cwd", "logs", "exit_code"}
        argv = execution.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise StageRecordError("execution.argv must be a non-empty list of strings")
        if not isinstance(execution.get("cwd"), str) or not execution["cwd"]:
            raise StageRecordError("execution.cwd must be a non-empty string")
        logs = execution.get("logs")
        if logs is not None:
            if not isinstance(logs, Mapping) or set(logs) != {"stdout", "stderr"}:
                raise StageRecordError("execution.logs must be null or {stdout, stderr}")
            for stream, fingerprint in logs.items():
                _validate_fingerprint(fingerprint, f"execution.logs.{stream}", kind="file")
        _require_int(execution.get("exit_code"), "execution.exit_code", allow_none=True)
    elif execution_kind == "in_process":
        expected = {"kind", "callable"}
        if not isinstance(execution.get("callable"), str) or not execution["callable"]:
            raise StageRecordError("execution.callable must be a non-empty string")
    else:
        raise StageRecordError(f"execution.kind must be one of {EXECUTION_KINDS}")
    if set(execution) != expected:
        raise StageRecordError(
            f"execution keys for {execution_kind} must be {sorted(expected)}, "
            f"got {sorted(execution, key=str)}"
        )
    if kind == "running" and execution_kind == "subprocess" and execution["exit_code"] is not None:
        raise StageRecordError("running subprocess record must have exit_code null")


def _validate_fingerprint(fingerprint: Any, name: str, *, kind: str | None = None) -> None:
    if not isinstance(fingerprint, Mapping):
        raise StageRecordError(f"{name} must be a fingerprint mapping")
    fingerprint_kind = fingerprint.get("kind")
    if fingerprint_kind not in FINGERPRINT_KINDS or (kind is not None and fingerprint_kind != kind):
        raise StageRecordError(f"{name}.kind is invalid: {fingerprint_kind!r}")
    if not isinstance(fingerprint.get("path"), str) or not fingerprint["path"]:
        raise StageRecordError(f"{name}.path must be a non-empty string")
    if fingerprint_kind == "file":
        if set(fingerprint) != {"kind", "path", "sha256", "size"}:
            raise StageRecordError(f"{name} file fingerprint keys are invalid")
        _require_pattern(fingerprint["sha256"], SHA256_RE, f"{name}.sha256")
        _require_int(fingerprint["size"], f"{name}.size")
    else:
        expected = {
            "kind",
            "path",
            "manifest_schema",
            "manifest_sha256",
            "n_files",
            "total_size",
            "manifest_path",
        }
        if set(fingerprint) != expected:
            raise StageRecordError(f"{name} directory fingerprint keys are invalid")
        if fingerprint["manifest_schema"] != DIRECTORY_MANIFEST_SCHEMA:
            raise StageRecordError(f"{name}.manifest_schema must be {DIRECTORY_MANIFEST_SCHEMA}")
        _require_pattern(fingerprint["manifest_sha256"], SHA256_RE, f"{name}.manifest_sha256")
        _require_int(fingerprint["n_files"], f"{name}.n_files")
        _require_int(fingerprint["total_size"], f"{name}.total_size")
        if fingerprint["manifest_path"] is not None and not isinstance(
            fingerprint["manifest_path"], str
        ):
            raise StageRecordError(f"{name}.manifest_path must be null or a string")


def _validate_fingerprint_map(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise StageRecordError(f"{name} must be a mapping of logical name to fingerprint")
    for logical_name, fingerprint in value.items():
        if not isinstance(logical_name, str) or not logical_name:
            raise StageRecordError(f"{name} logical names must be non-empty strings")
        if fingerprint is None:
            continue
        _validate_fingerprint(fingerprint, f"{name}.{logical_name}")


def _validate_dependencies(value: Any) -> None:
    if not isinstance(value, list):
        raise StageRecordError("dependencies must be a list")
    for index, dependency in enumerate(value):
        if not isinstance(dependency, Mapping) or set(dependency) != {
            "stage_id",
            "record_path",
            "record_sha256",
        }:
            raise StageRecordError(
                f"dependencies[{index}] must be {{stage_id, record_path, record_sha256}}"
            )
        validate_stage_id(dependency["stage_id"])
        if not isinstance(dependency["record_path"], str) or not dependency["record_path"]:
            raise StageRecordError(f"dependencies[{index}].record_path must be a string")
        _require_pattern(dependency["record_sha256"], SHA256_RE, f"dependencies[{index}]")


def _validate_post_validation(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"status", "checks"}:
        raise StageRecordError("post_validation must be {status, checks}")
    if value["status"] != "passed":
        raise StageRecordError("completed stage record requires post_validation.status passed")
    checks = value["checks"]
    if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
        raise StageRecordError("post_validation.checks must be a list of strings")


def _validate_failure(value: Any) -> None:
    expected = {"reason", "exit_code", "exception", "partial_outputs_present"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StageRecordError(f"failure must have keys {sorted(expected)}")
    if not isinstance(value["reason"], str) or not value["reason"]:
        raise StageRecordError("failure.reason must be a non-empty string")
    _require_int(value["exit_code"], "failure.exit_code", allow_none=True)
    if value["exception"] is not None and not isinstance(value["exception"], str):
        raise StageRecordError("failure.exception must be null or a string")
    if not isinstance(value["partial_outputs_present"], bool):
        raise StageRecordError("failure.partial_outputs_present must be a boolean")
