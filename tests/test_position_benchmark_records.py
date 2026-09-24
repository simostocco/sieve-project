"""Tests for Phase 12C3B2A execution-stage records, fingerprints, and atomic writes."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path

import pytest
import yaml

from scripts import position_benchmark_records as records
from scripts.position_benchmark_records import (
    DIRECTORY_MANIFEST_SCHEMA,
    StageRecordError,
    atomic_write_bytes,
    build_stage_record,
    dependency_reference,
    directory_fingerprint,
    directory_manifest,
    existing_stage_records,
    file_fingerprint,
    load_stage_record,
    require_completed_record,
    sha256_file,
    stage_record_path,
    validate_stage_record,
    write_directory_manifest_jsonl,
    write_stage_record,
)

REVISION = "a" * 40
PLAN_SHA = "b" * 64
MANIFEST_SHA = "c" * 64
STARTED = "2026-09-23T10:00:00+00:00"
COMPLETED = "2026-09-23T11:00:00+00:00"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path, files: dict[str, bytes], order: list[str] | None = None) -> Path:
    for name in order or list(files):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name])
    return root


TREE = {
    "attributions.npz": b"npz-bytes",
    "analysis_metadata.yaml": b"meta: 1\n",
    "attributions_per_sample/sample_0.npz": b"s0",
    "attributions_per_sample/sample_1.npz": b"s1",
    "attributions_per_sample/nested/deep.txt": b"deep",
}


def _file_fp(path: Path) -> dict:
    return file_fingerprint(path)


def _common(**overrides) -> dict:
    fields = {
        "stage_id": "runs/L3_rope/real_training",
        "stage_type": "training",
        "run_id": "L3_rope",
        "side": "real",
        "repository_revision": REVISION,
        "resolved_plan_sha256": PLAN_SHA,
        "manifest_file_sha256": MANIFEST_SHA,
        "inputs": {},
        "dependencies": [],
        "environment": {"hostname": "test-host"},
        "started_at": STARTED,
    }
    fields.update(overrides)
    return fields


def _subprocess_execution(exit_code=None, logs=None) -> dict:
    return {
        "kind": "subprocess",
        "argv": ["/usr/bin/python", "scripts/train.py", "--seed", "42"],
        "cwd": "/repo",
        "logs": logs,
        "exit_code": exit_code,
    }


def _running(**overrides) -> dict:
    return build_stage_record(
        record_kind="running", execution=_subprocess_execution(), **_common(**overrides)
    )


def _completed(tmp_path: Path, **overrides) -> dict:
    output = tmp_path / "out.bin"
    output.write_bytes(b"output")
    return build_stage_record(
        record_kind="completed",
        execution=_subprocess_execution(exit_code=0),
        completed_at=COMPLETED,
        outputs={"config": _file_fp(output)},
        post_validation={"status": "passed", "checks": ["files_exist"]},
        **_common(**overrides),
    )


def _failed(**overrides) -> dict:
    return build_stage_record(
        record_kind="failed",
        execution=_subprocess_execution(exit_code=2),
        completed_at=COMPLETED,
        failure={
            "reason": "subprocess_failed",
            "exit_code": 2,
            "exception": "stage exited with code 2",
            "partial_outputs_present": True,
        },
        **_common(**overrides),
    )


# ---------------------------------------------------------------------------
# File fingerprints
# ---------------------------------------------------------------------------


def test_file_fingerprint_is_streaming_sha_and_size(tmp_path):
    path = tmp_path / "data.bin"
    data = os.urandom(3 * 1024 * 1024 + 17)
    path.write_bytes(data)

    fingerprint = file_fingerprint(path)

    assert fingerprint == {
        "kind": "file",
        "path": str(path),
        "sha256": _sha(data),
        "size": len(data),
    }
    assert file_fingerprint(path) == fingerprint
    assert sha256_file(path) == _sha(data)


def test_file_fingerprint_ignores_mtime(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"same")
    before = file_fingerprint(path)
    os.utime(path, (1, 1))
    assert file_fingerprint(path) == before


def test_file_fingerprint_rejects_missing_directory_and_symlink(tmp_path):
    with pytest.raises(StageRecordError, match="does not exist"):
        file_fingerprint(tmp_path / "missing")
    with pytest.raises(StageRecordError, match="not a regular file"):
        file_fingerprint(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(StageRecordError, match="symlink"):
        file_fingerprint(link)


def test_file_fingerprint_rejects_fifo(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(StageRecordError, match="not a regular file"):
        file_fingerprint(fifo)


# ---------------------------------------------------------------------------
# Directory manifests
# ---------------------------------------------------------------------------


def test_directory_manifest_is_deterministic_and_sorted(tmp_path):
    first = _tree(tmp_path / "a", TREE)
    second = _tree(tmp_path / "b", TREE, order=list(reversed(list(TREE))))

    left = directory_manifest(first)
    right = directory_manifest(second)

    assert left == right
    assert left["schema"] == DIRECTORY_MANIFEST_SCHEMA
    assert left["n_files"] == len(TREE)
    assert left["total_size"] == sum(len(value) for value in TREE.values())
    paths = [entry["path"] for entry in left["entries"]]
    assert paths == sorted(TREE)
    assert "attributions_per_sample/nested/deep.txt" in paths
    for entry in left["entries"]:
        assert entry == {
            "path": entry["path"],
            "size": len(TREE[entry["path"]]),
            "sha256": _sha(TREE[entry["path"]]),
        }


def test_directory_manifest_is_location_and_mtime_independent(tmp_path):
    root = _tree(tmp_path / "a", TREE)
    before = directory_manifest(root)
    moved = tmp_path / "elsewhere" / "copy"
    shutil.copytree(root, moved)
    for path in moved.rglob("*"):
        os.utime(path, (5, 5))

    assert directory_manifest(moved)["sha256"] == before["sha256"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda root: (root / "extra.txt").write_bytes(b"x"), id="added"),
        pytest.param(lambda root: (root / "attributions.npz").unlink(), id="removed"),
        pytest.param(
            lambda root: (root / "attributions_per_sample" / "sample_0.npz").write_bytes(b"S0"),
            id="modified",
        ),
        pytest.param(
            lambda root: (root / "attributions.npz").rename(root / "attributions2.npz"),
            id="renamed",
        ),
    ],
)
def test_directory_manifest_hash_changes_on_content_change(tmp_path, mutate):
    root = _tree(tmp_path / "a", TREE)
    before = directory_manifest(root)["sha256"]
    mutate(root)
    assert directory_manifest(root)["sha256"] != before


def test_directory_manifest_rejects_file_and_directory_symlinks(tmp_path):
    root = _tree(tmp_path / "a", TREE)
    (root / "link.npz").symlink_to(root / "attributions.npz")
    with pytest.raises(StageRecordError, match="symlink"):
        directory_manifest(root)

    root2 = _tree(tmp_path / "b", TREE)
    (root2 / "dirlink").symlink_to(root2 / "attributions_per_sample", target_is_directory=True)
    with pytest.raises(StageRecordError, match="symlink"):
        directory_manifest(root2)


def test_directory_manifest_rejects_special_files_and_symlinked_root(tmp_path):
    root = _tree(tmp_path / "a", TREE)
    os.mkfifo(root / "attributions_per_sample" / "pipe")
    with pytest.raises(StageRecordError, match="special file"):
        directory_manifest(root)

    real = _tree(tmp_path / "real", TREE)
    link = tmp_path / "rootlink"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(StageRecordError, match="not a real directory"):
        directory_manifest(link)
    with pytest.raises(StageRecordError, match="does not exist"):
        directory_manifest(tmp_path / "missing")


def test_directory_manifest_jsonl_bytes_hash_to_manifest_sha(tmp_path):
    root = _tree(tmp_path / "a", TREE)
    manifest = directory_manifest(root)
    jsonl = tmp_path / "manifest.jsonl"

    write_directory_manifest_jsonl(jsonl, manifest)

    data = jsonl.read_bytes()
    assert _sha(data) == manifest["sha256"]
    lines = data.decode("utf-8").splitlines()
    assert lines[0] == f'{{"n_files":{len(TREE)},"schema":"{DIRECTORY_MANIFEST_SCHEMA}"}}'
    assert len(lines) == len(TREE) + 1
    assert "mtime" not in data.decode() and str(tmp_path) not in data.decode()
    with pytest.raises(FileExistsError):
        write_directory_manifest_jsonl(jsonl, manifest)


def test_directory_fingerprint_matches_manifest(tmp_path):
    root = _tree(tmp_path / "a", TREE)
    fingerprint = directory_fingerprint(root, manifest_path=tmp_path / "m.jsonl")
    assert fingerprint["kind"] == "directory"
    assert fingerprint["manifest_sha256"] == directory_manifest(root)["sha256"]
    assert fingerprint["manifest_sha256"] == sha256_file(tmp_path / "m.jsonl")
    assert fingerprint["n_files"] == len(TREE)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_publishes_exact_bytes_and_refuses_overwrite(tmp_path):
    target = tmp_path / "file.yaml"
    atomic_write_bytes(target, b"one\n")
    assert target.read_bytes() == b"one\n"
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"two\n")
    assert target.read_bytes() == b"one\n"
    atomic_write_bytes(target, b"two\n", overwrite=True)
    assert target.read_bytes() == b"two\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.yaml"]


def test_atomic_write_uses_fsync_and_replace(tmp_path, monkeypatch):
    calls = []
    real_replace, real_fsync = os.replace, os.fsync
    monkeypatch.setattr(
        records.os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))
    )
    monkeypatch.setattr(records.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd)))

    atomic_write_bytes(tmp_path / "x", b"data")

    assert calls[0] == "fsync"
    assert "replace" in calls
    assert calls.index("replace") > 0


def test_atomic_write_failure_leaves_no_target_and_no_temp(tmp_path, monkeypatch):
    def broken_fsync(fd):
        raise OSError("disk gone")

    monkeypatch.setattr(records.os, "fsync", broken_fsync)
    with pytest.raises(OSError, match="disk gone"):
        atomic_write_bytes(tmp_path / "x", b"data")
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_requires_existing_parent(tmp_path):
    with pytest.raises(FileNotFoundError):
        atomic_write_bytes(tmp_path / "missing" / "x", b"data")


# ---------------------------------------------------------------------------
# Stage records
# ---------------------------------------------------------------------------


def test_valid_records_of_each_kind(tmp_path):
    assert _running()["record_kind"] == "running"
    assert _completed(tmp_path)["record_kind"] == "completed"
    failed = _failed()
    assert failed["record_kind"] == "failed"
    assert "outputs" not in failed


def test_record_schema_rejects_missing_and_unexpected_keys(tmp_path):
    record = _completed(tmp_path)
    for key in ("environment", "resolved_plan_sha256", "outputs", "post_validation"):
        broken = copy.deepcopy(record)
        del broken[key]
        with pytest.raises(StageRecordError, match="missing"):
            validate_stage_record(broken)
    broken = copy.deepcopy(record)
    broken["status"] = "completed"
    with pytest.raises(StageRecordError, match="unexpected"):
        validate_stage_record(broken)


def test_running_and_failed_records_cannot_carry_outputs(tmp_path):
    running = _running()
    running["outputs"] = _completed(tmp_path)["outputs"]
    with pytest.raises(StageRecordError, match="unexpected"):
        validate_stage_record(running)
    failed = _failed()
    failed["outputs"] = running["outputs"]
    with pytest.raises(StageRecordError, match="unexpected"):
        validate_stage_record(failed)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(schema_version=True), "schema_version"),
        (lambda r: r.update(schema_version=2), "schema_version"),
        (lambda r: r.update(record_kind="done"), "record_kind"),
        (lambda r: r.update(stage_type="training_v2"), "stage_type"),
        (lambda r: r.update(side="both"), "side"),
        (lambda r: r.update(repository_revision="unknown"), "repository_revision"),
        (lambda r: r.update(resolved_plan_sha256="B" * 64), "resolved_plan_sha256"),
        (lambda r: r.update(stage_id="runs/../x"), "stage_id"),
        (lambda r: r.update(stage_id="runs//x"), "stage_id"),
        (lambda r: r.update(started_at="yesterday"), "started_at"),
        (lambda r: r["post_validation"].update(status="failed"), "post_validation"),
        (lambda r: r["execution"].update(exit_code=1), "exit_code 0"),
        (lambda r: r["execution"].update(argv="python train.py"), "argv"),
        (lambda r: r["execution"].update(shell=True), "execution keys"),
        (lambda r: r.update(outputs={}), "at least one output"),
        (lambda r: r["outputs"]["config"].update(sha256="x"), "sha256"),
        (lambda r: r.update(dependencies=[{"stage_id": "a"}]), "dependencies"),
    ],
)
def test_completed_record_field_validation(tmp_path, mutate, message):
    record = _completed(tmp_path)
    mutate(record)
    with pytest.raises(StageRecordError, match=message):
        validate_stage_record(record)


def test_running_subprocess_record_requires_null_exit_code():
    record = _running()
    record["execution"]["exit_code"] = 0
    with pytest.raises(StageRecordError, match="exit_code null"):
        validate_stage_record(record)


def test_failed_record_requires_failure_block():
    record = _failed()
    record["failure"]["partial_outputs_present"] = "yes"
    with pytest.raises(StageRecordError, match="partial_outputs_present"):
        validate_stage_record(record)


def test_in_process_execution_block(tmp_path):
    output = tmp_path / "report.yaml"
    output.write_bytes(b"ok")
    record = build_stage_record(
        record_kind="completed",
        execution={"kind": "in_process", "callable": "module:function"},
        completed_at=COMPLETED,
        outputs={"report": _file_fp(output)},
        post_validation={"status": "passed", "checks": []},
        **_common(
            stage_id="preflight/null_validation",
            stage_type="null_validation",
            run_id=None,
            side="benchmark",
        ),
    )
    assert record["execution"] == {"kind": "in_process", "callable": "module:function"}


def test_write_and_load_record_round_trip_at_canonical_path(tmp_path):
    execution_dir = tmp_path / "execution"
    record = _completed(tmp_path)

    path = write_stage_record(execution_dir, record)

    assert path == execution_dir / "stages" / "runs" / "L3_rope" / "real_training.completed.yaml"
    assert path == stage_record_path(execution_dir, record["stage_id"], "completed")
    assert load_stage_record(path) == record
    assert yaml.safe_load(path.read_bytes()) == record
    with pytest.raises(FileExistsError):
        write_stage_record(execution_dir, record)


def test_record_file_hash_is_deterministic(tmp_path):
    record = _completed(tmp_path)
    first = write_stage_record(tmp_path / "e1", record)
    second = write_stage_record(tmp_path / "e2", copy.deepcopy(record))
    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)


def test_completed_record_cannot_be_confused_with_running_or_failed(tmp_path):
    execution_dir = tmp_path / "execution"
    running_path = write_stage_record(execution_dir, _running())
    failed_path = write_stage_record(
        execution_dir, _failed(stage_id="runs/L3_rope/null_training", side="null")
    )

    with pytest.raises(StageRecordError, match="only a completed record"):
        require_completed_record(running_path)
    with pytest.raises(StageRecordError, match="only a completed record"):
        require_completed_record(failed_path)

    # A running record renamed to look completed is still rejected by its content.
    disguised = stage_record_path(execution_dir, "runs/L3_rope/real_training", "completed")
    shutil.copy(running_path, disguised)
    with pytest.raises(StageRecordError, match="only a completed record"):
        require_completed_record(disguised)


def test_completed_record_at_wrong_location_rejects(tmp_path):
    execution_dir = tmp_path / "execution"
    path = write_stage_record(execution_dir, _completed(tmp_path))
    moved = stage_record_path(execution_dir, "runs/L3_other/real_training", "completed")
    moved.parent.mkdir(parents=True)
    shutil.copy(path, moved)
    with pytest.raises(StageRecordError, match="location"):
        require_completed_record(moved)
    assert require_completed_record(path)["stage_id"] == "runs/L3_rope/real_training"


def test_empty_or_malformed_record_file_is_not_valid(tmp_path):
    path = stage_record_path(tmp_path, "runs/L3_rope/real_training", "completed")
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    with pytest.raises(StageRecordError):
        require_completed_record(path)
    path.write_text("record_kind: [", encoding="utf-8")
    with pytest.raises(StageRecordError):
        require_completed_record(path)


def test_existing_stage_records_and_dependency_reference(tmp_path):
    execution_dir = tmp_path / "execution"
    assert existing_stage_records(execution_dir, "runs/L3_rope/real_training") == {}
    path = write_stage_record(execution_dir, _completed(tmp_path))
    assert existing_stage_records(execution_dir, "runs/L3_rope/real_training") == {
        "completed": path
    }

    reference = dependency_reference(path)
    assert reference == {
        "stage_id": "runs/L3_rope/real_training",
        "record_path": str(path),
        "record_sha256": sha256_file(path),
    }
    running = write_stage_record(
        execution_dir, _running(stage_id="runs/L3_rope/real_explanation", stage_type="explanation")
    )
    with pytest.raises(StageRecordError):
        dependency_reference(running)
