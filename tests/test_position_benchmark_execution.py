"""Tests for Phase 12C3B2A execution foundation and full null preflight.

No benchmark stage is executed here: subprocess tests use tiny fake Python
scripts, git tests use a fake runner or a throwaway repository, and the null
preflight runs the real 12C3A ``validate_null_pair`` on the tiny strict null
fixture shared with the 12C3B1 paired-manifest tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import yaml

from scripts import (
    create_null_baseline,
    position_benchmark_execution,
    position_benchmark_manifest,
    run_position_benchmark,
)
from scripts.position_benchmark_execution import (
    ENVIRONMENT_PROBE,
    EnvironmentProbeError,
    ExecutionLock,
    ExecutionLockError,
    NullPreflightError,
    OutputLocationError,
    PlanAuthorityError,
    RepositoryGateError,
    StageFailure,
    StageInterrupted,
    StageStateError,
    benchmark_root_from_plan,
    bind_benchmark_plan,
    build_null_validation_report,
    execute_subprocess_stage,
    load_persisted_execution_plan,
    null_validation_path,
    probe_execution_environment,
    publish_completed_record,
    record_stage_failure,
    require_execution_locations_safe,
    require_output_location_safe,
    require_repository_gate,
    run_null_preflight,
    run_subprocess_stage,
    verify_plan_rebuild,
    write_null_validation_report,
)
from scripts.position_benchmark_manifest import (
    build_human_summary,
    build_resolved_plan,
    resolved_plan_file_sha256,
    write_resolved_plan,
)
from scripts.position_benchmark_records import (
    build_stage_record,
    existing_stage_records,
    file_fingerprint,
    load_stage_record,
    sha256_file,
    stage_record_path,
)
from src.data import null_lineage
from tests.test_position_benchmark_manifest import _base_manifest, _write_yaml
from tests.test_position_benchmark_paired_manifest import _build, _paired_manifest

REVISION = "a" * 40
OTHER_REVISION = "f" * 40
COMMON = {
    "repository_revision": REVISION,
    "resolved_plan_sha256": "b" * 64,
    "manifest_file_sha256": "c" * 64,
}
ENVIRONMENT = {"hostname": "test-host"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plan_fixture(tmp_path: Path, mutate_manifest=None, mode: str = "cv"):
    manifest_path, manifest = _paired_manifest(tmp_path, mode=mode)
    if mutate_manifest is not None:
        mutate_manifest(tmp_path, manifest)
        _write_yaml(manifest_path, manifest)
    plan = _build(manifest_path)
    plan_path = tmp_path / "plans" / "plan.yaml"
    write_resolved_plan(plan_path, plan)
    return manifest_path, manifest, plan, plan_path


def _with_covariates(tmp_path: Path, manifest: dict) -> None:
    (tmp_path / "sex.tsv").write_text("sample_id\tsex\ns0\tXY\n", encoding="utf-8")
    (tmp_path / "pcs.tsv").write_text("sample_id\tPC1\ns0\t0.1\n", encoding="utf-8")
    manifest["training"]["sex_map"] = "sex.tsv"
    manifest["training"]["pc_map"] = "pcs.tsv"
    manifest["training"]["num_pcs"] = 1


def _write_mutated_plan(tmp_path: Path, plan: dict, mutate, name: str = "mutated.yaml") -> Path:
    data = copy.deepcopy(plan)
    mutate(data)
    path = tmp_path / "plans" / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# v2 input_files
# ---------------------------------------------------------------------------


def test_v2_plan_records_input_file_raw_byte_hashes(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)
    files = plan["input_files"]
    data = tmp_path / "data"

    assert list(files) == [
        "preprocessed_data",
        "null_artifact",
        "null_lineage_sidecar",
        "split_plan",
        "sex_map",
        "pc_map",
    ]
    assert files["preprocessed_data"] == {
        "path": str(data / "cohort.pt"),
        "sha256": _sha((data / "cohort.pt").read_bytes()),
    }
    assert files["null_artifact"] == {
        "path": str(data / "cohort.null.pt"),
        "sha256": _sha((data / "cohort.null.pt").read_bytes()),
    }
    sidecar = null_lineage.sidecar_path_for(data / "cohort.null.pt")
    assert files["null_lineage_sidecar"] == {
        "path": str(sidecar),
        "sha256": _sha(sidecar.read_bytes()),
    }
    split = tmp_path / "splits" / "split_plan.yaml"
    assert files["split_plan"] == {"path": str(split), "sha256": _sha(split.read_bytes())}
    assert files["sex_map"] is None and files["pc_map"] is None
    # input_files overlaps null_binding by design; both describe the same bytes.
    assert files["preprocessed_data"]["sha256"] == plan["null_binding"]["source_artifact_sha256"]
    assert files["null_artifact"]["sha256"] == plan["null_binding"]["null_artifact_sha256"]


def test_v2_plan_records_optional_covariate_hashes(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path, _with_covariates)
    for key, name in (("sex_map", "sex.tsv"), ("pc_map", "pcs.tsv")):
        path = tmp_path / name
        assert plan["input_files"][key] == {"path": str(path), "sha256": _sha(path.read_bytes())}


def test_input_file_byte_change_changes_plan(tmp_path):
    manifest_path, _, plan, _ = _plan_fixture(tmp_path, _with_covariates)
    _append(tmp_path / "sex.tsv", "s1\tXX\n")
    rebuilt = _build(manifest_path)
    assert rebuilt["input_files"]["sex_map"]["sha256"] != plan["input_files"]["sex_map"]["sha256"]


def test_split_file_hash_is_distinct_from_membership_hash(tmp_path):
    manifest_path, _, plan, _ = _plan_fixture(tmp_path)
    _append(tmp_path / "splits" / "split_plan.yaml", "# formatting-only change\n")
    rebuilt = _build(manifest_path)
    # Scientific membership identity is unchanged; reviewed file bytes are not.
    assert rebuilt["split_plan"]["membership_sha256"] == plan["split_plan"]["membership_sha256"]
    assert (
        rebuilt["input_files"]["split_plan"]["sha256"]
        != plan["input_files"]["split_plan"]["sha256"]
    )


def test_v1_plan_has_no_input_files_and_unchanged_top_level_keys(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")
    assert list(plan) == [
        "schema_version",
        "manifest_path",
        "manifest_file_sha256",
        "repository_root",
        "repository_revision",
        "benchmark",
        "dataset",
        "split_plan",
        "runtime",
        "runs",
        "comparisons",
        "warnings",
        "null_execution",
    ]


def test_write_resolved_plan_bytes_unchanged_and_atomic(tmp_path, monkeypatch):
    manifest_path, _ = _base_manifest(tmp_path)
    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(
        "scripts.position_benchmark_records.os.replace",
        lambda a, b: (replaced.append((a, b)), real_replace(a, b)),
    )
    out = tmp_path / "plans" / "plan.yaml"

    write_resolved_plan(out, plan)

    assert out.read_text(encoding="utf-8") == yaml.safe_dump(plan, sort_keys=False)
    assert len(replaced) == 1 and Path(replaced[0][1]) == out
    assert sorted(path.name for path in out.parent.iterdir()) == ["plan.yaml"]
    with pytest.raises(
        position_benchmark_manifest.BenchmarkManifestError, match="--out-plan already exists"
    ):
        write_resolved_plan(out, plan)


def test_resolved_plan_file_sha256_hashes_exact_bytes(tmp_path):
    _, _, _, plan_path = _plan_fixture(tmp_path)
    original = plan_path.read_bytes()
    assert resolved_plan_file_sha256(plan_path) == _sha(original)
    reformatted = tmp_path / "plans" / "reformatted.yaml"
    reformatted.write_bytes(original + b"\n")
    # A semantically identical file with different bytes is a different plan.
    assert yaml.safe_load(reformatted.read_bytes()) == yaml.safe_load(original)
    assert resolved_plan_file_sha256(reformatted) != resolved_plan_file_sha256(plan_path)


def test_dry_run_never_binds_or_creates_execution_state(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan_path = tmp_path / "plans" / "plan.yaml"
    code = run_position_benchmark.main(
        [str(manifest_path), "--dry-run", "--out-plan", str(plan_path), "--python", sys.executable]
    )
    assert code == 0
    plan = yaml.safe_load(plan_path.read_bytes())
    root = benchmark_root_from_plan(plan)
    assert not root.exists()
    assert "input_files" in plan
    assert not (root / "execution").exists()


def test_public_cli_still_refuses_execution(tmp_path, capsys):
    manifest_path, _ = _paired_manifest(tmp_path)
    assert run_position_benchmark.main([str(manifest_path)]) == 2
    assert "execution is not implemented yet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_position_benchmark.main([str(manifest_path), "--execute-plan", "plan.yaml"])


# ---------------------------------------------------------------------------
# Persisted plan loading and rebuild authority
# ---------------------------------------------------------------------------


def test_load_persisted_plan_round_trips_and_hashes_exact_bytes(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    loaded, plan_bytes, plan_sha = load_persisted_execution_plan(plan_path)
    assert loaded == plan
    assert plan_bytes == plan_path.read_bytes()
    assert plan_sha == _sha(plan_bytes) == resolved_plan_file_sha256(plan_path)


def test_verify_plan_rebuild_accepts_unchanged_plan(tmp_path):
    manifest_path, _, plan, plan_path = _plan_fixture(tmp_path)
    verified = verify_plan_rebuild(plan_path, manifest_path=manifest_path)
    assert verified.plan == plan
    assert verified.resolved_plan_sha256 == _sha(plan_path.read_bytes())


def test_rebuild_ignores_only_new_output_warnings(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    training = Path(plan["runs"][0]["directories"]["training"])
    training.mkdir(parents=True)
    (training / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    verify_plan_rebuild(plan_path)


def _rewrite_manifest(manifest_path: Path, manifest: dict, mutate) -> None:
    data = copy.deepcopy(manifest)
    mutate(data)
    _write_yaml(manifest_path, data)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda tmp, mp, m: _rewrite_manifest(
                mp, m, lambda d: d["calibration"].update(n_bootstrap=999)
            ),
            id="manifest-content",
        ),
        pytest.param(lambda tmp, mp, m: _append(mp, "# comment only\n"), id="manifest-bytes"),
        pytest.param(
            lambda tmp, mp, m: _append(tmp / "splits" / "split_plan.yaml", "# x\n"),
            id="split-plan-bytes",
        ),
        pytest.param(
            lambda tmp, mp, m: _append(
                null_lineage.sidecar_path_for(tmp / "data" / "cohort.null.pt"), "# x\n"
            ),
            id="sidecar-bytes",
        ),
        pytest.param(
            lambda tmp, mp, m: _append(tmp / "data" / "cohort.pt", "x"), id="real-dataset"
        ),
        pytest.param(
            lambda tmp, mp, m: _append(tmp / "data" / "cohort.null.pt", "x"), id="null-artifact"
        ),
    ],
)
def test_rebuild_rejects_input_changes(tmp_path, change):
    manifest_path, manifest, _, plan_path = _plan_fixture(tmp_path)
    change(tmp_path, manifest_path, manifest)
    with pytest.raises(PlanAuthorityError):
        verify_plan_rebuild(plan_path)


@pytest.mark.parametrize("name", ["sex.tsv", "pcs.tsv"])
def test_rebuild_rejects_covariate_changes(tmp_path, name):
    _, _, _, plan_path = _plan_fixture(tmp_path, _with_covariates)
    _append(tmp_path / name, "s1\t0\n")
    with pytest.raises(PlanAuthorityError, match="input_files"):
        verify_plan_rebuild(plan_path)


def test_rebuild_rejects_repository_revision_change(tmp_path, monkeypatch):
    _, _, _, plan_path = _plan_fixture(tmp_path)
    monkeypatch.setattr(
        position_benchmark_manifest, "_read_repository_revision", lambda root: OTHER_REVISION
    )
    with pytest.raises(PlanAuthorityError, match="repository_revision"):
        verify_plan_rebuild(plan_path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["runs"][0]["train_argv"].append("--extra"), "train_argv"),
        (lambda p: p["runs"][0]["directories"].update(training="/elsewhere/training"), "training"),
        (lambda p: p["calibration"].update(n_bootstrap=5), "n_bootstrap"),
    ],
)
def test_rebuild_rejects_edited_persisted_plan(tmp_path, mutate, message):
    _, _, plan, _ = _plan_fixture(tmp_path)
    edited = _write_mutated_plan(tmp_path, plan, mutate)
    with pytest.raises(PlanAuthorityError, match=message):
        verify_plan_rebuild(edited)


def test_rebuild_rejects_changed_runtime_python(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)
    other = tmp_path / "otherpython"
    other.write_text("", encoding="utf-8")

    def mutate(data):
        data["runtime"]["python"] = str(other)
        for run in data["runs"]:
            for key in position_benchmark_execution.RUN_ARGV_KEYS:
                run[key][0] = str(other)
            run["calibration"]["argv"][0] = str(other)
        for comparison in data["comparisons"].values():
            comparison["argv"][0] = str(other)

    edited = _write_mutated_plan(tmp_path, plan, mutate)
    load_persisted_execution_plan(edited)
    verify_plan_rebuild(edited)  # self-consistent: rebuild uses the plan's own python
    mixed = _write_mutated_plan(
        tmp_path, plan, lambda d: d["runtime"].update(python=str(other)), name="mixed.yaml"
    )
    with pytest.raises(PlanAuthorityError, match="runtime.python"):
        verify_plan_rebuild(mixed)


def test_rebuild_rejects_supplied_manifest_mismatch(tmp_path):
    _, _, _, plan_path = _plan_fixture(tmp_path)
    other = tmp_path / "other.yaml"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(PlanAuthorityError, match="not the plan's manifest"):
        verify_plan_rebuild(plan_path, manifest_path=other)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(warnings=["planned output directory is non-empty: x"]), "warnings"),
        (lambda p: p.update(repository_revision="unknown"), "repository_revision"),
        (lambda p: p.update(repository_revision="abc"), "repository_revision"),
        (lambda p: p["null_execution"].update(execution_authorized=True), "execution_authorized"),
        (lambda p: p["null_binding"].update(execution_authorized=True), "execution_authorized"),
        (lambda p: p["null_binding"].pop("lineage_sha256"), "lineage_sha256"),
        (lambda p: p["null_binding"].update(n_samples=True), "n_samples"),
        (lambda p: p.pop("input_files"), "input_files"),
        (lambda p: p["input_files"].pop("pc_map"), "input_files"),
        (lambda p: p["input_files"]["split_plan"].update(sha256="X"), "split_plan.sha256"),
        (
            lambda p: p["input_files"]["preprocessed_data"].update(sha256="0" * 64),
            "preprocessed_data.sha256",
        ),
        (lambda p: p["input_files"]["null_artifact"].update(path="/x.pt"), "null_artifact.path"),
        (lambda p: p["runtime"].update(python="python"), "runtime.python"),
        (lambda p: p["runtime"].update(device="tpu"), "runtime.device"),
        (lambda p: p["runs"][0].update(explain_argv="python explain.py"), "explain_argv"),
        (lambda p: p["runs"][0]["null_train_argv"].append(3), "null_train_argv"),
        (lambda p: p["comparisons"]["performance"].update(argv=[]), "performance.argv"),
        (lambda p: p["runs"][0]["calibration"]["argv"].__setitem__(0, "/x/python"), "runtime"),
    ],
)
def test_persisted_plan_structural_validation(tmp_path, mutate, message):
    _, _, plan, _ = _plan_fixture(tmp_path)
    edited = _write_mutated_plan(tmp_path, plan, mutate)
    with pytest.raises(PlanAuthorityError, match=message):
        load_persisted_execution_plan(edited)


def test_schema_v1_plan_is_never_executable(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")
    plan_path = tmp_path / "v1.yaml"
    write_resolved_plan(plan_path, plan)
    with pytest.raises(PlanAuthorityError, match="schema_version 2"):
        load_persisted_execution_plan(plan_path)


def test_non_mapping_and_malformed_plans_reject(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n", encoding="utf-8")
    with pytest.raises(PlanAuthorityError, match="mapping"):
        load_persisted_execution_plan(bad)
    bad.write_text("a: [", encoding="utf-8")
    with pytest.raises(PlanAuthorityError, match="YAML"):
        load_persisted_execution_plan(bad)
    with pytest.raises(PlanAuthorityError, match="cannot read"):
        load_persisted_execution_plan(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Permanent benchmark plan binding
# ---------------------------------------------------------------------------


def test_bind_benchmark_plan_copies_exact_bytes_once(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    verified = verify_plan_rebuild(plan_path)
    root = benchmark_root_from_plan(plan)

    first = bind_benchmark_plan(root, verified)

    bound = root / "execution" / "resolved_plan.yaml"
    assert first["status"] == "bound"
    assert bound.read_bytes() == plan_path.read_bytes()
    binding = yaml.safe_load((root / "execution" / "plan_binding.yaml").read_bytes())
    assert binding["resolved_plan_sha256"] == verified.resolved_plan_sha256
    assert binding["repository_revision"] == plan["repository_revision"]
    stat_before = bound.stat()

    second = bind_benchmark_plan(root, verified)

    assert second["status"] == "verified"
    assert bound.stat().st_ino == stat_before.st_ino
    assert bound.stat().st_mtime_ns == stat_before.st_mtime_ns


def test_bind_rejects_different_plan_bytes(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    root = benchmark_root_from_plan(plan)
    bind_benchmark_plan(root, verify_plan_rebuild(plan_path))
    other = position_benchmark_execution.VerifiedPlan(
        path=plan_path,
        plan=plan,
        plan_bytes=plan_path.read_bytes() + b"\n",
        resolved_plan_sha256=_sha(plan_path.read_bytes() + b"\n"),
    )
    with pytest.raises(PlanAuthorityError, match="bound to a different plan"):
        bind_benchmark_plan(root, other)


def test_bind_completes_missing_binding_record_and_rejects_tampered_one(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    verified = verify_plan_rebuild(plan_path)
    root = benchmark_root_from_plan(plan)
    bind_benchmark_plan(root, verified)
    binding_path = root / "execution" / "plan_binding.yaml"
    binding_path.unlink()
    assert bind_benchmark_plan(root, verified)["status"] == "verified"
    assert binding_path.is_file()
    data = yaml.safe_load(binding_path.read_bytes())
    data["resolved_plan_sha256"] = "0" * 64
    binding_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(PlanAuthorityError, match="disagrees"):
        bind_benchmark_plan(root, verified)


def test_benchmark_root_derivation_rejects_inconsistent_layout(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)
    assert benchmark_root_from_plan(plan) == tmp_path / "outputs" / "posenc_l3_paired" / "L3"
    broken = copy.deepcopy(plan)
    broken["comparisons"]["performance"]["directory"] = "/other/comparisons/performance"
    with pytest.raises(PlanAuthorityError, match="disagree"):
        benchmark_root_from_plan(broken)


# ---------------------------------------------------------------------------
# Repository gate (fake git runner)
# ---------------------------------------------------------------------------


class FakeGit:
    """Record git invocations and answer from a table keyed by the git subcommand args."""

    def __init__(self, answers: dict[tuple[str, ...], tuple[int, str, str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        assert isinstance(argv, list) and argv[0] == "git"
        assert kwargs["shell"] is False
        code, out, err = self.answers[tuple(argv[1:])]
        return subprocess.CompletedProcess(argv, code, out, err)


def _gate_plan(tmp_path: Path) -> dict:
    return {"repository_root": str(tmp_path), "repository_revision": REVISION}


def _git_answers(tmp_path: Path, **overrides) -> dict:
    answers = {
        ("rev-parse", "--show-toplevel"): (0, f"{tmp_path}\n", ""),
        ("rev-parse", "HEAD"): (0, f"{REVISION}\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=all"): (0, "", ""),
    }
    answers.update(overrides)
    return answers


def test_repository_gate_passes_clean_matching_checkout(tmp_path):
    runner = FakeGit(_git_answers(tmp_path))
    result = require_repository_gate(_gate_plan(tmp_path), runner=runner)
    assert result == {"repository_root": str(tmp_path.resolve()), "repository_revision": REVISION}
    assert [call[0] for call in runner.calls] == [
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    ]
    assert all(call[1]["cwd"] == str(tmp_path) for call in runner.calls)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({("rev-parse", "--show-toplevel"): (0, "/somewhere/else\n", "")}, "repository root"),
        ({("rev-parse", "HEAD"): (0, f"{OTHER_REVISION}\n", "")}, "does not match plan"),
        (
            {
                ("status", "--porcelain=v1", "--untracked-files=all"): (
                    0,
                    " M scripts/train.py\n",
                    "",
                )
            },
            "not clean",
        ),
        (
            {
                ("status", "--porcelain=v1", "--untracked-files=all"): (
                    0,
                    "M  scripts/train.py\n",
                    "",
                )
            },
            "not clean",
        ),
        (
            {("status", "--porcelain=v1", "--untracked-files=all"): (0, "?? new_file.py\n", "")},
            "not clean",
        ),
        ({("rev-parse", "HEAD"): (128, "", "fatal: bad")}, "failed"),
    ],
)
def test_repository_gate_rejects(tmp_path, overrides, message):
    runner = FakeGit(_git_answers(tmp_path, **{}) | overrides)
    with pytest.raises(RepositoryGateError, match=message):
        require_repository_gate(_gate_plan(tmp_path), runner=runner)


@pytest.mark.parametrize("revision", ["unknown", "", None, "A" * 40, "a" * 39])
def test_repository_gate_rejects_invalid_revision_without_running_git(tmp_path, revision):
    runner = FakeGit({})
    plan = {"repository_root": str(tmp_path), "repository_revision": revision}
    with pytest.raises(RepositoryGateError, match="full Git SHA"):
        require_repository_gate(plan, runner=runner)
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Output location safety (throwaway git repository)
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    return repo


def test_output_location_outside_repository_passes(tmp_path, scratch_repo):
    assert (
        require_output_location_safe(tmp_path / "bench", repository_root=scratch_repo)
        == "outside_repository"
    )


def test_output_location_ignored_inside_repository_passes(scratch_repo):
    target = scratch_repo / "documentation" / "outputs" / "posenc" / "L3"
    assert require_output_location_safe(target, repository_root=scratch_repo) == "git_ignored"


def test_output_location_not_ignored_inside_repository_rejects(scratch_repo):
    with pytest.raises(OutputLocationError, match="not ignored"):
        require_output_location_safe(
            scratch_repo / "results_here" / "L3", repository_root=scratch_repo
        )
    with pytest.raises(OutputLocationError, match="repository root"):
        require_output_location_safe(scratch_repo, repository_root=scratch_repo)


def test_output_location_git_error_rejects(tmp_path):
    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 128, "", "fatal: not a git repository")

    target = tmp_path / "inside" / "x"
    with pytest.raises(OutputLocationError, match="check-ignore failed"):
        require_output_location_safe(target, repository_root=tmp_path, runner=failing)


def test_execution_locations_check_benchmark_root_and_plan(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    plan = copy.deepcopy(plan)
    plan["repository_root"] = str(tmp_path / "unrelated_repo")
    assert require_execution_locations_safe(plan, plan_path) == {
        "benchmark_root": "outside_repository",
        "resolved_plan": "outside_repository",
    }


# ---------------------------------------------------------------------------
# Execution lock
# ---------------------------------------------------------------------------


def test_execution_lock_contention_in_process(tmp_path):
    first = ExecutionLock(tmp_path, resolved_plan_sha256="b" * 64).acquire()
    try:
        with pytest.raises(ExecutionLockError, match="another executor"):
            ExecutionLock(tmp_path, resolved_plan_sha256="b" * 64).acquire()
        holder = json.loads((tmp_path / "execution" / "execution.lock").read_text())
        assert holder["pid"] == os.getpid()
        assert holder["resolved_plan_sha256"] == "b" * 64
        assert {"hostname", "started_at"} <= set(holder)
    finally:
        first.release()
    with ExecutionLock(tmp_path, resolved_plan_sha256="b" * 64) as again:
        assert again.held
    assert not again.held
    assert (tmp_path / "execution" / "execution.lock").exists()


def test_execution_lock_contention_across_processes_and_release_on_death(tmp_path):
    lock_path = tmp_path / "execution" / "execution.lock"
    lock_path.parent.mkdir(parents=True)
    holder_script = textwrap.dedent(f"""
        import fcntl, os, sys, time
        fd = os.open({str(lock_path)!r}, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        print("locked", flush=True)
        time.sleep(60)
        """)
    child = subprocess.Popen(
        [sys.executable, "-c", holder_script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(ExecutionLockError):
            ExecutionLock(tmp_path, resolved_plan_sha256="b" * 64).acquire()
    finally:
        child.kill()
        child.wait()
    # The kernel released the dead holder's lock; no stale-lock cleanup needed.
    ExecutionLock(tmp_path, resolved_plan_sha256="b" * 64).acquire().release()


# ---------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------


def _probe_runner(payload: dict | str, returncode: int = 0, calls: list | None = None):
    def runner(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        stdout = payload if isinstance(payload, str) else "noise\n" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(argv, returncode, stdout, "probe stderr")

    return runner


PROBE_PAYLOAD = {
    "executable": "/env/bin/python",
    "python_version": "3.10.20",
    "platform": "Linux-6.6",
    "hostname": "gpu-node",
    "cuda_visible_devices": "0",
    "torch_version": "2.13.0",
    "torch_cuda_version": "13.0",
    "cuda_available": True,
    "cuda_device_names": ["NVIDIA Test GPU"],
}


def test_environment_probe_parses_output_and_uses_argv(tmp_path):
    calls: list = []
    plan = {"runtime": {"python": "/env/bin/python", "device": "cuda"}}
    environment = probe_execution_environment(
        plan, runner=_probe_runner(PROBE_PAYLOAD, calls=calls)
    )
    assert environment == {
        "plan_python": "/env/bin/python",
        "probe_executable": "/env/bin/python",
        "python_version": "3.10.20",
        "torch_version": "2.13.0",
        "torch_cuda_version": "13.0",
        "cuda_available": True,
        "cuda_device_names": ["NVIDIA Test GPU"],
        "platform": "Linux-6.6",
        "hostname": "gpu-node",
        "cuda_visible_devices": "0",
    }
    argv, kwargs = calls[0]
    assert argv == ["/env/bin/python", "-c", ENVIRONMENT_PROBE]
    assert kwargs["shell"] is False


def test_environment_probe_rejects_cuda_unavailable():
    payload = dict(PROBE_PAYLOAD, cuda_available=False, cuda_device_names=[])
    plan = {"runtime": {"python": "/env/bin/python", "device": "cuda"}}
    with pytest.raises(EnvironmentProbeError, match="CUDA is unavailable"):
        probe_execution_environment(plan, runner=_probe_runner(payload))
    cpu_plan = {"runtime": {"python": "/env/bin/python", "device": "cpu"}}
    assert (
        probe_execution_environment(cpu_plan, runner=_probe_runner(payload))["cuda_available"]
        is False
    )


@pytest.mark.parametrize(
    "payload, returncode, message",
    [
        ({"torch_import_error": "ModuleNotFoundError('torch')"}, 0, "cannot import torch"),
        ("not json\n", 0, "JSON"),
        (PROBE_PAYLOAD, 1, "failed"),
        ({"executable": "/x"}, 0, "missing"),
    ],
)
def test_environment_probe_failures(payload, returncode, message):
    plan = {"runtime": {"python": "/env/bin/python", "device": "cpu"}}
    with pytest.raises(EnvironmentProbeError, match=message):
        probe_execution_environment(plan, runner=_probe_runner(payload, returncode))


def test_environment_probe_real_interpreter_cpu():
    environment = probe_execution_environment(
        {"runtime": {"python": sys.executable, "device": "cpu"}}
    )
    assert environment["plan_python"] == sys.executable
    assert environment["python_version"].startswith(f"{sys.version_info.major}.")
    assert isinstance(environment["cuda_available"], bool)


# ---------------------------------------------------------------------------
# Full null preflight and null_validation.yaml
# ---------------------------------------------------------------------------


def test_null_preflight_passes_with_real_validator(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)
    result = run_null_preflight(plan)
    report = result["validator_report"]
    for key in position_benchmark_execution.NULL_BINDING_IDENTITY_FIELDS:
        assert report[key] == plan["null_binding"][key]
    assert result["validator"] == "src.data.null_lineage.validate_null_pair"
    assert result["input_files"]["sex_map"] is None


@pytest.mark.parametrize("field", position_benchmark_execution.NULL_BINDING_IDENTITY_FIELDS)
def test_null_preflight_rejects_each_binding_field_mismatch(tmp_path, field):
    _, _, plan, _ = _plan_fixture(tmp_path)
    plan = copy.deepcopy(plan)
    plan["null_binding"][field] = (
        plan["null_binding"][field] + 1 if field == "n_samples" else "0" * 64
    )
    with pytest.raises(NullPreflightError, match=field):
        run_null_preflight(plan)


@pytest.mark.parametrize("field", position_benchmark_execution.NULL_BINDING_IDENTITY_FIELDS)
def test_null_preflight_rejects_validator_report_disagreement(tmp_path, field):
    _, _, plan, _ = _plan_fixture(tmp_path)

    def validator(real, null, sidecar):
        report = dict(null_lineage.validate_null_pair(real, null, sidecar))
        report[field] = report[field] + 1 if field == "n_samples" else "1" * 64
        return report

    with pytest.raises(NullPreflightError, match=field):
        run_null_preflight(plan, validator=validator)


def test_null_preflight_wraps_validator_failure(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)

    def validator(real, null, sidecar):
        raise ValueError("labels do not match permutation")

    with pytest.raises(NullPreflightError, match="labels do not match permutation"):
        run_null_preflight(plan, validator=validator)


@pytest.mark.parametrize(
    "target",
    [
        lambda tmp: null_lineage.sidecar_path_for(tmp / "data" / "cohort.null.pt"),
        lambda tmp: tmp / "splits" / "split_plan.yaml",
    ],
    ids=["sidecar", "split_plan"],
)
def test_null_preflight_rejects_raw_file_hash_change(tmp_path, target):
    _, _, plan, _ = _plan_fixture(tmp_path)
    _append(target(tmp_path), "# byte change\n")
    with pytest.raises(NullPreflightError, match="bytes changed since planning"):
        run_null_preflight(plan)


def test_null_preflight_rejects_covariate_hash_change(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path, _with_covariates)
    _append(tmp_path / "pcs.tsv", "s1\t0.2\n")
    with pytest.raises(NullPreflightError, match="input_files.pc_map"):
        run_null_preflight(plan)


def test_null_preflight_never_regenerates_null(tmp_path, monkeypatch):
    _, _, plan, _ = _plan_fixture(tmp_path)
    null_path = tmp_path / "data" / "cohort.null.pt"
    sidecar = null_lineage.sidecar_path_for(null_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (null_path, sidecar)}

    def forbidden(*args, **kwargs):
        raise AssertionError("null regeneration must never be called")

    monkeypatch.setattr(create_null_baseline, "create_strict_single_permutation", forbidden)
    monkeypatch.setattr(null_lineage, "build_null_samples", forbidden)
    monkeypatch.setattr(null_lineage, "write_sidecar", forbidden)

    run_null_preflight(plan)

    for path, (data, mtime) in before.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == mtime


def test_null_validation_report_contents_and_atomic_write(tmp_path):
    _, _, plan, plan_path = _plan_fixture(tmp_path)
    plan_sha = _sha(plan_path.read_bytes())
    preflight = run_null_preflight(plan)
    report = build_null_validation_report(
        plan,
        resolved_plan_sha256=plan_sha,
        repository_revision=plan["repository_revision"],
        preflight=preflight,
        validated_at="2026-09-23T12:00:00+00:00",
    )
    files = plan["input_files"]
    assert report == {
        "schema_version": 1,
        "status": "passed",
        "validated_at": "2026-09-23T12:00:00+00:00",
        "repository_revision": plan["repository_revision"],
        "resolved_plan_sha256": plan_sha,
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "source": files["preprocessed_data"],
        "null": files["null_artifact"],
        "lineage_sidecar": files["null_lineage_sidecar"],
        "lineage_sha256": plan["null_binding"]["lineage_sha256"],
        "sample_ids_sha256": plan["null_binding"]["sample_ids_sha256"],
        "n_samples": plan["null_binding"]["n_samples"],
        "split_plan": {
            "path": files["split_plan"]["path"],
            "file_sha256": files["split_plan"]["sha256"],
            "membership_sha256": plan["split_plan"]["membership_sha256"],
            "sample_ids_sha256": plan["split_plan"]["sample_ids_sha256"],
        },
        "validator": "src.data.null_lineage.validate_null_pair",
        "validator_report": preflight["validator_report"],
        "plan_binding_match": True,
    }
    root = benchmark_root_from_plan(plan)
    path = write_null_validation_report(root, report)
    assert path == null_validation_path(root) == root / "null_binding" / "null_validation.yaml"
    assert yaml.safe_load(path.read_bytes()) == report
    assert sorted(p.name for p in path.parent.iterdir()) == ["null_validation.yaml"]
    with pytest.raises(FileExistsError):
        write_null_validation_report(root, report)
    # Same inputs and timestamp produce the same bytes (deterministic report).
    again = build_null_validation_report(
        plan,
        resolved_plan_sha256=plan_sha,
        repository_revision=plan["repository_revision"],
        preflight=run_null_preflight(plan),
        validated_at="2026-09-23T12:00:00+00:00",
    )
    assert yaml.safe_dump(again, sort_keys=False) == path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Generic subprocess stage primitive
# ---------------------------------------------------------------------------


FAKE_STAGE = textwrap.dedent("""
    import json, os, pathlib, sys
    out_dir = pathlib.Path(sys.argv[1])
    exit_code = int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "partial.txt").write_text("partial output", encoding="utf-8")
    print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))
    print("diagnostic on stderr", file=sys.stderr)
    sys.exit(exit_code)
    """)


@pytest.fixture()
def fake_stage(tmp_path):
    script = tmp_path / "fake_stage.py"
    script.write_text(FAKE_STAGE, encoding="utf-8")
    return script


WEIRD_ARGS = ["--name", "a b", "$HOME", "*.pt", "x;rm -rf /", "'quoted'", "--flag=1"]


def test_subprocess_stage_passes_argv_exactly_and_persists_logs(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    argv = [sys.executable, str(fake_stage), str(out_dir), "0", *WEIRD_ARGS]

    result = run_subprocess_stage(argv, cwd=cwd, log_dir=tmp_path / "logs", log_stem="stage")

    payload = json.loads((tmp_path / "logs" / "stage.stdout.log").read_text().splitlines()[0])
    assert payload["argv"] == argv[2:]
    assert payload["cwd"] == str(cwd)
    assert result.exit_code == 0
    assert result.argv == argv and result.argv is not argv
    assert "diagnostic on stderr" in (tmp_path / "logs" / "stage.stderr.log").read_text()
    for stream in ("stdout", "stderr"):
        log = tmp_path / "logs" / f"stage.{stream}.log"
        assert getattr(result, f"{stream}_log") == file_fingerprint(log)
    block = result.execution_block()
    assert block["argv"] == argv and block["exit_code"] == 0
    assert block["logs"]["stdout"]["sha256"] == sha256_file(tmp_path / "logs" / "stage.stdout.log")


def test_subprocess_stage_uses_shell_false_and_unmodified_list(tmp_path):
    seen = {}

    class SpyPopen:
        def __init__(self, args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

        def wait(self, timeout=None):
            return 0

    argv = ["/env/bin/python", "/repo/scripts/train.py", "--seed", "42"]
    run_subprocess_stage(
        argv, cwd=tmp_path, log_dir=tmp_path / "logs", log_stem="s", popen=SpyPopen
    )
    assert seen["args"] == argv
    assert isinstance(seen["args"], list)
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert seen["kwargs"]["start_new_session"] is True


def test_subprocess_stage_nonzero_exit_raises_with_logs(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    argv = [sys.executable, str(fake_stage), str(out_dir), "3"]
    with pytest.raises(StageFailure, match="code 3") as info:
        run_subprocess_stage(argv, cwd=tmp_path, log_dir=tmp_path / "logs", log_stem="stage")
    assert info.value.result.exit_code == 3
    assert "diagnostic on stderr" in (tmp_path / "logs" / "stage.stderr.log").read_text()
    assert (out_dir / "partial.txt").read_text() == "partial output"


def test_subprocess_stage_refuses_existing_logs_and_bad_inputs(tmp_path, fake_stage):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stage.stdout.log").write_text("old", encoding="utf-8")
    argv = [sys.executable, str(fake_stage), str(tmp_path / "out"), "0"]
    with pytest.raises(StageStateError, match="existing stage log"):
        run_subprocess_stage(argv, cwd=tmp_path, log_dir=tmp_path / "logs", log_stem="stage")
    assert (tmp_path / "logs" / "stage.stdout.log").read_text() == "old"
    with pytest.raises(StageStateError, match="absolute"):
        run_subprocess_stage(argv, cwd="relative", log_dir=tmp_path / "l2", log_stem="s")
    with pytest.raises(StageStateError, match="argv"):
        run_subprocess_stage("python x.py", cwd=tmp_path, log_dir=tmp_path / "l3", log_stem="s")


class FakeProcessGroup:
    """Fake POSIX process group for a stage leader plus possible workers.

    ``group_dies_on`` is the signal that empties the whole group (``None``
    means the group is already gone); ``leader_dies_on`` is the signal that
    ends the leader (defaults to ``group_dies_on``), so workers can outlive
    the leader. ``killpg`` records every non-probe signal.
    """

    def __init__(self, *, pid=987654, group_dies_on=signal.SIGINT, leader_dies_on="same"):
        self.pid = pid
        self.group_dies_on = group_dies_on
        self.leader_dies_on = group_dies_on if leader_dies_on == "same" else leader_dies_on
        self.group_alive = group_dies_on is not None
        self.leader_alive = self.group_alive
        self.signals: list = []
        self.direct_signals: list = []

    def killpg(self, pgid, sig):
        assert pgid == self.pid, "only the stage group may be signalled"
        assert pgid != os.getpgrp(), "the executor's own group must never be signalled"
        if not self.group_alive:
            raise ProcessLookupError
        if sig == 0:
            return
        self.signals.append(sig)
        if sig == self.leader_dies_on:
            self.leader_alive = False
        if sig == self.group_dies_on:
            self.group_alive = False
            self.leader_alive = False

    def popen(self, interrupt_first: bool = True):
        group = self

        class FakeProcess:
            def __init__(self, args, **kwargs):
                self.pid = group.pid
                self.kwargs = kwargs
                self.waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if interrupt_first and self.waits == 1:
                    raise KeyboardInterrupt
                if group.leader_alive:
                    raise subprocess.TimeoutExpired("fake", timeout)
                return -2

            def poll(self):
                return None if group.leader_alive else -2

            def send_signal(self, sig):
                group.direct_signals.append(sig)
                if sig == group.leader_dies_on:
                    group.leader_alive = False

        return FakeProcess


@pytest.fixture()
def fast_grace(monkeypatch):
    for name in ("INTERRUPT_GRACE_SECONDS", "TERMINATE_GRACE_SECONDS", "KILL_GRACE_SECONDS"):
        monkeypatch.setattr(position_benchmark_execution, name, 0.05)
    monkeypatch.setattr(position_benchmark_execution, "GROUP_POLL_SECONDS", 0.01)


def _interrupted_run(tmp_path, group, monkeypatch):
    monkeypatch.setattr(position_benchmark_execution.os, "killpg", group.killpg)
    with pytest.raises(StageInterrupted) as info:
        run_subprocess_stage(
            ["/env/bin/python", "x.py"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            log_stem="s",
            popen=group.popen(),
        )
    return info.value


def test_subprocess_stage_keyboard_interrupt(tmp_path, monkeypatch, fast_grace):
    group = FakeProcessGroup()
    interrupt = _interrupted_run(tmp_path, group, monkeypatch)
    assert isinstance(interrupt, KeyboardInterrupt)
    assert interrupt.result.interrupted
    assert interrupt.result.exit_code == -2
    assert group.signals == [signal.SIGINT]
    assert group.direct_signals == []


@pytest.mark.parametrize(
    "group_dies_on, leader_dies_on, expected",
    [
        (signal.SIGINT, "same", [signal.SIGINT]),
        (signal.SIGTERM, "same", [signal.SIGINT, signal.SIGTERM]),
        (signal.SIGKILL, "same", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]),
        # Workers outlive the leader: escalation continues until the group is empty.
        (signal.SIGTERM, signal.SIGINT, [signal.SIGINT, signal.SIGTERM]),
        (signal.SIGKILL, signal.SIGINT, [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]),
    ],
)
def test_interrupt_escalates_on_process_group(
    tmp_path, monkeypatch, fast_grace, group_dies_on, leader_dies_on, expected
):
    group = FakeProcessGroup(group_dies_on=group_dies_on, leader_dies_on=leader_dies_on)
    interrupt = _interrupted_run(tmp_path, group, monkeypatch)
    assert group.signals == expected
    assert interrupt.result.exit_code == -2


def test_interrupt_with_group_already_gone_sends_nothing(tmp_path, monkeypatch, fast_grace):
    group = FakeProcessGroup(group_dies_on=None)
    interrupt = _interrupted_run(tmp_path, group, monkeypatch)
    assert group.signals == [] and group.direct_signals == []
    assert interrupt.result.exit_code == -2


def test_signal_group_tolerates_exited_group(monkeypatch):
    def vanished(pgid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(position_benchmark_execution.os, "killpg", vanished)

    class Exited:
        pid = 987654

    position_benchmark_execution._signal_group(Exited(), signal.SIGTERM)


def test_interrupt_never_signals_executor_process_group(tmp_path, monkeypatch, fast_grace):
    group = FakeProcessGroup(pid=os.getpgrp())

    def forbidden(pgid, sig):
        raise AssertionError(f"killpg({pgid}, {sig}) must not target the executor group")

    monkeypatch.setattr(position_benchmark_execution.os, "killpg", forbidden)
    with pytest.raises(StageInterrupted):
        run_subprocess_stage(
            ["/env/bin/python", "x.py"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            log_stem="s",
            popen=group.popen(),
        )
    assert group.direct_signals == [signal.SIGINT]


def _stage_kwargs(tmp_path: Path, argv: list[str], owned: list[Path]) -> dict:
    return {
        "execution_dir": tmp_path / "execution",
        "stage_id": "runs/L3_rope/real_training",
        "stage_type": "training",
        "run_id": "L3_rope",
        "side": "real",
        "argv": argv,
        "cwd": tmp_path,
        "owned_paths": owned,
        "common": COMMON,
        "inputs": {},
        "dependencies": [],
        "environment": ENVIRONMENT,
    }


def test_execute_stage_success_leaves_only_running_until_completed_published(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )

    result, running_path = execute_subprocess_stage(**kwargs)

    execution_dir = kwargs["execution_dir"]
    assert set(existing_stage_records(execution_dir, kwargs["stage_id"])) == {"running"}
    running = load_stage_record(running_path)
    assert running["execution"]["argv"] == kwargs["argv"]
    assert running["execution"]["exit_code"] is None

    completed = build_stage_record(
        record_kind="completed",
        stage_id=kwargs["stage_id"],
        stage_type="training",
        run_id="L3_rope",
        side="real",
        execution=result.execution_block(),
        inputs={},
        dependencies=[],
        environment=ENVIRONMENT,
        started_at=running["started_at"],
        completed_at=result.completed_at,
        outputs={"partial": file_fingerprint(out_dir / "partial.txt")},
        post_validation={"status": "passed", "checks": ["files_exist"]},
        **COMMON,
    )
    path = publish_completed_record(execution_dir, running_path, completed)
    assert set(existing_stage_records(execution_dir, kwargs["stage_id"])) == {"completed"}
    assert load_stage_record(path)["execution"]["logs"]["stderr"]["sha256"] == sha256_file(
        execution_dir / "logs" / "runs" / "L3_rope" / "real_training.stderr.log"
    )


def test_execute_stage_failure_writes_failed_record_and_keeps_partial_output(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "4"], [out_dir]
    )

    with pytest.raises(StageFailure):
        execute_subprocess_stage(**kwargs)

    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"failed"}
    failed = load_stage_record(records["failed"])
    assert failed["failure"] == {
        "reason": "subprocess_failed",
        "exit_code": 4,
        "exception": "stage exited with code 4",
        "partial_outputs_present": True,
    }
    assert failed["execution"]["logs"]["stderr"]["sha256"]
    assert (out_dir / "partial.txt").read_text() == "partial output"
    # A failed stage blocks any fresh start until manual review.
    with pytest.raises(StageStateError, match="manual review"):
        execute_subprocess_stage(**kwargs)


def test_execute_stage_interrupt_writes_failed_record_never_completed(
    tmp_path, monkeypatch, fast_grace
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [out_dir])
    group = FakeProcessGroup(group_dies_on=signal.SIGTERM)
    monkeypatch.setattr(position_benchmark_execution.os, "killpg", group.killpg)

    with pytest.raises(StageInterrupted):
        execute_subprocess_stage(**kwargs, popen=group.popen())

    assert group.signals == [signal.SIGINT, signal.SIGTERM]

    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"failed"}
    assert load_stage_record(records["failed"])["failure"]["reason"] == "interrupted"
    assert load_stage_record(records["failed"])["failure"]["partial_outputs_present"] is False


def test_execute_stage_real_sigint_leaves_no_completed_record(tmp_path):
    script = tmp_path / "sleeper.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)\n"
        "(pathlib.Path(sys.argv[1]) / 'partial.bin').write_bytes(b'x')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(tmp_path, [sys.executable, str(script), str(out_dir)], [out_dir])
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(StageInterrupted):
            execute_subprocess_stage(**kwargs)
    finally:
        timer.cancel()
    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"failed"}
    assert (out_dir / "partial.bin").read_bytes() == b"x"


def test_execute_stage_refuses_non_empty_output_without_record(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "old.pt").write_bytes(b"old")
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )
    with pytest.raises(StageStateError, match="without a completed record"):
        execute_subprocess_stage(**kwargs)
    assert existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"]) == {}
    assert (out_dir / "old.pt").read_bytes() == b"old"


def test_execute_stage_allows_missing_or_empty_output(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )
    execute_subprocess_stage(**kwargs)


def test_record_stage_failure_after_post_validation(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )
    result, running_path = execute_subprocess_stage(**kwargs)

    path = record_stage_failure(
        execution_dir=kwargs["execution_dir"],
        running_path=running_path,
        result=result,
        reason="post_validation_failed",
        owned_paths=[out_dir],
        exception="cv_results.yaml missing",
    )

    assert path == stage_record_path(kwargs["execution_dir"], kwargs["stage_id"], "failed")
    assert set(existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])) == {"failed"}
    failed = load_stage_record(path)
    assert failed["failure"]["reason"] == "post_validation_failed"
    assert failed["failure"]["exit_code"] == 0
    assert (out_dir / "partial.txt").exists()


def test_publish_completed_rejects_non_completed_record(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )
    _, running_path = execute_subprocess_stage(**kwargs)
    running = load_stage_record(running_path)
    with pytest.raises(Exception, match="completed record"):
        publish_completed_record(kwargs["execution_dir"], running_path, running)
    assert set(existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])) == {"running"}


def test_human_summary_unchanged_for_v2_input_files(tmp_path):
    _, _, plan, _ = _plan_fixture(tmp_path)
    assert "input_files" not in build_human_summary(plan)


# ---------------------------------------------------------------------------
# Launch preconditions and defensive failure (review corrections)
# ---------------------------------------------------------------------------


def _forbidden_popen(*args, **kwargs):
    raise AssertionError("no subprocess may be launched when a launch precondition fails")


def _assert_untouched(kwargs: dict, out_dir: Path) -> None:
    assert existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"]) == {}
    assert not (kwargs["execution_dir"] / "stages").exists()
    assert sorted(path.name for path in out_dir.iterdir()) == ["keep.txt"]
    assert (out_dir / "keep.txt").read_text() == "untouched"


@pytest.fixture()
def empty_out(tmp_path):
    out_dir = tmp_path / "out_parent"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("untouched", encoding="utf-8")
    return out_dir


@pytest.mark.parametrize(
    "argv",
    ["python x.py", [], ["/env/bin/python", 3], ("/env/bin/python", None), None],
    ids=["string", "empty", "int-item", "none-item", "none"],
)
def test_execute_stage_malformed_argv_leaves_no_record(tmp_path, empty_out, argv):
    kwargs = _stage_kwargs(tmp_path, argv, [tmp_path / "out"])
    with pytest.raises(StageStateError, match="argv"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)
    assert not (kwargs["execution_dir"] / "logs").exists()


def test_execute_stage_relative_cwd_leaves_no_record(tmp_path, empty_out):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    kwargs["cwd"] = "relative/dir"
    with pytest.raises(StageStateError, match="absolute"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)


def test_execute_stage_missing_cwd_leaves_no_record(tmp_path, empty_out):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    kwargs["cwd"] = tmp_path / "does_not_exist"
    with pytest.raises(StageStateError, match="existing directory"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)


def test_execute_stage_cwd_file_leaves_no_record(tmp_path, empty_out):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    kwargs["cwd"] = empty_out / "keep.txt"
    with pytest.raises(StageStateError, match="existing directory"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_execute_stage_existing_log_leaves_no_record(tmp_path, empty_out, stream):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    log = kwargs["execution_dir"] / "logs" / "runs" / "L3_rope" / f"real_training.{stream}.log"
    log.parent.mkdir(parents=True)
    log.write_text("previous attempt", encoding="utf-8")
    with pytest.raises(StageStateError, match="existing stage log"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)
    assert log.read_text() == "previous attempt"
    assert sorted(p.name for p in log.parent.iterdir()) == [log.name]


def test_execute_stage_blocked_log_directory_leaves_no_record(tmp_path, empty_out):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    blocker = kwargs["execution_dir"] / "logs" / "runs"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(StageStateError, match="non-directory"):
        execute_subprocess_stage(**kwargs, popen=_forbidden_popen)
    _assert_untouched(kwargs, empty_out)


def test_launch_preconditions_reject_bad_log_layout(tmp_path):
    for stem in ("", "a/b", ".", ".."):
        with pytest.raises(StageStateError, match="log stem"):
            position_benchmark_execution.validate_launch_preconditions(
                ["/env/bin/python"], cwd=tmp_path, log_dir=tmp_path / "logs", log_stem=stem
            )
    with pytest.raises(StageStateError, match="log directory must be absolute"):
        position_benchmark_execution.validate_launch_preconditions(
            ["/env/bin/python"], cwd=tmp_path, log_dir="logs", log_stem="s"
        )
    assert not (tmp_path / "logs").exists()


def test_execute_stage_unexpected_launch_exception_records_failure(tmp_path, fake_stage):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )

    def exploding_popen(*args, **kwargs):
        raise RuntimeError("unexpected launch bug")

    with pytest.raises(StageFailure, match="unexpected launch bug") as info:
        execute_subprocess_stage(**kwargs, popen=exploding_popen)

    assert isinstance(info.value.__cause__, RuntimeError)
    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"failed"}
    failed = load_stage_record(records["failed"])
    assert failed["failure"] == {
        "reason": "unexpected_error",
        "exit_code": None,
        "exception": "RuntimeError: unexpected launch bug",
        "partial_outputs_present": False,
    }
    assert failed["execution"]["argv"] == kwargs["argv"]
    assert failed["execution"]["cwd"] == str(tmp_path)
    # Both logs had been opened before launch, so the record fingerprints them.
    assert failed["execution"]["logs"]["stdout"]["size"] == 0


def test_execute_stage_unexpected_exception_after_launch_keeps_partial_output(
    tmp_path, fake_stage, monkeypatch
):
    out_dir = tmp_path / "out"
    kwargs = _stage_kwargs(
        tmp_path, [sys.executable, str(fake_stage), str(out_dir), "0"], [out_dir]
    )
    real_stage_result = position_benchmark_execution._stage_result

    def broken_stage_result(*args, **kwargs):
        raise ValueError("hashing bug after the child finished")

    monkeypatch.setattr(position_benchmark_execution, "_stage_result", broken_stage_result)
    with pytest.raises(StageFailure, match="hashing bug"):
        execute_subprocess_stage(**kwargs)
    monkeypatch.setattr(position_benchmark_execution, "_stage_result", real_stage_result)

    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"failed"}
    failed = load_stage_record(records["failed"])
    assert failed["failure"]["reason"] == "unexpected_error"
    assert failed["failure"]["partial_outputs_present"] is True
    assert (out_dir / "partial.txt").read_text() == "partial output"


def test_execute_stage_failed_record_publication_failure_keeps_running(tmp_path, monkeypatch):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    real_write = position_benchmark_execution.write_stage_record

    def write_running_only(execution_dir, record):
        if record["record_kind"] != "running":
            raise OSError("disk full")
        return real_write(execution_dir, record)

    def exploding_popen(*args, **kwargs):
        raise RuntimeError("launch bug")

    monkeypatch.setattr(position_benchmark_execution, "write_stage_record", write_running_only)
    with pytest.raises(
        position_benchmark_execution.StageRecordError, match="running marker"
    ) as info:
        execute_subprocess_stage(**kwargs, popen=exploding_popen)

    assert isinstance(info.value.__cause__, RuntimeError)
    assert "launch bug" in str(info.value) and "disk full" in str(info.value)
    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"running"}


def test_execute_stage_does_not_convert_system_exit(tmp_path):
    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])

    def exiting_popen(*args, **kwargs):
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        execute_subprocess_stage(**kwargs, popen=exiting_popen)
    # BaseException is not converted: the running marker stays (fail closed).
    records = existing_stage_records(kwargs["execution_dir"], kwargs["stage_id"])
    assert set(records) == {"running"}


def test_execute_stage_launches_in_new_session(tmp_path):
    seen = {}

    class SpyPopen:
        def __init__(self, args, **kwargs):
            seen.update(kwargs)
            self.pid = 987654

        def wait(self, timeout=None):
            return 0

    kwargs = _stage_kwargs(tmp_path, ["/env/bin/python", "x.py"], [tmp_path / "out"])
    execute_subprocess_stage(**kwargs, popen=SpyPopen)
    assert seen["start_new_session"] is True
    assert seen["shell"] is False
