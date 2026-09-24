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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import (
    compare_ablation_rankings,
    create_null_baseline,
    explain,
    position_benchmark_execution,
    position_benchmark_manifest,
    run_position_benchmark,
    train,
)
from scripts.position_benchmark_execution import (
    ENVIRONMENT_PROBE,
    NULL_VALIDATION_STAGE_ID,
    SHARED_NULL_STAGE_ID,
    CalibrationInputGateError,
    CompletedStageMismatchError,
    EnvironmentProbeError,
    ExecutionLock,
    ExecutionLockError,
    NullPreflightError,
    OutputLocationError,
    PlanAuthorityError,
    RepositoryGateError,
    StageFailure,
    StageInterrupted,
    StagePostValidationError,
    StageStateError,
    benchmark_root_from_plan,
    bind_benchmark_plan,
    build_benchmark_stages,
    build_null_validation_report,
    execute_benchmark_plan,
    execute_subprocess_stage,
    load_persisted_execution_plan,
    null_validation_path,
    output_manifest_path,
    probe_execution_environment,
    publish_completed_record,
    record_stage_failure,
    require_execution_locations_safe,
    require_output_location_safe,
    require_real_only_comparison,
    require_repository_gate,
    run_null_preflight,
    run_subprocess_stage,
    scan_benchmark_state,
    shared_null_validation_path,
    verify_plan_rebuild,
    write_null_validation_report,
)
from scripts.position_benchmark_manifest import (
    build_human_summary,
    build_resolved_plan,
    resolved_plan_file_sha256,
    write_resolved_plan,
)
from scripts.position_benchmark_pairing import (
    CONTENT_BASELINE_POLICY,
    PairCompatibilityError,
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
from src.data.dataset_provenance import build_dataset_provenance
from src.encoding import AnnotationLevel
from src.training.split_plan import ordered_sample_ids
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


def test_public_cli_without_mode_is_a_concise_usage_error(tmp_path, capsys):
    # Phase 12C3B2B replaced the 12C3B2A "execution is not implemented yet"
    # refusal; a bare invocation is still an exit-2 usage error, never execution.
    manifest_path, _ = _paired_manifest(tmp_path)
    capsys.readouterr()
    assert run_position_benchmark.main([str(manifest_path)]) == 2
    err = capsys.readouterr().err
    assert "choose --dry-run or --execute-plan PLAN" in err
    assert "Traceback" not in err


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


# ===========================================================================
# Phase 12C3B2B: paired benchmark DAG execution, post-validation, and resume
# ===========================================================================
#
# No GPU and no real training: a fake ``Popen`` intercepts every planned argv
# and writes small deterministic outputs with the SAME metadata helpers that
# train.py / explain.py use (split-plan replay, dataset provenance, resolved
# position encoding, fold config/info, model provenance), over the tiny
# 8-sample strict-null fixture. The real 12C3A validate_null_pair and 12C3B1
# pair rules run unmodified.

RUN_IDS = ("legacy", "no_position")


def _argv_value(argv, flag):
    return argv[argv.index(flag) + 1]


def _stage_id_for_argv(argv) -> str:
    script = Path(argv[1]).name
    if script == "train.py":
        out = Path(_argv_value(argv, "--output-dir"))
        return f"runs/{out.parent.name}/{out.name}_training"
    if script == "explain.py":
        out = Path(_argv_value(argv, "--output-dir"))
        return f"runs/{out.parent.parent.name}/{out.parent.name}_explanation"
    if script == "bootstrap_null_calibration.py":
        return f"runs/{Path(_argv_value(argv, '--output')).parent.parent.name}/calibration"
    return {
        "ablation_compare.py": "comparisons/performance",
        "compare_ablation_rankings.py": "comparisons/raw_rankings",
        "compare_position_attributions.py": "comparisons/raw_attributions",
    }[script]


def _fake_train(argv) -> None:
    """Write train.py-shaped outputs using train.py's own metadata helpers."""
    ns = train.build_arg_parser().parse_args(argv[2:])
    output_dir = Path(ns.output_dir) / ns.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed = torch.load(ns.preprocessed_data, weights_only=False)
    samples = preprocessed["samples"]
    labels = np.array([sample.label for sample in samples])
    ns.pc_map_sha256 = None
    ns.num_covariates = 0
    config = dict(vars(ns))
    chrom_index = {"1": 0}
    resolved = train.prepare_training_position_encoding(
        ns, AnnotationLevel[ns.level], num_chromosomes=len(chrom_index)
    )
    identity = train.write_dataset_mappings_artifact(output_dir, {"GENE1": 0}, chrom_index)
    mode = "cv" if ns.cv is not None else "single_split"
    split_plan, split_metadata, _ = train.prepare_training_split_plan(
        args=ns, output_dir=output_dir, sample_ids=ordered_sample_ids(samples), labels=labels
    )
    run_metadata = train.build_training_run_metadata(
        input_dim=resolved.input_dim,
        num_genes=1,
        num_chromosomes=len(chrom_index),
        genome_build=ns.genome_build,
        resolved_position_encoding=resolved,
        chrom_index=chrom_index,
        gene_mapping_sha256=str(identity["gene_mapping_sha256"]),
        chromosome_mapping_sha256=str(identity["chromosome_mapping_sha256"]),
        training_mode=mode,
    )
    run_metadata["split_plan"] = split_metadata
    run_metadata["dataset_provenance"] = build_dataset_provenance(
        preprocessed, path=Path(ns.preprocessed_data)
    )
    config.update(run_metadata)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = {"auc": 0.5, "accuracy": 0.5, "best_epoch": 1, "epochs_trained": 1}
    # Class-weighting metadata is produced ONLY through train.py's own helpers
    # (_update_saved_config, _resolve_pos_weight, save_fold_config) so this fake
    # cannot drift from the real serialization contract.
    if mode == "cv":
        train._update_saved_config(
            config_path, class_weighting_applied=None, class_weighting_pos_weight=None
        )
        for fold in split_plan["folds"]:
            index = fold["fold_index"]
            fold_dir = output_dir / f"fold_{index}"
            fold_dir.mkdir()
            # Side-specific bytes: real and null weights are never identical.
            (fold_dir / "best_model.pt").write_bytes(f"weights {output_dir} {index}".encode())
            ns._fold_pos_weight = train._resolve_pos_weight(
                labels[fold["train_indices"]], ns.class_weighting
            )
            train.save_fold_config(fold_dir, index, ns, run_metadata=run_metadata)
            now = train.datetime.now(train.timezone.utc)
            train.save_fold_info(
                fold_dir=fold_dir,
                fold_idx=index,
                n_folds=ns.cv,
                seed=ns.seed,
                train_indices=fold["train_indices"],
                val_indices=fold["val_indices"],
                labels=labels,
                fold_metrics=metrics,
                training_started=now,
                training_completed=now,
            )
        results = {"mean_auc": 0.5, "fold_results": [{"auc": 0.5, "accuracy": 0.5}] * ns.cv}
        (output_dir / "cv_results.yaml").write_text(yaml.safe_dump(results), encoding="utf-8")
    else:
        pos_weight = train._resolve_pos_weight(
            labels[split_plan["train_indices"]], ns.class_weighting
        )
        weight_value = float(pos_weight.item()) if pos_weight is not None else None
        train._update_saved_config(
            config_path,
            class_weighting_applied=pos_weight is not None,
            class_weighting_pos_weight=weight_value,
        )
        (output_dir / "best_model.pt").write_bytes(f"weights {output_dir}".encode())
        results = {
            "auc": 0.5,
            "accuracy": 0.5,
            "class_weighting_applied": pos_weight is not None,
            "class_weighting_pos_weight": weight_value,
        }
        (output_dir / "results.yaml").write_text(yaml.safe_dump(results), encoding="utf-8")


def _fake_explain(argv) -> None:
    """Write explain.py-shaped outputs, including exact model provenance."""
    ns = explain.build_arg_parser().parse_args(argv[2:])
    experiment_dir = Path(ns.experiment_dir).resolve()
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    if ns.fold_index is not None:
        checkpoint = experiment_dir / f"fold_{ns.fold_index}" / "best_model.pt"
        mode, auc, cv_results = "cv_explicit_fold", 0.5, experiment_dir / "cv_results.yaml"
    else:
        checkpoint = experiment_dir / "best_model.pt"
        mode, auc, cv_results = "single_run_best_model", None, None
    provenance = explain._build_model_provenance(
        checkpoint_selection_mode=mode,
        checkpoint_path=checkpoint,
        config_path=experiment_dir / "config.yaml",
        selected_fold=ns.fold_index,
        selected_fold_auc=auc,
        cv_results_path=cv_results,
    )
    preprocessed = torch.load(ns.preprocessed_data, weights_only=False)
    samples = preprocessed["samples"]
    scores = np.empty(len(samples), dtype=object)
    metadata = np.empty(len(samples), dtype=object)
    per_sample = output_dir / "attributions_per_sample"
    per_sample.mkdir()
    for index, sample in enumerate(samples):
        scores[index] = np.array([0.1 * (index + 1)])
        metadata[index] = {
            "sample_idx": index,
            "sample_id": sample.sample_id,
            "chromosomes": np.array([variant.chrom for variant in sample.variants]),
            "positions": np.array([variant.pos for variant in sample.variants]),
            "gene_ids": np.array([0 for _ in sample.variants]),
        }
        np.savez(per_sample / f"sample_{index}.npz", variant_scores=scores[index])
    np.savez(output_dir / "attributions.npz", variant_scores=scores, metadata=metadata)
    (output_dir / "sieve_variant_rankings.csv").write_text(
        "chromosome,position,gene_name,mean_attribution\n1,100,GENE1,0.5\n", encoding="utf-8"
    )
    (output_dir / "sieve_gene_rankings.csv").write_text(
        "gene_name,gene_score\nGENE1,0.5\n", encoding="utf-8"
    )
    integrated_gradients = {
        "executed": True,
        "attribution_schema_version": explain.ATTRIBUTION_SCHEMA_VERSION,
        "requested_ig_mode": ns.ig_mode,
        "resolved_ig_mode": "content",
        "attribution_feature_space": "content",
        "attribution_width": config["content_dim"],
        "content_dim": config["content_dim"],
        "input_dim": config["input_dim"],
        "variant_score_aggregation": explain.VARIANT_SCORE_AGGREGATION,
        "baseline_policy": CONTENT_BASELINE_POLICY,
        "n_steps": ns.n_steps,
        "max_variants": min(ns.max_variants, 2000),
        "sampling_policy": explain.SAMPLING_POLICY,
        "sampling_seed": None,
        "comparability_warning": None,
    }
    analysis = {
        "is_null_baseline": ns.is_null_baseline,
        "experiment_dir": str(ns.experiment_dir),
        "genome_build": ns.genome_build,
        "n_samples": len(samples),
        "annotation_level": config["level"],
        "n_integration_steps": ns.n_steps,
        "max_variants_per_sample": ns.max_variants,
        "aggregation_method": ns.aggregation_method,
        "skip_attention": ns.skip_attention,
        "skip_ig": ns.skip_ig,
        "model_provenance": provenance,
        "dataset_provenance": build_dataset_provenance(
            preprocessed, path=Path(ns.preprocessed_data)
        ),
        "integrated_gradients": integrated_gradients,
        "attention_threshold_mode": ns.attention_threshold_mode,
        "attention_threshold": ns.attention_threshold,
        "attention_percentile": ns.attention_percentile,
    }
    (output_dir / "analysis_metadata.yaml").write_text(
        yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8"
    )


def _fake_bootstrap(argv) -> None:
    """Write bootstrap_null_calibration.py-shaped outputs with its actual summary keys."""
    with np.load(_argv_value(argv, "--null-attributions"), allow_pickle=True) as data:
        n_null = len(data["metadata"])
    Path(_argv_value(argv, "--output")).write_text("variant,delta_rank\nv,1\n", encoding="utf-8")
    Path(_argv_value(argv, "--output-gene-stats")).write_text("gene\nGENE1\n", encoding="utf-8")
    summary = {
        "n_bootstrap": int(_argv_value(argv, "--n-bootstrap")),
        "genome_build": _argv_value(argv, "--genome-build"),
        "n_real_variants": 1,
        "n_null_samples": n_null,
        "n_unique_null_variants": 8,
        "excluded_sex_chroms": "--exclude-sex-chroms" in argv,
        "n_real_variants_removed_sex_chroms": 0,
        "n_null_rows_removed_sex_chroms": 0,
        "n_real_variants_missing_from_null": 0,
        "per_gene": {
            "n_tested": 1,
            "gene_delta_rank_aggregation": _argv_value(argv, "--gene-delta-rank-aggregation"),
        },
    }
    Path(_argv_value(argv, "--output-summary")).write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )


def _fake_comparison(argv) -> None:
    for index, token in enumerate(argv):
        if token.startswith("--out-"):
            path = Path(argv[index + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{Path(argv[1]).name} output\n", encoding="utf-8")


_FAKE_SCRIPTS = {
    "train.py": _fake_train,
    "explain.py": _fake_explain,
    "bootstrap_null_calibration.py": _fake_bootstrap,
    "ablation_compare.py": _fake_comparison,
    "compare_ablation_rankings.py": _fake_comparison,
    "compare_position_attributions.py": _fake_comparison,
}


class _FinishedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.pid = 2**22 + 12345

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class FakeBenchmarkWorld:
    """A fake ``Popen`` that records exact argv and synthesizes stage outputs.

    ``before`` / ``after`` hooks (keyed by stage ID) let a test tamper with
    files at a precise DAG point; ``exit_codes`` makes a stage exit non-zero.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.exit_codes: dict[str, int] = {}
        self.before: dict = {}
        self.after: dict = {}

    def __call__(self, argv, **kwargs):
        assert isinstance(argv, list) and kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        stage_id = _stage_id_for_argv(argv)
        self.calls.append((stage_id, list(argv)))
        if stage_id in self.before:
            self.before[stage_id](argv)
        code = self.exit_codes.get(stage_id, 0)
        if code == 0:
            _FAKE_SCRIPTS[Path(argv[1]).name](argv)
            if stage_id in self.after:
                self.after[stage_id](argv)
        return _FinishedProcess(code)

    def stage_ids(self) -> list[str]:
        return [stage_id for stage_id, _ in self.calls]


def _clean_git(plan: dict, overrides: dict | None = None) -> FakeGit:
    answers = {
        ("rev-parse", "--show-toplevel"): (0, f"{plan['repository_root']}\n", ""),
        ("rev-parse", "HEAD"): (0, f"{plan['repository_revision']}\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=all"): (0, "", ""),
    }
    answers.update(overrides or {})
    return FakeGit(answers)


@pytest.fixture()
def bench(tmp_path):
    return _bench(tmp_path)


def _bench(tmp_path: Path, mode: str = "cv") -> SimpleNamespace:
    manifest_path, _, plan, plan_path = _plan_fixture(tmp_path, mode=mode)
    root = benchmark_root_from_plan(plan)
    return SimpleNamespace(
        manifest_path=manifest_path,
        plan=plan,
        plan_path=plan_path,
        root=root,
        execution=root / "execution",
        world=FakeBenchmarkWorld(),
        stages=build_benchmark_stages(plan),
    )


def _run(bench, *, resume: bool = False, git: FakeGit | None = None) -> dict:
    return execute_benchmark_plan(
        bench.plan_path,
        manifest_path=bench.manifest_path,
        resume=resume,
        git_runner=git or _clean_git(bench.plan),
        probe_runner=_probe_runner(PROBE_PAYLOAD),
        popen=bench.world,
    )


def _record_kinds(bench) -> dict[str, list[str]]:
    return {
        stage.stage_id: sorted(existing_stage_records(bench.execution, stage.stage_id))
        for stage in bench.stages
    }


def _completed_record(bench, stage_id: str) -> dict:
    return load_stage_record(stage_record_path(bench.execution, stage_id, "completed"))


def _edit_yaml(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run_dir(bench, run_id: str) -> Path:
    return bench.root / "runs" / run_id


EXPECTED_ORDER = [
    NULL_VALIDATION_STAGE_ID,
    *[
        f"runs/{run_id}/{name}"
        for run_id in RUN_IDS
        for name in (
            "real_training",
            "real_explanation",
            "null_training",
            "null_explanation",
            "pair_validation",
            "calibration",
        )
    ],
    SHARED_NULL_STAGE_ID,
    "comparisons/performance",
    "comparisons/raw_rankings",
    "comparisons/raw_attributions",
]
SUBPROCESS_ORDER = [
    stage_id
    for stage_id in EXPECTED_ORDER
    if not stage_id.startswith("benchmark/") and not stage_id.endswith("pair_validation")
]


# --- DAG --------------------------------------------------------------------


def test_stage_dag_order_ids_and_exact_plan_argv(bench):
    stages = bench.stages
    assert [stage.stage_id for stage in stages] == EXPECTED_ORDER
    by_id = {stage.stage_id: stage for stage in stages}
    for run in bench.plan["runs"]:
        prefix = f"runs/{run['run_id']}"
        assert by_id[f"{prefix}/real_training"].argv == tuple(run["train_argv"])
        assert by_id[f"{prefix}/real_explanation"].argv == tuple(run["explain_argv"])
        assert by_id[f"{prefix}/null_training"].argv == tuple(run["null_train_argv"])
        assert by_id[f"{prefix}/null_explanation"].argv == tuple(run["null_explain_argv"])
        assert by_id[f"{prefix}/calibration"].argv == tuple(run["calibration"]["argv"])
        assert by_id[f"{prefix}/pair_validation"].argv is None
        assert "--fold-index" in run["explain_argv"] and "--fold-index" in run["null_explain_argv"]
        assert _argv_value(run["null_explain_argv"], "--fold-index") == "0"
        assert "--is-null-baseline" in run["null_explain_argv"]
        assert "--is-null-baseline" not in run["explain_argv"]
    for name in ("performance", "raw_rankings", "raw_attributions"):
        assert by_id[f"comparisons/{name}"].argv == tuple(bench.plan["comparisons"][name]["argv"])
    assert (
        by_id[NULL_VALIDATION_STAGE_ID].callable_name == "src.data.null_lineage.validate_null_pair"
    )
    assert by_id["runs/legacy/pair_validation"].callable_name == (
        "scripts.position_benchmark_pairing.require_compatible_real_null_pair"
    )
    assert by_id[SHARED_NULL_STAGE_ID].callable_name == (
        "scripts.position_benchmark_pairing.require_shared_null_across_pairs"
    )
    # The second strategy's pair validation binds the first's (cumulative shared null).
    assert "runs/legacy/pair_validation" in by_id["runs/no_position/pair_validation"].dependency_ids


def test_full_dag_executes_exact_argv_in_order_and_completes(bench):
    result = _run(bench)

    assert bench.world.stage_ids() == SUBPROCESS_ORDER
    by_id = {stage.stage_id: stage for stage in bench.stages}
    for stage_id, argv in bench.world.calls:
        assert argv == list(by_id[stage_id].argv)
    assert result["executed"] == EXPECTED_ORDER and result["reused"] == []
    assert all(kinds == ["completed"] for kinds in _record_kinds(bench).values())
    summary = result["summary"]
    assert summary["completed_stages"] == len(EXPECTED_ORDER)
    assert summary["real_training_completed"] == summary["null_training_completed"] == 2
    assert summary["pair_validations_completed"] == summary["calibrations_completed"] == 2
    assert summary["shared_null_validation"] == "passed"
    assert summary["raw_comparisons_completed"] == [
        "performance",
        "raw_rankings",
        "raw_attributions",
    ]
    assert summary["calibrated_cross_strategy_comparison"] == (
        "not_executed_gate_closed_until_phase_12c3c"
    )
    assert "calibrated_rankings" not in {
        path.name for path in (bench.root / "comparisons").iterdir()
    }
    # Every record binds the exact plan bytes and revision.
    plan_sha = resolved_plan_file_sha256(bench.plan_path)
    for stage_id in EXPECTED_ORDER:
        record = _completed_record(bench, stage_id)
        assert record["resolved_plan_sha256"] == plan_sha
        assert record["repository_revision"] == bench.plan["repository_revision"]
        assert record["inputs"]["resolved_plan"]["sha256"] == plan_sha


def test_full_dag_single_split(tmp_path):
    bench = _bench(tmp_path, mode="single_split")
    result = _run(bench)
    assert result["summary"]["completed_stages"] == len(EXPECTED_ORDER)
    record = _completed_record(bench, "runs/legacy/real_training")
    assert "results.yaml" in record["outputs"] and "cv_results.yaml" not in record["outputs"]
    explanation = _completed_record(bench, "runs/legacy/real_explanation")
    assert "model_provenance.checkpoint_selection_mode" in explanation["post_validation"]["checks"]


def test_null_validation_stage_record_binds_preflight_inputs(bench):
    _run(bench)
    record = _completed_record(bench, NULL_VALIDATION_STAGE_ID)
    files = bench.plan["input_files"]
    assert record["execution"] == {
        "kind": "in_process",
        "callable": "src.data.null_lineage.validate_null_pair",
    }
    assert record["side"] == "benchmark" and record["run_id"] is None
    assert record["inputs"]["real_dataset"]["sha256"] == files["preprocessed_data"]["sha256"]
    assert record["inputs"]["null_artifact"]["sha256"] == files["null_artifact"]["sha256"]
    assert (
        record["inputs"]["null_lineage_sidecar"]["sha256"]
        == files["null_lineage_sidecar"]["sha256"]
    )
    assert record["inputs"]["split_plan"]["sha256"] == files["split_plan"]["sha256"]
    report = yaml.safe_load(null_validation_path(bench.root).read_text(encoding="utf-8"))
    assert report["resolved_plan_sha256"] == resolved_plan_file_sha256(bench.plan_path)
    assert record["outputs"]["null_validation.yaml"]["sha256"] == sha256_file(
        null_validation_path(bench.root)
    )
    training = _completed_record(bench, "runs/legacy/real_training")
    assert training["dependencies"][0]["stage_id"] == NULL_VALIDATION_STAGE_ID
    assert training["dependencies"][0]["record_sha256"] == sha256_file(
        stage_record_path(bench.execution, NULL_VALIDATION_STAGE_ID, "completed")
    )


def test_null_preflight_failure_runs_no_stage(bench, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("labels are not a permutation")

    with pytest.raises(NullPreflightError, match="validate_null_pair failed"):
        execute_benchmark_plan(
            bench.plan_path,
            manifest_path=bench.manifest_path,
            git_runner=_clean_git(bench.plan),
            probe_runner=_probe_runner(PROBE_PAYLOAD),
            null_validator=broken,
            popen=bench.world,
        )
    assert bench.world.calls == []
    assert not (bench.execution / "stages").exists()


@pytest.mark.parametrize(
    "failing",
    ["runs/legacy/null_training", "runs/legacy/calibration", "comparisons/raw_rankings"],
)
def test_first_failed_stage_stops_the_dag(bench, failing):
    bench.world.exit_codes[failing] = 3
    with pytest.raises(StageFailure, match=failing):
        _run(bench)
    kinds = _record_kinds(bench)
    position = EXPECTED_ORDER.index(failing)
    assert all(kinds[stage_id] == ["completed"] for stage_id in EXPECTED_ORDER[:position])
    assert kinds[failing] == ["failed"]
    assert all(kinds[stage_id] == [] for stage_id in EXPECTED_ORDER[position + 1 :])
    assert bench.world.stage_ids()[-1] == failing


def test_repository_change_during_stage_fails_without_completed_record(bench):
    head = ("rev-parse", "HEAD")
    git = _clean_git(bench.plan)
    original = git.answers[head]
    counter = {"calls": 0}
    base_call = git.__call__

    class ChangingGit(FakeGit):
        def __call__(self, argv, **kwargs):
            if tuple(argv[1:]) == head:
                counter["calls"] += 1
                # Gate 1-3: pre-execution, NV before, NV after; 4: training before.
                self.answers[head] = (
                    (0, OTHER_REVISION + "\n", "") if counter["calls"] > 4 else original
                )
            return base_call(argv, **kwargs)

    changing = ChangingGit(git.answers)
    with pytest.raises(StagePostValidationError, match="repository_gate_failed"):
        _run(bench, git=changing)
    kinds = _record_kinds(bench)
    assert kinds["runs/legacy/real_training"] == ["failed"]
    failed = load_stage_record(
        stage_record_path(bench.execution, "runs/legacy/real_training", "failed")
    )
    assert failed["failure"]["reason"] == "repository_gate_failed"
    assert failed["failure"]["partial_outputs_present"] is True


def test_stage_keyboard_interrupt_writes_failed_record(bench, monkeypatch):
    original = position_benchmark_execution._POST_VALIDATORS["training"]

    def interrupted(ctx, stage):
        raise KeyboardInterrupt

    monkeypatch.setitem(position_benchmark_execution._POST_VALIDATORS, "training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(bench)
    monkeypatch.setitem(position_benchmark_execution._POST_VALIDATORS, "training", original)
    failed = load_stage_record(
        stage_record_path(bench.execution, "runs/legacy/real_training", "failed")
    )
    assert failed["failure"]["reason"] == "interrupted"
    assert _record_kinds(bench)["runs/legacy/real_explanation"] == []


# --- training post-validation ------------------------------------------------


def _config_edit(mutate):
    def hook(argv):
        out = Path(_argv_value(argv, "--output-dir")) / "training" / "config.yaml"
        _edit_yaml(out, mutate)

    return hook


def _remove(relative):
    def hook(argv):
        (Path(_argv_value(argv, "--output-dir")) / "training" / relative).unlink()

    return hook


def _set(dotted, value):
    def mutate(data):
        *parents, leaf = dotted.split(".")
        for part in parents:
            data = data[part]
        data[leaf] = value

    return mutate


TRAINING_REJECTIONS = [
    ("real", _remove("config.yaml"), "missing"),
    ("real", _remove("dataset_mappings.json"), "missing"),
    ("real", _remove("split_plan.yaml"), "missing"),
    ("real", _remove("cv_results.yaml"), "missing"),
    ("real", _remove("fold_0/config.yaml"), "missing"),
    ("real", _remove("fold_1/fold_info.yaml"), "missing"),
    ("real", _remove("fold_0/best_model.pt"), "missing"),
    (
        "real",
        _config_edit(_set("dataset_provenance.preprocessed_data_sha256", "0" * 64)),
        "preprocessed_data_sha256",
    ),
    ("real", _config_edit(_set("dataset_provenance.is_null_baseline", True)), "is_null_baseline"),
    (
        "null",
        _config_edit(_set("dataset_provenance.null_lineage.lineage_sha256", "1" * 64)),
        "lineage_sha256",
    ),
    (
        "real",
        _config_edit(_set("dataset_provenance.sample_ids_sha256", "2" * 64)),
        "dataset_provenance.sample_ids_sha256",
    ),
    ("real", _config_edit(_set("split_plan.sha256", "3" * 64)), "config.split_plan.sha256"),
    ("real", _config_edit(_set("split_plan.source", "generated")), "config.split_plan.source"),
    ("real", _config_edit(_set("seed", 43)), "config.seed"),
    ("real", _config_edit(_set("class_weighting", "auto")), "config.class_weighting"),
    ("real", _config_edit(_set("level", "L2")), "config.level"),
    ("real", _config_edit(_set("latent_dim", 32)), "config.latent_dim"),
    (
        "real",
        _config_edit(_set("position_encoding_execution.relative_position_encoding", "rope")),
        "position_encoding_execution",
    ),
    (
        "real",
        _config_edit(_set("position_encoding.relative.type", "none")),
        "position",
    ),
]


@pytest.mark.parametrize(("side", "hook", "message"), TRAINING_REJECTIONS)
def test_training_post_validation_rejects(bench, side, hook, message):
    stage_id = f"runs/legacy/{side}_training"
    bench.world.after[stage_id] = hook
    with pytest.raises(StagePostValidationError, match=message):
        _run(bench)
    kinds = _record_kinds(bench)
    assert kinds[stage_id] == ["failed"]
    failed = load_stage_record(stage_record_path(bench.execution, stage_id, "failed"))
    assert failed["failure"]["reason"] == "post_validation_failed"
    assert failed["failure"]["partial_outputs_present"] is True
    # Partial outputs are retained, never deleted.
    assert (_run_dir(bench, "legacy") / side / "training").is_dir()
    assert bench.world.stage_ids()[-1] == stage_id


def test_training_post_validation_rejects_wrong_fold_split(bench):
    def hook(argv):
        info = Path(_argv_value(argv, "--output-dir")) / "training" / "fold_0" / "fold_info.yaml"
        _edit_yaml(info, _set("train_sample_indices", [0, 1, 2, 3]))

    bench.world.after["runs/legacy/real_training"] = hook
    with pytest.raises(StagePostValidationError, match="replayed split plan"):
        _run(bench)


def _training_dir(bench, side: str = "real") -> Path:
    return _run_dir(bench, "legacy") / side / "training"


def test_cv_class_weighting_matches_real_train_py_contract(bench):
    _run(bench)
    training = _training_dir(bench)
    root = yaml.safe_load((training / "config.yaml").read_text(encoding="utf-8"))
    assert root["class_weighting"] == "off"
    # train.py's CV branch writes both keys to the root config as null.
    assert "class_weighting_applied" in root and root["class_weighting_applied"] is None
    assert "class_weighting_pos_weight" in root and root["class_weighting_pos_weight"] is None
    for fold in range(bench.plan["split_plan"]["n_folds"]):
        fold_config = yaml.safe_load(
            (training / f"fold_{fold}" / "config.yaml").read_text(encoding="utf-8")
        )
        assert fold_config["class_weighting_applied"] is False
        assert fold_config["class_weighting_pos_weight"] is None
    checks = _completed_record(bench, "runs/legacy/real_training")["post_validation"]["checks"]
    for label in (
        "config.class_weighting",
        "config.class_weighting_applied",
        "config.class_weighting_pos_weight",
        "fold_0/config.yaml.class_weighting_applied",
        "fold_1/config.yaml.class_weighting_pos_weight",
    ):
        assert label in checks


def _fold_edit(fold: int, mutate):
    def hook(argv):
        path = Path(_argv_value(argv, "--output-dir")) / "training" / f"fold_{fold}" / "config.yaml"
        _edit_yaml(path, mutate)

    return hook


@pytest.mark.parametrize(
    ("hook", "message"),
    [
        (_config_edit(_set("class_weighting", "auto")), "config.class_weighting must"),
        (_config_edit(_set("class_weighting_applied", False)), "config.class_weighting_applied"),
        (
            _config_edit(_set("class_weighting_pos_weight", 1.5)),
            "config.class_weighting_pos_weight",
        ),
        (
            _fold_edit(1, _set("class_weighting_applied", True)),
            "fold_1/config.yaml.class_weighting_applied",
        ),
        (
            _fold_edit(0, _set("class_weighting_pos_weight", 2.0)),
            "fold_0/config.yaml.class_weighting_pos_weight",
        ),
    ],
)
def test_cv_class_weighting_mutations_reject(bench, hook, message):
    bench.world.after["runs/legacy/real_training"] = hook
    with pytest.raises(StagePostValidationError, match=message):
        _run(bench)
    assert _record_kinds(bench)["runs/legacy/real_training"] == ["failed"]


def test_single_split_class_weighting_matches_real_train_py_contract(tmp_path):
    bench = _bench(tmp_path, mode="single_split")
    _run(bench)
    training = _training_dir(bench)
    root = yaml.safe_load((training / "config.yaml").read_text(encoding="utf-8"))
    results = yaml.safe_load((training / "results.yaml").read_text(encoding="utf-8"))
    assert root["class_weighting"] == "off"
    assert root["class_weighting_applied"] is False
    assert root["class_weighting_pos_weight"] is None
    assert results["class_weighting_applied"] is False
    assert results["class_weighting_pos_weight"] is None


def _results_edit(mutate):
    def hook(argv):
        _edit_yaml(Path(_argv_value(argv, "--output-dir")) / "training" / "results.yaml", mutate)

    return hook


@pytest.mark.parametrize(
    ("hook", "message"),
    [
        (_config_edit(_set("class_weighting", "on")), "config.class_weighting must"),
        (_config_edit(_set("class_weighting_applied", True)), "config.class_weighting_applied"),
        (
            _config_edit(_set("class_weighting_pos_weight", 1.0)),
            "config.class_weighting_pos_weight",
        ),
        (_results_edit(_set("class_weighting_applied", True)), "results.class_weighting_applied"),
        (
            _results_edit(_set("class_weighting_pos_weight", 1.0)),
            "results.class_weighting_pos_weight",
        ),
    ],
)
def test_single_split_class_weighting_mutations_reject(tmp_path, hook, message):
    bench = _bench(tmp_path, mode="single_split")
    bench.world.after["runs/legacy/real_training"] = hook
    with pytest.raises(StagePostValidationError, match=message):
        _run(bench)
    assert _record_kinds(bench)["runs/legacy/real_training"] == ["failed"]


def test_static_producer_contract_real_train_helpers_are_accepted(tmp_path):
    """Build class-weighting artifacts with train.py's own helpers, no training.

    A root config is updated exactly as train.py's CV branch does and a fold
    config is written by ``train.save_fold_config`` with the pos_weight that
    ``train._resolve_pos_weight`` returns for ``class_weighting="off"``. The
    executor's class-weighting checks must accept both, so neither the fake
    producer nor the validator can drift from the real serialization.
    """
    root_path = tmp_path / "config.yaml"
    root_path.write_text(yaml.safe_dump({"class_weighting": "off"}), encoding="utf-8")
    train._update_saved_config(
        root_path, class_weighting_applied=None, class_weighting_pos_weight=None
    )
    ns = train.build_arg_parser().parse_args(
        ["--preprocessed-data", "x.pt", "--level", "L3", "--class-weighting", "off"]
    )
    ns.experiment_name = "training"
    ns.num_covariates = 0
    ns.pc_map_sha256 = None
    ns._fold_pos_weight = train._resolve_pos_weight(np.array([0, 1, 1, 1, 1]), "off")
    assert ns._fold_pos_weight is None
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    train.save_fold_config(fold_dir, 0, ns, run_metadata=None)

    root = yaml.safe_load(root_path.read_text(encoding="utf-8"))
    fold = yaml.safe_load((fold_dir / "config.yaml").read_text(encoding="utf-8"))
    stage = position_benchmark_execution.BenchmarkStage(
        "runs/r/real_training",
        "training",
        "real",
        "r",
        ("python", "train.py"),
        None,
        (tmp_path,),
        (),
    )
    checks: list[str] = []
    check = position_benchmark_execution._check
    check(stage, "config.class_weighting", root.get("class_weighting"), "off", checks)
    check(
        stage,
        "config.class_weighting_applied",
        root.get("class_weighting_applied", "<missing>"),
        None,
        checks,
    )
    check(
        stage,
        "config.class_weighting_pos_weight",
        root.get("class_weighting_pos_weight", "<missing>"),
        None,
        checks,
    )
    check(
        stage,
        "fold_0/config.yaml.class_weighting_applied",
        fold.get("class_weighting_applied"),
        False,
        checks,
    )
    check(
        stage,
        "fold_0/config.yaml.class_weighting_pos_weight",
        fold.get("class_weighting_pos_weight", "<missing>"),
        None,
        checks,
    )
    assert len(checks) == 5


def test_training_completed_record_hashes_named_outputs(bench):
    _run(bench)
    record = _completed_record(bench, "runs/legacy/real_training")
    training = _run_dir(bench, "legacy") / "real" / "training"
    for name in (
        "config.yaml",
        "dataset_mappings.json",
        "split_plan.yaml",
        "cv_results.yaml",
        "fold_0/config.yaml",
        "fold_0/fold_info.yaml",
        "fold_0/best_model.pt",
        "fold_1/best_model.pt",
    ):
        assert record["outputs"][name]["sha256"] == sha256_file(training / name)
    assert record["outputs"]["selected_checkpoint"]["path"] == str(
        training / "fold_0" / "best_model.pt"
    )
    tree = record["outputs"]["output_tree"]
    assert tree["kind"] == "directory"
    assert tree["manifest_path"] == str(
        output_manifest_path(bench.execution, "runs/legacy/real_training")
    )
    assert (
        record["inputs"]["dataset"]["sha256"]
        == bench.plan["input_files"]["preprocessed_data"]["sha256"]
    )
    null_record = _completed_record(bench, "runs/legacy/null_training")
    assert (
        null_record["inputs"]["dataset"]["sha256"]
        == bench.plan["input_files"]["null_artifact"]["sha256"]
    )
    assert null_record["inputs"]["split_plan"] == record["inputs"]["split_plan"]


# --- explanation post-validation --------------------------------------------


def _explain_edit(mutate):
    def hook(argv):
        _edit_yaml(Path(_argv_value(argv, "--output-dir")) / "analysis_metadata.yaml", mutate)

    return hook


def _explain_remove(relative):
    def hook(argv):
        path = Path(_argv_value(argv, "--output-dir")) / relative
        if path.is_dir():
            shutil_rmtree(path)
        else:
            path.unlink()

    return hook


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


def _empty_per_sample(argv):
    per_sample = Path(_argv_value(argv, "--output-dir")) / "attributions_per_sample"
    for child in per_sample.iterdir():
        child.unlink()


def _lingering_tmp(argv):
    (Path(_argv_value(argv, "--output-dir")) / "_tmp_attributions").mkdir()


EXPLANATION_REJECTIONS = [
    ("real", _explain_remove("analysis_metadata.yaml"), "missing"),
    ("real", _explain_remove("attributions.npz"), "missing"),
    ("real", _explain_remove("sieve_variant_rankings.csv"), "missing"),
    ("real", _explain_remove("attributions_per_sample"), "attributions_per_sample"),
    ("real", _empty_per_sample, "attributions_per_sample"),
    ("real", _lingering_tmp, "_tmp_attributions"),
    (
        "real",
        _explain_edit(_set("dataset_provenance.sample_ids_sha256", "4" * 64)),
        "dataset_provenance",
    ),
    ("real", _explain_edit(_set("model_provenance.config_sha256", "5" * 64)), "config"),
    ("real", _explain_edit(_set("model_provenance.checkpoint_sha256", "6" * 64)), "checkpoint"),
    (
        "null",
        _explain_edit(_set("model_provenance.checkpoint_path", "/elsewhere/best_model.pt")),
        "checkpoint_path",
    ),
    ("real", _explain_edit(_set("model_provenance.selected_fold", 1)), "fold"),
    (
        "real",
        _explain_edit(_set("integrated_gradients.resolved_ig_mode", "legacy")),
        "resolved_ig_mode",
    ),
    ("real", _explain_edit(_set("integrated_gradients.n_steps", 7)), "n_steps"),
    ("real", _explain_edit(_set("n_integration_steps", 7)), "n_integration_steps"),
    ("real", _explain_edit(_set("max_variants_per_sample", 9)), "max_variants_per_sample"),
    ("real", _explain_edit(_set("aggregation_method", "max")), "aggregation_method"),
    ("real", _explain_edit(_set("is_null_baseline", True)), "is_null_baseline"),
    ("null", _explain_edit(_set("is_null_baseline", False)), "is_null_baseline"),
    ("real", _explain_edit(_set("n_samples", 7)), "n_samples"),
]


@pytest.mark.parametrize(("side", "hook", "message"), EXPLANATION_REJECTIONS)
def test_explanation_post_validation_rejects(bench, side, hook, message):
    stage_id = f"runs/legacy/{side}_explanation"
    bench.world.after[stage_id] = hook
    with pytest.raises(StagePostValidationError, match=message):
        _run(bench)
    assert _record_kinds(bench)[stage_id] == ["failed"]
    assert bench.world.stage_ids()[-1] == stage_id


def test_explanation_record_binds_training_authority(bench):
    _run(bench)
    record = _completed_record(bench, "runs/legacy/real_explanation")
    training = _completed_record(bench, "runs/legacy/real_training")
    assert record["inputs"]["training/config.yaml"] == training["outputs"]["config.yaml"]
    assert (
        record["inputs"]["training/selected_checkpoint"]["sha256"]
        == training["outputs"]["selected_checkpoint"]["sha256"]
    )
    assert [dep["stage_id"] for dep in record["dependencies"]] == [
        "runs/legacy/real_training",
        NULL_VALIDATION_STAGE_ID,
    ]
    for name in (
        "analysis_metadata.yaml",
        "attributions.npz",
        "sieve_variant_rankings.csv",
        "sieve_gene_rankings.csv",
        "output_tree",
    ):
        assert name in record["outputs"]


# --- pair validation and calibration ----------------------------------------


def test_pair_report_is_pure_and_written_before_calibration(bench):
    _run(bench)
    path = _run_dir(bench, "legacy") / "calibration" / "paired_compatibility.yaml"
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert report["compatible"] is True and report["run_id"] == "legacy"
    assert set(report) == {
        "schema_version",
        "run_id",
        "compatible",
        "null_binding",
        "shared",
        "real",
        "null",
        "checked_rules",
        "mismatches",
    }
    assert "validated_at" not in path.read_text(encoding="utf-8")
    pair = _completed_record(bench, "runs/legacy/pair_validation")
    for name in (
        "real/config.yaml",
        "null/config.yaml",
        "real/analysis_metadata.yaml",
        "null/analysis_metadata.yaml",
        "real/attributions.npz",
        "null/attributions.npz",
        "real/sieve_variant_rankings.csv",
        "real/checkpoint",
        "null/checkpoint",
        "null_validation.yaml",
        "resolved_plan",
    ):
        assert name in pair["inputs"]
    assert pair["outputs"]["paired_compatibility.yaml"]["sha256"] == sha256_file(path)
    calibration = _completed_record(bench, "runs/legacy/calibration")
    assert calibration["dependencies"][0]["stage_id"] == "runs/legacy/pair_validation"
    assert (
        calibration["inputs"]["paired_compatibility.yaml"]
        == pair["outputs"]["paired_compatibility.yaml"]
    )
    assert set(calibration["outputs"]) == {
        "bootstrap_calibrated_variant_rankings.csv",
        "bootstrap_calibrated_variant_rankings_gene_stats.csv",
        "bootstrap_calibrated_variant_rankings_summary.yaml",
    }


def test_pair_failure_prevents_any_calibration(bench):
    # A difference the per-side post-validators do not police but the 12C3B1
    # pair rules do (the full integrated_gradients block must be equal).
    bench.world.after["runs/legacy/null_explanation"] = _explain_edit(
        _set("integrated_gradients.sampling_seed", 99)
    )
    with pytest.raises(StagePostValidationError, match="incompatible"):
        _run(bench)
    kinds = _record_kinds(bench)
    assert kinds["runs/legacy/pair_validation"] == ["failed"]
    failed = load_stage_record(
        stage_record_path(bench.execution, "runs/legacy/pair_validation", "failed")
    )
    assert failed["failure"]["reason"] == "validator_failed"
    assert failed["execution"]["kind"] == "in_process"
    assert not any(stage_id.endswith("calibration") for stage_id in bench.world.stage_ids())


def _tamper_before_calibration(monkeypatch, tamper):
    original = position_benchmark_execution.calibration_input_gate

    def gate(ctx, stage):
        tamper(ctx, stage)
        return original(ctx, stage)

    monkeypatch.setattr(position_benchmark_execution, "calibration_input_gate", gate)


@pytest.mark.parametrize(
    "relative",
    [
        "real/explanation/sieve_variant_rankings.csv",
        "real/explanation/analysis_metadata.yaml",
        "null/explanation/attributions.npz",
        "calibration/paired_compatibility.yaml",
    ],
)
def test_calibration_hash_gate_rejects_changed_inputs(bench, monkeypatch, relative):
    _tamper_before_calibration(
        monkeypatch,
        lambda ctx, stage: _append(_run_dir(bench, stage.run_id) / relative, "\n# tampered\n"),
    )
    with pytest.raises(CalibrationInputGateError, match="changed after pair validation"):
        _run(bench)
    assert not any(stage_id.endswith("calibration") for stage_id in bench.world.stage_ids())
    assert _record_kinds(bench)["runs/legacy/calibration"] == []


def test_calibration_hash_gate_rejects_changed_pair_record(bench, monkeypatch):
    def tamper(ctx, stage):
        _append(
            stage_record_path(bench.execution, "runs/legacy/pair_validation", "completed"),
            "# edited\n",
        )

    _tamper_before_calibration(monkeypatch, tamper)
    with pytest.raises(CalibrationInputGateError, match="pair-validation record changed"):
        _run(bench)
    assert not any(stage_id.endswith("calibration") for stage_id in bench.world.stage_ids())


def _summary_edit(mutate):
    def hook(argv):
        _edit_yaml(Path(_argv_value(argv, "--output-summary")), mutate)

    return hook


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("n_bootstrap", 5), "n_bootstrap"),
        (_set("n_null_samples", 7), "n_null_samples"),
        (_set("excluded_sex_chroms", True), "excluded_sex_chroms"),
        (_set("per_gene.gene_delta_rank_aggregation", "mean"), "gene_delta_rank_aggregation"),
        (_set("genome_build", "GRCh38"), "genome_build"),
        (_set("n_real_variants", 0), "n_real_variants"),
        (_set("n_real_variants_missing_from_null", 1), "missing_from_null"),
    ],
)
def test_calibration_summary_mismatch_rejects(bench, mutate, message):
    bench.world.after["runs/legacy/calibration"] = _summary_edit(mutate)
    with pytest.raises(StagePostValidationError, match=message):
        _run(bench)
    assert _record_kinds(bench)["runs/legacy/calibration"] == ["failed"]


def test_calibration_outputs_must_not_preexist(bench):
    calibration = _run_dir(bench, "legacy") / "calibration"

    def plant(argv):
        calibration.mkdir(parents=True, exist_ok=True)
        (calibration / "bootstrap_calibrated_variant_rankings.csv").write_text("x", "utf-8")

    bench.world.after["runs/legacy/null_explanation"] = plant
    with pytest.raises(StageStateError, match="without a completed record"):
        _run(bench)
    assert not any(stage_id.endswith("calibration") for stage_id in bench.world.stage_ids())


# --- shared null ----------------------------------------------------------------


def test_two_pairs_share_one_null_and_final_stage_is_recorded(bench):
    _run(bench)
    report = yaml.safe_load(shared_null_validation_path(bench.root).read_text(encoding="utf-8"))
    identity = {
        key: bench.plan["null_binding"][key]
        for key in position_benchmark_execution.NULL_BINDING_IDENTITY_FIELDS
    }
    assert report == {"schema_version": 1, "run_ids": list(RUN_IDS), "null_binding": identity}
    record = _completed_record(bench, SHARED_NULL_STAGE_ID)
    assert [dep["stage_id"] for dep in record["dependencies"]] == [
        "runs/legacy/pair_validation",
        "runs/no_position/pair_validation",
        NULL_VALIDATION_STAGE_ID,
    ]
    assert "validated_at" not in shared_null_validation_path(bench.root).read_text("utf-8")


@pytest.mark.parametrize(
    "field_name",
    ["lineage_sha256", "null_artifact_sha256", "source_artifact_sha256", "sample_ids_sha256"],
)
def test_divergent_null_binding_rejects_before_second_calibration(bench, field_name):
    first_report = _run_dir(bench, "legacy") / "calibration" / "paired_compatibility.yaml"
    bench.world.before["runs/no_position/real_training"] = lambda argv: _edit_yaml(
        first_report, _set(f"null_binding.{field_name}", "7" * 64)
    )
    with pytest.raises(StagePostValidationError, match="different null bindings"):
        _run(bench)
    assert _record_kinds(bench)["runs/no_position/pair_validation"] == ["failed"]
    assert "runs/no_position/calibration" not in bench.world.stage_ids()


def test_shared_null_failure_blocks_comparisons(bench, monkeypatch):
    def diverged(ctx):
        raise PairCompatibilityError("strategy pairs reference different null bindings")

    monkeypatch.setattr(position_benchmark_execution, "_compute_shared_null_report", diverged)
    with pytest.raises(StagePostValidationError, match="different null bindings"):
        _run(bench)
    assert _record_kinds(bench)[SHARED_NULL_STAGE_ID] == ["failed"]
    assert not any(stage_id.startswith("comparisons/") for stage_id in bench.world.stage_ids())


# --- real-only comparisons ----------------------------------------------------


def test_raw_comparisons_are_real_only_exact_argv(bench):
    _run(bench)
    calls = dict(bench.world.calls)
    for name in ("performance", "raw_rankings", "raw_attributions"):
        argv = calls[f"comparisons/{name}"]
        assert argv == bench.plan["comparisons"][name]["argv"]
        for token in argv:
            assert "/null/" not in token and "/calibration" not in token
            assert "bootstrap_calibrated" not in token
    ranking = _completed_record(bench, "comparisons/raw_rankings")
    names = [fp["path"] for fp in ranking["inputs"].values()]
    assert any(path.endswith("real/explanation/sieve_variant_rankings.csv") for path in names)
    attribution = _completed_record(bench, "comparisons/raw_attributions")
    per_sample = [fp for fp in attribution["inputs"].values() if fp["kind"] == "directory"]
    assert per_sample and all(fp["path"].endswith("attributions_per_sample") for fp in per_sample)
    performance = _completed_record(bench, "comparisons/performance")
    assert [dep["stage_id"] for dep in performance["dependencies"]] == [
        "runs/legacy/real_training",
        "runs/no_position/real_training",
    ]
    assert compare_ablation_rankings.DEFERRED_CALIBRATED_SCORE_COLUMNS == {
        "delta_rank",
        "z_attribution",
        "p_rank_boot",
        "rank_real",
        "median_rank_null_boot",
        "corrected_rank",
    }


@pytest.mark.parametrize(
    "token",
    [
        "runs/legacy/null/explanation/attributions.npz",
        "runs/legacy/calibration/bootstrap_calibrated_variant_rankings.csv",
    ],
)
def test_real_only_guard_rejects_null_or_calibrated_paths(bench, token):
    stage = next(s for s in bench.stages if s.stage_id == "comparisons/raw_rankings")
    tainted = position_benchmark_execution.BenchmarkStage(
        stage.stage_id,
        stage.stage_type,
        stage.side,
        stage.run_id,
        (*stage.argv, str(bench.root / token)),
        None,
        stage.owned_paths,
        stage.dependency_ids,
    )
    with pytest.raises(StageStateError):
        require_real_only_comparison(bench.plan, tainted)


# --- directory manifests and logs -------------------------------------------


def test_output_manifests_and_logs_live_outside_scientific_trees(bench):
    _run(bench)
    for stage_id in (
        "runs/legacy/real_training",
        "runs/legacy/null_explanation",
        "comparisons/raw_attributions",
    ):
        record = _completed_record(bench, stage_id)
        tree = record["outputs"]["output_tree"]
        manifest = Path(tree["manifest_path"])
        assert manifest.is_file() and bench.execution / "manifests" in manifest.parents
        assert sha256_file(manifest) == tree["manifest_sha256"]
        recomputed = position_benchmark_execution.directory_manifest(tree["path"])
        assert recomputed["sha256"] == tree["manifest_sha256"]
        for entry in recomputed["entries"]:
            assert not entry["path"].endswith((".log", ".jsonl"))
        logs = record["execution"]["logs"]
        for fingerprint in logs.values():
            assert bench.execution / "logs" in Path(fingerprint["path"]).parents
            assert sha256_file(fingerprint["path"]) == fingerprint["sha256"]
    assert (bench.execution / "logs" / "runs" / "legacy" / "real_training.stdout.log").is_file()


# --- resume -------------------------------------------------------------------


def test_resume_after_complete_run_reuses_everything(bench):
    first = _run(bench)
    bench.world.calls.clear()
    second = _run(bench, resume=True)
    assert bench.world.calls == []
    assert second["reused"] == EXPECTED_ORDER and second["executed"] == []
    assert second["summary"] == first["summary"]


def test_without_resume_completed_state_rejects(bench):
    _run(bench)
    bench.world.calls.clear()
    with pytest.raises(StageStateError, match="--resume"):
        _run(bench)
    assert bench.world.calls == []


def _manual_recover(bench, stage_id: str, output: Path) -> None:
    """Simulate deliberate manual recovery of one stage (records, logs, manifest, output)."""
    import shutil

    for path in existing_stage_records(bench.execution, stage_id).values():
        path.unlink()
    for stream in ("stdout", "stderr"):
        (bench.execution / "logs" / f"{stage_id}.{stream}.log").unlink(missing_ok=True)
    output_manifest_path(bench.execution, stage_id).unlink(missing_ok=True)
    if output.is_dir():
        shutil.rmtree(output)


def test_resume_revalidates_prefix_and_executes_remaining_stages(bench):
    _run(bench)
    for name in ("raw_attributions", "raw_rankings"):
        _manual_recover(bench, f"comparisons/{name}", bench.root / "comparisons" / name)
    (bench.execution / "benchmark_execution_summary.yaml").unlink()
    bench.world.calls.clear()
    result = _run(bench, resume=True)
    assert bench.world.stage_ids() == ["comparisons/raw_rankings", "comparisons/raw_attributions"]
    assert result["reused"] == EXPECTED_ORDER[:-2]


@pytest.mark.parametrize("kind", ["failed", "running"])
def test_running_or_failed_record_always_rejects(bench, kind):
    bench.world.exit_codes["runs/legacy/real_explanation"] = 1
    with pytest.raises(StageFailure):
        _run(bench)
    stage_id = "runs/legacy/real_explanation"
    if kind == "running":
        failed = stage_record_path(bench.execution, stage_id, "failed")
        record = load_stage_record(failed)
        failed.unlink()
        running = build_stage_record(
            record_kind="running",
            **{
                key: record[key]
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
            },
            execution={**record["execution"], "logs": None, "exit_code": None},
        )
        position_benchmark_execution.write_stage_record(bench.execution, running)
    bench.world.calls.clear()
    for resume in (False, True):
        with pytest.raises(StageStateError, match=kind if resume else "--resume"):
            _run(bench, resume=resume)
    assert bench.world.calls == []


def _resume_rejects(bench, message, *, git=None):
    bench.world.calls.clear()
    with pytest.raises(
        (CompletedStageMismatchError, PlanAuthorityError, RepositoryGateError), match=message
    ):
        _run(bench, resume=True, git=git)
    assert bench.world.calls == []


def test_resume_rejects_changed_output_file(bench):
    _run(bench)
    _append(_run_dir(bench, "legacy") / "real" / "training" / "cv_results.yaml", "# x\n")
    _resume_rejects(bench, "output cv_results.yaml changed")


def test_resume_rejects_added_output(bench):
    _run(bench)
    (_run_dir(bench, "legacy") / "real" / "explanation" / "extra.txt").write_text("x", "utf-8")
    _resume_rejects(bench, r"added=\['extra.txt'\]")


def test_resume_rejects_removed_output(bench):
    _run(bench)
    (
        _run_dir(bench, "legacy")
        / "null"
        / "explanation"
        / "attributions_per_sample"
        / "sample_3.npz"
    ).unlink()
    _resume_rejects(bench, "removed=")


def test_resume_rejects_changed_stage_input(bench):
    # The paired compatibility report is an input of calibration and shared_null;
    # the pair stage that produced it detects the change first.
    _run(bench)
    _append(_run_dir(bench, "no_position") / "calibration" / "paired_compatibility.yaml", "#\n")
    _resume_rejects(bench, "runs/no_position/pair_validation cannot be resumed")


def test_resume_rejects_changed_planned_input_bytes(bench):
    _run(bench)
    _append(Path(bench.plan["input_files"]["split_plan"]["path"]), "# edited\n")
    _resume_rejects(bench, "differs from a fresh rebuild")


def _edit_record(bench, stage_id, mutate):
    path = stage_record_path(bench.execution, stage_id, "completed")
    _edit_yaml(path, mutate)


def test_resume_rejects_changed_argv_in_record(bench):
    _run(bench)
    _edit_record(
        bench,
        "runs/legacy/null_training",
        lambda record: record["execution"]["argv"].append("--extra"),
    )
    _resume_rejects(bench, "recorded argv differs")


def test_resume_rejects_repository_revision_change(bench):
    _run(bench)
    git = _clean_git(bench.plan, {("rev-parse", "HEAD"): (0, OTHER_REVISION + "\n", "")})
    _resume_rejects(bench, "does not match plan repository_revision", git=git)
    _edit_record(bench, "runs/legacy/real_training", _set("repository_revision", OTHER_REVISION))
    _resume_rejects(bench, "repository_revision")


def test_resume_rejects_changed_dependency_record(bench):
    _run(bench)
    # A benign-looking edit to the null-validation record still changes its
    # bytes, so every stage that bound its hash refuses to resume.
    _edit_record(bench, NULL_VALIDATION_STAGE_ID, _set("environment.hostname", "elsewhere"))
    _resume_rejects(bench, "dependency record benchmark/null_validation changed")


def test_resume_rejects_corrupted_record(bench):
    _run(bench)
    stage_record_path(bench.execution, "runs/legacy/calibration", "completed").write_text(
        "record_kind: completed\n", encoding="utf-8"
    )
    _resume_rejects(bench, "completed record is invalid")


def test_resume_rejects_when_post_validation_no_longer_passes(bench, monkeypatch):
    _run(bench)

    def now_fails(ctx, stage):
        raise StagePostValidationError("comparison outputs no longer acceptable")

    monkeypatch.setitem(position_benchmark_execution._POST_VALIDATORS, "comparison", now_fails)
    _resume_rejects(bench, "post-validation no longer passes")


def test_resume_rejects_tampered_persisted_manifest(bench):
    _run(bench)
    _append(output_manifest_path(bench.execution, "runs/legacy/real_explanation"), "{}\n")
    _resume_rejects(bench, "persisted output manifest changed")


def test_scan_rejects_records_after_an_incomplete_stage(bench):
    _run(bench)
    # Remove only the calibration record: later stages still hold records.
    _manual_recover(bench, "runs/no_position/calibration", Path("/nonexistent"))
    with pytest.raises(StageStateError, match="earlier stage"):
        scan_benchmark_state(bench.execution, bench.stages, resume=True)


def test_no_resume_rejects_leftover_output_without_record(bench):
    stray = _run_dir(bench, "legacy") / "real" / "training"
    stray.mkdir(parents=True)
    (stray / "config.yaml").write_text("stale: true\n", encoding="utf-8")
    with pytest.raises(StageStateError, match="without a completed record"):
        _run(bench)
    assert bench.world.calls == []


def test_execution_lock_is_held_for_the_whole_run(bench):
    seen = []

    def probe_lock(argv):
        lock = ExecutionLock(bench.root, resolved_plan_sha256="0" * 64)
        with pytest.raises(ExecutionLockError):
            lock.acquire()
        seen.append(True)

    bench.world.before["comparisons/raw_attributions"] = probe_lock
    _run(bench)
    assert seen == [True]
