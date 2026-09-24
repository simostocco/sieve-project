"""Tests for the Phase 12C3C provenance-gated calibrated positional ranking comparison.

A completed paired benchmark is produced ONCE per module by the Phase 12C3B2B
executor over the fake stage world of ``test_position_benchmark_execution``
(tiny strict-null fixture, fake train/explain/bootstrap, no GPU, no real
training). Every test then works on that benchmark and the fixture restores
the exact bytes of every file afterwards, so tests stay independent.

Most provenance failures are exercised through
``load_calibrated_position_benchmark``; a few go through the public
comparator CLI. Some tests deliberately *forge* a consistent record (rewrite
an artifact and its recorded fingerprint together): that isolates one
defect behind an otherwise-valid chain, and it also documents that unsigned
records prove consistency, not authorship.
"""

from __future__ import annotations

import copy
import csv
import os
import shutil
from pathlib import Path

import pytest
import yaml

from scripts import (
    compare_ablation_rankings,
    position_benchmark_calibrated,
    position_benchmark_execution,
    position_benchmark_manifest,
)
from scripts.position_benchmark_calibrated import (
    CalibratedProvenanceError,
    expected_calibration_dependency_ids,
    expected_pair_dependency_ids,
    expected_shared_null_dependency_ids,
    load_calibrated_position_benchmark,
    planned_benchmark_root,
    require_unchanged,
)
from scripts.position_benchmark_execution import (
    SHARED_NULL_STAGE_ID,
    benchmark_root_from_plan,
    build_benchmark_stages,
)
from scripts.position_benchmark_records import (
    file_fingerprint,
    sha256_file,
    stage_record_path,
)
from tests.test_position_benchmark_execution import (
    RUN_IDS,
    _append,
    _bench,
    _edit_yaml,
    _run,
    _set,
)

CALIBRATED_NAME = "bootstrap_calibrated_variant_rankings.csv"
SUMMARY_NAME = "bootstrap_calibrated_variant_rankings_summary.yaml"
GENE_STATS_NAME = "bootstrap_calibrated_variant_rankings_gene_stats.csv"


# ---------------------------------------------------------------------------
# One completed benchmark per module, restored byte-for-byte after each test
# ---------------------------------------------------------------------------


def _snapshot(root: Path) -> tuple[dict[str, bytes], set[str]]:
    files, directories = {}, set()
    for current, dirnames, filenames in os.walk(root):
        for name in dirnames:
            directories.add(os.path.join(current, name))
        for name in filenames:
            path = os.path.join(current, name)
            files[path] = Path(path).read_bytes()
    return files, directories


def _restore(root: Path, snapshot: tuple[dict[str, bytes], set[str]]) -> None:
    files, directories = snapshot
    for current, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            path = os.path.join(current, name)
            if path not in files:
                os.unlink(path)
        for name in dirnames:
            path = os.path.join(current, name)
            if os.path.islink(path):
                os.unlink(path)
            elif path not in directories:
                shutil.rmtree(path)
    for path, data in files.items():
        target = Path(path)
        if target.is_symlink():
            target.unlink()
        if not target.is_file() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


@pytest.fixture(scope="module")
def _completed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calibrated_benchmark")
    bench = _bench(tmp)
    _run(bench)
    return bench, tmp, _snapshot(tmp)


@pytest.fixture()
def bench(_completed):
    bench, tmp, snapshot = _completed
    yield bench
    _restore(tmp, snapshot)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_path(bench, stage_id: str, kind: str = "completed") -> Path:
    return stage_record_path(bench.execution, stage_id, kind)


def _record(bench, stage_id: str) -> dict:
    return yaml.safe_load(_record_path(bench, stage_id).read_text(encoding="utf-8"))


def _edit_record(bench, stage_id: str, mutate) -> None:
    _edit_yaml(_record_path(bench, stage_id), mutate)


def _calibration_dir(bench, run_id: str) -> Path:
    return bench.root / "runs" / run_id / "calibration"


def _refingerprint(bench, stage_id: str, section: str, name: str) -> None:
    """Forge a consistent record: set ``section[name]`` to the file's current fingerprint."""

    def mutate(record):
        record[section][name] = file_fingerprint(record[section][name]["path"])

    _edit_record(bench, stage_id, mutate)


def _forge_calibrated_rows(bench, run_id: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Rewrite a calibrated CSV and make summary + calibration record agree with it."""
    path = _calibration_dir(bench, run_id) / CALIBRATED_NAME
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _edit_yaml(_calibration_dir(bench, run_id) / SUMMARY_NAME, _set("n_real_variants", len(rows)))
    for name in (CALIBRATED_NAME, SUMMARY_NAME):
        _refingerprint(bench, f"runs/{run_id}/calibration", "outputs", name)


def _calibrated_rows(bench, run_id: str) -> tuple[list[dict], list[str]]:
    path = _calibration_dir(bench, run_id) / CALIBRATED_NAME
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _compare(bench, out_dir: Path, *extra: str, score: str | None = "delta_rank", root=None):
    argv = [
        "--comparison-axis",
        "position",
        "--position-calibrated-benchmark",
        str(root or bench.root),
    ]
    if score is not None:
        argv += ["--score-column", score]
    argv += [
        "--top-k",
        "1,2,3",
        "--high-rank-threshold",
        "2",
        "--low-rank-threshold",
        "3",
        "--out-comparison",
        str(out_dir / "comparison.yaml"),
        "--out-jaccard",
        str(out_dir / "jaccard.tsv"),
        "--out-level-specific",
        str(out_dir / "specific.tsv"),
        *extra,
    ]
    return compare_ablation_rankings.main(argv)


def _rejects(bench, message: str) -> None:
    with pytest.raises(CalibratedProvenanceError, match=message):
        load_calibrated_position_benchmark(bench.root)


def _read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _load_calibrated_runs(bench):
    """Load each planned run exactly as the calibrated comparator does."""
    benchmark = load_calibrated_position_benchmark(bench.root)
    return [
        compare_ablation_rankings._load_position_run(
            compare_ablation_rankings.PositionRunSpec(
                run_id=run.run_id,
                config_path=run.config_path,
                ranking_path=run.calibrated_ranking_path,
                analysis_metadata_path=run.analysis_metadata_path,
            ),
            score_column="delta_rank",
            resolve_score_column=compare_ablation_rankings._resolve_calibrated_position_score_column,
            sort_orders=compare_ablation_rankings.CALIBRATED_POSITION_SCORE_COLUMNS,
        )
        for run in benchmark.runs
    ]


# ---------------------------------------------------------------------------
# Mirrored layout constants stay locked to the executor and planner
# ---------------------------------------------------------------------------


def test_mirrored_constants_equal_executor_and_planner():
    calibrated = position_benchmark_calibrated
    execution = position_benchmark_execution
    manifest = position_benchmark_manifest
    for name in (
        "EXECUTION_DIRNAME",
        "BOUND_PLAN_NAME",
        "PLAN_BINDING_NAME",
        "PLAN_BINDING_SCHEMA_VERSION",
        "NULL_BINDING_DIRNAME",
        "SHARED_NULL_NAME",
        "NULL_VALIDATION_STAGE_ID",
        "SHARED_NULL_STAGE_ID",
        "NULL_VALIDATOR_NAME",
        "PAIR_VALIDATOR_NAME",
        "SHARED_NULL_VALIDATOR_NAME",
        "NULL_BINDING_IDENTITY_FIELDS",
    ):
        assert getattr(calibrated, name) == getattr(execution, name), name
    for name in (
        "PAIRED_SCHEMA_VERSION",
        "CALIBRATION_RANKINGS_NAME",
        "CALIBRATION_GENE_STATS_NAME",
        "CALIBRATION_SUMMARY_NAME",
        "PAIRED_COMPATIBILITY_NAME",
    ):
        assert getattr(calibrated, name) == getattr(manifest, name), name


@pytest.mark.parametrize("mode", ["cv", "single_split"])
def test_mirrored_dependency_ids_and_root_equal_executor_dag(tmp_path, mode):
    plan = _bench(tmp_path, mode=mode).plan
    stages = {stage.stage_id: stage for stage in build_benchmark_stages(plan)}
    run_ids = [run["run_id"] for run in plan["runs"]]
    for index, run_id in enumerate(run_ids):
        assert stages[f"runs/{run_id}/pair_validation"].dependency_ids == (
            expected_pair_dependency_ids(run_ids, index)
        )
        assert stages[f"runs/{run_id}/calibration"].dependency_ids == (
            expected_calibration_dependency_ids(run_id)
        )
    assert stages[SHARED_NULL_STAGE_ID].dependency_ids == expected_shared_null_dependency_ids(
        run_ids
    )
    assert planned_benchmark_root(plan) == benchmark_root_from_plan(plan)


def test_calibrated_module_does_not_import_training_stack():
    import subprocess
    import sys

    code = (
        "import sys; import scripts.position_benchmark_calibrated; "
        "print('torch' in sys.modules, 'scripts.position_benchmark_execution' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------------------
# Valid benchmark
# ---------------------------------------------------------------------------


def test_valid_completed_benchmark_passes_calibrated_comparison(bench, tmp_path):
    assert _compare(bench, tmp_path) == 0
    summary = yaml.safe_load((tmp_path / "comparison.yaml").read_text(encoding="utf-8"))

    assert summary["comparison_axis"] == "position"
    assert summary["comparison_mode"] == "calibrated"
    assert summary["score"] == {"column": "delta_rank", "sort_order": "descending"}
    shared = _record_path(bench, SHARED_NULL_STAGE_ID)
    report = bench.root / "null_binding" / "shared_null_validation.yaml"
    identity = {
        key: bench.plan["null_binding"][key]
        for key in position_benchmark_execution.NULL_BINDING_IDENTITY_FIELDS
    }
    assert summary["execution_provenance"] == {
        "benchmark_root": str(bench.root),
        "resolved_plan_path": str(bench.execution / "resolved_plan.yaml"),
        "resolved_plan_sha256": sha256_file(bench.plan_path),
        "plan_binding_path": str(bench.execution / "plan_binding.yaml"),
        "plan_binding_sha256": sha256_file(bench.execution / "plan_binding.yaml"),
        "repository_revision": bench.plan["repository_revision"],
        "manifest_file_sha256": bench.plan["manifest_file_sha256"],
        "shared_null_record_path": str(shared),
        "shared_null_record_sha256": sha256_file(shared),
        "shared_null_report_path": str(report),
        "shared_null_report_sha256": sha256_file(report),
        "null_binding": identity,
    }
    assert [run["run_id"] for run in summary["runs"]] == sorted(RUN_IDS)
    for run in summary["runs"]:
        run_id = run["run_id"]
        run_root = bench.root / "runs" / run_id
        calibration = _record_path(bench, f"runs/{run_id}/calibration")
        pair = _record_path(bench, f"runs/{run_id}/pair_validation")
        expected_files = {
            "config": run_root / "real" / "training" / "config.yaml",
            "analysis_metadata": run_root / "real" / "explanation" / "analysis_metadata.yaml",
            "calibrated_ranking": run_root / "calibration" / CALIBRATED_NAME,
            "calibration_summary": run_root / "calibration" / SUMMARY_NAME,
            "calibration_record": calibration,
            "pair_validation_record": pair,
            "paired_compatibility": run_root / "calibration" / "paired_compatibility.yaml",
        }
        for label, path in expected_files.items():
            assert run[f"{label}_path"] == str(path)
            assert run[f"{label}_sha256"] == sha256_file(path)
        assert run["n_variants"] == 6
        assert run["position_strategy_id"].startswith(run["position_strategy_name"])
        assert "ranking_path" not in run
    assert summary["variant_universe"]["n_variants"] == 6
    assert "timestamp" not in (tmp_path / "comparison.yaml").read_text(encoding="utf-8")


def test_executor_stage_output_equals_standalone_rerun_math(bench, tmp_path):
    """The executor-run stage and a standalone rerun agree (same defaults, same math)."""
    planned = bench.plan["comparisons"]["calibrated_rankings"]
    argv = list(planned["argv"][2:])
    for flag, name in (
        ("--out-comparison", "comparison.yaml"),
        ("--out-jaccard", "jaccard.tsv"),
        ("--out-level-specific", "specific.tsv"),
    ):
        argv[argv.index(flag) + 1] = str(tmp_path / name)
    assert compare_ablation_rankings.main(argv) == 0
    outputs = [Path(path) for path in planned["expected_outputs"]]
    assert (tmp_path / "jaccard.tsv").read_bytes() == outputs[1].read_bytes()
    assert (tmp_path / "specific.tsv").read_bytes() == outputs[2].read_bytes()
    executed = yaml.safe_load(outputs[0].read_text(encoding="utf-8"))
    rerun = yaml.safe_load((tmp_path / "comparison.yaml").read_text(encoding="utf-8"))
    # Only the shared-null/plan identities are equal; the comparison stage's own
    # completed record does not feed back into the comparison.
    assert executed == rerun


def test_jaccard_and_strategy_specific_equal_existing_pure_functions(bench, tmp_path):
    assert _compare(bench, tmp_path) == 0
    runs = _load_calibrated_runs(bench)
    matrices = compare_ablation_rankings.compute_position_jaccard_matrices(
        runs, [1, 2, 3], score_column="delta_rank", score_sort_order="descending"
    )
    expected_jaccard = [
        {key: str(value) for key, value in row.items()}
        for top_k in (1, 2, 3)
        for row in matrices[top_k]
    ]
    assert _read_tsv(tmp_path / "jaccard.tsv") == expected_jaccard
    specific = compare_ablation_rankings.find_strategy_specific_variants(runs, 2, 3)
    assert _read_tsv(tmp_path / "specific.tsv") == [
        {key: str(value) for key, value in row.items()} for row in specific
    ]
    summary = yaml.safe_load((tmp_path / "comparison.yaml").read_text(encoding="utf-8"))
    assert summary["jaccard_matrices"] == {f"top_{k}": matrices[k] for k in (1, 2, 3)}
    # The fake strategies rank differently, so the comparison is not trivial.
    assert any(row["jaccard"] < 1.0 for row in matrices[1] + matrices[2])


def test_calibrated_rankings_sort_descending_by_delta_rank(bench):
    for run in _load_calibrated_runs(bench):
        scores = [row["score"] for row in run.rankings]
        assert scores == sorted(scores, reverse=True)
        assert [row["rank"] for row in run.rankings] == list(range(1, len(scores) + 1))
        assert run.rankings[0]["score"] > 0


def test_calibrated_resolver_ties_break_by_variant_id(tmp_path):
    path = tmp_path / "calibrated.csv"
    path.write_text("variant_id,delta_rank\nv3,1.5\nv1,1.5\nv2,4\nv0,-2\n", encoding="utf-8")
    rows, column, order = compare_ablation_rankings.load_position_rankings(
        path,
        run_id="r",
        score_column="delta_rank",
        resolve_score_column=compare_ablation_rankings._resolve_calibrated_position_score_column,
        sort_orders=compare_ablation_rankings.CALIBRATED_POSITION_SCORE_COLUMNS,
    )
    assert (column, order) == ("delta_rank", "descending")
    assert [row["variant_id"] for row in rows] == ["v2", "v1", "v3", "v0"]


@pytest.mark.parametrize(
    "score",
    [
        None,
        "mean_attribution",
        "rank",
        "max_attribution",
        "z_attribution",
        "p_rank_boot",
        "fdr_rank_boot",
        "rank_real",
        "median_rank_null_boot",
        "iqr_rank_null_boot",
        "at_resolution_floor",
        "DELTA_RANK",
    ],
)
def test_calibrated_mode_accepts_only_delta_rank(bench, tmp_path, capsys, score):
    assert _compare(bench, tmp_path, score=score) == 1
    assert "delta_rank" in capsys.readouterr().err
    assert not (tmp_path / "comparison.yaml").exists()


def test_calibrated_mode_rejects_position_run(bench, tmp_path, capsys):
    run_root = bench.root / "runs" / "legacy"
    extra = [
        "--position-run",
        "legacy",
        str(run_root / "real" / "training" / "config.yaml"),
        str(run_root / "calibration" / CALIBRATED_NAME),
        str(run_root / "real" / "explanation" / "analysis_metadata.yaml"),
    ]
    assert _compare(bench, tmp_path, *extra) == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_calibrated_mode_rejects_level_inputs(bench, tmp_path, capsys):
    assert _compare(bench, tmp_path, "--ranking-dir", str(tmp_path)) == 1
    assert "rejects --ranking-dir" in capsys.readouterr().err


def test_at_least_two_runs_required(bench, tmp_path, capsys, monkeypatch):
    benchmark = load_calibrated_position_benchmark(bench.root)
    single = copy.copy(benchmark)
    object.__setattr__(single, "runs", benchmark.runs[:1])
    monkeypatch.setattr(
        position_benchmark_calibrated,
        "load_calibrated_position_benchmark",
        lambda root: single,
    )
    assert _compare(bench, tmp_path) == 1
    assert "at least two planned runs" in capsys.readouterr().err


def test_run_set_comes_from_plan_not_directories(bench, tmp_path):
    # A stray, well-formed calibration for an unplanned run is ignored.
    stray = bench.root / "runs" / "stray" / "calibration"
    shutil.copytree(_calibration_dir(bench, "legacy"), stray)
    stray_record = _record_path(bench, "runs/stray/calibration")
    stray_record.parent.mkdir(parents=True)
    shutil.copy(_record_path(bench, "runs/legacy/calibration"), stray_record)

    benchmark = load_calibrated_position_benchmark(bench.root)
    assert [run.run_id for run in benchmark.runs] == list(RUN_IDS)
    assert _compare(bench, tmp_path) == 0
    summary = yaml.safe_load((tmp_path / "comparison.yaml").read_text(encoding="utf-8"))
    assert [run["run_id"] for run in summary["runs"]] == sorted(RUN_IDS)


def test_universe_mismatch_rejects(bench, tmp_path, capsys):
    rows, fieldnames = _calibrated_rows(bench, "no_position")
    _forge_calibrated_rows(bench, "no_position", rows[:-1], fieldnames)
    load_calibrated_position_benchmark(bench.root)  # provenance is consistent
    assert _compare(bench, tmp_path) == 1
    assert "variant universe mismatch" in capsys.readouterr().err


def test_duplicate_variant_ids_reject(bench, tmp_path, capsys):
    rows, fieldnames = _calibrated_rows(bench, "legacy")
    _forge_calibrated_rows(bench, "legacy", [*rows, dict(rows[0])], fieldnames)
    assert _compare(bench, tmp_path) == 1
    assert "duplicate variant_id" in capsys.readouterr().err


def test_missing_delta_rank_column_rejects(bench, tmp_path, capsys):
    rows, fieldnames = _calibrated_rows(bench, "legacy")
    kept = [name for name in fieldnames if name != "delta_rank"]
    _forge_calibrated_rows(bench, "legacy", [{key: row[key] for key in kept} for row in rows], kept)
    assert _compare(bench, tmp_path) == 1
    assert "no 'delta_rank' column" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("abc", "non-numeric"),
        ("", "empty"),
        ("nan", "non-finite"),
        ("inf", "non-finite"),
        ("-inf", "non-finite"),
    ],
)
def test_invalid_delta_rank_values_reject(bench, tmp_path, capsys, value, message):
    rows, fieldnames = _calibrated_rows(bench, "legacy")
    rows[2]["delta_rank"] = value
    _forge_calibrated_rows(bench, "legacy", rows, fieldnames)
    assert _compare(bench, tmp_path) == 1
    assert message in capsys.readouterr().err


def test_file_changed_during_comparison_is_detected(bench):
    benchmark = load_calibrated_position_benchmark(bench.root)
    require_unchanged(benchmark)
    _append(_calibration_dir(bench, "legacy") / CALIBRATED_NAME, "\n")
    with pytest.raises(CalibratedProvenanceError, match="changed during calibrated comparison"):
        require_unchanged(benchmark)


# ---------------------------------------------------------------------------
# Plan authority
# ---------------------------------------------------------------------------


def test_missing_bound_plan_rejects(bench):
    (bench.execution / "resolved_plan.yaml").unlink()
    _rejects(bench, "does not exist")


def test_changed_bound_plan_bytes_reject(bench):
    _append(bench.execution / "resolved_plan.yaml", "# edited\n")
    _rejects(bench, "plan binding resolved_plan_sha256")


def test_wrong_plan_binding_sha_rejects(bench):
    _edit_yaml(bench.execution / "plan_binding.yaml", _set("resolved_plan_sha256", "0" * 64))
    _rejects(bench, "plan binding resolved_plan_sha256")


@pytest.mark.parametrize("field", ["repository_revision", "manifest_file_sha256"])
def test_wrong_plan_binding_identity_rejects(bench, field):
    _edit_yaml(bench.execution / "plan_binding.yaml", _set(field, "f" * 40))
    _rejects(bench, f"plan binding {field}")


def test_benchmark_root_mismatch_rejects(bench, tmp_path):
    other = tmp_path / "copy"
    shutil.copytree(bench.execution, other / "execution")
    with pytest.raises(CalibratedProvenanceError, match="is not the bound plan's benchmark root"):
        load_calibrated_position_benchmark(other)


def test_execution_directory_is_not_accepted_as_root(bench):
    with pytest.raises(CalibratedProvenanceError):
        load_calibrated_position_benchmark(bench.execution)


def test_extra_planned_run_without_records_rejects(bench):
    # Forge a consistent plan + binding that plans one more strategy.
    plan_path = bench.execution / "resolved_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    extra = copy.deepcopy(plan["runs"][0])
    old_root = extra["directories"]["run_root"]
    new_root = str(Path(old_root).parent / "extra")

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value.replace(old_root, new_root) if isinstance(value, str) else value

    extra = relocate(extra)
    extra["run_id"] = "extra"
    plan["runs"].append(extra)
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    _edit_yaml(
        bench.execution / "plan_binding.yaml", _set("resolved_plan_sha256", sha256_file(plan_path))
    )
    # Every record carries the old plan hash, so the first record already refuses.
    _rejects(bench, "resolved_plan_sha256")


def test_missing_planned_run_calibration_rejects(bench):
    _record_path(bench, "runs/no_position/calibration").unlink()
    _rejects(bench, "runs/no_position/calibration has no completed record")


# ---------------------------------------------------------------------------
# Shared-null chain
# ---------------------------------------------------------------------------


def test_missing_shared_null_record_rejects(bench):
    _record_path(bench, SHARED_NULL_STAGE_ID).unlink()
    _rejects(bench, "benchmark/shared_null has no completed record")


@pytest.mark.parametrize("kind", ["running", "failed"])
def test_shared_null_sibling_record_rejects(bench, kind):
    shutil.copy(
        _record_path(bench, SHARED_NULL_STAGE_ID), _record_path(bench, SHARED_NULL_STAGE_ID, kind)
    )
    _rejects(bench, f"has a {kind} record")


def test_corrupt_shared_null_record_rejects(bench):
    _record_path(bench, SHARED_NULL_STAGE_ID).write_text("record_kind: completed\n", "utf-8")
    _rejects(bench, "keys are invalid")


def test_incomplete_shared_null_dependency_set_rejects(bench):
    _edit_record(bench, SHARED_NULL_STAGE_ID, lambda r: r["dependencies"].pop(0))
    _rejects(bench, "differ from the planned dependencies")


def test_extra_shared_null_dependency_rejects(bench):
    calibration = {
        "stage_id": "runs/legacy/calibration",
        "record_path": str(_record_path(bench, "runs/legacy/calibration")),
        "record_sha256": sha256_file(_record_path(bench, "runs/legacy/calibration")),
    }
    _edit_record(bench, SHARED_NULL_STAGE_ID, lambda r: r["dependencies"].append(calibration))
    _rejects(bench, "differ from the planned dependencies")


def test_shared_null_dependency_hash_mismatch_rejects(bench):
    _edit_record(
        bench, SHARED_NULL_STAGE_ID, lambda r: r["dependencies"][1].update(record_sha256="0" * 64)
    )
    _rejects(bench, "dependency record runs/no_position/pair_validation changed")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("execution.callable", "somewhere.else"), "was not produced by"),
        (_set("stage_type", "comparison"), "stage_type"),
        (_set("resolved_plan_sha256", "0" * 64), "resolved_plan_sha256"),
    ],
)
def test_changed_shared_null_record_rejects(bench, mutate, message):
    _edit_record(bench, SHARED_NULL_STAGE_ID, mutate)
    _rejects(bench, message)


def test_changed_shared_null_record_output_fingerprint_rejects(bench):
    def mutate(record):
        record["outputs"]["shared_null_validation.yaml"]["sha256"] = "0" * 64

    _edit_record(bench, SHARED_NULL_STAGE_ID, mutate)
    _rejects(bench, "changed since its stage completed")


def test_changed_shared_null_report_rejects(bench):
    _append(bench.root / "null_binding" / "shared_null_validation.yaml", "# edited\n")
    _rejects(bench, "shared_null_validation.yaml changed since its stage completed")


def test_wrong_shared_null_binding_rejects(bench):
    report = bench.root / "null_binding" / "shared_null_validation.yaml"
    _edit_yaml(report, _set("null_binding.lineage_sha256", "7" * 64))
    _refingerprint(bench, SHARED_NULL_STAGE_ID, "outputs", "shared_null_validation.yaml")
    _rejects(bench, "does not equal require_shared_null_across_pairs")


def test_wrong_shared_null_run_order_rejects(bench):
    report = bench.root / "null_binding" / "shared_null_validation.yaml"
    _edit_yaml(report, lambda data: data["run_ids"].reverse())
    _refingerprint(bench, SHARED_NULL_STAGE_ID, "outputs", "shared_null_validation.yaml")
    _rejects(bench, "does not equal require_shared_null_across_pairs")


# ---------------------------------------------------------------------------
# Pair-validation chain
# ---------------------------------------------------------------------------


def test_missing_pair_validation_record_rejects(bench):
    _record_path(bench, "runs/legacy/pair_validation").unlink()
    _rejects(bench, "runs/legacy/pair_validation has no completed record")


def test_changed_pair_validation_record_rejects(bench):
    # Even a cosmetic edit changes the bytes that calibration and shared_null bound.
    _edit_record(bench, "runs/legacy/pair_validation", _set("environment.hostname", "elsewhere"))
    _rejects(bench, "dependency record runs/legacy/pair_validation changed")


def test_pair_validation_wrong_callable_rejects(bench):
    _edit_record(bench, "runs/legacy/pair_validation", _set("execution.callable", "x.y"))
    _rejects(bench, "was not produced by")


def test_incompatible_paired_report_rejects(bench):
    report = _calibration_dir(bench, "legacy") / "paired_compatibility.yaml"
    _edit_yaml(report, _set("compatible", False))
    _refingerprint(bench, "runs/legacy/pair_validation", "outputs", "paired_compatibility.yaml")
    _rejects(bench, "paired compatibility report is not passing")


def test_paired_report_wrong_null_binding_rejects(bench):
    report = _calibration_dir(bench, "legacy") / "paired_compatibility.yaml"
    _edit_yaml(report, _set("null_binding.n_samples", 9))
    _refingerprint(bench, "runs/legacy/pair_validation", "outputs", "paired_compatibility.yaml")
    _rejects(bench, "null binding differs from plan null_binding")


def test_changed_paired_compatibility_rejects(bench):
    _append(_calibration_dir(bench, "no_position") / "paired_compatibility.yaml", "# x\n")
    _rejects(bench, "paired_compatibility.yaml changed since its stage completed")


# ---------------------------------------------------------------------------
# Calibration chain
# ---------------------------------------------------------------------------


def test_missing_calibration_record_rejects(bench):
    _record_path(bench, "runs/legacy/calibration").unlink()
    _rejects(bench, "runs/legacy/calibration has no completed record")


@pytest.mark.parametrize("kind", ["running", "failed"])
def test_calibration_sibling_record_rejects(bench, kind):
    stage_id = "runs/legacy/calibration"
    shutil.copy(_record_path(bench, stage_id), _record_path(bench, stage_id, kind))
    _rejects(bench, f"has a {kind} record")


def test_changed_calibration_record_rejects(bench):
    def mutate(record):
        record["inputs"]["null/attributions.npz"]["sha256"] = "0" * 64

    _edit_record(bench, "runs/legacy/calibration", mutate)
    _rejects(bench, "input null/attributions.npz is not the pair-validated file")


def test_calibration_paired_input_not_pair_output_rejects(bench):
    def mutate(record):
        record["inputs"]["paired_compatibility.yaml"]["size"] += 1

    _edit_record(bench, "runs/legacy/calibration", mutate)
    _rejects(bench, "is not the pair validation output")


def test_wrong_calibration_argv_rejects(bench):
    _edit_record(bench, "runs/legacy/calibration", lambda r: r["execution"]["argv"].append("--x"))
    _rejects(bench, "recorded argv differs from the planned calibration argv")


def test_wrong_calibration_cwd_rejects(bench):
    _edit_record(bench, "runs/legacy/calibration", _set("execution.cwd", "/elsewhere"))
    _rejects(bench, "is not the plan repository_root")


def test_wrong_calibration_dependency_order_rejects(bench):
    _edit_record(bench, "runs/legacy/calibration", lambda r: r["dependencies"].reverse())
    _rejects(bench, "differ from the planned dependencies")


def test_wrong_calibration_dependency_hash_rejects(bench):
    _edit_record(
        bench,
        "runs/legacy/calibration",
        lambda r: r["dependencies"][1].update(record_sha256="0" * 64),
    )
    _rejects(bench, "dependency record benchmark/null_validation changed")


def test_wrong_calibrated_output_path_rejects(bench):
    gene_stats = _calibration_dir(bench, "legacy") / GENE_STATS_NAME

    def mutate(record):
        record["outputs"][CALIBRATED_NAME] = file_fingerprint(gene_stats)

    _edit_record(bench, "runs/legacy/calibration", mutate)
    _rejects(bench, f"output {CALIBRATED_NAME} is not the planned path")


def test_external_calibrated_csv_is_never_accepted(bench, tmp_path):
    external = tmp_path / CALIBRATED_NAME
    shutil.copy(_calibration_dir(bench, "legacy") / CALIBRATED_NAME, external)

    def mutate(record):
        record["outputs"][CALIBRATED_NAME] = file_fingerprint(external)

    _edit_record(bench, "runs/legacy/calibration", mutate)
    _rejects(bench, "is not the planned path")


def test_changed_calibrated_csv_rejects(bench):
    _append(_calibration_dir(bench, "legacy") / CALIBRATED_NAME, "1,999,GENE1,0\n")
    _rejects(bench, f"{CALIBRATED_NAME} changed since its stage completed")


def test_changed_calibration_summary_rejects(bench):
    _append(_calibration_dir(bench, "legacy") / SUMMARY_NAME, "# x\n")
    _rejects(bench, f"{SUMMARY_NAME} changed since its stage completed")


def test_changed_gene_stats_rejects(bench):
    _append(_calibration_dir(bench, "legacy") / GENE_STATS_NAME, "GENE2\n")
    _rejects(bench, f"{GENE_STATS_NAME} changed since its stage completed")


def test_wrong_config_bytes_reject(bench):
    _append(bench.root / "runs" / "legacy" / "real" / "training" / "config.yaml", "# x\n")
    _rejects(bench, "real/config.yaml changed since its stage completed")


def test_wrong_analysis_metadata_bytes_reject(bench):
    _append(
        bench.root / "runs" / "legacy" / "real" / "explanation" / "analysis_metadata.yaml", "# x\n"
    )
    _rejects(bench, "real/analysis_metadata.yaml changed since its stage completed")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("n_bootstrap", 5), "n_bootstrap"),
        (_set("n_null_samples", 7), "n_null_samples"),
        (_set("genome_build", "GRCh38"), "genome_build"),
        (_set("excluded_sex_chroms", True), "excluded_sex_chroms"),
        (
            _set("per_gene.gene_delta_rank_aggregation", "mean"),
            "per_gene.gene_delta_rank_aggregation",
        ),
        (_set("n_real_variants_missing_from_null", 1), "n_real_variants_missing_from_null"),
        (_set("n_real_variants", 5), "n_real_variants"),
        (_set("n_bootstrap", True), "n_bootstrap"),
    ],
)
def test_calibration_summary_mismatch_rejects(bench, mutate, message):
    _edit_yaml(_calibration_dir(bench, "legacy") / SUMMARY_NAME, mutate)
    _refingerprint(bench, "runs/legacy/calibration", "outputs", SUMMARY_NAME)
    _rejects(bench, f"calibration summary {message}")


def test_provenance_failure_through_cli_is_concise(bench, tmp_path, capsys):
    _append(_calibration_dir(bench, "legacy") / CALIBRATED_NAME, "\n")
    assert _compare(bench, tmp_path) == 1
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and "Traceback" not in err
    assert not (tmp_path / "comparison.yaml").exists()
