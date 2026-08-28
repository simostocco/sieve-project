"""Tests for Phase 12C2B dry-run command orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_position_benchmark
from scripts.position_benchmark_manifest import (
    POSITION_SCORE_COLUMN,
    BenchmarkManifestError,
    build_resolved_plan,
)
from tests.test_position_benchmark_manifest import _base_manifest, _write_yaml


def _argv_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _malformed_yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "bad.yaml"
    path.write_text("manifest: [", encoding="utf-8")
    return path


def test_train_argv_is_deterministic_and_explicit_for_legacy(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    first = build_resolved_plan(
        manifest_path, python_override=sys.executable, device_override="cpu"
    )
    second = build_resolved_plan(
        manifest_path, python_override=sys.executable, device_override="cpu"
    )

    assert first["runs"][0]["train_argv"] == second["runs"][0]["train_argv"]
    argv = first["runs"][0]["train_argv"]
    assert "--position-preset" in argv
    assert _argv_after(argv, "--position-preset") == "legacy"
    assert _argv_after(argv, "--level") == "L3"
    assert _argv_after(argv, "--val-split") == "0.2"
    assert _argv_after(argv, "--cv") == "2"
    assert _argv_after(argv, "--split-plan") == str(tmp_path / "splits" / "split_plan.yaml")
    assert "--gradient-clip" not in argv


def test_train_argv_maps_output_dir_and_experiment_name_without_extra_training_nesting(tmp_path):
    plan = build_resolved_plan(
        _base_manifest(tmp_path)[0], python_override=sys.executable, device_override="cpu"
    )
    argv = plan["runs"][0]["train_argv"]
    run_root = Path(plan["runs"][0]["directories"]["run_root"])

    assert _argv_after(argv, "--output-dir") == str(run_root / "real")
    assert _argv_after(argv, "--experiment-name") == "training"
    assert Path(_argv_after(argv, "--output-dir")) / _argv_after(argv, "--experiment-name") == Path(
        plan["runs"][0]["directories"]["training"]
    )


def test_gradient_clip_and_covariate_args_are_emitted_when_present(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    sex_map = tmp_path / "sex.tsv"
    pc_map = tmp_path / "pcs.tsv"
    sex_map.write_text("sample_id\tsex\nS1\tXX\n", encoding="utf-8")
    pc_map.write_text("sample_id\tPC1\nS1\t0.1\n", encoding="utf-8")
    manifest["training"]["sex_map"] = "sex.tsv"
    manifest["training"]["pc_map"] = "pcs.tsv"
    manifest["training"]["num_pcs"] = 1
    manifest["training"]["gradient_clip"] = 1.5
    _write_yaml(manifest_path, manifest)

    argv = build_resolved_plan(
        manifest_path, python_override=sys.executable, device_override="cpu"
    )["runs"][0]["train_argv"]

    assert _argv_after(argv, "--gradient-clip") == "1.5"
    assert _argv_after(argv, "--sex-map") == str(sex_map)
    assert _argv_after(argv, "--pc-map") == str(pc_map)
    assert _argv_after(argv, "--num-pcs") == "1"


@pytest.mark.parametrize(
    ("run_index", "expected_flags"),
    [
        (1, ["--absolute-position-encoding", "none", "--relative-position-encoding", "none"]),
    ],
)
def test_no_position_exact_flags(tmp_path, run_index, expected_flags):
    plan = build_resolved_plan(
        _base_manifest(tmp_path)[0], python_override=sys.executable, device_override="cpu"
    )
    argv = plan["runs"][run_index]["train_argv"]

    for expected in expected_flags:
        assert expected in argv


@pytest.mark.parametrize(
    ("run_id", "position", "expected"),
    [
        (
            "sinusoidal",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "sinusoidal",
                "relative_position_encoding": "none",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "position_dim": 64,
                "sinusoidal_coordinate_scale": 1.0,
                "sinusoidal_max_wavelength": 10000.0,
            },
            [
                "--position-dim",
                "64",
                "--sinusoidal-coordinate-scale",
                "1.0",
                "--sinusoidal-max-wavelength",
                "10000.0",
            ],
        ),
        (
            "learned_binned",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "learned_binned",
                "relative_position_encoding": "none",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "position_dim": 64,
                "position_bin_size": 10000,
            },
            ["--absolute-position-encoding", "learned_binned", "--position-bin-size", "10000"],
        ),
        (
            "t5",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "none",
                "relative_position_encoding": "t5_bucket",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "num_position_buckets": 32,
                "max_position_distance": 100000,
            },
            ["--num-position-buckets", "32", "--max-position-distance", "100000"],
        ),
        (
            "rope",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "none",
                "relative_position_encoding": "rope",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "rope_coordinate_scale": 10000.0,
                "rope_base": 10000.0,
            },
            ["--rope-coordinate-scale", "10000.0", "--rope-base", "10000.0"],
        ),
        (
            "alibi_fixed",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "none",
                "relative_position_encoding": "alibi_fixed",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "alibi_distance_function": "log1p",
                "alibi_distance_scale": 10000.0,
            },
            ["--alibi-distance-function", "log1p", "--alibi-distance-scale", "10000.0"],
        ),
        (
            "alibi_learned",
            {
                "position_preset": "custom",
                "absolute_position_encoding": "none",
                "relative_position_encoding": "alibi_learned",
                "chromosome_encoding": "none",
                "cross_chromosome_policy": "separate",
                "alibi_distance_function": "log1p",
                "alibi_distance_scale": 10000.0,
            },
            ["--relative-position-encoding", "alibi_learned", "--alibi-distance-scale", "10000.0"],
        ),
    ],
)
def test_primary_strategy_position_flags_round_trip(tmp_path, run_id, position, expected):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"] = [
        {"run_id": "legacy", "position": {"position_preset": "legacy"}},
        {"run_id": run_id, "position": position},
    ]
    _write_yaml(manifest_path, manifest)

    argv = build_resolved_plan(
        manifest_path, python_override=sys.executable, device_override="cpu"
    )["runs"][1]["train_argv"]

    for token in expected:
        assert token in argv
    assert "--position-preset" in argv
    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")
    assert all("position_strategy_id" not in run for run in plan["runs"])


def test_explain_cv_uses_parent_experiment_and_fold(tmp_path):
    plan = build_resolved_plan(
        _base_manifest(tmp_path)[0], python_override=sys.executable, device_override="cpu"
    )
    argv = plan["runs"][0]["explain_argv"]

    assert _argv_after(argv, "--experiment-dir") == plan["runs"][0]["directories"]["training"]
    assert _argv_after(argv, "--fold-index") == "0"
    assert "--ig-mode" in argv
    assert _argv_after(argv, "--ig-mode") == "content"
    assert "--is-null-baseline" not in argv


def test_explain_single_split_omits_fold_index(tmp_path):
    plan = build_resolved_plan(
        _base_manifest(tmp_path, mode="single_split")[0],
        python_override=sys.executable,
        device_override="cpu",
    )

    assert "--fold-index" not in plan["runs"][0]["explain_argv"]


def test_b1_b2_b3_commands_preserve_manifest_order_and_output_paths(tmp_path):
    plan = build_resolved_plan(
        _base_manifest(tmp_path)[0], python_override=sys.executable, device_override="cpu"
    )
    run_ids = [run["run_id"] for run in plan["runs"]]
    comparisons = plan["comparisons"]

    assert run_ids == ["legacy", "no_position"]
    b1 = comparisons["performance"]["argv"]
    assert "--comparison-axis" in b1
    assert _argv_after(b1, "--comparison-axis") == "position"
    assert [b1[index + 1] for index, token in enumerate(b1) if token == "--run-dir"] == [
        run["directories"]["training"] for run in plan["runs"]
    ]
    b2 = comparisons["raw_rankings"]["argv"]
    assert _argv_after(b2, "--score-column") == POSITION_SCORE_COLUMN
    assert "delta_rank" not in b2
    assert "position_ranking_jaccard.tsv" in b2[-3]
    b3 = comparisons["raw_attributions"]["argv"]
    assert b3.count("--position-run") == 2
    assert (
        "position_attribution_summary.tsv" in comparisons["raw_attributions"]["expected_outputs"][0]
    )


def test_output_safety_missing_and_empty_leaf_allowed(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    empty_training = (
        tmp_path / "outputs" / "posenc_l3_primary" / "L3" / "runs" / "legacy" / "real" / "training"
    )
    empty_training.mkdir(parents=True)

    build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_output_safety_non_empty_leaf_fails_or_warns(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    training = (
        tmp_path / "outputs" / "posenc_l3_primary" / "L3" / "runs" / "legacy" / "real" / "training"
    )
    training.mkdir(parents=True)
    (training / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="non-empty"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")

    plan = build_resolved_plan(
        manifest_path,
        python_override=sys.executable,
        device_override="cpu",
        allow_existing_outputs=True,
    )
    assert any("non-empty" in warning for warning in plan["warnings"])


def test_file_where_output_directory_expected_fails(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    training = (
        tmp_path / "outputs" / "posenc_l3_primary" / "L3" / "runs" / "legacy" / "real" / "training"
    )
    training.parent.mkdir(parents=True)
    training.write_text("not a dir", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="existing file"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_dry_run_cli_requires_dry_run_and_prints_without_creating_outputs(tmp_path, capsys):
    manifest_path, _ = _base_manifest(tmp_path)

    assert run_position_benchmark.main([str(manifest_path)]) == 2
    assert (
        run_position_benchmark.main(
            [str(manifest_path), "--dry-run", "--python", sys.executable, "--device", "cpu"]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "POSITION BENCHMARK DRY RUN" in output
    assert "NULL EXECUTION: deferred to Phase 12C3" in output
    assert not (tmp_path / "outputs").exists()


def test_out_plan_writes_one_yaml_only_and_rejects_existing(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    out_plan = tmp_path / "plans" / "resolved.yaml"

    assert (
        run_position_benchmark.main(
            [
                str(manifest_path),
                "--dry-run",
                "--out-plan",
                str(out_plan),
                "--python",
                sys.executable,
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    data = yaml.safe_load(out_plan.read_text(encoding="utf-8"))
    assert isinstance(data["runs"][0]["train_argv"], list)
    assert not (tmp_path / "outputs").exists()
    assert (
        run_position_benchmark.main(
            [
                str(manifest_path),
                "--dry-run",
                "--out-plan",
                str(out_plan),
                "--python",
                sys.executable,
                "--device",
                "cpu",
            ]
        )
        == 2
    )


def test_runtime_interpreter_and_device_override_hierarchy(tmp_path):
    plan = build_resolved_plan(
        _base_manifest(tmp_path)[0], python_override=sys.executable, device_override="cpu"
    )

    assert Path(plan["runtime"]["python"]).is_absolute()
    assert plan["runtime"]["device"] == "cpu"


def test_manifest_relative_python_path_resolves_against_manifest_parent(tmp_path, monkeypatch):
    manifest_path, manifest = _base_manifest(tmp_path)
    python_path = tmp_path / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    manifest["runtime"]["python"] = "./venv/bin/python"
    _write_yaml(manifest_path, manifest)
    monkeypatch.chdir("/")

    plan = build_resolved_plan(manifest_path, device_override="cpu")

    assert plan["runtime"]["python"] == str(python_path)


def test_cli_relative_python_override_is_invocation_relative(tmp_path, monkeypatch):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runtime"]["python"] = sys.executable
    _write_yaml(manifest_path, manifest)
    invocation_dir = tmp_path / "invocation"
    python_path = invocation_dir / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(invocation_dir)

    plan = build_resolved_plan(
        manifest_path, python_override="./venv/bin/python", device_override="cpu"
    )

    assert plan["runtime"]["python"] == str(python_path)


def test_python_path_directory_rejects(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)

    with pytest.raises(BenchmarkManifestError, match="Python executable"):
        build_resolved_plan(manifest_path, python_override=str(tmp_path), device_override="cpu")


@pytest.mark.parametrize(
    ("prepare", "expected"),
    [
        (lambda tmp_path: tmp_path / "missing.yaml", "cannot be read"),
        (lambda tmp_path: tmp_path, "existing file"),
        (_malformed_yaml_path, "malformed"),
    ],
)
def test_cli_manifest_file_failures_are_concise(tmp_path, capsys, prepare, expected):
    manifest_path = prepare(tmp_path)

    assert run_position_benchmark.main([str(manifest_path), "--dry-run"]) == 2

    captured = capsys.readouterr()
    assert expected in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda tmp_path, manifest: manifest["training"].__setitem__("split_plan", "splits"),
            "split_plan",
        ),
        (
            lambda tmp_path, manifest: (tmp_path / "splits" / "split_plan.yaml").write_text(
                "split: [",
                encoding="utf-8",
            ),
            "malformed",
        ),
        (
            lambda tmp_path, manifest: manifest["dataset"].__setitem__("preprocessed_data", "data"),
            "preprocessed_data",
        ),
        (
            lambda tmp_path, manifest: (
                manifest["training"].__setitem__("sex_map", "sex_dir"),
                (tmp_path / "sex_dir").mkdir(),
            ),
            "sex_map",
        ),
        (
            lambda tmp_path, manifest: (
                manifest["training"].__setitem__("pc_map", "pc_dir"),
                manifest["training"].__setitem__("num_pcs", 1),
                (tmp_path / "pc_dir").mkdir(),
            ),
            "pc_map",
        ),
        (
            lambda tmp_path, manifest: (
                manifest["runtime"].__setitem__("python", "python_dir"),
                (tmp_path / "python_dir").mkdir(),
            ),
            "Python executable",
        ),
    ],
)
def test_cli_ordinary_input_failures_are_concise(tmp_path, capsys, mutate, expected):
    manifest_path, manifest = _base_manifest(tmp_path)
    mutate(tmp_path, manifest)
    _write_yaml(manifest_path, manifest)
    argv = [str(manifest_path), "--dry-run", "--device", "cpu"]
    if expected != "Python executable":
        argv.extend(["--python", sys.executable])

    assert run_position_benchmark.main(argv) == 2

    captured = capsys.readouterr()
    assert expected in captured.err
    assert "Traceback" not in captured.err


def test_cli_help_has_no_benchmark_side_effects(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run_position_benchmark.build_arg_parser().parse_args(["--help"])

    assert error.value.code == 0
    assert "dry-run" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
