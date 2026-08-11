"""Tests for strategy-aware predictive performance comparison."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import ablation_compare


def _position_encoding(relative_type="alibi_fixed", scale=10000.0):
    relative = {
        "type": relative_type,
        "num_buckets": None,
        "total_bias_rows": None,
        "max_distance_bp": None,
        "rope_coordinate_scale": None,
        "rope_base": None,
        "alibi_distance_function": None,
        "alibi_distance_scale": None,
    }
    if relative_type == "none":
        pass
    elif relative_type == "t5_bucket":
        relative.update({"num_buckets": 32, "max_distance_bp": 100000})
    elif relative_type == "alibi_fixed":
        relative.update(
            {
                "alibi_distance_function": "log1p",
                "alibi_distance_scale": scale,
            }
        )
    else:
        raise ValueError(relative_type)

    return {
        "schema_version": 1,
        "preset": "custom",
        "annotation_level": "L3",
        "absolute": {
            "type": "none",
            "fusion": None,
            "dim": None,
            "coordinate_scale": None,
            "max_wavelength": None,
            "bin_size_bp": None,
        },
        "relative": relative,
        "chromosome": {
            "encoding": "none",
            "cross_chromosome_policy": "separate",
            "num_chromosomes": 3,
            "requires_chrom_ids": relative_type != "none",
            "cross_chromosome_parameter": (
                "learned_bias" if relative_type == "alibi_fixed" else None
            ),
            "mapping": {"0": "1", "1": "2", "2": "X"},
        },
        "attribution": {"default_ig_mode": "content"},
        "content_dim": 7,
        "input_dim": 7,
    }


def _config(relative_type="alibi_fixed", scale=10000.0, **overrides):
    data = {
        "config_schema_version": 2,
        "level": "L3",
        "content_dim": 7,
        "input_dim": 7,
        "num_genes": 100,
        "num_chromosomes": 3,
        "dataset_identity": {
            "genome_build": "GRCh37",
            "gene_mapping_sha256": "genehash",
            "chromosome_mapping_sha256": "chromhash",
        },
        "seed": 42,
        "position_encoding": _position_encoding(relative_type, scale),
        "position_encoding_execution": {
            "schema_version": 2,
            "source": "resolved_position_encoding_applied_to_model",
            "resolved_config_applied_to_model": True,
            "training_mode": "cv",
        },
        "cv": 5,
        "val_split": 0.2,
        "latent_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_attention_layers": 2,
        "aggregation_method": "mean",
        "classifier_type": "flatten",
        "num_covariates": 0,
        "batch_size": 32,
        "epochs": 100,
        "lr": 0.001,
        "lambda_attr": 0.0,
        "early_stopping": 10,
        "gradient_clip": None,
        "gradient_accumulation_steps": 1,
        "class_weighting": "auto",
        "chunk_size": 3000,
        "chunk_overlap": 0,
        "preprocessed_data": "/data/cohort.pt",
        "vcf": None,
        "phenotypes": None,
        "sex_map": None,
        "pc_map": None,
        "pc_map_sha256": None,
        "num_pcs": 0,
    }
    data.update(overrides)
    return data


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _make_run(tmp_path, name, config, results=None, cv_results=None):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    _write_yaml(run_dir / "config.yaml", config)
    if cv_results is not None:
        _write_yaml(run_dir / "cv_results.yaml", cv_results)
    else:
        _write_yaml(run_dir / "results.yaml", results or {"auc": 0.7, "accuracy": 0.6, "loss": 0.4})
    return run_dir


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["ablation_compare.py", *map(str, argv)])
    return ablation_compare.main()


def test_level_mode_preserves_historical_results_yaml_header_and_yaml(tmp_path, monkeypatch):
    run_dir = tmp_path / "ablation_L2"
    run_dir.mkdir()
    _write_yaml(run_dir / "config.yaml", {"level": "L2"})
    _write_yaml(run_dir / "results.yaml", {"auc": 0.61, "accuracy": 0.55, "loss": 0.8})
    out_tsv = tmp_path / "summary.tsv"
    out_yaml = tmp_path / "summary.yaml"

    assert (
        _run_main(
            monkeypatch,
            "--run-dir",
            run_dir,
            "--out-summary-tsv",
            out_tsv,
            "--out-summary-yaml",
            out_yaml,
        )
        == 0
    )

    rows = list(csv.reader(out_tsv.open(encoding="utf-8"), delimiter="\t"))
    assert rows[0] == ["level", "run_id", "auc", "std_auc", "accuracy", "loss", "results_yaml"]
    summary = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert summary["best_level"] == "L2"
    assert summary["ranking_metric_priority"] == ["auc", "accuracy", "loss"]
    assert "levels" in summary


def test_direct_script_help_supports_documented_invocation():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/ablation_compare.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--comparison-axis" in result.stdout


def test_level_mode_infers_directory_level_without_config(tmp_path, monkeypatch):
    run_dir = tmp_path / "ablation_L1"
    run_dir.mkdir()
    _write_yaml(run_dir / "results.yaml", {"auc": 0.61, "accuracy": 0.55, "loss": 0.8})
    out_tsv = tmp_path / "summary.tsv"
    out_yaml = tmp_path / "summary.yaml"

    assert (
        _run_main(
            monkeypatch,
            "--run-dir",
            run_dir,
            "--out-summary-tsv",
            out_tsv,
            "--out-summary-yaml",
            out_yaml,
        )
        == 0
    )

    summary = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert summary["best_level"] == "L1"


def test_level_mode_extracts_cv_results_without_position_metadata(tmp_path, monkeypatch):
    run_dir = tmp_path / "ablation_L3"
    run_dir.mkdir()
    _write_yaml(
        run_dir / "cv_results.yaml", {"mean_auc": 0.77, "std_auc": 0.02, "mean_accuracy": 0.66}
    )
    out_tsv = tmp_path / "summary.tsv"
    out_yaml = tmp_path / "summary.yaml"

    assert (
        _run_main(
            monkeypatch,
            "--run-dir",
            run_dir,
            "--out-summary-tsv",
            out_tsv,
            "--out-summary-yaml",
            out_yaml,
        )
        == 0
    )

    rows = list(csv.DictReader(out_tsv.open(encoding="utf-8"), delimiter="\t"))
    assert rows[0]["level"] == "L3"
    assert rows[0]["auc"] == "0.77"
    assert rows[0]["std_auc"] == "0.02"
    assert rows[0]["accuracy"] == "0.66"


def test_position_mode_writes_strategy_aware_outputs_for_compatible_runs(tmp_path, monkeypatch):
    alibi = _make_run(
        tmp_path,
        "alibi",
        _config("alibi_fixed"),
        cv_results={"mean_auc": 0.8, "std_auc": 0.03, "mean_accuracy": 0.7},
    )
    none = _make_run(
        tmp_path,
        "none",
        _config("none", input_dim=7),
        results={"auc": 0.75, "accuracy": 0.69, "loss": 0.5},
    )
    out_tsv = tmp_path / "position.tsv"
    out_yaml = tmp_path / "position.yaml"

    assert (
        _run_main(
            monkeypatch,
            "--comparison-axis",
            "position",
            "--run-dir",
            alibi,
            "--run-dir",
            none,
            "--out-summary-tsv",
            out_tsv,
            "--out-summary-yaml",
            out_yaml,
        )
        == 0
    )

    rows = list(csv.DictReader(out_tsv.open(encoding="utf-8"), delimiter="\t"))
    assert list(rows[0]) == [
        "position_strategy_id",
        "position_strategy_name",
        "position_strategy_hash",
        "run_id",
        "level",
        "preset",
        "absolute_position_encoding",
        "relative_position_encoding",
        "chromosome_encoding",
        "cross_chromosome_policy",
        "auc",
        "std_auc",
        "accuracy",
        "loss",
        "config_path",
        "results_yaml",
    ]
    assert [row["position_strategy_id"] for row in rows] == sorted(
        row["position_strategy_id"] for row in rows
    )
    assert rows[0]["position_strategy_hash"]

    summary = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert summary["comparison_axis"] == "position"
    assert summary["compatibility"]["compatible"] is True
    assert summary["best_run_id"] == "alibi"
    assert summary["best_position_strategy_id"] == next(
        row["position_strategy_id"] for row in rows if row["run_id"] == "alibi"
    )
    assert summary["runs"][0]["position_strategy"]


def test_position_mode_winner_uses_auc_accuracy_loss_priority(tmp_path, monkeypatch):
    high_accuracy = _make_run(
        tmp_path,
        "high_accuracy",
        _config("none"),
        results={"auc": 0.8, "accuracy": 0.9, "loss": 0.9},
    )
    low_loss = _make_run(
        tmp_path,
        "low_loss",
        _config("alibi_fixed"),
        results={"auc": 0.8, "accuracy": 0.8, "loss": 0.1},
    )
    out_tsv = tmp_path / "position.tsv"
    out_yaml = tmp_path / "position.yaml"

    assert (
        _run_main(
            monkeypatch,
            "--comparison-axis",
            "position",
            "--run-dir",
            low_loss,
            "--run-dir",
            high_accuracy,
            "--out-summary-tsv",
            out_tsv,
            "--out-summary-yaml",
            out_yaml,
        )
        == 0
    )

    summary = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert summary["best_run_id"] == "high_accuracy"


def test_position_mode_rejects_results_dir(tmp_path, monkeypatch):
    assert _run_main(monkeypatch, "--comparison-axis", "position", "--results-dir", tmp_path) == 1


def test_position_mode_rejects_single_run_missing_required_context_field(
    tmp_path,
    monkeypatch,
):
    config = _config("none")
    config.pop("class_weighting")
    run_dir = _make_run(tmp_path, "missing_context", config)

    assert _run_main(monkeypatch, "--comparison-axis", "position", "--run-dir", run_dir) == 1


def test_position_mode_rejects_duplicate_run_ids_from_same_basename(tmp_path, monkeypatch):
    first = _make_run(tmp_path / "left", "same_name", _config("none"))
    second = _make_run(tmp_path / "right", "same_name", _config("alibi_fixed"))

    assert (
        _run_main(
            monkeypatch,
            "--comparison-axis",
            "position",
            "--run-dir",
            first,
            "--run-dir",
            second,
        )
        == 1
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cfg: cfg.pop("position_encoding"),
        lambda cfg: cfg["position_encoding_execution"].update(
            {"resolved_config_applied_to_model": False}
        ),
        lambda cfg: cfg["dataset_identity"].update({"chromosome_mapping_sha256": "different"}),
        lambda cfg: cfg.pop("class_weighting"),
    ],
)
def test_position_mode_rejects_missing_metadata_and_context_mismatches(
    tmp_path,
    monkeypatch,
    mutator,
):
    good = _make_run(tmp_path, "good", _config("none"))
    bad_config = _config("alibi_fixed")
    mutator(bad_config)
    bad = _make_run(tmp_path, "bad", bad_config)

    assert (
        _run_main(monkeypatch, "--comparison-axis", "position", "--run-dir", good, "--run-dir", bad)
        == 1
    )
