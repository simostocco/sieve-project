"""Tests for Phase 12C2B positional benchmark manifest validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from scripts.position_benchmark_manifest import (
    DEFERRED_SAMPLE_BINDING,
    DEFERRED_STRATEGY_IDENTITY,
    BenchmarkManifestError,
    build_resolved_plan,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _cv_split_plan() -> dict:
    return {
        "schema_version": 1,
        "mode": "cv",
        "n_samples": 4,
        "sample_ids_sha256": HASH_A,
        "seed": 42,
        "split_source": "generated",
        "n_folds": 2,
        "folds": [
            {
                "fold_index": 0,
                "train_indices": [1, 3],
                "val_indices": [0, 2],
                "train_sample_ids_sha256": HASH_B,
                "val_sample_ids_sha256": HASH_C,
            },
            {
                "fold_index": 1,
                "train_indices": [0, 2],
                "val_indices": [1, 3],
                "train_sample_ids_sha256": HASH_D,
                "val_sample_ids_sha256": HASH_E,
            },
        ],
    }


def _single_split_plan() -> dict:
    return {
        "schema_version": 1,
        "mode": "single_split",
        "n_samples": 4,
        "sample_ids_sha256": HASH_A,
        "seed": 42,
        "split_source": "generated",
        "train_indices": [0, 2],
        "val_indices": [1, 3],
        "train_sample_ids_sha256": HASH_B,
        "val_sample_ids_sha256": HASH_C,
    }


def _position_none() -> dict:
    return {
        "position_preset": "custom",
        "absolute_position_encoding": "none",
        "relative_position_encoding": "none",
        "chromosome_encoding": "none",
        "cross_chromosome_policy": "separate",
    }


def _base_manifest(
    tmp_path: Path, *, mode: str = "cv", role: str = "primary", level: str = "L3"
) -> tuple[Path, dict]:
    dataset = tmp_path / "data" / "cohort.pt"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("placeholder", encoding="utf-8")
    split_plan_path = tmp_path / "splits" / "split_plan.yaml"
    split_plan = _cv_split_plan() if mode == "cv" else _single_split_plan()
    _write_yaml(split_plan_path, split_plan)
    training = {
        "mode": mode,
        "seed": 42,
        "val_split": 0.2,
        "split_plan": "splits/split_plan.yaml",
        "latent_dim": 64,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_attention_layers": 2,
        "aggregation_method": "mean",
        "classifier_type": "flatten",
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
        "sex_map": None,
        "pc_map": None,
        "num_pcs": 0,
    }
    explanation = {
        "ig_mode": "content",
        "n_steps": 50,
        "max_variants": 2000,
        "aggregation_method": "mean",
    }
    if mode == "cv":
        training["cv_folds"] = 2
        explanation["fold_index"] = 0
    manifest = {
        "schema_version": 1,
        "benchmark_id": f"posenc_{level.lower()}_{role}",
        "annotation_level": level,
        "benchmark_role": role,
        "paths": {"output_root": "outputs"},
        "dataset": {
            "preprocessed_data": "data/cohort.pt",
            "genome_build": "GRCh37",
        },
        "training": training,
        "explanation": explanation,
        "runtime": {
            "python": "python",
            "device": "cpu",
            "train_num_workers": 0,
            "explain_batch_size": 4,
        },
        "runs": [
            {"run_id": "legacy", "position": {"position_preset": "legacy"}},
            {"run_id": "no_position", "position": _position_none()},
        ],
    }
    path = _write_yaml(tmp_path / "manifest.yaml", manifest)
    return path, manifest


def _plan(tmp_path: Path, **kwargs) -> dict:
    manifest_path, _ = _base_manifest(tmp_path, **kwargs)
    return build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_valid_l3_primary_manifest_builds_resolved_plan(tmp_path):
    plan = _plan(tmp_path)

    assert plan["benchmark"]["annotation_level"] == "L3"
    assert plan["benchmark"]["benchmark_role"] == "primary"
    assert plan["split_plan"]["dataset_sample_binding_validation"] == DEFERRED_SAMPLE_BINDING
    assert plan["runs"][0]["canonical_position_strategy_identity"] == DEFERRED_STRATEGY_IDENTITY
    assert plan["runs"][0]["position_intent"] == {"position_preset": "legacy"}


def test_valid_l0_sensitivity_manifest(tmp_path):
    plan = _plan(tmp_path, role="sensitivity", level="L0")

    assert plan["benchmark"]["annotation_level"] == "L0"
    assert plan["benchmark"]["benchmark_role"] == "sensitivity"


def test_schema_mismatch_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["schema_version"] = 2
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="schema_version"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_unknown_top_level_key_rejects_null_block(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["null"] = {"enabled": False}
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="unknown top-level"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


@pytest.mark.parametrize(
    ("role", "level"),
    [("primary", "L0"), ("sensitivity", "L3"), ("primary", "L4")],
)
def test_role_level_contract_rejects_unsupported_pairs(tmp_path, role, level):
    manifest_path, _ = _base_manifest(tmp_path, role=role, level=level)

    with pytest.raises(BenchmarkManifestError, match="annotation_level|L4"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


@pytest.mark.parametrize("bad_id", ["", ".", "..", "bad/id", "bad id", " bad", "bad\tid"])
def test_unsafe_benchmark_id_rejects(tmp_path, bad_id):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["benchmark_id"] = bad_id
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="benchmark_id"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_duplicate_run_id_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"][1]["run_id"] = "legacy"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="duplicate run_id"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_zero_runs_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"] = []
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="at least two"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_one_run_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"] = manifest["runs"][:1]
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="at least two"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_two_runs_succeeds(tmp_path):
    plan = _plan(tmp_path)

    assert [run["run_id"] for run in plan["runs"]] == ["legacy", "no_position"]


def test_relative_paths_resolve_against_manifest_parent_and_not_cwd(tmp_path, monkeypatch):
    manifest_path, _ = _base_manifest(tmp_path)
    monkeypatch.chdir("/")

    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")

    assert plan["dataset"]["preprocessed_data"] == str(tmp_path / "data" / "cohort.pt")
    assert plan["split_plan"]["input_path"] == str(tmp_path / "splits" / "split_plan.yaml")


def test_absolute_paths_remain_absolute(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    dataset = tmp_path / "absolute.pt"
    dataset.write_text("placeholder", encoding="utf-8")
    manifest["dataset"]["preprocessed_data"] = str(dataset)
    _write_yaml(manifest_path, manifest)

    plan = build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")

    assert plan["dataset"]["preprocessed_data"] == str(dataset)


def test_missing_dataset_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["dataset"]["preprocessed_data"] = "data/missing.pt"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="preprocessed_data"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_dataset_directory_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    directory = tmp_path / "data" / "directory_dataset"
    directory.mkdir()
    manifest["dataset"]["preprocessed_data"] = "data/directory_dataset"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="preprocessed_data"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_optional_covariate_files_are_validated(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["training"]["pc_map"] = "missing_pcs.tsv"
    manifest["training"]["num_pcs"] = 2
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="pc_map"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_sex_map_directory_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    directory = tmp_path / "sex_dir"
    directory.mkdir()
    manifest["training"]["sex_map"] = "sex_dir"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="sex_map"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_pc_map_directory_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    directory = tmp_path / "pc_dir"
    directory.mkdir()
    manifest["training"]["pc_map"] = "pc_dir"
    manifest["training"]["num_pcs"] = 1
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="pc_map"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_pc_map_num_pcs_relationship_matches_train_py(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    pc_map = tmp_path / "pcs.tsv"
    pc_map.write_text("sample_id\tPC1\nS1\t0.1\n", encoding="utf-8")
    manifest["training"]["pc_map"] = "pcs.tsv"
    manifest["training"]["num_pcs"] = 0
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="pc_map requires"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_structural_cv_split_plan_accepted(tmp_path):
    plan = _plan(tmp_path)

    assert plan["split_plan"]["mode"] == "cv"
    assert plan["split_plan"]["n_folds"] == 2
    assert [fold["fold_index"] for fold in plan["split_plan"]["folds"]] == [0, 1]


def test_cv_split_membership_hash_is_independent_of_fold_serialization_order(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path)
    ordered_plan = build_resolved_plan(
        manifest_path,
        python_override=sys.executable,
        device_override="cpu",
    )
    split_plan = _cv_split_plan()
    split_plan["folds"] = list(reversed(split_plan["folds"]))
    _write_yaml(tmp_path / "splits" / "split_plan.yaml", split_plan)

    reversed_plan = build_resolved_plan(
        manifest_path,
        python_override=sys.executable,
        device_override="cpu",
    )

    assert (
        reversed_plan["split_plan"]["membership_sha256"]
        == ordered_plan["split_plan"]["membership_sha256"]
    )
    assert [fold["fold_index"] for fold in reversed_plan["split_plan"]["folds"]] == [0, 1]


def test_structural_single_split_plan_accepted(tmp_path):
    plan = _plan(tmp_path, mode="single_split")

    assert plan["split_plan"]["mode"] == "single_split"
    assert "n_folds" not in plan["split_plan"]
    assert plan["runs"][0]["explain_argv"].count("--fold-index") == 0


def test_split_mode_mismatch_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path, mode="cv")
    _write_yaml(tmp_path / "splits" / "split_plan.yaml", _single_split_plan())
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="mode"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_split_n_folds_mismatch_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path, mode="cv")
    manifest["training"]["cv_folds"] = 3
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="n_folds"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_fold_index_out_of_range_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path, mode="cv")
    manifest["explanation"]["fold_index"] = 2
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="fold_index"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_malformed_split_hash_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    split_plan = _cv_split_plan()
    split_plan["sample_ids_sha256"] = "not-a-hash"
    _write_yaml(tmp_path / "splits" / "split_plan.yaml", split_plan)
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="SHA256"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan.pop("seed"), "seed"),
        (lambda plan: plan.__setitem__("split_source", "mystery"), "split_source"),
        (lambda plan: plan.__setitem__("train_indices", [0, 0]), "duplicate"),
        (lambda plan: plan.__setitem__("train_indices", [0, 4]), "out-of-range"),
        (lambda plan: plan.__setitem__("val_indices", [0, 3]), "disjoint"),
        (lambda plan: plan.__setitem__("val_indices", [3]), "cover exactly"),
    ],
)
def test_malformed_single_split_structural_invariants_reject(tmp_path, mutate, message):
    manifest_path, _ = _base_manifest(tmp_path, mode="single_split")
    split_plan = _single_split_plan()
    mutate(split_plan)
    _write_yaml(tmp_path / "splits" / "split_plan.yaml", split_plan)

    with pytest.raises(BenchmarkManifestError, match=message):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_repeated_cv_validation_sample_rejects(tmp_path):
    manifest_path, _ = _base_manifest(tmp_path, mode="cv")
    split_plan = _cv_split_plan()
    split_plan["folds"][1]["val_indices"] = [0, 3]
    split_plan["folds"][1]["train_indices"] = [1, 2]
    _write_yaml(tmp_path / "splits" / "split_plan.yaml", split_plan)

    with pytest.raises(BenchmarkManifestError, match="each sample exactly once"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_legacy_extra_custom_field_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"][0]["position"]["absolute_position_encoding"] = "none"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="legacy preset rejects"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_missing_applicable_numeric_field_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"][1]["position"] = {
        "position_preset": "custom",
        "absolute_position_encoding": "sinusoidal",
        "relative_position_encoding": "none",
        "chromosome_encoding": "none",
        "cross_chromosome_policy": "separate",
        "position_dim": 64,
        "sinusoidal_coordinate_scale": 1.0,
    }
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="sinusoidal_max_wavelength"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_irrelevant_position_field_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"][1]["position"]["rope_base"] = 10000.0
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="irrelevant"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_duplicate_position_intent_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runs"].append({"run_id": "same_intent", "position": _position_none()})
    _write_yaml(manifest_path, manifest)

    with pytest.raises(
        BenchmarkManifestError, match="duplicates an existing explicit position intent"
    ):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_output_root_existing_file_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    output_root = tmp_path / "existing_file"
    output_root.write_text("not a directory", encoding="utf-8")
    manifest["paths"]["output_root"] = "existing_file"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="output_root"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_output_root_equal_to_dataset_path_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["paths"]["output_root"] = "data/cohort.pt"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="output_root"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")


def test_output_root_equal_to_split_plan_path_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["paths"]["output_root"] = "splits/split_plan.yaml"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="output_root"):
        build_resolved_plan(manifest_path, python_override=sys.executable, device_override="cpu")
