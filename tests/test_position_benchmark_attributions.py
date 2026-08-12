"""Tests for raw attribution stability comparison across position strategies."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from scripts import compare_position_attributions as cmp


def _position_encoding(relative_type: str = "none", scale: float = 10000.0) -> dict:
    relative: dict[str, object] = {"type": relative_type}
    input_dim = 7
    if relative_type == "t5_bucket":
        relative.update({"num_buckets": 32, "max_distance_bp": 100000})
        input_dim = 15
    elif relative_type == "alibi_fixed":
        relative.update(
            {
                "alibi_distance_function": "linear",
                "alibi_distance_scale": scale,
            }
        )
        input_dim = 7
    elif relative_type != "none":
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
                "learned_bias" if relative_type == "t5_bucket" else None
            ),
            "mapping": {"0": "1", "1": "2", "2": "X"},
        },
        "attribution": {"default_ig_mode": "content"},
        "content_dim": 7,
        "input_dim": input_dim,
    }


def _config(relative_type: str = "none", scale: float = 10000.0) -> dict:
    position_encoding = _position_encoding(relative_type, scale)
    return {
        "config_schema_version": 2,
        "level": "L3",
        "content_dim": 7,
        "input_dim": position_encoding["input_dim"],
        "num_genes": 100,
        "num_chromosomes": 3,
        "dataset_identity": {
            "genome_build": "GRCh37",
            "gene_mapping_sha256": "genehash",
            "chromosome_mapping_sha256": "chromhash",
        },
        "seed": 42,
        "position_encoding": position_encoding,
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


def _analysis_metadata(config: dict, provenance: dict, *, n_samples: int = 2) -> dict:
    position_encoding = config["position_encoding"]
    return {
        "is_null_baseline": False,
        "annotation_level": config["level"],
        "genome_build": config["dataset_identity"]["genome_build"],
        "n_samples": n_samples,
        "aggregation_method": config["aggregation_method"],
        "model_provenance": provenance,
        "integrated_gradients": {
            "executed": True,
            "attribution_schema_version": 1,
            "requested_ig_mode": "auto",
            "resolved_ig_mode": "content",
            "attribution_feature_space": "content",
            "attribution_width": config["content_dim"],
            "content_dim": config["content_dim"],
            "input_dim": config["input_dim"],
            "variant_score_aggregation": "l2",
            "baseline_policy": "zero_content_observed_absolute_position",
            "n_steps": 16,
            "max_variants": 100,
            "sampling_policy": "manual_chunk_full_coverage_no_random_subsampling",
            "sampling_seed": None,
            "comparability_warning": None,
            "absolute_position_encoding": position_encoding["absolute"]["type"],
            "relative_position_encoding": position_encoding["relative"]["type"],
            "chromosome_encoding": position_encoding["chromosome"]["encoding"],
            "position_encoding_metadata_source": "reconstructed_resolved_config",
        },
    }


def _base_samples() -> dict[str, dict[str, Any]]:
    return {
        "s1": {
            "sample_idx": 0,
            "sample_id": "s1",
            "label": 1,
            "positions": np.array([100, 200]),
            "gene_ids": np.array([10, 20]),
            "chromosomes": np.array(["1", "1"], dtype=object),
            "attributions": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        },
        "s2": {
            "sample_idx": 1,
            "sample_id": "s2",
            "label": 0,
            "positions": np.array([300]),
            "gene_ids": np.array([30]),
            "chromosomes": np.array(["2"], dtype=object),
            "attributions": np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npz_scalars(analysis: dict) -> dict[str, object]:
    ig = analysis["integrated_gradients"]
    return {
        "attribution_schema_version": ig["attribution_schema_version"],
        "requested_ig_mode": ig["requested_ig_mode"],
        "resolved_ig_mode": ig["resolved_ig_mode"],
        "attribution_feature_space": ig["attribution_feature_space"],
        "attribution_width": ig["attribution_width"],
        "content_dim": ig["content_dim"],
        "input_dim": ig["input_dim"],
        "absolute_position_encoding": ig["absolute_position_encoding"],
        "relative_position_encoding": ig["relative_position_encoding"],
        "chromosome_encoding": ig["chromosome_encoding"],
        "position_encoding_metadata_source": ig["position_encoding_metadata_source"],
        "variant_score_aggregation": ig["variant_score_aggregation"],
        "baseline_policy": ig["baseline_policy"],
        "n_steps": ig["n_steps"],
        "max_variants": ig["max_variants"],
        "sampling_policy": ig["sampling_policy"],
        "sampling_seed": -1,
        "comparability_warning": "",
    }


def _make_run(
    tmp_path: Path,
    run_id: str,
    *,
    relative_type: str = "none",
    scale: float = 10000.0,
    sample_order: list[str] | None = None,
    reverse_sample_variants: set[str] | None = None,
    sample_updates: Callable[[dict[str, dict[str, Any]]], None] | None = None,
    config_updates: Callable[[dict], None] | None = None,
    analysis_updates: Callable[[dict], None] | None = None,
    aggregate_scalar_updates: dict[str, object] | None = None,
    per_sample_scalar_updates: dict[str, object] | None = None,
    aggregate_scores_override: object | None = None,
    checkpoint_bytes: bytes = b"checkpoint bytes",
    checkpoint_selection_mode: str = "cv_explicit_fold",
    selected_fold: int | None = 0,
    selected_fold_auc: float | None = 0.75,
    cv_results_path: Path | None = None,
) -> cmp.PositionAttributionRunSpec:
    run_dir = tmp_path / run_id
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "model.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    cv_path = cv_results_path
    if cv_path is None and checkpoint_selection_mode in {"cv_explicit_fold", "cv_best_fold"}:
        cv_path = run_dir / "cv_results.yaml"
    if cv_path is not None:
        cv_path.write_text("fold_results: []\n", encoding="utf-8")
    config = _config(relative_type, scale)
    if config_updates is not None:
        config_updates(config)
    config_path = run_dir / "config.yaml"
    provenance = {
        "schema_version": 1,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_path": str(config_path.resolve()),
        "selected_fold": selected_fold,
        "selected_fold_auc": selected_fold_auc,
        "cv_results_path": str(cv_path.resolve()) if cv_path is not None else None,
    }
    samples = _base_samples()
    if sample_updates is not None:
        sample_updates(samples)
    if sample_order is None:
        sample_order = ["s1", "s2"]
    analysis = _analysis_metadata(config, provenance, n_samples=len(sample_order))
    if analysis_updates is not None:
        analysis_updates(analysis)
    _write_yaml(config_path, config)
    analysis_path = run_dir / "analysis_metadata.yaml"
    _write_yaml(analysis_path, analysis)

    reverse_sample_variants = reverse_sample_variants or set()
    aggregate_scores = []
    aggregate_metadata = []
    per_sample_scalar_updates = per_sample_scalar_updates or {}
    for sample_id in sample_order:
        sample = samples[sample_id]
        attributions = np.asarray(sample["attributions"], dtype=float)
        positions = np.asarray(sample["positions"], dtype=object)
        gene_ids = np.asarray(sample["gene_ids"], dtype=object)
        chromosomes = np.asarray(sample["chromosomes"], dtype=object)
        if sample_id in reverse_sample_variants:
            attributions = attributions[::-1]
            positions = positions[::-1]
            gene_ids = gene_ids[::-1]
            chromosomes = chromosomes[::-1]
        scores = np.linalg.norm(attributions, axis=1)
        metadata = {
            "sample_idx": sample["sample_idx"],
            "sample_id": sample["sample_id"],
            "label": sample["label"],
            "positions": positions,
            "gene_ids": gene_ids,
            "chromosomes": chromosomes,
        }
        aggregate_metadata.append(metadata)
        aggregate_scores.append(scores)
        per_sample_scalars = {
            key: value
            for key, value in _npz_scalars(analysis).items()
            if key in cmp.PER_SAMPLE_SCALAR_KEYS
        }
        per_sample_scalars.update(per_sample_scalar_updates)
        np.savez(
            per_sample_dir / f"sample_{sample['sample_idx']}.npz",
            attributions=attributions,
            variant_scores=scores,
            **per_sample_scalars,
        )
    aggregate_scalars = _npz_scalars(analysis)
    if aggregate_scalar_updates is not None:
        aggregate_scalars.update(aggregate_scalar_updates)
    np.savez(
        run_dir / "attributions.npz",
        variant_scores=(
            aggregate_scores_override
            if aggregate_scores_override is not None
            else np.array(aggregate_scores, dtype=object)
        ),
        metadata=np.array(aggregate_metadata, dtype=object),
        **aggregate_scalars,
    )
    return cmp.PositionAttributionRunSpec(
        run_id=run_id,
        config_path=config_path,
        analysis_metadata_path=analysis_path,
        attributions_path=run_dir / "attributions.npz",
        per_sample_dir=per_sample_dir,
    )


def _compare(tmp_path: Path, specs: list[cmp.PositionAttributionRunSpec]) -> dict:
    return cmp.compare_position_attributions(
        specs,
        out_summary_tsv=tmp_path / "nested" / "summary.tsv",
        out_sample_tsv=tmp_path / "nested2" / "sample.tsv",
        out_feature_tsv=tmp_path / "nested3" / "feature.tsv",
        out_comparison_yaml=tmp_path / "nested4" / "comparison.yaml",
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_direct_script_help_succeeds_and_shows_five_token_position_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/compare_position_attributions.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert (
        "--position-run RUN_ID CONFIG_YAML ANALYSIS_METADATA_YAML ATTRIBUTIONS_NPZ ATTRIBUTIONS_PER_SAMPLE_DIR"
        in result.stdout
    )


def test_position_run_parser_rejects_one_run_and_duplicate_ids(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a")
    entry = [
        run_a.run_id,
        str(run_a.config_path),
        str(run_a.analysis_metadata_path),
        str(run_a.attributions_path),
        str(run_a.per_sample_dir),
    ]
    with pytest.raises(ValueError, match="at least two"):
        cmp.parse_position_run_specs([entry])
    with pytest.raises(ValueError, match="duplicate run ID"):
        cmp.parse_position_run_specs([entry, entry])


@pytest.mark.parametrize(
    ("path_field", "message"),
    [
        ("config_path", "config file"),
        ("analysis_metadata_path", "analysis metadata file"),
        ("attributions_path", "aggregate attributions file"),
        ("per_sample_dir", "per-sample directory"),
    ],
)
def test_position_run_parser_rejects_missing_artifacts(
    tmp_path: Path,
    path_field: str,
    message: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b")
    entries = []
    for run in (run_a, run_b):
        values = {
            "run_id": run.run_id,
            "config_path": run.config_path,
            "analysis_metadata_path": run.analysis_metadata_path,
            "attributions_path": run.attributions_path,
            "per_sample_dir": run.per_sample_dir,
        }
        if run is run_b:
            values[path_field] = tmp_path / "missing"
        entries.append(
            [
                values["run_id"],
                str(values["config_path"]),
                str(values["analysis_metadata_path"]),
                str(values["attributions_path"]),
                str(values["per_sample_dir"]),
            ]
        )
    with pytest.raises(ValueError, match=message):
        cmp.parse_position_run_specs(entries)


def test_valid_comparison_aligns_samples_and_variants_and_writes_outputs(
    tmp_path: Path,
) -> None:
    run_a = _make_run(tmp_path, "run_a", relative_type="none")
    run_b = _make_run(
        tmp_path,
        "run_b",
        relative_type="alibi_fixed",
        sample_order=["s2", "s1"],
        reverse_sample_variants={"s1"},
    )
    comparison = _compare(tmp_path, [run_a, run_b])

    sample_rows = _read_rows(tmp_path / "nested2" / "sample.tsv")
    feature_rows = _read_rows(tmp_path / "nested3" / "feature.tsv")
    summary_rows = _read_rows(tmp_path / "nested" / "summary.tsv")
    assert list(sample_rows[0]) == cmp.SAMPLE_TSV_COLUMNS
    assert list(feature_rows[0]) == cmp.FEATURE_TSV_COLUMNS
    assert list(summary_rows[0]) == cmp.SUMMARY_TSV_COLUMNS
    assert [row["sample_id"] for row in sample_rows] == ["s1", "s2"]
    assert len(feature_rows) == 7
    assert summary_rows[0]["n_samples"] == "2"
    assert comparison["schema_version"] == 1
    assert comparison["comparison_axis"] == "position"
    assert comparison["analysis_type"] == "attribution_stability"
    assert comparison["alignment"]["exact_sample_universe_required"] is True
    assert comparison["artifact_integrity"]["aggregate_pickle_required"] is True
    assert "trusted local SIEVE" in comparison["warnings"][0]
    assert isinstance(comparison["pairwise_summary"], list)
    assert (tmp_path / "nested4" / "comparison.yaml").exists()


def test_outputs_are_genuine_tab_separated_tsv_files(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", relative_type="alibi_fixed")
    _compare(tmp_path, [run_a, run_b])

    expected_headers = {
        tmp_path / "nested2" / "sample.tsv": cmp.SAMPLE_TSV_COLUMNS,
        tmp_path / "nested3" / "feature.tsv": cmp.FEATURE_TSV_COLUMNS,
        tmp_path / "nested" / "summary.tsv": cmp.SUMMARY_TSV_COLUMNS,
    }
    for path, expected in expected_headers.items():
        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert "\t" in header
        assert header.split("\t") == expected


def test_input_dim_difference_alone_is_allowed_for_content_attributions(
    tmp_path: Path,
) -> None:
    run_a = _make_run(tmp_path, "run_a", relative_type="none")
    run_b = _make_run(tmp_path, "run_b", relative_type="t5_bucket")

    comparison = _compare(tmp_path, [run_a, run_b])

    compared_fields = comparison["compatibility"]["explanation_context"]["compared_fields"]
    assert "integrated_gradients.input_dim" not in compared_fields
    assert (
        comparison["runs"][0]["position_strategy_id"]
        != comparison["runs"][1]["position_strategy_id"]
    )


def test_strategy_identity_comes_from_config_and_not_analysis_metadata(
    tmp_path: Path,
) -> None:
    run_a = _make_run(tmp_path, "run_a", relative_type="none")
    run_b = _make_run(
        tmp_path,
        "run_b",
        relative_type="alibi_fixed",
        scale=25.0,
        analysis_updates=lambda analysis: analysis["integrated_gradients"].update(
            {"relative_position_encoding": "alibi_fixed"}
        ),
    )
    comparison = _compare(tmp_path, [run_a, run_b])
    strategy_ids = [run["position_strategy_id"] for run in comparison["runs"]]
    assert strategy_ids[0] != strategy_ids[1]

    altered = _make_run(tmp_path, "altered", relative_type="alibi_fixed", scale=50.0)
    altered_identity = cmp.load_position_attribution_run(altered).identity
    assert altered_identity.strategy_id != comparison["runs"][1]["position_strategy_id"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda config: config.update({"seed": 99}),
            "position comparison context mismatch",
        ),
        (
            lambda config: config["dataset_identity"].update({"gene_mapping_sha256": "different"}),
            "position comparison context mismatch",
        ),
        (
            lambda config: config.update({"content_dim": 6}),
            "feature names",
        ),
    ],
)
def test_training_context_mismatches_reject(
    tmp_path: Path,
    mutator: Callable[[dict], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", config_updates=mutator)
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


@pytest.mark.parametrize(
    ("analysis_mutator", "match"),
    [
        (
            lambda analysis: analysis["integrated_gradients"].update(
                {"resolved_ig_mode": "legacy"}
            ),
            "resolved_ig_mode",
        ),
        (lambda analysis: analysis.update({"is_null_baseline": True}), "is_null_baseline"),
        (
            lambda analysis: analysis["integrated_gradients"].update({"n_steps": 17}),
            "position explanation context mismatch",
        ),
        (
            lambda analysis: analysis["integrated_gradients"].update(
                {"relative_position_encoding": "t5_bucket"}
            ),
            "relative_position_encoding",
        ),
    ],
)
def test_explanation_and_ig_metadata_mismatches_reject(
    tmp_path: Path,
    analysis_mutator: Callable[[dict], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", analysis_updates=analysis_mutator)
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


@pytest.mark.parametrize(
    ("analysis_mutator", "match"),
    [
        (lambda analysis: analysis.pop("model_provenance"), "model_provenance"),
        (
            lambda analysis: analysis["model_provenance"].update({"schema_version": 2}),
            "schema_version",
        ),
        (
            lambda analysis: analysis["model_provenance"].update({"checkpoint_sha256": "bad"}),
            "checkpoint_sha256",
        ),
        (
            lambda analysis: analysis["model_provenance"].update({"checkpoint_sha256": "0" * 64}),
            "checkpoint hash mismatch",
        ),
        (
            lambda analysis: analysis["model_provenance"].update(
                {"checkpoint_path": str(Path("/tmp/missing-checkpoint.pt"))}
            ),
            "checkpoint file is missing",
        ),
        (
            lambda analysis: analysis["model_provenance"].update(
                {"config_path": str(Path("/tmp/other-config.yaml"))}
            ),
            "config_path mismatch",
        ),
        (
            lambda analysis: analysis["model_provenance"].update(
                {"checkpoint_selection_mode": "cv_best_fold"}
            ),
            "cv_explicit_fold",
        ),
    ],
)
def test_checkpoint_provenance_rejects_malformed_inputs(
    tmp_path: Path,
    analysis_mutator: Callable[[dict], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", analysis_updates=analysis_mutator)
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


def test_cv_requires_same_explicit_fold_but_auc_may_differ(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a", selected_fold=1, selected_fold_auc=0.7)
    run_b = _make_run(tmp_path, "run_b", selected_fold=1, selected_fold_auc=0.9)
    comparison = _compare(tmp_path, [run_a, run_b])
    assert comparison["compatibility"]["checkpoint_policy"]["selected_fold"] == 1

    run_c = _make_run(tmp_path, "run_c", selected_fold=2, selected_fold_auc=0.9)
    with pytest.raises(ValueError, match="selected_fold mismatch"):
        _compare(tmp_path, [run_a, run_c])


def test_single_split_policy_accepts_same_mode_and_rejects_mixed_modes(
    tmp_path: Path,
) -> None:
    def single_split(config: dict) -> None:
        config["position_encoding_execution"]["training_mode"] = "single_split"
        config["cv"] = False

    run_a = _make_run(
        tmp_path,
        "run_a",
        config_updates=single_split,
        checkpoint_selection_mode="explicit_checkpoint",
        selected_fold=None,
        selected_fold_auc=None,
        cv_results_path=None,
    )
    run_b = _make_run(
        tmp_path,
        "run_b",
        config_updates=single_split,
        checkpoint_selection_mode="explicit_checkpoint",
        selected_fold=None,
        selected_fold_auc=None,
        cv_results_path=None,
    )
    comparison = _compare(tmp_path, [run_a, run_b])
    assert comparison["compatibility"]["checkpoint_policy"]["selected_fold"] is None

    run_c = _make_run(
        tmp_path,
        "run_c",
        config_updates=single_split,
        checkpoint_selection_mode="single_run_best_model",
        selected_fold=None,
        selected_fold_auc=None,
        cv_results_path=None,
    )
    with pytest.raises(ValueError, match="checkpoint_selection_mode mismatch"):
        _compare(tmp_path, [run_a, run_c])


@pytest.mark.parametrize(
    ("sample_updates", "match"),
    [
        (
            lambda samples: samples["s2"].update({"sample_id": "missing_s2"}),
            "sample universe mismatch",
        ),
        (lambda samples: samples["s1"].update({"label": 0}), "label mismatch"),
    ],
)
def test_sample_universe_and_labels_are_exact(
    tmp_path: Path,
    sample_updates: Callable[[dict[str, dict[str, Any]]], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", sample_updates=sample_updates)
    with pytest.raises(ValueError, match=match) as excinfo:
        _compare(tmp_path, [run_a, run_b])
    if "sample universe" in match:
        message = str(excinfo.value)
        assert "reference sample count 2" in message
        assert "other sample count 2" in message
        assert "missing count 1" in message
        assert "extra count 1" in message


def test_per_sample_files_must_match_aggregate_sample_indices(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b")
    (run_b.per_sample_dir / "sample_1.npz").unlink()
    with pytest.raises(ValueError, match="missing per-sample files"):
        _compare(tmp_path, [run_a, run_b])
    _make_run(tmp_path, "run_c")
    run_c = cmp.PositionAttributionRunSpec(
        "run_c",
        tmp_path / "run_c" / "config.yaml",
        tmp_path / "run_c" / "analysis_metadata.yaml",
        tmp_path / "run_c" / "attributions.npz",
        tmp_path / "run_c" / "per_sample",
    )
    np.savez(run_c.per_sample_dir / "sample_99.npz", attributions=np.zeros((1, 7)))
    with pytest.raises(ValueError, match="extra sample_N"):
        _compare(tmp_path, [run_a, run_c])


@pytest.mark.parametrize(
    ("sample_updates", "match"),
    [
        (lambda samples: samples["s1"].update({"positions": np.array([100])}), "equal length"),
        (
            lambda samples: samples["s1"].update(
                {"positions": np.array([100, 100]), "gene_ids": np.array([10, 10])}
            ),
            "duplicate variant key",
        ),
        (
            lambda samples: samples["s1"].update(
                {"chromosomes": np.array(["", "1"], dtype=object)}
            ),
            "chromosome",
        ),
        (
            lambda samples: samples["s1"].update(
                {"positions": np.array([True, 200], dtype=object)}
            ),
            "position",
        ),
    ],
)
def test_malformed_aggregate_sample_metadata_rejects(
    tmp_path: Path,
    sample_updates: Callable[[dict[str, dict[str, Any]]], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", sample_updates=sample_updates)
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


@pytest.mark.parametrize(
    ("sample_updates", "match"),
    [
        (
            lambda samples: samples["s1"].update({"gene_ids": np.array([10, 21])}),
            "variant universe mismatch",
        ),
        (
            lambda samples: samples["s1"].update(
                {
                    "positions": np.array([100, 200, 300]),
                    "gene_ids": np.array([10, 20, 30]),
                    "chromosomes": np.array(["1", "1", "1"], dtype=object),
                    "attributions": np.vstack([samples["s1"]["attributions"], np.zeros((1, 7))]),
                }
            ),
            "variant universe mismatch",
        ),
    ],
)
def test_per_sample_variant_universe_is_exact(
    tmp_path: Path,
    sample_updates: Callable[[dict[str, dict[str, Any]]], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", sample_updates=sample_updates)
    with pytest.raises(ValueError, match=match) as excinfo:
        _compare(tmp_path, [run_a, run_b])
    message = str(excinfo.value)
    assert "reference variant count" in message
    assert "other variant count" in message
    assert "missing count" in message
    assert "extra count" in message


def test_rectangular_equal_length_aggregate_object_scores_succeed(tmp_path: Path) -> None:
    def equal_length(samples: dict[str, dict[str, Any]]) -> None:
        samples["s2"].update(
            {
                "positions": np.array([300, 400]),
                "gene_ids": np.array([30, 40]),
                "chromosomes": np.array(["2", "2"], dtype=object),
                "attributions": np.array(
                    [
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            }
        )

    run_a = _make_run(tmp_path, "run_a", sample_updates=equal_length)
    run_b = _make_run(tmp_path, "run_b", sample_updates=equal_length)

    comparison = _compare(tmp_path, [run_a, run_b])

    loaded = np.load(run_a.attributions_path, allow_pickle=True)["variant_scores"]
    assert loaded.dtype == object
    assert loaded.ndim == 2
    assert comparison["pairwise_summary"][0]["n_samples"] == 2


def test_malformed_aggregate_object_scores_reject_cleanly(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(
        tmp_path,
        "run_b",
        aggregate_scores_override=np.array(
            [np.array(["bad", "score"], dtype=object), np.array([1.0], dtype=object)],
            dtype=object,
        ),
    )

    with pytest.raises(ValueError, match="aggregate variant_scores.*numeric"):
        _compare(tmp_path, [run_a, run_b])


def test_zero_sample_comparison_rejects(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a", sample_order=[])
    run_b = _make_run(tmp_path, "run_b", sample_order=[])

    with pytest.raises(ValueError, match="analysis_metadata.n_samples"):
        _compare(tmp_path, [run_a, run_b])


@pytest.mark.parametrize(
    ("writer", "match"),
    [
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.zeros((2, 7, 1)),
                variant_scores=np.zeros(2),
                **scalars,
            ),
            "two-dimensional",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.zeros((2, 6)),
                variant_scores=np.zeros(2),
                **scalars,
            ),
            "shape mismatch",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.zeros((1, 7)),
                variant_scores=np.zeros(1),
                **scalars,
            ),
            "shape mismatch",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.zeros((2, 7)),
                variant_scores=np.zeros((2, 1)),
                **scalars,
            ),
            "one-dimensional",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.array([["x"] * 7, ["y"] * 7]),
                variant_scores=np.zeros(2),
                **scalars,
            ),
            "numeric",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.array([[np.inf] * 7, [0.0] * 7]),
                variant_scores=np.array([np.inf, 0.0]),
                **scalars,
            ),
            "finite",
        ),
        (
            lambda path, scalars: np.savez(
                path,
                attributions=np.zeros((2, 7)),
                variant_scores=np.ones(2),
                **scalars,
            ),
            "aggregate/per-sample score mismatch|self-consistency",
        ),
    ],
)
def test_per_sample_matrix_integrity_rejects(
    tmp_path: Path,
    writer: Callable[[Path, dict[str, object]], None],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b")
    analysis = yaml.safe_load(run_b.analysis_metadata_path.read_text(encoding="utf-8"))
    scalars = {
        key: value
        for key, value in _npz_scalars(analysis).items()
        if key in cmp.PER_SAMPLE_SCALAR_KEYS
    }
    writer(run_b.per_sample_dir / "sample_0.npz", scalars)
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


@pytest.mark.parametrize(
    ("aggregate_scalar_updates", "match"),
    [
        ({"resolved_ig_mode": "legacy"}, "resolved_ig_mode"),
        ({"relative_position_encoding": "unavailable"}, "unavailable"),
    ],
)
def test_aggregate_scalar_metadata_is_cross_validated(
    tmp_path: Path,
    aggregate_scalar_updates: dict[str, object],
    match: str,
) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(
        tmp_path,
        "run_b",
        aggregate_scalar_updates=aggregate_scalar_updates,
    )
    with pytest.raises(ValueError, match=match):
        _compare(tmp_path, [run_a, run_b])


def test_per_sample_scalar_metadata_is_cross_validated(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(
        tmp_path,
        "run_b",
        per_sample_scalar_updates={"resolved_ig_mode": "legacy"},
    )
    with pytest.raises(ValueError, match="resolved_ig_mode"):
        _compare(tmp_path, [run_a, run_b])


def test_metric_semantics_cover_scaling_sign_and_zero_policies() -> None:
    assert cmp.signed_cosine(np.array([1.0, 2.0]), np.array([2.0, 4.0])).value == pytest.approx(1.0)
    assert cmp.signed_cosine(np.array([1.0]), np.array([-1.0])).value == -1.0
    both_zero = cmp.signed_cosine(np.zeros(2), np.zeros(2))
    assert (both_zero.value, both_zero.reason) == (1.0, "both_zero")
    one_zero = cmp.signed_cosine(np.zeros(2), np.ones(2))
    assert (one_zero.value, one_zero.reason) == (0.0, "one_zero")
    assert cmp.normalized_l2(np.array([1.0]), np.array([3.0])) == pytest.approx(0.5)
    assert cmp.normalized_l2(np.zeros(2), np.zeros(2)) == 0.0


@pytest.mark.parametrize(
    ("values_a", "values_b", "expected_reason", "valid"),
    [
        ([1.0], [1.0], "insufficient_values", False),
        ([1.0, 1.0], [2.0, 2.0], "both_constant", False),
        ([1.0, 1.0], [2.0, 3.0], "one_constant", False),
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "ok", True),
    ],
)
def test_pearson_policies(
    values_a: list[float],
    values_b: list[float],
    expected_reason: str,
    valid: bool,
) -> None:
    result = cmp.pearson(np.array(values_a), np.array(values_b))
    assert result.reason == expected_reason
    assert result.valid is valid


def test_feature_sufficient_statistics_match_numpy() -> None:
    stats = cmp.FeatureStats()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    stats.update(a, b)
    metrics = cmp.feature_metrics_from_stats(stats)
    assert metrics["cosine"] == pytest.approx(
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    )
    assert metrics["pearson"] == pytest.approx(float(np.corrcoef(a, b)[0, 1]))
    assert metrics["mean_abs_a"] == pytest.approx(2.0)


def test_invalid_pearson_writes_empty_tsv_field(tmp_path: Path) -> None:
    def constant(samples: dict[str, dict[str, Any]]) -> None:
        samples["s1"]["attributions"] = np.zeros((2, 7))
        samples["s2"]["attributions"] = np.zeros((1, 7))

    run_a = _make_run(tmp_path, "run_a", sample_updates=constant)
    run_b = _make_run(tmp_path, "run_b", sample_updates=constant)
    _compare(tmp_path, [run_a, run_b])
    rows = _read_rows(tmp_path / "nested2" / "sample.tsv")
    assert rows[0]["signed_pearson"] == ""
    assert rows[0]["signed_pearson_reason"] == "both_constant"


def test_pairwise_processing_holds_at_most_two_full_sample_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = [
        _make_run(tmp_path, "run_a"),
        _make_run(tmp_path, "run_b", relative_type="alibi_fixed"),
        _make_run(tmp_path, "run_c", relative_type="t5_bucket"),
    ]
    active = 0
    maximum = 0
    full_matrix_shape_seen = False
    feature_temporary_shapes: list[tuple[int, ...]] = []
    score_temporary_shapes: list[tuple[int, ...]] = []
    original_load = cmp._load_per_sample_attributions
    original_feature_values = cmp._ordered_feature_values
    original_ordered_scores = cmp._ordered_scores

    def tracking_load(run: cmp.LoadedPositionAttributionRun, sample_id: str) -> cmp.LoadedSample:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        return original_load(run, sample_id)

    @contextmanager
    def tracking_open(run: cmp.LoadedPositionAttributionRun, sample_id: str) -> Any:
        nonlocal active
        sample = tracking_load(run, sample_id)
        try:
            yield sample
        finally:
            active -= 1

    def tracking_feature_values(sample: cmp.LoadedSample, feature_index: int) -> np.ndarray:
        nonlocal full_matrix_shape_seen
        values = original_feature_values(sample, feature_index)
        feature_temporary_shapes.append(values.shape)
        full_matrix_shape_seen = full_matrix_shape_seen or values.shape == sample.attributions.shape
        return values

    def tracking_ordered_scores(sample: cmp.LoadedSample) -> np.ndarray:
        values = original_ordered_scores(sample)
        score_temporary_shapes.append(values.shape)
        return values

    monkeypatch.setattr(cmp, "_open_ordered_sample", tracking_open)
    monkeypatch.setattr(cmp, "_ordered_feature_values", tracking_feature_values)
    monkeypatch.setattr(cmp, "_ordered_scores", tracking_ordered_scores)
    _compare(tmp_path, runs)
    assert maximum <= 2
    assert feature_temporary_shapes
    assert score_temporary_shapes
    assert not full_matrix_shape_seen
    assert all(len(shape) == 1 for shape in feature_temporary_shapes)
    assert all(len(shape) == 1 for shape in score_temporary_shapes)
