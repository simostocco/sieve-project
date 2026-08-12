"""Tests for strategy-aware positional ranking stability comparison."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import compare_ablation_rankings as rankings
from scripts.position_benchmark_metadata import (
    extract_explanation_context,
    require_compatible_explanation_contexts,
)


def _position_encoding(relative_type: str = "none", scale: float = 10000.0) -> dict:
    relative: dict[str, object] = {"type": relative_type}
    if relative_type == "t5_bucket":
        relative.update({"num_buckets": 32, "max_distance_bp": 100000})
    elif relative_type == "alibi_fixed":
        relative.update(
            {
                "alibi_distance_function": "log1p",
                "alibi_distance_scale": scale,
            }
        )
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
            "cross_chromosome_parameter": ("learned_bias" if relative_type != "none" else None),
            "mapping": {"0": "1", "1": "2", "2": "X"},
        },
        "attribution": {"default_ig_mode": "content"},
        "content_dim": 7,
        "input_dim": 7 if relative_type == "none" else 15,
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


def _analysis_metadata(config: dict) -> dict:
    position_encoding = config["position_encoding"]
    return {
        "is_null_baseline": False,
        "annotation_level": config["level"],
        "genome_build": config["dataset_identity"]["genome_build"],
        "n_samples": 4,
        "aggregation_method": "rank_average",
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


def _ranking_rows(order: list[str] | None = None) -> list[dict[str, object]]:
    by_id = {
        "v1": {
            "variant_id": "v1",
            "gene_name": "GENE1",
            "gene_id": 1,
            "chromosome": "1",
            "position": 100,
            "rank": 1,
            "mean_attribution": 9.0,
            "max_attribution": 11.0,
            "resolved_ig_mode": "content",
            "attribution_feature_space": "content",
            "variant_score_aggregation": "l2",
        },
        "v2": {
            "variant_id": "v2",
            "gene_name": "GENE2",
            "gene_id": 2,
            "chromosome": "1",
            "position": 200,
            "rank": 2,
            "mean_attribution": 8.0,
            "max_attribution": 10.0,
            "resolved_ig_mode": "content",
            "attribution_feature_space": "content",
            "variant_score_aggregation": "l2",
        },
        "v3": {
            "variant_id": "v3",
            "gene_name": "GENE3",
            "gene_id": 3,
            "chromosome": "2",
            "position": 300,
            "rank": 3,
            "mean_attribution": 7.0,
            "max_attribution": 9.0,
            "resolved_ig_mode": "content",
            "attribution_feature_space": "content",
            "variant_score_aggregation": "l2",
        },
        "v4": {
            "variant_id": "v4",
            "gene_name": "GENE4",
            "gene_id": 4,
            "chromosome": "X",
            "position": 400,
            "rank": 4,
            "mean_attribution": 6.0,
            "max_attribution": 8.0,
            "resolved_ig_mode": "content",
            "attribution_feature_space": "content",
            "variant_score_aggregation": "l2",
        },
    }
    return [by_id[key] for key in (order or ["v1", "v2", "v3", "v4"])]


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_rankings(
    path: Path,
    rows: list[dict[str, object]],
    *,
    omit_columns: set[str] | None = None,
) -> None:
    omit_columns = omit_columns or set()
    headers = [
        "variant_id",
        "gene_name",
        "gene_id",
        "chromosome",
        "position",
        "rank",
        "mean_attribution",
        "max_attribution",
        "resolved_ig_mode",
        "attribution_feature_space",
        "variant_score_aggregation",
    ]
    headers = [header for header in headers if header not in omit_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_run(
    tmp_path: Path,
    run_id: str,
    config: dict,
    rows: list[dict[str, object]],
    analysis: dict | None = None,
    omit_ranking_columns: set[str] | None = None,
) -> tuple[str, Path, Path, Path]:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    ranking_path = run_dir / "rankings.csv"
    analysis_path = run_dir / "analysis_metadata.yaml"
    _write_yaml(config_path, config)
    _write_yaml(analysis_path, analysis or _analysis_metadata(config))
    _write_rankings(ranking_path, rows, omit_columns=omit_ranking_columns)
    return run_id, config_path, ranking_path, analysis_path


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runs: list[tuple[str, Path, Path, Path]],
    *,
    score_column: str | None = "rank",
    top_k: str = "1,2,10",
    out_comparison: Path | None = None,
    out_jaccard: Path | None = None,
    out_specific: Path | None = None,
    extra_args: list[str] | None = None,
) -> int:
    argv = [
        "compare_ablation_rankings.py",
        "--comparison-axis",
        "position",
        "--top-k",
        top_k,
        "--high-rank-threshold",
        "1",
        "--low-rank-threshold",
        "2",
        "--out-comparison",
        str(out_comparison or tmp_path / "comparison.yaml"),
        "--out-jaccard",
        str(out_jaccard or tmp_path / "jaccard.tsv"),
        "--out-level-specific",
        str(out_specific or tmp_path / "specific.tsv"),
    ]
    if extra_args:
        argv.extend(extra_args)
    if score_column is not None:
        argv.extend(["--score-column", score_column])
    for run_id, config_path, ranking_path, analysis_path in runs:
        argv.extend(
            [
                "--position-run",
                run_id,
                str(config_path),
                str(ranking_path),
                str(analysis_path),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    return rankings.main()


def test_direct_script_help_supports_position_ranking_options() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/compare_ablation_rankings.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--comparison-axis" in result.stdout
    assert "--position-run" in result.stdout


def test_parser_rejects_invalid_comparison_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_ablation_rankings.py", "--comparison-axis", "genes"],
    )

    with pytest.raises(SystemExit):
        rankings.parse_args()


def test_position_mode_writes_strategy_aware_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "baseline", _config("none"), _ranking_rows())
    second_rows = _ranking_rows(["v3", "v4", "v1", "v2"])
    for rank, row in enumerate(second_rows, start=1):
        row["rank"] = rank
    second = _make_run(
        tmp_path,
        "alibi",
        _config("alibi_fixed", scale=20000.0),
        second_rows,
    )

    assert _run_main(monkeypatch, tmp_path, [first, second]) == 0

    jaccard_rows = list(
        csv.DictReader(
            (tmp_path / "jaccard.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert jaccard_rows[0].keys() == {
        "top_k",
        "run_id_a",
        "run_id_b",
        "position_strategy_id_a",
        "position_strategy_id_b",
        "jaccard",
        "overlap",
        "size_a",
        "size_b",
        "union",
        "score_column",
        "score_sort_order",
    }
    assert jaccard_rows[0]["run_id_a"] == "alibi"
    assert jaccard_rows[0]["run_id_b"] == "baseline"
    assert jaccard_rows[0]["size_a"] == "1"
    assert jaccard_rows[2]["size_a"] == "4"
    assert jaccard_rows[2]["size_b"] == "4"

    summary = yaml.safe_load((tmp_path / "comparison.yaml").read_text(encoding="utf-8"))
    assert summary["comparison_axis"] == "position"
    assert summary["score"] == {"column": "rank", "sort_order": "ascending"}
    assert summary["compatibility"]["training_context"]["compatible"] is True
    assert summary["compatibility"]["explanation_context"]["compatible"] is True
    assert summary["variant_universe"] == {
        "n_variants": 4,
        "key_rule": "explicit_variant_id_else_chromosome_position_gene_id",
    }
    assert isinstance(summary["jaccard_matrices"]["top_1"], list)
    assert [run["run_id"] for run in summary["runs"]] == ["alibi", "baseline"]
    assert summary["runs"][0]["position_strategy"]["relative"]["type"] == "alibi_fixed"
    assert summary["strategy_specific_variant_counts"] == [
        {
            "run_id": "alibi",
            "position_strategy_id": summary["runs"][0]["position_strategy_id"],
            "count": 1,
        },
        {
            "run_id": "baseline",
            "position_strategy_id": summary["runs"][1]["position_strategy_id"],
            "count": 1,
        },
    ]

    specific_rows = list(
        csv.DictReader((tmp_path / "specific.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    assert specific_rows[0].keys() == {
        "variant_id",
        "gene",
        "chrom",
        "pos",
        "specific_to_run_id",
        "specific_to_position_strategy_id",
        "rank_at_specific_strategy",
        "score_at_specific_strategy",
        "other_run_id",
        "other_position_strategy_id",
        "rank_at_other_strategy",
    }
    assert {row["variant_id"] for row in specific_rows} == {"v1", "v3"}


def test_position_mode_creates_independent_nested_output_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second = _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows())
    out_comparison = tmp_path / "summary" / "deep" / "comparison.yaml"
    out_jaccard = tmp_path / "jaccard" / "deep" / "matrix.tsv"
    out_specific = tmp_path / "specific" / "deep" / "variants.tsv"

    assert (
        _run_main(
            monkeypatch,
            tmp_path,
            [first, second],
            out_comparison=out_comparison,
            out_jaccard=out_jaccard,
            out_specific=out_specific,
        )
        == 0
    )

    assert out_comparison.exists()
    assert out_jaccard.exists()
    assert out_specific.exists()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--ranking-dir", "rankings"],
        ["--rankings", "L0:rankings.csv"],
    ],
)
def test_position_mode_rejects_level_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second = _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows())
    rewritten_args = [
        str(tmp_path / value) if value.startswith("rankings") else value for value in extra_args
    ]

    assert (
        _run_main(
            monkeypatch,
            tmp_path,
            [first, second],
            extra_args=rewritten_args,
        )
        == 1
    )

    assert "rejects --ranking-dir and --rankings" in capsys.readouterr().err


def test_level_mode_rejects_position_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--comparison-axis",
            "level",
            "--position-run",
            "run",
            str(tmp_path / "config.yaml"),
            str(tmp_path / "ranking.csv"),
            str(tmp_path / "analysis.yaml"),
        ],
    )

    assert rankings.main() == 1

    assert "level comparison rejects --position-run" in capsys.readouterr().err


def test_position_ranking_csv_provenance_is_checked_against_analysis_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second = _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows())

    assert _run_main(monkeypatch, tmp_path, [first, second]) == 0


def test_position_mode_rejects_missing_ranking_provenance_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _ranking_rows()
    runs = [
        _make_run(
            tmp_path,
            "a",
            _config("none"),
            rows,
            omit_ranking_columns={"variant_score_aggregation"},
        ),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    assert "variant_score_aggregation" in capsys.readouterr().err


def test_position_mode_rejects_ranking_provenance_analysis_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _ranking_rows()
    for row in rows:
        row["variant_score_aggregation"] = "max"
    runs = [
        _make_run(tmp_path, "a", _config("none"), rows),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "run 'a'" in stderr
    assert "variant_score_aggregation" in stderr
    assert "ranking CSV value" in stderr
    assert "analysis metadata value" in stderr


def test_position_mode_rejects_inconsistent_ranking_provenance_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _ranking_rows()
    rows[1]["attribution_feature_space"] = "legacy"
    runs = [
        _make_run(tmp_path, "a", _config("none"), rows),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "run 'a'" in stderr
    assert "attribution_feature_space" in stderr
    assert "inconsistent" in stderr


def test_position_mode_rejects_legacy_ranking_provenance_even_if_analysis_says_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _ranking_rows()
    for row in rows:
        row["resolved_ig_mode"] = "legacy"
    runs = [
        _make_run(tmp_path, "a", _config("none"), rows),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "resolved_ig_mode" in stderr
    assert "ranking CSV value 'legacy'" in stderr


def test_position_score_column_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs, score_column=None) == 1

    assert "explicit --score-column" in capsys.readouterr().err


@pytest.mark.parametrize(
    "score_column",
    [
        "delta_rank",
        "z_attribution",
        "empirical_p_variant",
        "fdr_variant",
        "p_rank_boot",
        "rank_real",
        "median_rank_null_boot",
        "corrected_rank",
    ],
)
def test_position_mode_rejects_calibrated_or_null_derived_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    score_column: str,
) -> None:
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs, score_column=score_column) == 1

    assert "Phase 12C" in capsys.readouterr().err


def test_position_mode_rejects_training_context_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mismatched = _config("alibi_fixed")
    mismatched["seed"] = 43
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", mismatched, _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "position comparison context mismatch" in stderr
    assert "seed" in stderr


def test_position_mode_allows_different_input_dim_per_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second = _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows())

    assert _run_main(monkeypatch, tmp_path, [first, second]) == 0


def test_position_mode_rejects_per_run_config_analysis_link_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config("none")
    analysis = _analysis_metadata(config)
    analysis["integrated_gradients"]["input_dim"] = 999
    runs = [
        _make_run(tmp_path, "a", config, _ranking_rows(), analysis),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "run 'a'" in stderr
    assert "integrated_gradients.input_dim" in stderr
    assert "config value" in stderr
    assert "analysis metadata value" in stderr


@pytest.mark.parametrize(
    "field",
    [
        "annotation_level",
        "genome_build",
        "content_dim",
        "attribution_width",
        "absolute_position_encoding",
        "relative_position_encoding",
        "chromosome_encoding",
        "position_encoding_metadata_source",
    ],
)
def test_position_mode_rejects_representative_per_run_link_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    config = _config("none")
    analysis = _analysis_metadata(config)
    ig = analysis["integrated_gradients"]
    if field == "annotation_level":
        analysis["annotation_level"] = "L2"
    elif field == "genome_build":
        analysis["genome_build"] = "GRCh38"
    elif field == "content_dim":
        ig["content_dim"] = 99
    elif field == "attribution_width":
        ig["attribution_width"] = 99
    elif field == "absolute_position_encoding":
        ig["absolute_position_encoding"] = "sinusoidal"
    elif field == "relative_position_encoding":
        ig["relative_position_encoding"] = "t5_bucket"
    elif field == "chromosome_encoding":
        ig["chromosome_encoding"] = "learned"
    elif field == "position_encoding_metadata_source":
        ig["position_encoding_metadata_source"] = "unavailable"
    else:
        raise AssertionError(field)
    runs = [
        _make_run(tmp_path, "a", config, _ranking_rows(), analysis),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "run 'a'" in stderr
    assert field in stderr


def test_position_mode_rejects_same_wrong_attribution_width_before_cross_run_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_config = _config("none")
    first_analysis = _analysis_metadata(first_config)
    first_analysis["integrated_gradients"]["attribution_width"] = 99
    second_config = _config("alibi_fixed")
    second_analysis = _analysis_metadata(second_config)
    second_analysis["integrated_gradients"]["attribution_width"] = 99
    runs = [
        _make_run(tmp_path, "a", first_config, _ranking_rows(), first_analysis),
        _make_run(tmp_path, "b", second_config, _ranking_rows(), second_analysis),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "run 'a'" in stderr
    assert "integrated_gradients.attribution_width" in stderr
    assert "position explanation context mismatch" not in stderr


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda metadata: metadata["integrated_gradients"].update({"executed": False}),
            "executed",
        ),
        (lambda metadata: metadata.update({"is_null_baseline": True}), "is_null_baseline"),
        (
            lambda metadata: metadata["integrated_gradients"].update(
                {"resolved_ig_mode": "legacy"}
            ),
            "resolved_ig_mode",
        ),
        (
            lambda metadata: metadata["integrated_gradients"].update(
                {"attribution_feature_space": "legacy"}
            ),
            "attribution_feature_space",
        ),
        (
            lambda metadata: metadata["integrated_gradients"].update(
                {"comparability_warning": "old config"}
            ),
            "comparability_warning",
        ),
        (
            lambda metadata: metadata["integrated_gradients"].update(
                {"baseline_policy": "zeros_like_variant_features"}
            ),
            "baseline_policy",
        ),
    ],
)
def test_position_mode_rejects_non_comparable_ig_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutator,
    message: str,
) -> None:
    config = _config("none")
    analysis = _analysis_metadata(config)
    mutator(analysis)
    runs = [
        _make_run(tmp_path, "a", config, _ranking_rows(), analysis),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    assert message in capsys.readouterr().err


def test_position_mode_rejects_explanation_context_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config("alibi_fixed")
    analysis = _analysis_metadata(config)
    analysis["integrated_gradients"]["n_steps"] = 32
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", config, _ranking_rows(), analysis),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "position explanation context mismatch" in stderr
    assert "integrated_gradients.n_steps" in stderr


@pytest.mark.parametrize(
    "field",
    [
        "n_samples",
        "aggregation_method",
        "integrated_gradients.attribution_width",
        "integrated_gradients.variant_score_aggregation",
        "integrated_gradients.n_steps",
        "integrated_gradients.max_variants",
        "integrated_gradients.sampling_policy",
        "integrated_gradients.sampling_seed",
    ],
)
def test_explanation_context_helper_rejects_representative_mismatches(
    field: str,
) -> None:
    first = _analysis_metadata(_config("none"))
    second = _analysis_metadata(_config("alibi_fixed"))
    if field == "n_samples":
        second["n_samples"] = 5
    elif field == "aggregation_method":
        second["aggregation_method"] = "mean"
    else:
        ig_field = field.split(".", 1)[1]
        second["integrated_gradients"][ig_field] = {
            "attribution_width": 99,
            "variant_score_aggregation": "max",
            "n_steps": 32,
            "max_variants": 10,
            "sampling_policy": "random",
            "sampling_seed": 123,
        }[ig_field]

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        require_compatible_explanation_contexts(
            [
                extract_explanation_context(first, run_id="a"),
                extract_explanation_context(second, run_id="b"),
            ]
        )


def test_position_mode_uses_strict_fallback_variant_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows()
    for row in rows:
        row["variant_id"] = ""
    _write_rankings(path, rows)

    loaded, _, _ = rankings.load_position_rankings(
        path,
        run_id="a",
        score_column="rank",
    )

    assert [row["variant_id"] for row in loaded] == [
        "1:100_1",
        "1:200_2",
        "2:300_3",
        "X:400_4",
    ]


def test_position_mode_rejects_missing_gene_id_for_fallback_key(tmp_path: Path) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows()
    rows[0]["variant_id"] = ""
    rows[0]["gene_id"] = ""
    _write_rankings(path, rows)

    with pytest.raises(ValueError, match="run 'a' row 2.*gene_id"):
        rankings.load_position_rankings(path, run_id="a", score_column="rank")


def test_position_mode_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows()
    rows[1]["variant_id"] = "v1"
    _write_rankings(path, rows)

    with pytest.raises(ValueError, match="run 'a'.*duplicate.*v1"):
        rankings.load_position_rankings(path, run_id="a", score_column="rank")


@pytest.mark.parametrize("bad_score", ["", "not-a-number", "nan", "inf", "-inf"])
def test_position_mode_rejects_invalid_scores(tmp_path: Path, bad_score: str) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows()
    rows[0]["mean_attribution"] = bad_score
    _write_rankings(path, rows)

    with pytest.raises(ValueError, match="run 'a'.*v1.*mean_attribution"):
        rankings.load_position_rankings(
            path,
            run_id="a",
            score_column="mean_attribution",
        )


def test_position_mode_tie_breaks_by_variant_id_not_csv_order(tmp_path: Path) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows(["v2", "v1", "v3", "v4"])
    rows[0]["mean_attribution"] = 5.0
    rows[1]["mean_attribution"] = 5.0
    rows[2]["mean_attribution"] = 4.0
    rows[3]["mean_attribution"] = 3.0
    _write_rankings(path, rows)

    loaded, _, sort_order = rankings.load_position_rankings(
        path,
        run_id="a",
        score_column="mean_attribution",
    )

    assert sort_order == "descending"
    assert [row["variant_id"] for row in loaded[:2]] == ["v1", "v2"]


def test_position_mode_sorts_max_attribution_descending(tmp_path: Path) -> None:
    path = tmp_path / "rankings.csv"
    rows = _ranking_rows()
    rows[0]["max_attribution"] = 1.0
    rows[1]["max_attribution"] = 9.0
    rows[2]["max_attribution"] = 5.0
    rows[3]["max_attribution"] = 3.0
    _write_rankings(path, rows)

    loaded, resolved_col, sort_order = rankings.load_position_rankings(
        path,
        run_id="a",
        score_column="max_attribution",
    )

    assert resolved_col == "max_attribution"
    assert sort_order == "descending"
    assert [row["variant_id"] for row in loaded] == ["v2", "v3", "v4", "v1"]


def test_position_mode_exact_variant_universe_allows_different_csv_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second = _make_run(
        tmp_path,
        "b",
        _config("alibi_fixed"),
        _ranking_rows(["v4", "v3", "v2", "v1"]),
    )

    assert _run_main(monkeypatch, tmp_path, [first, second]) == 0


def test_position_mode_jaccard_values_are_exact_at_multiple_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second_rows = _ranking_rows(["v2", "v3", "v1", "v4"])
    for rank, row in enumerate(second_rows, start=1):
        row["rank"] = rank
    second = _make_run(tmp_path, "b", _config("alibi_fixed"), second_rows)

    assert _run_main(monkeypatch, tmp_path, [first, second], top_k="1,2") == 0

    rows = list(csv.DictReader((tmp_path / "jaccard.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert rows[0]["top_k"] == "1"
    assert rows[0]["overlap"] == "0"
    assert rows[0]["union"] == "2"
    assert rows[0]["jaccard"] == "0.0"
    assert rows[1]["top_k"] == "2"
    assert rows[1]["overlap"] == "1"
    assert rows[1]["union"] == "3"
    assert rows[1]["jaccard"] == "0.3333"


def test_position_mode_rejects_variant_universe_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    second_rows = _ranking_rows()
    second_rows[-1]["variant_id"] = "v5"
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", _config("alibi_fixed"), second_rows),
    ]

    assert _run_main(monkeypatch, tmp_path, runs) == 1

    stderr = capsys.readouterr().err
    assert "variant universe mismatch" in stderr
    assert "missing count" in stderr
    assert "extra count" in stderr


def test_position_mode_rejects_duplicate_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    _, config_path, ranking_path, analysis_path = _make_run(
        tmp_path,
        "b",
        _config("alibi_fixed"),
        _ranking_rows(),
    )

    assert (
        _run_main(monkeypatch, tmp_path, [first, ("a", config_path, ranking_path, analysis_path)])
        == 1
    )

    assert "duplicate position run IDs" in capsys.readouterr().err


def test_position_mode_requires_at_least_two_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _make_run(tmp_path, "a", _config("none"), _ranking_rows())

    assert _run_main(monkeypatch, tmp_path, [run]) == 1

    assert "at least two --position-run" in capsys.readouterr().err


def test_position_mode_rejects_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _make_run(tmp_path, "a", _config("none"), _ranking_rows())
    second_run_id, config_path, _, analysis_path = _make_run(
        tmp_path,
        "b",
        _config("alibi_fixed"),
        _ranking_rows(),
    )

    assert (
        _run_main(
            monkeypatch,
            tmp_path,
            [first, (second_run_id, config_path, tmp_path / "missing.csv", analysis_path)],
        )
        == 1
    )

    assert "path not found" in capsys.readouterr().err


def test_position_mode_rejects_non_positive_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = [
        _make_run(tmp_path, "a", _config("none"), _ranking_rows()),
        _make_run(tmp_path, "b", _config("alibi_fixed"), _ranking_rows()),
    ]

    assert _run_main(monkeypatch, tmp_path, runs, top_k="0") == 1

    assert "positive integers" in capsys.readouterr().err
