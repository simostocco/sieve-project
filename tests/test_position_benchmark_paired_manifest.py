"""Tests for Phase 12C3B1 schema-v2 paired real/null benchmark planning."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import create_null_baseline, run_position_benchmark
from scripts.position_benchmark_manifest import (
    SIDECAR_SAMPLE_BINDING,
    BenchmarkManifestError,
    build_human_summary,
    build_resolved_plan,
)
from src.data import null_lineage
from src.data.vcf_parser import SampleVariants, VariantRecord
from src.training.split_plan import (
    build_cv_split_plan,
    build_single_split_plan,
    ordered_sample_ids,
    write_split_plan,
)
from tests.test_position_benchmark_manifest import _base_manifest, _write_yaml

N_SAMPLES = 8


def _samples(prefix: str = "s") -> list[SampleVariants]:
    return [
        SampleVariants(
            f"{prefix}{index}",
            label=index % 2,
            variants=[
                VariantRecord(
                    chrom="1",
                    pos=100 + index,
                    ref="A",
                    alt="T",
                    gene="GENE1",
                    consequence="missense_variant",
                    genotype=1,
                    annotations={"sift": 0.05, "polyphen": 0.9},
                )
            ],
            sex="M",
        )
        for index in range(N_SAMPLES)
    ]


def _split_plan(sample_ids: list[str], mode: str) -> dict:
    if mode == "cv":
        return build_cv_split_plan(
            folds=[([1, 3, 5, 7], [0, 2, 4, 6]), ([0, 2, 4, 6], [1, 3, 5, 7])],
            sample_ids=sample_ids,
            seed=42,
            split_source="generated",
            n_folds=2,
        )
    return build_single_split_plan(
        train_indices=[0, 2, 4, 6],
        val_indices=[1, 3, 5, 7],
        sample_ids=sample_ids,
        seed=42,
        split_source="generated",
    )


def _paired_manifest(tmp_path: Path, *, mode: str = "cv") -> tuple[Path, dict]:
    """Write a valid v2 manifest over a real strict 12C3A null fixture."""
    manifest_path, manifest = _base_manifest(tmp_path, mode=mode)
    real_path = tmp_path / "data" / "cohort.pt"
    samples = _samples()
    torch.save({"samples": samples, "metadata": {"genome_build": "GRCh37"}}, real_path)
    null_path = tmp_path / "data" / "cohort.null.pt"
    create_null_baseline.create_strict_single_permutation(
        str(real_path), str(null_path), seed=7, reuse=False, argv=["create_null_baseline.py"]
    )
    write_split_plan(
        tmp_path / "splits" / "split_plan.yaml",
        _split_plan(ordered_sample_ids(samples), mode),
    )
    manifest["schema_version"] = 2
    manifest["benchmark_id"] = "posenc_l3_paired"
    manifest["training"]["class_weighting"] = "off"
    manifest["null_baseline"] = {"artifact": "data/cohort.null.pt"}
    manifest["calibration"] = {
        "n_bootstrap": 1000,
        "seed": 42,
        "top_k": [50, 100, 200],
        "exclude_sex_chroms": False,
        "min_variants_per_gene": 10,
        "gene_delta_rank_aggregation": "max",
    }
    manifest["runtime"]["calibration_n_jobs"] = 2
    _write_yaml(manifest_path, manifest)
    return manifest_path, manifest


def _build(manifest_path: Path, **kwargs) -> dict:
    return build_resolved_plan(
        manifest_path, python_override=sys.executable, device_override="cpu", **kwargs
    )


def _rewrite(manifest_path: Path, manifest: dict, mutate) -> None:
    manifest = copy.deepcopy(manifest)
    mutate(manifest)
    _write_yaml(manifest_path, manifest)


def _argv_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _differing_positions(left: list[str], right: list[str]) -> list[int]:
    assert len(left) == len(right)
    return [index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]


# ---------------------------------------------------------------------------
# Null block and shared null binding
# ---------------------------------------------------------------------------


def test_v2_plan_records_one_benchmark_level_null_binding(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)

    null_path = (tmp_path / "data" / "cohort.null.pt").resolve()
    real_path = (tmp_path / "data" / "cohort.pt").resolve()
    sidecar = null_lineage.load_sidecar(null_lineage.sidecar_path_for(null_path))
    binding = plan["null_binding"]
    assert plan["schema_version"] == 2
    assert binding["lineage_sha256"] == sidecar["lineage_sha256"]
    assert binding["source_artifact_sha256"] == null_lineage.compute_file_sha256(real_path)
    assert binding["null_artifact_sha256"] == null_lineage.compute_file_sha256(null_path)
    assert binding["sample_ids_sha256"] == plan["split_plan"]["sample_ids_sha256"]
    assert binding["n_samples"] == N_SAMPLES
    assert binding["sidecar_path"] == str(null_lineage.sidecar_path_for(null_path))
    assert binding["execution_authorized"] is False
    assert plan["null_execution"]["execution_authorized"] is False
    assert plan["split_plan"]["dataset_sample_binding_validation"] == SIDECAR_SAMPLE_BINDING
    for run in plan["runs"]:
        assert _argv_after(run["null_train_argv"], "--preprocessed-data") == str(null_path)
        assert _argv_after(run["null_explain_argv"], "--preprocessed-data") == str(null_path)
        assert _argv_after(run["train_argv"], "--preprocessed-data") == str(real_path)


def test_v2_plan_generation_is_repeatable_and_deterministic(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)

    assert _build(manifest_path) == _build(manifest_path)


def test_planner_never_unpickles_cohort_artifacts(tmp_path, monkeypatch):
    manifest_path, _ = _paired_manifest(tmp_path)

    def _forbidden(*args, **kwargs):
        raise AssertionError("dry-run planning must not call torch.load")

    monkeypatch.setattr(torch, "load", _forbidden)
    plan = _build(manifest_path)

    assert plan["null_binding"]["validation_level"] == "sidecar_schema_and_file_sha256_only"


def test_strategy_level_null_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(
        manifest_path,
        manifest,
        lambda m: m["runs"][1].update({"null_baseline": {"artifact": "data/cohort.null.pt"}}),
    )

    with pytest.raises(BenchmarkManifestError, match="benchmark-level manifest.null_baseline"):
        _build(manifest_path)


def test_strategy_level_plain_null_key_also_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(
        manifest_path,
        manifest,
        lambda m: m["runs"][1].update({"null": {"artifact": "data/cohort.null.pt"}}),
    )

    with pytest.raises(BenchmarkManifestError, match=r"runs\[1\]\.null is not allowed"):
        _build(manifest_path)


@pytest.mark.parametrize(
    "key",
    [
        "lineage_sidecar",
        "permutation_seed",
        "lineage_sha256",
        "source_artifact_sha256",
        "sample_ids_sha256",
        "null_artifact_sha256",
        "source_dataset",
    ],
)
def test_null_block_rejects_restated_sidecar_identity(tmp_path, key):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["null_baseline"].update({key: "x"}))

    with pytest.raises(BenchmarkManifestError, match="must not restate"):
        _build(manifest_path)


def test_null_block_rejects_unknown_key(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["null_baseline"].update({"enabled": True}))

    with pytest.raises(BenchmarkManifestError, match="exactly artifact"):
        _build(manifest_path)


@pytest.mark.parametrize("block", ["null_baseline", "calibration"])
def test_v2_requires_null_and_calibration_blocks(tmp_path, block):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m.pop(block))

    with pytest.raises(BenchmarkManifestError, match=f"manifest.{block} is required"):
        _build(manifest_path)


def test_sidecar_path_is_derived_and_missing_sidecar_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    null_lineage.sidecar_path_for(tmp_path / "data" / "cohort.null.pt").unlink()

    with pytest.raises(BenchmarkManifestError, match="deterministic path"):
        _build(manifest_path)


def test_source_sha_mismatch_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    with (tmp_path / "data" / "cohort.pt").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(BenchmarkManifestError, match="source.sha256"):
        _build(manifest_path)


def test_null_sha_mismatch_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    with (tmp_path / "data" / "cohort.null.pt").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(BenchmarkManifestError, match="null.sha256"):
        _build(manifest_path)


def _mutate_sidecar(tmp_path: Path, mutate) -> None:
    sidecar_path = null_lineage.sidecar_path_for(tmp_path / "data" / "cohort.null.pt")
    sidecar = null_lineage.load_sidecar(sidecar_path)
    mutate(sidecar)
    null_lineage.write_sidecar(sidecar_path, sidecar)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("lineage_sha256"),
        lambda s: s["samples"].pop("sample_ids_sha256"),
        lambda s: s["permutation"].update({"seed": True}),
        lambda s: s.update({"schema_version": 2}),
    ],
)
def test_malformed_sidecar_schema_rejects(tmp_path, mutate):
    manifest_path, _ = _paired_manifest(tmp_path)
    _mutate_sidecar(tmp_path, mutate)

    with pytest.raises(BenchmarkManifestError, match="sidecar schema is invalid"):
        _build(manifest_path)


def test_malformed_sidecar_yaml_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    sidecar_path = null_lineage.sidecar_path_for(tmp_path / "data" / "cohort.null.pt")
    sidecar_path.write_text("lineage: [", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="malformed"):
        _build(manifest_path)


def test_sidecar_sample_ids_mismatch_with_split_plan_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    other_ids = [f"other{index}" for index in range(N_SAMPLES)]
    write_split_plan(tmp_path / "splits" / "split_plan.yaml", _split_plan(other_ids, "cv"))

    with pytest.raises(BenchmarkManifestError, match="sample_ids_sha256"):
        _build(manifest_path)


def test_sidecar_n_samples_mismatch_with_split_plan_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    _mutate_sidecar(tmp_path, lambda s: s["samples"].update({"n_samples": N_SAMPLES + 1}))

    with pytest.raises(BenchmarkManifestError, match="n_samples"):
        _build(manifest_path)


def test_null_artifact_equal_to_real_dataset_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(
        manifest_path, manifest, lambda m: m["null_baseline"].update({"artifact": "data/cohort.pt"})
    )

    with pytest.raises(BenchmarkManifestError, match="must differ"):
        _build(manifest_path)


def test_missing_null_artifact_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(
        manifest_path, manifest, lambda m: m["null_baseline"].update({"artifact": "data/none.pt"})
    )

    with pytest.raises(BenchmarkManifestError, match="manifest.null_baseline.artifact"):
        _build(manifest_path)


# ---------------------------------------------------------------------------
# Paired protocol restrictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "on"])
def test_class_weighting_other_than_off_rejects(tmp_path, value):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["training"].update({"class_weighting": value}))

    with pytest.raises(BenchmarkManifestError, match="class_weighting must be 'off'"):
        _build(manifest_path)


def test_class_weighting_yaml_boolean_off_gets_quoting_hint(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    text = yaml.safe_dump(manifest, sort_keys=False).replace(
        "class_weighting: 'off'", "class_weighting: off"
    )
    manifest_path.write_text(text, encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="boolean false"):
        _build(manifest_path)


def test_paired_cv_fold_index_other_than_zero_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["explanation"].update({"fold_index": 1}))

    with pytest.raises(BenchmarkManifestError, match="fold_index must be 0"):
        _build(manifest_path)


def test_paired_single_split_is_allowed_without_fold_index(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path, mode="single_split")
    plan = _build(manifest_path)

    for run in plan["runs"]:
        assert "--fold-index" not in run["explain_argv"]
        assert "--fold-index" not in run["null_explain_argv"]
        assert "--cv" not in run["null_train_argv"]
    assert plan["paired_policy"]["checkpoint_selection"] == "single_run_best_model"


# ---------------------------------------------------------------------------
# Real/null command pairing
# ---------------------------------------------------------------------------


def test_real_and_null_train_argv_differ_only_in_dataset_and_output_dir(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)

    for run in plan["runs"]:
        real, null = run["train_argv"], run["null_train_argv"]
        positions = _differing_positions(real, null)
        assert [real[index - 1] for index in positions] == ["--preprocessed-data", "--output-dir"]
        assert _argv_after(null, "--output-dir") == str(
            Path(run["directories"]["run_root"]) / "null"
        )
        assert _argv_after(null, "--experiment-name") == "training"
        for flag in ("--split-plan", "--seed", "--cv", "--class-weighting", "--position-preset"):
            assert _argv_after(real, flag) == _argv_after(null, flag)
        assert _argv_after(null, "--class-weighting") == "off"
        assert _argv_after(null, "--split-plan") == str(
            (tmp_path / "splits" / "split_plan.yaml").resolve()
        )
        assert "--is-null-baseline" not in null


def test_real_and_null_explain_argv_differ_only_where_expected(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)

    for run in plan["runs"]:
        real, null = run["explain_argv"], run["null_explain_argv"]
        assert null[-1] == "--is-null-baseline"
        assert "--is-null-baseline" not in real
        positions = _differing_positions(real, null[:-1])
        assert [real[index - 1] for index in positions] == [
            "--experiment-dir",
            "--preprocessed-data",
            "--output-dir",
        ]
        assert _argv_after(null, "--experiment-dir") == run["directories"]["null_training"]
        assert _argv_after(null, "--output-dir") == run["directories"]["null_explanation"]
        for argv in (real, null):
            assert _argv_after(argv, "--fold-index") == "0"
            assert _argv_after(argv, "--ig-mode") == "content"
            assert _argv_after(argv, "--n-steps") == "50"
            assert _argv_after(argv, "--max-variants") == "2000"
            assert _argv_after(argv, "--aggregation-method") == "mean"
            assert "--checkpoint" not in argv


def test_paired_directories_follow_real_null_calibration_layout(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)

    for run in plan["runs"]:
        root = Path(run["directories"]["run_root"])
        assert run["directories"] == {
            "run_root": str(root),
            "training": str(root / "real" / "training"),
            "explanation": str(root / "real" / "explanation"),
            "null_training": str(root / "null" / "training"),
            "null_explanation": str(root / "null" / "explanation"),
            "calibration": str(root / "calibration"),
        }
        assert run["expected_artifacts"]["null_training"][-1] == str(
            root / "null" / "training" / "fold_0" / "best_model.pt"
        )
    # Phase 12C3C: the former reserved placeholder is now a planned comparison.
    assert "reserved_comparisons" not in plan
    assert list(plan["comparisons"]) == [
        "performance",
        "raw_rankings",
        "raw_attributions",
        "calibrated_rankings",
    ]


def test_v2_plan_contains_calibrated_rankings_comparison(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)
    root = Path(plan["runs"][0]["directories"]["run_root"]).parent.parent
    directory = root / "comparisons" / "calibrated_rankings"
    outputs = [
        directory / "position_calibrated_ranking_comparison.yaml",
        directory / "position_calibrated_ranking_jaccard.tsv",
        directory / "position_calibrated_strategy_specific_variants.tsv",
    ]

    assert plan["comparisons"]["calibrated_rankings"] == {
        "directory": str(directory),
        "score_column": "delta_rank",
        "argv": [
            plan["runtime"]["python"],
            str(Path(plan["repository_root"]) / "scripts" / "compare_ablation_rankings.py"),
            "--comparison-axis",
            "position",
            "--position-calibrated-benchmark",
            str(root),
            "--score-column",
            "delta_rank",
            "--out-comparison",
            str(outputs[0]),
            "--out-jaccard",
            str(outputs[1]),
            "--out-level-specific",
            str(outputs[2]),
        ],
        "expected_outputs": [str(path) for path in outputs],
    }
    argv = plan["comparisons"]["calibrated_rankings"]["argv"]
    # Same comparator defaults as the raw ranking comparison: no threshold flags.
    raw = plan["comparisons"]["raw_rankings"]["argv"]
    for flag in ("--top-k", "--high-rank-threshold", "--low-rank-threshold"):
        assert flag not in argv and flag not in raw
    assert "--position-run" not in argv


def test_non_empty_calibrated_comparison_directory_fails_or_warns(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    directory = Path(_build(manifest_path)["comparisons"]["calibrated_rankings"]["directory"])
    directory.mkdir(parents=True)
    (directory / "stale.tsv").write_text("x", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="non-empty"):
        _build(manifest_path)
    warnings = _build(manifest_path, allow_existing_outputs=True)["warnings"]
    assert warnings == [f"planned output directory is non-empty: {directory}"]


def test_calibrated_comparison_directory_that_is_a_file_rejects(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    directory = Path(_build(manifest_path)["comparisons"]["calibrated_rankings"]["directory"])
    directory.parent.mkdir(parents=True)
    directory.write_text("not a directory", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="existing file"):
        _build(manifest_path, allow_existing_outputs=True)


def test_null_artifact_inside_calibrated_comparison_directory_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    leaf = Path(_build(manifest_path)["comparisons"]["calibrated_rankings"]["directory"])
    leaf.mkdir(parents=True)
    for name in ("cohort.null.pt", "cohort.null.pt.null-lineage.yaml"):
        (leaf / name).write_bytes((tmp_path / "data" / name).read_bytes())
    relative = leaf.relative_to(tmp_path) / "cohort.null.pt"
    _rewrite(
        manifest_path, manifest, lambda m: m["null_baseline"].update({"artifact": str(relative)})
    )

    with pytest.raises(BenchmarkManifestError, match="null_baseline.artifact path collides"):
        _build(manifest_path, allow_existing_outputs=True)


def test_calibration_argv_is_explicit_and_deterministic(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)
    plan = _build(manifest_path)
    run = plan["runs"][0]
    root = Path(run["directories"]["run_root"])
    calibration_dir = root / "calibration"

    assert run["calibration"]["argv"] == [
        plan["runtime"]["python"],
        str(Path(plan["repository_root"]) / "scripts" / "bootstrap_null_calibration.py"),
        "--real-rankings",
        str(root / "real" / "explanation" / "sieve_variant_rankings.csv"),
        "--null-attributions",
        str(root / "null" / "explanation" / "attributions.npz"),
        "--output",
        str(calibration_dir / "bootstrap_calibrated_variant_rankings.csv"),
        "--output-gene-stats",
        str(calibration_dir / "bootstrap_calibrated_variant_rankings_gene_stats.csv"),
        "--output-summary",
        str(calibration_dir / "bootstrap_calibrated_variant_rankings_summary.yaml"),
        "--n-bootstrap",
        "1000",
        "--seed",
        "42",
        "--top-k",
        "50,100,200",
        "--min-variants-per-gene",
        "10",
        "--gene-delta-rank-aggregation",
        "max",
        "--genome-build",
        "GRCh37",
        "--n-jobs",
        "2",
    ]
    assert run["calibration"]["paired_compatibility"] == str(
        calibration_dir / "paired_compatibility.yaml"
    )
    assert run["calibration"]["status"] == "planned_not_executed"
    assert plan["calibration"]["delta_rank"] == "median_rank_null_boot - rank_real"


def test_calibration_exclude_sex_chroms_flag_is_emitted_only_when_true(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    assert "--exclude-sex-chroms" not in _build(manifest_path)["runs"][0]["calibration"]["argv"]

    _rewrite(
        manifest_path, manifest, lambda m: m["calibration"].update({"exclude_sex_chroms": True})
    )
    assert _build(manifest_path)["runs"][0]["calibration"]["argv"][-1] == "--exclude-sex-chroms"


@pytest.mark.parametrize(
    ("top_k", "message"),
    [
        ([], "non-empty list"),
        ("50,100", "non-empty list"),
        ([50, 0], "positive"),
        ([50, True], "integer"),
        ([50, 1.5], "integer"),
        ([50, 50], "duplicates"),
    ],
)
def test_calibration_top_k_is_validated(tmp_path, top_k, message):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["calibration"].update({"top_k": top_k}))

    with pytest.raises(BenchmarkManifestError, match=message):
        _build(manifest_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.pop("seed"), "missing required keys"),
        (lambda c: c.update({"n_jobs": 4}), "unknown keys"),
        (lambda c: c.update({"n_bootstrap": 0}), "positive"),
        (lambda c: c.update({"seed": -1}), "non-negative"),
        (lambda c: c.update({"exclude_sex_chroms": "no"}), "bool"),
        (lambda c: c.update({"gene_delta_rank_aggregation": "sum"}), "one of"),
    ],
)
def test_calibration_block_is_validated(tmp_path, mutate, message):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: mutate(m["calibration"]))

    with pytest.raises(BenchmarkManifestError, match=message):
        _build(manifest_path)


@pytest.mark.parametrize("value", [0, -2, True, "4"])
def test_calibration_n_jobs_runtime_is_validated(tmp_path, value):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m["runtime"].update({"calibration_n_jobs": value}))

    with pytest.raises(BenchmarkManifestError, match="calibration_n_jobs"):
        _build(manifest_path)


# ---------------------------------------------------------------------------
# Raw B1/B2/B3 remain real-only and identical to v1
# ---------------------------------------------------------------------------


def test_v2_real_side_and_raw_comparisons_match_equivalent_v1_plan(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    v1 = copy.deepcopy(manifest)
    v1["schema_version"] = 1
    for key in ("null_baseline", "calibration"):
        v1.pop(key)
    v1["runtime"].pop("calibration_n_jobs")
    v1_path = _write_yaml(tmp_path / "manifest_v1.yaml", v1)

    paired = _build(manifest_path)
    real_only = _build(v1_path)

    raw_names = ["performance", "raw_rankings", "raw_attributions"]
    assert list(real_only["comparisons"]) == raw_names
    assert {name: paired["comparisons"][name] for name in raw_names} == real_only["comparisons"]
    assert "null_binding" not in real_only
    for paired_run, real_run in zip(paired["runs"], real_only["runs"], strict=True):
        assert paired_run["train_argv"] == real_run["train_argv"]
        assert paired_run["explain_argv"] == real_run["explain_argv"]
        assert "null_train_argv" not in real_run
        assert "calibration" not in real_run
    for argv in (
        paired["comparisons"]["raw_rankings"]["argv"],
        paired["comparisons"]["raw_attributions"]["argv"],
        paired["comparisons"]["performance"]["argv"],
    ):
        assert not any("/null/" in token for token in argv)
        assert "delta_rank" not in argv


def test_v1_runtime_still_rejects_calibration_n_jobs(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["runtime"]["calibration_n_jobs"] = 1
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="unknown keys"):
        _build(manifest_path)


def test_v1_manifest_with_calibration_block_still_rejects(tmp_path):
    manifest_path, manifest = _base_manifest(tmp_path)
    manifest["calibration"] = {}
    _write_yaml(manifest_path, manifest)

    with pytest.raises(BenchmarkManifestError, match="unknown top-level"):
        _build(manifest_path)


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf", ["null_training", "null_explanation", "calibration"])
def test_non_empty_paired_leaf_fails_or_warns(tmp_path, leaf):
    manifest_path, _ = _paired_manifest(tmp_path)
    leaf_path = Path(_build(manifest_path)["runs"][0]["directories"][leaf])
    leaf_path.mkdir(parents=True)
    (leaf_path / "stale.txt").write_text("x", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="non-empty"):
        _build(manifest_path)
    assert any(
        str(leaf_path) in warning
        for warning in _build(manifest_path, allow_existing_outputs=True)["warnings"]
    )


def test_null_artifact_inside_planned_output_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    leaf = Path(_build(manifest_path)["runs"][0]["directories"]["calibration"])
    leaf.mkdir(parents=True)
    for name in ("cohort.null.pt", "cohort.null.pt.null-lineage.yaml"):
        (leaf / name).write_bytes((tmp_path / "data" / name).read_bytes())
    relative = leaf.relative_to(tmp_path) / "cohort.null.pt"
    _rewrite(
        manifest_path, manifest, lambda m: m["null_baseline"].update({"artifact": str(relative)})
    )

    with pytest.raises(BenchmarkManifestError, match="null_baseline.artifact path collides"):
        _build(manifest_path, allow_existing_outputs=True)


def test_output_root_equal_to_null_artifact_rejects(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(
        manifest_path, manifest, lambda m: m["paths"].update({"output_root": "data/cohort.null.pt"})
    )

    with pytest.raises(BenchmarkManifestError, match="output_root is an existing file"):
        _build(manifest_path)


def test_v2_rejects_top_level_plain_null_block(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    _rewrite(manifest_path, manifest, lambda m: m.update({"null": m.pop("null_baseline")}))

    with pytest.raises(BenchmarkManifestError, match="unknown top-level keys: \\['null'\\]"):
        _build(manifest_path)


def test_v2_rejects_bare_yaml_null_key_without_crashing(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    text = yaml.safe_dump(manifest, sort_keys=False).replace("null_baseline:", "null:")
    manifest_path.write_text(text, encoding="utf-8")
    assert None in yaml.safe_load(text)

    with pytest.raises(BenchmarkManifestError, match="unknown top-level keys: \\[None\\]"):
        _build(manifest_path)


def test_null_baseline_block_needs_no_yaml_quoting(tmp_path):
    manifest_path, manifest = _paired_manifest(tmp_path)
    text = manifest_path.read_text(encoding="utf-8")

    assert "\nnull_baseline:\n" in text
    assert "'null_baseline'" not in text
    assert yaml.safe_load(text)["null_baseline"] == {"artifact": "data/cohort.null.pt"}
    assert _build(manifest_path)["null_binding"]["n_samples"] == N_SAMPLES


def test_dry_run_cli_prints_paired_plan_without_creating_outputs(tmp_path, capsys):
    manifest_path, _ = _paired_manifest(tmp_path)

    code = run_position_benchmark.main(
        [str(manifest_path), "--dry-run", "--python", sys.executable, "--device", "cpu"]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Null lineage SHA256:" in output
    assert "null train:" in output
    assert "calibration:" in output
    assert "  calibrated_rankings: " in output
    assert "--position-calibrated-benchmark" in output
    assert "not authorized until the Phase 12C3B2" in output
    assert not (tmp_path / "outputs").exists()


def test_paired_human_summary_is_deterministic(tmp_path):
    manifest_path, _ = _paired_manifest(tmp_path)

    assert build_human_summary(_build(manifest_path)) == build_human_summary(_build(manifest_path))


def test_example_paired_manifest_uses_null_baseline_and_quoted_off():
    example = Path(__file__).resolve().parent.parent / (
        "documentation/examples/position-benchmark-L3-paired.yaml"
    )
    data = yaml.safe_load(example.read_text(encoding="utf-8"))

    assert data["schema_version"] == 2
    assert data["training"]["class_weighting"] == "off"
    assert data["explanation"]["fold_index"] == 0
    assert data["null_baseline"] == {"artifact": "../data/cohort.null.pt"}
    assert "null" not in data and None not in data
    assert "null_baseline:\n  artifact:" in example.read_text(encoding="utf-8")
    assert all("null" not in run and "null_baseline" not in run for run in data["runs"])


def test_example_paired_manifest_plans_end_to_end_over_fixture_files(tmp_path):
    example = Path(__file__).resolve().parent.parent / (
        "documentation/examples/position-benchmark-L3-paired.yaml"
    )
    manifest_path = tmp_path / "examples" / "position-benchmark-L3-paired.yaml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "data").mkdir()
    samples = _samples() + [
        SampleVariants(f"t{index}", label=index % 2, variants=[], sex="F") for index in range(2)
    ]
    real_path = tmp_path / "data" / "cohort.preprocessed.pt"
    torch.save({"samples": samples, "metadata": {"genome_build": "GRCh37"}}, real_path)
    create_null_baseline.create_strict_single_permutation(
        str(real_path),
        str(tmp_path / "data" / "cohort.null.pt"),
        seed=7,
        reuse=False,
        argv=["create_null_baseline.py"],
    )
    folds = [
        (
            [i for i in range(10) if i not in (fold, fold + 5)],
            [fold, fold + 5],
        )
        for fold in range(5)
    ]
    write_split_plan(
        tmp_path / "splits" / "l3_primary_split_plan.yaml",
        build_cv_split_plan(
            folds=folds,
            sample_ids=ordered_sample_ids(samples),
            seed=42,
            split_source="generated",
            n_folds=5,
        ),
    )

    plan = _build(manifest_path)

    assert plan["schema_version"] == 2
    assert len(plan["runs"]) == 8
    assert len({_argv_after(run["calibration"]["argv"], "--output") for run in plan["runs"]}) == 8
    assert {_argv_after(run["null_train_argv"], "--preprocessed-data") for run in plan["runs"]} == {
        plan["null_binding"]["null_artifact_path"]
    }
