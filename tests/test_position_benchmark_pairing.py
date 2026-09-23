"""Tests for the Phase 12C3B1 real/null pair compatibility validator."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.position_benchmark_pairing import (
    PairCompatibilityError,
    PairSide,
    compare_real_null_pair,
    load_pair_side,
    require_compatible_real_null_pair,
    require_shared_null_across_pairs,
    variant_universe_sha256,
)
from tests.test_position_benchmark_rankings import _analysis_metadata, _config

SOURCE_SHA = "1" * 64
NULL_SHA = "2" * 64
LINEAGE_SHA = "3" * 64
IDS_SHA = "4" * 64
SPLIT_SHA = "5" * 64
UNIVERSE_SHA = "6" * 64
REVISION = "a" * 40
N_SAMPLES = 4

BINDING = {
    "schema_version": 1,
    "lineage_sha256": LINEAGE_SHA,
    "source_artifact_sha256": SOURCE_SHA,
    "null_artifact_sha256": NULL_SHA,
    "sample_ids_sha256": IDS_SHA,
    "n_samples": N_SAMPLES,
    "null_artifact_path": "/data/cohort.null.pt",
    "sidecar_path": "/data/cohort.null.pt.null-lineage.yaml",
}


def _dataset_provenance(side: str) -> dict:
    if side == "real":
        return {
            "schema_version": 1,
            "preprocessed_data_path": "/data/cohort.pt",
            "preprocessed_data_sha256": SOURCE_SHA,
            "sample_ids_sha256": IDS_SHA,
            "is_null_baseline": False,
            "null_metadata_kind": "none",
            "null_lineage": None,
        }
    return {
        "schema_version": 1,
        "preprocessed_data_path": "/data/cohort.null.pt",
        "preprocessed_data_sha256": NULL_SHA,
        "sample_ids_sha256": IDS_SHA,
        "is_null_baseline": True,
        "null_metadata_kind": "strict_v1",
        "null_lineage": {
            "lineage_sha256": LINEAGE_SHA,
            "source_artifact_sha256": SOURCE_SHA,
            "permutation_indices_sha256": "7" * 64,
            "original_labels_sha256": "8" * 64,
            "permuted_labels_sha256": "9" * 64,
        },
    }


def _side_config(side: str, relative_type: str = "none") -> dict:
    config = _config(relative_type)
    config["class_weighting"] = "off"
    config["preprocessed_data"] = f"/data/{side}.pt"
    config["split_plan"] = {
        "schema_version": 1,
        "source": "replayed",
        "path": f"/bench/runs/x/{side}/training/split_plan.yaml",
        "sha256": SPLIT_SHA,
        "sample_ids_sha256": IDS_SHA,
        "input_path": "/splits/split_plan.yaml",
        "input_sha256": SPLIT_SHA,
    }
    config["dataset_provenance"] = _dataset_provenance(side)
    return config


def _side_analysis(config: dict, side: str) -> dict:
    analysis = _analysis_metadata(config)
    analysis.update(
        {
            "is_null_baseline": side == "null",
            "n_samples": N_SAMPLES,
            "max_variants_per_sample": 2000,
            "skip_ig": False,
            "skip_attention": False,
            "attention_threshold_mode": "absolute",
            "attention_threshold": 0.1,
            "attention_percentile": 99.9,
            "model_provenance": {
                "schema_version": 1,
                "checkpoint_selection_mode": "cv_explicit_fold",
                "checkpoint_path": f"/bench/runs/x/{side}/training/fold_0/best_model.pt",
                "checkpoint_sha256": ("c" if side == "real" else "d") * 64,
                "config_path": f"/bench/runs/x/{side}/training/config.yaml",
                "selected_fold": 0,
                "selected_fold_auc": 0.71 if side == "real" else 0.49,
                "cv_results_path": f"/bench/runs/x/{side}/training/cv_results.yaml",
            },
            "dataset_provenance": copy.deepcopy(config["dataset_provenance"]),
        }
    )
    return analysis


def _side(side: str, relative_type: str = "none") -> PairSide:
    config = _side_config(side, relative_type)
    return PairSide(
        config=config,
        analysis_metadata=_side_analysis(config, side),
        variant_universe_sha256=UNIVERSE_SHA,
        repository_revision=REVISION,
    )


def _replace(side: PairSide, **changes) -> PairSide:
    values = {
        "config": copy.deepcopy(dict(side.config)),
        "analysis_metadata": copy.deepcopy(dict(side.analysis_metadata)),
        "variant_universe_sha256": side.variant_universe_sha256,
        "repository_revision": side.repository_revision,
    }
    values.update(changes)
    return PairSide(**values)


def _set(mapping: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        mapping = mapping[part]
    mapping[parts[-1]] = value


def _report(real: PairSide, null: PairSide, **kwargs) -> dict:
    return compare_real_null_pair(
        run_id=kwargs.pop("run_id", "no_position"),
        real=real,
        null=null,
        null_binding=kwargs.pop("null_binding", BINDING),
        expected_fold_index=kwargs.pop("expected_fold_index", 0),
    )


def _failed_fields(report: dict) -> set[str]:
    return {mismatch["field"] for mismatch in report["mismatches"]}


# ---------------------------------------------------------------------------
# Matched pair
# ---------------------------------------------------------------------------


def test_matched_pair_passes_with_deterministic_serializable_report():
    report = require_compatible_real_null_pair(
        run_id="no_position",
        real=_side("real"),
        null=_side("null"),
        null_binding=BINDING,
        expected_fold_index=0,
    )

    assert report["compatible"] is True
    assert report["mismatches"] == []
    assert report["null_binding"] == {
        key: BINDING[key]
        for key in (
            "lineage_sha256",
            "source_artifact_sha256",
            "null_artifact_sha256",
            "sample_ids_sha256",
            "n_samples",
        )
    }
    assert report["shared"]["fold_index"] == 0
    assert report["shared"]["training_seed"] == 42
    assert report["shared"]["split_plan_sha256"] == SPLIT_SHA
    assert report["real"]["checkpoint_sha256"] != report["null"]["checkpoint_sha256"]
    assert report == _report(_side("real"), _side("null"))
    assert yaml.safe_load(yaml.safe_dump(report, sort_keys=False)) == report
    assert "config.preprocessed_data" not in report["checked_rules"]


def test_single_split_pair_uses_single_run_selection():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.config["position_encoding_execution"]["training_mode"] = "single_split"
        side.config["cv"] = None
        side.analysis_metadata["model_provenance"].update(
            {"checkpoint_selection_mode": "single_run_best_model", "selected_fold": None}
        )

    assert _report(real, null, expected_fold_index=None)["compatible"] is True
    assert _report(real, null, expected_fold_index=0)["compatible"] is False


# ---------------------------------------------------------------------------
# Controlled-configuration mismatches (each independently rejects)
# ---------------------------------------------------------------------------

CONFIG_MUTATIONS = [
    ("level", "L2", "config.level"),
    ("position_encoding.chromosome.mapping", {"0": "1"}, "config.position_encoding"),
    ("position_encoding_execution.source", "other", "config.position_encoding_execution"),
    ("latent_dim", 32, "config.latent_dim"),
    ("hidden_dim", 64, "config.hidden_dim"),
    ("num_heads", 2, "config.num_heads"),
    ("num_attention_layers", 1, "config.num_attention_layers"),
    ("aggregation_method", "max", "config.aggregation_method"),
    ("classifier_type", "attention_pool", "config.classifier_type"),
    ("num_covariates", 1, "config.num_covariates"),
    ("lr", 0.01, "config.lr"),
    ("epochs", 5, "config.epochs"),
    ("batch_size", 8, "config.batch_size"),
    ("early_stopping", 3, "config.early_stopping"),
    ("gradient_clip", 1.0, "config.gradient_clip"),
    ("gradient_accumulation_steps", 2, "config.gradient_accumulation_steps"),
    ("lambda_attr", 0.1, "config.lambda_attr"),
    ("chunk_size", 100, "config.chunk_size"),
    ("chunk_overlap", 10, "config.chunk_overlap"),
    ("seed", 7, "config.seed"),
    ("cv", 3, "config.cv"),
    ("val_split", 0.3, "config.val_split"),
    ("input_dim", 99, "config.input_dim"),
    ("num_genes", 99, "config.num_genes"),
    ("split_plan.sha256", "0" * 64, "config.split_plan.sha256"),
    ("split_plan.input_sha256", "0" * 64, "config.split_plan.input_sha256"),
    ("split_plan.source", "generated", "config.split_plan.source"),
    (
        "dataset_identity.gene_mapping_sha256",
        "other",
        "config.dataset_identity.gene_mapping_sha256",
    ),
    (
        "dataset_identity.chromosome_mapping_sha256",
        "other",
        "config.dataset_identity.chromosome_mapping_sha256",
    ),
    ("dataset_identity.genome_build", "GRCh38", "config.dataset_identity.genome_build"),
    ("sex_map", "/maps/sex.tsv", "config.sex_map"),
    ("pc_map_sha256", "0" * 64, "config.pc_map_sha256"),
    ("num_pcs", 2, "config.num_pcs"),
    ("class_weighting", "auto", "config.class_weighting"),
]


@pytest.mark.parametrize(("dotted", "value", "field"), CONFIG_MUTATIONS)
def test_null_config_mismatch_rejects(dotted, value, field):
    null = _side("null")
    _set(null.config, dotted, value)
    report = _report(_side("real"), null)

    assert report["compatible"] is False
    assert field in _failed_fields(report)
    with pytest.raises(PairCompatibilityError, match="incompatible"):
        require_compatible_real_null_pair(
            run_id="no_position",
            real=_side("real"),
            null=null,
            null_binding=BINDING,
            expected_fold_index=0,
        )


def test_strategy_identity_mismatch_rejects():
    report = _report(_side("real"), _side("null", relative_type="t5_bucket"))

    assert {"position_strategy_identity.hash", "config.position_encoding"} <= _failed_fields(report)


ANALYSIS_MUTATIONS = [
    ("annotation_level", "L2", "analysis.annotation_level"),
    ("n_samples", 5, "analysis.n_samples"),
    ("aggregation_method", "mean", "analysis.aggregation_method"),
    ("genome_build", "GRCh38", "analysis.genome_build"),
    ("max_variants_per_sample", 100, "analysis.max_variants_per_sample"),
    ("skip_attention", True, "analysis.skip_attention"),
    ("attention_threshold", 0.5, "analysis.attention_threshold"),
    ("integrated_gradients.resolved_ig_mode", "legacy", "analysis.integrated_gradients"),
    ("integrated_gradients.n_steps", 32, "analysis.integrated_gradients.n_steps"),
    ("integrated_gradients.max_variants", 50, "analysis.integrated_gradients.max_variants"),
    ("integrated_gradients.sampling_seed", 0, "analysis.integrated_gradients.sampling_seed"),
    (
        "integrated_gradients.attribution_width",
        3,
        "analysis.integrated_gradients.attribution_width",
    ),
    (
        "integrated_gradients.baseline_policy",
        "zero",
        "analysis.integrated_gradients.baseline_policy",
    ),
    (
        "model_provenance.checkpoint_selection_mode",
        "cv_best_fold",
        "analysis.model_provenance.checkpoint_selection_mode",
    ),
    ("model_provenance.selected_fold", 1, "analysis.model_provenance.selected_fold"),
]


@pytest.mark.parametrize(("dotted", "value", "field"), ANALYSIS_MUTATIONS)
def test_null_explanation_mismatch_rejects(dotted, value, field):
    null = _side("null")
    _set(null.analysis_metadata, dotted, value)
    report = _report(_side("real"), null)

    assert report["compatible"] is False
    assert field in _failed_fields(report)


def test_identical_but_forbidden_ig_mode_on_both_sides_rejects():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.analysis_metadata["integrated_gradients"]["resolved_ig_mode"] = "legacy"
    report = _report(real, null)

    assert "analysis.integrated_gradients.resolved_ig_mode" in _failed_fields(report)


def test_best_fold_selection_on_both_sides_rejects():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.analysis_metadata["model_provenance"]["checkpoint_selection_mode"] = "cv_best_fold"

    assert "analysis.model_provenance.checkpoint_selection_mode" in _failed_fields(
        _report(real, null)
    )


def test_fold_one_on_both_sides_rejects_for_primary_cv():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.analysis_metadata["model_provenance"]["selected_fold"] = 1

    assert "analysis.model_provenance.selected_fold" in _failed_fields(_report(real, null))
    assert "expected_fold_index" in _failed_fields(_report(real, null, expected_fold_index=1))


def test_class_weighting_auto_on_both_sides_rejects():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.config["class_weighting"] = "auto"

    assert "config.class_weighting" in _failed_fields(_report(real, null))


def test_generated_split_on_both_sides_rejects():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.config["split_plan"]["source"] = "generated"

    assert "config.split_plan.source" in _failed_fields(_report(real, null))


def test_sample_ids_mismatch_with_binding_rejects():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.config["split_plan"]["sample_ids_sha256"] = "0" * 64
        side.config["dataset_provenance"]["sample_ids_sha256"] = "0" * 64
        side.analysis_metadata["dataset_provenance"]["sample_ids_sha256"] = "0" * 64
    failed = _failed_fields(_report(real, null))

    assert {
        "config.split_plan.sample_ids_sha256",
        "config.dataset_provenance.sample_ids_sha256",
    } <= failed


# ---------------------------------------------------------------------------
# Dataset relation, lineage, and binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side_name", "dotted", "value", "field"),
    [
        (
            "real",
            "dataset_provenance.preprocessed_data_sha256",
            "0" * 64,
            "config.dataset_provenance.preprocessed_data_sha256",
        ),
        (
            "null",
            "dataset_provenance.preprocessed_data_sha256",
            "0" * 64,
            "config.dataset_provenance.preprocessed_data_sha256",
        ),
        (
            "null",
            "dataset_provenance.null_lineage.lineage_sha256",
            "0" * 64,
            "config.dataset_provenance.null_lineage.lineage_sha256",
        ),
        (
            "null",
            "dataset_provenance.null_lineage.source_artifact_sha256",
            "0" * 64,
            "config.dataset_provenance.null_lineage.source_artifact_sha256",
        ),
        (
            "null",
            "dataset_provenance.null_metadata_kind",
            "legacy_unversioned",
            "config.dataset_provenance.null_metadata_kind",
        ),
        (
            "real",
            "dataset_provenance.is_null_baseline",
            True,
            "config.dataset_provenance.is_null_baseline",
        ),
    ],
)
def test_dataset_binding_mismatch_rejects(side_name, dotted, value, field):
    real, null = _side("real"), _side("null")
    side = real if side_name == "real" else null
    _set(side.config, dotted, value)
    _set(side.analysis_metadata, dotted, value)

    assert field in _failed_fields(_report(real, null))


def test_explanation_dataset_differs_from_training_dataset_rejects():
    null = _side("null")
    null.analysis_metadata["dataset_provenance"]["preprocessed_data_sha256"] = SOURCE_SHA

    assert "analysis.dataset_provenance.preprocessed_data_sha256" in _failed_fields(
        _report(_side("real"), null)
    )


@pytest.mark.parametrize(("side_name", "value"), [("real", True), ("null", False)])
def test_explanation_null_flag_mismatch_rejects(side_name, value):
    real, null = _side("real"), _side("null")
    side = real if side_name == "real" else null
    side.analysis_metadata["is_null_baseline"] = value

    assert "analysis.is_null_baseline" in _failed_fields(_report(real, null))


def test_historical_runs_without_dataset_provenance_fail_pairing():
    real, null = _side("real"), _side("null")
    for side in (real, null):
        side.config.pop("dataset_provenance")
        side.analysis_metadata["dataset_provenance"] = None
    report = _report(real, null)

    assert report["compatible"] is False
    assert {
        "config.dataset_provenance.preprocessed_data_sha256",
        "config.dataset_provenance.null_lineage.lineage_sha256",
        "config.dataset_provenance.sample_ids_sha256",
    } <= _failed_fields(report)


def test_legacy_unversioned_null_run_fails_pairing():
    null = _side("null")
    for block in (null.config, null.analysis_metadata):
        block["dataset_provenance"]["null_metadata_kind"] = "legacy_unversioned"
        block["dataset_provenance"]["null_lineage"] = None

    assert {
        "config.dataset_provenance.null_metadata_kind",
        "config.dataset_provenance.null_lineage.lineage_sha256",
    } <= _failed_fields(_report(_side("real"), null))


def test_real_and_null_datasets_swapped_rejects():
    assert _report(_side("null"), _side("real"))["compatible"] is False


def test_missing_null_binding_field_rejects():
    binding = dict(BINDING)
    binding.pop("lineage_sha256")

    with pytest.raises(PairCompatibilityError, match="null_binding is missing"):
        _report(_side("real"), _side("null"), null_binding=binding)


def test_missing_required_provenance_fails_closed():
    null = _side("null")
    null.config.pop("split_plan")
    real = _side("real")
    real.config.pop("split_plan")

    assert "config.split_plan.sha256" in _failed_fields(_report(real, null))


# ---------------------------------------------------------------------------
# Repository revision, variant universe, checkpoint sanity
# ---------------------------------------------------------------------------


def test_repository_revision_mismatch_rejects():
    report = _report(_side("real"), _replace(_side("null"), repository_revision="b" * 40))

    assert "repository_revision" in _failed_fields(report)


def test_unknown_repository_revision_rejects():
    real = _replace(_side("real"), repository_revision="unknown")
    null = _replace(_side("null"), repository_revision="unknown")

    assert "repository_revision" in _failed_fields(_report(real, null))


def test_variant_universe_mismatch_rejects():
    report = _report(_side("real"), _replace(_side("null"), variant_universe_sha256="0" * 64))

    assert "variant_universe_sha256" in _failed_fields(report)


def test_identical_real_and_null_checkpoints_fail_sanity_check():
    null = _side("null")
    null.analysis_metadata["model_provenance"]["checkpoint_sha256"] = "c" * 64

    report = _report(_side("real"), null)
    assert [m["rule"] for m in report["mismatches"]] == ["sanity_must_differ"]
    with pytest.raises(PairCompatibilityError, match="checkpoints are byte-identical"):
        require_compatible_real_null_pair(
            run_id="no_position",
            real=_side("real"),
            null=null,
            null_binding=BINDING,
            expected_fold_index=0,
        )


# ---------------------------------------------------------------------------
# Benchmark-level shared null
# ---------------------------------------------------------------------------


def _strategy_reports(bindings=None) -> list[dict]:
    reports = []
    for index, relative in enumerate(("none", "t5_bucket", "alibi_fixed")):
        real, null = _side("real", relative), _side("null", relative)
        real.analysis_metadata["model_provenance"]["checkpoint_sha256"] = f"{index}c".ljust(64, "c")
        null.analysis_metadata["model_provenance"]["checkpoint_sha256"] = f"{index}d".ljust(64, "d")
        binding = BINDING if bindings is None else bindings[index]
        reports.append(
            require_compatible_real_null_pair(
                run_id=f"run_{relative}",
                real=real,
                null=null,
                null_binding=binding,
                expected_fold_index=0,
            )
        )
    return reports


def test_all_strategies_have_distinct_checkpoints_but_share_one_null():
    reports = _strategy_reports()
    summary = require_shared_null_across_pairs(reports)

    assert summary["run_ids"] == ["run_none", "run_t5_bucket", "run_alibi_fixed"]
    assert len({r["real"]["checkpoint_sha256"] for r in reports}) == 3
    assert len({r["shared"]["position_strategy_hash"] for r in reports}) == 3
    assert summary["null_binding"]["lineage_sha256"] == LINEAGE_SHA


def test_strategy_specific_null_binding_rejects_at_benchmark_level():
    other = dict(BINDING, lineage_sha256="e" * 64)
    reports = _strategy_reports()
    reports[2] = dict(
        reports[2],
        null_binding=dict(
            reports[2]["null_binding"], **{"lineage_sha256": other["lineage_sha256"]}
        ),
    )

    with pytest.raises(PairCompatibilityError, match="different null bindings"):
        require_shared_null_across_pairs(reports)


def test_benchmark_level_rejects_incompatible_or_duplicate_reports():
    reports = _strategy_reports()
    with pytest.raises(PairCompatibilityError, match="duplicate run_id"):
        require_shared_null_across_pairs([reports[0], reports[0]])
    bad = _report(_side("real"), _replace(_side("null"), repository_revision="b" * 40))
    with pytest.raises(PairCompatibilityError, match="incompatible pair reports"):
        require_shared_null_across_pairs([reports[0], dict(bad, run_id="bad")])
    with pytest.raises(PairCompatibilityError, match="at least one"):
        require_shared_null_across_pairs([])


# ---------------------------------------------------------------------------
# Variant-universe fingerprint from attributions.npz
# ---------------------------------------------------------------------------


def _write_attributions(path: Path, samples: list[dict], *, scores_scale: float = 1.0) -> Path:
    """Write an attributions.npz in explain.py's layout (object arrays)."""
    variant_scores = [
        np.asarray(sample["scores"], dtype=float) * scores_scale for sample in samples
    ]
    metadata = [
        {
            "positions": np.asarray(sample["positions"]),
            "gene_ids": np.asarray(sample["gene_ids"]),
            "chromosomes": np.asarray(sample["chromosomes"]),
            "sample_idx": index,
            "sample_id": f"s{index}",
            "label": sample.get("label", 0),
        }
        for index, sample in enumerate(samples)
    ]
    scores_array = np.empty(len(variant_scores), dtype=object)
    scores_array[:] = variant_scores
    np.savez(path, variant_scores=scores_array, metadata=np.array(metadata, dtype=object))
    return path


def _universe() -> list[dict]:
    return [
        {"positions": [100, 200], "gene_ids": [0, 1], "chromosomes": ["1", "2"], "scores": [1, 2]},
        {"positions": [300], "gene_ids": [2], "chromosomes": ["X"], "scores": [3]},
    ]


def test_variant_universe_ignores_scores_and_labels(tmp_path):
    real = _write_attributions(tmp_path / "real.npz", _universe())
    null_samples = _universe()
    for sample in null_samples:
        sample["label"] = 1
    null = _write_attributions(tmp_path / "null.npz", null_samples, scores_scale=0.5)

    assert variant_universe_sha256(real) == variant_universe_sha256(null)
    assert variant_universe_sha256(real) == variant_universe_sha256(real)


def test_variant_universe_preserves_sample_boundaries(tmp_path):
    base = _write_attributions(tmp_path / "base.npz", _universe())
    moved = _universe()
    moved[1]["positions"].insert(0, moved[0]["positions"].pop())
    moved[1]["gene_ids"].insert(0, moved[0]["gene_ids"].pop())
    moved[1]["chromosomes"].insert(0, moved[0]["chromosomes"].pop())
    moved[0]["scores"].pop()
    moved[1]["scores"].insert(0, 2)
    shifted = _write_attributions(tmp_path / "shifted.npz", moved)

    assert variant_universe_sha256(base) != variant_universe_sha256(shifted)


@pytest.mark.parametrize(
    ("mutate"),
    [
        lambda u: u[0]["positions"].__setitem__(0, 101),
        lambda u: u[0]["gene_ids"].__setitem__(0, 9),
        lambda u: u[0]["chromosomes"].__setitem__(0, "3"),
        lambda u: u.append({"positions": [], "gene_ids": [], "chromosomes": [], "scores": []}),
    ],
)
def test_variant_universe_changes_with_any_variant_identity(tmp_path, mutate):
    base = _write_attributions(tmp_path / "base.npz", _universe())
    changed = _universe()
    mutate(changed)

    assert variant_universe_sha256(base) != variant_universe_sha256(
        _write_attributions(tmp_path / "changed.npz", changed)
    )


def test_variant_universe_rejects_inconsistent_lengths(tmp_path):
    broken = _universe()
    broken[0]["scores"].append(9)

    with pytest.raises(PairCompatibilityError, match="mismatched"):
        variant_universe_sha256(_write_attributions(tmp_path / "broken.npz", broken))


def test_variant_universe_rejects_missing_arrays(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, other=np.zeros(1))

    with pytest.raises(PairCompatibilityError, match="variant_scores"):
        variant_universe_sha256(path)


# ---------------------------------------------------------------------------
# load_pair_side from completed artifacts
# ---------------------------------------------------------------------------


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_side(
    root: Path, side: PairSide, *, mode: str = "cv_explicit_fold", fold: int | None = 0
) -> tuple[Path, Path, Path]:
    """Materialize one completed side whose provenance points at its real training_dir."""
    training = root / "training"
    explanation = root / "explanation"
    explanation.mkdir(parents=True)
    config = copy.deepcopy(dict(side.config))
    analysis = copy.deepcopy(dict(side.analysis_metadata))
    if mode == "cv_explicit_fold":
        checkpoint = training / f"fold_{fold}" / "best_model.pt"
    else:
        checkpoint = training / "best_model.pt"
        config["position_encoding_execution"]["training_mode"] = "single_split"
        config["cv"] = None
        analysis["model_provenance"].update({"selected_fold_auc": None, "cv_results_path": None})
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(root.name.encode("utf-8"))
    config_path = training / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    analysis["model_provenance"].update(
        {
            "checkpoint_selection_mode": mode,
            "selected_fold": fold,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_bytes(checkpoint),
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256_bytes(config_path),
        }
    )
    _write_analysis(explanation, analysis)
    _write_attributions(explanation / "attributions.npz", _universe())
    return training, explanation, checkpoint


def _write_analysis(explanation: Path, analysis: dict) -> None:
    (explanation / "analysis_metadata.yaml").write_text(yaml.safe_dump(analysis), encoding="utf-8")


def _read_analysis(explanation: Path) -> dict:
    return yaml.safe_load((explanation / "analysis_metadata.yaml").read_text(encoding="utf-8"))


def _load(training: Path, explanation: Path) -> PairSide:
    return load_pair_side(training, explanation, repository_revision=REVISION)


# A. normal matched pair


def test_load_pair_side_round_trips_and_validates_pair(tmp_path):
    real_dirs = _write_side(tmp_path / "real", _side("real"))
    null_dirs = _write_side(tmp_path / "null", _side("null"))

    real = _load(real_dirs[0], real_dirs[1])
    null = _load(null_dirs[0], null_dirs[1])
    report = require_compatible_real_null_pair(
        run_id="no_position", real=real, null=null, null_binding=BINDING, expected_fold_index=0
    )

    assert report["shared"]["variant_universe_sha256"] == variant_universe_sha256(
        real_dirs[1] / "attributions.npz"
    )
    recorded = real.analysis_metadata["model_provenance"]
    assert recorded["config_path"] == str((real_dirs[0] / "config.yaml").resolve())
    assert recorded["checkpoint_path"] == str((real_dirs[0] / "fold_0" / "best_model.pt").resolve())


# B. explanation from another training directory


def test_explanation_from_another_identical_training_dir_rejects(tmp_path):
    run_a = _write_side(tmp_path / "run_a" / "real", _side("real"))
    run_b = _write_side(tmp_path / "run_b" / "real", _side("real"))
    config_a = (run_a[0] / "config.yaml").read_bytes()
    assert config_a == (run_b[0] / "config.yaml").read_bytes()

    with pytest.raises(PairCompatibilityError, match="does not belong to the supplied"):
        _load(run_a[0], run_b[1])


# C. config_path pointing elsewhere


def test_config_path_pointing_elsewhere_rejects(tmp_path):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    elsewhere = tmp_path / "elsewhere" / "config.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes((training / "config.yaml").read_bytes())
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["config_path"] = str(elsewhere)
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="config_path does not belong"):
        _load(training, explanation)


def test_config_path_with_same_filename_in_other_dir_rejects(tmp_path):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["config_path"] = str(tmp_path / "other" / "config.yaml")
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="config_path does not belong"):
        _load(training, explanation)


def test_equivalent_relative_config_path_is_accepted(tmp_path, monkeypatch):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    monkeypatch.chdir(tmp_path)
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["config_path"] = "real/training/../training/config.yaml"
    _write_analysis(explanation, analysis)

    assert _load(training, explanation).config["class_weighting"] == "off"


# D. config bytes modified after explanation


def test_config_modified_after_explanation_rejects(tmp_path):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    with (training / "config.yaml").open("a", encoding="utf-8") as handle:
        handle.write("# edited after explanation\n")

    with pytest.raises(PairCompatibilityError, match="config bytes changed"):
        _load(training, explanation)


@pytest.mark.parametrize("value", [None, ""])
def test_missing_config_sha256_rejects_for_paired_provenance(tmp_path, value):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    analysis = _read_analysis(explanation)
    if value is None:
        analysis["model_provenance"].pop("config_sha256")
    else:
        analysis["model_provenance"]["config_sha256"] = value
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="config_sha256 is required"):
        _load(training, explanation)


# E. checkpoint outside the supplied training_dir


def test_checkpoint_outside_training_dir_rejects(tmp_path):
    training, explanation, checkpoint = _write_side(tmp_path / "real", _side("real"))
    outside = tmp_path / "outside" / "fold_0" / "best_model.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(checkpoint.read_bytes())
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["checkpoint_path"] = str(outside)
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="checkpoint_path does not match"):
        _load(training, explanation)


def test_changed_checkpoint_bytes_still_rejects(tmp_path):
    training, explanation, checkpoint = _write_side(tmp_path / "real", _side("real"))
    checkpoint.write_bytes(b"retrained")

    with pytest.raises(PairCompatibilityError, match="checkpoint bytes changed"):
        _load(training, explanation)


# F. cv_explicit_fold selected_fold=0 points exactly to training/fold_0/best_model.pt


def test_cv_explicit_fold_zero_binds_to_fold_zero_checkpoint(tmp_path):
    training, explanation, checkpoint = _write_side(tmp_path / "real", _side("real"))

    assert checkpoint == training / "fold_0" / "best_model.pt"
    assert _load(training, explanation).analysis_metadata["model_provenance"]["selected_fold"] == 0
    for wrong in (training / "best_model.pt", training / "fold_0" / "last_model.pt"):
        wrong.write_bytes(checkpoint.read_bytes())
        analysis = _read_analysis(explanation)
        analysis["model_provenance"]["checkpoint_path"] = str(wrong)
        _write_analysis(explanation, analysis)
        with pytest.raises(PairCompatibilityError, match="checkpoint_path does not match"):
            _load(training, explanation)


@pytest.mark.parametrize("fold", [None, -1, True, "0", 0.0])
def test_cv_explicit_fold_requires_plain_non_negative_selected_fold(tmp_path, fold):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["selected_fold"] = fold
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="plain non-negative integer"):
        _load(training, explanation)


# G. checkpoint from fold_1 while provenance says fold_0


def test_fold_one_checkpoint_with_fold_zero_provenance_rejects(tmp_path):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    fold_one = training / "fold_1" / "best_model.pt"
    fold_one.parent.mkdir()
    fold_one.write_bytes(b"fold one weights")
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["checkpoint_path"] = str(fold_one)
    analysis["model_provenance"]["checkpoint_sha256"] = _sha256_bytes(fold_one)
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="checkpoint_path does not match"):
        _load(training, explanation)


def test_fold_one_explanation_loads_but_fails_primary_pair_rule(tmp_path):
    real_dirs = _write_side(tmp_path / "real", _side("real"), fold=1)
    null_dirs = _write_side(tmp_path / "null", _side("null"), fold=1)
    real, null = _load(*real_dirs[:2]), _load(*null_dirs[:2])

    with pytest.raises(PairCompatibilityError, match="selected_fold must be 0"):
        require_compatible_real_null_pair(
            run_id="no_position", real=real, null=null, null_binding=BINDING, expected_fold_index=0
        )


# H. single_run_best_model points exactly to training/best_model.pt


def test_single_run_best_model_binds_to_root_checkpoint(tmp_path):
    training, explanation, checkpoint = _write_side(
        tmp_path / "real", _side("real"), mode="single_run_best_model", fold=None
    )

    assert checkpoint == training / "best_model.pt"
    assert _load(training, explanation).config["cv"] is None
    fold_zero = training / "fold_0" / "best_model.pt"
    fold_zero.parent.mkdir()
    fold_zero.write_bytes(checkpoint.read_bytes())
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["checkpoint_path"] = str(fold_zero)
    _write_analysis(explanation, analysis)
    with pytest.raises(PairCompatibilityError, match="checkpoint_path does not match"):
        _load(training, explanation)


def test_single_run_best_model_requires_null_selected_fold(tmp_path):
    training, explanation, _ = _write_side(
        tmp_path / "real", _side("real"), mode="single_run_best_model", fold=None
    )
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["selected_fold"] = 0
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="selected_fold must be null"):
        _load(training, explanation)


@pytest.mark.parametrize("mode", ["cv_best_fold", "explicit_checkpoint", None])
def test_non_paired_selection_modes_fail_closed_in_loader(tmp_path, mode):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    analysis = _read_analysis(explanation)
    analysis["model_provenance"]["checkpoint_selection_mode"] = mode
    _write_analysis(explanation, analysis)

    with pytest.raises(PairCompatibilityError, match="cannot be bound to a paired training_dir"):
        _load(training, explanation)


def test_load_pair_side_rejects_missing_artifacts(tmp_path):
    training, explanation, _ = _write_side(tmp_path / "real", _side("real"))
    (explanation / "analysis_metadata.yaml").unlink()

    with pytest.raises(PairCompatibilityError, match="does not exist"):
        _load(training, explanation)
