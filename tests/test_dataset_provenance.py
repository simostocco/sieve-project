"""Tests for Phase 12C3B1 training/explanation dataset provenance."""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from scripts import create_null_baseline, explain, train
from src.data import null_lineage
from src.data.dataset_provenance import (
    build_dataset_provenance,
    require_explanation_dataset_matches_training,
    resolve_explanation_dataset_provenance,
)
from src.data.vcf_parser import SampleVariants, VariantRecord
from src.training.split_plan import ordered_sample_ids, sample_ids_sha256


def _samples(n: int = 6) -> list[SampleVariants]:
    return [
        SampleVariants(
            f"s{index}",
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
        for index in range(n)
    ]


@pytest.fixture
def artifacts(tmp_path: Path) -> dict[str, Path]:
    real_path = tmp_path / "cohort.pt"
    torch.save({"samples": _samples(), "metadata": {"genome_build": "GRCh37"}}, real_path)
    null_path = tmp_path / "cohort.null.pt"
    create_null_baseline.create_strict_single_permutation(
        str(real_path), str(null_path), seed=3, reuse=False, argv=["create_null_baseline.py"]
    )
    return {
        "real": real_path,
        "null": null_path,
        "sidecar": null_lineage.sidecar_path_for(null_path),
    }


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# build_dataset_provenance
# ---------------------------------------------------------------------------


def test_real_dataset_provenance(artifacts):
    data = _load(artifacts["real"])
    provenance = build_dataset_provenance(data, path=artifacts["real"])

    assert provenance == {
        "schema_version": 1,
        "preprocessed_data_path": str(artifacts["real"].resolve()),
        "preprocessed_data_sha256": null_lineage.compute_file_sha256(artifacts["real"]),
        "sample_ids_sha256": sample_ids_sha256(ordered_sample_ids(data["samples"])),
        "is_null_baseline": False,
        "null_metadata_kind": "none",
        "null_lineage": None,
    }


def test_strict_null_dataset_provenance_matches_lineage(artifacts):
    data = _load(artifacts["null"])
    provenance = build_dataset_provenance(data, path=artifacts["null"])
    report = null_lineage.validate_null_pair(
        artifacts["real"], artifacts["null"], artifacts["sidecar"]
    )
    embedded = data["_null_baseline_metadata"]

    assert provenance["is_null_baseline"] is True
    assert provenance["null_metadata_kind"] == "strict_v1"
    assert provenance["preprocessed_data_sha256"] == report["null_artifact_sha256"]
    assert provenance["sample_ids_sha256"] == report["sample_ids_sha256"]
    assert provenance["null_lineage"] == {
        "lineage_sha256": report["lineage_sha256"],
        "source_artifact_sha256": report["source_artifact_sha256"],
        "permutation_indices_sha256": report["permutation_indices_sha256"],
        "original_labels_sha256": report["original_labels_sha256"],
        "permuted_labels_sha256": report["permuted_labels_sha256"],
    }
    assert provenance["null_lineage"]["lineage_sha256"] == embedded["lineage_sha256"]


def test_precomputed_sha_is_used_verbatim(artifacts):
    provenance = build_dataset_provenance(
        _load(artifacts["real"]), path=artifacts["real"], preprocessed_data_sha256="f" * 64
    )

    assert provenance["preprocessed_data_sha256"] == "f" * 64


def test_legacy_unversioned_null_metadata_is_null_without_lineage(artifacts):
    data = _load(artifacts["real"])
    data["_null_baseline_metadata"] = {"is_null_baseline": True, "permutation_seed": 42}

    provenance = build_dataset_provenance(data, path=artifacts["real"])

    assert provenance["is_null_baseline"] is True
    assert provenance["null_metadata_kind"] == "legacy_unversioned"
    assert provenance["null_lineage"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e.pop("lineage_sha256"), "invalid strict"),
        (lambda e: e.update({"lineage_sha256": "ABC"}), "invalid strict"),
        (lambda e: e.update({"is_null_baseline": False}), "invalid strict"),
        (lambda e: e.update({"n_samples": True}), "invalid strict"),
        (lambda e: e.update({"sample_ids_sha256": "0" * 64}), "sample_ids_sha256 does not match"),
        (lambda e: e.update({"n_samples": 99}), "n_samples does not match"),
        (lambda e: e.update({"permuted_labels_sha256": "0" * 64}), "permuted_labels_sha256"),
    ],
)
def test_structurally_invalid_strict_null_metadata_rejects(artifacts, mutate, message):
    data = _load(artifacts["null"])
    mutate(data["_null_baseline_metadata"])

    with pytest.raises(ValueError, match=message):
        build_dataset_provenance(data, path=artifacts["null"])


def test_null_labels_tampered_after_embedding_rejects(artifacts):
    data = _load(artifacts["null"])
    data["samples"][0].label = 1 - data["samples"][0].label

    with pytest.raises(ValueError, match="permuted_labels_sha256"):
        build_dataset_provenance(data, path=artifacts["null"])


@pytest.mark.parametrize(
    "metadata",
    [["not", "a", "mapping"], {"is_null_baseline": False}, {"permutation_seed": 1}],
)
def test_malformed_null_metadata_never_passes_as_real(artifacts, metadata):
    data = _load(artifacts["real"])
    data["_null_baseline_metadata"] = metadata

    with pytest.raises(ValueError, match="_null_baseline_metadata"):
        build_dataset_provenance(data, path=artifacts["real"])


def test_missing_samples_list_rejects(artifacts):
    with pytest.raises(ValueError, match="samples"):
        build_dataset_provenance({"metadata": {}}, path=artifacts["real"])


# ---------------------------------------------------------------------------
# train.py plumbing (metadata only)
# ---------------------------------------------------------------------------


def test_train_main_records_dataset_provenance_in_run_metadata():
    source = inspect.getsource(train.main)

    load = "preprocessed = torch.load(args.preprocessed_data, weights_only=False)"
    build = "dataset_provenance = build_dataset_provenance("
    assert source.count(build) == 1
    assert source.index(load) < source.index(build) < source.index("if sex_map:")
    assert "dataset_provenance = None" in source
    assign = 'run_metadata["dataset_provenance"] = dataset_provenance'
    assert source.count(assign) == 1
    assert source.index(assign) < source.index("_update_saved_config(config_path, **run_metadata)")
    # Checkpoint and fold-config propagation reuse the existing run_metadata path.
    assert source.count("checkpoint_metadata=run_metadata") == 2
    assert (
        source.count("save_fold_config(fold_dir, fold_idx, args, run_metadata=run_metadata)") == 1
    )


def test_fold_config_persists_dataset_provenance(tmp_path, artifacts):
    provenance = build_dataset_provenance(_load(artifacts["null"]), path=artifacts["null"])
    args = argparse.Namespace(
        experiment_name="training",
        level="L3",
        latent_dim=8,
        hidden_dim=8,
        num_attention_layers=1,
        num_heads=1,
        chunk_size=10,
        chunk_overlap=0,
        aggregation_method="mean",
        classifier_type="flatten",
        lr=0.001,
        lambda_attr=0.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        gradient_clip=None,
        early_stopping=1,
        epochs=1,
        seed=42,
        genome_build="GRCh37",
        sex_map=None,
        pc_map=None,
        num_pcs=0,
        pc_map_sha256=None,
        preprocessed_data=str(artifacts["null"]),
        vcf=None,
        phenotypes=None,
    )
    train.save_fold_config(tmp_path, 0, args, run_metadata={"dataset_provenance": provenance})

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["dataset_provenance"] == provenance


def test_update_saved_config_persists_dataset_provenance_at_root(tmp_path, artifacts):
    provenance = build_dataset_provenance(_load(artifacts["real"]), path=artifacts["real"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"level": "L3"}), encoding="utf-8")

    train._update_saved_config(config_path, dataset_provenance=provenance)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["dataset_provenance"] == provenance


# ---------------------------------------------------------------------------
# explain.py validation
# ---------------------------------------------------------------------------


def _provenances(artifacts) -> tuple[dict, dict]:
    real = build_dataset_provenance(_load(artifacts["real"]), path=artifacts["real"])
    null = build_dataset_provenance(_load(artifacts["null"]), path=artifacts["null"])
    return real, null


def _resolve(artifacts, *, dataset: str, training_config: dict, flag: bool):
    return resolve_explanation_dataset_provenance(
        _load(artifacts[dataset]),
        path=artifacts[dataset],
        training_config=training_config,
        is_null_baseline_flag=flag,
    )


# A. New provenance-aware training configs: strict, fail-closed.


def test_new_real_training_real_dataset_without_flag_passes(artifacts):
    real, _ = _provenances(artifacts)

    recorded = _resolve(
        artifacts, dataset="real", training_config={"dataset_provenance": real}, flag=False
    )
    assert recorded == real


def test_new_null_training_null_dataset_with_flag_passes(artifacts):
    _, null = _provenances(artifacts)

    recorded = _resolve(
        artifacts, dataset="null", training_config={"dataset_provenance": null}, flag=True
    )
    assert recorded == null


def test_new_real_training_with_null_flag_rejects(artifacts):
    real, _ = _provenances(artifacts)

    for dataset in ("real", "null"):
        with pytest.raises(ValueError, match="checkpoint was trained on real data"):
            _resolve(
                artifacts,
                dataset=dataset,
                training_config={"dataset_provenance": real},
                flag=True,
            )


def test_new_null_training_missing_flag_rejects(artifacts):
    _, null = _provenances(artifacts)

    for dataset in ("null", "real"):
        with pytest.raises(ValueError, match="pass --is-null-baseline"):
            _resolve(
                artifacts,
                dataset=dataset,
                training_config={"dataset_provenance": null},
                flag=False,
            )


def test_new_training_and_explanation_dataset_sha_mismatch_rejects(artifacts):
    real, null = _provenances(artifacts)
    other = copy.deepcopy(real)
    other["preprocessed_data_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="preprocessed_data_sha256 mismatch"):
        _resolve(
            artifacts, dataset="real", training_config={"dataset_provenance": other}, flag=False
        )
    with pytest.raises(ValueError, match="preprocessed_data_sha256 mismatch"):
        _resolve(artifacts, dataset="real", training_config={"dataset_provenance": null}, flag=True)


def test_new_training_with_invalid_explanation_dataset_rejects(artifacts):
    _, null = _provenances(artifacts)
    data = _load(artifacts["null"])
    data["_null_baseline_metadata"].pop("lineage_sha256")

    with pytest.raises(ValueError, match="invalid strict"):
        resolve_explanation_dataset_provenance(
            data,
            path=artifacts["null"],
            training_config={"dataset_provenance": null},
            is_null_baseline_flag=True,
        )


def test_malformed_training_provenance_rejects(artifacts):
    real, _ = _provenances(artifacts)

    with pytest.raises(ValueError, match="must be a mapping"):
        require_explanation_dataset_matches_training(
            training_config={"dataset_provenance": "x"},
            explanation_provenance=real,
            is_null_baseline_flag=False,
        )
    with pytest.raises(ValueError, match="must be a bool"):
        require_explanation_dataset_matches_training(
            training_config={"dataset_provenance": {"is_null_baseline": "no"}},
            explanation_provenance=real,
            is_null_baseline_flag=False,
        )


# B. Historical configs without dataset_provenance: historical behaviour.


@pytest.mark.parametrize(
    ("dataset", "flag"),
    [("real", False), ("real", True), ("null", True), ("null", False)],
)
def test_historical_config_keeps_metadata_only_null_flag(artifacts, dataset, flag):
    real, null = _provenances(artifacts)

    recorded = _resolve(artifacts, dataset=dataset, training_config={"level": "L3"}, flag=flag)

    assert recorded == (real if dataset == "real" else null)


def test_historical_legacy_null_without_flag_still_explains(artifacts):
    data = _load(artifacts["real"])
    data["_null_baseline_metadata"] = {"is_null_baseline": True, "permutation_seed": 42}

    recorded = resolve_explanation_dataset_provenance(
        data, path=artifacts["real"], training_config={"level": "L3"}, is_null_baseline_flag=False
    )

    assert recorded["null_metadata_kind"] == "legacy_unversioned"


def test_historical_unclassifiable_dataset_records_none_instead_of_failing(artifacts):
    data = _load(artifacts["real"])
    data["_null_baseline_metadata"] = ["not", "a", "mapping"]

    assert (
        resolve_explanation_dataset_provenance(
            data,
            path=artifacts["real"],
            training_config={"level": "L3"},
            is_null_baseline_flag=False,
        )
        is None
    )


def test_require_is_a_no_op_for_historical_configs(artifacts):
    real, null = _provenances(artifacts)

    for provenance, flag in ((real, True), (null, False)):
        require_explanation_dataset_matches_training(
            training_config={"level": "L3"},
            explanation_provenance=provenance,
            is_null_baseline_flag=flag,
        )


def test_explain_main_validates_before_ig_and_persists_dataset_provenance():
    source = inspect.getsource(explain.main)

    resolve = "dataset_provenance = resolve_explanation_dataset_provenance("
    assert source.count(resolve) == 1
    load = "preprocessed = torch.load(args.preprocessed_data, weights_only=False)"
    assert source.index(load) < source.index(resolve)
    assert source.index(resolve) < source.index("# === INTEGRATED GRADIENTS (CHUNKED) ===")
    assert source.index(resolve) < source.index("sex_map_path = config.get('sex_map')")
    assert "training_config=config," in source
    assert "'dataset_provenance': dataset_provenance," in source


def test_explain_null_flag_help_describes_contract_boundary():
    help_text = " ".join(explain.build_arg_parser().format_help().split())

    assert "fail closed" in help_text
    assert "historical runs keep metadata-only behaviour" in help_text


def test_explain_model_provenance_records_config_sha256_additively(tmp_path):
    config_path = tmp_path / "training" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("level: L3\n", encoding="utf-8")
    checkpoint = tmp_path / "training" / "best_model.pt"
    checkpoint.write_bytes(b"weights")

    provenance = explain._build_model_provenance(
        checkpoint_selection_mode="single_run_best_model",
        checkpoint_path=checkpoint,
        config_path=config_path,
        selected_fold=None,
        selected_fold_auc=None,
        cv_results_path=None,
    )

    assert list(provenance) == [
        "schema_version",
        "checkpoint_selection_mode",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "selected_fold",
        "selected_fold_auc",
        "cv_results_path",
    ]
    assert provenance["config_path"] == str(config_path.resolve())
    assert provenance["config_sha256"] == null_lineage.compute_file_sha256(config_path)
    assert provenance["checkpoint_sha256"] == null_lineage.compute_file_sha256(checkpoint)
