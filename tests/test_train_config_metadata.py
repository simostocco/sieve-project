"""Tests for Phase 4C training metadata helpers."""

import json
from types import MappingProxyType

import pytest
import yaml

from scripts import train
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    PositionEncodingRequest,
    resolve_position_encoding_config,
)


def resolved_l3():
    return resolve_position_encoding_config(
        PositionEncodingRequest(),
        AnnotationLevel.L3,
        latent_dim=64,
        num_heads=4,
        num_chromosomes=3,
    )


def test_checksum_independent_of_insertion_order():
    assert train.mapping_sha256({"BRCA1": 0, "TP53": 1}) == train.mapping_sha256(
        {"TP53": 1, "BRCA1": 0}
    )


def test_checksum_deterministic_across_repeated_calls():
    mapping = {"BRCA1": 0, "TP53": 1}
    assert train.mapping_sha256(mapping) == train.mapping_sha256(mapping)


def test_checksum_changes_when_id_changes():
    assert train.mapping_sha256({"BRCA1": 0, "TP53": 1}) != train.mapping_sha256(
        {"BRCA1": 1, "TP53": 0}
    )


def test_checksum_changes_when_name_changes():
    assert train.mapping_sha256({"BRCA1": 0, "TP53": 1}) != train.mapping_sha256(
        {"BRCA1": 0, "MYC": 1}
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"BRCA1": True}, "bool"),
        ({"BRCA1": 0.0}, "integer"),
        ({"BRCA1": "0"}, "integer"),
        ({"BRCA1": -1}, "non-negative"),
        ({"BRCA1": 0, "TP53": 0}, "unique"),
        ({"BRCA1": 0, "TP53": 2}, "contiguous"),
        ({1: 0}, "non-string"),
    ],
)
def test_mapping_validation_rejects_malformed_mappings(mapping, message):
    with pytest.raises(ValueError, match=message):
        train.mapping_sha256(mapping, mapping_name="gene_index")


def test_mapping_checksum_does_not_mutate_input_mapping():
    mapping = {"TP53": 1, "BRCA1": 0}
    before = dict(mapping)
    train.mapping_sha256(mapping)
    assert mapping == before


def test_chromosome_inversion_is_id_to_name_in_numeric_order():
    assert train.build_chromosome_id_to_name({"X": 2, "1": 0, "2": 1}) == {
        "0": "1",
        "1": "2",
        "2": "X",
    }


def test_mapping_artifact_contains_complete_mappings_and_hashes(tmp_path):
    gene_index = {"TP53": 1, "BRCA1": 0}
    chrom_index = {"X": 2, "1": 0, "2": 1}

    identity = train.write_dataset_mappings_artifact(tmp_path, gene_index, chrom_index)
    artifact_path = tmp_path / "dataset_mappings.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["gene_index"] == {"BRCA1": 0, "TP53": 1}
    assert payload["chrom_index"] == {"1": 0, "2": 1, "X": 2}
    assert payload["chromosome_id_to_name"] == {"0": "1", "1": "2", "2": "X"}
    assert payload["gene_mapping_sha256"] == train.mapping_sha256(gene_index)
    assert payload["chromosome_mapping_sha256"] == train.mapping_sha256(chrom_index)
    assert identity == {
        "gene_mapping_sha256": payload["gene_mapping_sha256"],
        "chromosome_mapping_sha256": payload["chromosome_mapping_sha256"],
        "mappings_artifact": "dataset_mappings.json",
        "mappings_artifact_base": "experiment_root",
    }


def test_mapping_artifact_json_is_deterministic_and_has_one_newline(tmp_path):
    gene_index = {"TP53": 1, "BRCA1": 0}
    chrom_index = {"X": 2, "1": 0, "2": 1}

    train.write_dataset_mappings_artifact(tmp_path, gene_index, chrom_index)
    first = (tmp_path / "dataset_mappings.json").read_text(encoding="utf-8")
    train.write_dataset_mappings_artifact(
        tmp_path,
        dict(reversed(gene_index.items())),
        chrom_index,
    )
    second = (tmp_path / "dataset_mappings.json").read_text(encoding="utf-8")

    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_mapping_artifact_contains_no_sample_or_phenotype_data(tmp_path):
    train.write_dataset_mappings_artifact(tmp_path, {"BRCA1": 0}, {"1": 0})
    payload = json.loads((tmp_path / "dataset_mappings.json").read_text(encoding="utf-8"))
    assert "sample_ids" not in payload
    assert "labels" not in payload
    assert "phenotypes" not in payload


def test_serialized_position_config_embeds_chromosome_mapping_without_gene_index():
    position_encoding = train.serialize_position_encoding_for_training(
        resolved_l3(),
        {"X": 2, "1": 0, "2": 1},
    )

    assert position_encoding["chromosome"]["mapping"] == {
        "0": "1",
        "1": "2",
        "2": "X",
    }
    assert "gene_index" not in position_encoding


def test_serialized_position_config_does_not_mutate_resolved_config():
    resolved = resolved_l3()
    before = resolved.to_dict()

    train.serialize_position_encoding_for_training(resolved, {"1": 0})

    assert resolved.to_dict() == before


def test_run_metadata_contains_required_dimensions_identity_and_resolved_config():
    resolved = resolved_l3()
    metadata = train.build_training_run_metadata(
        input_dim=71,
        num_genes=2,
        num_chromosomes=3,
        genome_build="GRCh37",
        resolved_position_encoding=resolved,
        chrom_index={"1": 0, "2": 1, "X": 2},
        gene_mapping_sha256="genehash",
        chromosome_mapping_sha256="chromhash",
        training_mode="cv",
    )

    assert metadata["metadata_schema_version"] == 1
    assert metadata["input_dim"] == 71
    assert metadata["content_dim"] == resolved.content_dim
    assert metadata["num_genes"] == 2
    assert metadata["num_chromosomes"] == 3
    assert metadata["position_encoding"]["input_dim"] == resolved.input_dim
    assert metadata["dataset_identity"] == {
        "genome_build": "GRCh37",
        "gene_mapping_sha256": "genehash",
        "chromosome_mapping_sha256": "chromhash",
        "mappings_artifact": "dataset_mappings.json",
        "mappings_artifact_base": "experiment_root",
    }


def test_cv_execution_metadata_is_accurate():
    assert train.build_position_encoding_execution_metadata(
        training_mode="cv",
        dataset_num_chromosomes=3,
    ) == {
        "schema_version": 1,
        "source": "legacy_existing_model_paths",
        "resolved_config_applied_to_model": False,
        "model_num_chromosomes": 3,
        "chrom_ids_passed_to_attention": True,
        "chromosome_embedding_executed": True,
        "chromosome_aware_relative_bias_executed": True,
    }


def test_single_split_execution_metadata_records_relative_routing_active():
    metadata = train.build_position_encoding_execution_metadata(
        training_mode="single_split",
        dataset_num_chromosomes=3,
    )

    assert metadata["model_num_chromosomes"] == 0
    assert metadata["chrom_ids_passed_to_attention"] is True
    assert metadata["chromosome_embedding_executed"] is False
    assert metadata["chromosome_aware_relative_bias_executed"] is True
    assert metadata["resolved_config_applied_to_model"] is False


def test_invalid_training_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported training_mode"):
        train.build_position_encoding_execution_metadata(
            training_mode="holdout",
            dataset_num_chromosomes=3,
        )


def test_helpers_accept_read_only_mappings_without_mutating():
    gene_index = MappingProxyType({"BRCA1": 0, "TP53": 1})
    chrom_index = MappingProxyType({"1": 0, "2": 1})

    metadata = train.build_training_run_metadata(
        input_dim=71,
        num_genes=2,
        num_chromosomes=2,
        genome_build="GRCh37",
        resolved_position_encoding=resolved_l3(),
        chrom_index=chrom_index,
        gene_mapping_sha256=train.mapping_sha256(gene_index),
        chromosome_mapping_sha256=train.mapping_sha256(chrom_index),
        training_mode="cv",
    )

    assert metadata["dataset_identity"]["mappings_artifact"] == "dataset_mappings.json"


def test_metadata_update_preserves_raw_config_fields_and_later_class_weight_updates(tmp_path):
    config_path = tmp_path / "config.yaml"
    raw_config = {
        "level": "L3",
        "position_preset": "legacy",
        "absolute_position_encoding": None,
        "lr": 1e-5,
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    run_metadata = {
        "metadata_schema_version": 1,
        "input_dim": 71,
        "position_encoding": {"preset": "legacy"},
    }
    train._update_saved_config(config_path, **run_metadata)
    train._update_saved_config(
        config_path,
        class_weighting_applied=True,
        class_weighting_pos_weight=2.5,
    )

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["position_preset"] == "legacy"
    assert saved["absolute_position_encoding"] is None
    assert saved["lr"] == 1e-5
    assert saved["metadata_schema_version"] == 1
    assert saved["position_encoding"] == {"preset": "legacy"}
    assert saved["class_weighting_applied"] is True


def test_helpers_do_not_construct_datasets_or_models(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected construction")

    monkeypatch.setattr(train, "ChunkedVariantDataset", fail)
    monkeypatch.setattr(train, "create_model", fail)

    train.mapping_sha256({"BRCA1": 0})
    train.build_chromosome_id_to_name({"1": 0})
    train.build_training_run_metadata(
        input_dim=1,
        num_genes=1,
        num_chromosomes=1,
        genome_build="GRCh37",
        resolved_position_encoding=resolved_l3(),
        chrom_index={"1": 0},
        gene_mapping_sha256="genehash",
        chromosome_mapping_sha256="chromhash",
        training_mode="single_split",
    )
