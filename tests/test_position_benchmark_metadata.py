"""Tests for positional benchmark metadata identity and context checks."""

import copy

import pytest

from scripts.position_benchmark_metadata import (
    canonical_strategy_json,
    compare_contexts,
    extract_comparison_context,
    position_strategy_identity,
    require_authoritative_position_metadata,
    require_compatible_contexts,
)


def _base_position_encoding(relative=None):
    if relative is None:
        relative = {
            "type": "alibi_fixed",
            "num_buckets": None,
            "total_bias_rows": None,
            "max_distance_bp": None,
            "rope_coordinate_scale": None,
            "rope_base": None,
            "alibi_distance_function": "log1p",
            "alibi_distance_scale": 10000.0,
        }
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
            "requires_chrom_ids": True,
            "cross_chromosome_parameter": "learned_bias",
            "mapping": {"0": "1", "1": "2", "2": "X"},
        },
        "attribution": {"default_ig_mode": "content"},
        "content_dim": 7,
        "input_dim": 7,
    }


def _base_config():
    return {
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
        "position_encoding": _base_position_encoding(),
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


def _identity(config=None):
    return position_strategy_identity(_base_config() if config is None else config)


def _contexts(*configs):
    return [
        extract_comparison_context(config, run_id=f"run_{idx}")
        for idx, config in enumerate(configs)
    ]


def test_canonical_strategy_json_and_hash_are_deterministic():
    first = _identity()
    second = _identity()

    assert canonical_strategy_json(first.payload) == canonical_strategy_json(second.payload)
    assert first.hash == second.hash
    assert first.strategy_id == second.strategy_id


def test_same_architecture_different_dataset_hashes_keeps_strategy_id_but_fails_context():
    first = _base_config()
    second = copy.deepcopy(first)
    second["dataset_identity"]["gene_mapping_sha256"] = "different"

    assert (
        position_strategy_identity(first).strategy_id
        == position_strategy_identity(second).strategy_id
    )

    report = compare_contexts(_contexts(first, second))
    assert not report.compatible
    assert report.mismatches[0].field == "dataset_identity.gene_mapping_sha256"


def test_alibi_scale_changes_strategy_id_but_context_remains_compatible():
    first = _base_config()
    second = copy.deepcopy(first)
    second["position_encoding"]["relative"]["alibi_distance_scale"] = 20000.0

    assert (
        position_strategy_identity(first).strategy_id
        != position_strategy_identity(second).strategy_id
    )
    assert require_compatible_contexts(_contexts(first, second)).compatible


def test_mapping_and_binning_extensions_are_ignored_for_strategy_identity():
    first = _base_config()
    second = copy.deepcopy(first)
    second["position_encoding"]["chromosome"]["mapping"] = {"0": "chr1"}
    second["position_encoding"]["absolute"]["binning"] = {
        "schema_version": 1,
        "chromosome_lengths_bp": [100],
        "bins_per_chromosome": [1],
        "num_embeddings": 1,
    }

    assert (
        position_strategy_identity(first).strategy_id
        == position_strategy_identity(second).strategy_id
    )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda cfg: cfg["position_encoding"]["absolute"].pop("type"),
            "position_encoding.absolute.type",
        ),
        (
            lambda cfg: cfg["position_encoding"]["relative"].pop("alibi_distance_scale"),
            "position_encoding.relative.alibi_distance_scale",
        ),
        (
            lambda cfg: cfg["position_encoding"]["chromosome"].pop("encoding"),
            "position_encoding.chromosome.encoding",
        ),
    ],
)
def test_required_strategy_fields_are_validated(mutator, message):
    config = _base_config()
    mutator(config)

    with pytest.raises(ValueError, match=message):
        position_strategy_identity(config)


def test_authoritative_metadata_is_required():
    config = _base_config()
    config.pop("position_encoding")

    with pytest.raises(ValueError, match="position_encoding"):
        require_authoritative_position_metadata(config)


def test_unapplied_position_metadata_is_rejected():
    config = _base_config()
    config["position_encoding_execution"]["resolved_config_applied_to_model"] = False

    with pytest.raises(ValueError, match="resolved_config_applied_to_model"):
        position_strategy_identity(config)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda cfg: cfg["position_encoding"].update({"preset": "experimental"}),
            "position_encoding.preset",
        ),
        (
            lambda cfg: cfg["position_encoding"]["absolute"].update(
                {
                    "type": "sinusoidal",
                    "dim": True,
                    "coordinate_scale": 1.0,
                    "max_wavelength": 10000.0,
                }
            ),
            "position_encoding.absolute.dim",
        ),
        (
            lambda cfg: cfg["position_encoding"]["absolute"].update(
                {
                    "type": "sinusoidal",
                    "dim": 0,
                    "coordinate_scale": 1.0,
                    "max_wavelength": 10000.0,
                }
            ),
            "position_encoding.absolute.dim",
        ),
        (
            lambda cfg: cfg["position_encoding"]["absolute"].update(
                {"type": "learned_binned", "dim": 8, "bin_size_bp": 0}
            ),
            "position_encoding.absolute.bin_size_bp",
        ),
        (
            lambda cfg: cfg["position_encoding"].update(
                {
                    "relative": {
                        "type": "t5_bucket",
                        "num_buckets": 0,
                        "max_distance_bp": 100000,
                    }
                }
            ),
            "position_encoding.relative.num_buckets",
        ),
        (
            lambda cfg: cfg["position_encoding"].update(
                {
                    "relative": {
                        "type": "rope",
                        "rope_coordinate_scale": 1.0,
                        "rope_base": "bad",
                    }
                }
            ),
            "position_encoding.relative.rope_base",
        ),
        (
            lambda cfg: cfg["position_encoding"]["relative"].update({"alibi_distance_scale": -1.0}),
            "position_encoding.relative.alibi_distance_scale",
        ),
        (
            lambda cfg: cfg["position_encoding"]["relative"].update(
                {"alibi_distance_function": "sqrt"}
            ),
            "position_encoding.relative.alibi_distance_function",
        ),
        (
            lambda cfg: cfg["position_encoding"]["chromosome"].update({"encoding": "one_hot"}),
            "position_encoding.chromosome.encoding",
        ),
        (
            lambda cfg: cfg["position_encoding"]["chromosome"].update(
                {"cross_chromosome_policy": "bucket"}
            ),
            "position_encoding.chromosome.cross_chromosome_policy",
        ),
    ],
)
def test_malformed_strategy_values_are_rejected_before_hashing(mutator, message):
    config = _base_config()
    mutator(config)

    with pytest.raises(ValueError, match=message):
        position_strategy_identity(config)


@pytest.mark.parametrize(
    "field, value",
    [
        ("seed", 123),
        ("level", "L0"),
        ("content_dim", 5),
        ("preprocessed_data", "/data/other.pt"),
        ("vcf", "/data/other.vcf.gz"),
        ("phenotypes", "/data/other.tsv"),
        ("pc_map_sha256", "different-pc-hash"),
    ],
)
def test_context_mismatches_are_incompatible(field, value):
    first = _base_config()
    second = copy.deepcopy(first)
    second[field] = value

    report = compare_contexts(_contexts(first, second))

    assert not report.compatible
    assert any(mismatch.field == field for mismatch in report.mismatches)


def test_input_dim_difference_is_not_a_context_mismatch():
    first = _base_config()
    second = copy.deepcopy(first)
    second["input_dim"] = 71
    second["position_encoding"]["input_dim"] = 71

    assert require_compatible_contexts(_contexts(first, second)).compatible


def test_missing_required_context_field_is_a_compatibility_failure():
    config = _base_config()
    config.pop("class_weighting")

    with pytest.raises(ValueError, match="run 'missing_weighting'.*class_weighting"):
        extract_comparison_context(config, run_id="missing_weighting")


def test_two_configs_missing_same_required_context_field_cannot_compare_compatible():
    first = _base_config()
    second = copy.deepcopy(first)
    first.pop("class_weighting")
    second.pop("class_weighting")

    with pytest.raises(ValueError, match="run 'run_0'.*class_weighting"):
        _contexts(first, second)


def test_present_none_context_values_are_not_missing():
    config = _base_config()
    config["preprocessed_data"] = None

    context = extract_comparison_context(config, run_id="none_values_are_present")

    assert context.fields["gradient_clip"] is None
    assert context.fields["preprocessed_data"] is None
    assert context.fields["vcf"] is None
    assert context.fields["pc_map_sha256"] is None


def test_duplicate_run_ids_are_rejected_before_field_comparison():
    config = _base_config()
    contexts = [
        extract_comparison_context(config, run_id="duplicate"),
        extract_comparison_context(copy.deepcopy(config), run_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="duplicate.*'duplicate'"):
        compare_contexts(contexts)


def test_diagnostic_preserves_run_ids_and_values():
    first = _base_config()
    second = copy.deepcopy(first)
    second["seed"] = 999
    contexts = [
        extract_comparison_context(first, run_id="alibi_10k"),
        extract_comparison_context(second, run_id="alibi_20k"),
    ]

    report = compare_contexts(contexts)

    mismatch = next(item for item in report.mismatches if item.field == "seed")
    assert mismatch.values_by_run == {"alibi_10k": 42, "alibi_20k": 999}
