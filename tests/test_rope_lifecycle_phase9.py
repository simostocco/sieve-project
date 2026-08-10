"""Phase 9C RoPE lifecycle diagnostics and integration tests."""

from __future__ import annotations

import argparse
import copy

import pytest
import torch

from scripts import train
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import learned_binned_layout_from_position_encoding_dict
from src.models.reconstruction import reconstruct_sieve_from_checkpoint
from src.models.sieve import SIEVE

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 2,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 12,
    "dropout": 0.0,
    "aggregation": "max",
    "aggregation_method": "mean",
    "num_covariates": 0,
    "classifier_type": "flatten",
}
CHROM_INDEX = {"1": 0, "2": 1, "X": 2}
SWAPPED_CHROM_INDEX = {"1": 1, "2": 0, "X": 2}
RENAMED_CHROM_INDEX = {"1": 0, "2": 1, "Y": 2}
DIFFERENT_CHROM_INDEX = {"A": 0, "B": 1, "C": 2}


def _args(*extra: str) -> argparse.Namespace:
    return train.parse_args(
        [
            "--level",
            "L3",
            "--latent-dim",
            str(MODEL_KWARGS["latent_dim"]),
            "--hidden-dim",
            str(MODEL_KWARGS["hidden_dim"]),
            "--num-heads",
            str(MODEL_KWARGS["num_heads"]),
            "--num-attention-layers",
            str(MODEL_KWARGS["num_attention_layers"]),
            *extra,
        ]
    )


def _rope_args(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    rope_coordinate_scale: float = 12345.0,
    rope_base: float = 4321.0,
) -> argparse.Namespace:
    args = [
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        absolute.value,
        "--relative-position-encoding",
        "rope",
        "--chromosome-encoding",
        chromosome.value,
        "--cross-chromosome-policy",
        cross_policy.value,
        "--rope-coordinate-scale",
        str(rope_coordinate_scale),
        "--rope-base",
        str(rope_base),
    ]
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        args.extend(["--position-dim", "4"])
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        args.extend(["--position-dim", "8", "--position-bin-size", "100000000"])
    return _args(*args)


def _resolve_custom(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.ROPE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
):
    kwargs = {}
    if relative is RelativePositionEncoding.ROPE:
        kwargs = {"rope_coordinate_scale": 12345.0, "rope_base": 4321.0}
    elif relative is RelativePositionEncoding.T5_BUCKET:
        kwargs = {"num_position_buckets": 8, "max_position_distance": 1000}
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        kwargs.update({"position_dim": 8, "position_bin_size": 100000000})
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            **kwargs,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=3,
    )


def _resolve_rope(
    *,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
):
    return train.prepare_training_position_encoding(
        _rope_args(
            chromosome=chromosome,
            absolute=absolute,
            cross_policy=cross_policy,
        ),
        AnnotationLevel.L3,
        num_chromosomes=3,
    )


def _run_metadata(config, *, chrom_index=CHROM_INDEX):
    return train.build_training_run_metadata(
        input_dim=config.input_dim,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        genome_build="GRCh37",
        resolved_position_encoding=config,
        chrom_index=chrom_index,
        gene_mapping_sha256="genehash",
        chromosome_mapping_sha256="chromhash",
        training_mode="cv",
    )


def _case_a_config(config, metadata):
    return {
        **MODEL_KWARGS,
        "level": "L3",
        **copy.deepcopy(metadata),
    }


def _layout_from_metadata(metadata):
    serialized = metadata["position_encoding"]
    return learned_binned_layout_from_position_encoding_dict(
        serialized,
        chromosome_mapping=serialized["chromosome"]["mapping"],
    )


def _source_model(config, layout=None) -> SIEVE:
    torch.manual_seed(1234)
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=config.chromosome.num_chromosomes,
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
        learned_binned_position_layout=layout,
    )
    model.eval()
    return model


def _old_source_model() -> SIEVE:
    model = SIEVE(
        input_dim=71,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        aggregation=MODEL_KWARGS["aggregation"],
        num_covariates=MODEL_KWARGS["num_covariates"],
        num_chromosomes=0,
        classifier_type=MODEL_KWARGS["classifier_type"],
    )
    model.eval()
    return model


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def test_rope_only_reconstruction_does_not_require_exact_chromosome_name_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.NONE)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=SWAPPED_CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_rope_with_learned_chromosome_embedding_requires_exact_chromosome_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)
    serialized = _case_a_config(config, metadata)
    checkpoint = _checkpoint(source)

    matching = reconstruct_sieve_from_checkpoint(
        serialized,
        checkpoint,
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    assert matching.is_new_schema is True
    assert matching.resolved_position_encoding == config

    with pytest.raises(ValueError, match="chromosome|mapping|chrom_index"):
        reconstruct_sieve_from_checkpoint(
            serialized,
            checkpoint,
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_learned_chromosome_embedding_rejects_wrong_chromosome_name_identity():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=RENAMED_CHROM_INDEX,
        )


def test_t5_with_learned_chromosome_embedding_uses_same_identity_guard():
    config = _resolve_custom(
        chromosome=ChromosomeEncoding.LEARNED,
        relative=RelativePositionEncoding.T5_BUCKET,
    )
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_rope_only_reconstruction_allows_different_valid_chromosome_names():
    config = _resolve_rope(chromosome=ChromosomeEncoding.NONE)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=DIFFERENT_CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_learned_binned_still_requires_exact_mapping_after_identity_refactor():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        relative=RelativePositionEncoding.ROPE,
        chromosome=ChromosomeEncoding.NONE,
    )
    metadata = _run_metadata(config)
    source = _source_model(config, _layout_from_metadata(metadata))

    matching = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )
    assert matching.is_new_schema is True

    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=SWAPPED_CHROM_INDEX,
        )


def test_learned_chromosome_embedding_rejects_dataset_count_without_mapping():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    with pytest.raises(ValueError, match="dataset_chrom_index"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
        )


def test_learned_chromosome_embedding_pure_checkpoint_reconstruction_succeeds():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    source = _source_model(config)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
    )

    assert result.is_new_schema is True
    assert result.resolved_position_encoding == config


def test_learned_chromosome_embedding_requires_saved_mapping():
    config = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    metadata = _run_metadata(config)
    metadata["position_encoding"]["chromosome"].pop("mapping")
    source = _source_model(config)

    with pytest.raises(ValueError, match="position_encoding.chromosome.mapping"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
        )


def test_case_b_and_case_c_remain_state_driven_compatibility_paths():
    intended = _resolve_rope(chromosome=ChromosomeEncoding.LEARNED)
    old_source = _old_source_model()
    transitional = {
        **MODEL_KWARGS,
        "level": "L3",
        "input_dim": 71,
        "position_encoding": intended.to_dict(),
        "position_encoding_execution": {"resolved_config_applied_to_model": False},
    }

    case_b = reconstruct_sieve_from_checkpoint(
        {**MODEL_KWARGS, "level": "L3"},
        _checkpoint(old_source),
        num_genes=5,
        dataset_num_chromosomes=3,
    )
    case_c = reconstruct_sieve_from_checkpoint(
        transitional,
        _checkpoint(old_source),
        num_genes=5,
        dataset_num_chromosomes=3,
    )

    assert case_b.is_new_schema is False
    assert case_c.is_new_schema is False
    assert not any("cross_chromosome_bias" in key for key in case_b.model.state_dict())
    assert not any("cross_chromosome_bias" in key for key in case_c.model.state_dict())


def test_direct_resolver_fixture_keeps_rope_non_default_values():
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.ROPE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            rope_coordinate_scale=12345.0,
            rope_base=4321.0,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=3,
    )

    assert config.relative.rope_coordinate_scale == 12345.0
    assert config.relative.rope_base == 4321.0
