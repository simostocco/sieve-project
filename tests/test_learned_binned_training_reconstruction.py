"""Phase 8B3 learned-binned training and strict reconstruction tests."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable

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
from src.models.chunked_sieve import ChunkedSIEVEModel
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


def _learned_args(*extra: str) -> argparse.Namespace:
    return _args(
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        "learned_binned",
        "--relative-position-encoding",
        "none",
        "--chromosome-encoding",
        "none",
        "--cross-chromosome-policy",
        "separate",
        "--position-dim",
        "8",
        "--position-bin-size",
        "100000000",
        *extra,
    )


def _resolved_learned(*extra: str):
    return train.prepare_training_position_encoding(
        _learned_args(*extra),
        AnnotationLevel.L3,
        num_chromosomes=3,
    )


def _resolved_none():
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.NONE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=0,
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


def _layout_from_metadata(metadata):
    serialized = metadata["position_encoding"]
    return learned_binned_layout_from_position_encoding_dict(
        serialized,
        chromosome_mapping=serialized["chromosome"]["mapping"],
    )


def _training_layout(config, metadata):
    return train._training_learned_binned_layout_from_metadata(
        config,
        metadata["position_encoding"],
    )


def _case_a_config(config, metadata):
    return {
        **MODEL_KWARGS,
        "config_schema_version": 2,
        "metadata_schema_version": 1,
        "position_encoding_schema_version": config.schema_version,
        "input_dim": config.input_dim,
        "content_dim": config.content_dim,
        "num_genes": 5,
        "num_chromosomes": config.chromosome.num_chromosomes,
        "position_encoding": copy.deepcopy(metadata["position_encoding"]),
        "dataset_identity": {"mappings_artifact": "dataset_mappings.json"},
        "position_encoding_execution": {
            "schema_version": 2,
            "resolved_config_applied_to_model": True,
        },
    }


def _base_model(config, layout=None):
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


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def _chunked_checkpoint(base_model):
    chunked = ChunkedSIEVEModel(
        base_model,
        aggregation_method=MODEL_KWARGS["aggregation_method"],
    )
    return {"model_state_dict": copy.deepcopy(chunked.state_dict())}


def _assert_state_exact(source_state, target_state):
    assert set(source_state) == set(target_state)
    for key, tensor in source_state.items():
        assert target_state[key].shape == tensor.shape
        assert torch.equal(target_state[key], tensor)


def test_prepare_training_position_encoding_accepts_learned_binned_with_rope():
    resolved = _resolved_learned()

    assert resolved.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED

    rope_args = _learned_args(
        "--relative-position-encoding",
        "rope",
        "--rope-coordinate-scale",
        "1.0",
        "--rope-base",
        "10000.0",
    )
    rope_resolved = train.prepare_training_position_encoding(
        rope_args,
        AnnotationLevel.L3,
        num_chromosomes=3,
    )

    assert rope_resolved.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED
    assert rope_resolved.relative.encoding is RelativePositionEncoding.ROPE


def test_training_serialized_layout_is_model_construction_authority():
    config = _resolved_learned()
    metadata = _run_metadata(config)
    serialized_layout = _layout_from_metadata(metadata)
    training_layout = _training_layout(config, metadata)

    assert training_layout == serialized_layout
    assert training_layout.chromosome_lengths_bp == serialized_layout.chromosome_lengths_bp
    assert training_layout.bins_per_chromosome == serialized_layout.bins_per_chromosome
    assert training_layout.num_embeddings == serialized_layout.num_embeddings
    assert training_layout.chromosome_offsets == serialized_layout.chromosome_offsets

    model = train.create_training_model(
        args=_learned_args(),
        resolved_position_encoding=config,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        num_covariates=0,
        learned_binned_position_layout=training_layout,
    )

    assert model.base_model._absolute_position_runtime.layout == serialized_layout
    assert model.base_model.absolute_position_embedding.weight.shape == (
        serialized_layout.num_embeddings,
        config.absolute.position_dim,
    )
    assert torch.equal(
        model.base_model.absolute_position_embedding.weight,
        torch.zeros_like(model.base_model.absolute_position_embedding.weight),
    )


def test_training_model_split_forward_backward_reaches_learned_embedding_rows():
    config = _resolved_learned()
    layout = _training_layout(config, _run_metadata(config))
    model = train.create_training_model(
        args=_learned_args(),
        resolved_position_encoding=config,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        num_covariates=0,
        learned_binned_position_layout=layout,
    )

    content = torch.randn(1, 3, config.content_dim)
    observed_absolute = torch.zeros(1, 3, config.absolute.position_dim)
    positions = torch.tensor([[1, 100000001, 1]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 2]], dtype=torch.long)
    gene_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mask = torch.tensor([[True, True, True]])

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed_absolute,
    )
    logits.sum().backward()

    grad = model.base_model.absolute_position_embedding.weight.grad
    assert grad is not None
    assert torch.any(grad[0] != 0)
    assert torch.any(grad[1] != 0)
    assert torch.any(grad[6] != 0)


def test_non_learned_training_state_surface_is_unchanged():
    config = _resolved_none()
    metadata = _run_metadata(config, chrom_index={})
    layout = _training_layout(config, metadata)

    model = train.create_training_model(
        args=_args(),
        resolved_position_encoding=config,
        num_genes=5,
        num_chromosomes=0,
        num_covariates=0,
        learned_binned_position_layout=layout,
    )

    assert layout is None
    assert not any("absolute_position_embedding.weight" in key for key in model.state_dict())


@pytest.mark.parametrize("chunked", [False, True])
def test_case_a_learned_binned_strict_round_trip(chunked):
    config = _resolved_learned()
    metadata = _run_metadata(config)
    layout = _layout_from_metadata(metadata)
    source = _base_model(config, layout)
    checkpoint = _chunked_checkpoint(source) if chunked else _checkpoint(source)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        checkpoint,
        num_genes=5,
        dataset_num_chromosomes=3,
        dataset_chrom_index=CHROM_INDEX,
    )

    assert result.is_new_schema is True
    assert result.is_chunked_checkpoint is chunked
    assert result.resolved_position_encoding == config
    assert result.base_model.absolute_position_embedding.weight.shape == (
        layout.num_embeddings,
        config.absolute.position_dim,
    )
    if chunked:
        assert "base_model.absolute_position_embedding.weight" in checkpoint["model_state_dict"]
    else:
        assert "absolute_position_embedding.weight" in checkpoint["model_state_dict"]
    _assert_state_exact(checkpoint["model_state_dict"], result.model.state_dict())


def test_case_a_learned_binned_pure_checkpoint_reconstruction_allows_no_live_dataset():
    config = _resolved_learned()
    metadata = _run_metadata(config)
    layout = _layout_from_metadata(metadata)
    source = _base_model(config, layout)

    result = reconstruct_sieve_from_checkpoint(
        _case_a_config(config, metadata),
        _checkpoint(source),
        num_genes=5,
    )

    assert result.base_model.absolute_position_embedding.weight.shape == (
        layout.num_embeddings,
        config.absolute.position_dim,
    )


@pytest.mark.parametrize(
    ("current_mapping", "message"),
    [
        ({"1": 0, "X": 1, "2": 2}, "exactly match"),
        ({"1": 1, "2": 0, "X": 2}, "exactly match"),
        ({"1": 0, "3": 1, "X": 2}, "exactly match"),
        ({"1": 0, "2": True, "X": 2}, "non-negative integers"),
    ],
)
def test_case_a_learned_binned_rejects_live_mapping_mismatch(current_mapping, message):
    config = _resolved_learned()
    metadata = _run_metadata(config)
    source = _base_model(config, _layout_from_metadata(metadata))

    with pytest.raises(ValueError, match=message):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
            dataset_chrom_index=current_mapping,
        )


def test_case_a_learned_binned_rejects_count_without_live_mapping():
    config = _resolved_learned()
    metadata = _run_metadata(config)
    source = _base_model(config, _layout_from_metadata(metadata))

    with pytest.raises(ValueError, match="dataset_chrom_index"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            _checkpoint(source),
            num_genes=5,
            dataset_num_chromosomes=3,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data["absolute"].pop("binning"), "binning"),
        (
            lambda data: data["absolute"]["binning"].update(num_embeddings=99),
            "num_embeddings",
        ),
    ],
)
def test_case_a_learned_binned_requires_valid_serialized_binning(mutator, message):
    config = _resolved_learned()
    metadata = _run_metadata(config)
    source = _base_model(config, _layout_from_metadata(metadata))
    serialized = _case_a_config(config, metadata)
    mutator(serialized["position_encoding"])

    with pytest.raises(ValueError, match=message):
        reconstruct_sieve_from_checkpoint(
            serialized,
            _checkpoint(source),
            num_genes=5,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.update(
            {"absolute_position_embedding.weight": state["absolute_position_embedding.weight"][:-1]}
        ),
        lambda state: state.update(
            {
                "absolute_position_embedding.weight": state["absolute_position_embedding.weight"][
                    :, :-1
                ]
            }
        ),
        lambda state: state.pop("absolute_position_embedding.weight"),
    ],
)
def test_case_a_learned_binned_strict_load_rejects_embedding_corruption(
    mutator: Callable[[dict[str, torch.Tensor]], object],
):
    config = _resolved_learned()
    metadata = _run_metadata(config)
    source = _base_model(config, _layout_from_metadata(metadata))
    checkpoint = _checkpoint(source)
    mutator(checkpoint["model_state_dict"])

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
        )


def test_case_a_non_learned_strict_load_rejects_unexpected_learned_embedding():
    config = _resolved_none()
    metadata = _run_metadata(config, chrom_index={})
    source = _base_model(config)
    checkpoint = _checkpoint(source)
    checkpoint["model_state_dict"]["absolute_position_embedding.weight"] = torch.zeros(1, 8)

    with pytest.raises(RuntimeError):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
        )


def test_case_a_reconciliation_rejects_learned_binning_metadata_conflict():
    config = _resolved_learned()
    metadata = _run_metadata(config)
    source = _base_model(config, _layout_from_metadata(metadata))
    checkpoint = _checkpoint(source)
    checkpoint["metadata"] = _case_a_config(config, metadata)
    checkpoint["metadata"]["position_encoding"]["absolute"]["binning"]["num_embeddings"] += 1

    with pytest.raises(ValueError, match="position_encoding.absolute.binning.num_embeddings"):
        reconstruct_sieve_from_checkpoint(
            _case_a_config(config, metadata),
            checkpoint,
            num_genes=5,
        )


def test_case_b_and_case_c_do_not_execute_learned_binned_layout_metadata():
    old_source = SIEVE(
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
    intended = _resolved_learned()
    transitional = {
        **MODEL_KWARGS,
        "input_dim": 71,
        "position_encoding": intended.to_dict(),
        "position_encoding_execution": {"resolved_config_applied_to_model": False},
    }

    case_b = reconstruct_sieve_from_checkpoint(
        {**MODEL_KWARGS},
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
    assert not any("absolute_position_embedding.weight" in key for key in case_b.model.state_dict())
    assert not any("absolute_position_embedding.weight" in key for key in case_c.model.state_dict())
