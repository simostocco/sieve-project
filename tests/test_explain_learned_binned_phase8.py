"""Phase 8B4 learned-binned explanation integration tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import explain, train
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedIGMode,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import learned_binned_layout_from_position_encoding_dict
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.ig_mode import resolve_ig_mode
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


def _resolve_learned(
    *,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
):
    kwargs = {}
    if relative is RelativePositionEncoding.T5_BUCKET:
        kwargs = {"num_position_buckets": 8, "max_position_distance": 1000}
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.LEARNED_BINNED,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            position_dim=8,
            position_bin_size=100000000,
            **kwargs,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=3,
    )


def _resolve_none():
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


def _case_a_config(config, metadata):
    return {
        **MODEL_KWARGS,
        "level": "L3",
        **copy.deepcopy(metadata),
    }


def _source_model(config, layout=None):
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
    if config.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        with torch.no_grad():
            values = torch.arange(
                model.absolute_position_embedding.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(model.absolute_position_embedding.weight)
            model.absolute_position_embedding.weight.copy_(values / 10.0)
    model.eval()
    return model


def _checkpoint(model):
    return {"model_state_dict": copy.deepcopy(model.state_dict())}


def _dataset(*, chrom_index=CHROM_INDEX, num_chromosomes=3):
    return SimpleNamespace(
        num_genes=5,
        num_chromosomes=num_chromosomes,
        chrom_index=chrom_index,
    )


def _learned_fixture(
    *,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
):
    config = _resolve_learned(relative=relative, chromosome=chromosome)
    metadata = _run_metadata(config)
    layout = _layout_from_metadata(metadata)
    source = _source_model(config, layout)
    checkpoint = _checkpoint(source)
    reconstruction = explain._reconstruct_model_for_explanation(
        _case_a_config(config, metadata),
        checkpoint,
        _dataset(),
    )
    return config, metadata, layout, source, checkpoint, reconstruction


def _chunk(config, *, observed_value: float = 0.0):
    return {
        "content_features": torch.tensor(
            [
                [0.2, 1.0, 0.0, 0.0, 1.0, 0.3, 0.7],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.6, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "absolute_position_features": torch.full(
            (3, config.absolute.position_dim),
            observed_value,
            dtype=torch.float32,
        ),
        "positions": torch.tensor([1, 100000001, 0], dtype=torch.long),
        "chrom_ids": torch.tensor([0, 0, 999], dtype=torch.long),
        "gene_ids": torch.tensor([0, 1, 0], dtype=torch.long),
        "mask": torch.tensor([True, True, False]),
    }


def _content_ig(reconstruction, chunk):
    explainer = IntegratedGradientsExplainer(
        reconstruction.base_model,
        device="cpu",
        n_steps=4,
        ig_mode=ResolvedIGMode.CONTENT,
    )
    before = reconstruction.base_model.absolute_position_embedding.weight.detach().clone()
    attributions, positions, gene_ids, mask, chrom_ids = explain._attribute_chunk_for_ig(
        explainer=explainer,
        chunk=chunk,
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        device="cpu",
    )
    after = reconstruction.base_model.absolute_position_embedding.weight.detach().clone()
    return attributions, positions, gene_ids, mask, chrom_ids, before, after


def test_explain_reconstruction_for_learned_binned_restores_embedding_and_mapping():
    config, _metadata, layout, source, _checkpoint, reconstruction = _learned_fixture()

    assert reconstruction.is_new_schema is True
    assert reconstruction.resolved_position_encoding.absolute.encoding is (
        AbsolutePositionEncoding.LEARNED_BINNED
    )
    assert reconstruction.base_model.absolute_position_embedding.weight.shape == (
        layout.num_embeddings,
        config.absolute.position_dim,
    )
    assert torch.equal(
        reconstruction.base_model.absolute_position_embedding.weight,
        source.absolute_position_embedding.weight,
    )


@pytest.mark.parametrize(
    "chrom_index",
    [
        {"1": 0, "2": 1, "Y": 2},
        {"1": 1, "2": 0, "X": 2},
    ],
)
def test_explain_reconstruction_rejects_live_chromosome_mapping_mismatch(chrom_index):
    config = _resolve_learned()
    metadata = _run_metadata(config)
    source = _source_model(config, _layout_from_metadata(metadata))

    with pytest.raises(ValueError, match="exactly match"):
        explain._reconstruct_model_for_explanation(
            _case_a_config(config, metadata),
            _checkpoint(source),
            _dataset(chrom_index=chrom_index),
        )


def test_learned_binned_ig_policy_and_provenance_are_content_only():
    config, _metadata, _layout, _source, _checkpoint, reconstruction = _learned_fixture()
    run_config = reconstruction.effective_config

    assert resolve_ig_mode("auto", config=run_config, is_new_schema=True) is (
        ResolvedIGMode.CONTENT
    )
    assert resolve_ig_mode("content", config=run_config, is_new_schema=True) is (
        ResolvedIGMode.CONTENT
    )
    with pytest.raises(ValueError, match="custom positional execution"):
        explain._validate_ig_mode_for_reconstruction(ResolvedIGMode.LEGACY, reconstruction)

    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="auto",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        reconstruction=reconstruction,
        content_dim=config.content_dim,
        n_steps=4,
        max_variants=3,
    )

    assert metadata["absolute_position_encoding"] == "learned_binned"
    assert metadata["position_encoding_metadata_source"] == "reconstructed_resolved_config"
    assert metadata["resolved_ig_mode"] == "content"
    assert metadata["attribution_feature_space"] == "content"
    assert metadata["attribution_width"] == config.content_dim


def test_learned_binned_content_ig_uses_content_width_and_fixed_embedding_context():
    config, _metadata, _layout, source, _checkpoint, reconstruction = _learned_fixture()

    assert torch.equal(
        reconstruction.base_model.absolute_position_embedding.weight,
        source.absolute_position_embedding.weight,
    )

    attributions, positions, _gene_ids, mask, chrom_ids, before, after = _content_ig(
        reconstruction,
        _chunk(config),
    )

    assert attributions.shape == (1, 3, config.content_dim)
    assert attributions.shape[-1] != config.input_dim
    assert torch.isfinite(attributions).all()
    assert chrom_ids is not None
    assert torch.equal(chrom_ids, torch.tensor([[0, 0, 999]], dtype=torch.long))
    assert torch.equal(positions, torch.tensor([[1, 100000001, 0]], dtype=torch.long))
    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert torch.equal(before, after)


def test_learned_binned_content_ig_ignores_observed_absolute_feature_values():
    config, _metadata, _layout, _source, _checkpoint, reconstruction = _learned_fixture()

    attributions_a, *_rest_a, before, after = _content_ig(
        reconstruction,
        _chunk(config, observed_value=0.0),
    )
    attributions_b, *_rest_b, before_b, after_b = _content_ig(
        reconstruction,
        _chunk(config, observed_value=1000.0),
    )

    assert torch.equal(attributions_a, attributions_b)
    assert torch.equal(before, after)
    assert torch.equal(before_b, after_b)


def test_learned_binned_attention_split_path_ignores_observed_absolute_values():
    config, _metadata, _layout, _source, _checkpoint, reconstruction = _learned_fixture()
    analyzer = AttentionAnalyzer(reconstruction.model, device="cpu")
    chunk_a = _chunk(config, observed_value=0.0)
    chunk_b = _chunk(config, observed_value=1000.0)

    attention_a = analyzer.extract_attention_weights(
        None,
        chunk_a["positions"].unsqueeze(0),
        chunk_a["gene_ids"].unsqueeze(0),
        chunk_a["mask"].unsqueeze(0),
        chrom_ids=chunk_a["chrom_ids"].unsqueeze(0),
        content_features=chunk_a["content_features"].unsqueeze(0),
        absolute_position_features=chunk_a["absolute_position_features"].unsqueeze(0),
    )
    attention_b = analyzer.extract_attention_weights(
        None,
        chunk_b["positions"].unsqueeze(0),
        chunk_b["gene_ids"].unsqueeze(0),
        chunk_b["mask"].unsqueeze(0),
        chrom_ids=chunk_b["chrom_ids"].unsqueeze(0),
        content_features=chunk_b["content_features"].unsqueeze(0),
        absolute_position_features=chunk_b["absolute_position_features"].unsqueeze(0),
    )

    assert reconstruction.resolved_position_encoding.absolute.encoding is (
        AbsolutePositionEncoding.LEARNED_BINNED
    )
    assert len(attention_a) == MODEL_KWARGS["num_attention_layers"]
    assert attention_a[0].shape == (1, MODEL_KWARGS["num_heads"], 3, 3)
    assert torch.equal(attention_a[0], attention_b[0])


def test_learned_binned_t5_and_chromosome_embedding_reconstruct_for_explanation():
    config, _metadata, _layout, _source, _checkpoint, reconstruction = _learned_fixture(
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
    )

    layer = reconstruction.base_model.attention.attention_layers[0]
    assert reconstruction.resolved_position_encoding.relative.encoding is (
        RelativePositionEncoding.T5_BUCKET
    )
    assert layer.position_bias.weight.shape == (9, MODEL_KWARGS["num_heads"])
    assert layer.chrom_embedding.weight.shape == (4, MODEL_KWARGS["latent_dim"])
    assert reconstruction.base_model.absolute_position_embedding.weight.shape[-1] == (
        config.absolute.position_dim
    )


def test_non_learned_case_a_and_historical_cases_still_reconstruct_for_explanation():
    none_config = _resolve_none()
    none_metadata = _run_metadata(none_config, chrom_index={})
    none_source = _source_model(none_config)
    case_a = explain._reconstruct_model_for_explanation(
        _case_a_config(none_config, none_metadata),
        _checkpoint(none_source),
        _dataset(chrom_index={}, num_chromosomes=0),
    )
    assert case_a.is_new_schema is True
    assert case_a.resolved_position_encoding == none_config

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
    old_source.eval()
    case_b = explain._reconstruct_model_for_explanation(
        {**MODEL_KWARGS, "level": "L3"},
        _checkpoint(old_source),
        _dataset(),
    )
    transitional = {
        **MODEL_KWARGS,
        "level": "L3",
        "input_dim": 71,
        "position_encoding": _run_metadata(_resolve_learned())["position_encoding"],
        "position_encoding_execution": {"resolved_config_applied_to_model": False},
    }
    case_c = explain._reconstruct_model_for_explanation(
        transitional,
        _checkpoint(old_source),
        _dataset(),
    )

    assert case_b.is_new_schema is False
    assert case_c.is_new_schema is False
    assert case_b.resolved_position_encoding is None
    assert case_c.resolved_position_encoding is None
