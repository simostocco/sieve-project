"""Tests for Phase 7B3 training positional configuration integration."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from scripts import explain, train
from src.encoding.chunked_dataset import collate_chunks
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedIGMode,
)
from src.explain.gradients import ContentSIEVEWrapper
from src.models.position_runtime import (
    NoAbsolutePositionRuntime,
    NoRelativePositionRuntime,
    T5RelativePositionRuntime,
)
from src.models.reconstruction import reconstruct_sieve_from_checkpoint
from src.training.loss import SIEVELoss
from src.training.trainer import Trainer


def _args(*extra: str) -> argparse.Namespace:
    return train.parse_args(["--level", "L3", *extra])


def _custom_args(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    extra: list[str] | None = None,
) -> argparse.Namespace:
    return _args(
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        absolute.value,
        "--relative-position-encoding",
        relative.value,
        "--chromosome-encoding",
        chromosome.value,
        "--cross-chromosome-policy",
        cross_policy.value,
        *(extra or []),
    )


def _resolve(
    args: argparse.Namespace,
    level: AnnotationLevel = AnnotationLevel.L3,
    *,
    num_chromosomes: int = 3,
):
    return train.prepare_training_position_encoding(
        args,
        level,
        num_chromosomes=num_chromosomes,
    )


def _model_args() -> argparse.Namespace:
    return argparse.Namespace(
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
        aggregation_method="mean",
        classifier_type="flatten",
    )


def _metadata(config, *, training_mode: str = "cv"):
    chrom_index = {} if config.chromosome.num_chromosomes == 0 else {"1": 0, "2": 1, "X": 2}
    return train.build_training_run_metadata(
        input_dim=config.input_dim,
        num_genes=5,
        num_chromosomes=config.chromosome.num_chromosomes,
        genome_build="GRCh37",
        resolved_position_encoding=config,
        chrom_index=chrom_index,
        gene_mapping_sha256="genehash",
        chromosome_mapping_sha256="chromhash",
        training_mode=training_mode,
    )


@pytest.mark.parametrize(
    "absolute",
    [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL],
)
@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.NONE, RelativePositionEncoding.T5_BUCKET],
)
@pytest.mark.parametrize(
    "chromosome",
    [ChromosomeEncoding.NONE, ChromosomeEncoding.LEARNED],
)
@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_supported_phase7_custom_matrix_resolves_for_training(
    absolute,
    relative,
    chromosome,
    cross_policy,
):
    args = _custom_args(
        absolute=absolute,
        relative=relative,
        chromosome=chromosome,
        cross_policy=cross_policy,
    )

    resolved = _resolve(args, num_chromosomes=3)

    assert resolved.preset is PositionPreset.CUSTOM
    assert resolved.absolute.encoding is absolute
    assert resolved.relative.encoding is relative
    assert resolved.chromosome.encoding is chromosome
    assert resolved.chromosome.cross_chromosome_policy is cross_policy


@pytest.mark.parametrize(
    "absolute",
    [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL],
)
def test_non_chromosome_aware_control_resolves_with_zero_chromosomes(absolute):
    args = _custom_args(absolute=absolute)

    resolved = _resolve(args, num_chromosomes=0)

    assert resolved.chromosome.requires_chrom_ids is False
    assert resolved.chromosome.num_chromosomes == 0


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_alibi_training_position_encoding_is_supported(relative):
    args = _args(
        "--position-preset",
        "custom",
        "--absolute-position-encoding",
        "none",
        "--relative-position-encoding",
        relative.value,
        "--chromosome-encoding",
        "none",
        "--cross-chromosome-policy",
        "separate",
        "--alibi-distance-function",
        "linear",
        "--alibi-distance-scale",
        "25.0",
    )

    resolved = _resolve(args, num_chromosomes=3)

    assert resolved.relative.encoding is relative
    assert resolved.relative.alibi_distance_scale == 25.0
    assert resolved.chromosome.cross_chromosome_parameter == "learned_bias"

    model = train.create_model(
        input_dim=resolved.input_dim,
        num_genes=5,
        latent_dim=8,
        # The genomic ALiBi prior (fixed and learned) is defined only for 4 heads.
        num_heads=4,
        num_attention_layers=1,
        hidden_dim=10,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        position_encoding=resolved,
    )
    layer = model.base_model.attention.attention_layers[0]
    if relative is RelativePositionEncoding.ALIBI_LEARNED:
        assert layer.alibi_slope_logits.shape == (4,)
        assert layer.alibi_slope_logits.requires_grad is True
    else:
        assert layer.alibi_slope_logits is None


def test_input_dim_authority_comes_from_resolved_config():
    absolute_none = _resolve(_custom_args(absolute=AbsolutePositionEncoding.NONE))
    sinusoidal = _resolve(
        _custom_args(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            extra=["--position-dim", "8"],
        )
    )
    legacy = _resolve(_args("--position-preset", "legacy"))
    l0_none = _resolve(
        _custom_args(absolute=AbsolutePositionEncoding.NONE),
        AnnotationLevel.L0,
        num_chromosomes=0,
    )

    assert absolute_none.input_dim == 7
    assert sinusoidal.input_dim == 15
    assert legacy.input_dim == 71
    assert l0_none.input_dim == 1


def test_create_model_keeps_no_config_historical_compatibility():
    model = train.create_model(
        input_dim=71,
        num_genes=5,
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
    )

    assert model.base_model.position_encoding is None
    assert model.base_model.variant_encoder.encoder[0].in_features == 71


def test_create_model_applies_custom_absolute_none_relative_none_surface():
    resolved = _resolve(_custom_args(absolute=AbsolutePositionEncoding.NONE), num_chromosomes=0)

    model = train.create_model(
        input_dim=resolved.input_dim,
        num_genes=5,
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
        num_chromosomes=0,
        position_encoding=resolved,
    )

    layer = model.base_model.attention.attention_layers[0]
    assert model.base_model.position_encoding is resolved
    assert model.base_model.variant_encoder.encoder[0].in_features == resolved.input_dim
    assert isinstance(model.base_model._absolute_position_runtime, NoAbsolutePositionRuntime)
    assert isinstance(layer._relative_position_runtime, NoRelativePositionRuntime)
    assert not any("position_bias.weight" in key for key in model.state_dict())
    assert not any("chrom_embedding.weight" in key for key in model.state_dict())


def test_create_model_applies_t5_mask_learned_chromosome_surface():
    resolved = _resolve(
        _custom_args(
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            cross_policy=CrossChromosomePolicy.MASK,
            extra=["--num-position-buckets", "8", "--max-position-distance", "1000"],
        )
    )

    model = train.create_model(
        input_dim=resolved.input_dim,
        num_genes=5,
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        position_encoding=resolved,
    )

    layer = model.base_model.attention.attention_layers[0]
    assert model.base_model.position_encoding is resolved
    assert isinstance(layer._relative_position_runtime, T5RelativePositionRuntime)
    assert layer.position_bias.weight.shape == (8, 2)
    assert layer.chrom_embedding.weight.shape == (4, 8)


def test_create_model_applies_explicit_resolved_legacy_chromosome_embedding():
    resolved = _resolve(_args("--position-preset", "legacy"))

    model = train.create_model(
        input_dim=resolved.input_dim,
        num_genes=5,
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        position_encoding=resolved,
    )

    layer = model.base_model.attention.attention_layers[0]
    assert model.base_model.position_encoding is resolved
    assert layer.chrom_embedding.weight.shape == (4, 8)


def test_create_training_model_uses_same_resolved_authority_for_cv_and_single_split():
    resolved = _resolve(
        _custom_args(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            extra=["--position-dim", "8"],
        )
    )
    args = _model_args()

    cv_model = train.create_training_model(
        args=args,
        resolved_position_encoding=resolved,
        num_genes=5,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        num_covariates=0,
    )
    single_model = train.create_training_model(
        args=args,
        resolved_position_encoding=resolved,
        num_genes=5,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        num_covariates=0,
    )

    assert cv_model.base_model.position_encoding is resolved
    assert single_model.base_model.position_encoding is resolved
    assert cv_model.base_model.variant_encoder.encoder[0].in_features == resolved.input_dim
    assert single_model.base_model.variant_encoder.encoder[0].in_features == resolved.input_dim
    assert cv_model.base_model.num_chromosomes == resolved.chromosome.num_chromosomes
    assert single_model.base_model.num_chromosomes == resolved.chromosome.num_chromosomes


def test_new_normalized_legacy_uses_learned_chromosome_in_training_model():
    resolved = _resolve(_args("--position-preset", "legacy"), num_chromosomes=3)
    # The historical single-split omission is old-checkpoint compatibility only;
    # new training always applies the resolved legacy chromosome configuration.
    model = train.create_training_model(
        args=_model_args(),
        resolved_position_encoding=resolved,
        num_genes=5,
        num_chromosomes=3,
        num_covariates=0,
    )

    assert model.base_model.attention.attention_layers[0].chrom_embedding is not None


def test_execution_metadata_for_no_position_control():
    resolved = _resolve(_custom_args(absolute=AbsolutePositionEncoding.NONE), num_chromosomes=0)

    metadata = train.build_position_encoding_execution_metadata(
        resolved_position_encoding=resolved,
        training_mode="cv",
    )

    assert metadata == {
        "schema_version": 2,
        "source": "resolved_position_encoding_applied_to_model",
        "resolved_config_applied_to_model": True,
        "training_mode": "cv",
        "preset": "custom",
        "absolute_position_encoding": "none",
        "absolute_position_dim": 0,
        "relative_position_encoding": "none",
        "position_bias_rows": None,
        "chromosome_encoding": "none",
        "chromosome_embedding_executed": False,
        "cross_chromosome_policy": "separate",
        "cross_chromosome_mask_executed": False,
        "requires_chrom_ids": False,
        "chrom_ids_passed_to_attention": True,
        "model_num_chromosomes": 0,
        "input_dim": resolved.input_dim,
        "content_dim": resolved.content_dim,
    }


def test_execution_metadata_for_t5_mask_learned_chromosome_and_training_modes():
    resolved = _resolve(
        _custom_args(
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            cross_policy=CrossChromosomePolicy.MASK,
            extra=["--num-position-buckets", "8", "--max-position-distance", "1000"],
        )
    )

    cv = train.build_position_encoding_execution_metadata(
        resolved_position_encoding=resolved,
        training_mode="cv",
    )
    single = train.build_position_encoding_execution_metadata(
        resolved_position_encoding=resolved,
        training_mode="single_split",
    )

    assert cv["position_bias_rows"] == 8
    assert cv["chromosome_embedding_executed"] is True
    assert cv["cross_chromosome_mask_executed"] is True
    assert cv["requires_chrom_ids"] is True
    assert {k: v for k, v in cv.items() if k != "training_mode"} == {
        k: v for k, v in single.items() if k != "training_mode"
    }


def test_execution_metadata_for_normalized_legacy():
    resolved = _resolve(_args("--position-preset", "legacy"))

    metadata = train.build_position_encoding_execution_metadata(
        resolved_position_encoding=resolved,
        training_mode="single_split",
    )

    assert metadata["preset"] == "legacy"
    assert metadata["absolute_position_encoding"] == "sinusoidal"
    assert metadata["relative_position_encoding"] == "t5_bucket"
    assert metadata["chromosome_encoding"] == "learned"
    assert metadata["cross_chromosome_policy"] == "separate"
    assert metadata["resolved_config_applied_to_model"] is True


def test_training_run_metadata_schema_versions_and_consistency_guards():
    resolved = _resolve(_custom_args(absolute=AbsolutePositionEncoding.NONE), num_chromosomes=0)

    metadata = _metadata(resolved, training_mode="single_split")

    assert metadata["config_schema_version"] == 2
    assert metadata["metadata_schema_version"] == 1
    assert metadata["position_encoding_schema_version"] == resolved.schema_version
    assert metadata["input_dim"] == resolved.input_dim
    assert metadata["content_dim"] == resolved.content_dim
    assert metadata["num_chromosomes"] == resolved.chromosome.num_chromosomes
    assert metadata["position_encoding"]["input_dim"] == resolved.input_dim
    assert metadata["position_encoding_execution"]["input_dim"] == resolved.input_dim
    with pytest.raises(ValueError, match="input_dim"):
        train.build_training_run_metadata(
            input_dim=resolved.input_dim + 1,
            num_genes=5,
            num_chromosomes=resolved.chromosome.num_chromosomes,
            genome_build="GRCh37",
            resolved_position_encoding=resolved,
            chrom_index={},
            gene_mapping_sha256="genehash",
            chromosome_mapping_sha256="chromhash",
            training_mode="cv",
        )
    with pytest.raises(ValueError, match="num_chromosomes"):
        train.build_training_run_metadata(
            input_dim=resolved.input_dim,
            num_genes=5,
            num_chromosomes=1,
            genome_build="GRCh37",
            resolved_position_encoding=resolved,
            chrom_index={},
            gene_mapping_sha256="genehash",
            chromosome_mapping_sha256="chromhash",
            training_mode="cv",
        )


def test_fold_config_sections_match_parent_run_metadata(tmp_path):
    resolved = _resolve(
        _custom_args(
            absolute=AbsolutePositionEncoding.SINUSOIDAL,
            relative=RelativePositionEncoding.T5_BUCKET,
            chromosome=ChromosomeEncoding.LEARNED,
            extra=["--position-dim", "8"],
        )
    )
    run_metadata = _metadata(resolved, training_mode="cv")
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    args = argparse.Namespace(
        experiment_name="phase7",
        level="L3",
        latent_dim=8,
        hidden_dim=10,
        num_attention_layers=1,
        num_heads=2,
        chunk_size=3000,
        chunk_overlap=0,
        aggregation_method="mean",
        classifier_type="flatten",
        lr=1e-3,
        lambda_attr=0.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        gradient_clip=None,
        early_stopping=2,
        epochs=1,
        seed=42,
        genome_build="GRCh37",
        sex_map=None,
        pc_map=None,
        num_pcs=0,
        pc_map_sha256=None,
        preprocessed_data=None,
        vcf=None,
        phenotypes=None,
    )

    train.save_fold_config(fold_dir, 0, args, run_metadata=run_metadata)
    config = yaml.safe_load((fold_dir / "config.yaml").read_text(encoding="utf-8"))

    for key in (
        "config_schema_version",
        "input_dim",
        "content_dim",
        "num_genes",
        "num_chromosomes",
        "position_encoding",
        "position_encoding_execution",
        "dataset_identity",
    ):
        assert config[key] == run_metadata[key]


def test_dataset_mapping_identity_payload_is_unchanged_by_strategy_metadata():
    payload = train.build_dataset_mappings_payload({"BRCA1": 0}, {"1": 0, "X": 1})

    assert payload["gene_index"] == {"BRCA1": 0}
    assert payload["chrom_index"] == {"1": 0, "X": 1}
    assert payload["chromosome_id_to_name"] == {"0": "1", "1": "X"}
    assert payload["schema_version"] == 1


def test_checkpoint_metadata_and_strategy_state_surface_round_trip(tmp_path):
    resolved = _resolve(_custom_args(absolute=AbsolutePositionEncoding.NONE), num_chromosomes=0)
    model = train.create_model(
        input_dim=resolved.input_dim,
        num_genes=5,
        latent_dim=8,
        num_heads=2,
        num_attention_layers=1,
        hidden_dim=10,
        num_chromosomes=0,
        position_encoding=resolved,
    )
    run_metadata = _metadata(resolved, training_mode="single_split")
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=SIEVELoss(),
        device="cpu",
        checkpoint_dir=tmp_path,
        checkpoint_metadata=run_metadata,
    )

    trainer.save_checkpoint("model.pt", {"auc": 0.5})
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)

    assert checkpoint["metadata"] == run_metadata
    assert checkpoint["metadata"]["position_encoding"] == run_metadata["position_encoding"]
    assert (
        checkpoint["metadata"]["position_encoding_execution"]
        == run_metadata["position_encoding_execution"]
    )
    assert "metadata" in checkpoint
    assert not any("position_bias.weight" in key for key in checkpoint["model_state_dict"])
    assert not any("chrom_embedding.weight" in key for key in checkpoint["model_state_dict"])


# Thesis RQ1.2-RQ1.4 on test_data/small (199 genes, chromosome 1) with configuration G.
# The seven thesis strategies. Values: (axes, expected positional state).
_LAYER0 = "attention.attention_layers.0."
_ALIBI = {RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED}
_ZERO_INITIALISED_POSITIONAL_TABLES = (
    "absolute_position_embedding",
    "position_bias",
    "cross_chromosome_bias",
)
THESIS_STRATEGIES = {
    "none": ({}, {}),
    "sinusoidal": ({"absolute": AbsolutePositionEncoding.SINUSOIDAL}, {}),
    "learned_binned": (
        {"absolute": AbsolutePositionEncoding.LEARNED_BINNED},
        {"absolute_position_embedding.weight": (24926, 64)},
    ),
    "t5_bucket": (
        {"relative": RelativePositionEncoding.T5_BUCKET},
        {_LAYER0 + "position_bias.weight": (33, 4)},
    ),
    "rope": (
        {"relative": RelativePositionEncoding.ROPE},
        {_LAYER0 + "cross_chromosome_bias": (4,)},
    ),
    "alibi_fixed": (
        {"relative": RelativePositionEncoding.ALIBI_FIXED},
        {_LAYER0 + "cross_chromosome_bias": (4,)},
    ),
    "alibi_learned": (
        {"relative": RelativePositionEncoding.ALIBI_LEARNED},
        {_LAYER0 + "cross_chromosome_bias": (4,), _LAYER0 + "alibi_slope_logits": (4,)},
    ),
}


def _config_g_training_model(dataset, config_g_args, axes):
    """Resolve, serialise and build the chunked model through the train.py steps."""
    args = _custom_args(**axes, extra=config_g_args)
    config = _resolve(args, num_chromosomes=dataset.num_chromosomes)
    metadata = train.build_training_run_metadata(
        input_dim=config.input_dim,
        num_genes=dataset.num_genes,
        num_chromosomes=dataset.num_chromosomes,
        genome_build=args.genome_build,
        resolved_position_encoding=config,
        chrom_index=dataset.chrom_index,
        gene_mapping_sha256=train.mapping_sha256(dataset.gene_index),
        chromosome_mapping_sha256=train.mapping_sha256(dataset.chrom_index),
        training_mode="single_split",
    )
    layout = train._training_learned_binned_layout_from_metadata(
        config, metadata["position_encoding"]
    )
    model = train.create_training_model(
        args=args,
        resolved_position_encoding=config,
        num_genes=dataset.num_genes,
        num_chromosomes=dataset.num_chromosomes,
        num_covariates=0,
        learned_binned_position_layout=layout,
    )
    return args, config, metadata, layout, model


@pytest.mark.parametrize("strategy", THESIS_STRATEGIES)
def test_rq1_2_position_strategy_instantiates_with_configuration_g(
    small_dataset, config_g_args, strategy
):
    axes, positional_state = THESIS_STRATEGIES[strategy]
    args, config, _, layout, chunked_model = _config_g_training_model(
        small_dataset, config_g_args, axes
    )
    model = chunked_model.base_model
    (layer,) = model.attention.attention_layers
    absolute_runtime = model._absolute_position_runtime
    relative_runtime = layer._relative_position_runtime
    a, r = config.absolute, config.relative

    # Configuration G reaches the model; widths follow the resolved config.
    assert (args.lr, args.lambda_attr) == (1e-5, 0.1)
    assert model.position_encoding == config
    assert config.input_dim == config.content_dim + (a.position_dim or 0)
    assert model.variant_encoder.encoder[0].weight.shape == (64, config.input_dim)
    assert (layer.latent_dim, layer.num_heads, layer.head_dim) == (32, 4, 8)
    assert {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if any(name in key for name in ("position", "chrom", "alibi"))
    } == positional_state

    # Positional runtime parameters are the resolved values.
    if a.encoding is AbsolutePositionEncoding.SINUSOIDAL:
        assert (
            absolute_runtime.position_dim,
            absolute_runtime.coordinate_scale,
            absolute_runtime.max_wavelength,
        ) == (a.position_dim, a.coordinate_scale, a.max_wavelength)
    elif a.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        assert absolute_runtime.embedding is model.absolute_position_embedding
        assert (absolute_runtime.position_dim, absolute_runtime.bin_size_bp) == (
            a.position_dim,
            a.bin_size_bp,
        )
        assert absolute_runtime.layout == layout
    else:
        assert isinstance(absolute_runtime, NoAbsolutePositionRuntime)

    if r.encoding is RelativePositionEncoding.T5_BUCKET:
        assert (relative_runtime.num_position_buckets, relative_runtime.max_distance) == (
            r.num_buckets,
            r.max_distance_bp,
        )
        assert layer.position_bias.num_embeddings == r.total_bias_rows
    elif r.encoding is RelativePositionEncoding.ROPE:
        assert (relative_runtime.coordinate_scale, relative_runtime.rope_base) == (
            r.rope_coordinate_scale,
            r.rope_base,
        )
        assert relative_runtime.head_dim == layer.head_dim
    elif r.encoding in _ALIBI:
        assert (relative_runtime.distance_function, relative_runtime.distance_scale) == (
            r.alibi_distance_function,
            r.alibi_distance_scale,
        )
    else:
        assert isinstance(relative_runtime, NoRelativePositionRuntime)


def _small_data_logits(model, dataset) -> torch.Tensor:
    """Eval-mode logits for all 20 samples, fed the split tensors training uses."""
    batch = collate_chunks([dataset[i] for i in range(len(dataset))])
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            None,
            batch["positions"],
            batch["gene_ids"],
            batch["mask"],
            chrom_ids=batch["chrom_ids"],
            content_features=batch["content_features"],
            absolute_position_features=batch["absolute_position_features"],
        )
    return logits.flatten()


@pytest.mark.parametrize("topology", ["base", "chunked"])
@pytest.mark.parametrize("strategy", THESIS_STRATEGIES)
def test_rq1_3_strict_reconstruction_restores_state_and_outputs(
    small_dataset, config_g_args, strategy, topology
):
    args, config, metadata, _, model = _config_g_training_model(
        small_dataset, config_g_args, THESIS_STRATEGIES[strategy][0]
    )
    # The positional tables start at zero, so a fresh model would match them
    # even if they were not restored. Give them deterministic non-zero values.
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(0)
        for name, parameter in model.named_parameters():
            if any(table in name for table in _ZERO_INITIALISED_POSITIONAL_TABLES):
                parameter.copy_(0.1 * torch.randn_like(parameter))
    source = model if topology == "chunked" else model.base_model
    checkpoint = {
        "model_state_dict": copy.deepcopy(source.state_dict()),
        "metadata": copy.deepcopy(metadata),
    }

    # train.py writes vars(args) plus the run metadata to config.yaml.
    result = reconstruct_sieve_from_checkpoint(
        {**vars(args), **metadata},
        checkpoint,
        num_genes=small_dataset.num_genes,
        dataset_num_chromosomes=small_dataset.num_chromosomes,
        dataset_chrom_index=small_dataset.chrom_index,
    )

    assert result.is_new_schema is True
    assert result.is_chunked_checkpoint is (topology == "chunked")
    assert result.resolved_position_encoding == config
    restored = result.model.state_dict()
    assert restored.keys() == checkpoint["model_state_dict"].keys()
    for key, tensor in checkpoint["model_state_dict"].items():
        assert torch.equal(restored[key], tensor), key
    assert torch.equal(
        _small_data_logits(result.base_model, small_dataset),
        _small_data_logits(model.base_model, small_dataset),
    )


def _run_script_main(monkeypatch, script, args) -> None:
    """Run ``script.main()`` on ``args`` without the CO2 measurement wrapper."""
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(
        script,
        "run_with_co2_tracking",
        lambda _stage, _output_dir, operation, *operation_args: operation(*operation_args),
    )
    script.main()


@pytest.mark.parametrize("strategy", THESIS_STRATEGIES)
def test_rq1_4_training_checkpoint_reconstruction_explanation_lifecycle(
    small_dataset, config_g_args, strategy, tmp_path, monkeypatch
):
    data_path = Path(__file__).parent.parent / "test_data" / "small" / "preprocessed_test.pt"
    run_dir, explain_dir = tmp_path / "run", tmp_path / "explain"
    # Minimum training: one epoch on CPU with SIEVE's default single split.
    train_args = _custom_args(
        **THESIS_STRATEGIES[strategy][0],
        extra=[
            *config_g_args,
            "--preprocessed-data", str(data_path),
            "--epochs", "1",
            "--device", "cpu",
            "--output-dir", str(tmp_path),
            "--experiment-name", run_dir.name,
        ],
    )
    config = _resolve(train_args, num_chromosomes=small_dataset.num_chromosomes)

    _run_script_main(monkeypatch, train, train_args)

    # Checkpoint: written by training and strictly reconstructed from config.yaml.
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    reconstruction = explain._reconstruct_model_for_explanation(
        saved_config, checkpoint, small_dataset
    )
    assert reconstruction.is_new_schema is True
    assert reconstruction.is_chunked_checkpoint is True
    assert reconstruction.resolved_position_encoding == config

    # Explanation through explain.py on the trained run.
    _run_script_main(
        monkeypatch,
        explain,
        explain.parse_args(
            [
                "--experiment-dir", str(run_dir),
                "--preprocessed-data", str(data_path),
                "--output-dir", str(explain_dir),
                "--device", "cpu",
                "--n-steps", "4",
            ]
        ),
    )

    analysis = yaml.safe_load((explain_dir / "analysis_metadata.yaml").read_text())
    ig = analysis["integrated_gradients"]
    assert analysis["n_samples"] == len(small_dataset.samples)
    assert analysis["model_provenance"]["checkpoint_path"] == str(
        (run_dir / "best_model.pt").resolve()
    )
    assert ig["resolved_ig_mode"] == "content"
    assert ig["attribution_width"] == config.content_dim
    assert (
        ig["absolute_position_encoding"],
        ig["relative_position_encoding"],
        ig["chromosome_encoding"],
    ) == (
        config.absolute.encoding.value,
        config.relative.encoding.value,
        config.chromosome.encoding.value,
    )
    per_sample = sorted((explain_dir / "attributions_per_sample").glob("sample_*.npz"))
    assert len(per_sample) == len(small_dataset.samples)
    for path in per_sample:
        with np.load(path, allow_pickle=False) as sample:
            assert sample["attributions"].shape[1] == config.content_dim
            assert np.isfinite(sample["attributions"]).all()
    assert (explain_dir / "sieve_gene_rankings_mean.csv").is_file()


@pytest.mark.parametrize("ig_mode", [ResolvedIGMode.CONTENT, ResolvedIGMode.DOSAGE])
@pytest.mark.parametrize("strategy", THESIS_STRATEGIES)
def test_rq3_ig_completeness_and_fixed_context_in_explain_and_attribute_batch(
    small_dataset, config_g_args, strategy, ig_mode
):
    """RQ3.1 completeness and RQ3.2 fixed context through both production IG paths.

    The base model is what explain.py attributes. Completeness is checked
    against the content wrapper with absolute position, positions,
    chromosomes, genes and mask held at their observed values.
    """
    *_, chunked_model = _config_g_training_model(
        small_dataset, config_g_args, THESIS_STRATEGIES[strategy][0]
    )
    model = chunked_model.base_model.eval()
    explainer = explain._create_integrated_gradients_explainer(
        model=model, device="cpu", n_steps=64, max_variants=3000, resolved_ig_mode=ig_mode
    )
    indices = [0, 1]
    batch_attributions, _, batch_metadata = explainer.attribute_batch(
        DataLoader(Subset(small_dataset, indices), batch_size=1, collate_fn=collate_chunks)
    )
    wrapper = ContentSIEVEWrapper(model)

    for index, batch_attribution, metadata in zip(
        indices, batch_attributions, batch_metadata, strict=True
    ):
        chunk = small_dataset[index]
        attributions, _, _, mask, _ = explain._attribute_chunk_for_ig(
            explainer=explainer, chunk=chunk, resolved_ig_mode=ig_mode, device="cpu"
        )
        attributions = attributions.detach()[0][mask[0]]

        # explain.py and attribute_batch() attribute the same thing.
        assert metadata["sampling_applied"] is False
        assert attributions.shape == (
            int(chunk["mask"].sum()),
            model.position_encoding.content_dim,
        )
        torch.testing.assert_close(
            torch.as_tensor(batch_attribution), attributions, rtol=1e-5, atol=1e-7
        )

        # Completeness: sum(attributions) = f(observed) - f(baseline).
        inputs = {
            key: chunk[key].unsqueeze(0)
            for key in (
                "content_features",
                "absolute_position_features",
                "positions",
                "gene_ids",
                "mask",
                "chrom_ids",
            )
        }
        baseline = inputs["content_features"].clone()
        if ig_mode is ResolvedIGMode.DOSAGE:
            baseline[..., 0] = 0
            assert torch.count_nonzero(attributions[:, 1:]) == 0
        else:
            baseline.zero_()
        with torch.no_grad():
            delta = (
                wrapper(**inputs) - wrapper(**{**inputs, "content_features": baseline})
            ).item()
        residual = abs(attributions.sum().item() - delta)
        # Provisional tolerance; residuals are printed for the thesis record (-rP).
        print(
            f"{strategy} {ig_mode.value} sample {index}: f(x)-f(baseline)={delta:.6e} "
            f"abs_residual={residual:.3e} rel_residual={residual / max(abs(delta), 1e-12):.3e}"
        )
        assert residual <= max(1e-3, 1e-2 * abs(delta))
