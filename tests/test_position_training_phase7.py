"""Tests for Phase 7B3 training positional configuration integration."""

import argparse

import pytest
import torch
import yaml

from scripts import train
from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionPreset,
    RelativePositionEncoding,
)
from src.models.position_runtime import (
    NoAbsolutePositionRuntime,
    NoRelativePositionRuntime,
    T5RelativePositionRuntime,
)
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
    "extra",
    [
        [
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "alibi_fixed",
            "--chromosome-encoding",
            "none",
        ],
        [
            "--absolute-position-encoding",
            "none",
            "--relative-position-encoding",
            "alibi_learned",
            "--chromosome-encoding",
            "none",
        ],
    ],
)
def test_unsupported_future_strategies_fail_before_model_construction(extra):
    args = _args("--position-preset", "custom", *extra)

    with pytest.raises(NotImplementedError, match="not implemented"):
        _resolve(args, num_chromosomes=3)


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
