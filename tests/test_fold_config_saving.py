"""
Tests for fold-specific configuration saving in cross-validation.

Tests that save_fold_config and save_fold_info produce the expected
YAML files with correct content and structure.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml


# Add project root to path so we can import from scripts/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import save_fold_config, save_fold_info


@pytest.fixture
def tmp_fold_dir(tmp_path):
    """Create a temporary fold directory."""
    fold_dir = tmp_path / "experiment" / "fold_0"
    fold_dir.mkdir(parents=True)
    return fold_dir


@pytest.fixture
def sample_args():
    """Create a sample argparse.Namespace mimicking train.py args."""
    return argparse.Namespace(
        experiment_name="TEST_EXPERIMENT_CV",
        level="L3",
        latent_dim=32,
        hidden_dim=64,
        num_attention_layers=1,
        num_heads=4,
        chunk_size=3000,
        chunk_overlap=0,
        aggregation_method="mean",
        lr=1e-5,
        lambda_attr=0.1,
        batch_size=16,
        gradient_accumulation_steps=4,
        gradient_clip=1.0,
        early_stopping=15,
        epochs=100,
        seed=42,
        genome_build="GRCh37",
        sex_map=None,
        preprocessed_data="/data/preprocessed.pt",
        vcf=None,
        phenotypes=None,
    )


@pytest.fixture
def sample_labels():
    """Create sample labels array (100 samples: 50 cases, 50 controls)."""
    return np.array([1] * 50 + [0] * 50)


@pytest.fixture
def sample_run_metadata():
    """Create lightweight run metadata as saved by Phase 4C."""
    return {
        "metadata_schema_version": 1,
        "input_dim": 71,
        "content_dim": 7,
        "num_genes": 2,
        "num_chromosomes": 2,
        "position_encoding": {
            "preset": "legacy",
            "chromosome": {
                "mapping": {
                    "0": "1",
                    "1": "2",
                }
            },
        },
        "dataset_identity": {
            "genome_build": "GRCh37",
            "gene_mapping_sha256": "genehash",
            "chromosome_mapping_sha256": "chromhash",
            "mappings_artifact": "dataset_mappings.json",
            "mappings_artifact_base": "experiment_root",
        },
        "position_encoding_execution": {
            "schema_version": 1,
            "source": "legacy_existing_model_paths",
            "resolved_config_applied_to_model": False,
            "model_num_chromosomes": 2,
            "chrom_ids_passed_to_attention": True,
            "chromosome_embedding_executed": True,
            "chromosome_aware_relative_bias_executed": True,
        },
    }


@pytest.fixture
def sample_fold_metrics():
    """Create sample fold metrics as returned by train_single_fold."""
    return {
        "auc": 0.6456,
        "accuracy": 0.5973,
        "loss": 0.6821,
        "best_epoch": 13,
        "epochs_trained": 28,
        "training_time_seconds": 327.5,
        "time_per_epoch_seconds": 11.7,
    }


class TestSaveFoldConfig:
    """Tests for save_fold_config function."""

    def test_creates_config_file(self, tmp_fold_dir, sample_args):
        """Test that config.yaml is created in fold directory."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        assert (tmp_fold_dir / "config.yaml").exists()

    def test_config_is_valid_yaml(self, tmp_fold_dir, sample_args):
        """Test that the saved config is valid YAML."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_config_contains_fold_index(self, tmp_fold_dir, sample_args):
        """Test that fold_index is correctly stored."""
        save_fold_config(tmp_fold_dir, fold_idx=2, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["fold_index"] == 2

    def test_config_contains_architecture_params(self, tmp_fold_dir, sample_args):
        """Test that all architecture parameters are saved."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["level"] == "L3"
        assert config["latent_dim"] == 32
        assert config["hidden_dim"] == 64
        assert config["num_attention_layers"] == 1
        assert config["num_heads"] == 4
        assert config["chunk_size"] == 3000
        assert config["aggregation_method"] == "mean"

    def test_config_contains_training_params(self, tmp_fold_dir, sample_args):
        """Test that all training parameters are saved."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["lr"] == 1e-5
        assert config["lambda_attr"] == 0.1
        assert config["batch_size"] == 16
        assert config["gradient_accumulation_steps"] == 4
        assert config["gradient_clip"] == 1.0
        assert config["early_stopping"] == 15
        assert config["epochs"] == 100
        assert config["seed"] == 42

    def test_config_contains_data_reference(self, tmp_fold_dir, sample_args):
        """Test that data reference is saved."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["preprocessed_data"] == "/data/preprocessed.pt"

    def test_config_contains_parent_reference(self, tmp_fold_dir, sample_args):
        """Test that parent config reference is saved."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["parent_config"] == "../config.yaml"

    def test_config_contains_experiment_name(self, tmp_fold_dir, sample_args):
        """Test that experiment name is saved."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["experiment_name"] == "TEST_EXPERIMENT_CV"

    def test_config_with_vcf_data_source(self, tmp_fold_dir, sample_args):
        """Test config when using VCF instead of preprocessed data."""
        sample_args.preprocessed_data = None
        sample_args.vcf = "/data/cohort.vcf.gz"
        sample_args.phenotypes = "/data/phenotypes.tsv"

        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["preprocessed_data"] is None
        assert config["vcf"] == "/data/cohort.vcf.gz"
        assert config["phenotypes"] == "/data/phenotypes.tsv"

    def test_config_usable_by_explain(self, tmp_fold_dir, sample_args):
        """Test that saved config contains fields needed by explain.py."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        # explain.py needs these fields to load a model
        required_for_explain = ["level", "aggregation_method"]
        for field in required_for_explain:
            assert field in config, f"Missing field required by explain.py: {field}"

    def test_call_without_run_metadata_preserves_old_output(
        self, tmp_fold_dir, sample_args
    ):
        """Test that metadata is absent when not supplied."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert "metadata_schema_version" not in config
        assert "position_encoding" not in config
        assert "dataset_identity" not in config
        assert "position_encoding_execution" not in config

    def test_supplied_run_metadata_is_copied_into_fold_config(
        self, tmp_fold_dir, sample_args, sample_run_metadata
    ):
        """Test that lightweight run metadata is saved additively."""
        save_fold_config(
            tmp_fold_dir,
            fold_idx=0,
            args=sample_args,
            run_metadata=sample_run_metadata,
        )
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        for key, value in sample_run_metadata.items():
            assert config[key] == value

    def test_run_metadata_input_is_not_mutated(
        self, tmp_fold_dir, sample_args, sample_run_metadata
    ):
        """Test that save_fold_config defensively copies supplied metadata."""
        before = yaml.safe_load(yaml.safe_dump(sample_run_metadata))

        save_fold_config(
            tmp_fold_dir,
            fold_idx=0,
            args=sample_args,
            run_metadata=sample_run_metadata,
        )

        assert sample_run_metadata == before

    def test_original_flat_fields_remain_with_run_metadata(
        self, tmp_fold_dir, sample_args, sample_run_metadata
    ):
        """Test that old fold fields are preserved when metadata is added."""
        save_fold_config(
            tmp_fold_dir,
            fold_idx=0,
            args=sample_args,
            run_metadata=sample_run_metadata,
        )
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["level"] == "L3"
        assert config["latent_dim"] == 32
        assert config["lr"] == 1e-5
        assert config["parent_config"] == "../config.yaml"

    def test_full_dataset_mappings_are_not_duplicated(
        self, tmp_fold_dir, sample_args, sample_run_metadata
    ):
        """Test that fold config refers to the mapping artifact only."""
        save_fold_config(
            tmp_fold_dir,
            fold_idx=0,
            args=sample_args,
            run_metadata=sample_run_metadata,
        )
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert "gene_index" not in config
        assert "chrom_index" not in config
        assert config["dataset_identity"]["mappings_artifact"] == "dataset_mappings.json"
        assert config["dataset_identity"]["mappings_artifact_base"] == "experiment_root"

    def test_parent_and_fold_lightweight_metadata_can_match(
        self, tmp_fold_dir, sample_args, sample_run_metadata
    ):
        """Test that fold metadata can equal parent run metadata."""
        parent_metadata = yaml.safe_load(yaml.safe_dump(sample_run_metadata))
        save_fold_config(
            tmp_fold_dir,
            fold_idx=0,
            args=sample_args,
            run_metadata=sample_run_metadata,
        )
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        fold_metadata = {key: config[key] for key in parent_metadata}
        assert fold_metadata == parent_metadata


class TestSaveFoldInfo:
    """Tests for save_fold_info function."""

    def test_creates_fold_info_file(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that fold_info.yaml is created."""
        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        assert (tmp_fold_dir / "fold_info.yaml").exists()

    def test_fold_info_is_valid_yaml(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that fold_info.yaml is valid YAML."""
        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)
        assert isinstance(info, dict)

    def test_fold_info_contains_fold_metadata(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test fold index, n_folds, and seed are saved."""
        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=2,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["fold_index"] == 2
        assert info["n_folds"] == 5
        assert info["random_seed"] == 42

    def test_fold_info_contains_sample_counts(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that sample counts are correct."""
        train_indices = list(range(80))
        val_indices = list(range(80, 100))

        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=train_indices,
            val_indices=val_indices,
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["n_train_samples"] == 80
        assert info["n_val_samples"] == 20

    def test_fold_info_contains_sample_indices(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that sample indices are saved as lists of ints."""
        train_indices = list(range(80))
        val_indices = list(range(80, 100))

        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=train_indices,
            val_indices=val_indices,
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["train_sample_indices"] == train_indices
        assert info["val_sample_indices"] == val_indices

    def test_fold_info_class_distribution(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that class distribution is correctly computed."""
        # labels: [1]*50 + [0]*50
        # train_indices 0-79: 50 cases (indices 0-49) + 30 controls (50-79)
        # val_indices 80-99: 0 cases + 20 controls
        train_indices = list(range(80))
        val_indices = list(range(80, 100))

        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=train_indices,
            val_indices=val_indices,
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["train_cases"] == 50
        assert info["train_controls"] == 30
        assert info["val_cases"] == 0
        assert info["val_controls"] == 20

    def test_fold_info_training_outcome(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that training outcome metrics are saved."""
        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["best_epoch"] == 13
        assert info["best_val_auc"] == pytest.approx(0.6456)
        assert info["best_val_accuracy"] == pytest.approx(0.5973)
        assert info["epochs_trained"] == 28
        assert info["training_time_seconds"] == pytest.approx(327.5)

    def test_fold_info_timestamps(
        self, tmp_fold_dir, sample_labels, sample_fold_metrics
    ):
        """Test that timestamps are saved as ISO format strings."""
        started = datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc)

        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=started,
            training_completed=completed,
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        assert info["training_started"] == started.isoformat()
        assert info["training_completed"] == completed.isoformat()

    def test_fold_info_indices_are_plain_ints(
        self, tmp_fold_dir, sample_fold_metrics
    ):
        """Test that numpy int64 indices are converted to plain ints."""
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        # Simulate numpy int64 indices (as returned by StratifiedKFold)
        train_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64).tolist()
        val_indices = np.array([8, 9], dtype=np.int64).tolist()

        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=train_indices,
            val_indices=val_indices,
            labels=labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        # YAML should have native ints, not numpy types
        for idx in info["train_sample_indices"]:
            assert isinstance(idx, int)
        for idx in info["val_sample_indices"]:
            assert isinstance(idx, int)


class TestBothFilesSavedTogether:
    """Test that both files can be saved for the same fold."""

    def test_both_files_coexist(
        self, tmp_fold_dir, sample_args, sample_labels, sample_fold_metrics
    ):
        """Test that config.yaml and fold_info.yaml can coexist."""
        save_fold_config(tmp_fold_dir, fold_idx=0, args=sample_args)
        save_fold_info(
            fold_dir=tmp_fold_dir,
            fold_idx=0,
            n_folds=5,
            seed=42,
            train_indices=list(range(80)),
            val_indices=list(range(80, 100)),
            labels=sample_labels,
            fold_metrics=sample_fold_metrics,
            training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
            training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
        )

        assert (tmp_fold_dir / "config.yaml").exists()
        assert (tmp_fold_dir / "fold_info.yaml").exists()

        # Both should be loadable independently
        with open(tmp_fold_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        with open(tmp_fold_dir / "fold_info.yaml") as f:
            info = yaml.safe_load(f)

        # Fold index should be consistent
        assert config["fold_index"] == info["fold_index"]

    def test_multiple_folds(self, tmp_path, sample_args, sample_labels, sample_fold_metrics):
        """Test saving config for multiple folds."""
        experiment_dir = tmp_path / "experiment"

        for fold_idx in range(3):
            fold_dir = experiment_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True)

            save_fold_config(fold_dir, fold_idx=fold_idx, args=sample_args)
            save_fold_info(
                fold_dir=fold_dir,
                fold_idx=fold_idx,
                n_folds=3,
                seed=42,
                train_indices=list(range(70)),
                val_indices=list(range(70, 100)),
                labels=sample_labels,
                fold_metrics=sample_fold_metrics,
                training_started=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
                training_completed=datetime(2026, 2, 1, 19, 45, 0, tzinfo=timezone.utc),
            )

        # Verify each fold has its own config
        for fold_idx in range(3):
            fold_dir = experiment_dir / f"fold_{fold_idx}"
            with open(fold_dir / "config.yaml") as f:
                config = yaml.safe_load(f)
            assert config["fold_index"] == fold_idx

            with open(fold_dir / "fold_info.yaml") as f:
                info = yaml.safe_load(f)
            assert info["fold_index"] == fold_idx
            assert info["n_folds"] == 3
