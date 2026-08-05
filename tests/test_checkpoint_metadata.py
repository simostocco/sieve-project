"""Tests for additive training checkpoint metadata."""

import pytest
import torch

from scripts import train
from src.training.loss import SIEVELoss
from src.training.trainer import Trainer


def make_trainer(tmp_path, *, checkpoint_metadata=None, scheduler=None):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=SIEVELoss(),
        device="cpu",
        checkpoint_dir=tmp_path,
        scheduler=scheduler,
        checkpoint_metadata=checkpoint_metadata,
    )


def test_trainer_without_checkpoint_metadata_preserves_historical_key_set(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.save_checkpoint("model.pt", {"auc": 0.7})
    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert set(checkpoint) == {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "metrics",
        "best_val_auc",
        "history",
    }
    assert "metadata" not in checkpoint


def test_checkpoint_metadata_is_saved_as_defensive_copy(tmp_path):
    metadata = {
        "metadata_schema_version": 1,
        "input_dim": 71,
        "dataset_identity": {"mappings_artifact": "dataset_mappings.json"},
    }
    trainer = make_trainer(tmp_path, checkpoint_metadata=metadata)
    metadata["input_dim"] = 99
    # Mutate a nested field too, so the test distinguishes deep copy from shallow copy.
    metadata["dataset_identity"]["mappings_artifact"] = "mutated.json"

    trainer.save_checkpoint("model.pt", {"auc": 0.7})
    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["metadata"] == {
        "metadata_schema_version": 1,
        "input_dim": 71,
        "dataset_identity": {"mappings_artifact": "dataset_mappings.json"},
    }
    assert "gene_index" not in checkpoint["metadata"]
    assert "chrom_index" not in checkpoint["metadata"]


def test_checkpoint_metadata_rejects_non_dict(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_metadata"):
        make_trainer(tmp_path, checkpoint_metadata=["not", "a", "dict"])


def test_checkpoint_state_dict_keys_are_unchanged_when_metadata_is_present(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_metadata={"metadata_schema_version": 1})
    expected_keys = set(trainer.model.state_dict())

    trainer.save_checkpoint("model.pt", {"auc": 0.7})
    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert set(checkpoint["model_state_dict"]) == expected_keys
    assert "optimizer_state_dict" in checkpoint


def test_scheduler_state_dict_behavior_is_unchanged(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=SIEVELoss(),
        device="cpu",
        checkpoint_dir=tmp_path,
        scheduler=scheduler,
        checkpoint_metadata={"metadata_schema_version": 1},
    )

    trainer.save_checkpoint("model.pt", {"auc": 0.7})
    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert "scheduler_state_dict" in checkpoint


def test_metrics_history_epoch_and_best_auc_are_unchanged_with_metadata(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_metadata={"metadata_schema_version": 1})
    trainer.current_epoch = 4
    trainer.best_val_auc = 0.8
    trainer.history["train_loss"].append(0.3)

    trainer.save_checkpoint("model.pt", {"auc": 0.7})
    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["epoch"] == 4
    assert checkpoint["metrics"] == {"auc": 0.7}
    assert checkpoint["best_val_auc"] == 0.8
    assert checkpoint["history"]["train_loss"] == [0.3]


def test_load_checkpoint_works_with_metadata_present(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_metadata={"metadata_schema_version": 1})
    trainer.save_checkpoint("model.pt", {"auc": 0.7})

    loader = make_trainer(tmp_path)
    metrics = loader.load_checkpoint("model.pt")

    assert metrics == {"auc": 0.7}


def test_old_checkpoint_without_metadata_still_loads(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.save_checkpoint("model.pt", {"auc": 0.7})

    loader = make_trainer(tmp_path, checkpoint_metadata={"metadata_schema_version": 1})
    metrics = loader.load_checkpoint("model.pt")

    assert metrics == {"auc": 0.7}


def test_checkpoint_metadata_loads_with_current_torch_load_behavior(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_metadata={"metadata_schema_version": 1})
    trainer.save_checkpoint("model.pt", {"auc": 0.7})

    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["metadata"] == {"metadata_schema_version": 1}


def test_train_single_fold_forwards_checkpoint_metadata_without_training(monkeypatch, tmp_path):
    calls = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls["checkpoint_metadata"] = kwargs["checkpoint_metadata"]
            self.current_epoch = 0

        def train(self, **kwargs):
            calls["trained"] = True

        def load_checkpoint(self, filename):
            calls["loaded"] = filename
            return {"auc": 0.7}

    monkeypatch.setattr(train, "Trainer", FakeTrainer)

    model = torch.nn.Linear(2, 1)
    args = type(
        "Args",
        (),
        {
            "lr": 1e-3,
            "device": "cpu",
            "lambda_attr": 0.0,
            "early_stopping": 2,
            "gradient_clip": None,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
        },
    )()
    metadata = {"metadata_schema_version": 1}

    metrics = train.train_single_fold(
        train_loader=[],
        val_loader=[],
        model=model,
        args=args,
        checkpoint_dir=tmp_path,
        checkpoint_metadata=metadata,
    )

    assert calls["checkpoint_metadata"] == metadata
    assert calls["trained"] is True
    assert calls["loaded"] == "best_model.pt"
    assert metrics["auc"] == 0.7
