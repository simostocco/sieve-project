"""Tests for train.py split-plan generation, replay, and metadata plumbing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import train
from src.training.split_plan import (
    build_cv_split_plan,
    build_single_split_plan,
    load_split_plan,
    ordered_sample_ids,
    split_plan_sha256,
    write_split_plan,
)


def _args(*, cv: int | None = None, split_plan: str | None = None, seed: int = 42):
    return argparse.Namespace(
        cv=cv,
        val_split=0.5,
        seed=seed,
        split_plan=split_plan,
    )


def _sample_ids() -> list[str]:
    return ["S0", "S1", "S2", "S3"]


def _labels() -> np.ndarray:
    return np.array([0, 0, 1, 1])


def _cv_plan(sample_ids=None, *, split_source="generated"):
    sample_ids = _sample_ids() if sample_ids is None else sample_ids
    return build_cv_split_plan(
        folds=[
            ([1, 3], [0, 2]),
            ([0, 2], [1, 3]),
        ],
        sample_ids=sample_ids,
        seed=42,
        split_source=split_source,
        n_folds=2,
    )


def _single_plan(sample_ids=None, *, split_source="generated"):
    sample_ids = _sample_ids() if sample_ids is None else sample_ids
    return build_single_split_plan(
        train_indices=[0, 2],
        val_indices=[1, 3],
        sample_ids=sample_ids,
        seed=42,
        split_source=split_source,
    )


def test_parser_help_includes_split_plan():
    parser = train.build_arg_parser()

    assert "--split-plan" in parser.format_help()
    assert train.parse_args(["--level", "L3"]).split_plan is None


def test_generated_cv_uses_existing_stratified_fold_helper(monkeypatch, tmp_path):
    expected_folds = [
        (np.array([1, 3]), np.array([0, 2])),
        (np.array([0, 2]), np.array([1, 3])),
    ]
    calls = []

    def fake_create_stratified_folds(labels, n_folds, random_state):
        calls.append((labels.copy(), n_folds, random_state))
        return expected_folds

    monkeypatch.setattr(train, "create_stratified_folds", fake_create_stratified_folds)

    plan, metadata, input_plan = train.prepare_training_split_plan(
        args=_args(cv=2, seed=99),
        output_dir=tmp_path,
        sample_ids=_sample_ids(),
        labels=_labels(),
    )

    assert len(calls) == 1
    assert calls[0][1:] == (2, 99)
    assert input_plan is None
    assert [fold["val_indices"] for fold in plan["folds"]] == [[0, 2], [1, 3]]
    assert metadata["source"] == "generated"
    assert metadata["sha256"] == split_plan_sha256(plan)
    assert (tmp_path / "split_plan.yaml").exists()


def test_generated_single_uses_existing_train_test_split_semantics(monkeypatch, tmp_path):
    calls = []

    def fake_train_test_split(indices, *, test_size, stratify, random_state):
        calls.append((indices.copy(), test_size, stratify.copy(), random_state))
        return np.array([0, 2]), np.array([1, 3])

    monkeypatch.setattr(train, "train_test_split", fake_train_test_split)

    plan, metadata, _ = train.prepare_training_split_plan(
        args=_args(seed=77),
        output_dir=tmp_path,
        sample_ids=_sample_ids(),
        labels=_labels(),
    )

    assert len(calls) == 1
    assert calls[0][1:] == (0.5, pytest.approx(_labels()), 77)
    assert plan["train_indices"] == [0, 2]
    assert plan["val_indices"] == [1, 3]
    assert metadata["source"] == "generated"


def test_replay_cv_uses_exact_indices_without_stratified_generation(monkeypatch, tmp_path):
    real_plan_path = tmp_path / "real" / "split_plan.yaml"
    write_split_plan(real_plan_path, _cv_plan(split_source="generated"))

    def fail_create_stratified_folds(*args, **kwargs):
        raise AssertionError("replayed CV split must not regenerate folds")

    monkeypatch.setattr(train, "create_stratified_folds", fail_create_stratified_folds)

    plan, metadata, input_plan = train.prepare_training_split_plan(
        args=_args(cv=2, split_plan=str(real_plan_path), seed=1234),
        output_dir=tmp_path / "null",
        sample_ids=_sample_ids(),
        labels=np.array([1, 0, 0, 1]),
    )

    assert [fold["train_indices"] for fold in plan["folds"]] == [[1, 3], [0, 2]]
    assert [fold["val_indices"] for fold in plan["folds"]] == [[0, 2], [1, 3]]
    assert metadata["source"] == "replayed"
    assert metadata["input_path"] == str(real_plan_path.resolve())
    assert metadata["input_sha256"] == split_plan_sha256(input_plan)


def test_replay_single_uses_exact_indices_without_train_test_split(monkeypatch, tmp_path):
    real_plan_path = tmp_path / "real" / "split_plan.yaml"
    write_split_plan(real_plan_path, _single_plan(split_source="generated"))

    def fail_train_test_split(*args, **kwargs):
        raise AssertionError("replayed single split must not call train_test_split")

    monkeypatch.setattr(train, "train_test_split", fail_train_test_split)

    plan, metadata, _ = train.prepare_training_split_plan(
        args=_args(split_plan=str(real_plan_path), seed=1234),
        output_dir=tmp_path / "null",
        sample_ids=_sample_ids(),
        labels=np.array([1, 0, 0, 1]),
    )

    assert plan["train_indices"] == [0, 2]
    assert plan["val_indices"] == [1, 3]
    assert metadata["source"] == "replayed"


def test_replay_with_different_args_seed_succeeds_at_split_plan_layer(tmp_path):
    real_plan_path = tmp_path / "real" / "split_plan.yaml"
    write_split_plan(real_plan_path, _single_plan(split_source="generated"))

    plan, metadata, _ = train.prepare_training_split_plan(
        args=_args(split_plan=str(real_plan_path), seed=999),
        output_dir=tmp_path / "null",
        sample_ids=_sample_ids(),
        labels=np.array([1, 0, 0, 1]),
    )

    assert plan["seed"] == 42
    assert metadata["source"] == "replayed"


def test_same_existing_membership_is_reused_but_metadata_records_current_source(tmp_path):
    output_dir = tmp_path / "experiment"
    existing = _single_plan(split_source="generated")
    write_split_plan(output_dir / "split_plan.yaml", existing)
    before = (output_dir / "split_plan.yaml").read_text(encoding="utf-8")
    replay_path = tmp_path / "real" / "split_plan.yaml"
    replay = _single_plan(split_source="replayed")
    write_split_plan(replay_path, replay)

    plan, metadata, _ = train.prepare_training_split_plan(
        args=_args(split_plan=str(replay_path)),
        output_dir=output_dir,
        sample_ids=_sample_ids(),
        labels=np.array([1, 0, 0, 1]),
    )

    assert plan["split_source"] == "generated"
    assert (output_dir / "split_plan.yaml").read_text(encoding="utf-8") == before
    assert metadata["source"] == "replayed"
    assert metadata["input_sha256"] == split_plan_sha256(replay)


def test_different_existing_membership_rejects(tmp_path):
    output_dir = tmp_path / "experiment"
    write_split_plan(output_dir / "split_plan.yaml", _single_plan())
    replay_path = tmp_path / "real" / "split_plan.yaml"
    different = build_single_split_plan(
        train_indices=[0, 1],
        val_indices=[2, 3],
        sample_ids=_sample_ids(),
        seed=42,
        split_source="generated",
    )
    write_split_plan(replay_path, different)

    with pytest.raises(ValueError, match="different sample membership"):
        train.prepare_training_split_plan(
            args=_args(split_plan=str(replay_path)),
            output_dir=output_dir,
            sample_ids=_sample_ids(),
            labels=_labels(),
        )


def test_fold_info_indices_can_match_replayed_plan(tmp_path):
    plan = _cv_plan()
    fold = plan["folds"][0]
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    train.save_fold_info(
        fold_dir=fold_dir,
        fold_idx=0,
        n_folds=2,
        seed=42,
        train_indices=fold["train_indices"],
        val_indices=fold["val_indices"],
        labels=_labels(),
        fold_metrics={"auc": 0.5, "accuracy": 0.5},
        training_started=train.datetime(2026, 1, 1, tzinfo=train.timezone.utc),
        training_completed=train.datetime(2026, 1, 1, tzinfo=train.timezone.utc),
    )

    saved = yaml.safe_load((fold_dir / "fold_info.yaml").read_text(encoding="utf-8"))
    assert saved["train_sample_indices"] == fold["train_indices"]
    assert saved["val_sample_indices"] == fold["val_indices"]


def test_ordered_sample_ids_are_all_samples_authority():
    samples = [argparse.Namespace(sample_id="S0"), argparse.Namespace(sample_id="S1")]

    assert ordered_sample_ids(samples) == ["S0", "S1"]


def test_load_saved_experiment_plan_round_trips(tmp_path):
    plan, metadata, _ = train.prepare_training_split_plan(
        args=_args(),
        output_dir=tmp_path,
        sample_ids=_sample_ids(),
        labels=_labels(),
    )

    saved = load_split_plan(Path(metadata["path"]))
    assert split_plan_sha256(saved) == split_plan_sha256(plan)
