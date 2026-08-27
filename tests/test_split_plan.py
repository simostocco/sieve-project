"""Tests for versioned training split-plan validation and hashing."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from src.training.split_plan import (
    build_cv_split_plan,
    build_single_split_plan,
    build_split_plan_metadata,
    load_split_plan,
    ordered_sample_ids,
    sample_ids_sha256,
    split_plan_sha256,
    validate_split_plan,
    write_or_validate_existing_split_plan,
)


@dataclass
class Sample:
    sample_id: str


def _sample_ids() -> list[str]:
    return ["S0", "S1", "S2", "S3", "S4", "S5"]


def _cv_plan(sample_ids=None):
    sample_ids = _sample_ids() if sample_ids is None else sample_ids
    return build_cv_split_plan(
        folds=[
            ([2, 3, 4, 5], [0, 1]),
            ([0, 1, 4, 5], [2, 3]),
            ([0, 1, 2, 3], [4, 5]),
        ],
        sample_ids=sample_ids,
        seed=42,
        split_source="generated",
        n_folds=3,
    )


def _single_plan(sample_ids=None):
    sample_ids = _sample_ids() if sample_ids is None else sample_ids
    return build_single_split_plan(
        train_indices=[0, 2, 4],
        val_indices=[1, 3, 5],
        sample_ids=sample_ids,
        seed=42,
        split_source="generated",
    )


def test_ordered_sample_ids_validate_exact_strings():
    samples = [Sample(" S0 "), Sample("S1")]

    assert ordered_sample_ids(samples) == [" S0 ", "S1"]
    assert sample_ids_sha256([" S0 ", "S1"]) != sample_ids_sha256(["S0", "S1"])


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([Sample("")], "non-empty"),
        ([Sample("   ")], "non-empty"),
        ([Sample("S0"), Sample("S0")], "duplicate"),
        ([object()], "string"),
    ],
)
def test_ordered_sample_ids_reject_malformed_sample_ids(samples, message):
    with pytest.raises(ValueError, match=message):
        ordered_sample_ids(samples)


def test_generated_cv_plan_records_exact_folds_and_hashes():
    plan = _cv_plan()

    assert plan["mode"] == "cv"
    assert plan["n_folds"] == 3
    assert plan["sample_ids_sha256"] == sample_ids_sha256(_sample_ids())
    assert [fold["val_indices"] for fold in plan["folds"]] == [[0, 1], [2, 3], [4, 5]]
    assert sorted(index for fold in plan["folds"] for index in fold["val_indices"]) == list(
        range(6)
    )
    for fold in plan["folds"]:
        assert "train_sample_ids_sha256" in fold
        assert "val_sample_ids_sha256" in fold


def test_generated_single_plan_records_exact_membership_and_hashes():
    plan = _single_plan()

    assert plan["mode"] == "single_split"
    assert plan["train_indices"] == [0, 2, 4]
    assert plan["val_indices"] == [1, 3, 5]
    assert plan["sample_ids_sha256"] == sample_ids_sha256(_sample_ids())
    assert plan["train_sample_ids_sha256"] != plan["val_sample_ids_sha256"]


def test_canonical_membership_hash_ignores_seed_and_split_source():
    first = _single_plan()
    second = copy.deepcopy(first)
    second["seed"] = 999
    second["split_source"] = "replayed"

    assert split_plan_sha256(first) == split_plan_sha256(second)


def test_canonical_membership_hash_changes_when_index_changes():
    first = _single_plan()
    second = build_single_split_plan(
        train_indices=[0, 2, 5],
        val_indices=[1, 3, 4],
        sample_ids=_sample_ids(),
        seed=42,
        split_source="generated",
    )

    assert split_plan_sha256(first) != split_plan_sha256(second)


def test_yaml_formatting_and_key_order_do_not_change_membership_hash(tmp_path):
    plan = _single_plan()
    path = tmp_path / "split_plan.yaml"
    path.write_text(
        "\n".join(
            [
                "split_source: generated",
                "seed: 42",
                "val_sample_ids_sha256: " + plan["val_sample_ids_sha256"],
                "train_sample_ids_sha256: " + plan["train_sample_ids_sha256"],
                "val_indices: [1, 3, 5]",
                "train_indices:",
                "  - 0",
                "  - 2",
                "  - 4",
                "sample_ids_sha256: " + plan["sample_ids_sha256"],
                "n_samples: 6",
                "mode: single_split",
                "schema_version: 1",
            ]
        ),
        encoding="utf-8",
    )
    loaded = validate_split_plan(
        load_split_plan(path),
        sample_ids=_sample_ids(),
        expected_mode="single_split",
    )

    assert split_plan_sha256(loaded) == split_plan_sha256(plan)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda plan: plan.update({"schema_version": 2}), "schema_version"),
        (lambda plan: plan.update({"mode": "cv"}), "mode"),
        (lambda plan: plan.update({"n_samples": 7}), "n_samples"),
        (lambda plan: plan.update({"sample_ids_sha256": "bad"}), "sample_ids_sha256"),
        (lambda plan: plan.update({"train_indices": [0, 2, 6]}), "out-of-range"),
        (lambda plan: plan.update({"train_indices": [0, True, 4]}), "integer"),
        (lambda plan: plan.update({"train_indices": [0, 0, 4]}), "duplicate"),
        (lambda plan: plan.update({"train_indices": [0, 1, 4]}), "disjoint"),
        (lambda plan: plan.update({"train_indices": [0, 2]}), "cover all samples"),
        (lambda plan: plan.update({"train_sample_ids_sha256": "bad"}), "train_sample"),
    ],
)
def test_single_split_validation_rejects_malformed_plans(mutator, message):
    plan = _single_plan()
    mutator(plan)

    with pytest.raises(ValueError, match=message):
        validate_split_plan(
            plan,
            sample_ids=_sample_ids(),
            expected_mode="single_split",
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda plan: plan.update({"n_folds": 2}), "n_folds"),
        (lambda plan: plan["folds"].pop(), "folds length"),
        (lambda plan: plan["folds"][1].update({"fold_index": 0}), "duplicate"),
        (lambda plan: plan["folds"][1].update({"fold_index": 4}), "0..n_folds"),
        (
            lambda plan: (
                plan["folds"][1].update({"train_indices": [1, 2, 4, 5], "val_indices": [0, 3]}),
                plan["folds"][1].pop("train_sample_ids_sha256"),
                plan["folds"][1].pop("val_sample_ids_sha256"),
            ),
            "validation folds",
        ),
        (lambda plan: plan["folds"][0].update({"train_indices": [0, 3, 4, 5]}), "disjoint"),
        (lambda plan: plan["folds"][0].update({"train_indices": [3, 4, 5]}), "cover"),
        (lambda plan: plan["folds"][0].update({"val_sample_ids_sha256": "bad"}), "val_sample"),
    ],
)
def test_cv_validation_rejects_malformed_plans(mutator, message):
    plan = _cv_plan()
    mutator(plan)

    with pytest.raises(ValueError, match=message):
        validate_split_plan(
            plan,
            sample_ids=_sample_ids(),
            expected_mode="cv",
            expected_n_folds=3,
        )


def test_replay_validation_allows_different_seed_and_different_labels_boundary():
    plan = _single_plan()
    plan["seed"] = 1234
    plan["split_source"] = "replayed"

    validated = validate_split_plan(
        plan,
        sample_ids=_sample_ids(),
        expected_mode="single_split",
    )

    assert validated["seed"] == 1234
    assert validated["split_source"] == "replayed"


def test_existing_split_plan_with_same_membership_is_reused_without_rewrite(tmp_path):
    existing = _single_plan()
    existing["split_source"] = "generated"
    path = tmp_path / "split_plan.yaml"
    path.write_text("sentinel: keep me\n", encoding="utf-8")
    path.unlink()
    write_or_validate_existing_split_plan(
        path,
        existing,
        sample_ids=_sample_ids(),
        expected_mode="single_split",
    )
    before = path.read_text(encoding="utf-8")
    requested = copy.deepcopy(existing)
    requested["split_source"] = "replayed"

    reused = write_or_validate_existing_split_plan(
        path,
        requested,
        sample_ids=_sample_ids(),
        expected_mode="single_split",
    )

    assert split_plan_sha256(reused) == split_plan_sha256(existing)
    assert path.read_text(encoding="utf-8") == before


def test_existing_split_plan_with_different_membership_rejects(tmp_path):
    path = tmp_path / "split_plan.yaml"
    write_or_validate_existing_split_plan(
        path,
        _single_plan(),
        sample_ids=_sample_ids(),
        expected_mode="single_split",
    )
    different = build_single_split_plan(
        train_indices=[0, 1, 4],
        val_indices=[2, 3, 5],
        sample_ids=_sample_ids(),
        seed=42,
        split_source="generated",
    )

    with pytest.raises(ValueError, match="different sample membership"):
        write_or_validate_existing_split_plan(
            path,
            different,
            sample_ids=_sample_ids(),
            expected_mode="single_split",
        )


def test_split_plan_metadata_records_current_invocation_source(tmp_path):
    plan = _single_plan()
    input_plan = copy.deepcopy(plan)
    input_path = tmp_path / "real" / "split_plan.yaml"
    experiment_path = tmp_path / "null" / "split_plan.yaml"

    metadata = build_split_plan_metadata(
        source="replayed",
        experiment_plan_path=experiment_path,
        plan=plan,
        input_plan_path=input_path,
        input_plan=input_plan,
    )

    assert metadata["source"] == "replayed"
    assert metadata["path"] == str(experiment_path.resolve())
    assert metadata["sha256"] == split_plan_sha256(plan)
    assert metadata["input_path"] == str(input_path.resolve())
    assert metadata["input_sha256"] == split_plan_sha256(input_plan)
