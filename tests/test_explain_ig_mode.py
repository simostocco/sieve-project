"""Focused tests for explain.py IG mode selection and attribution metadata."""

import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import explain
from src.encoding import AnnotationLevel
from src.encoding.position_config import ResolvedIGMode
from src.explain import load_sample_attributions
from src.explain.ig_mode import IGModeCompatibilityWarning


@pytest.fixture
def base_argv(tmp_path) -> list[str]:
    """Minimum argv accepted by the explain parser."""
    return [
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--config",
        str(tmp_path / "config.yaml"),
        "--preprocessed-data",
        str(tmp_path / "preprocessed.pt"),
        "--output-dir",
        str(tmp_path / "out"),
    ]


def parse_with(base_argv: list[str], *extra: str):
    return explain.parse_args([*base_argv, *extra])


def _new_config(default_ig_mode: str = "content") -> dict[str, object]:
    return {
        "level": "L3",
        "input_dim": 71,
        "content_dim": 7,
        "position_encoding": {
            "absolute": {"type": "sinusoidal"},
            "relative": {"type": "t5_bucket"},
            "chromosome": {"encoding": "learned"},
            "attribution": {"default_ig_mode": default_ig_mode},
        },
    }


class RecordingExplainer:
    """Small explainer double recording calls into attribute()."""

    def __init__(self, output_width: int | None = None):
        self.output_width = output_width
        self.calls = []

    def attribute(self, variant_features, positions, gene_ids, mask, **kwargs):
        self.calls.append(
            {
                "variant_features": variant_features,
                "positions": positions,
                "gene_ids": gene_ids,
                "mask": mask,
                **kwargs,
            }
        )
        source = kwargs["content_features"] if variant_features is None else variant_features
        width = source.shape[-1] if self.output_width is None else self.output_width
        return torch.ones(source.shape[0], source.shape[1], width)


def _chunk(include_features: bool = True) -> dict[str, torch.Tensor]:
    chunk = {
        "content_features": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "absolute_position_features": torch.tensor([[5.0], [6.0]]),
        "positions": torch.tensor([10, 20]),
        "gene_ids": torch.tensor([0, 1]),
        "mask": torch.tensor([True, True]),
        "chrom_ids": torch.tensor([1, 2]),
    }
    if include_features:
        chunk["features"] = torch.tensor([[1.0, 5.0, 2.0], [3.0, 6.0, 4.0]])
    return chunk


def test_parse_args_default_ig_mode_is_auto(base_argv):
    assert parse_with(base_argv).ig_mode == "auto"


@pytest.mark.parametrize("mode", ["auto", "content", "legacy"])
def test_parse_args_accepts_ig_modes(base_argv, mode):
    assert parse_with(base_argv, "--ig-mode", mode).ig_mode == mode


def test_parse_args_rejects_unsupported_ig_mode(base_argv):
    with pytest.raises(SystemExit):
        parse_with(base_argv, "--ig-mode", "full")


def test_parser_help_explains_ig_modes_and_keeps_unrelated_defaults():
    parser = explain.build_arg_parser()
    help_text = parser.format_help()
    ig_help = next(
        action.help for action in parser._actions if "--ig-mode" in action.option_strings
    )

    assert "--ig-mode" in help_text
    assert "saved attribution policy" in ig_help
    assert "absolute position remains fixed" in ig_help
    assert "complete historical feature representation" in ig_help
    assert (
        parser.parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--config",
                "config.yaml",
                "--preprocessed-data",
                "data.pt",
                "--output-dir",
                "out",
            ]
        ).aggregation_method
        == "mean"
    )


def test_new_schema_auto_resolves_to_saved_content():
    assert explain.resolve_ig_mode("auto", config=_new_config("content")) is ResolvedIGMode.CONTENT


def test_new_schema_saved_legacy_resolves_to_legacy():
    assert explain.resolve_ig_mode("auto", config=_new_config("legacy")) is ResolvedIGMode.LEGACY


def test_old_config_modes_use_existing_compatibility_warnings():
    with pytest.warns(IGModeCompatibilityWarning, match="historical legacy"):
        assert explain.resolve_ig_mode("auto", config={}) is ResolvedIGMode.LEGACY

    with pytest.warns(IGModeCompatibilityWarning, match="content-only attribution override"):
        assert explain.resolve_ig_mode("content", config={}) is ResolvedIGMode.CONTENT


def test_skip_ig_metadata_does_not_resolve_or_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = explain._build_skipped_ig_metadata("auto")

    assert metadata == {
        "executed": False,
        "requested_ig_mode": "auto",
        "resolved_ig_mode": None,
    }
    assert not caught


def test_production_skip_branch_does_not_validate_or_resolve_ig_metadata():
    source = inspect.getsource(explain.main)
    after_skip_condition = source.split("if args.skip_ig:", maxsplit=1)[1]
    skip_branch, ig_branch = after_skip_condition.split("else:", maxsplit=1)

    assert "_validate_config_content_dim" not in skip_branch
    assert "resolve_ig_mode" not in skip_branch
    assert "_read_position_strategy_metadata" not in skip_branch
    assert "_create_integrated_gradients_explainer" not in skip_branch
    assert "_validate_config_content_dim" in ig_branch


def test_create_integrated_gradients_explainer_preserves_parameters(monkeypatch):
    captured = {}

    class FakeIntegratedGradientsExplainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        explain,
        "IntegratedGradientsExplainer",
        FakeIntegratedGradientsExplainer,
    )

    result = explain._create_integrated_gradients_explainer(
        model="model",
        device="cpu",
        n_steps=17,
        max_variants=23,
        resolved_ig_mode=ResolvedIGMode.CONTENT,
    )

    assert isinstance(result, FakeIntegratedGradientsExplainer)
    assert captured == {
        "model": "model",
        "device": "cpu",
        "n_steps": 17,
        "max_variants": 23,
        "ig_mode": ResolvedIGMode.CONTENT,
    }


def test_legacy_chunk_execution_passes_historical_features_without_splits():
    recorder = RecordingExplainer()
    covariates = torch.tensor([[0.5]])

    attr, positions, gene_ids, mask, chrom_ids = explain._attribute_chunk_for_ig(
        explainer=recorder,
        chunk=_chunk(),
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        device="cpu",
        chunk_covariates=covariates,
    )

    call = recorder.calls[-1]
    assert torch.equal(call["variant_features"], _chunk()["features"].unsqueeze(0))
    assert "content_features" not in call
    assert "absolute_position_features" not in call
    assert torch.equal(call["covariates"], covariates)
    assert torch.equal(attr, torch.ones(1, 2, 3))
    assert torch.equal(positions, _chunk()["positions"].unsqueeze(0))
    assert torch.equal(gene_ids, _chunk()["gene_ids"].unsqueeze(0))
    assert torch.equal(mask, _chunk()["mask"].unsqueeze(0))
    assert torch.equal(chrom_ids, _chunk()["chrom_ids"].unsqueeze(0))


def test_content_chunk_execution_uses_splits_without_features_key():
    recorder = RecordingExplainer()
    covariates = torch.tensor([[0.5]])
    chunk = _chunk(include_features=False)

    attr, positions, gene_ids, mask, chrom_ids = explain._attribute_chunk_for_ig(
        explainer=recorder,
        chunk=chunk,
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        device="cpu",
        chunk_covariates=covariates,
    )

    call = recorder.calls[-1]
    assert call["variant_features"] is None
    assert torch.equal(call["content_features"], chunk["content_features"].unsqueeze(0))
    assert torch.equal(
        call["absolute_position_features"],
        chunk["absolute_position_features"].unsqueeze(0),
    )
    assert torch.equal(call["positions"], chunk["positions"].unsqueeze(0))
    assert torch.equal(call["gene_ids"], chunk["gene_ids"].unsqueeze(0))
    assert torch.equal(call["mask"], chunk["mask"].unsqueeze(0))
    assert torch.equal(call["chrom_ids"], chunk["chrom_ids"].unsqueeze(0))
    assert torch.equal(call["covariates"], covariates)
    assert torch.equal(attr, torch.ones(1, 2, 2))
    assert torch.equal(positions, chunk["positions"].unsqueeze(0))
    assert torch.equal(gene_ids, chunk["gene_ids"].unsqueeze(0))
    assert torch.equal(mask, chunk["mask"].unsqueeze(0))
    assert torch.equal(chrom_ids, chunk["chrom_ids"].unsqueeze(0))


@pytest.mark.parametrize("missing_key", ["content_features", "absolute_position_features"])
def test_content_chunk_execution_requires_both_split_tensors(missing_key):
    chunk = _chunk(include_features=False)
    del chunk[missing_key]

    with pytest.raises(ValueError, match="content_features.*absolute_position_features"):
        explain._attribute_chunk_for_ig(
            explainer=RecordingExplainer(),
            chunk=chunk,
            resolved_ig_mode=ResolvedIGMode.CONTENT,
            device="cpu",
        )


def test_legacy_chunk_execution_requires_historical_features():
    with pytest.raises(ValueError, match="features"):
        explain._attribute_chunk_for_ig(
            explainer=RecordingExplainer(),
            chunk=_chunk(include_features=False),
            resolved_ig_mode=ResolvedIGMode.LEGACY,
            device="cpu",
        )


def test_expected_widths_follow_resolved_mode_metadata():
    content = explain._build_ig_run_metadata(
        requested_ig_mode="content",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        config=_new_config(),
        content_dim=7,
        input_dim=71,
        n_steps=50,
        max_variants=2000,
    )
    legacy = explain._build_ig_run_metadata(
        requested_ig_mode="legacy",
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        config=_new_config(),
        content_dim=7,
        input_dim=71,
        n_steps=50,
        max_variants=2000,
    )

    assert content["attribution_width"] == 7
    assert legacy["attribution_width"] == 71


def test_validate_attribution_width_rejects_wrong_width_before_filtering():
    raw_attributions = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unexpected attribution feature width"):
        explain._validate_attribution_width(raw_attributions, expected_width=2)


def test_mask_filtering_excludes_padded_rows_from_raw_scores_and_metadata():
    raw_attributions = np.array(
        [[1.0, 0.0], [99.0, 99.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    mask = np.array([True, False, True])
    positions = np.array([10, 999, 30])
    gene_ids = np.array([0, 99, 1])
    chroms = np.array(["1", "bad", "2"])

    explain._validate_attribution_width(raw_attributions, expected_width=2)
    attrs = raw_attributions[mask]
    scores = np.linalg.norm(attrs, ord=2, axis=1)

    assert attrs.shape == (2, 2)
    np.testing.assert_allclose(scores, [1.0, 5.0])
    assert positions[mask].tolist() == [10, 30]
    assert gene_ids[mask].tolist() == [0, 1]
    assert chroms[mask].tolist() == ["1", "2"]


def test_config_content_dim_uses_structural_authority_and_accepts_match():
    assert explain._validate_config_content_dim({"level": "L2"}, AnnotationLevel.L2) == 5
    assert (
        explain._validate_config_content_dim(
            _new_config(default_ig_mode="content"),
            AnnotationLevel.L3,
        )
        == 7
    )


def test_old_schema_content_dim_is_not_enforced_for_ig_compatibility():
    assert explain._validate_config_content_dim({"content_dim": 999}, AnnotationLevel.L2) == 5


def test_new_schema_conflicting_content_dim_raises():
    config = _new_config()
    config["content_dim"] = 6

    with pytest.raises(ValueError, match="content_dim"):
        explain._validate_config_content_dim(config, AnnotationLevel.L3)


@pytest.mark.parametrize("value", [4, "5", True])
def test_config_content_dim_rejects_conflicts_and_non_integer_values(value):
    config = _new_config()
    config["content_dim"] = value

    with pytest.raises(ValueError, match="content_dim"):
        explain._validate_config_content_dim(config, AnnotationLevel.L3)


def test_input_dim_remains_config_width_authority():
    config = _new_config()
    config["input_dim"] = 123
    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="legacy",
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        config=config,
        content_dim=7,
        input_dim=config["input_dim"],
        n_steps=9,
        max_variants=11,
    )

    assert metadata["input_dim"] == 123
    assert metadata["attribution_width"] == 123


def test_content_metadata_values_match_contract():
    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="auto",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        config=_new_config(),
        content_dim=7,
        input_dim=71,
        n_steps=9,
        max_variants=11,
    )

    assert metadata["attribution_schema_version"] == 1
    assert metadata["requested_ig_mode"] == "auto"
    assert metadata["resolved_ig_mode"] == "content"
    assert metadata["attribution_feature_space"] == "content"
    assert metadata["attribution_width"] == 7
    assert metadata["content_dim"] == 7
    assert metadata["input_dim"] == 71
    assert metadata["variant_score_aggregation"] == "l2"
    assert metadata["baseline_policy"] == "zero_content_observed_absolute_position"
    assert metadata["n_steps"] == 9
    assert metadata["max_variants"] == 11
    assert metadata["sampling_policy"] == "manual_chunk_full_coverage_no_random_subsampling"
    assert metadata["sampling_seed"] is None
    assert metadata["comparability_warning"] is None


def test_legacy_metadata_values_match_contract():
    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="legacy",
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        config=_new_config(),
        content_dim=7,
        input_dim=71,
        n_steps=9,
        max_variants=11,
    )

    assert metadata["resolved_ig_mode"] == "legacy"
    assert metadata["attribution_feature_space"] == "legacy"
    assert metadata["attribution_width"] == 71
    assert metadata["baseline_policy"] == "zero_historical_features"
    assert "not directly comparable" in metadata["comparability_warning"]


def test_position_strategy_metadata_new_schema_and_old_config():
    assert explain._read_position_strategy_metadata(_new_config()) == {
        "absolute_position_encoding": "sinusoidal",
        "relative_position_encoding": "t5_bucket",
        "chromosome_encoding": "learned",
        "position_encoding_metadata_source": "config",
    }
    assert explain._read_position_strategy_metadata({}) == {
        "absolute_position_encoding": None,
        "relative_position_encoding": None,
        "chromosome_encoding": None,
        "position_encoding_metadata_source": "unavailable_old_config",
    }


@pytest.mark.parametrize(
    "config",
    [
        {"position_encoding": "bad"},
        {"position_encoding": {"absolute": {}, "relative": {}, "chromosome": {}}},
        {
            "position_encoding": {
                "absolute": {"type": "sinusoidal"},
                "relative": {"type": 1},
                "chromosome": {"encoding": "learned"},
            }
        },
    ],
)
def test_malformed_position_strategy_metadata_raises(config):
    with pytest.raises(ValueError, match="position_encoding|must be a string"):
        explain._read_position_strategy_metadata(config)


def test_per_sample_npz_metadata_is_pickle_free_and_keeps_existing_keys(tmp_path):
    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="content",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        config=_new_config(),
        content_dim=2,
        input_dim=66,
        n_steps=3,
        max_variants=4,
    )
    path = tmp_path / "sample_0.npz"
    np.savez(
        path,
        attributions=np.ones((2, 2), dtype=np.float32),
        variant_scores=np.ones(2, dtype=np.float32),
        **explain._npz_scalar_metadata(metadata, per_sample=True),
    )

    with np.load(path, allow_pickle=False) as data:
        assert {"attributions", "variant_scores"}.issubset(data.files)
        assert data["attribution_width"].item() == data["attributions"].shape[1]
        for key in data.files:
            assert data[key].dtype != object


def test_top_level_npz_metadata_is_pickle_free_except_historical_arrays(tmp_path):
    metadata = explain._build_ig_run_metadata(
        requested_ig_mode="legacy",
        resolved_ig_mode=ResolvedIGMode.LEGACY,
        config={},
        content_dim=2,
        input_dim=66,
        n_steps=3,
        max_variants=4,
    )
    path = tmp_path / "attributions.npz"
    np.savez(
        path,
        variant_scores=np.array([np.ones(2)], dtype=object),
        metadata=np.array([{"sample_idx": 0}], dtype=object),
        **explain._npz_scalar_metadata(metadata, per_sample=False),
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["resolved_ig_mode"].item() == "legacy"
        assert data["attribution_feature_space"].item() == "legacy"
        assert data["absolute_position_encoding"].item() == "unavailable"
        assert data["sampling_seed"].item() == -1
        for key in set(data.files) - {"variant_scores", "metadata"}:
            assert data[key].dtype != object

    with np.load(path, allow_pickle=True) as data:
        assert "variant_scores" in data.files
        assert "metadata" in data.files


def test_analysis_metadata_shapes_for_executed_and_skipped_ig():
    ig_metadata = explain._build_ig_run_metadata(
        requested_ig_mode="content",
        resolved_ig_mode=ResolvedIGMode.CONTENT,
        config=_new_config(),
        content_dim=7,
        input_dim=71,
        n_steps=9,
        max_variants=11,
    )
    executed = {"integrated_gradients": {"executed": True, **ig_metadata}}
    skipped = {"integrated_gradients": explain._build_skipped_ig_metadata("auto")}

    assert executed["integrated_gradients"]["executed"] is True
    assert executed["integrated_gradients"]["resolved_ig_mode"] == "content"
    assert skipped["integrated_gradients"] == {
        "executed": False,
        "requested_ig_mode": "auto",
        "resolved_ig_mode": None,
    }


def test_ranking_metadata_annotation_preserves_numeric_columns():
    original = pd.DataFrame(
        {
            "score": [0.2, 0.1],
            "rank": [1, 2],
            "gene_score": [3.0, 1.0],
        }
    )
    metadata = {
        "resolved_ig_mode": "content",
        "attribution_feature_space": "content",
        "variant_score_aggregation": "l2",
    }

    annotated = explain._annotate_ranking_metadata(original, metadata)

    pd.testing.assert_frame_equal(annotated[original.columns], original)
    assert set(annotated.columns) - set(original.columns) == {
        "resolved_ig_mode",
        "attribution_feature_space",
        "variant_score_aggregation",
    }
    assert annotated["resolved_ig_mode"].tolist() == ["content", "content"]


def test_load_sample_attributions_returns_historical_two_keys_and_docstring_is_mode_aware(tmp_path):
    per_sample_dir = tmp_path / "attributions_per_sample"
    per_sample_dir.mkdir()
    np.savez(
        per_sample_dir / "sample_0.npz",
        attributions=np.ones((2, 3), dtype=np.float32),
        variant_scores=np.ones(2, dtype=np.float32),
        attribution_width=np.asarray(3),
    )

    result = load_sample_attributions(per_sample_dir, 0)

    assert set(result) == {"attributions", "variant_scores"}
    doc = load_sample_attributions.__doc__
    assert "input_dim" in doc
    assert "content_dim" in doc


def test_production_main_uses_mode_aware_chunk_helper():
    assert "_attribute_chunk_for_ig(" in inspect.getsource(explain.main)


def test_attention_code_path_remains_historical_feature_based():
    source = inspect.getsource(explain.main)
    attention_section = source.split("# === ATTENTION ANALYSIS ===", maxsplit=1)[1]

    assert "features = batch['features'].to(args.device)" in attention_section
    assert "content_features" not in attention_section


def test_no_custom_position_strategy_cli_flags_are_exposed():
    help_text = explain.build_arg_parser().format_help()

    assert "--absolute-position-encoding" not in help_text
    assert "--relative-position-encoding" not in help_text
