"""
Unit tests for Phase 3 explainability components.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import explain
from src.encoding import AnnotationLevel, VariantDataset, collate_samples
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.shap_epistasis import SHAPEpistasisDetector
from src.explain.variant_ranking import VariantRanker
from src.models.sieve import create_sieve_model, load_state_dict_with_legacy_upgrade


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "test_data" / "small"


@pytest.fixture
def preprocessed_data(test_data_dir):
    """Load preprocessed test data."""
    data_path = test_data_dir / "preprocessed_test.pt"
    if not data_path.exists():
        pytest.skip(f"Test data not found: {data_path}")
    return torch.load(data_path, weights_only=False)


@pytest.fixture
def test_checkpoint(test_data_dir):
    """Load test model checkpoint."""
    checkpoint_path = test_data_dir / "test_model" / "L3_run" / "fold_0" / "best_model.pt"
    if not checkpoint_path.exists():
        pytest.skip(f"Test checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, weights_only=False, map_location='cpu')


@pytest.fixture
def test_dataset(preprocessed_data):
    """Create test dataset."""
    samples = preprocessed_data['samples']
    return VariantDataset(samples, annotation_level=AnnotationLevel.L3)


@pytest.fixture
def test_model(test_checkpoint, test_dataset):
    """Create and load test model."""
    config = {
        'input_dim': 71,
        'latent_dim': 64,
        'hidden_dim': 32,
        'num_heads': 2,
        'num_attention_layers': 1,
        'dropout': 0.1,
    }
    model = create_sieve_model(config, num_genes=test_dataset.num_genes)
    # Legacy checkpoints predate the chromosome-aware position bias and
    # chrom_embedding; the helper pads grown tensors and tolerates new keys.
    load_state_dict_with_legacy_upgrade(model, test_checkpoint['model_state_dict'])
    model.eval()
    return model


def _minimum_explain_argv(tmp_path, *extra):
    return [
        "--experiment-dir",
        str(tmp_path / "experiment"),
        "--preprocessed-data",
        str(tmp_path / "preprocessed.pt"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def _write_config(path: Path) -> None:
    path.write_text(yaml.safe_dump({"level": "L3"}), encoding="utf-8")


def _write_checkpoint(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_cv_results(path: Path, aucs) -> None:
    path.write_text(
        yaml.safe_dump({"fold_results": [{"auc": auc} for auc in aucs]}),
        encoding="utf-8",
    )


def _patch_torch_load(monkeypatch):
    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append(Path(path))
        return {"model_state_dict": {}}

    monkeypatch.setattr(explain.torch, "load", fake_load)
    return calls


def _experiment_dir(tmp_path, *, aucs=None, checkpoint_payloads=None):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    _write_config(exp_dir / "config.yaml")
    if aucs is None:
        _write_checkpoint(exp_dir / "best_model.pt", b"single-run")
    else:
        _write_cv_results(exp_dir / "cv_results.yaml", aucs)
        checkpoint_payloads = checkpoint_payloads or [
            f"fold-{i}".encode() for i in range(len(aucs))
        ]
        for i, payload in enumerate(checkpoint_payloads):
            _write_checkpoint(exp_dir / f"fold_{i}" / "best_model.pt", payload)
    return exp_dir


def test_parse_args_accepts_fold_index_with_experiment_dir(tmp_path):
    args = explain.parse_args(_minimum_explain_argv(tmp_path, "--fold-index", "1"))

    assert args.fold_index == 1


def test_parse_args_rejects_fold_index_with_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        explain.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--config",
                str(tmp_path / "config.yaml"),
                "--preprocessed-data",
                str(tmp_path / "preprocessed.pt"),
                "--output-dir",
                str(tmp_path / "out"),
                "--fold-index",
                "1",
            ]
        )


def test_parse_args_rejects_negative_fold_index(tmp_path):
    with pytest.raises(SystemExit):
        explain.parse_args(_minimum_explain_argv(tmp_path, "--fold-index", "-1"))


def test_sha256_file_uses_exact_checkpoint_bytes(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint-a")

    assert explain._sha256_file(checkpoint_path) == hashlib.sha256(
        b"checkpoint-a"
    ).hexdigest()

    checkpoint_path.write_bytes(b"checkpoint-b")
    assert explain._sha256_file(checkpoint_path) == hashlib.sha256(
        b"checkpoint-b"
    ).hexdigest()


def test_load_model_explicit_cv_fold_records_selected_checkpoint(monkeypatch, tmp_path):
    exp_dir = _experiment_dir(
        tmp_path,
        aucs=[0.91, 0.73, 0.99],
        checkpoint_payloads=[b"fold0", b"fold1-selected", b"fold2-best"],
    )
    calls = _patch_torch_load(monkeypatch)
    args = explain.parse_args(
        [
            "--experiment-dir",
            str(exp_dir),
            "--preprocessed-data",
            str(tmp_path / "preprocessed.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            "--fold-index",
            "1",
        ]
    )

    result = explain.load_model_and_config(args)
    provenance = result.model_provenance
    selected_checkpoint = (exp_dir / "fold_1" / "best_model.pt").resolve()

    assert calls == [selected_checkpoint]
    assert provenance["checkpoint_selection_mode"] == "cv_explicit_fold"
    assert provenance["selected_fold"] == 1
    assert provenance["selected_fold_auc"] == 0.73
    assert provenance["checkpoint_path"] == str(selected_checkpoint)
    assert provenance["checkpoint_sha256"] == hashlib.sha256(
        b"fold1-selected"
    ).hexdigest()
    assert provenance["config_path"] == str((exp_dir / "config.yaml").resolve())
    assert provenance["cv_results_path"] == str((exp_dir / "cv_results.yaml").resolve())


def test_load_model_preserves_historical_cv_best_fold_selection(monkeypatch, tmp_path):
    exp_dir = _experiment_dir(tmp_path, aucs=[0.71, 0.95, 0.82])
    calls = _patch_torch_load(monkeypatch)
    args = explain.parse_args(_minimum_explain_argv(tmp_path))

    result = explain.load_model_and_config(args)
    provenance = result.model_provenance
    selected_checkpoint = (exp_dir / "fold_1" / "best_model.pt").resolve()

    assert calls == [selected_checkpoint]
    assert provenance["checkpoint_selection_mode"] == "cv_best_fold"
    assert provenance["selected_fold"] == 1
    assert provenance["selected_fold_auc"] == 0.95


def test_load_model_single_run_records_best_model(monkeypatch, tmp_path):
    exp_dir = _experiment_dir(tmp_path, aucs=None)
    calls = _patch_torch_load(monkeypatch)
    args = explain.parse_args(_minimum_explain_argv(tmp_path))

    result = explain.load_model_and_config(args)
    provenance = result.model_provenance
    selected_checkpoint = (exp_dir / "best_model.pt").resolve()

    assert calls == [selected_checkpoint]
    assert provenance["checkpoint_selection_mode"] == "single_run_best_model"
    assert provenance["selected_fold"] is None
    assert provenance["selected_fold_auc"] is None
    assert provenance["cv_results_path"] is None


def test_load_model_explicit_checkpoint_records_supplied_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "model.pt"
    _write_config(config_path)
    checkpoint_path.write_bytes(b"explicit")
    calls = _patch_torch_load(monkeypatch)
    args = explain.parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--preprocessed-data",
            str(tmp_path / "preprocessed.pt"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    result = explain.load_model_and_config(args)
    provenance = result.model_provenance

    assert calls == [checkpoint_path.resolve()]
    assert provenance["checkpoint_selection_mode"] == "explicit_checkpoint"
    assert provenance["checkpoint_path"] == str(checkpoint_path.resolve())
    assert provenance["config_path"] == str(config_path.resolve())
    assert provenance["selected_fold"] is None
    assert provenance["selected_fold_auc"] is None
    assert provenance["cv_results_path"] is None


def test_load_model_rejects_fold_index_out_of_range(monkeypatch, tmp_path):
    _experiment_dir(tmp_path, aucs=[0.1])
    _patch_torch_load(monkeypatch)
    args = explain.parse_args(
        _minimum_explain_argv(tmp_path, "--fold-index", "2")
    )

    with pytest.raises(ValueError, match="out of range"):
        explain.load_model_and_config(args)


def test_load_model_rejects_fold_index_without_cv_results(monkeypatch, tmp_path):
    _experiment_dir(tmp_path, aucs=None)
    _patch_torch_load(monkeypatch)
    args = explain.parse_args(
        _minimum_explain_argv(tmp_path, "--fold-index", "0")
    )

    with pytest.raises(ValueError, match="cv_results.yaml"):
        explain.load_model_and_config(args)


@pytest.mark.parametrize(
    "payload, match",
    [
        ([], "mapping"),
        ({"fold_results": []}, "non-empty fold_results"),
        ({"fold_results": ["bad"]}, "fold_results\\[0\\] must be a mapping"),
        ({"fold_results": [{}]}, "must contain 'auc'"),
        ({"fold_results": [{"auc": "bad"}]}, "finite number"),
        ({"fold_results": [{"auc": float("nan")}]}, "finite number"),
        ({"fold_results": [{"auc": float("inf")}]}, "finite number"),
    ],
)
def test_load_model_rejects_malformed_cv_results(monkeypatch, tmp_path, payload, match):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    _write_config(exp_dir / "config.yaml")
    (exp_dir / "cv_results.yaml").write_text(
        yaml.safe_dump(payload),
        encoding="utf-8",
    )
    _patch_torch_load(monkeypatch)
    args = explain.parse_args(_minimum_explain_argv(tmp_path))

    with pytest.raises(ValueError, match=match):
        explain.load_model_and_config(args)


def test_load_model_rejects_missing_selected_checkpoint(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    _write_config(exp_dir / "config.yaml")
    _write_cv_results(exp_dir / "cv_results.yaml", [0.9])
    _patch_torch_load(monkeypatch)
    args = explain.parse_args(_minimum_explain_argv(tmp_path))

    with pytest.raises(FileNotFoundError, match=str(exp_dir / "fold_0" / "best_model.pt")):
        explain.load_model_and_config(args)


def test_load_model_rejects_missing_explicit_checkpoint(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    missing_checkpoint = tmp_path / "missing.pt"
    _write_config(config_path)
    _patch_torch_load(monkeypatch)
    args = explain.parse_args(
        [
            "--checkpoint",
            str(missing_checkpoint),
            "--config",
            str(config_path),
            "--preprocessed-data",
            str(tmp_path / "preprocessed.pt"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(FileNotFoundError, match=str(missing_checkpoint.resolve())):
        explain.load_model_and_config(args)


def test_analysis_metadata_writes_top_level_model_provenance_for_all_ig_modes():
    source = Path("scripts/explain.py").read_text(encoding="utf-8")
    metadata_section = source.split("# === SAVE ANALYSIS METADATA ===", maxsplit=1)[1]

    assert "'model_provenance': model_load.model_provenance" in metadata_section
    assert "'integrated_gradients': integrated_gradients_metadata" in metadata_section
    assert "_npz_scalar_metadata" not in metadata_section


class TestIntegratedGradients:
    """Test integrated gradients explainer."""

    def test_initialization(self, test_model):
        """Test explainer initialization."""
        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)
        assert explainer.model is test_model
        assert explainer.device == 'cpu'
        assert explainer.n_steps == 5

    def test_attribute_single_sample(self, test_model, test_dataset):
        """Test attribution for a single sample."""
        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)

        # Get a sample
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Compute attributions
        attributions = explainer.attribute(features, positions, gene_ids, mask)

        # Check output shape
        assert attributions.shape == features.shape
        assert not torch.isnan(attributions).any()

    def test_attribute_batch(self, test_model, test_dataset):
        """Test batch attribution."""
        from torch.utils.data import DataLoader

        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)
        dataloader = DataLoader(
            test_dataset,
            batch_size=4,
            collate_fn=collate_samples,
            shuffle=False
        )

        all_attrs, all_scores, all_meta = explainer.attribute_batch(
            dataloader, aggregate='l2'
        )

        # Check we got results for all samples
        assert len(all_attrs) == len(test_dataset)
        assert len(all_scores) == len(test_dataset)
        assert len(all_meta) == len(test_dataset)

        # Check shapes and values
        for attrs, scores in zip(all_attrs, all_scores):
            assert attrs.ndim == 2  # (n_variants, n_features)
            assert scores.ndim == 1  # (n_variants,)
            assert len(attrs) == len(scores)
            assert not np.isnan(attrs).any()
            assert not np.isnan(scores).any()


class TestAttentionAnalysis:
    """Test attention pattern analysis."""

    def test_initialization(self, test_model):
        """Test analyzer initialization."""
        analyzer = AttentionAnalyzer(test_model, device='cpu')
        assert analyzer.model is not None

    def test_extract_attention(self, test_model, test_dataset):
        """Test attention extraction."""
        analyzer = AttentionAnalyzer(test_model, device='cpu')

        # Get a sample
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Extract attention weights
        attention_weights = analyzer.extract_attention_weights(
            features, positions, gene_ids, mask
        )

        # Should return a list of attention weight tensors
        assert isinstance(attention_weights, list)
        if len(attention_weights) > 0:
            # Each element should be a tensor
            assert isinstance(attention_weights[0], torch.Tensor)
            # Find interactions
            interactions = analyzer.find_top_interactions(
                attention_weights, positions, gene_ids, mask, top_k=10
            )
            assert isinstance(interactions, list)


class TestVariantRanking:
    """Test variant ranking."""

    def test_rank_variants(self):
        """Test variant ranking."""
        ranker = VariantRanker()

        # Create dummy attribution data (chromosomes now required)
        all_scores = [
            np.array([0.5, 0.3, 0.8]),
            np.array([0.2, 0.9]),
        ]
        all_meta = [
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 0]),
                'chromosomes': np.array(['1', '1', '1']),
                'label': 1
            },
            {
                'positions': np.array([100, 400]),
                'gene_ids': np.array([0, 2]),
                'chromosomes': np.array(['1', '2']),
                'label': 0
            }
        ]

        rankings = ranker.rank_variants(all_scores, all_meta)

        # Check output structure (check only fields that are actually returned)
        assert 'position' in rankings.columns
        assert 'gene_id' in rankings.columns
        assert 'mean_attribution' in rankings.columns
        assert 'num_samples' in rankings.columns
        assert 'chromosome' in rankings.columns

        # Check we got all unique (chrom, pos, gene) combinations
        unique_keys = set()
        for meta in all_meta:
            for chrom, pos, gene in zip(meta['chromosomes'], meta['positions'], meta['gene_ids']):
                unique_keys.add((chrom, pos, gene))
        assert len(rankings) == len(unique_keys)

    def test_rank_variants_chromosome_collision_prevention(self):
        """Test that same position on different chromosomes creates separate entries."""
        ranker = VariantRanker()

        # Create data where position 100 with gene 0 exists on BOTH chr1 and chrX
        # This would have caused a collision with the old (pos, gene) key
        all_scores = [
            np.array([0.9, 0.1]),  # High score for chr1:100, low for chrX:100
        ]
        all_meta = [
            {
                'positions': np.array([100, 100]),  # Same position!
                'gene_ids': np.array([0, 0]),        # Same gene!
                'chromosomes': np.array(['1', 'X']), # Different chromosomes
                'label': 1
            },
        ]

        rankings = ranker.rank_variants(all_scores, all_meta)

        # CRITICAL: We should get TWO rows, not one
        assert len(rankings) == 2, (
            f"Expected 2 rows for same pos/gene on different chromosomes, got {len(rankings)}"
        )

        # Verify both chromosomes are present
        chroms = set(rankings['chromosome'].tolist())
        assert chroms == {'1', 'X'}, f"Expected chromosomes {{'1', 'X'}}, got {chroms}"

        # Verify the attribution scores are correct (not merged)
        chr1_row = rankings[rankings['chromosome'] == '1'].iloc[0]
        chrX_row = rankings[rankings['chromosome'] == 'X'].iloc[0]
        assert abs(chr1_row['mean_attribution'] - 0.9) < 0.01, "Chr1 attribution should be 0.9"
        assert abs(chrX_row['mean_attribution'] - 0.1) < 0.01, "ChrX attribution should be 0.1"

    def test_rank_variants_missing_chromosomes_raises_error(self):
        """Test that missing chromosomes in metadata raises ValueError."""
        ranker = VariantRanker()

        all_scores = [np.array([0.5])]
        all_meta = [
            {
                'positions': np.array([100]),
                'gene_ids': np.array([0]),
                # NO 'chromosomes' key - should raise error
                'label': 1
            },
        ]

        with pytest.raises(ValueError, match="chromosomes"):
            ranker.rank_variants(all_scores, all_meta)

    # ------------------------------------------------------------------
    # Rank convention: rank 1 = best variant, for all aggregation methods
    # ------------------------------------------------------------------

    def _make_ranked(self, aggregation: str):
        """Return rankings for a 3-variant, 2-sample dataset."""
        # Variant A (chr1:100, gene 0): high attribution in both samples → best
        # Variant B (chr1:200, gene 1): medium attribution
        # Variant C (chr1:300, gene 2): low attribution
        all_scores = [
            np.array([0.9, 0.5, 0.1]),
            np.array([0.8, 0.4, 0.2]),
        ]
        all_meta = [
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 2]),
                'chromosomes': np.array(['1', '1', '1']),
            },
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 2]),
                'chromosomes': np.array(['1', '1', '1']),
            },
        ]
        ranker = VariantRanker(aggregation=aggregation)
        return ranker.rank_variants(all_scores, all_meta)

    def test_rank_convention_mean_best_variant_is_rank1(self):
        """aggregation='mean': variant with highest mean_attribution must get rank 1."""
        rankings = self._make_ranked('mean')
        best = rankings.loc[rankings['mean_attribution'].idxmax()]
        assert best['rank'] == 1, (
            f"Expected rank 1 for best mean variant, got {best['rank']}"
        )

    def test_rank_convention_max_best_variant_is_rank1(self):
        """aggregation='max': variant with highest max_attribution must get rank 1."""
        rankings = self._make_ranked('max')
        best = rankings.loc[rankings['max_attribution'].idxmax()]
        assert best['rank'] == 1, (
            f"Expected rank 1 for best max variant, got {best['rank']}"
        )

    def test_rank_convention_rank_average_best_variant_is_rank1(self):
        """aggregation='rank_average': variant with lowest composite score must get rank 1."""
        rankings = self._make_ranked('rank_average')
        # Lowest composite (rank_mean + rank_max + rank_samples) / 3 = best
        best = rankings.loc[rankings['score'].idxmin()]
        assert best['rank'] == 1, (
            f"Expected rank 1 for best composite variant, got {best['rank']}"
        )
        # Sanity-check: the overall best-attribution variant should also top the composite
        best_mean_pos = rankings.loc[rankings['mean_attribution'].idxmax(), 'position']
        rank1_pos = rankings.loc[rankings['rank'] == 1, 'position'].iloc[0]
        assert best_mean_pos == rank1_pos, (
            "Variant with highest mean_attribution should also top rank_average ranking"
        )

    def test_aggregation_mean_score_equals_mean_attribution(self):
        """With aggregation='mean', score must equal mean_attribution for every row."""
        import pandas as pd
        ranker = VariantRanker(aggregation='mean')

        all_scores = [
            np.array([0.9, 0.2, 0.5]),
            np.array([0.7, 0.4, 0.5]),
        ]
        all_meta = [
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 2]),
                'chromosomes': np.array(['1', '1', '1']),
            },
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 2]),
                'chromosomes': np.array(['1', '1', '1']),
            },
        ]
        rankings = ranker.rank_variants(all_scores, all_meta)
        pd.testing.assert_series_equal(
            rankings['score'].reset_index(drop=True),
            rankings['mean_attribution'].reset_index(drop=True),
            check_names=False,
        )

    def test_rank_genes(self):
        """Test gene ranking."""
        ranker = VariantRanker()

        # Create dummy variant rankings
        import pandas as pd
        variant_rankings = pd.DataFrame({
            'position': [100, 200, 300, 400],
            'gene_id': [0, 1, 0, 2],
            'mean_attribution': [0.5, 0.3, 0.8, 0.2],
            'num_samples': [2, 1, 1, 1]
        })

        gene_rankings = ranker.rank_genes(variant_rankings)

        # Check output structure
        assert 'gene_id' in gene_rankings.columns
        assert 'num_variants' in gene_rankings.columns
        assert 'gene_score' in gene_rankings.columns
        assert 'top_variant_pos' in gene_rankings.columns

        # Check gene with 2 variants has higher count
        gene0 = gene_rankings[gene_rankings['gene_id'] == 0].iloc[0]
        assert gene0['num_variants'] == 2


class TestSHAPEpistasis:
    """Test SHAP epistasis detection."""

    def test_initialization(self, test_model):
        """Test detector initialization."""
        detector = SHAPEpistasisDetector(test_model, device='cpu')
        assert detector.model is test_model
        assert detector.device == 'cpu'

    def test_validate_interaction(self, test_model, test_dataset):
        """Test counterfactual perturbation."""
        detector = SHAPEpistasisDetector(test_model, device='cpu')

        # Get a sample with multiple variants
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Find two variants to test
        variant_indices = torch.where(mask[0])[0]
        if len(variant_indices) < 2:
            pytest.skip("Sample needs at least 2 variants")

        v1_idx = variant_indices[0].item()
        v2_idx = variant_indices[1].item()

        # Validate interaction
        result = detector.validate_interaction_with_perturbation(
            features, positions, gene_ids, mask, v1_idx, v2_idx
        )

        # Check result structure (use actual keys returned)
        assert 'pred_both' in result
        assert 'pred_variant1_only' in result
        assert 'pred_variant2_only' in result
        assert 'pred_neither' in result
        assert 'effect_variant1' in result
        assert 'effect_variant2' in result
        assert 'effect_combined' in result
        assert 'synergy' in result
        assert 'interaction_type' in result

        # Check interaction type is valid
        assert result['interaction_type'] in ['synergistic', 'antagonistic', 'independent']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_validate_epistasis_empty_file():
    """Test validate_epistasis.py handles empty interaction files."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Create temporary empty interactions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")  # Empty file
        empty_file = f.name

    try:
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(project_root)

        # Run validate_epistasis.py script
        result = subprocess.run(
            [sys.executable, 'scripts/validate_epistasis.py',
             '--interactions', empty_file,
             '--checkpoint', 'test_data/small/test_model/L3_run/fold_0/best_model.pt',
             '--config', 'test_data/small/test_model/L3_run/config.yaml',
             '--preprocessed-data', 'test_data/small/preprocessed_test.pt',
             '--output-dir', '/tmp/test_epistasis',
             '--device', 'cpu'],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit cleanly (not crash) with helpful error message
        assert 'ERROR: Interactions file is empty' in result.stdout
        assert 'No interactions were found' in result.stdout
        
    finally:
        # Clean up
        Path(empty_file).unlink(missing_ok=True)


def test_validate_epistasis_no_data():
    """Test validate_epistasis.py handles CSV with headers but no data."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Create CSV with headers but no data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("pos1,pos2,gene1,gene2,weight\n")  # Header only
        header_only_file = f.name

    try:
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(project_root)

        result = subprocess.run(
            [sys.executable, 'scripts/validate_epistasis.py',
             '--interactions', header_only_file,
             '--checkpoint', 'test_data/small/test_model/L3_run/fold_0/best_model.pt',
             '--config', 'test_data/small/test_model/L3_run/config.yaml',
             '--preprocessed-data', 'test_data/small/preprocessed_test.pt',
             '--output-dir', '/tmp/test_epistasis',
             '--device', 'cpu'],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit cleanly with helpful error message
        assert 'ERROR: Interactions file has no rows' in result.stdout
        assert 'No interactions available' in result.stdout
        
    finally:
        Path(header_only_file).unlink(missing_ok=True)


class TestLoadSampleAttributions:
    """Test the load_sample_attributions helper function."""

    def test_load_single_sample(self, tmp_path):
        """Test loading a single sample's attributions from per-sample directory."""
        from src.explain import load_sample_attributions

        # Create mock per-sample directory
        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        # Save mock data for 3 samples
        for i in range(3):
            n_variants = 10 + i * 5
            input_dim = 6
            attrs = np.random.randn(n_variants, input_dim).astype(np.float32)
            scores = np.linalg.norm(attrs, ord=2, axis=1)
            np.savez(per_sample_dir / f'sample_{i}.npz',
                     attributions=attrs, variant_scores=scores)

        # Load sample 1
        result = load_sample_attributions(per_sample_dir, 1)

        assert 'attributions' in result
        assert 'variant_scores' in result
        assert result['attributions'].shape == (15, 6)
        assert result['variant_scores'].shape == (15,)

    def test_load_nonexistent_sample_raises(self, tmp_path):
        """Test that loading a nonexistent sample raises FileNotFoundError."""
        from src.explain import load_sample_attributions

        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_sample_attributions(per_sample_dir, 999)

    def test_scores_match_attributions(self, tmp_path):
        """Test that variant_scores are the L2 norm of raw attributions."""
        from src.explain import load_sample_attributions

        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        attrs = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        scores = np.linalg.norm(attrs, ord=2, axis=1)
        np.savez(per_sample_dir / 'sample_0.npz',
                 attributions=attrs, variant_scores=scores)

        result = load_sample_attributions(per_sample_dir, 0)
        np.testing.assert_allclose(result['variant_scores'], [1.0, 5.0, 0.0])
