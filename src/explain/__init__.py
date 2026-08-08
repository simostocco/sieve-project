"""
Explainability module for SIEVE.

This module provides tools for interpreting SIEVE models and discovering
disease-associated variants through:
- Integrated gradients for variant-level attribution
- Attention weight analysis for epistatic interactions
- Counterfactual epistasis detection and validation
- Variant ranking and prioritization
- Biological validation against databases
- Visualisation of discovered variants

Author: Francesco Lescai
"""

from pathlib import Path

import numpy as np

from .gradients import IntegratedGradientsExplainer
from .attention_analysis import AttentionAnalyzer
from .variant_ranking import VariantRanker
from .counterfactual_epistasis import CounterfactualEpistasisDetector
from .biological_validation import BiologicalValidator


def load_sample_attributions(per_sample_dir: str | Path, sample_idx: int) -> dict:
    """Load raw attributions for a single sample from the per-sample directory.

    Parameters
    ----------
    per_sample_dir : str or Path
        Path to the ``attributions_per_sample/`` directory produced by explain.py.
    sample_idx : int
        Zero-based sample index.

    Returns
    -------
    dict
        Keys ``'attributions'`` and ``'variant_scores'``. For legacy output
        files, the attribution matrix width is ``input_dim``. For content-mode
        output files, the attribution matrix width is ``content_dim``.
    """
    path = Path(per_sample_dir) / f'sample_{sample_idx}.npz'
    with np.load(path, allow_pickle=False) as data:
        attributions = data['attributions']
        variant_scores = data['variant_scores']
    return {
        'attributions': attributions,
        'variant_scores': variant_scores,
    }


__all__ = [
    'IntegratedGradientsExplainer',
    'AttentionAnalyzer',
    'VariantRanker',
    'CounterfactualEpistasisDetector',
    'BiologicalValidator',
    'load_sample_attributions',
]
