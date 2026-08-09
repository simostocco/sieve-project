"""
Attention pattern analysis for epistasis detection.

Analyzes attention weights from SIEVE's position-aware attention layers
to identify variant-variant interactions (epistasis).

Author: Francesco Lescai
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class AttentionAnalyzer:
    """
    Analyse attention patterns for epistatic interactions.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model
    device : str
        Device to run computations ('cuda' or 'cpu')
    attention_threshold : float
        Minimum attention weight to consider as interaction (default: 0.1)

    Attributes
    ----------
    model : nn.Module
        The SIEVE model
    device : str
        Computation device
    attention_threshold : float
        Threshold for significant attention

    Examples
    --------
    >>> analyzer = AttentionAnalyzer(model, device='cuda')
    >>> interactions = analyzer.extract_interactions(features, positions, gene_ids, mask)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        attention_threshold: float = 0.1,
        threshold_mode: str = 'absolute',
        attention_percentile: float = 99.9,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.attention_threshold = attention_threshold
        self.threshold_mode = threshold_mode
        self.attention_percentile = attention_percentile

        if self.threshold_mode not in {'absolute', 'percentile'}:
            raise ValueError(
                f"Unknown threshold_mode: {self.threshold_mode}. "
                "Expected 'absolute' or 'percentile'."
            )
        if not 0.0 <= self.attention_percentile <= 100.0:
            raise ValueError("attention_percentile must be between 0 and 100.")

    def extract_attention_weights(
        self,
        variant_features: Tensor | None,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        chrom_ids: Optional[Tensor] = None,
        *,
        content_features: Tensor | None = None,
        absolute_position_features: Tensor | None = None,
    ) -> List[Tensor]:
        """
        Extract attention weights from all layers.

        Parameters
        ----------
        variant_features : Optional[Tensor]
            Historical variant features, shape (batch, num_variants,
            input_dim). Must be supplied without split tensors in historical
            mode.
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Tensor
            Validity mask, shape (batch, num_variants)
        chrom_ids : Optional[Tensor]
            Chromosome indices, shape (batch, num_variants). When provided
            (and the model was constructed with ``num_chromosomes > 0``),
            enables chromosome-aware position bias and chromosome embedding.
        content_features, absolute_position_features : Optional[Tensor]
            Split feature pair for custom positional execution. Must be
            supplied together, with ``variant_features=None``.

        Returns
        -------
        attention_weights : List[Tensor]
            Attention weights from each layer
            Each tensor: (batch, num_heads, num_variants, num_variants)
        """
        has_historical = variant_features is not None
        has_content = content_features is not None
        has_position = absolute_position_features is not None

        if has_historical and (has_content or has_position):
            raise ValueError(
                "variant_features cannot be mixed with content_features or "
                "absolute_position_features"
            )
        if not has_historical and not has_content and not has_position:
            raise ValueError(
                "attention extraction requires either variant_features or the "
                "content_features/absolute_position_features pair"
            )
        if has_content != has_position:
            raise ValueError(
                "content_features and absolute_position_features must be supplied "
                "together"
            )

        if variant_features is not None:
            variant_features = variant_features.to(self.device)
        if content_features is not None:
            content_features = content_features.to(self.device)
        if absolute_position_features is not None:
            absolute_position_features = absolute_position_features.to(self.device)
        positions = positions.to(self.device)
        gene_ids = gene_ids.to(self.device)
        mask = mask.to(self.device)
        if chrom_ids is not None:
            chrom_ids = chrom_ids.to(self.device)

        with torch.no_grad():
            if has_historical:
                attention_weights = self.model.get_attention_patterns(
                    variant_features,
                    positions,
                    gene_ids,
                    mask,
                    chrom_ids=chrom_ids,
                )
            else:
                attention_weights = self.model.get_attention_patterns(
                    None,
                    positions,
                    gene_ids,
                    mask,
                    chrom_ids=chrom_ids,
                    content_features=content_features,
                    absolute_position_features=absolute_position_features,
                )

        return attention_weights

    def find_top_interactions(
        self,
        attention_weights: List[Tensor],
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        top_k: int = 100,
        aggregate_layers: str = 'mean',
        aggregate_heads: str = 'mean',
        sample_indices: Optional[Tensor] = None,
        chunk_indices: Optional[Tensor] = None,
    ) -> List[Dict]:
        """
        Find top variant-variant interactions based on attention.

        Parameters
        ----------
        attention_weights : List[Tensor]
            Attention weights from each layer
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Tensor
            Validity mask, shape (batch, num_variants)
        top_k : int
            Number of top interactions to return
        aggregate_layers : str
            How to aggregate across layers: 'mean', 'max', 'last'
        aggregate_heads : str
            How to aggregate across heads: 'mean', 'max'
        sample_indices : Tensor, optional
            Original sample indices aligned with the batch dimension.
        chunk_indices : Tensor, optional
            Chunk indices aligned with the batch dimension.

        Returns
        -------
        top_interactions : List[Dict]
            List of top interactions, each with:
            - variant1_pos, variant2_pos: genomic positions
            - variant1_gene, variant2_gene: gene IDs
            - attention_score: aggregated attention weight
            - same_gene: whether variants are in same gene
        """
        # Aggregate across layers
        if aggregate_layers == 'mean':
            attn = torch.mean(torch.stack(attention_weights), dim=0)
        elif aggregate_layers == 'max':
            attn = torch.max(torch.stack(attention_weights), dim=0)[0]
        elif aggregate_layers == 'last':
            attn = attention_weights[-1]
        else:
            raise ValueError(f"Unknown layer aggregation: {aggregate_layers}")

        # attn shape: (batch, num_heads, num_variants, num_variants)

        # Aggregate across heads
        if aggregate_heads == 'mean':
            attn = attn.mean(dim=1)  # (batch, num_variants, num_variants)
        elif aggregate_heads == 'max':
            attn = attn.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown head aggregation: {aggregate_heads}")

        # Process each sample in batch
        all_interactions = []

        for b in range(attn.shape[0]):
            # Get valid variants
            valid_mask = mask[b].cpu().numpy()
            n_valid = valid_mask.sum()

            if n_valid < 2:
                continue

            # Get attention for valid variants
            attn_matrix = attn[b].cpu().numpy()[:n_valid, :n_valid]
            pos = positions[b].cpu().numpy()[:n_valid]
            genes = gene_ids[b].cpu().numpy()[:n_valid]

            # Get upper triangle (avoid self-attention and duplicates)
            i_upper, j_upper = np.triu_indices(n_valid, k=1)

            # Get attention scores for pairs
            pair_scores = attn_matrix[i_upper, j_upper]

            if len(pair_scores) == 0:
                continue

            if self.threshold_mode == 'absolute':
                threshold_value = float(self.attention_threshold)
            else:
                threshold_value = float(
                    np.percentile(pair_scores, self.attention_percentile)
                )

            # Filter by threshold
            significant = pair_scores >= threshold_value
            i_sig = i_upper[significant]
            j_sig = j_upper[significant]
            scores_sig = pair_scores[significant]

            if len(scores_sig) == 0:
                continue

            # Sort by score
            sorted_indices = np.argsort(scores_sig)[::-1][:top_k]

            original_sample_idx = (
                int(sample_indices[b].item()) if sample_indices is not None else int(b)
            )
            chunk_idx = (
                int(chunk_indices[b].item()) if chunk_indices is not None else 0
            )

            # Collect interactions
            for idx in sorted_indices:
                i, j = i_sig[idx], j_sig[idx]
                interaction = {
                    'sample_idx': original_sample_idx,
                    'chunk_idx': chunk_idx,
                    'variant1_idx': int(i),
                    'variant2_idx': int(j),
                    'variant1_pos': int(pos[i]),
                    'variant2_pos': int(pos[j]),
                    'variant1_gene': int(genes[i]),
                    'variant2_gene': int(genes[j]),
                    'attention_score': float(scores_sig[idx]),
                    'attention_threshold_mode': self.threshold_mode,
                    'attention_threshold_value': threshold_value,
                    'attention_percentile': (
                        float(self.attention_percentile)
                        if self.threshold_mode == 'percentile'
                        else np.nan
                    ),
                    'same_gene': genes[i] == genes[j],
                    'distance': abs(int(pos[j]) - int(pos[i]))
                }
                all_interactions.append(interaction)

        return all_interactions

    def aggregate_interactions_across_samples(
        self,
        all_sample_interactions: List[List[Dict]],
        min_samples: int = 2
    ) -> List[Dict]:
        """
        Aggregate interactions across multiple samples.

        Identifies variant pairs that show strong attention in multiple samples,
        suggesting consistent epistatic interactions.

        Parameters
        ----------
        all_sample_interactions : List[List[Dict]]
            List of interaction lists, one per sample
        min_samples : int
            Minimum number of samples where interaction must appear

        Returns
        -------
        aggregated_interactions : List[Dict]
            Aggregated interactions with:
            - variant1_pos, variant2_pos, variant1_gene, variant2_gene
            - num_samples: number of samples with this interaction
            - mean_attention: mean attention score
            - max_attention: max attention score
        """
        # Collect all variant pairs
        pair_stats = {}

        for interactions in all_sample_interactions:
            for inter in interactions:
                # Create pair key (sorted to handle symmetry)
                pair = tuple(sorted([
                    (inter['variant1_pos'], inter['variant1_gene']),
                    (inter['variant2_pos'], inter['variant2_gene'])
                ]))

                if pair not in pair_stats:
                    pair_stats[pair] = {
                        'scores': [],
                        'sample_ids': set(),
                        'same_gene': inter['same_gene'],
                        'distance': inter['distance'],
                        'threshold_mode': inter.get('attention_threshold_mode', 'absolute'),
                    }

                pair_stats[pair]['scores'].append(inter['attention_score'])
                sample_id = inter.get('sample_idx')
                if sample_id is not None:
                    pair_stats[pair]['sample_ids'].add(int(sample_id))

        # Filter and aggregate
        aggregated = []
        for pair, stats in pair_stats.items():
            num_samples = len(stats['sample_ids']) if stats['sample_ids'] else len(stats['scores'])
            if num_samples >= min_samples:
                (pos1, gene1), (pos2, gene2) = pair
                aggregated.append({
                    'variant1_pos': pos1,
                    'variant1_gene': gene1,
                    'variant2_pos': pos2,
                    'variant2_gene': gene2,
                    'num_samples': num_samples,
                    'n_occurrences': len(stats['scores']),
                    'mean_attention': float(np.mean(stats['scores'])),
                    'max_attention': float(np.max(stats['scores'])),
                    'attention_threshold_mode': stats['threshold_mode'],
                    'same_gene': stats['same_gene'],
                    'distance': stats['distance']
                })

        # Sort by number of samples, then mean attention
        aggregated.sort(
            key=lambda x: (x['num_samples'], x['mean_attention']),
            reverse=True
        )

        return aggregated

    def compute_attention_entropy(
        self,
        attention_weights: Tensor,
        mask: Tensor
    ) -> np.ndarray:
        """
        Compute attention entropy for each variant.

        High entropy = attention spread across many variants
        Low entropy = attention focused on few variants

        Parameters
        ----------
        attention_weights : Tensor
            Attention from one layer, shape (batch, heads, variants, variants)
        mask : Tensor
            Validity mask, shape (batch, variants)

        Returns
        -------
        entropy : np.ndarray
            Entropy values, shape (batch, variants)
        """
        # Average across heads
        attn = attention_weights.mean(dim=1)  # (batch, variants, variants)

        # Compute entropy for each variant (query)
        entropy = []
        for b in range(attn.shape[0]):
            n_valid = mask[b].sum().item()
            attn_valid = attn[b, :n_valid, :n_valid].cpu().numpy()

            # Entropy = -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            attn_valid = attn_valid + eps
            sample_entropy = -np.sum(attn_valid * np.log(attn_valid), axis=1)
            entropy.append(sample_entropy)

        return np.array(entropy, dtype=object)
