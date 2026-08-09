"""
SIEVE: Sparse Interpretable Exome Variant Explainer

This module implements the complete SIEVE model that combines:
1. Variant encoding (feature projection)
2. Position-aware dense self-attention over the sparse variant set
3. Gene aggregation (variant → gene level)
4. Phenotype classification (case vs control)

Author: Francesco Lescai
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.encoding.position_config import PositionPreset, ResolvedPositionEncodingConfig

from .aggregation import EfficientGeneAggregator
from .attention import MultiLayerAttention
from .classifier import AttentionPoolingClassifier, PhenotypeClassifier
from .encoder import VariantEncoder
from .feature_composition import compose_legacy_variant_features_torch
from .position_runtime import (
    ObservedAbsolutePositionRuntime,
    build_absolute_position_runtime,
    validate_phase7_runtime_support,
)


class SIEVE(nn.Module):
    """
    Complete SIEVE model for variant discovery.

    This integrates all components into a single end-to-end model:
    1. Encode variant features to latent space
    2. Apply position-aware attention layers
    3. Aggregate variants to gene level
    4. Classify phenotype

    Parameters
    ----------
    input_dim : int
        Input feature dimension (depends on annotation level)
    num_genes : int
        Number of unique genes in dataset
    latent_dim : int
        Latent embedding dimension (default: 64)
    hidden_dim : int
        Hidden layer dimension for encoder (default: 128)
    num_heads : int
        Number of attention heads (default: 4)
    num_attention_layers : int
        Number of attention layers (default: 2)
    classifier_hidden_dim : int
        Hidden dimension for classifier (default: 256)
    dropout : float
        Dropout probability (default: 0.1)
    aggregation : str
        Gene aggregation method: 'max', 'mean', 'sum' (default: 'max')
    num_position_buckets : int
        Number of relative position buckets (default: 32)
    max_distance : int
        Maximum genomic distance for bucketing (default: 100,000 bp)

    Attributes
    ----------
    variant_encoder : VariantEncoder
        Encodes variant features to latent space
    attention : MultiLayerAttention
        Position-aware attention layers
    gene_aggregator : EfficientGeneAggregator
        Aggregates variants to genes
    classifier : PhenotypeClassifier
        Predicts phenotype from gene embeddings

    Examples
    --------
    >>> model = SIEVE(input_dim=71, num_genes=100, latent_dim=64)
    >>> features = torch.randn(2, 50, 71)  # [batch, variants, features]
    >>> positions = torch.randint(100, 10000, (2, 50))
    >>> gene_ids = torch.randint(0, 100, (2, 50))
    >>> mask = torch.ones(2, 50, dtype=torch.bool)
    >>> logits, attn_weights = model(features, positions, gene_ids, mask, return_attention=True)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        input_dim: int,
        num_genes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'max',
        num_position_buckets: int = 32,
        max_distance: int = 100000,
        num_covariates: int = 0,
        num_chromosomes: int = 0,
        classifier_type: str = 'flatten',
        position_encoding: ResolvedPositionEncodingConfig | None = None,
    ):
        super().__init__()

        if position_encoding is not None:
            if not isinstance(position_encoding, ResolvedPositionEncodingConfig):
                raise ValueError("position_encoding must be a ResolvedPositionEncodingConfig.")
            validate_phase7_runtime_support(position_encoding)
            if input_dim != position_encoding.input_dim:
                raise ValueError(
                    "input_dim must match position_encoding.input_dim when "
                    "position_encoding is supplied."
                )
            resolved_num_chromosomes = position_encoding.chromosome.num_chromosomes
            if num_chromosomes not in {0, resolved_num_chromosomes}:
                raise ValueError(
                    "num_chromosomes must be 0 or match "
                    "position_encoding.chromosome.num_chromosomes when position_encoding "
                    "is supplied."
                )
        else:
            resolved_num_chromosomes = num_chromosomes

        self.input_dim = input_dim
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.num_covariates = num_covariates
        self.num_chromosomes = resolved_num_chromosomes
        self.classifier_type = classifier_type
        self.position_encoding = position_encoding
        self._absolute_position_runtime = (
            ObservedAbsolutePositionRuntime()
            if position_encoding is None
            else build_absolute_position_runtime(position_encoding)
        )

        # 1. Variant encoder
        self.variant_encoder = VariantEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # 2. Position-aware attention
        self.attention = MultiLayerAttention(
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            dropout=dropout,
            num_position_buckets=num_position_buckets,
            max_distance=max_distance,
            num_chromosomes=self.num_chromosomes,
            position_encoding=position_encoding,
        )

        # 3. Gene aggregation
        self.gene_aggregator = EfficientGeneAggregator(
            num_genes=num_genes,
            latent_dim=latent_dim,
            aggregation=aggregation,
        )

        # 4. Phenotype classifier
        if classifier_type == 'flatten':
            self.classifier = PhenotypeClassifier(
                num_genes=num_genes,
                latent_dim=latent_dim,
                hidden_dim=classifier_hidden_dim,
                dropout=dropout,
                num_covariates=num_covariates,
            )
        elif classifier_type == 'attention_pool':
            self.classifier = AttentionPoolingClassifier(
                num_genes=num_genes,
                latent_dim=latent_dim,
                hidden_dim=classifier_hidden_dim,
                dropout=dropout,
                num_covariates=num_covariates,
            )
        else:
            raise ValueError(
                f"Unknown classifier_type '{classifier_type}'. "
                "Must be 'flatten' or 'attention_pool'."
            )

    def forward(
        self,
        variant_features: Tensor | None,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None,
        covariates: Optional[Tensor] = None,
        return_attention: bool = False,
        return_intermediate: bool = False,
        return_embeddings: bool = False,
        chrom_ids: Optional[Tensor] = None,
        *,
        content_features: Tensor | None = None,
        absolute_position_features: Tensor | None = None,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass through SIEVE model.

        Parameters
        ----------
        variant_features : Optional[Tensor]
            Historical variant features, shape (batch, num_variants, input_dim).
            Used unchanged when the split tensors are not supplied.
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)
            True = real variant, False = padding
        covariates : Optional[Tensor]
            Sample-level covariates, shape (batch, num_covariates).
            When provided, concatenated to gene embeddings in the classifier.
        return_attention : bool
            Whether to return attention weights (for explainability)
        return_intermediate : bool
            Whether to return intermediate representations
        return_embeddings : bool
            If True, return gene embeddings instead of logits (for chunked aggregation)
        chrom_ids : Optional[Tensor]
            Chromosome indices, shape (batch, num_variants). When provided
            (and the model was constructed with ``num_chromosomes > 0``),
            enables chromosome-aware position bias and chromosome embedding
            in attention. Cross-chromosome attention itself is **not** masked.
        content_features : Optional[Tensor]
            Split content features. When supplied together with
            ``absolute_position_features``, these are the primary legacy
            execution input and are composed into the unchanged historical
            VariantEncoder width.
        absolute_position_features : Optional[Tensor]
            Split historical absolute-position features. Must be supplied
            together with ``content_features``. For L0 this tensor has zero
            final-dimension width.

        Returns
        -------
        output : Tensor
            If return_embeddings=False: Phenotype prediction logits, shape (batch, 1)
            If return_embeddings=True: Gene embeddings, shape (batch, num_genes, latent_dim)
        intermediates : Optional[Dict]
            Dictionary containing intermediate outputs if requested:
            - 'variant_embeddings': After encoder
            - 'attended_embeddings': After attention
            - 'gene_embeddings': After aggregation
            - 'attention_weights': Attention weights from each layer (if return_attention=True)
        """
        intermediates = {} if (return_attention or return_intermediate or return_embeddings) else None

        # 1. Encode variants
        # Dataset batches now provide split tensors. Compose them immediately
        # before VariantEncoder so the split path is primary while old callers
        # and explanation paths can still use historical variant_features.
        # Historical checkpoints remain compatible because model state is unchanged.
        encoder_input = self._resolve_variant_encoder_input(
            variant_features,
            positions=positions,
            chrom_ids=chrom_ids,
            mask=mask,
            content_features=content_features,
            absolute_position_features=absolute_position_features,
        )
        variant_embeddings = self.variant_encoder(encoder_input)
        if return_intermediate or return_embeddings:
            intermediates['variant_embeddings'] = variant_embeddings

        # 2. Apply attention
        attended_embeddings, attention_weights = self.attention(
            variant_embeddings,
            positions,
            mask,
            return_attention=return_attention,
            chrom_ids=chrom_ids,
        )
        if return_intermediate or return_embeddings:
            intermediates['attended_embeddings'] = attended_embeddings
        if return_attention:
            intermediates['attention_weights'] = attention_weights

        # 3. Aggregate to genes
        gene_embeddings = self.gene_aggregator(
            attended_embeddings,
            gene_ids,
            mask
        )
        if return_intermediate or return_embeddings:
            intermediates['gene_embeddings'] = gene_embeddings

        # 4. Return embeddings or classify
        if return_embeddings:
            return gene_embeddings, intermediates
        else:
            logits = self.classifier(gene_embeddings, covariates=covariates)
            return logits, intermediates

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model architecture.

        Returns
        -------
        Dict[str, Any]
            Model configuration and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'input_dim': self.input_dim,
            'num_genes': self.num_genes,
            'latent_dim': self.latent_dim,
            'classifier_type': self.classifier_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_params': sum(p.numel() for p in self.variant_encoder.parameters()),
            'attention_params': sum(p.numel() for p in self.attention.parameters()),
            'aggregator_params': sum(p.numel() for p in self.gene_aggregator.parameters()),
            'classifier_params': sum(p.numel() for p in self.classifier.parameters()),
        }

    def get_attention_patterns(
        self,
        variant_features: Tensor | None,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None,
        chrom_ids: Optional[Tensor] = None,
        *,
        content_features: Tensor | None = None,
        absolute_position_features: Tensor | None = None,
    ) -> List[Tensor]:
        """
        Extract attention patterns for explainability.

        Parameters
        ----------
        variant_features : Optional[Tensor]
            Historical variant features, shape (batch, num_variants, input_dim).
            Used as the fallback when split tensors are absent.
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)
        chrom_ids : Optional[Tensor]
            Chromosome indices, shape (batch, num_variants). Enables
            chromosome-aware attention bias when the model was constructed
            with ``num_chromosomes > 0``.
        content_features, absolute_position_features : Optional[Tensor]
            Complete split feature pair. When both are supplied, attention is
            extracted after composing the unchanged historical VariantEncoder
            input width. Supplying only one split tensor raises ``ValueError``.

        Returns
        -------
        List[Tensor]
            Attention weights from each layer
            Each tensor: (batch, num_heads, num_variants, num_variants)
        """
        with torch.no_grad():
            _, intermediates = self.forward(
                variant_features,
                positions,
                gene_ids,
                mask,
                return_attention=True,
                chrom_ids=chrom_ids,
                content_features=content_features,
                absolute_position_features=absolute_position_features,
            )
            return intermediates['attention_weights']

    def _resolve_variant_encoder_input(
        self,
        variant_features: Tensor | None,
        *,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        content_features: Tensor | None,
        absolute_position_features: Tensor | None,
    ) -> Tensor:
        """Resolve the tensor that is fed to VariantEncoder."""
        has_content = content_features is not None
        has_position = absolute_position_features is not None

        if has_content != has_position:
            raise ValueError(
                "content_features and absolute_position_features must be supplied "
                "together."
            )

        if has_content and has_position:
            resolved_absolute_position_features = self._absolute_position_runtime.resolve(
                absolute_position_features,
                positions=positions,
                chrom_ids=chrom_ids,
                mask=mask,
                reference=content_features,
            )
            composed = compose_legacy_variant_features_torch(
                content_features,
                resolved_absolute_position_features,
            )
            if composed.shape[-1] != self.input_dim:
                raise ValueError(
                    "Composed legacy VariantEncoder input width does not match "
                    f"model input_dim: got {composed.shape[-1]}, expected "
                    f"{self.input_dim}"
            )
            return composed

        if (
            self.position_encoding is not None
            and self.position_encoding.preset is PositionPreset.CUSTOM
        ):
            raise ValueError(
                "custom positional execution requires content_features and "
                "absolute_position_features."
            )

        if variant_features is None:
            raise ValueError(
                "variant_features is required when content_features and "
                "absolute_position_features are not supplied."
            )

        return variant_features


def create_sieve_model(
    config: Dict,
    num_genes: int
) -> SIEVE:
    """
    Create SIEVE model from configuration dictionary.

    Parameters
    ----------
    config : Dict
        Configuration dictionary with model hyperparameters
    num_genes : int
        Number of unique genes in dataset

    Returns
    -------
    SIEVE
        Initialized SIEVE model

    Examples
    --------
    >>> config = {
    ...     'input_dim': 71,
    ...     'latent_dim': 64,
    ...     'num_heads': 4,
    ...     'num_attention_layers': 2,
    ... }
    >>> model = create_sieve_model(config, num_genes=100)
    """
    return SIEVE(
        input_dim=config.get('input_dim'),
        num_genes=num_genes,
        latent_dim=config.get('latent_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_attention_layers=config.get('num_attention_layers', 2),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 256),
        dropout=config.get('dropout', 0.1),
        aggregation=config.get('aggregation', 'max'),
        num_position_buckets=config.get('num_position_buckets', 32),
        max_distance=config.get('max_distance', 100000),
        num_covariates=config.get('num_covariates', 0),
        num_chromosomes=config.get('num_chromosomes', 0),
        classifier_type=config.get('classifier_type', 'flatten'),
    )


def load_state_dict_with_legacy_upgrade(
    model: nn.Module,
    state_dict: Dict[str, Tensor],
) -> None:
    """
    Load a checkpoint into ``model``, padding tensors that grew shape between
    versions and tolerating newly added parameters.

    Two model changes break ``strict=True`` for legacy checkpoints:
    ``position_bias.weight`` gained one row (cross-chromosome bucket) and
    ``chrom_embedding.weight`` is newly added. This helper copies overlapping
    rows from the checkpoint and leaves new entries at their fresh init
    (zero), matching option 2 in the migration plan.

    Parameters
    ----------
    model : nn.Module
        Target model; its ``state_dict()`` defines the destination shapes.
    state_dict : Dict[str, Tensor]
        Source checkpoint state dict. Mutated locally only.
    """
    current_state = model.state_dict()
    upgraded = dict(state_dict)
    for key, tensor in list(upgraded.items()):
        if key in current_state and current_state[key].shape != tensor.shape:
            target = current_state[key].clone()
            slices = tuple(slice(0, s) for s in tensor.shape)
            target[slices] = tensor
            upgraded[key] = target
    model.load_state_dict(upgraded, strict=False)
