"""
Chunked SIEVE model wrapper for whole-genome processing.

Wraps the base SIEVE model to handle chunked inputs and aggregate
chunk-level embeddings into sample-level predictions while preserving
interpretability for explainability analysis.

Key features:
- Aggregates gene embeddings across chunks (not logits)
- Preserves gene embeddings for integrated gradients
- Supports gene-level embedding sparsity regularisation
- Provides chunk-wise attention patterns for epistasis detection

Author: Francesco Lescai
"""

from typing import Dict, Tuple, Optional, List, Union
import torch
import torch.nn as nn


def _build_split_feature_kwargs(
    content_features: torch.Tensor | None,
    absolute_position_features: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Return split-feature kwargs only when a caller supplied a split tensor."""
    if content_features is None and absolute_position_features is None:
        return {}
    return {
        'content_features': content_features,
        'absolute_position_features': absolute_position_features,
    }


def build_sample_covariates(
    batch_sex: Optional[torch.Tensor],
    num_covariates: int,
    num_samples: int,
    device: torch.device,
    batch_covariates: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Build a sample-level covariate tensor from batch sex values.

    Sex occupies column 0.  Any remaining covariate columns (columns 1+)
    are left as zero here; callers that support additional covariates (e.g.
    ancestry PCs) should fill those columns after calling this helper.

    Parameters
    ----------
    batch_sex : Optional[torch.Tensor]
        Sex tensor, shape ``[num_samples]``, already on ``device``.
        If ``None`` or ``num_covariates == 0``, returns ``None``.
    num_covariates : int
        Total number of covariates expected by the model.
    num_samples : int
        Number of samples in this batch.
    device : torch.device
        Target device for the output tensor.
    batch_covariates : Optional[torch.Tensor]
        Pre-built covariate tensor with shape ``[num_samples, num_covariates]``.
        When supplied, it is validated and returned directly.

    Returns
    -------
    Optional[torch.Tensor]
        ``[num_samples, num_covariates]`` float tensor, or ``None``.
    """
    if num_covariates == 0:
        if batch_covariates is not None:
            raise ValueError("Received covariates but num_covariates=0.")
        return None

    if batch_covariates is not None:
        batch_covariates = batch_covariates.to(device)
        if batch_covariates.dim() != 2:
            raise ValueError(
                f"batch_covariates must be 2D, got shape {tuple(batch_covariates.shape)}"
            )
        if batch_covariates.shape[0] != num_samples:
            raise ValueError(
                "batch_covariates first dimension must match num_samples "
                f"({batch_covariates.shape[0]} vs {num_samples})"
            )
        if batch_covariates.shape[1] != num_covariates:
            raise ValueError(
                "batch_covariates second dimension must match num_covariates "
                f"({batch_covariates.shape[1]} vs {num_covariates})"
            )
        return batch_covariates

    if batch_sex is None:
        return None

    sample_covariates = torch.zeros(
        num_samples, num_covariates,
        device=device, dtype=batch_sex.dtype,
    )
    sample_covariates[:, 0] = batch_sex
    return sample_covariates


class ChunkedSIEVEModel(nn.Module):
    """
    Wrapper around SIEVE model to handle chunked variant processing.

    Processes each chunk through the base SIEVE model to get gene embeddings,
    aggregates embeddings across chunks, then applies classification.
    This preserves interpretability for explainability analysis.

    Parameters
    ----------
    base_model : nn.Module
        The base SIEVE model
    aggregation_method : str
        How to aggregate gene embeddings across chunks:
        - 'mean': Average gene embeddings (default)
        - 'max': Element-wise max of gene embeddings
    embedding_dim : Optional[int]
        Unused; retained for API compatibility

    Key Methods
    -----------
    forward() : Returns logits and intermediates (including gene embeddings)
    get_gene_embeddings() : Extract aggregated gene embeddings for explainability
    get_attention_patterns() : Extract chunk-wise attention patterns
    train_step() : Training with support for embedding sparsity regularisation

    Examples
    --------
    >>> base_model = create_sieve_model(config, num_genes=1000)
    >>> chunked_model = ChunkedSIEVEModel(base_model, aggregation_method='mean')
    >>> # Training processes chunks, aggregates embeddings, preserves interpretability
    """

    def __init__(
        self,
        base_model: nn.Module,
        aggregation_method: str = 'mean',
        embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.aggregation_method = aggregation_method

        if aggregation_method in ('attention', 'logit_mean'):
            raise NotImplementedError(
                f"aggregation_method='{aggregation_method}' is not implemented. "
                "Only 'mean' and 'max' are supported. "
                "'logit_mean' was a provisional alias for 'mean' and has been removed. "
                "'attention' (learned chunk-level attention) has not been implemented."
            )

    def forward(
        self,
        features: torch.Tensor | None,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_intermediate: bool = False,
        chrom_ids: Optional[torch.Tensor] = None,
        *,
        content_features: torch.Tensor | None = None,
        absolute_position_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with automatic chunk aggregation.

        If chunk metadata is provided, aggregates gene embeddings across chunks
        before classification. This preserves interpretability for explainability.

        Parameters
        ----------
        features : Optional[torch.Tensor]
            Historical features [batch_size, max_variants, feature_dim]. Used
            as a compatibility fallback when split tensors are absent.
        positions : torch.Tensor
            [batch_size, max_variants]
        gene_ids : torch.Tensor
            [batch_size, max_variants]
        mask : torch.Tensor
            [batch_size, max_variants]
        chunk_indices : Optional[torch.Tensor]
            [batch_size] - which chunk within the sample
        total_chunks : Optional[torch.Tensor]
            [batch_size] - total chunks for each sample
        original_sample_indices : Optional[torch.Tensor]
            [batch_size] - which original sample this chunk belongs to
        covariates : Optional[torch.Tensor]
            Sample-level covariates [num_samples, num_covariates].
            Already aggregated to sample level (not per-chunk).
        return_attention : bool
            If True, collect attention weights from chunks
        return_intermediate : bool
            If True, return intermediate embeddings
        content_features, absolute_position_features : Optional[torch.Tensor]
            Complete split feature pair. When both are supplied, the base model
            composes them into the unchanged historical VariantEncoder input
            width. Supplying only one split tensor raises ``ValueError`` in the
            base model.

        Returns
        -------
        logits : torch.Tensor
            Sample-level predictions [num_samples] or [num_samples, 1]
        intermediates : Optional[Dict]
            Dictionary containing:
            - 'gene_embeddings': Aggregated gene embeddings
            - 'attention_weights': List of attention weights per chunk (if return_attention=True)
            - 'chunk_metadata': Mapping from chunks to samples
        """
        # If no chunk metadata, process as regular batch
        # NOTE: Delegates directly to base_model. The base model should be on the
        # same device as ChunkedSIEVEModel to ensure output tensors match input device.
        if original_sample_indices is None:
            # Only pass covariates if the base model supports them
            kwargs = dict(
                return_attention=return_attention,
                return_intermediate=return_intermediate,
            )
            if hasattr(self.base_model, 'num_covariates') and self.base_model.num_covariates > 0:
                kwargs['covariates'] = covariates
            if chrom_ids is not None:
                kwargs['chrom_ids'] = chrom_ids
            kwargs.update(
                _build_split_feature_kwargs(
                    content_features,
                    absolute_position_features,
                )
            )
            return self.base_model(
                features, positions, gene_ids, mask,
                **kwargs,
            )

        # Process all chunks through base model to get gene embeddings
        base_kwargs = dict(
            return_embeddings=True,  # Get gene embeddings, not logits
            return_attention=return_attention,
            return_intermediate=return_intermediate,
        )
        if chrom_ids is not None:
            base_kwargs['chrom_ids'] = chrom_ids
        base_kwargs.update(
            _build_split_feature_kwargs(
                content_features,
                absolute_position_features,
            )
        )
        chunk_gene_embeddings, chunk_intermediates = self.base_model(
            features, positions, gene_ids, mask,
            **base_kwargs,
        )
        # chunk_gene_embeddings: [num_chunks, num_genes, latent_dim]
        device = chunk_gene_embeddings.device

        # Get unique samples and map chunks to samples
        unique_samples = original_sample_indices.unique(sorted=True)
        num_samples = len(unique_samples)
        sample_mapping = torch.searchsorted(unique_samples, original_sample_indices)

        # Get dimensions
        num_genes = chunk_gene_embeddings.shape[1]
        latent_dim = chunk_gene_embeddings.shape[2]

        # Aggregate gene embeddings across chunks
        if self.aggregation_method == 'mean':
            # Average gene embeddings across chunks per sample
            # For each gene, average the embeddings from chunks containing that gene

            # Initialize aggregated embeddings
            aggregated_embeddings = torch.zeros(
                num_samples, num_genes, latent_dim,
                dtype=chunk_gene_embeddings.dtype,
                device=device
            )

            # Count how many chunks contribute to each gene per sample
            counts = torch.zeros(
                num_samples, num_genes,
                dtype=torch.float32,
                device=device
            )

            # Vectorized accumulation of embeddings per sample
            # Sum embeddings from all chunks into their corresponding samples
            aggregated_embeddings.index_add_(0, sample_mapping, chunk_gene_embeddings)

            # Compute per-chunk gene presence mask: genes with non-zero embeddings
            # Use L2 norm for robustness to sign cancellations
            gene_has_variants = (chunk_gene_embeddings.pow(2).sum(dim=-1) > 1e-9).float()

            # Accumulate counts of contributing chunks per gene and sample
            counts.index_add_(0, sample_mapping, gene_has_variants)

            # Average, explicitly avoiding division by zero:
            # Only divide where at least one chunk contributed to a gene,
            # and leave zero embeddings unchanged when there are no variants.
            counts_expanded = counts.unsqueeze(-1)
            nonzero_mask = counts_expanded > 0
            safe_counts_expanded = torch.where(
                nonzero_mask,
                counts_expanded,
                torch.ones_like(counts_expanded)
            )
            aggregated_embeddings = torch.where(
                nonzero_mask,
                aggregated_embeddings / safe_counts_expanded,
                aggregated_embeddings
            )

        elif self.aggregation_method == 'max':
            # Max-pool gene embeddings across chunks per sample
            aggregated_embeddings = torch.zeros(
                num_samples, num_genes, latent_dim,
                dtype=chunk_gene_embeddings.dtype,
                device=device
            )

            # Vectorized element-wise max across chunks per sample using scatter_reduce
            if chunk_gene_embeddings.numel() > 0:
                # sample_mapping: [num_chunks] -> expand to match chunk_gene_embeddings
                index = sample_mapping.view(-1, 1, 1).expand(-1, num_genes, latent_dim)
                aggregated_embeddings.scatter_reduce_(
                    dim=0,
                    index=index,
                    src=chunk_gene_embeddings,
                    reduce='amax',
                    include_self=True  # Keep initial zeros, matching previous behaviour
                )

        elif self.aggregation_method == 'attention':
            raise NotImplementedError(
                "aggregation_method='attention' is not implemented. "
                "Only 'mean' and 'max' are supported."
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Apply classifier on aggregated embeddings
        # NOTE: This intentionally bypasses base_model.forward() and assumes
        # that the base model exposes a callable 'classifier' attribute that
        # can operate directly on aggregated gene embeddings of shape
        # [num_samples, num_genes, latent_dim].
        # If using a different base model, it must conform to this interface.
        classifier = self.base_model.classifier
        if hasattr(classifier, 'num_covariates') and classifier.num_covariates > 0:
            logits = classifier(aggregated_embeddings, covariates=covariates)
        else:
            logits = classifier(aggregated_embeddings)

        # Prepare intermediates if requested
        intermediates = None
        if return_intermediate or return_attention:
            intermediates = {
                'gene_embeddings': aggregated_embeddings,
                'chunk_metadata': {
                    'sample_mapping': sample_mapping,
                    'unique_samples': unique_samples
                }
            }
            if return_attention and chunk_intermediates is not None:
                intermediates['attention_weights'] = chunk_intermediates.get('attention_weights', [])

        return logits, intermediates

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[Union[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Training step that handles chunk aggregation.

        Supports embedding sparsity regularisation (lambda_attr > 0) by computing
        gene-level sparsity on aggregated embeddings.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from ChunkedVariantDataset
        criterion : nn.Module
            Loss function. Supported types:
            - SIEVELoss: Returns dict {'total': scalar, ...} when lambda_attr > 0
            - BCEWithLogitsLoss: Returns scalar tensor
            Note: For embedding sparsity regularisation support, loss function must have
            a 'lambda_attr' attribute that can be checked via hasattr().
            Custom loss functions should follow this interface convention.
        device : torch.device
            Device to use

        Returns
        -------
        loss_output : Union[Dict[str, torch.Tensor], torch.Tensor]
            If criterion returns a dict (SIEVELoss): dict with keys
            'total', 'classification', 'attribution_sparsity'.
            If criterion returns a scalar (BCEWithLogitsLoss): scalar tensor.
            The trainer is responsible for extracting the scalar for backprop.
        predictions : torch.Tensor
            Sample-level predictions [num_samples] or [num_samples, 1]

        Notes
        -----
        Attribution regularisation uses gene-level sparsity (not variant-level)
        since chunked processing operates on aggregated gene embeddings.
        This encourages the model to rely on fewer genes rather than fewer variants.
        """
        # Move to device
        features = batch.get('features')
        if features is not None:
            features = features.to(device)
        content_features = batch.get('content_features')
        absolute_position_features = batch.get('absolute_position_features')
        if content_features is not None:
            content_features = content_features.to(device)
        if absolute_position_features is not None:
            absolute_position_features = absolute_position_features.to(device)
        positions = batch['positions'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        chrom_ids = batch.get('chrom_ids')
        if chrom_ids is not None:
            chrom_ids = chrom_ids.to(device)

        chunk_indices = batch.get('chunk_indices')
        total_chunks = batch.get('total_chunks')
        original_sample_indices = batch.get('original_sample_indices')
        batch_sex = batch.get('sex')
        batch_covariates = batch.get('covariates')

        # Build sample-level covariates from sex if the base model uses them
        sample_covariates = None

        if chunk_indices is not None:
            chunk_indices = chunk_indices.to(device)
            total_chunks = total_chunks.to(device)
            original_sample_indices = original_sample_indices.to(device)

            # Aggregate chunk labels to sample labels (vectorized)
            # All chunks from same sample have same label.
            # Get unique samples in sorted order (matching forward method)
            unique_samples = original_sample_indices.unique(sorted=True)

            # For each unique sample, find its first occurrence in original_sample_indices
            # and extract the label at that position
            sample_labels = torch.zeros(len(unique_samples), dtype=labels.dtype, device=device)
            first_chunk_indices = torch.zeros(len(unique_samples), dtype=torch.long, device=device)
            for i, sample_idx in enumerate(unique_samples):
                # Find first chunk belonging to this sample
                first_chunk_idx = (original_sample_indices == sample_idx).nonzero(as_tuple=True)[0][0]
                sample_labels[i] = labels[first_chunk_idx]
                first_chunk_indices[i] = first_chunk_idx

            # Aggregate sex to sample level (same value for all chunks of a sample)
            num_covariates = getattr(self.base_model, 'num_covariates', 0)
            if num_covariates > 0:
                sample_sex = None
                if batch_sex is not None:
                    batch_sex = batch_sex.to(device)
                    sample_sex = batch_sex[first_chunk_indices]
                sample_batch_covariates = None
                if batch_covariates is not None:
                    sample_batch_covariates = batch_covariates.to(device)[first_chunk_indices]
                sample_covariates = build_sample_covariates(
                    sample_sex, num_covariates, len(unique_samples), device,
                    batch_covariates=sample_batch_covariates,
                )
        else:
            sample_labels = labels
            num_covariates = getattr(self.base_model, 'num_covariates', 0)
            if num_covariates > 0:
                batch_sex_dev = batch_sex.to(device) if batch_sex is not None else None
                sample_covariates = build_sample_covariates(
                    batch_sex_dev, num_covariates, labels.shape[0], device,
                    batch_covariates=batch_covariates,
                )

        # Forward pass (aggregates chunks automatically)
        # Get intermediates for embedding sparsity regularisation if needed
        need_embeddings = hasattr(criterion, 'lambda_attr') and criterion.lambda_attr > 0

        predictions, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            covariates=sample_covariates,
            return_intermediate=need_embeddings,
            chrom_ids=chrom_ids,
            **_build_split_feature_kwargs(
                content_features,
                absolute_position_features,
            ),
        )
        # Ensure 1D tensor for loss computation
        if predictions.dim() > 1:
            predictions = predictions.view(-1)

        # Compute loss at sample level
        if need_embeddings and intermediates is not None:
            # Pass gene embeddings for embedding sparsity regularisation
            loss_output = criterion(
                predictions, sample_labels.float(),
                gene_embeddings=intermediates['gene_embeddings']
            )
        else:
            # Standard classification loss only
            loss_output = criterion(predictions, sample_labels.float())

        # Return the full loss_output (dict or scalar) so the trainer
        # can log the decomposition (classification vs attribution).
        # For dict (SIEVELoss): contains 'total', 'classification', 'attribution_sparsity'
        # For scalar (BCEWithLogitsLoss): plain tensor
        return loss_output, predictions

    def get_gene_embeddings(
        self,
        features: torch.Tensor | None,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
        chrom_ids: Optional[torch.Tensor] = None,
        *,
        content_features: torch.Tensor | None = None,
        absolute_position_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get aggregated gene embeddings for explainability.

        Parameters
        ----------
        features, positions, gene_ids, mask : torch.Tensor
            Variant data. ``features`` is the historical fallback and may be
            ``None`` when the complete split pair is supplied.
        chunk_indices, total_chunks, original_sample_indices : Optional[torch.Tensor]
            Chunking metadata
        chrom_ids : Optional[torch.Tensor]
            Chromosome indices, shape (batch, num_variants).
        content_features, absolute_position_features : Optional[torch.Tensor]
            Complete split feature pair for split-primary legacy execution.

        Returns
        -------
        torch.Tensor
            Aggregated gene embeddings [num_samples, num_genes, latent_dim]
        """
        _, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            return_intermediate=True,
            chrom_ids=chrom_ids,
            **_build_split_feature_kwargs(
                content_features,
                absolute_position_features,
            ),
        )

        if intermediates is None:
            raise RuntimeError(
                "ChunkedSIEVEModel.forward did not return intermediates "
                "despite return_intermediate=True when calling get_gene_embeddings."
            )

        if 'gene_embeddings' not in intermediates:
            raise KeyError(
                "Intermediates returned by ChunkedSIEVEModel.forward do not contain "
                "'gene_embeddings'. Ensure the base model is configured to produce "
                "gene embeddings when return_intermediate=True."
            )

        return intermediates['gene_embeddings']

    def get_attention_patterns(
        self,
        features: torch.Tensor | None,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
        chrom_ids: Optional[torch.Tensor] = None,
        *,
        content_features: torch.Tensor | None = None,
        absolute_position_features: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """
        Get attention patterns for explainability.

        NOTE: For chunked data, returns attention weights from each chunk.
        These are within-chunk attention patterns, not full sample attention.

        Parameters
        ----------
        features, positions, gene_ids, mask : torch.Tensor
            Variant data. ``features`` is the historical fallback and may be
            ``None`` when the complete split pair is supplied.
        chunk_indices, total_chunks, original_sample_indices : Optional[torch.Tensor]
            Chunking metadata
        chrom_ids : Optional[torch.Tensor]
            Chromosome indices, shape (batch, num_variants).
        content_features, absolute_position_features : Optional[torch.Tensor]
            Complete split feature pair for split-primary legacy execution.

        Returns
        -------
        List[torch.Tensor]
            Attention weights per chunk. Returns empty list if:
            - Base model does not support attention (intermediates is None)
            - Base model did not return attention weights
            - return_attention=True was not honored by base model
        """
        _, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            return_attention=True,
            chrom_ids=chrom_ids,
            **_build_split_feature_kwargs(
                content_features,
                absolute_position_features,
            ),
        )

        if intermediates is None or 'attention_weights' not in intermediates:
            return []

        return intermediates['attention_weights']
