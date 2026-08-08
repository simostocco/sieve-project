"""
Integrated Gradients for variant attribution.

Uses Captum's IntegratedGradients to compute variant-level importance scores.
This allows us to identify which variants most strongly influence the model's
predictions for each sample.

Author: Francesco Lescai
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from captum.attr import IntegratedGradients

from src.encoding.position_config import ResolvedIGMode


class IntegratedGradientsExplainer:
    """
    Compute variant attributions using Integrated Gradients.

    This class wraps Captum's IntegratedGradients to work with SIEVE's
    multi-input architecture. By default it preserves historical legacy
    behavior, where the complete ``variant_features`` tensor is differentiable.
    Content mode instead makes only ``content_features`` differentiable while
    observed absolute-position features and all IDs/masks/covariates stay fixed.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model
    device : str
        Device to run computations ('cuda' or 'cpu')
    n_steps : int
        Number of integration steps (default: 50)
    ig_mode : ResolvedIGMode or str
        Resolved attribution mode. ``legacy`` preserves the historical Python
        API. ``content`` uses SIEVE's split-primary model path. ``auto`` must be
        resolved from configuration before constructing this explainer.

    Attributes
    ----------
    model : nn.Module
        The SIEVE model
    device : str
        Computation device
    ig : IntegratedGradients
        Captum's IntegratedGradients instance

    Examples
    --------
    >>> explainer = IntegratedGradientsExplainer(model, device='cuda')
    >>> attributions = explainer.attribute(features, positions, gene_ids, mask)
    >>> # legacy attributions shape: (batch, num_variants, input_dim)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        n_steps: int = 50,
        max_variants: int = 2000,
        ig_mode: ResolvedIGMode | str = ResolvedIGMode.LEGACY,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_steps = n_steps
        self.max_variants = max_variants
        self.ig_mode = _coerce_resolved_ig_mode(ig_mode)

        if self.ig_mode is ResolvedIGMode.LEGACY:
            self.model_wrapper = SIEVEWrapper(model)
        else:
            self.model_wrapper = ContentSIEVEWrapper(model)

        # Create IntegratedGradients instance
        self.ig = IntegratedGradients(self.model_wrapper)

    def attribute(
        self,
        variant_features: Tensor | None,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        target: Optional[int] = None,
        baseline: Optional[Tensor] = None,
        covariates: Optional[Tensor] = None,
        chrom_ids: Optional[Tensor] = None,
        *,
        content_features: Tensor | None = None,
        absolute_position_features: Tensor | None = None,
    ) -> Tensor:
        """
        Compute integrated gradients attributions for variants.

        Legacy mode attributes the complete historical ``variant_features``
        tensor. Content mode requires ``variant_features=None`` and attributes
        only ``content_features`` while observed absolute-position features are
        fixed throughout integration.

        Parameters
        ----------
        variant_features : Optional[Tensor]
            Variant features, shape (batch, num_variants, input_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Tensor
            Validity mask, shape (batch, num_variants)
        target : Optional[int]
            Target class (0 or 1). If None, uses predicted class
        baseline : Optional[Tensor]
            Baseline input for integration. In legacy mode, shape must equal
            ``variant_features.shape`` and the default is
            ``zeros_like(variant_features)``. In content mode, shape must equal
            ``content_features.shape`` and the default is
            ``zeros_like(content_features)``. ``absolute_position_features`` is
            not part of the baseline and remains at its observed fixed value
            throughout integration.
        covariates : Optional[Tensor]
            Sample-level covariates, shape (batch, num_covariates).
            Must be provided when the model was trained with covariates
            (``num_covariates > 0``).  Omitting covariates for a model that
            expects them will explain a different function than was trained.
        content_features : Optional[Tensor]
            Split content features, shape (batch, num_variants, content_dim).
            Required in content mode and forbidden in legacy mode.
        absolute_position_features : Optional[Tensor]
            Observed split absolute-position features. Required in content mode
            and fixed as a non-Captum forward argument.

        Returns
        -------
        attributions : Tensor
            Legacy mode shape is (batch, num_variants, input_dim). Content mode
            shape is (batch, num_variants, content_dim).
        """
        positions = positions.to(self.device)
        gene_ids = gene_ids.to(self.device)
        mask = mask.to(self.device)
        if covariates is not None:
            covariates = covariates.to(self.device)
        if chrom_ids is not None:
            chrom_ids = chrom_ids.to(self.device)

        if self.ig_mode is ResolvedIGMode.LEGACY:
            if variant_features is None:
                raise ValueError("variant_features is required in legacy IG mode")
            if content_features is not None or absolute_position_features is not None:
                raise ValueError(
                    "content_features and absolute_position_features are only valid "
                    "in content IG mode"
                )
            variant_features = variant_features.to(self.device)
            baseline = _prepare_baseline(baseline, variant_features, self.device)
            additional = (positions, gene_ids, mask, covariates, chrom_ids)
            return self.ig.attribute(
                inputs=variant_features,
                baselines=baseline,
                target=target,
                additional_forward_args=additional,
                n_steps=self.n_steps
            )

        if variant_features is not None:
            raise ValueError("variant_features must be None in content IG mode")
        if content_features is None and absolute_position_features is None:
            raise ValueError(
                "content_features and absolute_position_features are required "
                "in content IG mode"
            )
        if content_features is None or absolute_position_features is None:
            raise ValueError(
                "content_features and absolute_position_features must be supplied "
                "together in content IG mode"
            )

        content_features = content_features.to(self.device)
        absolute_position_features = absolute_position_features.to(self.device)
        baseline = _prepare_baseline(baseline, content_features, self.device)

        # Absolute position remains part of the observed function, but not the
        # attribution input. Detaching enforces that boundary even if a caller
        # supplies a tensor that requires gradients.
        fixed_absolute_position = absolute_position_features.detach()
        additional = (
            fixed_absolute_position,
            positions,
            gene_ids,
            mask,
            covariates,
            chrom_ids,
        )
        return self.ig.attribute(
            inputs=content_features,
            baselines=baseline,
            target=target,
            additional_forward_args=additional,
            n_steps=self.n_steps
        )

    def attribute_batch(
        self,
        dataloader,
        aggregate: str = 'l2',
        num_covariates: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Compute attributions for a full dataset.

        IMPORTANT: Processes samples one at a time to avoid OOM errors.
        Integrated gradients requires storing all intermediate activations
        for gradient computation. With attention mechanisms over thousands
        of variants, batch processing exceeds GPU memory.

        Legacy mode reads ``batch['features']`` and preserves historical
        attribution widths. Content mode reads ``batch['content_features']`` and
        ``batch['absolute_position_features']``; historical ``features`` is not
        required for IG execution.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding batches of samples
        aggregate : str
            How to aggregate feature attributions to variant-level scores:
            - 'l2': L2 norm across features (default)
            - 'l1': L1 norm across features
            - 'sum': Sum across features
            - 'mean': Mean across features
        num_covariates : int
            Number of covariates expected by the model (default 0).
            When > 0, a 'sex' tensor must be present in each batch.

        Returns
        -------
        all_attributions : List[np.ndarray]
            List of attribution arrays, one per sample. In legacy mode, each
            attribution matrix has width ``input_dim``. In content mode, each
            attribution matrix has width ``content_dim``.
        all_variant_scores : List[np.ndarray]
            List of aggregated variant scores, one per sample
        all_metadata : List[Dict]
            List of metadata dicts containing positions, genes, etc.
        """
        all_attributions = []
        all_variant_scores = []
        all_metadata = []

        self.model.eval()
        # NOTE: Do NOT use torch.no_grad() - we need gradients for integrated gradients!

        # Calculate total samples for progress tracking
        total_samples = len(dataloader.dataset)
        print(f"Processing {total_samples} samples individually (required for memory efficiency)...")

        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            if self.ig_mode is ResolvedIGMode.LEGACY:
                features = batch['features']
                content_features = None
                absolute_position_features = None
                batch_size = features.shape[0]
            else:
                if 'content_features' not in batch or 'absolute_position_features' not in batch:
                    raise ValueError(
                        "content IG mode requires batch['content_features'] and "
                        "batch['absolute_position_features']"
                    )
                features = None
                content_features = batch['content_features']
                absolute_position_features = batch['absolute_position_features']
                batch_size = content_features.shape[0]
            positions = batch['positions']
            gene_ids = batch['gene_ids']
            mask = batch['mask']
            chrom_ids_batch = batch.get('chrom_ids')

            # --- Covariate handling ---
            # Build the per-sample covariate vector using the same logic as
            # ChunkedSIEVEModel.train_step (via build_sample_covariates).
            batch_sex = batch.get('sex')
            batch_covariates = batch.get('covariates')
            if num_covariates > 0:
                if batch_sex is None and batch_covariates is None:
                    raise ValueError(
                        f"Model has num_covariates={num_covariates} but the batch "
                        "contains no covariate tensor. Cannot build covariate vector."
                    )
                # Move covariate tensors to the model device before building
                target_device = torch.device(self.device)
                if batch_sex is not None:
                    batch_sex = batch_sex.to(target_device)
                if batch_covariates is not None:
                    batch_covariates = batch_covariates.to(target_device)
                # Import here to avoid circular dependency at module level
                from src.models.chunked_sieve import build_sample_covariates
                batch_covariates_full = build_sample_covariates(
                    batch_sex, num_covariates, batch_size,
                    target_device,
                    batch_covariates=batch_covariates,
                )
            elif (batch_sex is not None or batch_covariates is not None) and num_covariates == 0:
                raise ValueError(
                    "A covariate tensor is present in the batch but num_covariates=0. "
                    "Either set num_covariates to the correct value or remove the "
                    "covariates from the batch."
                )
            else:
                batch_covariates_full = None

            # CRITICAL: Process each sample individually to avoid OOM
            # Integrated gradients requires storing all intermediate activations,
            # which for attention mechanisms with many variants becomes huge
            for i in range(batch_size):
                # Progress update
                sample_num = len(all_metadata) + 1
                if sample_num % 10 == 0 or sample_num == total_samples:
                    print(f"  Processed {sample_num}/{total_samples} samples...", flush=True)

                # Extract single sample (keep batch dimension)
                sample_features = (
                    features[i:i+1] if self.ig_mode is ResolvedIGMode.LEGACY else None
                )
                sample_content = (
                    content_features[i:i+1]
                    if self.ig_mode is ResolvedIGMode.CONTENT else None
                )
                sample_absolute_position = (
                    absolute_position_features[i:i+1]
                    if self.ig_mode is ResolvedIGMode.CONTENT else None
                )
                sample_positions = positions[i:i+1]
                sample_gene_ids = gene_ids[i:i+1]
                sample_mask = mask[i:i+1]
                sample_chrom_ids = (
                    chrom_ids_batch[i:i+1] if chrom_ids_batch is not None else None
                )

                # Per-sample covariate slice
                sample_covariates = (
                    batch_covariates_full[i:i+1] if batch_covariates_full is not None else None
                )

                # CRITICAL: Limit variants to avoid OOM
                # Count valid variants for this sample
                num_valid_variants = sample_mask[0].sum().item()

                if num_valid_variants > self.max_variants:
                    # Too many variants - need to subsample
                    valid_indices = torch.where(sample_mask[0])[0]

                    # Random sampling of variant indices
                    selected_indices = valid_indices[torch.randperm(len(valid_indices))[:self.max_variants]]
                    selected_indices = selected_indices.sort()[0]  # Keep sorted for locality

                    # Truncate to selected variants
                    sample_features_truncated = (
                        sample_features[:, selected_indices, :]
                        if sample_features is not None else None
                    )
                    sample_content_truncated = (
                        sample_content[:, selected_indices, :]
                        if sample_content is not None else None
                    )
                    sample_absolute_position_truncated = (
                        sample_absolute_position[:, selected_indices, :]
                        if sample_absolute_position is not None else None
                    )
                    sample_positions_truncated = sample_positions[:, selected_indices]
                    sample_gene_ids_truncated = sample_gene_ids[:, selected_indices]
                    sample_mask_truncated = sample_mask[:, selected_indices]
                    sample_chrom_ids_truncated = (
                        sample_chrom_ids[:, selected_indices]
                        if sample_chrom_ids is not None else None
                    )

                    # Track original indices for metadata
                    original_indices = selected_indices
                else:
                    # Use all variants
                    sample_features_truncated = sample_features
                    sample_content_truncated = sample_content
                    sample_absolute_position_truncated = sample_absolute_position
                    sample_positions_truncated = sample_positions
                    sample_gene_ids_truncated = sample_gene_ids
                    sample_mask_truncated = sample_mask
                    sample_chrom_ids_truncated = sample_chrom_ids
                    original_indices = None

                # Compute attributions for this single sample (possibly truncated)
                sample_attributions = self.attribute(
                    sample_features_truncated, sample_positions_truncated,
                    sample_gene_ids_truncated, sample_mask_truncated,
                    covariates=sample_covariates,
                    chrom_ids=sample_chrom_ids_truncated,
                    content_features=sample_content_truncated,
                    absolute_position_features=sample_absolute_position_truncated,
                )

                # Convert to numpy
                attributions_np = sample_attributions[0].cpu().numpy()
                mask_np_truncated = sample_mask_truncated[0].cpu().numpy()

                # Aggregate feature attributions to variant scores
                if aggregate == 'l2':
                    variant_scores = np.linalg.norm(attributions_np, ord=2, axis=1)
                elif aggregate == 'l1':
                    variant_scores = np.linalg.norm(attributions_np, ord=1, axis=1)
                elif aggregate == 'sum':
                    variant_scores = np.sum(attributions_np, axis=1)
                elif aggregate == 'mean':
                    variant_scores = np.mean(attributions_np, axis=1)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregate}")

                # Get valid variants only
                valid_mask = mask_np_truncated

                all_attributions.append(attributions_np[valid_mask])
                all_variant_scores.append(variant_scores[valid_mask])

                # Store metadata (use truncated positions/genes if applicable)
                if original_indices is not None:
                    # Was truncated - use the selected subset
                    metadata = {
                        'positions': sample_positions_truncated[0][sample_mask_truncated[0]].cpu().numpy(),
                        'gene_ids': sample_gene_ids_truncated[0][sample_mask_truncated[0]].cpu().numpy(),
                        'sample_idx': len(all_metadata),
                        'num_variants_original': num_valid_variants,
                        'num_variants_analyzed': self.max_variants,
                        'truncated': True,
                    }
                else:
                    # Not truncated - use all
                    metadata = {
                        'positions': positions[i][mask[i]].cpu().numpy(),
                        'gene_ids': gene_ids[i][mask[i]].cpu().numpy(),
                        'sample_idx': len(all_metadata),
                        'num_variants_original': num_valid_variants,
                        'num_variants_analyzed': num_valid_variants,
                        'truncated': False,
                    }
                # Fix: use sample_ids (plural) not sample_id
                if 'sample_ids' in batch:
                    metadata['sample_id'] = batch['sample_ids'][i]
                if 'labels' in batch:
                    metadata['label'] = batch['labels'][i].cpu().item()

                all_metadata.append(metadata)

                # Clear GPU cache after each sample
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        return all_attributions, all_variant_scores, all_metadata

    def get_top_variants(
        self,
        variant_scores: np.ndarray,
        metadata: Dict,
        top_k: int = 100
    ) -> List[Tuple[int, int, float]]:
        """
        Get top K variants by attribution score.

        Parameters
        ----------
        variant_scores : np.ndarray
            Variant attribution scores, shape (num_variants,)
        metadata : Dict
            Metadata dict with 'positions' and 'gene_ids'
        top_k : int
            Number of top variants to return

        Returns
        -------
        top_variants : List[Tuple[int, int, float]]
            List of (position, gene_id, score) tuples for top variants
        """
        # Get absolute scores (importance regardless of direction)
        abs_scores = np.abs(variant_scores)

        # Get top K indices
        top_indices = np.argsort(abs_scores)[::-1][:top_k]

        # Collect results
        top_variants = []
        for idx in top_indices:
            position = int(metadata['positions'][idx])
            gene_id = int(metadata['gene_ids'][idx])
            score = float(variant_scores[idx])
            top_variants.append((position, gene_id, score))

        return top_variants


class SIEVEWrapper(nn.Module):
    """
    Wrapper for SIEVE model to work with Captum.

    Captum expects a model that takes a single input tensor.
    This wrapper takes variant_features as the differentiable input and
    passes all other arguments (positions, gene_ids, mask, and optionally
    covariates) through ``additional_forward_args``.

    Covariates are passed as the optional last positional argument so that
    the function signature is identical whether or not the model uses them.
    SIEVEWrapper intentionally remains the public legacy wrapper so existing
    callers keep historical full-feature attribution semantics.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        covariates: Optional[Tensor] = None,
        chrom_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning only logits."""
        logits, _ = self.model(
            variant_features,
            positions,
            gene_ids,
            mask,
            covariates=covariates,
            return_attention=False,
            return_intermediate=False,
            chrom_ids=chrom_ids,
        )
        return logits


class ContentSIEVEWrapper(nn.Module):
    """
    Wrapper for content-only Integrated Gradients.

    ``content_features`` is the sole Captum differentiable input.
    ``absolute_position_features`` is an observed fixed forward argument, while
    positions, chromosome IDs, gene IDs, masks, and covariates remain fixed.
    The wrapper uses the split-primary model path and is mathematically
    different from slicing legacy full-feature attributions.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        content_features: Tensor,
        absolute_position_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        covariates: Tensor | None = None,
        chrom_ids: Tensor | None = None,
    ) -> Tensor:
        """Forward pass returning only logits from the split-primary path."""
        logits, _ = self.model(
            None,
            positions,
            gene_ids,
            mask,
            covariates=covariates,
            return_attention=False,
            return_intermediate=False,
            chrom_ids=chrom_ids,
            content_features=content_features,
            absolute_position_features=absolute_position_features,
        )
        return logits


def _coerce_resolved_ig_mode(ig_mode: ResolvedIGMode | str) -> ResolvedIGMode:
    if isinstance(ig_mode, ResolvedIGMode):
        return ig_mode
    if isinstance(ig_mode, str):
        if ig_mode == ResolvedIGMode.CONTENT.value:
            return ResolvedIGMode.CONTENT
        if ig_mode == ResolvedIGMode.LEGACY.value:
            return ResolvedIGMode.LEGACY
        if ig_mode == "auto":
            raise ValueError(
                "ig_mode='auto' must be resolved from configuration before "
                "constructing IntegratedGradientsExplainer"
            )
    raise ValueError("ig_mode must be resolved to 'content' or 'legacy'")


def _prepare_baseline(
    baseline: Tensor | None,
    differentiable_input: Tensor,
    device: str,
) -> Tensor:
    if baseline is None:
        return torch.zeros_like(differentiable_input)
    if not isinstance(baseline, torch.Tensor):
        raise ValueError(
            f"baseline must be a torch.Tensor; got {type(baseline).__name__}"
        )
    if baseline.shape != differentiable_input.shape:
        raise ValueError(
            "baseline shape must match the differentiable input shape: got "
            f"{tuple(baseline.shape)}, expected {tuple(differentiable_input.shape)}"
        )
    baseline = baseline.to(device)
    if baseline.dtype != differentiable_input.dtype:
        raise ValueError(
            "baseline dtype must match the differentiable input dtype: got "
            f"{baseline.dtype}, expected {differentiable_input.dtype}"
        )
    return baseline
