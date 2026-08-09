"""Parameterless positional-runtime boundaries for legacy SIEVE execution.

The classes in this module separate *where* positional information enters the
model from *how* the historical tensors and parameters are represented. They
intentionally own no trainable state: legacy checkpoints must keep seeing the
same ``position_bias`` and ``chrom_embedding`` keys on the attention modules.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from src.encoding import relative_position_bucket


class AbsolutePositionRuntime(Protocol):
    """Resolve absolute-position features before VariantEncoder composition."""

    def resolve(
        self,
        observed_absolute_position_features: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        """
        Return absolute-position features aligned to ``reference``.

        Implementations may use genomic positions, chromosomes, or masks in
        later phases. The legacy implementation returns the observed split
        tensor unchanged so sinusoidal features are never recomputed here.
        """


@dataclass(frozen=True)
class ObservedAbsolutePositionRuntime:
    """Legacy adapter that returns observed absolute-position features as-is."""

    def resolve(
        self,
        observed_absolute_position_features: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        """
        Return the exact observed absolute-position tensor after alignment checks.

        ``positions``, ``chrom_ids``, and ``mask`` are accepted to establish the
        future interface, but the legacy path is tensor-observed: it does not
        recompute, clone, detach, cast, or move absolute features.
        """
        if not isinstance(observed_absolute_position_features, Tensor):
            raise ValueError("observed_absolute_position_features must be a torch.Tensor.")
        if not isinstance(reference, Tensor):
            raise ValueError("reference must be a torch.Tensor.")
        if observed_absolute_position_features.ndim != reference.ndim:
            raise ValueError(
                "observed_absolute_position_features and reference must have the same rank."
            )
        if observed_absolute_position_features.shape[:-1] != reference.shape[:-1]:
            raise ValueError(
                "observed_absolute_position_features and reference must have matching "
                "leading batch/variant dimensions."
            )
        return observed_absolute_position_features


class RelativePositionRuntime(Protocol):
    """Adjust attention scores with a relative-position strategy."""

    def adjust_attention_scores(
        self,
        base_scores: Tensor,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        position_bias: nn.Embedding | None,
    ) -> Tensor:
        """
        Return attention scores after relative-position adjustment.

        The score-level boundary keeps future RoPE implementations possible:
        same-chromosome pairs can alter query/key scores while cross-chromosome
        pairs can retain unrotated scores plus an explicit bias.
        """


@dataclass(frozen=True)
class LegacyT5RelativePositionRuntime:
    """Parameterless wrapper around the historical T5-style bucket bias."""

    num_position_buckets: int
    max_distance: int

    def compute_bias(
        self,
        query_positions: Tensor,
        key_positions: Tensor,
        position_bias: nn.Embedding | None,
        query_chroms: Tensor | None = None,
        key_chroms: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the historical relative-position bias tensor.

        The batch loop and embedding lookup intentionally mirror the original
        attention method so state-dict ownership and numerical behavior remain
        unchanged.
        """
        if position_bias is None:
            raise ValueError("position_bias embedding is required for legacy T5 bias.")

        position_buckets_list = []
        for batch_idx in range(query_positions.shape[0]):
            query_chroms_b = query_chroms[batch_idx] if query_chroms is not None else None
            key_chroms_b = key_chroms[batch_idx] if key_chroms is not None else None
            position_buckets_list.append(
                relative_position_bucket(
                    query_positions[batch_idx],
                    key_positions[batch_idx],
                    num_buckets=self.num_position_buckets,
                    max_distance=self.max_distance,
                    query_chroms=query_chroms_b,
                    key_chroms=key_chroms_b,
                )
            )

        position_buckets = torch.stack(position_buckets_list, dim=0)
        return position_bias(position_buckets).permute(0, 3, 1, 2)

    def adjust_attention_scores(
        self,
        base_scores: Tensor,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        position_bias: nn.Embedding | None,
    ) -> Tensor:
        """Add historical T5-style relative bias to precomputed QK scores."""
        bias = self.compute_bias(
            positions,
            positions,
            position_bias,
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )
        return base_scores + bias
