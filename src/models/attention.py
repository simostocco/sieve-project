"""
Position-aware self-attention for SIEVE.

This module implements position-aware dense self-attention over the sparse
variant set: attention runs over the variant-present positions of a sample
(not all genomic positions) while preserving spatial relationships through
relative position bias.

Key features:
- Multi-head attention over the observed variant positions
- Relative position bias with logarithmic bucketing
- Proper masking for variable-length sequences
- Numerically stable implementation

Author: Francesco Lescai
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.encoding.position_config import (
    ChromosomeEncoding,
    CrossChromosomePolicy,
    RelativePositionEncoding,
    ResolvedPositionEncodingConfig,
)

from .position_runtime import (
    LegacyT5RelativePositionRuntime,
    build_relative_position_runtime,
    build_same_chromosome_pair_mask,
    validate_phase7_runtime_support,
)


class PositionAwareSparseAttention(nn.Module):
    """
    Position-aware self-attention over variant positions.

    The name is retained from an earlier design and is kept stable for
    compatibility. Attention here is dense over the set of variants a sample
    carries; the sparsity is a property of the input representation, which
    materialises only alternate-allele sites, not of the attention pattern.
    Cost is therefore quadratic in the number of variants per sample rather
    than in the number of genomic positions.

    Unlike DeepRVAT (which uses permutation-invariant deep sets), this
    attention mechanism preserves genomic position information through a
    learnable relative position bias.

    The attention computation includes:
    1. Standard multi-head attention (Q, K, V)
    2. Relative position bias based on genomic distance
    3. Proper masking for padding and invalid positions

    Attention formula:
        attention = softmax((Q @ K.T) / sqrt(d_k) + position_bias) @ V

    Parameters
    ----------
    latent_dim : int
        Dimension of input embeddings
    num_heads : int
        Number of attention heads (default: 4)
    dropout : float
        Dropout probability (default: 0.1)
    num_position_buckets : int
        Number of relative position buckets (default: 32)
    max_distance : int
        Maximum genomic distance to consider (default: 100,000 bp)

    Attributes
    ----------
    query : nn.Linear
        Query projection
    key : nn.Linear
        Key projection
    value : nn.Linear
        Value projection
    position_bias : nn.Embedding
        Learnable position bias for each bucket and head
    output_proj : nn.Linear
        Output projection

    Examples
    --------
    >>> attention = PositionAwareSparseAttention(latent_dim=64, num_heads=4)
    >>> x = torch.randn(2, 10, 64)  # [batch, variants, latent_dim]
    >>> positions = torch.tensor([[100, 200, 300, ...], [...]])  # [batch, variants]
    >>> mask = torch.ones(2, 10, dtype=torch.bool)  # [batch, variants]
    >>> output, attn_weights = attention(x, positions, mask)
    >>> output.shape
    torch.Size([2, 10, 64])
    >>> attn_weights.shape
    torch.Size([2, 4, 10, 10])  # [batch, heads, queries, keys]
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_position_buckets: int = 32,
        max_distance: int = 100000,
        num_chromosomes: int = 0,
        position_encoding: ResolvedPositionEncodingConfig | None = None,
    ):
        super().__init__()

        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        if position_encoding is not None:
            if not isinstance(position_encoding, ResolvedPositionEncodingConfig):
                raise ValueError("position_encoding must be a ResolvedPositionEncodingConfig.")
            validate_phase7_runtime_support(position_encoding)
            resolved_num_chromosomes = position_encoding.chromosome.num_chromosomes
            if num_chromosomes not in {0, resolved_num_chromosomes}:
                raise ValueError(
                    "num_chromosomes must be 0 or match "
                    "position_encoding.chromosome.num_chromosomes when position_encoding "
                    "is supplied."
                )
        else:
            resolved_num_chromosomes = num_chromosomes

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.position_encoding = position_encoding
        self.num_chromosomes = resolved_num_chromosomes
        if position_encoding is None:
            self.num_position_buckets = num_position_buckets
            self.max_distance = max_distance
            self._relative_position_runtime = LegacyT5RelativePositionRuntime(
                num_position_buckets=num_position_buckets,
                max_distance=max_distance,
            )
        else:
            self.num_position_buckets = position_encoding.relative.num_buckets or 0
            self.max_distance = position_encoding.relative.max_distance_bp or 0
            self._relative_position_runtime = build_relative_position_runtime(position_encoding)

        # Attention projections
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

        # Learnable position bias. In historical/no-config mode the allocation
        # is exactly the Phase 6 surface. Explicit configs allocate only the
        # rows required by the resolved relative strategy.
        if position_encoding is None:
            self.position_bias = nn.Embedding(num_position_buckets + 1, num_heads)
        elif position_encoding.relative.encoding is RelativePositionEncoding.NONE:
            self.position_bias = None
        elif position_encoding.relative.encoding is RelativePositionEncoding.T5_BUCKET:
            total_bias_rows = position_encoding.relative.total_bias_rows
            if not isinstance(total_bias_rows, int) or total_bias_rows <= 0:
                raise ValueError("position_encoding.relative.total_bias_rows must be positive.")
            self.position_bias = nn.Embedding(total_bias_rows, num_heads)
        else:
            raise NotImplementedError(
                f"relative_position_encoding={position_encoding.relative.encoding.value} "
                "is not implemented."
            )

        # Optional chromosome embedding added to inputs before computing Q/K/V.
        # Disambiguates variants that share a coordinate on different
        # chromosomes. Allocated only when num_chromosomes > 0; one extra row
        # covers padding sentinel values that may be passed during forward().
        if position_encoding is None:
            use_chrom_embedding = num_chromosomes > 0
        else:
            use_chrom_embedding = (
                position_encoding.chromosome.encoding is ChromosomeEncoding.LEARNED
            )
            if use_chrom_embedding and resolved_num_chromosomes <= 0:
                raise ValueError("learned chromosome encoding requires num_chromosomes > 0.")
        if use_chrom_embedding:
            self.chrom_embedding = nn.Embedding(resolved_num_chromosomes + 1, latent_dim)
            nn.init.zeros_(self.chrom_embedding.weight)
        else:
            self.chrom_embedding = None

        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize position bias to zero (no bias initially)
        if self.position_bias is not None:
            nn.init.zeros_(self.position_bias.weight)

    def _compute_position_bias(
        self,
        query_positions: Tensor,
        key_positions: Tensor,
        query_chroms: Optional[Tensor] = None,
        key_chroms: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute relative position bias.

        Parameters
        ----------
        query_positions : Tensor
            Query positions, shape (batch, num_queries)
        key_positions : Tensor
            Key positions, shape (batch, num_keys)
        query_chroms : Optional[Tensor]
            Query chromosome ids, shape (batch, num_queries). When provided,
            cross-chromosome pairs are routed to the dedicated bucket.
        key_chroms : Optional[Tensor]
            Key chromosome ids, shape (batch, num_keys).

        Returns
        -------
        Tensor
            Position bias, shape (batch, num_heads, num_queries, num_keys)
        """
        if self.position_bias is None:
            raise ValueError("No position bias exists for relative_position_encoding=none.")
        return self._relative_position_runtime.compute_bias(
            query_positions,
            key_positions,
            self.position_bias,
            query_chroms=query_chroms,
            key_chroms=key_chroms,
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
        chrom_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply position-aware self-attention.

        Parameters
        ----------
        x : Tensor
            Input embeddings, shape (batch, num_variants, latent_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        mask : Optional[Tensor]
            Attention mask, shape (batch, num_variants)
            True = valid position, False = padding
        return_attention : bool
            Whether to return attention weights (for explainability)
        chrom_ids : Optional[Tensor]
            Chromosome indices, shape (batch, num_variants). When provided,
            two effects:
            (1) ``self.chrom_embedding(chrom_ids)`` is added to ``x`` before
            computing Q/K/V, disambiguating same-coordinate variants on
            different chromosomes; this requires ``num_chromosomes > 0`` at
            construction time.
            (2) cross-chromosome pairs are routed to the dedicated relative-
            position bias bucket. Cross-chromosome attention itself is **not**
            masked, only the position-bias prior changes.

        Returns
        -------
        output : Tensor
            Attended features, shape (batch, num_variants, latent_dim)
        attention_weights : Optional[Tensor]
            Attention weights if return_attention=True,
            shape (batch, num_heads, num_variants, num_variants)
        """
        batch_size, num_variants, _ = x.shape
        configured_execution = self.position_encoding is not None
        if configured_execution:
            self._validate_configured_attention_inputs(positions, mask, chrom_ids)

        # Add chromosome embedding to inputs (when configured). This is the
        # absolute-disambiguation half of the chromosome-aware fix and is
        # independent of the relative-bias half below.
        if chrom_ids is not None and self.chrom_embedding is not None:
            x = x + self.chrom_embedding(chrom_ids)

        # Project to Q, K, V
        # Shape: (batch, num_variants, latent_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        # Shape: (batch, num_variants, num_heads, head_dim)
        Q = Q.view(batch_size, num_variants, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_variants, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_variants, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, num_variants, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        # (batch, num_heads, num_variants, head_dim) @ (batch, num_heads, head_dim, num_variants)
        # -> (batch, num_heads, num_variants, num_variants)
        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add relative position bias. When chrom_ids is supplied, cross-
        # chromosome pairs use the dedicated bucket; otherwise the legacy
        # chromosome-blind bucketing is preserved for backward compatibility.
        attn_scores = self._relative_position_runtime.adjust_attention_scores(
            base_scores,
            query=Q,
            key=K,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=self.position_bias,
        )

        if configured_execution:
            if (
                self.position_encoding.chromosome.cross_chromosome_policy
                is CrossChromosomePolicy.MASK
            ):
                same_chromosome = build_same_chromosome_pair_mask(chrom_ids)
                attn_scores = attn_scores.masked_fill(
                    ~same_chromosome.unsqueeze(1),
                    float("-inf"),
                )
            if mask is not None:
                valid_pairs = mask[:, :, None] & mask[:, None, :]
                attn_scores = attn_scores.masked_fill(
                    ~valid_pairs.unsqueeze(1),
                    float("-inf"),
                )
        else:
            # Apply mask if provided
            if mask is not None:
                # Create attention mask: (batch, 1, 1, num_variants)
                # Broadcasting will expand to (batch, num_heads, num_variants, num_variants)
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_variants)

                # Set masked positions to large negative value
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

        # Softmax over key dimension
        # Shape: (batch, num_heads, num_variants, num_variants)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Handle all-masked rows (will have NaN after softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply dropout
        attn_weights_dropout = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, num_variants, num_variants) @ (batch, num_heads, num_variants, head_dim)
        # -> (batch, num_heads, num_variants, head_dim)
        attended = torch.matmul(attn_weights_dropout, V)

        # Transpose back to (batch, num_variants, num_heads, head_dim)
        attended = attended.transpose(1, 2)

        # Concatenate heads
        # Shape: (batch, num_variants, latent_dim)
        attended = attended.contiguous().view(batch_size, num_variants, self.latent_dim)

        # Output projection
        output = self.output_proj(attended)

        if return_attention:
            return output, attn_weights
        else:
            return output, None

    def _validate_configured_attention_inputs(
        self,
        positions: Tensor,
        mask: Tensor | None,
        chrom_ids: Tensor | None,
    ) -> None:
        """Validate explicit new-schema chromosome and padding inputs."""
        if not isinstance(positions, Tensor) or positions.ndim != 2:
            raise ValueError("positions must be a rank-2 torch.Tensor in explicit-config mode.")
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise ValueError("mask must be a torch.Tensor in explicit-config mode.")
            if mask.dtype is not torch.bool:
                raise ValueError("mask must be a boolean torch.Tensor in explicit-config mode.")
            if mask.shape != positions.shape:
                raise ValueError("mask shape must match positions shape in explicit-config mode.")

        requires_chrom_ids = self.position_encoding.chromosome.requires_chrom_ids
        if requires_chrom_ids and chrom_ids is None:
            raise ValueError("chrom_ids are required by the resolved position encoding.")
        if chrom_ids is None:
            return
        if not isinstance(chrom_ids, Tensor):
            raise ValueError("chrom_ids must be a torch.Tensor in explicit-config mode.")
        if chrom_ids.ndim != 2:
            raise ValueError("chrom_ids must have shape [batch, variants].")
        if chrom_ids.shape != positions.shape:
            raise ValueError("chrom_ids shape must match positions shape.")

        real_chrom_ids = chrom_ids[mask] if mask is not None else chrom_ids.reshape(-1)
        if real_chrom_ids.numel() == 0:
            return
        if torch.any(real_chrom_ids < 0) or torch.any(real_chrom_ids >= self.num_chromosomes):
            raise ValueError(
                "real chromosome ids must satisfy 0 <= chrom_id < "
                "position_encoding.chromosome.num_chromosomes."
            )


class MultiLayerAttention(nn.Module):
    """
    Multiple layers of position-aware attention with residual connections.

    Parameters
    ----------
    latent_dim : int
        Dimension of embeddings
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of attention layers (default: 2)
    dropout : float
        Dropout probability
    num_position_buckets : int
        Number of position buckets
    max_distance : int
        Maximum genomic distance

    Examples
    --------
    >>> multi_attn = MultiLayerAttention(latent_dim=64, num_heads=4, num_layers=2)
    >>> x = torch.randn(2, 10, 64)
    >>> positions = torch.randint(100, 1000, (2, 10))
    >>> mask = torch.ones(2, 10, dtype=torch.bool)
    >>> output, attn_list = multi_attn(x, positions, mask, return_attention=True)
    >>> len(attn_list)
    2
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_position_buckets: int = 32,
        max_distance: int = 100000,
        num_chromosomes: int = 0,
        position_encoding: ResolvedPositionEncodingConfig | None = None,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.attention_layers = nn.ModuleList([
            PositionAwareSparseAttention(
                latent_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_position_buckets=num_position_buckets,
                max_distance=max_distance,
                num_chromosomes=num_chromosomes,
                position_encoding=position_encoding,
            )
            for _ in range(num_layers)
        ])

        # Layer norm for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
        chrom_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Apply multiple attention layers with residual connections.

        Parameters
        ----------
        x : Tensor
            Input embeddings, shape (batch, num_variants, latent_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        mask : Optional[Tensor]
            Attention mask, shape (batch, num_variants)
        return_attention : bool
            Whether to return attention weights from all layers
        chrom_ids : Optional[Tensor]
            Chromosome indices, shape (batch, num_variants). Forwarded to each
            ``PositionAwareSparseAttention`` layer.

        Returns
        -------
        output : Tensor
            Final output, shape (batch, num_variants, latent_dim)
        attention_list : Optional[list]
            List of attention weights from each layer if return_attention=True
        """
        attention_list = [] if return_attention else None

        for i, (attn_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Attention with residual connection
            attn_out, attn_weights = attn_layer(
                x,
                positions,
                mask,
                return_attention=return_attention,
                chrom_ids=chrom_ids,
            )

            # Residual connection + layer norm
            x = layer_norm(x + attn_out)

            if return_attention:
                attention_list.append(attn_weights)

        return x, attention_list
