"""Parameterless positional-runtime boundaries for SIEVE position execution.

The classes in this module separate *where* positional information enters the
model from *how* the historical tensors and parameters are represented. They
intentionally own no trainable state: legacy checkpoints must keep seeing the
same ``position_bias`` and ``chrom_embedding`` keys on the attention modules.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from src.encoding import relative_position_bucket
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedPositionEncodingConfig,
)
from src.encoding.position_layout import (
    LEARNED_BINNED_COORDINATE_ORIGIN,
    LEARNED_BINNED_LAYOUT,
    LEARNED_BINNED_LAYOUT_SCHEMA_VERSION,
    LearnedBinnedAbsolutePositionLayout,
)


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


@dataclass(frozen=True)
class NoAbsolutePositionRuntime:
    """Custom absolute-position runtime for ``absolute_position_encoding=none``."""

    def resolve(
        self,
        observed_absolute_position_features: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        """Return a zero-width absolute-position view aligned to ``reference``."""
        if not isinstance(reference, Tensor):
            raise ValueError("reference must be a torch.Tensor.")
        if reference.ndim < 2:
            raise ValueError("reference must have rank at least 2.")
        return reference[..., :0]


@dataclass(frozen=True)
class SinusoidalAbsolutePositionRuntime:
    """Custom model-side sinusoidal absolute-position runtime."""

    position_dim: int
    coordinate_scale: float
    max_wavelength: float

    def __post_init__(self) -> None:
        """Validate direct construction outside the pure resolver."""
        _validate_positive_int("position_dim", self.position_dim)
        if self.position_dim % 2 != 0:
            raise ValueError("position_dim must be even for sinusoidal absolute position.")
        _validate_positive_finite_number("coordinate_scale", self.coordinate_scale)
        _validate_positive_finite_number("max_wavelength", self.max_wavelength)

    def resolve(
        self,
        observed_absolute_position_features: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        """
        Compute sinusoidal features from positions while ignoring observed features.

        Padding rows are zeroed by ``mask`` so padded coordinate zero does not
        contribute ``cos(0)=1`` channels.
        """
        if not isinstance(reference, Tensor):
            raise ValueError("reference must be a torch.Tensor.")
        if not isinstance(positions, Tensor):
            raise ValueError("positions must be a torch.Tensor.")
        if reference.ndim < 2:
            raise ValueError("reference must have rank at least 2.")
        if positions.shape != reference.shape[:-1]:
            raise ValueError("positions shape must equal reference.shape[:-1].")
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise ValueError("mask must be a torch.Tensor when provided.")
            if mask.shape != positions.shape:
                raise ValueError("mask shape must match positions shape.")
            if mask.dtype is not torch.bool:
                raise ValueError("mask must be a boolean torch.Tensor.")

        scaled_positions = positions.to(device=reference.device, dtype=reference.dtype)
        scaled_positions = scaled_positions / self.coordinate_scale
        div = torch.exp(
            torch.arange(
                0,
                self.position_dim,
                2,
                device=reference.device,
                dtype=reference.dtype,
            )
            * -(math.log(self.max_wavelength) / self.position_dim)
        )
        positional_features = torch.empty(
            (*positions.shape, self.position_dim),
            device=reference.device,
            dtype=reference.dtype,
        )
        angles = scaled_positions[..., None] * div
        positional_features[..., 0::2] = torch.sin(angles)
        positional_features[..., 1::2] = torch.cos(angles)
        if mask is not None:
            positional_features = positional_features.masked_fill(
                ~mask.to(device=reference.device).unsqueeze(-1),
                0,
            )
        return positional_features


@dataclass(frozen=True)
class LearnedBinnedAbsolutePositionRuntime:
    """Resolve learned absolute-position rows from chromosome-local bins.

    The trainable table is registered on ``SIEVE`` as
    ``absolute_position_embedding``. This runtime is a plain routing object: it
    holds a reference to that registered embedding and to immutable layout
    metadata, but it does not own independent parameters or buffers.
    """

    embedding: nn.Embedding
    layout: LearnedBinnedAbsolutePositionLayout
    bin_size_bp: int
    position_dim: int

    def __post_init__(self) -> None:
        """Validate direct construction outside SIEVE."""
        if not isinstance(self.embedding, nn.Embedding):
            raise ValueError("embedding must be an nn.Embedding.")
        validate_learned_binned_layout_settings(
            self.layout,
            bin_size_bp=self.bin_size_bp,
            position_dim=self.position_dim,
            num_chromosomes=len(self.layout.chromosome_lengths_bp),
        )
        if self.embedding.num_embeddings != self.layout.num_embeddings:
            raise ValueError("embedding.num_embeddings must match layout.num_embeddings.")
        if self.embedding.embedding_dim != self.position_dim:
            raise ValueError("embedding.embedding_dim must match position_dim.")

    def resolve(
        self,
        observed_absolute_position_features: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        mask: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        """Look up learned absolute features for real variants only.

        Padding is mask-authoritative: padded rows may contain coordinate
        sentinels such as position 0 or nonsense chromosome IDs, so lookup rows
        are computed only where ``mask`` is true. Padded outputs are forced to
        exact zeros after embedding lookup.
        """
        del observed_absolute_position_features
        if not isinstance(reference, Tensor):
            raise ValueError("reference must be a torch.Tensor.")
        if reference.ndim < 2:
            raise ValueError("reference must have rank at least 2.")
        if not isinstance(positions, Tensor):
            raise ValueError("positions must be a torch.Tensor.")
        if chrom_ids is None:
            raise ValueError("chrom_ids are required for learned_binned absolute position.")
        if not isinstance(chrom_ids, Tensor):
            raise ValueError("chrom_ids must be a torch.Tensor.")
        if mask is None:
            raise ValueError("mask is required for learned_binned absolute position.")
        if not isinstance(mask, Tensor):
            raise ValueError("mask must be a torch.Tensor.")
        if positions.shape != reference.shape[:-1]:
            raise ValueError("positions shape must equal reference.shape[:-1].")
        if chrom_ids.shape != reference.shape[:-1]:
            raise ValueError("chrom_ids shape must equal reference.shape[:-1].")
        if mask.shape != reference.shape[:-1]:
            raise ValueError("mask shape must equal reference.shape[:-1].")
        if mask.dtype is not torch.bool:
            raise ValueError("mask must be a boolean torch.Tensor.")
        if not _is_integer_tensor(positions):
            raise ValueError("positions must use an integer dtype.")
        if not _is_integer_tensor(chrom_ids):
            raise ValueError("chrom_ids must use an integer dtype.")

        device = self.embedding.weight.device
        positions_on_device = positions.to(device=device, dtype=torch.long)
        chrom_ids_on_device = chrom_ids.to(device=device, dtype=torch.long)
        mask_on_device = mask.to(device=device)

        safe_rows = torch.zeros_like(positions_on_device, dtype=torch.long, device=device)
        real_positions = positions_on_device[mask_on_device]
        real_chrom_ids = chrom_ids_on_device[mask_on_device]
        if real_positions.numel() > 0:
            if torch.any(real_positions < 1):
                raise ValueError("real learned_binned positions must be >= 1.")
            if torch.any(real_chrom_ids < 0) or torch.any(
                real_chrom_ids >= len(self.layout.chromosome_lengths_bp)
            ):
                raise ValueError("real chrom_ids must satisfy 0 <= chrom_id < num_chromosomes.")

            lengths = torch.tensor(
                self.layout.chromosome_lengths_bp,
                dtype=torch.long,
                device=device,
            )
            real_lengths = lengths[real_chrom_ids]
            if torch.any(real_positions > real_lengths):
                raise ValueError("real learned_binned positions must not exceed chromosome length.")

            offsets = torch.tensor(
                self.layout.chromosome_offsets,
                dtype=torch.long,
                device=device,
            )
            local_bins = (real_positions - 1) // self.bin_size_bp
            global_rows = offsets[real_chrom_ids] + local_bins.to(dtype=torch.long)
            safe_rows[mask_on_device] = global_rows

        resolved = self.embedding(safe_rows)
        if resolved.dtype != reference.dtype:
            resolved = resolved.to(dtype=reference.dtype)
        resolved = resolved.masked_fill(~mask_on_device.unsqueeze(-1), 0)
        return resolved


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
        cross_chromosome_bias: Tensor | None = None,
    ) -> Tensor:
        """
        Return attention scores after relative-position adjustment.

        The score-level boundary keeps future RoPE implementations possible:
        same-chromosome pairs can alter query/key scores while cross-chromosome
        pairs can retain unrotated scores plus an explicit bias.
        """


@dataclass(frozen=True)
class NoRelativePositionRuntime:
    """Custom relative-position runtime for ``relative_position_encoding=none``."""

    def adjust_attention_scores(
        self,
        base_scores: Tensor,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        position_bias: nn.Embedding | None,
        cross_chromosome_bias: Tensor | None = None,
    ) -> Tensor:
        """Return the exact base score tensor object unchanged."""
        if cross_chromosome_bias is not None:
            raise ValueError("cross_chromosome_bias is not used for relative_position_encoding=none.")
        return base_scores


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
        cross_chromosome_bias: Tensor | None = None,
    ) -> Tensor:
        """Add historical T5-style relative bias to precomputed QK scores."""
        if cross_chromosome_bias is not None:
            raise ValueError("cross_chromosome_bias is not used for legacy T5 bias.")
        bias = self.compute_bias(
            positions,
            positions,
            position_bias,
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )
        return base_scores + bias


@dataclass(frozen=True)
class T5RelativePositionRuntime:
    """Custom T5-style relative-position runtime without owned parameters."""

    num_position_buckets: int
    max_distance: int
    cross_chromosome_policy: CrossChromosomePolicy

    def __post_init__(self) -> None:
        """Validate direct construction outside the pure resolver."""
        _validate_t5_settings(self.num_position_buckets, self.max_distance)
        if not isinstance(self.cross_chromosome_policy, CrossChromosomePolicy):
            raise ValueError("cross_chromosome_policy must be a CrossChromosomePolicy enum.")

    def compute_bias(
        self,
        query_positions: Tensor,
        key_positions: Tensor,
        position_bias: nn.Embedding | None,
        query_chroms: Tensor | None = None,
        key_chroms: Tensor | None = None,
    ) -> Tensor:
        """Compute custom T5 bias according to the resolved cross-chromosome policy."""
        if position_bias is None:
            raise ValueError("position_bias embedding is required for T5 bias.")

        if self.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE:
            if query_chroms is None or key_chroms is None:
                raise ValueError(
                    "query_chroms and key_chroms are required for cross_chromosome_policy=separate."
                )
            expected_rows = self.num_position_buckets + 1
            use_chroms = True
        else:
            expected_rows = self.num_position_buckets
            use_chroms = False

        if position_bias.num_embeddings != expected_rows:
            raise ValueError(
                "position_bias.num_embeddings must match the resolved T5 bucket row count."
            )

        position_buckets_list = []
        for batch_idx in range(query_positions.shape[0]):
            position_buckets_list.append(
                relative_position_bucket(
                    query_positions[batch_idx],
                    key_positions[batch_idx],
                    num_buckets=self.num_position_buckets,
                    max_distance=self.max_distance,
                    query_chroms=query_chroms[batch_idx] if use_chroms else None,
                    key_chroms=key_chroms[batch_idx] if use_chroms else None,
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
        cross_chromosome_bias: Tensor | None = None,
    ) -> Tensor:
        """Add custom T5-style relative bias to precomputed QK scores."""
        if cross_chromosome_bias is not None:
            raise ValueError("cross_chromosome_bias is not used for T5 bias.")
        bias = self.compute_bias(
            positions,
            positions,
            position_bias,
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )
        return base_scores + bias


@dataclass(frozen=True)
class RopeRelativePositionRuntime:
    """Parameterless RoPE relative-position runtime for Q/K score routing.

    RoPE is chromosome-local in this project. Same-chromosome pairs use
    rotary-transformed Q/K scores. Cross-chromosome pairs never subtract
    coordinates; under ``separate`` they use the unrotated base QK score plus
    one attention-owned learned scalar per head.
    """

    head_dim: int
    coordinate_scale: float
    rope_base: float
    cross_chromosome_policy: CrossChromosomePolicy

    def __post_init__(self) -> None:
        """Validate direct construction outside the pure resolver."""
        _validate_positive_int("head_dim", self.head_dim)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        _validate_positive_finite_number("coordinate_scale", self.coordinate_scale)
        _validate_positive_finite_number("rope_base", self.rope_base)
        if not isinstance(self.cross_chromosome_policy, CrossChromosomePolicy):
            raise ValueError("cross_chromosome_policy must be a CrossChromosomePolicy enum.")

    def rotate(self, values: Tensor, positions: Tensor) -> Tensor:
        """Rotate adjacent Q/K feature pairs using raw genomic positions."""
        self._validate_token_inputs(values, positions, name="values")
        compute_dtype = torch.float64 if values.dtype is torch.float64 else torch.float32
        values_for_rotation = values.to(dtype=compute_dtype)
        positions_for_rotation = positions.to(device=values.device, dtype=compute_dtype)
        scaled_positions = positions_for_rotation / self.coordinate_scale
        inverse_frequency = torch.pow(
            torch.as_tensor(self.rope_base, device=values.device, dtype=compute_dtype),
            -torch.arange(
                0,
                self.head_dim,
                2,
                device=values.device,
                dtype=compute_dtype,
            )
            / self.head_dim,
        )
        angles = scaled_positions[:, None, :, None] * inverse_frequency.view(1, 1, 1, -1)
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        even = values_for_rotation[..., 0::2]
        odd = values_for_rotation[..., 1::2]
        rotated = torch.empty_like(values_for_rotation)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated

    def adjust_attention_scores(
        self,
        base_scores: Tensor,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
        position_bias: nn.Embedding | None,
        cross_chromosome_bias: Tensor | None = None,
    ) -> Tensor:
        """Route same-chromosome RoPE scores and cross-chromosome policy scores."""
        if position_bias is not None:
            raise ValueError("position_bias must be None for relative_position_encoding=rope.")
        self._validate_score_inputs(base_scores, query, key, positions, chrom_ids)
        if chrom_ids is None:
            raise ValueError("chrom_ids are required for relative_position_encoding=rope.")

        query_rot = self.rotate(query, positions)
        key_rot = self.rotate(key, positions)
        rotated_scores = torch.matmul(query_rot, key_rot.transpose(-2, -1)) / (
            self.head_dim**0.5
        )
        if rotated_scores.dtype != base_scores.dtype:
            rotated_scores = rotated_scores.to(dtype=base_scores.dtype)

        same_chromosome = build_same_chromosome_pair_mask(chrom_ids).to(device=base_scores.device)
        if self.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE:
            self._validate_cross_chromosome_bias(cross_chromosome_bias, base_scores)
            bias = cross_chromosome_bias.to(device=base_scores.device, dtype=base_scores.dtype)
            cross_scores = base_scores + bias.view(1, -1, 1, 1)
            return torch.where(same_chromosome.unsqueeze(1), rotated_scores, cross_scores)

        if self.cross_chromosome_policy is CrossChromosomePolicy.MASK:
            if cross_chromosome_bias is not None:
                raise ValueError(
                    "cross_chromosome_bias must be None for cross_chromosome_policy=mask."
                )
            return torch.where(same_chromosome.unsqueeze(1), rotated_scores, base_scores)

        raise ValueError(f"unsupported cross_chromosome_policy: {self.cross_chromosome_policy!r}")

    def _validate_token_inputs(
        self,
        values: Tensor,
        positions: Tensor,
        *,
        name: str,
    ) -> None:
        if not isinstance(values, Tensor):
            raise ValueError(f"{name} must be a torch.Tensor.")
        if values.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, heads, variants, head_dim].")
        if values.shape[-1] != self.head_dim:
            raise ValueError(f"{name} final dimension must match head_dim.")
        if not values.dtype.is_floating_point:
            raise ValueError(f"{name} must use a floating dtype.")
        if not isinstance(positions, Tensor):
            raise ValueError("positions must be a torch.Tensor.")
        if positions.ndim != 2:
            raise ValueError("positions must have shape [batch, variants].")
        if positions.shape != (values.shape[0], values.shape[2]):
            raise ValueError("positions shape must match query/key batch and variant dimensions.")
        if not _is_integer_tensor(positions):
            raise ValueError("positions must use an integer dtype.")

    def _validate_score_inputs(
        self,
        base_scores: Tensor,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        chrom_ids: Tensor | None,
    ) -> None:
        if not isinstance(base_scores, Tensor):
            raise ValueError("base_scores must be a torch.Tensor.")
        if base_scores.ndim != 4:
            raise ValueError("base_scores must have shape [batch, heads, queries, keys].")
        if not base_scores.dtype.is_floating_point:
            raise ValueError("base_scores must use a floating dtype.")
        self._validate_token_inputs(query, positions, name="query")
        self._validate_token_inputs(key, positions, name="key")
        expected_scores = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
        if tuple(base_scores.shape) != expected_scores:
            raise ValueError("base_scores shape must match query/key score dimensions.")
        if query.shape != key.shape:
            raise ValueError("query and key must have identical shapes for RoPE.")
        if chrom_ids is None:
            return
        if not isinstance(chrom_ids, Tensor):
            raise ValueError("chrom_ids must be a torch.Tensor.")
        if chrom_ids.ndim != 2:
            raise ValueError("chrom_ids must have shape [batch, variants].")
        if chrom_ids.shape != positions.shape:
            raise ValueError("chrom_ids shape must match positions shape.")
        if not _is_integer_tensor(chrom_ids):
            raise ValueError("chrom_ids must use an integer dtype.")

    def _validate_cross_chromosome_bias(
        self,
        cross_chromosome_bias: Tensor | None,
        base_scores: Tensor,
    ) -> None:
        if cross_chromosome_bias is None:
            raise ValueError(
                "cross_chromosome_bias is required for relative_position_encoding=rope "
                "with cross_chromosome_policy=separate."
            )
        if not isinstance(cross_chromosome_bias, Tensor):
            raise ValueError("cross_chromosome_bias must be a torch.Tensor.")
        if cross_chromosome_bias.shape != (base_scores.shape[1],):
            raise ValueError("cross_chromosome_bias shape must equal (num_heads,).")
        if not cross_chromosome_bias.dtype.is_floating_point:
            raise ValueError("cross_chromosome_bias must use a floating dtype.")


def build_same_chromosome_pair_mask(chrom_ids: Tensor) -> Tensor:
    """
    Build a chromosome-routing mask without applying padding or attention logic.

    ``True`` means the query/key pair has the same zero-based chromosome id.
    Chromosome id 0 is treated as an ordinary real chromosome id.
    """
    if not isinstance(chrom_ids, Tensor):
        raise ValueError("chrom_ids must be a torch.Tensor.")
    if chrom_ids.ndim != 2:
        raise ValueError("chrom_ids must have shape [batch, variants].")
    return chrom_ids[:, :, None] == chrom_ids[:, None, :]


def validate_attention_runtime_support(config: ResolvedPositionEncodingConfig) -> None:
    """Validate the positional strategies executed inside attention.

    Absolute-position fusion is owned by ``SIEVE`` before VariantEncoder and is
    therefore not an attention support decision.
    """
    if not isinstance(config, ResolvedPositionEncodingConfig):
        raise ValueError("config must be a ResolvedPositionEncodingConfig.")
    if config.preset not in {PositionPreset.LEGACY, PositionPreset.CUSTOM}:
        raise NotImplementedError(f"position preset {config.preset!r} is not supported.")
    if config.relative.encoding in {
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }:
        raise NotImplementedError(
            f"relative_position_encoding={config.relative.encoding.value} is not implemented."
        )
    if config.relative.encoding not in {
        RelativePositionEncoding.NONE,
        RelativePositionEncoding.T5_BUCKET,
        RelativePositionEncoding.ROPE,
    }:
        raise NotImplementedError(
            f"relative_position_encoding={config.relative.encoding.value} is not implemented."
        )
    if config.chromosome.encoding not in {ChromosomeEncoding.NONE, ChromosomeEncoding.LEARNED}:
        raise NotImplementedError(
            f"chromosome_encoding={config.chromosome.encoding.value} is not supported."
        )
    if config.chromosome.cross_chromosome_policy not in {
        CrossChromosomePolicy.SEPARATE,
        CrossChromosomePolicy.MASK,
    }:
        raise NotImplementedError(
            "cross_chromosome_policy="
            f"{config.chromosome.cross_chromosome_policy.value} is not supported."
        )


def validate_learned_binned_layout_settings(
    layout: LearnedBinnedAbsolutePositionLayout,
    *,
    bin_size_bp: int,
    position_dim: int,
    num_chromosomes: int,
) -> None:
    """Validate layout metadata against model-side learned-binned dimensions."""
    if not isinstance(layout, LearnedBinnedAbsolutePositionLayout):
        raise ValueError(
            "learned_binned_position_layout must be a LearnedBinnedAbsolutePositionLayout."
        )
    _validate_positive_int(
        "learned_binned_position_layout.schema_version",
        layout.schema_version,
    )
    if layout.schema_version != LEARNED_BINNED_LAYOUT_SCHEMA_VERSION:
        raise ValueError("learned_binned_position_layout.schema_version is unsupported.")
    _validate_positive_int(
        "learned_binned_position_layout.coordinate_origin",
        layout.coordinate_origin,
    )
    if layout.coordinate_origin != LEARNED_BINNED_COORDINATE_ORIGIN:
        raise ValueError("learned_binned_position_layout.coordinate_origin must be 1.")
    if not isinstance(layout.layout, str):
        raise ValueError("learned_binned_position_layout.layout must be a string.")
    if layout.layout != LEARNED_BINNED_LAYOUT:
        raise ValueError(
            "learned_binned_position_layout.layout must be chromosome_local_contiguous."
        )
    _validate_positive_int(
        "learned_binned_position_layout.num_embeddings",
        layout.num_embeddings,
    )
    _validate_positive_int("position_encoding.absolute.bin_size_bp", bin_size_bp)
    _validate_positive_int("position_encoding.absolute.position_dim", position_dim)
    _validate_positive_int("position_encoding.chromosome.num_chromosomes", num_chromosomes)
    if len(layout.chromosome_lengths_bp) != num_chromosomes:
        raise ValueError(
            "learned_binned_position_layout.chromosome_lengths_bp length must match "
            "position_encoding.chromosome.num_chromosomes."
        )
    if len(layout.bins_per_chromosome) != num_chromosomes:
        raise ValueError(
            "learned_binned_position_layout.bins_per_chromosome length must match "
            "position_encoding.chromosome.num_chromosomes."
        )
    if layout.num_embeddings != sum(layout.bins_per_chromosome):
        raise ValueError(
            "learned_binned_position_layout.num_embeddings must equal sum(bins_per_chromosome)."
        )
    for idx, (length, bins) in enumerate(
        zip(layout.chromosome_lengths_bp, layout.bins_per_chromosome, strict=True)
    ):
        _validate_positive_int(
            f"learned_binned_position_layout.chromosome_lengths_bp[{idx}]",
            length,
        )
        _validate_positive_int(
            f"learned_binned_position_layout.bins_per_chromosome[{idx}]",
            bins,
        )
        expected_bins = (length + bin_size_bp - 1) // bin_size_bp
        if bins != expected_bins:
            raise ValueError(
                "learned_binned_position_layout.bins_per_chromosome must match "
                "chromosome_lengths_bp and bin_size_bp."
            )


def validate_model_runtime_support(
    config: ResolvedPositionEncodingConfig,
    *,
    learned_binned_position_layout: LearnedBinnedAbsolutePositionLayout | None = None,
) -> None:
    """Validate positional strategies executed by SIEVE as a complete model."""
    validate_attention_runtime_support(config)
    if config.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        if config.absolute.bin_size_bp is None or config.absolute.position_dim is None:
            raise ValueError("resolved learned_binned absolute-position settings are incomplete.")
        validate_learned_binned_layout_settings(
            learned_binned_position_layout,
            bin_size_bp=config.absolute.bin_size_bp,
            position_dim=config.absolute.position_dim,
            num_chromosomes=config.chromosome.num_chromosomes,
        )
        return
    if learned_binned_position_layout is not None:
        raise ValueError(
            "learned_binned_position_layout is only valid for absolute_position_encoding=learned_binned."
        )
    if config.absolute.encoding not in {
        AbsolutePositionEncoding.NONE,
        AbsolutePositionEncoding.SINUSOIDAL,
    }:
        raise NotImplementedError(
            f"absolute_position_encoding={config.absolute.encoding.value} is not implemented."
        )


def validate_phase7_runtime_support(config: ResolvedPositionEncodingConfig) -> None:
    """Validate that a resolved config is supported by external Phase 7 entry points."""
    validate_attention_runtime_support(config)
    if config.relative.encoding is RelativePositionEncoding.ROPE:
        raise NotImplementedError("relative_position_encoding=rope is not supported by Phase 7.")
    if config.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        raise NotImplementedError("absolute_position_encoding=learned_binned is not implemented.")
    if config.absolute.encoding not in {
        AbsolutePositionEncoding.NONE,
        AbsolutePositionEncoding.SINUSOIDAL,
    }:
        raise NotImplementedError(
            f"absolute_position_encoding={config.absolute.encoding.value} is not implemented."
        )


def build_absolute_position_runtime(
    config: ResolvedPositionEncodingConfig,
    *,
    learned_binned_position_layout: LearnedBinnedAbsolutePositionLayout | None = None,
    absolute_position_embedding: nn.Embedding | None = None,
) -> AbsolutePositionRuntime:
    """Build the absolute-position runtime for a resolved config."""
    validate_model_runtime_support(
        config,
        learned_binned_position_layout=learned_binned_position_layout,
    )
    if config.preset is PositionPreset.LEGACY:
        return ObservedAbsolutePositionRuntime()
    if config.absolute.encoding is AbsolutePositionEncoding.NONE:
        return NoAbsolutePositionRuntime()
    if config.absolute.encoding is AbsolutePositionEncoding.SINUSOIDAL:
        if (
            config.absolute.position_dim is None
            or config.absolute.coordinate_scale is None
            or config.absolute.max_wavelength is None
        ):
            raise ValueError("resolved sinusoidal absolute-position settings are incomplete.")
        return SinusoidalAbsolutePositionRuntime(
            position_dim=config.absolute.position_dim,
            coordinate_scale=config.absolute.coordinate_scale,
            max_wavelength=config.absolute.max_wavelength,
        )
    if config.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        if absolute_position_embedding is None:
            raise ValueError("absolute_position_embedding is required for learned_binned.")
        return LearnedBinnedAbsolutePositionRuntime(
            embedding=absolute_position_embedding,
            layout=learned_binned_position_layout,
            bin_size_bp=config.absolute.bin_size_bp,
            position_dim=config.absolute.position_dim,
        )
    raise NotImplementedError(
        f"absolute_position_encoding={config.absolute.encoding.value} is not implemented."
    )


def build_relative_position_runtime(
    config: ResolvedPositionEncodingConfig,
    *,
    head_dim: int | None = None,
) -> RelativePositionRuntime:
    """Build the parameterless relative-position runtime for a resolved config."""
    validate_attention_runtime_support(config)
    if config.preset is PositionPreset.LEGACY:
        if config.relative.num_buckets is None or config.relative.max_distance_bp is None:
            raise ValueError("resolved legacy T5 settings are incomplete.")
        return LegacyT5RelativePositionRuntime(
            num_position_buckets=config.relative.num_buckets,
            max_distance=config.relative.max_distance_bp,
        )
    if config.relative.encoding is RelativePositionEncoding.NONE:
        return NoRelativePositionRuntime()
    if config.relative.encoding is RelativePositionEncoding.T5_BUCKET:
        if config.relative.num_buckets is None or config.relative.max_distance_bp is None:
            raise ValueError("resolved T5 settings are incomplete.")
        return T5RelativePositionRuntime(
            num_position_buckets=config.relative.num_buckets,
            max_distance=config.relative.max_distance_bp,
            cross_chromosome_policy=config.chromosome.cross_chromosome_policy,
        )
    if config.relative.encoding is RelativePositionEncoding.ROPE:
        if head_dim is None:
            raise ValueError("head_dim is required for relative_position_encoding=rope.")
        if config.relative.rope_coordinate_scale is None or config.relative.rope_base is None:
            raise ValueError("resolved RoPE settings are incomplete.")
        return RopeRelativePositionRuntime(
            head_dim=head_dim,
            coordinate_scale=config.relative.rope_coordinate_scale,
            rope_base=config.relative.rope_base,
            cross_chromosome_policy=config.chromosome.cross_chromosome_policy,
        )
    raise NotImplementedError(
        f"relative_position_encoding={config.relative.encoding.value} is not implemented."
    )


def _validate_positive_int(field_name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _validate_positive_finite_number(field_name: str, value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be positive and finite.")


def _validate_t5_settings(num_position_buckets: int, max_distance: int) -> None:
    _validate_positive_int("num_position_buckets", num_position_buckets)
    _validate_positive_int("max_distance", max_distance)
    if num_position_buckets < 4:
        raise ValueError("num_position_buckets must be at least 4.")
    if num_position_buckets % 2 != 0:
        raise ValueError("num_position_buckets must be even.")
    if max_distance <= num_position_buckets // 4:
        raise ValueError("max_distance must be greater than num_position_buckets // 4.")


def _is_integer_tensor(value: Tensor) -> bool:
    return value.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
