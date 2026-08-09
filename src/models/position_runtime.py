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
    ) -> Tensor:
        """Return the exact base score tensor object unchanged."""
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
    ) -> Tensor:
        """Add custom T5-style relative bias to precomputed QK scores."""
        bias = self.compute_bias(
            positions,
            positions,
            position_bias,
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        )
        return base_scores + bias


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


def validate_phase7_runtime_support(config: ResolvedPositionEncodingConfig) -> None:
    """Validate that a resolved config is supported by the Phase 7 runtime subset."""
    if not isinstance(config, ResolvedPositionEncodingConfig):
        raise ValueError("config must be a ResolvedPositionEncodingConfig.")
    if config.preset not in {PositionPreset.LEGACY, PositionPreset.CUSTOM}:
        raise NotImplementedError(f"position preset {config.preset!r} is not supported.")
    if config.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED:
        raise NotImplementedError("absolute_position_encoding=learned_binned is not implemented.")
    if config.absolute.encoding not in {
        AbsolutePositionEncoding.NONE,
        AbsolutePositionEncoding.SINUSOIDAL,
    }:
        raise NotImplementedError(
            f"absolute_position_encoding={config.absolute.encoding.value} is not implemented."
        )
    if config.relative.encoding in {
        RelativePositionEncoding.ROPE,
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }:
        raise NotImplementedError(
            f"relative_position_encoding={config.relative.encoding.value} is not implemented."
        )
    if config.relative.encoding not in {
        RelativePositionEncoding.NONE,
        RelativePositionEncoding.T5_BUCKET,
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


def build_absolute_position_runtime(
    config: ResolvedPositionEncodingConfig,
) -> AbsolutePositionRuntime:
    """Build the parameterless absolute-position runtime for a resolved config."""
    validate_phase7_runtime_support(config)
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
    raise NotImplementedError(
        f"absolute_position_encoding={config.absolute.encoding.value} is not implemented."
    )


def build_relative_position_runtime(
    config: ResolvedPositionEncodingConfig,
) -> RelativePositionRuntime:
    """Build the parameterless relative-position runtime for a resolved config."""
    validate_phase7_runtime_support(config)
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
