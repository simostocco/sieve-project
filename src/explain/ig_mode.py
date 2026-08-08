"""
Pure Integrated Gradients mode resolution for positional attribution.

This module intentionally does not import Torch, Captum, model code, datasets,
or filesystem helpers. It resolves an already-authoritative configuration
mapping into the attribution mode used by explanation code.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum

from src.encoding.position_config import ResolvedIGMode


class RequestedIGMode(str, Enum):
    """User-requested Integrated Gradients attribution mode."""

    AUTO = "auto"
    CONTENT = "content"
    LEGACY = "legacy"


class IGModeCompatibilityWarning(UserWarning):
    """Warning emitted when historical configs require compatibility behavior."""


def resolve_ig_mode(
    requested_mode: RequestedIGMode | str,
    *,
    config: Mapping[str, object],
) -> ResolvedIGMode:
    """
    Resolve an IG mode from a request and an authoritative config mapping.

    New-schema configs contain ``position_encoding`` and must include valid
    nested attribution metadata, even when the caller explicitly requests a
    concrete mode. Old configs lack ``position_encoding``; ``auto`` resolves to
    legacy attribution with a compatibility warning, while an explicit content
    override is allowed but warned because it depends on split tensors from the
    current dataset.
    """
    mode = _coerce_requested_mode(requested_mode)

    if "position_encoding" in config:
        saved_default = _read_saved_default_ig_mode(config)
        if mode is RequestedIGMode.AUTO:
            return saved_default
        if mode is RequestedIGMode.CONTENT:
            return ResolvedIGMode.CONTENT
        if mode is RequestedIGMode.LEGACY:
            return ResolvedIGMode.LEGACY

    if mode is RequestedIGMode.AUTO:
        # Old configs predate the content/position split, so automatic mode
        # must preserve the historical attribution target instead of guessing.
        warnings.warn(
            "Old config has no position_encoding metadata; resolving "
            "ig_mode='auto' to historical legacy attribution.",
            IGModeCompatibilityWarning,
            stacklevel=2,
        )
        return ResolvedIGMode.LEGACY
    if mode is RequestedIGMode.CONTENT:
        warnings.warn(
            "Old config has no position_encoding metadata; using explicit "
            "content-only attribution override. This requires split tensors "
            "from the current dataset.",
            IGModeCompatibilityWarning,
            stacklevel=2,
        )
        return ResolvedIGMode.CONTENT
    if mode is RequestedIGMode.LEGACY:
        return ResolvedIGMode.LEGACY

    raise ValueError(f"unsupported ig_mode: {requested_mode!r}")


def _coerce_requested_mode(requested_mode: RequestedIGMode | str) -> RequestedIGMode:
    if isinstance(requested_mode, RequestedIGMode):
        return requested_mode
    if isinstance(requested_mode, str):
        try:
            return RequestedIGMode(requested_mode)
        except ValueError as exc:
            raise ValueError("requested_mode must be one of: auto, content, legacy") from exc
    raise ValueError("requested_mode must be a RequestedIGMode or exact mode string")


def _read_saved_default_ig_mode(config: Mapping[str, object]) -> ResolvedIGMode:
    position_encoding = config["position_encoding"]
    if not isinstance(position_encoding, Mapping):
        raise ValueError("position_encoding must be a mapping")

    if "attribution" not in position_encoding:
        raise ValueError("position_encoding.attribution is required")
    attribution = position_encoding["attribution"]
    if not isinstance(attribution, Mapping):
        raise ValueError("position_encoding.attribution must be a mapping")

    if "default_ig_mode" not in attribution:
        raise ValueError("position_encoding.attribution.default_ig_mode is required")
    saved_default = attribution["default_ig_mode"]
    if saved_default == ResolvedIGMode.CONTENT.value:
        return ResolvedIGMode.CONTENT
    if saved_default == ResolvedIGMode.LEGACY.value:
        return ResolvedIGMode.LEGACY
    raise ValueError("position_encoding.attribution.default_ig_mode must be 'content' or 'legacy'")
