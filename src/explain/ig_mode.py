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
    is_new_schema: bool | None = None,
) -> ResolvedIGMode:
    """
    Resolve an IG mode from a request and an authoritative config mapping.

    ``is_new_schema`` is the execution-authority result from checkpoint
    reconstruction. When omitted, the historical Python API is preserved:
    ``position_encoding`` presence is treated as new schema. Explanation code
    should pass the reconstruction result explicitly so transitional configs
    with non-executed positional metadata keep historical attribution policy.
    """
    mode = _coerce_requested_mode(requested_mode)
    if is_new_schema is None:
        is_new_schema = "position_encoding" in config
    elif type(is_new_schema) is not bool:
        raise ValueError("is_new_schema must be None or an exact boolean")

    if is_new_schema:
        saved_default = _read_saved_default_ig_mode(config)
        if mode is RequestedIGMode.AUTO:
            return saved_default
        if mode is RequestedIGMode.CONTENT:
            return ResolvedIGMode.CONTENT
        if mode is RequestedIGMode.LEGACY:
            return ResolvedIGMode.LEGACY

    if mode is RequestedIGMode.AUTO:
        warnings.warn(
            _compatibility_warning(config, explicit_content=False),
            IGModeCompatibilityWarning,
            stacklevel=2,
        )
        return ResolvedIGMode.LEGACY
    if mode is RequestedIGMode.CONTENT:
        warnings.warn(
            _compatibility_warning(config, explicit_content=True),
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


def _compatibility_warning(
    config: Mapping[str, object],
    *,
    explicit_content: bool,
) -> str:
    if "position_encoding" in config:
        if explicit_content:
            return (
                "Config contains position_encoding metadata, but checkpoint "
                "reconstruction identified historical/transitional execution; "
                "using explicit content-only attribution override. This "
                "requires split tensors from the current dataset."
            )
        return (
            "Config contains position_encoding metadata, but checkpoint "
            "reconstruction identified historical/transitional execution; "
            "resolving ig_mode='auto' to historical legacy attribution."
        )
    if explicit_content:
        return (
            "Old config has no position_encoding metadata; using explicit "
            "content-only attribution override. This requires split tensors "
            "from the current dataset."
        )
    return (
        "Old config has no position_encoding metadata; resolving "
        "ig_mode='auto' to historical legacy attribution."
    )
