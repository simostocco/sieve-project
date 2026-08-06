"""
Torch feature composition helpers for legacy SIEVE model execution.

The helpers in this module contain no learned state. They exist to rebuild the
historical VariantEncoder input ordering from the split content and absolute
position tensors introduced by the positional-encoding migration.
"""

import torch
from torch import Tensor


def compose_legacy_variant_features_torch(
    content_features: Tensor,
    absolute_position_features: Tensor,
) -> Tensor:
    """
    Compose historical VariantEncoder features from split tensors.

    This is the model-side Torch equivalent of the Phase 5B1 NumPy composer. It
    performs deterministic column composition only, contains no learned state,
    and preserves the legacy VariantEncoder ordering:
    ``[dosage, absolute_position, remaining_content]``. For L0, the absolute
    position tensor has zero width and ``content_features`` is returned directly.

    Parameters
    ----------
    content_features : Tensor
        Content tensor with shape ``[..., content_dim]``. The final dimension
        must contain at least the dosage column.
    absolute_position_features : Tensor
        Historical absolute-position tensor with shape ``[..., position_dim]``.
        For L0, ``position_dim`` is zero.

    Returns
    -------
    Tensor
        Historical feature tensor with the same leading dimensions, dtype, and
        device as the inputs.
    """
    if not isinstance(content_features, torch.Tensor):
        raise ValueError(
            "content_features must be a torch.Tensor; " f"got {type(content_features).__name__}"
        )
    if not isinstance(absolute_position_features, torch.Tensor):
        raise ValueError(
            "absolute_position_features must be a torch.Tensor; "
            f"got {type(absolute_position_features).__name__}"
        )
    if content_features.ndim != absolute_position_features.ndim:
        raise ValueError(
            "content_features and absolute_position_features must have the same rank; "
            f"got {content_features.ndim} and {absolute_position_features.ndim}"
        )
    if content_features.ndim < 2:
        raise ValueError(
            "content_features and absolute_position_features must have rank at least 2; "
            f"got rank {content_features.ndim}"
        )
    if content_features.shape[:-1] != absolute_position_features.shape[:-1]:
        raise ValueError(
            "content_features and absolute_position_features leading dimensions "
            "must match; got "
            f"{tuple(content_features.shape[:-1])} and "
            f"{tuple(absolute_position_features.shape[:-1])}"
        )
    if content_features.dtype != absolute_position_features.dtype:
        raise ValueError(
            "content_features and absolute_position_features must have the same dtype; "
            f"got {content_features.dtype} and {absolute_position_features.dtype}"
        )
    if content_features.device != absolute_position_features.device:
        raise ValueError(
            "content_features and absolute_position_features must be on the same device; "
            f"got {content_features.device} and {absolute_position_features.device}"
        )
    if content_features.shape[-1] < 1:
        raise ValueError("content_features must have at least one final-dimension column")

    if absolute_position_features.shape[-1] == 0:
        return content_features

    return torch.cat(
        [
            content_features[..., :1],
            absolute_position_features,
            content_features[..., 1:],
        ],
        dim=-1,
    )
