"""Config and checkpoint reconstruction policy for SIEVE positional schemas.

This module is intentionally separate from ``SIEVE`` and attention execution.
It decides how a serialized config/checkpoint pair should be interpreted, then
uses the approved model constructors without changing forward behavior.

Case A is the only format produced by new training from Phase 7B3 onward:
schema-v2 configs whose resolved position encoding actually constructed the
model. These checkpoints load with ``strict=True``.

Cases B and C are read-only compatibility paths for historical checkpoints.
They reconstruct no-config historical SIEVE models from checkpoint state and
allow only the known 32-row to 33-row T5 position-bias migration.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    ResolvedPositionEncodingConfig,
    resolved_position_encoding_from_dict,
)
from src.encoding.position_layout import (
    LearnedBinnedAbsolutePositionLayout,
    learned_binned_layout_from_position_encoding_dict,
    validate_saved_chromosome_mapping_matches_chrom_index,
)
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.position_runtime import validate_model_runtime_support
from src.models.sieve import SIEVE, load_state_dict_with_legacy_upgrade


@dataclass(frozen=True)
class ReconstructedSIEVEModel:
    """Model plus provenance returned by checkpoint reconstruction."""

    model: nn.Module
    base_model: SIEVE
    resolved_position_encoding: ResolvedPositionEncodingConfig | None
    is_new_schema: bool
    is_chunked_checkpoint: bool
    effective_config: dict[str, object]


def reconstruct_sieve_from_checkpoint(
    config: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    num_genes: int,
    dataset_num_chromosomes: int | None = None,
    dataset_chrom_index: Mapping[str, int] | None = None,
) -> ReconstructedSIEVEModel:
    """Reconstruct a SIEVE or ChunkedSIEVEModel from in-memory metadata.

    The caller supplies already loaded config and checkpoint mappings. New
    schema configs are exact architecture contracts and load strictly.
    Historical and transitional configs infer architecture from checkpoint
    tensors because old configs did not reliably record positional state.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_dict = _checkpoint_state_dict(checkpoint)
    topology = _detect_state_topology(state_dict)

    case = _classify_reconstruction_case(config, checkpoint)
    if case == "A":
        return _reconstruct_case_a(
            config,
            checkpoint,
            state_dict=state_dict,
            is_chunked=topology == "chunked",
            num_genes=num_genes,
            dataset_num_chromosomes=dataset_num_chromosomes,
            dataset_chrom_index=dataset_chrom_index,
        )
    return _reconstruct_compatibility_case(
        config,
        state_dict=state_dict,
        is_chunked=topology == "chunked",
        num_genes=num_genes,
    )


def _checkpoint_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint.model_state_dict is required")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint.model_state_dict must be a mapping")
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint.model_state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint.model_state_dict[{key!r}] must be a tensor")
    return state_dict


def _classify_reconstruction_case(
    config: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> str:
    has_position_encoding = "position_encoding" in config
    if not has_position_encoding:
        return "B"
    if "config_schema_version" in config:
        schema_version = _strict_int_value(
            config["config_schema_version"],
            "config_schema_version",
        )
        if schema_version == 2:
            return "A"

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping) and "config_schema_version" in metadata:
        metadata_schema_version = _strict_int_value(
            metadata["config_schema_version"],
            "checkpoint.metadata.config_schema_version",
        )
        if metadata_schema_version == 2:
            raise ValueError(
                "transitional config cannot be promoted by schema-v2 checkpoint metadata"
            )
    execution = config.get("position_encoding_execution")
    if not isinstance(execution, Mapping):
        raise ValueError(
            "pre-v2 config with position_encoding requires position_encoding_execution"
        )
    applied = execution.get("resolved_config_applied_to_model")
    if not isinstance(applied, bool):
        raise ValueError(
            "position_encoding_execution.resolved_config_applied_to_model must be boolean"
        )
    if applied:
        raise ValueError(
            "pre-v2 position_encoding metadata claims resolved config execution; "
            "schema-v2 config is required"
        )
    return "C"


def _detect_state_topology(state_dict: Mapping[str, torch.Tensor]) -> str:
    if not state_dict:
        raise ValueError("checkpoint.model_state_dict must not be empty")
    prefixed = [key.startswith("base_model.") for key in state_dict]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed base_model-prefixed and unprefixed state_dict keys")
    return "chunked" if all(prefixed) else "base"


def _reconstruct_case_a(
    config: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    state_dict: Mapping[str, torch.Tensor],
    is_chunked: bool,
    num_genes: int,
    dataset_num_chromosomes: int | None,
    dataset_chrom_index: Mapping[str, int] | None,
) -> ReconstructedSIEVEModel:
    effective_config = _reconcile_case_a_config(config, checkpoint.get("metadata"))
    _require_top_level(effective_config, "config_schema_version")
    if _strict_int_value(effective_config["config_schema_version"], "config_schema_version") != 2:
        raise ValueError("config_schema_version must be 2 for new-schema reconstruction")

    latent_dim = _int_config(effective_config, "latent_dim", 64)
    num_heads = _int_config(effective_config, "num_heads", 4)
    position_data = _mapping_config(effective_config, "position_encoding")
    resolved = resolved_position_encoding_from_dict(
        position_data,
        latent_dim=latent_dim,
        num_heads=num_heads,
    )
    _validate_case_a_chromosome_row_identity(
        resolved,
        position_data,
        dataset_num_chromosomes=dataset_num_chromosomes,
        dataset_chrom_index=dataset_chrom_index,
    )
    learned_binned_position_layout = _case_a_learned_binned_layout(
        resolved,
        position_data,
    )
    validate_model_runtime_support(
        resolved,
        learned_binned_position_layout=learned_binned_position_layout,
    )
    _validate_case_a_structure(
        effective_config,
        resolved,
        num_genes=num_genes,
        dataset_num_chromosomes=dataset_num_chromosomes,
    )
    base_model = _build_base_model(
        effective_config,
        input_dim=resolved.input_dim,
        num_genes=num_genes,
        num_chromosomes=resolved.chromosome.num_chromosomes,
        position_encoding=resolved,
        learned_binned_position_layout=learned_binned_position_layout,
    )
    model = _wrap_if_chunked(base_model, effective_config, is_chunked=is_chunked)
    model.load_state_dict(dict(state_dict), strict=True)
    return ReconstructedSIEVEModel(
        model=model,
        base_model=base_model,
        resolved_position_encoding=resolved,
        is_new_schema=True,
        is_chunked_checkpoint=is_chunked,
        effective_config=effective_config,
    )


def _reconstruct_compatibility_case(
    config: Mapping[str, object],
    *,
    state_dict: Mapping[str, torch.Tensor],
    is_chunked: bool,
    num_genes: int,
) -> ReconstructedSIEVEModel:
    effective_config = copy.deepcopy(dict(config))
    input_dim = _infer_old_input_dim(state_dict)
    if "input_dim" in effective_config and effective_config["input_dim"] != input_dim:
        raise ValueError(
            "old config input_dim conflicts with checkpoint state: "
            f"{effective_config['input_dim']} != {input_dim}"
        )
    effective_config["input_dim"] = input_dim

    latent_dim = _int_config(effective_config, "latent_dim", 64)
    num_chromosomes = _infer_old_num_chromosomes(state_dict, latent_dim=latent_dim)
    effective_config["num_chromosomes"] = num_chromosomes

    base_model = _build_base_model(
        effective_config,
        input_dim=input_dim,
        num_genes=num_genes,
        num_chromosomes=num_chromosomes,
        position_encoding=None,
    )
    model = _wrap_if_chunked(base_model, effective_config, is_chunked=is_chunked)
    _preflight_compatibility_load(model.state_dict(), state_dict)
    load_state_dict_with_legacy_upgrade(model, dict(state_dict))
    return ReconstructedSIEVEModel(
        model=model,
        base_model=base_model,
        resolved_position_encoding=None,
        is_new_schema=False,
        is_chunked_checkpoint=is_chunked,
        effective_config=effective_config,
    )


def _build_base_model(
    config: Mapping[str, object],
    *,
    input_dim: int,
    num_genes: int,
    num_chromosomes: int,
    position_encoding: ResolvedPositionEncodingConfig | None,
    learned_binned_position_layout: LearnedBinnedAbsolutePositionLayout | None = None,
) -> SIEVE:
    return SIEVE(
        input_dim=input_dim,
        num_genes=num_genes,
        latent_dim=_int_config(config, "latent_dim", 64),
        hidden_dim=_int_config(config, "hidden_dim", 128),
        num_heads=_int_config(config, "num_heads", 4),
        num_attention_layers=_int_config(config, "num_attention_layers", 2),
        classifier_hidden_dim=_int_config(config, "classifier_hidden_dim", 256),
        dropout=float(config.get("dropout", 0.1)),
        aggregation=str(config.get("aggregation", "max")),
        num_position_buckets=_nullable_int_config(config, "num_position_buckets", 32),
        max_distance=_nullable_int_config(config, "max_distance", 100000),
        num_covariates=_int_config(config, "num_covariates", 0),
        num_chromosomes=num_chromosomes,
        classifier_type=str(config.get("classifier_type", "flatten")),
        position_encoding=position_encoding,
        learned_binned_position_layout=learned_binned_position_layout,
    )


def _wrap_if_chunked(
    base_model: SIEVE,
    config: Mapping[str, object],
    *,
    is_chunked: bool,
) -> nn.Module:
    if not is_chunked:
        return base_model
    return ChunkedSIEVEModel(
        base_model=base_model,
        aggregation_method=str(config.get("aggregation_method", "mean")),
    )


def _reconcile_case_a_config(
    config: Mapping[str, object],
    metadata: object,
) -> dict[str, object]:
    effective = copy.deepcopy(dict(config))
    if metadata is None:
        return effective
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint.metadata must be a mapping when present")
    _merge_allowed_metadata(effective, metadata, path="")
    return effective


def _merge_allowed_metadata(
    target: dict[str, object],
    metadata: Mapping[str, object],
    *,
    path: str,
) -> None:
    allowed = {
        "config_schema_version",
        "metadata_schema_version",
        "position_encoding_schema_version",
        "input_dim",
        "content_dim",
        "num_genes",
        "num_chromosomes",
        "position_encoding",
        "dataset_identity",
        "position_encoding_execution",
    }
    for key in allowed:
        if key not in metadata:
            continue
        dotted = f"{path}.{key}" if path else key
        value = metadata[key]
        if key not in target:
            target[key] = copy.deepcopy(value)
            continue
        existing = target[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            nested = copy.deepcopy(dict(existing))
            _merge_mapping_exact(nested, value, path=dotted)
            target[key] = nested
        elif not _same_scalar_value(existing, value):
            raise ValueError(f"checkpoint metadata conflict at {dotted}")


def _merge_mapping_exact(
    target: dict[str, object],
    metadata: Mapping[str, object],
    *,
    path: str,
) -> None:
    for key, value in metadata.items():
        dotted = f"{path}.{key}"
        if key not in target:
            target[key] = copy.deepcopy(value)
            continue
        existing = target[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            nested = copy.deepcopy(dict(existing))
            _merge_mapping_exact(nested, value, path=dotted)
            target[key] = nested
        elif not _same_scalar_value(existing, value):
            raise ValueError(f"checkpoint metadata conflict at {dotted}")


def _case_a_learned_binned_layout(
    resolved: ResolvedPositionEncodingConfig,
    position_data: Mapping[str, object],
) -> LearnedBinnedAbsolutePositionLayout | None:
    """Return the authoritative learned-binned layout for schema-v2 execution.

    For learned-binned checkpoints, ``position_encoding.absolute.binning`` is
    the model architecture contract. Reconstruction must allocate the embedding
    table from that saved layout before strict state loading; checkpoint tensor
    shapes are not a source of layout inference or migration.
    """
    if resolved.absolute.encoding is not AbsolutePositionEncoding.LEARNED_BINNED:
        return None
    chromosome_mapping = _case_a_saved_chromosome_mapping(position_data)

    return learned_binned_layout_from_position_encoding_dict(
        position_data,
        chromosome_mapping=chromosome_mapping,
    )


def _validate_case_a_chromosome_row_identity(
    resolved: ResolvedPositionEncodingConfig,
    position_data: Mapping[str, object],
    *,
    dataset_num_chromosomes: int | None,
    dataset_chrom_index: Mapping[str, int] | None,
) -> None:
    """Validate schema-v2 chromosome row identity when model state is row-indexed.

    Learned chromosome embeddings are indexed directly by ``chrom_id``.
    Learned-binned absolute-position tables are also chromosome-ID dependent.
    For those strategies, chromosome IDs carry learned row identity and a live
    dataset must use the same chromosome-name-to-ID mapping as training. RoPE
    and T5 cross-chromosome routing alone do not require exact chromosome-name
    identity because they do not own chromosome-row-indexed parameters.
    """
    if not _requires_exact_chromosome_mapping(resolved):
        return

    saved_mapping = _case_a_saved_chromosome_mapping(position_data)
    if dataset_chrom_index is not None:
        validate_saved_chromosome_mapping_matches_chrom_index(
            saved_mapping,
            dataset_chrom_index,
        )
    elif dataset_num_chromosomes is not None:
        raise ValueError(
            "dataset_chrom_index is required for schema-v2 reconstruction when "
            "chromosome row-indexed state is present"
        )


def _requires_exact_chromosome_mapping(
    resolved: ResolvedPositionEncodingConfig,
) -> bool:
    return (
        resolved.chromosome.encoding is ChromosomeEncoding.LEARNED
        or resolved.absolute.encoding is AbsolutePositionEncoding.LEARNED_BINNED
    )


def _case_a_saved_chromosome_mapping(
    position_data: Mapping[str, object],
) -> Mapping[str, object]:
    chromosome = position_data.get("chromosome")
    if not isinstance(chromosome, Mapping):
        raise ValueError("position_encoding.chromosome must be a mapping")
    chromosome_mapping = chromosome.get("mapping")
    if not isinstance(chromosome_mapping, Mapping):
        raise ValueError("position_encoding.chromosome.mapping must be a mapping")
    return chromosome_mapping


def _validate_case_a_structure(
    config: Mapping[str, object],
    resolved: ResolvedPositionEncodingConfig,
    *,
    num_genes: int,
    dataset_num_chromosomes: int | None,
) -> None:
    expectations = {
        "input_dim": resolved.input_dim,
        "content_dim": resolved.content_dim,
        "num_chromosomes": resolved.chromosome.num_chromosomes,
        "position_encoding_schema_version": resolved.schema_version,
    }
    for key, expected in expectations.items():
        _require_top_level(config, key)
        if _strict_int_value(config[key], key) != expected:
            raise ValueError(f"{key} conflicts with resolved position encoding")
    if "num_genes" in config:
        if _strict_int_value(config["num_genes"], "num_genes") != num_genes:
            raise ValueError("num_genes conflicts with requested reconstruction num_genes")
    if dataset_num_chromosomes is None:
        return
    if (
        _strict_int_value(dataset_num_chromosomes, "dataset_num_chromosomes")
        != resolved.chromosome.num_chromosomes
    ):
        raise ValueError("dataset_num_chromosomes conflicts with resolved chromosome count")


def _infer_old_input_dim(state_dict: Mapping[str, torch.Tensor]) -> int:
    matches = [
        tensor
        for key, tensor in state_dict.items()
        if key.endswith("variant_encoder.encoder.0.weight")
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint must contain one variant_encoder.encoder.0.weight tensor")
    tensor = matches[0]
    if tensor.ndim != 2:
        raise ValueError("variant_encoder.encoder.0.weight must be rank 2")
    return int(tensor.shape[1])


def _infer_old_num_chromosomes(
    state_dict: Mapping[str, torch.Tensor],
    *,
    latent_dim: int,
) -> int:
    tensors = [
        tensor for key, tensor in state_dict.items() if key.endswith("chrom_embedding.weight")
    ]
    if not tensors:
        return 0
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("old checkpoint chrom_embedding.weight shapes disagree")
    rows, width = next(iter(shapes))
    if rows < 1:
        raise ValueError("old checkpoint chrom_embedding.weight must have at least one row")
    if width != latent_dim:
        raise ValueError("old checkpoint chrom_embedding.weight width conflicts with latent_dim")
    return int(rows - 1)


def _preflight_compatibility_load(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> None:
    target_keys = set(target_state)
    source_keys = set(source_state)
    missing = target_keys - source_keys
    unexpected = source_keys - target_keys
    if missing:
        raise ValueError(f"old checkpoint missing model key: {sorted(missing)[0]}")
    if unexpected:
        raise ValueError(f"old checkpoint has unexpected model key: {sorted(unexpected)[0]}")

    allowed_migrations = 0
    for key in sorted(target_keys):
        target = target_state[key]
        source = source_state[key]
        if target.shape == source.shape:
            continue
        if _is_allowed_position_bias_migration(key, source, target):
            allowed_migrations += 1
            continue
        raise ValueError(
            f"old checkpoint tensor shape mismatch for {key}: "
            f"{tuple(source.shape)} != {tuple(target.shape)}"
        )
    if allowed_migrations:
        return


def _is_allowed_position_bias_migration(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
) -> bool:
    return (
        key.endswith("position_bias.weight")
        and source.ndim == 2
        and target.ndim == 2
        and source.shape[0] == 32
        and target.shape[0] == 33
        and source.shape[1] == target.shape[1]
    )


def _require_top_level(config: Mapping[str, object], key: str) -> None:
    if key not in config:
        raise ValueError(f"{key} is required for new-schema reconstruction")


def _mapping_config(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    _require_top_level(config, key)
    value = config[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _int_config(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _nullable_int_config(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer when not None")
    return value


def _strict_int_value(value: object, dotted_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{dotted_name} must be an integer")
    return value


def _same_scalar_value(existing: object, value: object) -> bool:
    return type(existing) is type(value) and existing == value
