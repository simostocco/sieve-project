#!/usr/bin/env python3
"""
Training script for SIEVE models.

This script trains a SIEVE model on VCF data with specified annotation level.
Supports single train/val split or k-fold cross-validation.

Usage:
    python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv --level L3
    python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv --level L3 --cv 5

Author: Francesco Lescai
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_sample_variants
from src.data.covariates import (
    attach_pc_covariates_to_samples,
    compute_file_sha256,
    load_pc_map,
)
from src.encoding import (
    AnnotationLevel,
    ChunkedVariantDataset,
    collate_chunks,
    get_feature_dimension
)
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedPositionEncodingConfig,
    resolve_position_encoding_config,
)
from src.models import SIEVE, ChunkedSIEVEModel
from src.models.position_runtime import validate_phase7_runtime_support
from src.training import (
    SIEVELoss,
    Trainer,
    create_stratified_folds,
    print_fold_stats,
)
from src.training.loss import compute_class_weights


def _enum_choices(enum_cls) -> list[str]:
    """Return argparse choices from a string-valued Enum."""
    return [member.value for member in enum_cls]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""
    parser = argparse.ArgumentParser(
        description='Train SIEVE model on VCF data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--vcf', type=str, default=None,
                        help='Path to VCF file (required if not using --preprocessed-data)')
    parser.add_argument('--phenotypes', type=str, default=None,
                        help='Path to phenotypes file (required if not using --preprocessed-data)')
    parser.add_argument('--preprocessed-data', type=str, default=None,
                        help='Path to preprocessed data file (.pt from preprocess.py)')
    parser.add_argument('--level', type=str, required=True,
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                        help='Annotation level to use')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lambda-attr', type=float, default=0.0,
                        help='Embedding sparsity regularisation weight (penalises L2 norms of variant/gene embeddings)')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of batches to accumulate gradients over (simulates larger batch)')
    parser.add_argument('--chunk-size', type=int, default=3000,
                        help='Chunk size for processing variants (default: 3000, ensures whole-genome coverage)')
    parser.add_argument('--chunk-overlap', type=int, default=0,
                        help='Overlap between adjacent chunks (default: 0)')
    parser.add_argument('--aggregation-method', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='How to aggregate chunk outputs into sample predictions')

    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent dimension for embeddings')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-attention-layers', type=int, default=2,
                        help='Number of attention layers')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension in encoder')

    # Position encoding arguments
    parser.add_argument('--position-preset', type=str, default=PositionPreset.LEGACY.value,
                        choices=_enum_choices(PositionPreset),
                        help='Position encoding preset')
    parser.add_argument('--absolute-position-encoding', type=str, default=None,
                        choices=_enum_choices(AbsolutePositionEncoding),
                        help='Absolute position encoding strategy')
    parser.add_argument('--relative-position-encoding', type=str, default=None,
                        choices=_enum_choices(RelativePositionEncoding),
                        help='Relative position encoding strategy')
    parser.add_argument('--chromosome-encoding', type=str, default=None,
                        choices=_enum_choices(ChromosomeEncoding),
                        help='Chromosome encoding strategy')
    parser.add_argument('--cross-chromosome-policy', type=str, default=None,
                        choices=_enum_choices(CrossChromosomePolicy),
                        help='Cross-chromosome attention policy')
    parser.add_argument('--position-dim', type=int, default=None,
                        help='Absolute position embedding dimension')
    parser.add_argument('--sinusoidal-coordinate-scale', type=float, default=None,
                        help='Coordinate scale for sinusoidal absolute position encoding')
    parser.add_argument('--sinusoidal-max-wavelength', type=float, default=None,
                        help='Maximum wavelength for sinusoidal absolute position encoding')
    parser.add_argument('--position-bin-size', type=int, default=None,
                        help='Bin size in base pairs for learned binned absolute position encoding')
    parser.add_argument('--num-position-buckets', type=int, default=None,
                        help='Number of ordinary T5-style relative position buckets')
    parser.add_argument('--max-position-distance', type=int, default=None,
                        help='Maximum distance for T5-style relative position bucketing')
    parser.add_argument('--rope-coordinate-scale', type=float, default=None,
                        help='Coordinate scale for RoPE relative position encoding')
    parser.add_argument('--rope-base', type=float, default=None,
                        help='Base wavelength for RoPE relative position encoding')
    parser.add_argument('--alibi-distance-function', type=str, default=None,
                        choices=_enum_choices(AlibiDistanceFunction),
                        help='Distance transform for ALiBi relative position encoding')
    parser.add_argument('--alibi-distance-scale', type=float, default=None,
                        help='Distance scale for ALiBi relative position encoding')

    # Cross-validation arguments
    parser.add_argument('--cv', '--cv-folds', dest='cv', type=int, default=None,
                        help='Number of CV folds (if None, use single train/val split)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (if not using CV)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: {level}_run)')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Genome build
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build (GRCh37 or GRCh38)')

    # Covariate arguments
    parser.add_argument(
        '--sex-map', type=str, default=None,
        help=(
            'Path to sample_sex.tsv from infer_sex.py. When training directly '
            'from VCF, sex is used for ploidy-aware dosage encoding and also as '
            'a covariate in the classifier head to adjust for sex imbalance '
            'between cases and controls. When using --preprocessed-data, this '
            'script cannot change existing dosages and only uses sex as a '
            'covariate; ensure your preprocessed data were generated with '
            'sex-aware ploidy encoding if required.'
        )
    )
    parser.add_argument(
        '--pc-map',
        type=str,
        default=None,
        help=(
            "Optional TSV with columns: sample_id, PC1, PC2, ... "
            "PCs are concatenated to the covariate vector after sex."
        ),
    )
    parser.add_argument(
        '--num-pcs',
        type=int,
        default=0,
        help='Number of PCs to use from --pc-map (default: 0 = do not use).',
    )

    parser.add_argument(
        '--class-weighting',
        choices=['auto', 'on', 'off'],
        default='auto',
        help=(
            "Whether to apply inverse-frequency class weighting in the BCE loss. "
            "'auto' (default): enabled only if the training fold has case fraction "
            "outside [0.4, 0.6]. 'on': always enabled. 'off': never enabled."
        ),
    )

    parser.add_argument(
        '--classifier-type',
        choices=['flatten', 'attention_pool'],
        default='flatten',
        help=(
            "Classifier head architecture. "
            "'flatten' (default, backward-compatible): flattens gene embeddings and applies "
            "Linear(num_genes*latent_dim → hidden_dim). "
            "'attention_pool': attention-pools gene embeddings to latent_dim before the "
            "classifier, reducing the first linear to Linear(latent_dim → hidden_dim) "
            "and addressing p>>n overparameterisation."
        ),
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args(argv)


def build_position_encoding_request(
    args: argparse.Namespace,
) -> PositionEncodingRequest:
    """Convert parsed CLI strings into a pure position-encoding request."""
    return PositionEncodingRequest(
        preset=PositionPreset(args.position_preset),
        absolute_position_encoding=(
            None
            if args.absolute_position_encoding is None
            else AbsolutePositionEncoding(args.absolute_position_encoding)
        ),
        relative_position_encoding=(
            None
            if args.relative_position_encoding is None
            else RelativePositionEncoding(args.relative_position_encoding)
        ),
        chromosome_encoding=(
            None
            if args.chromosome_encoding is None
            else ChromosomeEncoding(args.chromosome_encoding)
        ),
        cross_chromosome_policy=(
            None
            if args.cross_chromosome_policy is None
            else CrossChromosomePolicy(args.cross_chromosome_policy)
        ),
        position_dim=args.position_dim,
        sinusoidal_coordinate_scale=args.sinusoidal_coordinate_scale,
        sinusoidal_max_wavelength=args.sinusoidal_max_wavelength,
        position_bin_size=args.position_bin_size,
        num_position_buckets=args.num_position_buckets,
        max_position_distance=args.max_position_distance,
        rope_coordinate_scale=args.rope_coordinate_scale,
        rope_base=args.rope_base,
        alibi_distance_function=(
            None
            if args.alibi_distance_function is None
            else AlibiDistanceFunction(args.alibi_distance_function)
        ),
        alibi_distance_scale=args.alibi_distance_scale,
    )


def prepare_training_position_encoding(
    args: argparse.Namespace,
    annotation_level: AnnotationLevel,
    *,
    num_chromosomes: int,
) -> ResolvedPositionEncodingConfig:
    """Resolve and validate the position-encoding configuration used for training."""
    request = build_position_encoding_request(args)
    resolved = resolve_position_encoding_config(
        request,
        annotation_level,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_chromosomes=num_chromosomes,
    )
    validate_phase7_runtime_support(resolved)

    if request.preset is PositionPreset.LEGACY:
        historical_input_dim = get_feature_dimension(annotation_level)
        if resolved.input_dim != historical_input_dim:
            raise ValueError(
                "legacy position configuration resolved input_dim "
                f"{resolved.input_dim}, expected historical input_dim {historical_input_dim}"
            )

    return resolved


def _canonical_index_items(
    mapping: Mapping[str, int],
    *,
    mapping_name: str,
) -> list[tuple[str, int]]:
    """Validate an index mapping and return deterministic name-sorted items."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{mapping_name} must implement Mapping")

    ids = []
    items = []
    for name, idx in mapping.items():
        if not isinstance(name, str):
            raise ValueError(f"{mapping_name} contains a non-string name: {name!r}")
        if isinstance(idx, bool):
            raise ValueError(f"{mapping_name}[{name!r}] must be an integer ID, not bool")
        if not isinstance(idx, int):
            raise ValueError(f"{mapping_name}[{name!r}] must be an integer ID")
        if idx < 0:
            raise ValueError(f"{mapping_name}[{name!r}] must be non-negative")
        ids.append(idx)
        items.append((name, idx))

    if len(set(ids)) != len(ids):
        raise ValueError(f"{mapping_name} must have unique integer IDs")
    expected_ids = set(range(len(items)))
    if set(ids) != expected_ids:
        raise ValueError(
            f"{mapping_name} IDs must be contiguous and exactly range(len(mapping))"
        )

    return sorted(items, key=lambda item: item[0])


def mapping_sha256(
    mapping: Mapping[str, int],
    *,
    mapping_name: str = "mapping",
) -> str:
    """Return a stable SHA-256 checksum for a validated index mapping."""
    canonical_items = _canonical_index_items(mapping, mapping_name=mapping_name)
    payload = [{"name": name, "id": idx} for name, idx in canonical_items]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_chromosome_id_to_name(
    chrom_index: Mapping[str, int],
) -> dict[str, str]:
    """Invert chromosome name-to-ID mapping for serialized model-row metadata."""
    canonical_items = _canonical_index_items(
        chrom_index,
        mapping_name="chrom_index",
    )
    return {
        str(idx): name
        for name, idx in sorted(canonical_items, key=lambda item: item[1])
    }


def build_dataset_mappings_payload(
    gene_index: Mapping[str, int],
    chrom_index: Mapping[str, int],
) -> dict[str, object]:
    """Build the complete deterministic dataset mapping sidecar payload."""
    gene_items = _canonical_index_items(gene_index, mapping_name="gene_index")
    chrom_items = _canonical_index_items(chrom_index, mapping_name="chrom_index")
    return {
        "schema_version": 1,
        "gene_index": {name: idx for name, idx in gene_items},
        "chrom_index": {
            name: idx for name, idx in sorted(chrom_items, key=lambda item: item[1])
        },
        "chromosome_id_to_name": {
            str(idx): name
            for name, idx in sorted(chrom_items, key=lambda item: item[1])
        },
        "gene_mapping_sha256": mapping_sha256(
            gene_index,
            mapping_name="gene_index",
        ),
        "chromosome_mapping_sha256": mapping_sha256(
            chrom_index,
            mapping_name="chrom_index",
        ),
    }


def write_dataset_mappings_artifact(
    output_dir: Path,
    gene_index: Mapping[str, int],
    chrom_index: Mapping[str, int],
) -> dict[str, object]:
    """Write dataset_mappings.json and return lightweight identity metadata."""
    payload = build_dataset_mappings_payload(gene_index, chrom_index)
    artifact_path = output_dir / "dataset_mappings.json"
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return {
        "gene_mapping_sha256": payload["gene_mapping_sha256"],
        "chromosome_mapping_sha256": payload["chromosome_mapping_sha256"],
        "mappings_artifact": "dataset_mappings.json",
        "mappings_artifact_base": "experiment_root",
    }


def serialize_position_encoding_for_training(
    resolved_position_encoding: ResolvedPositionEncodingConfig,
    chrom_index: Mapping[str, int],
) -> dict[str, object]:
    """Serialize resolved position config with chromosome row mapping attached."""
    position_encoding = copy.deepcopy(resolved_position_encoding.to_dict())
    chromosome_mapping = build_chromosome_id_to_name(chrom_index)
    resolved_num_chromosomes = resolved_position_encoding.chromosome.num_chromosomes
    if len(chromosome_mapping) != resolved_num_chromosomes:
        raise ValueError(
            "chrom_index cardinality must match resolved chromosome count "
            f"({len(chromosome_mapping)} != {resolved_num_chromosomes})"
        )
    chromosome_config = position_encoding.setdefault("chromosome", {})
    chromosome_config["mapping"] = chromosome_mapping
    return position_encoding


def build_position_encoding_execution_metadata(
    *,
    resolved_position_encoding: ResolvedPositionEncodingConfig,
    training_mode: Literal["cv", "single_split"],
) -> dict[str, object]:
    """Describe the resolved position configuration applied to new training runs."""
    if training_mode not in {"cv", "single_split"}:
        raise ValueError(f"unsupported training_mode: {training_mode!r}")

    relative = resolved_position_encoding.relative
    chromosome = resolved_position_encoding.chromosome
    position_bias_rows = (
        None
        if relative.encoding is RelativePositionEncoding.NONE
        else relative.total_bias_rows
    )
    return {
        "schema_version": 2,
        "source": "resolved_position_encoding_applied_to_model",
        "resolved_config_applied_to_model": True,
        "training_mode": training_mode,
        "preset": resolved_position_encoding.preset.value,
        "absolute_position_encoding": resolved_position_encoding.absolute.encoding.value,
        "absolute_position_dim": resolved_position_encoding.absolute.position_dim or 0,
        "relative_position_encoding": relative.encoding.value,
        "position_bias_rows": position_bias_rows,
        "chromosome_encoding": chromosome.encoding.value,
        "chromosome_embedding_executed": chromosome.encoding is ChromosomeEncoding.LEARNED,
        "cross_chromosome_policy": chromosome.cross_chromosome_policy.value,
        "cross_chromosome_mask_executed": (
            chromosome.cross_chromosome_policy is CrossChromosomePolicy.MASK
        ),
        "requires_chrom_ids": chromosome.requires_chrom_ids,
        "chrom_ids_passed_to_attention": True,
        "model_num_chromosomes": chromosome.num_chromosomes,
        "input_dim": resolved_position_encoding.input_dim,
        "content_dim": resolved_position_encoding.content_dim,
    }


def build_training_run_metadata(
    *,
    input_dim: int,
    num_genes: int,
    num_chromosomes: int,
    genome_build: str,
    resolved_position_encoding: ResolvedPositionEncodingConfig,
    chrom_index: Mapping[str, int],
    gene_mapping_sha256: str,
    chromosome_mapping_sha256: str,
    training_mode: Literal["cv", "single_split"],
) -> dict[str, object]:
    """Build lightweight run metadata for configs and checkpoints."""
    if input_dim != resolved_position_encoding.input_dim:
        raise ValueError(
            "input_dim must match resolved_position_encoding.input_dim before serialization"
        )
    if num_chromosomes != resolved_position_encoding.chromosome.num_chromosomes:
        raise ValueError(
            "num_chromosomes must match "
            "resolved_position_encoding.chromosome.num_chromosomes before serialization"
        )
    return {
        "config_schema_version": 2,
        "metadata_schema_version": 1,
        "position_encoding_schema_version": resolved_position_encoding.schema_version,
        "input_dim": input_dim,
        "content_dim": resolved_position_encoding.content_dim,
        "num_genes": num_genes,
        "num_chromosomes": num_chromosomes,
        "position_encoding": serialize_position_encoding_for_training(
            resolved_position_encoding,
            chrom_index,
        ),
        "dataset_identity": {
            "genome_build": genome_build,
            "gene_mapping_sha256": gene_mapping_sha256,
            "chromosome_mapping_sha256": chromosome_mapping_sha256,
            "mappings_artifact": "dataset_mappings.json",
            "mappings_artifact_base": "experiment_root",
        },
        "position_encoding_execution": build_position_encoding_execution_metadata(
            resolved_position_encoding=resolved_position_encoding,
            training_mode=training_mode,
        ),
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(
    input_dim: int,
    num_genes: int,
    latent_dim: int,
    num_heads: int,
    num_attention_layers: int,
    hidden_dim: int,
    aggregation_method: str = 'mean',
    num_covariates: int = 0,
    num_chromosomes: int = 0,
    classifier_type: str = 'flatten',
    position_encoding: ResolvedPositionEncodingConfig | None = None,
) -> ChunkedSIEVEModel:
    """
    Create Chunked SIEVE model for whole-genome processing.

    This wraps the base SIEVE model to handle chunked variant processing,
    ensuring all chromosomes are seen (not just chr1/chr2).
    """
    # Create base SIEVE model
    base_model = SIEVE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_genes=num_genes,
        num_attention_layers=num_attention_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_covariates=num_covariates,
        num_chromosomes=num_chromosomes,
        classifier_type=classifier_type,
        position_encoding=position_encoding,
    )

    # Wrap in chunked model for whole-genome coverage
    model = ChunkedSIEVEModel(
        base_model=base_model,
        aggregation_method=aggregation_method,
        embedding_dim=latent_dim if aggregation_method == 'attention' else None
    )

    return model


def create_training_model(
    *,
    args: argparse.Namespace,
    resolved_position_encoding: ResolvedPositionEncodingConfig,
    num_genes: int,
    num_chromosomes: int,
    num_covariates: int,
) -> ChunkedSIEVEModel:
    """Create the new-schema training model from the resolved positional config."""
    return create_model(
        input_dim=resolved_position_encoding.input_dim,
        num_genes=num_genes,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_attention_layers=args.num_attention_layers,
        hidden_dim=args.hidden_dim,
        aggregation_method=args.aggregation_method,
        num_covariates=num_covariates,
        num_chromosomes=num_chromosomes,
        classifier_type=args.classifier_type,
        position_encoding=resolved_position_encoding,
    )


def save_fold_config(
    fold_dir: Path,
    fold_idx: int,
    args,
    run_metadata: dict[str, object] | None = None,
) -> None:
    """
    Save fold-specific config.yaml with architecture and training parameters.

    This config can be used standalone with explain.py, without needing
    the parent experiment config.

    Parameters
    ----------
    fold_dir : Path
        Directory for the fold (e.g., experiment_dir/fold_0).
    fold_idx : int
        Index of the fold (0-based).
    args : argparse.Namespace
        Command-line arguments containing all config values.
    """
    num_covariates = getattr(args, 'num_covariates', 1 if args.sex_map else 0)
    pos_weight_val = getattr(args, '_fold_pos_weight', None)
    fold_config = {
        'fold_index': fold_idx,
        'experiment_name': args.experiment_name,
        # Architecture parameters
        'level': args.level,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'num_attention_layers': args.num_attention_layers,
        'num_heads': args.num_heads,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'aggregation_method': args.aggregation_method,
        'classifier_type': getattr(args, 'classifier_type', 'flatten'),
        # Training parameters
        'lr': args.lr,
        'lambda_attr': args.lambda_attr,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'gradient_clip': args.gradient_clip,
        'early_stopping': args.early_stopping,
        'epochs': args.epochs,
        'seed': args.seed,
        'genome_build': args.genome_build,
        # Covariate parameters
        'sex_map': str(args.sex_map) if args.sex_map else None,
        'num_covariates': num_covariates,
        'pc_map': str(args.pc_map) if getattr(args, 'pc_map', None) else None,
        'num_pcs': int(getattr(args, 'num_pcs', 0)),
        'pc_map_sha256': getattr(args, 'pc_map_sha256', None),
        # Class weighting
        'class_weighting_applied': pos_weight_val is not None,
        'class_weighting_pos_weight': float(pos_weight_val.item()) if pos_weight_val is not None else None,
        # Data reference
        'preprocessed_data': str(args.preprocessed_data) if args.preprocessed_data else None,
        'vcf': str(args.vcf) if args.vcf else None,
        'phenotypes': str(args.phenotypes) if args.phenotypes else None,
        # Reference to parent config
        'parent_config': '../config.yaml',
    }
    if run_metadata is not None:
        fold_config.update(copy.deepcopy(run_metadata))

    with open(fold_dir / 'config.yaml', 'w') as f:
        yaml.dump(fold_config, f, default_flow_style=False, sort_keys=False)


def _update_saved_config(config_path: Path, **updates) -> None:
    """Merge key/value updates into the run config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    config.update(updates)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_fold_info(
    fold_dir: Path,
    fold_idx: int,
    n_folds: int,
    seed: int,
    train_indices: List[int],
    val_indices: List[int],
    labels: np.ndarray,
    fold_metrics: Dict[str, float],
    training_started: datetime,
    training_completed: datetime,
) -> None:
    """
    Save fold_info.yaml with fold-specific metadata for reproducibility.

    This includes sample split indices (critical for null baseline generation),
    class distributions, and training outcome.

    Parameters
    ----------
    fold_dir : Path
        Directory for the fold.
    fold_idx : int
        Index of the fold (0-based).
    n_folds : int
        Total number of CV folds.
    seed : int
        Random seed used for splitting.
    train_indices : list of int
        Sample-level indices used for training.
    val_indices : list of int
        Sample-level indices used for validation.
    labels : np.ndarray
        Full label array for the dataset.
    fold_metrics : dict
        Metrics returned by train_single_fold.
    training_started : datetime
        Timestamp when fold training started.
    training_completed : datetime
        Timestamp when fold training completed.
    """
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    fold_info = {
        'fold_index': fold_idx,
        'n_folds': n_folds,
        'random_seed': seed,
        # Sample split information
        'n_train_samples': len(train_indices),
        'n_val_samples': len(val_indices),
        'train_sample_indices': [int(i) for i in train_indices],
        'val_sample_indices': [int(i) for i in val_indices],
        # Class distribution
        'train_cases': int(train_labels.sum()),
        'train_controls': int(len(train_labels) - train_labels.sum()),
        'val_cases': int(val_labels.sum()),
        'val_controls': int(len(val_labels) - val_labels.sum()),
        # Training outcome
        'best_epoch': fold_metrics.get('best_epoch'),
        'best_val_auc': float(fold_metrics['auc']),
        'best_val_accuracy': float(fold_metrics['accuracy']),
        'epochs_trained': fold_metrics.get('epochs_trained'),
        # Timestamps
        'training_started': training_started.isoformat(),
        'training_completed': training_completed.isoformat(),
        'training_time_seconds': fold_metrics.get('training_time_seconds'),
    }

    with open(fold_dir / 'fold_info.yaml', 'w') as f:
        yaml.dump(fold_info, f, default_flow_style=False, sort_keys=False)


def _resolve_pos_weight(
    train_labels: np.ndarray,
    class_weighting: str,
) -> Optional[torch.Tensor]:
    """
    Return the BCE pos_weight tensor (or None) based on the policy.

    Parameters
    ----------
    train_labels : np.ndarray
        Binary labels (0/1) for the training fold.
    class_weighting : str
        'auto' | 'on' | 'off'

    Returns
    -------
    Optional[torch.Tensor]
        Positive class weight, or None when class weighting is disabled.
    """
    case_fraction = float(train_labels.mean())
    apply = (
        class_weighting == 'on'
        or (class_weighting == 'auto' and not (0.4 <= case_fraction <= 0.6))
    )
    if apply:
        pos_weight = compute_class_weights(torch.as_tensor(train_labels, dtype=torch.float32))
        print(
            f"Applying class weighting: case_fraction={case_fraction:.3f}, "
            f"pos_weight={pos_weight.item():.3f}"
        )
        return pos_weight
    return None


def train_single_fold(
    train_loader,
    val_loader,
    model: SIEVE,
    args,
    checkpoint_dir: Path,
    pos_weight: Optional[torch.Tensor] = None,
    checkpoint_metadata: dict[str, object] | None = None,
) -> Dict[str, float]:
    """Train model on a single fold."""
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize AUC
        factor=0.5,
        patience=5,
    )

    # Create loss function (move pos_weight to training device to avoid device mismatch)
    if pos_weight is not None:
        pos_weight = pos_weight.to(args.device)
    loss_fn = SIEVELoss(lambda_attr=args.lambda_attr, pos_weight=pos_weight)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping,
        gradient_clip_value=args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_metadata=checkpoint_metadata,
    )

    # Train
    print("\nTraining...")
    train_start = time.time()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=True,
    )
    train_time = time.time() - train_start

    # Capture actual epochs trained before loading checkpoint
    actual_epochs_trained = trainer.current_epoch + 1

    # Load best model
    best_metrics = trainer.load_checkpoint('best_model.pt')

    # Add timing information (use actual epochs, not best checkpoint epoch)
    best_metrics['training_time_seconds'] = train_time
    best_metrics['epochs_trained'] = actual_epochs_trained
    best_metrics['time_per_epoch_seconds'] = train_time / actual_epochs_trained
    best_metrics['best_epoch'] = trainer.current_epoch + 1  # When best model was found

    return best_metrics


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Validate arguments
    if args.preprocessed_data is None and (args.vcf is None or args.phenotypes is None):
        raise ValueError("Must provide either --preprocessed-data OR both --vcf and --phenotypes")
    if args.pc_map is not None and args.num_pcs == 0:
        raise ValueError("--pc-map requires --num-pcs > 0")
    if args.num_pcs > 0 and args.pc_map is None:
        raise ValueError("--num-pcs requires --pc-map")

    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.level}_run"

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sex map / PC map if provided
    sex_map = None
    pc_map = None
    num_covariates = 0
    if args.sex_map is not None:
        import pandas as pd
        print(f"\nLoading sex map from {args.sex_map}...")
        sex_df = pd.read_csv(args.sex_map, sep='\t')
        sex_map = dict(zip(sex_df['sample_id'], sex_df['inferred_sex']))
        # Filter to M/F only
        sex_map = {k: v for k, v in sex_map.items() if v in ('M', 'F')}
        print(f"  {len(sex_map)} samples with definitive sex (M or F)")
        num_covariates = 1  # sex is 1 covariate
        print(f"  Sex will be used as a covariate in the classifier head")
    if args.pc_map is not None:
        print(f"\nLoading PC map from {args.pc_map}...")
        pc_map = load_pc_map(args.pc_map, args.num_pcs)
        args.pc_map_sha256 = compute_file_sha256(args.pc_map)
        print(f"  Loaded {len(pc_map)} samples with {args.num_pcs} PC(s)")
        print(f"  PC map SHA256: {args.pc_map_sha256}")
        num_covariates += args.num_pcs
    else:
        args.pc_map_sha256 = None
    args.num_covariates = num_covariates

    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)
    print(f"Config saved to {config_path}")

    # Load data
    if args.preprocessed_data is not None:
        # Load from preprocessed file
        print(f"\nLoading preprocessed data from {args.preprocessed_data}...")
        start_time = time.time()

        preprocessed = torch.load(args.preprocessed_data, weights_only=False)
        all_samples = preprocessed['samples']
        metadata = preprocessed.get('metadata', {})

        load_time = time.time() - start_time
        print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")
        if metadata:
            print(f"  Original VCF: {metadata.get('vcf_path', 'unknown')}")
            print(f"  Cases: {metadata.get('num_cases', 'unknown')}")
            print(f"  Controls: {metadata.get('num_controls', 'unknown')}")

        # If sex_map provided, update samples with sex information
        if sex_map:
            n_updated = 0
            for sample in all_samples:
                if sample.sample_id in sex_map:
                    sample.sex = sex_map[sample.sample_id]
                    n_updated += 1
            print(f"  Updated {n_updated}/{len(all_samples)} samples with sex info")
    else:
        # Load from VCF
        print(f"\nLoading data from {args.vcf}...")
        start_time = time.time()
        all_samples = build_sample_variants(
            vcf_path=args.vcf,
            phenotype_file=args.phenotypes,
            sex_map=sex_map,
        )
        load_time = time.time() - start_time
        print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")

    if pc_map is not None:
        attach_pc_covariates_to_samples(
            all_samples,
            pc_map=pc_map,
            include_sex=sex_map is not None,
        )
        print(f"  Attached ancestry PCs to {len(all_samples)} samples")

    # Create chunked dataset (ensures whole-genome coverage)
    annotation_level = AnnotationLevel[args.level]
    print(f"\nCreating CHUNKED dataset with annotation level {args.level}...")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Chunk overlap: {args.chunk_overlap}")
    print(f"  Aggregation method: {args.aggregation_method}")
    dataset = ChunkedVariantDataset(
        samples=all_samples,
        annotation_level=annotation_level,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap
    )
    if dataset.num_covariates not in (0, num_covariates):
        raise ValueError(
            "Covariate tensor width in the dataset does not match the model configuration "
            f"({dataset.num_covariates} vs {num_covariates})."
        )

    # Get dimensions. New training runs are always explicit new-schema runs, so
    # the resolved positional configuration is the model-width authority.
    resolved_position_encoding = prepare_training_position_encoding(
        args,
        annotation_level,
        num_chromosomes=dataset.num_chromosomes,
    )
    input_dim = resolved_position_encoding.input_dim
    num_genes = dataset.num_genes
    dataset_identity = write_dataset_mappings_artifact(
        output_dir,
        dataset.gene_index,
        dataset.chrom_index,
    )
    training_mode = "cv" if args.cv is not None else "single_split"
    run_metadata = build_training_run_metadata(
        input_dim=input_dim,
        num_genes=num_genes,
        num_chromosomes=dataset.num_chromosomes,
        genome_build=args.genome_build,
        resolved_position_encoding=resolved_position_encoding,
        chrom_index=dataset.chrom_index,
        gene_mapping_sha256=str(dataset_identity["gene_mapping_sha256"]),
        chromosome_mapping_sha256=str(dataset_identity["chromosome_mapping_sha256"]),
        training_mode=training_mode,
    )
    _update_saved_config(config_path, **run_metadata)
    print(f"Content dimension: {resolved_position_encoding.content_dim}")
    print(f"Resolved model input dimension: {input_dim}")
    print(f"Number of genes: {num_genes}")
    print(f"Position encoding preset: {resolved_position_encoding.preset.value}")
    print(
        "Position strategies: "
        f"absolute={resolved_position_encoding.absolute.encoding.value}, "
        f"relative={resolved_position_encoding.relative.encoding.value}, "
        f"chromosome={resolved_position_encoding.chromosome.encoding.value}, "
        f"cross={resolved_position_encoding.chromosome.cross_chromosome_policy.value}"
    )
    print(f"CRITICAL: Using chunked processing for FULL GENOME coverage (not just chr1/chr2)!")

    # Get labels
    labels = np.array([sample.label for sample in all_samples])
    n_cases = labels.sum()
    n_controls = len(labels) - n_cases
    print(f"Cases: {n_cases}, Controls: {n_controls} ({n_cases/len(labels):.1%} case rate)")

    # Report sex covariate summary
    if num_covariates > 0:
        n_with_sex = sum(1 for s in all_samples if s.sex is not None)
        n_male = sum(1 for s in all_samples if s.sex == 'M')
        n_female = sum(1 for s in all_samples if s.sex == 'F')
        print(f"\nSex covariate enabled:")
        print(f"  Samples with sex: {n_with_sex}/{len(all_samples)}")
        print(f"  Male: {n_male}, Female: {n_female}")
        male_cases = sum(1 for s in all_samples if s.sex == 'M' and s.label == 1)
        female_cases = sum(1 for s in all_samples if s.sex == 'F' and s.label == 1)
        print(f"  Male cases: {male_cases}, Female cases: {female_cases}")
    if args.num_pcs > 0:
        print(f"Ancestry PCs enabled: {args.num_pcs}")

    if args.cv is not None:
        # Cross-validation
        _update_saved_config(
            config_path,
            class_weighting_applied=None,
            class_weighting_pos_weight=None,
        )
        print(f"\n{'='*60}")
        print(f"Running {args.cv}-fold cross-validation")
        print(f"{'='*60}")

        folds = create_stratified_folds(labels, n_folds=args.cv, random_state=args.seed)
        cv_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{args.cv}")
            print(f"{'='*60}")

            # Print fold statistics
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            print_fold_stats(fold_idx, train_labels, val_labels)

            # Convert sample indices to chunk indices
            # CRITICAL: train_idx/val_idx are sample-level indices
            # but dataset is indexed by chunks. Must convert!
            train_chunk_idx = []
            for sample_idx in train_idx:
                train_chunk_idx.extend(dataset.get_chunks_for_sample(int(sample_idx)))

            val_chunk_idx = []
            for sample_idx in val_idx:
                val_chunk_idx.extend(dataset.get_chunks_for_sample(int(sample_idx)))

            print(f"\nChunk-level split:")
            print(f"  Train: {len(train_chunk_idx)} chunks from {len(train_idx)} samples")
            print(f"  Val: {len(val_chunk_idx)} chunks from {len(val_idx)} samples")

            # Create data loaders with chunked processing
            from torch.utils.data import DataLoader, Subset
            train_dataset = Subset(dataset, train_chunk_idx)
            val_dataset = Subset(dataset, val_chunk_idx)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_chunks,
                num_workers=args.num_workers,
                pin_memory=True if args.num_workers > 0 else False,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_chunks,
                num_workers=args.num_workers,
                pin_memory=True if args.num_workers > 0 else False,
            )

            # Create chunked model (for whole-genome processing)
            model = create_training_model(
                args=args,
                resolved_position_encoding=resolved_position_encoding,
                num_genes=num_genes,
                num_covariates=num_covariates,
                num_chromosomes=dataset.num_chromosomes,
            )

            # Create fold checkpoint directory
            fold_dir = output_dir / f'fold_{fold_idx}'

            # Resolve class weighting for this fold
            fold_pos_weight = _resolve_pos_weight(train_labels, args.class_weighting)
            # Store on args so save_fold_config can access it without signature change
            args._fold_pos_weight = fold_pos_weight

            # Train
            training_started = datetime.now(timezone.utc)
            fold_metrics = train_single_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                args=args,
                checkpoint_dir=fold_dir,
                pos_weight=fold_pos_weight,
                checkpoint_metadata=run_metadata,
            )
            training_completed = datetime.now(timezone.utc)

            cv_results.append(fold_metrics)

            # Save fold-specific config and metadata
            save_fold_config(fold_dir, fold_idx, args, run_metadata=run_metadata)
            save_fold_info(
                fold_dir=fold_dir,
                fold_idx=fold_idx,
                n_folds=args.cv,
                seed=args.seed,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                labels=labels,
                fold_metrics=fold_metrics,
                training_started=training_started,
                training_completed=training_completed,
            )
            print(f"\nFold {fold_idx + 1} config and metadata saved to {fold_dir}")

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Val AUC: {fold_metrics['auc']:.4f}")
            print(f"  Val Accuracy: {fold_metrics['accuracy']:.4f}")

        # Print CV summary
        print(f"\n{'='*60}")
        print(f"Cross-Validation Summary")
        print(f"{'='*60}")
        mean_auc = np.mean([r['auc'] for r in cv_results])
        std_auc = np.std([r['auc'] for r in cv_results])
        mean_acc = np.mean([r['accuracy'] for r in cv_results])
        std_acc = np.std([r['accuracy'] for r in cv_results])

        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save CV results
        results_path = output_dir / 'cv_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump({
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'data_loading_time_seconds': float(load_time),
                'fold_results': [
                    {k: float(v) for k, v in r.items()}
                    for r in cv_results
                ],
            }, f)
        print(f"\nResults saved to {results_path}")

    else:
        # Single train/val split
        print(f"\nUsing single train/val split ({1-args.val_split:.0%}/{args.val_split:.0%})")

        # Create stratified split
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            stratify=labels,
            random_state=args.seed,
        )

        # Print split statistics
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        print_fold_stats(0, train_labels, val_labels)

        # Convert sample indices to chunk indices
        # CRITICAL: train_idx/val_idx are sample-level indices (0-1967)
        # but dataset is indexed by chunks (0-18282). Must convert!
        train_chunk_idx = [i for i, info in enumerate(dataset.chunk_info) if info['sample_idx'] in train_idx]
        val_chunk_idx = [i for i, info in enumerate(dataset.chunk_info) if info['sample_idx'] in val_idx]

        print(f"\nChunk-level split:")
        print(f"  Train: {len(train_chunk_idx)} chunks from {len(train_idx)} samples")
        print(f"  Val: {len(val_chunk_idx)} chunks from {len(val_idx)} samples")

        # Create data loaders with chunked processing
        from torch.utils.data import DataLoader, Subset
        train_dataset = Subset(dataset, train_chunk_idx)
        val_dataset = Subset(dataset, val_chunk_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_chunks,
            num_workers=args.num_workers,
            pin_memory=True if args.num_workers > 0 else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_chunks,
            num_workers=args.num_workers,
            pin_memory=True if args.num_workers > 0 else False,
        )

        # Create chunked model (for whole-genome processing)
        model = create_training_model(
            args=args,
            resolved_position_encoding=resolved_position_encoding,
            num_genes=num_genes,
            num_covariates=num_covariates,
            num_chromosomes=dataset.num_chromosomes,
        )

        # Resolve class weighting for this split
        pos_weight = _resolve_pos_weight(train_labels, args.class_weighting)
        args._fold_pos_weight = pos_weight
        _update_saved_config(
            config_path,
            class_weighting_applied=pos_weight is not None,
            class_weighting_pos_weight=float(pos_weight.item()) if pos_weight is not None else None,
        )

        # Train
        metrics = train_single_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            args=args,
            checkpoint_dir=output_dir,
            pos_weight=pos_weight,
            checkpoint_metadata=run_metadata,
        )

        print(f"\nFinal Results:")
        print(f"  Val AUC: {metrics['auc']:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")

        # Save results
        results_path = output_dir / 'results.yaml'
        results_with_timing = {k: float(v) for k, v in metrics.items()}
        results_with_timing['data_loading_time_seconds'] = float(load_time)
        results_with_timing['class_weighting_applied'] = pos_weight is not None
        results_with_timing['class_weighting_pos_weight'] = float(pos_weight.item()) if pos_weight is not None else None
        with open(results_path, 'w') as f:
            yaml.dump(results_with_timing, f)
        print(f"\nResults saved to {results_path}")

    print(f"\nTraining complete! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
