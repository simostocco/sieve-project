#!/usr/bin/env python3
"""
Explainability analysis for trained SIEVE models.

Computes variant attributions using integrated gradients and analyzes
attention patterns to discover disease-associated variants and epistatic
interactions.

Usage:
    # Basic usage - analyse best model from experiment
    python scripts/explain.py \
        --experiment-dir outputs/L3_attr_medium \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability

    # Analyse specific fold
    python scripts/explain.py \
        --checkpoint outputs/L3_attr_medium/fold_0/best_model.pt \
        --config outputs/L3_attr_medium/config.yaml \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability_fold0

    # Only compute attributions (skip attention analysis)
    python scripts/explain.py \
        --experiment-dir outputs/L3_attr_medium \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability \
        --skip-attention

Notes:
    Attribution magnitudes (mean_attribution, score) are model-specific and
    not directly comparable across annotation levels or model architectures.
    For cross-level ablation comparison use rank-based metrics (Jaccard on
    top-K sets) rather than raw score differences, and rank by delta_rank,
    which is scale-free and stable across annotation levels.

Author: Francesco Lescai
"""

import argparse
import gc
import shutil
from collections import Counter
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data.covariates import attach_pc_covariates_to_samples, load_pc_map
from src.encoding import (
    ChunkedVariantDataset,
    collate_chunks,
    get_content_feature_dimension,
    AnnotationLevel
)
from src.encoding.position_config import PositionPreset, ResolvedIGMode
from src.models.reconstruction import (
    ReconstructedSIEVEModel,
    reconstruct_sieve_from_checkpoint,
)
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.ig_mode import RequestedIGMode, resolve_ig_mode
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.variant_ranking import VariantRanker


ATTRIBUTION_SCHEMA_VERSION = 1
VARIANT_SCORE_AGGREGATION = 'l2'
SAMPLING_POLICY = 'manual_chunk_full_coverage_no_random_subsampling'
LEGACY_COMPARABILITY_WARNING = (
    "Legacy IG includes historical positional channels where present; raw "
    "attribution magnitudes are not directly comparable with content-only "
    "benchmark attribution."
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the explainability CLI parser."""
    parser = argparse.ArgumentParser(
        description='Run explainability analysis on trained SIEVE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model input
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--experiment-dir', type=str,
                             help='Path to experiment directory (will use best fold)')
    model_group.add_argument('--checkpoint', type=str,
                             help='Path to specific model checkpoint')

    parser.add_argument('--config', type=str,
                        help='Path to config.yaml (required if using --checkpoint)')

    # Data input
    parser.add_argument('--preprocessed-data', type=str, required=True,
                        help='Path to preprocessed data (.pt file)')
    parser.add_argument(
        '--pc-map',
        type=str,
        default=None,
        help=(
            "Optional TSV with columns: sample_id, PC1, PC2, ... "
            "Used to rebuild the covariate vector for models trained with PCs."
        ),
    )
    parser.add_argument(
        '--num-pcs',
        type=int,
        default=0,
        help='Number of PCs to use from --pc-map (default: 0).',
    )

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Analysis options
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Number of integration steps for IG')
    parser.add_argument('--max-variants', type=int, default=2000,
                        help='Maximum variants per sample for IG (to avoid OOM). Samples with more variants are randomly sampled.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for dataloader (samples processed individually during IG to avoid OOM)')
    parser.add_argument('--skip-attention', action='store_true',
                        help='Skip attention analysis (faster)')
    parser.add_argument('--skip-ig', action='store_true',
                        help='Skip Integrated Gradients computation (use if you only need attention analysis)')
    parser.add_argument(
        '--ig-mode',
        type=str,
        default=RequestedIGMode.AUTO.value,
        choices=[mode.value for mode in RequestedIGMode],
        help=(
            "Integrated Gradients mode. auto uses the saved attribution policy "
            "for new-schema configs and preserves historical legacy attribution "
            "for old configs; content attributes biological content while "
            "absolute position remains fixed; legacy attributes the complete "
            "historical feature representation."
        ),
    )
    parser.add_argument('--top-k-variants', type=int, default=100,
                        help='Number of top variants to extract')
    parser.add_argument('--top-k-interactions', type=int, default=100,
                        help='Number of top interactions to extract')
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Minimum attention weight for interactions')
    parser.add_argument('--attention-threshold-mode', type=str, default='absolute',
                        choices=['absolute', 'percentile'],
                        help='How to threshold pairwise attention scores')
    parser.add_argument('--attention-percentile', type=float, default=99.9,
                        help='Percentile cutoff for attention interactions when using percentile mode')
    parser.add_argument('--aggregation-method', type=str, default='mean',
                        choices=['mean', 'max', 'rank_average'],
                        help=(
                            'How to aggregate per-sample variant scores into a '
                            'population-level ranking score. '
                            "'mean': score == mean_attribution (default, most transparent). "
                            "'max': score == max_attribution. "
                            "'rank_average': composite rank across mean, max, and sample count."
                        ))
    parser.add_argument('--is-null-baseline', action='store_true',
                        help='Flag indicating this is a null baseline analysis (for metadata)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    # Genome build
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build (GRCh37 or GRCh38)')

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args(argv)


def load_model_and_config(args):
    """Load model and configuration."""
    if args.experiment_dir:
        # Load from experiment directory
        exp_dir = Path(args.experiment_dir)

        # Load config
        config_path = exp_dir / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Find best fold (highest AUC in CV results)
        cv_results_path = exp_dir / 'cv_results.yaml'
        if cv_results_path.exists():
            with open(cv_results_path) as f:
                cv_results = yaml.safe_load(f)

            # Find best fold
            best_fold = 0
            best_auc = 0
            for i, result in enumerate(cv_results['fold_results']):
                if result['auc'] > best_auc:
                    best_auc = result['auc']
                    best_fold = i

            checkpoint_path = exp_dir / f'fold_{best_fold}' / 'best_model.pt'
            print(f"Using fold {best_fold} (AUC: {best_auc:.4f})")
        else:
            # Single run - use best_model.pt directly
            checkpoint_path = exp_dir / 'best_model.pt'
            print("Using single run model")

    else:
        # Load specific checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not args.config:
            raise ValueError("--config required when using --checkpoint")

        config_path = Path(args.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)

    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    # Note: weights_only=False is safe here since these are our own trusted checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    return config, checkpoint


def _validate_config_content_dim(config: dict, annotation_level: AnnotationLevel) -> int:
    """Return structural content width and reject conflicting serialized width."""
    content_dim = get_content_feature_dimension(annotation_level)
    if 'position_encoding' not in config:
        return content_dim
    if 'content_dim' not in config:
        return content_dim

    serialized_content_dim = config['content_dim']
    if isinstance(serialized_content_dim, bool) or not isinstance(serialized_content_dim, int):
        raise ValueError("config['content_dim'] must be an integer when present")
    if serialized_content_dim != content_dim:
        raise ValueError(
            "config['content_dim'] does not match annotation level content width: "
            f"{serialized_content_dim} != {content_dim}"
        )
    return content_dim


def _content_dim_for_reconstruction(
    reconstruction: ReconstructedSIEVEModel,
    annotation_level: AnnotationLevel,
) -> int:
    """Return the content attribution width from execution authority.

    Case A stores the resolved positional architecture that actually
    constructed the model. Cases B/C are historical compatibility paths, so
    structural annotation-level content width is the only execution authority.
    """
    if reconstruction.is_new_schema:
        if reconstruction.resolved_position_encoding is None:
            raise ValueError("new-schema reconstruction is missing resolved position encoding")
        return reconstruction.resolved_position_encoding.content_dim
    return get_content_feature_dimension(annotation_level)


def _validate_ig_mode_for_reconstruction(
    resolved_ig_mode: ResolvedIGMode,
    reconstruction: ReconstructedSIEVEModel,
) -> None:
    """Reject legacy IG only for authoritative custom positional execution."""
    if not reconstruction.is_new_schema:
        return
    resolved_position_encoding = reconstruction.resolved_position_encoding
    if resolved_position_encoding is None:
        raise ValueError("new-schema reconstruction is missing resolved position encoding")
    if (
        resolved_position_encoding.preset is PositionPreset.CUSTOM
        and resolved_ig_mode is ResolvedIGMode.LEGACY
    ):
        raise ValueError(
            "legacy IG is not supported for custom positional execution; use "
            "ig_mode='content' or ig_mode='auto'."
        )


def _attention_uses_split_inputs(reconstruction: ReconstructedSIEVEModel) -> bool:
    """Return True only when attention must use custom split-primary inputs."""
    return (
        reconstruction.is_new_schema
        and reconstruction.resolved_position_encoding is not None
        and reconstruction.resolved_position_encoding.preset is PositionPreset.CUSTOM
    )


def _read_position_strategy_metadata(
    reconstruction: ReconstructedSIEVEModel,
) -> dict[str, object]:
    """Read positional strategy provenance from reconstruction authority."""
    if reconstruction.is_new_schema:
        resolved = reconstruction.resolved_position_encoding
        if resolved is None:
            raise ValueError("new-schema reconstruction is missing resolved position encoding")
        return {
            'absolute_position_encoding': resolved.absolute.encoding.value,
            'relative_position_encoding': resolved.relative.encoding.value,
            'chromosome_encoding': resolved.chromosome.encoding.value,
            'position_encoding_metadata_source': 'reconstructed_resolved_config',
        }

    if 'position_encoding' not in reconstruction.effective_config:
        return {
            'absolute_position_encoding': None,
            'relative_position_encoding': None,
            'chromosome_encoding': None,
            'position_encoding_metadata_source': 'unavailable_old_config',
        }

    return {
        'absolute_position_encoding': None,
        'relative_position_encoding': None,
        'chromosome_encoding': None,
        'position_encoding_metadata_source': 'transitional_historical_execution',
    }


def _build_ig_run_metadata(
    *,
    requested_ig_mode: str,
    resolved_ig_mode: ResolvedIGMode,
    reconstruction: ReconstructedSIEVEModel,
    content_dim: int,
    n_steps: int,
    max_variants: int,
) -> dict[str, object]:
    """Build semantic metadata describing the Integrated Gradients run."""
    input_dim = reconstruction.base_model.input_dim
    if resolved_ig_mode is ResolvedIGMode.CONTENT:
        attribution_feature_space = 'content'
        attribution_width = content_dim
        baseline_policy = 'zero_content_observed_absolute_position'
        comparability_warning = None
    elif resolved_ig_mode is ResolvedIGMode.LEGACY:
        attribution_feature_space = 'legacy'
        attribution_width = input_dim
        baseline_policy = 'zero_historical_features'
        comparability_warning = LEGACY_COMPARABILITY_WARNING
    else:
        raise ValueError(f"unsupported resolved IG mode: {resolved_ig_mode!r}")

    metadata = {
        'attribution_schema_version': ATTRIBUTION_SCHEMA_VERSION,
        'requested_ig_mode': requested_ig_mode,
        'resolved_ig_mode': resolved_ig_mode.value,
        'attribution_feature_space': attribution_feature_space,
        'attribution_width': attribution_width,
        'content_dim': content_dim,
        'input_dim': input_dim,
        'variant_score_aggregation': VARIANT_SCORE_AGGREGATION,
        'baseline_policy': baseline_policy,
        'n_steps': n_steps,
        'max_variants': max_variants,
        'sampling_policy': SAMPLING_POLICY,
        'sampling_seed': None,
        'comparability_warning': comparability_warning,
    }
    metadata.update(_read_position_strategy_metadata(reconstruction))
    return metadata


def _build_skipped_ig_metadata(requested_ig_mode: str) -> dict[str, object]:
    """Build analysis metadata for attention-only runs without resolving IG mode."""
    return {
        'executed': False,
        'requested_ig_mode': requested_ig_mode,
        'resolved_ig_mode': None,
    }


def _create_integrated_gradients_explainer(
    *,
    model,
    device: str,
    n_steps: int,
    max_variants: int,
    resolved_ig_mode: ResolvedIGMode,
) -> IntegratedGradientsExplainer:
    """Construct the 5B3B explainer with the already resolved IG mode."""
    return IntegratedGradientsExplainer(
        model=model,
        device=device,
        n_steps=n_steps,
        max_variants=max_variants,
        ig_mode=resolved_ig_mode,
    )


def _npz_scalar_metadata(
    metadata: dict[str, object],
    *,
    per_sample: bool,
) -> dict[str, np.ndarray]:
    """Convert IG metadata scalars to NPZ-safe arrays without object dtype.

    Semantic metadata keeps Python ``None`` values. NPZ scalar fields use
    explicit sentinels because NumPy would otherwise store ``None`` as object
    dtype, which cannot be read with ``allow_pickle=False``.
    """
    if per_sample:
        keys = [
            'attribution_schema_version',
            'requested_ig_mode',
            'resolved_ig_mode',
            'attribution_feature_space',
            'attribution_width',
            'content_dim',
            'input_dim',
            'variant_score_aggregation',
            'baseline_policy',
        ]
    else:
        keys = [
            'attribution_schema_version',
            'requested_ig_mode',
            'resolved_ig_mode',
            'attribution_feature_space',
            'attribution_width',
            'content_dim',
            'input_dim',
            'absolute_position_encoding',
            'relative_position_encoding',
            'chromosome_encoding',
            'position_encoding_metadata_source',
            'variant_score_aggregation',
            'baseline_policy',
            'n_steps',
            'max_variants',
            'sampling_policy',
            'sampling_seed',
            'comparability_warning',
        ]

    scalar_metadata = {}
    for key in keys:
        value = metadata[key]
        if value is None:
            if key in {
                'absolute_position_encoding',
                'relative_position_encoding',
                'chromosome_encoding',
            }:
                value = 'unavailable'
            elif key == 'sampling_seed':
                value = -1
            else:
                value = ''
        scalar_metadata[key] = np.asarray(value)
        if scalar_metadata[key].dtype == object:
            raise ValueError(f"metadata field {key!r} cannot be serialized without pickle")
    return scalar_metadata


def _validate_attribution_width(
    attributions: np.ndarray,
    expected_width: int,
) -> None:
    """Validate raw attribution feature width before padded-row filtering."""
    if attributions.ndim != 2:
        raise ValueError(
            "chunk attributions must be a 2D matrix before mask filtering; "
            f"got shape {attributions.shape}"
        )
    actual_width = attributions.shape[1]
    if actual_width != expected_width:
        raise ValueError(
            "unexpected attribution feature width before mask filtering: "
            f"got {actual_width}, expected {expected_width}"
        )


def _attribute_chunk_for_ig(
    *,
    explainer: IntegratedGradientsExplainer,
    chunk: dict,
    resolved_ig_mode: ResolvedIGMode,
    device: str,
    chunk_covariates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Invoke 5B3B IG for one chunk while preserving the selected feature boundary."""
    positions = chunk['positions'].unsqueeze(0).to(device)
    gene_ids = chunk['gene_ids'].unsqueeze(0).to(device)
    mask = chunk['mask'].unsqueeze(0).to(device)
    chrom_ids = (
        chunk['chrom_ids'].unsqueeze(0).to(device)
        if 'chrom_ids' in chunk else None
    )
    if chunk_covariates is not None:
        chunk_covariates = chunk_covariates.to(device)

    if resolved_ig_mode is ResolvedIGMode.LEGACY:
        if 'features' not in chunk:
            raise ValueError("legacy IG mode requires chunk['features']")
        features = chunk['features'].unsqueeze(0).to(device)
        attributions = explainer.attribute(
            features,
            positions,
            gene_ids,
            mask,
            covariates=chunk_covariates,
            chrom_ids=chrom_ids,
        )
    elif resolved_ig_mode is ResolvedIGMode.CONTENT:
        if 'content_features' not in chunk or 'absolute_position_features' not in chunk:
            raise ValueError(
                "content IG mode requires chunk['content_features'] and "
                "chunk['absolute_position_features']"
            )
        content_features = chunk['content_features'].unsqueeze(0).to(device)
        absolute_position_features = chunk['absolute_position_features'].unsqueeze(0).to(device)
        attributions = explainer.attribute(
            None,
            positions,
            gene_ids,
            mask,
            covariates=chunk_covariates,
            chrom_ids=chrom_ids,
            content_features=content_features,
            absolute_position_features=absolute_position_features,
        )
    else:
        raise ValueError(f"unsupported resolved IG mode: {resolved_ig_mode!r}")

    return attributions, positions, gene_ids, mask, chrom_ids


def _annotate_ranking_metadata(df, ig_metadata: dict[str, object]):
    """Add informational IG provenance columns after ranking calculations."""
    annotated = df.copy()
    annotated['resolved_ig_mode'] = ig_metadata['resolved_ig_mode']
    annotated['attribution_feature_space'] = ig_metadata['attribution_feature_space']
    annotated['variant_score_aggregation'] = ig_metadata['variant_score_aggregation']
    return annotated


def _reconstruct_model_for_explanation(
    config: dict,
    checkpoint: dict,
    dataset: ChunkedVariantDataset,
) -> ReconstructedSIEVEModel:
    """Reconstruct the model without mutating loaded config metadata."""
    return reconstruct_sieve_from_checkpoint(
        config,
        checkpoint,
        num_genes=dataset.num_genes,
        dataset_num_chromosomes=dataset.num_chromosomes,
        dataset_chrom_index=dataset.chrom_index,
    )


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SIEVE Explainability Analysis")
    print("="*60)

    # Load model and config
    config, checkpoint = load_model_and_config(args)

    # Load data
    print("\nLoading data...")
    preprocessed = torch.load(args.preprocessed_data, weights_only=False)
    all_samples = preprocessed['samples']
    metadata = preprocessed.get('metadata', {})

    print(f"Loaded {len(all_samples)} samples")
    if metadata:
        print(f"  Cases: {metadata.get('num_cases', 'unknown')}")
        print(f"  Controls: {metadata.get('num_controls', 'unknown')}")

    # If the model was trained with sex covariates, apply the same sex map here
    # so that covariates propagated through IG match the training configuration.
    sex_map_path = config.get('sex_map')
    if sex_map_path and Path(sex_map_path).exists():
        import pandas as _pd
        sex_df = _pd.read_csv(sex_map_path, sep='\t')
        sex_map = dict(zip(sex_df['sample_id'], sex_df['inferred_sex']))
        sex_map = {k: v for k, v in sex_map.items() if v in ('M', 'F')}
        n_updated = 0
        for sample in all_samples:
            if sample.sample_id in sex_map:
                sample.sex = sex_map[sample.sample_id]
                n_updated += 1
        print(f"  Applied sex map from config: {n_updated}/{len(all_samples)} samples updated")
    elif sex_map_path:
        print(f"  WARNING: Sex map path from config not found ({sex_map_path}); "
              "sex covariates will use values embedded in preprocessed data (if any)")

    pc_map_path = args.pc_map or config.get('pc_map')
    num_pcs = args.num_pcs or config.get('num_pcs', 0)
    if pc_map_path is not None and num_pcs == 0:
        raise ValueError("--pc-map requires --num-pcs > 0 (or num_pcs in config)")
    if num_pcs > 0 and pc_map_path is None:
        raise ValueError("num_pcs > 0 but no PC map was provided")
    if pc_map_path is not None:
        pc_map = load_pc_map(pc_map_path, num_pcs)
        attach_pc_covariates_to_samples(
            all_samples,
            pc_map=pc_map,
            include_sex=sex_map_path is not None,
        )
        print(f"  Attached {num_pcs} PC covariate(s) from {pc_map_path}")

    # Get annotation level
    annotation_level = AnnotationLevel[config['level']]

    # Create CHUNKED dataset for whole-genome coverage
    print("\nCreating CHUNKED dataset for FULL GENOME explainability...")
    chunk_size = min(args.max_variants, 2000)  # Smaller chunks for IG (memory-intensive)
    print(f"  Chunk size: {chunk_size}")
    print(f"  This ensures ALL chromosomes are analyzed, not just chr1/chr2!")

    dataset = ChunkedVariantDataset(
        samples=all_samples,
        annotation_level=annotation_level,
        chunk_size=chunk_size,
        overlap=0
    )

    # Create model through Phase 7B4A reconstruction. That result is the sole
    # architecture authority: Case A uses resolved schema-v2 metadata, while
    # Cases B/C infer historical structure from checkpoint tensors.
    print("\nCreating model...")
    reconstruction = _reconstruct_model_for_explanation(config, checkpoint, dataset)
    model = reconstruction.model

    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # For IG, we need the reconstructed base model (not the chunked wrapper).
    ig_model = reconstruction.base_model
    if reconstruction.is_chunked_checkpoint:
        print("  Using base model for Integrated Gradients (chunk-level attributions)")
    else:
        print("  Using model directly for Integrated Gradients")

    # Detect covariate requirements from loaded model
    ig_num_covariates = getattr(ig_model, 'num_covariates', 0)
    if ig_num_covariates > 0:
        print(f"  Model uses {ig_num_covariates} covariate(s), which will be propagated through IG")
    if dataset.num_covariates not in (0, ig_num_covariates):
        raise ValueError(
            "Covariate tensor width in the dataset does not match the loaded model "
            f"({dataset.num_covariates} vs {ig_num_covariates})."
        )

    # === INTEGRATED GRADIENTS (CHUNKED) ===
    if args.skip_ig:
        print("\n" + "="*60)
        print("Skipping Integrated Gradients (--skip-ig specified)")
        print("="*60)
        variant_rankings = None
        gene_rankings = None
        case_enriched = None
        integrated_gradients_metadata = _build_skipped_ig_metadata(args.ig_mode)
    else:
        print("\n" + "="*60)
        print("Computing Integrated Gradients Attributions (CHUNKED)")
        print("="*60)

        content_dim = _content_dim_for_reconstruction(reconstruction, annotation_level)
        resolved_ig_mode = resolve_ig_mode(
            args.ig_mode,
            config=config,
            is_new_schema=reconstruction.is_new_schema,
        )
        _validate_ig_mode_for_reconstruction(resolved_ig_mode, reconstruction)
        ig_metadata = _build_ig_run_metadata(
            requested_ig_mode=args.ig_mode,
            resolved_ig_mode=resolved_ig_mode,
            reconstruction=reconstruction,
            content_dim=content_dim,
            n_steps=args.n_steps,
            max_variants=chunk_size,
        )
        expected_attribution_width = ig_metadata['attribution_width']
        integrated_gradients_metadata = {
            'executed': True,
            **ig_metadata,
        }

        explainer = _create_integrated_gradients_explainer(
            model=ig_model,
            device=args.device,
            n_steps=args.n_steps,
            max_variants=chunk_size,  # Process full chunks (no truncation within chunks)
            resolved_ig_mode=resolved_ig_mode,
        )

        print(f"IG Configuration:")
        print(f"  Requested IG mode: {args.ig_mode}")
        print(f"  Resolved IG mode: {resolved_ig_mode.value}")
        print(f"  Integration steps: {args.n_steps}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Attribution width: {expected_attribution_width}")
        print(f"  Processing ALL chunks per sample for FULL GENOME coverage")

        # === BUILD VARIANT INFO MAP (before IG loop, needed for ranker) ===
        # Map (chrom, position, gene_id) -> {gene_name} for annotation
        # CRITICAL: Include chromosome in key to prevent position collisions!
        # Same position number can exist on different chromosomes.
        print("\nBuilding variant info map...")

        # Use the dataset's gene_index (not a new one!)
        gene_index = dataset.gene_index

        variant_info_map = {}
        for sample in all_samples:
            for variant in sample.variants:
                pos = variant.pos
                gene_symbol = variant.gene
                chrom = variant.chrom

                # Skip genes not in dataset's gene_index (shouldn't happen but be safe)
                if gene_symbol not in gene_index:
                    print(f"WARNING: Gene {gene_symbol} not in dataset gene_index!")
                    continue

                gene_id = gene_index[gene_symbol]

                # FIXED: Include chromosome in key to prevent collisions
                key = (chrom, pos, gene_id)
                if key not in variant_info_map:
                    variant_info_map[key] = {
                        'gene_name': gene_symbol
                    }

        print(f"Mapped {len(variant_info_map)} unique (chrom, position, gene_id) combinations")

        # Diagnostic: Check chromosome distribution in variant_info_map
        # Chromosome is now part of the KEY (chrom, pos, gene_id), not the value
        chrom_counts = {}
        for (chrom, pos, gene_id) in variant_info_map.keys():
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1

        print(f"Variant info map chromosome distribution:")
        for chrom in sorted(chrom_counts.keys(), key=lambda x: (x.isdigit() and int(x) or 999, x))[:10]:
            print(f"  Chr {chrom}: {chrom_counts[chrom]} unique variants")
        if len(chrom_counts) > 10:
            print(f"  ... and {len(chrom_counts) - 10} more chromosomes")

        # === PREPARE INCREMENTAL PROCESSING ===
        # Create ranker upfront for incremental sample accumulation
        ranker = VariantRanker(aggregation=args.aggregation_method, variant_info_map=variant_info_map)

        # Determine case/control sets upfront
        case_indices = set(i for i in range(len(all_samples)) if all_samples[i].label == 1)
        control_indices = set(i for i in range(len(all_samples)) if all_samples[i].label == 0)
        print(f"Cases: {len(case_indices)}, Controls: {len(control_indices)}")

        # Temp directory for incremental attribution saving (avoids holding all in RAM)
        tmp_dir = output_dir / '_tmp_attributions'
        tmp_dir.mkdir(exist_ok=True)

        # Lightweight metadata list (small 1D arrays + scalars per sample)
        all_metadata = []

        num_samples = len(all_samples)
        metadata_variant_count = 0
        print(f"\nProcessing {num_samples} samples (chunk-by-chunk)...")

        for sample_idx in range(num_samples):
            if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == num_samples:
                print(f"  Sample {sample_idx + 1}/{num_samples}...")

            # Get all chunks for this sample
            chunk_indices = dataset.get_chunks_for_sample(sample_idx)

            # Process each chunk
            chunk_attributions = []
            chunk_positions = []
            chunk_gene_ids = []
            chunk_chromosomes = []

            for chunk_idx in chunk_indices:
                chunk = dataset[chunk_idx]

                # Get chunk info to map back to original variants
                chunk_info = dataset.chunk_info[chunk_idx]
                start_idx = chunk_info['start_idx']
                end_idx = chunk_info['end_idx']
                original_variants = all_samples[sample_idx].variants[start_idx:end_idx]

                # Build covariate tensor for this sample if the model needs it
                chunk_covariates = None
                if ig_num_covariates > 0:
                    from src.models.chunked_sieve import build_sample_covariates
                    sex_val = chunk.get('sex')
                    sex_tensor = None
                    if sex_val is not None:
                        sex_tensor = torch.tensor(
                            [float(sex_val)],
                            dtype=torch.float32,
                            device=torch.device(args.device),
                        )
                    chunk_covariates_tensor = None
                    if 'covariates' in chunk:
                        chunk_covariates_tensor = chunk['covariates'].unsqueeze(0).to(args.device)
                    if sex_tensor is None and chunk_covariates_tensor is None:
                        raise ValueError(
                            f"Model has num_covariates={ig_num_covariates} but the "
                            "chunk contains no covariate information."
                        )
                    chunk_covariates = build_sample_covariates(
                        sex_tensor, ig_num_covariates, 1,
                        torch.device(args.device),
                        batch_covariates=chunk_covariates_tensor,
                    )

                # Compute attributions for this chunk
                attr, positions, gene_ids, mask, chrom_ids = _attribute_chunk_for_ig(
                    explainer=explainer,
                    chunk=chunk,
                    resolved_ig_mode=resolved_ig_mode,
                    device=args.device,
                    chunk_covariates=chunk_covariates,
                )

                # Extract valid variants (non-padded) to CPU numpy immediately
                valid_mask = mask[0].cpu().numpy()
                attr_matrix = attr[0].cpu().numpy()
                _validate_attribution_width(attr_matrix, expected_attribution_width)
                attr_valid = attr_matrix[valid_mask]

                # Get chromosomes from original variants (matching valid positions)
                valid_chroms = np.array([v.chrom for v in original_variants])[valid_mask]

                chunk_attributions.append(attr_valid)
                chunk_positions.append(positions[0][valid_mask].cpu().numpy())
                chunk_gene_ids.append(gene_ids[0][valid_mask].cpu().numpy())
                chunk_chromosomes.append(valid_chroms)

                # Free GPU tensors immediately after extracting to CPU
                del positions, gene_ids, mask, attr
                if chrom_ids is not None:
                    del chrom_ids

            # Combine all chunks for this sample
            sample_attributions = np.concatenate(chunk_attributions, axis=0)
            sample_positions = np.concatenate(chunk_positions, axis=0)
            sample_gene_ids = np.concatenate(chunk_gene_ids, axis=0)
            sample_chromosomes = np.concatenate(chunk_chromosomes, axis=0)

            # Aggregate to variant scores (L2 norm across features)
            if sample_attributions.ndim > 1:
                sample_variant_scores = np.linalg.norm(sample_attributions, ord=2, axis=1)
            else:
                sample_variant_scores = np.abs(sample_attributions)

            # Save this sample's full data to disk immediately (freed from RAM after)
            np.savez(
                tmp_dir / f'sample_{sample_idx}.npz',
                attributions=sample_attributions,
                variant_scores=sample_variant_scores,
                **_npz_scalar_metadata(ig_metadata, per_sample=True),
            )

            # Feed scores into ranker incrementally (then discard per-sample arrays)
            is_case = True if sample_idx in case_indices else (
                False if sample_idx in control_indices else None
            )
            ranker.accumulate_sample(
                variant_scores=sample_variant_scores,
                positions=sample_positions,
                gene_ids=sample_gene_ids,
                chromosomes=sample_chromosomes,
                sample_idx=sample_idx,
                is_case=is_case,
            )

            # Keep only lightweight metadata (small 1D arrays + scalars)
            sample_meta = {
                'positions': sample_positions,
                'gene_ids': sample_gene_ids,
                'chromosomes': sample_chromosomes,
                'sample_idx': sample_idx,
                'sample_id': all_samples[sample_idx].sample_id,
                'label': all_samples[sample_idx].label
            }
            all_metadata.append(sample_meta)
            metadata_variant_count += len(sample_positions)

            # Free per-sample arrays (full attributions + scores now on disk)
            del sample_attributions, sample_variant_scores
            del chunk_attributions, chunk_positions, chunk_gene_ids, chunk_chromosomes

            # Periodic garbage collection to reclaim Python overhead
            if (sample_idx + 1) % 50 == 0:
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()

        print(f"\nComputed attributions for {num_samples} samples")
        print(f"CRITICAL: All chunks processed - FULL GENOME coverage achieved!")

        # Diagnostic: Check what variants are in the metadata
        print(f"\nDiagnostic: Checking metadata variant distribution...")
        print(f"  Total variants in metadata across all samples: {metadata_variant_count:,}")

        # Check chromosome distribution in original data
        chrom_dist = Counter()
        for sample in all_samples:
            for variant in sample.variants:
                chrom_dist[variant.chrom] += 1

        print(f"  Chromosomes in original data: {len(chrom_dist)}")
        for chrom in sorted(chrom_dist.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x))[:10]:
            print(f"    Chr {chrom}: {chrom_dist[chrom]:,} variants")
        if len(chrom_dist) > 10:
            print(f"    ... and {len(chrom_dist) - 10} more chromosomes")

        # Check a sample of positions and genes from metadata
        if len(all_metadata) > 0 and len(all_metadata[0]['positions']) > 0:
            sample_meta = all_metadata[0]
            print(f"\n  First sample has {len(sample_meta['positions']):,} variants in attributions")
            print(f"  First 5 positions: {sample_meta['positions'][:5].tolist()}")
            print(f"  First 5 gene_ids: {sample_meta['gene_ids'][:5].tolist()}")

        # Recombine only lightweight variant_scores + metadata into attributions.npz
        # Raw per-feature attributions stay in per-sample files (too large to fit in RAM together)
        print("\nSaving variant scores and metadata...")
        all_variant_scores = []
        for sidx in range(num_samples):
            with np.load(tmp_dir / f'sample_{sidx}.npz', allow_pickle=False) as data:
                all_variant_scores.append(data['variant_scores'])

        attributions_path = output_dir / 'attributions.npz'
        np.savez(
            attributions_path,
            variant_scores=np.array(all_variant_scores, dtype=object),
            metadata=np.array(all_metadata, dtype=object),
            **_npz_scalar_metadata(ig_metadata, per_sample=False),
        )
        print(f"Saved variant scores + metadata to {attributions_path}")

        del all_variant_scores, all_metadata
        gc.collect()

        # Promote temp dir to permanent per-sample output (rename, no copy)
        per_sample_dir = output_dir / 'attributions_per_sample'
        if per_sample_dir.exists():
            shutil.rmtree(per_sample_dir)
        tmp_dir.rename(per_sample_dir)
        print(f"Per-sample raw attributions preserved in {per_sample_dir}/")
        print(f"  {num_samples} files, each containing 'attributions' and 'variant_scores' arrays")

        # === VARIANT RANKING (from incremental accumulation) ===
        print("\n" + "="*60)
        print("Ranking Variants")
        print("="*60)

        variant_rankings = ranker.finalize_rankings()

        print(f"Ranked {len(variant_rankings)} unique variants")

        # Diagnostic: Check chromosome distribution in variant rankings
        if 'chromosome' in variant_rankings.columns:
            ranking_chrom_counts = variant_rankings['chromosome'].value_counts()
            print(f"\nDiagnostic: Variant rankings chromosome distribution:")
            print(f"  Unique chromosomes: {len(ranking_chrom_counts)}")
            for chrom in sorted(ranking_chrom_counts.index, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x))[:10]:
                print(f"  Chr {chrom}: {ranking_chrom_counts[chrom]} variants")
            if len(ranking_chrom_counts) > 10:
                print(f"  ... and {len(ranking_chrom_counts) - 10} more chromosomes")

            if len(ranking_chrom_counts) == 1:
                print(f"  ❌ CRITICAL: Only 1 chromosome in rankings! This is the bug we're looking for.")
        else:
            print("  ⚠️ WARNING: No chromosome column in variant rankings!")

        # Rank genes, primary (max) and two alternatives
        gene_rankings = ranker.rank_genes(
            variant_rankings=variant_rankings,
            aggregation='max'
        )
        gene_rankings_mean = ranker.rank_genes(
            variant_rankings=variant_rankings,
            aggregation='mean'
        )
        gene_rankings_size_norm = ranker.rank_genes(
            variant_rankings=variant_rankings,
            aggregation='size_normalised'
        )

        print(f"Ranked {len(gene_rankings)} genes")

        # Get case-enriched variants
        if case_indices and control_indices:
            try:
                case_enriched = ranker.get_case_enriched_variants(
                    variant_rankings=variant_rankings,
                    min_case_samples=min(5, len(case_indices) // 4),
                    min_diff=0.05,
                    top_k=args.top_k_variants
                )
                print(f"Identified {len(case_enriched)} case-enriched variants")
            except (ValueError, KeyError):
                print("Not enough data for case-enriched analysis")
                case_enriched = None
        else:
            case_enriched = None

        variant_rankings = _annotate_ranking_metadata(variant_rankings, ig_metadata)
        gene_rankings = _annotate_ranking_metadata(gene_rankings, ig_metadata)
        gene_rankings_mean = _annotate_ranking_metadata(gene_rankings_mean, ig_metadata)
        gene_rankings_size_norm = _annotate_ranking_metadata(
            gene_rankings_size_norm, ig_metadata
        )

        # Export rankings
        ranker.export_rankings(
            variant_rankings=variant_rankings,
            gene_rankings=gene_rankings,
            output_dir=str(output_dir),
            prefix='sieve'
        )

        # Export alternative gene rankings
        gene_rankings_mean.to_csv(output_dir / 'sieve_gene_rankings_mean.csv', index=False)
        gene_rankings_size_norm.to_csv(
            output_dir / 'sieve_gene_rankings_size_normalised.csv', index=False
        )
        print(f"Alternative gene rankings saved:")
        print(f"  {output_dir / 'sieve_gene_rankings_mean.csv'}")
        print(f"  {output_dir / 'sieve_gene_rankings_size_normalised.csv'}")

    # === ATTENTION ANALYSIS ===
    if not args.skip_attention:
        print("\n" + "="*60)
        print("Analyzing Attention Patterns")
        print("="*60)

        # Create dataloader for attention analysis
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_chunks,
            num_workers=0
        )
        total_chunks = len(dataset)
        num_samples = len(all_samples)
        print(f"Created dataloader with {len(dataloader)} batches ({total_chunks} chunks from {num_samples} samples)")

        analyzer = AttentionAnalyzer(
            model=model,
            device=args.device,
            attention_threshold=args.attention_threshold,
            threshold_mode=args.attention_threshold_mode,
            attention_percentile=args.attention_percentile,
        )

        all_interactions = []
        interactions_by_sample = {}
        use_split_attention = _attention_uses_split_inputs(reconstruction)

        print("Extracting attention weights...")
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            positions = batch['positions'].to(args.device)
            gene_ids = batch['gene_ids'].to(args.device)
            mask = batch['mask'].to(args.device)
            chrom_ids = (
                batch['chrom_ids'].to(args.device)
                if 'chrom_ids' in batch else None
            )

            if use_split_attention:
                if (
                    'content_features' not in batch
                    or 'absolute_position_features' not in batch
                ):
                    raise ValueError(
                        "custom positional attention requires batch['content_features'] "
                        "and batch['absolute_position_features']"
                    )
                content_features = batch['content_features'].to(args.device)
                absolute_position_features = batch['absolute_position_features'].to(args.device)
                attention_weights = analyzer.extract_attention_weights(
                    variant_features=None,
                    positions=positions,
                    gene_ids=gene_ids,
                    mask=mask,
                    chrom_ids=chrom_ids,
                    content_features=content_features,
                    absolute_position_features=absolute_position_features,
                )
            else:
                features = batch['features'].to(args.device)
                attention_weights = analyzer.extract_attention_weights(
                    variant_features=features,
                    positions=positions,
                    gene_ids=gene_ids,
                    mask=mask,
                    chrom_ids=chrom_ids,
                )

            # Find interactions
            interactions = analyzer.find_top_interactions(
                attention_weights=attention_weights,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask,
                top_k=args.top_k_interactions,
                aggregate_layers='mean',
                aggregate_heads='mean',
                sample_indices=batch['original_sample_indices'],
                chunk_indices=batch['chunk_indices'],
            )

            all_interactions.extend(interactions)
            for interaction in interactions:
                interactions_by_sample.setdefault(interaction['sample_idx'], []).append(interaction)

            # Free GPU tensors after each batch
            del positions, gene_ids, mask, attention_weights
            if use_split_attention:
                del content_features, absolute_position_features
            else:
                del features
            if chrom_ids is not None:
                del chrom_ids
            if args.device == 'cuda':
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                chunks_done = min((batch_idx + 1) * args.batch_size, total_chunks)
                print(f"  Processed {chunks_done}/{total_chunks} chunks ({num_samples} samples)")

        print(f"Extracted {len(all_interactions)} interactions")

        # Aggregate across samples
        aggregated_interactions = analyzer.aggregate_interactions_across_samples(
            all_sample_interactions=list(interactions_by_sample.values()),
            min_samples=2
        )

        print(f"Found {len(aggregated_interactions)} recurring interactions")

        # Save interactions
        import pandas as pd
        interactions_df = pd.DataFrame(aggregated_interactions)
        interactions_path = output_dir / 'sieve_interactions.csv'
        interactions_df.to_csv(interactions_path, index=False)
        print(f"Saved interactions to {interactions_path}")

        # Free attention analysis data
        del all_interactions, interactions_by_sample, aggregated_interactions
        gc.collect()

    # === SAVE ANALYSIS METADATA ===
    analysis_metadata = {
        'is_null_baseline': args.is_null_baseline,
        'experiment_dir': str(args.experiment_dir) if args.experiment_dir else str(args.checkpoint),
        'genome_build': args.genome_build,
        'n_samples': len(all_samples),
        'annotation_level': config['level'],
        'n_integration_steps': args.n_steps,
        'max_variants_per_sample': args.max_variants,
        'aggregation_method': args.aggregation_method,
        'skip_attention': args.skip_attention,
        'skip_ig': args.skip_ig,
        'integrated_gradients': integrated_gradients_metadata,
        'attention_threshold_mode': args.attention_threshold_mode,
        'attention_threshold': args.attention_threshold,
        'attention_percentile': args.attention_percentile,
    }

    if variant_rankings is not None:
        analysis_metadata['n_ranked_variants'] = len(variant_rankings)
        analysis_metadata['n_ranked_genes'] = len(gene_rankings)

    metadata_path = output_dir / 'analysis_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(analysis_metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Analysis metadata saved to {metadata_path}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if variant_rankings is not None:
        print(f"\nTop 10 Variants by Attribution:")
        print(variant_rankings.head(10)[['position', 'gene_id', 'mean_attribution', 'num_samples']])

        print(f"\nTop 10 Genes:")
        print(gene_rankings.head(10)[['gene_id', 'num_variants', 'gene_score', 'top_variant_pos']])

        if case_enriched is not None and len(case_enriched) > 0:
            print(f"\nTop 10 Case-Enriched Variants:")
            print(case_enriched.head(10)[[
                'position', 'gene_id', 'case_attribution', 'control_attribution', 'case_control_diff'
            ]])
    else:
        print("\n(Integrated Gradients skipped - no variant rankings to display)")

    print(f"\nResults saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
