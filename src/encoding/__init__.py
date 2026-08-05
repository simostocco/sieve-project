"""
Feature encoding module for SIEVE.

This module handles:
1. Multi-level annotation encoding (L0-L4 ablation experiments)
2. Positional encoding (sinusoidal for features, bucketing for attention)
3. Sparse tensor construction and batching

Author: Francesco Lescai
"""

from .levels import (
    AnnotationLevel,
    CONTENT_FEATURE_DIMENSIONS,
    FEATURE_DIMENSIONS,
    get_content_feature_dimension,
    get_feature_dimension,
    encode_genotype,
    encode_consequence_severity,
    encode_functional_scores,
    encode_variant_L0,
    encode_variant_L1,
    encode_variant_L2,
    encode_variant_L3,
    encode_variant_L4,
    encode_variants,
    get_level_description,
    summarize_level_features,
)

from .position_config import (
    AbsolutePositionEncoding,
    AbsolutePositionFusion,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    ResolvedAbsolutePositionConfig,
    ResolvedAttributionConfig,
    ResolvedChromosomeConfig,
    ResolvedIGMode,
    ResolvedPositionEncodingConfig,
    ResolvedRelativePositionConfig,
    resolve_position_encoding_config,
)

from .positional import (
    sinusoidal_position_encoding,
    relative_position_bucket,
    compute_sinusoidal_encodings_batch,
    get_relative_distance_statistics,
    visualize_positional_encoding,
    test_encoding_consistency,
)

from .sparse_tensor import (
    build_gene_index,
    build_chrom_index,
    build_variant_tensor,
    collate_samples,
    VariantDataset,
    test_sparse_tensor,
)

from .chunked_dataset import (
    ChunkedVariantDataset,
    collate_chunks,
)

__all__ = [
    # Annotation levels
    'AnnotationLevel',
    'CONTENT_FEATURE_DIMENSIONS',
    'FEATURE_DIMENSIONS',
    'get_content_feature_dimension',
    'get_feature_dimension',
    'encode_genotype',
    'encode_consequence_severity',
    'encode_functional_scores',
    'encode_variant_L0',
    'encode_variant_L1',
    'encode_variant_L2',
    'encode_variant_L3',
    'encode_variant_L4',
    'encode_variants',
    'get_level_description',
    'summarize_level_features',

    # Position encoding configuration
    'AbsolutePositionEncoding',
    'AbsolutePositionFusion',
    'AlibiDistanceFunction',
    'ChromosomeEncoding',
    'CrossChromosomePolicy',
    'PositionEncodingRequest',
    'PositionPreset',
    'RelativePositionEncoding',
    'ResolvedAbsolutePositionConfig',
    'ResolvedAttributionConfig',
    'ResolvedChromosomeConfig',
    'ResolvedIGMode',
    'ResolvedPositionEncodingConfig',
    'ResolvedRelativePositionConfig',
    'resolve_position_encoding_config',

    # Positional encoding
    'sinusoidal_position_encoding',
    'relative_position_bucket',
    'compute_sinusoidal_encodings_batch',
    'get_relative_distance_statistics',
    'visualize_positional_encoding',
    'test_encoding_consistency',

    # Sparse tensor construction
    'build_gene_index',
    'build_chrom_index',
    'build_variant_tensor',
    'collate_samples',
    'VariantDataset',
    'test_sparse_tensor',

    # Chunked processing (whole-genome coverage)
    'ChunkedVariantDataset',
    'collate_chunks',
]
