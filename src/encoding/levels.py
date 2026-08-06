"""
Annotation level encodings for SIEVE ablation experiments.

This module implements the annotation-ablation protocol, which quantifies how
much of a variant ranking is carried by genome structure and how much by the
supplied functional annotation scores.

There are four operational annotation levels, L0 to L3, with dimensions 1, 65,
69 and 71 respectively. A fifth enumerator, L4, exists as a compatibility
placeholder and is currently identical to L3.

Annotation Levels:
- L0: Genotype dosage only (ablation floor)
- L1: L0 + genomic position (test positional signal)
- L2: L1 + consequence class (minimal VEP)
- L3: L2 + SIFT + PolyPhen (standard functional scores)
- L4: compatibility placeholder, currently identical to L3

Author: Francesco Lescai
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.data import VariantRecord


class AnnotationLevel(Enum):
    """
    Enumeration of annotation levels for ablation experiments.

    Each level adds incremental annotation information:
    - L0: Genotype only (no annotations)
    - L1: + Position (test spatial signal)
    - L2: + Consequence (minimal VEP)
    - L3: + SIFT/PolyPhen (functional scores)
    - L4: compatibility placeholder, currently identical to L3
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


# Feature dimensions for each level
FEATURE_DIMENSIONS = {
    AnnotationLevel.L0: 1,      # genotype dosage
    AnnotationLevel.L1: 65,     # genotype + positional encoding (64D)
    AnnotationLevel.L2: 69,     # L1 + one-hot consequence severity (4D)
    AnnotationLevel.L3: 71,     # L2 + SIFT + PolyPhen (2D)
    AnnotationLevel.L4: 71,     # L3 + additional (currently same as L3)
}


CONTENT_FEATURE_DIMENSIONS = {
    AnnotationLevel.L0: 1,      # genotype dosage
    AnnotationLevel.L1: 1,      # genotype dosage, with position separated
    AnnotationLevel.L2: 5,      # genotype + one-hot consequence severity
    AnnotationLevel.L3: 7,      # L2 + SIFT + PolyPhen
    AnnotationLevel.L4: 7,      # L3 + additional (currently same as L3)
}


def get_feature_dimension(level: AnnotationLevel) -> int:
    """
    Get the feature dimension for a given annotation level.

    Parameters
    ----------
    level : AnnotationLevel
        The annotation level

    Returns
    -------
    int
        Feature dimension for this level

    Examples
    --------
    >>> get_feature_dimension(AnnotationLevel.L0)
    1
    >>> get_feature_dimension(AnnotationLevel.L3)
    71
    """
    return FEATURE_DIMENSIONS[level]


def get_content_feature_dimension(level: AnnotationLevel) -> int:
    """
    Get the non-positional content feature dimension for an annotation level.

    This does not alter the historical feature dimensions returned by
    get_feature_dimension(). It is used by the pure position-encoding
    configuration resolver to describe the future content/position split.

    Parameters
    ----------
    level : AnnotationLevel
        The annotation level

    Returns
    -------
    int
        Non-positional content feature dimension for this level
    """
    return CONTENT_FEATURE_DIMENSIONS[level]


def get_legacy_absolute_position_dimension(level: AnnotationLevel) -> int:
    """
    Get the historical absolute-position feature width for an annotation level.

    L0 never carried sinusoidal input features, so it uses a zero-width block.
    L1-L4 currently carry the fixed 64-dimensional sinusoidal block inside the
    historical ``features`` tensor.

    Parameters
    ----------
    level : AnnotationLevel
        The annotation level.

    Returns
    -------
    int
        Width of the historical absolute-position block.
    """
    if level == AnnotationLevel.L0:
        return 0
    return get_feature_dimension(level) - get_content_feature_dimension(level)


def _validate_feature_matrix(name: str, features: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional feature matrix; "
            f"got shape {features.shape}"
        )


def split_legacy_variant_features(
    features: np.ndarray,
    annotation_level: AnnotationLevel,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split historical variant features into content and absolute position.

    This helper treats the historical ``features`` matrix as the runtime
    authority. It does not independently re-encode biological annotations, which
    avoids drifting from the exact dosage, consequence, SIFT, PolyPhen,
    imputation, dtype, and ordering semantics that existing checkpoints learned.

    Parameters
    ----------
    features : np.ndarray
        Historical variant feature matrix with shape
        ``[num_variants, get_feature_dimension(annotation_level)]``.
    annotation_level : AnnotationLevel
        Annotation level used to create ``features``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(content_features, absolute_position_features)``. For L0, the
        absolute-position matrix has shape ``[num_variants, 0]`` so downstream
        code can handle all levels uniformly.
    """
    if not isinstance(annotation_level, AnnotationLevel):
        raise ValueError(f"Unknown annotation level: {annotation_level!r}")

    _validate_feature_matrix("features", features)
    expected_width = get_feature_dimension(annotation_level)
    if features.shape[1] != expected_width:
        raise ValueError(
            "features width does not match historical input dimension for "
            f"{annotation_level.value}: got {features.shape[1]}, "
            f"expected {expected_width}"
        )

    if annotation_level == AnnotationLevel.L0:
        content_features = features.copy()
        # A zero-width block records that L0 has no historical absolute
        # positional input while keeping the split representation rectangular.
        absolute_position_features = np.empty(
            (features.shape[0], 0),
            dtype=features.dtype,
        )
    else:
        # Derive the boundary from the dimension authorities instead of
        # duplicating the historical 64-column constant here. The legacy order
        # is [dosage, absolute position, annotations], so content is the dosage
        # prefix plus any annotation suffix.
        position_start = 1
        position_width = get_legacy_absolute_position_dimension(annotation_level)
        position_end = position_start + position_width
        absolute_position_features = features[:, position_start:position_end].copy()
        content_features = np.concatenate(
            [
                features[:, :position_start],
                features[:, position_end:],
            ],
            axis=1,
        )

    return (
        np.ascontiguousarray(content_features),
        np.ascontiguousarray(absolute_position_features),
    )


def compose_legacy_variant_features(
    content_features: np.ndarray,
    absolute_position_features: np.ndarray,
    annotation_level: AnnotationLevel,
) -> np.ndarray:
    """
    Recompose historical variant features from the split representation.

    The composed matrix preserves the exact legacy input width and ordering used
    by ``VariantEncoder``. This compatibility layer keeps historical
    ``features`` as the execution authority until model-side positional
    integration is implemented in a later phase.

    Parameters
    ----------
    content_features : np.ndarray
        Non-positional content feature matrix.
    absolute_position_features : np.ndarray
        Historical absolute-position feature matrix. For L0 this must have zero
        columns.
    annotation_level : AnnotationLevel
        Annotation level to compose.

    Returns
    -------
    np.ndarray
        Historical feature matrix with shape
        ``[num_variants, get_feature_dimension(annotation_level)]``.
    """
    if not isinstance(annotation_level, AnnotationLevel):
        raise ValueError(f"Unknown annotation level: {annotation_level!r}")

    _validate_feature_matrix("content_features", content_features)
    _validate_feature_matrix(
        "absolute_position_features", absolute_position_features
    )

    if content_features.dtype != absolute_position_features.dtype:
        raise ValueError(
            "content_features and absolute_position_features must have the "
            f"same dtype; got {content_features.dtype} and "
            f"{absolute_position_features.dtype}"
        )

    if content_features.shape[0] != absolute_position_features.shape[0]:
        raise ValueError(
            "content_features and absolute_position_features must have the "
            "same number of rows; got "
            f"{content_features.shape[0]} and {absolute_position_features.shape[0]}"
        )

    expected_content_width = get_content_feature_dimension(annotation_level)
    if content_features.shape[1] != expected_content_width:
        raise ValueError(
            "content_features width does not match content dimension for "
            f"{annotation_level.value}: got {content_features.shape[1]}, "
            f"expected {expected_content_width}"
        )

    expected_position_width = get_legacy_absolute_position_dimension(annotation_level)
    if absolute_position_features.shape[1] != expected_position_width:
        raise ValueError(
            "absolute_position_features width does not match historical "
            f"absolute-position dimension for {annotation_level.value}: got "
            f"{absolute_position_features.shape[1]}, expected "
            f"{expected_position_width}"
        )

    if annotation_level == AnnotationLevel.L0:
        features = content_features.copy()
    else:
        # The position block is inserted after dosage, not appended, because old
        # L2-L4 checkpoints learned [dosage, position, annotations].
        features = np.concatenate(
            [
                content_features[:, :1],
                absolute_position_features,
                content_features[:, 1:],
            ],
            axis=1,
        )

    return np.ascontiguousarray(features)


def encode_genotype(variant: VariantRecord) -> np.ndarray:
    """
    Encode genotype dosage as single feature.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode

    Returns
    -------
    np.ndarray
        Array of shape (1,) containing genotype dosage (0, 1, or 2)

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {})
    >>> encode_genotype(var)
    array([1.])
    """
    return np.array([float(variant.genotype)], dtype=np.float32)


def encode_consequence_severity(variant: VariantRecord) -> np.ndarray:
    """
    Encode consequence severity as one-hot vector.

    Severity levels:
    - 0: Unknown
    - 1: MODIFIER (intron, intergenic, etc.)
    - 2: LOW (synonymous, UTR, splice region)
    - 3: MODERATE (missense, inframe indels)
    - 4: HIGH (LoF: stop_gained, frameshift, splice donor/acceptor)

    One-hot encoding: [is_MODIFIER, is_LOW, is_MODERATE, is_HIGH]
    Unknown (0) is encoded as all zeros.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode

    Returns
    -------
    np.ndarray
        One-hot encoded consequence severity, shape (4,)

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {})
    >>> encode_consequence_severity(var)
    array([0., 0., 1., 0.])
    """
    from src.data import map_consequence_to_severity

    severity = map_consequence_to_severity(variant.consequence)

    # One-hot encode (skip level 0 which is unknown)
    one_hot = np.zeros(4, dtype=np.float32)
    if severity > 0:
        one_hot[severity - 1] = 1.0

    return one_hot


def encode_functional_scores(
    variant: VariantRecord,
    impute_value: float = 0.5
) -> np.ndarray:
    """
    Encode SIFT and PolyPhen scores.

    Both scores are normalised to [0, 1] where higher = more deleterious.
    Missing values are imputed with neutral value (default 0.5).

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode
    impute_value : float
        Value to use for missing scores (default: 0.5 = neutral)

    Returns
    -------
    np.ndarray
        Array [SIFT, PolyPhen], shape (2,)

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1,
    ...                     {'sift': 0.05, 'polyphen': 0.9})
    >>> encode_functional_scores(var)
    array([0.95, 0.9])
    """
    from src.data import normalize_sift_score, normalize_polyphen_score

    # Get raw scores
    sift_raw = variant.annotations.get('sift')
    polyphen_raw = variant.annotations.get('polyphen')

    # Normalise (SIFT is inverted, PolyPhen stays same)
    if sift_raw is not None:
        sift = normalize_sift_score(sift_raw)
    else:
        sift = impute_value

    if polyphen_raw is not None:
        polyphen = normalize_polyphen_score(polyphen_raw)
    else:
        polyphen = impute_value

    return np.array([sift, polyphen], dtype=np.float32)


def encode_variant_L0(variant: VariantRecord) -> np.ndarray:
    """
    Encode variant at Level 0: genotype only.

    This is the ablation floor of the protocol: it measures what the model
    learns from genotype patterns alone, with every functional annotation
    removed.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode

    Returns
    -------
    np.ndarray
        Feature vector of shape (1,) containing genotype dosage

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 2, {})
    >>> encode_variant_L0(var)
    array([2.])
    """
    return encode_genotype(variant)


def encode_variant_L1(
    variant: VariantRecord,
    position_encoding: np.ndarray
) -> np.ndarray:
    """
    Encode variant at Level 1: genotype + position.

    Tests whether genomic position carries disease signal beyond genotype alone.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode
    position_encoding : np.ndarray
        Pre-computed positional encoding for this variant's position, shape (64,)

    Returns
    -------
    np.ndarray
        Feature vector of shape (65,): [genotype, position_encoding(64)]

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {})
    >>> pos_enc = np.random.randn(64).astype(np.float32)
    >>> encoded = encode_variant_L1(var, pos_enc)
    >>> encoded.shape
    (65,)
    """
    genotype_feature = encode_genotype(variant)
    return np.concatenate([genotype_feature, position_encoding])


def encode_variant_L2(
    variant: VariantRecord,
    position_encoding: np.ndarray
) -> np.ndarray:
    """
    Encode variant at Level 2: genotype + position + consequence.

    Tests whether minimal VEP annotation (consequence type) improves discovery
    beyond position alone.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode
    position_encoding : np.ndarray
        Pre-computed positional encoding, shape (64,)

    Returns
    -------
    np.ndarray
        Feature vector of shape (69,):
        [genotype, position_encoding(64), consequence_one_hot(4)]

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {})
    >>> pos_enc = np.random.randn(64).astype(np.float32)
    >>> encoded = encode_variant_L2(var, pos_enc)
    >>> encoded.shape
    (69,)
    """
    l1_features = encode_variant_L1(variant, position_encoding)
    consequence_features = encode_consequence_severity(variant)
    return np.concatenate([l1_features, consequence_features])


def encode_variant_L3(
    variant: VariantRecord,
    position_encoding: np.ndarray,
    impute_value: float = 0.5
) -> np.ndarray:
    """
    Encode variant at Level 3: L2 + SIFT + PolyPhen.

    Tests whether standard functional prediction scores improve discovery
    beyond consequence annotation alone. This is the comparison point with
    traditional annotation-based methods.

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode
    position_encoding : np.ndarray
        Pre-computed positional encoding, shape (64,)
    impute_value : float
        Value for missing SIFT/PolyPhen scores (default: 0.5 = neutral)

    Returns
    -------
    np.ndarray
        Feature vector of shape (71,):
        [genotype, position_encoding(64), consequence_one_hot(4), SIFT, PolyPhen]

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1,
    ...                     {'sift': 0.05, 'polyphen': 0.9})
    >>> pos_enc = np.random.randn(64).astype(np.float32)
    >>> encoded = encode_variant_L3(var, pos_enc)
    >>> encoded.shape
    (71,)
    """
    l2_features = encode_variant_L2(variant, position_encoding)
    functional_features = encode_functional_scores(variant, impute_value)
    return np.concatenate([l2_features, functional_features])


def encode_variant_L4(
    variant: VariantRecord,
    position_encoding: np.ndarray,
    impute_value: float = 0.5
) -> np.ndarray:
    """
    Encode variant at Level 4: L3 + additional annotations.

    Full annotation including any additional features (currently same as L3,
    but can be extended to include CADD, LoF flags, etc.).

    Parameters
    ----------
    variant : VariantRecord
        Variant to encode
    position_encoding : np.ndarray
        Pre-computed positional encoding, shape (64,)
    impute_value : float
        Value for missing scores (default: 0.5 = neutral)

    Returns
    -------
    np.ndarray
        Feature vector of shape (71,) (currently same as L3)

    Notes
    -----
    Can be extended to include:
    - CADD scores
    - LoF confidence flags
    - Conservation scores (PhyloP, PhastCons)
    - Regulatory annotations
    """
    # Currently same as L3, but extensible
    return encode_variant_L3(variant, position_encoding, impute_value)


def encode_variants(
    variants: List[VariantRecord],
    level: AnnotationLevel,
    position_encodings: Optional[np.ndarray] = None,
    impute_value: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Encode multiple variants at specified annotation level.

    Parameters
    ----------
    variants : List[VariantRecord]
        List of variants to encode
    level : AnnotationLevel
        Annotation level to use
    position_encodings : Optional[np.ndarray]
        Pre-computed positional encodings for all variants, shape (n_variants, 64).
        Required for L1-L4, ignored for L0.
    impute_value : float
        Value for missing functional scores (default: 0.5 = neutral)

    Returns
    -------
    features : np.ndarray
        Encoded features, shape (n_variants, feature_dim)
    positions : np.ndarray
        Genomic positions, shape (n_variants,)
    gene_symbols : List[str]
        Gene symbols for each variant

    Raises
    ------
    ValueError
        If position_encodings is None for levels L1-L4

    Examples
    --------
    >>> from src.data import VariantRecord
    >>> vars = [
    ...     VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {}),
    ...     VariantRecord('1', 200, 'C', 'G', 'GENE2', 'synonymous_variant', 2, {}),
    ... ]
    >>> features, positions, genes = encode_variants(vars, AnnotationLevel.L0)
    >>> features.shape
    (2, 1)
    >>> positions
    array([100, 200])
    >>> genes
    ['GENE1', 'GENE2']
    """
    if len(variants) == 0:
        feature_dim = get_feature_dimension(level)
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            []
        )

    # Check position encodings for L1-L4
    if level != AnnotationLevel.L0 and position_encodings is None:
        raise ValueError(f"Position encodings required for {level.value}")

    # Encode each variant
    encoded_features = []
    positions = []
    gene_symbols = []

    for i, variant in enumerate(variants):
        # Get position encoding if needed
        pos_enc = position_encodings[i] if position_encodings is not None else None

        # Encode based on level
        if level == AnnotationLevel.L0:
            features = encode_variant_L0(variant)
        elif level == AnnotationLevel.L1:
            features = encode_variant_L1(variant, pos_enc)
        elif level == AnnotationLevel.L2:
            features = encode_variant_L2(variant, pos_enc)
        elif level == AnnotationLevel.L3:
            features = encode_variant_L3(variant, pos_enc, impute_value)
        elif level == AnnotationLevel.L4:
            features = encode_variant_L4(variant, pos_enc, impute_value)
        else:
            raise ValueError(f"Unknown annotation level: {level}")

        encoded_features.append(features)
        positions.append(variant.pos)
        gene_symbols.append(variant.gene)

    # Stack into arrays
    features_array = np.stack(encoded_features, axis=0)
    positions_array = np.array(positions, dtype=np.int64)

    return features_array, positions_array, gene_symbols


def get_level_description(level: AnnotationLevel) -> str:
    """
    Get human-readable description of annotation level.

    Parameters
    ----------
    level : AnnotationLevel
        The annotation level

    Returns
    -------
    str
        Description of what features are included

    Examples
    --------
    >>> get_level_description(AnnotationLevel.L0)
    'L0: Genotype only (ablation floor)'
    >>> get_level_description(AnnotationLevel.L3)
    'L3: Genotype + Position + Consequence + SIFT + PolyPhen'
    """
    descriptions = {
        AnnotationLevel.L0: "L0: Genotype only (ablation floor)",
        AnnotationLevel.L1: "L1: Genotype + Position",
        AnnotationLevel.L2: "L2: Genotype + Position + Consequence",
        AnnotationLevel.L3: "L3: Genotype + Position + Consequence + SIFT + PolyPhen",
        AnnotationLevel.L4: "L4: Compatibility placeholder (currently identical to L3)",
    }
    return descriptions[level]


def summarize_level_features(level: AnnotationLevel) -> Dict[str, any]:
    """
    Get summary of features included in annotation level.

    Parameters
    ----------
    level : AnnotationLevel
        The annotation level

    Returns
    -------
    Dict[str, any]
        Summary dictionary with keys:
        - 'level': Level name
        - 'feature_dim': Total feature dimension
        - 'includes_genotype': bool
        - 'includes_position': bool
        - 'includes_consequence': bool
        - 'includes_functional_scores': bool

    Examples
    --------
    >>> summary = summarize_level_features(AnnotationLevel.L2)
    >>> summary['feature_dim']
    69
    >>> summary['includes_consequence']
    True
    >>> summary['includes_functional_scores']
    False
    """
    return {
        'level': level.value,
        'feature_dim': get_feature_dimension(level),
        'includes_genotype': True,  # All levels include genotype
        'includes_position': level in [AnnotationLevel.L1, AnnotationLevel.L2,
                                       AnnotationLevel.L3, AnnotationLevel.L4],
        'includes_consequence': level in [AnnotationLevel.L2, AnnotationLevel.L3,
                                          AnnotationLevel.L4],
        'includes_functional_scores': level in [AnnotationLevel.L3, AnnotationLevel.L4],
    }
