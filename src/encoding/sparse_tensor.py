"""
Sparse tensor construction for SIEVE.

This module handles the conversion from variant records to PyTorch tensors,
including batching, padding, and gene index mapping. It addresses the fundamental
challenge that exomes have millions of positions but individuals have variants
at only thousands of positions.

Key functions:
- build_variant_tensor: Convert single sample to tensors
- collate_samples: Batch multiple samples with padding
- build_gene_index: Create gene symbol → integer mapping
- VariantDataset: PyTorch Dataset wrapper

Author: Francesco Lescai
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data import SampleVariants
from .levels import (
    AnnotationLevel,
    encode_variants,
    get_content_feature_dimension,
    get_feature_dimension,
    get_legacy_absolute_position_dimension,
    split_legacy_variant_features,
)
from .positional import sinusoidal_position_encoding


_SPLIT_FEATURE_KEYS = ('content_features', 'absolute_position_features')


def _validate_split_feature_schema(batch: list[dict[str, Any]]) -> bool:
    """Validate whether a batch consistently carries split feature tensors."""
    split_presence = [
        tuple(key in sample for key in _SPLIT_FEATURE_KEYS)
        for sample in batch
    ]

    if all(presence == (False, False) for presence in split_presence):
        return False
    if not all(presence == (True, True) for presence in split_presence):
        raise ValueError(
            "Batches must either omit both split feature keys or include both "
            "'content_features' and 'absolute_position_features' for every sample."
        )

    # Validate rank and row alignment before collation. Otherwise padding or
    # truncation could silently hide a malformed split tensor that no longer
    # corresponds row-for-row to the historical ``features`` authority.
    for sample_idx, sample in enumerate(batch):
        for key in ('features', 'content_features', 'absolute_position_features'):
            tensor = sample[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"Sample {sample_idx} tensor '{key}' must be a torch.Tensor; "
                    f"got {type(tensor).__name__}"
                )
            if tensor.ndim != 2:
                raise ValueError(
                    f"Sample {sample_idx} tensor '{key}' must be two-dimensional; "
                    f"got shape {tuple(tensor.shape)}"
                )

        feature_rows = sample['features'].shape[0]
        for key in ('content_features', 'absolute_position_features'):
            split_rows = sample[key].shape[0]
            if split_rows != feature_rows:
                raise ValueError(
                    f"Sample {sample_idx} tensor '{key}' row count must match "
                    f"'features': got {split_rows}, expected {feature_rows}"
                )

    content_dim = batch[0]['content_features'].shape[1]
    position_dim = batch[0]['absolute_position_features'].shape[1]
    for sample_idx, sample in enumerate(batch):
        if sample['content_features'].shape[1] != content_dim:
            raise ValueError(
                f"Inconsistent content feature width for sample {sample_idx}: "
                f"got {sample['content_features'].shape[1]}, expected {content_dim}"
            )
        if sample['absolute_position_features'].shape[1] != position_dim:
            raise ValueError(
                f"Inconsistent absolute-position feature width for sample {sample_idx}: got "
                f"{sample['absolute_position_features'].shape[1]}, expected "
                f"{position_dim}"
            )

    return True


def build_gene_index(all_samples: List[SampleVariants]) -> Dict[str, int]:
    """
    Build mapping from gene symbols to integer indices.

    This creates a consistent gene index across all samples for the gene
    aggregation layer in the model.

    Parameters
    ----------
    all_samples : List[SampleVariants]
        All samples in the dataset

    Returns
    -------
    Dict[str, int]
        Mapping from gene symbol to integer index

    Examples
    --------
    >>> from src.data import SampleVariants, VariantRecord
    >>> var1 = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense', 1, {})
    >>> var2 = VariantRecord('1', 200, 'C', 'G', 'GENE2', 'missense', 1, {})
    >>> sample = SampleVariants('sample1', 1, [var1, var2])
    >>> gene_idx = build_gene_index([sample])
    >>> gene_idx
    {'GENE1': 0, 'GENE2': 1}
    """
    # Collect all unique gene symbols
    gene_symbols = set()
    for sample in all_samples:
        for variant in sample.variants:
            gene_symbols.add(variant.gene)

    # Sort for deterministic ordering
    sorted_genes = sorted(gene_symbols)

    # Create mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(sorted_genes)}

    return gene_to_idx


def _chrom_sort_key(chrom: str):
    """Natural sort key for chromosome names: 1..22, X, Y, MT, then anything else."""
    s = chrom[3:] if chrom.startswith('chr') else chrom
    if s.isdigit():
        return (0, int(s), '')
    if s == 'X':
        return (1, 0, '')
    if s == 'Y':
        return (1, 1, '')
    if s in ('MT', 'M'):
        return (1, 2, '')
    return (2, 0, s)


def build_chrom_index(all_samples: List[SampleVariants]) -> Dict[str, int]:
    """
    Build mapping from chromosome names to integer indices.

    Used by the model to disambiguate variants that share genomic coordinates
    on different chromosomes (positional features alone are chromosome-blind)
    and to route cross-chromosome relative-position pairs to a dedicated
    attention bias bucket.

    Parameters
    ----------
    all_samples : List[SampleVariants]
        All samples in the dataset.

    Returns
    -------
    Dict[str, int]
        Mapping from chromosome name (as it appears on ``VariantRecord.chrom``)
        to integer index. Standard human chromosomes are ordered 1..22, X, Y,
        MT; any non-standard names are appended in lexical order.

    Examples
    --------
    >>> from src.data import SampleVariants, VariantRecord
    >>> v1 = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense', 1, {})
    >>> v2 = VariantRecord('X', 200, 'C', 'G', 'GENE2', 'missense', 1, {})
    >>> sample = SampleVariants('s1', 1, [v1, v2])
    >>> idx = build_chrom_index([sample])
    >>> idx['1'] < idx['X']
    True
    """
    chrom_names = set()
    for sample in all_samples:
        for variant in sample.variants:
            chrom_names.add(variant.chrom)

    sorted_chroms = sorted(chrom_names, key=_chrom_sort_key)
    return {chrom: idx for idx, chrom in enumerate(sorted_chroms)}


def build_variant_tensor(
    sample: SampleVariants,
    annotation_level: AnnotationLevel,
    gene_index: Dict[str, int],
    max_variants: Optional[int] = None,
    impute_value: float = 0.5,
    chrom_index: Optional[Dict[str, int]] = None,
) -> Dict[str, Tensor]:
    """
    Convert a single sample's variants to PyTorch tensors.

    Parameters
    ----------
    sample : SampleVariants
        Sample containing variants and label
    annotation_level : AnnotationLevel
        Annotation level to use for encoding
    gene_index : Dict[str, int]
        Mapping from gene symbol to integer index
    max_variants : Optional[int]
        If specified, limit to first N variants (for debugging/testing)
    impute_value : float
        Value for missing functional scores (default: 0.5)
    chrom_index : Optional[Dict[str, int]]
        Mapping from chromosome name to integer index. When provided, the
        returned dict includes a ``chrom_ids`` tensor used by chromosome-aware
        positional bias and chromosome embedding in the model. When ``None``,
        ``chrom_ids`` is omitted (legacy chromosome-blind path).

    Returns
    -------
    Dict[str, Tensor]
        Dictionary containing:
        - 'features': Historical execution features
          [num_variants, feature_dim]. This remains the tensor consumed by the
          current model path.
        - 'content_features': Non-positional content features
          [num_variants, content_dim]
        - 'absolute_position_features': Historical absolute-position features
          [num_variants, absolute_position_dim]. For L0 this is a zero-width
          tensor with shape [num_variants, 0].
        - 'positions': Genomic positions [num_variants]
        - 'gene_ids': Gene indices [num_variants]
        - 'mask': Valid variant mask [num_variants] (all 1s, no padding yet)
        - 'label': Phenotype label [scalar]
        - 'sample_id': Sample identifier (string, not tensor)
        - 'chrom_ids': Chromosome indices [num_variants] (only if
          ``chrom_index`` is provided)

    Examples
    --------
    >>> from src.data import SampleVariants, VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense', 1, {})
    >>> sample = SampleVariants('sample1', 1, [var])
    >>> gene_idx = {'GENE1': 0}
    >>> tensors = build_variant_tensor(sample, AnnotationLevel.L0, gene_idx)
    >>> tensors['features'].shape
    torch.Size([1, 1])
    >>> tensors['label']
    tensor(1)
    """
    variants = sample.variants
    if max_variants is not None:
        variants = variants[:max_variants]

    n_variants = len(variants)
    feature_dim = get_feature_dimension(annotation_level)
    content_dim = get_content_feature_dimension(annotation_level)
    absolute_position_dim = get_legacy_absolute_position_dimension(annotation_level)

    # Handle empty variant case
    if n_variants == 0:
        # ``features`` remains the runtime tensor consumed by the model. The
        # split tensors are derived alongside it so later phases can separate
        # content from absolute position without changing empty-sample padding
        # semantics.
        empty = {
            'features': torch.zeros((0, feature_dim), dtype=torch.float32),
            'content_features': torch.zeros((0, content_dim), dtype=torch.float32),
            'absolute_position_features': torch.zeros(
                (0, absolute_position_dim),
                dtype=torch.float32,
            ),
            'positions': torch.zeros(0, dtype=torch.long),
            'gene_ids': torch.zeros(0, dtype=torch.long),
            'mask': torch.zeros(0, dtype=torch.bool),
            'label': torch.tensor(sample.label, dtype=torch.long),
            'sample_id': sample.sample_id,
        }
        if chrom_index is not None:
            empty['chrom_ids'] = torch.zeros(0, dtype=torch.long)
        return empty

    # Extract positions for positional encoding
    positions_np = np.array([v.pos for v in variants], dtype=np.int64)

    # Compute positional encodings if needed (L1-L4)
    if annotation_level != AnnotationLevel.L0:
        position_encodings = sinusoidal_position_encoding(positions_np, d_model=64)
    else:
        position_encodings = None

    # Encode variants
    features_np, positions_np, gene_symbols = encode_variants(
        variants,
        annotation_level,
        position_encodings,
        impute_value
    )
    content_features_np, absolute_position_features_np = split_legacy_variant_features(
        features_np,
        annotation_level,
    )

    # Map gene symbols to indices
    gene_ids_np = np.array([gene_index[gene] for gene in gene_symbols], dtype=np.int64)

    # Create mask (all True since no padding yet)
    mask_np = np.ones(n_variants, dtype=bool)

    # Convert to tensors
    out = {
        # Historical features remain the execution authority. The split is
        # derived from this matrix rather than independently re-encoding
        # annotations, preserving legacy ordering, dtype, and checkpoint input
        # compatibility exactly.
        'features': torch.from_numpy(np.ascontiguousarray(features_np)),
        'content_features': torch.from_numpy(content_features_np),
        'absolute_position_features': torch.from_numpy(absolute_position_features_np),
        'positions': torch.from_numpy(positions_np),
        'gene_ids': torch.from_numpy(gene_ids_np),
        'mask': torch.from_numpy(mask_np),
        'label': torch.tensor(sample.label, dtype=torch.long),
        'sample_id': sample.sample_id,
    }
    if chrom_index is not None:
        chrom_ids_np = np.array(
            [chrom_index[v.chrom] for v in variants], dtype=np.int64
        )
        out['chrom_ids'] = torch.from_numpy(chrom_ids_np)
    return out


def collate_samples(
    batch: List[Dict[str, Any]],
    max_variants_per_batch: Optional[int] = 3000
) -> Dict[str, Tensor]:
    """
    Collate multiple samples into a padded batch with optional variant limit.

    Pads all samples to the maximum number of variants within the batch.
    This is more memory-efficient than padding to a global maximum.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of sample tensors from build_variant_tensor

    Returns
    -------
    Dict[str, Tensor]
        Batched tensors:
        - 'features': [batch_size, max_variants, feature_dim]
        - 'content_features': [batch_size, max_variants, content_dim] when
          every input item contains both split feature keys
        - 'absolute_position_features':
          [batch_size, max_variants, absolute_position_dim] when every input
          item contains both split feature keys
        - 'positions': [batch_size, max_variants]
        - 'gene_ids': [batch_size, max_variants]
        - 'mask': [batch_size, max_variants] (1=real, 0=padding)
        - 'labels': [batch_size]
        - 'sample_ids': List of sample IDs (not a tensor)

    Notes
    -----
    Padding is done with zeros, and the mask indicates which positions are real.
    The model should use the mask to ignore padding positions.
    Split-aware batches include the two split feature tensors only when every
    input item contains both split keys. Legacy manually constructed
    dictionaries that omit both keys retain the historical output schema.

    Examples
    --------
    >>> # Assume we have two samples with different numbers of variants
    >>> sample1 = {
    ...     'features': torch.randn(3, 5),
    ...     'positions': torch.tensor([100, 200, 300]),
    ...     'gene_ids': torch.tensor([0, 1, 2]),
    ...     'mask': torch.ones(3, dtype=torch.bool),
    ...     'label': torch.tensor(1),
    ...     'sample_id': 'sample1'
    ... }
    >>> sample2 = {
    ...     'features': torch.randn(5, 5),
    ...     'positions': torch.tensor([100, 200, 300, 400, 500]),
    ...     'gene_ids': torch.tensor([0, 1, 2, 3, 4]),
    ...     'mask': torch.ones(5, dtype=torch.bool),
    ...     'label': torch.tensor(0),
    ...     'sample_id': 'sample2'
    ... }
    >>> batch = collate_samples([sample1, sample2])
    >>> batch['features'].shape
    torch.Size([2, 5, 5])
    >>> batch['mask'].sum(dim=1)  # Number of real variants per sample
    tensor([3, 5])
    """
    batch_size = len(batch)
    has_split_features = _validate_split_feature_schema(batch)

    # Get max number of variants in this batch
    max_variants = max(sample['features'].shape[0] for sample in batch)

    # Cap at max_variants_per_batch to prevent OOM
    if max_variants_per_batch is not None and max_variants > max_variants_per_batch:
        max_variants = max_variants_per_batch

    # Detect whether the per-sample tensors carry chrom_ids. All samples in a
    # batch must agree (mixed batches are not supported).
    has_chrom_ids = any('chrom_ids' in sample for sample in batch)
    if has_chrom_ids and not all('chrom_ids' in sample for sample in batch):
        raise ValueError(
            "Mixed batches with and without 'chrom_ids' are not supported."
        )

    # Handle edge case where all samples have zero variants
    if max_variants == 0:
        # Find feature dimension from any sample with known dimension
        # If all samples have 0 variants, use the expected feature dimension from first sample
        # Even with shape [0, feature_dim], we can get feature_dim
        feature_dim = batch[0]['features'].shape[1] if len(batch[0]['features'].shape) > 1 else 0

        # Return empty batch with proper structure
        empty = {
            'features': torch.zeros((batch_size, 0, feature_dim), dtype=torch.float32),
            'positions': torch.zeros((batch_size, 0), dtype=torch.long),
            'gene_ids': torch.zeros((batch_size, 0), dtype=torch.long),
            'mask': torch.zeros((batch_size, 0), dtype=torch.bool),
            'labels': torch.tensor([sample['label'] for sample in batch], dtype=torch.long),
            'sample_ids': [sample['sample_id'] for sample in batch],
        }
        if has_chrom_ids:
            empty['chrom_ids'] = torch.zeros((batch_size, 0), dtype=torch.long)
        if has_split_features:
            content_dim = batch[0]['content_features'].shape[1]
            absolute_position_dim = batch[0]['absolute_position_features'].shape[1]
            empty['content_features'] = torch.zeros(
                (batch_size, 0, content_dim),
                dtype=torch.float32,
            )
            empty['absolute_position_features'] = torch.zeros(
                (batch_size, 0, absolute_position_dim),
                dtype=torch.float32,
            )
        return empty

    # Get feature dimension from first sample (safe now since max_variants > 0)
    feature_dim = batch[0]['features'].shape[1]
    content_dim = batch[0]['content_features'].shape[1] if has_split_features else 0
    absolute_position_dim = (
        batch[0]['absolute_position_features'].shape[1]
        if has_split_features else 0
    )

    # Initialize padded tensors
    features_padded = torch.zeros(
        (batch_size, max_variants, feature_dim),
        dtype=torch.float32
    )
    # Padded rows intentionally remain zeros for both split tensors, matching
    # historical ``features`` padding and avoiding positional values for masked
    # slots.
    content_features_padded = (
        torch.zeros((batch_size, max_variants, content_dim), dtype=torch.float32)
        if has_split_features else None
    )
    absolute_position_features_padded = (
        torch.zeros(
            (batch_size, max_variants, absolute_position_dim),
            dtype=torch.float32,
        )
        if has_split_features else None
    )
    positions_padded = torch.zeros(
        (batch_size, max_variants),
        dtype=torch.long
    )
    gene_ids_padded = torch.zeros(
        (batch_size, max_variants),
        dtype=torch.long
    )
    mask_padded = torch.zeros(
        (batch_size, max_variants),
        dtype=torch.bool
    )
    labels = torch.zeros(batch_size, dtype=torch.long)
    sample_ids = []
    # Pad chrom_ids with zeros (a real chromosome index). The attention mask
    # already prevents padded slots from contributing to the loss; the value
    # is irrelevant operationally as long as it indexes a valid embedding row.
    chrom_ids_padded = (
        torch.zeros((batch_size, max_variants), dtype=torch.long)
        if has_chrom_ids else None
    )

    # Fill in the data
    for i, sample in enumerate(batch):
        n_variants = sample['features'].shape[0]

        if n_variants > 0:
            # Truncate to max_variants if necessary
            n_to_copy = min(n_variants, max_variants)
            features_padded[i, :n_to_copy] = sample['features'][:n_to_copy]
            if has_split_features:
                content_features_padded[i, :n_to_copy] = (
                    sample['content_features'][:n_to_copy]
                )
                absolute_position_features_padded[i, :n_to_copy] = (
                    sample['absolute_position_features'][:n_to_copy]
                )
            positions_padded[i, :n_to_copy] = sample['positions'][:n_to_copy]
            gene_ids_padded[i, :n_to_copy] = sample['gene_ids'][:n_to_copy]
            mask_padded[i, :n_to_copy] = sample['mask'][:n_to_copy]
            if has_chrom_ids:
                chrom_ids_padded[i, :n_to_copy] = sample['chrom_ids'][:n_to_copy]

        labels[i] = sample['label']
        sample_ids.append(sample['sample_id'])

    out = {
        'features': features_padded,
        'positions': positions_padded,
        'gene_ids': gene_ids_padded,
        'mask': mask_padded,
        'labels': labels,
        'sample_ids': sample_ids,
    }
    if has_chrom_ids:
        out['chrom_ids'] = chrom_ids_padded
    if has_split_features:
        out['content_features'] = content_features_padded
        out['absolute_position_features'] = absolute_position_features_padded
    return out


class VariantDataset(Dataset):
    """
    PyTorch Dataset for variant data.

    Wraps SampleVariants data for use with PyTorch DataLoader.

    Parameters
    ----------
    samples : List[SampleVariants]
        List of all samples
    annotation_level : AnnotationLevel
        Annotation level to use
    gene_index : Optional[Dict[str, int]]
        Gene index mapping. If None, will be built from samples.
    max_variants : Optional[int]
        Limit number of variants per sample (for debugging)
    impute_value : float
        Value for missing functional scores (default: 0.5)

    Attributes
    ----------
    samples : List[SampleVariants]
        The sample data
    annotation_level : AnnotationLevel
        Annotation level being used
    gene_index : Dict[str, int]
        Gene symbol to index mapping
    num_genes : int
        Total number of unique genes

    Examples
    --------
    >>> from src.data import SampleVariants, VariantRecord
    >>> from torch.utils.data import DataLoader
    >>> var = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense', 1, {})
    >>> sample = SampleVariants('sample1', 1, [var])
    >>> dataset = VariantDataset([sample], AnnotationLevel.L0)
    >>> len(dataset)
    1
    >>> dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_samples)
    >>> for batch in dataloader:
    ...     print(batch['features'].shape)
    ...     break
    torch.Size([1, 1, 1])
    """

    def __init__(
        self,
        samples: List[SampleVariants],
        annotation_level: AnnotationLevel,
        gene_index: Optional[Dict[str, int]] = None,
        max_variants: Optional[int] = None,
        impute_value: float = 0.5,
        chrom_index: Optional[Dict[str, int]] = None,
    ):
        self.samples = samples
        self.annotation_level = annotation_level
        self.max_variants = max_variants
        self.impute_value = impute_value

        # Build gene index if not provided
        if gene_index is None:
            self.gene_index = build_gene_index(samples)
        else:
            self.gene_index = gene_index

        self.num_genes = len(self.gene_index)

        # Build chromosome index if not provided. Always populated so the model
        # can run in chromosome-aware mode by default.
        if chrom_index is None:
            self.chrom_index = build_chrom_index(samples)
        else:
            self.chrom_index = chrom_index

        self.num_chromosomes = len(self.chrom_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample as tensors."""
        sample = self.samples[idx]
        return build_variant_tensor(
            sample,
            self.annotation_level,
            self.gene_index,
            self.max_variants,
            self.impute_value,
            chrom_index=self.chrom_index,
        )

    def get_sample_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the dataset.

        Returns
        -------
        Dict[str, Any]
            Statistics including:
            - 'n_samples': Number of samples
            - 'n_genes': Number of unique genes
            - 'mean_variants_per_sample': Mean variants per sample
            - 'max_variants': Maximum variants in any sample
            - 'min_variants': Minimum variants in any sample
            - 'feature_dim': Feature dimension
        """
        variant_counts = [len(s.variants) for s in self.samples]

        return {
            'n_samples': len(self.samples),
            'n_genes': self.num_genes,
            'mean_variants_per_sample': np.mean(variant_counts),
            'max_variants': np.max(variant_counts) if variant_counts else 0,
            'min_variants': np.min(variant_counts) if variant_counts else 0,
            'feature_dim': get_feature_dimension(self.annotation_level),
            'annotation_level': self.annotation_level.value,
        }

    def get_label_distribution(self) -> Dict[int, int]:
        """
        Get distribution of labels in dataset.

        Returns
        -------
        Dict[int, int]
            Mapping from label to count
        """
        labels = [s.label for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}


def test_sparse_tensor():
    """
    Built-in test function for sparse tensor construction.

    Returns
    -------
    bool
        True if all tests pass
    """
    from src.data import SampleVariants, VariantRecord

    # Create test data
    var1 = VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1,
                        {'sift': 0.05, 'polyphen': 0.9})
    var2 = VariantRecord('1', 200, 'C', 'G', 'GENE2', 'synonymous_variant', 2, {})

    sample1 = SampleVariants('sample1', 1, [var1, var2])
    sample2 = SampleVariants('sample2', 0, [var1])  # Only one variant

    # Test 1: Build gene index
    gene_idx = build_gene_index([sample1, sample2])
    assert len(gene_idx) == 2, "Wrong number of genes"
    assert 'GENE1' in gene_idx, "Missing GENE1"
    assert 'GENE2' in gene_idx, "Missing GENE2"

    # Test 2: Build variant tensor L0
    tensor_l0 = build_variant_tensor(sample1, AnnotationLevel.L0, gene_idx)
    assert tensor_l0['features'].shape == (2, 1), f"Wrong shape L0: {tensor_l0['features'].shape}"
    assert tensor_l0['label'] == 1, "Wrong label"

    # Test 3: Build variant tensor L3
    tensor_l3 = build_variant_tensor(sample1, AnnotationLevel.L3, gene_idx)
    assert tensor_l3['features'].shape == (2, 71), f"Wrong shape L3: {tensor_l3['features'].shape}"

    # Test 4: Collate samples
    batch = collate_samples([tensor_l0, build_variant_tensor(sample2, AnnotationLevel.L0, gene_idx)])
    assert batch['features'].shape == (2, 2, 1), f"Wrong batch shape: {batch['features'].shape}"
    assert batch['mask'].sum() == 3, f"Wrong mask sum: {batch['mask'].sum()}"  # 2 + 1 real variants

    # Test 5: Dataset
    dataset = VariantDataset([sample1, sample2], AnnotationLevel.L0)
    assert len(dataset) == 2, "Wrong dataset length"

    stats = dataset.get_sample_statistics()
    assert stats['n_samples'] == 2, "Wrong n_samples"
    assert stats['n_genes'] == 2, "Wrong n_genes"

    print("✓ All sparse tensor tests passed")
    return True
