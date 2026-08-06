"""
Chunked variant dataset for processing all variants with bounded memory.

This module implements chunked processing to ensure the model sees all variants
across the entire genome, not just the first N variants (which would bias toward
chr1/chr2 due to VCF ordering).

Key innovation: Split each sample into multiple chunks, process chunks independently,
then aggregate chunk-level outputs into sample-level predictions.

Author: Francesco Lescai
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import SampleVariants
from src.encoding.levels import AnnotationLevel
from src.encoding.sparse_tensor import (
    _validate_split_feature_schema,
    build_chrom_index,
    build_gene_index,
    build_variant_tensor,
)
from src.data.covariates import encode_sex_for_covariate


def _encode_sex(sex: 'Optional[str]') -> float:
    """Encode sex as a float for use as a model covariate.

    Parameters
    ----------
    sex : str or None
        'M', 'F', or None.

    Returns
    -------
    float
        0.0 for female, 1.0 for male, -1.0 for unknown/missing.
    """
    return encode_sex_for_covariate(sex)


class ChunkedVariantDataset(Dataset):
    """
    Dataset that splits each sample into chunks for memory-efficient processing.

    Instead of truncating samples to max_variants (which loses chr3-22),
    this dataset yields multiple chunks per sample. Each chunk contains
    up to chunk_size variants from different parts of the genome.

    The model processes each chunk independently, then aggregates chunk
    embeddings/logits to produce a final sample-level prediction.

    Parameters
    ----------
    samples : List[SampleVariants]
        All samples
    annotation_level : AnnotationLevel
        Annotation level to use
    chunk_size : int
        Maximum variants per chunk (default: 3000, safe for memory)
    overlap : int
        Number of overlapping variants between adjacent chunks (default: 0)
    gene_index : Optional[Dict[str, int]]
        Gene index. If None, built from samples.
    impute_value : float
        Imputation value for missing scores

    Attributes
    ----------
    chunk_info : List[Dict]
        Metadata for each chunk:
        - sample_idx: index into samples
        - chunk_idx: which chunk within the sample
        - start_idx: start variant index in original sample
        - end_idx: end variant index in original sample
        - total_chunks: total chunks for this sample

    Examples
    --------
    >>> dataset = ChunkedVariantDataset(samples, AnnotationLevel.L3, chunk_size=3000)
    >>> print(f"Original samples: {len(samples)}")
    >>> print(f"Total chunks: {len(dataset)}")
    >>> # Train on chunks, aggregate during forward pass
    """

    def __init__(
        self,
        samples: List[SampleVariants],
        annotation_level: AnnotationLevel,
        chunk_size: int = 3000,
        overlap: int = 0,
        gene_index: Optional[Dict[str, int]] = None,
        impute_value: float = 0.5,
        chrom_index: Optional[Dict[str, int]] = None,
    ):
        self.samples = samples
        self.annotation_level = annotation_level
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        covariate_lengths = set()

        # Build chunk metadata
        self.chunk_info = []
        for sample_idx, sample in enumerate(samples):
            n_variants = len(sample.variants)

            # Encode sex as float for covariate use
            sex_code = _encode_sex(sample.sex)
            sample_covariates = getattr(sample, 'covariates', None)
            if sample_covariates is not None:
                sample_covariates = np.asarray(sample_covariates, dtype=np.float32)
                if sample_covariates.ndim != 1:
                    raise ValueError(
                        f"Sample {sample.sample_id!r} covariates must be 1D; "
                        f"got shape {sample_covariates.shape}"
                    )
                covariate_lengths.add(int(sample_covariates.shape[0]))

            if n_variants == 0:
                # Empty sample - create one empty chunk
                self.chunk_info.append({
                    'sample_idx': sample_idx,
                    'chunk_idx': 0,
                    'start_idx': 0,
                    'end_idx': 0,
                    'total_chunks': 1,
                    'sample_id': sample.sample_id,
                    'label': sample.label,
                    'sex': sex_code,
                    'covariates': sample_covariates,
                })
            else:
                # Calculate chunks with overlap
                stride = chunk_size - overlap
                total_chunks = max(1, int(np.ceil(n_variants / stride)))

                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * stride
                    end_idx = min(start_idx + chunk_size, n_variants)

                    self.chunk_info.append({
                        'sample_idx': sample_idx,
                        'chunk_idx': chunk_idx,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'total_chunks': total_chunks,
                        'sample_id': sample.sample_id,
                        'label': sample.label,
                        'sex': sex_code,
                        'covariates': sample_covariates,
                    })

        if len(covariate_lengths) > 1:
            raise ValueError(
                f"Inconsistent covariate lengths across samples: {sorted(covariate_lengths)}"
            )
        self.num_covariates = next(iter(covariate_lengths), 0)

        print(f"ChunkedVariantDataset created:")
        print(f"  Samples: {len(samples)}")
        print(f"  Total chunks: {len(self.chunk_info)}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Overlap: {overlap}")
        print(f"  Avg chunks per sample: {len(self.chunk_info) / len(samples):.1f}")

    def __len__(self) -> int:
        return len(self.chunk_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single chunk as tensors."""
        info = self.chunk_info[idx]
        sample = self.samples[info['sample_idx']]

        # Extract chunk of variants
        chunk_variants = sample.variants[info['start_idx']:info['end_idx']]

        # Create temporary SampleVariants for this chunk
        chunk_sample = SampleVariants(
            sample_id=sample.sample_id,
            label=sample.label,
            variants=chunk_variants
        )

        # Build tensor for this chunk
        chunk_tensor = build_variant_tensor(
            chunk_sample,
            self.annotation_level,
            self.gene_index,
            max_variants=None,  # Don't truncate - chunk is already sized
            impute_value=self.impute_value,
            chrom_index=self.chrom_index,
        )

        # Add chunk metadata
        chunk_tensor['chunk_idx'] = info['chunk_idx']
        chunk_tensor['total_chunks'] = info['total_chunks']
        chunk_tensor['original_sample_idx'] = info['sample_idx']
        chunk_tensor['sex'] = info['sex']
        if info['covariates'] is not None:
            chunk_tensor['covariates'] = torch.as_tensor(
                info['covariates'], dtype=torch.float32
            )

        return chunk_tensor

    def get_chunks_for_sample(self, sample_idx: int) -> List[int]:
        """Get list of chunk indices for a given sample."""
        return [i for i, info in enumerate(self.chunk_info)
                if info['sample_idx'] == sample_idx]


def collate_chunks(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate multiple chunks into a padded batch.

    This is similar to collate_samples but simpler because chunks
    are already sized appropriately.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of chunk tensors from ChunkedVariantDataset

    Returns
    -------
    Dict[str, Any]
        Batched tensors with chunk metadata. Split-aware batches include
        ``content_features`` and ``absolute_position_features`` only when every
        input item contains both split feature keys. Legacy manually constructed
        chunk dictionaries that omit both keys retain the historical output
        schema.
    """
    batch_size = len(batch)
    has_covariates = any('covariates' in sample for sample in batch)
    has_chrom_ids = any('chrom_ids' in sample for sample in batch)
    if has_chrom_ids and not all('chrom_ids' in sample for sample in batch):
        raise ValueError(
            "Mixed chunk batches with and without 'chrom_ids' are not supported."
        )
    has_split_features = _validate_split_feature_schema(batch)

    # Get max variants in this batch of chunks
    max_variants = max(sample['features'].shape[0] for sample in batch)

    if max_variants == 0:
        feature_dim = batch[0]['features'].shape[1] if len(batch[0]['features'].shape) > 1 else 0
        collated = {
            'features': torch.zeros((batch_size, 0, feature_dim), dtype=torch.float32),
            'positions': torch.zeros((batch_size, 0), dtype=torch.long),
            'gene_ids': torch.zeros((batch_size, 0), dtype=torch.long),
            'mask': torch.zeros((batch_size, 0), dtype=torch.bool),
            'labels': torch.tensor([s['label'] for s in batch], dtype=torch.long),
            'sex': torch.tensor([s['sex'] for s in batch], dtype=torch.float32),
            'sample_ids': [s['sample_id'] for s in batch],
            'chunk_indices': torch.tensor([s['chunk_idx'] for s in batch], dtype=torch.long),
            'total_chunks': torch.tensor([s['total_chunks'] for s in batch], dtype=torch.long),
            'original_sample_indices': torch.tensor([s['original_sample_idx'] for s in batch], dtype=torch.long),
        }
        if has_covariates:
            first_with_cov = next(s for s in batch if 'covariates' in s)
            cov_dim = first_with_cov['covariates'].shape[0]
            collated['covariates'] = torch.zeros((batch_size, cov_dim), dtype=torch.float32)
        if has_chrom_ids:
            collated['chrom_ids'] = torch.zeros((batch_size, 0), dtype=torch.long)
        if has_split_features:
            content_dim = batch[0]['content_features'].shape[1]
            absolute_position_dim = batch[0]['absolute_position_features'].shape[1]
            collated['content_features'] = torch.zeros(
                (batch_size, 0, content_dim),
                dtype=torch.float32,
            )
            collated['absolute_position_features'] = torch.zeros(
                (batch_size, 0, absolute_position_dim),
                dtype=torch.float32,
            )
        return collated

    feature_dim = batch[0]['features'].shape[1]
    content_dim = batch[0]['content_features'].shape[1] if has_split_features else 0
    absolute_position_dim = (
        batch[0]['absolute_position_features'].shape[1]
        if has_split_features else 0
    )

    # Initialize padded tensors
    features_padded = torch.zeros((batch_size, max_variants, feature_dim), dtype=torch.float32)
    # Split-aware chunks keep the same zero-padding contract as historical
    # ``features``: real rows are copied, padding rows stay all-zero and masked.
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
    positions_padded = torch.zeros((batch_size, max_variants), dtype=torch.long)
    gene_ids_padded = torch.zeros((batch_size, max_variants), dtype=torch.long)
    mask_padded = torch.zeros((batch_size, max_variants), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    sex = torch.zeros(batch_size, dtype=torch.float32)
    sample_ids = []
    chunk_indices = torch.zeros(batch_size, dtype=torch.long)
    total_chunks = torch.zeros(batch_size, dtype=torch.long)
    original_sample_indices = torch.zeros(batch_size, dtype=torch.long)
    covariates = None
    if has_covariates:
        first_with_cov = next(s for s in batch if 'covariates' in s)
        cov_dim = first_with_cov['covariates'].shape[0]
        covariates = torch.zeros((batch_size, cov_dim), dtype=torch.float32)
    chrom_ids_padded = (
        torch.zeros((batch_size, max_variants), dtype=torch.long)
        if has_chrom_ids else None
    )

    # Fill in the data
    for i, sample in enumerate(batch):
        n_variants = sample['features'].shape[0]

        if n_variants > 0:
            features_padded[i, :n_variants] = sample['features']
            if has_split_features:
                content_features_padded[i, :n_variants] = sample['content_features']
                absolute_position_features_padded[i, :n_variants] = (
                    sample['absolute_position_features']
                )
            positions_padded[i, :n_variants] = sample['positions']
            gene_ids_padded[i, :n_variants] = sample['gene_ids']
            mask_padded[i, :n_variants] = sample['mask']
            if has_chrom_ids:
                chrom_ids_padded[i, :n_variants] = sample['chrom_ids']

        labels[i] = sample['label']
        sex[i] = sample['sex']
        sample_ids.append(sample['sample_id'])
        chunk_indices[i] = sample['chunk_idx']
        total_chunks[i] = sample['total_chunks']
        original_sample_indices[i] = sample['original_sample_idx']
        if has_covariates:
            if 'covariates' not in sample:
                raise ValueError(
                    "Mixed batches with and without covariates are not supported."
                )
            covariates[i] = sample['covariates']

    collated = {
        'features': features_padded,
        'positions': positions_padded,
        'gene_ids': gene_ids_padded,
        'mask': mask_padded,
        'labels': labels,
        'sex': sex,
        'sample_ids': sample_ids,
        'chunk_indices': chunk_indices,
        'total_chunks': total_chunks,
        'original_sample_indices': original_sample_indices,
    }
    if covariates is not None:
        collated['covariates'] = covariates
    if has_chrom_ids:
        collated['chrom_ids'] = chrom_ids_padded
    if has_split_features:
        collated['content_features'] = content_features_padded
        collated['absolute_position_features'] = absolute_position_features_padded
    return collated
