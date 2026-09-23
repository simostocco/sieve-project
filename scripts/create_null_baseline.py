#!/usr/bin/env python3
"""
Create null baseline datasets with permuted labels for attribution calibration.

This script creates copies of preprocessed data with randomly shuffled labels,
breaking any real genotype-phenotype relationship. Models trained on this data
establish the null distribution of attributions.

Usage:
    # Single permutation
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output data/preprocessed_NULL.pt \
        --seed 42

    # Multiple permutations for robust null distribution
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output-dir data/null_permutations \
        --n-permutations 5

    # Strict, machine-verifiable single-artifact lineage (Phase 12C3A)
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output data/preprocessed_NULL.pt \
        --seed 42 \
        --strict-lineage

    # Strict mode, reusing a previously validated artifact if its lineage matches
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output data/preprocessed_NULL.pt \
        --seed 42 \
        --strict-lineage \
        --reuse

Author: Francesco Lescai
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so `src` is importable when this script is run
# directly (python scripts/create_null_baseline.py ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import null_lineage  # noqa: E402


def create_single_permutation(
    input_path: str,
    output_path: str,
    seed: int = 42
) -> dict:
    """
    Create a single permuted dataset.

    Parameters
    ----------
    input_path : str
        Path to original preprocessed data
    output_path : str
        Path to save permuted data
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Statistics about the permutation
    """
    print(f"Loading original data from {input_path}...")
    data = torch.load(input_path, weights_only=False)

    # Handle different data structures
    if 'labels' in data:
        labels = data['labels']
    elif 'samples' in data:
        # If samples are stored as list of objects
        labels = torch.tensor([s.label if hasattr(s, 'label') else s['label']
                               for s in data['samples']])
    else:
        raise ValueError("Cannot find labels in preprocessed data. "
                        f"Available keys: {list(data.keys())}")

    n_samples = len(labels)
    n_cases = (labels == 1).sum().item()
    n_controls = (labels == 0).sum().item()

    print(f"Original data: {n_samples} samples ({n_cases} cases, {n_controls} controls)")

    # Report all keys found in the data, everything except 'labels'
    # (or the label field inside 'samples') will be copied verbatim.
    all_keys = sorted(data.keys())
    print(f"Data keys ({len(all_keys)}): {all_keys}")

    # Permute labels using a local RNG to avoid mutating global NumPy state
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(n_samples)

    if isinstance(labels, torch.Tensor):
        permuted_labels = labels[permuted_indices].clone()
    else:
        permuted_labels = [labels[i] for i in permuted_indices]

    # Verify permutation changed positions
    if isinstance(labels, torch.Tensor):
        same_position = (labels == permuted_labels).sum().item()
    else:
        same_position = sum(1 for i, l in enumerate(labels) if l == permuted_labels[i])

    print(f"Labels in same position after permutation: {same_position}/{n_samples} "
          f"({100*same_position/n_samples:.1f}%)")

    # Create permuted dataset
    permuted_data = {}
    for key, value in data.items():
        if key == 'labels':
            permuted_data[key] = permuted_labels
        elif key == 'samples' and isinstance(value, list):
            # Need to update labels within sample objects
            permuted_samples = []
            for i, sample in enumerate(value):
                if hasattr(sample, '_replace'):  # namedtuple
                    permuted_samples.append(sample._replace(label=permuted_labels[i].item()))
                elif isinstance(sample, dict):
                    new_sample = sample.copy()
                    new_sample['label'] = permuted_labels[i].item() if isinstance(permuted_labels[i], torch.Tensor) else permuted_labels[i]
                    permuted_samples.append(new_sample)
                else:
                    # Try to set attribute directly
                    sample_copy = sample  # May need deep copy depending on structure
                    sample_copy.label = permuted_labels[i].item() if isinstance(permuted_labels[i], torch.Tensor) else permuted_labels[i]
                    permuted_samples.append(sample_copy)
            permuted_data[key] = permuted_samples
        else:
            permuted_data[key] = value

    # Add metadata
    permuted_data['_null_baseline_metadata'] = {
        'is_null_baseline': True,
        'permutation_seed': seed,
        'original_path': str(input_path),
        'n_samples': n_samples,
        'n_cases': n_cases,
        'n_controls': n_controls,
        'same_position_count': same_position,
    }

    # Verify all original keys are preserved (only labels should differ)
    preserved_keys = sorted(k for k in permuted_data.keys()
                            if k != '_null_baseline_metadata')
    original_keys = sorted(data.keys())
    if preserved_keys != original_keys:
        missing = set(original_keys) - set(preserved_keys)
        extra = set(preserved_keys) - set(original_keys)
        raise RuntimeError(
            f"Key mismatch after permutation! "
            f"Missing: {missing}, Extra: {extra}"
        )
    print(f"Verified: all {len(original_keys)} original keys preserved")

    # Report non-label fields that were copied verbatim
    for key in original_keys:
        if key in ('labels', 'samples'):
            continue
        value = permuted_data[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape} {value.dtype} (unchanged)")
        elif isinstance(value, list):
            print(f"  {key}: list[{len(value)}] (unchanged)")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} entries (unchanged)")
        else:
            print(f"  {key}: {type(value).__name__} (unchanged)")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving permuted data to {output_path}...")
    torch.save(permuted_data, output_path)

    stats = {
        'n_samples': n_samples,
        'n_cases': n_cases,
        'n_controls': n_controls,
        'same_position': same_position,
        'seed': seed,
    }

    print("Done!")
    return stats


def create_multiple_permutations(
    input_path: str,
    output_dir: str,
    n_permutations: int = 5,
    base_seed: int = 42
) -> list:
    """
    Create multiple permuted datasets for robust null distribution.

    Parameters
    ----------
    input_path : str
        Path to original preprocessed data
    output_dir : str
        Directory to save permuted datasets
    n_permutations : int
        Number of permutations to create
    base_seed : int
        Base random seed (each permutation uses base_seed + i)

    Returns
    -------
    list
        List of statistics dicts for each permutation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {n_permutations} permuted datasets...")
    all_stats = []

    for i in range(n_permutations):
        seed = base_seed + i
        output_path = output_dir / f"preprocessed_NULL_perm{i}.pt"

        print(f"\n--- Permutation {i+1}/{n_permutations} (seed={seed}) ---")
        stats = create_single_permutation(input_path, str(output_path), seed)
        stats['permutation_index'] = i
        stats['output_path'] = str(output_path)
        all_stats.append(stats)

    # Save summary
    summary_path = output_dir / "permutation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Null Baseline Permutation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original data: {input_path}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Base seed: {base_seed}\n\n")
        for stats in all_stats:
            f.write(f"Permutation {stats['permutation_index']}: seed={stats['seed']}, "
                   f"same_position={stats['same_position']}/{stats['n_samples']}\n")

    print(f"\nSummary saved to {summary_path}")
    return all_stats


def _is_sha256ish(value: object, *, length: int) -> bool:
    """Return True if *value* looks like a lowercase hex digest of *length*."""
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(char in '0123456789abcdef' for char in value)


def _read_repository_revision(repo_root: Path) -> str:
    """Read the Git revision from repository files only; never spawn git.

    Mirrors the dry-run-safe revision reader used by
    scripts/position_benchmark_manifest.py so strict null-lineage provenance
    is captured without invoking a subprocess.
    """
    git_dir = repo_root / '.git'
    try:
        head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
        if not head.startswith('ref: '):
            return head if _is_sha256ish(head, length=40) else 'unknown'
        ref = head.removeprefix('ref: ').strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            value = ref_path.read_text(encoding='utf-8').strip()
            return value if _is_sha256ish(value, length=40) else 'unknown'
        packed_refs = git_dir / 'packed-refs'
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding='utf-8').splitlines():
                if line.startswith('#') or not line.strip():
                    continue
                revision, packed_ref = line.split(' ', 1)
                if packed_ref == ref and _is_sha256ish(revision, length=40):
                    return revision
    except OSError:
        return 'unknown'
    return 'unknown'


def create_strict_single_permutation(
    input_path: str,
    output_path: str,
    seed: int,
    *,
    reuse: bool,
    argv: list,
) -> dict:
    """
    Create, or validate-and-reuse, a strict machine-verifiable null baseline.

    Unlike ``create_single_permutation`` (which this function does not call
    and does not modify), strict mode:

    * never mutates the source ``SampleVariants`` objects (uses
      ``dataclasses.replace`` instead of the historical aliasing pattern);
    * records the full permutation vector, not just the seed, as the
      authoritative scientific transformation;
    * embeds a strict ``_null_baseline_metadata`` schema plus a
      ``<output>.null-lineage.yaml`` sidecar recording source/null byte
      identity, sample-order identity, label identity, permutation identity,
      and the semantic ``lineage_sha256``;
    * fails closed on any ambiguous existing-output state and never silently
      overwrites or reuses an artifact (see Phase 12C3A design notes in
      documentation/appendices/position-encoding-implementation-log.md).

    Parameters
    ----------
    input_path : str
        Path to the real preprocessed source ``.pt`` artifact.
    output_path : str
        Path to the strict null ``.pt`` artifact to create (or reuse).
    seed : int
        Seed used to draw a fresh permutation vector via
        ``numpy.random.default_rng(seed).permutation(n_samples)`` (the same
        RNG scheme as the historical generator). Reproduction provenance
        only; the resulting permutation vector is the scientific authority.
    reuse : bool
        If an existing strict artifact/sidecar pair is present, validate it
        and reuse it only if its lineage matches the requested one exactly.
    argv : list
        The exact CLI invocation (``sys.argv``), recorded as generator
        provenance.

    Returns
    -------
    dict
        Validation/identity report, plus ``reused``, ``output_path``, and
        ``sidecar_path``.
    """
    source_path = Path(input_path).resolve(strict=True)
    null_path = Path(output_path).resolve(strict=False)

    if null_path == source_path:
        raise ValueError(f"strict null output path must not equal the source path: {null_path}")

    sidecar_path = null_lineage.sidecar_path_for(null_path)
    output_exists = null_path.exists()
    sidecar_exists = sidecar_path.exists()

    # Fail-closed existing-output policy: never silently overwrite or reuse.
    if output_exists and not sidecar_exists:
        raise ValueError(
            f"strict null output already exists without a lineage sidecar: {null_path}. "
            "Refusing to overwrite a non-strict or foreign artifact."
        )
    if sidecar_exists and not output_exists:
        raise ValueError(
            f"strict lineage sidecar exists without its null output: {sidecar_path}. "
            "Refusing to proceed with an incomplete pair."
        )
    if output_exists and sidecar_exists and not reuse:
        raise ValueError(
            f"strict null output and sidecar already exist: {null_path}. "
            "Pass --reuse to validate and reuse an existing artifact; strict mode never "
            "silently overwrites an existing artifact."
        )

    print(f"Loading source data from {source_path}...")
    source_data = torch.load(source_path, weights_only=False)

    # Refuse to permute a source that is itself already a null baseline: the
    # scientific lineage this phase records is defined relative to a real
    # preprocessed cohort, not another null derivative.
    source_baseline_metadata = source_data.get('_null_baseline_metadata')
    if isinstance(source_baseline_metadata, dict) and source_baseline_metadata.get(
        'is_null_baseline'
    ) is True:
        raise ValueError(
            f"source artifact {source_path} is itself already a null baseline "
            "(_null_baseline_metadata.is_null_baseline is True); refusing to permute "
            "a null dataset"
        )

    source_samples = null_lineage.extract_samples(source_data, label='source')
    source_artifact_sha256 = null_lineage.compute_file_sha256(source_path)

    # The full permutation vector, not the seed, is the scientific authority
    # (see documentation/appendices/position-encoding-implementation-log.md,
    # Phase 12C3A). The seed only drives reproducible generation here.
    rng = np.random.default_rng(seed)
    requested_indices = rng.permutation(len(source_samples))
    lineage = null_lineage.build_null_lineage(
        samples=source_samples,
        source_artifact_sha256=source_artifact_sha256,
        permutation_indices=requested_indices,
    )

    if output_exists and sidecar_exists and reuse:
        print(f"Existing strict null artifact found at {null_path}; validating for reuse...")
        report = null_lineage.validate_null_pair(source_path, null_path, sidecar_path)
        if report['lineage_sha256'] != lineage['lineage_sha256']:
            raise ValueError(
                "existing strict null artifact has a different lineage_sha256 than requested "
                f"(existing={report['lineage_sha256']}, requested={lineage['lineage_sha256']}); "
                "refusing to reuse a mismatched artifact"
            )
        print(
            f"Reusing valid existing strict null artifact "
            f"(lineage_sha256={report['lineage_sha256']})."
        )
        return {
            'reused': True,
            'output_path': str(null_path),
            'sidecar_path': str(sidecar_path),
            **report,
        }

    n_samples = lineage['n_samples']
    print(
        f"Source data: {n_samples} samples "
        f"({lineage['n_cases']} cases, {lineage['n_controls']} controls)"
    )
    print(
        f"Labels in same position after permutation: "
        f"{lineage['same_position_count']}/{n_samples} "
        f"({100 * lineage['same_position_count'] / n_samples:.1f}%)"
    )

    repo_root = Path(__file__).resolve().parent.parent
    repository_revision = _read_repository_revision(repo_root)
    generator_script = 'scripts/create_null_baseline.py'

    embedded_metadata = null_lineage.build_embedded_metadata(
        lineage=lineage,
        source_artifact_path=str(source_path),
        original_path=str(input_path),
        permutation_seed=seed,
        generator_script=generator_script,
        repository_revision=repository_revision,
        argv=list(argv),
    )

    null_data = dict(source_data)
    null_data['samples'] = lineage['null_samples']
    null_data['_null_baseline_metadata'] = embedded_metadata

    null_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_null_path = null_path.with_name(f'.{null_path.name}.tmp-{os.getpid()}')
    tmp_sidecar_path = sidecar_path.with_name(f'.{sidecar_path.name}.tmp-{os.getpid()}')

    created_final_null = False
    try:
        print(f"Saving strict null artifact to {null_path}...")
        torch.save(null_data, tmp_null_path)
        os.replace(tmp_null_path, null_path)
        created_final_null = True

        null_artifact_sha256 = null_lineage.compute_file_sha256(null_path)
        sidecar_payload = null_lineage.build_sidecar_payload(
            lineage=lineage,
            source_path=str(source_path),
            source_sha256=source_artifact_sha256,
            null_path=str(null_path),
            null_sha256=null_artifact_sha256,
            permutation_seed=seed,
            generator_script=generator_script,
            repository_revision=repository_revision,
            argv=list(argv),
        )
        null_lineage.write_sidecar(tmp_sidecar_path, sidecar_payload)
        os.replace(tmp_sidecar_path, sidecar_path)
    except BaseException as exc:
        for tmp_path in (tmp_null_path, tmp_sidecar_path):
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        if created_final_null and not sidecar_path.exists():
            try:
                null_path.unlink()
            except OSError as cleanup_exc:
                raise RuntimeError(
                    f"strict null artifact {null_path} was created by this invocation but its "
                    "sidecar failed to publish, and best-effort rollback of the artifact also "
                    f"failed ({cleanup_exc}); the artifact is orphaned and must be removed "
                    f"manually. Original error: {exc}"
                ) from exc
        raise

    print("Validating freshly written strict null artifact...")
    try:
        report = null_lineage.validate_null_pair(source_path, null_path, sidecar_path)
    except BaseException as exc:
        # Both files were just published by this invocation; a failed
        # self-check must not leave an artifact/sidecar pair that claims to
        # be strict-validated on disk.
        for path in (null_path, sidecar_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        raise RuntimeError(
            f"freshly written strict null artifact {null_path} failed its own "
            f"self-check validation; both it and its sidecar have been removed "
            f"on a best-effort basis. Original error: {exc}"
        ) from exc
    print(f"Strict null artifact validated. lineage_sha256={report['lineage_sha256']}")

    return {
        'reused': False,
        'output_path': str(null_path),
        'sidecar_path': str(sidecar_path),
        **report,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create null baseline datasets with permuted labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Path to original preprocessed data (.pt file)')

    # Single permutation mode
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single permuted dataset')

    # Multiple permutation mode
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for multiple permuted datasets')
    parser.add_argument('--n-permutations', type=int, default=5,
                       help='Number of permutations (only with --output-dir)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (or base seed for multiple permutations)')

    # Strict single-artifact lineage mode (Phase 12C3A). Additive and opt-in:
    # historical invocations without these flags are entirely unaffected.
    parser.add_argument('--strict-lineage', action='store_true',
                       help='Create a machine-verifiable null-dataset lineage sidecar '
                            '(single-artifact --output mode only)')
    parser.add_argument('--reuse', action='store_true',
                       help='With --strict-lineage: validate and reuse an existing strict '
                            'artifact only if its lineage matches the requested one exactly')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output and args.output_dir:
        raise ValueError("Specify either --output (single) or --output-dir (multiple), not both")

    if args.reuse and not args.strict_lineage:
        raise ValueError("--reuse requires --strict-lineage")

    if args.strict_lineage and args.output_dir:
        raise ValueError(
            "--strict-lineage does not support --output-dir / multi-permutation mode "
            "(Phase 12C3A supports single-artifact strict lineage only)"
        )

    if not args.output and not args.output_dir:
        # Default to single permutation with auto-named output
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_NULL{input_path.suffix}")

    if args.strict_lineage:
        create_strict_single_permutation(
            args.input, args.output, args.seed, reuse=args.reuse, argv=list(sys.argv)
        )
    elif args.output:
        create_single_permutation(args.input, args.output, args.seed)
    else:
        create_multiple_permutations(args.input, args.output_dir,
                                    args.n_permutations, args.seed)


if __name__ == '__main__':
    main()
