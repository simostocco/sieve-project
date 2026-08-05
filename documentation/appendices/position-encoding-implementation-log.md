# Position-Encoding Implementation Log

## Purpose

This document records the chronological implementation history for the
position-encoding benchmark work. Each entry should explain:

- what changed;
- why it changed;
- runtime and compatibility effects;
- validation performed;
- known limitations;
- future work.

The position-encoding documentation has three separate roles:

- `position-encoding-audit.md` is the historical description of the original
  repository behavior at the audited baseline.
- `position-encoding-cli-contract.md` is the intended public interface and
  architectural contract for selectable position encodings.
- `position-encoding-implementation-log.md` is this chronological record of
  implementation decisions and results.

The audit should not be rewritten merely because implementation progressed. The
contract should change only when the intended public interface or architecture
contract changes. This log records what actually happened in each implementation
phase.

## Repository baseline

- Upstream repository: `lescailab/sieve-project`
- Working fork: `simostocco/sieve-project`
- Branch: `simostocco/position-encoding-benchmark`
- Starting commit: `434fc095ed0de4c2f32c2c9b3f024a0310b8f611`
- Starting release: `v1.3.0`

## Project objective

The project objective is to prepare SIEVE for controlled comparison of multiple
position-encoding strategies:

- fixed sinusoidal encoding;
- learned binned absolute encoding;
- T5-style relative bias;
- RoPE;
- fixed and learned ALiBi;
- later genomic-aware positional encodings.

The evaluation goals are:

- predictive performance;
- ranking stability;
- attribution quality;
- compatibility and legacy equivalence.

## Development principles

- Preserve original SIEVE behavior unless an approved phase changes it.
- Inspect relevant implementation, tests, config writers, config readers, and
  checkpoint paths before editing.
- Keep commits narrow and reviewable.
- Separate requested, resolved, and executed configuration.
- Prove legacy equivalence before enabling new strategies.
- Add focused tests for each compatibility or serialization behavior.
- Document compatibility behavior directly where it matters.
- Report actual validation honestly; do not claim skipped, blocked, or
  unexecuted checks passed.

## Phase 2 - Existing Architecture Audit

Commit: `52d6f39`

Goal:

Backfill a read-only audit of the original positional architecture before
introducing implementation changes.

Files changed:

- `AGENTS.md`
- `documentation/appendices/position-encoding-audit.md`

Findings:

- Absolute sinusoidal features were generated during sparse tensor construction
  for L1-L4 and concatenated into `variant_features`. L0 retained only dosage as
  the input feature.
- The current model also used learned T5-style relative-position bucket bias in
  attention. This relative mechanism was active even at L0.
- Chromosome-aware relative routing depended on `chrom_ids`. When present,
  cross-chromosome pairs used a dedicated bucket instead of coordinate
  subtraction.
- Learned chromosome embedding was path-dependent. CV training constructed the
  model with `num_chromosomes=dataset.num_chromosomes`; single-split training
  omitted `num_chromosomes`, so no learned chromosome embedding was constructed
  in that path.
- Gene-slot identity remained architecturally important because variants were
  aggregated into fixed gene slots before classification.
- Integrated Gradients differentiated `variant_features` while positions,
  gene IDs, masks, covariates, and chromosome IDs were fixed auxiliary inputs.

Reasoning:

The audit made clear that position was not one isolated feature. It was spread
across input features, attention bias, chromosome-aware routing, optional
chromosome embeddings, and gene aggregation. That meant later implementation
phases needed to separate content from position carefully rather than simply
toggle one code path.

Runtime behavior changed: no.

Validation:

This phase was documentation-only. No runtime validation was required by the
commit itself.

Known limitations:

- The audit described baseline behavior but did not implement selectable
  position encodings.
- It identified the CV versus single-split chromosome discrepancy but did not
  resolve it.
- It did not define the final CLI or serialized configuration contract.

## Phase 3 - Position-Encoding CLI Contract

Commit: `ba40d58`

Goal:

Define the public CLI and serialized configuration contract for selectable
position encodings before implementation began.

Files changed:

- `documentation/appendices/position-encoding-cli-contract.md`

Contract decisions:

- Absolute-position configuration, relative-attention configuration, chromosome
  embedding, and cross-chromosome policy were separated.
- The contract introduced `legacy` and `custom` presets.
- Planned strategies were `none`, `sinusoidal`, and `learned_binned` for
  absolute position; `none`, `t5_bucket`, `rope`, `alibi_fixed`, and
  `alibi_learned` for relative position; and `none` or `learned` for chromosome
  encoding.
- Learned binned absolute position was specified as an in-model trainable
  strategy rather than a preprocessing-only input feature.
- Chromosome IDs remained zero-based to match existing dataset and attention
  behavior.

Reasoning:

The contract was written before implementation so CLI names, validation rules,
old-checkpoint compatibility expectations, and cross-chromosome semantics could
be reviewed independently from code changes. Separating the strategy axes avoids
making chromosome identity, absolute position, and pairwise relative mechanisms
accidental side effects of one flag.

Runtime behavior changed: no.

Known limitations:

- The contract described intended behavior, not implemented behavior.
- It did not make the resolved configuration control model construction yet.
- It deferred old-checkpoint compatibility and explanation-time behavior to
  later phases.

## Phase 4A - Pure Configuration Resolver

Commit: `1d3fcc6`

Goal:

Add pure position-encoding configuration definitions and a resolver without
changing preprocessing, model construction, attention, training, explanation, or
checkpoint behavior.

Exact files changed:

- `src/encoding/__init__.py`
- `src/encoding/levels.py`
- `src/encoding/position_config.py`
- `tests/test_position_encoding_config.py`

Implementation:

- Added immutable enums and dataclasses for unresolved requests and resolved
  nested configuration.
- Added content-only dimensions for L0-L4 while preserving existing
  `get_feature_dimension()` behavior.
- Added resolver validation for presets, strategy combinations, inactive
  method-specific arguments, positive integers, finite positive numeric scales,
  even sinusoidal dimensions, T5 bucket constraints, RoPE head-dimension
  constraints, and chromosome-count requirements.
- Treated `None` as "use default" while allowing explicit zero values to reach
  validation and fail.
- Added primitive `to_dict()` serialization for resolved configuration.
- Set the resolved attribution default to content-only mode.

Reasoning:

The resolver was kept pure so the repository could validate and serialize the
intended position-encoding choices before any runtime model path depended on
them. This made it possible to test custom configurations while continuing to
execute the original model.

Runtime behavior changed: no.

Backward compatibility:

- Existing feature dimensions and preprocessing remained unchanged.
- Legacy resolution produced dimensions matching historical input widths.
- The resolver accepted enum instances only; CLI string conversion was deferred.

Validation:

- 101 tests passed.

Known limitations:

- The resolved configuration did not control model construction.
- RoPE, ALiBi, learned binned absolute position, and model execution changes
  were not implemented.
- CLI integration and checkpoint compatibility were deferred.

## Phase 4B - Training CLI Plumbing

Commit: `7314d6f`

Goal:

Add training CLI parsing and resolver plumbing without changing model
computation.

Exact files changed:

- `scripts/train.py`
- `tests/test_train_position_cli.py`

Implementation:

- Added parser options for the position preset, absolute-position encoding,
  relative-position encoding, chromosome encoding, cross-chromosome policy, and
  method-specific numeric parameters.
- Added `build_arg_parser()`, `parse_args()`,
  `build_position_encoding_request()`, and
  `prepare_training_position_encoding()`.
- Converted CLI strings to enum instances in the training helper layer.
- Kept the resolver as the validation authority instead of duplicating resolver
  validation inside `train.py`.
- Verified that legacy resolved input width matched
  `get_feature_dimension(annotation_level)`.
- Kept historical `input_dim` as the execution authority for model
  construction.
- Allowed valid custom requests to resolve but blocked custom execution with
  `NotImplementedError`.
- Allowed invalid custom configurations to fail through resolver validation
  before the deferred-execution guard.

Reasoning:

This phase made the CLI surface testable and made future train-time integration
possible, but it deliberately avoided silently training a model whose runtime
path ignored custom positional choices. The explicit `NotImplementedError`
protected users from believing custom strategies were already active.

Runtime computation remains legacy.

Validation:

- 74 related tests passed.

Known limitations:

- Resolved configuration was not serialized as normalized metadata yet.
- Custom strategies remained non-executable.
- Model constructors, preprocessing, attention, checkpoints, and explanation
  reconstruction were unchanged.

## Phase 4C - Reproducibility and Checkpoint Metadata

Commit: `eb13c4f1781b3bb806ca308eaf611ecd47d8a1b2`

Goal:

Persist normalized position-encoding metadata, dataset mapping identity, and
actual legacy execution metadata without changing model computation.

Exact files changed:

- `scripts/train.py`
- `src/training/trainer.py`
- `tests/test_checkpoint_metadata.py`
- `tests/test_fold_config_saving.py`
- `tests/test_train_config_metadata.py`

Implementation:

- Added deterministic mapping validation for dataset index mappings. Valid
  mappings must be mappings from string names to non-negative, unique,
  contiguous integer IDs; boolean, float, string, negative, duplicate, and
  gapped IDs are rejected.
- Added SHA-256 fingerprints for gene and chromosome mappings using stable,
  compact UTF-8 JSON canonicalization.
- Added one experiment-level `dataset_mappings.json` sidecar containing the full
  `gene_index`, full `chrom_index`, chromosome ID-to-name mapping, and mapping
  fingerprints.
- Stored the complete chromosome mapping in model-row orientation inside the
  serialized copy of the resolved position configuration.
- Stored the full gene mapping only once, in the sidecar, instead of duplicating
  it in parent configs, fold configs, or checkpoints.
- Added lightweight parent config, fold config, and checkpoint metadata:
  `metadata_schema_version`, `input_dim`, `content_dim`, `num_genes`,
  `num_chromosomes`, `position_encoding`, `dataset_identity`, and
  `position_encoding_execution`.
- Preserved separate requested, resolved, and executed configuration:
  requested configuration remains the raw CLI fields; resolved configuration is
  `resolved_position_encoding.to_dict()` plus serialized chromosome mapping;
  executed configuration records the actual current legacy path.
- Recorded actual CV versus single-split chromosome behavior. CV records
  `model_num_chromosomes=dataset.num_chromosomes`, learned chromosome embedding
  executed, and chromosome-aware relative routing active. Single-split records
  `model_num_chromosomes=0`, learned chromosome embedding not executed, and
  chromosome-aware relative routing active.
- Added optional checkpoint metadata to `Trainer`; when not supplied, historical
  checkpoint keys are preserved.
- Added defensive deep copying for fold and checkpoint metadata so caller
  mutation cannot alter saved metadata.
- Kept model state-dict shapes and computation unchanged.

Runtime behavior:

- No model, preprocessing, attention, optimization, or prediction computation
  changed.
- Checkpoint files receive an additional `metadata` key only when metadata is
  supplied.

Compatibility:

- Old `Trainer` calls remain valid.
- Old `save_fold_config()` calls remain valid.
- Old checkpoints remain loadable.
- `scripts/explain.py` and validation scripts remain unchanged.

Validation:

- 115 focused tests passed.
- 585 full-suite tests passed.
- 6 non-failing existing warnings were observed.
- `compileall` passed.
- `git diff --check` passed.
- New tests passed Ruff.
- New tests passed Black check.
- New tests passed isort check.
- Modified legacy files did not introduce additional Ruff/isort debt.
- `Trainer` Ruff findings remained 14 in both baseline and current code.

Known limitations:

- Normalized configuration still does not control model execution.
- `scripts/explain.py` does not consume checkpoint metadata.
- Source genomic files are not hashed.
- Metadata schema is transitional.
- Content and positional channels are not yet separated.

## Next planned phase

The next objective is to separate content features from positional features.
That phase should first prove exact legacy equivalence, including feature
shapes, logits, attention behavior, checkpoint compatibility, and attribution
setup. New strategies should not be enabled until equivalence tests pass.
