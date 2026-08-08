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

## Phase 5B1 - Additive Content and Absolute-Position Data Representation

Goal:

Expose content features and historical absolute-position features as explicit
dataset tensors while preserving the existing `features` tensor as the runtime
authority.

Exact files changed:

- `src/encoding/__init__.py`
- `src/encoding/chunked_dataset.py`
- `src/encoding/levels.py`
- `src/encoding/sparse_tensor.py`
- `tests/test_content_position_split.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Authoritative historical ordering:

- L0: `[dosage]`
- L1: `[dosage, sinusoidal_64]`
- L2: `[dosage, sinusoidal_64, consequence_4]`
- L3: `[dosage, sinusoidal_64, consequence_4, SIFT, PolyPhen]`
- L4: current L3-compatible representation

Implementation:

- Added pure NumPy helpers to split and recompose historical variant feature
  matrices.
- Derived `content_features` and `absolute_position_features` from the already
  encoded historical `features` matrix instead of independently re-encoding
  biological annotations. This avoids drifting from the exact dosage,
  consequence, SIFT, PolyPhen, imputation, dtype, and ordering semantics used by
  old checkpoints.
- Added a uniform L0 zero-width absolute-position representation with shape
  `[num_variants, 0]`.
- Extended `build_variant_tensor()` so dataset-generated samples return
  `features`, `content_features`, and `absolute_position_features`.
- Extended `collate_samples()` and `collate_chunks()` to pad the split tensors
  only when every input item carries the split pair.
- Preserved compatibility with old manually constructed dictionaries that omit
  both split keys.
- Rejected mixed legacy/split-aware batches, partial split pairs, and
  malformed split tensors, including wrong ranks, row-count mismatches, and
  inconsistent split widths.
- Exported the split, compose, and legacy absolute-position dimension helpers as
  public encoding APIs.

Runtime behavior:

- Historical `features` remains the tensor consumed by training, explanation,
  validation, and model forward paths.
- Model construction, attention, chromosome handling, Integrated Gradients,
  checkpoint tensor shapes, CLI behavior, chunk boundaries, gene IDs,
  chromosome IDs, masks, labels, covariates, and sample IDs are unchanged.
- No positional strategy beyond the existing legacy sinusoidal input features is
  executable yet.

Compatibility:

- Existing code that reads `batch["features"]` continues to receive the same
  values and shapes.
- Legacy collator inputs without split keys retain the old output schema.
- Split-aware collator inputs receive padded split tensors with zero-filled
  padding rows, matching the historical `features` padding contract.

Validation:

- `tests/test_content_position_split.py`: 49 passed.
- The focused positional/data/model metadata test command passed 183 tests.
- Additional relevant dataset, chunking, covariate, explanation, and validation
  tests passed 94 tests with 1 existing non-failing warning.
- The full test suite passed 634 tests with 6 existing non-failing warnings.
- `compileall` passed for the changed encoding and new test files.
- `git diff --check` passed.
- The new test file passed Ruff, Black check, and isort check.
- Modified legacy encoding files retained the same Ruff finding count as the
  committed baseline comparison: 53 current findings versus 53 baseline
  findings.
- Modified legacy encoding files retained the same isort finding set as the
  committed baseline comparison.

Baseline static debt:

- The modified legacy encoding files still carry pre-existing Ruff and isort
  debt, including import-order, old typing-style, and unrelated unused-import
  findings.
- Black check reported that each modified legacy encoding file would be
  reformatted. A baseline Black comparison process for the same files hung in
  this environment and was interrupted, so Black equivalence for legacy files
  could not be conclusively compared.

Known limitations:

- The model does not consume `content_features` or
  `absolute_position_features`.
- The split representation is legacy-only and recomposes to the historical
  `VariantEncoder` input ordering.
- The additive migration temporarily carries historical `features` plus the two
  split tensors, increasing CPU and potentially pinned-memory use until
  model-side composition removes the duplicate representation.
- Learned binned absolute position, model-side composition, RoPE, ALiBi, and
  content-only Integrated Gradients remain deferred.

## Phase 5B2 - Model-Side Legacy Feature Composition

Goal:

Execute the legacy model path through a parameter-free Torch composition of
`content_features` and `absolute_position_features` while preserving historical
`variant_features` as a compatibility fallback.

Exact files changed:

- `src/models/feature_composition.py`
- `src/models/sieve.py`
- `src/models/chunked_sieve.py`
- `src/training/trainer.py`
- `tests/test_legacy_feature_composition.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `compose_legacy_variant_features_torch()`, a pure Torch function with
  no parameters, registered buffers, module state, NumPy conversion, casts,
  device moves, detaches, or input mutation.
- Rejected an `nn.Module` composer for this phase because a module would alter
  `named_modules()` and architecture-rendering surface even without adding
  state-dict keys. The pure function keeps the model state surface unchanged.
- Added validation that the split tensors are Torch tensors with matching rank,
  leading dimensions, dtype, and device, and that content has at least one
  column.
- Preserved the L0 zero-width position path by returning `content_features`
  directly when `absolute_position_features.shape[-1] == 0`. The empty
  absolute-position tensor is therefore not required to receive a gradient.
- Added split-primary input resolution inside `SIEVE.forward()` immediately
  before `VariantEncoder`. When both split tensors are present, the model
  composes and executes the historical input ordering from them. When neither
  split tensor is present, historical `variant_features` is passed through
  unchanged. Partial split pairs and composed-width mismatches raise clear
  `ValueError`s.
- Threaded optional split tensors through `SIEVE.get_attention_patterns()`,
  `ChunkedSIEVEModel.forward()`, `ChunkedSIEVEModel.train_step()`,
  `ChunkedSIEVEModel.get_gene_embeddings()`,
  `ChunkedSIEVEModel.get_attention_patterns()`, and the standard
  `Trainer.train_epoch()` / `Trainer.validate()` paths.
- Passed split kwargs to base models only when at least one split tensor was
  supplied, preserving compatibility with legacy model doubles that accept only
  the historical feature signature.
- Left explanation modules and scripts unchanged. Integrated Gradients still
  differentiates historical `variant_features`.

Runtime behavior:

- Dataset-backed training now executes composed split tensors whenever batches
  contain both `content_features` and `absolute_position_features`.
- Historical `variant_features` remains a fully compatible fallback for old
  callers, explanation paths, attention-analysis scripts, and counterfactual
  callers.
- Custom positional strategies remain non-executable.
- The additive dataset representation still carries all three tensors:
  historical `features`, `content_features`, and
  `absolute_position_features`.

Compatibility:

- `VariantEncoder` construction and first-layer input width are unchanged.
- Model state-dict keys and tensor shapes are unchanged across historical and
  split-primary forwards.
- No composer-related state-dict keys exist.
- `load_state_dict_with_legacy_upgrade()` still accepts state dicts without any
  composer state.
- State-dict compatibility is what preserves historical checkpoints; checkpoint
  serialization and migration logic were not changed.
- CV versus single-split chromosome behavior is unchanged.
- Checkpoint serialization, CLI/configuration, attention implementation,
  encoding/preprocessing, explanation code, architecture rendering, and
  validation scripts were not modified.

Validation:

- `tests/test_legacy_feature_composition.py`: 32 passed, 1 skipped. The skipped
  test is the optional CUDA device-mismatch check on a CPU-only run.
- Focused regression command passed 253 tests, 1 skipped, with 1 existing
  non-failing deprecation warning from `tests/test_phase3_explain.py`.
- Full test suite passed 666 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for the changed model, trainer, and new test files.
- `git diff --check` passed.
- The new test file passed Ruff, Black check with Python 3.10 target, and isort
  check.
- The new pure helper passed Ruff, Black check with Python 3.10 target, and
  isort check.

Baseline static debt:

- Modified legacy Python files still carry pre-existing Ruff and isort debt.
- Ruff comparison for `src/models/sieve.py`, `src/models/chunked_sieve.py`, and
  `src/training/trainer.py` reported 61 baseline findings and 61 current
  findings.
- isort comparison failed for the same three legacy files in both baseline and
  current code, reflecting pre-existing import-order debt rather than new debt.
- Black baseline/current comparison for the three modified legacy files did not
  complete reliably in this environment and was interrupted. The new helper and
  new test file passed Black check.

Known limitations:

- Integrated Gradients remains historical-feature attribution; content-only
  attribution is still deferred.
- The model only composes the legacy sinusoidal input representation. Learned
  binned absolute position, RoPE, ALiBi, and other custom strategies remain
  deferred and are not executable.
- Training batches still carry historical `features` plus the two split tensors,
  so the temporary additive CPU and pinned-memory overhead remains.
- Explanation and attention-analysis scripts still pass historical
  `variant_features` and rely on the fallback path.

## Phase 5B3B - Core Content-Only Integrated Gradients Boundary

Goal:

Add the core Python API boundary for choosing legacy full-feature Integrated
Gradients or content-only Integrated Gradients, without changing the explanation
CLI, model state, checkpoint metadata, or output schemas.

Exact files changed:

- `src/explain/ig_mode.py`
- `src/explain/gradients.py`
- `tests/test_ig_content_mode.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added a pure `resolve_ig_mode()` helper plus `RequestedIGMode` and
  `IGModeCompatibilityWarning`. The resolver imports no Torch, Captum, model,
  dataset, or filesystem code.
- Required new-schema configs containing `position_encoding` to include valid
  nested `position_encoding.attribution.default_ig_mode` metadata, even when
  callers explicitly request `content` or `legacy`.
- Preserved old-config compatibility: `auto` resolves to legacy with a warning,
  explicit `legacy` resolves silently, and explicit `content` resolves with a
  warning that the current dataset must provide split tensors.
- Kept `SIEVEWrapper` as the public legacy wrapper and added
  `ContentSIEVEWrapper` for split-primary content attribution.
- Extended `IntegratedGradientsExplainer` with `ig_mode`, defaulting to
  `ResolvedIGMode.LEGACY`. The explainer rejects unresolved `auto`, selects
  exactly one wrapper, and constructs exactly one Captum
  `IntegratedGradients` object.
- Preserved legacy attribution semantics: `variant_features` remains the
  differentiable input, split tensors are rejected, and the default baseline is
  `zeros_like(variant_features)`.
- Added content attribution semantics: `variant_features` must be `None`,
  `content_features` is the only differentiable Captum input, and
  `absolute_position_features` is moved to the explainer device, detached, and
  passed as a fixed observed forward argument.
- Added strict baseline validation for both modes: explicit baselines must be
  Torch tensors with the exact differentiable-input shape and matching dtype
  after movement to the explainer device.
- Extended `attribute_batch()` so content mode requires split batch keys,
  does not require historical `features`, and applies the existing per-sample
  variant truncation indices to content, absolute position, positions, gene IDs,
  masks, and chromosome IDs together.

Runtime behavior:

- Existing Python callers keep legacy full-feature attribution unless they pass
  `ig_mode=ResolvedIGMode.CONTENT`.
- Content mode is available only through the Python explainer API in this
  phase. `scripts/explain.py` and downstream output schemas still use their
  existing legacy paths.
- Absolute position remains observed and active in content mode, but it is not
  a differentiable attribution target.
- No model, attention, training, checkpoint, config, state-dict, CLI, or
  ranking behavior changed.

Compatibility:

- `SIEVEWrapper` kept its public name and call signature.
- Model state-dict keys and tensor shapes are unchanged.
- Historical checkpoints remain compatible through unchanged model state.
- The legacy Python explainer API remains the default execution path.
- Old configs without `position_encoding` do not silently switch attribution
  target when `auto` is requested.

Validation:

- `tests/test_ig_content_mode.py`: 39 passed.
- Focused regression command passed 249 tests, 1 skipped, with 1 existing
  non-failing deprecation warning from `tests/test_phase3_explain.py`.
- Full test suite passed 705 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for the new resolver, modified gradients module, and new
  content-mode test file.
- `git diff --check` passed.
- The new resolver and new content-mode test file passed Ruff, isort, and
  Black check with Python 3.10 target. The combined two-file Black invocation
  hung in this environment and was interrupted, but each new file passed the
  same Black check individually.

Baseline static debt:

- Modified legacy `src/explain/gradients.py` still carries pre-existing Ruff,
  isort, and Black debt.
- Ruff comparison for committed baseline `src/explain/gradients.py` and the
  current modified file reported 19 findings in both versions.
- isort comparison failed for the committed baseline and current
  `src/explain/gradients.py`, reflecting pre-existing import-order debt rather
  than new debt.
- Black comparison reported that committed baseline and current
  `src/explain/gradients.py` would both be reformatted.

Known limitations:

- `scripts/explain.py` still has no `--ig-mode` CLI, config merge, metadata
  emission, or content-mode output schema.
- Manual explanation paths outside `IntegratedGradientsExplainer` still operate
  on historical feature tensors.
- Deterministic sampling behavior was intentionally not changed.
- Custom positional strategies remain non-executable.
- Content mode depends on current split tensors being present in batches; old
  datasets or hand-built batches without split keys cannot use content IG.

## Next planned phase

Phase 5B3C should integrate `--ig-mode` into `scripts/explain.py`, merge the
request with saved configuration metadata, and record explicit attribution-mode
metadata in explanation outputs while keeping custom positional strategies
disabled.
