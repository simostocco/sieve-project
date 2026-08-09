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

## Phase 5B3C - Explanation CLI and Attribution Metadata

Goal:

Integrate the 5B3B Integrated Gradients mode boundary into `scripts/explain.py`
and persist enough provenance to identify whether saved attribution files are
legacy full-feature or content-only outputs.

Exact files changed:

- `scripts/explain.py`
- `src/explain/__init__.py`
- `tests/test_explain_ig_mode.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `--ig-mode {auto,content,legacy}` to the explanation CLI. The default
  is `auto`.
- Extracted `build_arg_parser()` and kept `parse_args(argv=None)` compatible
  with normal command-line execution.
- Resolved IG mode only when IG executes. `auto` uses saved attribution policy
  for new-schema configs and preserves historical legacy attribution for old
  configs through the existing resolver warning.
- Preserved explicit old-config `--ig-mode content` behavior from the 5B3B
  resolver: it resolves to content and warns that split tensors are required.
- Kept `--skip-ig` attention-only execution from resolving IG mode, validating
  IG metadata, validating IG content dimensions or positional strategy metadata,
  constructing an explainer, or emitting compatibility warnings.
- Routed the production manual `sample -> chunks -> explainer.attribute()` loop
  through `_attribute_chunk_for_ig()`.
- In legacy mode, the chunk helper requires historical `features` and passes no
  split tensor keyword arguments.
- In content mode, the chunk helper requires `content_features` and
  `absolute_position_features`, does not require historical `features`, passes
  `None` as the historical feature input, and leaves fixed-position behavior to
  the 5B3B `ContentSIEVEWrapper`.
- Added structural content-dimension validation using
  `get_content_feature_dimension(annotation_level)`. If a new config includes
  top-level `content_dim`, it must be an integer matching that structural
  content width.
- Validated raw attribution width before mask-based row filtering. Content
  outputs must have width `content_dim`; legacy outputs must have width
  `input_dim`.
- Preserved the existing boolean mask authority for excluding padded variants
  from raw attributions, L2 scores, positions, gene IDs, and chromosome
  metadata.
- Kept per-variant score aggregation unchanged as L2 norm over the active
  attribution feature axis.
- Added semantic IG run metadata including requested/resolved mode, attribution
  feature space, widths, baseline policy, n-step/chunk parameters, sampling
  policy, and comparability warning.
- Added position-strategy metadata extraction for new-schema configs from
  `position_encoding.absolute.type`, `position_encoding.relative.type`, and
  `position_encoding.chromosome.encoding`. Old configs record those strategy
  fields as unavailable rather than inferred.
- Added NPZ-safe scalar metadata to per-sample attribution files while
  preserving the existing `attributions` and `variant_scores` keys.
- Added NPZ-safe scalar run metadata to top-level `attributions.npz` while
  preserving historical `variant_scores` and `metadata` arrays.
- Kept semantic Python metadata values as `None` where applicable. NPZ scalar
  serialization uses explicit non-object sentinels: unavailable positional
  strategies become `"unavailable"`, `sampling_seed=None` becomes `-1`, and
  `comparability_warning=None` becomes an empty string.
- Added `analysis_metadata.yaml` nested `integrated_gradients` metadata. Skipped
  IG records `executed: false`, the requested mode, and `resolved_ig_mode:
  null`.
- Added informational IG provenance columns to variant rankings, primary gene
  rankings, mean gene rankings, and size-normalised gene rankings after all
  ranking calculations complete.
- Updated `load_sample_attributions()` documentation so raw attribution width is
  described as `input_dim` for legacy files and `content_dim` for content files.

Runtime behavior:

- Attention analysis remains unchanged and continues to consume historical
  `batch["features"]`.
- 5B3B IG mathematics are unchanged. The script selects between those already
  implemented legacy/content explainer paths.
- No deterministic or random sampling behavior changed. The manual script path
  still processes deterministic dataset chunks without `attribute_batch()`
  random subsampling.
- No model, training, checkpoint, attention, ranking algorithm, or downstream
  comparison code changed.

Compatibility:

- Existing `--skip-ig` attention-only usage remains compatible and does not
  require IG content-dimension validation, positional-strategy metadata
  validation, or IG mode resolution.
- Existing per-sample NPZ consumers still find `attributions` and
  `variant_scores`.
- Existing top-level `attributions.npz` consumers still find `variant_scores`
  and the historical object-array `metadata`.
- `load_sample_attributions()` still returns exactly `attributions` and
  `variant_scores`.
- Old configs do not receive guessed positional strategy identifiers.

Validation:

- `tests/test_explain_ig_mode.py`: 41 passed.
- Focused regression command passed 279 tests, 1 skipped, with 1 existing
  non-failing deprecation warning from `tests/test_phase3_explain.py`.
- Full test suite passed 746 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `scripts/explain.py`, `src/explain/__init__.py`, and
  `tests/test_explain_ig_mode.py`.
- `git diff --check` passed.
- The new test file passed Ruff, Black check with Python 3.10 target, and isort
  check.

Baseline static debt:

- `scripts/explain.py` retained 19 Ruff findings in both committed baseline and
  current code.
- `src/explain/__init__.py` retained 1 Ruff finding in both committed baseline
  and current code.
- Black check reported that committed baseline and current `scripts/explain.py`
  would both be reformatted.
- Black check reported that committed baseline and current
  `src/explain/__init__.py` would both be reformatted.
- isort check failed for committed baseline and current `scripts/explain.py`,
  reflecting pre-existing import-order debt rather than new debt.
- isort check failed for committed baseline and current
  `src/explain/__init__.py`, reflecting pre-existing import-order debt rather
  than new debt.

Known limitations:

- Downstream comparison tools do not yet reject or warn on incompatible
  attribution modes.
- Deterministic sampling and selected-index persistence remain deferred.
- Custom positional strategies remain non-executable.
- New scalar metadata is added to explainability outputs, but checkpoint and
  training serialization are unchanged in this phase.

## Phase 5B3D - Deterministic IG Batch Sampling

Goal:

Make `IntegratedGradientsExplainer.attribute_batch()` reproducible when it
must subsample valid variants, and persist the exact original per-sample
variant rows that produced returned attributions, scores, and metadata.

Exact files changed:

- `src/explain/gradients.py`
- `tests/test_ig_sampling_reproducibility.py`
- `tests/test_ig_content_mode.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Preserved the historical `attribute_batch()` variant limit, mask filtering,
  attribution baselines, content/legacy differentiable input boundary,
  covariate handling, chromosome handling, and aggregation definitions.
- Replaced historical truncation sampling that depended on Torch's global RNG
  and was not independently reproducible unless external code controlled that
  state. The new explicit constructor argument defaults to `sampling_seed=0`
  for reproducible scientific attribution. Passing `sampling_seed=None`
  explicitly preserves nondeterministic compatibility behavior for Python
  callers.
- Added `MAX_TORCH_SEED = 2**63 - 1` and strict seed validation. The explainer
  accepts `None` or integer seeds from zero through `MAX_TORCH_SEED`, and
  rejects booleans, floats, strings, negative integers, other types, and
  integers above the Torch seed maximum without coercion.
- Added deterministic per-sample seed derivation using the global sample index:
  `(sampling_seed + global_sample_idx) % (MAX_TORCH_SEED + 1)`. This keeps
  selected subsets stable across DataLoader batch-size changes when sample
  order is unchanged, and documents the remaining limitation that changing
  sample order changes global-index seed assignment.
- Generated deterministic permutations with a local CPU `torch.Generator`, so
  unrelated global Torch RNG state and CUDA RNG state do not affect selected
  rows and are not consumed.
- Added no fallback to global RNG when `sampling_seed` is an integer. The only
  intentional nondeterministic compatibility path is `sampling_seed=None`.
- Preserved historical no-truncation execution by passing the original full
  padded per-sample tensors and original mask to attribution when
  `num_valid_variants <= max_variants`. No-truncation samples only add
  selected-index provenance after attribution.
- Updated the existing content-mode truncation test so it asserts tensor
  alignment through the persisted `selected_variant_indices` contract rather
  than monkeypatching the historical global `torch.randperm` call.
- Centralised variant-row selection around original row indices from the
  padded per-sample variant axis. The same selected row set is applied to
  historical `features`, `content_features`, `absolute_position_features`,
  `positions`, `gene_ids`, `mask`, and `chrom_ids`.
- Added always-present `selected_variant_indices` metadata as a sorted
  `np.int64` array of original valid row indices. Its length equals
  `num_variants_analyzed` whether or not truncation occurred.
- Added per-sample metadata fields `sampling_seed`,
  `effective_sampling_seed`, and `sampling_applied`. In this phase,
  `sampling_applied == truncated`; the separate field records the
  reproducibility semantics explicitly.
- Guaranteed that legacy and content IG modes select the same row indices when
  run over the same sample order, mask, `max_variants`, and `sampling_seed`.

Runtime behavior:

- `attribute_batch()` now defaults to deterministic variant subsampling when a
  sample has more valid variants than `max_variants`.
- No-truncation samples preserve historical full-padded-tensor attribution
  execution and now also persist their full valid original row indices in
  metadata, with `sampling_applied=False`, `truncated=False`, and
  `effective_sampling_seed=None`.
- `scripts/explain.py` remains unchanged. Its production manual chunk path
  still processes deterministic full chunks through `attribute()` and does not
  use `attribute_batch()` random subsampling.
- No model, training, checkpoint, config, attention, ranking, NPZ/YAML schema,
  or positional-encoding execution behavior changed.

Compatibility:

- Existing constructor call sites remain valid because `sampling_seed` is an
  optional keyword with a default.
- Existing `attribute_batch()` callers still receive exactly the same
  three-element return tuple: attributions, variant scores, metadata.
- Existing metadata keys remain present; the phase only adds new per-sample
  metadata keys.
- Explicit `sampling_seed=None` remains available for callers that need the
  historical nondeterministic subsampling path.

Validation:

- `tests/test_ig_sampling_reproducibility.py`: 33 passed.
- Focused regression command passed 141 tests with 1 existing non-failing
  deprecation warning from `tests/test_phase3_explain.py`.
- Full test suite passed 779 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/explain/gradients.py` and
  `tests/test_ig_sampling_reproducibility.py`, and
  `tests/test_ig_content_mode.py`.
- `git diff --check` passed.
- The new sampling test file passed Ruff, Black check with Python 3.10 target,
  and isort check.

Baseline static debt:

- Modified legacy `src/explain/gradients.py` retained 19 Ruff findings in both
  the committed baseline and current code.
- Black check reported that committed baseline and current
  `src/explain/gradients.py` would both be reformatted.
- isort check failed for committed baseline and current
  `src/explain/gradients.py`, reflecting pre-existing import-order debt rather
  than new debt.
- `tests/test_ig_content_mode.py` retained zero Ruff findings in both the
  committed baseline and current code. Black and isort checks improved from
  failing on the committed baseline to passing on the current file.

Known limitations:

- `attribute_batch()` deterministic seed assignment is tied to global sample
  order. Changing DataLoader/sample order changes which sample receives which
  effective seed.
- `scripts/explain.py` still records `sampling_seed=None` because its manual
  chunk path does not perform random subsampling.
- Downstream comparison tools do not yet enforce selected-index compatibility.
- Custom positional strategies remain non-executable.

## Next planned phase

Phase 6A - common positional-encoding interface design.

## Phase 6A - Positional Runtime Interface Design

Goal:

Define the model-side positional-runtime boundary before adding selectable
non-legacy strategies.

Design decisions:

- The observed absolute-position tensor is the legacy absolute boundary.
  Historical sinusoidal feature values are not recomputed inside the model.
- Absolute position remains separate from content features until the last
  compatibility step before `VariantEncoder`.
- Relative position uses a score-level runtime abstraction. The runtime sees
  base attention scores, query/key tensors, positions, chromosome ids, and the
  existing bias owner. This boundary is required for future RoPE behavior where
  same-chromosome scores may be rotated while cross-chromosome scores remain
  unrotated plus explicit bias.
- Cross-chromosome policy remains separate from relative strategy. The current
  legacy behavior still routes cross-chromosome pairs to a dedicated bucket
  only when chromosome ids are supplied; it does not mask cross-chromosome
  attention.
- Existing `position_bias` and `chrom_embedding` parameter ownership must stay
  on `PositionAwareSparseAttention`. Runtime objects must not own parameters,
  buffers, checkpoint metadata, or state-dict namespaces.
- Phase 6B was planned as a zero-intended-runtime-change refactor.

## Phase 6B - Legacy Positional Runtime Interfaces

Goal:

Introduce parameterless runtime interfaces for the executed legacy positional
behavior without enabling custom strategies or changing model computation.

Exact files changed:

- `src/models/position_runtime.py`
- `src/models/attention.py`
- `src/models/sieve.py`
- `tests/test_position_runtime_legacy_equivalence.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `AbsolutePositionRuntime`, a protocol whose `resolve()` method receives
  observed absolute-position features, genomic positions, chromosome ids, mask,
  and a reference content tensor.
- Added `ObservedAbsolutePositionRuntime`, a frozen, parameterless dataclass
  that returns the exact observed absolute-position tensor after minimal
  rank/leading-dimension validation. It does not clone, detach, cast, move, or
  recompute sinusoidal features. The L0 zero-width absolute-position tensor and
  L1-L4 64-channel observed tensors pass through unchanged.
- Added `RelativePositionRuntime`, a score-level protocol whose
  `adjust_attention_scores()` method receives base scores, query/key tensors,
  positions, chromosome ids, and the externally owned position-bias embedding.
- Added `LegacyT5RelativePositionRuntime`, a frozen, parameterless dataclass
  that reproduces the historical T5-style relative bucket bias.
- Added `compute_bias()` on `LegacyT5RelativePositionRuntime` as the
  compatibility helper. It preserves the historical per-batch loop, direct
  `relative_position_bucket()` call, direct `position_bias` lookup, and
  `[batch, heads, queries, keys]` permutation.
- Updated `PositionAwareSparseAttention._compute_position_bias()` to delegate
  to the runtime helper while preserving the method name and signature.
- Updated attention forward execution so Q/K/V projection, reshape, base QK
  scores, score-level bias adjustment, padding mask, softmax, `nan_to_num`,
  dropout, value aggregation, reshape, and output projection remain in the
  historical order.
- Added `ObservedAbsolutePositionRuntime` to `SIEVE` as a plain attribute, not
  an `nn.Module`. The split-primary path resolves the observed
  absolute-position tensor and then calls the unchanged historical feature
  composer before `VariantEncoder`.
- Preserved the historical `variant_features` fallback path. When split tensors
  are absent, the absolute runtime is not called.

Runtime behavior:

- Historical feature composition remains runtime authority for executed legacy
  model input.
- `position_bias` ownership is unchanged on each
  `PositionAwareSparseAttention` layer.
- `chrom_embedding` ownership is unchanged and remains allocated only when
  `num_chromosomes > 0`.
- State-dict keys are unchanged. No key contains `_absolute_position_runtime`,
  `_relative_position_runtime`, or `position_runtime`.
- State-dict tensor shapes are unchanged, including direct
  `attention.attention_layers.<N>.position_bias.weight` and
  `attention.attention_layers.<N>.chrom_embedding.weight` when configured.
- Parameter count is unchanged because runtime objects are not modules and own
  no parameters or buffers.
- L0-L4 historical input dimensions and content/absolute split dimensions are
  unchanged.
- Chromosome semantics are unchanged: chromosome ids are zero-based as passed,
  zero can still be a real chromosome id under `mask=True`, padding authority
  remains the mask, and cross-chromosome attention remains allowed.
- `scripts/train.py`, `scripts/explain.py`, `src/models/chunked_sieve.py`, and
  `src/encoding/*` were not changed.

Compatibility:

- Existing checkpoint key names and tensor ownership remain compatible.
- The synthetic old-checkpoint test with 32-row `position_bias.weight` loads
  through `load_state_dict_with_legacy_upgrade()`. The overlapping 32 rows are
  preserved exactly, and the destination cross-chromosome row remains valid
  according to the existing upgrade semantics.
- Existing chunked tests continue to prove split tensors and chromosome ids
  reach base `SIEVE`; no positional strategy logic was added to
  `ChunkedSIEVEModel`.
- Existing explainability tests continue to prove content-mode IG attribution
  width is `content_dim` and legacy-mode attribution width is `input_dim`.

Validation:

- `tests/test_position_runtime_legacy_equivalence.py`: 23 passed.
- Focused regression command passed 286 tests, 1 skipped, with 1 existing
  non-failing deprecation warning from `tests/test_phase3_explain.py`.
- Full test suite passed 802 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/models/position_runtime.py`,
  `src/models/attention.py`, `src/models/sieve.py`, and
  `tests/test_position_runtime_legacy_equivalence.py`.
- `git diff --check` passed.
- The two new Python files passed Ruff, Black check with Python 3.10 target,
  and isort check.
- Modified legacy files retained no net Ruff debt: committed baseline and
  current `src/models/attention.py` plus `src/models/sieve.py` both reported
  29 Ruff findings.
- Modified legacy files retained matching pre-existing Black debt: committed
  baseline and current `src/models/attention.py` plus `src/models/sieve.py`
  would both be reformatted.
- Modified legacy files retained matching pre-existing isort debt: committed
  baseline and current `src/models/sieve.py` both report import sorting debt.

Known limitations:

- Custom positional strategies remain non-executable.
- `cross_chromosome_policy=mask` is not implemented.
- Learned-binned absolute position is not implemented.
- RoPE is not implemented.
- ALiBi is not implemented.
- The current chromosome padding/zero-ID ambiguity remains unchanged.
- Normalized metadata/runtime discrepancies are not reconciled.

## Next planned phase

Phase 7 - selectable baseline positional strategies.

## Phase 7A - Selectable Baseline Strategy Design

Goal:

Define the baseline custom strategy architecture before adding executable
runtime implementations.

Design decisions:

- New-schema legacy runs will execute the normalized resolved configuration
  consistently in later Phase 7 integration.
- Historical CV versus single-split chromosome discrepancies remain old-schema
  compatibility behavior rather than new-schema execution semantics.
- New-schema checkpoints, whether `preset=legacy` or `preset=custom`, will
  require exact reconstruction from serialized configuration and state-dict
  surface.
- Custom sinusoidal absolute position is computed model-side and honors the
  configured coordinate scale, wavelength, and width.
- Custom sinusoidal padding rows are zeroed by mask so padded coordinate zero
  does not contribute `cos(0)=1` channels.
- `relative_position_encoding=none` owns no relative-position bias.
- T5 with `cross_chromosome_policy=mask` uses ordinary position buckets only;
  cross-chromosome score removal is a separate attention-mask concern.
- Chromosome policy remains separate from chromosome embedding. Chromosome ids
  can be required for pair routing even when learned chromosome embedding is
  disabled.
- Future custom attention execution will enforce query/key padding validity
  before softmax.
- Legacy IG will not be used for custom positional models.

Runtime behavior changed: no.

Known limitations:

- The design phase did not implement runtime algorithms.
- Model construction, attention wiring, training, explanation, checkpoint
  compatibility, and custom-strategy execution remained unchanged.

## Phase 7B1 - Baseline Positional Runtime Implementations

Goal:

Implement parameterless baseline positional runtime algorithms without wiring
them into SIEVE, attention, training, explanation, checkpoints, or preprocessing.

Exact files changed:

- `src/models/position_runtime.py`
- `tests/test_position_runtime_phase7_baselines.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `NoAbsolutePositionRuntime`, which validates the reference tensor and
  returns the zero-width `reference[..., :0]` view for custom
  `absolute_position_encoding=none`.
- Added `SinusoidalAbsolutePositionRuntime`, which computes custom model-side
  sinusoidal features from genomic positions using the resolved position width,
  coordinate scale, and max wavelength. It ignores observed historical absolute
  tensors numerically and zeroes masked padding rows.
- Added `NoRelativePositionRuntime`, which returns the exact base attention
  score tensor object unchanged for custom `relative_position_encoding=none`.
- Added `T5RelativePositionRuntime`, a parameterless custom T5 bias runtime
  that receives the attention-owned `position_bias` embedding explicitly.
- Implemented custom T5 `separate` behavior with required query/key chromosome
  ids, ordinary within-chromosome buckets, and a dedicated cross-chromosome row
  at `num_position_buckets`.
- Implemented custom T5 `mask` behavior with ordinary position buckets only and
  no cross-chromosome bucket lookup; future attention masking remains separate.
- Added `build_same_chromosome_pair_mask()` for zero-based chromosome routing
  masks without padding or softmax behavior.
- Added `validate_phase7_runtime_support()` for the Phase 7 supported runtime
  subset: absolute `none`/`sinusoidal`, relative `none`/`t5_bucket`, chromosome
  `none`/`learned`, and cross policy `separate`/`mask`.
- Added `build_absolute_position_runtime()` and
  `build_relative_position_runtime()` factories. Legacy configs still build
  `ObservedAbsolutePositionRuntime` for every level, including L0, and
  `LegacyT5RelativePositionRuntime` for relative position.
- Direct runtime construction now validates malformed custom sinusoidal and T5
  settings clearly, while leaving the pure resolver as the normal construction
  authority.

Runtime behavior:

- No model, attention, training, explanation, preprocessing, checkpoint, or
  feature-tensor execution path was changed.
- Split-primary batches are recomposed into the historical `VariantEncoder`
  representation, preserving historical feature semantics and ordering.
  `features` remains the compatibility fallback when split tensors are absent.
- `ObservedAbsolutePositionRuntime` and `LegacyT5RelativePositionRuntime`
  behavior remains unchanged.
- Runtime objects are plain frozen dataclasses, not `nn.Module` instances, and
  own no parameters, buffers, embeddings, or tensor configuration state.

Compatibility:

- Existing legacy state-dict ownership remains unchanged because factories are
  not wired into attention or SIEVE yet.
- Existing checkpoint compatibility remains governed by the Phase 6 state-dict
  surface.
- `src/models/attention.py`, `src/models/sieve.py`, `scripts/train.py`, and
  `scripts/explain.py` were not changed.

Validation:

- `tests/test_position_runtime_phase7_baselines.py`: 64 passed.
- Focused regression command passed 177 tests.
- Full test suite passed 866 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/models/position_runtime.py` and
  `tests/test_position_runtime_phase7_baselines.py`.
- `git diff --check` passed.
- The new Phase 7B1 test file passed Ruff, Black check with Python 3.10 target,
  and isort check.

Baseline static debt:

- Modified legacy `src/models/position_runtime.py` retained zero Ruff findings
  in both the committed baseline and current code.
- Black check improved for `src/models/position_runtime.py`: the committed
  baseline would be reformatted, while the current file passes Black check with
  Python 3.10 target.
- isort check passed for both committed baseline and current
  `src/models/position_runtime.py`.

Known limitations:

- Phase 7B1 does not allocate or remove model parameters for custom strategies.
- The new runtimes are not yet selected by SIEVE, attention, training, or
  explanation.
- Learned binned absolute position, RoPE, fixed ALiBi, and learned ALiBi remain
  unsupported and raise `NotImplementedError` through the Phase 7 runtime
  support validator.
- `cross_chromosome_policy=mask` computes no score mask yet; Phase 7B1 only
  provides ordinary T5 bias behavior and the same-chromosome pair-mask helper.
- Historical CV/single-split chromosome differences remain old-schema
  compatibility concerns until later integration phases.

## Phase 7B2 - Baseline Model Selection and State Surfaces

Goal:

Wire the Phase 7B1 baseline positional runtimes into SIEVE and attention while
preserving historical/no-config construction as the old-checkpoint-compatible
execution path.

Exact files changed:

- `src/models/attention.py`
- `src/models/sieve.py`
- `tests/test_position_model_phase7_selection.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added an optional `ResolvedPositionEncodingConfig` constructor boundary to
  `SIEVE`, `MultiLayerAttention`, and `PositionAwareSparseAttention`. Existing
  callers remain compatible because the argument is final and defaults to
  `None`.
- Kept historical/no-config mode as the Phase 6B path: observed absolute
  runtime, legacy T5 runtime, historical `position_bias` allocation with an
  extra row, historical `num_chromosomes` authority, permissive missing
  chromosome ids, historical key-only padding mask, and historical
  `variant_features` fallback.
- Added explicit new-schema mode when `position_encoding` is supplied. The
  resolved config is validated, `input_dim` must match
  `position_encoding.input_dim`, and nonzero conflicting constructor
  `num_chromosomes` values are rejected.
- Made explicit configs select absolute and relative runtimes through the
  Phase 7B1 factories. Explicit legacy configs still select observed absolute
  and legacy T5 runtimes, while custom configs can select absolute none,
  sinusoidal, relative none, or T5.
- Enforced the custom split-primary rule in SIEVE: explicit custom models must
  receive both `content_features` and `absolute_position_features`. Historical
  `variant_features` fallback remains allowed for no-config and explicit
  resolved legacy models.
- Preserved the model-side composer. Split inputs are still recomposed into the
  historical `VariantEncoder` order: dosage, absolute block, remaining content.
- Allocated `position_bias` according to resolved relative strategy in explicit
  mode: no parameter for relative none, `total_bias_rows` rows for T5, and no
  runtime-owned parameters.
- Allocated chromosome embeddings according to resolved chromosome strategy in
  explicit mode: no parameter for chromosome none, direct zero-initialized
  `chrom_embedding.weight` with `num_chromosomes + 1` rows for learned
  chromosome encoding.
- Kept chromosome routing independent from chromosome embedding. Chromosome ids
  can still be required for T5 separate or cross-policy mask when
  `chromosome_encoding=none`.
- Added explicit-config chromosome-id validation in attention. Required
  chromosome ids must be present, rank-2 tensors with shape matching positions,
  and real, unmasked ids must satisfy `0 <= chrom_id < num_chromosomes`. Padding
  remains mask-defined; chromosome id zero remains a valid real id.
- Added explicit-config cross-policy masking. `separate` applies no generic
  pair mask, while `mask` applies `build_same_chromosome_pair_mask()` after
  relative score adjustment and before padding validity masking.
- Added explicit-config query/key padding safety. When `mask` is supplied, both
  invalid query rows and invalid key columns are set to `-inf` before softmax,
  with existing `torch.nan_to_num(..., nan=0.0)` preserving zero weights for
  fully masked padded query rows.
- Kept `_compute_position_bias()` for T5-compatible callers and made it raise
  clearly when explicit relative none has no position bias.

Runtime behavior:

- Historical/no-config construction remains the compatibility path for old
  Python callers and old checkpoints.
- Explicit resolved legacy is new-schema execution and consistently allocates
  according to the resolved config, including learned chromosome embedding when
  the resolved config requires it.
- Explicit custom models now execute baseline absolute none, custom sinusoidal,
  relative none, T5 separate, T5 mask, and relative-none mask behavior through
  the selected runtimes.
- Training CLI custom execution is still blocked by `scripts/train.py`; this
  phase does not make custom training available.
- `scripts/train.py`, `scripts/explain.py`, `src/models/chunked_sieve.py`,
  `src/explain/*`, `src/training/*`, preprocessing, checkpoint serialization,
  and `src/models/position_runtime.py` were not changed.

State surfaces:

- Explicit relative none creates no `position_bias.weight` state-dict key.
- Explicit T5 separate creates direct per-layer `position_bias.weight` with
  `num_buckets + 1` rows.
- Explicit T5 mask creates direct per-layer `position_bias.weight` with
  `num_buckets` rows.
- Explicit chromosome none creates no `chrom_embedding.weight` state-dict key.
- Explicit learned chromosome creates direct per-layer
  `chrom_embedding.weight` with `num_chromosomes + 1` rows.
- No absolute-position runtime in Phase 7B2 owns trainable state.
- Resolved config objects and runtime dataclasses remain plain Python state; no
  `position_encoding`, `position_runtime`, `_relative_position_runtime`, or
  `_absolute_position_runtime` namespace appears in `state_dict()`.

Validation:

- `tests/test_position_model_phase7_selection.py`: 40 passed.
- Focused regression command passed 178 tests, 1 skipped.
- `tests/test_ig_content_mode.py`: 39 passed.
- Full test suite passed 906 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/models/attention.py`, `src/models/sieve.py`,
  and `tests/test_position_model_phase7_selection.py`.
- `git diff --check` passed.
- The new Phase 7B2 test file passed Ruff, Black check with Python 3.10 target,
  and isort check.

Baseline static debt:

- Modified legacy production files improved Ruff findings: committed baseline
  `src/models/attention.py` plus `src/models/sieve.py` reported 29 findings,
  while current code reports 28.
- Black check retained matching pre-existing formatting debt: committed
  baseline and current `src/models/attention.py` plus `src/models/sieve.py`
  would be reformatted.
- isort check improved: committed baseline reported import-sorting debt in
  `src/models/sieve.py`, while current `src/models/attention.py` and
  `src/models/sieve.py` pass isort check.

Known limitations:

- Training still blocks custom execution until Phase 7B3.
- Config deserialization, checkpoint metadata reconstruction, and exact
  new-schema checkpoint restoration are not yet wired.
- `scripts/explain.py` does not yet reconstruct custom positional models.
- Legacy IG custom rejection and custom-model attribution compatibility are
  deferred.
- Learned-binned absolute position, RoPE, fixed ALiBi, and learned ALiBi remain
  unsupported.
- Old checkpoint compatibility still uses the no-config historical path.

## Phase 7B3 - Training, Serialization, and Checkpoint Integration

Goal:

Enable new training runs to use the supported Phase 7 baseline positional
strategies through the resolved configuration, while preserving old no-config
construction as the compatibility path for existing callers and checkpoints.

Exact files changed:

- `scripts/train.py`
- `tests/test_position_training_phase7.py`
- `tests/test_train_position_cli.py`
- `tests/test_train_config_metadata.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Removed the Phase 4B blanket custom-training guard from
  `prepare_training_position_encoding()` and replaced it with
  `validate_phase7_runtime_support()`. Supported configs now resolve for
  training; learned-binned absolute position, RoPE, fixed ALiBi, and learned
  ALiBi still raise `NotImplementedError` before model construction.
- Kept resolver validation as the source of invalid-config `ValueError`
  failures. `scripts/train.py` does not duplicate position-configuration
  validation.
- Made the resolved configuration the new training model-width authority.
  `main()` now uses `resolved_position_encoding.input_dim` after dataset
  construction and before the CV/single-split branch.
- Added `create_training_model()` so CV and single-split training construct
  models through the same resolved-config path.
- Extended `create_model()` with an optional final `position_encoding`
  argument. Existing callers can omit it and retain historical/no-config model
  construction.
- Passed the resolved config to `SIEVE` for new training runs. CV and
  single-split training now both pass `dataset.num_chromosomes` to model
  construction.
- Updated execution metadata to schema version 2, recording the applied
  resolved strategy surface rather than the older transitional
  path-dependent legacy description.
- Validated serialized chromosome mapping cardinality against the resolved
  chromosome count before attaching the mapping to the nested position config.
- Added consistency checks before serialization so top-level `input_dim` and
  `num_chromosomes` must match the resolved configuration being recorded.
- Added `config_schema_version=2` and
  `position_encoding_schema_version=<resolved schema>` to run metadata while
  preserving existing raw CLI fields and nested `position_encoding`.

Runtime behavior:

- New training runs execute supported custom baselines:
  absolute `none`/`sinusoidal`, relative `none`/`t5_bucket`, chromosome
  `none`/`learned`, and cross-chromosome `separate`/`mask`.
- Default CLI behavior still resolves the legacy preset, but new training now
  uses explicit resolved legacy model construction rather than the old
  path-dependent no-config model path.
- Split-primary batches remain recomposed into the historical
  `VariantEncoder` representation before encoding, preserving historical
  feature semantics and ordering.
- The historical `features` tensor remains the compatibility fallback when
  split tensors are absent in model calls that allow fallback.
- No preprocessing, feature tensor generation, attention math, model runtime
  classes, checkpoint-writing machinery, or explanation code was changed in
  this phase.

Serialization and checkpoint compatibility:

- Parent training configs and CV fold configs now receive the same normalized
  run metadata produced from the resolved position config.
- Checkpoint metadata receives the same run metadata through the existing
  `Trainer` path; no `Trainer` code changed.
- New-schema checkpoints preserve strategy state through ordinary model
  `state_dict()` keys selected by the resolved config, such as direct T5
  `position_bias.weight` and learned chromosome `chrom_embedding.weight` when
  configured.
- Old checkpoints and old Python callers remain compatible through omitted
  `position_encoding`, which still constructs the historical/no-config model
  surface.
- Explanation-time reconstruction of new custom positional models is still
  deferred; `scripts/explain.py` was not changed.

Validation:

- `tests/test_position_training_phase7.py`: 36 passed.
- Focused regression command including train parser/config helpers passed 375
  tests.
- Full test suite passed 944 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `scripts/train.py`,
  `tests/test_position_training_phase7.py`, `tests/test_train_position_cli.py`,
  and `tests/test_train_config_metadata.py`.
- `git diff --check` passed.
- `tests/test_position_training_phase7.py` passed Ruff, Black check with
  Python 3.10 target, and isort check.
- The modified train/config test files also passed Ruff, Black check with
  Python 3.10 target, and isort check.

Baseline static debt:

- `scripts/train.py` retained matching Ruff debt: committed baseline and
  current code both report 20 Ruff findings.
- `scripts/train.py` retained matching Black formatting debt: committed
  baseline and current code both would be reformatted.
- `scripts/train.py` retained matching isort import-order debt: committed
  baseline and current code both fail isort check.

Known limitations:

- Config-driven reconstruction in `scripts/explain.py` and validation scripts
  is still deferred.
- This phase does not implement learned-binned absolute position, RoPE, fixed
  ALiBi, or learned ALiBi.
- No real training was run; coverage uses deterministic unit tests and helper
  integration tests only.
- Historical no-config behavior remains necessary for old checkpoints and old
  callers, so new-schema training and old-checkpoint reconstruction remain
  intentionally separate paths.

## Next planned phase

Phase 7B4 - config/checkpoint reconstruction readers for new-schema positional
models, including explanation-time compatibility boundaries.
