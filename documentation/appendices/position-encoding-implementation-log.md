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

## Phase 7B4A - Config and Checkpoint Reconstruction Foundation

Goal:

Add the reusable config/checkpoint reconstruction layer that Phase 7B4B will
use to make explanation new-schema aware, without changing explanation,
training, or model execution paths yet.

Exact files changed:

- `src/encoding/position_config.py`
- `src/models/reconstruction.py`
- `tests/test_position_reconstruction_phase7.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Reconstruction classes:

- Case A is the authoritative schema-v2 new-run format produced by all training
  from Phase 7B3 onward. It is triggered only by config-level
  `position_encoding` plus `config_schema_version == 2`.
- Case B is old historical config compatibility for configs without
  `position_encoding`. Checkpoint state is the structural authority.
- Case C is transitional metadata-only compatibility for pre-v2 configs that
  contain `position_encoding` but explicitly record
  `position_encoding_execution.resolved_config_applied_to_model == False`.
  These configs did not execute the resolved config and therefore reconstruct
  through the same state-driven historical path as Case B.
- Cases B and C are read-only compatibility paths. No future training path
  should create them.

Implementation:

- Added `resolved_position_encoding_from_dict()`, a pure deserializer that does
  not import Torch, construct models, access files, or apply runtime support
  gates.
- The deserializer rebuilds a `PositionEncodingRequest` from serialized active
  strategy fields, calls `resolve_position_encoding_config()`, and compares the
  result against the serialized canonical fields. This keeps the resolver as
  the single source of positional configuration math.
- The training-only `position_encoding.chromosome.mapping` extension is allowed
  and validated. It must be a mapping from the real zero-based chromosome IDs
  `"0"` through `str(num_chromosomes - 1)` to chromosome-name strings. It is
  not retained in the resolved dataclass.
- Unknown fields in canonical position-encoding sections are rejected, and
  corruption in resolved fields such as `input_dim`, `content_dim`,
  `total_bias_rows`, `requires_chrom_ids`, `cross_chromosome_parameter`, and
  `default_ig_mode` raises explicit `ValueError`s.
- Added `src/models/reconstruction.py` with `ReconstructedSIEVEModel` and
  `reconstruct_sieve_from_checkpoint()`.
- Case A reconciles only Phase 7B3 architecture/provenance metadata from
  checkpoint metadata, without mutating caller mappings. Config values remain
  primary; matching metadata succeeds, missing metadata is tolerated, allowed
  missing config fields may be filled from metadata, and conflicts raise.
- Case A validates structural consistency between top-level config fields and
  the resolved config, including `input_dim`, `content_dim`,
  `num_chromosomes`, and `position_encoding_schema_version`.
- Case A validates current Phase-7 runtime support during model reconstruction,
  constructs explicit resolved SIEVE, wraps chunked checkpoints when the state
  keys are consistently `base_model.`-prefixed, and loads with
  `strict=True`.
- Schema-v2 normalized `preset=legacy` is still Case A. It reconstructs an
  explicit resolved legacy SIEVE, allocates the normalized learned chromosome
  embedding, and strict-loads exact state.
- Case B/C historical reconstruction infers old `input_dim` from the unique
  `variant_encoder.encoder.0.weight` state tensor and infers chromosome module
  presence from `chrom_embedding.weight` state tensors. Dataset chromosome
  count is not used as old-checkpoint architecture authority.
- Compatibility loading now has a preflight before calling the existing
  `load_state_dict_with_legacy_upgrade()`. The only accepted shape migration
  is historical T5 `position_bias.weight` rows `32 -> 33` with the same head
  width; arbitrary missing keys, unexpected keys, and tensor-shape corruption
  are rejected.
- Checkpoint metadata alone cannot promote Case B or C into Case A.
- Raw nullable training CLI fields, such as `num_position_buckets=None`, do not
  override authoritative resolved positional metadata or historical defaults
  during reconstruction.
- Canonical serialized resolved fields are type-strict as well as
  value-strict, so malformed metadata such as `True` for integer fields or
  float representations of integer dimensions is rejected.
- `position_encoding.chromosome.mapping` may be absent, but if present it must
  be a real mapping; explicit null is invalid.
- Authoritative outer schema and architecture metadata are type-strict as well
  as value-strict.

Runtime behavior:

- No explanation, IG, attention analysis, training, preprocessing, SIEVE,
  attention, position runtime, chunked wrapper, or trainer execution file was
  changed.
- The new reconstruction helper only consumes existing model APIs.

Validation:

- `tests/test_position_reconstruction_phase7.py`: 87 passed.
- Focused regression command passed 354 tests.
- Historical legacy reconstruction regression passed 23 tests.
- Full test suite passed 1031 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/encoding/position_config.py`,
  `src/models/reconstruction.py`, and
  `tests/test_position_reconstruction_phase7.py`.
- `git diff --check` passed.
- The new reconstruction module and test file passed Ruff, Black check with
  Python 3.10 target, and isort check.

Baseline static debt:

- Modified legacy `src/encoding/position_config.py` retained zero Ruff
  findings in both committed baseline and current code.
- `src/encoding/position_config.py` retained matching Black formatting debt:
  committed baseline and current code both would be reformatted.
- `src/encoding/position_config.py` passed isort check in both committed
  baseline and current code.

Known limitations:

- `scripts/explain.py` still uses the historical reconstruction path and has
  not yet consumed `reconstruct_sieve_from_checkpoint()`.
- IG policy, attention analysis inputs, validation scripts, and downstream
  reconstruction consumers remain deferred.
- Dataset mapping checksum verification is not implemented in this phase.
- Learned-binned absolute position, RoPE, fixed ALiBi, and learned ALiBi can
  deserialize if resolver-valid, but reconstruction currently rejects them
  through Phase-7 runtime support validation.

## Next planned phase

## Phase 7B4B - Explanation, IG Policy, and Attention Integration

Goal:

Make explanation consume Phase 7B4A reconstruction so supported Phase-7
positional models are explainable through the same architecture authority used
for checkpoint loading.

Exact files changed:

- `scripts/explain.py`
- `src/explain/ig_mode.py`
- `src/explain/attention_analysis.py`
- `tests/test_explain_position_phase7.py`
- `tests/test_explain_ig_mode.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- `scripts/explain.py` now calls `reconstruct_sieve_from_checkpoint()` after
  creating the chunked dataset and uses the returned model/base model for
  attention and chunk-level IG.
- Explanation no longer mutates loaded config to fill `input_dim` or
  `num_chromosomes` for reconstruction. Actual model input width comes from
  `reconstruction.base_model.input_dim`.
- IG metadata also derives `input_dim` and legacy attribution width directly
  from `reconstruction.base_model.input_dim`; callers no longer pass a
  duplicate width.
- Content attribution width comes from the reconstructed resolved config for
  authoritative schema-v2 Case A and from structural annotation-level content
  width for historical/transitional Cases B/C.
- `resolve_ig_mode()` accepts an explicit `is_new_schema` execution-authority
  flag while preserving the old Python API when omitted. The flag accepts only
  `None` or exact booleans, rejecting integer/string lookalikes.
- Case A uses saved attribution metadata for `auto`; explicit `content` is
  allowed; explicit `legacy` is resolved but rejected later for custom
  positional execution.
- Case A normalized legacy still allows legacy IG because the historical
  complete feature representation remains meaningful for that preset.
- Cases B/C resolve through historical compatibility. Case C positional
  metadata is not treated as executed architecture, and warnings describe
  historical/transitional execution.
- Custom schema-v2 legacy IG is rejected before attribution execution because
  custom positional execution no longer corresponds to the historical complete
  `features` attribution target.
- `AttentionAnalyzer.extract_attention_weights()` now supports two explicit
  modes: historical `variant_features` or split `content_features` plus
  `absolute_position_features`. Missing, partial, or mixed feature inputs raise
  clear `ValueError`s.
- `scripts/explain.py` routes attention through split-primary inputs only for
  Case A custom positional models. Case A legacy and Cases B/C keep historical
  feature attention routing.
- Position strategy provenance is now reconstruction-aware: Case A reports
  executed resolved strategies, Case B reports unavailable old-config
  strategies, and Case C reports transitional historical execution without
  claiming intended positional metadata was executed.
- `--skip-ig` still does not resolve IG mode, validate custom legacy IG,
  validate attribution content dimensions, or construct the IG explainer.

Runtime behavior:

- No training, preprocessing, model, attention runtime, checkpoint
  reconstruction, or gradients implementation files were changed.
- Existing content IG remains differentiable only over `content_features`;
  absolute position, positions, chromosome IDs, gene IDs, masks, and covariates
  remain fixed.
- Existing old-schema explanation compatibility remains intact through Case B/C
  reconstruction and historical IG/attention routing.

Validation:

- `tests/test_explain_position_phase7.py`: 19 passed.
- IG/content/sampling command passed 110 tests.
- `tests/test_attention_analysis.py`: 2 passed.
- `tests/test_position_reconstruction_phase7.py`: 87 passed.
- `tests/test_phase3_explain.py`: 20 passed, with 1 existing deprecation
  warning.
- Attention/reconstruction/explanation regression command passed 109 tests,
  with 1 existing deprecation warning.
- Broader focused command passed 314 tests, with 1 existing deprecation
  warning.
- Full test suite passed 1047 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `scripts/explain.py`, `src/explain/ig_mode.py`,
  `src/explain/attention_analysis.py`, and
  `tests/test_explain_position_phase7.py`.
- `git diff --check` passed.
- `tests/test_explain_position_phase7.py` passed Ruff, Black check with
  Python 3.10 target, and isort check.
- Modified legacy `tests/test_explain_ig_mode.py` passed Ruff and isort; Black
  check retained matching pre-existing formatting debt relative to committed
  baseline.
- Modified legacy `scripts/explain.py` retained matching Ruff, Black, and
  isort debt relative to committed baseline.
- Modified legacy `src/explain/attention_analysis.py` retained matching Ruff,
  Black, and isort debt relative to committed baseline.
- Modified legacy `src/explain/ig_mode.py` passed Ruff and isort checks;
  Black check improved from baseline formatting debt to clean current output.

Known limitations:

- Explanation support is limited to the Phase-7 strategies already accepted by
  reconstruction/runtime support: legacy, none, sinusoidal, and T5 bucket
  combinations.
- Learned-binned absolute position, RoPE, fixed ALiBi, and learned ALiBi remain
  deferred even if their serialized configs can be parsed.
- Dataset mapping checksum verification is still deferred.
- Downstream validation and analysis scripts have not yet been aligned to the
  Phase 7B4A reconstruction helper.

## Phase 8B1 - Learned-Binned Genome Layout Metadata

Goal:

Add the reference-genome and serialization metadata needed to make
`absolute_position_encoding=learned_binned` reconstructable later, without
allocating learned positional parameters or enabling learned-binned execution.

Exact files changed:

- `src/data/genome.py`
- `src/encoding/position_config.py`
- `src/encoding/position_layout.py`
- `scripts/train.py`
- `tests/test_learned_binned_position_layout.py`
- `tests/test_position_encoding_config.py`
- `tests/test_train_config_metadata.py`
- `tests/test_position_runtime_phase7_baselines.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added authoritative canonical chromosome lengths for GRCh37 and GRCh38 for
  normalized chromosomes `1..22`, `X`, `Y`, and `MT`, plus copy-safe lookup
  helpers in `src/data/genome.py`.
- Added pure learned-binned layout metadata in
  `src/encoding/position_layout.py`. The layout records schema version,
  `coordinate_origin=1`, the `chromosome_local_contiguous` layout identifier,
  chromosome lengths ordered by chromosome ID, per-chromosome bin counts, and
  total `num_embeddings`.
- The coordinate convention is explicit: repository positions are 1-based, so
  future local bin lookup is `(position_bp - 1) // bin_size_bp`.
- Layout construction validates saved chromosome ID-to-name mappings as
  contiguous real zero-based IDs, uses chromosome ID order rather than mapping
  insertion or lexical order, rejects non-standard contigs, and derives all bin
  counts by ceiling division from authoritative chromosome lengths. Serialized
  chromosome ID keys must use canonical string spellings such as `"0"` and
  `"1"`; alternate decimal spellings such as `"00"` or `"01"` are rejected.
- Added a pure mapping validator that checks serialized
  `position_encoding.chromosome.mapping` is the exact inverse of the current
  `chrom_index`; it does not shift IDs, invent padding rows, or accept
  non-mapping saved metadata.
- `ResolvedAbsolutePositionConfig` remains pure strategy configuration only:
  `type`, `fusion`, `dim`, coordinate parameters, and `bin_size_bp`. It does
  not retain chromosome lengths, offsets, or table size.
- `resolved_position_encoding_from_dict()` now allows and validates the
  training-only learned-binned `position_encoding.absolute.binning` extension,
  strips it before canonical resolver equality, and still rejects the extension
  on non-learned-binned absolute encodings.
- Learned-binned absolute encoding now requires chromosome IDs and positive
  `num_chromosomes` even when chromosome embedding, relative position, and
  cross-chromosome routing are otherwise disabled.
- `serialize_position_encoding_for_training()` attaches the learned-binned
  `absolute.binning` extension from the resolved `bin_size_bp`, genome build,
  and validated chromosome mapping. Non-learned-binned configs remain
  unchanged and omit `absolute.binning`.

Runtime behavior:

- No `nn.Embedding`, runtime lookup, model constructor, attention, explanation,
  reconstruction, preprocessing tensor, checkpoint state-dict, or training
  execution behavior was changed.
- `validate_phase7_runtime_support()` still rejects learned-binned absolute
  position before model construction.
- Historical feature composition remains the executed authority for supported
  Phase-7 paths; learned-binned is metadata/config foundation only.

Validation:

- `tests/test_learned_binned_position_layout.py`: 19 passed.
- Phase 8B1 focused command passed 262 tests.
- Full test suite passed 1069 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for changed Python files.
- `git diff --check` passed.
- New files `src/encoding/position_layout.py` and
  `tests/test_learned_binned_position_layout.py` passed Ruff and isort.
- Black check passed for each new file individually. The combined two-file
  Black invocation hung twice in this environment and was interrupted; per-file
  checks were clean.

Baseline static debt:

- Modified legacy files retained matching Ruff debt: committed baseline and
  current code both report 30 findings across the touched legacy file set.
- Modified legacy files retained no worse isort debt: committed baseline had 4
  import-order errors, current code has 1 remaining pre-existing
  `scripts/train.py` import-order error.
- Modified legacy files retained no worse Black formatting debt: committed
  baseline had 6 failing files, current code has 3 remaining production files
  that Black would reformat.

Known limitations:

- The learned-binned layout is not yet executed by SIEVE or ChunkedSIEVEModel.
- The serialized layout does not duplicate chromosome names; the authoritative
  names remain in `position_encoding.chromosome.mapping`.
- Chromosome offsets are derived prefix sums, not serialized independent state.
- Non-standard contigs are rejected for learned-binned layout construction.

## Phase 8B2 - Learned-Binned Registered Embedding and Runtime Lookup

Goal:

Implement model-side learned-binned absolute-position execution for direct
SIEVE construction while keeping training, checkpoint reconstruction, and
explanation unchanged.

Exact files changed:

- `src/models/position_runtime.py`
- `src/models/sieve.py`
- `src/models/attention.py`
- `tests/test_learned_binned_position_runtime.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Split runtime validation by ownership. `validate_attention_runtime_support()`
  validates only attention-owned relative/chromosome/cross-chromosome
  behavior and does not reject absolute-position strategies. This lets
  attention receive the truthful resolved config when SIEVE owns learned
  absolute fusion.
- Added `validate_model_runtime_support()` for complete SIEVE execution. It
  calls the attention validator, allows `none` and `sinusoidal` without a
  layout, and allows `learned_binned` only with a valid
  `LearnedBinnedAbsolutePositionLayout`.
- Preserved `validate_phase7_runtime_support()` as the external Phase-7 gate
  used by training and reconstruction. It still rejects
  `absolute_position_encoding=learned_binned`.
- Added `LearnedBinnedAbsolutePositionRuntime`, a plain dataclass that owns no
  registered state. It references the SIEVE-registered embedding and immutable
  layout metadata.
- Extended `SIEVE` with
  `learned_binned_position_layout: LearnedBinnedAbsolutePositionLayout | None`.
  Direct learned-binned construction requires this layout. Supplying a layout
  for any non-learned-binned absolute strategy raises.
- For learned-binned only, SIEVE registers exactly one trainable parameter:
  `absolute_position_embedding.weight` with shape
  `(layout.num_embeddings, resolved.absolute.position_dim)`.
- The learned absolute embedding is initialized to exact zeros so the strategy
  is neutral at construction while selected rows can diverge through gradients.
- The global row formula is chromosome-local contiguous:
  `local_bin = (position_bp - 1) // bin_size_bp` and
  `global_row = layout.chromosome_offsets[chrom_id] + local_bin`.
- The 1-based boundary is explicit: positions `1` and `bin_size_bp` map to
  local bin 0, while `bin_size_bp + 1` maps to local bin 1.
- Accepted integer runtime inputs are canonicalized internally to `torch.long`
  before row/index arithmetic. This preserves the public integer-dtype
  validation rule while avoiding narrow-int and `uint8` indexing semantics for
  genomic coordinates.
- Padding is mask-authoritative. The learned runtime validates only
  `mask=True` rows, initializes padded lookup rows to 0, writes real global
  rows only into real positions, performs embedding lookup, and forces
  `mask=False` outputs to exact zeros. Real `chrom_id=0` remains valid.
- Real rows require integer positions/chromosome IDs, position `>= 1`,
  `0 <= chrom_id < num_chromosomes`, and position not exceeding that
  chromosome's serialized length. Invalid real coordinates raise `ValueError`;
  they are not clamped.
- Exact chromosome-end coordinates are valid, but coordinates past the
  serialized chromosome length are rejected even when they would fall inside
  the final allocated bin.
- Direct layout validation is type-strict for scalar architecture fields:
  `schema_version`, `coordinate_origin`, and `num_embeddings` must be non-bool
  integers before equality checks are applied, and `layout` must be a string.
- The observed historical `absolute_position_features` tensor is ignored as
  learned-binned execution authority. Learned-binned uses positions,
  chromosome IDs, mask, layout, and the registered embedding.

Runtime behavior:

- Direct SIEVE construction with custom learned-binned absolute position and a
  valid layout now succeeds.
- Direct SIEVE passes the original authoritative resolved config through to
  attention unchanged; no synthetic absolute=`none` attention view is created.
- Learned-binned coexists with supported attention-owned strategies, including
  T5 bucket relative bias and learned chromosome embedding.
- No training CLI behavior changed. Training still calls
  `validate_phase7_runtime_support()` and rejects learned-binned before model
  construction.
- No checkpoint reconstruction behavior changed. Reconstruction still calls
  `validate_phase7_runtime_support()` and rejects learned-binned.
- No explanation, Integrated Gradients, chunked wrapper, feature-composition,
  preprocessing, or data tensor generation code changed.

Compatibility and state-dict effects:

- Custom `none`, custom `sinusoidal`, explicit legacy, and no-config historical
  models do not contain `absolute_position_embedding.weight`.
- Learned-binned direct SIEVE models contain exactly one
  `absolute_position_embedding.weight` key.
- Existing Phase-7 relative-position, chromosome embedding, attention,
  VariantEncoder, and historical checkpoint state keys remain unchanged.

Validation:

- `tests/test_learned_binned_position_runtime.py`: 33 passed.
- Focused runtime/model command passed 272 tests.
- Training/reconstruction gate regression command passed 123 tests.
- Full test suite passed 1102 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/models/position_runtime.py`,
  `tests/test_learned_binned_position_runtime.py`.
- `git diff --check` passed.
- `src/models/position_runtime.py` and
  `tests/test_learned_binned_position_runtime.py` passed Ruff and isort.
- A combined two-file Black check hung twice in this environment and was
  interrupted; per-file Black checks with Python 3.10 target passed for both
  touched Python files.

Baseline static debt:

- Modified legacy files retained matching Ruff debt: committed baseline and
  current code both report 28 findings across `src/models/position_runtime.py`,
  `src/models/sieve.py`, and `src/models/attention.py`.
- Modified legacy files improved isort status from 3 baseline import-order
  errors to 0 current errors.
- Modified legacy files retained matching Black formatting debt: committed
  baseline and current code both have 3 files that Black would reformat.

Known limitations:

- Learned-binned training serialization consumption and strict checkpoint
  reconstruction remain deliberately disabled until Phase 8B3.
- Explanation does not yet reconstruct or attribute learned-binned models.
- Layout-to-live-dataset mapping verification remains deferred.

## Phase 8B3 - Learned-Binned Training and Strict Reconstruction

Goal:

Enable learned-binned absolute position for new training model construction
and authoritative schema-v2 checkpoint reconstruction, without changing
explanation or compatibility checkpoint semantics.

Exact files changed:

- `scripts/train.py`
- `src/models/reconstruction.py`
- `tests/test_learned_binned_training_reconstruction.py`
- `tests/test_position_training_phase7.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Training now validates the current training strategy surface with
  `validate_attention_runtime_support()` during pure resolution, so
  `absolute_position_encoding=learned_binned` is accepted while unsupported
  attention-owned RoPE and ALiBi strategies still fail.
- The serialized learned-binned layout is the sole architecture authority for
  model construction. Training resolves the config, builds `run_metadata`,
  reads the exact `run_metadata["position_encoding"]`, parses
  `position_encoding.absolute.binning`, validates model support with that
  parsed layout, and only then persists the normalized metadata to
  `config.yaml`.
- CV and single-split training reuse the same immutable parsed layout object
  for every model created in the run. No per-fold or single-split layout is
  rebuilt from genome/chromosome inputs.
- `create_model()` and `create_training_model()` now accept
  `learned_binned_position_layout` and pass it to `SIEVE`; `ChunkedSIEVEModel`
  itself remains unchanged.
- Authoritative Case-A reconstruction parses learned-binned layout metadata
  from the saved `position_encoding.absolute.binning` extension, validates the
  complete model strategy with `validate_model_runtime_support()`, allocates
  the exact embedding table shape, and then loads with `strict=True`.
- Missing or malformed learned-binned `absolute.binning` rejects Case-A
  reconstruction. Reconstruction does not infer table architecture from
  checkpoint tensor shape, current genome metadata, observed positions, or
  defaults.
- `reconstruct_sieve_from_checkpoint()` now accepts optional
  `dataset_chrom_index`. For Case-A learned-binned reconstruction, a supplied
  live dataset mapping must exactly invert the saved
  `position_encoding.chromosome.mapping`, because learned-bin row identity is
  chromosome-ID identity. Supplying only `dataset_num_chromosomes` for
  learned-binned is rejected; pure checkpoint reconstruction with neither live
  dataset argument remains allowed.
- No learned-binned checkpoint migration was added. Wrong embedding row count,
  wrong embedding width, missing learned embedding, or unexpected learned
  embedding under a non-learned strategy fail through native strict PyTorch
  state loading.

Runtime behavior:

- New training can construct learned-binned models with supported relative
  strategies (`none` and `t5_bucket`).
- Learned-binned training models contain
  `base_model.absolute_position_embedding.weight` in chunked model state, with
  shape `(serialized_layout.num_embeddings, resolved.absolute.position_dim)`
  and zero initialization before training updates.
- Tiny split-primary forward/backward through a training-created learned-binned
  model reaches selected embedding rows.
- Custom `none`, custom `sinusoidal`, explicit legacy, and historical no-config
  paths keep their existing state surfaces and do not gain learned-binned
  embedding keys.

Compatibility effects:

- Case A is strict and config-primary. Config/checkpoint metadata conflicts in
  nested learned-binned binning or chromosome mapping continue to reject before
  model loading.
- Cases B and C remain unchanged and state-driven. They do not parse or execute
  learned-binned metadata as architecture, and the only compatibility migration
  remains the historical T5 32-row to 33-row position-bias upgrade.
- Explanation remains deferred. `scripts/explain.py` still passes only
  `dataset_num_chromosomes`, so learned-binned explanation may fail until the
  Phase 8B4 caller supplies the live chromosome mapping.

Validation:

- `tests/test_learned_binned_training_reconstruction.py`: 20 passed.
- Training focused command passed 127 tests.
- Reconstruction focused command passed 107 tests.
- Runtime regression command passed 156 tests.
- Broader focused command passed 389 tests.
- Full test suite passed 1121 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `scripts/train.py`,
  `src/models/reconstruction.py`, and
  `tests/test_learned_binned_training_reconstruction.py`.
- `git diff --check` passed.
- New test file `tests/test_learned_binned_training_reconstruction.py` passed
  Ruff, Black check with Python 3.10 target, and isort.

Baseline static debt:

- Modified production files retained matching Ruff debt: committed baseline
  and current code both report 20 findings across `scripts/train.py` and
  `src/models/reconstruction.py`.
- Modified production files improved isort status from 2 baseline import-order
  errors to 1 current `scripts/train.py` import-order error.
- Modified production files improved Black status from 2 baseline files that
  Black would reformat to 1 current file (`scripts/train.py`). Combined
  two-file production Black checks hung and were interrupted; per-file checks
  completed.

Known limitations:

- Learned-binned explanation and content-IG end-to-end integration remain
  deferred.
- Live dataset mapping compatibility is enforced only for authoritative
  schema-v2 learned-binned reconstruction.
- Training still does not run real datasets in this development environment;
  coverage uses deterministic unit tensors only.

## Phase 8B4 - Learned-Binned Explanation and Content-IG Integration

Goal:

Enable schema-v2 learned-binned models to reconstruct through the explanation
entry point and prove content-mode Integrated Gradients and attention analysis
use the learned-binned runtime path without changing training, preprocessing,
model runtime, reconstruction internals, or attribution internals.

Exact files changed:

- `scripts/explain.py`
- `tests/test_explain_position_phase7.py`
- `tests/test_explain_learned_binned_phase8.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- `_reconstruct_model_for_explanation()` now forwards the live
  `dataset.chrom_index` into `reconstruct_sieve_from_checkpoint()`.
- Explanation reconstruction remains config/checkpoint-primary. For
  schema-v2 learned-binned models, the saved resolved positional metadata and
  strict state dict define the architecture, while the live chromosome mapping
  is used only to verify identity compatibility with the saved mapping.
- Added end-to-end learned-binned explanation tests that build schema-v2
  metadata through the training serializer, reconstruct through the explanation
  helper, and verify nonzero `absolute_position_embedding.weight` values are
  restored exactly.
- Added tests proving live chromosome mapping name or ID mismatches reject
  learned-binned explanation reconstruction.
- Added tests proving `auto` and explicit `content` IG resolve to content mode
  for learned-binned schema-v2 custom models, while legacy IG is rejected for
  custom positional execution.
- Added tests proving content IG differentiates only `content_features`, reports
  `content_dim` attribution width, preserves the learned embedding as fixed
  model context, and ignores the observed historical
  `absolute_position_features` compatibility tensor for learned-binned absolute
  execution.
- Added tests proving attention analysis works on split-primary learned-binned
  inputs and ignores observed historical absolute-position feature values.
- Added tests proving learned-binned absolute position can reconstruct for
  explanation with T5 relative bias and learned chromosome embedding.
- Added regression coverage showing non-learned Case-A and historical
  Case-B/C explanation reconstruction remain unchanged.

Runtime behavior:

- Learned-binned schema-v2 explanation reconstruction can now succeed when the
  live dataset supplies a chromosome mapping that exactly matches the saved
  mapping.
- Content-mode IG remains the comparable attribution path: position stays active
  in the model through the learned embedding, but only content features are
  integrated.
- Historical `features` remain a compatibility fallback for old callers and
  historical explanation paths. Split-primary learned-binned explanation uses
  content features plus model-side positional runtime lookup.
- No training, preprocessing, model construction, attention implementation,
  reconstruction implementation, gradient implementation, or IG policy
  implementation changed.

Compatibility effects:

- Old-schema and transitional explanation reconstruction continue to use their
  historical compatibility paths.
- Schema-v2 learned-binned explanation rejects incompatible live chromosome
  mappings rather than remapping IDs or inferring padding rows.
- `src/explain/gradients.py`, `src/explain/ig_mode.py`,
  `src/explain/attention_analysis.py`, model runtime files, reconstruction, and
  training remained unchanged in this phase.

Validation:

- `tests/test_explain_learned_binned_phase8.py`: 9 passed.
- New plus existing explanation reconstruction helper command passed 28 tests.
- Explanation-focused command passed 140 tests.
- Learned-binned lifecycle command passed 81 tests.
- Phase-7 explanation/reconstruction command passed 126 tests with 1 existing
  deprecation warning.
- Broader focused command passed 319 tests with 1 existing deprecation warning.
- Full test suite passed 1130 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `scripts/explain.py` and
  `tests/test_explain_learned_binned_phase8.py`.
- `git diff --check` passed.
- New test file `tests/test_explain_learned_binned_phase8.py` passed Ruff,
  Black check with Python 3.10 target, and isort.

Baseline static debt:

- Modified legacy files retained matching Ruff debt: committed baseline and
  current code both report 19 findings across `scripts/explain.py` and
  `tests/test_explain_position_phase7.py`.
- Modified legacy files retained matching Black formatting debt: committed
  baseline and current code both have 2 files that Black would reformat.
- Modified legacy files improved isort status from 2 baseline import-order
  errors to 1 current import-order error in `scripts/explain.py`.

Known limitations:

- Learned-binned explanation depends on exact live chromosome mapping identity
  for schema-v2 reconstruction; no remapping or repair path is provided.
- Learned-binned content IG is now covered through deterministic unit tensors,
  but no real explanation dataset or training run was executed.
- RoPE and ALiBi remain future positional strategies.

## Phase 9B - RoPE Runtime and Attention State Surface

Goal:

Implement RoPE relative-position execution inside the existing positional
runtime and attention interfaces while preserving historical T5/none numerics,
state-dict compatibility for existing strategies, and all training,
reconstruction, explanation, preprocessing, and checkpoint lifecycle behavior.

Exact files changed:

- `src/models/position_runtime.py`
- `src/models/attention.py`
- `tests/test_rope_position_runtime.py`
- `tests/test_position_runtime_phase7_baselines.py`
- `tests/test_position_training_phase7.py`
- `tests/test_position_reconstruction_phase7.py`
- `tests/test_learned_binned_position_runtime.py`
- `tests/test_learned_binned_training_reconstruction.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `RopeRelativePositionRuntime`, a parameter-free runtime that rotates Q
  and K using adjacent feature pairs and the standard inverse-frequency formula
  `rope_base ** (-arange(0, head_dim, 2) / head_dim)`.
- RoPE uses the raw 1-based positions already present in batches. Padded
  `position=0` is accepted by the runtime itself because it has no mask; the
  attention layer validates that real, unmasked RoPE positions are `>= 1`.
- RoPE rotates only Q/K. V is never rotated.
- RoPE dtype policy is local to the runtime: float64 Q/K use float64 trig,
  rotation, and score matmul; every other floating Q/K dtype uses float32 for
  trig, rotation, and score matmul. Routed scores are cast back to
  `base_scores.dtype`.
- RoPE requires `base_scores` to be floating-point before score routing, so
  rotated scores cannot be silently quantized by an integer compatibility score
  tensor.
- Added a `cross_chromosome_bias` argument to the relative runtime API.
  T5/legacy/none runtimes reject a supplied cross bias so accidental wiring is
  visible rather than silently ignored.
- For RoPE with cross-chromosome `separate`, attention registers one learned
  per-head `cross_chromosome_bias` parameter with shape `(num_heads,)`; no key
  is registered for RoPE `mask`, T5, none, or legacy paths.
- RoPE same-chromosome pairs use rotated scores. RoPE separate
  cross-chromosome pairs use unrotated base scores plus the per-head learned
  cross bias. RoPE mask cross-chromosome pairs use the unrotated base score and
  rely on the existing chromosome-aware attention mask to remove them.
- `build_relative_position_runtime()` now accepts `head_dim` for RoPE runtime
  construction. Non-RoPE construction keeps its previous defaults and state
  surfaces.
- `validate_attention_runtime_support()` now allows RoPE. The older
  `validate_phase7_runtime_support()` helper explicitly continues to reject
  RoPE so Phase-7 contract tests remain frozen.

Runtime behavior:

- Historical no-config legacy attention, custom none, and custom T5 retain
  their existing score mathematics and state-dict keys.
- RoPE requires chromosome IDs for same/cross-chromosome routing. It does not
  add learned-binned-style chromosome-name identity validation; exact mapping
  identity remains required only for strategies with chromosome-row-indexed
  parameters.
- RoPE separate introduces only the dedicated
  `attention.attention_layers.*.cross_chromosome_bias` parameter. RoPE mask
  introduces no relative-position parameter.
- Model forward/backward, returned attention weights, padding behavior, learned
  chromosome embeddings, learned-binned absolute embeddings, and chunked model
  state-prefix behavior are covered with deterministic unit tensors.
- A direct split-primary model test covers learned-binned absolute runtime and
  RoPE relative runtime composing in one forward/backward pass.
- No training CLI, config serialization, checkpoint reconstruction,
  explanation, gradients, attention analysis, preprocessing, or dataset
  implementation changed in this phase.

Compatibility effects:

- Existing state dicts for historical legacy, custom none, custom T5, and
  learned-binned absolute runtime remain protected by exact key-set tests.
- The cross-bias parameter exists only when the resolved architecture is
  `relative=rope` with `cross_chromosome_policy=separate`; there is no
  zero-sized compatibility parameter or buffer placeholder.
- Lifecycle helpers that use the attention-runtime support gate may now accept
  RoPE. Tests that intentionally describe unsupported future strategies now use
  ALiBi, while the Phase-7-specific gate directly proves RoPE remains rejected
  there.

Validation:

- `tests/test_rope_position_runtime.py`: 43 passed.
- RoPE plus Phase-7 runtime/model/config focused command passed 263 tests.
- Training, reconstruction, and explanation lifecycle command passed 140
  tests.
- Learned-binned compatibility command passed 62 tests.
- Full test suite passed 1172 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for `src/models/position_runtime.py`,
  `src/models/attention.py`, and `tests/test_rope_position_runtime.py`.
- `git diff --check` passed.
- New test file `tests/test_rope_position_runtime.py` passed Ruff, Black check
  with Python 3.10 target, and isort.
- Modified pre-existing Python files retained matching Ruff debt: committed
  baseline and current code both report 13 findings across the touched
  production and test files.
- Current touched pre-existing Python files passed isort. The committed
  baseline has import-order findings in those files, so this phase introduced
  no isort debt.
- Per-file Black checks match the pre-existing production formatting debt in
  `src/models/position_runtime.py` and `src/models/attention.py`; touched test
  files are Black-clean in the current diff.

Known limitations:

- RoPE training, schema-v2 strict reconstruction, checkpoint compatibility
  policy, explanation/IG policy, and downstream validation script alignment are
  deferred to the next lifecycle phase.
- ALiBi strategies remain unsupported.
- No real training, dataset generation, or checkpoint migration was executed.

## Phase 9C Prerequisite - Schema-v2 Chromosome Row Identity

Goal:

Fix a schema-v2 reconstruction correctness gap discovered by the Phase 9C
diagnostic before continuing broader RoPE lifecycle validation.

Exact files changed:

- `src/models/reconstruction.py`
- `tests/test_rope_lifecycle_phase9.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Diagnostic:

- A true schema-v2 model with `relative=rope` and `chromosome=learned`
  correctly reconstructed when the live dataset mapping matched the saved
  `position_encoding.chromosome.mapping`, but also reconstructed when live
  chromosome IDs for chr1/chr2 were swapped.
- That was unsafe because `chrom_embedding.weight` rows are indexed by
  `chrom_id`, so row identity is biological architecture metadata, not merely
  tensor shape.

Implementation:

- Added centralized Case-A chromosome row-identity validation in
  `src/models/reconstruction.py`.
- Exact chromosome mapping identity is now required when the resolved
  schema-v2 architecture has chromosome-row-indexed learned state:
  `chromosome=learned` or `absolute=learned_binned`.
- The helper reuses
  `validate_saved_chromosome_mapping_matches_chrom_index()` from
  `src.encoding.position_layout`; no parallel comparator was introduced.
- Authoritative schema-v2 configs with row-indexed chromosome state must carry
  `position_encoding.chromosome.mapping`. Missing saved mapping now rejects
  reconstruction.
- When live `dataset_chrom_index` is supplied, exact inverse mapping equality
  is required. When only `dataset_num_chromosomes` is supplied, reconstruction
  rejects because cardinality cannot prove row identity. Pure checkpoint
  reconstruction without a live dataset remains allowed when saved mapping is
  complete.
- `_case_a_learned_binned_layout()` now focuses on learned-bin layout parsing;
  live mapping compatibility is enforced once by the centralized Case-A
  identity helper before layout parsing.

Runtime and compatibility effects:

- RoPE-only configurations with `chromosome=none` still do not require exact
  chromosome-name identity; same/cross routing depends on IDs, but RoPE owns no
  chromosome-row-indexed parameter.
- T5 plus `chromosome=learned` is now protected by the same row-identity guard,
  because the learned chromosome embedding is independent of the relative
  strategy.
- Learned-binned exact mapping behavior remains protected after the refactor.
- Historical Case B and transitional Case C remain state-driven compatibility
  paths and do not acquire schema-v2 mapping policy.
- No remapping, migration, checkpoint repair, attention-forward validation, or
  RoPE-specific mapping check was added.

Validation:

- `tests/test_rope_lifecycle_phase9.py`: 11 passed.
- Reconstruction-focused command passed 127 tests.
- Broader positional/explanation regression command passed 317 tests.
- Full test suite passed 1183 tests, 1 skipped, with 6 existing non-failing
  warnings.

Known limitations:

- This prerequisite fixes schema-v2 row identity only. The broader Phase 9C
  RoPE training, strict reconstruction, explanation, IG, provenance, and
  attention lifecycle validation remains to be completed.

## Phase 9C - RoPE Training, Strict Reconstruction, and Explanation Lifecycle

Goal:

Complete RoPE lifecycle validation across training metadata, strict
schema-v2 reconstruction, explanation reconstruction, content IG, attention
analysis, and learned-binned composition without changing production runtime
code.

Exact files changed:

- `tests/test_rope_lifecycle_phase9.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added deterministic lifecycle tests proving training resolves and serializes
  RoPE `separate` and `mask` configurations with the expected normalized
  position metadata and execution metadata.
- Added tests proving serialized RoPE training metadata round-trips through
  `resolved_position_encoding_from_dict()` without losing non-default
  `rope_coordinate_scale`, `rope_base`, cross-chromosome policy, or resolved
  content/input dimensions.
- Added training-created model tests for RoPE `separate` and `mask`, including
  exact relative-position state surfaces, split-primary forward execution, and
  finite backward propagation for the `separate` path.
- Added strict schema-v2 reconstruction tests for base SIEVE and chunked SIEVE
  RoPE checkpoints, including nonzero cross-chromosome bias restoration,
  multi-layer required-key coverage, corrupt relative-state rejection, and
  metadata/config conflict rejection before state authority is applied.
- Added explanation lifecycle tests proving schema-v2 RoPE reconstruction
  restores state, selects content IG policy with the expected provenance,
  keeps RoPE position context fixed during content IG, and exposes split-primary
  attention through `AttentionAnalyzer`.
- Added a deterministic position-sensitivity regression showing RoPE
  same-chromosome attention changes when positions change.
- Added a learned-binned absolute plus RoPE relative lifecycle test proving the
  strict reconstruction path restores both learned absolute embedding state and
  RoPE cross-chromosome bias state together.

Runtime behavior:

- No production code changed in this phase.
- RoPE model behavior remains the Phase 9B implementation: split-primary
  batches are recomposed into the historical VariantEncoder representation,
  RoPE rotates Q/K for same-chromosome pairs, and RoPE `separate`
  cross-chromosome pairs use base scores plus the learned per-head bias.
- Historical feature semantics and ordering remain preserved.
- `features` remains the compatibility fallback when split tensors are absent.
- No training, reconstruction, explanation, attention-analysis, or IG runtime
  path was modified.

Compatibility effects:

- Schema-v2 RoPE configs are now covered end to end by tests for training
  metadata, strict reconstruction, explanation reconstruction, content IG, and
  attention extraction.
- RoPE `separate` state is protected by exact required-key and forbidden-key
  tests; RoPE `mask` is protected against accidental relative state.
- State-dict compatibility remains the mechanism for restoring historical
  checkpoints; no migration path or checkpoint metadata promotion was added.
- The Phase 9C prerequisite chromosome row-identity guard remains in force for
  schema-v2 architectures with chromosome-row-indexed learned state.
- Learned-binned absolute and RoPE relative state surfaces are covered together
  without adding chromosome-name identity validation specifically for RoPE-only
  routing.

Validation:

- `tests/test_rope_lifecycle_phase9.py`: 35 passed.
- RoPE runtime plus lifecycle command passed 78 tests.
- Training and reconstruction lifecycle command passed 203 tests.
- Explanation and attention lifecycle command passed 142 tests.
- Broader focused command passed 353 tests.
- Full test suite passed 1207 tests, 1 skipped, with 6 existing non-failing
  warnings.

Known limitations:

- No real training, dataset generation, checkpoint migration, or production
  explanation run was executed.
- Downstream validation scripts beyond the explanation and attention-analysis
  paths covered by unit tests remain later-roadmap work.
- ALiBi strategies remain unsupported.

## Phase 10B1 - Fixed ALiBi Runtime and Attention State Surface

Goal:

Implement fixed ALiBi relative-position execution while preserving historical,
T5, RoPE, learned-binned absolute, reconstruction, training, and explanation
behavior outside the newly supported fixed-ALiBi strategy.

Exact files changed:

- `src/models/position_runtime.py`
- `src/models/attention.py`
- `tests/test_alibi_position_runtime.py`
- `tests/test_position_training_phase7.py`
- `tests/test_position_reconstruction_phase7.py`
- `tests/test_learned_binned_position_runtime.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `build_alibi_fixed_slopes(num_heads)`, the deterministic standard
  ALiBi head-slope schedule. The exact locked schedules are:
  - 1 head: `(0.00390625,)`
  - 2 heads: `(0.0625, 0.00390625)`
  - 4 heads: `(0.25, 0.0625, 0.015625, 0.00390625)`
  - 6 heads: `(0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125)`
- Added `FixedAlibiRelativePositionRuntime`, a frozen plain dataclass that
  stores fixed slopes as a Python tuple and owns no `nn.Parameter`, buffer,
  tensor field, or state-dict key.
- Implemented chromosome-local bidirectional genomic distance:
  `abs(position_i_bp - position_j_bp)`.
- Implemented scale placement as `distance_bp / alibi_distance_scale`, with
  `linear` using the scaled value and `log1p` using
  `log1p(distance_bp / alibi_distance_scale)`.
- Implemented the same-chromosome score penalty:
  `base_score - fixed_slope[h] * transformed_distance`.
- Reused the existing attention-owned `cross_chromosome_bias` parameter for
  fixed ALiBi with `cross_chromosome_policy=separate`; it has shape
  `(num_heads,)` and zero initialization.
- Kept fixed ALiBi `position_bias=None`; no fake T5 embedding row is allocated.
- Kept the runtime API unchanged for Phase 10B1. Learned slope arguments are
  deferred until learned ALiBi exists in Phase 10B2.
- Extended `build_relative_position_runtime()` with an optional `num_heads`
  keyword required only for fixed ALiBi. Existing none/T5/legacy callers remain
  compatible, and RoPE retains its `head_dim` requirement.
- Extended configured attention's mask-aware real-position validation from RoPE
  to RoPE plus fixed ALiBi: real variants must have integer positions `>= 1`,
  while padded rows may still carry position `0`.
- Enabled `validate_attention_runtime_support()` for `ALIBI_FIXED` only.
  `ALIBI_LEARNED` remains unsupported.
- Kept `validate_phase7_runtime_support()` frozen for historical Phase-7 entry
  points; it now explicitly rejects fixed ALiBi after attention support accepts
  it.

Runtime behavior:

- Fixed ALiBi is score-bias-only. It does not rotate Q, K, or V, does not
  inspect V, and does not perform a second QK matmul.
- The runtime mathematically depends only on `base_scores`, `positions`,
  `chrom_ids`, fixed slopes, and optional cross-chromosome bias. Tests prove
  changing Q/K while holding `base_scores` fixed does not change the output.
- For `cross_chromosome_policy=separate`, same-chromosome pairs receive the
  ALiBi penalty and cross-chromosome pairs receive `base_score +
  cross_chromosome_bias[h]`. No cross-chromosome genomic distance penalty is
  applied.
- For `cross_chromosome_policy=mask`, same-chromosome pairs receive the ALiBi
  penalty and cross-chromosome pairs are left as base scores inside the
  runtime; the existing outer attention mask sets those pairs to `-inf`.
- Positions are promoted to int64 before subtraction, and exact integer
  base-pair distance is computed before float32 or float64 transform math.
- Distance and bias transform math uses float64 when `base_scores` is float64;
  otherwise it uses float32 locally and casts the routed result back to
  `base_scores.dtype`.

Compatibility effects:

- Historical/no-config, custom none, custom T5, and RoPE relative-position
  state surfaces remain covered by existing regression tests.
- Fixed ALiBi `mask` creates no relative-position state key.
- Fixed ALiBi `separate` creates only
  `attention.attention_layers.<N>.cross_chromosome_bias`.
- Learned chromosome embeddings and learned-binned absolute embeddings remain
  independent state surfaces that compose with fixed ALiBi.
- Schema-v2 strict reconstruction now supports fixed ALiBi through ordinary
  resolved-config model construction and strict state loading.
- No checkpoint migration, explanation logic, training serialization logic, or
  preprocessing/data path changed.
- `ALIBI_LEARNED` remains unsupported and is still rejected before model
  execution.

Validation:

- `tests/test_alibi_position_runtime.py`: 54 passed.
- Position runtime/model focused command passed 294 tests.
- Training/reconstruction support regression command passed 177 tests.
- Broader focused command passed 504 tests.
- Full test suite passed 1262 tests, 1 skipped, with 6 existing non-failing
  warnings.

Known limitations:

- Learned ALiBi slopes and their state surface are deferred.
- Fixed ALiBi lifecycle coverage is limited to direct runtime/model tests and
  existing training/reconstruction support regressions; full training,
  strict-reconstruction, explanation, IG, and attention-analysis lifecycle tests
  analogous to RoPE remain for a later lifecycle phase.
- No real training, dataset generation, checkpoint migration, or production
  explanation run was executed.

## Phase 10B2 - Learned ALiBi Slopes and State Surface

Goal:

Implement learned ALiBi relative-position execution and attention-owned
checkpoint state without adding the full training, reconstruction, explanation,
or IG lifecycle matrix deferred to Phase 10C.

Exact files changed:

- `src/models/position_runtime.py`
- `src/models/attention.py`
- `tests/test_alibi_position_runtime.py`
- `tests/test_position_training_phase7.py`
- `tests/test_position_reconstruction_phase7.py`
- `tests/test_learned_binned_position_runtime.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added `build_alibi_initial_slope_logits(num_heads)`, a deterministic pure
  helper that computes raw learned-ALiBi initialization logits from the fixed
  ALiBi schedule using inverse softplus, `log(expm1(slope))`.
- Added `LearnedAlibiRelativePositionRuntime`, a frozen plain dataclass that
  owns no tensor, parameter, buffer, or state-dict key.
- Extended the relative runtime API with `alibi_slope_logits`; non-learned
  runtimes reject accidental non-`None` logits instead of silently ignoring
  incorrectly wired learned state.
- Added attention-owned `alibi_slope_logits` as the exact checkpoint parameter
  name for `relative_position_encoding=alibi_learned`.
- Kept raw logits distinct from physical slopes. The runtime computes effective
  slopes as `softplus(alibi_slope_logits)` at execution time. Mathematically
  softplus is strictly positive; in finite precision, sufficiently negative
  logits may underflow to exactly zero. The enforced semantic invariant is that
  effective slopes cannot become negative and therefore cannot turn genomic
  distance into a reward.
- Initialized learned effective slopes to match the fixed ALiBi schedule.
- Shared fixed and learned ALiBi score adjustment through one implementation
  path: positions are promoted to int64, pairwise subtraction and absolute
  base-pair distance are computed exactly as integers, and only the resulting
  distance enters float32 or float64 transform math.
- Fixed ALiBi constructs its deterministic slope tensor directly in the selected
  ALiBi compute dtype, preserving Phase 10B1 float64 fixed-slope precision.
- Reused the existing per-head `cross_chromosome_bias` parameter for learned
  ALiBi with `cross_chromosome_policy=separate`; `mask` uses no cross-bias
  parameter.
- Kept both ALiBi variants `position_bias=None`; no T5-style bias rows are
  allocated for fixed or learned ALiBi.
- Generalized configured numeric relative-position validation to RoPE, fixed
  ALiBi, and learned ALiBi: real variants require integer positions `>= 1`,
  while padded rows may carry position `0`.
- Enabled `validate_attention_runtime_support()` and model construction for
  `ALIBI_LEARNED`.
- Kept `validate_phase7_runtime_support()` frozen; it still explicitly rejects
  RoPE, fixed ALiBi, and learned ALiBi.

Runtime behavior:

- Learned ALiBi is score-bias-only. It does not rotate Q/K, transform V, or
  compute an additional QK product.
- Same-chromosome scores use
  `base_score - softplus(alibi_slope_logits[h]) * transformed_distance`.
- With `cross_chromosome_policy=separate`, cross-chromosome scores use
  `base_score + cross_chromosome_bias[h]`; no cross-chromosome genomic distance
  penalty is applied.
- With `cross_chromosome_policy=mask`, cross-chromosome pairs remain base
  scores inside the runtime and are masked by the existing outer configured
  attention mask.
- Learned ALiBi inherits the Phase 10B1 exact-distance policy for large genomic
  coordinates, including the float32 score path.

Compatibility effects:

- Fixed ALiBi behavior remains covered by the existing exact schedule,
  no-slope-state, query/key-independence, mask/separate state, and
  large-coordinate one-base-pair regressions.
- Learned ALiBi state surfaces per attention layer are now:
  - mask: `alibi_slope_logits`;
  - separate: `alibi_slope_logits` plus `cross_chromosome_bias`.
- Chunked models naturally prefix learned ALiBi state with `base_model.`.
- Learned chromosome embeddings and learned-binned absolute embeddings remain
  independent state surfaces that compose with learned ALiBi.
- Learned ALiBi slope state is indexed by attention head, not chromosome, so it
  adds no chromosome-name identity requirement. Existing schema-v2 mapping
  guards remain limited to chromosome-row-indexed learned state.
- Generic schema-v2 strict reconstruction now round-trips learned ALiBi model
  state, including deterministic non-default raw logits and cross bias, without
  changing reconstruction production code.
- Training preparation and model construction now accept learned ALiBi, but the
  full Phase 10C lifecycle remains deferred.

Validation:

- `tests/test_alibi_position_runtime.py`: 82 passed.
- Position runtime/model focused command passed 322 tests.
- Training/reconstruction support regression command passed 177 tests.
- Broader focused command passed 532 tests.
- Full test suite passed 1290 tests, 1 skipped, with 6 existing non-failing
  warnings.

Known limitations:

- Full learned-ALiBi training, strict reconstruction, explanation, Integrated
  Gradients, attention-analysis, and corrupt-checkpoint lifecycle coverage is
  deferred to Phase 10C.
- No real training, dataset generation, checkpoint migration, or production
  explanation run was executed.

## Phase 10C - ALiBi Training, Strict Reconstruction, and Explanation Lifecycle

Goal:

Complete end-to-end lifecycle validation for fixed and learned ALiBi without
changing production code, ALiBi mathematics, checkpoint state semantics,
training code, reconstruction code, or explanation code.

Exact files changed:

- `tests/test_alibi_lifecycle_phase10.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added a focused Phase 10C lifecycle test module covering ALiBi training
  resolution, metadata serialization, strict reconstruction, explanation,
  content IG, attention analysis, mapping identity, and learned-binned
  composition.
- Used real `train.prepare_training_position_encoding()` for fixed and learned
  ALiBi with `separate` and `mask` cross-chromosome policies.
- Used real `train.build_training_run_metadata()` and verified non-default
  `alibi_distance_function=linear` and `alibi_distance_scale=23456.0` serialize
  and round-trip through `resolved_position_encoding_from_dict()`.
- Verified execution metadata records the applied relative strategy, no
  T5-style `position_bias_rows`, and the selected cross-chromosome policy.
- Constructed training-created models through the real training model factory
  and locked exact state surfaces:
  - fixed + mask: no relative state;
  - fixed + separate: `cross_chromosome_bias`;
  - learned + mask: raw `alibi_slope_logits`;
  - learned + separate: raw `alibi_slope_logits` plus `cross_chromosome_bias`.
- Proved training-created forward/backward execution for all four fixed/learned
  and separate/mask combinations, including finite content gradients and
  nonzero learned raw-logit/cross-bias gradients where those parameters exist.
- Added strict schema-v2 base and chunked reconstruction round trips for fixed
  and learned ALiBi `separate`, including exact tensor-state equality.
- Verified learned ALiBi restores raw `alibi_slope_logits` exactly from
  checkpoint state; tests do not compare only effective softplus outputs.
- Added strict corrupt-state rejection for fixed separate, fixed mask, learned
  separate, and learned mask checkpoints.
- Added a two-layer learned-ALiBi state ownership regression proving per-layer
  raw logits and cross-bias tensors are independent and required.
- Added config/checkpoint metadata conflict tests for ALiBi distance function
  and scale before checkpoint tensor state is accepted as architecture
  authority.
- Verified historical Case B and transitional Case C remain historical and do
  not infer ALiBi state from aspirational metadata.
- Proved ALiBi-only reconstruction does not require exact chromosome-name
  identity because ALiBi state is indexed by attention head, not chromosome.
- Proved the existing chromosome-row identity guard remains active when learned
  chromosome embeddings or learned-binned absolute embeddings are present.
- Used `explain._reconstruct_model_for_explanation()` for fixed and learned
  ALiBi schema-v2 checkpoints and verified strategy/function/scale plus ALiBi
  state restoration.
- Verified `auto` and `content` IG policy resolves to content mode for custom
  schema-v2 ALiBi, while legacy IG remains rejected through the existing generic
  custom positional execution policy.
- Exercised content-only Integrated Gradients for fixed and learned ALiBi with
  zero-width absolute-position features for `absolute=none`; attributions have
  content width, finite values, and preserved positions/chromosomes/masks.
- Verified learned ALiBi raw logits and cross-bias parameters remain exactly
  stable through content IG.
- Added post-reconstruction distance-sensitivity tests proving fixed and learned
  ALiBi attention weights change when same-chromosome relative distances change.
- Exercised `AttentionAnalyzer` on fixed and learned ALiBi `separate`, and
  fixed ALiBi `mask`, including exact zero cross-chromosome attention
  probability for a valid masked cross-chromosome pair.
- Verified IG provenance records the reconstructed resolved config as metadata
  source, content attribution space, content width, and the correct ALiBi
  relative strategy without serializing ALiBi distance matrices or slope
  tensors.
- Added learned-binned absolute plus fixed/learned ALiBi strict lifecycle
  coverage, including exact learned absolute table restoration, ALiBi state
  restoration, matching chromosome mapping success, and swapped mapping
  rejection through the existing learned-binned row-identity guard.

Runtime behavior:

- No production code changed.
- Fixed and learned ALiBi mathematics remain the Phase 10B1/10B2
  implementation: exact int64 distance subtraction before floating transform
  math, fixed deterministic slopes or learned raw logits with softplus
  effective slopes, no Q/K/V transformation, no T5 `position_bias`, and
  existing separate/mask cross-chromosome routing.

Compatibility effects:

- Existing training, reconstruction, explanation, IG, attention-analysis,
  learned-binned, RoPE, and historical compatibility paths are exercised by
  tests but not modified.
- Checkpoint tensor state remains authoritative for learned raw
  `alibi_slope_logits`.
- ALiBi adds no chromosome-name identity rule on its own; existing row-identity
  policy remains tied to chromosome-row-indexed learned state.

Validation:

- `tests/test_alibi_lifecycle_phase10.py`: 52 passed.
- ALiBi focused command passed 134 tests.
- Training/reconstruction command passed 221 tests.
- Explanation command passed 159 tests.
- Cross-strategy regression command passed 369 tests.
- Full test suite passed 1342 tests, 1 skipped, with 6 existing non-failing
  warnings.

Known limitations:

- No real training, dataset generation, checkpoint migration, or production
  explanation run was executed.
- Downstream validation and benchmark scripts remain later-roadmap work.

## Phase 12B1 - Positional Strategy Identity and Predictive Performance

Goal:

Implement the first downstream benchmark layer without changing training,
explanation, model, reconstruction, encoding, data, null-baseline, ranking, or
attribution production paths.

Exact files changed:

- `scripts/position_benchmark_metadata.py`
- `scripts/ablation_compare.py`
- `tests/test_position_benchmark_metadata.py`
- `tests/test_position_benchmark_performance.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Implementation:

- Added pure, read-only positional benchmark metadata helpers. The new module
  imports no Torch code, loads no checkpoints, constructs no models, and reads
  only saved YAML/config dictionaries supplied by callers.
- Separated position strategy identity from predictive comparison context.
  Strategy identity describes only the configured positional architecture;
  comparison context describes non-positional training and dataset conditions
  that must match before predictive metrics can be interpreted as a positional
  comparison.
- Defined a canonical strategy payload with `strategy_schema_version=1`,
  `preset`, whitelisted `absolute`, whitelisted `relative`, and whitelisted
  `chromosome` sections.
- Used committed serialized field names in the canonical payload, including
  `absolute.dim`, `absolute.coordinate_scale`, `absolute.max_wavelength`,
  `absolute.bin_size_bp`, `relative.num_buckets`,
  `relative.max_distance_bp`, `relative.rope_coordinate_scale`,
  `relative.rope_base`, `relative.alibi_distance_function`, and
  `relative.alibi_distance_scale`.
- Ignored legitimate non-strategy serialized extensions such as
  `chromosome.mapping` and `absolute.binning` when computing strategy identity.
- Added deterministic canonical JSON serialization with `sort_keys=True`,
  compact separators, and `allow_nan=False`, then SHA-256 hashing over UTF-8.
- Exposed a human-readable strategy name and stable strategy ID of the form
  `<readable-name>__<first-12-hex-of-full-hash>`, while preserving the full
  hash separately.
- Required authoritative new-schema position metadata for position benchmark
  mode: `position_encoding` must be present and
  `position_encoding_execution.resolved_config_applied_to_model` must be true
  with the resolved-config execution source.
- Added predictive comparison-context extraction with required fields for
  annotation level, content width, dataset mapping identity, randomization,
  split protocol, non-positional model settings, training settings, chunking,
  and recorded data/covariate provenance.
- Made missing required comparison-context fields compatibility failures rather
  than silently skipping them. Required fields now reject on absence even when
  every compared run omits the same field; explicit `None` remains a legitimate
  saved value.
- Added lightweight pure-Python type/domain validation for required canonical
  strategy values before hashing, including preset, absolute/relative strategy
  types, chromosome encoding, cross-chromosome policy, positive integer
  strategy widths/distances, positive finite numeric scales, and ALiBi distance
  functions.
- Rejected duplicate run IDs before comparison because run IDs key the
  compatibility diagnostics.
- Allowed `input_dim` to differ across strategies because positional width may
  differ; required `content_dim` to match.
- Preserved parent-run class-weighting scope by comparing the requested
  `class_weighting` policy only. Fold-specific `class_weighting_applied` and
  `class_weighting_pos_weight` validation remains later work.
- Generalized `scripts/ablation_compare.py` with
  `--comparison-axis {level,position}`. The default `level` path preserves the
  historical annotation-ablation behavior and does not require positional
  metadata.
- Preserved the documented direct CLI invocation
  `python scripts/ablation_compare.py ...` as well as package import usage.
- Added explicit `position` mode for strategy-aware predictive performance
  comparison over explicit `--run-dir` inputs only. It rejects `--results-dir`
  discovery to avoid hidden inference from directory names.
- Position mode reuses existing AUC, accuracy, loss, and std-AUC metric
  extraction semantics for `results.yaml` and `cv_results.yaml`.
- Position-mode TSV output is tidy, one row per run, and includes strategy ID,
  readable name, full hash, strategy type columns, metrics, config path, and
  results path.
- Position-mode YAML output records `comparison_axis: position`, metric
  priority, best strategy/run, compatibility report, canonical strategy
  payloads, metrics, and paths.

Runtime behavior:

- Historical level-mode `ablation_compare.py` commands remain the default and
  keep the original TSV header, YAML keys (`best_level`, `best_run_id`,
  `ranking_metric_priority`, `levels`), level sorting, metric priority, and
  permissive config handling.
- Position mode performs no statistical testing, no confidence intervals, no
  ranking-stability analysis, and no attribution-stability analysis.
- No training, explanation, data generation, checkpoint loading, model
  construction, or positional runtime code changed.

Compatibility effects:

- A historical/legacy reference for the new positional benchmark must come from
  a new-schema resolved legacy run. Position mode does not infer a strategy
  from raw CLI flags, directory names, checkpoint state, or incomplete old
  metadata.
- Dataset mapping hashes are compared, but they identify gene/chromosome
  mappings only. They do not cryptographically identify the sample cohort,
  phenotype contents, VCF contents, or preprocessed-data contents. Phase 12B1
  therefore also compares the available data-source and covariate provenance
  paths, while leaving stronger cohort hashing to later work.
- L3 is the operational rich-content primary benchmark level. L0 is the
  dosage-only sensitivity analysis level. L4 remains a compatibility
  placeholder identical to L3 and is not a preferred primary benchmark level.

Validation:

- New Phase 12B1 benchmark tests passed 45 tests.
- Explicit direct CLI smoke test `scripts/ablation_compare.py --help` passed.
- Relevant existing training/config focused command passed 192 tests.
- Full test suite passed 1387 tests, 1 skipped, with 6 existing non-failing
  warnings.
- `compileall` passed for the changed benchmark script/module/test files.
- Ruff, Black, and isort passed on the new Python files.
- `scripts/ablation_compare.py` was compared against its committed baseline:
  Ruff current count is lower than baseline, isort now passes on the current
  file, and remaining Black findings match pre-existing legacy formatting
  debt.
- Forbidden training, explanation, model, reconstruction, encoding, data,
  null-baseline, ranking, and attribution production paths were not modified.

Known limitations:

- Ranking stability and attribution stability remain unimplemented.
- Null-baseline positional config propagation remains a later phase.
- Fold-specific applied class-weighting compatibility is not validated in this
  phase.
- Existing dataset metadata does not prove identical cohorts beyond the
  recorded mapping hashes and data-source/covariate provenance.

## Phase 12B2 - Position-Encoding Ranking Stability

Goal:

- Add read-only, strategy-aware ranking-stability comparison for completed
  positional benchmark explanation runs.
- Preserve the historical annotation-level ablation ranking comparison as the
  default behavior.

Files changed:

- `scripts/compare_ablation_rankings.py`
- `scripts/position_benchmark_metadata.py`
- `tests/test_compare_ablation_rankings.py`
- `tests/test_position_benchmark_rankings.py`
- `documentation/appendices/position-encoding-implementation-log.md`

Decisions and reasoning:

- Added `--comparison-axis {level,position}` to
  `scripts/compare_ablation_rankings.py`, defaulting to `level`.
- Moved the historical annotation-level CLI execution into a private
  level-mode runner while preserving the existing ranking-file discovery,
  score semantics, warning behavior, TSV headers, YAML keys, and output
  filenames.
- Added explicit repeated `--position-run RUN_ID CONFIG_YAML RANKING_CSV
  ANALYSIS_METADATA_YAML` input for position mode. Position strategy identity
  is always derived from the authoritative training `config.yaml`; ranking
  CSVs, analysis metadata, paths, and run IDs are not used to infer strategy.
- Reused Phase 12B1 training-context helpers for strategy identity and
  non-positional compatibility. `input_dim` may differ across positional
  strategies, while `content_dim` and other non-positional training/data
  context fields must match.
- Added explanation-context extraction and compatibility checks for content
  Integrated Gradients. Position ranking mode requires raw content-space IG
  metadata with observed absolute position held fixed.
- Required raw ranking CSV provenance columns (`resolved_ig_mode`,
  `attribution_feature_space`, and `variant_score_aggregation`) in position
  mode and checked their constant per-file values against
  `analysis_metadata.integrated_gradients`.
- Kept full positional strategy identity exclusively config-derived. Ranking
  provenance fields validate attribution-mode compatibility only; they do not
  replace or augment config-derived strategy identity.
- Enforced per-run content attribution width:
  `integrated_gradients.attribution_width` must equal both serialized
  `content_dim` and IG `content_dim`.
- Restricted position-mode score columns to raw explanation ranking columns:
  `rank`, `mean_attribution`, and `max_attribution`. Calibrated, null-derived,
  bootstrap-derived, or chromosome-corrected ranking columns are rejected until
  provenance validation is added in Phase 12C.
- Required an exact variant universe across compared position runs before
  computing top-k Jaccard values.
- Made position-mode variant keys strict: use non-empty `variant_id` when
  present, otherwise require all of chromosome, position, and gene ID. The
  historical chromosome-position-only fallback remains available only in
  level mode.
- Made position-mode tie-breaking independent of CSV row order by sorting on
  score first and `variant_id` second.
- Added tidy position-mode outputs: pairwise top-k Jaccard rows with strategy
  IDs and one row per strategy-specific variant/other-run comparison.
- Rejected mixed-mode CLI inputs: position mode rejects `--ranking-dir` and
  `--rankings`, while level mode rejects `--position-run`.
- Created parent directories independently for all three position-mode outputs:
  comparison YAML, Jaccard TSV, and strategy-specific TSV.

Runtime behavior:

- Historical level mode remains the default and does not require positional
  config or analysis metadata.
- Position mode performs no training, explanation, checkpoint loading, model
  construction, statistical testing, null-baseline integration, bootstrap
  integration, or attribution-magnitude stability analysis.
- Position mode compares completed raw explanation ranking CSVs only.
- Position mode refuses ranking CSVs whose available attribution provenance
  does not match analysis metadata, including legacy-IG ranking CSVs paired
  with content-IG metadata.

Compatibility effects:

- Existing `compare_ablation_rankings.py` level-mode commands keep the
  historical default `z_attribution` behavior through the level runner.
- Direct script invocation (`python scripts/compare_ablation_rankings.py
  --help`) and package import both work after importing Phase 12B1 helpers.
- Position mode refuses ambiguous or non-comparable inputs rather than
  producing a partially comparable ranking report.
- Independent output directories are supported for all position-mode outputs.

Validation:

- Ranking-focused command passed 82 tests:
  `tests/test_compare_ablation_rankings.py` and
  `tests/test_position_benchmark_rankings.py`.
- Direct CLI help command passed:
  `/home/simostocco/miniforge3/envs/sieve-posenc/bin/python
  scripts/compare_ablation_rankings.py --help`.
- Phase 12B1 companion metadata/performance command passed 45 tests.
- Full-suite regression coverage completed via four non-overlapping
  test-file shards, not a monolithic full-suite run:
  - shard 1 rerun: 367 passed, 1 warning in 44.30 seconds. The first shard-1
    attempt had one load-sensitive timeout in
    `tests/test_phase3_explain.py::test_validate_epistasis_no_data`; that test
    passed alone and shard 1 passed on rerun.
  - shard 2: 313 passed in 6.12 seconds.
  - shard 3: 331 passed, 1 skipped, 5 warnings in 98.48 seconds.
  - shard 4: 443 passed in 144.95 seconds.
  - summed shard coverage: 1454 passed, 1 skipped, 6 warnings.
- `compileall` passed for the changed ranking script/module/test files.
- `git diff --check` passed.
- Ruff, Black check, and isort passed on
  `scripts/position_benchmark_metadata.py` and
  `tests/test_position_benchmark_rankings.py`.
- Legacy modified files were compared against committed baselines:
  `scripts/compare_ablation_rankings.py` and
  `tests/test_compare_ablation_rankings.py`. Ruff current findings decreased
  from 71 to 70. Baseline and current Black checks both report the same two
  legacy files would be reformatted. Baseline isort reported both legacy files;
  current isort reports only `tests/test_compare_ablation_rankings.py`.

Known limitations:

- Position-mode ranking stability compares top-k Jaccard and
  strategy-specific high-ranking variants only.
- Calibrated/null-derived ranking comparison remains deferred until Phase 12C
  can validate provenance.
- Attribution magnitude stability is not implemented in this phase.
- Regression coverage was completed in four shards because the prior
  monolithic full-suite run did not complete within the bounded window.

## Next planned phase

Phase 12B3 - Position-Encoding Attribution Stability
