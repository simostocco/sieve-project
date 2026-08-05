# Position-Encoding Architecture Audit

**Repository:** `lescailab/sieve-project`  
**Working branch:** `simostocco/position-encoding-benchmark`  
**Audited baseline commit:** `434fc095ed0de4c2f32c2c9b3f024a0310b8f611` (`v1.3.0`)  
**Purpose:** prepare SIEVE so training, evaluation, explanation, null-model analysis, and downstream validation can run with a selectable positional-encoding strategy.

## 1. Scope and constraints

This audit is read-only. It describes the current architecture and the changes that will eventually be required; it does not define an implementation patch yet.

Current constraints:

- GPU training is postponed until the remote server becomes available.
- No real training runs will be performed during this implementation month.
- No synthetic cohort or smoke-training pipeline will be added.
- Focused unit tests using small deterministic tensors are allowed and required for shape, masking, mathematical, serialization, and gradient checks.
- Existing behavior must remain available through an explicit `legacy` configuration.
- Positional settings must be selected during training and restored automatically during explanation and downstream analysis.
- Changes must be split into small, reviewable commits.

## 2. Baseline repository state

The audited code baseline is commit `434fc095ed0de4c2f32c2c9b3f024a0310b8f611`.

At the time of the independent audit verification, the working tree contained only the two untracked documentation files created for this project:

- `AGENTS.md`
- `documentation/appendices/position-encoding-audit.md`

Those files do not alter the audited source behavior. A repository-wide Ruff run reported 834 pre-existing violations, 653 automatically fixable. Therefore:

- repository-wide Ruff success is not a valid acceptance criterion for this branch;
- unrelated lint errors must not be fixed as part of positional-encoding work;
- each task should lint only the files changed by that task;
- full-repository lint output is baseline technical debt, not a regression introduced by this project.

The baseline test summary should be recorded separately if it has not already been saved.

## 3. Current end-to-end execution flow

```text
preprocessed SampleVariants / VariantRecord objects
    |
    v
build_variant_tensor / encode_variants
    |
    +--> variant_features  [B, V, input_dim]
    +--> positions         [B, V]
    +--> gene_ids          [B, V]
    +--> chrom_ids         [B, V]
    +--> mask              [B, V]
    |
    v
ChunkedVariantDataset and collate_chunks
    |
    v
ChunkedSIEVEModel
    |
    v
SIEVE.forward
    |
    +--> VariantEncoder
    +--> MultiLayerAttention
    +--> GeneAggregator
    +--> classifier
    |
    v
phenotype logits
```

Explanation follows a related path:

```text
saved config + checkpoint
    |
    v
scripts/explain.py reconstructs the model
    |
    v
IntegratedGradientsExplainer
    |
    +--> differentiable input: variant_features
    +--> fixed additional arguments:
         positions, gene_ids, mask, covariates, chrom_ids
    |
    v
feature attributions
    |
    v
per-variant score, metadata, rankings
```

## 4. Current positional-information channels

Position is not implemented as one isolated mechanism. It reaches the prediction through three separate channels.

### 4.1 Fixed absolute sinusoidal features

Files involved:

- `src/encoding/positional.py`
- `src/encoding/sparse_tensor.py`
- `src/encoding/levels.py`

For annotation levels L1-L4, `build_variant_tensor` computes a fixed 64-dimensional sinusoidal representation from each genomic coordinate before the model forward pass. That representation is concatenated to dosage and annotation features.

```text
L1 feature = [dosage ; sinusoidal_position_64]
L2 feature = [dosage ; sinusoidal_position_64 ; consequence_4]
L3 feature = [dosage ; sinusoidal_position_64 ; consequence_4 ; scores_2]
L4 feature = current L3-compatible representation
```

Current hard-coded dimensions:

| Level | Current feature dimension | Composition |
|---|---:|---|
| L0 | 1 | dosage |
| L1 | 65 | dosage + 64 positional coordinates |
| L2 | 69 | L1 + 4 consequence coordinates |
| L3 | 71 | L2 + 2 prediction scores |
| L4 | 71 | currently compatible with L3 |

Consequences:

- preprocessing decides part of the neural architecture;
- input dimension depends on the positional strategy;
- the sinusoidal strategy cannot be disabled cleanly without changing preprocessing and model input dimensions;
- learned absolute embeddings do not naturally belong in a preprocessing-only representation because they are trainable parameters;
- RoPE and ALiBi do not naturally fit input-feature concatenation.

### 4.2 Learned relative-distance bucket bias

Files involved:

- `src/encoding/positional.py`
- `src/models/attention.py`

For every pair of variants, the attention module computes a relative-distance bucket. A learned embedding table maps each bucket to one scalar bias per attention head.

For sample `b`, head `h`, query variant `i`, and key variant `j`:

\[
S_{bhij}
=
\frac{Q_{bhi}^{\top}K_{bhj}}{\sqrt{d_h}}
+
\beta_{bhij}
\]

where:

\[
\beta_{bhij}
=
W_{\mathrm{bucket}}[\operatorname{bucket}(p_{bi}-p_{bj}),h]
\]

The bias tensor has shape `[B, H, V, V]` and is added before softmax.

When `chrom_ids` are passed to attention, cross-chromosome pairs use a dedicated bucket instead of coordinate subtraction. When `chrom_ids` are absent, the current fallback is chromosome-blind and buckets the numerical coordinate difference directly.

This relative mechanism remains active at L0 even though L0 does not contain sinusoidal input features.

The module-level documentation in `src/encoding/positional.py` says sinusoidal and relative bucket encodings are not used simultaneously, but the actual L1-L4 execution path does use both. The implementation is authoritative; the conflicting docstring should be corrected in a later focused task.

### 4.3 Learned chromosome embedding

File involved:

- `src/models/attention.py`

A learned chromosome embedding exists only when the model is constructed with `num_chromosomes > 0`. When that module exists and chromosome IDs are supplied, it is added to the latent variant representation before Q/K/V projection:

\[
\widetilde X_v = X_v + E_{\mathrm{chrom}}[c_v]
\]

followed by:

\[
Q=\widetilde XW_Q,\qquad K=\widetilde XW_K,\qquad V=\widetilde XW_V
\]

Current behavior differs by construction path:

- cross-validation training passes `num_chromosomes=dataset.num_chromosomes`, so the learned chromosome embedding is present;
- the current single-split training path omits `num_chromosomes`, so no chromosome embedding module is created;
- both paths can still pass `chrom_ids` to attention for chromosome-aware bucket routing.

Therefore L0 is not a true position-free model in either path because relative position remains active. In the CV path it additionally uses learned chromosome identity. Gene identity also remains available downstream through gene aggregation.

## 5. Attention tensor shapes

Let:

- `B`: batch size;
- `V`: variants in the current padded chunk;
- `D`: latent dimension;
- `H`: attention heads;
- `Dh = D / H`: head dimension.

```text
variant_features         [B, V, input_dim]
variant embeddings       [B, V, D]
positions                [B, V]
chrom_ids                [B, V]
gene_ids                 [B, V]
mask                     [B, V]

projected Q/K/V           [B, V, D]
multi-head Q/K/V          [B, H, V, Dh]
content scores            [B, H, V, V]
relative-position bias    [B, H, V, V]
attention probabilities   [B, H, V, V]
head outputs              [B, H, V, Dh]
concatenated output       [B, V, D]
```

The implementation requires `D % H == 0`. RoPE will later act on Q and K after they have shape `[B, H, V, Dh]`; its rotary dimension must be even.

## 6. Chunking boundary

File involved:

- `src/encoding/chunked_dataset.py`

Samples are divided into chunks before the base SIEVE model is applied. Training and `ChunkedVariantDataset` default to a chunk size of 3,000 variants with zero overlap. The current explanation path instead limits explanation chunks to `min(args.max_variants, 2000)`, with `max_variants` defaulting to 2,000.

Benchmark implications:

- attention interactions are restricted to variants in the same chunk;
- positional encodings do not recover cross-chunk interactions;
- chunk size and overlap must remain fixed across positional conditions;
- chunking must not be refactored in the same commit as positional encoding;
- chunking must be documented as a fixed benchmark constraint.

## 7. Model construction and reconstruction call sites

Primary construction path:

- `scripts/train.py`
  - determines `input_dim` through `get_feature_dimension(annotation_level)`;
  - constructs the base SIEVE model;
  - wraps it in `ChunkedSIEVEModel`;
  - trains and saves experiment outputs;
  - currently passes `num_chromosomes` in the CV path but omits it in the single-split path, producing different chromosome-embedding architectures.

Primary reconstruction paths:

- `scripts/explain.py`
  - loads `config.yaml` and a checkpoint;
  - reconstructs the base or chunked model;
  - loads state through `load_state_dict_with_legacy_upgrade`;
  - runs Integrated Gradients and attention analysis.

- `scripts/validate_epistasis.py`
  - reconstructs a model from config and checkpoint;
  - must eventually restore the same positional strategy.

Additional architecture/checkpoint and positional-behavior consumers discovered by the audit:

- `src/encoding/__init__.py`
- `src/explain/attention_analysis.py`
- `src/explain/counterfactual_epistasis.py`
- `scripts/plot_detailed_architecture.py`
- `scripts/render_model_architecture.py`
- `scripts/run_null_baseline_analysis.sh`
- `scripts/ablation_compare.py`
- `utilities/demos/test_encoding_pipeline.py`
- `utilities/demos/test_training_pipeline.py`
- `tests/test_phase3_explain.py`
- `tests/test_chrom_aware_attention.py`
- `tests/test_ig_covariates.py`
- `tests/test_sex_covariate.py`
- `tests/test_classifier_type_switch.py`
- `tests/test_chunked_sieve.py`
- `tests/test_class_weighting.py`
- downstream scripts that consume explanation and ranking outputs

Any configuration added to training must be propagated to every code path that reconstructs a trained model. Adding CLI arguments only to `scripts/train.py` is insufficient.

## 8. Configuration and checkpoint lifecycle

```text
CLI arguments
    |
    v
training configuration / model constructor
    |
    v
config.yaml in experiment output
    |
    +--> checkpoint state_dict
    |
    v
scripts/explain.py or validation script
    |
    v
model reconstruction
    |
    v
load_state_dict_with_legacy_upgrade
```

Current behavior:

- `config.yaml` is initially based on `vars(args)` before data/model construction;
- current configs do not reliably serialize resolved `input_dim`, `num_genes`, or `num_chromosomes`;
- explanation and validation recompute some architecture values from the supplied data;
- this means the current config is not by itself a complete architecture specification.

Required future behavior:

- positional strategy and method-specific parameters are written to the experiment config;
- explanation and validation load those values automatically;
- users do not repeat positional CLI arguments during explanation;
- the saved config is sufficient to reconstruct tensor shapes and parameter names;
- old configs lacking positional fields resolve to historical behavior;
- old checkpoints remain loadable where technically possible;
- incompatible configuration/checkpoint combinations fail with a clear error.

Architecture-rendering scripts infer some dimensions directly from state-dict shapes. In particular:

- `render_model_architecture.py` reads `position_bias.weight.shape[0]`; this row count currently includes the extra cross-chromosome bucket, so it is not identical to the configured ordinary-bucket count;
- `plot_detailed_architecture.py` reconstructs a model with strict `load_state_dict`, does not use the legacy-upgrade helper, and currently creates dummy variant features with hard-coded width 71 even after resolving another `input_dim`.

These tools may need updates for new positional parameters, but they must not drive the core model design.

## 9. Current Integrated Gradients behavior

File involved:

- `src/explain/gradients.py`

The current Captum call treats `variant_features` as the differentiable input. These values remain fixed along the integration path:

- positions;
- chromosome IDs;
- gene IDs;
- masks;
- covariates.

With a zero feature baseline, the current object is:

\[
IG(\text{variant features};0\mid
\text{positions, chromosomes, genes, mask, covariates fixed})
\]

Current asymmetry:

- sinusoidal absolute position is inside `variant_features` at L1-L4 and receives direct feature attribution;
- relative bucket bias, chromosome embedding, RoPE, and ALiBi live inside the model and do not receive direct input-feature attribution.

Target benchmark definition:

- primary variant ranking uses IG over dosage and non-positional annotations only;
- position remains active and changes the gradients;
- every positional strategy answers the same question: “How important is this variant's content under this positional architecture?”;
- position-specific reliance is a separate diagnostic, not part of the primary variant score.

The refactor must not silently change attribution semantics. The selected IG mode must be explicit in output metadata.

A second current implementation detail must be preserved in the audit: `scripts/explain.py` runs a manual chunk-level IG loop and converts feature attributions to a per-variant score using the L2 norm directly. This path is distinct from the configurable aggregation available in `IntegratedGradientsExplainer.attribute_batch`.

## 10. Why a refactor is required

The required change is organizational, not cosmetic.

Current responsibility split:

```text
preprocessing:
    fixed absolute sinusoidal representation
    annotation-level dimensions

attention module:
    chromosome embedding
    relative-distance bucket bias

training/explanation:
    assume historical feature dimensions
```

Target responsibility split:

```text
preprocessing:
    dosage and non-positional annotations
    raw genomic positions
    chromosome IDs
    gene IDs
    masks

model:
    selected absolute-position encoder
    selected Q/K transformation
    selected attention-logit bias
    selected chromosome behavior

configuration:
    records selected strategies and parameters
```

Without this separation:

- one CLI flag cannot safely switch every relevant code path;
- feature dimensions differ for architectural rather than biological reasons;
- learned embeddings cannot be trained naturally;
- RoPE and ALiBi require unrelated special cases;
- Integrated Gradients compares different input spaces;
- explanation may reconstruct a model different from training;
- a true no-position control cannot be expressed.

## 11. Target conceptual architecture

### 11.1 Content representation

Input:

```text
dosage
consequence representation
SIFT / PolyPhen and other non-positional annotations
```

Output:

```text
content embedding H  [B, V, D]
```

### 11.2 Absolute-position mechanism

Initial strategies:

- `none`
- `sinusoidal`
- `learned_binned`

Later genomics-aware absolute encodings can use the same interface.

### 11.3 Relative-attention mechanism

Initial strategies:

- `none`
- `t5_bucket`
- `rope`
- `alibi_fixed`
- `alibi_learned`

A relative mechanism may transform Q/K, add an attention-logit bias, or both. Chromosome handling must be explicit.

## 12. Historical behavior that must be preserved

Historical behavior is currently path-dependent and must be defined carefully before introducing the `legacy` preset.

### L0 historical behavior

```text
absolute sinusoidal input: disabled
relative bucket bias: enabled
cross-chromosome dedicated bucket: enabled only when chrom_ids are passed
chromosome embedding:
    CV training: enabled because num_chromosomes is supplied
    single-split training: disabled because num_chromosomes is omitted
```

### L1-L4 historical behavior

```text
absolute sinusoidal input: enabled, 64 dimensions
relative bucket bias: enabled
cross-chromosome dedicated bucket: enabled only when chrom_ids are passed
chromosome embedding:
    CV training: enabled because num_chromosomes is supplied
    single-split training: disabled because num_chromosomes is omitted
```

The CLI-contract phase must decide whether `legacy` preserves this path-dependent discrepancy or normalizes all new runs to one explicit chromosome policy while retaining a separate compatibility path for old checkpoints.

Legacy invariants:

1. Existing CLI commands without new flags keep historical behavior.
2. Existing annotation-level feature semantics remain available.
3. Legacy outputs remain numerically equivalent after refactoring within an agreed tolerance.
4. Padding behavior is unchanged.
5. Cross-chromosome bucket behavior is unchanged.
6. Gene aggregation, chunk aggregation, classifier selection, covariates, loss, and training defaults are unchanged.
7. Existing configs without positional fields resolve to legacy mode.
8. New checkpoints/configs contain complete positional configuration.
9. Explanation reconstructs the exact training architecture.
10. Integrated Gradients compatibility behavior is explicit and documented.

## 13. Files expected to change

### Core encoding and model files

| File | Current responsibility | Expected future responsibility |
|---|---|---|
| `src/encoding/levels.py` | feature dimensions include sinusoidal PE | content dimensions plus legacy compatibility |
| `src/encoding/sparse_tensor.py` | computes sinusoidal PE | produces content features and positional metadata |
| `src/encoding/positional.py` | sinusoidal function and relative bucketing; conflicting module docstring | reusable utilities and/or strategy components; corrected documentation |
| `src/models/encoder.py` | maps current feature tensor to latent representation | maps content or legacy features according to migration design |
| `src/models/attention.py` | chromosome embedding + T5 bias | delegates Q/K transforms and bias generation to configured strategies |
| `src/models/sieve.py` | constructs fixed architecture | constructs configured positional components |
| `src/models/chunked_sieve.py` | chunk wrapper | remains behaviorally stable; configuration plumbing only if needed |

### Training, reconstruction, and explanation

| File | Required change |
|---|---|
| `scripts/train.py` | add and validate positional CLI options; save resolved config |
| `scripts/explain.py` | restore positional config; expose explicit IG mode; document/resolve explanation chunk-size behavior |
| `scripts/validate_epistasis.py` | restore positional config |
| `src/explain/gradients.py` | standardize content-only IG and metadata |
| `src/training/trainer.py` | likely no mathematical change; checkpoint metadata may need extension |
| `scripts/plot_detailed_architecture.py` | understand configured architecture or fail clearly |
| `scripts/render_model_architecture.py` | understand configured architecture or fail clearly |

### Tests

Existing tests likely affected:

- `tests/test_chrom_aware_attention.py`
- `tests/test_phase3_explain.py`
- `tests/test_fold_config_saving.py`
- `tests/test_validate_epistasis_chunking.py`
- `tests/test_classifier_type_switch.py`
- `tests/test_ig_covariates.py`
- `tests/test_sex_covariate.py`
- `tests/test_chunked_sieve.py`
- `tests/test_class_weighting.py`

Likely new focused tests:

- positional configuration resolution;
- legacy preset resolution by annotation level;
- no-position configuration;
- strategy tensor shapes;
- RoPE identity and relative-rotation properties;
- ALiBi monotonicity and slope constraints;
- learned-bin boundaries;
- cross-chromosome policy;
- config/checkpoint round trip;
- content-only Integrated Gradients dimensions;
- old-config fallback to legacy.

## 14. Non-goals

Out of scope until separately approved:

- changing chunk size or chunk aggregation;
- replacing gene aggregation;
- changing the classifier;
- changing training hyperparameters;
- adding new biological annotations;
- changing null-model methodology;
- adding a synthetic cohort generator;
- running a smoke-training benchmark;
- resolving repository-wide Ruff debt;
- formatting unrelated files;
- optimizing GPU performance before correctness is established.

## 15. Development and review workflow

Every change follows this sequence:

1. Define exact behavior and invariants.
2. Give Codex one narrow task.
3. Codex inspects relevant files and all call sites.
4. Codex proposes a plan and stops before editing.
5. After approval, Codex modifies only agreed files.
6. Review `git diff`.
7. Run focused tests and changed-file static checks.
8. Commit the isolated change.

Recommended checks after every implementation task:

```bash
git diff --check
git status --short
python -m pytest <focused test files>
python -m ruff check <changed Python files>
python -m black --check <changed Python files>
python -m isort --check-only <changed Python files>
```

Do not use repository-wide Ruff success as a gate until pre-existing lint debt is addressed separately.

## 16. Verified unresolved decisions for the CLI-contract phase

The independent repository verification identified these concrete decisions:

- Does `legacy` reproduce the current CV/single-split chromosome-embedding discrepancy, or does it define one normalized new-run behavior plus a separate old-checkpoint compatibility rule?
- Should explanation intentionally keep its current 2,000-variant cap, or inherit the training chunk size recorded in config?
- Should `num_chromosomes`, chromosome-index mapping, gene index size, and gene ordering be serialized rather than recomputed?
- Should primary content-only IG be implemented by slicing historical feature tensors or by completing the model-level content/position separation?
- What explicit cross-chromosome behavior should RoPE and ALiBi use?
- Should rendering scripts be authoritative config readers, state-dict inspectors, or best-effort tools that fail clearly on unknown strategies?

## 17. Decisions for the CLI-contract phase

Before implementation, Phase 3 must settle:

1. whether the user-facing API exposes a `legacy` preset plus low-level custom options;
2. whether chromosome encoding is an independent flag or part of each preset;
3. stable names for saved configuration fields;
4. valid and invalid strategy combinations;
5. absolute-position fusion by concatenation or addition;
6. coordinate scaling for sinusoidal encoding and RoPE;
7. bin width and out-of-range policy for learned embeddings;
8. cross-chromosome behavior for RoPE and ALiBi;
9. fallback behavior for old configs without positional fields;
10. whether legacy feature-level IG remains available alongside the new content-only IG mode.

No source-code change should begin until the first CLI/configuration contract is accepted.
