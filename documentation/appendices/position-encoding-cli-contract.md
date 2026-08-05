# Position-Encoding CLI and Configuration Contract

**Repository:** `lescailab/sieve-project`  
**Working branch:** `simostocco/position-encoding-benchmark`  
**Baseline source commit:** `434fc095ed0de4c2f32c2c9b3f024a0310b8f611`  
**Audit commit:** `52d6f39`  
**Contract schema:** position encoding v1  
**Status:** revised after read-only repository review; implementation must not begin until this document is reviewed and accepted.

## 1. Purpose

This document defines the user-facing and serialized contract for selecting positional encoding in SIEVE.

The contract covers:

- training CLI arguments;
- resolved model configuration;
- checkpoint/config reconstruction;
- old-checkpoint compatibility;
- valid and invalid method combinations;
- chromosome-aware behavior;
- explanation-time Integrated Gradients mode;
- downstream scripts that reconstruct or compare models.

The implementation must make positional encoding selectable without changing unrelated training, chunking, aggregation, classifier, covariate, loss, or optimization behavior.

## 2. Design principles

### 2.1 Separate biological content from position

The implementation must represent these as separate inputs:

```text
content_features
positions
chrom_ids
gene_ids
mask
covariates
```

`content_features` contains dosage and non-positional annotations only.

Position is transformed inside the model according to the resolved positional configuration.

### 2.2 Separate three positional decisions

The configuration distinguishes:

1. **Absolute-position encoding**
   - `none`
   - `sinusoidal`
   - `learned_binned`

2. **Relative-attention encoding**
   - `none`
   - `t5_bucket`
   - `rope`
   - `alibi_fixed`
   - `alibi_learned`

3. **Chromosome embedding**
   - `none`
   - `learned`

Cross-chromosome attention policy is separate from chromosome embedding. The model may use chromosome IDs for pair routing even when no learned chromosome embedding is added.

### 2.3 Configuration is resolved once

Training resolves the CLI into one complete configuration before constructing the model.

The resolved configuration is:

- printed before training;
- saved in `config.yaml`;
- copied into fold-specific configs where those exist;
- included in checkpoint metadata where practical;
- loaded by explanation and validation;
- treated as authoritative for all new-schema runs.

### 2.4 New-run behavior and old-checkpoint compatibility are different

New training runs use an explicit, internally consistent configuration.

Old checkpoints and configs are handled by a compatibility resolver that infers historical architecture details. Compatibility inference is not exposed as a new-training preset.

## 3. Training CLI

### 3.1 Preset selector

```text
--position-preset {legacy,custom}
```

Default:

```text
legacy
```

Rules:

- `legacy` is the default for new training runs.
- `custom` enables explicit low-level strategy selection.
- When `legacy` is selected, explicitly supplied custom strategy flags are rejected rather than silently ignored.
- The compatibility loader for old checkpoints is internal and is not a valid training preset.

### 3.2 Absolute-position strategy

```text
--absolute-position-encoding {none,sinusoidal,learned_binned}
```

Rules:

- required when `--position-preset custom`;
- forbidden when `--position-preset legacy`;
- valid for all annotation levels, including L0;
- independent of the relative-position strategy.

### 3.3 Relative-position strategy

```text
--relative-position-encoding {none,t5_bucket,rope,alibi_fixed,alibi_learned}
```

Rules:

- required when `--position-preset custom`;
- forbidden when `--position-preset legacy`;
- independent of the absolute-position strategy.

### 3.4 Chromosome embedding

```text
--chromosome-encoding {none,learned}
```

Rules:

- required when `--position-preset custom`;
- forbidden when `--position-preset legacy`;
- controls only the learned chromosome embedding added before Q/K/V projection;
- does not disable use of `chrom_ids` for same-chromosome/cross-chromosome routing.

### 3.5 Cross-chromosome policy

```text
--cross-chromosome-policy {separate,mask}
```

Default for `custom`:

```text
separate
```

Meaning:

- `separate`
  - with `t5_bucket`, one dedicated learned cross-chromosome bucket is used;
  - with `rope`, same-chromosome scores use rotated Q/K, while cross-chromosome scores use unrotated Q/K plus a learned per-head cross-chromosome bias;
  - with `alibi_fixed`, slopes remain fixed but the cross-chromosome term is a learned per-head bias; this method is therefore “fixed slopes plus learned cross-chromosome bias”;
  - with `alibi_learned`, both positive slopes and the cross-chromosome bias are learned;
  - with relative `none`, `separate` is a no-op and adds no pairwise positional term.

- `mask`
  - cross-chromosome attention logits are masked before softmax;
  - this applies to every relative strategy, including `none`;
  - each valid query retains its self-key, so the combined padding/chromosome mask cannot remove every valid key.

Chromosome-ID requirements:

- `mask` always requires `chrom_ids`;
- `separate` requires `chrom_ids` for `t5_bucket`, `rope`, and both ALiBi variants;
- relative `none` with `separate` requires `chrom_ids` only when `chromosome_encoding=learned`.

The implementation must never subtract coordinates from different chromosomes and interpret the result as physical genomic distance.

### 3.6 Shared absolute-position dimension

```text
--position-dim INT
```

Default for applicable custom absolute methods:

```text
64
```

Rules:

- applies to `sinusoidal` and `learned_binned`;
- must be positive;
- must be even for `sinusoidal`;
- forbidden when absolute encoding is `none`;
- legacy resolves to 64 for L1-L4 and no absolute block for L0.

### 3.7 Sinusoidal options

```text
--sinusoidal-coordinate-scale FLOAT
--sinusoidal-max-wavelength FLOAT
```

Custom defaults:

```text
sinusoidal_coordinate_scale: 1.0
sinusoidal_max_wavelength: 10000.0
```

Definition:

\[
u_v = \frac{p_v}{s_{\mathrm{sin}}}
\]

The sinusoidal function is evaluated at `u_v`.

Rules:

- both values must be positive;
- valid only with `absolute_position_encoding=sinusoidal`;
- legacy L1-L4 resolves to the current historical implementation:
  - dimension 64;
  - raw base-pair coordinate scale 1.0;
  - current maximum wavelength 10000.0.

### 3.8 Learned-binned options

```text
--position-bin-size INT
```

Default:

```text
10000
```

Definition:

\[
b_v = \left\lfloor \frac{p_v}{\Delta} \right\rfloor
\]

where `Delta` is `position_bin_size` in base pairs.

Rules:

- must be a positive integer;
- valid only with `absolute_position_encoding=learned_binned`;
- `position_dim` is the embedding width; it does not change the number or layout of bins;
- tables are chromosome-aware;
- table sizes are derived from a new versioned chromosome-length metadata table for each supported genome build;
- introducing that metadata table is an explicit implementation addition, not existing repository behavior;
- a coordinate outside the declared genome-build range raises a clear error;
- coordinates are not silently clipped or wrapped;
- padded variants are neutralized by the existing boolean mask rather than by changing the historical chromosome-ID mapping.

### 3.9 T5-style bucket options

```text
--num-position-buckets INT
--max-position-distance INT
```

Defaults:

```text
num_position_buckets: 32
max_position_distance: 100000
```

Rules:

- valid only with `relative_position_encoding=t5_bucket`;
- `num_position_buckets` counts ordinary same-chromosome buckets;
- `num_position_buckets` must be even and at least 4;
- the current bidirectional implementation defines `max_exact = num_position_buckets // 4`;
- `max_position_distance` must be strictly greater than `max_exact` so the logarithmic denominator is valid;
- with cross-chromosome policy `separate`, one additional learned row is allocated for the cross-chromosome bucket;
- saved config must distinguish configured ordinary buckets from total embedding rows;
- bucket behavior remains bidirectional.

### 3.10 RoPE options

```text
--rope-coordinate-scale FLOAT
--rope-base FLOAT
```

Defaults:

```text
rope_coordinate_scale: 10000.0
rope_base: 10000.0
```

Definition:

\[
u_v = \frac{p_v}{s_{\mathrm{rope}}}
\]

RoPE rotates the full Q and K head dimension using `u_v`.

Rules:

- valid only with `relative_position_encoding=rope`;
- both values must be positive;
- `latent_dim % num_heads == 0`;
- `head_dim = latent_dim / num_heads` must be even;
- value vectors are not rotated;
- when chromosome embedding is enabled, it is added before Q/K/V projection for both score paths;
- same-chromosome pairs use rotated Q/K scores;
- with `separate`, cross-chromosome pairs use the unrotated Q/K computed from the same chromosome-enriched latent inputs, plus a learned per-head cross-chromosome bias;
- with `mask`, cross-chromosome logits are masked.

### 3.11 ALiBi options

```text
--alibi-distance-function {linear,log1p}
--alibi-distance-scale FLOAT
```

Defaults:

```text
alibi_distance_function: log1p
alibi_distance_scale: 10000.0
```

Same-chromosome distance:

\[
d_{ij}
=
\frac{|p_i-p_j|}{s_{\mathrm{alibi}}}
\]

Distance transforms:

```text
linear: d_ij
log1p:  log(1 + d_ij)
```

Bias:

\[
B_{hij} = -m_h \psi(d_{ij})
\]

Rules:

- valid only with `alibi_fixed` or `alibi_learned`;
- scale must be positive;
- fixed ALiBi uses deterministic head-specific positive slopes;
- learned ALiBi initializes from the fixed slopes;
- learned slopes are parameterized to remain strictly positive;
- zero same-chromosome distance gives zero ALiBi penalty;
- with `separate`, cross-chromosome pairs use a learned per-head bias instead of genomic distance;
- therefore `alibi_fixed` means fixed within-chromosome slopes plus a learned cross-chromosome bias;
- with `mask`, cross-chromosome logits are masked.

## 4. Preset resolution

### 4.1 `legacy` preset for new runs

The `legacy` preset represents the intended current SIEVE positional architecture, normalized so CV and single-split training construct the same model.

For L0:

```yaml
absolute:
  type: none
relative:
  type: t5_bucket
  num_buckets: 32
  max_distance_bp: 100000
chromosome:
  encoding: learned
  cross_chromosome_policy: separate
```

For L1-L4:

```yaml
absolute:
  type: sinusoidal
  dim: 64
  fusion: input_concat
  coordinate_scale: 1.0
  max_wavelength: 10000.0
relative:
  type: t5_bucket
  num_buckets: 32
  max_distance_bp: 100000
chromosome:
  encoding: learned
  cross_chromosome_policy: separate
```

Additional rules:

- both CV and single-split training pass the resolved `num_chromosomes`;
- `chrom_ids` are required;
- all existing non-positional training defaults remain unchanged;
- legacy sinusoidal fusion reproduces the historical input-concatenation mathematics;
- legacy T5-style bucket behavior reproduces the current bucket function and learned per-head bias.

### 4.2 `custom` preset

`custom` requires explicit values for:

```text
absolute_position_encoding
relative_position_encoding
chromosome_encoding
```

The resolver fills method-specific defaults only after those three strategies are known.

All absolute/relative combinations are valid unless a mathematical constraint below is violated.

Examples of valid combinations:

| Absolute | Relative | Chromosome | Purpose |
|---|---|---|---|
| none | none | none | true position-free control |
| none | none | learned | chromosome-identity-only control |
| sinusoidal | none | none | absolute-only control |
| none | t5_bucket | none | relative-bucket-only control |
| sinusoidal | t5_bucket | learned | legacy-equivalent architecture |
| learned_binned | none | learned | learned absolute encoding |
| none | rope | learned | RoPE condition |
| none | alibi_fixed | learned | fixed ALiBi condition |
| none | alibi_learned | learned | learned ALiBi condition |
| learned_binned | rope | learned | combined absolute + relative condition |

## 5. Absolute fusion contract

For position block \(P\) and content features \(C\):

```text
absolute none:
    encoder_input = C

absolute sinusoidal or learned_binned:
    encoder_input = concat(C, P, dim=-1)
```

This fusion is named:

```text
input_concat
```

Reasons:

- it permits exact reproduction of the historical sinusoidal input path;
- learned absolute embeddings can use the same fusion location;
- position remains model-controlled rather than preprocessing-controlled;
- Integrated Gradients can differentiate only `C` while holding `P` fixed.

Resolved dimensions:

```text
content_dim = annotation-level non-positional dimension

absolute none:
    input_dim = content_dim

absolute active:
    input_dim = content_dim + position_dim
```

Expected non-positional content dimensions after separation:

| Level | Content dimension |
|---|---:|
| L0 | 1 |
| L1 | 1 |
| L2 | 5 |
| L3 | 7 |
| L4 | 7 unless L4 annotations are expanded in a separate task |

These values must be verified and locked by tests during the content/position separation phase.

## 6. Validation rules

The resolver must reject invalid configurations before constructing the model.

### 6.1 Preset conflicts

Reject:

```text
position_preset=legacy
plus any explicitly supplied custom positional flag
```

The error must list the conflicting arguments.

### 6.2 Missing custom strategies

Reject `position_preset=custom` unless all three are explicit:

```text
absolute_position_encoding
relative_position_encoding
chromosome_encoding
```

### 6.3 Method-specific arguments

Reject method-specific arguments supplied for an inactive method.

Examples:

```text
--rope-base with relative=t5_bucket
--position-bin-size with absolute=none
--alibi-distance-scale with relative=rope
```

Parser-level defaults for method-specific arguments should be `None`. The resolver applies defaults after strategy selection, allowing it to distinguish an omitted value from an incompatible explicit value.

### 6.4 Shape and mathematical constraints

Reject:

- `latent_dim % num_heads != 0`;
- RoPE with odd `head_dim`;
- non-positive dimensions, scales, bin sizes, bucket counts, or distances;
- sinusoidal with odd `position_dim`;
- T5 bucketing with an odd `num_position_buckets`;
- T5 bucketing with `num_position_buckets < 4`;
- T5 bucketing when `max_position_distance <= num_position_buckets // 4`.

### 6.5 Required metadata

Reject new training when:

- `cross_chromosome_policy=mask` and `chrom_ids` or a chromosome mapping are unavailable;
- `cross_chromosome_policy=separate`, the relative strategy is T5/RoPE/ALiBi, and chromosome routing metadata are unavailable;
- `chromosome_encoding=learned` but `num_chromosomes <= 0`;
- learned-binned encoding lacks a supported genome build or versioned chromosome-length metadata.

### 6.6 Padding and attention safety

The implementation must guarantee:

- padded variants cannot become valid query/key pairs;
- `mask` cross-chromosome policy cannot leave a valid query with no valid key;
- self-attention remains available for each valid variant unless explicitly changed in another task;
- the boolean mask, not a newly invented chromosome padding ID, is authoritative for padding.

Historical chromosome indices start at 0 for real chromosomes, while padded tensors also contain zeros. New positional work must not silently shift these IDs because existing preprocessed data and checkpoints depend on the mapping. The mask distinguishes padding from a real chromosome whose index is 0. The complete chromosome-name mapping must be serialized for new runs.

## 7. Serialized configuration schema

New runs write:

```yaml
config_schema_version: 2

# Existing training fields remain at their current top-level locations.

input_dim: 65
content_dim: 1
num_genes: 16089
num_chromosomes: 24

position_encoding:
  schema_version: 1
  preset: legacy

  absolute:
    type: sinusoidal
    fusion: input_concat
    dim: 64
    coordinate_scale: 1.0
    max_wavelength: 10000.0
    bin_size_bp: null

  relative:
    type: t5_bucket
    num_buckets: 32
    total_bias_rows: 33
    max_distance_bp: 100000
    rope_coordinate_scale: null
    rope_base: null
    alibi_distance_function: null
    alibi_distance_scale: null

  chromosome:
    encoding: learned
    cross_chromosome_policy: separate
    num_chromosomes: 24
    mapping:
      "0": "1"
      "1": "2"
      # complete resolved dataset mapping saved by implementation;
      # padding is distinguished by the boolean mask
    cross_chromosome_parameter: dedicated_bucket

  attribution:
    default_ig_mode: content

dataset_identity:
  genome_build: GRCh37
  chromosome_mapping_sha256: "..."
  gene_mapping_sha256: "..."
  preprocessed_data_sha256: "..."
```

Rules:

- inactive method-specific fields are serialized as `null`;
- values are resolved values, not unresolved parser defaults;
- `total_bias_rows` is saved separately from ordinary `num_buckets`;
- the complete chromosome mapping is serialized because a state dict can reveal row count but not chromosome-name-to-index identity;
- gene mapping is represented by a stable checksum and gene count;
- the preprocessed data file or canonical dataset manifest receives a stable identifier/checksum;
- learned-binned metadata also records the chromosome-length table version;
- these mapping/checksum rules are explicit new design choices, not descriptions of current repository behavior;
- fold configs contain the same resolved positional and dataset identity sections.

## 8. Checkpoint metadata

New checkpoints should include lightweight reconstruction metadata in addition to the state dict:

```yaml
config_schema_version: 2
position_encoding_schema_version: 1
position_encoding: <resolved nested mapping>
input_dim: <resolved int>
content_dim: <resolved int>
num_genes: <resolved int>
num_chromosomes: <resolved int>
```

`config.yaml` remains the primary human-readable configuration.

Checkpoint metadata is a fallback and consistency check.

Merge rules:

- for a new-schema run, `config.yaml` is primary;
- checkpoint metadata may fill only fields missing from an otherwise valid config;
- every field present in both sources must agree;
- any overlapping conflict raises a clear error;
- an incomplete old config is resolved through the compatibility rules below rather than being treated as a new-schema config.

## 9. Old config and checkpoint compatibility

### 9.1 Trigger

Compatibility resolution is used only when:

```text
position_encoding is absent from config
```

It is not selectable for new training.

### 9.2 Inference rules

The loader infers:

1. annotation level from config;
2. historical absolute sinusoidal behavior:
   - L0: none;
   - L1-L4: historical 64-dimensional sinusoidal input;
3. T5-style relative bias from the historical architecture/state-dict keys;
4. chromosome embedding presence from `chrom_embedding.weight` or its prefixed equivalent in the state dict;
5. input dimension:
   - use the encoder weight shape as structural authority;
   - use config only when it agrees;
   - use historical annotation-level fallback only when neither source provides the value;
6. number of chromosome embedding rows from the state dict when present;
7. T5 bias row interpretation:
   - older 32-row checkpoints represent ordinary buckets without the new extra row;
   - current 33-row checkpoints represent 32 ordinary buckets plus one cross-chromosome row.

The resolved compatibility object records:

```yaml
position_encoding:
  schema_version: 1
  preset: legacy_checkpoint
  source: compatibility_inference
  warnings:
    - "..."
```

### 9.3 No guessing across the CV/single-split discrepancy

The compatibility loader must not assume chromosome embedding is present merely because the run used historical training code.

It must inspect the checkpoint.

### 9.4 Compatibility warnings

Warnings are emitted when:

- chromosome mapping cannot be verified;
- `chrom_ids` are unavailable;
- input dimension is inferred rather than serialized;
- old explanation behavior is selected automatically.

Errors are raised when:

- config `input_dim` conflicts with the encoder input width in the state dict;
- config and checkpoint metadata overlap and disagree;
- checkpoint tensor shapes cannot be reconciled with the inferred architecture.

A state dict cannot recover the original chromosome-name-to-index mapping; that mapping must come from matching data or serialized metadata.

## 10. Explanation and evaluation contract

### 10.1 Configuration authority

For new-schema runs:

- explanation and validation load positional configuration from `config.yaml`;
- checkpoint metadata is checked for consistency;
- positional architecture cannot be overridden casually through explanation CLI flags;
- a supplied dataset must match recorded genome build and mapping identity.

### 10.2 No duplicated positional flags

`scripts/explain.py` and validation scripts do not expose copies of all training positional flags.

They reconstruct the training architecture from config.

This prevents:

```text
train with RoPE
explain accidentally with T5 buckets
```

### 10.3 Explanation chunking

Training chunk size and explanation maximum variants remain separate concepts.

Existing arguments remain:

```text
training:    --chunk-size
explanation: --max-variants
```

Rules:

- explanation does not silently inherit training chunk size;
- explanation records `max_variants`;
- `scripts/explain.py` currently performs manual chunk-level attribution and does not use `attribute_batch` subsampling;
- `IntegratedGradientsExplainer.attribute_batch` currently uses `torch.randperm` without a local saved seed;
- the later IG task must make both paths explicit;
- any path that randomly subsamples variants uses a saved deterministic seed;
- sampled indices are saved so multiple positional models can be compared on identical variants;
- changing explanation chunk/subsampling behavior is handled in the later Integrated Gradients task, not in initial config plumbing.

## 11. Integrated Gradients CLI

Add to `scripts/explain.py`:

```text
--ig-mode {auto,content,legacy}
```

Default:

```text
auto
```

Resolution:

- new-schema checkpoint:
  - `auto` resolves to saved `position_encoding.attribution.default_ig_mode`;
  - default saved value is `content`;

- old config/checkpoint:
  - `auto` resolves to `legacy`;
  - a warning explains the historical behavior.

### 11.1 `content`

Differentiable input:

```text
dosage and non-positional annotations only
```

Fixed during integration:

```text
position encoding
positions
chromosome IDs
gene IDs
masks
covariates
```

Per-variant ranking:

- retain signed feature-level attributions for completeness diagnostics;
- L0/L1 dosage-only score may use absolute dosage attribution;
- richer levels use a documented aggregation over content channels;
- the exact aggregation implementation is finalized in the later IG task.

### 11.2 `legacy`

Reproduces the current feature attribution path as closely as possible:

- historical `variant_features` are the differentiable input;
- current L2 feature norm is used for the main per-variant score;
- for L1-L4, sinusoidal channels are included in that norm;
- output metadata marks the result as non-comparable to content-only benchmark attributions.

## 12. Downstream propagation

The following consumers and public surfaces must preserve, restore, or report positional metadata:

- `src/encoding/__init__.py`
- `src/models/encoder.py`
- `src/models/classifier.py` where architecture inference relies on classifier shapes
- `src/training/trainer.py`
- `scripts/explain.py`
- `scripts/validate_epistasis.py`
- `src/explain/attention_analysis.py`
- `src/explain/counterfactual_epistasis.py`
- `scripts/run_null_baseline_analysis.sh`
- `scripts/select_best_cv_fold.py`
- `scripts/ablation_compare.py`
- `scripts/render_model_architecture.py`
- `scripts/plot_detailed_architecture.py`
- `utilities/demos/test_encoding_pipeline.py`
- `utilities/demos/test_training_pipeline.py`
- `utilities/demos/test_model_architecture.py`
- experiment comparison and attribution comparison outputs

Focused test inventory includes:

- `tests/test_chrom_aware_attention.py`
- `tests/test_fold_config_saving.py`
- `tests/test_class_weighting.py`
- `tests/test_phase3_explain.py`
- `tests/test_ig_covariates.py`
- `tests/test_classifier_type_switch.py`
- `tests/test_chunked_sieve.py`
- `tests/test_validate_epistasis_chunking.py`
- `tests/test_null_baseline_protocol_matching.py`

Null-baseline training must copy the real experiment's complete positional configuration rather than reconstructing it from partial hyperparameters.

Ablation comparison outputs must include positional strategy identifiers so experiments are not grouped only by annotation level or classifier.

## 13. Human-readable run identifiers

The implementation may derive a compact identifier:

```text
abs-<absolute>__rel-<relative>__chr-<chromosome>
```

Examples:

```text
abs-none__rel-none__chr-none
abs-sinusoidal__rel-t5_bucket__chr-learned
abs-none__rel-rope__chr-learned
abs-none__rel-alibi_fixed__chr-learned
```

This identifier may be printed and stored in results metadata.

It must not replace the full serialized configuration.

## 14. CLI examples

### 14.1 New normalized legacy run

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset legacy \
  --experiment-name cad_L1_legacy \
  ...
```

Because `legacy` is the default, omitting `--position-preset legacy` produces the same resolved configuration.

The learned chromosome embedding is initialized to zero, matching the current CV initialization. It may learn during training. Exact historical single-split architecture is preserved only through old-checkpoint compatibility, because the historical single-split path did not construct this module.

### 14.2 True no-position control

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding none \
  --relative-position-encoding none \
  --chromosome-encoding none \
  --cross-chromosome-policy separate \
  --experiment-name cad_L1_no_position \
  ...
```

With relative `none` and chromosome encoding `none`, the `separate` policy adds no positional term; chromosome IDs remain available as metadata.

### 14.3 Absolute sinusoidal only

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding sinusoidal \
  --relative-position-encoding none \
  --chromosome-encoding none \
  --position-dim 64 \
  --sinusoidal-coordinate-scale 1.0 \
  --sinusoidal-max-wavelength 10000 \
  ...
```

### 14.4 T5-style relative bias only

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding none \
  --relative-position-encoding t5_bucket \
  --chromosome-encoding none \
  --cross-chromosome-policy separate \
  --num-position-buckets 32 \
  --max-position-distance 100000 \
  ...
```

### 14.5 Learned absolute bins

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding learned_binned \
  --relative-position-encoding none \
  --chromosome-encoding learned \
  --position-dim 64 \
  --position-bin-size 10000 \
  ...
```

### 14.6 RoPE

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding none \
  --relative-position-encoding rope \
  --chromosome-encoding learned \
  --cross-chromosome-policy separate \
  --rope-coordinate-scale 10000 \
  --rope-base 10000 \
  ...
```

### 14.7 Fixed ALiBi

```bash
python scripts/train.py \
  --preprocessed-data /path/to/data.pt \
  --level L1 \
  --position-preset custom \
  --absolute-position-encoding none \
  --relative-position-encoding alibi_fixed \
  --chromosome-encoding learned \
  --cross-chromosome-policy separate \
  --alibi-distance-function log1p \
  --alibi-distance-scale 10000 \
  ...
```

### 14.8 Explanation

```bash
python scripts/explain.py \
  --experiment-dir outputs/cad_L1_legacy \
  --preprocessed-data /path/to/data.pt \
  --output-dir results/cad_L1_legacy \
  --ig-mode auto \
  ...
```

No positional architecture flags are repeated.

## 15. Error-message requirements

Validation errors must name:

- the invalid argument;
- its supplied value;
- the active method/preset;
- the allowed remedy.

Example:

```text
--rope-base was supplied, but --relative-position-encoding is t5_bucket.
Remove --rope-base or select --relative-position-encoding rope.
```

Compatibility errors must state which evidence conflicted:

```text
Config declares no chromosome embedding, but checkpoint contains
base_model.attention.attention_layers.0.chrom_embedding.weight.
```

## 16. Current implementation gaps versus contract choices

The following statements in this contract are intentional future design choices rather than descriptions of code that already exists:

- content-only dimensions and model-level content/position separation;
- learned chromosome-aware binned embeddings;
- versioned chromosome-length metadata;
- resolved nested configuration and dataset identity checksums;
- checkpoint reconstruction metadata;
- RoPE;
- ALiBi;
- the normalized new-run `legacy` preset;
- `--ig-mode`;
- deterministic attribution subsampling metadata.

Current repository facts that constrain those choices include:

- current `input_dim` is selected from fixed annotation-level dimensions;
- current sinusoidal channels are created in preprocessing;
- current T5-style relative bias already exists;
- current checkpoint metadata is minimal;
- current explanation has a manual chunk-level IG path with an L2 feature norm;
- current CV and single-split paths differ in chromosome-embedding construction.

## 17. Implementation order governed by this contract

After this document is accepted:

1. add configuration dataclasses/enums and pure resolver functions;
2. add CLI arguments with no model behavior change;
3. save resolved configuration;
4. load and validate resolved configuration in explanation/validation;
5. add focused configuration tests;
6. add legacy regression tests;
7. separate content and positional inputs;
8. introduce strategy interfaces;
9. make existing mechanisms independently selectable;
10. implement learned binned encoding;
11. implement RoPE;
12. implement fixed and learned ALiBi;
13. standardize Integrated Gradients;
14. update downstream comparison and rendering tools.

The first implementation task must be configuration plumbing only. It must not add RoPE, ALiBi, learned bins, or alter the forward pass.

## 18. Acceptance criteria for this contract

This contract is accepted when:

- the flag names and saved-field names are approved;
- `legacy` semantics are approved;
- custom required fields are approved;
- the absolute fusion location is approved;
- RoPE and ALiBi cross-chromosome behavior is approved;
- old-checkpoint inference rules are approved;
- explanation chunking remains explicitly separate;
- IG mode resolution is approved;
- no unresolved ambiguity remains that would force the first coding task to invent behavior.
