# AGENTS.md — SIEVE positional-encoding development rules

## Purpose

This repository is being extended to benchmark multiple positional-encoding strategies while preserving the historical SIEVE implementation.

Read this file before every task. Also read:

- `documentation/appendices/position-encoding-audit.md`
- the task-specific prompt supplied by the user

The task prompt may narrow scope further, but it must not silently override the compatibility rules below.

## Required workflow

For every task:

1. Inspect the relevant implementation, tests, constructors, config writers, and config readers.
2. Report all relevant call sites and current behavior.
3. Propose a narrow implementation plan.
4. Stop and wait for explicit approval before editing.
5. After approval, modify only the agreed files.
6. Summarize the diff and unresolved risks.
7. Run focused tests and changed-file checks.
8. Do not commit or push unless explicitly requested.

Never combine planning and implementation in the first response to a new task.

## Project invariants

- Preserve historical behavior through an explicit `legacy` configuration.
- Existing CLI commands without new positional flags retain historical behavior.
- Existing configs without positional fields resolve through an explicit compatibility rule; do not guess whether CV or single-split chromosome behavior should apply.
- Training, explanation, validation, and checkpoint reconstruction use the same resolved positional configuration.
- Do not change chunking, gene aggregation, classifier behavior, covariates, loss functions, optimizer defaults, or training hyperparameters unless explicitly required.
- Do not modify unrelated files.
- Do not add dependencies without approval.
- Do not perform repository-wide formatting or lint cleanup.
- Do not create synthetic cohorts or add smoke-training pipelines.
- Do not run real training.
- Small deterministic tensors in unit tests are allowed and expected.
- Do not remove backward-compatibility code without explicit approval.
- Do not rename accepted public CLI/config fields without a migration plan.

## Positional-encoding boundaries

Keep these concepts separate:

1. **Content features**
   - dosage;
   - consequence representation;
   - SIFT, PolyPhen, and other non-positional annotations.

2. **Absolute positional representation**
   - none;
   - sinusoidal;
   - learned binned;
   - later genomics-aware encodings.

3. **Relative attention mechanism**
   - none;
   - current T5-style bucket bias;
   - RoPE;
   - fixed ALiBi;
   - learned ALiBi;
   - later genomics-aware pairwise mechanisms.

4. **Chromosome handling**
   - must be explicit;
   - cross-chromosome coordinate subtraction must never be treated as meaningful physical distance;
   - legacy chromosome behavior must remain reproducible.

## Historical and legacy behavior

Historical positional behavior is currently path-dependent:

- L0 has no sinusoidal input feature.
- L1-L4 include the current 64-dimensional sinusoidal input encoding.
- The current relative bucket bias is enabled.
- Cross-chromosome dedicated-bucket routing occurs only when `chrom_ids` are supplied.
- CV training supplies `num_chromosomes`, so a learned chromosome embedding is created.
- Single-split training currently omits `num_chromosomes`, so no learned chromosome embedding is created.

Do not assume that `legacy` means chromosome embedding is always enabled. Until the CLI/config contract explicitly resolves this discrepancy, preserve and report the current path-specific behavior.

A refactor is not complete until focused regression tests cover the agreed legacy/compatibility semantics to the agreed numerical tolerance.

## Integrated Gradients

For the positional benchmark, the primary comparable attribution target is dosage and non-positional content while position remains fixed and active in the model.

Do not silently aggregate sinusoidal attribution channels for one method while comparing them with dosage-only attribution for another.

Any retained legacy attribution behavior must be explicitly named and documented.

## Baseline lint condition

The untouched repository has substantial pre-existing Ruff debt. Do not use repository-wide Ruff success as a task gate and do not fix unrelated violations.

Run Ruff, Black, and isort only on changed Python files unless the task specifically concerns global formatting.

## Required checks

After an implementation task, run at minimum:

```bash
git diff --check
git status --short
python -m pytest <focused test files>
python -m ruff check <changed Python files>
python -m black --check <changed Python files>
python -m isort --check-only <changed Python files>
```

Run broader tests when the change affects shared construction, checkpoint loading, or explanation.

Report:

- commands run;
- pass/fail/skip counts;
- pre-existing failures distinguished from new failures;
- checks not run and why.

## Diff discipline

Before editing:

- identify the minimum file set;
- identify every constructor and reconstruction call site;
- identify config serialization and deserialization paths;
- identify tests that lock current behavior.

After editing:

- show `git diff --stat`;
- summarize behavior changed;
- summarize behavior deliberately unchanged;
- highlight compatibility risks;
- never hide generated or unrelated edits.

## Commit discipline

One conceptual change per commit.

Preferred sequence:

1. documentation audit;
2. CLI/config contract;
3. configuration plumbing without behavior change;
4. legacy regression coverage;
5. separation of content and position;
6. positional strategy interfaces;
7. selectable existing mechanisms;
8. learned absolute encoding;
9. RoPE;
10. ALiBi;
11. Integrated Gradients standardization;
12. downstream reconstruction and documentation.

Do not commit or push unless explicitly requested.

## Task prompt template

```text
Read AGENTS.md and documentation/appendices/position-encoding-audit.md first.

Task:
[one narrowly defined change]

Current phase:
[phase name]

Required behavior:
- [...]
- [...]

Required invariants:
- Legacy behavior remains unchanged unless this task explicitly introduces a new selectable mode.
- Existing CLI defaults remain unchanged.
- Do not modify unrelated files.
- Do not add dependencies.
- Do not create data or run training.
- Add or update focused unit tests.
- Run focused tests and changed-file static checks.

Before editing:
1. Inspect all relevant implementation and test files.
2. Find every constructor, config writer, config reader, and checkpoint reconstruction call site affected.
3. Propose a file-by-file plan.
4. Identify compatibility risks and ambiguities.
5. Stop and wait for approval.

After approval:
1. Implement only the approved plan.
2. Show changed files and a concise diff summary.
3. Run the agreed checks.
4. Report failures honestly.
5. Do not commit or push until explicitly instructed.
```
