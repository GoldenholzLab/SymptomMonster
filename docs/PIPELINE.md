# Pipeline

This is the end-to-end walkthrough: every stage, which command produces which
file, and how those files feed the next stage. Field-level schemas for each
artifact are in [FORMATS.md](FORMATS.md); the statistical reasoning is in
[METHODS.md](METHODS.md). Placeholder names (`Drugzol`, `Tokenax`, `headache`)
stand in for whatever your study uses.

## Data flow at a glance

```
                  alias.json
                      |
  notes ---------->  extract  ----->  signal.extractions.jsonl --+
    |                                                            |
    |                normalize build  <--------------------------+
    |                      |                                     |
    |                mapping.csv (review)                        |
    |                      |                                     |
    |                normalize apply  <--------------------------+
    |                      |
    |             signal.normalized.jsonl ---------------------+
    |                                                          |
    +--> noise --> surrogate.notes.jsonl --> extract -->       |
                                             normalize apply   |
                                                  |            |
                                      noise.normalized.jsonl   |
                                                  |            |
                                                  v            v
                                                stats   /   bayes
                                                  |
                                             summary.csv
                                                  |
             +---------------+---------------+---------------+
             v               v               v               v
          subgroup        reference        figures         tables

  model_outputs/*.jsonl + annotations/*.csv ---> benchmark ---> benchmark.csv
```

The signal path and the noise path run the **same** `extract` and
`normalize apply` steps. Only the input notes differ: real notes for the signal,
surrogate notes for the noise. Sharing the extractor, the prompt, and the mapping
across both is what makes the per-patient pairing valid.

## 1. extract

```bash
symptommonster extract \
  --notes notes/ \
  --alias-map alias.json \
  --model my-local-model \
  --backend ollama \
  --out signal.extractions.jsonl
```

For each patient the extractor joins the pre-treatment notes and the
post-treatment notes into two windows, replaces every drug name in them with its
fictitious token (the index drug and each co-medication get distinct tokens),
checks that no real name survived, and prompts the model to list the symptoms
attributable to the index drug. The index drug is named in the prompt only by its
masked token, taken from the patient's `group`.

Output: one `signal.extractions.jsonl` row per patient, with the fields
`{patient_id, group, symptoms[], raw_response?, extraction_time_s?}`.

Useful flags: `--limit N` and `--seed S` run a reproducible subset (the seed
fixes the patient order, then the limit takes a prefix); `--resume` skips
patients already present in `--out` so an interrupted run continues; `--prompt
FILE` overrides the packaged extraction template; `--temperature T` controls
sampling (0 by default). One failing patient is logged and skipped, never fatal.

## 2. normalize (build, review, apply)

Normalization is split into a build phase and an apply phase so a human can
inspect and correct the mapping before it ever touches the data.

```bash
symptommonster normalize build \
  --in signal.extractions.jsonl \
  --ctcae vocab.json \
  --out-mapping mapping.csv
```

`build` collects every unique raw term across the input extractions, splits
compound terms, preprocesses each part, and routes it through three tiers: fuzzy
match to your preferred-term vocabulary, then curated regex synonym rules, then
(optionally) a local model for the residue. It writes `mapping.csv` with one row
per raw term: `raw_term, preprocessed_term, normalized_term, tier`. Terms judged
not to be symptoms are written with the `filtered` tier and dropped on apply.

Review or hand-edit `mapping.csv` here. This is the human checkpoint: change any
`normalized_term`, or set it to the filter sentinel to drop a term.

```bash
symptommonster normalize apply \
  --in signal.extractions.jsonl \
  --mapping mapping.csv \
  --out signal.normalized.jsonl
```

`apply` re-splits and preprocesses each raw symptom exactly as `build` did, looks
it up in the reviewed mapping, keeps only terms that resolve to a real canonical
value, and deduplicates per patient. Output: `signal.normalized.jsonl`, with the
fields `{patient_id, group, symptoms[], symptoms_raw?}`.

Tier 3 (the model residue grouping) only runs when you pass `--model`; without it,
residual terms pass through unchanged rather than being dropped, so the pipeline
still runs end to end on a machine with no model.

## 3. noise (the empirical null cohort)

```bash
symptommonster noise \
  --notes notes/ \
  --out surrogate.notes.jsonl \
  --seed 0
```

`noise` constructs a surrogate cohort that carries no treatment signal by
construction. Within each `group` it pools every patient's **pre-treatment**
notes, then gives each surrogate patient both a "pre" window and a "post" window
drawn from that pre-pool, preferentially from other patients. Because both
windows are pre-treatment text, no drug exposure separates them, so any pre/post
difference the extractor later reports is background noise, not a treatment
effect. Each surrogate keeps the original patient's id, group, and window sizes,
so the surrogate cohort has the same shape as the real one.

Output: `surrogate.notes.jsonl` in the note-source JSONL shape (`{patient_id,
group, pre_notes[], post_notes[]}`), which feeds straight back into `extract`.
The construction is deterministic given `--seed`.

## 4. extract + normalize the surrogate (the noise floor)

Run the identical extraction and the identical mapping over the surrogate notes:

```bash
symptommonster extract \
  --notes surrogate.notes.jsonl \
  --alias-map alias.json \
  --model my-local-model \
  --out noise.extractions.jsonl

symptommonster normalize apply \
  --in noise.extractions.jsonl \
  --mapping mapping.csv \
  --out noise.normalized.jsonl
```

Reuse `mapping.csv` from step 2 rather than building a new one, because the two
cohorts must be normalized the same way. The result, `noise.normalized.jsonl`, is
the null each real rate is tested against.

## 5. stats

```bash
symptommonster stats \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --out-dir results/ \
  --fdr-q 0.10 \
  --alpha 0.05
```

For each group, the candidate symptoms are everything its signal run reported.
For each symptom, the patients present in **both** the signal and noise runs are
paired, giving each a 0/1 signal indicator and a 0/1 noise indicator. A
one-sample paired t-test on the per-patient difference yields a p-value,
Benjamini-Hochberg FDR is applied within the group, and rates and their
difference are reported with Clopper-Pearson (exact) and t intervals.

Two optional flags extend this. `--scope global` corrects FDR across every pair
at once instead of within each drug, as a sensitivity analysis. `--covariates
covariates.csv` adds a covariate-adjusted confirmation: the per-patient noise
indicator is fit on the covariates to give an expected background, and the
drugged residuals are tested against zero with their own FDR pass, which adds the
`lr_resid_*` columns to each per-group table.

Output, written under `--out-dir`:

- `stats/<group>.csv`, the full per-pair table for each group.
- `summary.csv`, a thin cross-group slice: `group, symptom, signal_pct,
  noise_pct, signal_minus_noise, p_value, q_value, significant, direction`.

`summary.csv` is the file the downstream `subgroup`, `reference`, `figures`, and
`tables` stages read.

## 6. bayes

```bash
symptommonster bayes \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --out-dir results/ \
  --draws 10000 \
  --threshold 0.95
```

A Beta-Binomial sensitivity analysis that asks the same question without the
t-test's distributional assumptions. Per pair it places a Beta posterior on the
signal rate and the noise rate, draws from both, and reports P(signal > noise)
and the posterior of their difference. Output: `bayes/<group>.csv` per group and
`bayes_summary.csv` across groups. Deterministic given `--seed`.

## 7. subgroup

```bash
symptommonster subgroup \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --covariates covariates.csv \
  --strata institution,sex,age \
  --top-from results/summary.csv \
  --out results/subgroups.csv
```

Takes the top signals (selected via `--top-from summary.csv`), joins each patient
to their covariate row, and re-computes the signal-minus-noise difference within
each stratum of each named covariate. A chi-square test reports whether the effect
is heterogeneous across strata. Output: one `subgroups.csv` row per (group,
symptom, stratum_type, stratum_value).

## 8. benchmark

```bash
symptommonster benchmark \
  --extractions model_outputs/ \
  --annotations annotations/ \
  --gt-mode union \
  --out results/benchmark.csv
```

Independent of the signal/noise path. `--extractions` is a directory with one
normalized JSONL per model; `--annotations` is a directory with one CSV per
annotator. Ground truth is aggregated across annotators (`union`, `majority`, or
`intersection`), all terms are canonicalized so synonyms align, and each model is
scored: macro F1, symptomatic-only F1, weighted MAE, mean inference time, and
extraction bias, one CSV row per model. Inter-annotator agreement
(Krippendorff's alpha over the shared patients) is printed to stderr as context.

## 9. reference

```bash
symptommonster reference \
  --dumps reference_dumps/ \
  --signal results/summary.csv \
  --out results/reference_comparison.csv
```

`--dumps` is a directory of `<group>.json` files, each a list of `{term, rate}`
entries from the literature. Each rate string is parsed to a midpoint percentage
and matched (case-insensitively) to a symptom in `summary.csv`. Output: one
`reference_comparison.csv` row per matched (group, symptom), with the columns
`group, symptom, observed_signal_pct, reference_rate, source`.

## 10. figures and tables

```bash
symptommonster figures --which all \
  --signal results/summary.csv \
  --benchmark results/benchmark.csv \
  --reference results/reference_comparison.csv \
  --matrix matrix.csv \
  --out-dir figures/

symptommonster tables \
  --signal results/summary.csv \
  --demographics demographics.csv \
  --out-dir tables/
```

`figures` renders from the documented CSV formats; pass `--which all` or a single
figure name. The figure names and the input each one consumes are:

| `--which` | Input flag | Reads |
| --- | --- | --- |
| `pipeline` | (none) | data-free schematic |
| `signal_grid` | `--signal` | signal summary CSV |
| `benchmark_lollipop` | `--benchmark` | benchmark CSV |
| `time_vs_efficacy` | `--benchmark` | benchmark CSV |
| `noise_vs_literature` | `--reference` | reference comparison CSV |
| `dendrogram` | `--matrix` | drug-by-symptom matrix CSV |

A figure whose input was not supplied is skipped with a note on stderr rather
than failing the batch, so `--which all` renders whatever the provided inputs
support. `tables` renders manuscript tables from `summary.csv` (written as
`table2.csv`) and, when `--demographics` is given, a `table1.csv`.
