# Formats

Field-level schemas for every file the pipeline reads and writes. JSONL files
hold one JSON object per line. CSV files have a header row. Optional fields are
marked; a `?` after a field name means it may be absent. Placeholder values use a
drug `Drugzol` (token `Tokenax`) and the symptom `headache`.

## Note source

A note source is either a directory tree or a JSONL export.

Directory layout:

```
root/<patient_id>/pre/*.txt      baseline (pre-treatment) notes
root/<patient_id>/post/*.txt     follow-up (post-treatment) notes
```

JSONL export, one object per line:

| Field | Type | Notes |
| --- | --- | --- |
| `patient_id` | string | Unique per patient. |
| `pre_notes` | string[] | Baseline note texts. |
| `post_notes` | string[] | Follow-up note texts. |
| `group` | string \| null | Cohort label; the index drug name (see alias map). |

```json
{"patient_id": "patient_0001", "pre_notes": ["..."], "post_notes": ["..."], "group": "Drugzol"}
```

The `noise` stage writes this exact shape.

## Alias map (JSON)

A flat object: real drug name to fictitious token.

```json
{"Drugzol": "Tokenax", "Comedazine": "Tokenbis"}
```

A patient's `group` must be one of the keys; its token is what the prompt names.

## Extractions JSONL

Written by `extract`, read by `normalize`.

| Field | Type | Notes |
| --- | --- | --- |
| `patient_id` | string | |
| `group` | string \| null | Carried through from the note source. |
| `symptoms` | string[] | Raw model-attributed terms, verbatim. |
| `raw_response?` | string | The model's full response, when retained. |
| `extraction_time_s?` | number | Wall-clock seconds for this patient. |

```json
{"patient_id": "patient_0001", "group": "Drugzol", "symptoms": ["headache", "mild nausea"], "extraction_time_s": 4.2}
```

## Mapping CSV

Written by `normalize build`, reviewed by a human, read by `normalize apply`.
One row per raw term (a compound raw term contributes one row per part).

| Column | Notes |
| --- | --- |
| `raw_term` | The term as the model emitted it. |
| `preprocessed_term` | The cleaned, split form the tiers matched on. |
| `normalized_term` | The canonical term, Title Case; the filter sentinel if dropped. |
| `tier` | Which tier resolved it: `ctcae`, `rule`, `llm`, or `filtered`. |

```csv
raw_term,preprocessed_term,normalized_term,tier
mild nausea,nausea,Nausea,rule
headache,headache,Headache,ctcae
none noted,,[FILTERED],filtered
```

Rows whose `normalized_term` is the filter sentinel are dropped on apply. Edit
`normalized_term` to correct a mapping, or set it to the sentinel to drop a term.

## Normalized JSONL

Written by `normalize apply`, read by `stats`, `bayes`, and `subgroup`. The
signal run and the noise run share this schema.

| Field | Type | Notes |
| --- | --- | --- |
| `patient_id` | string | |
| `group` | string \| null | |
| `symptoms` | string[] | Canonical terms, deduplicated and sorted. |
| `symptoms_raw?` | string[] | The original raw terms, retained for audit. |

```json
{"patient_id": "patient_0001", "group": "Drugzol", "symptoms": ["Headache", "Nausea"], "symptoms_raw": ["headache", "mild nausea"]}
```

## Covariates CSV

Read by `subgroup` and by `stats --covariates`. Keyed by `patient_id`, one
column per covariate.

```csv
patient_id,institution,sex,age
patient_0001,site_a,F,adult
```

## Annotator CSV

Read by `benchmark`. One file per annotator.

| Column | Notes |
| --- | --- |
| `patient_id` | |
| `symptoms` | Comma-separated terms; empty / `none` / `n/a` means no symptoms. |

```csv
patient_id,symptoms
patient_0001,"headache, nausea"
patient_0002,
```

## Stats outputs

`stats` writes a per-group table and a cross-group summary under `--out-dir`.

`stats/<group>.csv`, the full per-pair table:

| Column | Notes |
| --- | --- |
| `symptom` | |
| `n_patients` | Patients with the symptom in the signal run. |
| `total_patients` | Patients paired across both runs (the denominator). |
| `signal_pct` | Signal prevalence, percent. |
| `noise_n` | Patients with the symptom in the noise run. |
| `noise_pct` | Noise prevalence, percent. |
| `signal_minus_noise` | Difference of the two percentages. |
| `t_stat` | Paired one-sample t statistic. |
| `p_value` | Two-sided p-value. |
| `q_value` | Benjamini-Hochberg q-value within the group. |
| `significant` | Whether `q_value <= --fdr-q`. |
| `direction` | `increased` or `decreased`. |
| `diff_ci_lower`, `diff_ci_upper` | t interval on the mean difference, percent. |
| `signal_ci_lower`, `signal_ci_upper` | Clopper-Pearson interval on the signal rate, percent. |
| `noise_ci_lower`, `noise_ci_upper` | Clopper-Pearson interval on the noise rate, percent. |

When `stats` runs with `--covariates`, four more columns follow, carrying the
covariate-adjusted residual test: `lr_resid_mean` (mean of signal minus the
fitted background, percent), `lr_resid_p`, `lr_resid_q` (its own FDR pass), and
`lr_resid_significant`. The `q_value` and `significant` columns reflect the FDR
`--scope`: within each group by default, or pooled across all pairs with
`--scope global`.

`summary.csv`, a thin slice across all groups and the file the downstream stages
read:

| Column |
| --- |
| `group` |
| `symptom` |
| `signal_pct` |
| `noise_pct` |
| `signal_minus_noise` |
| `p_value` |
| `q_value` |
| `significant` |
| `direction` |

## Bayes outputs

`bayes` writes `bayes/<group>.csv` per group and `bayes_summary.csv` across
groups. The columns carry the identifying and rate fields plus the posterior
summary:

| Column | Notes |
| --- | --- |
| `group`, `symptom` | |
| `n_patients`, `total_patients`, `noise_n` | Counts, as in the stats table. |
| `signal_pct`, `noise_pct`, `signal_minus_noise` | Rates, percent. |
| `posterior_prob_signal_gt_noise` | P(signal rate > noise rate). |
| `posterior_mean_diff` | Posterior mean of the difference, percent. |
| `bayes_ci_lower`, `bayes_ci_upper` | Central 95% credible interval, percent. |
| `bayes_significant` | Whether the posterior probability meets `--threshold`. |

## Subgroups CSV

Written by `subgroup`. One row per (group, symptom, stratum).

| Column | Notes |
| --- | --- |
| `group` | |
| `symptom` | |
| `stratum_type` | The covariate name (for example `institution`). |
| `stratum_value` | The level within that covariate (for example `site_a`). |
| `n_patients` | Patients in this stratum. |
| `signal_pct` | Signal prevalence within the stratum, percent. |
| `noise_pct` | Noise prevalence within the stratum, percent. |
| `signal_minus_noise` | Difference within the stratum, percent. |
| `heterogeneity_p` | Chi-square p-value for variation across this covariate's strata. |

## Benchmark CSV

Written by `benchmark`. One row per model (the model name is the JSONL filename
stem). Inter-annotator agreement is reported separately to stderr.

| Column | Notes |
| --- | --- |
| `model` | Filename stem of the model's JSONL. |
| `macro_f1` | Mean per-patient set F1 over all judged patients. |
| `symptomatic_f1` | Mean per-patient F1 over patients with a non-empty truth set. |
| `weighted_mae` | Prevalence-weighted mean absolute error of per-symptom rates. |
| `mean_inference_s` | Mean per-patient inference time; blank if unavailable. |
| `bias` | Mean `len(predicted) - len(truth)` per patient. |

The `time_vs_efficacy` figure reads this same table and additionally recognizes
two optional boolean columns when present: `reasoning` (drawn as a diamond
marker) and `production` (forces the highlighted point).

## Reference dump JSON and comparison CSV

Reference dump, `<group>.json`, a list of entries:

| Field | Type | Notes |
| --- | --- | --- |
| `term` | string | Adverse-event term as published. |
| `rate` | string | Percentage, range, or bound (`"8%"`, `"10% to 25%"`, `"<5%"`). |

```json
[{"term": "headache", "rate": "10% to 25%"}, {"term": "nausea", "rate": "<5%"}]
```

`reference_comparison.csv`, written by `reference`:

| Column | Notes |
| --- | --- |
| `group` | |
| `symptom` | The matched symptom from the signal summary. |
| `observed_signal_pct` | Observed signal prevalence, percent. |
| `reference_rate` | Parsed midpoint of the published rate, percent. |
| `source` | The dump the rate came from (the group). |
