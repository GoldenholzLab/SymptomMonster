# SymptomMonster

SymptomMonster detects drug-attributed symptom signals from paired pre- and
post-treatment clinical notes. A locally served large language model reads each
patient's notes, with every drug name replaced by a fictitious token, and
extracts the symptoms it judges attributable to the index drug. A per-pair
empirical null, built by reassembling pre-treatment notes from other patients in
the same cohort, separates real treatment signal from the background rate at
which clinicians record a given word regardless of treatment. The result is a per
drug-symptom table of signal-minus-noise differences with frequentist and
Bayesian support, optional subgroup heterogeneity, model benchmarking against
human annotators, and comparison to published reference rates.

This repository ships **method code only**. It contains no clinical data and no
vocabularies. You supply every input: the notes, a drug alias map, a
preferred-term vocabulary, and any optional covariates or reference dumps.

## Method overview

1. **extract**: mask the index drug and every co-medication to distinct
   fictitious tokens, then have a local model extract symptoms attributable to
   the index drug from the masked notes.
2. **normalize**: fold raw extracted terms onto a canonical vocabulary in three
   tiers (fuzzy vocabulary match, curated regex rules, then a model for the
   residue), through a reviewable mapping CSV you can audit before applying.
3. **noise**: build scrambled-surrogate pre/post pairs from other patients'
   pre-treatment notes, then run extract and normalize over them to obtain the
   empirical null.
4. **stats**: per drug-symptom pair, a paired one-sample t-test on the
   per-patient signal-minus-noise difference, Benjamini-Hochberg FDR within each
   drug at q < 0.10, and Clopper-Pearson confidence intervals. FDR can instead be
   pooled across all pairs, and a covariate-adjusted residual confirmation is
   available when covariates are supplied.
5. **bayes**: a Beta-Binomial posterior P(signal > noise) per pair, as a
   distribution-free sensitivity analysis.
6. **subgroup**: stratify the top signals by user-supplied covariates and test
   heterogeneity across strata.
7. **benchmark**: score several local models against annotator ground truth
   (macro F1, symptomatic-only F1, weighted MAE, inference time, bias) and report
   inter-annotator agreement.
8. **reference**: parse published adverse-event reference-rate dumps and join
   them to the observed signals.
9. **figures / tables**: render figures and manuscript tables from the documented
   CSV formats.

```
  signal lane:  notes  -> extract -> normalize (build, review, apply) -> signal
  noise lane:   notes  -> noise   -> extract   -> normalize (apply)    -> noise

  signal + noise -> stats -> summary.csv -> subgroup, reference, figures, tables
  signal + noise -> bayes
  model outputs + annotations -> benchmark
```

## Installation

```bash
pip install -e .
```

The pure-Python statistics, normalization, and figure code installs with no
model dependencies. The model stages (`extract`, and tier 3 of `normalize`) need
a local model server. Install the matching extra:

```bash
pip install -e ".[ollama]"   # Ollama backend
pip install -e ".[mlx]"      # Apple MLX backend
pip install -e ".[dev]"      # test and lint tooling
```

Python 3.11 or newer is required.

## Inputs you supply

No data or vocabularies are bundled. Every stage reads explicit paths you
provide; there are no hidden defaults. See [docs/INPUTS.md](docs/INPUTS.md) for
the exact layouts and [docs/FORMATS.md](docs/FORMATS.md) for the field-level
schemas.

- **Notes**: either a directory tree (`root/<patient_id>/pre/*.txt` and
  `.../post/*.txt`) or a JSONL of `{patient_id, pre_notes[], post_notes[], group}`.
- **Alias map**: a JSON object mapping each real drug name to a fictitious token,
  for example `{"Drugzol": "Tokenax"}`. A patient's `group` names their index
  drug, whose token the extraction prompt refers to.
- **Preferred-term vocabulary**: a JSON list of canonical terms (for example a
  CTCAE export) used by the tier-1 fuzzy match.
- **Covariates** (optional): a CSV keyed by `patient_id` with one column per
  covariate, for subgroup analysis and the covariate-adjusted stats check.
- **Reference dumps** (optional): per-group JSON files of published adverse-event
  rates, for the literature comparison.

## Quickstart

The commands below run a full study end to end with placeholder filenames. The
noise floor is produced by running the same extract and normalize steps over the
surrogate notes the noise stage emits.

```bash
# 1. Extract symptoms from the real cohort (drugs are masked internally).
symptommonster extract \
  --notes notes/ \
  --alias-map alias.json \
  --model my-local-model \
  --backend ollama \
  --out signal.extractions.jsonl

# 2. Build a reviewable raw -> canonical mapping, then apply it.
symptommonster normalize build \
  --in signal.extractions.jsonl \
  --ctcae vocab.json \
  --out-mapping mapping.csv
#    ... review / hand-edit mapping.csv here ...
symptommonster normalize apply \
  --in signal.extractions.jsonl \
  --mapping mapping.csv \
  --out signal.normalized.jsonl

# 3. Build the scrambled-surrogate note pairs (the empirical null cohort).
symptommonster noise \
  --notes notes/ \
  --out surrogate.notes.jsonl

# 4. Run the SAME extract and normalize over the surrogate notes -> noise floor.
symptommonster extract \
  --notes surrogate.notes.jsonl \
  --alias-map alias.json \
  --model my-local-model \
  --out noise.extractions.jsonl
symptommonster normalize apply \
  --in noise.extractions.jsonl \
  --mapping mapping.csv \
  --out noise.normalized.jsonl

# 5. Per-pair statistics: paired t-test, within-drug BH-FDR, exact CIs.
symptommonster stats \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --out-dir results/

# 6. Bayesian sensitivity check.
symptommonster bayes \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --out-dir results/

# 7. Optional: subgroup heterogeneity, reference comparison, figures, tables.
symptommonster subgroup \
  --signal signal.normalized.jsonl \
  --noise noise.normalized.jsonl \
  --covariates covariates.csv \
  --top-from results/summary.csv \
  --out results/subgroups.csv
symptommonster reference \
  --dumps reference_dumps/ \
  --signal results/summary.csv \
  --out results/reference_comparison.csv
symptommonster figures --which all --signal results/summary.csv --out-dir figures/
symptommonster tables --signal results/summary.csv --out-dir tables/
```

Reuse the same `mapping.csv` for both the signal and the noise runs so the two
cohorts are normalized identically, which is what keeps the paired test valid.

The `--model` id is whatever your backend serves. The published study used
`ministral-3:14b-instruct-2512-fp16` for extraction; see
[docs/METHODS.md](docs/METHODS.md) for the full reference configuration.

## Stage reference

| Command | Reads | Writes |
| --- | --- | --- |
| `extract` | notes, alias map, model | extractions JSONL |
| `normalize build` | extractions, vocabulary, rules | mapping CSV |
| `normalize apply` | extractions, mapping | normalized JSONL |
| `noise` | notes | surrogate notes JSONL |
| `stats` | signal + noise normalized JSONL | `stats/<group>.csv`, `summary.csv` |
| `bayes` | signal + noise normalized JSONL | `bayes/<group>.csv`, `bayes_summary.csv` |
| `subgroup` | signal + noise + covariates | subgroups CSV |
| `benchmark` | model extractions dir + annotations dir | benchmark CSV |
| `reference` | reference dumps dir + signal summary | reference comparison CSV |
| `figures` | summary / benchmark / reference / matrix CSVs | figure files |
| `tables` | summary (+ demographics) | table files |

Every subcommand supports `--help`. See [docs/PIPELINE.md](docs/PIPELINE.md) for
which command feeds which file, and [docs/METHODS.md](docs/METHODS.md) for the
statistical and normalization methods.

## Two-layer masking and the leak guard

Drug identity is hidden before any note text reaches the model, in two layers by
construction: the index drug maps to one token and every co-medication maps to a
different one. The model can therefore neither recognize the drug under study nor
infer its class from the company it keeps. Matching is case-insensitive,
case-preserving, and anchored on word boundaries with longest-name-first
precedence, so a multi-word name is masked before any of its substrings and
partial words are never touched.

After masking, a leak guard re-scans the text for any source name and aborts the
patient if one survived. The error is deliberately vague: it confirms that a leak
occurred without echoing the name that leaked, so logs never reintroduce the
identity the masking removed.

## Testing

```bash
pytest
```

The suite exercises the pure functions (statistics, agreement coefficients,
masking, normalization, parsing) and needs neither a model nor a network.

## License

MIT. See [LICENSE](LICENSE).
