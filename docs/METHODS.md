# Methods

This document describes the methods SymptomMonster implements: how drug identity
is hidden from the model, how the empirical null is constructed, and the
statistical machinery that turns paired symptom indicators into signals. The body
is a generic description of the software's behavior; the closing section records
the configuration the published study used, for anyone reproducing it.

## Masking

Before any note text reaches the model, every drug name in it is replaced with a
fictitious token drawn from a user-supplied alias map. Masking is two-layer by
construction: the index drug maps to one token and each co-medication maps to a
different one. The model therefore cannot recognize the drug under study, and it
cannot infer the drug's class from the identities of the drugs prescribed
alongside it, both of which an unmasked note would leak.

Substitution is case-insensitive but case-preserving: a replaced token takes on
the capitalization pattern of the text it replaced. Matching is anchored on word
boundaries and tries longer names before shorter ones, so a multi-word drug name
is masked as a unit before any of its component words, and a name never matches
inside a larger word.

The prompt names the index drug only by its token, and the same token is what the
masker writes into the notes, so the model reads one consistent fictitious name
throughout. After masking, a leak guard re-scans the text for any source name; if
one survived, the patient is aborted. The guard's error is intentionally vague: it
reports that a leak occurred without repeating the leaked name, so the identity
removed by masking is never reintroduced through logs.

## Symptom extraction

A local large language model reads the masked pre-treatment and post-treatment
note windows and returns the symptoms attributable to the index drug. The
attribution criteria, supplied in the prompt, prioritize an explicit documented
link between a symptom and the drug, then a genuinely new post-treatment symptom,
then a clearly worsened pre-existing one. The criteria exclude the treated
condition itself, findings better explained by another cause, and stable
pre-existing findings. The model returns a JSON list of terms, which a tolerant
parser recovers even when the model wraps it in prose or code fences. Extraction
is backend-agnostic: any locally served model exposed through the Ollama or MLX
backend works, and the model id is an opaque string passed straight through.

## Normalization

Raw model terms vary in phrasing ("drowsiness", "somnolence", "sleepiness") and
in form (compounds, modifiers, qualifiers). Normalization folds them onto a
canonical vocabulary in three tiers, behind a reviewable mapping so a human can
audit and correct the result before it touches the data.

Each raw term is first split on common compound separators and preprocessed:
lowercased, stripped of leading intensity/temporal modifiers and of trailing
location, laterality, and generic-noun qualifiers, so "severe headache in the
left temple" reduces to "headache". A filter step drops anything that is plainly
not a symptom (negations, hedging preambles, schema bookkeeping words, bare dates
and numbers, tiny fragments, and pathologically long strings) before any tier
runs.

The surviving term passes through the tiers in order, first match wins:

1. **Fuzzy vocabulary match.** The term is compared against the user-supplied
   preferred-term list; a close match (above a similarity cutoff) maps to that
   canonical term. This tier is skipped when no vocabulary is provided.
2. **Curated regex synonym rules.** An ordered set of rules maps variant
   phrasings to a canonical symptom; more specific rules precede broader ones so
   they are not swallowed. The packaged rules are clinically generic and carry no
   study, drug, or cohort specifics; they can be overridden wholesale.
3. **Model residue grouping.** The long tail of colloquial terms the first two
   tiers miss is batched to a local model, which groups them onto canonical
   terms. This tier runs only when a model is configured; otherwise residual
   terms pass through unchanged rather than being dropped, so the pipeline still
   completes without a model.

The build phase writes one mapping row per term recording the raw term, its
preprocessed form, the chosen canonical term, and the resolving tier. The apply
phase re-derives each term's preprocessed form identically, looks it up, keeps
only terms that resolve to a real canonical value, and deduplicates per patient.

## The scrambled-surrogate empirical null

Some apparent symptom rate is not a treatment effect at all. It is the base rate
at which clinicians write a given word in any note. To separate signal from this
background, the pipeline builds a surrogate cohort that carries no on-treatment
signal by construction and runs the identical extraction over it.

The construction proceeds per cohort. All of a cohort's pre-treatment notes are
pooled. Each surrogate patient is then given both a "pre" window and a "post"
window resampled from that pre-pool, preferentially from other patients' notes,
with each patient's original window sizes preserved. Because both windows are
drawn from pre-treatment text, no drug exposure separates them: any pre/post
difference the extractor reports on the surrogate is, by construction, noise.
Preserving each patient's id and window shape keeps the surrogate cohort aligned
one-to-one with the real one, which is what makes the subsequent paired test
valid. The whole construction is deterministic given a seed.

## Per-pair statistics

For each cohort, the candidate symptoms are everything its signal run reported.
For each (cohort, symptom) pair, the patients present in both the signal and the
noise runs are paired. Each contributes a 0/1 signal indicator (did the real run
report the symptom for this patient) and a 0/1 noise indicator (did the surrogate
run), and the analysis operates on the per-patient difference of the two.

**Paired t-test.** A one-sample, two-sided t-test asks whether the mean
per-patient difference (signal minus noise) is non-zero. Degenerate cases are
handled explicitly: a single pair carries no variance estimate, and a difference
vector with zero variance is reported as no effect when its mean is zero and as a
perfectly consistent effect otherwise.

**Benjamini-Hochberg FDR.** Because many symptoms are tested per drug, p-values
are corrected for multiplicity. The step-up procedure yields a q-value per pair,
the smallest false-discovery-rate level at which the pair would be declared
significant, and a pair is called significant when its q-value is at or below the
chosen threshold (q < 0.10 by default). By default each drug is corrected on its
own, which keeps a drug with many symptoms from borrowing or lending significance
to an unrelated one; the correction can instead be pooled across every pair as a
global sensitivity analysis.

**Confidence intervals.** The signal rate and the noise rate each get an exact
Clopper-Pearson binomial interval, inverted from the Beta distribution so the
endpoints behave correctly at zero and full prevalence. The mean difference gets
a Student-t interval. All intervals are reported on the percentage scale to match
the rates.

**Covariate-adjusted confirmation.** When covariates are supplied, each pair also
gets a confirmation that the gap is not explained by patient mix. The per-patient
noise indicator is regressed on the covariates with logistic regression to give
an expected background rate for each patient, and a one-sample t-test then asks
whether the drugged residuals (signal minus expected) differ from zero. These
residual p-values carry their own FDR pass. A symptom that is all-present or
all-absent in the noise arm, or a rank-deficient design that the fit cannot
resolve, falls back to the unconditional background rate rather than failing.

## Bayesian sensitivity analysis

As a check that does not lean on the t-test's distributional assumptions, each
pair is also analyzed with a Beta-Binomial model. Under a uniform Beta(1, 1)
prior, the posterior for a rate observed as k of n is Beta(k + 1, n - k + 1). The
analysis draws from the signal posterior and the noise posterior, then reports
P(signal rate > noise rate) and the posterior mean and central 95% credible
interval of their difference. A pair is flagged when the posterior probability
meets a chosen threshold (0.95 by default). Equal counts over equal totals give a
probability near one-half; a strong signal over an empty noise floor pushes it
toward one. The Monte-Carlo estimate is deterministic given a seed.

## Subgroup heterogeneity

To check whether a signal is uniform across the population or concentrated in a
subgroup, the top signals are stratified by user-supplied covariates. Within each
stratum of each named covariate, the signal-minus-noise difference is recomputed,
and a chi-square test reports whether symptom occurrence varies across that
covariate's strata, where a small heterogeneity p-value indicates the effect is
not the same in every subgroup.

## Benchmark metrics and agreement

Model quality is measured against human annotation. Ground truth is aggregated
across annotators under a configurable rule: `union` (a symptom counts if any
annotator recorded it), `majority` (at least half of the annotators who saw the
patient), or `intersection` (all of them). Every term is folded to a canonical
representative first, so synonyms agree before voting.

Each model is then scored per patient on the set of symptoms it attributed:

- **Macro F1**: the mean per-patient set F1 over all judged patients, so a patient
  with two symptoms counts the same as one with ten. Two empty sets count as a
  perfect match (correctly agreeing a patient has no symptoms), and a patient the
  model skipped is treated as an empty prediction rather than dropped.
- **Symptomatic-only F1**: the same mean restricted to patients whose truth set is
  non-empty, isolating performance where it matters and removing the inflation
  from empty-versus-empty agreement.
- **Weighted MAE**: the weighted mean absolute error between the model's
  per-symptom prevalence vector and the truth's, aligned over the union of
  symptoms.
- **Bias**: the mean per-patient difference in the number of symptoms (predicted
  minus truth); positive is over-extraction, negative is under-extraction.
- **Mean inference time**: the average per-patient wall-clock time, when the
  extractions carry it.

Inter-annotator agreement is reported alongside the model scores as context. The
headline measure is pairwise set-F1, the same set-F1 used to score the models,
averaged per annotator pair over the patients both judged: information-retrieval
agreement has no stable negative class, so a chance-corrected coefficient
understates it. The package also implements three chance-corrected coefficients
from their definitions, as a conventional cross-check: Cohen's kappa (two raters
over paired labels), Fleiss' kappa (any number of raters, from a per-item
category-count table), and Krippendorff's alpha (a raters-by-items matrix tolerant
of missing judgements). The benchmark prints the pairwise-F1 range and, beneath it,
Krippendorff's alpha over the patients every annotator judged.

## Reference-rate comparison

Observed signals can be placed against published adverse-event rates. Each
cohort's reference dump lists terms with rates in free-form text: bare
percentages, explicit ranges, or open-ended bounds. Each rate is parsed to a
single representative midpoint percentage (the mean of a two-sided range, half of
an upper bound, or the bound itself for a lower bound), the term is matched
case-insensitively to a symptom in the signal summary, and the observed signal
prevalence is emitted next to the reference rate for plotting. This is a
descriptive comparison, not a statistical test.

## Reference configuration (as published)

The code is model- and vocabulary-agnostic; the models, vocabulary, and
thresholds are supplied per run. For reproduction, the published study used:

- **Extraction model**: `ministral-3:14b-instruct-2512-fp16`, served through
  Ollama or MLX at temperature 0.05 with a 131,072-token context window (the
  `extract` defaults for `--temperature` and `--num-ctx`).
- **Tier-3 normalization model**: `qwen3.5:122b-a10b-q4_K_M` at temperature 0.0.
- **Tier-1 vocabulary**: CTCAE v5.0 preferred terms.
- **FDR**: Benjamini-Hochberg within drug at q < 0.10.
