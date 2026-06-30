# Inputs

SymptomMonster ships **method code only**. No clinical notes, no drug names, no
preferred-term vocabulary, no synonym data, and no reference rates are bundled
with the package. You provide all of them. This document describes how to supply
each input; [FORMATS.md](FORMATS.md) gives the field-level schemas.

Keeping data out of the package is deliberate. It means the repository can be
shared freely, and it leaves de-identification of notes, the only input that can
carry protected information, in the hands of the data holder. Mask or
remove identifiers in your notes before running anything here; the drug masking
described below hides drug identity for the model, but it is not a substitute for
de-identifying patient information.

All examples use placeholder names: a drug `Drugzol` masked to the token
`Tokenax`, with the symptom `headache`.

## Notes

The note source feeds `extract` and `noise`. Two shapes are accepted; the source
is chosen automatically from the path (a directory versus a `.jsonl` file).

### Directory tree

One folder per patient, with a `pre` and a `post` subfolder of `.txt` files:

```
notes/
  patient_0001/
    pre/
      2019-03-01.txt
      2019-05-12.txt
    post/
      2019-09-20.txt
  patient_0002/
    pre/
      ...
    post/
      ...
```

`pre/` holds baseline (pre-treatment) notes and `post/` holds follow-up
(post-treatment) notes. Patient ids are the folder names and may be formatted
however you like. With a directory source, every patient's `group` is unset
unless you wire one in programmatically; for cohort-aware masking and surrogate
construction, the JSONL shape below is usually more convenient.

### JSONL export

One JSON object per line:

```json
{"patient_id": "patient_0001", "pre_notes": ["...", "..."], "post_notes": ["..."], "group": "Drugzol"}
```

`pre_notes` and `post_notes` are arrays of note strings. `group` is the cohort
label; see the group convention below. This is also the exact shape the `noise`
stage writes, so surrogate notes feed straight back through `extract`.

## Alias map and the group convention

The alias map is a JSON object mapping each real drug name to the fictitious
token that replaces it:

```json
{
  "Drugzol": "Tokenax",
  "Comedazine": "Tokenbis",
  "Otherstatin": "Tokenter"
}
```

Two rules make the masking work:

- **Include every drug you want hidden**: the index drugs under study and the
  common co-medications. Any name not in the map is left untouched in the notes,
  so omissions are how identity leaks. The leak guard will abort a patient whose
  index drug name survives, but it only checks names it knows.
- **Use distinct tokens** so the index drug and its co-medications are not
  collapsed to the same fictitious name. Distinct tokens are what give the
  two-layer guarantee: the model can neither recognize the index drug nor infer
  its class from the surrounding drugs.

The **group convention**: a patient's `group` is the real name of their index
drug, and that name must be a key in the alias map. The extractor looks up the
group in the map to find the token, and the prompt refers to the index drug by
that token only. A patient whose `group` is absent from the map is an error,
since the index drug could be named in the prompt but not masked in the notes. A
patient with no `group` is handled with a neutral phrase ("the index
medication") and no drug-specific masking beyond the map.

Matching is case-insensitive and case-preserving, on word boundaries, with
longer names tried first, so `"valproic acid"` is masked before `"valproic"`,
and a name never matches inside a larger word.

## Preferred-term vocabulary

Tier 1 of normalization fuzzy-matches each raw term against a preferred-term
vocabulary that you supply as JSON. It is **never bundled**; a common choice is a
CTCAE export, but any controlled term list works. Two shapes are accepted:

```json
["Headache", "Nausea", "Dizziness", "Fatigue"]
```

or a list of objects carrying a `term` key:

```json
[{"term": "Headache"}, {"term": "Nausea"}]
```

The vocabulary is optional: pass it with `--ctcae` to `normalize build`. Without
it, tier 1 is skipped and terms fall through to the rule and LLM tiers. The
package's curated regex synonym rules are generic and contain no study, drug, or
cohort specifics; override them with your own via `--rules` if you need a
different canonical set.

## Covariates

Both `subgroup` and `stats --covariates` read a covariates CSV keyed by
`patient_id`, with one column per covariate:

```csv
patient_id,institution,sex,age
patient_0001,site_a,F,adult
patient_0002,site_b,M,pediatric
```

Name the columns to test with `--strata` (default `institution,sex,age`). Values
should be the discrete strata you want compared; bin continuous variables (such
as age) into categories before passing them in.

## Reference dumps

`reference` reads a directory of per-group JSON files named `<group>.json`. Each
is a list of published adverse-event rate entries:

```json
[
  {"term": "headache", "rate": "10% to 25%"},
  {"term": "nausea", "rate": "<5%"},
  {"term": "dizziness", "rate": "8%"}
]
```

Rate strings may be bare percentages (`"8%"`), explicit ranges (`"10% to 25%"`,
`"1-10%"`), or open-ended bounds (`"<5%"`, `">10%"`); each is parsed to a single
midpoint percentage. Terms are matched case-insensitively to the symptoms in your
signal `summary.csv`. As with everything else, these dumps are yours to assemble
from the literature; none ship with the package.

## Annotations (benchmark)

`benchmark` reads a directory of annotator CSVs, one per annotator, each with a
`patient_id` column and a comma-separated `symptoms` column:

```csv
patient_id,symptoms
patient_0001,"headache, nausea"
patient_0002,
```

An empty cell (or `none` / `n/a`) means the annotator recorded no symptoms for
that patient, still a judged patient, just with an empty set. The model outputs
to score against these go in a separate directory, one normalized JSONL per
model.
