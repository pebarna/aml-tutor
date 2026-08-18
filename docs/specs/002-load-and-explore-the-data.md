# Load and explore the data

Get the real PaySim transactions into a DataFrame you can trust, and see the number that will
shape every decision for the rest of this tutorial: the fraud rate.

## Key concept

**At roughly 0.1% fraud prevalence, accuracy is meaningless.** A model that predicts "never fraud"
for every transaction scores above 99.8% accuracy on PaySim, while catching exactly zero fraud —
the one outcome the classifier exists to catch. Accuracy answers "how often is the model right?",
which on a dataset this imbalanced is dominated by the majority class no matter what the model
does about the minority class you actually care about. It cannot distinguish a useless model from
a useful one.

This is why the rest of the tutorial reports precision (of the transactions flagged as fraud, how
many actually were?) and recall (of the transactions that actually were fraud, how many did the
model flag?) instead. Neither metric can be gamed by the "always predict the majority class"
trick: a never-fraud model has undefined precision (it flags nothing) and zero recall (it catches
nothing). Both numbers reveal the failure that accuracy hides.

Before any of that is measurable, the data has to load correctly — right columns, right types, and
a fraud rate you have independently checked rather than assumed.

## Implementation order

1. **Download the full dataset.** Get "Synthetic Financial Datasets For Fraud Detection"
   (`ealaxi/paysim1`) from Kaggle and save it somewhere outside this repository — it is a few
   hundred MB, too large to commit, and the tutorial's fixture in `aml-tutor/tests/fixtures/` is a
   small carved-out slice, not a substitute for the real thing.
2. **Sample it down with a fixed seed.** From `aml-triage`, run:

   ```sh
   uv run python -c "import pandas as pd; df = pd.read_csv('<full path to the downloaded CSV>'); df.sample(n=200000, random_state=20260818).to_csv('data/paysim_sample.csv', index=False)"
   ```

   The `random_state` is fixed so that re-running this command reproduces the exact same sample —
   without it, every run would give you a different 200,000 rows and every later result would be
   irreproducible. This writes `aml-triage/data/paysim_sample.csv`. Add `data/` to
   `aml-triage/.gitignore` before you do anything else: this file is large and derived, and belongs
   next to your code but not inside your git history.
3. **Implement the loader.** Write `load_transactions(path) -> DataFrame` in
   `src/aml_triage/data.py`. It reads the CSV, checks that the columns you need are present, and
   coerces the numeric columns to explicit dtypes rather than trusting whatever pandas inferred —
   a column pandas reads as `object` because of one stray blank value will silently break every
   later lesson's arithmetic on it.
4. **Load, validate, and look at the fraud rate.** Call `load_transactions` on
   `data/paysim_sample.csv`, confirm the columns and dtypes are what you expect, and print
   `df["isFraud"].mean()`. Write that number down — it is the baseline every precision/recall
   result in later lessons has to be read against.

### If you ask the tutor to do this step for you

The doer has no shell access, so it cannot download the Kaggle dataset, run the sampling command,
or create `data/paysim_sample.csv` — you have to run steps 1 and 2 yourself, the same way lesson
001's `uv init` had to be yours to run. What the doer can do is write `src/aml_triage/data.py` by
hand, matching the `load_transactions(path) -> DataFrame` signature above. It cannot verify that
implementation against your actual sample file, since that file only exists once you have created
it — the baked-in check below is what verifies it, against the tutorial's fixture, once you run it.

## Checks

Ask the learner to answer these in their own words:

- Why would "99.8% accuracy" be a meaningless claim about this model, given what you know about
  the fraud rate?
- Why does a model that always predicts "not fraud" get zero recall and undefined precision,
  instead of some other number that also looks respectable?
- Why coerce dtypes explicitly in `load_transactions` instead of trusting whatever `pd.read_csv`
  infers?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/002_test_data_loading.py
```

```json validation
[
  {
    "id": "002-load-and-explore-the-data",
    "label": "Load and explore the data",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/002_test_data_loading.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

`load_transactions` coerces numeric columns to explicit dtypes, but `step` — the column lesson
003's entire split depends on — is exactly the kind of column that can silently come back as the
wrong dtype: one stray blank or non-numeric value in the source CSV is enough for `pd.read_csv` to
infer `object` (string) instead of an integer. If `load_transactions` didn't catch that, lesson
003's `temporal_split` would still run without error — `df[df["step"] <= split_step]` is valid on
strings too — but the comparison would be lexicographic, not numeric: `"9" <= "10"` is `False`
under string comparison even though `9 <= 10` is `True` under integer comparison. Rows would land
on the wrong side of the boundary in ways that don't show up as a crash, just as a split that's
subtly wrong. That failure would look like a bug in the split logic, not in the loading step that
actually caused it, because it traces back to a dtype `load_transactions` never validated. This is
exactly why lesson 002's dtype checks have to happen here, before lesson 003 ever runs.
