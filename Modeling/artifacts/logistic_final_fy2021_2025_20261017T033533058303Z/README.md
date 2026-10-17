# PBS Utah: selected logistic regression, final refit

> **Synthetic-data publication:** The project owner confirmed that the inputs are
> fictitious and authorized public sharing of this model and its supporting
> results. This artifact is a capstone demonstration, not a deployed model for
> real donors. Publication metadata has machine-specific paths redacted; the
> fitted model, feature specification, and evaluation results are unchanged.

## Status and intended use

This bundle contains the selected, untuned logistic-regression specification,
refitted on eligible active FY2021–FY2025 donor-year rows for `converted_1yr`.
Eligible former-major donors remain included. Lapsed donors are not included.
The target is major status in the next fiscal year, not first-ever major status.

The artifact contains fitted preprocessing and the classifier together. It has
not been deployed, and no FY2026 prospect list was produced during this refit.
See `metadata.json` for the final status, data provenance, settings, and versions.

## Frozen specification

- Fourteen approved core inputs plus `contacted_t` and `heldout_t`.
- Numeric/binary inputs: numeric conversion, training-median imputation, scaling.
- `gender` and `giving_change_band`: object conversion, constant `Unknown`
  imputation, one-hot encoding with unseen levels ignored.
- Logistic regression: L2 regularization, C=1, lbfgs, maximum 1,000 iterations,
  no class weights, no resampling, and no hyperparameter tuning.
- Actual treatment controls are retained during fitting. Both are forcibly
  neutralized by public prediction methods, including historical scoring.
- No calibration model was fitted. Saving/portability changes do not alter the
  selected preprocessing, regularization, feature set, or scoring rule.

## Evaluation provenance and limitations

Development used saved temporal splits: FY2021 to FY2022, FY2021–FY2022 to
FY2023, and FY2021–FY2023 to FY2024. Prior full-data EDA occurred, so these
results are exploratory validation, not an untouched-test estimate.

After the user selected logistic regression, the frozen pipeline was fitted on
FY2021–FY2024 and evaluated once on FY2025 as the labeled funding-disruption
check. The saved FY2025 results belong to that earlier development-trained
model, not to this FY2021–FY2025 refit. FY2025 was not used to tune parameters
or fit a calibrator. Its outcomes enter training only in the approved final refit.
No post-refit generalization estimate is available.

- Primary metric: recall at 1,000 per assessment cohort; cutoff ties fractional.
- Overall and never-major evaluations use separate 1,000-prospect lists.
- No-contact evaluation excludes actual contacted donors and uses saved holdout
  weights in both ranking allocation and metrics; it is not an uplift analysis.
- Intervals are 1,000 couple-clustered bootstrap replicates, conditional on fitted
  predictions; they do not incorporate the full model-selection process.
- The model's overall advantage in development was concentrated in former-major
  capture. Improvement for never-major outreach was not established.
- FY2025 overall recall was 67.90%, versus 80.60% pooled development recall.
- FY2025 probabilities underpredicted conversion on average: overall mean
  prediction 0.1081% versus observed 0.2463%; no-contact overall 0.0961% versus
  0.1590%. Refitting does not establish that calibration is repaired.
- Treat outputs as ranking scores/provisional probabilities, not validated
  individual conversion probabilities or estimates of contact effects.
- This dataset builder is a reconstruction with reconciled counts, not a claim
  of identical original features. Age is reconstructed from the FY2026 snapshot;
  pre-FY1999 records contribute to observed giving history and former-major flags.
- Prospect rows are person-level. Household deduplication remains an operational
  decision and is not performed by this artifact.

## Loading and scoring

Use the recorded package versions in `requirements.txt` and the Python version
in `metadata.json`. Compatibility across library versions is not guaranteed.
Keep the entire bundle together. Only load trusted artifacts: joblib uses pickle
and can execute code during deserialization. Checksums detect corruption, not
malicious replacement of both an artifact and its checksum file.

```python
import sys
from pathlib import Path

bundle = Path("path/to/this/bundle").resolve()
sys.path.insert(0, str(bundle))
from pbs_utah_logistic_artifact import load_pipeline

model = load_pipeline(bundle)
# X must be a pandas DataFrame with exactly the approved feature columns.
# Use the project's safe loader with purpose="predict" to construct X.
probabilities = model.predict_proba(X)[:, 1]
```

Input columns may be reordered, but missing, duplicate, or extra columns are
rejected. The wrapper copies its input and switches both treatment controls off
even if they arrive as True. Actual exposure flags must be retained separately
for any evaluation; do not derive exposure from neutralized model inputs.

Do not bypass the control-safe prediction methods or directly score with the
inner classifier. No donor-level training data, IDs, or predictions are exported
in this bundle. Inspect `reload_validation.json` for the portability checks.

## Contents

- `model.joblib`: fitted complete control-safe pipeline.
- `pbs_utah_logistic_artifact.py`: importable preprocessing and inference helpers.
- `metadata.json`, `feature_manifest.json`, `feature_schema.csv`: frozen settings
  and provenance.
- `development_logistic_*.csv`: existing exploratory results for the selected model.
- `fy2025_*.csv`: previously completed one-time check and calibration results.
- `requirements.txt`, `reload_validation.json`, `checksums.json`: reproducibility
  and artifact-integrity records.
