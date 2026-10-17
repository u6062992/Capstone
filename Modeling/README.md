# Modeling and FY2026 scoring

These datasets were rebuilt from the raw inputs using the finalized
[group EDA](../pbs_utah_group_eda.html). They do **not** use the earlier individual
EDA's first-time-only target. The selected logistic model has now been fitted,
saved, reload-verified, and used for FY2026 scoring.

> **Synthetic-data publication:** The project owner confirmed that the data are
> fictitious and authorized public sharing, including the fitted model and scored
> prospect examples. These are not real donor records or operational outreach lists.

## Final deliverables

- [Final modeling write-up (HTML)](reports/final_logistic_writeup_20261017T034811386440Z/final_modeling_writeup.html)
- [Quarto report source](reports/final_logistic_writeup_20261017T034811386440Z/final_modeling_writeup.qmd)
- [Saved model and loading instructions](artifacts/logistic_final_fy2021_2025_20261017T033533058303Z/README.md)
- [FY2026 synthetic scoring outputs](outputs/fy2026_logistic_20261017T034811386440Z/README.md)

Primary exploratory validation is the FY2022–FY2024 rolling assessment: logistic
overall recall@1,000 is **80.60%**, precision **3.60%**, and lift **26.11×**.
FY2025 was assessed once only after selection, as a separate labeled disruption
check, then included in the final FY2021–FY2025 refit. Its 67.90% recall does not
replace the primary validation estimate and is not a post-refit performance claim.

The final artifact was trained on 160,919 donor-year rows and scored 38,219 FY2026
synthetic prospects. The preliminary top list is person-level and not household
deduplicated. Calibration and never-major limitations remain documented. No
FY2026 outcome evaluation, contact sending, or live deployment was performed.

## Local datasets and rebuilding

The dataset links below refer to rebuild-managed local files, not files shipped
in the public publication set. Rebuilding requires the original synthetic inputs.

Use [development_1yr_fy2021_2024.parquet](data/development_1yr_fy2021_2024.parquet)
for initial development. Keep FY2025 out of model selection. Use the
[safe loading helper](scripts/load_modeling_inputs.py) to select approved
predictors rather than passing every column to a model.

| Dataset | Donor-year rows | Positive outcomes | Use |
|---|---:|---:|---|
| [Development, FY2021–2024](data/development_1yr_fy2021_2024.parquet) | 128,033 | 189 | One-year training and rolling testing |
| [Funding check, FY2025](data/funding_check_1yr_fy2025.parquet) | 32,886 | 81 | One labeled check after model selection |
| [Final training, FY2021–2025](data/final_training_1yr_fy2021_2025.parquet) | 160,919 | 270 | Retrain only after evaluation |
| [Three-year, FY2021–2023](data/active_3yr_fy2021_2023.parquet) | 95,249 | 343 | Complete three-year outcomes |
| [Five-year, FY2021](data/active_5yr_fy2021.parquet) | 30,843 | 169 | Complete five-year outcomes; only one modern cohort |
| [Scoring, FY2026](data/scoring_fy2026.parquet) | 38,219 | Unknown | Future prospect ranking |
| [Lapsed reporting](data/lapsed_reporting_fy2021_2025.parquet) | 70,668 | 9 | Reporting only; not primary model training |

Each of those seven datasets has both Parquet and CSV versions. Parquet preserves
nullable outcomes and column types; CSV blanks must not be converted to zeros.
The complete list, sizes, hashes and row counts are in
[dataset_inventory.csv](dataset_inventory.csv).

Additional Parquet files provide the positive-giving person-year ledger, modern
eligible-donor master, historical giving-only benchmark data, and
[rolling split membership](data/rolling_split_membership.parquet).

## Agreed definitions

- Individual donor-year snapshots at June 30; fiscal years are July–June.
- Active eligibility: positive giving in FY t, but not major in t. Former majors
  **remain eligible**, with a separate `former_major` flag.
- Recognized giving: positive direct individual payments plus assigned soft
  credits; linked legacy dollars go once to the first-listed individual.
- Major status: at least $1,200 individually, or combined spouse giving when
  neither spouse individually qualifies. Both receive combined-only labels.
- Lapsed: no gift in t, but a gift in t−1 or t−2.
- Targets: major in the next 1, 3 or 5 fiscal years. Incomplete windows remain
  missing, including **every FY2026 target**.

## Predictor and validation safeguards

The [feature manifest](feature_manifest.json) defines core, optional,
training-only and historical-giving-only inputs. The
[feature dictionary](feature_dictionary.csv) documents all 76 master columns.
Outcome amounts, conversion tiers, IDs, evaluation weights, and snapshot
classification fields are not default predictors.

- Train on earlier years and assess FY2022, FY2023, and FY2024 separately.
- Tune only within each training subset, grouping by `couple_id` so linked
  spouses and repeated donor records remain together in inner folds. Outer
  temporal tests intentionally can include donors seen in earlier training years.
- FY2025 is a disruption check, not a model-selection fold.
- Training controls `contacted_t` and `heldout_t` must be set to False whenever
  ranking prospects, including historical assessment. Actual flags remain for
  no-contact evaluation. The loader applies this with `purpose="predict"`.
- Missing Passport measurements remain missing; `viewed_t` distinguishes record
  availability. FY2021/FY2022 campaign lookbacks are partially covered and flagged.
- Dataset construction performs no fitted imputation, scaling, encoding,
  oversampling, or feature selection. Model preprocessing was learned inside each
  training fold; the final artifact refits its preprocessing on FY2021–FY2025.
- Longer-horizon temporal training must satisfy `training t + horizon <= test t`.
  The [historical split plan](historical_split_plan.csv) implements that condition.
  Its giving-only feature list excludes snapshot demographics and modern engagement.

Example from the repository root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Modeling/scripts").resolve()))
from load_modeling_inputs import load_inputs

X, y, groups, metadata = load_inputs()
# Apply rolling split membership before fitting; groups is for inner grouped CV.
# When ready to score, purpose="predict" neutralizes treatment controls.
```

## Reconciliation and reconstruction caveats

All report-count acceptance checks passed; see
[report_reconciliation.csv](audit/report_reconciliation.csv). Removed copy rows
were verified against valid content twins, not deleted solely by synthetic ID.
Raw inputs and previous EDA outputs were unchanged. Input hashes, missingness,
removal provenance and build checks are under [audit/](audit/).

The original group builder was unavailable. This is a reconstruction with
matching cohort counts, **not a claim of identical feature values**. Explicit
conventions requiring review against the original builder when available:

- Retain pre-FY1999 records for observed history and former-major flags; mark
  affected rows with `history_pre1999` for sensitivity analysis.
- Interpret tenure inclusively; reconstruct age from the FY2026 snapshot.
- Define a multi-year conversion tier from the first qualifying year. Observed
  conversion-year giving is not an estimated optimal ask amount.
- Preserve person-level prospect rows; `couple_id` groups linked spouses. Whether
  the final outreach list should deduplicate households remains an operational decision.

## Rebuild

```bash
python Modeling/scripts/build_group_modeling_data.py
```

Dependencies are pandas, NumPy and PyArrow (already available for this build).
A rebuild invalidates readiness first. The safe loader refuses files if the build
has not completed or their hashes differ from the inventory. Generated data and
audit files remain locally Git-ignored. The publication set includes reports,
source code, aggregate evaluation tables, the fitted model bundle, and the
explicitly authorized synthetic FY2026 prospect exports. Editor state, temporary
files, and the modified duplicate report are excluded. No credentials should be
included in this repository, regardless of the synthetic status of the data.
