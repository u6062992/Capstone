# PBS Utah Major Donor Prediction Capstone

This capstone explores how PBS Utah could prioritize eligible donors for major-donor outreach. It combines modern and legacy giving history with approved donor features, compares ranking models, and produces a reproducible logistic-regression artifact and FY2026 prospect scores.

> **Synthetic-data notice:** The project owner confirmed that the data are fictitious and approved public publication. Published donor identifiers, giving records, and prospect scores are synthetic examples, not real PBS Utah donor records. Results demonstrate this capstone workflow and should not be interpreted as verified findings about actual donors or used for real outreach.

## Start here: final modeling deliverables

- [Final modeling write-up — HTML](Modeling/reports/final_logistic_writeup_20261017T034811386440Z/final_modeling_writeup.html)
- [Final modeling write-up — Quarto source](Modeling/reports/final_logistic_writeup_20261017T034811386440Z/final_modeling_writeup.qmd) and [included report body](Modeling/reports/final_logistic_writeup_20261017T034811386440Z/_report_body.qmd)
- [Saved final logistic model and loading instructions](Modeling/artifacts/logistic_final_fy2021_2025_20261017T033533058303Z/README.md)
- [Modeling definitions, feature lists, and data-building scripts](Modeling/README.md)
- [All 38,219 synthetic FY2026 prospect scores](Modeling/outputs/fy2026_logistic_20261017T034811386440Z/fy2026_all_prospect_scores.csv)
- [Preliminary top-1,000 synthetic prospects](Modeling/outputs/fy2026_logistic_20261017T034811386440Z/fy2026_top1000_person_level.csv)
- [Scoring instructions and caveats](Modeling/outputs/fy2026_logistic_20261017T034811386440Z/README.md)

GitHub may show HTML as source or a download. Open the downloaded HTML locally to read the formatted report; keep the repository folder structure for its local links.

## Modeling workflow and repository guide

The final modeling workflow is complete: reconstruct donor-year datasets from the
finalized group EDA definitions, compare models using rolling FY2022–FY2024
assessments, run the separate FY2025 disruption check, refit the selected logistic
specification on FY2021–FY2025, and save/reload the model to rank FY2026 prospects.
This is a completed synthetic-data capstone workflow, not a deployed outreach system.

| Location | Contents and purpose |
|---|---|
| [Modeling overview](Modeling/README.md) | Target and eligibility definitions, validation safeguards, reconstruction caveats, and dataset rebuilding instructions |
| [Modeling scripts](Modeling/scripts/) | Dataset construction and safe input loading |
| [Feature manifest](Modeling/feature_manifest.json) and [feature dictionary](Modeling/feature_dictionary.csv) | Approved predictor roles and master-column definitions |
| [Final report](Modeling/reports/final_logistic_writeup_20261017T034811386440Z/final_modeling_writeup.html) | Model comparisons, evaluation results, and limitations |
| [Saved model bundle](Modeling/artifacts/logistic_final_fy2021_2025_20261017T033533058303Z/README.md) | Fitted pipeline and loading instructions |
| [FY2026 outputs](Modeling/outputs/fy2026_logistic_20261017T034811386440Z/README.md) | Synthetic prospect rankings, preliminary top-1,000 list, and scoring caveats |

Start with the final report to review findings, or the Modeling overview to inspect
the methodology. Rebuild-managed datasets and audits remain local and Git-ignored;
the published deliverables do not require those files merely to read the results.

## Selected model and primary performance

The selected model is **untuned logistic regression** with training-fold preprocessing and the approved core features. The target is `converted_1yr`: major status in the next fiscal year among active donors who are not currently major. **Eligible former-major donors are retained.**

Primary evidence comes from exploratory rolling assessments on **FY2022–FY2024**:

| Overall metric | Logistic regression |
|---|---:|
| Recall at 1,000 prospects per year | **80.60%** |
| Precision at 1,000 | **3.60%** |
| Lift at 1,000 | **26.11×** |
| Mean annual average precision | 0.2454 |
| Mean annual ROC AUC | 0.9342 |
| Classification accuracy at a 0.5 threshold | 99.856% |

Recall at 1,000, not accuracy, guided selection. The report compares the dummy, current-giving ranking, logistic regression, random forest, histogram gradient boosting, and the former-major-or-$500 rule, with separate never-major and weighted no-contact evaluations.

### FY2025 is a separate disruption check, not the primary validation set

1. Model selection used rolling FY2022–FY2024 assessments; FY2025 was excluded.
2. After selection, the frozen FY2021–FY2024-trained pipeline was evaluated once on FY2025 as a labeled influx/funding-disruption check. Overall recall was **67.90%**, reported separately.
3. The selected specification was then refitted on **FY2021–FY2025: 160,919 donor-year rows**. FY2025 was not permanently withheld from final training.
4. That saved final model scored FY2026 without another refit or repeat FY2025 evaluation.

Prior full-data EDA occurred, so validation is exploratory rather than an untouched-test claim. The final refit has no independent post-refit performance estimate.

## FY2026 scores and limitations

- **38,219** eligible synthetic prospects were scored; the preliminary overall top list contains **1,000** person-level rows, with no fractional cutoff selection needed in this release.
- FY2026 scores target FY2027 outcomes, which are unknown. No FY2026 performance metrics are available.
- Scores are **uncalibrated model probabilities for ranking**, not validated individual risks or estimates of outreach effects.
- Model improvements were concentrated in former-major capture; better never-major outreach performance was not established.
- The two training-only controls are retained during fitting and forced off during prediction. Actual exposure flags remain separate.
- Household deduplication and contact suppressions were not applied. No outreach was sent and no live deployment occurred.
- The AI-assisted report retains its human-review disclaimer. Synthetic-data publication approval is not a claim of comprehensive analytical review.

## Reproducibility and publication contents

The repository includes the report source/HTML, aggregate result snapshots, portable inference code, fitted model bundle, version requirements, checksums, and synthetic score exports. Only load trusted joblib artifacts and use the recorded package versions.

Raw working datasets, rebuild-managed modeling datasets/audits, editor state, and unrelated local changes are not part of this publication. Rebuilding datasets requires the original synthetic inputs; the published report can be regenerated from its included aggregate snapshots without refitting or rescoring.

The HTML report is a standalone companion, not a native Quarto render. Its Python code cells were executed and checked. To rebuild it with the recorded environment and IPython available:

```bash
python Modeling/reports/final_logistic_writeup_20261017T034811386440Z/build_html_companion.py --overwrite
```

Regenerating files changes checksums; refresh the corresponding checksum manifest before distributing a modified bundle.

## Project documentation

- [Business problem statement (rendered HTML)](pbs_utah_business_problem_statement.html) — View the rendered statement describing the project's business need, objectives, scope, deliverables, assumptions, and success criteria.
- [Business problem statement (Microsoft Word)](pbs_utah_business_problem_statement.docx) — Download or open the editable Word version.

## Exploratory data analysis

The [finalized group EDA](pbs_utah_group_eda.html) governs the modeling target, eligibility, spouse rules, treatment controls, and rolling validation. Earlier individual EDA reports use a different first-time-conversion framing and are retained as historical work, not as the final modeling specification.

- [Open the complete EDA folder](EDA/)
- [Earlier individual EDA report](EDA/pbs_utah_final_eda_report.md)
- [Earlier individual EDA Quarto source](EDA/pbs_utah_final_eda_report.qmd)
