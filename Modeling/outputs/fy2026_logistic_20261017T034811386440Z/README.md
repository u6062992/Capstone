# FY2026 logistic prospect scores

> **Synthetic examples:** The project owner confirmed that all data are fictitious
> and approved public publication of these scored records. Identifiers and giving
> fields do not describe real donors. These files are not real outreach lists.

This release scored 38,219 eligible active person-level prospects with the saved
FY2021–FY2025 logistic artifact. It did not refit the model or repeat FY2025
evaluation. Both training-only controls were forced off during scoring; actual
contact/holdout fields remain in the exports for audit context.

- `fy2026_all_prospect_scores.csv`: all scored prospect rows, descending score.
- `fy2026_top1000_person_level.csv`: preliminary overall top-1,000 allocation.
- `scoring_audit.json`: dataset/model hashes, controls, missingness, and run checks.

There are 1,000 rows and 1,000 allocated places in this release's top list; no
cutoff tie required fractional selection. The allocation column is still supplied
to make the convention explicit. Identifier sorting is display-only, not a
selection tiebreaker. Equal person weights are used; historical no-contact weights
are not used for operational list construction.

`score_uncalibrated` is a provisional model probability for ranking, not a
validated individual probability or estimate of contact effects. `rank_min` and
`rank_max` preserve ties. `top1000_allocation` is the fraction of a list place.
Missing FY2026 outcomes are expected: this cohort predicts FY2027 outcomes.
No FY2026 performance measures can yet be computed.

No households were deduplicated, no outreach/contact eligibility filters beyond
the modeled active-donor population were added, and no outreach was sent. Staff
must review household duplicates, suppressions, and eligibility before use.
Import `row_id`, `constituent_id`, and `couple_id` as text to preserve formatting.
These CSVs contain synthetic identifiers and are publicly shared with the
project owner's approval. Do not replace them with real donor information in
this public repository.
