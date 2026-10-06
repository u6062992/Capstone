# PBS Utah exploratory data analysis

This folder contains the final exploratory data analysis for the PBS Utah
major-donor conversion project. The analysis uses an individual donor–fiscal-
year cohort and defines conversion as the first fiscal year in which recognized
giving reaches at least $1,200.

## Final report

- [Rendered Markdown report](pbs_utah_final_eda_report.md)
- [Quarto source](pbs_utah_final_eda_report.qmd)

## Key findings

- The FY2023–FY2025 EDA cohort contains 154,247 donor-year observations,
  60,372 unique donors, and 101 conversions.
- Ninety-nine conversions came from active donors; only two came from lapsed
  donors.
- Current annual giving of $250–$499 produced approximately 9.6 times baseline
  lift.
- Current annual giving of $500–$1,199 produced approximately 108.8 times
  baseline lift.
- Donor-clustered bootstrap intervals confirmed that actionable high-giving
  tiers remain above baseline after repeated donor observations are considered.

## Folder structure

- `pbs_utah_final_eda_report.qmd` — final report source
- `pbs_utah_final_eda_report.md` — rendered Markdown report for GitHub
- `scripts/` — standard-library Python scripts used to construct and analyze
  the donor-year data
- `outputs/` — EDA summaries, lift tables, recommendations, and SVG figures
- `derived_data/` — locally generated donor-level CSVs; intentionally excluded
  from Git because they are reproducible and one exceeds GitHub's file limit
- `ARTIFACTS.md` — reproducibility order and publication manifest

## Reproduce the analysis

From the repository root, run:

```bash
python EDA/scripts/build_donor_fiscal_year_core.py
python EDA/scripts/build_donor_fiscal_year_features.py
python EDA/scripts/eda_eligible_fy2023_2025.py
python EDA/scripts/create_initial_eda_visualizations.py
python EDA/scripts/bootstrap_clustered_lift.py
quarto render EDA/pbs_utah_final_eda_report.qmd
```

The scripts use only the Python standard library. Quarto is required only to
render the final report.
