# PBS Utah Major Donor Prediction Capstone

This capstone project supports PBS Utah in identifying current donors who are most likely to become major donors. It combines modern and legacy donation records, campaign activity, constituent information, and PBS Passport viewing behavior to support donor prioritization, forecast prospect volume, and estimate when donors may make another discretionary gift.

## Project documentation

- [Business problem statement (rendered HTML)](pbs_utah_business_problem_statement.html) — View the rendered statement describing the project's business need, objectives, scope, deliverables, assumptions, and success criteria.
- [Business problem statement (Microsoft Word)](pbs_utah_business_problem_statement.docx) — Download or open the editable Word version.

## Exploratory data analysis

The EDA uses an individual donor–fiscal-year cohort and defines major-donor
conversion as the first fiscal year in which recognized giving reaches at least
$1,200. The FY2023–FY2025 cohort contains 154,247 donor-year observations,
60,372 unique donors, and 101 conversions. Recent giving level was the strongest
simple conversion signal, and donor-clustered bootstrap analysis confirmed that
the higher-giving tiers remained well above baseline after repeated donor
observations were considered.

- [Open the complete EDA folder](EDA/)
- [View the rendered final EDA report](EDA/pbs_utah_final_eda_report.html)
- [View the Quarto report source](EDA/pbs_utah_final_eda_report.qmd)
