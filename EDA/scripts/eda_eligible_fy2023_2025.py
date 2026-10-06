"""Create the initial EDA artifacts for eligible FY2023-FY2025 donor-years.

The script uses only the Python standard library. It filters the enriched
donor-fiscal-year table to eligible scoring observations with a complete
three-year modern history, then writes cohort, target-balance, feature-quality,
categorical-count, and converter-comparison outputs.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


EDA_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = EDA_DIR / "derived_data" / "donor_fiscal_year_features.csv"
COHORT_PATH = EDA_DIR / "derived_data" / "eda_eligible_fy2023_2025.csv"
OUTPUT_DIR = EDA_DIR / "outputs"
TARGET_PATH = OUTPUT_DIR / "target_balance_by_fy_and_segment.csv"
QUALITY_PATH = OUTPUT_DIR / "feature_quality.csv"
CATEGORY_PATH = OUTPUT_DIR / "categorical_counts.csv"
COMPARISON_PATH = OUTPUT_DIR / "key_feature_comparison_by_outcome.csv"
SUMMARY_PATH = OUTPUT_DIR / "initial_eda_summary.md"

EDA_FISCAL_YEARS = {2023, 2024, 2025}
UNIQUE_VALUE_CAP = 1001

COMPARISON_FEATURES = [
    "annual_giving",
    "annual_giving_3yr_total",
    "annual_giving_3yr_mean",
    "direct_giving_3yr_total",
    "soft_credit_giving_3yr_total",
    "recurring_giving_3yr_total",
    "nonrecurring_giving_3yr_total",
    "direct_payment_count_3yr",
    "giving_years_3yr",
    "max_direct_payment_3yr",
    "giving_change_vs_prior_fy",
    "years_since_last_positive_giving",
    "threshold_gap_current_fy",
    "campaign_solicitations_3yr",
    "campaign_responses_3yr",
    "campaign_response_rate_3yr",
    "major_gift_solicitations_3yr",
    "upgrade_solicitations_3yr",
    "passport_active_years_3yr",
    "passport_titles_viewed_3yr",
    "passport_genres_viewed_3yr",
    "passport_weighted_mean_percent_watched_3yr",
    "cultivation_contacted_3yr",
    "legacy_linked_gift_rows",
    "legacy_linked_giving_years",
    "years_since_last_legacy_gift",
]

CATEGORICAL_FEATURES = [
    "fiscal_year",
    "eligibility_segment",
    "primary_constituent_type",
    "address_state",
    "gender",
    "next_fy_conversion",
    "passport_observed_fy",
    "has_linked_legacy_history",
    "cultivation_selected_fy",
]


def is_cohort_row(row: dict[str, str]) -> bool:
    """Return whether a row belongs to the agreed primary EDA cohort."""
    return (
        row["eligible_for_next_fy_scoring"] == "True"
        and int(row["fiscal_year"]) in EDA_FISCAL_YEARS
        and int(row["rolling_years_available"]) == 3
        and row["next_fy_outcome_observed"] == "True"
    )


def quantile(sorted_values: list[float], probability: float) -> float:
    """Calculate a linearly interpolated quantile from sorted values."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * fraction


def fmt_number(value: float) -> str:
    """Format a summary value for CSV output."""
    return f"{value:.6f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    donor_ids: set[str] = set()
    donor_row_counts = Counter()
    target_counts = Counter()
    target_by_fy_segment = Counter()
    missing_counts = Counter()
    unique_values: dict[str, set[str]] = defaultdict(set)
    category_counts = Counter()
    comparison_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    comparison_missing = Counter()

    with INPUT_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])

        with COHORT_PATH.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if not is_cohort_row(row):
                    continue

                writer.writerow(row)
                row_count += 1
                donor_id = row["constituent_id"]
                donor_ids.add(donor_id)
                donor_row_counts[donor_id] += 1

                outcome = row["next_fy_conversion"]
                fy = row["fiscal_year"]
                segment = row["eligibility_segment"]
                target_counts[outcome] += 1
                target_by_fy_segment[(fy, segment, outcome)] += 1

                for field in fieldnames:
                    value = row[field]
                    if value == "":
                        missing_counts[field] += 1
                    if len(unique_values[field]) < UNIQUE_VALUE_CAP:
                        unique_values[field].add(value)

                for field in CATEGORICAL_FEATURES:
                    category_counts[(field, row[field])] += 1

                for field in COMPARISON_FEATURES:
                    value = row[field]
                    if value == "":
                        comparison_missing[(field, outcome)] += 1
                    else:
                        comparison_values[(field, outcome)].append(float(value))

    write_target_balance(target_by_fy_segment)
    write_feature_quality(
        fieldnames,
        row_count,
        missing_counts,
        unique_values,
    )
    write_categorical_counts(category_counts, row_count)
    write_feature_comparison(comparison_values, comparison_missing)
    write_summary(
        row_count,
        donor_ids,
        donor_row_counts,
        target_counts,
        target_by_fy_segment,
        fieldnames,
        missing_counts,
        unique_values,
    )

    print(f"Cohort output: {COHORT_PATH}")
    print(f"EDA output directory: {OUTPUT_DIR}")
    print(f"Rows: {row_count:,}")
    print(f"Unique donors: {len(donor_ids):,}")
    print(f"Converters: {target_counts['True']:,}")
    print(f"Non-converters: {target_counts['False']:,}")
    print(
        "Conversion rate: "
        f"{target_counts['True'] / row_count:.6%}"
    )
    print("Donor observations per cohort count:")
    for observation_count, donors in sorted(
        Counter(donor_row_counts.values()).items()
    ):
        print(f"  {observation_count} year(s): {donors:,} donors")


def write_target_balance(target_by_fy_segment: Counter) -> None:
    """Write target counts and rates by fiscal year and eligibility segment."""
    with TARGET_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "fiscal_year",
            "eligibility_segment",
            "observations",
            "conversions",
            "non_conversions",
            "conversion_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        groups = sorted({
            (fy, segment)
            for fy, segment, _ in target_by_fy_segment
        })
        for fy, segment in groups:
            conversions = target_by_fy_segment[(fy, segment, "True")]
            non_conversions = target_by_fy_segment[(fy, segment, "False")]
            total = conversions + non_conversions
            writer.writerow(
                {
                    "fiscal_year": fy,
                    "eligibility_segment": segment,
                    "observations": total,
                    "conversions": conversions,
                    "non_conversions": non_conversions,
                    "conversion_rate": fmt_number(conversions / total),
                }
            )


def write_feature_quality(
    fieldnames: list[str],
    row_count: int,
    missing_counts: Counter,
    unique_values: dict[str, set[str]],
) -> None:
    """Write missingness and capped cardinality for every field."""
    with QUALITY_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames_out = [
            "feature",
            "missing_count",
            "missing_rate",
            "observed_unique_values_capped",
            "unique_value_count_is_capped",
            "constant_in_cohort",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames_out)
        writer.writeheader()

        for field in fieldnames:
            unique_count = len(unique_values[field])
            capped = unique_count >= UNIQUE_VALUE_CAP
            writer.writerow(
                {
                    "feature": field,
                    "missing_count": missing_counts[field],
                    "missing_rate": fmt_number(missing_counts[field] / row_count),
                    "observed_unique_values_capped": unique_count,
                    "unique_value_count_is_capped": capped,
                    "constant_in_cohort": not capped and unique_count == 1,
                }
            )


def write_categorical_counts(category_counts: Counter, row_count: int) -> None:
    """Write frequency tables for selected categorical fields."""
    with CATEGORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["feature", "value", "count", "share"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for field in CATEGORICAL_FEATURES:
            values = [
                (value, count)
                for (feature, value), count in category_counts.items()
                if feature == field
            ]
            for value, count in sorted(values, key=lambda item: (-item[1], item[0])):
                writer.writerow(
                    {
                        "feature": field,
                        "value": value or "<missing>",
                        "count": count,
                        "share": fmt_number(count / row_count),
                    }
                )


def write_feature_comparison(
    comparison_values: dict[tuple[str, str], list[float]],
    comparison_missing: Counter,
) -> None:
    """Write robust univariate summaries by conversion outcome."""
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "feature",
            "next_fy_conversion",
            "nonmissing_count",
            "missing_count",
            "mean",
            "minimum",
            "p25",
            "median",
            "p75",
            "maximum",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for feature in COMPARISON_FEATURES:
            for outcome in ["False", "True"]:
                values = comparison_values[(feature, outcome)]
                values.sort()
                if values:
                    summary = {
                        "mean": fmt_number(fmean(values)),
                        "minimum": fmt_number(values[0]),
                        "p25": fmt_number(quantile(values, 0.25)),
                        "median": fmt_number(quantile(values, 0.50)),
                        "p75": fmt_number(quantile(values, 0.75)),
                        "maximum": fmt_number(values[-1]),
                    }
                else:
                    summary = {
                        "mean": "",
                        "minimum": "",
                        "p25": "",
                        "median": "",
                        "p75": "",
                        "maximum": "",
                    }
                writer.writerow(
                    {
                        "feature": feature,
                        "next_fy_conversion": outcome,
                        "nonmissing_count": len(values),
                        "missing_count": comparison_missing[(feature, outcome)],
                        **summary,
                    }
                )


def write_summary(
    row_count: int,
    donor_ids: set[str],
    donor_row_counts: Counter,
    target_counts: Counter,
    target_by_fy_segment: Counter,
    fieldnames: list[str],
    missing_counts: Counter,
    unique_values: dict[str, set[str]],
) -> None:
    """Write a concise Markdown record of the initial EDA cohort audit."""
    constant_fields = [
        field
        for field in fieldnames
        if len(unique_values[field]) == 1
    ]
    high_missing_fields = sorted(
        [
            (field, missing_counts[field] / row_count)
            for field in fieldnames
            if missing_counts[field] / row_count >= 0.20
        ],
        key=lambda item: (-item[1], item[0]),
    )
    observation_distribution = Counter(donor_row_counts.values())

    lines = [
        "# Initial EDA cohort summary",
        "",
        "## Cohort definition",
        "",
        "- Eligible for next-fiscal-year scoring",
        "- Fiscal years 2023 through 2025",
        "- Three complete years of modern giving history",
        "- Observed next-fiscal-year outcome",
        "",
        "## Cohort size",
        "",
        f"- Donor-year observations: {row_count:,}",
        f"- Unique donors: {len(donor_ids):,}",
        f"- Conversions: {target_counts['True']:,}",
        f"- Non-conversions: {target_counts['False']:,}",
        (
            "- Conversion rate: "
            f"{target_counts['True'] / row_count:.4%}"
        ),
        "",
        "## Repeated donor observations",
        "",
    ]
    for years, donors in sorted(observation_distribution.items()):
        lines.append(f"- {years} cohort year(s): {donors:,} donors")

    lines.extend(
        [
            "",
            "## Target balance by fiscal year and segment",
            "",
            "| Fiscal year | Segment | Observations | Conversions | Rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    groups = sorted({
        (fy, segment)
        for fy, segment, _ in target_by_fy_segment
    })
    for fy, segment in groups:
        conversions = target_by_fy_segment[(fy, segment, "True")]
        non_conversions = target_by_fy_segment[(fy, segment, "False")]
        total = conversions + non_conversions
        lines.append(
            f"| {fy} | {segment} | {total:,} | {conversions:,} | "
            f"{conversions / total:.4%} |"
        )

    lines.extend(["", "## Constant fields in the filtered cohort", ""])
    for field in constant_fields:
        lines.append(f"- `{field}`")

    lines.extend(["", "## Fields with at least 20% blank values", ""])
    for field, missing_rate in high_missing_fields:
        lines.append(f"- `{field}`: {missing_rate:.1%}")

    lines.extend(
        [
            "",
            "Detailed outputs are available in the other CSV files in this directory.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
