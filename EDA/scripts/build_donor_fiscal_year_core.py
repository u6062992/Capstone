"""Build the core individual donor-fiscal-year analytical table.

This script uses only the Python standard library. It combines positive Unite
payments and deduplicated positive soft credits for individual constituents,
then derives annual major-donor status, first observed conversion, active and
lapsed eligibility, and next-fiscal-year outcomes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path


EDA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
OUTPUT_DIR = EDA_DIR / "derived_data"
OUTPUT_PATH = OUTPUT_DIR / "donor_fiscal_year_core.csv"

FIRST_FY = 2021
LAST_FY = 2026
MAJOR_THRESHOLD = Decimal("1200.00")
ZERO = Decimal("0.00")


def fiscal_year(value: str) -> int:
    """Return the July-June fiscal-year ending year for a date string."""
    parsed = date.fromisoformat(value) if "-" in value else datetime.strptime(
        value,
        "%m/%d/%Y",
    ).date()
    return parsed.year + 1 if parsed.month >= 7 else parsed.year


def money(value: Decimal) -> str:
    """Format a monetary Decimal consistently for CSV output."""
    return f"{value.quantize(Decimal('0.01')):.2f}"


def load_individuals() -> tuple[dict[str, dict[str, str]], set[str]]:
    """Load eligible individual constituent IDs and known organization IDs."""
    individuals: dict[str, dict[str, str]] = {}
    organization_ids: set[str] = set()

    path = DATA_DIR / "constituents_w_memberships.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            constituent_id = row["Constituent ID"]
            if row["Primary Constituent Type"] == "Organization":
                organization_ids.add(constituent_id)
                continue
            individuals[constituent_id] = {
                "primary_constituent_type": row["Primary Constituent Type"],
                "address_state": row["Address State"],
                "gender": row["Gender"],
            }

    return individuals, organization_ids


def new_annual_record() -> dict[str, Decimal | int]:
    """Create an empty annual aggregation record."""
    return {
        "direct_giving": ZERO,
        "recurring_direct_giving": ZERO,
        "outright_direct_giving": ZERO,
        "soft_credit_giving": ZERO,
        "direct_payment_count": 0,
        "recurring_payment_count": 0,
        "outright_payment_count": 0,
        "soft_credit_count": 0,
        "max_direct_payment": ZERO,
    }


def aggregate_giving(
    individuals: dict[str, dict[str, str]],
) -> tuple[dict[tuple[str, int], dict[str, Decimal | int]], set[str], dict[str, int]]:
    """Aggregate positive direct payments and deduplicated soft credits."""
    annual = defaultdict(new_annual_record)
    observed_donors: set[str] = set()
    audit = defaultdict(int)

    payment_path = DATA_DIR / "unite_payments.txt"
    with payment_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit["payment_rows_read"] += 1
            donor_id = row["donor_id"]
            amount = Decimal(row["payment_amount"])

            if donor_id == "SYN-ORPHAN":
                audit["payment_rows_orphan_excluded"] += 1
                continue
            if donor_id not in individuals:
                audit["payment_rows_nonindividual_excluded"] += 1
                continue
            if amount <= ZERO:
                audit["payment_rows_nonpositive_excluded"] += 1
                continue

            fy = fiscal_year(row["credit_date"])
            if not FIRST_FY <= fy <= LAST_FY:
                audit["payment_rows_outside_period_excluded"] += 1
                continue

            record = annual[(donor_id, fy)]
            record["direct_giving"] += amount
            record["direct_payment_count"] += 1
            record["max_direct_payment"] = max(
                record["max_direct_payment"],
                amount,
            )

            if row["pledge_gift_type"] == "Recurring":
                record["recurring_direct_giving"] += amount
                record["recurring_payment_count"] += 1
            else:
                record["outright_direct_giving"] += amount
                record["outright_payment_count"] += 1

            observed_donors.add(donor_id)
            audit["payment_rows_included"] += 1

    soft_path = DATA_DIR / "soft_credits.csv"
    seen_soft_rows: set[tuple[str, ...]] = set()
    with soft_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        for row in reader:
            audit["soft_credit_rows_read"] += 1
            row_key = tuple(row[field] for field in fieldnames)
            if row_key in seen_soft_rows:
                audit["soft_credit_exact_duplicates_excluded"] += 1
                continue
            seen_soft_rows.add(row_key)

            donor_id = row["Soft Credit Constituent ID"]
            amount = Decimal(row["Credit Amount"])
            if donor_id == "SYN-ORPHAN":
                audit["soft_credit_rows_orphan_excluded"] += 1
                continue
            if donor_id not in individuals:
                audit["soft_credit_rows_nonindividual_excluded"] += 1
                continue
            if amount <= ZERO:
                audit["soft_credit_rows_nonpositive_excluded"] += 1
                continue

            fy = fiscal_year(row["Credit Date"])
            if not FIRST_FY <= fy <= LAST_FY:
                audit["soft_credit_rows_outside_period_excluded"] += 1
                continue

            record = annual[(donor_id, fy)]
            record["soft_credit_giving"] += amount
            record["soft_credit_count"] += 1
            observed_donors.add(donor_id)
            audit["soft_credit_rows_included"] += 1

    return annual, observed_donors, dict(audit)


def classify_segment(
    annual_totals: dict[int, Decimal],
    fy: int,
    prior_major: bool,
) -> str:
    """Classify a donor at fiscal year-end for next-year scoring."""
    current = annual_totals[fy]
    prior_positive_years = [
        year
        for year in range(FIRST_FY, fy)
        if annual_totals[year] > ZERO
    ]

    if prior_major:
        return "prior_major_ineligible"
    if current >= MAJOR_THRESHOLD:
        return "first_observed_major_year"
    if ZERO < current < MAJOR_THRESHOLD:
        return "active_normal"

    recent_normal = any(
        ZERO < annual_totals[year] < MAJOR_THRESHOLD
        for year in range(max(FIRST_FY, fy - 3), fy)
    )
    if recent_normal:
        return "lapsed_normal"
    if prior_positive_years:
        return "long_term_inactive"
    return "not_yet_donor"


def write_table(
    individuals: dict[str, dict[str, str]],
    annual: dict[tuple[str, int], dict[str, Decimal | int]],
    observed_donors: set[str],
) -> dict[str, int]:
    """Write the complete donor-fiscal-year core table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_audit = defaultdict(int)

    fieldnames = [
        "constituent_id",
        "fiscal_year",
        "primary_constituent_type",
        "address_state",
        "gender",
        "direct_giving",
        "recurring_direct_giving",
        "outright_direct_giving",
        "soft_credit_giving",
        "annual_giving",
        "nonrecurring_recognized_giving",
        "direct_payment_count",
        "recurring_payment_count",
        "outright_payment_count",
        "soft_credit_count",
        "max_direct_payment",
        "major_in_fy",
        "first_observed_major_in_fy",
        "recurring_alone_reaches_threshold",
        "left_censored_major_at_entry",
        "prior_observed_major",
        "eligibility_segment",
        "eligible_for_next_fy_scoring",
        "next_fy_conversion",
        "next_fy_outcome_observed",
        "history_years_available",
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for donor_id in sorted(observed_donors):
            annual_totals: dict[int, Decimal] = {}
            for fy in range(FIRST_FY, LAST_FY + 1):
                record = annual[(donor_id, fy)]
                annual_totals[fy] = (
                    record["direct_giving"] + record["soft_credit_giving"]
                )

            first_major_fy = next(
                (
                    fy
                    for fy in range(FIRST_FY, LAST_FY + 1)
                    if annual_totals[fy] >= MAJOR_THRESHOLD
                ),
                None,
            )

            for fy in range(FIRST_FY, LAST_FY + 1):
                record = annual[(donor_id, fy)]
                annual_giving = annual_totals[fy]
                prior_major = first_major_fy is not None and first_major_fy < fy
                first_major = first_major_fy == fy
                segment = classify_segment(annual_totals, fy, prior_major)
                outcome_observed = fy < LAST_FY
                next_conversion = (
                    outcome_observed
                    and first_major_fy == fy + 1
                    and not prior_major
                )
                eligible = (
                    outcome_observed
                    and segment in {"active_normal", "lapsed_normal"}
                )
                nonrecurring = (
                    record["outright_direct_giving"]
                    + record["soft_credit_giving"]
                )

                writer.writerow(
                    {
                        "constituent_id": donor_id,
                        "fiscal_year": fy,
                        **individuals[donor_id],
                        "direct_giving": money(record["direct_giving"]),
                        "recurring_direct_giving": money(
                            record["recurring_direct_giving"]
                        ),
                        "outright_direct_giving": money(
                            record["outright_direct_giving"]
                        ),
                        "soft_credit_giving": money(
                            record["soft_credit_giving"]
                        ),
                        "annual_giving": money(annual_giving),
                        "nonrecurring_recognized_giving": money(nonrecurring),
                        "direct_payment_count": record["direct_payment_count"],
                        "recurring_payment_count": record[
                            "recurring_payment_count"
                        ],
                        "outright_payment_count": record[
                            "outright_payment_count"
                        ],
                        "soft_credit_count": record["soft_credit_count"],
                        "max_direct_payment": money(
                            record["max_direct_payment"]
                        ),
                        "major_in_fy": annual_giving >= MAJOR_THRESHOLD,
                        "first_observed_major_in_fy": first_major,
                        "recurring_alone_reaches_threshold": (
                            record["recurring_direct_giving"]
                            >= MAJOR_THRESHOLD
                        ),
                        "left_censored_major_at_entry": (
                            first_major_fy == FIRST_FY
                        ),
                        "prior_observed_major": prior_major,
                        "eligibility_segment": segment,
                        "eligible_for_next_fy_scoring": eligible,
                        "next_fy_conversion": next_conversion,
                        "next_fy_outcome_observed": outcome_observed,
                        "history_years_available": fy - FIRST_FY + 1,
                    }
                )
                output_audit["rows_written"] += 1
                output_audit[f"segment_{segment}"] += 1
                output_audit["eligible_rows"] += int(eligible)
                output_audit["next_fy_conversions"] += int(next_conversion)

    output_audit["observed_individual_donors"] = len(observed_donors)
    return dict(output_audit)


def main() -> None:
    individuals, organization_ids = load_individuals()
    annual, observed_donors, source_audit = aggregate_giving(individuals)
    output_audit = write_table(individuals, annual, observed_donors)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Individual constituents available: {len(individuals):,}")
    print(f"Organization constituents excluded: {len(organization_ids):,}")
    print("\nSource audit")
    for key in sorted(source_audit):
        print(f"  {key}: {source_audit[key]:,}")
    print("\nOutput audit")
    for key in sorted(output_audit):
        print(f"  {key}: {output_audit[key]:,}")


if __name__ == "__main__":
    main()
