"""Enrich the core individual donor-fiscal-year analytical table.

The output preserves every core row and adds leak-free features available at
the end of each scoring fiscal year. Rolling features use the current and two
prior fiscal years. Legacy Team Approach gifts contribute relationship-history
indicators only; household-level gift amounts are never assigned to people.
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
DERIVED_DIR = EDA_DIR / "derived_data"
CORE_PATH = DERIVED_DIR / "donor_fiscal_year_core.csv"
OUTPUT_PATH = DERIVED_DIR / "donor_fiscal_year_features.csv"

FIRST_FY = 2021
LAST_FY = 2026
MAJOR_THRESHOLD = Decimal("1200.00")
ZERO = Decimal("0.00")

CAMPAIGN_SOLICITATIONS = 0
CAMPAIGN_RESPONSES = 1
CAMPAIGN_MAJOR = 2
CAMPAIGN_UPGRADE = 3
CAMPAIGN_RENEWAL = 4
CAMPAIGN_LAPSED = 5
CAMPAIGN_ACQUISITION = 6
CAMPAIGN_DIGITAL = 7
CAMPAIGN_WIDTH = 8


def fiscal_year(value: str) -> int:
    """Return the July-June fiscal-year ending year for a date string."""
    parsed = date.fromisoformat(value) if "-" in value else datetime.strptime(
        value,
        "%m/%d/%Y",
    ).date()
    return parsed.year + 1 if parsed.month >= 7 else parsed.year


def is_true(value: str) -> bool:
    """Parse common textual Boolean values."""
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def money(value: Decimal) -> str:
    """Format a monetary Decimal consistently for CSV output."""
    return f"{value.quantize(Decimal('0.01')):.2f}"


def rate(numerator: int, denominator: int) -> str:
    """Format a rate, treating no opportunities as a zero rate."""
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def load_core_donors() -> tuple[set[str], list[str]]:
    """Load donor IDs and the current core column order."""
    donor_ids: set[str] = set()
    with CORE_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        core_fields = list(reader.fieldnames or [])
        for row in reader:
            donor_ids.add(row["constituent_id"])
    return donor_ids, core_fields


def load_campaign_code_flags() -> dict[str, tuple[bool, ...]]:
    """Map each campaign code to modeling category flags."""
    flags: dict[str, tuple[bool, ...]] = {}
    path = DATA_DIR / "campaign_codes.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            activity_type = row["Activity Type"].strip()
            campaign_type = row["Campaign Type"].strip()
            communication_method = row["Communication Method"].strip()
            flags[row["Marketing Code"]] = (
                activity_type == "D",
                campaign_type == "U",
                campaign_type == "R",
                campaign_type == "L",
                campaign_type == "Q",
                communication_method in {"Digital", "Email"},
            )
    return flags


def aggregate_campaigns(
    donor_ids: set[str],
    code_flags: dict[str, tuple[bool, ...]],
) -> tuple[dict[tuple[str, int], list[int]], dict[str, int]]:
    """Aggregate solicitation and response activity by donor and fiscal year."""
    annual: dict[tuple[str, int], list[int]] = {}
    audit = defaultdict(int)
    path = DATA_DIR / "campaign_members.txt"

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit["rows_read"] += 1
            donor_id = row["constituent_id"]
            if donor_id not in donor_ids:
                audit["rows_nonmodeled_donor_excluded"] += 1
                continue
            fy = fiscal_year(row["campaign_start_date"])
            if not FIRST_FY <= fy <= LAST_FY:
                audit["rows_outside_period_excluded"] += 1
                continue

            record = annual.setdefault(
                (donor_id, fy),
                [0] * CAMPAIGN_WIDTH,
            )
            record[CAMPAIGN_SOLICITATIONS] += 1
            record[CAMPAIGN_RESPONSES] += int(is_true(row["responded"]))

            flags = code_flags.get(row["marketing_code"])
            if flags is None:
                audit["rows_missing_campaign_code"] += 1
            else:
                major, upgrade, renewal, lapsed, acquisition, digital = flags
                record[CAMPAIGN_MAJOR] += int(major)
                record[CAMPAIGN_UPGRADE] += int(upgrade)
                record[CAMPAIGN_RENEWAL] += int(renewal)
                record[CAMPAIGN_LAPSED] += int(lapsed)
                record[CAMPAIGN_ACQUISITION] += int(acquisition)
                record[CAMPAIGN_DIGITAL] += int(digital)

            audit["rows_included"] += 1

    audit["donor_fy_records"] = len(annual)
    return annual, dict(audit)


def aggregate_passport(
    donor_ids: set[str],
) -> tuple[dict[tuple[str, int], list[float | int]], dict[str, int]]:
    """Aggregate annual Passport viewing and encode genre breadth as a bitmask."""
    annual: dict[tuple[str, int], list[float | int]] = {}
    genre_index: dict[str, int] = {}
    audit = defaultdict(int)
    path = DATA_DIR / "passport_viewing.txt"

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit["rows_read"] += 1
            donor_id = row["constituent_id"]
            if donor_id not in donor_ids:
                audit["rows_nonmodeled_donor_excluded"] += 1
                continue
            fy = int(row["fiscal_year"])
            if not FIRST_FY <= fy <= LAST_FY:
                audit["rows_outside_period_excluded"] += 1
                continue

            titles = int(row["titles_viewed"])
            mean_percent = float(row["mean_percent_watched"])
            genre = row["genre"]
            genre_position = genre_index.setdefault(genre, len(genre_index))

            record = annual.setdefault((donor_id, fy), [0, 0.0, 0])
            record[0] += titles
            record[1] += titles * mean_percent
            record[2] |= 1 << genre_position
            audit["rows_included"] += 1

    audit["donor_fy_records"] = len(annual)
    audit["genres"] = len(genre_index)
    return annual, dict(audit)


def load_portfolios() -> dict[str, dict[str, str]]:
    """Load officer and capacity metadata by portfolio."""
    portfolios: dict[str, dict[str, str]] = {}
    path = DATA_DIR / "officer_portfolios.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rollout = row["rollout_fy"].strip()
            portfolios[row["portfolio_id"]] = {
                "officer_id": row["officer_id"],
                "portfolio_capacity": row["capacity"],
                "portfolio_rollout_fy": (
                    str(int(float(rollout))) if rollout else ""
                ),
            }
    return portfolios


def aggregate_cultivation(
    donor_ids: set[str],
) -> tuple[dict[tuple[str, int], dict[str, str | bool]], dict[str, int]]:
    """Load one donor-year cultivation assignment per documented key."""
    annual: dict[tuple[str, int], dict[str, str | bool]] = {}
    audit = defaultdict(int)
    path = DATA_DIR / "cultivation.csv"

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit["rows_read"] += 1
            donor_id = row["synthetic_id"]
            if donor_id not in donor_ids:
                audit["rows_nonmodeled_donor_excluded"] += 1
                continue
            fy = int(row["fiscal_year"])
            key = (donor_id, fy)
            if key in annual:
                audit["duplicate_donor_fy_rows"] += 1
                continue
            annual[key] = {
                "portfolio_id": row["portfolio_id"],
                "selected": is_true(row["selected"]),
                "contacted": is_true(row["contacted"]),
                "holdout": is_true(row["holdout"]),
                "program_adopted": is_true(row["program_adopted"]),
            }
            audit["rows_included"] += 1

    audit["donor_fy_records"] = len(annual)
    return annual, dict(audit)


def aggregate_legacy_history(
    donor_ids: set[str],
) -> tuple[dict[str, dict[str, int | set[int]]], dict[str, int]]:
    """Build linked legacy relationship indicators without gift amounts."""
    legacy: dict[str, dict[str, int | set[int]]] = {}
    audit = defaultdict(int)
    path = DATA_DIR / "team_approach_legacy_payments.txt"
    id_fields = [f"constituent_id_{number}" for number in range(1, 8)]

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit["rows_read"] += 1
            linked_ids = {
                row[field]
                for field in id_fields
                if row[field] and row[field] != "NA" and row[field] in donor_ids
            }
            if not linked_ids:
                audit["rows_without_modeled_individual_link"] += 1
                continue

            fy = fiscal_year(row["gift_date"])
            for donor_id in linked_ids:
                stats = legacy.setdefault(
                    donor_id,
                    {
                        "first_fy": fy,
                        "last_fy": fy,
                        "gift_rows": 0,
                        "sustaining_rows": 0,
                        "upgrade_rows": 0,
                        "giving_years": set(),
                    },
                )
                stats["first_fy"] = min(int(stats["first_fy"]), fy)
                stats["last_fy"] = max(int(stats["last_fy"]), fy)
                stats["gift_rows"] = int(stats["gift_rows"]) + 1
                stats["sustaining_rows"] = int(stats["sustaining_rows"]) + int(
                    row["gift_kind"] == "Sustaining"
                )
                stats["upgrade_rows"] = int(stats["upgrade_rows"]) + int(
                    row["gift_type"] == "Upgrade"
                )
                giving_years = stats["giving_years"]
                assert isinstance(giving_years, set)
                giving_years.add(fy)
                audit["individual_links_included"] += 1

            audit["rows_with_modeled_individual_link"] += 1

    audit["linked_individuals"] = len(legacy)
    return legacy, dict(audit)


def campaign_window(
    campaign: dict[tuple[str, int], list[int]],
    donor_id: str,
    fy: int,
) -> list[int]:
    """Sum campaign measures over the current and prior two fiscal years."""
    totals = [0] * CAMPAIGN_WIDTH
    for year in range(max(FIRST_FY, fy - 2), fy + 1):
        record = campaign.get((donor_id, year))
        if record is not None:
            totals = [left + right for left, right in zip(totals, record)]
    return totals


def passport_window(
    passport: dict[tuple[str, int], list[float | int]],
    donor_id: str,
    fy: int,
) -> tuple[int, int, float, int]:
    """Aggregate Passport activity over the current and prior two years."""
    active_years = 0
    titles = 0
    weighted_sum = 0.0
    genre_mask = 0
    for year in range(max(FIRST_FY, fy - 2), fy + 1):
        record = passport.get((donor_id, year))
        if record is None:
            continue
        active_years += 1
        titles += int(record[0])
        weighted_sum += float(record[1])
        genre_mask |= int(record[2])
    weighted_mean = weighted_sum / titles if titles else 0.0
    return active_years, titles, weighted_mean, genre_mask.bit_count()


def cultivation_window(
    cultivation: dict[tuple[str, int], dict[str, str | bool]],
    donor_id: str,
    fy: int,
) -> tuple[int, int, int]:
    """Count cultivation assignments over the current and prior two years."""
    selected = 0
    contacted = 0
    holdout = 0
    for year in range(max(FIRST_FY, fy - 2), fy + 1):
        record = cultivation.get((donor_id, year))
        if record is None:
            continue
        selected += int(bool(record["selected"]))
        contacted += int(bool(record["contacted"]))
        holdout += int(bool(record["holdout"]))
    return selected, contacted, holdout


def build_enriched_table(
    core_fields: list[str],
    campaign: dict[tuple[str, int], list[int]],
    passport: dict[tuple[str, int], list[float | int]],
    cultivation: dict[tuple[str, int], dict[str, str | bool]],
    portfolios: dict[str, dict[str, str]],
    legacy: dict[str, dict[str, int | set[int]]],
) -> dict[str, int]:
    """Stream the core table and append annual and rolling features."""
    feature_fields = [
        "rolling_years_available",
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
        "campaign_solicitations_fy",
        "campaign_responses_fy",
        "campaign_response_rate_fy",
        "major_gift_solicitations_fy",
        "upgrade_solicitations_fy",
        "renewal_solicitations_fy",
        "lapsed_solicitations_fy",
        "acquisition_solicitations_fy",
        "digital_solicitations_fy",
        "campaign_solicitations_3yr",
        "campaign_responses_3yr",
        "campaign_response_rate_3yr",
        "major_gift_solicitations_3yr",
        "upgrade_solicitations_3yr",
        "passport_observed_fy",
        "passport_titles_viewed_fy",
        "passport_genres_viewed_fy",
        "passport_weighted_mean_percent_watched_fy",
        "passport_active_years_3yr",
        "passport_titles_viewed_3yr",
        "passport_genres_viewed_3yr",
        "passport_weighted_mean_percent_watched_3yr",
        "cultivation_selected_fy",
        "cultivation_contacted_fy",
        "cultivation_holdout_fy",
        "cultivation_program_adopted_fy",
        "cultivation_portfolio_id_fy",
        "cultivation_officer_id_fy",
        "cultivation_portfolio_capacity_fy",
        "cultivation_portfolio_rollout_fy",
        "cultivation_selected_3yr",
        "cultivation_contacted_3yr",
        "cultivation_holdout_3yr",
        "has_linked_legacy_history",
        "legacy_first_fy",
        "legacy_last_fy",
        "legacy_linked_gift_rows",
        "legacy_linked_giving_years",
        "legacy_linked_sustaining_rows",
        "legacy_linked_upgrade_rows",
        "years_since_last_legacy_gift",
    ]
    fieldnames = core_fields + feature_fields
    audit = defaultdict(int)

    current_donor = ""
    history: list[dict[str, str]] = []
    last_positive_fy: int | None = None

    with CORE_PATH.open(encoding="utf-8", newline="") as source, OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            donor_id = row["constituent_id"]
            fy = int(row["fiscal_year"])
            if donor_id != current_donor:
                current_donor = donor_id
                history = []
                last_positive_fy = None

            history.append(row)
            rolling_rows = history[-3:]
            annual_values = [Decimal(item["annual_giving"]) for item in rolling_rows]
            direct_values = [Decimal(item["direct_giving"]) for item in rolling_rows]
            soft_values = [Decimal(item["soft_credit_giving"]) for item in rolling_rows]
            recurring_values = [
                Decimal(item["recurring_direct_giving"]) for item in rolling_rows
            ]
            nonrecurring_values = [
                Decimal(item["nonrecurring_recognized_giving"])
                for item in rolling_rows
            ]
            current_giving = Decimal(row["annual_giving"])
            prior_giving = (
                Decimal(history[-2]["annual_giving"])
                if len(history) >= 2
                else None
            )
            if current_giving > ZERO:
                last_positive_fy = fy

            annual_campaign = campaign.get(
                (donor_id, fy),
                [0] * CAMPAIGN_WIDTH,
            )
            rolling_campaign = campaign_window(campaign, donor_id, fy)

            annual_passport = passport.get((donor_id, fy))
            if annual_passport is None:
                passport_titles = 0
                passport_weighted_mean = ""
                passport_genres = 0
            else:
                passport_titles = int(annual_passport[0])
                passport_weighted_mean = (
                    f"{float(annual_passport[1]) / passport_titles:.6f}"
                    if passport_titles
                    else ""
                )
                passport_genres = int(annual_passport[2]).bit_count()
            (
                passport_active_years,
                passport_titles_3yr,
                passport_weighted_mean_3yr,
                passport_genres_3yr,
            ) = passport_window(passport, donor_id, fy)

            annual_cultivation = cultivation.get((donor_id, fy))
            cultivation_3yr = cultivation_window(cultivation, donor_id, fy)
            portfolio_id = (
                str(annual_cultivation["portfolio_id"])
                if annual_cultivation is not None
                else ""
            )
            portfolio = portfolios.get(portfolio_id, {})

            legacy_stats = legacy.get(donor_id)
            if legacy_stats is None:
                legacy_features = {
                    "has_linked_legacy_history": False,
                    "legacy_first_fy": "",
                    "legacy_last_fy": "",
                    "legacy_linked_gift_rows": 0,
                    "legacy_linked_giving_years": 0,
                    "legacy_linked_sustaining_rows": 0,
                    "legacy_linked_upgrade_rows": 0,
                    "years_since_last_legacy_gift": "",
                }
            else:
                giving_years = legacy_stats["giving_years"]
                assert isinstance(giving_years, set)
                legacy_last_fy = int(legacy_stats["last_fy"])
                legacy_features = {
                    "has_linked_legacy_history": True,
                    "legacy_first_fy": legacy_stats["first_fy"],
                    "legacy_last_fy": legacy_last_fy,
                    "legacy_linked_gift_rows": legacy_stats["gift_rows"],
                    "legacy_linked_giving_years": len(giving_years),
                    "legacy_linked_sustaining_rows": legacy_stats[
                        "sustaining_rows"
                    ],
                    "legacy_linked_upgrade_rows": legacy_stats["upgrade_rows"],
                    "years_since_last_legacy_gift": fy - legacy_last_fy,
                }

            enriched = {
                **row,
                "rolling_years_available": len(rolling_rows),
                "annual_giving_3yr_total": money(sum(annual_values, ZERO)),
                "annual_giving_3yr_mean": money(
                    sum(annual_values, ZERO) / len(annual_values)
                ),
                "direct_giving_3yr_total": money(sum(direct_values, ZERO)),
                "soft_credit_giving_3yr_total": money(sum(soft_values, ZERO)),
                "recurring_giving_3yr_total": money(
                    sum(recurring_values, ZERO)
                ),
                "nonrecurring_giving_3yr_total": money(
                    sum(nonrecurring_values, ZERO)
                ),
                "direct_payment_count_3yr": sum(
                    int(item["direct_payment_count"]) for item in rolling_rows
                ),
                "giving_years_3yr": sum(value > ZERO for value in annual_values),
                "max_direct_payment_3yr": money(
                    max(
                        Decimal(item["max_direct_payment"])
                        for item in rolling_rows
                    )
                ),
                "giving_change_vs_prior_fy": (
                    money(current_giving - prior_giving)
                    if prior_giving is not None
                    else ""
                ),
                "years_since_last_positive_giving": (
                    fy - last_positive_fy if last_positive_fy is not None else ""
                ),
                "threshold_gap_current_fy": money(
                    max(ZERO, MAJOR_THRESHOLD - current_giving)
                ),
                "campaign_solicitations_fy": annual_campaign[
                    CAMPAIGN_SOLICITATIONS
                ],
                "campaign_responses_fy": annual_campaign[CAMPAIGN_RESPONSES],
                "campaign_response_rate_fy": rate(
                    annual_campaign[CAMPAIGN_RESPONSES],
                    annual_campaign[CAMPAIGN_SOLICITATIONS],
                ),
                "major_gift_solicitations_fy": annual_campaign[CAMPAIGN_MAJOR],
                "upgrade_solicitations_fy": annual_campaign[CAMPAIGN_UPGRADE],
                "renewal_solicitations_fy": annual_campaign[CAMPAIGN_RENEWAL],
                "lapsed_solicitations_fy": annual_campaign[CAMPAIGN_LAPSED],
                "acquisition_solicitations_fy": annual_campaign[
                    CAMPAIGN_ACQUISITION
                ],
                "digital_solicitations_fy": annual_campaign[CAMPAIGN_DIGITAL],
                "campaign_solicitations_3yr": rolling_campaign[
                    CAMPAIGN_SOLICITATIONS
                ],
                "campaign_responses_3yr": rolling_campaign[CAMPAIGN_RESPONSES],
                "campaign_response_rate_3yr": rate(
                    rolling_campaign[CAMPAIGN_RESPONSES],
                    rolling_campaign[CAMPAIGN_SOLICITATIONS],
                ),
                "major_gift_solicitations_3yr": rolling_campaign[
                    CAMPAIGN_MAJOR
                ],
                "upgrade_solicitations_3yr": rolling_campaign[
                    CAMPAIGN_UPGRADE
                ],
                "passport_observed_fy": annual_passport is not None,
                "passport_titles_viewed_fy": passport_titles,
                "passport_genres_viewed_fy": passport_genres,
                "passport_weighted_mean_percent_watched_fy": (
                    passport_weighted_mean
                ),
                "passport_active_years_3yr": passport_active_years,
                "passport_titles_viewed_3yr": passport_titles_3yr,
                "passport_genres_viewed_3yr": passport_genres_3yr,
                "passport_weighted_mean_percent_watched_3yr": (
                    f"{passport_weighted_mean_3yr:.6f}"
                    if passport_titles_3yr
                    else ""
                ),
                "cultivation_selected_fy": (
                    bool(annual_cultivation["selected"])
                    if annual_cultivation is not None
                    else False
                ),
                "cultivation_contacted_fy": (
                    bool(annual_cultivation["contacted"])
                    if annual_cultivation is not None
                    else False
                ),
                "cultivation_holdout_fy": (
                    bool(annual_cultivation["holdout"])
                    if annual_cultivation is not None
                    else False
                ),
                "cultivation_program_adopted_fy": (
                    bool(annual_cultivation["program_adopted"])
                    if annual_cultivation is not None
                    else False
                ),
                "cultivation_portfolio_id_fy": portfolio_id,
                "cultivation_officer_id_fy": portfolio.get("officer_id", ""),
                "cultivation_portfolio_capacity_fy": portfolio.get(
                    "portfolio_capacity",
                    "",
                ),
                "cultivation_portfolio_rollout_fy": portfolio.get(
                    "portfolio_rollout_fy",
                    "",
                ),
                "cultivation_selected_3yr": cultivation_3yr[0],
                "cultivation_contacted_3yr": cultivation_3yr[1],
                "cultivation_holdout_3yr": cultivation_3yr[2],
                **legacy_features,
            }
            writer.writerow(enriched)
            audit["rows_written"] += 1
            audit["rows_with_campaign_fy"] += int(
                annual_campaign[CAMPAIGN_SOLICITATIONS] > 0
            )
            audit["rows_with_passport_fy"] += int(annual_passport is not None)
            audit["rows_with_cultivation_fy"] += int(
                annual_cultivation is not None
            )
            audit["rows_with_legacy_history"] += int(legacy_stats is not None)

    return dict(audit)


def print_audit(name: str, values: dict[str, int]) -> None:
    """Print a sorted audit section."""
    print(f"\n{name}")
    for key in sorted(values):
        print(f"  {key}: {values[key]:,}")


def main() -> None:
    donor_ids, core_fields = load_core_donors()
    code_flags = load_campaign_code_flags()
    campaign, campaign_audit = aggregate_campaigns(donor_ids, code_flags)
    passport, passport_audit = aggregate_passport(donor_ids)
    cultivation, cultivation_audit = aggregate_cultivation(donor_ids)
    portfolios = load_portfolios()
    legacy, legacy_audit = aggregate_legacy_history(donor_ids)
    output_audit = build_enriched_table(
        core_fields,
        campaign,
        passport,
        cultivation,
        portfolios,
        legacy,
    )

    print(f"Output: {OUTPUT_PATH}")
    print(f"Modeled donor IDs: {len(donor_ids):,}")
    print(f"Campaign codes: {len(code_flags):,}")
    print(f"Officer portfolios: {len(portfolios):,}")
    print_audit("Campaign audit", campaign_audit)
    print_audit("Passport audit", passport_audit)
    print_audit("Cultivation audit", cultivation_audit)
    print_audit("Legacy audit", legacy_audit)
    print_audit("Output audit", output_audit)


if __name__ == "__main__":
    main()
