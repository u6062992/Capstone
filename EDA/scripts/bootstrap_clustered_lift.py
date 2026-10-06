"""Estimate donor-clustered bootstrap intervals for pooled converter lift.

Whole donor histories are resampled with replacement so repeated donor-year
observations stay together. Donors with identical band histories are grouped
for computational efficiency; sampling probabilities remain proportional to
the number of donors represented by each signature.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path

from create_initial_eda_visualizations import (
    BACKGROUND,
    CONVERTER_COLOR,
    GRID_COLOR,
    LIFT_SPECS,
    PANEL_BACKGROUND,
    TEXT_COLOR,
    band_label,
    line,
    rect,
    svg_start,
    text,
)


EDA_DIR = Path(__file__).resolve().parents[1]
COHORT_PATH = EDA_DIR / "derived_data" / "eda_eligible_fy2023_2025.csv"
OUTPUT_DIR = EDA_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_CSV = OUTPUT_DIR / "converter_lift_cluster_bootstrap.csv"
OUTPUT_SVG = FIGURE_DIR / "converter_lift_cluster_bootstrap.svg"

BOOTSTRAP_REPLICATES = 1000
RANDOM_SEED = 20261005


def percentile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
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


def band_metadata() -> list[tuple[str, str, str]]:
    """Return ordered feature, display name, and band metadata."""
    return [
        (feature, title_value, label)
        for feature, title_value, bands in LIFT_SPECS
        for label, *_ in bands
    ]


def build_donor_signatures(
    metadata: list[tuple[str, str, str]],
) -> tuple[Counter, int, int]:
    """Build a frequency table of whole-donor pooled-band histories."""
    band_index = {
        (feature, label): index
        for index, (feature, _, label) in enumerate(metadata)
    }
    donor_records: dict[str, dict[str, object]] = {}

    with COHORT_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            donor_id = row["constituent_id"]
            converted = row["next_fy_conversion"] == "True"
            record = donor_records.setdefault(
                donor_id,
                {
                    "rows": 0,
                    "conversions": 0,
                    "band_rows": [0] * len(metadata),
                    "band_conversions": [0] * len(metadata),
                },
            )
            record["rows"] = int(record["rows"]) + 1
            record["conversions"] = int(record["conversions"]) + int(converted)

            band_rows = record["band_rows"]
            band_conversions = record["band_conversions"]
            assert isinstance(band_rows, list)
            assert isinstance(band_conversions, list)
            for feature, _, bands in LIFT_SPECS:
                label = band_label(float(row[feature]), bands)
                index = band_index[(feature, label)]
                band_rows[index] += 1
                band_conversions[index] += int(converted)

    signatures = Counter()
    total_rows = 0
    total_conversions = 0
    for record in donor_records.values():
        rows = int(record["rows"])
        conversions = int(record["conversions"])
        band_rows = record["band_rows"]
        band_conversions = record["band_conversions"]
        assert isinstance(band_rows, list)
        assert isinstance(band_conversions, list)
        signature = (
            rows,
            conversions,
            *band_rows,
            *band_conversions,
        )
        signatures[signature] += 1
        total_rows += rows
        total_conversions += conversions

    return signatures, total_rows, total_conversions


def run_bootstrap(
    signatures: Counter,
    metadata: list[tuple[str, str, str]],
) -> tuple[list[list[float]], list[list[float]], int]:
    """Run an exact whole-donor nonparametric cluster bootstrap."""
    signature_values = list(signatures)
    signature_weights = [signatures[value] for value in signature_values]
    cumulative_weights = list(accumulate(signature_weights))
    donor_count = sum(signature_weights)
    band_count = len(metadata)
    rate_samples = [[] for _ in range(band_count)]
    lift_samples = [[] for _ in range(band_count)]
    rng = random.Random(RANDOM_SEED)
    successful_replicates = 0

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_signature_indexes = Counter(
            rng.choices(
                range(len(signature_values)),
                cum_weights=cumulative_weights,
                k=donor_count,
            )
        )
        total_rows = 0
        total_conversions = 0
        band_rows = [0] * band_count
        band_conversions = [0] * band_count

        for signature_index, multiplicity in sampled_signature_indexes.items():
            signature = signature_values[signature_index]
            total_rows += multiplicity * signature[0]
            total_conversions += multiplicity * signature[1]
            row_offset = 2
            conversion_offset = 2 + band_count
            for band in range(band_count):
                band_rows[band] += multiplicity * signature[row_offset + band]
                band_conversions[band] += (
                    multiplicity * signature[conversion_offset + band]
                )

        if total_rows == 0 or total_conversions == 0:
            continue
        overall_rate = total_conversions / total_rows
        if any(value == 0 for value in band_rows):
            continue

        for band in range(band_count):
            conversion_rate = band_conversions[band] / band_rows[band]
            rate_samples[band].append(conversion_rate)
            lift_samples[band].append(conversion_rate / overall_rate)
        successful_replicates += 1

        if (replicate + 1) % 100 == 0:
            print(f"Completed {replicate + 1:,} bootstrap replicates")

    return rate_samples, lift_samples, successful_replicates


def observed_band_counts(
    signatures: Counter,
    band_count: int,
) -> tuple[int, int, list[int], list[int]]:
    """Recover observed row and conversion counts from donor signatures."""
    total_rows = 0
    total_conversions = 0
    band_rows = [0] * band_count
    band_conversions = [0] * band_count
    for signature, donor_frequency in signatures.items():
        total_rows += donor_frequency * signature[0]
        total_conversions += donor_frequency * signature[1]
        for band in range(band_count):
            band_rows[band] += donor_frequency * signature[2 + band]
            band_conversions[band] += donor_frequency * signature[
                2 + band_count + band
            ]
    return total_rows, total_conversions, band_rows, band_conversions


def write_results(
    metadata: list[tuple[str, str, str]],
    signatures: Counter,
    rate_samples: list[list[float]],
    lift_samples: list[list[float]],
    successful_replicates: int,
) -> list[dict[str, str | int]]:
    """Write observed estimates and clustered percentile intervals."""
    (
        total_rows,
        total_conversions,
        band_rows,
        band_conversions,
    ) = observed_band_counts(signatures, len(metadata))
    overall_rate = total_conversions / total_rows
    rows: list[dict[str, str | int]] = []

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "measure",
            "band",
            "observations",
            "conversions",
            "conversion_rate",
            "lift_vs_overall",
            "cluster_bootstrap_rate_ci_lower",
            "cluster_bootstrap_rate_ci_upper",
            "cluster_bootstrap_lift_ci_lower",
            "cluster_bootstrap_lift_ci_upper",
            "donor_clusters",
            "bootstrap_replicates",
            "successful_replicates",
            "random_seed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, (_, title_value, label) in enumerate(metadata):
            rates = sorted(rate_samples[index])
            lifts = sorted(lift_samples[index])
            conversion_rate = band_conversions[index] / band_rows[index]
            lift = conversion_rate / overall_rate
            row = {
                "measure": title_value,
                "band": label,
                "observations": band_rows[index],
                "conversions": band_conversions[index],
                "conversion_rate": f"{conversion_rate:.8f}",
                "lift_vs_overall": f"{lift:.4f}",
                "cluster_bootstrap_rate_ci_lower": (
                    f"{percentile(rates, 0.025):.8f}"
                ),
                "cluster_bootstrap_rate_ci_upper": (
                    f"{percentile(rates, 0.975):.8f}"
                ),
                "cluster_bootstrap_lift_ci_lower": (
                    f"{percentile(lifts, 0.025):.4f}"
                ),
                "cluster_bootstrap_lift_ci_upper": (
                    f"{percentile(lifts, 0.975):.4f}"
                ),
                "donor_clusters": sum(signatures.values()),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "successful_replicates": successful_replicates,
                "random_seed": RANDOM_SEED,
            }
            writer.writerow(row)
            rows.append(row)
    return rows


def create_chart(rows: list[dict[str, str | int]]) -> None:
    """Create pooled lift charts with donor-clustered bootstrap intervals."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 790
    elements = svg_start(width, height, "Donor-clustered bootstrap lift intervals")
    elements.extend(
        [
            text(60, 55, "Donor-clustered bootstrap lift intervals", "title"),
            text(
                60,
                82,
                (
                    f"{BOOTSTRAP_REPLICATES:,} whole-donor replicates; "
                    "error bars are 95% percentile intervals"
                ),
                "subtitle",
            ),
        ]
    )

    panel_positions = [(50, 120), (770, 120)]
    panel_width, panel_height = 680, 560
    grouped_rows = {
        title_value: [row for row in rows if row["measure"] == title_value]
        for _, title_value, _ in LIFT_SPECS
    }
    max_upper = max(
        float(row["cluster_bootstrap_lift_ci_upper"])
        for row in rows
    )
    if max_upper > 100:
        y_max = math.ceil(max_upper / 50) * 50
    elif max_upper > 20:
        y_max = math.ceil(max_upper / 10) * 10
    else:
        y_max = max(2.0, math.ceil(max_upper / 2) * 2)

    for (_, title_value, _), (panel_x, panel_y) in zip(
        LIFT_SPECS,
        panel_positions,
    ):
        panel_rows = grouped_rows[title_value]
        elements.append(
            rect(
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                PANEL_BACKGROUND,
                12,
            )
        )
        elements.append(
            text(panel_x + 24, panel_y + 36, title_value, "panel-title")
        )
        chart_x = panel_x + 65
        chart_y = panel_y + 70
        chart_width = panel_width - 90
        chart_height = 350

        for tick_index in range(6):
            tick = y_max * tick_index / 5
            y = chart_y + chart_height - (tick / y_max) * chart_height
            elements.append(line(chart_x, y, chart_x + chart_width, y, GRID_COLOR))
            elements.append(
                text(chart_x - 8, y + 4, f"{tick:.0f}×", "axis", "end")
            )

        baseline_y = chart_y + chart_height - (1 / y_max) * chart_height
        elements.append(
            line(
                chart_x,
                baseline_y,
                chart_x + chart_width,
                baseline_y,
                CONVERTER_COLOR,
                2,
                "6 5",
            )
        )

        group_width = chart_width / len(panel_rows)
        bar_width = min(48, group_width * 0.58)
        for index, row in enumerate(panel_rows):
            lift = float(row["lift_vs_overall"])
            lower = float(row["cluster_bootstrap_lift_ci_lower"])
            upper = float(row["cluster_bootstrap_lift_ci_upper"])
            observations = int(row["observations"])
            conversions = int(row["conversions"])
            conversion_rate = float(row["conversion_rate"])
            center = chart_x + group_width * (index + 0.5)
            bar_height = (lift / y_max) * chart_height
            x = center - bar_width / 2
            y = chart_y + chart_height - bar_height
            elements.append(rect(x, y, bar_width, bar_height, CONVERTER_COLOR, 3))

            low_y = chart_y + chart_height - (lower / y_max) * chart_height
            high_y = chart_y + chart_height - (upper / y_max) * chart_height
            elements.append(line(center, low_y, center, high_y, TEXT_COLOR, 1.5))
            elements.append(
                line(center - 6, low_y, center + 6, low_y, TEXT_COLOR, 1.5)
            )
            elements.append(
                line(center - 6, high_y, center + 6, high_y, TEXT_COLOR, 1.5)
            )
            elements.append(
                text(
                    center,
                    max(high_y - 8, chart_y + 12),
                    f"{lift:.1f}×",
                    "value",
                    "middle",
                )
            )

            label_y = chart_y + chart_height + 20
            elements.append(
                text(
                    center,
                    label_y,
                    str(row["band"]),
                    "axis",
                    "end",
                    f"rotate(-35 {center:.1f} {label_y:.1f})",
                )
            )
            elements.append(
                text(
                    center,
                    chart_y + chart_height + 73,
                    f"{conversions}/{observations:,}",
                    "axis",
                    "middle",
                )
            )
            elements.append(
                text(
                    center,
                    chart_y + chart_height + 91,
                    f"{conversion_rate:.3%}",
                    "axis",
                    "middle",
                )
            )

        elements.append(
            text(
                panel_x + 24,
                panel_y + panel_height - 24,
                "Whole donor histories are sampled together; dashed line = 1× baseline.",
                "note",
            )
        )

    elements.append("</svg>")
    OUTPUT_SVG.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = band_metadata()
    signatures, total_rows, total_conversions = build_donor_signatures(metadata)
    print(f"Donor clusters: {sum(signatures.values()):,}")
    print(f"Unique donor signatures: {len(signatures):,}")
    print(f"Donor-year rows: {total_rows:,}")
    print(f"Conversions: {total_conversions:,}")

    rate_samples, lift_samples, successful_replicates = run_bootstrap(
        signatures,
        metadata,
    )
    rows = write_results(
        metadata,
        signatures,
        rate_samples,
        lift_samples,
        successful_replicates,
    )
    create_chart(rows)

    print(f"Successful replicates: {successful_replicates:,}")
    print(f"Output data: {OUTPUT_CSV}")
    print(f"Output figure: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
