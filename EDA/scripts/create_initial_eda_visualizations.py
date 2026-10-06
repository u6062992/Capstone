"""Create dependency-free SVG visualizations for the initial donor EDA.

The charts use normalized percentages and conversion rates so the rare
converter group remains visible. Only the Python standard library is required.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


EDA_DIR = Path(__file__).resolve().parents[1]
COHORT_PATH = EDA_DIR / "derived_data" / "eda_eligible_fy2023_2025.csv"
OUTPUT_DIR = EDA_DIR / "outputs" / "figures"
TARGET_SVG = OUTPUT_DIR / "target_balance.svg"
GIVING_SVG = OUTPUT_DIR / "giving_distributions.svg"
LIFT_SVG = OUTPUT_DIR / "converter_lift.svg"
LIFT_CSV = EDA_DIR / "outputs" / "converter_lift_by_giving_band.csv"
FY_LIFT_SVG = OUTPUT_DIR / "converter_lift_by_fiscal_year.svg"
FY_LIFT_CSV = EDA_DIR / "outputs" / "converter_lift_by_fiscal_year.csv"
GALLERY_PATH = EDA_DIR / "outputs" / "eda_visualizations.md"

NONCONVERTER_COLOR = "#4C78A8"
CONVERTER_COLOR = "#F58518"
ACTIVE_COLOR = "#4C78A8"
LAPSED_COLOR = "#72B7B2"
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#243447"
MUTED_TEXT = "#5F6B76"
BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#F8FAFC"
FY_COLORS = {
    "2023": "#4C78A8",
    "2024": "#F58518",
    "2025": "#54A24B",
}
WILSON_Z = 1.959963984540054


ANNUAL_BANDS = [
    ("$0", 0.0, 0.0, True),
    ("$1–24", 1.0, 25.0, False),
    ("$25–49", 25.0, 50.0, False),
    ("$50–99", 50.0, 100.0, False),
    ("$100–249", 100.0, 250.0, False),
    ("$250–499", 250.0, 500.0, False),
    ("$500–1,199", 500.0, 1200.0, False),
]

THREE_YEAR_BANDS = [
    ("$0–99", 0.0, 100.0, False),
    ("$100–249", 100.0, 250.0, False),
    ("$250–499", 250.0, 500.0, False),
    ("$500–749", 500.0, 750.0, False),
    ("$750–999", 750.0, 1000.0, False),
    ("$1,000–1,499", 1000.0, 1500.0, False),
    ("$1,500+", 1500.0, math.inf, False),
]

MAX_GIFT_BANDS = [
    ("$0", 0.0, 0.0, True),
    ("$1–9", 1.0, 10.0, False),
    ("$10–24", 10.0, 25.0, False),
    ("$25–49", 25.0, 50.0, False),
    ("$50–99", 50.0, 100.0, False),
    ("$100–249", 100.0, 250.0, False),
    ("$250–499", 250.0, 500.0, False),
    ("$500+", 500.0, math.inf, False),
]

NONRECURRING_BANDS = [
    ("$0", 0.0, 0.0, True),
    ("$1–24", 1.0, 25.0, False),
    ("$25–49", 25.0, 50.0, False),
    ("$50–99", 50.0, 100.0, False),
    ("$100–249", 100.0, 250.0, False),
    ("$250–499", 250.0, 500.0, False),
    ("$500–999", 500.0, 1000.0, False),
    ("$1,000+", 1000.0, math.inf, False),
]

DISTRIBUTION_SPECS = [
    ("annual_giving", "Current annual giving", ANNUAL_BANDS),
    ("annual_giving_3yr_total", "Three-year total giving", THREE_YEAR_BANDS),
    ("max_direct_payment_3yr", "Maximum direct payment", MAX_GIFT_BANDS),
    (
        "nonrecurring_giving_3yr_total",
        "Three-year nonrecurring giving",
        NONRECURRING_BANDS,
    ),
]

LIFT_SPECS = [
    ("annual_giving", "Current annual giving", ANNUAL_BANDS),
    ("annual_giving_3yr_total", "Three-year total giving", THREE_YEAR_BANDS),
]

FY_LIFT_SPECS = [
    (
        "annual_giving",
        "Current annual giving",
        [
            ("Under $250", 0.0, 250.0, False),
            ("$250–499", 250.0, 500.0, False),
            ("$500–1,199", 500.0, 1200.0, False),
        ],
    ),
    (
        "annual_giving_3yr_total",
        "Three-year total giving",
        [
            ("Under $500", 0.0, 500.0, False),
            ("$500–999", 500.0, 1000.0, False),
            ("$1,000+", 1000.0, math.inf, False),
        ],
    ),
]


def svg_start(width: int, height: int, title: str) -> list[str]:
    """Create an SVG document header and shared styles."""
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="chart-title chart-desc">'
        ),
        f'<title id="chart-title">{escape(title)}</title>',
        '<desc id="chart-desc">PBS Utah donor conversion exploratory data analysis chart.</desc>',
        "<style>",
        (
            "text { font-family: Arial, Helvetica, sans-serif; fill: "
            f"{TEXT_COLOR}; }}"
        ),
        ".title { font-size: 26px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #5F6B76; }",
        ".panel-title { font-size: 17px; font-weight: 700; }",
        ".axis { font-size: 11px; fill: #5F6B76; }",
        ".label { font-size: 12px; }",
        ".value { font-size: 12px; font-weight: 700; }",
        ".note { font-size: 12px; fill: #5F6B76; }",
        "</style>",
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
    ]


def text(
    x: float,
    y: float,
    value: str,
    css_class: str = "label",
    anchor: str = "start",
    transform: str = "",
) -> str:
    """Return an escaped SVG text element."""
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}"{transform_attr}>{escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    radius: float = 0,
    opacity: float = 1.0,
) -> str:
    """Return an SVG rectangle."""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" '
        f'height="{max(height, 0):.1f}" rx="{radius:.1f}" '
        f'fill="{fill}" opacity="{opacity:.3f}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = GRID_COLOR,
    width: float = 1,
    dash: str = "",
) -> str:
    """Return an SVG line."""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def band_label(value: float, bands: list[tuple[str, float, float, bool]]) -> str:
    """Assign a numeric value to a labeled band."""
    for label, lower, upper, zero_only in bands:
        if zero_only and value == 0:
            return label
        if not zero_only and lower <= value < upper:
            return label
    raise ValueError(f"Value {value} did not match a configured band")


def wilson_interval(
    conversions: int,
    observations: int,
    z: float = WILSON_Z,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if observations == 0:
        return 0.0, 0.0
    proportion = conversions / observations
    z_squared = z * z
    denominator = 1 + z_squared / observations
    center = (
        proportion + z_squared / (2 * observations)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / observations
            + z_squared / (4 * observations * observations)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_counts() -> tuple[
    Counter,
    Counter,
    Counter,
    Counter,
    Counter,
    int,
    int,
]:
    """Scan the EDA cohort once and collect chart counts."""
    outcome_counts = Counter()
    target_groups = Counter()
    distribution_counts = Counter()
    lift_counts = Counter()
    fy_lift_counts = Counter()
    total_rows = 0
    total_conversions = 0

    with COHORT_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            outcome = row["next_fy_conversion"]
            converted = outcome == "True"
            total_conversions += int(converted)
            outcome_counts[outcome] += 1
            target_groups[(row["fiscal_year"], row["eligibility_segment"])] += 1
            target_groups[
                (row["fiscal_year"], row["eligibility_segment"], "converted")
            ] += int(converted)

            for feature, _, bands in DISTRIBUTION_SPECS:
                label = band_label(float(row[feature]), bands)
                distribution_counts[(feature, outcome, label)] += 1

            for feature, _, bands in LIFT_SPECS:
                label = band_label(float(row[feature]), bands)
                lift_counts[(feature, label, "rows")] += 1
                lift_counts[(feature, label, "conversions")] += int(converted)

            fiscal_year = row["fiscal_year"]
            fy_lift_counts[(fiscal_year, "all", "rows")] += 1
            fy_lift_counts[(fiscal_year, "all", "conversions")] += int(converted)
            for feature, _, bands in FY_LIFT_SPECS:
                label = band_label(float(row[feature]), bands)
                fy_lift_counts[(fiscal_year, feature, label, "rows")] += 1
                fy_lift_counts[
                    (fiscal_year, feature, label, "conversions")
                ] += int(converted)

    return (
        outcome_counts,
        target_groups,
        distribution_counts,
        lift_counts,
        fy_lift_counts,
        total_rows,
        total_conversions,
    )


def create_target_balance(
    outcome_counts: Counter,
    target_groups: Counter,
    total_rows: int,
    total_conversions: int,
) -> None:
    """Create a two-panel target count and conversion-rate chart."""
    width, height = 1400, 700
    elements = svg_start(width, height, "Target balance and conversion rates")
    elements.extend(
        [
            text(60, 55, "Target balance and conversion rates", "title"),
            text(
                60,
                82,
                "Eligible FY2023–FY2025 donor-years with complete three-year history",
                "subtitle",
            ),
        ]
    )

    panels = [(50, 115, 610, 490), (700, 115, 650, 490)]
    for x, y, panel_width, panel_height in panels:
        elements.append(rect(x, y, panel_width, panel_height, PANEL_BACKGROUND, 12))

    # Left panel: log-scaled outcome counts.
    px, py, pw, ph = panels[0]
    elements.append(text(px + 25, py + 38, "Outcome counts (log scale)", "panel-title"))
    chart_x = px + 155
    chart_y = py + 90
    chart_w = pw - 195
    max_log = math.log10(max(outcome_counts.values()))
    for exponent in range(0, 6):
        tick = 10**exponent
        tick_x = chart_x + chart_w * (math.log10(tick) / max_log)
        elements.append(line(tick_x, chart_y - 10, tick_x, chart_y + 245))
        elements.append(text(tick_x, chart_y + 268, f"{tick:,}", "axis", "middle"))

    outcomes = [
        ("Non-converter", outcome_counts["False"], NONCONVERTER_COLOR),
        ("Converter", outcome_counts["True"], CONVERTER_COLOR),
    ]
    for index, (label, count, color) in enumerate(outcomes):
        y = chart_y + 35 + index * 115
        bar_width = chart_w * (math.log10(max(count, 1)) / max_log)
        elements.append(text(chart_x - 18, y + 27, label, "label", "end"))
        elements.append(rect(chart_x, y, bar_width, 42, color, 5))
        elements.append(text(chart_x + bar_width + 10, y + 27, f"{count:,}", "value"))

    overall_rate = total_conversions / total_rows
    elements.append(
        text(
            px + 25,
            py + ph - 58,
            f"Overall conversion rate: {overall_rate:.4%}",
            "value",
        )
    )
    elements.append(
        text(
            px + 25,
            py + ph - 32,
            "The rare outcome is not visible on an ordinary linear count scale.",
            "note",
        )
    )

    # Right panel: rates by fiscal year and segment.
    px, py, pw, ph = panels[1]
    elements.append(text(px + 25, py + 38, "Conversion rate by cohort", "panel-title"))
    chart_x = px + 85
    chart_y = py + 75
    chart_w = pw - 125
    chart_h = 310
    max_rate = 0.0016
    for tick in [0, 0.0004, 0.0008, 0.0012, 0.0016]:
        y = chart_y + chart_h - (tick / max_rate) * chart_h
        elements.append(line(chart_x, y, chart_x + chart_w, y))
        elements.append(text(chart_x - 10, y + 4, f"{tick:.2%}", "axis", "end"))

    years = ["2023", "2024", "2025"]
    segments = [
        ("active_normal", "Active", ACTIVE_COLOR),
        ("lapsed_normal", "Lapsed", LAPSED_COLOR),
    ]
    group_width = chart_w / len(years)
    bar_width = 48
    for year_index, year in enumerate(years):
        center = chart_x + group_width * (year_index + 0.5)
        for segment_index, (segment, _, color) in enumerate(segments):
            count = target_groups[(year, segment)]
            conversions = target_groups[(year, segment, "converted")]
            conversion_rate = conversions / count if count else 0
            x = center + (segment_index - 0.5) * (bar_width + 10) - bar_width / 2
            bar_height = (conversion_rate / max_rate) * chart_h
            y = chart_y + chart_h - bar_height
            elements.append(rect(x, y, bar_width, bar_height, color, 3))
            elements.append(
                text(
                    x + bar_width / 2,
                    max(y - 8, chart_y + 10),
                    f"{conversion_rate:.3%}",
                    "axis",
                    "middle",
                )
            )
        elements.append(text(center, chart_y + chart_h + 28, f"FY{year}", "label", "middle"))

    legend_y = py + ph - 45
    for index, (_, label, color) in enumerate(segments):
        x = px + 170 + index * 170
        elements.append(rect(x, legend_y - 12, 16, 16, color, 2))
        elements.append(text(x + 24, legend_y + 1, label, "label"))

    elements.append(text(60, 655, "Source: eligible donor-year EDA cohort", "note"))
    elements.append("</svg>")
    TARGET_SVG.write_text("\n".join(elements), encoding="utf-8")


def create_giving_distributions(
    outcome_counts: Counter,
    distribution_counts: Counter,
) -> None:
    """Create four normalized giving-distribution panels."""
    width, height = 1500, 980
    elements = svg_start(width, height, "Giving distributions by conversion outcome")
    elements.extend(
        [
            text(60, 55, "Giving distributions by conversion outcome", "title"),
            text(
                60,
                82,
                "Within-outcome percentages make the rare converter group comparable",
                "subtitle",
            ),
        ]
    )
    legend_x = 1040
    elements.append(rect(legend_x, 44, 18, 18, NONCONVERTER_COLOR, 2))
    elements.append(text(legend_x + 26, 58, "Non-converter", "label"))
    elements.append(rect(legend_x + 170, 44, 18, 18, CONVERTER_COLOR, 2))
    elements.append(text(legend_x + 196, 58, "Converter", "label"))

    panel_positions = [
        (50, 115),
        (770, 115),
        (50, 535),
        (770, 535),
    ]
    panel_w, panel_h = 680, 370

    for (feature, title_value, bands), (px, py) in zip(
        DISTRIBUTION_SPECS,
        panel_positions,
    ):
        elements.append(rect(px, py, panel_w, panel_h, PANEL_BACKGROUND, 12))
        elements.append(text(px + 24, py + 34, title_value, "panel-title"))
        labels = [band[0] for band in bands]
        shares = {
            outcome: [
                distribution_counts[(feature, outcome, label)]
                / outcome_counts[outcome]
                for label in labels
            ]
            for outcome in ["False", "True"]
        }
        max_share = max(max(values) for values in shares.values())
        y_max = max(0.10, math.ceil(max_share * 10) / 10)
        chart_x = px + 65
        chart_y = py + 65
        chart_w = panel_w - 90
        chart_h = 230
        for tick_index in range(5):
            tick = y_max * tick_index / 4
            y = chart_y + chart_h - (tick / y_max) * chart_h
            elements.append(line(chart_x, y, chart_x + chart_w, y))
            elements.append(text(chart_x - 8, y + 4, f"{tick:.0%}", "axis", "end"))

        group_width = chart_w / len(labels)
        bar_width = min(22, group_width * 0.30)
        for index, label in enumerate(labels):
            center = chart_x + group_width * (index + 0.5)
            for outcome_index, (outcome, color) in enumerate(
                [("False", NONCONVERTER_COLOR), ("True", CONVERTER_COLOR)]
            ):
                share = shares[outcome][index]
                bar_height = (share / y_max) * chart_h
                x = center + (outcome_index - 0.5) * (bar_width + 3) - bar_width / 2
                y = chart_y + chart_h - bar_height
                elements.append(rect(x, y, bar_width, bar_height, color, 2))
            label_x = center
            label_y = chart_y + chart_h + 18
            elements.append(
                text(
                    label_x,
                    label_y,
                    label,
                    "axis",
                    "end",
                    f"rotate(-35 {label_x:.1f} {label_y:.1f})",
                )
            )

    elements.append(
        text(
            60,
            950,
            "Bars show the percentage within each outcome group, not raw counts.",
            "note",
        )
    )
    elements.append("</svg>")
    GIVING_SVG.write_text("\n".join(elements), encoding="utf-8")


def write_lift_data(
    lift_counts: Counter,
    overall_rate: float,
) -> list[dict[str, str | int | float]]:
    """Write band-level conversion rates and lift to CSV."""
    rows: list[dict[str, str | int | float]] = []
    with LIFT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "measure",
            "band",
            "observations",
            "conversions",
            "conversion_rate",
            "conversion_rate_ci_lower",
            "conversion_rate_ci_upper",
            "lift_vs_overall",
            "lift_ci_lower",
            "lift_ci_upper",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for feature, title_value, bands in LIFT_SPECS:
            for label, *_ in bands:
                observations = lift_counts[(feature, label, "rows")]
                conversions = lift_counts[(feature, label, "conversions")]
                conversion_rate = conversions / observations if observations else 0
                ci_lower, ci_upper = wilson_interval(conversions, observations)
                lift = conversion_rate / overall_rate if overall_rate else 0
                row = {
                    "measure": title_value,
                    "band": label,
                    "observations": observations,
                    "conversions": conversions,
                    "conversion_rate": f"{conversion_rate:.8f}",
                    "conversion_rate_ci_lower": f"{ci_lower:.8f}",
                    "conversion_rate_ci_upper": f"{ci_upper:.8f}",
                    "lift_vs_overall": f"{lift:.4f}",
                    "lift_ci_lower": f"{ci_lower / overall_rate:.4f}",
                    "lift_ci_upper": f"{ci_upper / overall_rate:.4f}",
                }
                writer.writerow(row)
                rows.append(row)
    return rows


def create_converter_lift(
    lift_rows: list[dict[str, str | int | float]],
    overall_rate: float,
) -> None:
    """Create lift charts for current and three-year giving bands."""
    width, height = 1500, 760
    elements = svg_start(width, height, "Converter lift by giving band")
    elements.extend(
        [
            text(60, 55, "Converter lift by giving band", "title"),
            text(
                60,
                82,
                (
                    f"Overall conversion rate = {overall_rate:.4%}; bars show lift, "
                    "error bars show 95% Wilson intervals"
                ),
                "subtitle",
            ),
        ]
    )

    panel_positions = [(50, 120), (770, 120)]
    panel_w, panel_h = 680, 540
    grouped_rows = {
        title_value: [row for row in lift_rows if row["measure"] == title_value]
        for _, title_value, _ in LIFT_SPECS
    }
    max_lift = max(float(row["lift_ci_upper"]) for row in lift_rows)
    if max_lift > 100:
        y_max = math.ceil(max_lift / 50) * 50
    elif max_lift > 20:
        y_max = math.ceil(max_lift / 10) * 10
    else:
        y_max = max(2.0, math.ceil(max_lift / 2) * 2)

    for (_, title_value, _), (px, py) in zip(LIFT_SPECS, panel_positions):
        rows = grouped_rows[title_value]
        elements.append(rect(px, py, panel_w, panel_h, PANEL_BACKGROUND, 12))
        elements.append(text(px + 24, py + 36, title_value, "panel-title"))
        chart_x = px + 65
        chart_y = py + 70
        chart_w = panel_w - 90
        chart_h = 335

        for tick_index in range(6):
            tick = y_max * tick_index / 5
            y = chart_y + chart_h - (tick / y_max) * chart_h
            elements.append(line(chart_x, y, chart_x + chart_w, y))
            elements.append(text(chart_x - 8, y + 4, f"{tick:.0f}×", "axis", "end"))

        baseline_y = chart_y + chart_h - (1 / y_max) * chart_h
        elements.append(
            line(
                chart_x,
                baseline_y,
                chart_x + chart_w,
                baseline_y,
                CONVERTER_COLOR,
                2,
                "6 5",
            )
        )

        group_width = chart_w / len(rows)
        bar_width = min(48, group_width * 0.58)
        for index, row in enumerate(rows):
            lift = float(row["lift_vs_overall"])
            lift_lower = float(row["lift_ci_lower"])
            lift_upper = float(row["lift_ci_upper"])
            conversion_rate = float(row["conversion_rate"])
            observations = int(row["observations"])
            conversions = int(row["conversions"])
            center = chart_x + group_width * (index + 0.5)
            bar_height = (lift / y_max) * chart_h
            x = center - bar_width / 2
            y = chart_y + chart_h - bar_height
            elements.append(rect(x, y, bar_width, bar_height, CONVERTER_COLOR, 3))
            error_low_y = chart_y + chart_h - (lift_lower / y_max) * chart_h
            error_high_y = chart_y + chart_h - (lift_upper / y_max) * chart_h
            elements.append(
                line(center, error_low_y, center, error_high_y, TEXT_COLOR, 1.5)
            )
            elements.append(
                line(center - 6, error_low_y, center + 6, error_low_y, TEXT_COLOR, 1.5)
            )
            elements.append(
                line(center - 6, error_high_y, center + 6, error_high_y, TEXT_COLOR, 1.5)
            )
            elements.append(
                text(
                    center,
                    max(error_high_y - 8, chart_y + 12),
                    f"{lift:.1f}×",
                    "value",
                    "middle",
                )
            )
            label_y = chart_y + chart_h + 20
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
                    chart_y + chart_h + 73,
                    f"{conversions}/{observations:,}",
                    "axis",
                    "middle",
                )
            )
            elements.append(
                text(
                    center,
                    chart_y + chart_h + 91,
                    f"{conversion_rate:.3%}",
                    "axis",
                    "middle",
                )
            )

        elements.append(
            text(
                px + 24,
                py + panel_h - 25,
                "Labels below bars show conversions/observations and conversion rate.",
                "note",
            )
        )

    elements.append("</svg>")
    LIFT_SVG.write_text("\n".join(elements), encoding="utf-8")


def write_fy_lift_data(
    fy_lift_counts: Counter,
) -> list[dict[str, str | int | float]]:
    """Write broad-tier conversion rates and lift for each fiscal year."""
    rows: list[dict[str, str | int | float]] = []
    with FY_LIFT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "fiscal_year",
            "measure",
            "band",
            "observations",
            "conversions",
            "conversion_rate",
            "conversion_rate_ci_lower",
            "conversion_rate_ci_upper",
            "fiscal_year_baseline_rate",
            "lift_vs_fiscal_year",
            "lift_ci_lower",
            "lift_ci_upper",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for fiscal_year in ["2023", "2024", "2025"]:
            year_rows = fy_lift_counts[(fiscal_year, "all", "rows")]
            year_conversions = fy_lift_counts[
                (fiscal_year, "all", "conversions")
            ]
            baseline_rate = year_conversions / year_rows
            for feature, title_value, bands in FY_LIFT_SPECS:
                for label, *_ in bands:
                    observations = fy_lift_counts[
                        (fiscal_year, feature, label, "rows")
                    ]
                    conversions = fy_lift_counts[
                        (fiscal_year, feature, label, "conversions")
                    ]
                    conversion_rate = conversions / observations if observations else 0
                    ci_lower, ci_upper = wilson_interval(conversions, observations)
                    row = {
                        "fiscal_year": fiscal_year,
                        "measure": title_value,
                        "band": label,
                        "observations": observations,
                        "conversions": conversions,
                        "conversion_rate": f"{conversion_rate:.8f}",
                        "conversion_rate_ci_lower": f"{ci_lower:.8f}",
                        "conversion_rate_ci_upper": f"{ci_upper:.8f}",
                        "fiscal_year_baseline_rate": f"{baseline_rate:.8f}",
                        "lift_vs_fiscal_year": f"{conversion_rate / baseline_rate:.4f}",
                        "lift_ci_lower": f"{ci_lower / baseline_rate:.4f}",
                        "lift_ci_upper": f"{ci_upper / baseline_rate:.4f}",
                    }
                    writer.writerow(row)
                    rows.append(row)
    return rows


def create_fy_lift_comparison(
    fy_rows: list[dict[str, str | int | float]],
) -> None:
    """Create fiscal-year rate comparisons with Wilson confidence intervals."""
    width, height = 1500, 820
    elements = svg_start(width, height, "Conversion rates by fiscal year and giving tier")
    elements.extend(
        [
            text(60, 55, "Conversion rates by fiscal year and giving tier", "title"),
            text(
                60,
                82,
                "Broad tiers reduce sparse year-by-band cells; error bars are 95% Wilson intervals",
                "subtitle",
            ),
        ]
    )
    for index, fiscal_year in enumerate(["2023", "2024", "2025"]):
        x = 1010 + index * 125
        elements.append(rect(x, 44, 16, 16, FY_COLORS[fiscal_year], 2))
        elements.append(text(x + 23, 57, f"FY{fiscal_year}", "label"))

    panel_positions = [(50, 120), (770, 120)]
    panel_w, panel_h = 680, 610
    grouped = {
        title_value: [row for row in fy_rows if row["measure"] == title_value]
        for _, title_value, _ in FY_LIFT_SPECS
    }

    for (_, title_value, bands), (px, py) in zip(FY_LIFT_SPECS, panel_positions):
        rows = grouped[title_value]
        max_upper = max(float(row["conversion_rate_ci_upper"]) for row in rows)
        if max_upper > 0.10:
            y_max = math.ceil(max_upper / 0.05) * 0.05
        elif max_upper > 0.02:
            y_max = math.ceil(max_upper / 0.01) * 0.01
        else:
            y_max = max(0.005, math.ceil(max_upper / 0.0025) * 0.0025)

        elements.append(rect(px, py, panel_w, panel_h, PANEL_BACKGROUND, 12))
        elements.append(text(px + 24, py + 36, title_value, "panel-title"))
        chart_x = px + 72
        chart_y = py + 72
        chart_w = panel_w - 105
        chart_h = 350
        for tick_index in range(6):
            tick = y_max * tick_index / 5
            y = chart_y + chart_h - (tick / y_max) * chart_h
            elements.append(line(chart_x, y, chart_x + chart_w, y))
            elements.append(text(chart_x - 9, y + 4, f"{tick:.1%}", "axis", "end"))

        labels = [band[0] for band in bands]
        group_width = chart_w / len(labels)
        bar_width = 32
        for band_index, label in enumerate(labels):
            center = chart_x + group_width * (band_index + 0.5)
            band_rows = [row for row in rows if row["band"] == label]
            band_rows.sort(key=lambda row: str(row["fiscal_year"]))
            for year_index, row in enumerate(band_rows):
                fiscal_year = str(row["fiscal_year"])
                conversion_rate = float(row["conversion_rate"])
                ci_lower = float(row["conversion_rate_ci_lower"])
                ci_upper = float(row["conversion_rate_ci_upper"])
                observations = int(row["observations"])
                conversions = int(row["conversions"])
                lift = float(row["lift_vs_fiscal_year"])
                x = center + (year_index - 1) * (bar_width + 6) - bar_width / 2
                bar_height = (conversion_rate / y_max) * chart_h
                y = chart_y + chart_h - bar_height
                color = FY_COLORS[fiscal_year]
                elements.append(rect(x, y, bar_width, bar_height, color, 2))

                error_low_y = chart_y + chart_h - (ci_lower / y_max) * chart_h
                error_high_y = chart_y + chart_h - (ci_upper / y_max) * chart_h
                error_center = x + bar_width / 2
                elements.append(
                    line(
                        error_center,
                        error_low_y,
                        error_center,
                        error_high_y,
                        TEXT_COLOR,
                        1.3,
                    )
                )
                elements.append(
                    line(
                        error_center - 5,
                        error_low_y,
                        error_center + 5,
                        error_low_y,
                        TEXT_COLOR,
                        1.3,
                    )
                )
                elements.append(
                    line(
                        error_center - 5,
                        error_high_y,
                        error_center + 5,
                        error_high_y,
                        TEXT_COLOR,
                        1.3,
                    )
                )
                elements.append(
                    text(
                        error_center,
                        chart_y + chart_h + 74 + year_index * 18,
                        f"FY{fiscal_year}: {conversions}/{observations:,}, {lift:.1f}×",
                        "axis",
                        "middle",
                    )
                )
            elements.append(
                text(center, chart_y + chart_h + 26, label, "label", "middle")
            )

        elements.append(
            text(
                px + 24,
                py + panel_h - 25,
                "Labels show conversions/observations and lift versus that year's baseline.",
                "note",
            )
        )

    elements.append("</svg>")
    FY_LIFT_SVG.write_text("\n".join(elements), encoding="utf-8")


def create_gallery() -> None:
    """Create a Markdown gallery linking the generated figures and data."""
    content = """# Initial EDA visualizations

## Target balance

![Target balance and conversion rates](figures/target_balance.svg)

## Giving distributions

![Giving distributions by conversion outcome](figures/giving_distributions.svg)

## Converter lift

![Converter lift by giving band](figures/converter_lift.svg)

Error bars are 95% Wilson intervals. Lift intervals divide the Wilson rate
interval by the overall cohort conversion rate; uncertainty in the baseline
rate is not propagated.

## Fiscal-year comparison

![Conversion rates by fiscal year and giving tier](figures/converter_lift_by_fiscal_year.svg)

## Donor-clustered bootstrap intervals

![Donor-clustered bootstrap lift intervals](figures/converter_lift_cluster_bootstrap.svg)

These intervals resample complete donor histories so repeated donor-year
observations remain together.

The underlying lift values are available in
[`converter_lift_by_giving_band.csv`](converter_lift_by_giving_band.csv) and
[`converter_lift_by_fiscal_year.csv`](converter_lift_by_fiscal_year.csv).
Clustered-bootstrap intervals are available in
[`converter_lift_cluster_bootstrap.csv`](converter_lift_cluster_bootstrap.csv).
"""
    GALLERY_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        outcome_counts,
        target_groups,
        distribution_counts,
        lift_counts,
        fy_lift_counts,
        total_rows,
        total_conversions,
    ) = load_counts()
    overall_rate = total_conversions / total_rows

    create_target_balance(
        outcome_counts,
        target_groups,
        total_rows,
        total_conversions,
    )
    create_giving_distributions(outcome_counts, distribution_counts)
    lift_rows = write_lift_data(lift_counts, overall_rate)
    create_converter_lift(lift_rows, overall_rate)
    fy_lift_rows = write_fy_lift_data(fy_lift_counts)
    create_fy_lift_comparison(fy_lift_rows)
    create_gallery()

    print(f"Rows analyzed: {total_rows:,}")
    print(f"Conversions: {total_conversions:,}")
    print(f"Overall conversion rate: {overall_rate:.6%}")
    print(f"Target balance chart: {TARGET_SVG}")
    print(f"Giving distribution chart: {GIVING_SVG}")
    print(f"Converter lift chart: {LIFT_SVG}")
    print(f"Lift data: {LIFT_CSV}")
    print(f"Fiscal-year lift chart: {FY_LIFT_SVG}")
    print(f"Fiscal-year lift data: {FY_LIFT_CSV}")
    print(f"Gallery: {GALLERY_PATH}")


if __name__ == "__main__":
    main()
