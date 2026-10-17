"""Point-in-time engagement features defined in ``pbs_utah_group_eda.html``.

This module has no import-time I/O and never writes datasets or caches. Call
``add_engagement_features`` from the modeling build after giving features exist.
Campaign-member rows are solicitations, including exact repeated rows: they must
not be deduplicated. Responses are dated independently of campaign starts.
Passport absence is unknown viewing, not zero viewing. Cultivation actuals and
evaluation weights must never be included automatically in a predictor matrix.

Authoritative report sections: Solicitation and Passport Viewing; Handling
Contacted Donors; Appendix / Master Data Dictionary. Published source coverage
is FY2021--FY2026; cultivation records themselves identify observed assignments.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_YEAR = 2021
LAST_YEAR = 2026
CHUNK_SIZE = 250_000

# Every appended column is documented, so callers can build an explicit manifest
# and select predictors by role rather than accidentally using officer metadata.
FEATURE_DOCUMENTATION = {
    "solicitations": {
        "role": "optional_predictor",
        "description": "Solicitation rows in FY t, including repeated asks; unknown outside source coverage.",
    },
    "responses": {
        "role": "optional_predictor",
        "description": "True responses with a valid response date in FY t (not campaign FY).",
    },
    "by_mail": {
        "role": "optional_predictor",
        "description": "Any Mail solicitation in FY t; unknown outside source coverage.",
    },
    "by_email": {
        "role": "optional_predictor",
        "description": "Any Email solicitation in FY t; unknown outside source coverage.",
    },
    "by_digital": {
        "role": "optional_predictor",
        "description": "Any Digital solicitation in FY t; unknown outside source coverage.",
    },
    "by_phone_or_in_person": {
        "role": "optional_predictor",
        "description": "Any Phone or In-Person solicitation in FY t; unknown outside source coverage.",
    },
    "solicitations_3yr": {
        "role": "optional_predictor",
        "description": "Solicitations in observed years of FY t-2 through t; use with coverage years.",
    },
    "solicitations_3yr_coverage_years": {
        "role": "metadata",
        "description": "Number of source-covered years in t-2:t (FY2021 has one; FY2022 has two).",
    },
    "titles": {
        "role": "optional_predictor",
        "description": "Sum of Passport titles viewed across genres in FY t; NaN without a viewing record.",
    },
    "genres": {
        "role": "optional_predictor",
        "description": "Distinct recorded Passport genres in FY t; NaN without a viewing record.",
    },
    "share_watched": {
        "role": "optional_predictor",
        "description": "Title-weighted mean_percent_watched (0-1) in FY t; NaN when unobserved/undefined.",
    },
    "viewed_t": {
        "role": "optional_predictor",
        "description": "A linked Passport record exists in FY t; False does not establish zero viewing.",
    },
    "viewed_by_t": {
        "role": "optional_predictor",
        "description": "Any Passport record from FY2021 through FY t, never using a later record.",
    },
    "cultivation_selected_actual": {
        "role": "evaluation",
        "description": "Actual selection flag in FY t; absent roster rows use False, not observed status.",
    },
    "cultivation_contacted_actual": {
        "role": "evaluation",
        "description": "Actual officer contact assignment in FY t, preserved even for FY2026 scoring.",
    },
    "cultivation_holdout_actual": {
        "role": "evaluation",
        "description": "Actual holdout flag in FY t; absent roster rows use False by report convention.",
    },
    "cultivation_observed": {
        "role": "metadata",
        "description": "This donor-year has a cultivation roster row; distinguishes missing history from recorded False.",
    },
    "contacted_t": {
        "role": "training_only",
        "description": "Contact assignment in FY t for training; forced False for FY2026 scoring only.",
    },
    "heldout_t": {
        "role": "training_only",
        "description": "Selected AND held out in FY t for training; forced False for FY2026 scoring only.",
    },
    "no_contact_pool": {
        "role": "evaluation",
        "description": "Not actually contacted in FY t; actuals, not neutralized training flags, define this pool.",
    },
    "no_contact_weight": {
        "role": "evaluation",
        "description": "0 for actual contact, full-roster selected/heldout for selected holdouts, otherwise 1.",
    },
    "portfolio_id": {
        "role": "metadata",
        "description": "Officer portfolio for the observed assignment; never a predictor; missing without a roster row.",
    },
    "program_adopted": {
        "role": "evaluation",
        "description": "Recorded portfolio adoption flag for analysis only; missing without a roster row.",
    },
}

_CHANNELS = {
    "mail": "by_mail",
    "email": "by_email",
    "digital": "by_digital",
    "phone": "by_phone_or_in_person",
    "in-person": "by_phone_or_in_person",
}
_KNOWN_OTHER_CHANNELS = {"legacy", "miscellaneous", "third party"}
_BOOLEAN_VALUES = {"true": True, "false": False, "1": True, "0": False}
_KEYS = ["constituent_id", "t"]


def fiscal_year(values: pd.Series) -> pd.Series:
    """Parse ISO or US m/d/y dates; return nullable July--June ending FYs.

    Invalid or empty dates return pd.NA. ``format='mixed'`` is intentional:
    pandas' single inferred format can otherwise reject valid mixed input.
    """
    dates = pd.to_datetime(values.astype("string").str.strip(), format="mixed", errors="coerce")
    return (dates.dt.year + dates.dt.month.ge(7)).astype("Int64")


def _read_csv(path: Path, columns: list[str], *, chunks: bool = False):
    """Read only required fields, preserving identifiers and literal NA text."""
    return pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=CHUNK_SIZE if chunks else None,
    )


def _missingness(frame: pd.DataFrame, counts: Counter) -> None:
    for column in frame.columns:
        counts[column] += int(frame[column].str.strip().eq("").sum())


def _value_counts(values: pd.Series, counts: Counter) -> None:
    counts.update({str(k): int(v) for k, v in values.value_counts(dropna=False).items()})


def _years(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    valid = numbers.notna() & np.isfinite(numbers) & numbers.mod(1).eq(0)
    return numbers.where(valid).astype("Int64")


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = values.str.strip().str.lower()
    parsed = normalized.map(_BOOLEAN_VALUES)
    if parsed.isna().any():
        bad = normalized[parsed.isna()].value_counts().to_dict()
        raise ValueError(f"cultivation.{name}: missing or invalid Boolean values: {bad}")
    return parsed.astype(bool)


def _campaign_features(keys: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    codes = _read_csv(data_dir / "campaign_codes.csv", ["Marketing Code", "Communication Method"])
    code_missing = Counter()
    _missingness(codes, code_missing)
    if codes["Marketing Code"].str.strip().eq("").any():
        raise ValueError("campaign_codes contains a missing Marketing Code")
    if codes["Marketing Code"].duplicated().any():
        raise ValueError("campaign_codes must have one row per Marketing Code (no join multiplication)")
    methods = codes["Communication Method"].str.strip().str.lower()
    method_map = pd.Series(methods.to_numpy(), index=codes["Marketing Code"])
    unknown_reference = Counter()
    _value_counts(
        codes.loc[~methods.isin(set(_CHANNELS) | _KNOWN_OTHER_CHANNELS), "Communication Method"],
        unknown_reference,
    )
    columns = ["constituent_id", "marketing_code", "campaign_start_date", "responded", "responded_date"]
    audit = Counter()
    missing = Counter()
    response_values = Counter()
    unknown_codes = Counter()
    unknown_methods = Counter()
    other_methods = Counter()
    source_year_counts = Counter()
    response_year_counts = Counter()
    totals = None
    donors = set(keys["constituent_id"])
    needed_years = {
        year for t in keys["t"].unique() for year in range(int(t) - 2, int(t) + 1)
        if FIRST_YEAR <= year <= LAST_YEAR
    }
    count_columns = ["solicitations", "responses", *dict.fromkeys(_CHANNELS.values())]

    with _read_csv(data_dir / "campaign_members.txt", columns, chunks=True) as reader:
        for chunk in reader:
            audit["rows_read"] += len(chunk)
            _missingness(chunk, missing)
            start_fy = fiscal_year(chunk["campaign_start_date"])
            response_fy = fiscal_year(chunk["responded_date"])
            start_blank = chunk["campaign_start_date"].str.strip().eq("")
            response_blank = chunk["responded_date"].str.strip().eq("")
            responded_raw = chunk["responded"].str.strip().str.lower()
            responded = responded_raw.map(_BOOLEAN_VALUES)
            is_response = responded.eq(True)
            valid_response = is_response & response_fy.notna()
            audit["invalid_campaign_start_dates"] += int((start_fy.isna() & ~start_blank).sum())
            audit["invalid_responded_dates"] += int((response_fy.isna() & ~response_blank).sum())
            audit["invalid_or_missing_response_flags"] += int(responded.isna().sum())
            audit["responded_true"] += int(is_response.sum())
            audit["responded_true_missing_date"] += int((is_response & response_blank).sum())
            audit["responded_true_invalid_date"] += int((is_response & response_fy.isna() & ~response_blank).sum())
            audit["responded_false_with_date"] += int((responded.eq(False) & ~response_blank).sum())
            audit["response_campaign_fy_mismatch"] += int((valid_response & start_fy.notna() & response_fy.ne(start_fy)).sum())
            audit["valid_responses"] += int(valid_response.sum())
            audit["campaigns_outside_published_coverage"] += int((start_fy.notna() & ~start_fy.between(FIRST_YEAR, LAST_YEAR)).sum())
            audit["responses_outside_published_coverage"] += int((valid_response & ~response_fy.between(FIRST_YEAR, LAST_YEAR)).sum())
            _value_counts(responded_raw, response_values)
            _value_counts(start_fy.dropna().astype(str), source_year_counts)
            _value_counts(response_fy[valid_response].astype(str), response_year_counts)

            methods_chunk = chunk["marketing_code"].map(method_map)
            unmapped = methods_chunk.isna()
            _value_counts(chunk.loc[unmapped, "marketing_code"], unknown_codes)
            unknown = ~unmapped & ~methods_chunk.isin(set(_CHANNELS) | _KNOWN_OTHER_CHANNELS)
            _value_counts(methods_chunk[unknown], unknown_methods)
            _value_counts(methods_chunk[methods_chunk.isin(_KNOWN_OTHER_CHANNELS)], other_methods)
            relevant = chunk["constituent_id"].isin(donors)
            sol_mask = relevant & start_fy.isin(needed_years)
            response_mask = relevant & valid_response & response_fy.isin(needed_years)

            sol = pd.DataFrame({
                "constituent_id": chunk.loc[sol_mask, "constituent_id"],
                "t": start_fy[sol_mask].astype("int64"),
                "solicitations": 1,
            })
            channels = methods_chunk[sol_mask].map(_CHANNELS)
            for column in dict.fromkeys(_CHANNELS.values()):
                sol[column] = channels.eq(column).astype("int64")
            sol = sol.groupby(_KEYS, sort=False).sum()
            resp = pd.DataFrame({
                "constituent_id": chunk.loc[response_mask, "constituent_id"],
                "t": response_fy[response_mask].astype("int64"),
                "responses": 1,
            }).groupby(_KEYS, sort=False).sum()
            part = sol.join(resp, how="outer").reindex(columns=count_columns).fillna(0)
            totals = part if totals is None else totals.add(part, fill_value=0)

    if totals is None:
        totals = pd.DataFrame(columns=count_columns, index=pd.MultiIndex.from_arrays([[], []], names=_KEYS))
    out = keys.merge(totals.reset_index(), on=_KEYS, how="left", sort=False, validate="one_to_one")
    covered = out["t"].between(FIRST_YEAR, LAST_YEAR)
    for column in ("solicitations", "responses"):
        out[column] = out[column].fillna(0).where(covered).astype(float)
    for column in dict.fromkeys(_CHANNELS.values()):
        out[column] = out[column].fillna(0).gt(0).astype("boolean").where(covered, pd.NA)

    # Look up each exact FY, rather than rolling over sparse donor-year rows.
    # This retains actual zero-event years and never turns gaps into extra years.
    rolling = np.zeros(len(out), dtype=float)
    coverage = np.zeros(len(out), dtype=np.int64)
    for lag in range(3):
        lookups = pd.MultiIndex.from_arrays([out["constituent_id"], out["t"] - lag], names=_KEYS)
        observed = (out["t"] - lag).between(FIRST_YEAR, LAST_YEAR).to_numpy()
        rolling += totals["solicitations"].reindex(lookups).fillna(0).to_numpy(dtype=float) * observed
        coverage += observed
    out["solicitations_3yr"] = np.where(coverage > 0, rolling, np.nan)
    out["solicitations_3yr_coverage_years"] = coverage
    return out, {
        **{k: int(v) for k, v in audit.items()},
        "missing_fields": dict(missing),
        "response_flag_values": dict(response_values),
        "campaign_rows_by_fy": dict(source_year_counts),
        "valid_response_rows_by_fy": dict(response_year_counts),
        "unknown_marketing_codes": dict(unknown_codes),
        "unknown_communication_methods_in_members": dict(unknown_methods),
        "known_other_channel_rows": dict(other_methods),
        "exact_repeats_policy": "All source rows counted; no deduplication.",
        "unknown_channel_policy": "Count in total solicitations, set no known-channel flag, audit explicitly.",
        "campaign_codes": {
            "rows_read": int(len(codes)),
            "missing_fields": dict(code_missing),
            "unknown_communication_methods": dict(unknown_reference),
        },
    }


def _passport_features(keys: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    columns = ["constituent_id", "fiscal_year", "genre", "titles_viewed", "mean_percent_watched"]
    missing = Counter()
    audit = Counter()
    parts = []
    donors = set(keys["constituent_id"])
    max_t = int(keys["t"].max()) if len(keys) else FIRST_YEAR - 1
    with _read_csv(data_dir / "passport_viewing.txt", columns, chunks=True) as reader:
        for chunk in reader:
            audit["rows_read"] += len(chunk)
            _missingness(chunk, missing)
            fy = _years(chunk["fiscal_year"])
            titles = pd.to_numeric(chunk["titles_viewed"], errors="coerce")
            share = pd.to_numeric(chunk["mean_percent_watched"], errors="coerce")
            bad_titles = titles.isna() | ~np.isfinite(titles) | titles.lt(0) | titles.mod(1).ne(0)
            bad_share = share.isna() | ~np.isfinite(share) | ~share.between(0, 1)
            audit["invalid_or_missing_fiscal_year"] += int(fy.isna().sum())
            audit["invalid_or_missing_title_count"] += int(bad_titles.sum())
            audit["invalid_or_missing_share"] += int(bad_share.sum())
            audit["rows_outside_published_coverage"] += int((fy.notna() & ~fy.between(FIRST_YEAR, LAST_YEAR)).sum())
            # Malformed measurement rows cannot be silently converted to zeros.
            if (bad_titles | bad_share).any():
                raise ValueError("passport_viewing has missing/invalid titles or mean_percent_watched; expected nonnegative integer titles and shares in [0, 1]")
            keep = chunk["constituent_id"].isin(donors) & fy.between(FIRST_YEAR, min(max_t, LAST_YEAR))
            part = chunk.loc[keep, ["constituent_id", "genre"]].copy()
            part["t"] = fy[keep].astype("int64")
            part["titles"] = titles[keep].astype(float)
            part["weighted_share"] = titles[keep] * share[keep]
            # Aggregate across chunks only after concatenation so distinct genres
            # are not double counted at a chunk boundary.
            parts.append(part)
    pv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["constituent_id", "genre", "t", "titles", "weighted_share"])
    pv["genre"] = pv["genre"].replace(r"^\s*$", pd.NA, regex=True)
    annual = pv.groupby(_KEYS, sort=False).agg(
        titles=("titles", "sum"),
        genres=("genre", "nunique"),
        weighted_share=("weighted_share", "sum"),
    ).reset_index()
    annual["share_watched"] = annual.pop("weighted_share").div(annual["titles"].where(annual["titles"].ne(0)))
    annual["viewed_t"] = True
    first_view = pv.groupby("constituent_id", sort=False)["t"].min()
    out = keys.merge(annual, on=_KEYS, how="left", sort=False, validate="one_to_one")
    out["viewed_t"] = out["viewed_t"].eq(True)
    out["viewed_by_t"] = out["constituent_id"].map(first_view).le(out["t"]).fillna(False).astype(bool)
    for column in ("titles", "genres", "share_watched"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    audit["modeled_donor_years_with_record"] = int(out["viewed_t"].sum())
    audit["modeled_donor_years_without_record"] = int((~out["viewed_t"]).sum())
    audit["observed_donor_years_zero_titles"] = int(annual["titles"].eq(0).sum())
    return out, {
        **{k: int(v) for k, v in audit.items()},
        "missing_fields": dict(missing),
        "absent_record_policy": "titles, genres and share_watched stay NaN; viewing indicators are False.",
        "weighted_mean": "sum(titles_viewed * mean_percent_watched) / sum(titles_viewed)",
    }


def _cultivation_features(keys: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    columns = ["synthetic_id", "fiscal_year", "selected", "contacted", "holdout", "portfolio_id", "program_adopted"]
    cult = _read_csv(data_dir / "cultivation.csv", columns)
    missing = Counter()
    _missingness(cult, missing)
    cult = cult.rename(columns={"synthetic_id": "constituent_id"})
    cult["t"] = _years(cult.pop("fiscal_year"))
    if cult["t"].isna().any() or cult["constituent_id"].str.strip().eq("").any():
        raise ValueError("cultivation requires a valid constituent identifier and integer fiscal_year")
    cult["t"] = cult["t"].astype("int64")
    if cult.duplicated(_KEYS).any():
        raise ValueError("cultivation must be unique by donor and fiscal year, including exact duplicate rows")
    for column in ("selected", "contacted", "holdout", "program_adopted"):
        cult[column] = _strict_bool(cult[column], column)
    inconsistent = (
        (cult["contacted"] & cult["holdout"])
        | ((cult["contacted"] | cult["holdout"]) & ~cult["selected"])
        | (cult["selected"] & ~(cult["contacted"] | cult["holdout"]))
    )
    if inconsistent.any():
        raise ValueError(f"cultivation has {int(inconsistent.sum())} inconsistent selection/contact/holdout rows")

    # Crucially, calculate design weights BEFORE restricting to modeled donors.
    roster = cult.assign(heldout=cult["selected"] & cult["holdout"]).groupby("t", sort=True).agg(
        roster_count=("selected", "size"),
        selected_count=("selected", "sum"),
        contacted_count=("contacted", "sum"),
        heldout_count=("heldout", "sum"),
    )
    if ((roster["selected_count"] > 0) & roster["heldout_count"].eq(0)).any():
        raise ValueError("A cultivation year has selected donors but no heldouts: no-contact weights are undefined")
    ratios = roster["selected_count"].div(roster["heldout_count"].where(roster["heldout_count"].ne(0)))
    cult["cultivation_observed"] = True
    cult["portfolio_id"] = cult["portfolio_id"].replace(r"^\s*$", pd.NA, regex=True).astype("string")
    cult = cult.rename(columns={
        "selected": "cultivation_selected_actual",
        "contacted": "cultivation_contacted_actual",
        "holdout": "cultivation_holdout_actual",
    })
    out = keys.merge(cult, on=_KEYS, how="left", sort=False, validate="one_to_one")
    for column in ("cultivation_selected_actual", "cultivation_contacted_actual", "cultivation_holdout_actual", "cultivation_observed"):
        out[column] = out[column].eq(True)
    out["program_adopted"] = out["program_adopted"].astype("boolean")
    selected = out["cultivation_selected_actual"]
    contacted = out["cultivation_contacted_actual"]
    heldout = selected & out["cultivation_holdout_actual"]
    out["contacted_t"] = contacted & out["t"].ne(2026)
    out["heldout_t"] = heldout & out["t"].ne(2026)
    out["no_contact_pool"] = ~contacted
    out["no_contact_weight"] = np.where(contacted, 0.0, np.where(heldout, out["t"].map(ratios), 1.0))
    if not np.isfinite(out["no_contact_weight"]).all():
        raise AssertionError("Non-finite no-contact weights")
    by_year = {
        str(int(year)): {column: int(value) for column, value in row.items()}
        for year, row in roster.iterrows()
    }
    return out, {
        "rows_read": int(len(cult)),
        "missing_fields": dict(missing),
        "roster_by_fy": by_year,
        "duplicate_donor_years": 0,
        "inconsistent_flag_rows": 0,
        "modeled_donor_years_observed": int(out["cultivation_observed"].sum()),
        "modeled_donor_years_not_observed": int((~out["cultivation_observed"]).sum()),
        "modeled_fys_without_roster": sorted(int(t) for t in set(keys["t"]) - set(roster.index)),
        "fy2026_training_flags_neutralized": int(out["t"].eq(2026).sum()),
        "weight_denominator": "Full cultivation roster per FY, not modeled donor subset.",
        "absent_roster_policy": "Observed=False; actual flags=False by convention, not evidence of recorded nonselection.",
    }


def add_engagement_features(frame: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Append engagement features without changing input rows, order or index.

    Required inputs: string ``constituent_id``, integer ``t``, string ``segment``
    and string ``couple_id``, with unique (constituent_id, t, segment) rows.
    Annual features are shared across segments without multiplying rows. Giving
    features and all original columns are preserved verbatim. ``data_dir`` must
    contain the four source files named in the helper functions above.

    Returns a new frame and an operational, JSON-serializable audit. No input
    table is loaded until this function is called. Raises on schema violations,
    duplicate cultivation assignments or inconsistent cultivation flags.
    """
    required = ["constituent_id", "t", "segment", "couple_id"]
    if not frame.columns.is_unique:
        raise ValueError("Input has duplicate column names")
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required modeling columns: {sorted(absent)}")
    if frame[required].isna().any().any():
        raise ValueError("Required modeling columns cannot contain missing values")
    for column in ("constituent_id", "segment", "couple_id"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    if not pd.api.types.is_integer_dtype(frame["t"].dtype) or pd.api.types.is_bool_dtype(frame["t"].dtype):
        raise ValueError("t must have an integer dtype")
    row_key = ["constituent_id", "t", "segment"]
    if frame.duplicated(row_key).any():
        raise ValueError("Modeling rows must be unique by constituent_id, t and segment")
    collisions = set(FEATURE_DOCUMENTATION) & set(frame.columns)
    if collisions:
        raise ValueError(f"Refusing to overwrite existing engagement columns: {sorted(collisions)}")

    keys = frame[_KEYS].drop_duplicates().reset_index(drop=True)
    keys["t"] = keys["t"].astype("int64")
    data_dir = Path(data_dir)
    campaign, campaign_audit = _campaign_features(keys, data_dir)
    passport, passport_audit = _passport_features(keys, data_dir)
    cultivation, cultivation_audit = _cultivation_features(keys, data_dir)
    features = campaign.merge(passport, on=_KEYS, how="left", sort=False, validate="one_to_one")
    features = features.merge(cultivation, on=_KEYS, how="left", sort=False, validate="one_to_one")
    if features.duplicated(_KEYS).any() or len(features) != len(keys):
        raise AssertionError("Engagement feature construction changed donor-year cardinality")
    # Reindex directly to input keys; this also preserves duplicate *index labels*
    # and avoids pandas merge index resets or accidental changes in giving dtypes.
    lookup = pd.MultiIndex.from_frame(frame[_KEYS])
    aligned = features.set_index(_KEYS).reindex(lookup)
    if set(aligned.columns) != set(FEATURE_DOCUMENTATION):
        raise AssertionError("Feature documentation does not match appended columns")
    out = frame.copy()
    for column in FEATURE_DOCUMENTATION:
        out[column] = aligned[column].array
    if len(out) != len(frame) or not out.index.equals(frame.index) or out.duplicated(row_key).any():
        raise AssertionError("Engagement joins changed input rows/order/index")
    if not out.loc[:, frame.columns].equals(frame):
        raise AssertionError("Engagement construction changed original modeling data")
    return out, {
        "input_rows": int(len(frame)),
        "output_rows": int(len(out)),
        "unique_donor_years": int(len(keys)),
        "row_order_and_original_columns_preserved": True,
        "source_coverage_first_fy": FIRST_YEAR,
        "source_coverage_last_fy": LAST_YEAR,
        "campaign": campaign_audit,
        "passport": passport_audit,
        "cultivation": cultivation_audit,
        "appended_columns": list(FEATURE_DOCUMENTATION),
    }
