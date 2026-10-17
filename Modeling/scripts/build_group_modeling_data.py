"""Reconstruct modeling inputs from the finalized group EDA, not the old EDA.

No models are fitted. Raw files and earlier datasets are never modified.
Requires the already-installed pandas, numpy and pyarrow packages.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from group_engagement_features import FEATURE_DOCUMENTATION, add_engagement_features


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "Modeling"
DATA = ROOT / "Data"
CHUNK = 150_000
LAST_FY = 2026
THRESHOLD = 120_000  # Integer cents throughout accounting.
INPUTS = [
    "constituents_w_memberships.csv", "unite_payments.txt",
    "team_approach_legacy_payments.txt", "soft_credits.csv",
    "campaign_codes.csv", "campaign_members.txt", "passport_viewing.txt",
    "cultivation.csv", "officer_portfolios.csv",
]


def read_strings(path, **kwargs):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", **kwargs)


def is_present(values):
    return ~values.str.strip().str.lower().isin(["", "na", "nan", "null", "none"])


def cents(values):
    numbers = pd.to_numeric(values, errors="raise") * 100
    if not np.isfinite(numbers).all() or not np.isclose(numbers, numbers.round(), atol=0.00001, rtol=0).all():
        raise ValueError("Payment amounts must be finite and exact to cents")
    return numbers.round().astype("int64")


def parse_dates(values):
    return pd.to_datetime(values, format="mixed", errors="raise")


def fiscal_year(dates):
    if dates.isna().any():
        raise ValueError("Missing gift dates cannot be assigned a fiscal year")
    return (dates.dt.year + dates.dt.month.ge(7)).astype("int64")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_constituents():
    c = read_strings(DATA / INPUTS[0])
    if c["Constituent ID"].duplicated().any():
        raise ValueError("Constituent identifiers are not unique")
    c = c.rename(columns={
        "Constituent ID": "constituent_id", "Spouse Constituent ID": "spouse_id",
        "Age": "age_snapshot", "Gender": "gender", "Address State": "state",
        "Primary Constituent Type": "constituent_type",
    })
    orgs = set(c.loc[c["constituent_type"].eq("Organization"), "constituent_id"])
    all_ids = set(c["constituent_id"])
    people = c.loc[~c["constituent_id"].isin(orgs), [
        "constituent_id", "spouse_id", "age_snapshot", "gender", "state", "constituent_type",
    ]].copy()
    people["age_snapshot"] = pd.to_numeric(people["age_snapshot"], errors="coerce")
    people["gender"] = people["gender"].replace("", "Unknown")
    people["state"] = people["state"].replace("", pd.NA)
    ids = set(people["constituent_id"])
    bad_spouse = is_present(people["spouse_id"]) & ~people["spouse_id"].isin(ids)
    if bad_spouse.any():
        raise ValueError("Unresolved spouse links require review before couple qualification")
    people["spouse_id"] = people["spouse_id"].where(is_present(people["spouse_id"]), "")
    links = dict(zip(people["constituent_id"], people["spouse_id"]))
    nonreciprocal = [(a, b) for a, b in links.items() if b and links.get(b) != a]
    if nonreciprocal:
        raise ValueError(f"Nonreciprocal spouse links need a policy: {nonreciprocal[:5]}")
    people["couple_id"] = [min(a, b) if b else a for a, b in links.items()]
    return people, all_ids, orgs


def failure_rules(frame, kind, all_ids, codes):
    if kind == "unite":
        return {
            "invalid_donor": ~frame["donor_id"].isin(all_ids),
            "invalid_campaign": ~frame["marketing_code"].isin(codes),
            "nonpositive_amount": cents(frame["payment_amount"]).le(0),
        }
    if kind == "soft":
        return {
            "invalid_recipient": ~frame["Soft Credit Constituent ID"].isin(all_ids),
            "nonpositive_amount": cents(frame["Credit Amount"]).le(0),
        }
    linked = pd.concat([frame[f"constituent_id_{i}"].isin(all_ids) for i in range(1, 8)], axis=1).any(axis=1)
    return {"unlinked_legacy": ~linked}


def projections(columns, kind):
    if kind == "unite":
        return {r: [c for c in columns if c not in {"payment_id", field}] for r, field in {
            "invalid_donor": "donor_id", "invalid_campaign": "marketing_code",
            "nonpositive_amount": "payment_amount",
        }.items()}
    if kind == "soft":
        return {r: [c for c in columns if c not in {"Credit ID", field}] for r, field in {
            "invalid_recipient": "Soft Credit Constituent ID", "nonpositive_amount": "Credit Amount",
        }.items()}
    return {"unlinked_legacy": [c for c in columns if c != "ta_account_id" and not c.startswith("constituent_id_")]}


def new_parts():
    return {"annual": [], "organizations": [], "removals": [], "cleaning": {}}


def append_gifts(frame, kind, people_ids, orgs, parts):
    if kind == "unite":
        donor, date_col, amt_col = "donor_id", "credit_date", "payment_amount"
    elif kind == "soft":
        donor, date_col, amt_col = "Soft Credit Constituent ID", "Credit Date", "Credit Amount"
    else:
        donor, date_col, amt_col = "constituent_id_1", "gift_date", "payment_amount"
    amounts = cents(frame[amt_col])
    dates = parse_dates(frame[date_col])
    years = fiscal_year(dates)
    if kind == "unite":
        organization = frame[donor].isin(orgs)
        org_frame = pd.DataFrame({"fy": years[organization], "organization_cents": amounts[organization]})
        parts["organizations"].append(org_frame.groupby("fy").sum())
    keep = frame[donor].isin(people_ids) & amounts.gt(0)
    if kind == "legacy" and (~frame[donor].isin(people_ids)).any():
        raise ValueError("A retained legacy record has no valid first-listed individual")
    if kind == "legacy" and (amounts <= 0).any():
        raise ValueError("Nonpositive retained legacy gifts need explicit handling")
    g = pd.DataFrame({
        "constituent_id": frame.loc[keep, donor], "fy": years[keep],
        "amount_cents": amounts[keep], "last_gift_date": dates[keep],
    })
    if not g["fy"].between(1980, LAST_FY).all():
        raise ValueError("Gift dates outside documented historical period")
    for source in ["unite", "soft", "legacy"]:
        g[f"{source}_cents"] = g["amount_cents"] if kind == source else 0
    if kind == "unite":
        recurring = frame.loc[keep, "pledge_gift_type"].eq("Recurring")
        g["upgrade_gift"] = frame.loc[keep, "gift_type"].eq("Upgrade")
        g["additional_gift"] = frame.loc[keep, "gift_type"].eq("Additional")
    elif kind == "legacy":
        recurring = frame.loc[keep, "gift_kind"].eq("Sustaining")
        g["upgrade_gift"] = False
        g["additional_gift"] = False
    else:
        recurring = pd.Series(False, index=g.index)
        g["upgrade_gift"] = False
        g["additional_gift"] = False
    g["recurring_cents"] = g["amount_cents"].where(recurring, 0)
    g["recurring_count"] = recurring.astype("int64")
    g["gift_count"] = 1
    g["legacy_spouse_listed"] = (is_present(frame.loc[keep, "constituent_id_2"]) if kind == "legacy" else False)
    aggregations = {c: "sum" for c in [
        "amount_cents", "unite_cents", "soft_cents", "legacy_cents",
        "recurring_cents", "recurring_count", "gift_count",
    ]}
    aggregations.update({"last_gift_date": "max", "upgrade_gift": "max", "additional_gift": "max", "legacy_spouse_listed": "max"})
    a = g.groupby(["constituent_id", "fy"], sort=False).agg(aggregations)
    a["largest_gift_cents"] = g.groupby(["constituent_id", "fy"], sort=False)["amount_cents"].max()
    parts["annual"].append(a)


def clean_and_aggregate(path, kind, people_ids, all_ids, orgs, codes, parts):
    """Two passes; verify exact content twins before any build can succeed.

    Hashes shortlist matches, then complete tuples are compared to avoid hash
    collision errors. Only failed records with a verified valid twin are removed.
    """
    small = read_strings(path) if kind == "soft" else None
    exact_removed = 0
    if small is not None:
        dup = small.duplicated(keep="first")
        exact_removed = int(dup.sum())
        for index, row in small.loc[dup].iterrows():
            parts["removals"].append({"source": path.name, "source_row": int(index) + 2, "row_id": row["Credit ID"], "reason": "exact_duplicate", "valid_twin_verified": True})
        small = small.loc[~dup]
    buckets = {}
    failure_counts = defaultdict(int)
    raw_rows = exact_removed
    row_id_col = {"unite": "payment_id", "soft": "Credit ID", "legacy": "ta_account_id"}[kind]
    get_chunks = lambda: [small.copy()] if small is not None else read_strings(path, chunksize=CHUNK)
    for frame in get_chunks():
        raw_rows += len(frame)
        rules = failure_rules(frame, kind, all_ids, codes)
        if pd.concat(rules, axis=1).sum(axis=1).gt(1).any():
            raise ValueError(f"{kind}: overlapping failed checks need a multi-field policy")
        project = projections(frame.columns, kind)
        for reason, mask in rules.items():
            failure_counts[reason] += int(mask.sum())
            subset = frame.loc[mask, project[reason]]
            hashes = pd.util.hash_pandas_object(subset, index=False)
            target = buckets.setdefault(reason, defaultdict(set))
            for h, values in zip(hashes, subset.itertuples(index=False, name=None)):
                target[int(h)].add(values)
            for index, identifier in frame.loc[mask, row_id_col].items():
                parts["removals"].append({"source": path.name, "source_row": int(index) + 2, "row_id": identifier, "reason": reason, "valid_twin_verified": True})
    valid_count = 0
    for frame in get_chunks():
        rules = failure_rules(frame, kind, all_ids, codes)
        good = frame.loc[~pd.concat(rules, axis=1).any(axis=1)]
        valid_count += len(good)
        project = projections(frame.columns, kind)
        for reason, target in buckets.items():
            if not target:
                continue
            subset = good[project[reason]]
            hashes = pd.util.hash_pandas_object(subset, index=False)
            hit = hashes.isin(target)
            for h, values in zip(hashes[hit], subset.loc[hit].itertuples(index=False, name=None)):
                key = int(h)
                if key in target:
                    target[key].discard(values)
                    if not target[key]:
                        del target[key]
        append_gifts(good, kind, people_ids, orgs, parts)
    unmatched = {reason: sum(map(len, keys.values())) for reason, keys in buckets.items()}
    if any(unmatched.values()):
        raise ValueError(f"{kind}: failed rows without an identical valid twin; no final datasets written: {unmatched}")
    parts["cleaning"][kind] = {
        "raw_rows": raw_rows, "exact_duplicates_removed": exact_removed,
        "copy_rows_removed": dict(failure_counts), "clean_rows": valid_count,
        "unverified_copy_signatures": unmatched,
    }


def finish_panel(parts, people):
    group = pd.concat(parts["annual"]).groupby(level=[0, 1], sort=True)
    summed = ["amount_cents", "unite_cents", "soft_cents", "legacy_cents", "recurring_cents", "recurring_count", "gift_count"]
    maximum = ["last_gift_date", "upgrade_gift", "additional_gift", "legacy_spouse_listed", "largest_gift_cents"]
    p = group.agg({**dict.fromkeys(summed, "sum"), **dict.fromkeys(maximum, "max")}).reset_index()
    p = p.merge(people[["constituent_id", "spouse_id", "couple_id"]], how="left", on="constituent_id", validate="many_to_one")
    lookup = p.set_index(["constituent_id", "fy"])["amount_cents"]
    spouse_keys = pd.MultiIndex.from_arrays([p["spouse_id"], p["fy"]])
    p["spouse_cents"] = lookup.reindex(spouse_keys).fillna(0).to_numpy(dtype="int64")
    p["major_individual"] = p["amount_cents"].ge(THRESHOLD)
    p["major_combined_only"] = (
        ~p["major_individual"] & p["spouse_cents"].lt(THRESHOLD)
        & p["spouse_id"].ne("") & (p["amount_cents"] + p["spouse_cents"]).ge(THRESHOLD)
    )
    p["major"] = p["major_individual"] | p["major_combined_only"]
    p["tier_basis_cents"] = p["amount_cents"] + p["spouse_cents"].where(p["major_combined_only"], 0)
    assert not p.duplicated(["constituent_id", "fy"]).any()
    assert p["amount_cents"].eq(p["unite_cents"] + p["soft_cents"] + p["legacy_cents"]).all()
    return p


def make_eligible(panel, first_year, last_year):
    parts = []
    active = panel.loc[panel["fy"].between(first_year, last_year) & ~panel["major"], ["constituent_id", "fy"]].rename(columns={"fy": "t"})
    active["segment"] = "active"
    parts.append(active)
    # Union of gifts one or two years before t, excluding ALL current-year givers.
    candidates = pd.concat([panel[["constituent_id", "fy"]].assign(fy=panel["fy"] + lag) for lag in [1, 2]]).drop_duplicates()
    candidates = candidates.loc[candidates["fy"].between(first_year, last_year)]
    candidates = candidates.merge(panel[["constituent_id", "fy"]].assign(gave=True), on=["constituent_id", "fy"], how="left", validate="one_to_one")
    lapsed = candidates.loc[candidates["gave"].isna(), ["constituent_id", "fy"]].rename(columns={"fy": "t"})
    lapsed["segment"] = "lapsed"
    parts.append(lapsed)
    return pd.concat(parts, ignore_index=True).sort_values(["t", "constituent_id"]).reset_index(drop=True)


def giving_features(eligible, panel, people):
    f = eligible.copy()
    p = panel.set_index(["constituent_id", "fy"]).sort_index()
    lookup = lambda col, lag=0: p[col].reindex(pd.MultiIndex.from_arrays([f["constituent_id"], f["t"] - lag])).reset_index(drop=True)
    f["prior_total"] = lookup("amount_cents").fillna(0).astype("int64") / 100
    f["prev_total"] = lookup("amount_cents", 1).fillna(0).astype("int64") / 100
    f["two_years_before"] = f["prev_total"] + lookup("amount_cents", 2).fillna(0).astype("int64") / 100
    f["years_last5"] = sum(lookup("amount_cents", lag).fillna(0).gt(0).astype(int) for lag in range(5))
    first = panel.groupby("constituent_id")["fy"].min()
    first_major = panel.loc[panel["major"]].groupby("constituent_id")["fy"].min()
    first_soft = panel.loc[panel["soft_cents"].gt(0)].groupby("constituent_id")["fy"].min()
    f["years_since_first_gift"] = f["t"] - f["constituent_id"].map(first) + 1
    f["former_major"] = f["constituent_id"].map(first_major).lt(f["t"])
    f["soft_before"] = f["constituent_id"].map(first_soft).lt(f["t"])
    f["soft_in_t"] = lookup("soft_cents").fillna(0).gt(0)
    f["soft_amount_t"] = lookup("soft_cents").fillna(0) / 100
    f["giving_change"] = f["prior_total"] - f["prev_total"]
    f["giving_change_band"] = np.select([
        f["prev_total"].eq(0), f["giving_change"].gt(0), f["giving_change"].lt(0),
    ], ["no gift last year", "gave more", "gave less"], default="gave the same")
    f["largest_gift_t"] = lookup("largest_gift_cents").fillna(0) / 100
    f["largest_gift_share"] = f["largest_gift_t"] / f["prior_total"].where(f["prior_total"].ne(0))
    f["recurring_amount_t"] = lookup("recurring_cents").fillna(0) / 100
    f["recurring_count_t"] = lookup("recurring_count").fillna(0).astype("int64")
    f["sustainer"] = f["recurring_count_t"].gt(0)
    f["monthly_amount"] = (f["recurring_amount_t"] / f["recurring_count_t"].where(f["sustainer"])).fillna(0)
    for col in ["upgrade_gift", "additional_gift", "legacy_spouse_listed"]:
        f[col] = lookup(col).eq(True)
    f["gift_count_t"] = lookup("gift_count").fillna(0).astype("int64")
    # Cumulative features on a sparse history; an as-of join includes only years <= t.
    hist = panel.sort_values(["constituent_id", "fy"]).copy()
    hist["legacy_years"] = hist["legacy_cents"].gt(0).groupby(hist["constituent_id"]).cumsum()
    hist["history_pre1999"] = hist["fy"].lt(1999).groupby(hist["constituent_id"]).cummax()
    left = f[["constituent_id", "t"]].assign(_order=np.arange(len(f))).sort_values("t")
    joined = pd.merge_asof(left, hist[["constituent_id", "fy", "last_gift_date", "legacy_years", "history_pre1999"]].sort_values("fy"), left_on="t", right_on="fy", by="constituent_id", direction="backward").sort_values("_order")
    f["legacy_years"] = joined["legacy_years"].to_numpy(dtype="int64")
    f["history_pre1999"] = joined["history_pre1999"].to_numpy(dtype=bool)
    f["feature_cutoff"] = pd.to_datetime(f["t"].astype(str) + "-06-30")
    f["last_gift_date"] = joined["last_gift_date"].to_numpy()
    f["days_since_last_gift"] = (f["feature_cutoff"] - f["last_gift_date"]).dt.days
    f = f.merge(people, on="constituent_id", how="left", sort=False, validate="many_to_one")
    f["age_at_t"] = f.pop("age_snapshot") - (LAST_FY - f["t"])
    f["age_at_t"] = f["age_at_t"].where(f["age_at_t"].between(0, 120))
    f["spouse_on_file"] = f["spouse_id"].ne("")
    f["row_id"] = f["constituent_id"] + ":" + f["t"].astype(str)
    f["in_validation"] = f["t"].between(2021, 2024)
    assert f["row_id"].is_unique
    assert f["days_since_last_gift"].ge(0).all()
    assert f["years_last5"].between(0, 5).all()
    return f


def add_outcomes(features, panel):
    f = features.copy()
    p = panel.set_index(["constituent_id", "fy"])
    for horizon in [1, 3, 5]:
        complete = f["t"].add(horizon).le(LAST_FY)
        converted = pd.Series(False, index=f.index)
        conv_year = pd.Series(pd.NA, index=f.index, dtype="Int64")
        conv_amount = pd.Series(np.nan, index=f.index)
        for ahead in range(1, horizon + 1):
            keys = pd.MultiIndex.from_arrays([f["constituent_id"], f["t"] + ahead])
            future_major = p["major"].reindex(keys).eq(True).to_numpy()
            first_conversion = ~converted & future_major & complete
            conv_year.loc[first_conversion] = f.loc[first_conversion, "t"] + ahead
            future_amount = p["tier_basis_cents"].reindex(keys).to_numpy() / 100
            conv_amount.loc[first_conversion] = future_amount[first_conversion]
            converted |= future_major
        f[f"outcome_observed_{horizon}yr"] = complete
        f[f"converted_{horizon}yr"] = converted.astype("Int64").where(complete, pd.NA)
        f[f"conversion_fy_{horizon}yr"] = conv_year
        f[f"conversion_amount_{horizon}yr"] = conv_amount
        tiers = pd.Series(pd.NA, index=f.index, dtype="string")
        tiers.loc[conv_amount.ge(1200) & conv_amount.lt(2000)] = "1200_to_under_2000"
        tiers.loc[conv_amount.ge(2000) & conv_amount.lt(5000)] = "2000_to_under_5000"
        tiers.loc[conv_amount.ge(5000)] = "5000_plus"
        f[f"conversion_tier_{horizon}yr"] = tiers
    return f


def write_dataset(frame, name, inventory, csv_copy=False):
    folder = DEST / "data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    frame.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    roundtrip = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), roundtrip, check_dtype=True)
    del roundtrip
    inventory.append({"file": str(path.relative_to(DEST)), "rows": len(frame), "columns": len(frame.columns), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    if csv_copy:
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        inventory.append({"file": str(csv_path.relative_to(DEST)), "rows": len(frame), "columns": len(frame.columns), "bytes": csv_path.stat().st_size, "sha256": file_hash(csv_path)})


def acceptance_checks(panel, modern, parts):
    checks = []
    def check(name, actual, expected):
        checks.append({"check": name, "expected": int(expected), "actual": int(actual), "passed": bool(actual == expected)})
    check("clean_unite_rows", parts["cleaning"]["unite"]["clean_rows"], 1686654)
    check("clean_soft_rows", parts["cleaning"]["soft"]["clean_rows"], 3953)
    check("clean_legacy_rows", parts["cleaning"]["legacy"]["clean_rows"], 986729)
    check("all_person_years", len(panel), 572487)
    check("all_people", panel["constituent_id"].nunique(), 144670)
    check("modern_positive_person_years", panel["fy"].ge(2021).sum(), 199772)
    active = modern.loc[modern["segment"].eq("active") & modern["t"].le(2025)]
    expected = {2021: (30843, 55, 165), 2022: (31198, 53, 181), 2023: (33208, 37, 187), 2024: (32784, 44, 213), 2025: (32886, 81, 205)}
    for year, (rows, positives, former) in expected.items():
        subset = active.loc[active["t"].eq(year)]
        check(f"active_{year}_rows", len(subset), rows)
        check(f"active_{year}_conversions", subset["converted_1yr"].sum(), positives)
        check(f"active_{year}_former_majors", subset["former_major"].sum(), former)
    for horizon, end, rows, positives in [(1, 2025, 160919, 270), (3, 2023, 95249, 343), (5, 2021, 30843, 169)]:
        subset = active.loc[active["t"].le(end)]
        check(f"active_{horizon}yr_rows", len(subset), rows)
        check(f"active_{horizon}yr_conversions", subset[f"converted_{horizon}yr"].sum(), positives)
    lapsed = modern.loc[modern["segment"].eq("lapsed") & modern["t"].le(2025)]
    check("lapsed_1yr_rows", len(lapsed), 70668)
    check("lapsed_1yr_conversions", lapsed["converted_1yr"].sum(), 9)
    for tier, n in [("1200_to_under_2000", 158), ("2000_to_under_5000", 99), ("5000_plus", 13)]:
        check(f"one_year_tier_{tier}", active["conversion_tier_1yr"].eq(tier).sum(), n)
    combined = panel.loc[panel["major_combined_only"], ["couple_id", "fy"]].drop_duplicates()
    check("combined_only_couple_years", len(combined), 1)
    never = active.loc[~active["former_major"]]
    check("never_major_conversions", never["converted_1yr"].sum(), 132)
    orgs = pd.concat(parts["organizations"]).groupby(level=0).sum()["organization_cents"]
    soft = panel.groupby("fy")["soft_cents"].sum().reindex(orgs.index)
    check("organization_soft_reconciliation_failures", orgs.ne(soft).sum(), 0)
    return pd.DataFrame(checks)


CORE = {
    "prior_total": "Recognized individual giving during FY t, in dollars.",
    "former_major": "Qualified as major in any FY before t, including legacy and combined-only status.",
    "years_last5": "Number of fiscal years with positive giving in t-4:t.",
    "years_since_first_gift": "Inclusive t minus first observed gift FY plus one; includes pre-1999 records.",
    "legacy_years": "Count of fiscal years with attributed legacy gifts through t.",
    "giving_change": "Dollar difference between t and t-1 giving.",
    "giving_change_band": "No gift last year, gave more, gave less, or gave the same.",
    "two_years_before": "Recognized giving in t-2 plus t-1.",
    "largest_gift_share": "Largest single recognized gift in t divided by FY t total; undefined for lapsed.",
    "sustainer": "At least one recurring payment during t, reconstructed from transactions.",
    "monthly_amount": "Recurring dollars divided by recurring payment count in t; zero for nonsustainers.",
    "soft_before": "At least one positive assigned soft credit dated in a fiscal year before t.",
    "age_at_t": "Snapshot age minus (2026-t); implausible negative or >120 ages left missing.",
    "gender": "Snapshot gender; blank recorded as Unknown; assumed stable.",
}
OPTIONAL = {
    "largest_gift_t": "Largest recognized gift in FY t.",
    "days_since_last_gift": "June 30 of t minus last observed positive gift date through t.",
    "upgrade_gift": "At least one Unite Upgrade gift during t.",
    "additional_gift": "At least one Unite Additional gift during t.",
    "soft_in_t": "Any positive assigned soft credit in FY t.",
    "state": "Snapshot state assumed stable; optional sensitivity feature.",
    "spouse_on_file": "Linked spouse in snapshot; assumed stable.",
}


def save_manifest(frame, inventory, checks, audit):
    documentation = dict(FEATURE_DOCUMENTATION)
    documentation.update({k: {"role": "core_predictor", "description": v} for k, v in CORE.items()})
    documentation.update({k: {"role": "optional_predictor", "description": v} for k, v in OPTIONAL.items()})
    fields = []
    for name in frame.columns:
        role = "metadata"
        if name.startswith(("converted_", "conversion_", "outcome_observed_")):
            role = "outcome"
        if name == "constituent_type":
            role = "sensitivity_only"
        entry = documentation.get(name, {"role": role, "description": "Audit, key, cutoff, or supporting accounting field; excluded from default predictors."})
        fields.append({"column": name, "dtype": str(frame[name].dtype), **entry})
    pd.DataFrame(fields).to_csv(DEST / "feature_dictionary.csv", index=False)
    baseline = list(CORE)
    optional = [f["column"] for f in fields if f["role"] == "optional_predictor"]
    manifest = {
        "primary_target": "converted_1yr", "row_key": ["constituent_id", "t"],
        "cv_group": "couple_id", "core_predictors": baseline,
        "optional_predictors": optional, "training_only_controls": ["contacted_t", "heldout_t"],
        "group_report_predictors": baseline + ["contacted_t", "heldout_t"],
        "historical_giving_only_predictors": [name for name in baseline if name not in {"age_at_t", "gender"}],
        "excluded_from_default_predictors": [f["column"] for f in fields if f["role"] not in {"core_predictor", "optional_predictor", "training_only"}],
        "scoring_rule": "Neutralize contacted_t and heldout_t to False for every prospect scoring call, including historical scoring; preserve actuals for evaluation.",
        "preprocessing": "No imputation, scaling, encoding, balancing, or feature selection performed. Fit all learned transforms within training folds only.",
        "fiscal_year": "July-June, ending year; cutoff June 30 t",
        "status": "counts_reconciled" if checks["passed"].all() else "requires_reconciliation",
    }
    (DEST / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(inventory).to_csv(DEST / "dataset_inventory.csv", index=False)
    audit["created_utc"] = datetime.now(timezone.utc).isoformat()
    audit["all_report_count_checks_passed"] = bool(checks["passed"].all())
    (DEST / "audit" / "build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (DEST / "audit" / "build_status.json").write_text(json.dumps({
        "state": "complete" if checks["passed"].all() else "incomplete",
        "completed_utc": audit["created_utc"],
        "inventory_sha256": file_hash(DEST / "dataset_inventory.csv"),
    }, indent=2), encoding="utf-8")


def build():
    (DEST / "audit").mkdir(parents=True, exist_ok=True)
    (DEST / "data").mkdir(parents=True, exist_ok=True)
    # Invalidate readiness BEFORE any output is replaced. A failed rebuild leaves
    # this marker incomplete and the safe loader refuses mixed-generation files.
    (DEST / "audit" / "build_status.json").write_text(
        json.dumps({"state": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    people, all_ids, orgs = load_constituents()
    people_ids = set(people["constituent_id"])
    codes = set(read_strings(DATA / "campaign_codes.csv", usecols=["Marketing Code"])["Marketing Code"])
    parts = new_parts()
    for filename, kind in [("unite_payments.txt", "unite"), ("soft_credits.csv", "soft"), ("team_approach_legacy_payments.txt", "legacy")]:
        clean_and_aggregate(DATA / filename, kind, people_ids, all_ids, orgs, codes, parts)
    panel = finish_panel(parts, people)
    # Checkpoint remains in the user's requested output directory, not a temp folder.
    panel.to_parquet(DEST / "data" / "person_year_giving.parquet", index=False)
    modern = add_outcomes(giving_features(make_eligible(panel, 2021, 2026), panel, people), panel)
    checks = acceptance_checks(panel, modern, parts)
    checks.to_csv(DEST / "audit" / "report_reconciliation.csv", index=False)
    pd.DataFrame(parts["removals"]).to_parquet(DEST / "audit" / "removed_rows.parquet", index=False)
    (DEST / "audit" / "cleaning_audit.json").write_text(json.dumps(parts["cleaning"], indent=2), encoding="utf-8")
    # Fail closed: no purported modeling-ready exports if cohort rules differ.
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"]].to_dict(orient="records")
        raise ValueError(f"Report reconciliation failed. Only audit/checkpoint files written: {failed}")
    modern, engagement_audit = add_engagement_features(modern, DATA)
    assert modern.loc[modern["t"].eq(2026), ["converted_1yr", "converted_3yr", "converted_5yr"]].isna().all().all()
    assert modern.loc[~modern["viewed_t"], ["titles", "genres", "share_watched"]].isna().all().all()
    inventory = []
    write_dataset(panel, "person_year_giving", inventory)
    write_dataset(modern, "donor_year_modeling_master", inventory)
    active = modern.loc[modern["segment"].eq("active")].copy()
    selected = {
        "development_1yr_fy2021_2024": active.loc[active["t"].le(2024)],
        "funding_check_1yr_fy2025": active.loc[active["t"].eq(2025)],
        "final_training_1yr_fy2021_2025": active.loc[active["t"].le(2025)],
        "active_3yr_fy2021_2023": active.loc[active["t"].le(2023)],
        "active_5yr_fy2021": active.loc[active["t"].eq(2021)],
        "scoring_fy2026": active.loc[active["t"].eq(2026)],
        "lapsed_reporting_fy2021_2025": modern.loc[modern["segment"].eq("lapsed") & modern["t"].le(2025)],
    }
    for name, frame in selected.items():
        write_dataset(frame.reset_index(drop=True), name, inventory, csv_copy=True)
    # Longer-horizon history-only benchmarks; no modern engagement back-filling.
    history = giving_features(make_eligible(panel, 2011, 2020), panel, people)
    history = add_outcomes(history.loc[history["segment"].eq("active")].reset_index(drop=True), panel)
    write_dataset(history, "historical_active_giving_fy2011_2020", inventory)
    splits = []
    for test_year in [2022, 2023, 2024]:
        for role, mask in [("train", active["t"].lt(test_year)), ("assessment", active["t"].eq(test_year))]:
            split = active.loc[mask, ["row_id", "constituent_id", "t", "couple_id"]].copy()
            split["split_id"] = f"rolling_test_{test_year}"
            split["role"] = role
            split["assessment_fy"] = test_year
            split["training_labels_known_by_fy"] = test_year
            splits.append(split)
    for role, mask in [("train", active["t"].le(2024)), ("assessment", active["t"].eq(2025))]:
        split = active.loc[mask, ["row_id", "constituent_id", "t", "couple_id"]].copy()
        split["split_id"] = "funding_check_2025_after_selection"
        split["role"] = role
        split["assessment_fy"] = 2025
        split["training_labels_known_by_fy"] = 2025
        splits.append(split)
    splits = pd.concat(splits, ignore_index=True)
    assert splits.loc[splits["role"].eq("train"), "t"].lt(splits.loc[splits["role"].eq("train"), "assessment_fy"]).all()
    write_dataset(splits, "rolling_split_membership", inventory)
    plan = []
    for horizon in [3, 5]:
        for test_year in range(2011 + horizon, 2021):
            plan.append({"horizon_years": horizon, "train_first_fy": 2011, "train_last_fy": test_year - horizon, "assessment_fy": test_year, "latest_training_outcome_fy": test_year, "feature_set": "historical_giving_only"})
    pd.DataFrame(plan).to_csv(DEST / "historical_split_plan.csv", index=False)
    cohort_rows = []
    for name, frame in selected.items():
        horizon = 3 if "_3yr_" in name else 5 if "_5yr_" in name else 1
        cohort_rows.append({"dataset": name, "rows": len(frame), "target": f"converted_{horizon}yr", "positives": int(frame[f"converted_{horizon}yr"].sum()) if frame[f"converted_{horizon}yr"].notna().any() else None, "missing_targets": int(frame[f"converted_{horizon}yr"].isna().sum())})
    pd.DataFrame(cohort_rows).to_csv(DEST / "audit" / "dataset_checks.csv", index=False)
    pd.DataFrame({"column": modern.columns, "missing_rows": modern.isna().sum().to_numpy()}).to_csv(DEST / "audit" / "missingness.csv", index=False)
    inputs = [{"file": str((DATA / name).relative_to(ROOT)), "bytes": (DATA / name).stat().st_size, "sha256": file_hash(DATA / name)} for name in INPUTS]
    pd.DataFrame(inputs).to_csv(DEST / "audit" / "input_manifest.csv", index=False)
    audit = {"cleaning": parts["cleaning"], "engagement": engagement_audit,
             "source_report_sha256": file_hash(ROOT / "pbs_utah_group_eda.html"),
             "script_sha256": {p.name: file_hash(p) for p in (DEST / "scripts").glob("*.py")},
             "software": {"pandas": pd.__version__, "numpy": np.__version__},
             "reconstruction": "Original shared builder was unavailable. Counts are reconciled; complete feature identity is not claimed.",
             "pre1999_policy": "Retained to match report full panel and legacy/former-major history; history_pre1999 available for sensitivity.",
             "multiyear_tier_policy": "First qualifying fiscal year within a completely observed horizon; tier is descriptive, not an optimal ask recommendation.",
             "unit_policy": "Individual donor-year rows, couple_id for grouping. Combined-only major labels both people. No automatic household deduplication of prospect lists.",
             "historical_features": "Use giving-only approved list, exclude snapshot demographics and modern engagement in historical benchmarks."}
    save_manifest(modern, inventory, checks, audit)
    return pd.DataFrame(cohort_rows)


if __name__ == "__main__":
    build()
