"""Load explicitly approved X/y/groups without outcome or identifier leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


MODELING = Path(__file__).resolve().parents[1]


def load_inputs(
    dataset="development_1yr_fy2021_2024",
    *,
    horizon=1,
    purpose="fit",
    include_optional=False,
    include_treatment_controls=True,
):
    """Return (X, y, couple_groups, metadata); never fit preprocessing or a model.

    purpose='predict' turns both treatment controls off for historical AND
    FY2026 scoring. Actual treatment flags stay untouched in metadata. Observed
    y is returned for assessment; unknown scoring outcomes remain nullable.
    Historical datasets automatically use their giving-only feature allowlist.
    """
    if purpose not in {"fit", "predict"} or horizon not in {1, 3, 5}:
        raise ValueError("purpose must be fit/predict; horizon must be 1, 3 or 5")
    if Path(dataset).name != dataset:
        raise ValueError("Pass a dataset stem from the inventory, not a path")
    status = json.loads((MODELING / "audit" / "build_status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete":
        raise RuntimeError("The most recent build is incomplete; do not use its files")
    inventory_path = MODELING / "dataset_inventory.csv"
    if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != status["inventory_sha256"]:
        raise RuntimeError("Dataset inventory changed after build completion")
    inventory = pd.read_csv(inventory_path)
    path = MODELING / "data" / f"{dataset}.parquet"
    relative = str(path.relative_to(MODELING)).replace("\\", "/")
    record = inventory.loc[inventory["file"].str.replace("\\", "/", regex=False).eq(relative)]
    if len(record) != 1 or hashlib.sha256(path.read_bytes()).hexdigest() != record.iloc[0]["sha256"]:
        raise RuntimeError("Dataset is missing from the verified inventory or has changed")
    manifest = json.loads((MODELING / "feature_manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path)
    historical = dataset.startswith("historical_")
    names = list(manifest["historical_giving_only_predictors"] if historical else manifest["core_predictors"])
    if include_optional:
        if historical:
            raise ValueError("Historical benchmarks use giving-only inputs, not modern optional features")
        names += manifest["optional_predictors"]
    controls = manifest["training_only_controls"] if include_treatment_controls and not historical else []
    names += controls
    X = frame.loc[:, names].copy()
    if purpose == "predict":
        for column in controls:
            X[column] = False
    y = frame[f"converted_{horizon}yr"].copy()
    if purpose == "fit" and y.isna().any():
        raise ValueError("Training includes incomplete outcomes; use a fully observed horizon-specific dataset")
    if frame.duplicated(["constituent_id", "t"]).any():
        raise ValueError("Donor-year rows must be unique")
    metadata = frame.drop(columns=names).copy()
    return X, y, frame["couple_id"].copy(), metadata
