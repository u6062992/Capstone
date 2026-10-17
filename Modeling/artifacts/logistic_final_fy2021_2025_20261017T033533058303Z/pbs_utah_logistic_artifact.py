"""Portable preprocessing and control-safe inference for the selected model.

Training uses actual contacted_t/heldout_t values. Public prediction methods
always set both controls to False without changing the caller's DataFrame.
Only deserialize artifacts from a trusted source.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


CONTROL_COLUMNS = ("contacted_t", "heldout_t")


def numeric_types(frame):
    """Same deterministic numeric conversion as the evaluated pipeline."""
    return frame.astype(float)


def category_types(frame):
    """Same deterministic category conversion as the evaluated pipeline."""
    return frame.astype(object).where(frame.notna(), np.nan)


class ControlSafePipeline(Pipeline):
    """A complete sklearn pipeline with mandatory scoring-time neutralization.

The inherited fit method is unchanged: it fits all preprocessing and the
estimator using actual training controls. No recalibration is performed.
"""

    def _scoring_frame(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provide an approved-feature pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("Scoring column names must be unique.")
        expected = list(self.feature_names_in_)
        missing = sorted(set(expected) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected))
        if missing or extra:
            raise ValueError(f"Feature mismatch: missing={missing}, extra={extra}")
        frame = X.loc[:, expected].copy()
        for column in CONTROL_COLUMNS:
            frame[column] = False
        return frame

    def predict_proba(self, X, **params):
        return super().predict_proba(self._scoring_frame(X), **params)

    def predict_log_proba(self, X, **params):
        return super().predict_log_proba(self._scoring_frame(X), **params)

    def decision_function(self, X, **params):
        return super().decision_function(self._scoring_frame(X), **params)

    def predict(self, X, **params):
        return super().predict(self._scoring_frame(X), **params)

    def score(self, X, y=None, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def load_pipeline(bundle_directory=None):
    """Verify the model checksum and load this trusted, version-pinned bundle.

Keep this module importable when loading. A checksum detects corruption; it
does not make pickle/joblib files safe to load from untrusted sources.
"""
    directory = Path(bundle_directory) if bundle_directory else Path(__file__).parent
    model_path = directory / "model.joblib"
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != checksums["model.joblib"]:
        raise ValueError("Model checksum differs from this bundle's manifest.")
    model = joblib.load(model_path)
    if not isinstance(model, ControlSafePipeline):
        raise TypeError("Artifact is not the expected control-safe pipeline.")
    return model
