from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    features: Tuple[str, ...] = ("severity", "duration_days", "affected_population")
    target: str = "economic_impact_million_usd"
    year_col: str = "year"


def to_numeric_clean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols).copy()
    return out


def preprocess(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    cols = list(spec.features) + [spec.target, spec.year_col]
    df = to_numeric_clean(df, cols)
    df = df[df[spec.target] >= 0].copy()

    df["duration_days"] = np.log1p(df["duration_days"].astype(float))
    df["affected_population"] = np.log1p(df["affected_population"].astype(float))
    return df


def split_by_year(
    df: pd.DataFrame,
    spec: FeatureSpec,
    train_end_year: int,
    val_year: int,
    test_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = spec.year_col
    train = df[df[y] <= train_end_year].copy()
    val = df[df[y] == val_year].copy()
    test = df[df[y] == test_year].copy()
    return train, val, test


def X_matrix(d: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    return d[list(spec.features)].astype(float).to_numpy()


def y_log1p(d: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    return np.log1p(d[spec.target].astype(float).to_numpy())


def optional_upper_clip(y: np.ndarray, q: float | None) -> tuple[np.ndarray, float | None]:
    if y.size == 0 or q is None:
        return y, None
    hi = float(np.quantile(y, q))
    return np.clip(y, None, hi), hi
