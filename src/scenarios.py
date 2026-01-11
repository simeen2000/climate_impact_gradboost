from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def apply_pct_changes(X: np.ndarray, feat_index: dict[str, int], changes_pct: Dict[str, float]) -> np.ndarray:
    Xs = X.copy()
    for k, v in changes_pct.items():
        j = feat_index[k]
        Xs[:, j] = Xs[:, j] * (1.0 + v / 100.0)
    return Xs


def scenario_sum_level(predict_log_fn, X_base: np.ndarray, feat_index: dict[str, int], changes_pct: Dict[str, float]) -> dict[str, float]:
    if X_base.size == 0:
        return {"sum_base_mUSD": 0.0, "sum_scn_mUSD": 0.0, "abs_change_mUSD": 0.0, "rel_change_%": 0.0}

    y0 = np.expm1(predict_log_fn(X_base))
    Xs = apply_pct_changes(X_base, feat_index, changes_pct)
    y1 = np.expm1(predict_log_fn(Xs))

    sum0 = float(y0.sum())
    sum1 = float(y1.sum())
    rel = 100.0 * (sum1 - sum0) / max(sum0, 1e-12)

    return {"sum_base_mUSD": sum0, "sum_scn_mUSD": sum1, "abs_change_mUSD": sum1 - sum0, "rel_change_%": rel}


def one_way_grid(
    predict_log_fn,
    X_base: np.ndarray,
    feat_index: dict[str, int],
    feat: str,
    deltas: Iterable[float] = (-20, -10, -5, 0, 5, 10, 20, 30, 50),
) -> pd.DataFrame:
    rows = []
    for d in deltas:
        rows.append({"feature": feat, "delta_%": float(d), **scenario_sum_level(predict_log_fn, X_base, feat_index, {feat: float(d)})})
    return pd.DataFrame(rows)


def representative_base(X: np.ndarray, method: str = "median") -> np.ndarray:
    if X.size == 0:
        return X
    if method == "median":
        x0 = np.median(X, axis=0)
    elif method == "mean":
        x0 = np.mean(X, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return np.repeat(x0.reshape(1, -1), X.shape[0], axis=0)
