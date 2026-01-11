from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def eval_metrics_log_and_level(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    r2 = r2_score(y_true_log, y_pred_log) if y_true_log.size > 1 else float("nan")
    out = {
        "R2_log": float(r2),
        "RMSE_log": rmse(y_true_log, y_pred_log),
        "RMSE_level": rmse(np.expm1(y_true_log), np.expm1(y_pred_log)),
    }
    return out


def empirical_coverage(y_true_log: np.ndarray, y_pred_q_log: np.ndarray) -> float:
    return float(np.mean(y_true_log <= y_pred_q_log)) if y_true_log.size else float("nan")


def tail_rmse(
    y_true_level: np.ndarray,
    y_pred_level: np.ndarray,
    top_q: float = 0.95,
) -> float:
    if y_true_level.size == 0:
        return float("nan")
    thr = float(np.quantile(y_true_level, top_q))
    m = y_true_level >= thr
    if not np.any(m):
        return float("nan")
    return rmse(y_true_level[m], y_pred_level[m])


def tail_overunder(
    y_true_level: np.ndarray,
    y_pred_level: np.ndarray,
    top_q: float = 0.95,
) -> dict[str, float]:
    if y_true_level.size == 0:
        return {"mean_error_tail": float("nan"), "frac_under_tail": float("nan")}
    thr = float(np.quantile(y_true_level, top_q))
    m = y_true_level >= thr
    if not np.any(m):
        return {"mean_error_tail": float("nan"), "frac_under_tail": float("nan")}
    err = y_pred_level[m] - y_true_level[m]
    return {
        "mean_error_tail": float(np.mean(err)),
        "frac_under_tail": float(np.mean(y_pred_level[m] < y_true_level[m])),
    }


def calibration_table(
    y_true_log: np.ndarray,
    preds: dict[float, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for a, qhat in sorted(preds.items()):
        rows.append({"alpha": a, "empirical_coverage": empirical_coverage(y_true_log, qhat), "gap": empirical_coverage(y_true_log, qhat) - a})
    return pd.DataFrame(rows)
