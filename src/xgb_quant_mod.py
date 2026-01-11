from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import xgboost as xgb


def make_smooth_pinball(alpha: float, eps: float = 0.2, h_floor: float = 1e-2):
    def obj(yhat: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        r = y - yhat
        grad = np.where(
            r > eps,
            -alpha,
            np.where(r < -eps, 1.0 - alpha, -r / (2.0 * eps)),
        )
        hess = np.where(np.abs(r) <= eps, 1.0 / (2.0 * eps), h_floor)
        return grad, hess

    return obj


def pinball_feval(alpha: float):
    def feval(yhat: np.ndarray, d: xgb.DMatrix):
        y = d.get_label()
        r = y - yhat
        loss = np.where(r >= 0.0, alpha * r, (alpha - 1.0) * r)
        return "pinball", float(np.mean(loss))

    return feval


@dataclass(frozen=True)
class XGBQuantileConfig:
    eta: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 3.0
    monotone_constraints: str = "(1,1,1)"
    tree_method: str = "hist"
    max_depth: int = 4
    min_child_weight: float = 0.1
    grow_policy: str = "lossguide"
    max_leaves: int = 24
    max_delta_step: float = 0.5
    verbosity: int = 0


def fit_quantile_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    feature_names: list[str],
    alpha: float,
    cfg: XGBQuantileConfig,
    rounds: int = 2000,
    early_stopping_rounds: int = 200,
) -> xgb.Booster:
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)

    params: Dict[str, object] = {
        "eta": cfg.eta,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "lambda": cfg.reg_lambda,
        "monotone_constraints": cfg.monotone_constraints,
        "tree_method": cfg.tree_method,
        "verbosity": cfg.verbosity,
        "disable_default_eval_metric": 1,
        "max_delta_step": cfg.max_delta_step,
        "max_depth": cfg.max_depth,
        "min_child_weight": cfg.min_child_weight,
        "grow_policy": cfg.grow_policy,
        "max_leaves": cfg.max_leaves,
        "base_score": float(np.median(y_tr)) if y_tr.size else 0.0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=rounds,
        evals=[(dtr, "train"), (dva, "val")],
        obj=make_smooth_pinball(alpha),
        custom_metric=pinball_feval(alpha),
        early_stopping_rounds=early_stopping_rounds,
        maximize=False,
        verbose_eval=False,
    )
    return booster


def predict_log(booster: xgb.Booster, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    dm = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(dm)


def gain_importance(booster: xgb.Booster) -> dict[str, float]:
    return booster.get_score(importance_type="gain")
