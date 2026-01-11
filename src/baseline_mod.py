from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear(X: np.ndarray, y_log: np.ndarray):
    model = LinearRegression()
    model.fit(X, y_log)
    return model


def fit_ridge(X: np.ndarray, y_log: np.ndarray, alpha: float = 1.0):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X, y_log)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
