from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def savefig(fig_dir: Path, name: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=200)
    plt.close()


def parity_plot(y_true_log: np.ndarray, y_pred_log: np.ndarray, title: str, fig_dir: Path, fname: str):
    if y_true_log.size == 0:
        return
    plt.figure()
    plt.scatter(y_true_log, y_pred_log, s=8)
    mn, mx = float(np.min(y_true_log)), float(np.max(y_true_log))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True log impact")
    plt.ylabel("Predicted log impact")
    plt.title(title)
    savefig(fig_dir, fname)


def scenario_curve(grid: pd.DataFrame, xcol: str, ycol: str, title: str, fig_dir: Path, fname: str):
    if grid.empty:
        return
    plt.figure()
    plt.plot(grid[xcol], grid[ycol], marker="o")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    savefig(fig_dir, fname)


def importance_barh(imp_gain: dict[str, float], fig_dir: Path, fname: str, title: str = "Feature importance (gain)"):
    if not imp_gain:
        return
    s = pd.Series(imp_gain, dtype=float).sort_values()
    plt.figure()
    s.plot(kind="barh")
    plt.xlabel("Gain")
    plt.title(title)
    savefig(fig_dir, fname)
