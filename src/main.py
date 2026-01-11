from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.features import FeatureSpec, preprocess, split_by_year, X_matrix, y_log1p, optional_upper_clip
from src.xgb_quant_mod import XGBQuantileConfig, fit_quantile_model, predict_log, gain_importance
from src.baseline_mod import fit_linear, fit_ridge, predict as predict_baseline
from src.eval import eval_metrics_log_and_level, calibration_table, tail_rmse, tail_overunder
from src.scenarios import representative_base, scenario_sum_level, one_way_grid
from src.plots import parity_plot, scenario_curve, importance_barh


def ensure_dirs(root: Path):
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_end_year", type=int, default=2023)
    p.add_argument("--val_year", type=int, default=2024)
    p.add_argument("--test_year", type=int, default=2025)

    p.add_argument("--clip_q", type=float, default=None, help="Optional upper clip quantile for y_log on TRAIN only (e.g. 0.99). Default None.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.50, 0.90, 0.95])

    p.add_argument("--base_method", type=str, default="median", choices=["median", "mean"], help="Representative base for scenarios.")
    p.add_argument("--scenario", type=str, default="severity:15 duration_days:10 affected_population:20", help="Space-separated feature:delta% pairs.")

    return p.parse_args()


def parse_scenario(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    parts = [x.strip() for x in s.split() if x.strip()]
    for part in parts:
        k, v = part.split(":")
        out[k] = float(v)
    return out


def main():
    args = parse_args()
    root = Path(".")
    ensure_dirs(root)

    spec = FeatureSpec()
    FEATS = list(spec.features)
    feat_index = {f: i for i, f in enumerate(FEATS)}

    df = load_dataset()
    df = preprocess(df, spec)
    train, val, test = split_by_year(df, spec, args.train_end_year, args.val_year, args.test_year)

    X_tr = X_matrix(train, spec)
    X_va = X_matrix(val, spec)
    X_te = X_matrix(test, spec)

    y_tr_raw = y_log1p(train, spec)
    y_va = y_log1p(val, spec)
    y_te = y_log1p(test, spec)

    y_tr, clip_hi = optional_upper_clip(y_tr_raw, args.clip_q)

    sizes = {"train": int(train.shape[0]), "val": int(val.shape[0]), "test": int(test.shape[0])}
    print("Sizes:", sizes)
    if clip_hi is not None:
        print(f"TRAIN y_log clipped at quantile={args.clip_q} -> hi={clip_hi:.6f}")

    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"

    baselines = {
        "linear": fit_linear(X_tr, y_tr_raw),
        "ridge": fit_ridge(X_tr, y_tr_raw, alpha=1.0),
    }

    base_rows = []
    for name, model in baselines.items():
        yhat_log = predict_baseline(model, X_te) if X_te.size else np.array([])
        m = eval_metrics_log_and_level(y_te, yhat_log) if y_te.size else {"R2_log": np.nan, "RMSE_log": np.nan, "RMSE_level": np.nan}
        y_true_lvl = np.expm1(y_te) if y_te.size else np.array([])
        y_pred_lvl = np.expm1(yhat_log) if yhat_log.size else np.array([])
        m["tail_RMSE_95"] = tail_rmse(y_true_lvl, y_pred_lvl, top_q=0.95)
        m.update({f"tail_{k}": v for k, v in tail_overunder(y_true_lvl, y_pred_lvl, top_q=0.95).items()})
        m["model"] = name
        base_rows.append(m)
    df_base = pd.DataFrame(base_rows).set_index("model")
    df_base.to_csv(tab_dir / "baseline_metrics_test.csv")
    print("\nBaseline metrics (test):")
    print(df_base)

    cfg = XGBQuantileConfig(monotone_constraints="(1,1,1)")
    boosters = {}
    preds_test = {}

    for a in args.alphas:
        boosters[a] = fit_quantile_model(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            feature_names=FEATS,
            alpha=float(a),
            cfg=cfg,
        )
        preds_test[a] = predict_log(boosters[a], X_te, FEATS) if X_te.size else np.array([])

    rows = []
    y_true_lvl = np.expm1(y_te) if y_te.size else np.array([])
    for a, yhat_log in preds_test.items():
        mm = eval_metrics_log_and_level(y_te, yhat_log) if y_te.size else {"R2_log": np.nan, "RMSE_log": np.nan, "RMSE_level": np.nan}
        y_pred_lvl = np.expm1(yhat_log) if yhat_log.size else np.array([])
        mm["tail_RMSE_95"] = tail_rmse(y_true_lvl, y_pred_lvl, top_q=0.95)
        mm.update({f"tail_{k}": v for k, v in tail_overunder(y_true_lvl, y_pred_lvl, top_q=0.95).items()})
        mm["alpha"] = float(a)
        rows.append(mm)

        parity_plot(y_te, yhat_log, f"Parity plot (log) – Q{int(a*100)}", fig_dir, f"parity_q{int(a*100)}.png")

    df_q = pd.DataFrame(rows).set_index("alpha")
    df_q.to_csv(tab_dir / "quantile_xgb_metrics_test.csv")
    print("\nQuantile XGB metrics (test):")
    print(df_q)

    cal = calibration_table(y_te, preds_test)
    cal.to_csv(tab_dir / "quantile_calibration_test.csv", index=False)
    print("\nQuantile calibration (test):")
    print(cal)

    a_max = float(max(args.alphas))
    imp = gain_importance(boosters[a_max])
    pd.Series(imp, dtype=float).sort_values(ascending=False).to_csv(tab_dir / f"importance_gain_q{int(a_max*100)}.csv")
    importance_barh(imp, fig_dir, f"importance_gain_q{int(a_max*100)}.png", title=f"Feature importance (gain) – Q{int(a_max*100)}")

    X_base = representative_base(X_te, method=args.base_method)
    scenario = parse_scenario(args.scenario)

    def predict_qmax_log(X: np.ndarray) -> np.ndarray:
        return predict_log(boosters[a_max], X, FEATS)

    scn = scenario_sum_level(predict_qmax_log, X_base, feat_index, scenario)
    pd.Series(scn).to_csv(tab_dir / f"scenario_q{int(a_max*100)}.csv")
    print(f"\nScenario (Q{int(a_max*100)}) on representative base ({args.base_method}):", scenario)
    print(scn)

    for feat in FEATS:
        grid = one_way_grid(predict_qmax_log, X_base, feat_index, feat)
        grid.to_csv(tab_dir / f"scenario_grid_q{int(a_max*100)}_{feat}.csv", index=False)
        scenario_curve(
            grid,
            xcol="delta_%",
            ycol="sum_scn_mUSD",
            title=f"Scenario curve (Q{int(a_max*100)}) – {feat}",
            fig_dir=fig_dir,
            fname=f"scenario_curve_q{int(a_max*100)}_{feat}.png",
        )

    summary = []
    for name in df_base.index:
        r = {"model": name, **df_base.loc[name].to_dict()}
        summary.append(r)
    for a in df_q.index:
        r = {"model": f"xgb_q{int(float(a)*100)}", **df_q.loc[a].to_dict()}
        summary.append(r)

    df_summary = pd.DataFrame(summary).set_index("model")
    df_summary.to_csv(tab_dir / "summary_metrics_test.csv")
    print("\nSummary metrics (test):")
    print(df_summary)


if __name__ == "__main__":
    main()
