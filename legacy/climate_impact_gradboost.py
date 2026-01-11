import os, numpy as np, pandas as pd
import kagglehub, xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("uom190346a/global-climate-events-and-economic-impact-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files: raise FileNotFoundError("Ingen CSV-filer i datasettmappa.")
df = pd.read_csv(os.path.join(path, csv_files[0]))

cols = ["severity","duration_days","affected_population","economic_impact_million_usd","year"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols).copy()

df = df[df["economic_impact_million_usd"] >= 0].copy()

df["duration_days"] = np.log1p(df["duration_days"].astype(float))
df["affected_population"] = np.log1p(df["affected_population"].astype(float))

train = df[df["year"] <= 2023].copy()
val   = df[df["year"] == 2024].copy()
test  = df[df["year"] == 2025].copy()

FEATS = ["severity","duration_days","affected_population"]

def X_only(d):
    return d[FEATS].astype(float).values

def y_log(d):
    return np.log1p(d["economic_impact_million_usd"].astype(float).values)

X_tr = X_only(train)
X_va = X_only(val)
X_te = X_only(test)

y_tr_raw = y_log(train)
hi_clip = np.quantile(y_tr_raw, 0.99) if y_tr_raw.size else 0.0
y_tr = np.clip(y_tr_raw, None, hi_clip)
y_va = y_log(val)
y_te = y_log(test)

print({k: v.shape[0] for k,v in {"train":train,"val":val,"test":test}.items()})

dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATS)
dva = xgb.DMatrix(X_va, label=y_va, feature_names=FEATS)
dte = xgb.DMatrix(X_te, label=y_te, feature_names=FEATS)

def make_smooth_pinball(alpha, eps=0.2, h_floor=1e-2):
    def obj(yhat, dtrain):
        y = dtrain.get_label()
        r = y - yhat
        grad = np.where(r >  eps, -alpha,
               np.where(r < -eps, 1.0 - alpha, -r/(2*eps)))
        hess = np.where(np.abs(r) <= eps, 1.0/(2*eps), h_floor)
        return grad, hess
    return obj

def pinball_feval(alpha):
    def feval(yhat, d):
        y = d.get_label()
        r = y - yhat
        loss = np.where(r >= 0.0, alpha*r, (alpha-1.0)*r)
        return "pinball", float(np.mean(loss))
    return feval

params = {
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 3.0,
    "monotone_constraints": "(1,1,1)",
    "tree_method": "hist",
    "verbosity": 0,
    "base_score": float(np.median(y_tr)) if len(y_tr) else 0.0,
    "disable_default_eval_metric": 1,
    "max_delta_step": 0.5,
    "max_depth": 4,
    "min_child_weight": 0.1,
    "grow_policy": "lossguide",
    "max_leaves": 24
}

def fit_quantile(alpha, rounds=2000, early_stopping_rounds=200):
    obj = make_smooth_pinball(alpha)
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=rounds,
        evals=[(dtr,"train"),(dva,"val")],
        obj=obj,
        custom_metric=pinball_feval(alpha),
        early_stopping_rounds=early_stopping_rounds,
        maximize=False,
        verbose_eval=False
    )
    print(f"alpha={alpha}: best_iteration=", getattr(booster, "best_iteration", None))
    return booster

m_q50 = fit_quantile(0.50)
m_q90 = fit_quantile(0.90)
m_q95 = fit_quantile(0.95)

def eval_level_metrics(booster, X, y_log_arr, label="set"):
    if X.size == 0:
        return {"label":label, "R2_log": np.nan, "RMSE_log": np.nan, "RMSE_level": np.nan}
    yhat_log = booster.predict(xgb.DMatrix(X, feature_names=FEATS))
    r2log = r2_score(y_log_arr, yhat_log) if len(y_log_arr) > 1 else np.nan
    rmse_log = float(np.sqrt(mean_squared_error(y_log_arr, yhat_log)))
    yhat = np.expm1(yhat_log); ylvl = np.expm1(y_log_arr)
    rmse_lvl = float(np.sqrt(mean_squared_error(ylvl, yhat)))
    return {"label":label, "R2_log":r2log, "RMSE_log":rmse_log, "RMSE_level":rmse_lvl}

print("Quantile XGBoost (test):")
print("Q50:", eval_level_metrics(m_q50, X_te, y_te, "test"))
print("Q90:", eval_level_metrics(m_q90, X_te, y_te, "test"))
print("Q95:", eval_level_metrics(m_q95, X_te, y_te, "test"))

def scen_sum(booster, X_base, changes_pct: dict):
    if X_base.size == 0:
        return {"sum_base_mUSD": 0.0, "sum_scn_mUSD": 0.0, "abs_change_mUSD": 0.0, "rel_change_%": 0.0}
    d_base = xgb.DMatrix(X_base, feature_names=FEATS)
    y0 = np.expm1(booster.predict(d_base))
    Xs = X_base.copy()
    for k,v in changes_pct.items():
        j = FEATS.index(k)
        Xs[:, j] = Xs[:, j] * (1.0 + v/100.0)
    d_scn = xgb.DMatrix(Xs, feature_names=FEATS)
    y1 = np.expm1(booster.predict(d_scn))
    sum0, sum1 = float(y0.sum()), float(y1.sum())
    rel = 100.0*(sum1 - sum0)/max(sum0, 1e-9)
    return {"sum_base_mUSD": sum0, "sum_scn_mUSD": sum1, "abs_change_mUSD": sum1 - sum0, "rel_change_%": rel}

def one_way_grid(booster, X_base, feat, deltas=(-20,-10,-5,0,5,10,20,30,50)):
    rows=[]
    for d in deltas:
        rows.append({"feature": feat, "delta_%": d, **scen_sum(booster, X_base, {feat:d})})
    return pd.DataFrame(rows)

print("Worst-case (Q95) scenario +15% severity, +10% duration, +20% population:")
print(scen_sum(m_q95, X_te, {"severity":15, "duration_days":10, "affected_population":20}))

print("Q95 severity grid:");   print(one_way_grid(m_q95, X_te, "severity"))
print("Q95 duration grid:");   print(one_way_grid(m_q95, X_te, "duration_days"))
print("Q95 population grid:"); print(one_way_grid(m_q95, X_te, "affected_population"))

def gain_importance(booster):
    imp = booster.get_score(importance_type="gain")
    if not imp: return pd.Series(dtype=float)
    s = pd.Series(imp, dtype=float)
    return s.sort_values(ascending=False)

print("Feature importance (gain) Q95:")
print(gain_importance(m_q95))

def quantile_coverage(booster, X, y_log, alpha):
    yhat = booster.predict(xgb.DMatrix(X, feature_names=FEATS))
    return float(np.mean(y_log <= yhat))

print("Coverage Q90:", quantile_coverage(m_q90, X_te, y_te, 0.90))
print("Coverage Q95:", quantile_coverage(m_q95, X_te, y_te, 0.95))


# --- PLOTS ---
yhat50_log = m_q50.predict(xgb.DMatrix(X_te, feature_names=FEATS))
yhat90_log = m_q90.predict(xgb.DMatrix(X_te, feature_names=FEATS))
yhat95_log = m_q95.predict(xgb.DMatrix(X_te, feature_names=FEATS))

plt.figure()
plt.scatter(y_te, yhat50_log, s=8)
mn, mx = float(np.min(y_te)), float(np.max(y_te))
plt.plot([mn,mx],[mn,mx])
plt.xlabel("True log impact")
plt.ylabel("Predicted log impact (Q50)")
plt.title("Parity plot (log) – Q50")
plt.show()

plt.figure()
plt.scatter(y_te, yhat90_log, s=8)
mn, mx = float(np.min(y_te)), float(np.max(y_te))
plt.plot([mn,mx],[mn,mx])
plt.xlabel("True log impact")
plt.ylabel("Predicted log impact (Q90)")
plt.title("Parity plot (log) – Q90")
plt.show()

plt.figure()
plt.scatter(y_te, yhat95_log, s=8)
mn, mx = float(np.min(y_te)), float(np.max(y_te))
plt.plot([mn,mx],[mn,mx])
plt.xlabel("True log impact")
plt.ylabel("Predicted log impact (Q95)")
plt.title("Parity plot (log) – Q95")
plt.show()

grid_sev = one_way_grid(m_q95, X_te, "severity")
grid_dur = one_way_grid(m_q95, X_te, "duration_days")
grid_pop = one_way_grid(m_q95, X_te, "affected_population")

plt.figure()
plt.plot(grid_sev["delta_%"], grid_sev["sum_scn_mUSD"], marker="o")
plt.xlabel("Severity delta (%)")
plt.ylabel("Sum predicted Q95 impact (USD mio)")
plt.title("Scenario curve (Q95) – Severity")
plt.show()

plt.figure()
plt.plot(grid_dur["delta_%"], grid_dur["sum_scn_mUSD"], marker="o")
plt.xlabel("Duration delta (%)")
plt.ylabel("Sum predicted Q95 impact (USD mio)")
plt.title("Scenario curve (Q95) – Duration")
plt.show()

plt.figure()
plt.plot(grid_pop["delta_%"], grid_pop["sum_scn_mUSD"], marker="o")
plt.xlabel("Affected population delta (%)")
plt.ylabel("Sum predicted Q95 impact (USD mio)")
plt.title("Scenario curve (Q95) – Affected population")
plt.show()

imp_q95 = gain_importance(m_q95)
if not imp_q95.empty:
    plt.figure()
    imp_q95[::-1].plot(kind="barh")
    plt.xlabel("Gain")
    plt.title("Feature importance (gain) – Q95")
    plt.tight_layout()
    plt.show()
