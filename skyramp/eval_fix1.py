"""EVAL GATE fix attempt 1 (claim B only). Appends to eval_results.json; never edits earlier entries.

DIAGNOSIS (written after first_run): the first run's constant-width interval covered only
13.95% of ramp minutes (non-ramp 88.9%). Split conformal guarantees MARGINAL coverage only;
absolute residuals are heteroscedastic (ramp-minute RMSE ~6 kW vs ~1.1 kW otherwise) and a
single quantile cannot adapt. The scale used in the 'normalized' variant (trailing 10-min PV
std) is a weak proxy for ramp risk (ramp coverage only 0.40).
CHANGE (one): replace the fixed width with a LEARNED local scale (locally-weighted split
conformal, Lei et al. 2018): LightGBM predicts |residual| from PV-history + sky-image features,
fitted on half A of the calibration days; the conformal quantile of |res|/scale is taken on
half B; test is untouched. Thresholds are unchanged (config.py).
"""
import json, datetime
import numpy as np, pandas as pd, lightgbm as lgb
from . import config as C, data, models

def main():
    X, img = data.make_dataset()
    P = pd.read_csv(data.ROOT / "results" / "predictions.csv.gz", parse_dates=["ts"], index_col="ts")
    X = X.loc[P.index]
    m = json.loads((data.ROOT / "eval_results.json").read_text())[0]["chosen_on_calib"]["img_plus_pv"]
    S = np.hstack([data.tab_features(X).values, models.img_summary(data.frames(X, img))])
    res = (P[m] - P["delta"]).values
    role = P["role"].values
    ca = role == "calib"; te = role == "test"
    cal_days = np.sort(P.loc[ca, "date"].unique())
    rng = np.random.default_rng(C.SEED)
    half_a = set(rng.choice(cal_days, len(cal_days) // 2, replace=False))
    in_a = ca & P["date"].isin(half_a).values
    in_b = ca & ~in_a
    sm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=C.SEED, verbose=-1).fit(S[in_a], np.abs(res[in_a]))
    scale = lambda idx: np.maximum(sm.predict(S[idx]), 0.3)
    n = in_b.sum()
    q = np.quantile(np.abs(res[in_b]) / scale(in_b), min(1.0, np.ceil((n + 1) * (1 - C.ALPHA)) / n))
    half = q * scale(te)
    cover = np.abs(res[te]) <= half
    ramp = P.loc[te, "ramp"].values
    v = {"coverage_all": float(cover.mean()), "coverage_ramp": float(cover[ramp].mean()),
         "coverage_nonramp": float(cover[~ramp].mean()), "mean_width_kw": float(2 * half.mean()),
         "mean_width_ramp_kw": float(2 * half[ramp].mean()), "n_calib_halfB": int(n)}
    b_pass = bool(C.CLAIM_B_COVERAGE_BAND[0] <= v["coverage_all"] <= C.CLAIM_B_COVERAGE_BAND[1]
                  and v["coverage_ramp"] >= C.CLAIM_B_RAMP_MIN_COVERAGE)
    out = {"tag": "fix1_learned_scale", "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "claim_B": {"variant": "img_plus_pv/learned_scale", **v,
                       "band": list(C.CLAIM_B_COVERAGE_BAND), "ramp_min": C.CLAIM_B_RAMP_MIN_COVERAGE,
                       "pass": b_pass}}
    path = data.ROOT / "eval_results.json"
    hist = json.loads(path.read_text()); hist.append(out)
    path.write_text(json.dumps(hist, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
