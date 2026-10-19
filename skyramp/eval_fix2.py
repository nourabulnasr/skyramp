"""EVAL GATE fix attempt 2 (claim B only). Appends to eval_results.json.

DIAGNOSIS (after fix1: ramp coverage rose 0.14 -> 0.53 but overall fell to 0.830 and non-ramp
to 0.840): fix1 conformalised on only half of the calibration days (~7 days). Minute residuals
within a day are strongly autocorrelated, so the effective calibration sample is ~7 weather
days, not 5.6k points; the quantile is a high-variance estimate and the interval was cut too
narrow for the test days.
CHANGE (one): use ALL calibration days for the conformal quantile by getting out-of-fold
local-scale predictions (5 folds split BY DAY over the calibration block); the scale model used
on test is fitted on all calibration days. Thresholds unchanged.
"""
import json, datetime
import numpy as np, pandas as pd, lightgbm as lgb
from . import config as C, data, models

def fit(S, y):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=C.SEED, verbose=-1).fit(S, y)

def main():
    X, img = data.make_dataset()
    P = pd.read_csv(data.ROOT / "results" / "predictions.csv.gz", parse_dates=["ts"], index_col="ts")
    X = X.loc[P.index]
    m = json.loads((data.ROOT / "eval_results.json").read_text())[0]["chosen_on_calib"]["img_plus_pv"]
    S = np.hstack([data.tab_features(X).values, models.img_summary(data.frames(X, img))])
    res = np.abs((P[m] - P["delta"]).values)
    ca = (P["role"] == "calib").values; te = (P["role"] == "test").values
    days = P["date"].values
    cal_days = np.unique(days[ca])
    rng = np.random.default_rng(C.SEED)
    fold_of = dict(zip(cal_days, rng.permutation(len(cal_days)) % 5))
    fold = np.array([fold_of.get(d, -1) for d in days])
    oof = np.zeros(len(P))
    for k in range(5):
        tr, va = ca & (fold != k), ca & (fold == k)
        oof[va] = fit(S[tr], res[tr]).predict(S[va])
    full = fit(S[ca], res[ca])
    sc_c = np.maximum(oof[ca], 0.3); sc_t = np.maximum(full.predict(S[te]), 0.3)
    n = ca.sum()
    q = np.quantile(res[ca] / sc_c, min(1.0, np.ceil((n + 1) * (1 - C.ALPHA)) / n))
    half = q * sc_t
    cover = res[te] <= half
    ramp = P.loc[te, "ramp"].values
    v = {"coverage_all": float(cover.mean()), "coverage_ramp": float(cover[ramp].mean()),
         "coverage_nonramp": float(cover[~ramp].mean()), "mean_width_kw": float(2 * half.mean()),
         "mean_width_ramp_kw": float(2 * half[ramp].mean()), "n_calib": int(n), "n_calib_days": int(len(cal_days))}
    b_pass = bool(C.CLAIM_B_COVERAGE_BAND[0] <= v["coverage_all"] <= C.CLAIM_B_COVERAGE_BAND[1]
                  and v["coverage_ramp"] >= C.CLAIM_B_RAMP_MIN_COVERAGE)
    out = {"tag": "fix2_oof_scale_all_calib_days", "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "claim_B": {"variant": "img_plus_pv/learned_scale_oof", **v,
                       "band": list(C.CLAIM_B_COVERAGE_BAND), "ramp_min": C.CLAIM_B_RAMP_MIN_COVERAGE,
                       "pass": b_pass}}
    path = data.ROOT / "eval_results.json"
    hist = json.loads(path.read_text()); hist.append(out)
    path.write_text(json.dumps(hist, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
