"""Eval gate. Reads results/predictions.csv.gz (test+calib blocks), tests the README claims.

CLAIM_UNDER_TEST (A): adding sky-image information to PV history reduces 15-min-ahead
  RMSE on RAMP minutes (|dPV over 15 min| >= RAMP_KW) by >= 5% versus the best PV-only
  model, with a day-block bootstrap 95% CI whose lower bound is > 0.
CLAIM_UNDER_TEST (B): split-conformal 90% intervals reach empirical coverage inside
  [0.87, 0.93] on the held-out test blocks AND keep >= 0.80 coverage on ramp minutes.
Model choice within each family is made on the CALIB block only, never on test.
Thresholds come from config.py (written before the first run). Results are APPENDED
to eval_results.json; the first entry is ground truth and is never rewritten.
"""
import json, datetime, sys
import numpy as np, pandas as pd
from . import config as C, data

PV_ONLY = ["smart_persistence", "lgbm_pv", "nn_pv"]
IMG_PV = ["cnn_img_pv", "lgbm_pv_img"]
ALL = ["persistence", "smart_persistence", "lgbm_pv", "nn_pv", "lgbm_pv_img", "cnn_img_pv", "cnn_img_only"]

rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))

def split_cov(res_cal, res_te, alpha):
    n = len(res_cal)
    q = np.quantile(res_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return q

def run(tag="first_run"):
    P = pd.read_csv(data.ROOT / "results" / "predictions.csv.gz", parse_dates=["date"])
    ca, te = P[P.role == "calib"], P[P.role == "test"]
    err = lambda d, m: d[m] - d["delta"]               # delta-space error == PV-space error
    out = {"tag": tag, "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "n_test": int(len(te)), "n_test_ramp": int(te.ramp.sum()),
           "n_calib": int(len(ca)), "n_calib_ramp": int(ca.ramp.sum()),
           "ramp_kw": C.RAMP_KW, "horizon_min": C.HORIZON_MIN}

    rows = {}
    for m in ALL:
        e = err(te, m)
        rows[m] = {"rmse_all": rmse(e), "rmse_ramp": rmse(e[te.ramp]), "rmse_nonramp": rmse(e[~te.ramp]),
                   "rmse_calib": rmse(err(ca, m))}
    out["models"] = rows

    best_pv = min(PV_ONLY, key=lambda m: rows[m]["rmse_calib"])
    best_img = min(IMG_PV, key=lambda m: rows[m]["rmse_calib"])
    out["chosen_on_calib"] = {"pv_only": best_pv, "img_plus_pv": best_img}

    # Claim A: day-block bootstrap on ramp minutes
    rng = np.random.default_rng(C.SEED)
    dsub = te[te.ramp]
    ea, eb = err(dsub, best_pv).values, err(dsub, best_img).values
    days = dsub["date"].values
    ud = np.unique(days)
    idx_by_day = [np.where(days == d)[0] for d in ud]
    def gain(sel):
        return 1 - rmse(eb[sel]) / rmse(ea[sel])
    g = gain(np.arange(len(ea)))
    boots = []
    for _ in range(C.N_BOOT):
        pick = rng.integers(0, len(ud), len(ud))
        boots.append(gain(np.concatenate([idx_by_day[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    a_pass = bool(g >= C.CLAIM_A_MIN_REL_RMSE_GAIN and lo > 0)
    # Same comparison on non-ramp minutes and overall (context, not part of the claim)
    ctx = {}
    for name, mask in (("all", np.ones(len(te), bool)), ("nonramp", ~te.ramp.values)):
        a_, b_ = err(te[mask], best_pv).values, err(te[mask], best_img).values
        ctx[name] = 1 - rmse(b_) / rmse(a_)
    out["claim_A"] = {"best_pv_only": best_pv, "best_img_plus_pv": best_img,
                      "ramp_rmse_pv_only": rmse(ea), "ramp_rmse_img_plus_pv": rmse(eb),
                      "rel_gain": float(g), "ci95": [float(lo), float(hi)],
                      "rel_gain_nonramp": ctx["nonramp"], "rel_gain_all": ctx["all"],
                      "threshold": C.CLAIM_A_MIN_REL_RMSE_GAIN, "pass": a_pass}

    # Claim B: conformal on the chosen image+PV model (+ PV-only for contrast)
    conf = {}
    for label, m in (("img_plus_pv", best_img), ("pv_only", best_pv)):
        for kind in ("plain", "normalized"):
            sc_c = np.ones(len(ca)) if kind == "plain" else (ca.pv_std10.fillna(0).values + 0.5)
            sc_t = np.ones(len(te)) if kind == "plain" else (te.pv_std10.fillna(0).values + 0.5)
            q = split_cov(np.abs(err(ca, m).values) / sc_c, None, C.ALPHA)
            half = q * sc_t
            cover = np.abs(err(te, m).values) <= half
            conf[f"{label}/{kind}"] = {"coverage_all": float(cover.mean()),
                                       "coverage_ramp": float(cover[te.ramp.values].mean()),
                                       "coverage_nonramp": float(cover[~te.ramp.values].mean()),
                                       "mean_width_kw": float(2 * half.mean()),
                                       "mean_width_ramp_kw": float(2 * half[te.ramp.values].mean())}
    main = conf["img_plus_pv/plain"]
    b_pass = bool(C.CLAIM_B_COVERAGE_BAND[0] <= main["coverage_all"] <= C.CLAIM_B_COVERAGE_BAND[1]
                  and main["coverage_ramp"] >= C.CLAIM_B_RAMP_MIN_COVERAGE)
    out["claim_B"] = {"variants": conf, "tested_variant": "img_plus_pv/plain",
                      "band": list(C.CLAIM_B_COVERAGE_BAND), "ramp_min": C.CLAIM_B_RAMP_MIN_COVERAGE,
                      "pass": b_pass}
    out["verdict"] = ("CONFIRMED" if a_pass and b_pass else "PARTIAL" if (a_pass or b_pass) else "REFUTED")
    return out

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "first_run"
    res = run(tag)
    path = data.ROOT / "eval_results.json"
    hist = json.loads(path.read_text()) if path.exists() else []
    hist.append(res)                      # append only; first entry is ground truth
    path.write_text(json.dumps(hist, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
