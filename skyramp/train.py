"""Fit every model on the train block, write predictions for calib + test blocks."""
import time, numpy as np, pandas as pd
from . import data, models, config as C

def main():
    X, img = data.make_dataset()
    tr, ca, te = (X[X.role == r] for r in ("train", "calib", "test"))
    F = {r: data.tab_features(d) for r, d in (("train", tr), ("calib", ca), ("test", te))}
    FR = {r: data.frames(d, img) for r, d in (("train", tr), ("calib", ca), ("test", te))}
    y_tr = tr["delta"].values
    out = pd.concat([ca, te])[["role", "pv", "pv_future", "delta", "ramp", "pv_std10", "date", "mod"]].copy()
    n_ca = len(ca)
    def put(name, p_ca, p_te):
        out[name] = np.concatenate([p_ca, p_te])
    def run(name, fn):
        t = time.time(); fn(); print(f"{name}: {time.time() - t:.0f}s", flush=True)

    put("persistence", models.persistence(ca), models.persistence(te))
    put("smart_persistence", models.smart_persistence(ca), models.smart_persistence(te))

    def lgb_pv():
        a, b = models.lgbm_fit_predict(F["train"], y_tr, [F["calib"], F["test"]]); put("lgbm_pv", a, b)
    run("lgbm_pv", lgb_pv)

    def lgb_img():
        S = {r: np.hstack([F[r].values, models.img_summary(FR[r])]) for r in F}
        a, b = models.lgbm_fit_predict(S["train"], y_tr, [S["calib"], S["test"]]); put("lgbm_pv_img", a, b)
    run("lgbm_pv_img", lgb_img)

    T = {r: F[r].values.astype(np.float32) for r in F}
    # tabular block for the image-only ablation: sun position / clear-sky only, no PV values
    nopv = [i for i, c in enumerate(F["train"].columns) if c in ("mod", "doy", "cs_now", "cs_fut", "cs_slope")]
    def nn_pv():
        a, b = models.nn_fit_predict(FR["train"], T["train"], y_tr, [(FR["calib"], T["calib"]), (FR["test"], T["test"])],
                                     use_img=False, use_tab=True); put("nn_pv", a, b)
    run("nn_pv", nn_pv)
    def cnn_fused():
        a, b = models.nn_fit_predict(FR["train"], T["train"], y_tr, [(FR["calib"], T["calib"]), (FR["test"], T["test"])],
                                     use_img=True, use_tab=True); put("cnn_img_pv", a, b)
    run("cnn_img_pv", cnn_fused)
    def cnn_img():
        sel = lambda r: T[r][:, nopv]
        a, b = models.nn_fit_predict(FR["train"], sel("train"), y_tr, [(FR["calib"], sel("calib")), (FR["test"], sel("test"))],
                                     use_img=True, use_tab=True); put("cnn_img_only", a, b)
    run("cnn_img_only", cnn_img)

    (data.ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(data.ROOT / "results" / "predictions.csv.gz", index_label="ts")
    print("wrote", len(out), "rows")

if __name__ == "__main__":
    main()
