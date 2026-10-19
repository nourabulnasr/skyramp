"""Load the cached slice, build the minute grid, features, targets and the block split."""
from pathlib import Path
import numpy as np, pandas as pd
from . import config as C

ROOT = Path(__file__).resolve().parents[1]

def load():
    z = np.load(ROOT / "cache" / "skippd_slice.npz")
    ts = pd.to_datetime(z["ts"]).floor("min")
    df = pd.DataFrame({"ts": ts, "pv": z["pv"], "row": np.arange(len(ts))})
    df = df.drop_duplicates("ts", keep="first").reset_index(drop=True)
    return df, z["img"]

def build(df):
    """One sample per minute t where pv(t+15) exists. Returns a frame of features/targets."""
    s = df.set_index("ts")["pv"]
    rowof = df.set_index("ts")["row"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="min")
    pv = s.reindex(full)
    row = rowof.reindex(full)
    X = pd.DataFrame(index=full)
    X["pv"] = pv
    for l in C.LAGS[1:]:
        X[f"pv_l{l}"] = pv.shift(l)
    X["pv_future"] = pv.shift(-C.HORIZON_MIN)
    X["row_now"] = row
    for l in C.IMG_LAGS[1:]:
        X[f"row_l{l}"] = row.shift(l)
    X["pv_std10"] = pv.rolling(10, min_periods=5).std()
    m = X.index.hour * 60 + X.index.minute
    X["mod"] = m
    X["doy"] = X.index.dayofyear
    X["date"] = X.index.normalize()
    return X.dropna(subset=["pv", "pv_future", "row_now"] + [f"pv_l{l}" for l in C.LAGS[1:]]
                    + [f"row_l{l}" for l in C.IMG_LAGS[1:]]).copy()

def clearsky_table(X_train):
    """Label-free clear-sky proxy: 90th pct of TRAIN pv by (5-min bucket, day-of-year +-20d).
    Returned as a function cs(doy, minute_of_day) -> kW. Fit on the train block only."""
    b = (X_train["mod"].values // 5).astype(int)
    d = X_train["doy"].values; v = X_train["pv"].values
    cache = {}
    def cs(doy, mod):
        out = np.empty(len(doy))
        for i, (dd, mm) in enumerate(zip(doy, (np.asarray(mod) // 5).astype(int))):
            k = (dd, mm)
            if k not in cache:
                sel = (np.abs(d - dd) <= 20) & (np.abs(b - mm) <= 1)
                cache[k] = np.quantile(v[sel], 0.9) if sel.sum() >= 20 else 0.0
            out[i] = cache[k]
        return out
    return cs

def assign_roles(X):
    days = np.sort(X["date"].unique())
    role = {}
    pos_in_block = {}
    for i, d in enumerate(days):
        blk, pos = divmod(i, C.BLOCK_DAYS)
        role[d] = C.ROLE_BY_POS.get(pos, "gap")
    return X["date"].map(role)

def make_dataset():
    df, img = load()
    X = build(df)
    X["role"] = assign_roles(X)
    X = X[X["role"] != "gap"].copy()
    cs = clearsky_table(X[X.role == "train"])
    X["cs_now"] = cs(X["doy"].values, X["mod"].values)
    X["cs_fut"] = cs(X["doy"].values, (X["mod"].values + C.HORIZON_MIN))
    X["delta"] = X["pv_future"] - X["pv"]
    X["ramp"] = (X["delta"].abs() >= C.RAMP_KW)
    return X, img

def tab_features(X):
    cols = ["pv"] + [f"pv_l{l}" for l in C.LAGS[1:]] + ["pv_std10", "mod", "doy", "cs_now", "cs_fut"]
    F = X[cols].copy()
    F["ratio_now"] = X["pv"] / np.maximum(X["cs_now"], 1.0)   # clear-sky index
    F["cs_slope"] = X["cs_fut"] - X["cs_now"]
    return F

def frames(X, img):
    a = img[X["row_now"].values.astype(int)]
    b = img[X["row_l5"].values.astype(int)]
    return np.concatenate([a, b], axis=-1)        # N,32,32,6 uint8
