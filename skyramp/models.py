"""Baselines + learned models. Every model predicts delta = pv(t+15) - pv(t) (kW)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
import lightgbm as lgb
from . import config as C

# ---- non-learned baselines -------------------------------------------------
def persistence(X):
    return np.zeros(len(X))

def smart_persistence(X):
    """Hold the clear-sky index constant: pv_hat(t+15) = pv(t) * cs(t+15)/cs(t)."""
    ok = X["cs_now"].values > 1.0
    ratio = np.where(ok, X["cs_fut"].values / np.maximum(X["cs_now"].values, 1.0), 1.0)
    return X["pv"].values * ratio - X["pv"].values

# ---- LightGBM ----------------------------------------------------------------
def img_summary(frames):
    """Cheap hand features: 4x4 pooled RGB of the current frame (48) + red/blue ratio
    stats + frame-to-frame brightness change (cloud motion proxy)."""
    now = frames[..., :3].astype(np.float32); old = frames[..., 3:].astype(np.float32)
    n = len(now)
    pooled = now.reshape(n, 4, 8, 4, 8, 3).mean((2, 4)).reshape(n, -1)
    rb = now[..., 0] / np.maximum(now[..., 2], 1.0)
    f = [pooled, rb.mean((1, 2))[:, None], rb.std((1, 2))[:, None],
         (now.mean((1, 2, 3)) - old.mean((1, 2, 3)))[:, None],
         np.abs(now - old).mean((1, 2, 3))[:, None]]
    return np.concatenate(f, 1)

def lgbm_fit_predict(Ftr, ytr, Fothers):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=C.SEED, verbose=-1)
    m.fit(Ftr, ytr)
    return [m.predict(F) for F in Fothers]

# ---- small CNN + tabular fusion ---------------------------------------------
class Net(nn.Module):
    def __init__(self, n_tab, use_img=True, use_tab=True):
        super().__init__()
        self.use_img, self.use_tab = use_img, use_tab
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        d = (64 if use_img else 0) + (64 if use_tab else 0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, img, tab):
        parts = []
        if self.use_img: parts.append(self.cnn(img))
        if self.use_tab: parts.append(self.tab(tab))
        return self.head(torch.cat(parts, 1)).squeeze(1)

def _to_t(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)

def nn_fit_predict(frames_tr, tab_tr, y_tr, others, use_img=True, use_tab=True,
                   seeds=(0, 1, 2), epochs=10):
    """others: list of (frames, tab). Returns list of prediction arrays (seed-averaged)."""
    mu, sd = tab_tr.mean(0), tab_tr.std(0) + 1e-6
    prep = lambda t: torch.from_numpy(((t - mu) / sd).astype(np.float32))
    Xi, Xt, y = _to_t(frames_tr), prep(tab_tr), torch.from_numpy(y_tr.astype(np.float32))
    outs = [np.zeros(len(o[0])) for o in others]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        net = Net(Xt.shape[1], use_img, use_tab)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * ((len(y) + 255) // 256))
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(y))
            for i in range(0, len(y), 256):
                idx = perm[i:i + 256]
                loss = nn.functional.mse_loss(net(Xi[idx], Xt[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            for k, (fr, tb) in enumerate(others):
                outs[k] += np.concatenate([net(_to_t(fr[i:i + 2048]), prep(tb[i:i + 2048])).numpy()
                                           for i in range(0, len(fr), 2048)]) / len(seeds)
    return outs
