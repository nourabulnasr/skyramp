# EVAL-FINDINGS

## CLAIM_UNDER_TEST
* **A.** Adding sky-image information to PV history reduces 15-min-ahead RMSE on ramp minutes (|ΔPV over 15 min| ≥ 5 kW)
  by ≥ 5% versus the best PV-only model, with a day-block bootstrap 95% CI whose lower bound is > 0.
* **B.** Split-conformal 90% intervals reach empirical coverage in [0.87, 0.93] on held-out test blocks AND keep ≥ 0.80 coverage on ramp minutes.

Thresholds were fixed in `skyramp/config.py` and committed before the first training/eval run. Model choice inside each
family (PV-only: smart persistence / LightGBM / NN; image+PV: CNN / LightGBM+image features) was made on the calibration block only.
Test: 10,229 minutes over 15 days, 344 ramp minutes.

## Run 1 — first_run (ground truth, never edited)
| model | RMSE all | RMSE ramp | RMSE non-ramp | RMSE calib |
|---|---|---|---|---|
| persistence | 2.10 | 9.18 | 1.28 | 1.72 |
| smart persistence | 1.88 | 9.05 | 0.90 | 1.48 |
| LightGBM PV-only | 1.67 | 6.39 | 1.22 | 1.56 |
| NN PV-only | 1.60 | 6.31 | 1.12 | 1.53 |
| LightGBM PV+image feats | 1.58 | 5.41 | 1.25 | 1.54 |
| CNN image+PV | 1.55 | 6.04 | 1.11 | 1.43 |
| CNN image-only (+sun/clear-sky, no PV values) | 1.91 | 8.72 | 1.06 | 1.49 |

**Claim A:** chosen on calib → PV-only = *smart persistence* (calib RMSE 1.484), image+PV = *CNN* (1.426).
Ramp RMSE 9.05 → 6.04, gain **33.3%**, 95% CI [27.0%, 43.9%] → **passes** the pre-declared rule.
Same pair on non-ramp minutes: **−23.3%** (images+learning are worse than smart persistence there).

**Claim B** (constant-width conformal on the CNN): coverage all **0.864** (band 0.87–0.93, just misses), non-ramp 0.889,
ramp **0.140** → **fails**. PV-only plain: ramp coverage 0.000. Trailing-std-normalised variant: ramp 0.401.

### Caveat on claim A the pre-declared rule hides
Smart persistence won the calibration-overall selection (it is excellent on non-ramp minutes) but is a poor *ramp* predictor.
Against the learned PV-only models on ramp minutes the picture is much weaker (test RMSE, descriptive, not re-selected):
* CNN image+PV 6.04 vs NN PV-only 6.31 → **4.4% gain (below the 5% threshold)**; vs LightGBM PV-only 6.39 → 5.5%.
* LightGBM + image features 5.41 vs LightGBM PV-only 6.39 → 15.4%, vs NN PV-only 6.31 → 14.3% (but it was not the calib pick, and it is *worse* on non-ramp: 1.25 vs 1.22).
* Image-only CNN (no PV values) 8.72 ≈ smart persistence → images alone carry little ramp signal; the gain is the *fusion*.
So "images help at ramps" is directionally supported, but the size of the effect depends on which PV-only baseline you pick, and
with 15 test days / 344 ramp minutes the CI does not cover model-selection noise. I did not re-pick the baseline after seeing test numbers.

## Fix attempt 1 (claim B) — `skyramp/eval_fix1.py`
* **Diagnosis:** split conformal only guarantees marginal coverage; |residual| is heteroscedastic (≈6 kW at ramps vs ≈1.1 kW otherwise) so one quantile can't adapt; trailing-std is a weak ramp-risk proxy.
* **Change:** learned local scale (LightGBM predicting |residual| from PV + image features, fit on half the calibration days; quantile of |res|/scale on the other half).
* **Effect:** ramp 0.14 → 0.529 but overall 0.864 → **0.830**, non-ramp 0.840. Still fails.

## Fix attempt 2 (claim B) — `skyramp/eval_fix2.py`
* **Diagnosis:** fix 1 calibrated on ~7 days; minute residuals are autocorrelated within a day so the effective calibration size is days, not minutes — the quantile was too noisy/low.
* **Change:** 5-fold-by-day out-of-fold scale predictions so all 15 calibration days set the quantile; test scale from a model fit on all calibration days.
* **Effect:** overall **0.867** (0.003 under the band), non-ramp 0.874, ramp **0.651** (threshold 0.80). Still fails. Mean ramp-interval width 11.8 kW (vs ≈2 kW normally).
Stopped here per the two-attempt limit.

## Verdict: **PARTIAL**
* Claim A: formally passes (33.3%, CI [27%, 44%]), but only 4.4% against the best learned PV-only model on ramps; non-ramp is no better. Works as "fusion beats naive persistence at ramps", not as a robust "images beat a strong PV-only model".
* Claim B: **REFUTED** as stated. Marginal coverage is close to nominal (0.86–0.87) but conditional coverage at ramps is far from it (0.14 → 0.65 after both fixes). The useful finding is that a headline marginal coverage number would have hidden a ramp-coverage failure.

Limits: one season-spanning slice from a single site, 15 test days, one seed set for the nets (3-seed average), no hyperparameter search.
