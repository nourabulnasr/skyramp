"""Everything that must be fixed BEFORE looking at results lives here."""
HORIZON_MIN = 15          # forecast PV this many minutes ahead (SKIPP'D standard horizon)
RAMP_KW = 5.0             # ramp = |pv(t+15) - pv(t)| >= 5 kW (~17% of the 29.6 kW observed max)
LAGS = [0, 1, 2, 5, 10, 15, 30]   # PV history lags in minutes
IMG_LAGS = [0, 5]         # frames stacked for the CNN: now and 5 min ago
SEED = 0

# Block split: days are chunked into 6-day blocks = 5 data days + 1 discarded gap day.
# Data-day position p in 0..4 -> train for p in {0,1,2}, calibration p=3, test p=4.
# Whole days never straddle sets and the gap day separates adjacent blocks.
BLOCK_DAYS = 6
ROLE_BY_POS = {0: "train", 1: "train", 2: "train", 3: "calib", 4: "test"}

ALPHA = 0.10              # conformal miscoverage -> 90% intervals

# Pre-declared eval thresholds (written before the first eval run)
CLAIM_A_MIN_REL_RMSE_GAIN = 0.05   # image+PV must cut ramp-minute RMSE by >=5% vs best PV-only
CLAIM_B_COVERAGE_BAND = (0.87, 0.93)   # empirical 90% coverage must land in this band
CLAIM_B_RAMP_MIN_COVERAGE = 0.80   # ... and ramp-minute coverage should not fall below this
N_BOOT = 2000
