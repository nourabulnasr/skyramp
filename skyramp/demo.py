"""30-second demo: no download, no training. Recomputes the eval from the committed
predictions (nothing is written to eval_results.json) and prints the headline table."""
import json
from . import eval as E, data

def main():
    r = E.run("demo_recompute")
    M = r["models"]
    print(f"\nSkyRamp - 15-min-ahead PV forecast, test blocks: n={r['n_test']}, ramp minutes n={r['n_test_ramp']} (|dPV|>={r['ramp_kw']} kW)")
    print(f"{'model':20s} {'RMSE all':>9s} {'RMSE ramp':>10s} {'RMSE non-ramp':>14s}")
    for k, v in M.items():
        print(f"{k:20s} {v['rmse_all']:9.2f} {v['rmse_ramp']:10.2f} {v['rmse_nonramp']:14.2f}")
    a = r["claim_A"]
    print(f"\nClaim A (images help at ramps): {a['best_img_plus_pv']} vs {a['best_pv_only']} (both picked on calib)"
          f" -> ramp RMSE gain {a['rel_gain']:.1%}, 95% CI [{a['ci95'][0]:.1%}, {a['ci95'][1]:.1%}], pass={a['pass']}")
    print("  non-ramp gain vs same baseline: %.1f%%" % (100 * a["rel_gain_nonramp"]))
    for k, v in r["claim_B"]["variants"].items():
        print(f"Claim B conformal {k:26s} coverage all={v['coverage_all']:.3f} ramp={v['coverage_ramp']:.3f}")
    print("\nStored history (eval_results.json):")
    for h in json.loads((data.ROOT / "eval_results.json").read_text()):
        if "claim_B" in h and h["tag"] != "first_run":
            b = h["claim_B"]
            print(f"  {h['tag']}: all={b['coverage_all']:.3f} ramp={b['coverage_ramp']:.3f} pass={b['pass']}")
        elif h["tag"] == "first_run":
            print(f"  first_run: verdict={h['verdict']} A.pass={h['claim_A']['pass']} B.pass={h['claim_B']['pass']}")

if __name__ == "__main__":
    main()
