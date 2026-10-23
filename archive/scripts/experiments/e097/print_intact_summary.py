import json
from pathlib import Path

p = Path("outputs/e097/intact_window/kilosort4_intact_analysis_results.json")
d = json.load(open(p))

print("=== SUBSAMPLING ===")
for r in d["subsampling"]["results_by_size"]:
    st = [round(x, 3) for x in r["state_f1_mean"]]
    print(f"N={r['N']:3d} ({r['fraction']*100:5.1f}%): ARI={r['ari_mean']:.3f} +- {r['ari_sem']:.3f}, MacroF1={r['macro_f1_mean']:.3f} +- {r['macro_f1_sem']:.3f}, States={st}")

print("\n=== ABLATIONS ===")
for ab in d["attribution"]["ablation_results"]:
    print(f"Drop {int(ab['frac']*100)}% ({ab['k_dropped']} units): No Top ARI={ab['ari_no_top']:.3f} (F1={ab['f1_no_top']:.3f}), No Rand ARI={ab['ari_no_random_mean']:.3f}+-{ab['ari_no_random_sem']:.3f} (F1={ab['f1_no_random_mean']:.3f}), No Bot ARI={ab['ari_no_bottom']:.3f} (F1={ab['f1_no_bottom']:.3f})")

print("\n=== COORDINATION ===")
c = d["coordination"]
for k, v in c.items():
    print(f"{k}: {v}")

print("\n=== TOP 10 DRIVERS ===")
sorted_u = sorted(d["attribution"]["unit_stats"], key=lambda u: u["mi_bits"], reverse=True)
for rank, u in enumerate(sorted_u[:10], 1):
    print(f"#{rank:2d}: Unit {u['unit_idx']:3d}, depth={u['depth_um']:.1f} um, fr={u['mean_fr']:.2f} Hz, MI={u['mi_bits']:.4f} bits")

print("\n=== STATES ===")
print("Rates (Hz):", [round(x, 1) for x in d["mean_state_rates_hz"]])
print("Occupancy (%):", [round(x, 1) for x in d["state_occupancy_pct"]])
