import json
from pathlib import Path

p = Path("outputs/e097/intact_window/kilosort4_intact_6state_analysis_results.json")
d = json.load(open(p))

print("=== 6-STATE SUBSAMPLING ===")
for r in d["subsampling"]["results_by_size"]:
    st = [round(x, 3) for x in r["state_f1_mean"]]
    print(f"N={r['N']:3d}: ARI={r['ari_mean']:.3f}+-{r['ari_sem']:.3f}, mF1={r['macro_f1_mean']:.3f}+-{r['macro_f1_sem']:.3f}, States={st}")

print("\n=== ABLATIONS ===")
for ab in d["attribution"]["ablation_results"]:
    print(f"Drop {int(ab['frac']*100)}%: NoTop={ab['ari_no_top']:.3f}, NoRand={ab['ari_no_random_mean']:.3f}+-{ab['ari_no_random_sem']:.3f}, NoBot={ab['ari_no_bottom']:.3f}")

print("\nGini:", round(d["attribution"]["gini_mi"], 3))
print("PR:", round(d["coordination"]["participation_ratio"], 2))
print("PC1 var %:", round(d["coordination"]["top1_pct"], 1))
print("Mean comod %:", round(d["coordination"]["mean_comod_fraction"] * 100, 1))
print("Rates:", [round(x, 1) for x in d["mean_state_rates_hz"]])
print("Occ:", [round(x, 1) for x in d["state_occupancy_pct"]])
