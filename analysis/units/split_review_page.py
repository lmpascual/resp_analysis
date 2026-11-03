"""One local HTML page to flip through every window's split candidates and record verdicts.

Reads <outputs>/grp/<animal>/<window>/split_review/split_candidates.tsv (split_pairs.py) and writes
<outputs>/grp/split_review.html next to them; open it in a browser from that folder (the figures are linked by
relative path). Keys: right / left = next / previous, m = merge, s = separate, u = unsure, n = note. Verdicts are kept
in the browser (localStorage) while you work; "Save verdicts" downloads split_verdicts.tsv. Read it back with
--ingest <split_verdicts.tsv>, which fills each window's your_verdict / note columns.

    python split_review_page.py [--root <outputs>/grp] [--ingest <downloaded split_verdicts.tsv>]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "outputs" / "grp"
# trace-review server port per window (the .claude/launch.json entries <animal>-<window>-review, 2026-10-30)
PORTS = {("e038", "anoxia1"): 8801, ("e038", "anoxia2"): 8802, ("e038", "post_trial"): 8803, ("e038", "episodes"): 8804,
         ("e038", "terminal"): 8805, ("e038", "baseline"): 8806, ("e040", "anoxia3"): 8807, ("e040", "death"): 8808,
         ("e040", "baseline"): 8809, ("e040", "cno"): 8810, ("e040", "sighs"): 8811}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Split review</title>
<style>
:root { --bg: #fbfaf7; --ink: #2f2e2b; --muted: #77756f; --merge: #1c9157; --sep: #d93636; --unsure: #c27a00; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, sans-serif; }
header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid #ddd; flex-wrap: wrap; }
header b { font-size: 15px; }
#fig { width: 100%; max-width: 1500px; display: block; margin: 8px auto; }
.btn { border: 1px solid #bbb; background: #fff; padding: 5px 12px; border-radius: 6px; cursor: pointer; }
.btn.merge.on { background: var(--merge); color: #fff; } .btn.separate.on { background: var(--sep); color: #fff; }
.btn.unsure.on { background: var(--unsure); color: #fff; }
#note { width: 320px; padding: 4px; }
#meta { color: var(--muted); }
#list { display: flex; flex-wrap: wrap; gap: 3px; padding: 6px 16px; }
#list span { width: 10px; height: 10px; border-radius: 2px; background: #ddd; cursor: pointer; }
#list span.merge { background: var(--merge); } #list span.separate { background: var(--sep); }
#list span.unsure { background: var(--unsure); } #list span.cur { outline: 2px solid var(--ink); }
</style></head><body>
<header>
  <b id="title"></b><span id="meta"></span>
  <button class="btn merge" onclick="setV('merge')">merge (m)</button>
  <button class="btn separate" onclick="setV('separate')">separate (s)</button>
  <button class="btn unsure" onclick="setV('unsure')">unsure (u)</button>
  <input id="note" placeholder="note (n)" oninput="setNote(this.value)">
  <button class="btn" onclick="go(-1)">&larr;</button><button class="btn" onclick="go(1)">&rarr;</button>
  <button class="btn" onclick="save()">Save verdicts</button><span id="count"></span>
  <span id="links"></span>
</header>
<div id="list"></div>
<img id="fig" alt="">
<script>
const ROWS = __ROWS__;
const KEY = "split_review_verdicts_v1";
let V = {};
try { V = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { V = {}; }
for (const r of ROWS) { const k = r.key; if (!V[k] && r.your_verdict) V[k] = {v: r.your_verdict, note: r.note || ""}; }
let i = 0;
function store() { try { localStorage.setItem(KEY, JSON.stringify(V)); } catch (e) {} }
function render() {
  const r = ROWS[i];
  document.getElementById("fig").src = r.fig;
  document.getElementById("title").textContent = `${i + 1}/${ROWS.length}  ${r.animal} ${r.window}: #${r.unit_first} -> #${r.unit_second}`;
  document.getElementById("meta").textContent = `cos ${r.template_cosine.toFixed(3)}, ${r.label_first}/${r.label_second}, handover ${r.first_last_active_run_s}/${r.second_first_active_run_s} s`;
  document.getElementById("links").innerHTML = r.port ? `traces (start ${r.animal}-${r.window}-review first): ` +
    `<a href="http://localhost:${r.port}/#unit=${r.unit_first}" target="_blank">#${r.unit_first}</a> · ` +
    `<a href="http://localhost:${r.port}/#unit=${r.unit_second}" target="_blank">#${r.unit_second}</a>` : "";
  const v = (V[r.key] || {}).v || "";
  for (const b of document.querySelectorAll(".btn.merge,.btn.separate,.btn.unsure")) b.classList.toggle("on", b.classList.contains(v));
  document.getElementById("note").value = (V[r.key] || {}).note || "";
  const L = document.getElementById("list");
  L.innerHTML = "";
  ROWS.forEach((x, j) => { const s = document.createElement("span"); s.className = ((V[x.key] || {}).v || "") + (j === i ? " cur" : "");
    s.title = `${x.animal} ${x.window} #${x.unit_first}->#${x.unit_second}`; s.onclick = () => { i = j; render(); }; L.appendChild(s); });
  document.getElementById("count").textContent = `${Object.values(V).filter(x => x.v).length} decided`;
}
function setV(v) { const k = ROWS[i].key; V[k] = Object.assign(V[k] || {}, {v}); store(); go(1); }
function setNote(t) { const k = ROWS[i].key; V[k] = Object.assign(V[k] || {}, {note: t}); store(); }
function go(d) { i = Math.max(0, Math.min(ROWS.length - 1, i + d)); render(); }
function save() {
  const lines = ["animal\\twindow\\tunit_first\\tunit_second\\tyour_verdict\\tnote"];
  for (const r of ROWS) { const x = V[r.key] || {}; lines.push([r.animal, r.window, r.unit_first, r.unit_second, x.v || "", (x.note || "").replace(/[\\t\\n]/g, " ")].join("\\t")); }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/tab-separated-values"}));
  a.download = "split_verdicts.tsv"; a.click();
}
document.addEventListener("keydown", e => {
  if (e.target.id === "note") { if (e.key === "Enter" || e.key === "Escape") e.target.blur(); return; }
  if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "m") setV("merge"); else if (e.key === "s") setV("separate"); else if (e.key === "u") setV("unsure");
  else if (e.key === "n") { e.preventDefault(); document.getElementById("note").focus(); }
});
render();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--ingest", default=None, help="split_verdicts.tsv downloaded from the page")
    a = p.parse_args()
    root = Path(a.root)
    files = [f for f in sorted(root.glob("*/*/split_review/split_candidates.tsv")) if f.stat().st_size > 10]
    if a.ingest:
        V = pd.read_csv(a.ingest, sep="\t", dtype=str).fillna("")
        for f in files:
            animal, window = f.parts[-4], f.parts[-3]
            C = pd.read_csv(f, sep="\t", dtype={"your_verdict": str, "note": str}).fillna({"your_verdict": "", "note": ""})
            v = V[(V["animal"] == animal) & (V["window"] == window)].copy()
            v["unit_first"], v["unit_second"] = v["unit_first"].astype(int), v["unit_second"].astype(int)
            C = C.drop(columns=["your_verdict", "note"]).merge(v[["unit_first", "unit_second", "your_verdict", "note"]],
                                                               on=["unit_first", "unit_second"], how="left")
            C[["your_verdict", "note"]] = C[["your_verdict", "note"]].fillna("")
            C.to_csv(f, sep="\t", index=False)
            print(f"{animal}/{window}: {int((C['your_verdict'] != '').sum())} of {len(C)} decided")
        return
    rows = []
    for f in files:
        animal, window = f.parts[-4], f.parts[-3]
        C = pd.read_csv(f, sep="\t").fillna({"your_verdict": "", "note": ""})
        for r in C.to_dict(orient="records"):
            fig = f"{animal}/{window}/split_review/figures/{r['template_cosine']:.3f}_{r['unit_first']}_{r['unit_second']}.png"
            rows.append({"key": f"{animal}|{window}|{r['unit_first']}|{r['unit_second']}", "animal": animal,
                         "port": PORTS.get((animal, window)),
                         "window": window, "fig": fig, **{k: r[k] for k in (
                             "unit_first", "unit_second", "template_cosine", "label_first", "label_second",
                             "first_last_active_run_s", "second_first_active_run_s", "your_verdict", "note")}})
    rows.sort(key=lambda r: (r["animal"], r["window"], -r["template_cosine"]))
    out = root / "split_review.html"
    out.write_text(PAGE.replace("__ROWS__", json.dumps(rows, default=str)), encoding="utf-8")
    print(f"{len(rows)} pairs from {len(files)} windows -> {out}")
    _ = html  # (escaping not needed: rows go in as JSON)


if __name__ == "__main__":
    main()
