// PowerPoint deck of the e097 single-unit figures (rasters, PETHs and phase curves), built from
// <figures>/units_slides/slides.json written by figures_signatures.write_slide_manifest.
//
//   npm install pptxgenjs@3.12.0          (in any folder; then point NODE_PATH at its node_modules)
//   node make_unit_slides.js <slides.json> <out.pptx> [apply_theme.js]
//
// The slide text is a draft generated from each unit's numbers; edit the interpretation in PowerPoint.
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const [manifestPath, outPath, applyThemePath] = process.argv.slice(2);
const M = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
// transition windows (transition_slides.py): figure slides after the breathing slide, two more slides per unit
const TR = M.transitions && fs.existsSync(M.transitions) ? JSON.parse(fs.readFileSync(M.transitions, "utf-8")) : null;
const hex = (c) => c.replace("#", "").toUpperCase();
const CONDS = M.conditions.map((c) => ({ ...c, colour: hex(c.colour) }));     // time order
const accent = (i) => CONDS[Math.min(i, CONDS.length - 1)].colour;

const THEME = {
  name: "e097 units",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F1F1F", lt1: "FFFFFF", dk2: "3A3F47", lt2: "F2F2F0",
    accent1: accent(0), accent2: accent(1), accent3: accent(2), accent4: accent(3), accent5: accent(CONDS.length - 1),
    accent6: "52514E", hlink: accent(0), folHlink: accent(CONDS.length - 1),
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = M.title;
const C = pres.SchemeColor;
const num = { x: 12.3, y: 7.0, w: 0.7, h: 0.3, fontSize: 10, color: C.text2, align: "right" };

pres.defineSlideMaster({
  title: "Title", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.4, fontSize: 40,
      bold: true, color: C.text1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.85, w: 11.7, h: 1.2, fontSize: 20,
      color: C.text2, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Section", background: { color: C.background2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.7, w: 11.7, h: 1.2, fontSize: 36,
      bold: true, color: C.text1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.0, fontSize: 18,
      color: C.text2, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: num,
});
pres.defineSlideMaster({
  title: "Raster", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: 9.35, y: 0.45, w: 3.5, h: 0.4, fontSize: 14,
      color: C.text2, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 9.35, y: 0.9, w: 3.5, h: 3.05, fontSize: 28,
      bold: true, color: C.text1, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 9.35, y: 4.05, w: 3.5, h: 2.85, fontSize: 15,
      color: C.text1, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: num,
});
pres.defineSlideMaster({          // transition raster: six windows of draft text, so a smaller title and body
  title: "RasterTransition", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: 9.35, y: 0.45, w: 3.6, h: 0.4, fontSize: 14,
      color: C.text2, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 9.35, y: 0.9, w: 3.6, h: 2.3, fontSize: 22,
      bold: true, color: C.text1, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 9.35, y: 3.3, w: 3.6, h: 3.65, fontSize: 13,
      color: C.text1, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: num,
});
pres.defineSlideMaster({
  title: "Peth", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 1.05, fontSize: 24,
      bold: true, color: C.text1, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: num,
});

// title
let s = pres.addSlide({ masterName: "Title" });
s.addText(M.title, { placeholder: "title" });
s.addText(M.subtitle, { placeholder: "body" });

// intro (transition_slides.INTRO): the larger question and hypothesis, then the questions these recordings address
const I = M.intro && fs.existsSync(M.intro) ? JSON.parse(fs.readFileSync(M.intro, "utf-8")) : null;
if (I) {
  const label = (sl, text, y) => sl.addText(text.toUpperCase(), { x: 0.6, y, w: 6, h: 0.3, fontSize: 12, bold: true,
    color: C.text2, charSpacing: 1, margin: 0, isTextBox: true, objectName: `label ${text}` });
  s = pres.addSlide({ masterName: "Peth" });
  s.addText(I.title, { placeholder: "title" });
  label(s, "The question", 1.5);
  s.addText(I.question, { x: 0.6, y: 1.82, w: 12.1, h: 0.85, fontSize: 20, color: C.text1, margin: 0, valign: "top",
    isTextBox: true, objectName: "question" });
  label(s, "Central hypothesis", 2.95);
  const bars = [CONDS[0].colour, (CONDS.find((c) => c.name === "gasping") || CONDS[0]).colour,
    (CONDS.find((c) => c.name === "early recovery") || CONDS[CONDS.length - 1]).colour];
  I.hypothesis.forEach((h, i) => {
    const y = 3.3 + i * 1.18;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.09, h: 1.0, fill: { color: bars[i] }, line: { color: bars[i] },
      objectName: `hypothesis bar ${i + 1}` });
    const [lead, ...rest] = h.split(" ");
    s.addText([{ text: lead + " ", options: { bold: true } }, { text: rest.join(" ") }], { x: 0.85, y, w: 5.9, h: 1.0,
      fontSize: 16, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `hypothesis ${i + 1}` });
  });
  if (I.figure && fs.existsSync(I.figure)) {
    const w = 5.9, h = (w * I.figure_h) / I.figure_w;
    s.addImage({ path: I.figure, x: 7.0, y: 3.35, w, h, altText: I.figure_caption, objectName: "trajectories" });
    s.addText(I.figure_caption, { x: 7.0, y: 3.35 + h + 0.08, w, h: 0.55, fontSize: 11, italic: true,
      color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "trajectories caption" });
  }
  s.addNotes("Question and hypothesis paraphrased from the K00 Research Strategy (Innovation; Aims 1–3).");

  s = pres.addSlide({ masterName: "Peth" });
  s.addText(I.subquestions_title, { placeholder: "title" });
  const n = I.subquestions.length, top = 1.5, rowH = (6.95 - top) / n;
  I.subquestions.forEach((q, i) => {
    const y = top + i * rowH;
    if (i > 0) s.addShape(pres.shapes.LINE, { x: 0.6, y: y - 0.06, w: 12.1, h: 0, line: { color: "D9D8D4", width: 0.75 },
      objectName: `divider ${i}` });
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.08, w: 0.55, h: 0.55, fill: { color: C.text2 },
      line: { color: C.text2 }, objectName: `number ${i + 1}` });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.08, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1,
      align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `number text ${i + 1}` });
    s.addText(q.q, { x: 1.4, y: y + 0.02, w: 8.5, h: 0.5, fontSize: 19, bold: true, color: C.text1, margin: 0,
      valign: "middle", isTextBox: true, objectName: `subquestion ${i + 1}` });
    s.addText(q.how, { x: 1.4, y: y + 0.52, w: 8.5, h: rowH - 0.62, fontSize: 13.5, color: C.text1, margin: 0,
      valign: "top", isTextBox: true, objectName: `measured ${i + 1}` });
    s.addText([{ text: q.aim, options: { bold: true, breakLine: true } }, { text: q.where, options: { fontSize: 11 } }],
      { x: 10.2, y: y + 0.08, w: 2.6, h: rowH - 0.2, fontSize: 13, color: C.text2, margin: 0, valign: "top",
        isTextBox: true, objectName: `where ${i + 1}` });
  });
  s.addNotes("The sub-questions map onto the K00 aims: network dynamics from eupnea to gasping (Aim 1), re-embedding " +
    "and post-inspiratory (PiCo) recruitment during recovery (Aims 1-3), cardiorespiratory coupling (Aim 3). These " +
    "recordings describe the unperturbed sequence that the aims' manipulations would be tested against.");
}

// how to read: the condition timeline as coloured blocks, proportional to time
s = pres.addSlide({ masterName: "Peth" });
s.addText("How to read the unit slides", { placeholder: "title" });
const blocks = CONDS.map((c) => [c.name, c.start_raw_s, c.end_raw_s, c.colour]);
const t0 = 1740, t1 = 2160, X0 = 0.8, W = 11.7, Y = 1.85;
const xOf = (t) => X0 + (W * (t - t0)) / (t1 - t0);
s.addShape(pres.shapes.RECTANGLE, { x: X0, y: Y, w: W, h: 0.55, fill: { color: C.background2 }, line: { color: C.background2 },
  objectName: "timeline background" });
blocks.forEach(([name, a, b, col], i) => {
  s.addShape(pres.shapes.RECTANGLE, { x: xOf(a), y: Y, w: xOf(b) - xOf(a), h: 0.55, fill: { color: col },
    line: { color: col }, objectName: `timeline ${name}` });
  const above = i % 2 === 0;
  s.addText(`${name}\n${Math.round(a)}–${Math.round(b)} s`, { x: xOf(a), y: above ? Y - 0.75 : Y + 0.6,
    w: Math.max(xOf(b) - xOf(a), 1.9), h: 0.7, fontSize: 13, color: C.text1, valign: above ? "bottom" : "top",
    margin: 0, isTextBox: true, objectName: `timeline label ${name}` });
});
s.addText("raw-run time; grey gaps belong to no condition" + (TR ? " here (they are the pre-gasp and post-gasp " +
  "windows of the transition slides)" : "") + ". Recovery is split at change points in breathing and population " +
  "state (epoch_change_points.py)", { x: X0, y: Y + 1.35, w: W,
  h: 0.35, fontSize: 12, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addText([
  { text: "Raster slide: ", options: { bold: true } },
  { text: "every breath is one row, in recording order, aligned to inspiration onset (left) and offset (right); the " +
      "coloured bars on the right mark the conditions.", options: { breakLine: true } },
  { text: "PETH slide: ", options: { bold: true } },
  { text: "firing rate around inspiration onset and offset, and over the breath phase (polar). Each numbered column " +
      "adds one condition in bold over the earlier ones (thin); the last column overlays all.", options: { breakLine: true } },
  ...(TR ? [{ text: "Transition slides: ", options: { bold: true } },
    { text: "each unit's spikes from late anoxia to early recovery (whole cycles), and PETHs with pre-gasp and " +
        "post-gasp as their own columns.", options: { breakLine: true } }] : []),
  { text: "Text: ", options: { bold: true } },
  { text: "a draft written from each unit's rates and tuning; check and edit the interpretation." },
], { x: X0, y: 3.85, w: W, h: 3.2, fontSize: TR ? 16 : 18, color: C.text1, paraSpaceAfter: 8, valign: "top",
  isTextBox: true });

// breathing rate and the conditions on one time axis (figures_population.fig_breathing_epochs, 13 x 7.4 in)
if (M.breathing_figure && fs.existsSync(M.breathing_figure)) {
  s = pres.addSlide({ masterName: "Peth" });
  s.addText("Breathing rate and conditions", { placeholder: "title" });
  const h = 5.75, w = (h * 13) / 7.4;
  s.addImage({ path: M.breathing_figure, x: (13.33 - w) / 2, y: 1.35, w, h,
    altText: "Breathing rate per breath coloured by condition, above the condition timeline",
    objectName: "breathing and conditions" });
}

// the transitions into and out of gasping: section divider, then one figure per slide (slide-sized PNGs); then the
// same for other trials (M.transition_sections, e.g. anoxia #2), figure slides only
function figureSection(T) {
  pres.addSection({ title: T.section.title });
  s = pres.addSlide({ masterName: "Section", sectionTitle: T.section.title });
  s.addText(T.section.title, { placeholder: "title" });
  s.addText(T.section.body, { placeholder: "body" });
  for (const f of T.figures) {
    if (!fs.existsSync(f.path)) continue;
    s = pres.addSlide({ masterName: "Peth", sectionTitle: T.section.title });
    s.addText(f.title, { placeholder: "title" });
    const w = Math.min(12.3, (5.75 * f.w) / f.h), h = (w * f.h) / f.w;
    s.addImage({ path: f.path, x: (13.33 - w) / 2, y: 1.4, w, h, altText: f.title,
      objectName: path.basename(f.path, ".png") });
    s.addNotes(f.notes);
  }
}
if (TR) figureSection(TR);
for (const p of M.transition_sections || []) {
  if (fs.existsSync(p)) figureSection(JSON.parse(fs.readFileSync(p, "utf-8")));
}

// one section per group, two slides per unit
for (const g of M.groups) {
  const units = M.units.filter((u) => u.group === g.key);
  if (!units.length) continue;
  pres.addSection({ title: g.title });
  s = pres.addSlide({ masterName: "Section", sectionTitle: g.title });
  s.addText(g.title, { placeholder: "title" });
  s.addText(`${units.length} unit${units.length > 1 ? "s" : ""}: ` + units.map((u) => `#${u.unit}`).join(", "),
    { placeholder: "body" });
  for (const u of units) {
    const notes = `Unit #${u.unit}. Draft text generated from unit_phase_tuning.tsv and unit_peth_features.tsv; ` +
      "check the interpretation before presenting.";
    s = pres.addSlide({ masterName: "Raster", sectionTitle: g.title });
    s.addImage({ path: u.raster, x: 0.35, y: 0.45, w: 8.7, h: 6.6, altText: `Unit ${u.unit} spike rasters`,
      objectName: `raster ${u.unit}` });
    s.addText(`Unit #${u.unit}  ·  ${Math.round(u.depth_um)} µm  ·  ${u.label}`, { placeholder: "tag" });
    s.addText(u.headline, { placeholder: "title" });
    s.addText(u.bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < u.bullets.length - 1 } })),
      { placeholder: "body", paraSpaceAfter: 8 });
    s.addNotes(notes);
    s = pres.addSlide({ masterName: "Peth", sectionTitle: g.title });
    s.addText(`#${u.unit}  ·  ${u.headline}`, { placeholder: "title" });
    s.addImage({ path: u.peth, x: 0.5, y: 1.45, w: 12.3, h: (12.3 * 5.75) / 12.5,
      altText: `Unit ${u.unit} PETHs and phase curves`, objectName: `peth ${u.unit}` });
    s.addNotes(notes);
    const t = TR && TR.units[String(u.unit)];
    if (t && fs.existsSync(t.raster) && fs.existsSync(t.peth)) {
      const tnotes = `Unit #${u.unit}, through the transitions. Draft text from transition_unit_tuning.tsv ` +
        "(windows with < 20 spikes get no phase); check the interpretation before presenting.";
      s = pres.addSlide({ masterName: "RasterTransition", sectionTitle: g.title });
      s.addImage({ path: t.raster, x: 0.35, y: 0.45, w: 8.7, h: 6.6, altText: `Unit ${u.unit} spikes through the ` +
        "transitions", objectName: `transition raster ${u.unit}` });
      s.addText(`Unit #${u.unit}  ·  through the transitions`, { placeholder: "tag" });
      s.addText(t.headline, { placeholder: "title" });
      s.addText(t.bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < t.bullets.length - 1 } })),
        { placeholder: "body", paraSpaceAfter: 5 });
      s.addNotes(tnotes);
      s = pres.addSlide({ masterName: "Peth", sectionTitle: g.title });
      s.addText(`#${u.unit} through the transitions  ·  ${t.headline}`, { placeholder: "title" });
      s.addImage({ path: t.peth, x: 0.5, y: 1.45, w: 12.3, h: (12.3 * 5.75) / 12.5,
        altText: `Unit ${u.unit} PETHs and phase curves with the transition windows`,
        objectName: `transition peth ${u.unit}` });
      s.addNotes(tnotes);
    }
  }
}

(async () => {
  await pres.writeFile({ fileName: outPath });
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(outPath, THEME);
  }
  console.log("wrote", outPath);
})();
