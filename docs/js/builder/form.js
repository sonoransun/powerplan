/**
 * form.js — the builder's left rail: start-from select, scale seg,
 * component cards with twin range/number inputs, and the add-type popover.
 *
 * Renders into the .rail container and talks only to builder/state.js and
 * builder/catalog.js. Param edits update headers in place (no re-render,
 * focus preserved); structural changes re-render the component lists.
 */

import {
  SCALE_ORDER, SCALES_INFO, typeDef, typeList, paramBounds,
  sourceKw, storageEnergyKwh, storagePowerKw, H2_KWH_PER_KG,
} from "./catalog.js";
import {
  state, subscribe, setMeta, setScale, addComponent, removeComponent,
  duplicateComponent, setParam, resetBlank, loadPresetTemplate,
} from "./state.js";
import {
  PALETTE, dataColor, sourceColor, storageColor,
  formatKw, formatKwh, specChip,
} from "../theme.js";
import { getManifest } from "../data.js";

const CATS = [
  { cat: "sources", kicker: "SOURCES", add: "+ Add source" },
  { cat: "storage", kicker: "STORAGE", add: "+ Add storage" },
  { cat: "controllers", kicker: "CONTROLLERS", add: "+ Add controller" },
];

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

function compColor(cat, def) {
  if (cat === "sources") return sourceColor(def.label);
  if (cat === "storage") return storageColor(def.label);
  const map = { mppt: PALETTE.solar, fusion: PALETTE.fusion, antimatter: PALETTE.antimatter,
    hydrogen: PALETTE.hydrogen, cryogenic: PALETTE.smes };
  return dataColor(map[def.key] ?? PALETTE.loss);
}

function headline(cat, comp) {
  if (cat === "sources") return formatKw(sourceKw(comp));
  if (cat === "storage") return `${formatKwh(storageEnergyKwh(comp))} · ${formatKw(storagePowerKw(comp))}`;
  return formatKw(Number(comp.params.rated_kw) || 0);
}

// ── public API ────────────────────────────────────────────────────────

let railEl = null;
let openCards = new Set(); // comp ids whose <details> are open

export function renderForm(container) {
  railEl = container;
  buildSkeleton();
  fillPresetSelect();
  renderScaleSeg();
  renderAllLists();

  subscribe((kind) => {
    if (kind === "param") {
      refreshHeadlines();
    } else { // structure | meta | reset
      renderScaleSeg();
      renderAllLists();
      syncMetaInputs();
    }
  });
}

// ── skeleton ──────────────────────────────────────────────────────────

function buildSkeleton() {
  railEl.innerHTML = `
    <section class="panel" style="--rule:${dataColor(PALETTE.accent)}">
      <div class="panel-head"><span class="kicker">Start From</span></div>
      <div class="panel-body">
        <label class="field"><span>Template</span>
          <select id="bf-preset"><option value="">Blank canvas</option></select>
        </label>
        <label class="field"><span>Configuration name</span>
          <input type="text" id="bf-name" maxlength="48">
        </label>
        <div id="bf-template-note"></div>
      </div>
    </section>
    <section class="panel" style="--rule:${dataColor(PALETTE.demand)}">
      <div class="panel-head"><span class="kicker">Deployment Scale</span>
        <span class="num" id="bf-scale-peak"></span></div>
      <div class="panel-body">
        <div class="seg" id="bf-scale-seg" role="group" aria-label="Deployment scale"></div>
        <p class="muted" id="bf-scale-desc" style="margin:8px 0 0;font-size:12px;"></p>
      </div>
    </section>
    ${CATS.map(({ cat, kicker, add }) => `
      <section class="panel" style="--rule:var(--ink-faint)" data-cat-panel="${cat}">
        <div class="panel-head"><span class="kicker">${kicker}</span>
          <span class="muted num" data-cat-count="${cat}"></span></div>
        <div class="panel-body">
          <div data-cat-list="${cat}" style="display:flex;flex-direction:column;gap:8px;"></div>
          <button class="btn" data-cat-add="${cat}" style="width:100%;margin-top:8px;">${add}</button>
        </div>
      </section>`).join("")}
  `;

  railEl.querySelector("#bf-name").addEventListener("change", (e) => {
    setMeta({ name: e.target.value.trim() || "Custom build" });
  });
  for (const btn of railEl.querySelectorAll("[data-cat-add]")) {
    btn.addEventListener("click", () => openTypePicker(btn, btn.dataset.catAdd));
  }
  syncMetaInputs();
}

function syncMetaInputs() {
  const nameEl = railEl.querySelector("#bf-name");
  if (nameEl && document.activeElement !== nameEl) nameEl.value = state.name;
}

// ── start-from select ─────────────────────────────────────────────────

async function fillPresetSelect() {
  const sel = railEl.querySelector("#bf-preset");
  try {
    const manifest = await getManifest();
    for (const p of manifest.presets ?? []) {
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.exotic ? `${p.name} — speculative` : p.name;
      sel.appendChild(opt);
    }
  } catch (e) {
    const opt = document.createElement("option");
    opt.disabled = true;
    opt.textContent = "presets unavailable (data/index.json failed)";
    sel.appendChild(opt);
  }
  sel.addEventListener("change", async () => {
    const note = railEl.querySelector("#bf-template-note");
    note.innerHTML = "";
    if (!sel.value) { resetBlank(); return; }
    sel.disabled = true;
    try {
      const { unmapped } = await loadPresetTemplate(sel.value);
      if (unmapped.length) {
        note.innerHTML = `<p class="muted" style="font-size:11.5px;margin:4px 0 0;">
          Cloned as editable template. Approximations: ${esc(unmapped.join("; "))}.</p>`;
      }
    } catch (e) {
      note.innerHTML = `<div class="callout error" style="margin:8px 0 0;">
        Could not load preset: ${esc(e.message)}</div>`;
    } finally {
      sel.disabled = false;
    }
  });
}

// ── scale seg ─────────────────────────────────────────────────────────

function renderScaleSeg() {
  const seg = railEl.querySelector("#bf-scale-seg");
  seg.innerHTML = SCALE_ORDER.map((k) =>
    `<button data-scale="${k}" class="${k === state.scale_key ? "on" : ""}"
       title="${esc(SCALES_INFO[k].description)}">${SCALES_INFO[k].label.replace("Single ", "")}</button>`
  ).join("");
  for (const b of seg.querySelectorAll("button")) {
    b.addEventListener("click", () => setScale(b.dataset.scale));
  }
  const info = SCALES_INFO[state.scale_key];
  railEl.querySelector("#bf-scale-peak").innerHTML =
    `${formatKw(info.peak_load_kw)}<span class="unit">peak</span>`;
  railEl.querySelector("#bf-scale-desc").textContent = info.description;
}

// ── component lists ───────────────────────────────────────────────────

function renderAllLists() {
  for (const { cat } of CATS) renderList(cat);
}

function renderList(cat) {
  const list = railEl.querySelector(`[data-cat-list="${cat}"]`);
  const count = railEl.querySelector(`[data-cat-count="${cat}"]`);
  count.textContent = state[cat].length ? `× ${state[cat].length}` : "";
  list.innerHTML = "";
  if (!state[cat].length) {
    list.innerHTML = `<p class="faint" style="font-size:11.5px;margin:2px 0;">
      ${cat === "controllers" ? "none — engine auto-sizes a default trio" : "none added"}</p>`;
    return;
  }
  for (const comp of state[cat]) list.appendChild(buildCard(cat, comp));
}

function buildCard(cat, comp) {
  const def = typeDef(cat, comp.type);
  const card = document.createElement("details");
  card.className = "comp-card";
  card.style.setProperty("--comp", compColor(cat, def));
  if (openCards.has(comp.id)) card.open = true;
  card.addEventListener("toggle", () => {
    if (card.open) openCards.add(comp.id); else openCards.delete(comp.id);
  });

  const summary = document.createElement("summary");
  summary.innerHTML = `
    <span>${esc(def.label)}</span>${def.speculative ? specChip(true) : ""}
    <span class="cap" data-headline>${headline(cat, comp)}</span>
    <button class="icon-btn" data-act="dup" title="Duplicate">⧉</button>
    <button class="icon-btn" data-act="del" title="Remove">✕</button>`;
  summary.querySelector('[data-act="dup"]').addEventListener("click", (e) => {
    e.preventDefault(); duplicateComponent(cat, comp.id);
  });
  summary.querySelector('[data-act="del"]').addEventListener("click", (e) => {
    e.preventDefault(); openCards.delete(comp.id); removeComponent(cat, comp.id);
  });
  card.appendChild(summary);

  const body = document.createElement("div");
  body.className = "body";
  const main = def.params.filter((p) => !p.advanced);
  const adv = def.params.filter((p) => p.advanced);
  for (const p of main) body.appendChild(buildField(cat, comp, p));
  if (def.h2) {
    const note = document.createElement("p");
    note.className = "muted";
    note.style.cssText = "font-size:11.5px;margin:6px 0 2px;";
    note.dataset.h2Readout = "";
    note.innerHTML = h2Readout(comp);
    body.appendChild(note);
  }
  if (adv.length) {
    const d = document.createElement("details");
    d.innerHTML = `<summary class="muted" style="cursor:pointer;font-size:11.5px;
      list-style:none;padding:4px 0;">▸ advanced (${adv.length})</summary>`;
    d.addEventListener("toggle", () => {
      d.querySelector("summary").textContent =
        `${d.open ? "▾" : "▸"} advanced (${adv.length})`;
    });
    for (const p of adv) d.appendChild(buildField(cat, comp, p));
    body.appendChild(d);
  }
  card.appendChild(body);
  return card;
}

function h2Readout(comp) {
  const kwh = storageEnergyKwh(comp);
  return `≈ <span class="num">${formatKwh(kwh)}</span> @ ${H2_KWH_PER_KG} kWh/kg (LHV)`;
}

// ── param fields ──────────────────────────────────────────────────────

function buildField(cat, comp, p) {
  const wrap = document.createElement("label");
  wrap.className = "field";

  if (p.kind === "bool") {
    wrap.style.display = "flex";
    wrap.style.alignItems = "center";
    wrap.style.gap = "8px";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.style.width = "auto";
    cb.checked = !!comp.params[p.name];
    cb.addEventListener("change", () => setParam(cat, comp.id, p.name, cb.checked));
    wrap.appendChild(cb);
    wrap.appendChild(Object.assign(document.createElement("span"),
      { textContent: p.label, style: "margin:0" }));
    return wrap;
  }

  if (p.kind === "enum") {
    wrap.innerHTML = `<span>${esc(p.label)}</span>`;
    const sel = document.createElement("select");
    for (const o of p.options) {
      const opt = document.createElement("option");
      opt.value = String(o.value);
      opt.textContent = o.label;
      sel.appendChild(opt);
    }
    sel.value = String(comp.params[p.name] ?? p.default);
    sel.addEventListener("change", () => {
      setParam(cat, comp.id, p.name, p.numeric ? Number(sel.value) : sel.value);
    });
    wrap.appendChild(sel);
    return wrap;
  }

  // range / int — twin slider + number
  const b = paramBounds(cat, comp.type, p, state.scale_key)
    ?? { min: p.min ?? 0, max: p.max ?? 100, step: p.step ?? 1, typical: null };
  const cur = comp.params[p.name];
  const shown = cur == null ? (autoValue(comp, p) ?? b.min) : cur;

  const head = document.createElement("span");
  head.style.cssText = "display:flex;justify-content:space-between;align-items:baseline;gap:6px;";
  head.innerHTML = `<span>${esc(p.label)}${p.unit ? ` <span class="unit">${p.unit}</span>` : ""}
    ${p.inert ? `<span class="faint" title="${esc(p.inert)}" style="cursor:help;">(inert)</span>` : ""}</span>
    ${cur == null && p.nullText ? `<span class="faint" data-null-tag style="font-size:10.5px;">${esc(p.nullText)}</span>` : ""}`;
  wrap.appendChild(head);

  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:1fr 86px;gap:8px;align-items:center;";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = b.min; slider.max = b.max; slider.step = b.step;
  slider.value = shown;
  const num = document.createElement("input");
  num.type = "number";
  num.min = p.kind === "int" ? b.min : 0;
  num.step = b.step;
  num.value = shown;

  const commit = (v) => {
    let val = Number(v);
    if (!isFinite(val)) return;
    if (p.kind === "int") val = Math.max(1, Math.round(val));
    slider.value = val; num.value = val;
    head.querySelector("[data-null-tag]")?.remove();
    setParam(cat, comp.id, p.name, val);
  };
  slider.addEventListener("input", () => { num.value = slider.value; });
  slider.addEventListener("change", () => commit(slider.value));
  num.addEventListener("change", () => commit(num.value));
  row.appendChild(slider);
  row.appendChild(num);
  wrap.appendChild(row);

  if (b.typical) {
    const [lo, hi] = b.typical;
    const band = document.createElement("div");
    const l = ((lo - b.min) / (b.max - b.min)) * 100;
    const w = Math.max(((hi - lo) / (b.max - b.min)) * 100, 1);
    band.innerHTML = `
      <div style="position:relative;height:3px;margin:2px 1px 3px;background:var(--hairline);
                  border-radius:2px;overflow:hidden;">
        <i style="position:absolute;left:${l.toFixed(1)}%;width:${w.toFixed(1)}%;top:0;bottom:0;
                  background:var(--ink-faint);border-radius:2px;"></i></div>
      <span class="faint" style="font-size:10.5px;">typical ${fmtBound(lo)}–${fmtBound(hi)}${p.unit ? ` ${p.unit}` : ""} at this scale</span>`;
    wrap.appendChild(band);
  }
  return wrap;
}

function fmtBound(x) {
  return x >= 1e6 ? `${x / 1e6}M` : x >= 1e4 ? `${x / 1e3}k` : String(x);
}

/** Value to display for null ("auto") params — e.g. tilt ≡ latitude. */
function autoValue(comp, p) {
  if (p.name === "tilt_deg") return Math.abs(Number(comp.params.latitude) || 35);
  if (p.name === "heat_rate_btu_kwh") return comp.params.plant_type === "peaker" ? 9500 : 6800;
  if (p.name === "thermal_efficiency") return comp.params.fuel_cycle === "pb11" ? 0.75 : 0.42;
  return null;
}

// ── live header refresh on param edits ────────────────────────────────

function refreshHeadlines() {
  for (const { cat } of CATS) {
    const list = railEl.querySelector(`[data-cat-list="${cat}"]`);
    const cards = list.querySelectorAll(".comp-card");
    state[cat].forEach((comp, i) => {
      const card = cards[i];
      if (!card) return;
      const h = card.querySelector("[data-headline]");
      if (h) h.textContent = headline(cat, comp);
      const h2 = card.querySelector("[data-h2-readout]");
      if (h2) h2.innerHTML = h2Readout(comp);
    });
  }
}

// ── type-picker popover ───────────────────────────────────────────────

let activePopover = null;

function closePopover() {
  if (activePopover) { activePopover.remove(); activePopover = null; }
  document.removeEventListener("pointerdown", onOutside, true);
  document.removeEventListener("keydown", onEscape, true);
}
function onOutside(e) { if (activePopover && !activePopover.contains(e.target)) closePopover(); }
function onEscape(e) { if (e.key === "Escape") closePopover(); }

function openTypePicker(anchorBtn, cat) {
  closePopover();
  const pop = document.createElement("div");
  pop.className = "popover";
  const rows = typeList(cat).map((t) => {
    const avail = cat === "controllers" ? true : !!t.availability[state.scale_key];
    const spec = cat !== "controllers" ? t.availability[state.scale_key] : null;
    const range = spec
      ? (cat === "sources"
          ? `${fmtBound(spec.kw[0])}–${fmtBound(spec.kw[1])} kW`
          : `${fmtBound(spec.kwh[0])}–${fmtBound(spec.kwh[1])} kWh`)
      : "";
    return `
      <button data-type="${t.key}" ${avail ? "" : "disabled"}
        style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;width:100%;
               text-align:left;font:500 12.5px var(--font-ui);padding:7px 8px;border:none;
               background:none;color:${avail ? "var(--ink)" : "var(--ink-faint)"};
               cursor:${avail ? "pointer" : "not-allowed"};border-radius:var(--radius);">
        <span>${esc(t.label)}${t.speculative ? specChip(true) : ""}</span>
        <span class="num faint" style="font-size:10.5px;white-space:nowrap;">
          ${avail ? range : "not modeled at this scale"}</span>
      </button>`;
  }).join("");
  pop.innerHTML = `
    <div class="kicker" style="margin-bottom:6px;">
      Add ${cat === "sources" ? "source" : cat === "storage" ? "storage" : "controller"}
      — ${esc(SCALES_INFO[state.scale_key].label)}</div>
    <div style="display:flex;flex-direction:column;">${rows}</div>`;

  const panel = anchorBtn.closest(".panel");
  panel.appendChild(pop);
  pop.style.left = "10px";
  pop.style.right = "10px";
  pop.style.width = "auto";
  pop.style.top = `${anchorBtn.offsetTop + anchorBtn.offsetHeight + 6}px`;

  for (const b of pop.querySelectorAll("button[data-type]")) {
    if (b.disabled) continue;
    b.addEventListener("mouseenter", () => { b.style.background = "var(--hairline)"; });
    b.addEventListener("mouseleave", () => { b.style.background = "none"; });
    b.addEventListener("click", () => {
      const comp = addComponent(cat, b.dataset.type);
      openCards.add(comp.id);
      closePopover();
      renderList(cat); // re-render to open the fresh card
    });
  }
  activePopover = pop;
  document.addEventListener("pointerdown", onOutside, true);
  document.addEventListener("keydown", onEscape, true);
}
