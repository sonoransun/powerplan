/**
 * pages/resilience.js — the Resilience page.
 *
 * Tab A "Stress one system": failure-timeline composer → sim.runFailure()
 *   → paired baseline/stressed KPIs + power-balance charts with event bands.
 * Tab B "Batch scenario lab": pre-baked sample (rendered on load, so the tab
 *   is alive without Pyodide) or live sim.runBatch() → histograms, component
 *   impact, failure-type severity, per-config pairs, source×storage heatmap,
 *   and a sortable, bin-filterable run table.
 */

import * as echarts from "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.esm.min.js";
import {
  PALETTE, dataColor, formatKwh, formatPct, formatHour,
  mountChart, specChip, SPEC_TEXT, isSpeculative, isDark, FONT_MONO,
} from "../theme.js";
import { getManifest, getResilienceSample, histogram } from "../data.js";
import { sim, wasmSupported } from "../pyodide/sim-client.js";
import { renderPowerBalance, kpiStrip } from "../charts/dashboard.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const truncate = (s, n) => { s = String(s == null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "…" : s; };

// ── failure-event vocabulary (colors per the cross-agent brief) ───────
const EV_TYPES = {
  demand_surge: { label: "demand surge", hex: PALETTE.demand, dur: 48 },
  weather_crisis: { label: "weather crisis", hex: PALETTE.curtail, dur: 168 },
  grid_disconnect: { label: "grid disconnect", hex: PALETTE.grid_import, dur: 24 },
  source_trip: { label: "source trip", hex: PALETTE.loss, dur: 72 },
  storage_fault: { label: "storage fault", hex: PALETTE.flywheel, dur: 48 },
  simultaneous: { label: "simultaneous", hex: PALETTE.gas, dur: 48 },
};

const STAGE_TEXT = {
  "loading-pyodide": "loading Python runtime (~15 MB, first time only)…",
  "loading-numpy": "loading numpy…",
  "fetching-package": "fetching simulation package…",
  "importing": "importing powerplan…",
  "ready": "engine ready",
};

const HEAT_RAMP = ["#00204D", "#31446B", "#666870", "#958F78", "#FFE945"];
const CHART_IDS = ["chHistLolp", "chHistEns", "chHistRec", "chImpact", "chFtype", "chPaired", "chHeat"];

const state = {
  manifest: null,
  hours: 720,
  events: [],          // {id, type, start_hour, duration_hours, severity?, multiplier?, num_components?, target?}
  evSeq: 1,
  stressHandles: [],
  days: 30,
  tier: "conventional",
  batch: { doc: null, handles: [], cleanups: [], filter: null, sortKey: "cfg", sortDir: 1 },
};

// ══════════════════════════════════════════════════════════════════════
// init
// ══════════════════════════════════════════════════════════════════════

export function init() {
  wireTabs();
  wireTabA();
  wireTabB();
  applyRules();
  loadManifest();
  loadSample();
  if (!wasmSupported()) disableLive();
  document.addEventListener("pp-theme-change", () => {
    applyRules();
    buildEventButtons();
    renderTimeline();
  });
}

/** Palette-keyed top rules, set via JS so dark-mode hex substitutions apply. */
function applyRules() {
  const set = (id, hex) => { const el = $(id); if (el) el.style.setProperty("--rule", dataColor(hex)); };
  set("cfgPanel", PALETTE.generation);
  set("tlPanel", PALETTE.demand);
  set("bCtl", PALETTE.loss);
  set("pHistLolp", PALETTE.discharge);
  set("pHistEns", PALETTE.demand);
  set("pHistRec", PALETTE.charge);
  set("pImpact", PALETTE.generation);
  set("pFtype", PALETTE.grid_import);
  set("pPaired", PALETTE.discharge);
  set("pHeat", PALETTE.grid_export);
  set("tblPanel", PALETTE.loss);
  set("pStressPB", PALETTE.unmet);
  set("pBasePB", PALETTE.discharge);
  set("pStressKpis", PALETTE.loss);
}

function wireTabs() {
  $("rzTabs").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => showTab(b.dataset.tab)));
  if (location.hash === "#batch") showTab("batch");
}

function showTab(tab) {
  $("rzTabs").querySelectorAll("button").forEach((b) =>
    b.classList.toggle("on", b.dataset.tab === tab));
  $("tabStress").hidden = tab !== "stress";
  $("tabBatch").hidden = tab !== "batch";
  try { history.replaceState(null, "", tab === "batch" ? "#batch" : "#stress"); } catch (e) { /* sandboxed */ }
}

function segWire(seg, fn) {
  seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    fn(b);
  }));
}

function disableLive() {
  $("aRun").disabled = true;
  $("bRun").disabled = true;
  const msg = `<div class="callout error"><strong>WebAssembly unavailable.</strong>
    <span class="muted">Live in-browser simulation needs WebAssembly, which this browser does not support.
    The pre-baked batch sample still renders below.</span></div>`;
  $("aError").innerHTML = msg;
  $("bError").innerHTML = msg;
}

// ── shared helpers ─────────────────────────────────────────────────────

function clampInt(v, lo, hi, dflt) {
  const n = Math.round(parseFloat(v));
  if (!isFinite(n)) return dflt;
  return Math.min(hi, Math.max(lo, n));
}

function progress(which, show, text, frac) {
  const line = $(which + "Prog");
  if (!line) return;
  line.hidden = !show;
  if (show) {
    $(which + "ProgText").textContent = text || "";
    $(which + "ProgBar").style.width = `${Math.round((frac || 0) * 100)}%`;
  }
}

function errorHtml(title, err) {
  const msg = err && err.message ? err.message : String(err);
  const tb = err && err.traceback
    ? `<details style="margin-top:8px"><summary class="muted" style="cursor:pointer;font-size:12px">Python traceback</summary>
       <div class="traceback">${esc(err.traceback)}</div></details>`
    : "";
  return `<div class="callout error"><strong>${esc(title)}.</strong> <span class="muted">${esc(msg)}</span>${tb}</div>`;
}

function ensureSpecCallout(container) {
  if (localStorage.getItem("pp-spec-dismissed") === "1") return;
  if (container.querySelector(".callout.spec")) return;
  container.innerHTML = `<div class="callout spec" style="display:flex;gap:12px;align-items:flex-start">
    <span style="flex:1">${esc(SPEC_TEXT)} <a href="reference.html#speculative">Model notes →</a></span>
    <button class="btn" style="padding:2px 8px;font-size:11px;flex:none" data-dismiss>dismiss</button></div>`;
  container.querySelector("[data-dismiss]").addEventListener("click", () => {
    localStorage.setItem("pp-spec-dismissed", "1");
    $("aSpecNote").innerHTML = "";
    $("bSpecNote").innerHTML = "";
  });
}

/** "LiquidElectrolyteBattery" → "Liquid Electrolyte"; "SolarPV" → "Solar PV". */
function shortType(name) {
  return String(name).replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/\s+(Battery|Storage|Turbine|Reactor)$/, "");
}
function axisName(name) {
  return isSpeculative(name) ? `${shortType(name)} [SPEC]` : shortType(name);
}
function typeList(types) {
  return (types || []).map((t) =>
    isSpeculative(t) ? `${esc(shortType(t))}&thinsp;${specChip(true)}` : esc(shortType(t))).join(", ");
}

/** Re-attach a chart click handler whenever mountChart rebuilds on theme flip. */
function bindChartClick(handle, fn) {
  const attach = () => { handle.chart.off("click"); handle.chart.on("click", fn); };
  attach();
  const onTheme = () => setTimeout(attach, 0);
  document.addEventListener("pp-theme-change", onTheme);
  return () => document.removeEventListener("pp-theme-change", onTheme);
}

// ══════════════════════════════════════════════════════════════════════
// TAB A — stress one system
// ══════════════════════════════════════════════════════════════════════

function wireTabA() {
  $("aPreset").addEventListener("change", onPresetChange);
  segWire($("aDur"), (b) => setHours(+b.dataset.h));
  buildEventButtons();
  renderTimeline();
  $("aRun").addEventListener("click", runStress);
  $("aCancel").addEventListener("click", () => sim.cancel());
  document.addEventListener("mousedown", (e) => {
    const pop = $("tlBody").querySelector(".popover");
    if (pop && !pop.contains(e.target) && !e.target.closest(".ev") && !e.target.closest(".ev-chips .chip")) {
      closePopover();
    }
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopover(); });
}

function readBuilderConfig() {
  try {
    const raw = localStorage.getItem("pp-builder-config");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && parsed.spec ? parsed : null;
  } catch (e) { return null; }
}

async function loadManifest() {
  const sel = $("aPreset");
  try {
    const m = await getManifest();
    state.manifest = m;
    const presetOpts = m.presets.map((p) =>
      `<option value="${esc(p.key)}">${esc(p.name)} — ${esc(p.scale_name)}${p.exotic ? " · SPECULATIVE" : ""}</option>`).join("");
    const builder = readBuilderConfig();
    const builderOpt = builder
      ? `<optgroup label="Custom"><option value="__builder__">From Builder — ${esc(builder.spec.name || "custom system")}</option></optgroup>`
      : "";
    sel.innerHTML = `<optgroup label="Presets">${presetOpts}</optgroup>${builderOpt}`;
    sel.value = "community";
    if (!sel.value) sel.selectedIndex = 0;
    if (builder && location.hash === "#from-builder") sel.value = "__builder__";
    onPresetChange();
  } catch (err) {
    sel.innerHTML = `<option value="">manifest unavailable</option>`;
    $("aError").innerHTML = errorHtml("Could not load data/index.json", err);
  }
}

function onPresetChange() {
  const key = $("aPreset").value;
  const presets = state.manifest ? state.manifest.presets : [];
  const p = presets.find((x) => x.key === key);
  const exotic = !!(p && p.exotic);
  $("aSpecMark").innerHTML = exotic ? specChip(true) : "";
  if (exotic) ensureSpecCallout($("aSpecNote"));
}

// ── timeline composer ──────────────────────────────────────────────────

function setHours(h) {
  state.hours = h;
  for (const ev of state.events) {
    ev.duration_hours = Math.min(ev.duration_hours, h);
    ev.start_hour = Math.min(ev.start_hour, Math.max(0, h - ev.duration_hours));
  }
  closePopover();
  renderTimeline();
}

function buildEventButtons() {
  const wrap = $("tlButtons");
  wrap.innerHTML = "";
  for (const [type, meta] of Object.entries(EV_TYPES)) {
    const b = document.createElement("button");
    b.className = "btn";
    b.innerHTML = `<i style="background:${dataColor(meta.hex)}"></i>+ ${esc(meta.label)}`;
    b.title = `add a ${meta.label} event (${meta.dur} h default)`;
    b.addEventListener("click", () => addEvent(type));
    wrap.appendChild(b);
  }
}

/** Centre of the largest event-free gap, so defaults don't pile up. */
function defaultStart(dur) {
  const H = state.hours;
  const spans = state.events
    .map((e) => [e.start_hour, e.start_hour + e.duration_hours])
    .sort((a, b) => a[0] - b[0]);
  const gaps = [];
  let cur = 0;
  for (const [s, e] of spans) {
    if (s > cur) gaps.push([cur, s]);
    cur = Math.max(cur, e);
  }
  if (cur < H) gaps.push([cur, H]);
  let best = null;
  for (const g of gaps) if (!best || g[1] - g[0] > best[1] - best[0]) best = g;
  if (!best) return Math.round(Math.max(0, (H - dur) / 2));
  const mid = (best[0] + best[1]) / 2;
  return Math.round(Math.min(Math.max(0, mid - dur / 2), Math.max(0, H - dur)));
}

function addEvent(type) {
  const meta = EV_TYPES[type];
  const dur = Math.min(meta.dur, state.hours);
  const ev = { id: state.evSeq++, type, start_hour: defaultStart(dur), duration_hours: dur };
  if (type === "source_trip" || type === "storage_fault") ev.severity = 1.0;
  if (type === "weather_crisis") ev.severity = 0.1;
  if (type === "demand_surge") ev.multiplier = 1.5;
  if (type === "simultaneous") ev.num_components = 2;
  state.events.push(ev);
  renderTimeline();
  openPopover(ev.id);
}

function removeEvent(id) {
  state.events = state.events.filter((e) => e.id !== id);
  closePopover();
  renderTimeline();
}

function evExtra(ev) {
  if (ev.type === "demand_surge") return `×${(+ev.multiplier).toFixed(1)}`;
  if (ev.type === "simultaneous") return `${ev.num_components} components`;
  if (ev.severity != null) return `sev ${(+ev.severity).toFixed(2)}`;
  return "";
}

function evTitle(ev) {
  const meta = EV_TYPES[ev.type];
  const extra = evExtra(ev);
  return `${meta.label} · ${formatHour(ev.start_hour)} · ${ev.duration_hours} h${extra ? ` · ${extra}` : ""}${ev.target ? ` · ${ev.target}` : ""}`;
}

function renderTimeline() {
  const H = state.hours;
  $("tlSpan").textContent = `0 – ${H} h · ${Math.round(H / 24)} d`;
  const track = $("tlTrack");
  const stepDays = H <= 720 ? 5 : H <= 2160 ? 15 : 60;
  let ticks = "", labels = "";
  for (let d = stepDays; d * 24 < H; d += stepDays) {
    const left = ((d * 24) / H * 100).toFixed(2);
    ticks += `<span class="tl-tick" style="left:${left}%"></span>`;
    labels += `<span style="left:${left}%">d${d}</span>`;
  }
  const evs = state.events.map((ev) => {
    const left = (ev.start_hour / H) * 100;
    const width = (ev.duration_hours / H) * 100;
    return `<span class="ev" data-id="${ev.id}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%;--ev:${dataColor(EV_TYPES[ev.type].hex)}" title="${esc(evTitle(ev))}"></span>`;
  }).join("");
  const empty = state.events.length === 0
    ? `<span class="tl-empty">no failure events — use the + buttons to add one</span>` : "";
  track.innerHTML = ticks + evs + empty + `<span class="axis">${labels}</span>`;
  track.querySelectorAll(".ev").forEach((el) =>
    el.addEventListener("click", () => openPopover(+el.dataset.id)));
  renderChips();
}

function renderChips() {
  const wrap = $("tlChips");
  if (!state.events.length) { wrap.innerHTML = ""; return; }
  wrap.innerHTML = [...state.events].sort((a, b) => a.start_hour - b.start_hour).map((ev) => {
    const meta = EV_TYPES[ev.type];
    return `<span class="chip" data-id="${ev.id}" title="${esc(evTitle(ev))}">
      <span class="dot" style="--dot:${dataColor(meta.hex)}"></span>
      ${esc(meta.label)} · d${Math.floor(ev.start_hour / 24)} · ${ev.duration_hours}h
      <span class="x" title="remove">✕</span></span>`;
  }).join("");
  wrap.querySelectorAll(".chip").forEach((el) => {
    el.querySelector(".x").addEventListener("click", (e) => {
      e.stopPropagation();
      removeEvent(+el.dataset.id);
    });
    el.addEventListener("click", () => openPopover(+el.dataset.id));
  });
}

// ── event popover ──────────────────────────────────────────────────────

function closePopover() {
  const pop = $("tlBody").querySelector(".popover");
  if (pop) pop.remove();
}

function openPopover(id) {
  closePopover();
  const ev = state.events.find((e) => e.id === id);
  if (!ev) return;
  const pop = document.createElement("div");
  pop.className = "popover";
  pop.innerHTML = popoverHtml(ev);
  $("tlBody").appendChild(pop);
  positionPopover(pop, ev);
  wirePopover(pop, ev);
}

function popoverHtml(ev) {
  const H = state.hours;
  const meta = EV_TYPES[ev.type];
  let extra = "";
  if (ev.type === "source_trip" || ev.type === "storage_fault" || ev.type === "weather_crisis") {
    const hint = ev.type === "weather_crisis" ? "weather floor — 0.1 ≈ severe storm" : "1 = full outage";
    extra += `<label class="field"><span>Severity 0–1 <span class="faint">(${hint})</span></span>
      <input type="number" data-k="severity" value="${ev.severity}" min="0" max="1" step="0.05"></label>`;
  }
  if (ev.type === "demand_surge") {
    extra += `<label class="field"><span>Demand multiplier 1.1–3</span>
      <input type="number" data-k="multiplier" value="${ev.multiplier}" min="1.1" max="3" step="0.1"></label>`;
  }
  if (ev.type === "simultaneous") {
    extra += `<label class="field"><span>Components hit 2–4</span>
      <input type="number" data-k="num_components" value="${ev.num_components}" min="2" max="4" step="1"></label>`;
  }
  if (ev.type === "source_trip" || ev.type === "storage_fault") {
    extra += `<label class="field"><span>Target component <span class="faint">(blank = random)</span></span>
      <input type="text" data-k="target" value="${esc(ev.target || "")}" placeholder="exact component name"></label>`;
  }
  return `
    <div class="kicker" style="margin-bottom:4px"><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${dataColor(meta.hex)};margin-right:6px;vertical-align:-1px"></span>${esc(meta.label)}</div>
    <div class="row">
      <label class="field" style="flex:1"><span>Start hour</span>
        <input type="number" data-k="start_hour" value="${ev.start_hour}" min="0" max="${H - 1}" step="1"></label>
      <label class="field" style="flex:1"><span>Duration (h)</span>
        <input type="number" data-k="duration_hours" value="${ev.duration_hours}" min="1" max="${H}" step="1"></label>
    </div>
    <div class="num muted" data-day style="font-size:11px">${esc(formatHour(ev.start_hour))}</div>
    ${extra}
    <div class="actions">
      <button class="btn danger" data-del>delete</button>
      <button class="btn" data-close>done</button>
    </div>`;
}

function positionPopover(pop, ev) {
  const body = $("tlBody");
  const track = $("tlTrack");
  const evEl = track.querySelector(`.ev[data-id="${ev.id}"]`);
  const left = evEl ? track.offsetLeft + evEl.offsetLeft : 0;
  const maxLeft = Math.max(0, body.clientWidth - pop.offsetWidth - 8);
  pop.style.left = `${Math.min(Math.max(0, left - 20), maxLeft)}px`;
  pop.style.top = `${track.offsetTop + track.offsetHeight + 6}px`;
}

function wirePopover(pop, ev) {
  pop.querySelectorAll("input").forEach((inp) => {
    inp.addEventListener("change", () => {
      const k = inp.dataset.k;
      if (k === "target") {
        ev.target = inp.value.trim() || undefined;
      } else {
        let v = parseFloat(inp.value);
        if (!isFinite(v)) return;
        const min = parseFloat(inp.min), max = parseFloat(inp.max);
        if (isFinite(min)) v = Math.max(min, v);
        if (isFinite(max)) v = Math.min(max, v);
        if (k === "num_components" || k === "start_hour" || k === "duration_hours") v = Math.round(v);
        ev[k] = v;
        // keep the event inside the horizon
        ev.start_hour = Math.min(ev.start_hour, Math.max(0, state.hours - ev.duration_hours));
        pop.querySelectorAll("input[data-k='start_hour'],input[data-k='duration_hours']")
          .forEach((i) => { i.value = ev[i.dataset.k]; });
        const day = pop.querySelector("[data-day]");
        if (day) day.textContent = formatHour(ev.start_hour);
      }
      renderTimeline(); // popover lives outside the track; re-render keeps it open
    });
  });
  pop.querySelector("[data-del]").addEventListener("click", () => removeEvent(ev.id));
  pop.querySelector("[data-close]").addEventListener("click", closePopover);
}

/** Events in webapi shape, only the fields each type understands. */
function exportEvents() {
  return [...state.events].sort((a, b) => a.start_hour - b.start_hour).map((ev) => {
    const out = { type: ev.type, start_hour: ev.start_hour, duration_hours: ev.duration_hours };
    if (ev.type === "source_trip" || ev.type === "storage_fault") {
      out.severity = ev.severity != null ? ev.severity : 1.0;
      if (ev.target) out.target = ev.target;
    }
    if (ev.type === "weather_crisis") out.severity = ev.severity != null ? ev.severity : 0.1;
    if (ev.type === "demand_surge") out.multiplier = ev.multiplier != null ? ev.multiplier : 1.5;
    if (ev.type === "simultaneous") out.num_components = ev.num_components != null ? ev.num_components : 2;
    return out;
  });
}

// ── paired run ─────────────────────────────────────────────────────────

async function runStress() {
  if (!wasmSupported()) return;
  $("aError").innerHTML = "";
  if (sim.busy) {
    $("aError").innerHTML = `<div class="callout error">The engine is busy — wait for the current run to finish (or cancel it).</div>`;
    return;
  }
  const sel = $("aPreset").value;
  let base = null;
  if (sel === "__builder__") {
    const b = readBuilderConfig();
    if (!b) {
      $("aError").innerHTML = `<div class="callout error">The Builder configuration is no longer in localStorage — rebuild it on the Builder page.</div>`;
      return;
    }
    base = { spec: b.spec };
  } else if (sel) {
    base = { preset: sel };
  } else {
    return;
  }
  const hours = state.hours;
  const seed = clampInt($("aSeed").value, 0, 2147483647, 42);
  const events = exportEvents();
  $("aRun").disabled = true;
  $("bRun").disabled = true;
  progress("a", true, "starting engine…", 0);
  try {
    sim.onStatus = (stage) => progress("a", true, STAGE_TEXT[stage] || stage, 0);
    await sim.ensureReady();
    const res = await sim.runFailure({ base, events, hours, seed }, (done, total, phase) => {
      const frac = total ? done / total : 0;
      const label = /base/i.test(phase || "") ? "baseline…"
        : /stress|fail/i.test(phase || "") ? "stressed…"
        : frac < 0.5 ? "baseline…" : "stressed…";
      progress("a", true, `${label} ${Math.round(frac * 100)}%`, frac);
    });
    renderStressResults(res.baseline, res.failed, events, hours);
  } catch (err) {
    if (err && err.name === "CancelledError") {
      $("aError").innerHTML = `<p class="muted" style="margin:6px 2px">Run cancelled.</p>`;
    } else {
      $("aError").innerHTML = errorHtml("Paired run failed", err);
    }
  } finally {
    $("aRun").disabled = false;
    $("bRun").disabled = false;
    progress("a", false);
  }
}

const PAIR_METRICS = [
  { label: "LOLP", get: (r) => r.lolp, higherWorse: true,
    fmt: (v) => v == null ? "—" : formatPct(v, 1),
    deltaFmt: (d) => `${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)} pp` },
  { label: "LOLE", get: (r) => r.lole_hours, higherWorse: true,
    fmt: (v) => v == null ? "—" : `${Math.round(v)} h`,
    deltaFmt: (d) => `${d >= 0 ? "+" : "−"}${Math.abs(Math.round(d))} h` },
  { label: "ENS", get: (r) => r.ens_kwh, higherWorse: true,
    fmt: (v) => v == null ? "—" : formatKwh(v),
    deltaFmt: (d) => `${d >= 0 ? "+" : "−"}${formatKwh(Math.abs(d))}` },
  { label: "MAX RECOVERY", get: (r) => r.recovery_time_max, higherWorse: true,
    fmt: (v) => v == null ? "—" : `${Math.round(v)} h`,
    deltaFmt: (d) => `${d >= 0 ? "+" : "−"}${Math.abs(Math.round(d))} h` },
  { label: "MIN RESERVE MARGIN", get: (r) => r.min_reserve_margin, higherWorse: false,
    fmt: (v) => v == null ? "—" : formatPct(v, 1),
    deltaFmt: (d) => `${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)} pp` },
];

function pairKpiHtml(m, rb, rf) {
  const b = m.get(rb), f = m.get(rf);
  const d = (f == null ? 0 : f) - (b == null ? 0 : b);
  const same = m.fmt(b) === m.fmt(f);
  let arrow = "·", color = "var(--ink-faint)";
  if (!same) {
    const worse = m.higherWorse ? d > 0 : d < 0;
    arrow = d > 0 ? "▲" : "▼";
    color = dataColor(worse ? PALETTE.demand : PALETTE.charge);
  }
  return `<div class="kpi" style="--rule:${color}">
    <div class="kicker">${m.label}</div>
    <div class="kpi-value pair"><span class="num">${m.fmt(b)}</span><span class="pair-arrow">→</span><span class="num">${m.fmt(f)}</span></div>
    <div class="pair-delta" style="color:${color}">${arrow} ${same ? "no change" : m.deltaFmt(d)}</div>
  </div>`;
}

function lcoeFootnote(baseline, failed, hours) {
  const fmt = (doc) => {
    const v = doc && doc.metrics ? doc.metrics.estimated_lcoe_usd_kwh : null;
    return v == null ? `<span class="num">∞</span>` : `<span class="num">$${v.toFixed(3)}</span><span class="unit">/kWh</span>`;
  };
  const note = hours < 8760
    ? ` — annualized estimate — short runs overstate LCOE ~${Math.round(8760 / hours)}×` : "";
  return `<p class="foot-note">LCOE ${fmt(baseline)} baseline · ${fmt(failed)} stressed${note}.</p>`;
}

function renderStressResults(baseline, failed, events, hours) {
  state.stressHandles.forEach((h) => { if (h && h.dispose) h.dispose(); });
  state.stressHandles = [];
  const root = $("aResults");
  const rb = baseline.resilience || {};
  const rf = failed.resilience || {};
  const desc = failed.failures && failed.failures.description ? failed.failures.description : "";
  root.innerHTML = `
    <div class="kpi-strip" style="margin:4px 0 2px">${PAIR_METRICS.map((m) => pairKpiHtml(m, rb, rf)).join("")}</div>
    ${lcoeFootnote(baseline, failed, hours)}
    <div class="grid" style="margin-top:14px">
      <section class="panel col-12" id="pStressPB">
        <div class="panel-head">
          <span class="kicker">Stressed dispatch</span>
          <span class="head-meta" title="${esc(desc)}">${esc(desc)}</span>
        </div>
        <div class="panel-body"><div class="chart tall" id="chStressPB"></div></div>
        <div class="fig-caption"><span class="fig-no">FIG. 2</span>Hourly power balance under the injected
          failures — shaded bands mark the failure windows; the red area is unmet demand.</div>
      </section>
      <section class="panel col-12" id="pBasePB">
        <div class="panel-head">
          <span class="kicker">Baseline dispatch</span>
          <span class="head-meta">identical weather, no failures</span>
        </div>
        <div class="panel-body"><div class="chart short" id="chBasePB"></div></div>
        <div class="fig-caption"><span class="fig-no">FIG. 3</span>The same system and weather draw with no
          failures injected — the reference for every delta above.</div>
      </section>
      <section class="panel col-12" id="pStressKpis">
        <div class="panel-head"><span class="kicker">Stressed run — full metrics</span></div>
        <div class="panel-body" id="aFullKpis"></div>
      </section>
    </div>`;
  applyRules();
  state.stressHandles.push(renderPowerBalance($("chStressPB"), failed, { events }));
  state.stressHandles.push(renderPowerBalance($("chBasePB"), baseline, { height: 220 }));
  kpiStrip($("aFullKpis"), failed);
  root.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ══════════════════════════════════════════════════════════════════════
// TAB B — batch scenario lab
// ══════════════════════════════════════════════════════════════════════

function wireTabB() {
  segWire($("bTier"), (b) => {
    state.tier = b.dataset.tier;
    if (state.tier !== "conventional") ensureSpecCallout($("bSpecNote"));
  });
  segWire($("bDays"), (b) => { state.days = +b.dataset.d; updateEstimate(); });
  $("bN").addEventListener("input", updateEstimate);
  $("bM").addEventListener("input", updateEstimate);
  $("bSample").addEventListener("click", loadSample);
  $("bRun").addEventListener("click", runBatch);
  $("bCancel").addEventListener("click", () => sim.cancel());
  $("bTier").querySelectorAll("button").forEach((b) => {
    if (b.dataset.tier !== "conventional") b.insertAdjacentHTML("beforeend", `&nbsp;${specChip(true)}`);
  });
  updateEstimate();
}

function updateEstimate() {
  const n = clampInt($("bN").value, 1, 20, 5);
  const m = clampInt($("bM").value, 1, 5, 2);
  const secs = n * (m + 1) * (state.days / 30) * 0.5;
  $("bEst").textContent = secs < 90 ? `≈ ${Math.round(secs)} s` : `≈ ${(secs / 60).toFixed(1)} min`;
}

async function loadSample() {
  $("bError").innerHTML = "";
  try {
    const doc = await getResilienceSample();
    const p = doc.params || {};
    renderBatch(doc,
      `pre-baked sample — scale ${p.scale}, tier ${p.tier}, N=${p.n_configs}, M=${p.n_failures}, seed ${p.seed}, ${p.hours} h`);
  } catch (err) {
    $("bMeta").textContent = "";
    $("bError").innerHTML = errorHtml("Could not load the pre-baked sample", err);
    for (const id of CHART_IDS) {
      $(id).innerHTML = `<div class="empty-state" style="padding:24px 12px">no batch loaded</div>`;
    }
    const tbody = $("bTable").querySelector("tbody");
    tbody.innerHTML = "";
    $("bTable").querySelector("thead").innerHTML = "";
  }
}

async function runBatch() {
  if (!wasmSupported()) return;
  $("bError").innerHTML = "";
  if (sim.busy) {
    $("bError").innerHTML = `<div class="callout error">The engine is busy — wait for the current run to finish (or cancel it).</div>`;
    return;
  }
  const n = clampInt($("bN").value, 1, 20, 5);
  const m = clampInt($("bM").value, 1, 5, 2);
  const seed = clampInt($("bSeed").value, 0, 2147483647, 42);
  const scale = $("bScale").value;
  const hours = state.days * 24;
  $("bRun").disabled = true;
  $("aRun").disabled = true;
  progress("b", true, "starting engine…", 0);
  try {
    sim.onStatus = (stage) => progress("b", true, STAGE_TEXT[stage] || stage, 0);
    await sim.ensureReady();
    const doc = await sim.runBatch(
      { scale, tier: state.tier, seed, nConfigs: n, nFailures: m, hours, includeBaseline: true },
      (done, total) => progress("b", true, `run ${done}/${total}`, total ? done / total : 0));
    renderBatch(doc, `live batch — scale ${scale}, tier ${state.tier}, N=${n}, M=${m}, seed ${seed}, ${hours} h`);
  } catch (err) {
    if (err && err.name === "CancelledError") {
      $("bError").innerHTML = `<p class="muted" style="margin:6px 2px">Batch cancelled.</p>`;
    } else {
      $("bError").innerHTML = errorHtml("Batch failed", err);
    }
  } finally {
    $("bRun").disabled = false;
    $("aRun").disabled = false;
    progress("b", false);
  }
}

// ── batch rendering ────────────────────────────────────────────────────

function renderBatch(doc, label) {
  const B = state.batch;
  B.handles.forEach((h) => h.dispose());
  B.cleanups.forEach((f) => f());
  B.handles = [];
  B.cleanups = [];
  B.doc = doc;
  B.filter = null;
  const a = doc.analysis || {};
  const fmtLolp = (v) => v == null ? "—" : formatPct(v, 1);
  $("bMeta").innerHTML = `${esc(label)} · <span class="num">${doc.runs.length}</span> runs
    · avg LOLP baseline <span class="num">${fmtLolp(a.baseline_avg_lolp)}</span>
    → stressed <span class="num">${fmtLolp(a.stressed_avg_lolp)}</span>`;
  renderHists(doc);
  renderImpact(doc);
  renderFtype(doc);
  renderPaired(doc);
  renderHeatmap(doc);
  renderTable();
}

function mountInto(elId, build) {
  const el = $(elId);
  el.innerHTML = ""; // clear skeleton / previous chart DOM
  const h = mountChart(echarts, el, build);
  h.render();
  state.batch.handles.push(h);
  return h;
}

const HIST_DEFS = [
  { el: "chHistLolp", metric: "lolp", title: "LOLP", statKey: "lolp",
    color: () => dataColor(PALETTE.discharge),
    get: (r) => r.resilience ? r.resilience.lolp : null,
    fmt: (v) => formatPct(v, 0) },
  { el: "chHistEns", metric: "ens", title: "ENS", statKey: "ens_kwh",
    color: () => dataColor(PALETTE.demand),
    get: (r) => r.resilience ? r.resilience.ens_kwh : null,
    fmt: (v) => formatKwh(v) },
  { el: "chHistRec", metric: "rec", title: "Max recovery", statKey: "recovery_time",
    color: () => dataColor(PALETTE.charge),
    get: (r) => r.resilience ? r.resilience.recovery_time_max : null,
    fmt: (v) => `${Math.round(v)}h` },
];

function renderHists(doc) {
  for (const def of HIST_DEFS) {
    const values = doc.runs.map(def.get).filter((v) => v != null && isFinite(v));
    const { bins, counts } = histogram(values, 14);
    if (!bins.length) {
      $(def.el).innerHTML = `<div class="empty-state" style="padding:24px 12px">no finite values</div>`;
      continue;
    }
    const stat = (doc.analysis || {})[def.statKey];
    const mean = stat && stat.mean != null
      ? stat.mean
      : values.reduce((s, x) => s + x, 0) / values.length;
    const lo = bins[0][0];
    const w = bins[0][1] - bins[0][0] || 1;
    const h = mountInto(def.el, () => histOption(def, bins, counts, mean, lo, w));
    state.batch.cleanups.push(bindChartClick(h, (p) => {
      if (p.componentType !== "series") return;
      const bin = bins[p.dataIndex];
      if (!bin) return;
      const last = p.dataIndex === bins.length - 1;
      setFilter({
        get: def.get, lo: bin[0], hi: bin[1], last,
        label: `${def.title} ∈ [${def.fmt(bin[0])}, ${def.fmt(bin[1])}${last ? "]" : ")"}`,
      });
    }));
  }
}

function histOption(def, bins, counts, mean, lo, w) {
  const labels = bins.map(([a, b]) => `${def.fmt(a)}–${def.fmt(b)}`);
  const meanIdx = Math.max(-0.5, Math.min(bins.length - 0.5, (mean - lo) / w - 0.5));
  return {
    animationDuration: 300,
    grid: { top: 26, left: 40, right: 12, bottom: 44 },
    tooltip: {
      trigger: "axis", axisPointer: { type: "shadow" },
      formatter: (ps) => {
        const p = ps[0];
        return `${labels[p.dataIndex]}<br>${p.value} run${p.value === 1 ? "" : "s"} — click to filter table`;
      },
    },
    xAxis: { type: "category", data: labels, axisLabel: { rotate: 28, fontSize: 9 } },
    yAxis: { type: "value", minInterval: 1, name: "runs", nameTextStyle: { fontSize: 9 } },
    series: [{
      type: "bar",
      data: counts,
      barCategoryGap: "12%",
      cursor: "pointer",
      itemStyle: { color: def.color() },
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { type: "dashed", width: 1, color: isDark() ? "#E8EAED" : "#2D3436" },
        label: { formatter: `mean ${def.fmt(mean)}`, fontFamily: FONT_MONO, fontSize: 9, position: "insideEndTop" },
        data: [{ xAxis: meanIdx }],
      },
    }],
  };
}

function renderImpact(doc) {
  const imp = (doc.analysis || {}).component_impact || {};
  const entries = Object.entries(imp)
    .map(([name, v]) => Object.assign({ name }, v))
    .sort((x, y) => Math.abs(y.resilience_impact) - Math.abs(x.resilience_impact))
    .slice(0, 8)
    .sort((x, y) => x.resilience_impact - y.resilience_impact);
  if (!entries.length) {
    $("chImpact").innerHTML = `<div class="empty-state" style="padding:24px 12px">no component-impact analysis in this batch</div>`;
    return;
  }
  mountInto("chImpact", () => ({
    animationDuration: 300,
    grid: { top: 12, left: 124, right: 44, bottom: 32 },
    tooltip: {
      trigger: "item",
      formatter: (p) => {
        const e = entries[p.dataIndex];
        const spec = isSpeculative(e.name) ? `<br><i>speculative technology</i>` : "";
        return `<b>${axisName(e.name)}</b><br>avg LOLP with: ${e.avg_lolp_with.toFixed(3)}
          <br>avg LOLP without: ${e.avg_lolp_without.toFixed(3)}<br>present in ${e.count} runs${spec}`;
      },
    },
    xAxis: {
      type: "value",
      axisLabel: { formatter: (v) => `${(v * 100).toFixed(0)}pp` },
      name: "Δ LOLP when present", nameLocation: "middle", nameGap: 24, nameTextStyle: { fontSize: 9 },
    },
    yAxis: { type: "category", data: entries.map((e) => axisName(e.name)), axisLabel: { fontSize: 9.5 } },
    series: [{
      type: "bar",
      barCategoryGap: "30%",
      data: entries.map((e) => ({
        value: e.resilience_impact,
        itemStyle: { color: dataColor(e.resilience_impact >= 0 ? PALETTE.charge : PALETTE.demand) },
        label: { position: e.resilience_impact >= 0 ? "right" : "left" },
      })),
      label: {
        show: true, fontFamily: FONT_MONO, fontSize: 9,
        formatter: (p) => `${p.value >= 0 ? "+" : ""}${(p.value * 100).toFixed(1)}pp`,
      },
    }],
  }));
}

function renderFtype(doc) {
  const fi = (doc.analysis || {}).failure_type_impact || {};
  const entries = Object.entries(fi)
    .map(([type, v]) => Object.assign({ type }, v))
    .sort((a, b) => b.avg_ens - a.avg_ens);
  if (!entries.length) {
    $("chFtype").innerHTML = `<div class="empty-state" style="padding:24px 12px">no failure-type analysis in this batch</div>`;
    return;
  }
  mountInto("chFtype", () => ({
    animationDuration: 300,
    grid: { top: 26, left: 70, right: 12, bottom: 48 },
    tooltip: {
      trigger: "axis", axisPointer: { type: "shadow" },
      formatter: (ps) => {
        const e = entries[ps[0].dataIndex];
        return `<b>${e.type.replace(/_/g, " ")}</b><br>avg ENS: ${formatKwh(e.avg_ens)}
          <br>avg LOLP: ${e.avg_lolp.toFixed(3)}<br>avg recovery: ${Math.round(e.avg_recovery)} h
          <br>${e.count} stressed run${e.count === 1 ? "" : "s"}`;
      },
    },
    xAxis: { type: "category", data: entries.map((e) => e.type.replace(/_/g, " ")), axisLabel: { rotate: 24, fontSize: 9 } },
    yAxis: { type: "value", axisLabel: { formatter: (v) => formatKwh(v) }, name: "avg ENS", nameTextStyle: { fontSize: 9 } },
    series: [{
      type: "bar",
      data: entries.map((e) => e.avg_ens),
      barCategoryGap: "25%",
      itemStyle: { color: dataColor(PALETTE.grid_import) },
    }],
  }));
}

function renderPaired(doc) {
  const byCfg = new Map();
  for (const r of doc.runs) {
    let c = byCfg.get(r.config_id);
    if (!c) {
      c = { name: r.config_name, sources: r.source_types || [], base: [], stress: [] };
      byCfg.set(r.config_id, c);
    }
    const lolp = r.resilience ? r.resilience.lolp : null;
    if (lolp == null) continue;
    (r.failure_id === -1 ? c.base : c.stress).push(lolp);
  }
  const ids = [...byCfg.keys()].sort((a, b) => a - b);
  if (!ids.length) {
    $("chPaired").innerHTML = `<div class="empty-state" style="padding:24px 12px">no runs in this batch</div>`;
    return;
  }
  const mean = (xs) => xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
  const cats = ids.map((i) => `C${i + 1}`);
  const baseVals = ids.map((i) => mean(byCfg.get(i).base));
  const stressVals = ids.map((i) => mean(byCfg.get(i).stress));
  mountInto("chPaired", () => ({
    animationDuration: 300,
    grid: { top: 32, left: 46, right: 12, bottom: 26 },
    legend: { top: 0, data: ["baseline", "stressed mean"] },
    tooltip: {
      trigger: "axis", axisPointer: { type: "shadow" },
      formatter: (ps) => {
        const i = ps[0].dataIndex;
        const c = byCfg.get(ids[i]);
        const rows = ps.map((p) => `${p.marker} ${p.seriesName}: ${p.value == null ? "—" : formatPct(p.value, 1)}`).join("<br>");
        return `<b>${esc(c.name)}</b><br><span style="font-size:10px">${c.sources.map(axisName).join(", ")}</span><br>${rows}`;
      },
    },
    xAxis: { type: "category", data: cats },
    yAxis: { type: "value", axisLabel: { formatter: (v) => formatPct(v, 0) }, name: "LOLP", nameTextStyle: { fontSize: 9 } },
    series: [
      { name: "baseline", type: "bar", data: baseVals, itemStyle: { color: dataColor(PALETTE.discharge) } },
      { name: "stressed mean", type: "bar", data: stressVals, itemStyle: { color: dataColor(PALETTE.demand) } },
    ],
  }));
}

function renderHeatmap(doc) {
  const stressed = doc.runs.filter((r) => r.failure_id !== -1 && r.resilience);
  const srcs = [...new Set(stressed.flatMap((r) => r.source_types || []))].sort();
  const stos = [...new Set(stressed.flatMap((r) => r.storage_types || []))].sort();
  const note = $("heatNote");
  const el = $("chHeat");
  if (srcs.length < 2 || stos.length < 2) {
    el.innerHTML = "";
    el.style.display = "none";
    note.innerHTML = `<div class="callout">Not enough component diversity for a cross-table — this batch
      spans ${srcs.length} source type${srcs.length === 1 ? "" : "s"} × ${stos.length} storage
      type${stos.length === 1 ? "" : "s"}; at least 2 × 2 are needed.</div>`;
    return;
  }
  note.innerHTML = "";
  el.style.display = "";
  const cells = [], naCells = [];
  let maxV = 0;
  stos.forEach((sto, x) => srcs.forEach((src, y) => {
    const hits = stressed.filter((r) =>
      r.source_types.includes(src) && r.storage_types.includes(sto));
    if (hits.length) {
      const v = hits.reduce((s, r) => s + r.resilience.lolp, 0) / hits.length;
      maxV = Math.max(maxV, v);
      cells.push({ x, y, v, n: hits.length });
    } else {
      naCells.push([x, y]);
    }
  }));
  mountInto("chHeat", () => heatOption(srcs, stos, cells, naCells, maxV));
}

function heatOption(srcs, stos, cells, naCells, maxV) {
  const mx = maxV || 1e-9;
  return {
    animation: false,
    grid: { top: 8, left: 110, right: 16, bottom: 92 },
    tooltip: {
      formatter: (p) => {
        if (p.seriesIndex === 1) return "no stressed runs pair these components";
        const c = cells[p.dataIndex];
        return `<b>${axisName(srcs[c.y])} × ${axisName(stos[c.x])}</b>
          <br>mean LOLP ${c.v.toFixed(3)} over ${c.n} stressed run${c.n === 1 ? "" : "s"}`;
      },
    },
    xAxis: { type: "category", data: stos.map(axisName), axisLabel: { rotate: 32, fontSize: 9 }, splitArea: { show: true } },
    yAxis: { type: "category", data: srcs.map(axisName), axisLabel: { fontSize: 9 }, splitArea: { show: true } },
    visualMap: {
      min: 0, max: mx, seriesIndex: 0, calculable: false,
      orient: "horizontal", left: "center", bottom: 0,
      inRange: { color: HEAT_RAMP },
      text: ["high LOLP", "low"], textStyle: { fontSize: 9 },
      formatter: (v) => (+v).toFixed(2),
    },
    series: [
      {
        type: "heatmap",
        data: cells.map((c) => ({
          value: [c.x, c.y, c.v],
          label: { color: c.v / mx > 0.55 ? "#1B2430" : "#F2F2F0" },
        })),
        label: { show: true, formatter: (p) => p.value[2].toFixed(3), fontFamily: FONT_MONO, fontSize: 9 },
        emphasis: { itemStyle: { borderColor: PALETTE.accent, borderWidth: 1 } },
      },
      {
        type: "heatmap",
        data: naCells.map(([x, y]) => ({ value: [x, y, 0] })),
        itemStyle: {
          color: isDark() ? "#262B33" : "#ECECEA",
          decal: {
            symbol: "rect", symbolSize: 1, rotation: Math.PI / 4,
            dashArrayX: [1, 0], dashArrayY: [2, 5],
            color: "rgba(128,128,128,0.4)",
          },
        },
        label: { show: true, formatter: "n/a", fontFamily: FONT_MONO, fontSize: 9, color: isDark() ? "#5F6368" : "#B2BEC3" },
      },
    ],
  };
}

// ── run table ──────────────────────────────────────────────────────────

const COLS = [
  { key: "cfg", label: "Config", get: (r) => r.config_id,
    html: (r) => `<td class="num" title="${esc(r.config_name)}">C${r.config_id + 1}</td>` },
  { key: "sources", label: "Sources", get: (r) => (r.source_types || []).join(", "),
    html: (r) => `<td>${typeList(r.source_types)}</td>` },
  { key: "storage", label: "Storage", get: (r) => (r.storage_types || []).join(", "),
    html: (r) => `<td>${typeList(r.storage_types)}</td>` },
  { key: "failure", label: "Failure scenario", get: (r) => r.failure_desc,
    html: (r) => `<td title="${esc(r.failure_desc)}">${esc(truncate(r.failure_desc, 56))}</td>` },
  { key: "lolp", label: "LOLP", get: (r) => r.resilience ? r.resilience.lolp : 0,
    html: (r) => `<td class="num">${(r.resilience ? r.resilience.lolp : 0).toFixed(3)}</td>` },
  { key: "ens", label: "ENS", get: (r) => r.resilience ? r.resilience.ens_kwh : 0,
    html: (r) => `<td class="num">${formatKwh(r.resilience ? r.resilience.ens_kwh : 0)}</td>` },
  { key: "rec", label: "Recovery max", get: (r) => r.resilience ? r.resilience.recovery_time_max : 0,
    html: (r) => `<td class="num">${Math.round(r.resilience ? r.resilience.recovery_time_max : 0)} h</td>` },
];

function setFilter(f) {
  state.batch.filter = f;
  renderTable();
  $("tblPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderTable() {
  const B = state.batch;
  const thead = $("bTable").querySelector("thead");
  const tbody = $("bTable").querySelector("tbody");
  if (!B.doc) { thead.innerHTML = ""; tbody.innerHTML = ""; renderFilterBar(0, 0); return; }
  thead.innerHTML = `<tr>${COLS.map((c) =>
    `<th data-key="${c.key}">${c.label}${B.sortKey === c.key ? `<span class="dir">${B.sortDir > 0 ? "▲" : "▼"}</span>` : ""}</th>`).join("")}</tr>`;
  thead.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.key;
    if (B.sortKey === k) B.sortDir *= -1;
    else { B.sortKey = k; B.sortDir = 1; }
    renderTable();
  }));
  let rows = B.doc.runs.slice();
  if (B.filter) {
    const { get, lo, hi, last } = B.filter;
    rows = rows.filter((r) => {
      const v = get(r);
      return v != null && v >= lo && (last ? v <= hi : v < hi);
    });
  }
  const col = COLS.find((c) => c.key === B.sortKey);
  if (col) {
    rows.sort((x, y) => {
      const a = col.get(x), b = col.get(y);
      const cmp = typeof a === "string" || typeof b === "string"
        ? String(a).localeCompare(String(b))
        : (a - b);
      return (cmp * B.sortDir) || (x.config_id - y.config_id) || (x.failure_id - y.failure_id);
    });
  }
  tbody.innerHTML = rows.length
    ? rows.map((r) => `<tr>${COLS.map((c) => c.html(r)).join("")}</tr>`).join("")
    : `<tr><td colspan="${COLS.length}" class="muted" style="text-align:center;padding:24px">no runs fall in the selected bin</td></tr>`;
  renderFilterBar(rows.length, B.doc.runs.length);
}

function renderFilterBar(shown, total) {
  const bar = $("bFilterBar");
  const B = state.batch;
  if (!B.doc) { bar.innerHTML = ""; return; }
  if (!B.filter) {
    bar.innerHTML = `click a histogram bin to filter · <span class="num">${total}</span> runs`;
    return;
  }
  bar.innerHTML = `<button class="chip on" title="clear filter">${esc(B.filter.label)} · ${shown}/${total} ✕</button>`;
  bar.querySelector("button").addEventListener("click", () => {
    B.filter = null;
    renderTable();
  });
}
