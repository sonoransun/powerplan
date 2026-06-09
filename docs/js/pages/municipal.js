/**
 * pages/municipal.js — growth projections.
 *
 * Five municipal archetypes with pre-baked full-year (8 760 h/plan-year)
 * fossil-to-renewable transition projections. Controls at their defaults
 * render the pre-baked doc instantly; any deviation (climate override,
 * year subset, fast fidelity) requires a live in-browser run through the
 * Pyodide worker (sim-client.js).
 */

import * as echarts from "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.esm.min.js";
import {
  PALETTE, dataColor, formatKw, formatKwh, formatUsd, formatPct,
  mountChart, FONT_MONO, isDark,
} from "../theme.js";
import { getManifest, getProjection } from "../data.js";
import { sim, wasmSupported } from "../pyodide/sim-client.js";

// ── constants ─────────────────────────────────────────────────────────

const BASE_YEAR = 2025;
const ALL_OFFSETS = [0, 5, 10, 15, 20, 25];
const DEFAULT_PROFILE = "college_town";
const FULL_YEAR_HOURS = 8760;
const FAST_HOURS = 720;

const STAGE_TEXT = {
  "loading-pyodide": "loading Pyodide runtime…",
  "loading-numpy": "loading NumPy…",
  "fetching-package": "fetching powerplan package…",
  "importing": "importing simulation engine…",
  "ready": "engine ready",
};

// ── formatters (drawer + labels) ──────────────────────────────────────

const fmtLcoe = (v) => (v == null ? "∞" : `$${v.toFixed(4)}/kWh`);
const fmtTonnes = (v) => {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M t`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k t`;
  return `${Math.round(v)} t`;
};
const fmtRatio = (v) => (v == null ? "—" : `${v.toFixed(2)}×`);
const fmtHours = (v) => (v == null ? "—" : `${Math.round(v).toLocaleString("en-US")} h`);
const compactNum = (v) => {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
  return String(v);
};
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const inkMid = () => (isDark() ? "#9AA0A6" : "#636E72");

// Year-row plan scalars (left column of the inspector).
const PLAN_ROWS = [
  ["fossil_capacity_kw", "fossil capacity", formatKw],
  ["renewable_capacity_kw", "renewable capacity", formatKw],
  ["storage_capacity_kwh", "storage capacity", formatKwh],
  ["peak_demand_kw", "planned peak load", formatKw],
  ["renewable_target_pct", "renewable target", (v) => formatPct(v, 0)],
  ["renewable_actual_pct", "renewable actual", (v) => formatPct(v)],
  ["total_capex", "cumulative capex", formatUsd],
  ["lcoe", "LCOE", fmtLcoe],
  ["emissions_tonnes", "CO₂ emissions", fmtTonnes],
];

// metrics{} scalars (data contract: scalars only).
const METRIC_ROWS = [
  ["avg_renewable_fraction", "renewable fraction", (v) => formatPct(v)],
  ["avg_system_efficiency", "system efficiency", (v) => formatPct(v)],
  ["self_sufficiency", "self-sufficiency", (v) => formatPct(v)],
  ["curtailment_fraction", "curtailment fraction", (v) => formatPct(v)],
  ["generation_to_demand_ratio", "generation / demand", fmtRatio],
  ["estimated_lcoe_usd_kwh", "LCOE (estimated)", fmtLcoe],
  ["peak_demand_kw", "peak demand (simulated)", formatKw],
  ["peak_generation_kw", "peak generation", formatKw],
  ["total_demand_kwh", "total demand", formatKwh],
  ["total_generation_kwh", "total generation", formatKwh],
  ["total_grid_import_kwh", "grid import", formatKwh],
  ["total_grid_export_kwh", "grid export", formatKwh],
  ["total_curtailment_kwh", "curtailed energy", formatKwh],
  ["total_controller_losses_kwh", "controller losses", formatKwh],
  ["total_capex_usd", "capex", formatUsd],
  ["simulation_hours", "simulated hours", fmtHours],
];

// ── module state ──────────────────────────────────────────────────────

const state = {
  profile: DEFAULT_PROFILE,
  climate: "",                       // "" = profile default
  years: new Set(ALL_OFFSETS),       // selected year offsets
  fidelity: "full",                  // "fast" | "full"
};

let prebaked = new Map();            // key → projection doc
let profileOrder = [];
let currentDoc = null;               // doc currently driving the figures
let renderedSig = "";                // control signature of the rendered doc
let renderedKind = "prebaked";       // "prebaked" | "live"
let running = false;
let charts = null;                   // {capacity, lcoe, target, peak, self, emissions}
let drawerDonut = null;
let statusTimer = null;
let els = {};

const controlSig = () => JSON.stringify(
  [state.profile, state.climate, [...state.years].sort((a, b) => a - b), state.fidelity]);
const defaultSig = () => JSON.stringify([state.profile, "", ALL_OFFSETS, "full"]);

// ── init ──────────────────────────────────────────────────────────────

export async function init() {
  els = {
    cards: byId("profile-cards"),
    climateSel: byId("climate-sel"),
    yearChips: byId("year-chips"),
    fidSeg: byId("fid-seg"),
    runBtn: byId("run-btn"),
    runNote: byId("run-note"),
    wasmNote: byId("wasm-note"),
    errorBox: byId("error-box"),
    header: byId("profile-header"),
    source: byId("run-source"),
    skeleton: byId("charts-skeleton"),
    grid: byId("charts-grid"),
    capNote: byId("cap-note"),
    lcoeNote: byId("lcoe-note"),
    emisBadge: byId("emis-badge"),
    drawer: byId("year-drawer"),
    drawerTitle: byId("drawer-title"),
    drawerSub: byId("drawer-sub"),
    drawerPlan: byId("drawer-plan"),
    drawerMetrics: byId("drawer-metrics"),
    drawerClose: byId("drawer-close"),
    status: byId("sim-status"),
    statusText: byId("sim-status-text"),
    statusBar: byId("sim-status-bar"),
    cancel: byId("sim-cancel"),
  };

  try {
    const manifest = await getManifest();
    profileOrder = (manifest.projections ?? []).map((p) => p.key);
    const docs = await Promise.all(profileOrder.map((k) => getProjection(k)));
    prebaked = new Map(profileOrder.map((k, i) => [k, docs[i]]));
    if (!prebaked.size) throw new Error("manifest lists no projections");
  } catch (err) {
    showFatal(err);
    return;
  }

  buildCards();
  buildYearChips();
  wireControls();
  wireDrawer();

  if (!wasmSupported()) {
    els.wasmNote.textContent = "WebAssembly unavailable — live overrides disabled";
  }
  sim.onStatus = (stage) => {
    if (running) statusShow(STAGE_TEXT[stage] ?? stage, null);
  };

  // reveal the figure grid, then mount (charts need a laid-out container)
  els.skeleton.remove();
  els.grid.style.display = "";
  mountAllCharts();
  applyRules();
  document.addEventListener("pp-theme-change", applyRules);

  const fromHash = location.hash.slice(1);
  const initial = prebaked.has(fromHash) ? fromHash : DEFAULT_PROFILE;
  if (initial !== fromHash) history.replaceState(null, "", `#${initial}`);
  selectProfile(initial);
  window.addEventListener("hashchange", () => {
    const k = location.hash.slice(1);
    if (prebaked.has(k) && k !== state.profile) selectProfile(k);
  });
}

const byId = (id) => document.getElementById(id);

// ── profile cards ─────────────────────────────────────────────────────

function buildCards() {
  els.cards.innerHTML = profileOrder.map((key) => {
    const p = prebaked.get(key).profile;
    return `
      <button type="button" class="preset-card" data-key="${key}">
        <div class="name">${esc(p.name)}</div>
        <div class="peak">pop ${p.population.toLocaleString("en-US")} · ${formatKw(p.peak_load_kw)} peak</div>
        <div class="card-sub">${esc(p.climate.name)}</div>
        <div class="card-sub">target <span class="num">${Math.round(p.renewable_target_pct * 100)}%</span>
          by <span class="num">${p.renewable_target_year}</span> ·
          net-zero <span class="num">${p.net_zero_year}</span></div>
      </button>`;
  }).join("");
  els.cards.querySelectorAll(".preset-card").forEach((card) => {
    card.addEventListener("click", () => {
      const k = card.dataset.key;
      if (k === state.profile) return;
      if (location.hash === `#${k}`) selectProfile(k);
      else location.hash = k;          // → hashchange → selectProfile
    });
  });
}

function syncCards() {
  els.cards.querySelectorAll(".preset-card").forEach((card) => {
    card.classList.toggle("on", card.dataset.key === state.profile);
  });
}

function selectProfile(key) {
  state.profile = key;
  syncCards();
  renderedKind = "prebaked";
  renderedSig = defaultSig();          // the pre-baked doc ≡ default controls
  renderAll(prebaked.get(key));
  updateRunUI();
}

// ── controls ──────────────────────────────────────────────────────────

function buildYearChips() {
  els.yearChips.innerHTML = ALL_OFFSETS.map((off) =>
    `<button type="button" class="chip on" data-off="${off}" title="${BASE_YEAR + off}">
       <span class="num">+${off}</span></button>`).join("");
  els.yearChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const off = Number(chip.dataset.off);
      if (state.years.has(off)) {
        if (state.years.size === 1) return;   // keep at least one plan year
        state.years.delete(off);
        chip.classList.remove("on");
      } else {
        state.years.add(off);
        chip.classList.add("on");
      }
      onControlChange();
    });
  });
}

function wireControls() {
  els.climateSel.addEventListener("change", () => {
    state.climate = els.climateSel.value;
    onControlChange();
  });
  els.fidSeg.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (state.fidelity === btn.dataset.fid) return;
      state.fidelity = btn.dataset.fid;
      els.fidSeg.querySelectorAll("button").forEach((b) =>
        b.classList.toggle("on", b === btn));
      onControlChange();
    });
  });
  els.runBtn.addEventListener("click", runLive);
  els.cancel.addEventListener("click", () => {
    try { sim.cancel(); } catch (e) { /* worker already gone */ }
  });
}

function onControlChange() {
  // Back at defaults → snap to the pre-baked doc instantly.
  const sig = controlSig();
  if (sig === defaultSig() && sig !== renderedSig) {
    renderedKind = "prebaked";
    renderedSig = sig;
    renderAll(prebaked.get(state.profile));
  }
  updateRunUI();
}

function updateRunUI() {
  const matches = controlSig() === renderedSig;
  els.runBtn.disabled = running || matches || !wasmSupported();
  if (running) {
    els.runNote.textContent = "projection running…";
  } else if (matches) {
    els.runNote.textContent = renderedKind === "prebaked"
      ? "pre-baked full-year result shown"
      : "live result shown — matches controls";
  } else if (!wasmSupported()) {
    els.runNote.textContent = "these overrides need a live run, but WebAssembly is unavailable in this browser";
  } else {
    els.runNote.textContent = "controls differ from the figures — project live to apply";
  }
}

// ── live run ──────────────────────────────────────────────────────────

async function runLive() {
  if (running || !wasmSupported()) return;
  if (sim.busy) {
    els.runNote.textContent = "engine is busy with another run — try again shortly";
    return;
  }
  const sig = controlSig();
  const years = [...state.years].sort((a, b) => a - b);
  const simHours = state.fidelity === "fast" ? FAST_HOURS : FULL_YEAR_HOURS;
  setRunning(true);
  statusShow("starting engine…", null);
  try {
    await sim.ensureReady();
    statusShow(`year 0/${years.length}`, 0);
    const doc = await sim.runProjection(
      {
        profile: state.profile,
        climate: state.climate || null,
        years,
        simHours,
        baseYear: BASE_YEAR,
      },
      (done, total) => statusShow(`year ${done}/${total}`, total ? done / total : null),
    );
    renderedKind = "live";
    renderedSig = sig;
    renderAll(doc);
    statusDone("projection complete");
  } catch (err) {
    if (err && err.name === "CancelledError") {
      statusDone("cancelled");
    } else if (err && err.name === "BusyError") {
      statusDone("engine busy");
    } else {
      statusDone("run failed");
      showError(err);
    }
  } finally {
    setRunning(false);
    updateRunUI();
  }
}

function setRunning(b) {
  running = b;
  els.runBtn.textContent = b ? "running…" : "▶ Project live";
  els.cancel.style.display = b ? "" : "none";
  updateRunUI();
}

// ── status chip ───────────────────────────────────────────────────────

function statusShow(text, frac) {
  clearTimeout(statusTimer);
  els.status.classList.remove("fade");
  els.statusText.textContent = text;
  if (frac == null) {
    els.statusBar.classList.add("indet");
    els.statusBar.style.width = "";
  } else {
    els.statusBar.classList.remove("indet");
    els.statusBar.style.width = `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  }
}

function statusDone(text) {
  statusShow(text, 1);
  statusTimer = setTimeout(() => els.status.classList.add("fade"), 2000);
}

// ── rendering ─────────────────────────────────────────────────────────

function renderAll(doc) {
  if (!doc || !Array.isArray(doc.years) || !doc.years.length) return;
  currentDoc = doc;
  closeDrawer();
  els.errorBox.innerHTML = "";
  els.header.innerHTML = headerLine(doc);
  els.source.innerHTML = sourceLine(doc);
  for (const key of Object.keys(charts)) charts[key].render(doc);
  updateEmisBadge(doc);
  updateNotes(doc);
}

function docHours(doc) {
  const h = doc.years[0] && doc.years[0].metrics
    ? doc.years[0].metrics.simulation_hours : null;
  return Math.round(h ?? FULL_YEAR_HOURS);
}

function headerLine(doc) {
  const p = doc.profile;
  return `<strong>${esc(p.name)}</strong>` +
    ` — pop <span class="num">${p.population.toLocaleString("en-US")}</span>` +
    ` · ${esc(p.climate.name)}` +
    ` · <span class="num">${Math.round(p.renewable_target_pct * 100)}%</span> renewable by` +
    ` <span class="num">${p.renewable_target_year}</span>,` +
    ` net-zero <span class="num">${p.net_zero_year}</span>` +
    ` · base year <span class="num">${doc.base_year}</span>`;
}

function sourceLine(doc) {
  const kind = renderedKind === "prebaked" ? "pre-baked projection" : "live projection";
  return `${kind} · ${docHours(doc).toLocaleString("en-US")} h simulated per plan year` +
    ` · ${doc.years.length} plan years · click any plotted year to inspect`;
}

function updateNotes(doc) {
  const hrs = docHours(doc);
  els.capNote.textContent = renderedKind === "prebaked"
    ? "Pre-baked result: each plan year was simulated over the full 8,760 h year."
    : `Live run: each plan year simulated over ${hrs.toLocaleString("en-US")} h.`;
  // LCOE footnote — required for live short runs (hours < 8760).
  els.lcoeNote.textContent = hrs < FULL_YEAR_HOURS
    ? ` Annualized estimate — short runs overstate LCOE ~${Math.round(FULL_YEAR_HOURS / hrs)}×.`
    : "";
}

function updateEmisBadge(doc) {
  const ys = doc.years;
  const el = els.emisBadge;
  if (ys.length >= 2 && ys[0].emissions_tonnes > 0) {
    const delta = 1 - ys[ys.length - 1].emissions_tonnes / ys[0].emissions_tonnes;
    const pct = Math.round(Math.abs(delta) * 100);
    el.style.display = "";
    el.classList.toggle("up", delta < 0);
    el.textContent = `${delta < 0 ? "+" : "−"}${pct}% vs base year`;
  } else {
    el.style.display = "none";
  }
}

function applyRules() {
  document.querySelectorAll("[data-rule]").forEach((el) => {
    el.style.setProperty("--rule", dataColor(PALETTE[el.dataset.rule] || "#B2BEC3"));
  });
}

// ── charts ────────────────────────────────────────────────────────────

function mountAllCharts() {
  charts = {
    capacity: mountChart(echarts, byId("ch-capacity"), buildCapacity),
    lcoe: mountChart(echarts, byId("ch-lcoe"), buildLcoe),
    target: mountChart(echarts, byId("ch-target"), buildTarget),
    peak: mountChart(echarts, byId("ch-peak"), buildPeak),
    self: mountChart(echarts, byId("ch-self"), buildSelf),
    emissions: mountChart(echarts, byId("ch-emissions"), buildEmissions),
  };
  for (const h of Object.values(charts)) bindClick(h, onChartClick);
}

/** Re-attach the click handler across theme-flip rebuilds (mountChart
 *  swaps the chart instance; listeners do not survive). */
function bindClick(handle, fn) {
  handle.chart.on("click", fn);
  document.addEventListener("pp-theme-change", () => {
    setTimeout(() => { handle.chart.off("click", fn); handle.chart.on("click", fn); }, 0);
  });
}

function onChartClick(params) {
  if (!params || params.componentType !== "series" || params.dataIndex == null) return;
  const row = currentDoc && currentDoc.years ? currentDoc.years[params.dataIndex] : null;
  if (row) openDrawer(row);
}

/** Interpolated fractional year where fossil capacity first drops to or
 *  below renewable capacity, or null if no crossing within the horizon. */
function crossoverYear(ys) {
  for (let i = 1; i < ys.length; i++) {
    const d0 = ys[i - 1].fossil_capacity_kw - ys[i - 1].renewable_capacity_kw;
    const d1 = ys[i].fossil_capacity_kw - ys[i].renewable_capacity_kw;
    if (d0 > 0 && d1 <= 0) {
      const t = d0 / (d0 - d1);
      return ys[i - 1].year + t * (ys[i].year - ys[i - 1].year);
    }
  }
  return null;
}

function yearAxis(yrs) {
  return {
    type: "value",
    min: yrs[0] - 1,
    max: yrs[yrs.length - 1] + 1,
    minInterval: 1,
    axisLabel: { formatter: (v) => String(Math.round(v)) },
    splitLine: { show: false },
  };
}

const catAxis = (ys) => ({ type: "category", data: ys.map((r) => String(r.year)), boundaryGap: true });

const ptLabel = (formatter) => ({
  show: true, position: "top",
  fontFamily: FONT_MONO, fontSize: 9, color: inkMid(),
  formatter,
});

// FIG. 1 — capacity evolution
function buildCapacity(doc) {
  const ys = doc.years;
  const yrs = ys.map((r) => r.year);
  const cross = crossoverYear(ys);
  const mid = inkMid();
  return {
    legend: { top: 0 },
    grid: { top: 34, left: 50, right: 52, bottom: 26 },
    tooltip: {
      trigger: "axis",
      formatter: (ps) => {
        if (!ps || !ps.length) return "";
        const rows = ps.map((p) =>
          `${p.marker} ${p.seriesName} <b>${p.seriesName === "storage"
            ? formatKwh(p.value[1] * 1000) : formatKw(p.value[1] * 1000)}</b>`);
        return [`<b>${Math.round(ps[0].value[0])}</b>`, ...rows].join("<br>");
      },
    },
    xAxis: yearAxis(yrs),
    yAxis: [
      { type: "value", name: "MW", axisLabel: { formatter: compactNum } },
      { type: "value", name: "MWh", splitLine: { show: false }, axisLabel: { formatter: compactNum } },
    ],
    series: [
      {
        name: "fossil", type: "line", stack: "cap",
        data: ys.map((r) => [r.year, r.fossil_capacity_kw / 1000]),
        color: dataColor(PALETTE.gas), symbolSize: 8,
        lineStyle: { width: 1.5 }, areaStyle: { opacity: 0.4 },
        emphasis: { focus: "series" },
      },
      {
        name: "renewable", type: "line", stack: "cap",
        data: ys.map((r) => [r.year, r.renewable_capacity_kw / 1000]),
        color: dataColor(PALETTE.charge), symbolSize: 8,
        lineStyle: { width: 1.5 }, areaStyle: { opacity: 0.4 },
        emphasis: { focus: "series" },
        markLine: cross == null ? undefined : {
          silent: true, symbol: "none",
          lineStyle: { type: "dashed", color: mid, width: 1 },
          label: {
            formatter: `crossover ≈ ${cross.toFixed(1)}`,
            position: "insideEndTop",
            fontFamily: FONT_MONO, fontSize: 9.5, color: mid,
          },
          data: [{ xAxis: cross }],
        },
      },
      {
        name: "storage", type: "line", yAxisIndex: 1,
        data: ys.map((r) => [r.year, r.storage_capacity_kwh / 1000]),
        color: dataColor(PALETTE.supercap),
        symbol: "diamond", symbolSize: 8,
        lineStyle: { width: 1.5, type: [5, 3] },
      },
    ],
  };
}

// FIG. 2 — LCOE trend
function buildLcoe(doc) {
  const ys = doc.years;
  return {
    grid: { top: 30, left: 54, right: 20, bottom: 26 },
    tooltip: {
      trigger: "axis",
      valueFormatter: (v) => (v == null ? "∞" : `$${Number(v).toFixed(4)}/kWh`),
    },
    xAxis: catAxis(ys),
    yAxis: { type: "value", axisLabel: { formatter: (v) => `$${v.toFixed(3)}` } },
    series: [{
      name: "LCOE", type: "line",
      data: ys.map((r) => r.lcoe),
      color: dataColor(PALETTE.grid_import), symbolSize: 8,
      lineStyle: { width: 1.5 }, areaStyle: { opacity: 0.1 },
      label: ptLabel((p) => (p.value == null ? "" : `$${Number(p.value).toFixed(4)}`)),
    }],
  };
}

// FIG. 3 — renewables vs target, with milestone markLines
function buildTarget(doc) {
  const ys = doc.years;
  const yrs = ys.map((r) => r.year);
  const p = doc.profile;
  const mid = inkMid();
  const lo = yrs[0] - 1, hi = yrs[yrs.length - 1] + 1;
  const marks = [];
  if (p.renewable_target_year >= lo && p.renewable_target_year <= hi) {
    marks.push({
      xAxis: p.renewable_target_year,
      label: { formatter: `target ${p.renewable_target_year}`, position: "insideEndTop" },
    });
  }
  if (p.net_zero_year >= lo && p.net_zero_year <= hi) {
    marks.push({
      xAxis: p.net_zero_year,
      label: { formatter: `net-zero ${p.net_zero_year}`, position: "insideMiddleBottom" },
    });
  }
  return {
    legend: { top: 0 },
    grid: { top: 34, left: 44, right: 20, bottom: 26 },
    tooltip: {
      trigger: "axis",
      formatter: (ps) => {
        if (!ps || !ps.length) return "";
        const rows = ps.map((p2) =>
          `${p2.marker} ${p2.seriesName} <b>${Number(p2.value[1]).toFixed(1)}%</b>`);
        return [`<b>${Math.round(ps[0].value[0])}</b>`, ...rows].join("<br>");
      },
    },
    xAxis: yearAxis(yrs),
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: [
      {
        name: "actual", type: "line",
        data: ys.map((r) => [r.year, r.renewable_actual_pct * 100]),
        color: dataColor(PALETTE.charge), symbolSize: 8,
        lineStyle: { width: 1.8 },
        markLine: marks.length ? {
          silent: true, symbol: "none",
          lineStyle: { type: "dashed", color: mid, width: 1 },
          label: { fontFamily: FONT_MONO, fontSize: 9.5, color: mid },
          data: marks,
        } : undefined,
      },
      {
        name: "target", type: "line", silent: true,
        data: ys.map((r) => [r.year, r.renewable_target_pct * 100]),
        color: dataColor(PALETTE.generation), symbol: "none",
        lineStyle: { width: 1.5, type: "dashed" },
      },
    ],
  };
}

// FIG. 4 — peak demand bars
function buildPeak(doc) {
  const ys = doc.years;
  return {
    grid: { top: 30, left: 48, right: 12, bottom: 26 },
    tooltip: { trigger: "axis", valueFormatter: (v) => formatKw(v * 1000) },
    xAxis: catAxis(ys),
    yAxis: { type: "value", name: "MW", axisLabel: { formatter: compactNum } },
    series: [{
      name: "peak demand", type: "bar", barWidth: "55%",
      data: ys.map((r) => r.peak_demand_kw / 1000),
      color: dataColor(PALETTE.discharge),
      label: ptLabel((p) => formatKw(ys[p.dataIndex].peak_demand_kw)),
    }],
  };
}

// FIG. 5 — self-sufficiency
function buildSelf(doc) {
  const ys = doc.years;
  return {
    grid: { top: 30, left: 44, right: 16, bottom: 26 },
    tooltip: {
      trigger: "axis",
      valueFormatter: (v) => (v == null ? "—" : `${Number(v).toFixed(1)}%`),
    },
    xAxis: catAxis(ys),
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: [{
      name: "self-sufficiency", type: "line",
      data: ys.map((r) => (r.metrics ? r.metrics.self_sufficiency * 100 : null)),
      color: dataColor(PALETTE.hydro), symbolSize: 8,
      lineStyle: { width: 1.8 }, areaStyle: { opacity: 0.16 },
      label: ptLabel((p) => (p.value == null ? "" : `${Number(p.value).toFixed(0)}%`)),
    }],
  };
}

// FIG. 6 — emissions bars
function buildEmissions(doc) {
  const ys = doc.years;
  return {
    grid: { top: 30, left: 50, right: 12, bottom: 26 },
    tooltip: { trigger: "axis", valueFormatter: fmtTonnes },
    xAxis: catAxis(ys),
    yAxis: { type: "value", name: "t CO₂", axisLabel: { formatter: compactNum } },
    series: [{
      name: "emissions", type: "bar", barWidth: "55%",
      data: ys.map((r) => r.emissions_tonnes),
      color: dataColor(PALETTE.gas),
      label: ptLabel((p) => fmtTonnes(p.value)),
    }],
  };
}

// FIG. 7 — drawer donut
function buildDonut(row) {
  return {
    tooltip: { trigger: "item", valueFormatter: (v) => formatKw(v) },
    series: [{
      type: "pie",
      radius: ["52%", "78%"],
      center: ["50%", "52%"],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: "transparent", borderWidth: 1 },
      label: {
        fontFamily: FONT_MONO, fontSize: 10, color: inkMid(),
        formatter: (p) => `${p.name}\n${formatKw(p.value)}`,
      },
      data: [
        { name: "fossil", value: Math.max(0, row.fossil_capacity_kw),
          itemStyle: { color: dataColor(PALETTE.gas) } },
        { name: "renewable", value: Math.max(0, row.renewable_capacity_kw),
          itemStyle: { color: dataColor(PALETTE.charge) } },
      ].filter((d) => d.value > 0),
    }],
  };
}

// ── year inspector drawer ─────────────────────────────────────────────

function wireDrawer() {
  els.drawerClose.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });
}

function openDrawer(row) {
  const p = currentDoc.profile;
  els.drawerTitle.innerHTML =
    `<span class="num">${row.year}</span><span class="unit">+${row.year_offset} yr</span>`;
  els.drawerSub.textContent = `${p.name} · ${p.climate.name}`;
  els.drawerPlan.innerHTML = rowsHtml(PLAN_ROWS, row);
  els.drawerMetrics.innerHTML = rowsHtml(METRIC_ROWS, row.metrics || {});
  if (!drawerDonut) drawerDonut = mountChart(echarts, byId("drawer-donut"), buildDonut);
  drawerDonut.render(row);
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
}

function rowsHtml(rows, obj) {
  const html = rows
    .filter(([k]) => obj[k] !== undefined)
    .map(([k, label, fmt]) => `<tr><td>${label}</td><td class="num">${fmt(obj[k])}</td></tr>`)
    .join("");
  return html || `<tr><td class="muted">no data for this year</td><td></td></tr>`;
}

// ── error states ──────────────────────────────────────────────────────

function showError(err) {
  const msg = err && err.message ? err.message : String(err);
  const tb = err && err.traceback
    ? `<details style="margin-top:8px"><summary class="muted" style="cursor:pointer">Python traceback</summary>
         <pre class="traceback">${esc(err.traceback)}</pre></details>`
    : "";
  els.errorBox.innerHTML = `
    <div class="callout error">
      <strong>Projection failed.</strong> <span class="muted">${esc(msg)}</span>${tb}
    </div>`;
}

function showFatal(err) {
  const msg = err && err.message ? err.message : String(err);
  const skel = byId("charts-skeleton");
  if (skel) {
    skel.innerHTML = `
      <div class="col-12">
        <div class="callout error">
          <strong>Could not load projection data.</strong> <span class="muted">${esc(msg)}</span>
          <div class="muted small" style="margin-top:6px">
            Expected pre-baked docs under <span class="num">data/projections/</span> —
            re-run <span class="num">scripts/prebake.py</span>.
          </div>
        </div>
      </div>`;
  }
  if (els.cards) {
    els.cards.innerHTML =
      `<div class="empty-state" style="grid-column:1/-1">projection data unavailable</div>`;
  }
}
