/**
 * pages/compare.js — the Compare page (pageId "compare").
 *
 * Entirely pre-baked from docs/data/comparison.json: chip selector →
 * radar (5 normalized axes, same normalizations as the matplotlib
 * comparison figure) + six bar small-multiples + sortable metrics table
 * with CSV export. Preset colors are a stable categoricalColors(10)
 * assignment in manifest preset order.
 */

import * as echarts from "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.esm.min.js";
import {
  PALETTE, dataColor, categoricalColors,
  formatKw, formatKwh, formatUsd, formatPct,
  FONT_MONO, mountChart, SPEC_TEXT, specChip,
} from "../theme.js";
import { getManifest, getComparison } from "../data.js";

// ── module state ──────────────────────────────────────────────────────

let presets = [];               // comparison docs, in manifest preset order
let selected = new Set();       // preset keys currently on
let sortKey = null;             // table sort column id (null = manifest order)
let sortDir = 1;                // 1 asc, -1 desc
let capexScale = "log";         // "log" | "lin"
const handles = {};             // chart id → mountChart handle

const SPEC_DISMISS_KEY = "pp-spec-dismissed";

// ── helpers ───────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);
const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp01 = (x) => Math.max(0, Math.min(1, x ?? 0));

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Selected presets, always in stable manifest order. */
const currentSel = () => presets.filter((p) => selected.has(p.key));

/** Stable color per preset: index in manifest order → categoricalColors(N). */
function colorOf(preset) {
  const colors = categoricalColors(presets.length);
  return colors[presets.indexOf(preset)];
}

// ── radar axes (EXACT matplotlib normalizations) ──────────────────────

const RADAR_AXES = [
  { label: "Self-sufficiency", rawLabel: "Self-sufficiency",
    norm: (m) => clamp01(m.self_sufficiency),
    raw: (m) => m.self_sufficiency, fmtRaw: (v) => formatPct(v, 1) },
  { label: "Efficiency", rawLabel: "Efficiency",
    norm: (m) => clamp01(m.avg_system_efficiency),
    raw: (m) => m.avg_system_efficiency, fmtRaw: (v) => formatPct(v, 1) },
  { label: "Renewable fraction", rawLabel: "Renewable fraction",
    norm: (m) => clamp01(m.avg_renewable_fraction),
    raw: (m) => m.avg_renewable_fraction, fmtRaw: (v) => formatPct(v, 1) },
  { label: "Gen/Demand (÷2)", rawLabel: "Gen/Demand ratio",
    norm: (m) => Math.min((m.generation_to_demand_ratio ?? 0) / 2, 1),
    raw: (m) => m.generation_to_demand_ratio, fmtRaw: (v) => `${(v ?? 0).toFixed(2)}×` },
  { label: "Low curtailment", rawLabel: "Curtailment",
    norm: (m) => 1 - Math.min(m.curtailment_fraction ?? 0, 1),
    raw: (m) => m.curtailment_fraction, fmtRaw: (v) => formatPct(v, 2) },
];

// ── bar small-multiples spec ──────────────────────────────────────────

const BAR_METRICS = [
  { id: "selfsuff", ruleKey: "charge",
    get: (m) => m.self_sufficiency,
    fmt: (v) => formatPct(v, 1), axisFmt: (v) => formatPct(v, 0) },
  { id: "eff", ruleKey: "discharge",
    get: (m) => m.avg_system_efficiency,
    fmt: (v) => formatPct(v, 1), axisFmt: (v) => formatPct(v, 0) },
  { id: "renew", ruleKey: "generation",
    get: (m) => m.avg_renewable_fraction,
    fmt: (v) => formatPct(v, 1), axisFmt: (v) => formatPct(v, 0) },
  { id: "lcoe", ruleKey: "grid_import",
    get: (m) => m.estimated_lcoe_usd_kwh,
    fmt: (v) => (v == null ? "∞" : `$${v.toFixed(4)}`),
    axisFmt: (v) => `$${(+v).toFixed(2)}` },
  { id: "capex", ruleKey: "grid_export",
    get: (m) => m.total_capex_usd,
    fmt: (v) => formatUsd(v), axisFmt: (v) => formatUsd(v) },
  { id: "curtail", ruleKey: "curtail",
    get: (m) => m.curtailment_fraction,
    fmt: (v) => formatPct(v, 2), axisFmt: (v) => formatPct(v, 1) },
];

// ── table columns ─────────────────────────────────────────────────────
// get() returns the sortable raw value; html() the formatted cell.

const COLUMNS = [
  { id: "name", label: "Preset", str: true, csv: "preset",
    get: (p) => p.name,
    html: (p) => `${escapeHtml(p.name)}${p.exotic ? " " + specChip(true) : ""}`,
    raw: (p) => p.name },
  { id: "scale", label: "Scale", str: true, csv: "scale",
    get: (p) => p.scale_name,
    html: (p) => escapeHtml(p.scale_name),
    raw: (p) => p.scale_name },
  { id: "peak", label: "Peak", csv: "peak_load_kw",
    get: (p) => p.peak_load_kw,
    html: (p) => formatKw(p.peak_load_kw),
    raw: (p) => p.peak_load_kw },
  { id: "demand", label: "Demand", csv: "total_demand_kwh",
    get: (p) => p.metrics.total_demand_kwh,
    html: (p) => formatKwh(p.metrics.total_demand_kwh),
    raw: (p) => p.metrics.total_demand_kwh },
  { id: "gen", label: "Generation", csv: "total_generation_kwh",
    get: (p) => p.metrics.total_generation_kwh,
    html: (p) => formatKwh(p.metrics.total_generation_kwh),
    raw: (p) => p.metrics.total_generation_kwh },
  { id: "renew", label: "Renewable", csv: "renewable_fraction",
    get: (p) => p.metrics.avg_renewable_fraction,
    html: (p) => formatPct(p.metrics.avg_renewable_fraction, 1),
    raw: (p) => p.metrics.avg_renewable_fraction },
  { id: "eff", label: "Efficiency", csv: "system_efficiency",
    get: (p) => p.metrics.avg_system_efficiency,
    html: (p) => formatPct(p.metrics.avg_system_efficiency, 1),
    raw: (p) => p.metrics.avg_system_efficiency },
  { id: "selfsuff", label: "Self-suff.", csv: "self_sufficiency",
    get: (p) => p.metrics.self_sufficiency,
    html: (p) => formatPct(p.metrics.self_sufficiency, 1),
    raw: (p) => p.metrics.self_sufficiency },
  { id: "curtail", label: "Curtailment", csv: "curtailment_fraction",
    get: (p) => p.metrics.curtailment_fraction,
    html: (p) => formatPct(p.metrics.curtailment_fraction, 2),
    raw: (p) => p.metrics.curtailment_fraction },
  { id: "import", label: "Grid import", csv: "grid_import_kwh",
    get: (p) => p.metrics.total_grid_import_kwh,
    html: (p) => formatKwh(p.metrics.total_grid_import_kwh),
    raw: (p) => p.metrics.total_grid_import_kwh },
  { id: "export", label: "Grid export", csv: "grid_export_kwh",
    get: (p) => p.metrics.total_grid_export_kwh,
    html: (p) => formatKwh(p.metrics.total_grid_export_kwh),
    raw: (p) => p.metrics.total_grid_export_kwh },
  { id: "capex", label: "CAPEX", csv: "total_capex_usd",
    get: (p) => p.metrics.total_capex_usd,
    html: (p) => formatUsd(p.metrics.total_capex_usd),
    raw: (p) => p.metrics.total_capex_usd },
  { id: "lcoe", label: "LCOE", csv: "lcoe_usd_per_kwh",
    get: (p) => p.metrics.estimated_lcoe_usd_kwh,
    html: (p) => {
      const v = p.metrics.estimated_lcoe_usd_kwh;
      return v == null ? '<span class="muted">∞</span>' : `$${v.toFixed(4)}`;
    },
    raw: (p) => p.metrics.estimated_lcoe_usd_kwh },
];

// ── entry point ───────────────────────────────────────────────────────

export async function init() {
  let manifest, comparison;
  try {
    [manifest, comparison] = await Promise.all([getManifest(), getComparison()]);
  } catch (err) {
    showFatal(err);
    return;
  }

  // Stable manifest order drives both chip layout and color assignment.
  const order = new Map((manifest.presets ?? []).map((p, i) => [p.key, i]));
  presets = [...(comparison.presets ?? [])]
    .sort((a, b) => (order.get(a.key) ?? 999) - (order.get(b.key) ?? 999));

  if (!presets.length) {
    showFatal(new Error("comparison.json contains no presets"));
    return;
  }

  selected = new Set(presets.filter((p) => !p.exotic).map((p) => p.key));

  buildChips();
  buildSpecCallout();
  mountAllCharts();
  wireControls();
  applyRules();
  renderAll();

  // mountChart handles its own theme rebuild; this listener (registered
  // after all mounts, so it fires last) refreshes the DOM-side colors and
  // re-attaches chart event handlers to the rebuilt instances.
  document.addEventListener("pp-theme-change", () => {
    paintChipDots();
    applyRules();
    bindBarClicks();
  });
}

// ── fatal error state ─────────────────────────────────────────────────

function showFatal(err) {
  const root = $("compare-root");
  if (!root) return;
  root.innerHTML = `
    <div class="col-12">
      <div class="callout error" style="margin:0">
        <strong>Failed to load comparison data.</strong>
        <div class="muted" style="margin:4px 0 8px">
          ${escapeHtml(err && err.message ? err.message : String(err))}
          — check that <code>docs/data/comparison.json</code> exists (re-run <code>scripts/prebake.py</code>).
        </div>
        <details>
          <summary style="cursor:pointer;font-size:12px">Traceback</summary>
          <pre class="traceback">${escapeHtml((err && err.stack) || String(err))}</pre>
        </details>
      </div>
    </div>`;
}

// ── selector chips ────────────────────────────────────────────────────

function buildChips() {
  const row = $("chip-row");
  row.innerHTML = presets.map((p) => `
    <button type="button" class="chip" data-key="${escapeHtml(p.key)}"
            title="${escapeHtml(p.scale_name)} · peak ${escapeHtml(formatKw(p.peak_load_kw))}">
      <span class="dot"></span>${escapeHtml(p.name)}${p.exotic ? " " + specChip(true) : ""}
    </button>`).join("");
  paintChipDots();

  row.addEventListener("click", (e) => {
    if (e.target.closest(".spec-chip")) return;   // chip-internal SPEC badge navigates
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const key = chip.dataset.key;
    if (selected.has(key)) selected.delete(key);
    else selected.add(key);
    renderAll();
  });
}

function paintChipDots() {
  const colors = categoricalColors(presets.length);
  document.querySelectorAll("#chip-row .chip").forEach((chip) => {
    const i = presets.findIndex((p) => p.key === chip.dataset.key);
    if (i >= 0) chip.style.setProperty("--dot", colors[i]);
  });
}

// ── speculative callout ───────────────────────────────────────────────

function buildSpecCallout() {
  const el = $("spec-callout");
  el.innerHTML = `
    <div>${escapeHtml(SPEC_TEXT)} <a href="reference.html#speculative">Model notes →</a></div>
    <button type="button" class="dismiss" title="Dismiss" aria-label="Dismiss">✕</button>`;
  el.querySelector(".dismiss").addEventListener("click", () => {
    try { localStorage.setItem(SPEC_DISMISS_KEY, "1"); } catch { /* private mode */ }
    el.hidden = true;
  });
}

function updateSpecCallout() {
  let dismissed = false;
  try { dismissed = localStorage.getItem(SPEC_DISMISS_KEY) === "1"; } catch { /* ignore */ }
  $("spec-callout").hidden = dismissed || !currentSel().some((p) => p.exotic);
}

// ── chart mounting ────────────────────────────────────────────────────

function mountAllCharts() {
  const radarEl = $("radar-chart");
  radarEl.classList.remove("skeleton");
  handles.radar = mountChart(echarts, radarEl, radarOption);

  for (const metric of BAR_METRICS) {
    const el = $(`${metric.id}-chart`);
    el.classList.remove("skeleton");
    handles[metric.id] = mountChart(echarts, el, () => barOption(metric));
  }
  bindBarClicks();
}

function bindBarClicks() {
  for (const metric of BAR_METRICS) {
    const h = handles[metric.id];
    if (!h || !h.chart) continue;
    h.chart.off("click");
    h.chart.on("click", (params) => {
      const p = currentSel()[params.dataIndex];
      if (p) location.href = `index.html#${p.key}`;
    });
  }
}

// ── radar option ──────────────────────────────────────────────────────

function radarOption() {
  const sel = currentSel();
  const hair = cssVar("--hairline");
  const exotic = new Set(presets.filter((p) => p.exotic).map((p) => p.name));

  return {
    legend: {
      top: 0, type: "scroll",
      data: sel.map((p) => p.name),
      formatter: (name) => (exotic.has(name) ? `${name} †` : name),
    },
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params) => {
        const d = params.data || {};
        const m = d.metrics || {};
        const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${params.color};margin-right:6px"></span>`;
        const rows = RADAR_AXES.map((ax) =>
          `<div style="display:flex;justify-content:space-between;gap:18px">
             <span>${ax.rawLabel}</span><b>${ax.fmtRaw(ax.raw(m))}</b></div>`).join("");
        return `<div style="font-weight:600;margin-bottom:4px">${dot}${escapeHtml(d.name || "")}</div>${rows}`;
      },
    },
    radar: {
      indicator: RADAR_AXES.map((ax) => ({ name: ax.label, min: 0, max: 1 })),
      radius: "62%",
      center: ["50%", "56%"],
      splitNumber: 4,
      axisName: { color: cssVar("--ink-mid"), fontSize: 10.5, fontFamily: FONT_MONO },
      axisLine: { lineStyle: { color: hair } },
      splitLine: { lineStyle: { color: hair } },
      splitArea: { show: false },
    },
    series: [{
      type: "radar",
      emphasis: { focus: "self" },
      symbol: "circle",
      symbolSize: 3,
      data: sel.map((p) => {
        const color = colorOf(p);
        return {
          name: p.name,
          value: RADAR_AXES.map((ax) => ax.norm(p.metrics)),
          metrics: p.metrics,
          itemStyle: { color },
          lineStyle: { color, width: 1.6 },
          areaStyle: { color, opacity: 0.12 },
        };
      }),
    }],
  };
}

// ── bar option (one metric) ───────────────────────────────────────────

function barOption(metric) {
  const sel = currentSel();
  const isLog = metric.id === "capex" && capexScale === "log";

  return {
    grid: { top: 30, left: 6, right: 10, bottom: 2, containLabel: true },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      confine: true,
      formatter: (ps) => {
        const pt = Array.isArray(ps) ? ps[0] : ps;
        const p = currentSel()[pt.dataIndex];
        return `<div style="font-weight:600;margin-bottom:2px">${escapeHtml(p ? p.name : pt.name)}</div>
                ${pt.marker} ${metric.fmt(pt.value == null || pt.value === "-" ? null : pt.value)}`;
      },
    },
    xAxis: {
      type: "category",
      data: sel.map((p) => p.key),
      axisLabel: { rotate: 30, interval: 0, fontSize: 9 },
    },
    yAxis: {
      type: isLog ? "log" : "value",
      ...(isLog ? { logBase: 10 } : { min: 0 }),
      axisLabel: { fontSize: 9, formatter: (v) => metric.axisFmt(v) },
    },
    series: [{
      type: "bar",
      cursor: "pointer",
      barMaxWidth: 36,
      data: sel.map((p) => ({
        value: metric.get(p.metrics),
        name: p.key,
        itemStyle: { color: colorOf(p), borderRadius: [2, 2, 0, 0] },
      })),
      label: {
        show: true,
        position: "top",
        fontFamily: FONT_MONO,
        fontSize: 9,
        color: "inherit",
        formatter: (pp) => metric.fmt(pp.value == null || pp.value === "-" ? null : pp.value),
      },
    }],
  };
}

// ── table ─────────────────────────────────────────────────────────────

const sortNum = (v) => (v == null || Number.isNaN(v) ? Infinity : v);

function sortedSel() {
  const sel = currentSel();
  if (!sortKey) return sel;
  const col = COLUMNS.find((c) => c.id === sortKey);
  if (!col) return sel;
  return [...sel].sort((a, b) => {
    const va = col.get(a), vb = col.get(b);
    const cmp = col.str
      ? String(va).localeCompare(String(vb))
      : sortNum(va) - sortNum(vb);
    return cmp * sortDir;
  });
}

function renderTable() {
  const wrap = $("table-wrap");
  const rows = sortedSel();

  if (!rows.length) {
    wrap.innerHTML = `<div class="empty-state">No presets selected — toggle a chip above to populate the table.</div>`;
    return;
  }

  const thead = `<tr>${COLUMNS.map((c) => {
    const ind = sortKey === c.id
      ? `<span class="sort-ind">${sortDir === 1 ? "▲" : "▼"}</span>` : "";
    return `<th data-col="${c.id}" title="Sort by ${escapeHtml(c.label)}">${escapeHtml(c.label)}${ind}</th>`;
  }).join("")}</tr>`;

  const tbody = rows.map((p) =>
    `<tr>${COLUMNS.map((c) =>
      c.str ? `<td>${c.html(p)}</td>` : `<td class="num">${c.html(p)}</td>`
    ).join("")}</tr>`).join("");

  wrap.innerHTML = `<table class="data"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;

  wrap.querySelector("thead").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-col]");
    if (!th) return;
    const id = th.dataset.col;
    if (sortKey === id) sortDir = -sortDir;
    else { sortKey = id; sortDir = 1; }
    renderTable();
  });
}

// ── CSV export ────────────────────────────────────────────────────────

function csvCell(v) {
  if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function downloadCsv() {
  const rows = sortedSel();
  if (!rows.length) return;
  const lines = [COLUMNS.map((c) => c.csv).join(",")];
  for (const p of rows) lines.push(COLUMNS.map((c) => csvCell(c.raw(p))).join(","));
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "powerplan_comparison.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ── controls ──────────────────────────────────────────────────────────

function wireControls() {
  const conv = $("btn-conv"), all = $("btn-all"), csv = $("btn-csv");
  conv.disabled = false;
  all.disabled = false;

  conv.addEventListener("click", () => {
    selected = new Set(presets.filter((p) => !p.exotic).map((p) => p.key));
    renderAll();
  });
  all.addEventListener("click", () => {
    selected = new Set(presets.map((p) => p.key));
    renderAll();
  });
  csv.addEventListener("click", downloadCsv);

  $("capex-seg").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-scale]");
    if (!btn || btn.dataset.scale === capexScale) return;
    capexScale = btn.dataset.scale;
    document.querySelectorAll("#capex-seg button").forEach((b) =>
      b.classList.toggle("on", b.dataset.scale === capexScale));
    if (currentSel().length) handles.capex.render();
  });
}

// ── panel rule colors (re-applied on theme flip for dark substitutes) ─

function applyRules() {
  const setRule = (id, color) => {
    const el = $(id);
    if (el) el.style.setProperty("--rule", color);
  };
  setRule("panel-select", PALETTE.accent);
  setRule("panel-radar", PALETTE.accent);
  for (const m of BAR_METRICS) setRule(`panel-bar-${m.id}`, dataColor(PALETTE[m.ruleKey]));
}

// ── master render ─────────────────────────────────────────────────────

function setEmpty(id, empty) {
  const chart = $(`${id}-chart`);
  const emptyEl = $(`${id}-empty`);
  if (chart) chart.hidden = empty;
  if (emptyEl) emptyEl.hidden = !empty;
}

function renderAll() {
  const sel = currentSel();

  document.querySelectorAll("#chip-row .chip").forEach((chip) =>
    chip.classList.toggle("on", selected.has(chip.dataset.key)));
  $("sel-count").textContent = `${sel.length} of ${presets.length} selected`;
  updateSpecCallout();

  const empty = sel.length === 0;
  setEmpty("radar", empty);
  for (const m of BAR_METRICS) setEmpty(m.id, empty);
  if (!empty) {
    handles.radar.render();
    for (const m of BAR_METRICS) handles[m.id].render();
  }

  renderTable();
  $("btn-csv").disabled = empty;
}
