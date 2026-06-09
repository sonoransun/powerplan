/**
 * charts/dashboard.js — the full single-run dashboard (FIXED interface).
 *
 *   renderDashboard(container, doc, {figStart=1, zoomGroup="dash"}) → {dispose()}
 *   renderPowerBalance(el, doc, {events=[], height})                → {dispose()}
 *   kpiStrip(container, doc)                                        → void
 *
 * Mirrors the matplotlib 9-panel dashboard against the simulation-doc JSON
 * contract. Consumed by the gallery, builder and resilience pages — keep the
 * exports stable. All data colors come from theme.js; never hardcode hex.
 */

import * as echarts from "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.esm.min.js";
import {
  PALETTE, dataColor, sourceColor, storageColor,
  formatKw, formatKwh, formatUsd, formatHour, formatDay,
  mountChart, isSpeculative, isDark, FONT_MONO,
} from "../theme.js";
import { rollingMean } from "../data.js";
import { renderSystemDiagram } from "../components/system-diagram.js";

// ── small helpers ─────────────────────────────────────────────────────

const NIGHT_ID = "pp-night";

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function rgba(hex, a) {
  const h = String(hex).replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}

const pctFmt = (v) => `${(+v).toFixed(1)}%`;
const cyc = (c) => (c == null ? "—" : c < 10 ? c.toFixed(1) : String(Math.round(c)));
const specName = (n) => (isSpeculative(n) ? `⚠ ${n}` : n);

function hourAxis(doc) {
  const n = doc.sim.n_steps;
  const dt = doc.sim.dt_hours;
  const start = (doc.series && doc.series.start_hour) || 0;
  const hours = new Array(n);
  for (let i = 0; i < n; i++) hours[i] = start + i * dt;
  return hours;
}

function catAxis(hours) {
  return {
    type: "category",
    data: hours,
    boundaryGap: false,
    axisLabel: { formatter: (h) => formatDay(+h), hideOverlap: true },
  };
}

function timeTooltip(fmt = formatKw) {
  return {
    trigger: "axis",
    confine: true,
    formatter(params) {
      const list = Array.isArray(params) ? params : [params];
      if (!list.length) return "";
      let html = `<div style="margin-bottom:4px">${formatHour(+list[0].axisValue)}</div>`;
      for (const p of list) {
        const v = Array.isArray(p.value) ? p.value[p.value.length - 1] : p.value;
        if (v == null || p.seriesName === NIGHT_ID) continue;
        const note = isSpeculative(p.seriesName)
          ? ` <span style="opacity:.6">⚠ speculative</span>` : "";
        html += `<div>${p.marker} ${esc(p.seriesName)}: <b>${fmt(v)}</b>${note}</div>`;
      }
      return html;
    },
  };
}

function zooms(slider) {
  const z = [{ type: "inside", xAxisIndex: 0 }];
  if (slider) {
    z.push({
      type: "slider", xAxisIndex: 0, height: 18, bottom: 6, brushSelect: false,
      labelFormatter: (value, valueStr) => {
        const h = Number(valueStr != null && valueStr !== "" ? valueStr : value);
        return Number.isFinite(h) ? formatDay(h) : "";
      },
    });
  }
  return z;
}

function area(name, data, color, { opacity = 0.3, stack, width = 1 } = {}) {
  return {
    name, type: "line", data, stack, symbol: "none", sampling: "lttb",
    lineStyle: { width, color },
    itemStyle: { color },
    areaStyle: { color, opacity },
  };
}

/** Invisible carrier series for the night-shading markAreas. */
function nightSeries() {
  return { id: NIGHT_ID, name: NIGHT_ID, type: "line", data: [], silent: true, showSymbol: false };
}

function nightColor() {
  return isDark() ? "rgba(232,234,237,0.05)" : "rgba(45,52,54,0.055)";
}

function nightBands(hours, dt, i0, i1) {
  const n = hours.length;
  if (!n) return [];
  const first = hours[0], last = hours[n - 1];
  const snap = (h) => hours[Math.min(n - 1, Math.max(0, Math.round((h - first) / dt)))];
  const bands = [];
  const lo = hours[Math.max(0, i0)], hi = hours[Math.min(n - 1, i1)];
  for (let d = Math.floor(lo / 24) - 1; d <= Math.floor(hi / 24); d++) {
    let a = d * 24 + 18, b = d * 24 + 30; // 18:00 → next-day 06:00
    a = Math.max(a, first);
    b = Math.min(b, last);
    if (b - a < dt) continue;
    bands.push([{ xAxis: snap(a) }, { xAxis: snap(b) }]);
  }
  return bands;
}

function eventColor(type) {
  const t = String(type || "").toLowerCase();
  if (t.includes("demand")) return PALETTE.demand;
  if (t.includes("weather")) return PALETTE.curtail;
  if (t.includes("grid")) return PALETTE.grid_import;
  return PALETTE.loss; // source / storage faults
}

function eventMarkArea(events, hours, dt) {
  const n = hours.length;
  const first = hours[0];
  const snap = (h) => hours[Math.min(n - 1, Math.max(0, Math.round((h - first) / dt)))];
  return {
    silent: true,
    data: events.map((ev) => {
      const c = dataColor(eventColor(ev.type));
      return [{
        name: String(ev.type || "event").replaceAll("_", " "),
        xAxis: snap(ev.start_hour || 0),
        itemStyle: { color: rgba(c, 0.25) },
        label: { show: true, position: "insideTop", fontSize: 9, fontFamily: FONT_MONO, color: c },
      }, { xAxis: snap((ev.start_hour || 0) + (ev.duration_hours || dt)) }];
    }),
  };
}

// ── chart option builders ─────────────────────────────────────────────

function powerBalanceOption(doc, events, { slider = true } = {}) {
  const s = doc.series;
  const hours = hourAxis(doc);
  const dt = doc.sim.dt_hours;
  const hasUnmet = (s.unmet_kw || []).some((v) => v > 1e-9);
  const series = [
    area("Generation", s.generation_kw, dataColor(PALETTE.generation), { stack: "supply", opacity: 0.25 }),
    area("Storage discharge", s.discharge_kw, dataColor(PALETTE.discharge), { stack: "supply", opacity: 0.45 }),
    area("Grid import", s.grid_import_kw, dataColor(PALETTE.grid_import), { stack: "supply", opacity: 0.18 }),
    area("Storage charge", (s.charge_kw || []).map((v) => -v), dataColor(PALETTE.charge), { opacity: 0.35 }),
  ];
  if (hasUnmet) series.push(area("Unmet demand", s.unmet_kw, dataColor(PALETTE.unmet), { opacity: 0.4 }));
  series.push({
    name: "Demand", type: "line", data: s.demand_kw, symbol: "none", sampling: "lttb", z: 10,
    lineStyle: { width: 2, color: dataColor(PALETTE.demand) },
    itemStyle: { color: dataColor(PALETTE.demand) },
  });
  if (events && events.length) series[0].markArea = eventMarkArea(events, hours, dt);
  series.push(nightSeries());
  return {
    animation: false,
    legend: { top: 0, type: "scroll", data: series.map((x) => x.name).filter((nm) => nm && nm !== NIGHT_ID) },
    grid: { top: 30, left: 64, right: 18, bottom: slider ? 56 : 30 },
    tooltip: timeTooltip(formatKw),
    xAxis: catAxis(hours),
    yAxis: { type: "value", axisLabel: { formatter: (v) => formatKw(v) } },
    dataZoom: zooms(slider),
    series,
  };
}

function sourceBreakdownOption(doc) {
  const hours = hourAxis(doc);
  const entries = Object.entries(doc.series.source_kw || {});
  const series = entries.map(([name, arr]) =>
    area(name, arr, sourceColor(name), { stack: "src", opacity: 0.55, width: 0.8 }));
  series.push({
    name: "Demand", type: "line", data: doc.series.demand_kw, symbol: "none", sampling: "lttb", z: 10,
    lineStyle: { width: 1.5, type: "dashed", color: dataColor(PALETTE.demand) },
    itemStyle: { color: dataColor(PALETTE.demand) },
  });
  series.push(nightSeries());
  return {
    animation: false,
    legend: {
      top: 0, type: "scroll",
      data: [...entries.map(([n]) => n), "Demand"],
      formatter: specName,
      tooltip: {
        show: true,
        formatter: (p) => (isSpeculative(p.name)
          ? "Speculative technology — intentionally optimistic what-if parameters"
          : p.name),
      },
    },
    grid: { top: 30, left: 64, right: 14, bottom: 28 },
    tooltip: timeTooltip(formatKw),
    xAxis: catAxis(hours),
    yAxis: { type: "value", axisLabel: { formatter: (v) => formatKw(v) } },
    dataZoom: zooms(false),
    series,
  };
}

function socOption(doc) {
  const hours = hourAxis(doc);
  const entries = Object.entries(doc.series.storage_soc || {});
  const series = entries.map(([name, arr]) => ({
    name, type: "line", symbol: "none", sampling: "lttb",
    data: arr.map((v) => +(v * 100).toFixed(2)),
    lineStyle: { width: 1.2, color: storageColor(name) },
    itemStyle: { color: storageColor(name) },
  }));
  series.push(nightSeries());
  return {
    animation: false,
    legend: { top: 0, type: "scroll", data: entries.map(([n]) => n) },
    grid: { top: 30, left: 44, right: 14, bottom: 28 },
    tooltip: timeTooltip(pctFmt),
    xAxis: catAxis(hours),
    yAxis: { type: "value", min: -5, max: 105, axisLabel: { formatter: "{value}%" } },
    dataZoom: zooms(false),
    series,
  };
}

function rollingOption(doc, key, color, label) {
  const hours = hourAxis(doc);
  const win = Math.max(1, Math.round(24 / doc.sim.dt_hours));
  const data = rollingMean(doc.series[key] || [], win).map((v) => +(v * 100).toFixed(2));
  return {
    animation: false,
    grid: { top: 12, left: 40, right: 10, bottom: 26 },
    tooltip: timeTooltip(pctFmt),
    xAxis: catAxis(hours),
    yAxis: { type: "value", min: 0, max: 105, axisLabel: { formatter: "{value}%" } },
    dataZoom: zooms(false),
    series: [area(label, data, dataColor(color), { opacity: 0.18, width: 1.2 })],
  };
}

function curtailGridOption(doc) {
  const hours = hourAxis(doc);
  const s = doc.series;
  const series = [
    area("Curtailment", s.curtailment_kw, dataColor(PALETTE.curtail), { opacity: 0.35 }),
    area("Grid import", s.grid_import_kw, dataColor(PALETTE.grid_import), { opacity: 0.35 }),
    area("Grid export", (s.grid_export_kw || []).map((v) => -v), dataColor(PALETTE.grid_export), { opacity: 0.35 }),
  ];
  series[0].markLine = {
    silent: true, symbol: "none",
    lineStyle: { color: isDark() ? "#5F6368" : "#B2BEC3", width: 1 },
    label: { show: false },
    data: [{ yAxis: 0 }],
  };
  series.push(nightSeries());
  return {
    animation: false,
    legend: { top: 0, data: series.map((x) => x.name).filter((nm) => nm && nm !== NIGHT_ID) },
    grid: { top: 30, left: 64, right: 14, bottom: 28 },
    tooltip: timeTooltip(formatKw),
    xAxis: catAxis(hours),
    yAxis: { type: "value", axisLabel: { formatter: (v) => formatKw(v) } },
    dataZoom: zooms(false),
    series,
  };
}

function waterfallRows(doc) {
  const m = doc.metrics, s = doc.series || {}, dt = doc.sim.dt_hours;
  const viaStorage = (s.discharge_kw || []).reduce((a, b) => a + b, 0) * dt;
  const rows = [
    { name: "Generated", delta: m.total_generation_kwh || 0, color: PALETTE.generation },
    { name: "Ctrl losses", delta: -(m.total_controller_losses_kwh || 0), color: PALETTE.loss },
    { name: "Direct use", delta: -((m.total_demand_kwh || 0) - (m.total_grid_import_kwh || 0)), color: PALETTE.demand },
    { name: "Via storage", delta: -viaStorage, color: PALETTE.discharge },
    { name: "Curtailed", delta: -(m.total_curtailment_kwh || 0), color: PALETTE.curtail },
    { name: "Grid import", delta: m.total_grid_import_kwh || 0, color: PALETTE.grid_import },
  ];
  let run = 0;
  for (const r of rows) {
    r.lo = Math.min(run, run + r.delta);
    r.len = Math.abs(r.delta);
    run += r.delta;
    r.run = run;
  }
  return rows;
}

function waterfallOption(doc) {
  const rows = waterfallRows(doc);
  return {
    grid: { top: 10, left: 96, right: 104, bottom: 28 },
    tooltip: {
      trigger: "item", confine: true,
      formatter(p) {
        const r = rows[p.dataIndex];
        return `${r.name}: <b>${(r.delta < 0 ? "−" : "+") + formatKwh(Math.abs(r.delta))}</b>` +
          `<br><span style="opacity:.65">running total ${formatKwh(r.run)}</span>`;
      },
    },
    xAxis: { type: "value", axisLabel: { formatter: (v) => formatKwh(v) } },
    yAxis: { type: "category", inverse: true, data: rows.map((r) => r.name), axisLabel: { fontSize: 10 } },
    series: [
      { // transparent offset carrier
        type: "bar", stack: "wf", barWidth: 16, silent: true,
        itemStyle: { color: "transparent" }, emphasis: { disabled: true },
        data: rows.map((r) => r.lo), tooltip: { show: false },
      },
      {
        type: "bar", stack: "wf", barWidth: 16,
        data: rows.map((r) => ({ value: r.len, itemStyle: { color: dataColor(r.color) } })),
        label: {
          show: true, position: "right", fontFamily: FONT_MONO, fontSize: 10,
          formatter: (p) => {
            const r = rows[p.dataIndex];
            return (p.dataIndex === 0 ? "" : r.delta < 0 ? "−" : "+") + formatKwh(Math.abs(r.delta));
          },
        },
      },
    ],
  };
}

function storageHealthOption(doc) {
  const det = doc.metrics.storage_details || [];
  return {
    grid: { top: 28, left: 44, right: 14, bottom: 44 },
    tooltip: {
      trigger: "item", confine: true,
      formatter(p) {
        const d = det[p.dataIndex];
        if (!d) return "";
        return `<b>${esc(d.name)}</b><br>health <b>${(d.health * 100).toFixed(1)}%</b> · ${cyc(d.cycles)} cycles` +
          `<br>final SOC ${(d.soc * 100).toFixed(0)}% · ${d.temperature_c.toFixed(1)} °C` +
          `<br>capex ${formatUsd(d.capital_cost)}`;
      },
    },
    xAxis: {
      type: "category",
      data: det.map((d) => d.name),
      axisLabel: {
        interval: 0, fontSize: 9.5, rotate: det.length > 4 ? 16 : 0,
        formatter: (n) => (n.length > 16 ? `${n.slice(0, 15)}…` : n),
      },
    },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: [{
      type: "bar", barMaxWidth: 46,
      data: det.map((d) => ({ value: +(d.health * 100).toFixed(2), itemStyle: { color: storageColor(d.name) } })),
      label: {
        show: true, position: "top", fontFamily: FONT_MONO, fontSize: 10,
        formatter: (p) => `${cyc(det[p.dataIndex] && det[p.dataIndex].cycles)} cy`,
      },
    }],
  };
}

function sankeyOption(doc) {
  const m = doc.metrics, s = doc.series || {}, dt = doc.sim.dt_hours;
  const srcs = (m.source_details || []).filter((d) => d.cumulative_kwh > 1e-6);
  const charge = (s.charge_kw || []).reduce((a, b) => a + b, 0) * dt;
  const losses = m.total_controller_losses_kwh || 0;
  const curt = m.total_curtailment_kwh || 0;
  const exp = m.total_grid_export_kwh || 0;
  const genIn = srcs.reduce((a, d) => a + d.cumulative_kwh, 0);
  const direct = Math.max(0, (m.total_generation_kwh || 0) - losses - charge - curt - exp);
  const dests = [
    { name: "Direct use", v: direct, color: PALETTE.demand },
    { name: "Storage charge", v: charge, color: PALETTE.charge },
    { name: "Curtailed", v: curt, color: PALETTE.curtail },
    { name: "Ctrl losses", v: losses, color: PALETTE.loss },
    { name: "Grid export", v: exp, color: PALETTE.grid_export },
  ].filter((d) => d.v > 1e-6);
  const destSum = dests.reduce((a, d) => a + d.v, 0);
  const k = destSum > 0 ? genIn / destSum : 1; // pro-rata rebalance to the bus input
  const FOOT = `<div style="margin-top:4px;font-size:10px;opacity:.6">` +
    `proportional allocation — dispatch does not track per-source routing</div>`;
  return {
    tooltip: {
      trigger: "item", confine: true,
      formatter(p) {
        if (p.dataType === "edge") {
          return `${esc(specName(p.data.source))} → ${esc(specName(p.data.target))}: ` +
            `<b>${formatKwh(p.data.value)}</b>${FOOT}`;
        }
        const specNote = isSpeculative(p.name)
          ? `<div style="margin-top:3px">⚠ speculative technology</div>` : "";
        return `${esc(specName(p.name))}: <b>${formatKwh(p.data.value)}</b>${specNote}${FOOT}`;
      },
    },
    series: [{
      type: "sankey", left: 8, right: 120, top: 14, bottom: 10,
      nodeWidth: 14, nodeGap: 16, nodeAlign: "justify",
      emphasis: { focus: "adjacency" },
      lineStyle: { color: "gradient", opacity: 0.3, curveness: 0.5 },
      label: { fontFamily: FONT_MONO, fontSize: 10, formatter: (p) => specName(p.name) },
      data: [
        ...srcs.map((d) => ({ name: d.name, value: d.cumulative_kwh, depth: 0, itemStyle: { color: sourceColor(d.name) } })),
        { name: "Bus", value: genIn, depth: 1, itemStyle: { color: dataColor(PALETTE.loss) } },
        ...dests.map((d) => ({ name: d.name, value: d.v * k, depth: 2, itemStyle: { color: dataColor(d.color) } })),
      ],
      links: [
        ...srcs.map((d) => ({ source: d.name, target: "Bus", value: d.cumulative_kwh })),
        ...dests.map((d) => ({ source: "Bus", target: d.name, value: d.v * k })),
      ],
    }],
  };
}

// ── KPI strip ─────────────────────────────────────────────────────────

function numUnit(str) {
  const i = String(str).lastIndexOf(" ");
  if (i < 0) return `<span class="num">${str}</span>`;
  return `<span class="num">${str.slice(0, i)}</span><span class="unit">${str.slice(i + 1)}</span>`;
}

function kpiCard({ label, html, rule, bar = null, tick = null, note = "" }) {
  const barHtml = bar == null ? "" :
    `<div class="kpi-bar"><i style="width:${(Math.max(0, Math.min(1, bar)) * 100).toFixed(1)}%"></i>` +
    `${tick == null ? "" : `<span class="tick" style="left:${(tick * 100).toFixed(0)}%"></span>`}</div>`;
  const noteHtml = note ? `<div class="muted" style="font-size:10px;line-height:1.4;margin-top:5px">${note}</div>` : "";
  return `<div class="kpi" style="--rule:${rule}">
    <div class="kicker">${label}</div>
    <div class="kpi-value">${html}</div>${barHtml}${noteHtml}
  </div>`;
}

export function kpiStrip(container, doc) {
  const m = doc.metrics || {};
  const r = doc.resilience;
  const lcoe = m.estimated_lcoe_usd_kwh;
  const ratio = m.generation_to_demand_ratio || 0;
  const simHours = (doc.sim && doc.sim.hours) || 8760;
  const pct = (v) => (v == null ? `<span class="num">—</span>` :
    `<span class="num">${(v * 100).toFixed(1)}</span><span class="unit">%</span>`);
  const cards = [
    kpiCard({ label: "Self-sufficiency", html: pct(m.self_sufficiency), rule: dataColor(PALETTE.charge), bar: m.self_sufficiency }),
    kpiCard({ label: "Avg efficiency", html: pct(m.avg_system_efficiency), rule: dataColor(PALETTE.discharge), bar: m.avg_system_efficiency }),
    kpiCard({ label: "Renewable fraction", html: pct(m.avg_renewable_fraction), rule: dataColor(PALETTE.generation), bar: m.avg_renewable_fraction }),
    kpiCard({
      label: "Gen / demand",
      html: `<span class="num">${ratio.toFixed(2)}</span><span class="unit">×</span>`,
      rule: dataColor(PALETTE.curtail), bar: Math.min(ratio / 2, 1), tick: 0.5,
    }),
    kpiCard({ label: "CAPEX", html: `<span class="num">${formatUsd(m.total_capex_usd)}</span>`, rule: dataColor(PALETTE.grid_export) }),
    kpiCard({
      label: "LCOE",
      html: lcoe == null
        ? `<span class="num">∞</span><span class="unit">$/kWh</span>`
        : `<span class="num">${lcoe.toFixed(4)}</span><span class="unit">$/kWh</span>`,
      rule: dataColor(PALETTE.grid_import),
      note: simHours < 8760
        ? `annualized estimate — short runs overstate LCOE ~${Math.round(8760 / simHours)}×` : "",
    }),
    kpiCard({ label: "Peak demand", html: numUnit(formatKw(m.peak_demand_kw)), rule: dataColor(PALETTE.demand) }),
  ];
  if (r) {
    cards.push(kpiCard({ label: "LOLP", html: pct(r.lolp), rule: dataColor(PALETTE.unmet), bar: r.lolp }));
    cards.push(kpiCard({ label: "Energy not served", html: numUnit(formatKwh(r.ens_kwh)), rule: dataColor(PALETTE.unmet) }));
  }
  container.innerHTML = `<div class="kpi-strip">${cards.join("")}</div>`;
}

// ── standalone power balance (resilience page) ────────────────────────

export function renderPowerBalance(elTarget, doc, { events = [], height } = {}) {
  if (height != null) {
    elTarget.style.height = typeof height === "number" ? `${height}px` : height;
  } else if (!elTarget.clientHeight) {
    elTarget.style.minHeight = "320px";
  }
  const h = mountChart(echarts, elTarget, () => powerBalanceOption(doc, events, { slider: true }));
  h.render();
  return { dispose: () => h.dispose() };
}

// ── full dashboard ────────────────────────────────────────────────────

const WINDOW_DEFS = [["24h", 24], ["7d", 168], ["30d", 720], ["90d", 2160], ["365d", 8760]];

export function renderDashboard(container, doc, { figStart = 1, zoomGroup = "dash" } = {}) {
  const hours = hourAxis(doc);
  const dt = doc.sim.dt_hours;
  const events = (doc.failures && doc.failures.events) || [];
  const handles = [];
  const cleanups = [];
  let fig = figStart;

  container.innerHTML = "";

  // KPI strip
  const kpiHost = el(`<div style="margin:0 0 16px"></div>`);
  container.appendChild(kpiHost);
  kpiStrip(kpiHost, doc);

  // time-window controls
  const controls = el(`<div style="display:flex;align-items:center;gap:12px;margin:0 0 14px;flex-wrap:wrap">
    <span class="kicker">Window</span>
    <div class="seg" data-role="windows"></div>
    <span class="muted" style="font-size:11.5px">scrub the slider or drag inside any time chart — panels stay in sync</span>
  </div>`);
  container.appendChild(controls);

  const grid = el(`<div class="grid"></div>`);
  container.appendChild(grid);

  function panel({ col, rule, kicker, caption, extra = "" }) {
    const p = el(`<section class="panel reveal col-${col}" style="--rule:${rule}">
      <div class="panel-head"><span class="kicker">${kicker}</span>${extra ? `<span class="muted" style="font-size:11px">${extra}</span>` : ""}</div>
      <div class="panel-body"></div>
      <div class="fig-caption"><span class="fig-no">FIG. ${fig++}</span>${caption}</div>
    </section>`);
    grid.appendChild(p);
    return p.querySelector(".panel-body");
  }

  function addChart(body, build, { cls = "chart", group } = {}) {
    const div = document.createElement("div");
    div.className = cls;
    body.appendChild(div);
    const h = mountChart(echarts, div, build, group ? { group } : {});
    h.render();
    handles.push(h);
    return h;
  }

  function emptyState(body, msg) {
    body.innerHTML = `<div class="empty-state" style="min-height:260px;display:flex;align-items:center;justify-content:center">${msg}</div>`;
  }

  const nStor = Object.keys(doc.series.storage_soc || {}).length;
  const nSrc = Object.keys(doc.series.source_kw || {}).length;

  // 1 · power balance (master chart — owns the slider)
  const pbBody = panel({
    col: 12, rule: dataColor(PALETTE.demand), kicker: "POWER BALANCE",
    extra: events.length ? `${events.length} injected event${events.length > 1 ? "s" : ""}` : "",
    caption: "Hourly dispatch — supply stack (generation → storage discharge → grid import) against the demand line; " +
      "storage charging plotted below zero." +
      (events.length ? " Shaded bands mark injected failure events." : "") +
      " Faint verticals are local nights (18:00–06:00), shown at windows ≤ 14 days.",
  });
  const master = addChart(pbBody, () => powerBalanceOption(doc, events, { slider: true }),
    { cls: "chart tall", group: zoomGroup });

  // 2 · source breakdown
  const srcBody = panel({
    col: 6, rule: dataColor(PALETTE.generation), kicker: "SOURCE BREAKDOWN",
    extra: `${nSrc} source${nSrc === 1 ? "" : "s"}`,
    caption: "Per-source output, stacked; dashed line is total demand. ⚠ marks speculative technologies.",
  });
  const srcChart = nSrc
    ? addChart(srcBody, () => sourceBreakdownOption(doc), { group: zoomGroup })
    : (emptyState(srcBody, "No generation sources in this configuration."), null);

  // 3 · storage state of charge
  const socBody = panel({
    col: 6, rule: dataColor(PALETTE.discharge), kicker: "STORAGE STATE OF CHARGE",
    extra: nStor ? `${nStor} unit${nStor === 1 ? "" : "s"}` : "",
    caption: "State of charge per unit, percent of effective capacity.",
  });
  const socChart = nStor
    ? addChart(socBody, () => socOption(doc), { group: zoomGroup })
    : (emptyState(socBody, "No storage units in this configuration — supply is direct generation and grid only."), null);

  // 4 · rolling efficiency
  const effBody = panel({
    col: 3, rule: dataColor(PALETTE.discharge), kicker: "ROLLING EFFICIENCY",
    caption: "System conversion efficiency, 24 h rolling mean.",
  });
  addChart(effBody, () => rollingOption(doc, "system_efficiency", PALETTE.discharge, "Efficiency"),
    { cls: "chart short", group: zoomGroup });

  // 5 · rolling renewable
  const renBody = panel({
    col: 3, rule: dataColor(PALETTE.charge), kicker: "ROLLING RENEWABLE",
    caption: "Renewable share of delivered energy, 24 h rolling mean.",
  });
  addChart(renBody, () => rollingOption(doc, "renewable_fraction", PALETTE.charge, "Renewable share"),
    { cls: "chart short", group: zoomGroup });

  // 6 · curtailment & grid
  const cgBody = panel({
    col: 6, rule: dataColor(PALETTE.curtail), kicker: "CURTAILMENT & GRID",
    caption: "Curtailed generation and grid trade; exports plotted below the zero line.",
  });
  const cgChart = addChart(cgBody, () => curtailGridOption(doc), { cls: "chart short", group: zoomGroup });

  echarts.connect(zoomGroup);

  // 7 · energy waterfall
  const wfBody = panel({
    col: 6, rule: dataColor(PALETTE.generation), kicker: "ENERGY WATERFALL",
    caption: "Cumulative energy balance from generation to disposition; grid import closes the gap.",
  });
  addChart(wfBody, () => waterfallOption(doc));

  // 8 · storage health
  const shBody = panel({
    col: 6, rule: dataColor(PALETTE.lithium), kicker: "STORAGE HEALTH",
    caption: "Remaining capacity health with accumulated equivalent full cycles. Hover for SOC, temperature and capex.",
  });
  if (doc.metrics.storage_details && doc.metrics.storage_details.length) {
    addChart(shBody, () => storageHealthOption(doc));
  } else {
    emptyState(shBody, "No storage units in this configuration.");
  }

  // 9 · system architecture
  const sysBody = panel({
    col: 12, rule: dataColor(PALETTE.loss), kicker: "SYSTEM ARCHITECTURE",
    caption: "Component topology — connector width ∝ log rated power. Hover any node for its end-of-run telemetry.",
  });
  const diagHost = document.createElement("div");
  sysBody.appendChild(diagHost);
  renderSystemDiagram(diagHost, doc.config, doc.metrics);
  cleanups.push(() => {
    const d = diagHost.__ppDiag;
    if (d) { d.ro.disconnect(); document.removeEventListener("pp-theme-change", d.onTheme); }
  });

  // 10 · energy flow sankey
  const skBody = panel({
    col: 12, rule: dataColor(PALETTE.charge), kicker: "ENERGY FLOW",
    caption: "Cumulative routing of generated energy through the bus. Per-source destination shares are " +
      "allocated pro-rata — the dispatch engine does not track per-source routing.",
  });
  if ((doc.metrics.source_details || []).some((d) => d.cumulative_kwh > 1e-6)) {
    addChart(skBody, () => sankeyOption(doc), { cls: "chart tall" });
  } else {
    emptyState(skBody, "No generated energy to route.");
  }

  // ── time-window seg buttons + shared zoom + night shading ──────────
  const segHost = controls.querySelector("[data-role=windows]");
  const simHours = hours.length ? hours[hours.length - 1] - hours[0] + dt : 0;
  const windows = WINDOW_DEFS.filter(([, h]) => h <= simHours + 1e-6);
  if (windows.length && simHours - windows[windows.length - 1][1] > 1) windows.push(["all", simHours]);
  const btns = windows.map(([label, h]) => {
    const b = el(`<button type="button">${label}</button>`);
    b.dataset.h = h;
    b.addEventListener("click", () => applyWindow(h));
    segHost.appendChild(b);
    return b;
  });
  if (!btns.length) controls.style.display = "none";

  const nightHandles = [master, srcChart, socChart, cgChart].filter(Boolean);

  function visibleRange() {
    const n = hours.length;
    let i0 = 0, i1 = Math.max(0, n - 1);
    try {
      const dz = (master.chart.getOption().dataZoom || [])[0];
      if (dz && dz.start != null && dz.end != null) {
        i0 = Math.max(0, Math.round((dz.start / 100) * (n - 1)));
        i1 = Math.min(n - 1, Math.round((dz.end / 100) * (n - 1)));
      } else if (dz && dz.startValue != null && dz.endValue != null) {
        // after a seg-button dispatchAction the percent pair is nulled;
        // startValue/endValue hold category indices for our axis
        i0 = Math.max(0, Math.min(n - 1, Math.round(+dz.startValue) || 0));
        i1 = Math.max(i0, Math.min(n - 1, Math.round(+dz.endValue) || 0));
      }
    } catch (e) { /* chart mid-rebuild */ }
    return [i0, i1];
  }

  function applyWindow(wHours) {
    const n = hours.length;
    const steps = Math.max(1, Math.round(wHours / dt));
    let [i0] = visibleRange();
    let i1 = i0 + steps - 1;
    if (i1 > n - 1) { i1 = n - 1; i0 = Math.max(0, i1 - steps + 1); }
    master.chart.dispatchAction({ type: "dataZoom", dataZoomIndex: 0, startValue: i0, endValue: i1 });
  }

  let nightTimer = null;
  function update() {
    const [i0, i1] = visibleRange();
    const visH = (i1 - i0 + 1) * dt;
    for (const b of btns) b.classList.toggle("on", Math.abs(visH - Number(b.dataset.h)) <= dt * 1.5);
    const bands = visH <= 24 * 14 + 1e-6 ? nightBands(hours, dt, i0, i1) : [];
    const ma = { silent: true, itemStyle: { color: nightColor() }, label: { show: false }, data: bands };
    for (const h of nightHandles) {
      try { h.chart.setOption({ series: [{ id: NIGHT_ID, markArea: ma }] }); } catch (e) { /* disposed */ }
    }
  }
  const schedule = () => { clearTimeout(nightTimer); nightTimer = setTimeout(update, 160); };
  const bindZoom = () => { try { master.chart.on("datazoom", schedule); } catch (e) { /* disposed */ } };
  const onTheme = () => setTimeout(() => { bindZoom(); update(); }, 0); // after mountChart rebuilds
  document.addEventListener("pp-theme-change", onTheme);
  bindZoom();
  update();
  cleanups.push(() => {
    clearTimeout(nightTimer);
    document.removeEventListener("pp-theme-change", onTheme);
  });

  return {
    dispose() {
      cleanups.forEach((f) => { try { f(); } catch (e) { /* already gone */ } });
      handles.forEach((h) => { try { h.dispose(); } catch (e) { /* already gone */ } });
      container.innerHTML = "";
    },
  };
}
