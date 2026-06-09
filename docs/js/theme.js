/**
 * theme.js — palette, color matchers, formatters, ECharts theme.
 *
 * The data colors are a verbatim port of powerplan/styles.py (Paul Tol's
 * muted scheme; the Python docstring says "Wong 2011" but the hexes are
 * Tol's). Never hardcode hex in chart configs — import from here, mirroring
 * the repo convention. Two hexes get contrast-adjusted substitutes in dark
 * mode only (see darkSub).
 */

export const PALETTE = {
  // energy flow roles
  demand: "#CC6677",
  generation: "#DDCC77",
  discharge: "#88CCEE",
  charge: "#44AA99",
  grid_import: "#AA4499",
  grid_export: "#332288",
  curtail: "#999933",
  loss: "#BBBBBB",
  // source types
  solar: "#DDCC77",
  wind: "#88CCEE",
  hydro: "#44AA99",
  geothermal: "#CC6677",
  gas: "#882255",
  fusion: "#AA4499",
  antimatter: "#6699CC",
  // storage types
  lithium: "#88CCEE",
  sodium: "#44AA99",
  flow: "#CC6677",
  flywheel: "#DDCC77",
  hydrogen: "#999933",
  supercap: "#332288",
  smes: "#AA4499",
  // semantic extras (web only)
  unmet: "#CC3311",
  accent: "#0984E3",
};

export const CATEGORY_CYCLE = [
  "#332288", "#88CCEE", "#44AA99", "#999933",
  "#DDCC77", "#CC6677", "#882255", "#AA4499",
  "#6699CC", "#117733", "#661100", "#BBBBBB",
];

// Dark-mode contrast substitutions — the ONLY two data-color overrides.
const darkSub = { "#332288": "#5B4FC0", "#882255": "#A8336B" };

export function dataColor(hex) {
  return isDark() && darkSub[hex] ? darkSub[hex] : hex;
}

/** Port of styles.source_color — substring match on the display name. */
export function sourceColor(name) {
  const n = String(name).toLowerCase();
  for (const key of ["solar", "wind", "hydro", "geothermal", "gas", "fusion", "antimatter"]) {
    if (n.includes(key)) return dataColor(PALETTE[key]);
  }
  return dataColor(PALETTE.generation);
}

/** Port of styles.storage_color. */
export function storageColor(name) {
  const n = String(name).toLowerCase();
  const keys = ["lithium", "li-ion", "sodium", "na ", "flow", "vanadium",
    "flywheel", "hydrogen", "h2", "supercap", "graphene", "smes"];
  const mapped = { "li-ion": "lithium", "na ": "sodium", vanadium: "flow", h2: "hydrogen", graphene: "supercap" };
  for (const key of keys) {
    if (n.includes(key)) return dataColor(PALETTE[mapped[key] ?? key]);
  }
  return dataColor(PALETTE.loss);
}

export function categoricalColors(n) {
  return Array.from({ length: n }, (_, i) => dataColor(CATEGORY_CYCLE[i % CATEGORY_CYCLE.length]));
}

// ── formatters (port of styles.format_kw / format_kwh) ───────────────

export function formatKw(x) {
  if (x == null) return "—";
  const a = Math.abs(x);
  if (a >= 1e6) return `${(x / 1e6).toFixed(1)} GW`;
  if (a >= 1e3) return `${(x / 1e3).toFixed(1)} MW`;
  return `${x.toFixed(a < 10 ? 1 : 0)} kW`;
}

export function formatKwh(x) {
  if (x == null) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return `${(x / 1e9).toFixed(2)} TWh`;
  if (a >= 1e6) return `${(x / 1e6).toFixed(1)} GWh`;
  if (a >= 1e3) return `${(x / 1e3).toFixed(1)} MWh`;
  return `${x.toFixed(0)} kWh`;
}

export function formatUsd(x) {
  if (x == null) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return `$${(x / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(x / 1e3).toFixed(0)}K`;
  return `$${x.toFixed(0)}`;
}

export function formatPct(x, dp = 1) {
  return x == null ? "—" : `${(x * 100).toFixed(dp)}%`;
}

/** hour-of-year → "Mar 14 · 06:00" (365-day no-leap convention). */
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
export function formatHour(h) {
  let day = Math.floor(h / 24), hod = Math.floor(h % 24), m = 0;
  while (m < 11 && day >= MDAYS[m]) day -= MDAYS[m++];
  return `${MONTHS[m]} ${day + 1} · ${String(hod).padStart(2, "0")}:00`;
}
export function formatDay(h) {
  let day = Math.floor(h / 24), m = 0;
  while (m < 11 && day >= MDAYS[m]) day -= MDAYS[m++];
  return `${MONTHS[m]} ${day + 1}`;
}

// ── theme state ───────────────────────────────────────────────────────

export function isDark() {
  return document.documentElement.dataset.theme === "dark";
}

export function setTheme(mode) {
  document.documentElement.dataset.theme = mode;
  localStorage.setItem("pp-theme", mode);
  document.dispatchEvent(new CustomEvent("pp-theme-change", { detail: mode }));
}

export function initTheme() {
  const saved = localStorage.getItem("pp-theme");
  if (saved) document.documentElement.dataset.theme = saved;
}

// ── ECharts theme registration ────────────────────────────────────────

export const FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace";
export const FONT_UI = "'IBM Plex Sans', ui-sans-serif, sans-serif";

function themeSpec(dark) {
  const ink = dark ? "#E8EAED" : "#2D3436";
  const mid = dark ? "#9AA0A6" : "#636E72";
  const hair = dark ? "#2A2F36" : "#E0E0E0";
  return {
    color: CATEGORY_CYCLE.map((c) => (dark && darkSub[c] ? darkSub[c] : c)),
    backgroundColor: "transparent",
    textStyle: { fontFamily: FONT_UI, color: ink },
    grid: { top: 28, left: 56, right: 16, bottom: 32, containLabel: false },
    categoryAxis: {
      axisLine: { lineStyle: { color: hair } },
      axisTick: { show: false },
      axisLabel: { color: mid, fontFamily: FONT_MONO, fontSize: 10 },
      splitLine: { show: false },
    },
    valueAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: mid, fontFamily: FONT_MONO, fontSize: 10 },
      splitLine: { lineStyle: { color: hair, type: [2, 3], width: 0.7 } },
    },
    legend: {
      textStyle: { color: mid, fontSize: 11, fontFamily: FONT_UI },
      icon: "rect", itemWidth: 10, itemHeight: 10, itemGap: 14,
    },
    tooltip: {
      backgroundColor: dark ? "#1C2026" : "#FFFFFF",
      borderColor: hair,
      borderWidth: 1,
      textStyle: { color: ink, fontSize: 11.5, fontFamily: FONT_MONO },
      extraCssText: "box-shadow: 0 4px 16px rgba(0,0,0,0.14); border-radius: 2px;",
    },
  };
}

let registered = false;
export function registerThemes(echarts) {
  if (registered) return;
  echarts.registerTheme("powerplan-light", themeSpec(false));
  echarts.registerTheme("powerplan-dark", themeSpec(true));
  registered = true;
}

export function currentTheme() {
  return isDark() ? "powerplan-dark" : "powerplan-light";
}

/**
 * Create a chart bound to theme changes and container resize.
 * Returns { chart, setOption, dispose }. On theme flip the chart is rebuilt
 * with the last option (ECharts themes are init-time only).
 */
export function mountChart(echarts, el, buildOption, { group } = {}) {
  registerThemes(echarts);
  let chart = echarts.init(el, currentTheme());
  if (group) { chart.group = group; }
  let lastArgs = null;
  const render = (...args) => {
    lastArgs = args;
    chart.setOption(buildOption(...args), true);
  };
  const onTheme = () => {
    const opt = lastArgs;
    chart.dispose();
    chart = echarts.init(el, currentTheme());
    if (group) { chart.group = group; echarts.connect(group); }
    if (opt) chart.setOption(buildOption(...opt), true);
    handle.chart = chart;
  };
  document.addEventListener("pp-theme-change", onTheme);
  const ro = new ResizeObserver(() => chart.resize());
  ro.observe(el);
  const handle = {
    chart,
    render,
    dispose() {
      document.removeEventListener("pp-theme-change", onTheme);
      ro.disconnect();
      chart.dispose();
    },
  };
  return handle;
}

// ── shared UI fragments ───────────────────────────────────────────────

export const SPEC_TEXT =
  "Speculative technology: micro-fusion and antimatter reactors are modeled at " +
  "parameters that do not exist today. Costs ($9k–27k/kW fusion, $50k–100k/kW " +
  "antimatter) and conversion efficiencies are intentionally optimistic " +
  "what-if inputs — not physics ground truth.";

export function specChip(compact = false) {
  return `<span class="spec-chip" title="${SPEC_TEXT.replaceAll('"', "&quot;")}"
    onclick="location.href='reference.html#speculative'">${compact ? "SPEC" : "SPECULATIVE"}</span>`;
}

export function isSpeculative(name) {
  const n = String(name).toLowerCase();
  return n.includes("fusion") || n.includes("antimatter");
}
