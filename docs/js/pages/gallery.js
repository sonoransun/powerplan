/**
 * pages/gallery.js — Dashboard page (pageId "index").
 *
 * Preset gallery from the manifest (sparkline cards, hash routing #<key>)
 * plus the full pre-baked dashboard via charts/dashboard.js. No Pyodide on
 * this page — everything is served from docs/data/.
 */

import { getManifest, getPreset } from "../data.js";
import { PALETTE, dataColor, formatKw, specChip, SPEC_TEXT } from "../theme.js";
import { renderDashboard } from "../charts/dashboard.js";

const SPEC_FLAG = "pp-spec-ack";
const DEFAULT_KEY = "community";

let manifest = null;
let dash = null;       // current renderDashboard handle
let current = null;    // selected preset key
let loadSeq = 0;       // stale-response guard

let stripEl, summaryEl, calloutEl, dashEl;

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── preset cards ──────────────────────────────────────────────────────

function sparkSvg(spark) {
  const d = (spark && spark.demand_kw) || [];
  const g = (spark && spark.generation_kw) || [];
  const n = Math.max(d.length, g.length);
  if (!n) return `<svg class="spark" aria-hidden="true"></svg>`;
  const max = Math.max(1, ...d, ...g);
  const pts = (arr) => arr
    .map((v, i) => `${((i / Math.max(1, n - 1)) * 100).toFixed(2)},${(32 - (v / max) * 28).toFixed(2)}`)
    .join(" ");
  return `<svg class="spark" viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true">
    <polyline points="${pts(g)}" fill="none" stroke="${dataColor(PALETTE.generation)}"
      stroke-width="1" vector-effect="non-scaling-stroke" opacity="0.85"/>
    <polyline points="${pts(d)}" fill="none" stroke="${dataColor(PALETTE.demand)}"
      stroke-width="1.2" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

function presetCard(p) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "preset-card";
  btn.dataset.key = p.key;
  btn.title = p.description || p.name;
  btn.innerHTML = `
    <div class="name">${esc(p.name)}${p.exotic ? " " + specChip(true) : ""}</div>
    <div class="peak">${esc(p.scale_name)} · ${formatKw(p.peak_load_kw)} peak</div>
    ${sparkSvg(p.spark)}`;
  btn.addEventListener("click", () => {
    if (location.hash.slice(1) !== p.key) location.hash = p.key;
    else select(p.key);
  });
  return btn;
}

// ── states ────────────────────────────────────────────────────────────

function dashSkeletons() {
  return `
    <div class="kpi-strip" style="margin-bottom:16px">
      ${`<div class="skeleton" style="min-height:86px"></div>`.repeat(7)}
    </div>
    <div class="skeleton" style="min-height:380px;margin-bottom:16px"></div>
    <div class="grid">
      <div class="skeleton col-6" style="min-height:300px"></div>
      <div class="skeleton col-6" style="min-height:300px"></div>
    </div>`;
}

function showError(host, err, what) {
  host.innerHTML = `<div class="callout error">
    <strong>Failed to load ${what}.</strong> ${esc((err && err.message) || String(err))}
    <details style="margin-top:8px">
      <summary class="muted" style="cursor:pointer;font-size:12px">details</summary>
      <pre class="traceback">${esc((err && (err.traceback || err.stack)) || String(err))}</pre>
    </details>
  </div>`;
}

function maybeSpecCallout(meta) {
  calloutEl.innerHTML = "";
  if (!meta || !meta.exotic) return;
  let seen = null;
  try { seen = localStorage.getItem(SPEC_FLAG); } catch (e) { /* storage blocked */ }
  if (seen) return;
  const div = document.createElement("div");
  div.className = "callout spec";
  div.innerHTML = `<div style="display:flex;gap:14px;align-items:flex-start">
    <div style="flex:1"><strong>Speculative technology.</strong> ${SPEC_TEXT}
      <a href="reference.html#speculative">Model notes →</a></div>
    <button type="button" class="btn" data-dismiss style="flex:none">Got it</button>
  </div>`;
  div.querySelector("[data-dismiss]").addEventListener("click", () => {
    try { localStorage.setItem(SPEC_FLAG, "1"); } catch (e) { /* storage blocked */ }
    div.remove();
  });
  calloutEl.appendChild(div);
}

function renderSummary(doc) {
  const c = doc.config || {};
  const nS = (c.sources || []).length;
  const nB = (c.storage || []).length;
  const nC = (c.controllers || []).length;
  const island = !(c.grid_interconnect_kw > 0);
  const sep = `<span class="faint" style="margin:0 8px">·</span>`;
  summaryEl.innerHTML = `
    <strong>${esc(c.name || "—")}</strong>
    <span class="muted"> — ${esc((c.scale && c.scale.description) || "")}</span>${sep}
    <span class="num">${nS}</span> source${nS === 1 ? "" : "s"}${sep}
    <span class="num">${nB}</span> storage unit${nB === 1 ? "" : "s"}${sep}
    <span class="num">${nC}</span> controller${nC === 1 ? "" : "s"}${sep}
    ${island
      ? `<span class="muted">island mode — no grid interconnect</span>`
      : `grid-tied <span class="num">±${formatKw(c.grid_interconnect_kw)}</span>`}`;
}

// ── selection ─────────────────────────────────────────────────────────

async function select(key) {
  current = key;
  for (const card of stripEl.querySelectorAll(".preset-card")) {
    const on = card.dataset.key === key;
    card.classList.toggle("on", on);
    if (on) card.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  const meta = manifest.presets.find((p) => p.key === key);
  maybeSpecCallout(meta);
  summaryEl.innerHTML = `<div class="skeleton" style="min-height:22px;max-width:560px"></div>`;
  if (dash) { dash.dispose(); dash = null; }
  dashEl.innerHTML = dashSkeletons();

  const seq = ++loadSeq;
  let doc;
  try {
    doc = await getPreset(key);
  } catch (err) {
    if (seq !== loadSeq) return;
    summaryEl.innerHTML = "";
    showError(dashEl, err, `preset “${key}”`);
    return;
  }
  if (seq !== loadSeq) return;
  renderSummary(doc);
  dash = renderDashboard(dashEl, doc, { figStart: 1, zoomGroup: "index-time" });
}

function onHash() {
  if (!manifest) return;
  const raw = decodeURIComponent(location.hash.slice(1));
  const key = manifest.presets.some((p) => p.key === raw)
    ? raw
    : (manifest.presets.some((p) => p.key === DEFAULT_KEY) ? DEFAULT_KEY
      : (manifest.presets[0] && manifest.presets[0].key));
  if (!key || key === current) return;
  select(key);
}

// ── entry point ───────────────────────────────────────────────────────

export async function init() {
  stripEl = document.getElementById("preset-strip");
  summaryEl = document.getElementById("config-summary");
  calloutEl = document.getElementById("spec-callout");
  dashEl = document.getElementById("dashboard-root");

  try {
    manifest = await getManifest();
  } catch (err) {
    stripEl.innerHTML = "";
    summaryEl.innerHTML = "";
    showError(dashEl, err, "the data manifest (data/index.json)");
    return;
  }

  const ver = document.getElementById("engine-version");
  if (ver) ver.textContent = manifest.engine_version || "?";

  stripEl.innerHTML = "";
  for (const p of manifest.presets || []) stripEl.appendChild(presetCard(p));
  if (!(manifest.presets || []).length) {
    dashEl.innerHTML = `<div class="empty-state">No presets in the manifest — re-run <span class="mono">scripts/prebake.py</span>.</div>`;
    return;
  }

  window.addEventListener("hashchange", onHash);
  onHash();
}
