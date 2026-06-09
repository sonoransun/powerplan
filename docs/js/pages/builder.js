/**
 * pages/builder.js — Builder page controller.
 *
 * Left rail (builder/form.js) mutates builder/state.js; this module owns the
 * right column: live system-diagram preview, validation readout, run controls,
 * the Pyodide run flow (js/pyodide/sim-client.js) and the results dashboard
 * (js/charts/dashboard.js).
 */

import { sim, wasmSupported } from "../pyodide/sim-client.js";
import { renderDashboard } from "../charts/dashboard.js";
import { renderSystemDiagram } from "../components/system-diagram.js";
import {
  state, subscribe, setMeta, validate, applyFix, toSpec, toUrlHash,
  fromUrlHash, restore, snapshot, loadSerialized, DURATION_CHOICES,
} from "../builder/state.js";
import { renderForm } from "../builder/form.js";
import {
  SCALES_INFO, typeDef, sourceKw, storageEnergyKwh, storagePowerKw,
  totalGenerationKw, totalStoragePowerKw, nice,
} from "../builder/catalog.js";
import { formatKw } from "../theme.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

const RUNTIME_EST = { 7: "~0.1 s", 30: "~0.3 s", 90: "~1 s", 365: "~4 s" };
const TOPOLOGY = { sic: "full_bridge", gan: "half_bridge", mppt: "boost",
  bidirectional: "h_bridge", hydrogen: "interleaved_boost", fusion: "multi_stage",
  cryogenic: "resonant_converter", antimatter: "triple_pathway" };

let dashHandle = null;       // renderDashboard disposer
let lastGood = null;         // serialized snapshot of the last successful run's config
let hasRun = false;
let currentFixes = [];

export function init() {
  // config precedence: shared URL > saved session > blank
  if (!fromUrlHash()) restore();

  renderForm($("b-rail"));
  buildDurationSeg();
  wireRunControls();
  wireShare();
  wireStatusChip();

  subscribe(onStateChange);
  onStateChange("reset");
  document.addEventListener("pp-theme-change", () => renderPreview());

  if (!wasmSupported()) {
    $("b-run").style.display = "none";
    $("b-duration").closest(".field").style.opacity = "0.4";
    $("b-wasm-note").innerHTML = `
      <div class="callout error" style="margin:12px 0 0;">
        <strong>WebAssembly is unavailable in this browser</strong> — live runs are
        disabled. Pre-baked results for the ten presets are on the
        <a href="index.html">Dashboard</a> page.</div>`;
  }
}

// ── state → right column ──────────────────────────────────────────────

let hashTimer = null;

function onStateChange(kind) {
  renderPreview();
  renderValidation();
  syncRunControls();
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    try { history.replaceState(null, "", toUrlHash()); } catch { /* file:// */ }
  }, 250);
  if (kind === "reset") clearError();
}

/** A SimDoc-shaped config descriptor so the shared diagram can draw the preview. */
function docLike() {
  const scale = SCALES_INFO[state.scale_key];
  const named = (cat, comp, i, n) => {
    const def = typeDef(cat, comp.type);
    return n > 1 ? `${def.label} #${i + 1}` : def.label;
  };
  const counts = {};
  for (const cat of ["sources", "storage", "controllers"])
    for (const c of state[cat]) counts[`${cat}:${c.type}`] = (counts[`${cat}:${c.type}`] ?? 0) + 1;
  const idx = {};
  const nm = (cat, comp) => {
    const k = `${cat}:${comp.type}`;
    idx[k] = (idx[k] ?? 0) + 1;
    return named(cat, comp, idx[k] - 1, counts[k]);
  };

  let controllers = state.controllers.map((c) => ({
    name: nm("controllers", c), type: typeDef("controllers", c.type).className,
    rated_kw: Number(c.params.rated_kw) || 0, topology: TOPOLOGY[c.type] ?? "full_bridge",
  }));
  if (!controllers.length && state.sources.length) {
    // mirror webapi.build_custom_config's auto-sized default trio
    const g = Math.max(totalGenerationKw(state), 1);
    const p = Math.max(totalStoragePowerKw(state), 1);
    controllers = [
      { name: "MPPT Controller (auto)", type: "MPPTController", rated_kw: nice(g), topology: "boost" },
      { name: "SiC Converter (auto)", type: "SiCConverter", rated_kw: nice(p), topology: "full_bridge" },
      { name: "Bidir Inverter (auto)", type: "BidirectionalInverter", rated_kw: nice(p), topology: "h_bridge" },
    ];
  }

  return {
    config: {
      name: state.name, preset_key: null,
      scale: { name: scale.label, peak_load_kw: scale.peak_load_kw,
        annual_consumption_kwh: scale.annual_consumption_kwh,
        num_endpoints: scale.num_endpoints, description: scale.description },
      grid_interconnect_kw: state.islandMode ? 0 : Number(state.grid_interconnect_kw) || 0,
      sources: state.sources.map((c) => ({
        name: nm("sources", c), type: typeDef("sources", c.type).className,
        rated_kw: sourceKw(c), units: Number(c.params.units) || 1,
        is_renewable: typeDef("sources", c.type).renewable !== false,
      })),
      storage: state.storage.map((c) => {
        const def = typeDef("storage", c.type);
        const units = Number(c.params.units) || 1;
        const charge = c.type === "hydrogen_fuel_cell"
          ? (Number(c.params.electrolyzer_kw) || 0) * units : storagePowerKw(c);
        return { name: nm("storage", c), type: def.className,
          capacity_kwh: storageEnergyKwh(c),
          max_charge_kw: charge, max_discharge_kw: storagePowerKw(c) };
      }),
      controllers,
    },
  };
}

function renderPreview() {
  const el = $("b-sysdiag");
  const gen = totalGenerationKw(state);
  const peak = SCALES_INFO[state.scale_key].peak_load_kw;
  $("b-capline").innerHTML =
    `Σ gen ${formatKw(gen)} <span class="unit">vs</span> peak ${formatKw(peak)}` +
    ` <span class="unit">(${peak > 0 ? (gen / peak).toFixed(2) : "—"}×)</span>`;

  if (!state.sources.length && !state.storage.length) {
    el.innerHTML = `<div class="empty-state" style="padding:28px 16px;">
      Nothing to draw yet — add a source from the rail.</div>`;
    return;
  }
  try {
    renderSystemDiagram(el, docLike().config, null);
  } catch (e) {
    // diagram module unavailable or choked — degrade to a text manifest
    el.innerHTML = `<p class="muted" style="font-size:12px;">
      ${state.sources.length} source(s) · ${state.storage.length} storage unit(s) ·
      ${state.controllers.length || "auto"} controller(s)</p>`;
    console.warn("builder: system diagram failed", e);
  }
}

function renderValidation() {
  const ul = $("b-validation");
  const issues = validate();
  currentFixes = [];
  ul.innerHTML = issues.map((i) => {
    let fixBtn = "";
    if (i.fix) {
      const idx = currentFixes.push(i.fix) - 1;
      fixBtn = ` <button class="btn" data-fix="${idx}"
        style="padding:2px 9px;font-size:11.5px;margin-left:6px;">${esc(i.fix.label)}</button>`;
    }
    return `<li class="v-${i.level}">${i.msg}${fixBtn}</li>`;
  }).join("");
  for (const b of ul.querySelectorAll("[data-fix]")) {
    b.addEventListener("click", () => applyFix(currentFixes[Number(b.dataset.fix)]));
  }
}

// ── run controls ──────────────────────────────────────────────────────

function buildDurationSeg() {
  const seg = $("b-duration");
  seg.innerHTML = DURATION_CHOICES.map((d) =>
    `<button data-days="${d}">${d} d</button>`).join("");
  for (const b of seg.querySelectorAll("button")) {
    b.addEventListener("click", () => setMeta({ durationDays: Number(b.dataset.days) }));
  }
}

// grid slider: log-ish cubic mapping 0..1000 → 0..4×peak
const gridFromT = (t, peak) => nice(4 * peak * Math.pow(t / 1000, 3));
const tFromGrid = (kw, peak) => Math.round(1000 * Math.cbrt(Math.max(kw, 0) / (4 * peak)));

function wireRunControls() {
  const slider = $("b-grid"), island = $("b-island");
  slider.addEventListener("input", () => {
    const peak = SCALES_INFO[state.scale_key].peak_load_kw;
    $("b-grid-val").textContent = formatKw(gridFromT(Number(slider.value), peak));
  });
  slider.addEventListener("change", () => {
    const peak = SCALES_INFO[state.scale_key].peak_load_kw;
    setMeta({ grid_interconnect_kw: gridFromT(Number(slider.value), peak) });
  });
  island.addEventListener("change", () => setMeta({ islandMode: island.checked }));
  $("b-run").addEventListener("click", run);
  $("b-to-resilience").addEventListener("click", () => {
    try { localStorage.setItem("pp-builder-config", JSON.stringify(toSpec())); } catch { }
    window.location.href = "resilience.html";
  });
}

function syncRunControls() {
  const peak = SCALES_INFO[state.scale_key].peak_load_kw;
  const seg = $("b-duration");
  for (const b of seg.querySelectorAll("button")) {
    b.classList.toggle("on", Number(b.dataset.days) === state.durationDays);
  }
  $("b-est").textContent =
    `runtime ${RUNTIME_EST[state.durationDays] ?? "~?"} after engine init`;
  const slider = $("b-grid");
  slider.value = tFromGrid(Number(state.grid_interconnect_kw) || 0, peak);
  slider.disabled = state.islandMode;
  $("b-grid-val").textContent = state.islandMode
    ? "0 kW (island)" : formatKw(Number(state.grid_interconnect_kw) || 0);
  $("b-island").checked = state.islandMode;
}

// ── status chip (engine init stages) ──────────────────────────────────

const STAGES = {
  "loading-pyodide": ["Loading Pyodide runtime…", 15],
  "loading-numpy": ["Loading numpy…", 45],
  "fetching-package": ["Fetching powerplan engine…", 70],
  "importing": ["Importing powerplan…", 90],
  "ready": ["Engine ready", 100],
};

function wireStatusChip() {
  sim.onStatus = (stage) => {
    const [label, pct] = STAGES[stage] ?? [stage, 50];
    const chip = $("b-status");
    $("b-status-text").textContent = label;
    $("b-status-bar").style.width = `${pct}%`;
    chip.classList.remove("fade");
    if (stage === "ready") setTimeout(() => chip.classList.add("fade"), 1600);
  };
}

// ── run flow ──────────────────────────────────────────────────────────

async function run() {
  if (sim.busy) return; // double-run guard
  const issues = validate();
  if (issues.some((i) => i.level === "err")) {
    $("b-validation-panel").scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  clearError();

  const hours = state.durationDays * 24;
  const days = state.durationDays;
  const btn = $("b-run"), prog = $("b-progress");
  btn.disabled = true;
  btn.style.display = "none";
  prog.style.display = "flex";
  $("b-progress-label").textContent = "Simulating… day 0/" + days;
  $("b-progress-bar").style.width = "0%";

  const results = $("b-results");
  if (!hasRun) {
    results.innerHTML = `
      <div class="skeleton" style="min-height:96px;margin-bottom:16px;"></div>
      <div class="skeleton" style="min-height:320px;margin-bottom:16px;"></div>
      <div class="skeleton" style="min-height:260px;"></div>`;
  }

  try {
    const doc = await sim.runCustom(
      { spec: toSpec(), hours, dtHours: 1.0 },
      (done, total) => {
        const day = Math.min(days, Math.ceil((done / Math.max(total, 1)) * days));
        $("b-progress-label").textContent = `Simulating… day ${day}/${days}`;
        $("b-progress-bar").style.width = `${(100 * done / Math.max(total, 1)).toFixed(1)}%`;
      });

    lastGood = snapshot();
    hasRun = true;
    if (dashHandle) { try { dashHandle.dispose(); } catch { } dashHandle = null; }
    results.innerHTML = `<div id="b-dash"></div><div id="b-lcoe-note"></div>`;
    dashHandle = renderDashboard(results.querySelector("#b-dash"), doc,
      { figStart: 2, zoomGroup: "builder-time" });
    if (hours < 8760) {
      results.querySelector("#b-lcoe-note").innerHTML = `
        <p class="muted" style="font-size:11.5px;margin:10px 2px 0;">
          <span class="num">LCOE</span> — annualized estimate; short runs overstate
          LCOE ~${Math.round(8760 / hours)}× (capital is amortized over hours simulated,
          see <a href="reference.html#lcoe">metric notes</a>).</p>`;
    }
  } catch (err) {
    if (err?.name !== "CancelledError") showError(err);
    if (!hasRun) {
      results.innerHTML = `<div class="empty-state">
        <p style="margin:0;">No results — the run did not complete.</p></div>`;
    }
  } finally {
    btn.disabled = false;
    btn.style.display = "";
    prog.style.display = "none";
  }
}

// ── error surface ─────────────────────────────────────────────────────

function clearError() { $("b-error").innerHTML = ""; }

function showError(err) {
  const tb = err?.traceback ? `
    <details style="margin-top:8px;">
      <summary class="muted" style="cursor:pointer;font-size:12px;">Python traceback</summary>
      <pre class="traceback" style="margin:8px 0 0;">${esc(err.traceback)}</pre>
    </details>` : "";
  $("b-error").innerHTML = `
    <div class="callout error">
      <strong>Run failed.</strong>
      <span class="muted">${esc(err?.message ?? String(err))}</span>${tb}
      ${lastGood ? `<div style="margin-top:10px;">
        <button class="btn" id="b-restore">Restore last good config</button></div>` : ""}
    </div>`;
  $("b-restore")?.addEventListener("click", () => {
    loadSerialized(lastGood);
    clearError();
  });
}

// ── share link ────────────────────────────────────────────────────────

function wireShare() {
  const btn = $("b-share");
  btn.addEventListener("click", async () => {
    const url = window.location.origin + window.location.pathname + toUrlHash();
    try { history.replaceState(null, "", toUrlHash()); } catch { }
    let ok = false;
    try { await navigator.clipboard.writeText(url); ok = true; } catch { }
    btn.textContent = ok ? "Copied ✓" : "Copy failed — use the URL bar";
    setTimeout(() => { btn.textContent = "Copy share link"; }, 2200);
  });
}
