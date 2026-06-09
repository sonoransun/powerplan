/**
 * state.js — the builder's single mutable store.
 *
 * Shape: { scale_key, name, grid_interconnect_kw, islandMode,
 *          durationDays ∈ 7|30|90|365, sources[], storage[], controllers[] }
 * where each component is { id, type, params:{...} }.
 *
 * Mutations emit "change" events ({kind:"structure"|"param"|"meta"|"reset"}),
 * persist to localStorage "pp-builder-state", and the page mirrors the
 * config into the URL via toUrlHash()/fromUrlHash() (#c= base64url JSON).
 */

import {
  SCALES_INFO, SCALE_ORDER, CLASSNAME_TO_KEY, CONTROLLER_REQUIREMENTS,
  typeDef, defaultParams, suggestController, sourceKw,
  totalGenerationKw, H2_KWH_PER_KG, nice,
} from "./catalog.js";
import { getPreset } from "../data.js";
import { formatKw } from "../theme.js";

export const DURATION_CHOICES = [7, 30, 90, 365];
const LS_KEY = "pp-builder-state";
let nextId = 1;
const newId = () => `c${nextId++}`;

export const state = blank();

function blank() {
  return {
    scale_key: "community",
    name: "Custom build",
    grid_interconnect_kw: SCALES_INFO.community.peak_load_kw * 0.25,
    islandMode: false,
    durationDays: 30,
    sources: [],
    storage: [],
    controllers: [],
  };
}

// ── change notification ───────────────────────────────────────────────

const listeners = new Set();
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function emit(kind) {
  persist();
  for (const fn of listeners) fn(kind);
}

// ── mutations ─────────────────────────────────────────────────────────

export function setMeta(patch) {
  Object.assign(state, patch);
  emit("meta");
}

export function setScale(key) {
  if (!SCALES_INFO[key] || key === state.scale_key) return;
  state.scale_key = key;
  // Re-clamp nothing: keep user numbers, validation flags out-of-band values.
  emit("structure");
}

export function addComponent(category, typeKey, paramsOverride = null) {
  const comp = {
    id: newId(),
    type: typeKey,
    params: paramsOverride ?? defaultParams(category, typeKey, state.scale_key),
  };
  state[category].push(comp);
  emit("structure");
  return comp;
}

export function removeComponent(category, id) {
  const i = state[category].findIndex((c) => c.id === id);
  if (i >= 0) { state[category].splice(i, 1); emit("structure"); }
}

export function duplicateComponent(category, id) {
  const src = state[category].find((c) => c.id === id);
  if (!src) return null;
  const copy = { id: newId(), type: src.type, params: { ...src.params } };
  state[category].splice(state[category].indexOf(src) + 1, 0, copy);
  emit("structure");
  return copy;
}

export function setParam(category, id, name, value) {
  const comp = state[category].find((c) => c.id === id);
  if (!comp) return;
  comp.params[name] = value;
  emit("param");
}

export function resetBlank() {
  Object.assign(state, blank());
  emit("reset");
}

// ── validation ────────────────────────────────────────────────────────

const POSITIVE = new Set(["rated_kw", "capacity_kwh", "h2_tank_kg",
  "electrolyzer_kw", "fuel_cell_kw", "max_power_kw", "units"]);

/**
 * → [{level:"err"|"warn"|"info"|"ok", msg (may contain HTML), fix?}]
 * fix = {label, category, type, params} for one-click adds.
 */
export function validate() {
  const issues = [];
  const peak = SCALES_INFO[state.scale_key].peak_load_kw;
  const genKw = totalGenerationKw(state);

  // errors — no sources
  if (state.sources.length === 0) {
    issues.push({ level: "err", msg: "No generation sources — add at least one source." });
  }

  // errors — NaN / negative / non-positive params
  for (const cat of ["sources", "storage", "controllers"]) {
    for (const comp of state[cat]) {
      const def = typeDef(cat, comp.type);
      if (!def) continue;
      for (const p of def.params) {
        if (p.kind !== "range" && p.kind !== "int") continue;
        const v = comp.params[p.name];
        if (v == null && (p.nullText || !p.scaleSpec)) continue; // auto-linked default
        if (typeof v !== "number" || !isFinite(v)) {
          issues.push({ level: "err", msg: `${def.label}: “${p.label}” is not a number.` });
        } else if (v < 0 || (POSITIVE.has(p.name) && v <= 0)) {
          issues.push({ level: "err",
            msg: `${def.label}: “${p.label}” must be ${POSITIVE.has(p.name) ? "positive" : "≥ 0"}.` });
        }
      }
    }
  }
  if (!isFinite(state.grid_interconnect_kw) || state.grid_interconnect_kw < 0) {
    issues.push({ level: "err", msg: "Grid interconnect must be a number ≥ 0." });
  }

  // errors — wind power-curve cut order
  for (const comp of state.sources.filter((c) => c.type === "wind")) {
    const { cut_in_ms: ci = 3, rated_wind_ms: rw = 12, cut_out_ms: co = 25 } = comp.params;
    if (!(ci < rw && rw < co)) {
      issues.push({ level: "err",
        msg: `Wind turbine: cut-in &lt; rated &lt; cut-out required (got ${ci} / ${rw} / ${co} m/s).` });
    }
  }

  // warnings — missing required controllers
  // With ZERO user controllers the engine auto-sizes MPPT+SiC+bidirectional,
  // which still misses fusion/antimatter/hydrogen/cryogenic requirements.
  const autoCovered = state.controllers.length === 0 ? new Set(["mppt"]) : new Set();
  for (const s of suggestController(state)) {
    if (autoCovered.has(s.type)) continue;
    issues.push({
      level: "warn",
      msg: `${s.requiredBy.join(" + ")} needs a ${s.label} (~${formatKw(s.rated_kw)}).`,
      fix: { label: `Add ${s.label}`, category: "controllers", type: s.type,
             params: { rated_kw: s.rated_kw } },
    });
  }
  if (state.controllers.length === 0 && state.sources.length > 0) {
    issues.push({ level: "info",
      msg: "No controllers specified — the engine will auto-size an MPPT + SiC + bidirectional trio." });
  }

  // warnings — island mode under-generation
  if (state.islandMode && genKw < 0.8 * peak && state.sources.length > 0) {
    issues.push({ level: "warn",
      msg: `Island mode with Σ generation ${formatKw(genKw)} &lt; 0.8 × peak (${formatKw(peak)}) — expect unmet demand.` });
  }

  // info — heavy over-generation
  if (genKw > 2 * peak) {
    issues.push({ level: "info",
      msg: `Σ generation is ${(genKw / peak).toFixed(1)}× peak load — expect heavy curtailment.` });
  }

  // info — speculative tech present
  const spec = [...state.sources, ...state.controllers]
    .filter((c) => /fusion|antimatter/.test(c.type));
  if (spec.length) {
    issues.push({ level: "info",
      msg: `Speculative technology in this build — parameters are optimistic what-ifs, not ground truth. <a href="reference.html#speculative">Model notes</a>` });
  }

  if (!issues.some((i) => i.level === "err")) {
    issues.unshift({ level: "ok", msg: "Configuration is runnable." });
  }
  return issues;
}

export function applyFix(fix) {
  const params = defaultParams(fix.category, fix.type, state.scale_key);
  Object.assign(params, fix.params);
  return addComponent(fix.category, fix.type, params);
}

// ── spec export (webapi.build_custom_config contract) ─────────────────

function cleanParams(comp) {
  const out = {};
  for (const [k, v] of Object.entries(comp.params)) {
    if (v == null) continue; // null = "use Python default" (tilt≡latitude, etc.)
    out[k] = v;
  }
  if (out.units === 1) delete out.units;
  return out;
}

/** The runCustom spec JSON. Controllers omitted entirely when user added none. */
export function toSpec() {
  const spec = {
    name: state.name || "Custom build",
    scale_key: state.scale_key,
    grid_interconnect_kw: state.islandMode ? 0 : Number(state.grid_interconnect_kw) || 0,
    sources: state.sources.map((c) => ({ type: c.type, params: cleanParams(c) })),
    storage: state.storage.map((c) => ({ type: c.type, params: cleanParams(c) })),
  };
  if (state.controllers.length) {
    spec.controllers = state.controllers.map((c) => ({ type: c.type, params: cleanParams(c) }));
  }
  return spec;
}

// ── serialization (URL hash + localStorage) ───────────────────────────

function serializable() {
  const strip = (c) => ({ type: c.type, params: c.params });
  return {
    v: 1,
    scale_key: state.scale_key,
    name: state.name,
    grid_interconnect_kw: state.grid_interconnect_kw,
    islandMode: state.islandMode,
    durationDays: state.durationDays,
    sources: state.sources.map(strip),
    storage: state.storage.map(strip),
    controllers: state.controllers.map(strip),
  };
}

export function loadSerialized(doc) {
  if (!doc || doc.v !== 1) return false;
  const fresh = blank();
  fresh.scale_key = SCALES_INFO[doc.scale_key] ? doc.scale_key : "community";
  fresh.name = String(doc.name ?? "Custom build");
  fresh.grid_interconnect_kw = Number(doc.grid_interconnect_kw) || 0;
  fresh.islandMode = !!doc.islandMode;
  fresh.durationDays = DURATION_CHOICES.includes(doc.durationDays) ? doc.durationDays : 30;
  for (const cat of ["sources", "storage", "controllers"]) {
    fresh[cat] = (Array.isArray(doc[cat]) ? doc[cat] : [])
      .filter((c) => typeDef(cat, c.type))
      .map((c) => ({ id: newId(), type: c.type, params: { ...(c.params ?? {}) } }));
  }
  Object.assign(state, fresh);
  emit("reset");
  return true;
}

export function snapshot() { return JSON.parse(JSON.stringify(serializable())); }

// base64url without unicode pitfalls
function b64uEncode(s) {
  return btoa(unescape(encodeURIComponent(s)))
    .replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}
function b64uDecode(s) {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  return decodeURIComponent(escape(atob(s.replaceAll("-", "+").replaceAll("_", "/") + pad)));
}

export function toUrlHash() {
  return "#c=" + b64uEncode(JSON.stringify(serializable()));
}

/** Parse "#c=..." (defaults to location.hash). True if a config was loaded. */
export function fromUrlHash(hash = window.location.hash) {
  const m = /^#c=([A-Za-z0-9_-]+)/.exec(hash || "");
  if (!m) return false;
  try {
    return loadSerialized(JSON.parse(b64uDecode(m[1])));
  } catch (e) {
    console.warn("builder: bad #c= hash ignored", e);
    return false;
  }
}

function persist() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(serializable())); }
  catch { /* private mode — fine */ }
}

export function restore() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? loadSerialized(JSON.parse(raw)) : false;
  } catch { return false; }
}

// ── presets as templates ──────────────────────────────────────────────

const SCALE_NAME_TO_KEY = {};
for (const k of SCALE_ORDER) SCALE_NAME_TO_KEY[SCALES_INFO[k].label] = k;

function paramsFromName(category, key, name) {
  // The data files carry display names, not constructor args — recover the
  // discrete choices encoded in them (best effort).
  const n = String(name).toLowerCase();
  const p = {};
  if (key === "lithium_ion") {
    if (n.includes("lfp")) p.chemistry = "lfp";
    else if (n.includes("nmc")) p.chemistry = "nmc";
  } else if (key === "liquid_electrolyte") {
    if (n.includes("zinc")) p.chemistry = "zinc_bromine";
  } else if (key === "natural_gas") {
    p.plant_type = n.includes("peaker") ? "peaker" : "ccgt";
  } else if (key === "micro_fusion") {
    if (n.includes("pb11") || n.includes("p-b11")) p.fuel_cycle = "pb11";
    else if (n.includes("dt") || n.includes("d-t")) p.fuel_cycle = "dt";
  } else if (key === "antimatter") {
    for (const t of ["xenon", "lead", "uranium"]) if (n.includes(t)) p.target_atom = t;
  }
  return p;
}

/**
 * Clone a pre-baked preset's config descriptor into the builder.
 * Returns {ok, unmapped:[...]} — unmapped lists details the data file does
 * not carry (those fall back to catalog defaults).
 */
export async function loadPresetTemplate(presetKey) {
  const doc = await getPreset(presetKey);
  const cfg = doc.config;
  const unmapped = [];
  const fresh = blank();

  fresh.scale_key = SCALE_NAME_TO_KEY[cfg.scale?.name] ?? "community";
  fresh.name = `${cfg.name} (copy)`;
  fresh.grid_interconnect_kw = Number(cfg.grid_interconnect_kw) || 0;
  fresh.islandMode = fresh.grid_interconnect_kw === 0;
  fresh.durationDays = state.durationDays;

  for (const s of cfg.sources ?? []) {
    const key = CLASSNAME_TO_KEY.sources[s.type];
    if (!key) { unmapped.push(`source “${s.name}” (${s.type})`); continue; }
    const units = Math.max(1, Number(s.units) || 1);
    const params = defaultParams("sources", key, fresh.scale_key);
    params.rated_kw = nice((Number(s.rated_kw) || 0) / units); // doc carries the total
    params.units = units;
    Object.assign(params, paramsFromName("sources", key, s.name));
    fresh.sources.push({ id: newId(), type: key, params });
  }

  for (const st of cfg.storage ?? []) {
    const key = CLASSNAME_TO_KEY.storage[st.type];
    if (!key) { unmapped.push(`storage “${st.name}” (${st.type})`); continue; }
    const params = defaultParams("storage", key, fresh.scale_key);
    if (key === "hydrogen_fuel_cell") {
      params.h2_tank_kg = nice((Number(st.capacity_kwh) || 0) / H2_KWH_PER_KG);
      params.electrolyzer_kw = nice(Number(st.max_charge_kw) || 0);
      params.fuel_cell_kw = nice(Number(st.max_discharge_kw) || 0);
    } else {
      params.capacity_kwh = Number(st.capacity_kwh) || 0;
      params.max_power_kw = Number(st.max_discharge_kw) || 0;
    }
    params.units = 1; // the doc folds units into the totals
    Object.assign(params, paramsFromName("storage", key, st.name));
    fresh.storage.push({ id: newId(), type: key, params });
  }

  for (const c of cfg.controllers ?? []) {
    const key = CLASSNAME_TO_KEY.controllers[c.type];
    if (!key) { unmapped.push(`controller “${c.name}” (${c.type})`); continue; }
    const params = defaultParams("controllers", key, fresh.scale_key);
    params.rated_kw = Number(c.rated_kw) || 1;
    fresh.controllers.push({ id: newId(), type: key, params });
  }

  unmapped.push("fine-grain preset details (tracking, hub heights, fuel cycles’ Q, reservoirs) are not in the data file — catalog defaults used");

  Object.assign(state, fresh);
  emit("reset");
  return { ok: true, unmapped };
}
