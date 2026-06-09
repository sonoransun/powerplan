/**
 * data.js — fetch + cache the pre-baked JSON under docs/data/.
 * All URLs are relative so the site works under a project-pages prefix.
 */

const SCHEMA_VERSION = 1;
const cache = new Map();

async function fetchJson(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`fetch ${path}: HTTP ${res.status}`);
  return res.json();
}

function checkSchema(doc, path) {
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new Error(
      `${path}: schema_version ${doc.schema_version}, expected ${SCHEMA_VERSION} — re-run scripts/prebake.py`);
  }
  return doc;
}

/** Manifest is fetched fresh each page load (it carries cache-busting hashes). */
export async function getManifest() {
  if (!cache.has("manifest")) {
    cache.set("manifest", fetchJson("data/index.json", { cache: "no-cache" })
      .then((d) => checkSchema(d, "data/index.json")));
  }
  return cache.get("manifest");
}

/** Data files are fetched with a ?v=<hash> from the manifest → safely immutable. */
export async function getData(relPath) {
  if (!cache.has(relPath)) {
    cache.set(relPath, (async () => {
      const manifest = await getManifest();
      const hash = manifest.files?.[relPath];
      const url = hash ? `${relPath}?v=${hash.slice(0, 12)}` : relPath;
      return checkSchema(await fetchJson(url), relPath);
    })());
  }
  return cache.get(relPath);
}

export const getPreset = (key) => getData(`data/presets/${key}.json`);
export const getComparison = () => getData("data/comparison.json");
export const getProjection = (key) => getData(`data/projections/${key}.json`);
export const getResilienceSample = () => getData("data/resilience_sample.json");

/** Average of arr over fixed-size windows (for 24 h rolling panels). */
export function rollingMean(arr, window) {
  if (window <= 1) return arr.slice();
  const out = new Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i] ?? 0;
    if (i >= window) sum -= arr[i - window] ?? 0;
    out[i] = sum / Math.min(i + 1, window);
  }
  return out;
}

/** Sturges-capped histogram bins for the resilience distributions. */
export function histogram(values, maxBins = 20) {
  const v = values.filter((x) => x != null && isFinite(x));
  if (!v.length) return { bins: [], counts: [] };
  const lo = Math.min(...v), hi = Math.max(...v);
  const n = Math.max(1, Math.min(maxBins, Math.ceil(Math.log2(v.length) + 1)));
  const w = (hi - lo) / n || 1;
  const counts = new Array(n).fill(0);
  for (const x of v) counts[Math.min(n - 1, Math.floor((x - lo) / w))]++;
  return { bins: Array.from({ length: n }, (_, i) => [lo + i * w, lo + (i + 1) * w]), counts };
}
