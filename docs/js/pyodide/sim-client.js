/**
 * sim-client.js — main-thread client for the Pyodide simulation worker.
 *
 * Pages import the `sim` singleton and call its run* methods; everything
 * heavy happens in js/pyodide/sim-worker.js (a CLASSIC worker — spawned
 * lazily on first ensureReady()/run*). Contract highlights:
 *
 *   - single-flight: a run* call while another is in flight rejects
 *     immediately with err.name === "BusyError" (ensureReady is exempt).
 *   - cancel(): terminates the worker, rejects the in-flight promise with
 *     err.name === "CancelledError", respawns lazily on the next call.
 *   - after runBatch resolves the worker is auto-recycled (terminate +
 *     lazy re-init) — the wasm heap never shrinks, so batches get a fresh one.
 *   - onProgress: (done, total, phase) with phase ∈ "simulate" | "batchRun"
 *     | "projectionYear". onStatus: (stage) with stage ∈ "loading-pyodide"
 *     | "loading-numpy" | "fetching-package" | "importing" | "ready".
 *   - failures reject with an Error carrying .traceback (string|undefined).
 *
 * Results cross the worker boundary as raw JSON strings and are parsed here,
 * on the main thread, into plain documents (schema v1 — identical shape to
 * the pre-baked files under data/).
 */

import { getManifest } from "../data.js";

const WORKER_URL = new URL("./sim-worker.js", import.meta.url);

const RUNNER_NOTE =
  "PowerPlan in-browser engine: the same powerplan package as the CLI, " +
  "running on Pyodide; emits schema-v1 JSON identical to scripts/prebake.py.";

export class BusyError extends Error {
  constructor(message = "SimClient is busy with another request") {
    super(message);
    this.name = "BusyError";
  }
}

export class CancelledError extends Error {
  constructor(message = "Simulation cancelled") {
    super(message);
    this.name = "CancelledError";
  }
}

/** Feature-detect WebAssembly + workers (Pyodide needs both). */
export function wasmSupported() {
  try {
    if (typeof Worker !== "function") return false;
    if (typeof WebAssembly !== "object" || WebAssembly === null) return false;
    if (typeof WebAssembly.instantiate !== "function") return false;
    // Smallest valid module: "\0asm" + version 1.
    const mod = new WebAssembly.Module(
      Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
    return mod instanceof WebAssembly.Module &&
      new WebAssembly.Instance(mod) instanceof WebAssembly.Instance;
  } catch (_err) {
    return false;
  }
}

export class SimClient {
  busy = false;
  ready = false;
  onStatus = null;            // (stage) => {}

  #worker = null;
  #pending = new Map();       // id → {resolve, reject, onProgress}
  #nextId = 1;
  #initPromise = null;
  #epoch = 0;                 // bumped by cancel(); stale async work checks it

  /** Lazy spawn + boot; resolves when the engine is importable. Safe to call
   *  repeatedly and concurrently — all callers share one boot promise. */
  async ensureReady() {
    if (this.ready && this.#worker) return;
    if (!this.#initPromise) this.#initPromise = this.#init();
    return this.#initPromise;
  }

  async runPreset({ key, hours = 720, dtHours = 1.0 } = {}, onProgress) {
    if (!key) throw new TypeError("runPreset: 'key' is required");
    return this.#run("runPreset", { key, hours, dtHours }, onProgress);
  }

  async runCustom({ spec, hours = 720, dtHours = 1.0 } = {}, onProgress) {
    if (!spec || typeof spec !== "object") {
      throw new TypeError("runCustom: 'spec' object is required");
    }
    return this.#run("runCustom", { spec, hours, dtHours }, onProgress);
  }

  /** base: {preset: "district"} | {spec: {...}} → {baseline, failed} docs. */
  async runFailure({ base, events, hours = 720, seed = 42 } = {}, onProgress) {
    if (!base || typeof base !== "object") {
      throw new TypeError("runFailure: 'base' ({preset}|{spec}) is required");
    }
    if (!Array.isArray(events)) {
      throw new TypeError("runFailure: 'events' array is required");
    }
    return this.#run("runFailure", { base, events, hours, seed }, onProgress);
  }

  async runBatch({ scale, tier, seed, nConfigs, nFailures,
                   hours = 720, includeBaseline = true } = {}, onProgress) {
    if (nConfigs == null || nFailures == null) {
      throw new TypeError("runBatch: 'nConfigs' and 'nFailures' are required");
    }
    const doc = await this.#run("runBatch",
      { scale, tier, seed, nConfigs, nFailures, hours, includeBaseline },
      onProgress);
    // Batches churn the wasm heap, which never shrinks — recycle the worker
    // (terminate now, respawn lazily on the next call).
    this.#recycle();
    return doc;
  }

  async runProjection({ profile, climate = null, years,
                        simHours = 720, baseYear = 2025 } = {}, onProgress) {
    if (!profile) throw new TypeError("runProjection: 'profile' is required");
    if (!Array.isArray(years) || !years.length) {
      throw new TypeError("runProjection: 'years' array is required");
    }
    return this.#run("runProjection",
      { profile, climate, years, simHours, baseYear }, onProgress);
  }

  /** Hard-stop the in-flight run: terminate the worker, reject pending
   *  promises with CancelledError; the next call respawns lazily. */
  cancel() {
    this.#epoch++;
    const entries = Array.from(this.#pending.values());
    this.#pending.clear();
    this.#teardown();
    this.#initPromise = null;
    this.busy = false;
    const err = new CancelledError();
    for (const entry of entries) entry.reject(err);
  }

  // ── internals ───────────────────────────────────────────────────────

  async #init() {
    const epoch = this.#epoch;
    try {
      // Spawn first: the worker starts importScripts(pyodide) immediately,
      // overlapping the big CDN download with the manifest fetch below.
      const worker = new Worker(WORKER_URL);
      this.#worker = worker;
      worker.onmessage = (e) => this.#handleMessage(worker, e.data || {});
      worker.onerror = (e) => this.#handleWorkerFailure(worker, e);
      worker.onmessageerror = () =>
        this.#handleWorkerFailure(worker, { message: "unreadable worker message" });

      const zipUrl = await this.#resolveZipUrl();
      if (this.#epoch !== epoch) throw new CancelledError();

      await this.#post("init", { zipUrl, runnerNote: RUNNER_NOTE });
      // this.ready is flipped by the "ready" message handler.
    } catch (err) {
      if (this.#epoch === epoch) {   // not already cleaned up by cancel()
        this.#initPromise = null;
        this.#teardown();
      }
      throw err;
    }
  }

  /** Absolute package URL, cache-busted with the manifest's pkg hash —
   *  data/index.json itself is fetched with cache:"no-cache" (data.js). */
  async #resolveZipUrl() {
    const manifest = await getManifest();
    const pkg = (manifest && manifest.pkg) || {};
    const rel = pkg.zip || "py/powerplan-pkg.zip";
    const suffix = pkg.sha256 ? `?v=${String(pkg.sha256).slice(0, 12)}` : "";
    return new URL(rel + suffix, document.baseURI).href;
  }

  async #run(op, payload, onProgress) {
    if (this.busy) {
      throw new BusyError(
        "SimClient is busy — wait for the current run or cancel() it");
    }
    this.busy = true;
    try {
      await this.ensureReady();
      const raw = await this.#post(op, payload, onProgress);
      return JSON.parse(raw);   // raw JSON string crosses; parsed here only
    } finally {
      this.busy = false;
    }
  }

  #post(op, payload, onProgress) {
    const worker = this.#worker;
    if (!worker) return Promise.reject(new CancelledError("worker was terminated"));
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ id, op, payload });
    });
  }

  #handleMessage(worker, msg) {
    if (worker !== this.#worker) return;   // stale worker post-cancel/recycle
    const entry = msg.id != null ? this.#pending.get(msg.id) : undefined;
    switch (msg.kind) {
      case "status":
        this.onStatus?.(msg.stage);
        break;
      case "progress":
        if (entry && typeof entry.onProgress === "function") {
          entry.onProgress(msg.done, msg.total, msg.phase);
        }
        break;
      case "ready":
        this.ready = true;
        this.onStatus?.("ready");
        if (entry) {
          this.#pending.delete(msg.id);
          entry.resolve();
        }
        break;
      case "result":
        if (entry) {
          this.#pending.delete(msg.id);
          entry.resolve(msg.result);
        }
        break;
      case "error": {
        const err = new Error(msg.message || "simulation worker error");
        if (msg.traceback) err.traceback = msg.traceback;
        if (entry) {
          this.#pending.delete(msg.id);
          entry.reject(err);
        } else {
          console.error("[sim-client] worker error:", msg.message,
            msg.traceback || "");
        }
        break;
      }
      default:
        break;
    }
  }

  /** Worker script crash (failed importScripts, OOM, …) — fail everything
   *  in flight and reset so the next call respawns from scratch. */
  #handleWorkerFailure(worker, event) {
    if (worker !== this.#worker) return;
    const detail = event && event.message ? `: ${event.message}` : "";
    const err = new Error(`simulation worker crashed${detail}`);
    const entries = Array.from(this.#pending.values());
    this.#pending.clear();
    this.#teardown();
    this.#initPromise = null;
    for (const entry of entries) entry.reject(err);
  }

  #teardown() {
    const worker = this.#worker;
    this.#worker = null;
    this.ready = false;
    if (worker) {
      worker.onmessage = null;
      worker.onerror = null;
      worker.onmessageerror = null;
      try { worker.terminate(); } catch (_err) { /* already dead */ }
    }
  }

  /** Terminate now; respawn lazily on the next ensureReady()/run*. */
  #recycle() {
    this.#teardown();
    this.#initPromise = null;
  }
}

export const sim = new SimClient();
