/**
 * sim-worker.js — CLASSIC web worker hosting the Pyodide simulation engine.
 *
 * This file must stay a classic script (no import/export): it is spawned by
 * js/pyodide/sim-client.js with `new Worker(url)` and pulls Pyodide in via
 * importScripts. All cross-thread traffic is structured-cloneable plain data:
 *
 *   client → worker : {id, op, payload}
 *     op "init"          payload {zipUrl, runnerNote}   (zipUrl is ABSOLUTE)
 *     op "runPreset" | "runCustom" | "runFailure" | "runBatch" | "runProjection"
 *                        payload = camelCase args from SimClient
 *   worker → client : {id, kind, ...}
 *     kind "status"      {stage}                        (boot stages)
 *     kind "progress"    {done, total, phase}
 *     kind "result"      {result}                       (raw JSON *string*;
 *                                                        parsed on the main
 *                                                        thread, not here)
 *     kind "ready"       {}
 *     kind "error"       {message, traceback}
 *
 * Memory discipline: the Python `dispatch` PyProxy is created ONCE and never
 * destroyed; every result crosses the boundary as a Python str → JS string
 * (converted by value), so no PyProxies leak per run. The client terminates
 * this worker outright after heavy batch runs — the wasm heap never shrinks.
 */

"use strict";

importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.js");

let pyodide = null;
let dispatchFn = null;     // PyProxy of Python dispatch() — kept forever
let initStarted = false;

// Tag progress messages with the request they belong to. reportProgress is
// called synchronously from Python mid-dispatch; postMessage still delivers.
let currentId = null;
let currentPhase = null;

const PHASES = {
  runPreset: "simulate",
  runCustom: "simulate",
  runFailure: "simulate",
  runBatch: "batchRun",
  runProjection: "projectionYear",
};

// Python reaches this via `from js import reportProgress`.
self.reportProgress = function (done, total) {
  self.postMessage({
    id: currentId,
    kind: "progress",
    done: Number(done),
    total: Number(total),
    phase: currentPhase,
  });
};

// Bootstrap source, run exactly once after the package zip is unpacked.
// payload_json arrives as a JS string → json.loads gives a pure Python dict
// (no JsProxy), so json.dumps(p["spec"]) etc. just work. Keys here are the
// client's camelCase; webapi's payloads are snake_case — mapped explicitly.
const BOOTSTRAP_PY = `
import json
import powerplan.webapi as webapi
from js import reportProgress


def _progress(done, total):
    reportProgress(done, total)


def dispatch(op, payload_json):
    p = json.loads(payload_json)
    if op == "runPreset":
        return webapi.run_preset(
            p["key"],
            hours=int(p.get("hours", 720)),
            dt_hours=float(p.get("dtHours", 1.0)),
            progress=_progress,
        )
    if op == "runCustom":
        return webapi.run_custom(
            json.dumps(p["spec"]),
            hours=int(p.get("hours", 720)),
            dt_hours=float(p.get("dtHours", 1.0)),
            progress=_progress,
        )
    if op == "runFailure":
        return webapi.run_failure(
            json.dumps({
                "base": p["base"],
                "events": p["events"],
                "hours": int(p.get("hours", 720)),
                "seed": int(p.get("seed", 42)),
            }),
            progress=_progress,
        )
    if op == "runBatch":
        return webapi.run_scenario_batch(
            json.dumps({
                "scale": p.get("scale", "community"),
                "tier": p.get("tier", "conventional"),
                "seed": int(p.get("seed", 42)),
                "n_configs": int(p["nConfigs"]),
                "n_failures": int(p["nFailures"]),
                "hours": int(p.get("hours", 720)),
                "include_baseline": bool(p.get("includeBaseline", True)),
            }),
            progress=_progress,
        )
    if op == "runProjection":
        return webapi.run_projection(
            json.dumps({
                "profile": p["profile"],
                "climate": p.get("climate"),
                "years": p["years"],
                "sim_hours": int(p.get("simHours", 720)),
                "base_year": int(p.get("baseYear", 2025)),
            }),
            progress=_progress,
        )
    raise ValueError(op)
`;

function postStatus(id, stage) {
  self.postMessage({ id: id, kind: "status", stage: stage });
}

/** Pyodide PythonError detection — its .message IS the full traceback. */
function isPythonError(err) {
  if (!err) return false;
  if (err.constructor && err.constructor.name === "PythonError") return true;
  if (err.type === "PythonError") return true;
  return typeof err.message === "string" &&
    err.message.indexOf("Traceback (most recent call last)") !== -1;
}

/** Short human-readable line: for tracebacks, the final exception line. */
function shortMessage(err) {
  if (isPythonError(err)) {
    const lines = String(err.message).trimEnd().split("\n");
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line) return line;
    }
  }
  return (err && err.message) ? String(err.message) : String(err);
}

/** Full detail: PythonError.message is the traceback; otherwise the JS stack. */
function tracebackOf(err) {
  if (isPythonError(err)) return String(err.message);
  return (err && err.stack) ? String(err.stack) : "";
}

async function initialize(id, payload) {
  const zipUrl = payload && payload.zipUrl;
  if (!zipUrl) throw new Error("init message missing zipUrl");
  if (payload.runnerNote) console.log("[sim-worker]", payload.runnerNote);

  postStatus(id, "loading-pyodide");
  pyodide = await loadPyodide();
  // Progress prints from ScenarioRunner / GrowthProjection land in the console.
  pyodide.setStdout({ batched: (s) => console.log("[py]", s) });
  pyodide.setStderr({ batched: (s) => console.warn("[py]", s) });

  postStatus(id, "loading-numpy");
  await pyodide.loadPackage("numpy");

  postStatus(id, "fetching-package");
  const resp = await fetch(zipUrl);
  if (!resp.ok) throw new Error("fetch " + zipUrl + ": HTTP " + resp.status);
  const buf = await resp.arrayBuffer();

  postStatus(id, "importing");
  pyodide.unpackArchive(buf, "zip");
  pyodide.runPython(BOOTSTRAP_PY);
  dispatchFn = pyodide.globals.get("dispatch"); // hold forever, never .destroy()

  self.postMessage({ id: id, kind: "ready" });
}

async function handleMessage(msg) {
  const id = msg.id;
  try {
    if (msg.op === "init") {
      if (dispatchFn) {                 // already booted (defensive; the
        self.postMessage({ id: id, kind: "ready" });  // client never re-inits
        return;                                       // a live worker)
      }
      if (initStarted) throw new Error("init already in progress");
      initStarted = true;
      try {
        await initialize(id, msg.payload || {});
      } catch (err) {
        initStarted = false;            // allow a retry message
        throw err;
      }
      return;
    }

    if (!dispatchFn) throw new Error("worker not initialized — send init first");
    if (!(msg.op in PHASES)) throw new Error("unknown op: " + String(msg.op));

    currentId = id;
    currentPhase = PHASES[msg.op];
    // Synchronous Python call; returns a Python str → arrives as a JS string.
    const result = dispatchFn(msg.op, JSON.stringify(msg.payload || {}));
    self.postMessage({ id: id, kind: "result", result: result });
  } catch (err) {
    self.postMessage({
      id: id,
      kind: "error",
      message: shortMessage(err),
      traceback: tracebackOf(err),
    });
  } finally {
    currentId = null;
    currentPhase = null;
  }
}

self.onmessage = function (event) {
  handleMessage(event.data || {});
};
