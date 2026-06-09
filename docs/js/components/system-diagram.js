/**
 * components/system-diagram.js — data-driven SVG system architecture.
 *
 * renderSystemDiagram(container, configDesc, metrics|null)
 *   configDesc = simulation doc `config` object; metrics enriches the hover
 *   popover (cf / health / cycles / losses) when available.
 *
 * Four columns — Sources → Controllers → Bus → Load/Grid — with the storage
 * row hung under the bus. Bezier connector width ∝ log(rated kW), clamped
 * 1.5–10. Under 640 px the layout reflows vertically (viewBox scales).
 * Colors come from theme.js; UI chrome from CSS variables.
 */

import {
  PALETTE, dataColor, sourceColor, storageColor,
  formatKw, formatKwh, formatUsd, formatPct, isSpeculative,
} from "../theme.js";

const NODE_H = 46;
const GAP = 14;

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function trunc(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s;
}

/** connector width ∝ log(rated kW), clamped to [1.5, 10]. */
function strokeW(kw) {
  return Math.max(1.5, Math.min(10, 1.15 * Math.log10(Math.max(kw || 0, 1) + 1)));
}

function cyc(c) {
  return c == null ? "—" : c < 10 ? c.toFixed(1) : String(Math.round(c));
}

// ── model ─────────────────────────────────────────────────────────────

function buildModel(cfg, metrics) {
  const md = metrics || {};
  const find = (list, name) => (list || []).find((d) => d.name === name) || null;
  const sources = (cfg.sources || []).map((s) => ({
    kind: "source", name: s.name, color: sourceColor(s.name), kw: s.rated_kw || 0,
    sub: (s.units || 1) > 1 ? `${s.units} × ${formatKw((s.rated_kw || 0) / s.units)}` : formatKw(s.rated_kw),
    det: find(md.source_details, s.name), raw: s,
  }));
  const controllers = (cfg.controllers || []).map((c) => ({
    kind: "controller", name: c.name, color: dataColor(PALETTE.loss), kw: c.rated_kw || 0,
    sub: `${formatKw(c.rated_kw)} · ${c.topology || "—"}`,
    det: find(md.controller_details, c.name), raw: c,
  }));
  const storage = (cfg.storage || []).map((b) => ({
    kind: "storage", name: b.name, color: storageColor(b.name),
    kw: Math.max(b.max_charge_kw || 0, b.max_discharge_kw || 0),
    sub: formatKwh(b.capacity_kwh),
    det: find(md.storage_details, b.name), raw: b,
  }));
  const load = {
    kind: "load", name: "Load", color: dataColor(PALETTE.demand),
    kw: (cfg.scale && cfg.scale.peak_load_kw) || 0,
    sub: `${formatKw(cfg.scale && cfg.scale.peak_load_kw)} peak`,
  };
  const island = !(cfg.grid_interconnect_kw > 0);
  const grid = island
    ? { kind: "grid", name: "Island", color: dataColor(PALETTE.loss), kw: 0, sub: "no interconnect", island: true }
    : { kind: "grid", name: "Grid", color: dataColor(PALETTE.grid_import), kw: cfg.grid_interconnect_kw, sub: `±${formatKw(cfg.grid_interconnect_kw)}` };
  return { sources, controllers, storage, load, grid };
}

// ── layouts ───────────────────────────────────────────────────────────

function placeCol(list, x, y0, w) {
  list.forEach((n, i) => { n.x = x; n.y = y0 + i * (NODE_H + GAP); n.w = w; n.h = NODE_H; });
}

function layoutH(m) {
  const VBW = 960, colW = 176;
  const sx = 18, cx = 268, busX = 560, rx = 742, top = 46;
  const colH = (n) => (n ? n * (NODE_H + GAP) - GAP : 0);
  const mainH = Math.max(colH(m.sources.length), colH(m.controllers.length), colH(2), 120);
  placeCol(m.sources, sx, top + (mainH - colH(m.sources.length)) / 2, colW);
  placeCol(m.controllers, cx, top + (mainH - colH(m.controllers.length)) / 2, colW);
  placeCol([m.load, m.grid], rx, top + (mainH - colH(2)) / 2, colW);
  const busTop = top - 6, busBot = top + mainH + 6;
  let vbH = busBot + 20;
  if (m.storage.length) {
    const n = m.storage.length;
    const sw = Math.min(160, Math.floor((VBW - 60) / n) - 10);
    const rowW = n * (sw + 10) - 10;
    const x0 = Math.min(Math.max(20, busX - rowW / 2), VBW - 20 - rowW);
    m.storage.forEach((b, i) => { b.x = x0 + i * (sw + 10); b.y = busBot + 56; b.w = sw; b.h = NODE_H; });
    vbH = busBot + 56 + NODE_H + 18;
  }
  return { VBW, vbH, busX, busTop, busBot, vertical: false };
}

function placeGrid(list, x0, y0, w, perRow) {
  list.forEach((n, i) => {
    n.x = x0 + (i % perRow) * (w + 12);
    n.y = y0 + Math.floor(i / perRow) * (NODE_H + 12);
    n.w = w; n.h = NODE_H;
  });
  return list.length ? y0 + Math.ceil(list.length / perRow) * (NODE_H + 12) - 12 : y0;
}

function layoutV(m) {
  const VBW = 520;
  const w = (VBW - 40 - 12) / 2;
  let y = 34;
  if (m.sources.length) {
    placeGrid(m.sources, 20, y, w, 2);
    y = m.sources[m.sources.length - 1].y + NODE_H + 44;
  } else {
    y += 30;
  }
  if (m.controllers.length) {
    placeGrid(m.controllers, 20, y, w, 2);
    y = m.controllers[m.controllers.length - 1].y + NODE_H + 44;
  }
  const busY = y;
  y += 52;
  if (m.storage.length) {
    placeGrid(m.storage, 20, y, w, 2);
    y = m.storage[m.storage.length - 1].y + NODE_H + 44;
  }
  m.load.x = 20; m.load.y = y; m.load.w = w; m.load.h = NODE_H;
  m.grid.x = 20 + w + 12; m.grid.y = y; m.grid.w = w; m.grid.h = NODE_H;
  return { VBW, vbH: y + NODE_H + 18, busY, busX1: 24, busX2: VBW - 24, vertical: true };
}

// ── edges ─────────────────────────────────────────────────────────────

function bezPath(x1, y1, x2, y2, horizontal) {
  if (horizontal) {
    const dx = (x2 - x1) * 0.45;
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }
  const dy = (y2 - y1) * 0.5;
  return `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`;
}

function flow(x1, y1, x2, y2, w, color, horizontal, dashed) {
  return `<path class="flow" d="${bezPath(x1, y1, x2, y2, horizontal)}" stroke="${color}"` +
    ` stroke-width="${w.toFixed(1)}"${dashed ? ' stroke-dasharray="5 4"' : ""}/>`;
}

function edgesH(m, L) {
  const out = [];
  const R = (n) => [n.x + n.w, n.y + n.h / 2];
  const Lt = (n) => [n.x, n.y + n.h / 2];
  if (m.controllers.length) {
    m.sources.forEach((s, i) => {
      const c = m.controllers[Math.min(i, m.controllers.length - 1)];
      out.push(flow(...R(s), ...Lt(c), strokeW(s.kw), s.color, true));
    });
    m.controllers.forEach((c) => out.push(flow(...R(c), L.busX, R(c)[1], strokeW(c.kw), c.color, true)));
  } else {
    m.sources.forEach((s) => out.push(flow(...R(s), L.busX, R(s)[1], strokeW(s.kw), s.color, true)));
  }
  out.push(flow(L.busX, Lt(m.load)[1], ...Lt(m.load), strokeW(m.load.kw), m.load.color, true));
  out.push(flow(L.busX, Lt(m.grid)[1], ...Lt(m.grid), strokeW(m.grid.kw || 1), m.grid.color, true, m.grid.island));
  m.storage.forEach((b) => {
    const x = b.x + b.w / 2;
    out.push(`<path class="flow" d="M ${x} ${b.y} C ${x} ${b.y - 28}, ${L.busX} ${L.busBot + 28}, ${L.busX} ${L.busBot}"` +
      ` stroke="${b.color}" stroke-width="${strokeW(b.kw).toFixed(1)}"/>`);
  });
  return out.join("");
}

function edgesV(m, L) {
  const out = [];
  const B = (n) => [n.x + n.w / 2, n.y + n.h];   // bottom center
  const T = (n) => [n.x + n.w / 2, n.y];          // top center
  if (m.controllers.length) {
    m.sources.forEach((s, i) => {
      const c = m.controllers[Math.min(i, m.controllers.length - 1)];
      out.push(flow(...B(s), ...T(c), strokeW(s.kw), s.color, false));
    });
    m.controllers.forEach((c) => out.push(flow(...B(c), B(c)[0], L.busY, strokeW(c.kw), c.color, false)));
  } else {
    m.sources.forEach((s) => out.push(flow(...B(s), B(s)[0], L.busY, strokeW(s.kw), s.color, false)));
  }
  m.storage.forEach((b) => out.push(flow(T(b)[0], L.busY, ...T(b), strokeW(b.kw), b.color, false)));
  out.push(flow(T(m.load)[0], L.busY, ...T(m.load), strokeW(m.load.kw), m.load.color, false));
  out.push(flow(T(m.grid)[0], L.busY, ...T(m.grid), strokeW(m.grid.kw || 1), m.grid.color, false, m.grid.island));
  return out.join("");
}

// ── headers, nodes ────────────────────────────────────────────────────

function header(x, y, label, anchor = "middle") {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}"` +
    ` style="font-size:9px;letter-spacing:.14em;fill:var(--ink-mid)">${label}</text>`;
}

function headers(m, L) {
  if (L.vertical) {
    return [
      m.sources.length ? header(20, m.sources[0].y - 8, "SOURCES", "start") : "",
      m.controllers.length ? header(20, m.controllers[0].y - 8, "CONVERSION", "start") : "",
      header(L.busX1, L.busY - 9, "BUS", "start"),
      m.storage.length ? header(20, m.storage[0].y - 8, "STORAGE", "start") : "",
      header(20, m.load.y - 8, "LOAD / GRID", "start"),
    ].join("");
  }
  return [
    header(18 + 88, 28, "SOURCES"),
    m.controllers.length ? header(268 + 88, 28, "CONVERSION") : "",
    header(L.busX, 28, "BUS"),
    header(742 + 88, 28, "LOAD / GRID"),
    m.storage.length ? header(m.storage[0].x, m.storage[0].y - 8, "STORAGE", "start") : "",
  ].join("");
}

function nodeSvg(n, idx) {
  const maxCh = Math.floor(n.w / 6.4);
  const spec = isSpeculative(n.name) ? " ⚠" : "";
  const dash = n.island ? ' stroke-dasharray="4 3"' : "";
  return `<g class="nd" data-i="${idx}">
    <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="2"
      fill="${n.color}" fill-opacity="0.13" stroke="${n.color}" stroke-width="1.4"${dash}/>
    <text class="node-label" x="${n.x + 9}" y="${n.y + 19}">${esc(trunc(n.name, maxCh))}${spec}</text>
    <text class="node-sub" x="${n.x + 9}" y="${n.y + 34}">${esc(trunc(n.sub, maxCh + 3))}</text>
  </g>`;
}

// ── hover popover (.sysdiag-tip) ──────────────────────────────────────

let tipEl = null;
function getTip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "sysdiag-tip";
    tipEl.style.display = "none";
    document.body.appendChild(tipEl);
  }
  return tipEl;
}

function row(k, v) {
  return `<div><span style="opacity:.6">${k}</span> ${v}</div>`;
}

function tipHtml(n) {
  let h = `<div style="font-weight:600;margin-bottom:3px">${esc(n.name)}</div>`;
  if (n.kind === "source") {
    h += row("rated", formatKw(n.kw));
    if ((n.raw && n.raw.units) > 1) h += row("units", n.raw.units);
    if (n.det) {
      h += row("capacity factor", formatPct(n.det.annual_cf));
      h += row("energy", formatKwh(n.det.cumulative_kwh));
      h += row("capex", formatUsd(n.det.capital_cost));
    }
  } else if (n.kind === "controller") {
    h += row("rated", formatKw(n.kw));
    h += row("topology", esc((n.raw && n.raw.topology) || "—"));
    if (n.det) {
      h += row("losses", formatKwh(n.det.cumulative_loss_kwh));
      h += row("op hours", `${Math.round(n.det.operating_hours)} h`);
    }
  } else if (n.kind === "storage") {
    h += row("capacity", formatKwh(n.raw && n.raw.capacity_kwh));
    if (n.raw) h += row("power", `+${formatKw(n.raw.max_charge_kw)} / −${formatKw(n.raw.max_discharge_kw)}`);
    if (n.det) {
      h += row("health", formatPct(n.det.health));
      h += row("cycles", cyc(n.det.cycles));
      h += row("temp", `${n.det.temperature_c.toFixed(1)} °C`);
      h += row("capex", formatUsd(n.det.capital_cost));
    }
  } else if (n.kind === "load") {
    h += row("peak demand", formatKw(n.kw));
  } else {
    h += n.island ? `<div>island mode — no interconnect</div>` : row("interconnect", `±${formatKw(n.kw)}`);
  }
  if (isSpeculative(n.name)) {
    h += `<div style="margin-top:4px;color:var(--spec-ink)">⚠ speculative technology</div>`;
  }
  return h;
}

function wireHover(container, nodes) {
  const tip = getTip();
  container.querySelectorAll("g.nd").forEach((g) => {
    const n = nodes[+g.dataset.i];
    if (!n) return;
    g.addEventListener("mouseenter", () => {
      tip.innerHTML = tipHtml(n);
      tip.style.display = "block";
    });
    g.addEventListener("mousemove", (e) => {
      tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 274)}px`;
      tip.style.top = `${Math.min(e.clientY + 14, window.innerHeight - 140)}px`;
    });
    g.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

// ── entry point ───────────────────────────────────────────────────────

export function renderSystemDiagram(container, configDesc, metrics = null) {
  container.classList.add("sysdiag");
  const prev = container.__ppDiag;
  if (prev) {
    prev.ro.disconnect();
    document.removeEventListener("pp-theme-change", prev.onTheme);
  }

  let mode = null;  // null | true (vertical) | false (horizontal)
  const draw = () => {
    const wpx = container.clientWidth;
    const vertical = wpx > 0 && wpx < 640;
    if (vertical === mode) return;
    mode = vertical;
    const m = buildModel(configDesc || {}, metrics);
    const nodes = [...m.sources, ...m.controllers, ...m.storage, m.load, m.grid];
    const L = vertical ? layoutV(m) : layoutH(m);
    const edges = vertical ? edgesV(m, L) : edgesH(m, L);
    const bus = vertical
      ? `<line class="bus" x1="${L.busX1}" y1="${L.busY}" x2="${L.busX2}" y2="${L.busY}"/>`
      : `<line class="bus" x1="${L.busX}" y1="${L.busTop}" x2="${L.busX}" y2="${L.busBot}"/>`;
    container.innerHTML =
      `<svg viewBox="0 0 ${L.VBW} ${Math.ceil(L.vbH)}" role="img" aria-label="System architecture diagram">` +
      headers(m, L) + edges + bus + nodes.map((n, i) => nodeSvg(n, i)).join("") +
      `</svg>`;
    wireHover(container, nodes);
  };

  const onTheme = () => { mode = null; draw(); };  // re-resolve dark-mode data colors
  document.addEventListener("pp-theme-change", onTheme);
  const ro = new ResizeObserver(() => draw());
  ro.observe(container);
  container.__ppDiag = { ro, onTheme };
  draw();
}
