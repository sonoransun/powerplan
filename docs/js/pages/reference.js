/**
 * pages/reference.js — Model reference page behavior.
 *
 * The page is deliberately static prose; this module only decorates it:
 *   - scroll-spy TOC (IntersectionObserver) + smooth-scroll anchors
 *   - palette-keyed paint: tech-card swatches, panel rules, the dispatch
 *     flow strip and step numerals (re-painted on theme flip, since two
 *     hues have dark-mode substitutes)
 *   - palette legend built from theme.js PALETTE
 *   - colophon engine version / run length from the data manifest
 *   - speculative chips injected from theme.js (single source of truth)
 */

import {
  PALETTE,
  CATEGORY_CYCLE,
  dataColor,
  sourceColor,
  storageColor,
  specChip,
} from "../theme.js";
import { getManifest } from "../data.js";

// flow roles used by the dispatch SVG (CSS vars --f-<key> on #dispatch-fig)
const FLOW_KEYS = [
  "generation", "charge", "discharge", "demand",
  "grid_import", "grid_export", "curtail",
];

const LEGEND_GROUPS = [
  ["Energy-flow roles", ["demand", "generation", "discharge", "charge",
    "grid_import", "grid_export", "curtail", "loss"]],
  ["Source types", ["solar", "wind", "hydro", "geothermal", "gas",
    "fusion", "antimatter"]],
  ["Storage types", ["lithium", "sodium", "flow", "flywheel", "hydrogen",
    "supercap", "smes"]],
  ["Web-only extras", ["unmet", "accent"]],
];

/** "src:Solar PV" | "sto:lithium" | "pal:fusion" | "cat:3" → resolved hex. */
function resolveSwatch(spec) {
  const i = spec.indexOf(":");
  const kind = i < 0 ? "pal" : spec.slice(0, i);
  const val = i < 0 ? spec : spec.slice(i + 1);
  if (kind === "src") return sourceColor(val);
  if (kind === "sto") return storageColor(val);
  if (kind === "cat") {
    const n = Number.parseInt(val, 10) || 0;
    return dataColor(CATEGORY_CYCLE[((n % CATEGORY_CYCLE.length) + CATEGORY_CYCLE.length) % CATEGORY_CYCLE.length]);
  }
  return dataColor(PALETTE[val] ?? PALETTE.loss);
}

/** Apply (or re-apply, on theme flip) every palette-derived color on the page. */
function paint() {
  // tech cards: data-swatch drives the --tech custom property (swatch,
  // top rule, availability dots all key off it in CSS)
  document.querySelectorAll("[data-swatch]").forEach((el) => {
    el.style.setProperty("--tech", resolveSwatch(el.dataset.swatch));
  });

  // palette-keyed panel top rules
  document.querySelectorAll("[data-rule]").forEach((el) => {
    el.style.setProperty("--rule", dataColor(PALETTE[el.dataset.rule] ?? PALETTE.loss));
  });

  // dispatch step numerals (SVG circles and the <ol> markers share --step-c)
  document.querySelectorAll("[data-flow]").forEach((el) => {
    el.style.setProperty("--step-c", dataColor(PALETTE[el.dataset.flow] ?? PALETTE.loss));
  });

  // dispatch flow strip arrow hues
  const fig = document.getElementById("dispatch-fig");
  if (fig) {
    for (const k of FLOW_KEYS) fig.style.setProperty(`--f-${k}`, dataColor(PALETTE[k]));
  }

  // palette legend swatches show the live (theme-adjusted) color;
  // the hex label stays canonical
  document.querySelectorAll(".pal-sw[data-pal]").forEach((el) => {
    el.style.background = dataColor(PALETTE[el.dataset.pal] ?? PALETTE.loss);
  });
}

function buildLegend() {
  const root = document.getElementById("palette-legend");
  if (!root) return;
  root.innerHTML = LEGEND_GROUPS.map(([title, keys]) => `
    <div class="pal-group">
      <div class="pal-title kicker">${title}</div>
      ${keys.map((k) => `
        <div class="pal-row">
          <i class="pal-sw" data-pal="${k}"></i>
          <span class="pal-name">${k}</span>
          <span class="pal-hex">${(PALETTE[k] || "").toUpperCase()}</span>
        </div>`).join("")}
    </div>`).join("");
}

function injectSpecChips() {
  document.querySelectorAll("[data-spec]").forEach((el) => {
    el.innerHTML = specChip(el.dataset.spec === "compact");
  });
}

async function fillColophon() {
  const vEl = document.getElementById("engine-version");
  const hEl = document.getElementById("engine-hours");
  const pEl = document.getElementById("engine-presets");
  try {
    const m = await getManifest();
    if (vEl) vEl.textContent = `powerplan v${m.engine_version}`;
    if (hEl) hEl.textContent = `${Number(m.sim_hours).toLocaleString("en-US")} h`;
    if (pEl) pEl.textContent = String((m.presets || []).length);
  } catch (err) {
    if (vEl) { vEl.textContent = "manifest unavailable"; vEl.classList.add("muted"); }
    if (hEl) hEl.textContent = "—";
    if (pEl) pEl.textContent = "—";
    console.warn("reference: data manifest not reachable —", err);
  }
}

function initToc() {
  const links = Array.from(document.querySelectorAll(".ref-toc a[href^='#']"));
  if (!links.length) return;

  const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const sections = Array.from(byId.keys())
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  if (!sections.length) return;

  const setActive = (id) => {
    for (const a of links) a.classList.toggle("on", a === byId.get(id));
  };

  // scroll-spy: the topmost section intersecting the spy band wins
  const visible = new Set();
  const pick = () => {
    for (const s of sections) {
      if (visible.has(s.id)) { setActive(s.id); return; }
    }
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    }
    pick();
  }, { rootMargin: "-64px 0px -55% 0px", threshold: 0 });
  sections.forEach((s) => io.observe(s));

  // short final section may never cross the band — pin it at page bottom
  let raf = false;
  window.addEventListener("scroll", () => {
    if (raf) return;
    raf = true;
    requestAnimationFrame(() => {
      raf = false;
      const last = sections[sections.length - 1];
      const atBottom = window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - 4;
      if (last && atBottom) setActive(last.id);
    });
  }, { passive: true });

  // smooth-scroll anchors (scroll-margin-top on .ref-section handles the
  // sticky-header offset); keep the hash shareable without a jump
  links.forEach((a) => {
    a.addEventListener("click", (ev) => {
      const id = decodeURIComponent(a.hash.slice(1));
      const target = document.getElementById(id);
      if (!target) return; // fall back to default anchor behavior
      ev.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", `#${id}`);
      setActive(id);
    });
  });

  // initial state — honor a deep link (e.g. a spec chip → #speculative)
  const initial = decodeURIComponent(location.hash.slice(1));
  setActive(byId.has(initial) ? initial : sections[0].id);
}

export function init() {
  buildLegend();
  injectSpecChips();
  paint();
  fillColophon();
  initToc();
  document.addEventListener("pp-theme-change", paint);
}
