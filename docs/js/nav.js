/**
 * nav.js — injects the shared sticky header into every page.
 * Usage: <script type="module"> import { initNav } from "./js/nav.js"; initNav("index"); </script>
 */

import { initTheme, isDark, setTheme } from "./theme.js";

const PAGES = [
  ["index", "Dashboard", "index.html"],
  ["compare", "Compare", "compare.html"],
  ["builder", "Builder", "builder.html"],
  ["resilience", "Resilience", "resilience.html"],
  ["municipal", "Municipal", "municipal.html"],
  ["reference", "Model", "reference.html"],
];

export function initNav(active) {
  initTheme();
  const header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML = `
    <div class="inner">
      <a class="brand" href="index.html">
        <span class="bolt">⚡</span> POWERPLAN
        <span class="sub">heterogeneous power system simulator</span>
      </a>
      <nav class="site-nav">
        ${PAGES.map(([id, label, href]) =>
          `<a href="${href}" class="${id === active ? "active" : ""}">${label}</a>`).join("")}
      </nav>
      <button class="theme-toggle" title="Toggle light/dark" aria-label="Toggle theme">◐</button>
    </div>`;
  document.body.prepend(header);
  header.querySelector(".theme-toggle").addEventListener("click", () => {
    setTheme(isDark() ? "light" : "dark");
  });
}
