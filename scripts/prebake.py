#!/usr/bin/env python3
"""
Pre-bake everything the static web frontend (docs/) needs.

    python scripts/prebake.py

Regenerates, atomically and deterministically (running twice leaves git clean):
  docs/data/presets/<key>.json      10 full-year preset simulations
  docs/data/comparison.json         metrics across all 10 presets
  docs/data/projections/<key>.json  5 municipal growth projections (full-year)
  docs/data/resilience_sample.json  pre-baked scenario batch for the lab page
  docs/py/powerplan-pkg.zip         the package for Pyodide (no matplotlib modules)
  docs/data/index.json              manifest: versions, sha256 hashes, preset cards

Everything goes through powerplan.webapi — the exact code path the browser
runs — so pre-baked and live results share one schema by construction.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from powerplan import webapi  # noqa: E402
from powerplan.presets import PRESETS, EXOTIC_PRESETS, ALL_PRESETS  # noqa: E402
from powerplan.municipal import MUNICIPAL_PROFILES  # noqa: E402
from powerplan.export import serialize_comparison, SCHEMA_VERSION  # noqa: E402

DOCS = REPO / "docs"
DATA = DOCS / "data"
PY = DOCS / "py"

SIM_HOURS = 8760
PROJECTION_YEARS = [0, 5, 10, 15, 20, 25]
# Decorative gallery sparklines: one summer week (day ~167) shows the
# diurnal solar/demand interplay better than a January week.
SPARK_START, SPARK_LEN = 4000, 168

# Modules shipped to the browser. visualize.py and styles.py import
# matplotlib at module level and are deliberately excluded — importing them
# in Pyodide must fail fast as ModuleNotFoundError, not pull 15 MB of wasm.
PKG_MODULES = [
    "__init__.py", "controllers.py", "export.py", "grid.py", "municipal.py",
    "presets.py", "profiles.py", "scenarios.py", "sources.py", "storage.py",
    "webapi.py",
]

RESILIENCE_SAMPLE_PARAMS = {
    "scale": "community", "tier": "conventional", "seed": 42,
    "n_configs": 10, "n_failures": 3, "hours": 720, "include_baseline": True,
}


def write_json(path: Path, payload, raw: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    path.write_text(text)
    print(f"  wrote {path.relative_to(REPO)}  ({len(text)/1e6:.2f} MB)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_zip() -> Path:
    PY.mkdir(parents=True, exist_ok=True)
    out = PY / "powerplan-pkg.zip"
    # Fixed timestamps + sorted names → byte-reproducible archive.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for mod in sorted(PKG_MODULES):
            src = REPO / "powerplan" / mod
            info = zipfile.ZipInfo(f"powerplan/{mod}", date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, src.read_bytes())
    print(f"  wrote {out.relative_to(REPO)}  ({out.stat().st_size/1e3:.0f} KB)")
    return out


def main() -> None:
    t0 = time.time()
    artifacts: dict[str, Path] = {}

    print(f"Pre-baking docs/ data ({SIM_HOURS} h presets) ...")

    # 1+2. Presets (one full-year run each) + comparison rows from the same runs
    comparison_rows = []
    preset_cards = []
    for key in ALL_PRESETS:
        print(f"  simulating preset {key} ...")
        raw = webapi.run_preset(key, hours=SIM_HOURS, source="prebake")
        doc = json.loads(raw)
        path = DATA / "presets" / f"{key}.json"
        write_json(path, None, raw=raw)
        artifacts[f"data/presets/{key}.json"] = path

        m = doc["metrics"]
        comparison_rows.append((key, key in EXOTIC_PRESETS, doc["config"], m))
        s = doc["series"]
        spark = slice(SPARK_START, SPARK_START + SPARK_LEN)
        preset_cards.append({
            "key": key,
            "name": doc["config"]["name"],
            "exotic": key in EXOTIC_PRESETS,
            "file": f"data/presets/{key}.json",
            "scale_name": doc["config"]["scale"]["name"],
            "peak_load_kw": doc["config"]["scale"]["peak_load_kw"],
            "description": doc["config"]["scale"]["description"],
            "headline": {
                "renewable_pct": m["avg_renewable_fraction"],
                "self_sufficiency": m["self_sufficiency"],
                "lcoe_usd_kwh": m["estimated_lcoe_usd_kwh"],
                "capex_usd": m["total_capex_usd"],
            },
            "spark": {
                "start_hour": SPARK_START,
                "demand_kw": s["demand_kw"][spark],
                "generation_kw": s["generation_kw"][spark],
            },
        })

    path = DATA / "comparison.json"
    write_json(path, serialize_comparison(comparison_rows))
    artifacts["data/comparison.json"] = path

    # 3. Municipal projections (all 5 profiles, full fidelity)
    projection_entries = []
    for key in MUNICIPAL_PROFILES:
        print(f"  projecting municipal {key} ...")
        raw = webapi.run_projection(json.dumps({
            "profile": key, "years": PROJECTION_YEARS, "sim_hours": SIM_HOURS,
        }), source="prebake")
        path = DATA / "projections" / f"{key}.json"
        write_json(path, None, raw=raw)
        artifacts[f"data/projections/{key}.json"] = path
        projection_entries.append({
            "key": key,
            "name": MUNICIPAL_PROFILES[key].name,
            "file": f"data/projections/{key}.json",
        })

    # 4. Resilience lab sample batch
    print("  running resilience sample batch ...")
    raw = webapi.run_scenario_batch(json.dumps(RESILIENCE_SAMPLE_PARAMS),
                                    source="prebake")
    path = DATA / "resilience_sample.json"
    write_json(path, None, raw=raw)
    artifacts["data/resilience_sample.json"] = path

    # 5. Package zip for Pyodide
    zip_path = build_zip()
    artifacts["py/powerplan-pkg.zip"] = zip_path

    # 6. Manifest (no timestamps — determinism is the staleness check)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": webapi.ENGINE_VERSION,
        "sim_hours": SIM_HOURS,
        "pkg": {"zip": "py/powerplan-pkg.zip",
                "sha256": sha256(zip_path)},
        "files": {rel: sha256(p) for rel, p in sorted(artifacts.items())},
        "presets": preset_cards,
        "projections": projection_entries,
        "resilience_sample": {"file": "data/resilience_sample.json",
                              "params": RESILIENCE_SAMPLE_PARAMS},
        "comparison": "data/comparison.json",
    }
    write_json(DATA / "index.json", manifest)

    total = sum(p.stat().st_size for p in artifacts.values())
    print(f"Done in {time.time()-t0:.0f}s — {len(artifacts)+1} artifacts, "
          f"{total/1e6:.1f} MB under docs/")


if __name__ == "__main__":
    main()
