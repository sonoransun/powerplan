"""
Web-API glue: the entry points the browser (Pyodide) and scripts/prebake.py
call to run simulations and get schema-v1 JSON strings back.

Rules this module enforces:
- Every entry point builds FRESH component instances inside function scope
  (components mutate across step() calls and have no reset).
- Results cross the JS boundary as JSON strings — never live objects.
- Memory is released promptly after serialization (the wasm heap never
  shrinks, so don't hold two runs' results at once).
- hydrogen_fuel_cell is special-cased: it takes h2_tank_kg / electrolyzer_kw /
  fuel_cell_kw, and passing capacity_kwh raises TypeError.

Browser-safe: never imports matplotlib (visualize.py / styles.py are excluded
from the browser package zip entirely).
"""

from __future__ import annotations

import dataclasses
import gc
import json

import numpy as np

from .grid import GridConfig, EnergyDispatcher
from .profiles import LoadProfile, SCALES
from .presets import PRESETS, EXOTIC_PRESETS, ALL_PRESETS
from .sources import create_source
from .storage import create_storage
from .controllers import (
    create_controller, MPPTController, SiCConverter, BidirectionalInverter,
)
from .scenarios import (
    FailureScenario, FailureAwareDispatcher, ResilienceMetrics, ScenarioRunner,
)
from .municipal import MUNICIPAL_PROFILES, CLIMATE_ZONES, GrowthProjection
from .export import (
    serialize_run, serialize_comparison, serialize_projection, serialize_batch,
)

ENGINE_VERSION = "0.1.0"


def _dumps(payload) -> str:
    # allow_nan=False is the hard guard: a NaN/inf sneaking past export.py
    # should fail loudly here, not emit invalid JSON. sort_keys makes output
    # canonical — scenarios.analyze() iterates string sets whose order varies
    # with per-process hash randomization, and byte-stable prebake output is
    # the staleness check.
    return json.dumps(payload, allow_nan=False, separators=(",", ":"),
                      sort_keys=True)


# ──────────────────────────────────────────────────────────────────────
# Custom config construction
# ──────────────────────────────────────────────────────────────────────

def build_custom_config(spec: dict) -> GridConfig:
    """Build a GridConfig from the builder UI's spec dict.

    spec = {
      "name": str, "scale_key": str, "grid_interconnect_kw": float,
      "sources":     [{"type": "solar_pv", "params": {...}}, ...],
      "storage":     [{"type": "lithium_ion", "params": {...}}, ...],
      "controllers": [{"type": "mppt", "params": {...}}, ...],   # optional
    }
    """
    scale_key = spec.get("scale_key", "home")
    if scale_key not in SCALES:
        raise ValueError(f"unknown scale_key {scale_key!r}; valid: {sorted(SCALES)}")
    scale = SCALES[scale_key]

    sources = [create_source(s["type"], **s.get("params", {}))
               for s in spec.get("sources", [])]
    if not sources:
        raise ValueError("config needs at least one source")

    storage = [create_storage(u["type"], **u.get("params", {}))
               for u in spec.get("storage", [])]

    ctrl_specs = spec.get("controllers")
    if ctrl_specs:
        controllers = [create_controller(c["type"], **c.get("params", {}))
                       for c in ctrl_specs]
    else:
        # Default trio sized like run_simulation.py --custom; an explicit
        # controller set avoids the engine's silent 0.95 no-controller default.
        total_src_kw = sum(s.rated_kw for s in sources)
        total_stor_kw = sum(u.max_discharge_kw for u in storage)
        controllers = [
            MPPTController(rated_kw=max(total_src_kw, 1)),
            SiCConverter(rated_kw=max(total_stor_kw, 1)),
            BidirectionalInverter(rated_kw=max(total_stor_kw, 1)),
        ]

    return GridConfig(
        name=spec.get("name", "Custom"),
        scale=scale,
        sources=sources,
        storage_units=storage,
        controllers=controllers,
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=float(spec.get("grid_interconnect_kw", 0.0)),
    )


def _base_config(base: dict) -> GridConfig:
    """base = {"preset": key} or {"spec": {...custom spec...}}."""
    if "preset" in base:
        key = base["preset"]
        if key not in ALL_PRESETS:
            raise ValueError(f"unknown preset {key!r}; valid: {sorted(ALL_PRESETS)}")
        return ALL_PRESETS[key]()
    if "spec" in base:
        return build_custom_config(base["spec"])
    raise ValueError("base must contain 'preset' or 'spec'")


# ──────────────────────────────────────────────────────────────────────
# Entry points — each returns a JSON string
# ──────────────────────────────────────────────────────────────────────

def run_preset(key: str, hours: int = 720, dt_hours: float = 1.0,
               progress=None, source: str = "browser") -> str:
    if key not in ALL_PRESETS:
        raise ValueError(f"unknown preset {key!r}; valid: {sorted(ALL_PRESETS)}")
    config = ALL_PRESETS[key]()
    return _run(config, hours, dt_hours, progress,
                preset_key=key, source=source)


def run_custom(spec_json: str, hours: int = 720, dt_hours: float = 1.0,
               progress=None, source: str = "browser") -> str:
    spec = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
    config = build_custom_config(spec)
    return _run(config, hours, dt_hours, progress, source=source)


def _run(config, hours, dt_hours, progress, preset_key=None,
         source="browser", with_resilience=True) -> str:
    dispatcher = EnergyDispatcher(config)
    dispatcher.simulate(hours=hours, dt_hours=dt_hours,
                        progress_callback=progress)
    resilience = ResilienceMetrics.compute(dispatcher) if with_resilience else None
    doc = serialize_run(dispatcher, preset_key=preset_key, source=source,
                        resilience=resilience, engine_version=ENGINE_VERSION)
    dispatcher.results.clear()
    out = _dumps(doc)
    del dispatcher, config, doc
    gc.collect()
    return out


def _add_events(scenario: FailureScenario, events: list[dict]) -> None:
    for e in events:
        etype = e["type"]
        start = float(e["start_hour"])
        dur = float(e["duration_hours"])
        if etype == "source_trip":
            scenario.add_source_trip(start, dur, target=e.get("target"),
                                     severity=float(e.get("severity", 1.0)))
        elif etype == "storage_fault":
            scenario.add_storage_fault(start, dur, target=e.get("target"),
                                       severity=float(e.get("severity", 1.0)))
        elif etype == "weather_crisis":
            scenario.add_weather_crisis(start, dur,
                                        severity=float(e.get("severity", 0.1)))
        elif etype == "grid_disconnect":
            scenario.add_grid_disconnect(start, dur)
        elif etype == "demand_surge":
            scenario.add_demand_surge(start, dur,
                                      multiplier=float(e.get("multiplier", 1.5)))
        elif etype == "simultaneous":
            scenario.add_simultaneous(start,
                                      num_components=int(e.get("num_components", 2)),
                                      duration_hours=dur)
        else:
            raise ValueError(f"unknown failure event type {etype!r}")


def run_failure(payload_json: str, progress=None, source: str = "browser") -> str:
    """Baseline + stressed runs of one config under explicit failure events.

    payload = {"base": {"preset": k} | {"spec": {...}}, "events": [...],
               "hours": 720, "dt_hours": 1.0, "seed": 42}

    Both runs use the same weather: apply()'s first RNG draw is the weather
    seed, and explicit add_* calls consume no RNG, so two FailureScenario
    instances with the same seed produce identical weather arrays.
    """
    p = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    hours = int(p.get("hours", 720))
    dt_hours = float(p.get("dt_hours", 1.0))
    seed = int(p.get("seed", 42))
    events = p.get("events", [])

    failed_scn = FailureScenario(seed=seed)
    _add_events(failed_scn, events)
    baseline_scn = FailureScenario(seed=seed)

    fail_cfg, fail_weather, fail_timeline = failed_scn.apply(
        _base_config(p["base"]), hours=hours, dt_hours=dt_hours)
    base_cfg, base_weather, _ = baseline_scn.apply(
        _base_config(p["base"]), hours=hours, dt_hours=dt_hours)
    assert np.array_equal(base_weather, fail_weather), \
        "baseline/failed weather diverged — RNG draw order changed"

    docs = {}
    total = 2 * int(hours / dt_hours)

    def _staged(offset):
        if progress is None:
            return None
        return lambda done, n: progress(offset + done, total)

    base_disp = FailureAwareDispatcher(base_cfg, {})
    base_disp.simulate(hours=hours, dt_hours=dt_hours,
                       weather_factors=base_weather,
                       progress_callback=_staged(0))
    docs["baseline"] = serialize_run(
        base_disp, source=source,
        resilience=ResilienceMetrics.compute(base_disp),
        engine_version=ENGINE_VERSION)
    base_disp.results.clear()
    del base_disp, base_cfg
    gc.collect()

    fail_disp = FailureAwareDispatcher(fail_cfg, fail_timeline)
    fail_disp.simulate(hours=hours, dt_hours=dt_hours,
                       weather_factors=fail_weather,
                       progress_callback=_staged(total // 2))
    docs["failed"] = serialize_run(
        fail_disp, source=source,
        resilience=ResilienceMetrics.compute(fail_disp),
        failures={"description": failed_scn.describe(),
                  "events": events},
        engine_version=ENGINE_VERSION)
    fail_disp.results.clear()
    del fail_disp, fail_cfg
    gc.collect()

    return _dumps(docs)


def run_scenario_batch(params_json: str, progress=None,
                       source: str = "browser") -> str:
    """params = {"scale", "tier", "seed", "n_configs", "n_failures",
                 "hours", "include_baseline"}"""
    p = json.loads(params_json) if isinstance(params_json, str) else params_json
    runner = ScenarioRunner(
        scale=p.get("scale", "community"),
        tier=p.get("tier", "conventional"),
        seed=int(p.get("seed", 42)),
        hours=int(p.get("hours", 720)),
        dt_hours=float(p.get("dt_hours", 1.0)),
    )
    results = runner.run(
        n_configs=int(p.get("n_configs", 5)),
        n_failures=int(p.get("n_failures", 2)),
        include_baseline=bool(p.get("include_baseline", True)),
        progress_callback=progress,
    )
    doc = serialize_batch(results, runner.analyze(), params=p)
    out = _dumps(doc)
    del runner, results, doc
    gc.collect()
    return out


def run_projection(params_json: str, progress=None,
                   source: str = "browser") -> str:
    """params = {"profile", "climate" (optional override), "years",
                 "sim_hours", "base_year"}

    Years run one at a time so progress can tick per year — bit-identical to
    a single run(years=[...]) because each year's weather is independently
    seeded (42 + year_offset) and configs are rebuilt per year.
    """
    p = json.loads(params_json) if isinstance(params_json, str) else params_json
    key = p["profile"]
    if key not in MUNICIPAL_PROFILES:
        raise ValueError(f"unknown profile {key!r}; valid: {sorted(MUNICIPAL_PROFILES)}")
    profile = MUNICIPAL_PROFILES[key]
    climate_key = p.get("climate")
    if climate_key:
        if climate_key not in CLIMATE_ZONES:
            raise ValueError(f"unknown climate {climate_key!r}; valid: {sorted(CLIMATE_ZONES)}")
        profile = dataclasses.replace(profile, climate=CLIMATE_ZONES[climate_key])

    years = sorted(int(y) for y in p.get("years", [0, 5, 10, 15, 20, 25]))
    sim_hours = int(p.get("sim_hours", 720))
    base_year = int(p.get("base_year", 2025))

    rows = []
    for i, y in enumerate(years):
        gp = GrowthProjection(profile, base_year=base_year)
        rows.extend(gp.run(years=[y], sim_hours=sim_hours))
        del gp
        gc.collect()
        if progress is not None:
            progress(i + 1, len(years))

    doc = serialize_projection(key, profile, rows, base_year)
    out = _dumps(doc)
    del rows, doc
    gc.collect()
    return out
