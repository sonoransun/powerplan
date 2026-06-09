"""
Serializers for the web frontend.

Turns dispatcher trajectories, comparison rows, scenario batches, and growth
projections into JSON-able dicts (schema_version 1). Shared by
scripts/prebake.py (CPython, pre-baked docs/data/*.json) and
powerplan/webapi.py (Pyodide, live in-browser runs) so the two paths cannot
drift. Browser-safe: numpy + stdlib only — no matplotlib, no file I/O.

Reads only dispatcher.results / compute_metrics() / config; never calls
demand_kw() (an out-of-band call would desynchronize the demand RNG).
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

SCHEMA_VERSION = 1


# ──────────────────────────────────────────────────────────────────────
# Sanitization helpers
# ──────────────────────────────────────────────────────────────────────

def _finite(x: float) -> float | None:
    """Non-finite floats (inf LCOE, NaN) become None so JSON stays valid."""
    x = float(x)
    return x if math.isfinite(x) else None


def _sig(x: float, n: int = 4) -> float | int | None:
    """Round to n significant digits — indistinguishable on charts, ~40% smaller."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0
    rounded = round(x, n - 1 - int(math.floor(math.log10(abs(x)))))
    return int(rounded) if rounded == int(rounded) else rounded


def _series(values, n: int = 4) -> list:
    return [_sig(v, n) for v in values]


def to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-able natives.

    np.int64 is not JSON-serializable (np.float64 is, but normalize both);
    non-finite floats become None.
    """
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return _finite(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ──────────────────────────────────────────────────────────────────────
# Config description
# ──────────────────────────────────────────────────────────────────────

def _describe_config(config, preset_key=None) -> dict:
    scale = config.scale
    return {
        "name": config.name,
        "preset_key": preset_key,
        "scale": {
            "name": scale.name,
            "peak_load_kw": _finite(scale.peak_load_kw),
            "annual_consumption_kwh": _finite(scale.annual_consumption_kwh),
            "num_endpoints": int(scale.num_endpoints),
            "description": scale.description,
        },
        "grid_interconnect_kw": _finite(config.grid_interconnect_kw),
        "sources": [
            {
                "name": s.name,
                "type": type(s).__name__,
                "rated_kw": _finite(s.rated_kw),
                "units": int(getattr(s, "units", 1)),
                "is_renewable": bool(s.is_renewable),
            }
            for s in config.sources
        ],
        "storage": [
            {
                "name": u.name,
                "type": type(u).__name__,
                "capacity_kwh": _finite(u.nominal_capacity_kwh),
                "max_charge_kw": _finite(u.max_charge_kw),
                "max_discharge_kw": _finite(u.max_discharge_kw),
            }
            for u in config.storage_units
        ],
        "controllers": [
            {
                "name": c.name,
                "type": type(c).__name__,
                "rated_kw": _finite(c.rated_kw),
                "topology": getattr(c, "topology", ""),
            }
            for c in config.controllers
        ],
    }


def _resilience_dict(resilience) -> dict:
    """ResilienceResult → dict, dropping base_metrics (duplicate of metrics)."""
    d = asdict(resilience)
    d.pop("base_metrics", None)
    return to_native(d)


# ──────────────────────────────────────────────────────────────────────
# Simulation run
# ──────────────────────────────────────────────────────────────────────

def serialize_run(dispatcher, *, preset_key=None, source="browser",
                  failures=None, resilience=None, engine_version="0.1.0") -> dict:
    """Full simulation document: config + metrics + hourly series."""
    results = dispatcher.results
    if not results:
        raise ValueError("dispatcher has no results — run simulate() first")

    dt = results[1].hour - results[0].hour if len(results) > 1 else 1.0
    n = len(results)

    source_names = list(results[0].source_outputs.keys())
    storage_names = list(results[0].storage_states.keys())

    series = {
        "start_hour": _finite(results[0].hour),
        "demand_kw": _series(r.demand_kw for r in results),
        "generation_kw": _series(r.total_generation_kw for r in results),
        "discharge_kw": _series(r.total_storage_discharge_kw for r in results),
        "charge_kw": _series(r.total_storage_charge_kw for r in results),
        "grid_import_kw": _series(r.grid_import_kw for r in results),
        "grid_export_kw": _series(r.grid_export_kw for r in results),
        "curtailment_kw": _series(r.curtailment_kw for r in results),
        "unmet_kw": _series(r.unmet_demand_kw for r in results),
        "system_efficiency": [round(float(r.system_efficiency), 4) for r in results],
        "renewable_fraction": [round(float(r.renewable_fraction), 4) for r in results],
        "source_kw": {
            name: _series(r.source_outputs.get(name, 0.0) for r in results)
            for name in source_names
        },
        "storage_soc": {
            name: [round(float(r.storage_states[name].soc), 4) for r in results]
            for name in storage_names
        },
    }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": "simulation",
        "engine": {"package": "powerplan", "version": engine_version},
        "source": source,
        "config": _describe_config(dispatcher.config, preset_key=preset_key),
        "sim": {"hours": _finite(n * dt), "dt_hours": _finite(dt), "n_steps": n},
        "metrics": to_native(dispatcher.compute_metrics()),
        "resilience": _resilience_dict(resilience) if resilience is not None else None,
        "failures": to_native(failures) if failures is not None else None,
        "series": series,
    }
    return doc


# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────

def serialize_comparison(rows) -> dict:
    """rows: iterable of (key, exotic_flag, config_desc_dict, metrics_dict)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "comparison",
        "presets": [
            {
                "key": key,
                "name": config_desc["name"],
                "exotic": bool(exotic),
                "scale_name": config_desc["scale"]["name"],
                "peak_load_kw": config_desc["scale"]["peak_load_kw"],
                "metrics": to_native(metrics),
            }
            for key, exotic, config_desc, metrics in rows
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Growth projection
# ──────────────────────────────────────────────────────────────────────

def _scalar_metrics(metrics: dict) -> dict:
    """Drop the three *_details lists — per-year docs only need scalars."""
    return to_native({k: v for k, v in metrics.items() if not k.endswith("_details")})


def serialize_projection(profile_key, profile, results, base_year) -> dict:
    """results: list[ProjectionYearResult] (possibly accumulated year-by-year)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "projection",
        "base_year": int(base_year),
        "profile": {
            "key": profile_key,
            "name": profile.name,
            "population": int(profile.population),
            "annual_load_growth_pct": _finite(profile.annual_load_growth_pct),
            "renewable_target_pct": _finite(profile.renewable_target_pct),
            "renewable_target_year": int(profile.renewable_target_year),
            "net_zero_year": int(profile.net_zero_year),
            "peak_load_kw": _finite(profile.scale.peak_load_kw),
            "climate": {
                "name": profile.climate.name,
                "solar_factor": _finite(profile.climate.solar_factor),
                "mean_wind_ms": _finite(getattr(profile.climate, "mean_wind_ms", 0)),
            },
        },
        "years": [
            {
                "year": int(r.year),
                "year_offset": int(r.year_offset),
                "fossil_capacity_kw": _finite(r.fossil_capacity_kw),
                "renewable_capacity_kw": _finite(r.renewable_capacity_kw),
                "storage_capacity_kwh": _finite(r.storage_capacity_kwh),
                "peak_demand_kw": _finite(r.peak_demand_kw),
                "renewable_target_pct": _finite(r.renewable_target_pct),
                "renewable_actual_pct": _finite(r.renewable_actual_pct),
                "total_capex": _finite(r.total_capex),
                "lcoe": _finite(r.lcoe),
                "emissions_tonnes": _finite(r.emissions_tonnes),
                "metrics": _scalar_metrics(r.metrics),
            }
            for r in results
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Scenario batch (resilience lab)
# ──────────────────────────────────────────────────────────────────────

def serialize_batch(results, analysis, params=None) -> dict:
    """results: list[ScenarioRunResult]; analysis: ScenarioRunner.analyze()."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "batch",
        "params": to_native(params) if params else None,
        "runs": [
            {
                "config_id": int(r.config_id),
                "failure_id": int(r.failure_id),
                "config_name": r.config_name,
                "source_types": list(r.source_types),
                "storage_types": list(r.storage_types),
                "failure_desc": r.failure_desc,
                "resilience": _resilience_dict(r.resilience),
            }
            for r in results
        ],
        "analysis": to_native(analysis),
    }
