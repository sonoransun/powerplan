"""
Programmatic scenario generation and resilience testing.

Generates infinite variations of power system configurations, injects
sporadic and catastrophic equipment failures, and measures system
resilience across all deployment scales.
"""

from __future__ import annotations

import copy
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from .storage import (
    StorageUnit, LithiumIonBattery, SodiumSolidStateBattery,
    LiquidElectrolyteBattery, FlywheelStorage, HydrogenFuelCell,
    GrapheneSupercapacitor, SMES,
)
from .sources import (
    EnergySource, SourceOutput, SolarPV, WindTurbine, MicroHydro,
    Geothermal, NaturalGasTurbine, MicroFusionReactor, AntimatterReactor,
)
from .controllers import (
    SiCConverter, GaNConverter, MPPTController, BidirectionalInverter,
    HydrogenPowerController, FusionPowerController, CryogenicPowerSupply,
    AntimatterPowerController,
)
from .profiles import LoadProfile, SCALES
from .grid import GridConfig, EnergyDispatcher, DispatchResult


# ──────────────────────────────────────────────────────────────────────
# Technology Specification Tables
# ──────────────────────────────────────────────────────────────────────

TECH_TIERS = {
    "conventional": {
        "sources": ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas"],
        "storage": ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
                     "flywheel", "hydrogen_fuel_cell"],
    },
    "exotic": {
        "sources": ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas",
                     "micro_fusion"],
        "storage": ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
                     "flywheel", "hydrogen_fuel_cell", "graphene_supercap", "smes"],
    },
    "antimatter": {
        "sources": ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas",
                     "micro_fusion", "antimatter"],
        "storage": ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
                     "flywheel", "hydrogen_fuel_cell", "graphene_supercap", "smes"],
    },
}

# Source specs: kw_range and selection weight per scale
# None means unavailable at that scale
SOURCE_SPECS = {
    "solar_pv": {
        "home": {"kw": (3, 12), "w": 0.8},
        "neighborhood": {"kw": (50, 200), "w": 0.7},
        "community": {"kw": (300, 1500), "w": 0.6},
        "district": {"kw": (3000, 20000), "w": 0.5},
        "metropolitan": {"kw": (50000, 300000), "w": 0.4},
    },
    "wind": {
        "home": {"kw": (0.5, 3), "w": 0.4},
        "neighborhood": {"kw": (10, 50), "w": 0.5},
        "community": {"kw": (200, 800), "w": 0.5},
        "district": {"kw": (2000, 15000), "w": 0.5},
        "metropolitan": {"kw": (8000, 80000), "w": 0.4},
    },
    "micro_hydro": {
        "community": {"kw": (20, 100), "w": 0.3},
        "district": {"kw": (200, 1000), "w": 0.3},
        "metropolitan": {"kw": (2000, 15000), "w": 0.2},
    },
    "geothermal": {
        "district": {"kw": (1000, 5000), "w": 0.25},
        "metropolitan": {"kw": (20000, 80000), "w": 0.3},
    },
    "micro_fusion": {
        "community": {"kw": (500, 2000), "w": 0.2},
        "district": {"kw": (5000, 20000), "w": 0.3},
        "metropolitan": {"kw": (30000, 150000), "w": 0.25},
    },
    "natural_gas": {
        "community": {"kw": (500, 3000), "w": 0.3},
        "district": {"kw": (5000, 50000), "w": 0.3},
        "metropolitan": {"kw": (50000, 500000), "w": 0.25},
    },
    "antimatter": {
        "district": {"kw": (10000, 30000), "w": 0.15},
        "metropolitan": {"kw": (50000, 200000), "w": 0.15},
    },
}

# Storage specs: kwh_range and power-to-energy ratio
STORAGE_SPECS = {
    "lithium_ion": {
        "home": {"kwh": (5, 20), "pr": 0.4},
        "neighborhood": {"kwh": (30, 150), "pr": 0.4},
        "community": {"kwh": (200, 1000), "pr": 0.5},
        "district": {"kwh": (2000, 15000), "pr": 0.5},
        "metropolitan": {"kwh": (50000, 300000), "pr": 0.5},
    },
    "sodium_solid_state": {
        "home": {"kwh": (5, 15), "pr": 0.3},
        "neighborhood": {"kwh": (20, 80), "pr": 0.3},
        "community": {"kwh": (100, 600), "pr": 0.3},
        "district": {"kwh": (1000, 8000), "pr": 0.3},
        "metropolitan": {"kwh": (20000, 100000), "pr": 0.3},
    },
    "liquid_electrolyte": {
        "community": {"kwh": (200, 1000), "pr": 0.2},
        "district": {"kwh": (2000, 20000), "pr": 0.2},
        "metropolitan": {"kwh": (50000, 400000), "pr": 0.2},
    },
    "flywheel": {
        "neighborhood": {"kwh": (1, 5), "pr": 15.0},
        "community": {"kwh": (3, 15), "pr": 15.0},
        "district": {"kwh": (20, 100), "pr": 10.0},
        "metropolitan": {"kwh": (50, 500), "pr": 10.0},
    },
    "hydrogen_fuel_cell": {
        "community": {"kwh": (1000, 5000), "pr": 0.02},
        "district": {"kwh": (10000, 200000), "pr": 0.02},
        "metropolitan": {"kwh": (100000, 5000000), "pr": 0.015},
    },
    "graphene_supercap": {
        "community": {"kwh": (10, 50), "pr": 150.0},
        "district": {"kwh": (50, 300), "pr": 200.0},
        "metropolitan": {"kwh": (200, 5000), "pr": 200.0},
    },
    "smes": {
        "district": {"kwh": (10, 80), "pr": 2000.0},
        "metropolitan": {"kwh": (30, 500), "pr": 2000.0},
    },
}

# Controller requirements: which controllers are needed for each source/storage type
CONTROLLER_REQUIREMENTS = {
    "solar_pv": ["mppt"],
    "natural_gas": [],
    "micro_fusion": ["fusion"],
    "antimatter": ["antimatter"],
    "hydrogen_fuel_cell": ["hydrogen"],
    "smes": ["cryogenic"],
}


# ──────────────────────────────────────────────────────────────────────
# Source/Storage Factory Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_source(type_key: str, rated_kw: float, rng) -> EnergySource:
    """Create a source with appropriate parameters for its type and size."""
    if type_key == "solar_pv":
        return SolarPV(rated_kw=rated_kw, tracking=rated_kw > 50,
                       latitude=25 + rng.random() * 25)
    elif type_key == "wind":
        hub = 15 + 85 * min(1, rated_kw / 5000)
        return WindTurbine(rated_kw=rated_kw, hub_height_m=hub)
    elif type_key == "micro_hydro":
        return MicroHydro(rated_kw=rated_kw, head_m=8 + rng.random() * 25)
    elif type_key == "geothermal":
        return Geothermal(rated_kw=rated_kw, well_temp_c=130 + rng.random() * 70)
    elif type_key == "natural_gas":
        ptype = rng.choice(["ccgt", "peaker"], p=[0.6, 0.4])
        return NaturalGasTurbine(rated_kw=rated_kw, plant_type=ptype)
    elif type_key == "micro_fusion":
        cycle = rng.choice(["dt", "pb11"], p=[0.7, 0.3])
        q = rng.uniform(8, 15) if cycle == "dt" else rng.uniform(3, 6)
        return MicroFusionReactor(rated_kw=rated_kw, fuel_cycle=cycle,
                                  q_engineering=q)
    elif type_key == "antimatter":
        target = rng.choice(["uranium", "xenon", "lead"], p=[0.5, 0.3, 0.2])
        layers = int(100 + rng.random() * 150)
        # Size fuel for ~1 year
        gross_est = rated_kw / 0.6
        fuel_rate_est = gross_est * 3600 / 180_000
        fuel = fuel_rate_est * 8760 * 1.1
        return AntimatterReactor(rated_kw=rated_kw, target_atom=target,
                                 graphene_electrode_layers=layers,
                                 fuel_reservoir_ug=fuel,
                                 electron_collection_efficiency=0.4 + rng.random() * 0.15)
    raise ValueError(f"Unknown source type: {type_key}")


def _make_storage(type_key: str, capacity_kwh: float, max_power_kw: float,
                  rng) -> StorageUnit:
    """Create a storage unit with appropriate parameters."""
    if type_key == "lithium_ion":
        chem = rng.choice(["lfp", "nmc"], p=[0.6, 0.4])
        return LithiumIonBattery(capacity_kwh=capacity_kwh,
                                 max_power_kw=max_power_kw, chemistry=chem)
    elif type_key == "sodium_solid_state":
        return SodiumSolidStateBattery(capacity_kwh=capacity_kwh,
                                       max_power_kw=max_power_kw)
    elif type_key == "liquid_electrolyte":
        return LiquidElectrolyteBattery(capacity_kwh=capacity_kwh,
                                        max_power_kw=max_power_kw)
    elif type_key == "flywheel":
        return FlywheelStorage(capacity_kwh=capacity_kwh,
                               max_power_kw=max_power_kw)
    elif type_key == "hydrogen_fuel_cell":
        h2_kg = capacity_kwh / 33.3
        return HydrogenFuelCell(h2_tank_kg=h2_kg,
                                electrolyzer_kw=max_power_kw * 1.2,
                                fuel_cell_kw=max_power_kw)
    elif type_key == "graphene_supercap":
        return GrapheneSupercapacitor(capacity_kwh=capacity_kwh,
                                      max_power_kw=max_power_kw)
    elif type_key == "smes":
        energy_j = capacity_kwh * 3.6e6
        inductance = 2 * energy_j / (1000 ** 2)  # I_max ~ 1000A
        return SMES(capacity_kwh=capacity_kwh, max_power_kw=max_power_kw,
                     inductance_h=max(1, inductance),
                     operating_temp_k=20 + rng.random() * 30)
    raise ValueError(f"Unknown storage type: {type_key}")


def _make_controllers(source_types: list[str], storage_types: list[str],
                      source_power: float, storage_power: float) -> list:
    """Auto-select and size controllers based on sources and storage."""
    controllers = []
    needed = set()

    for st in source_types:
        if st in CONTROLLER_REQUIREMENTS:
            needed.update(CONTROLLER_REQUIREMENTS[st])
    for st in storage_types:
        if st in CONTROLLER_REQUIREMENTS:
            needed.update(CONTROLLER_REQUIREMENTS[st])

    # Always need at least one general converter and inverter
    needed.add("sic")
    needed.add("bidirectional")

    total_power = source_power + storage_power
    for ctrl_type in needed:
        if ctrl_type == "mppt":
            controllers.append(MPPTController(rated_kw=max(source_power, 1)))
        elif ctrl_type == "sic":
            controllers.append(SiCConverter(rated_kw=max(total_power * 0.5, 1)))
        elif ctrl_type == "bidirectional":
            controllers.append(BidirectionalInverter(
                rated_kw=max(storage_power * 1.1, 1), phases=3))
        elif ctrl_type == "fusion":
            controllers.append(FusionPowerController(
                rated_kw=max(source_power * 0.5, 1)))
        elif ctrl_type == "antimatter":
            controllers.append(AntimatterPowerController(
                rated_kw=max(source_power * 0.5, 1), mhd_fraction=0.6))
        elif ctrl_type == "hydrogen":
            controllers.append(HydrogenPowerController(
                rated_kw=max(storage_power * 0.2, 1)))
        elif ctrl_type == "cryogenic":
            controllers.append(CryogenicPowerSupply(
                rated_kw=max(total_power * 0.02, 1)))

    return controllers


# ──────────────────────────────────────────────────────────────────────
# ConfigGenerator
# ──────────────────────────────────────────────────────────────────────

class ConfigGenerator:
    """Programmatic generation of valid power system configurations."""

    def __init__(
        self,
        scale: str = "community",
        tier: str = "conventional",
        seed: int | None = None,
        generation_margin: tuple[float, float] = (0.8, 1.5),
        backup_hours: tuple[float, float] = (2.0, 8.0),
        min_sources: int = 2,
        max_sources: int = 5,
        min_storage: int = 1,
        max_storage: int = 4,
        grid_interconnect: bool = True,
    ):
        self.scale_key = scale
        self.scale = SCALES[scale]
        self.tier = tier
        self.rng = np.random.default_rng(seed)
        self.generation_margin = generation_margin
        self.backup_hours = backup_hours
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.min_storage = min_storage
        self.max_storage = max_storage
        self.grid_interconnect = grid_interconnect
        self._config_count = 0

    def generate(self) -> GridConfig:
        """Generate one random valid configuration."""
        self._config_count += 1
        tier_def = TECH_TIERS[self.tier]
        peak = self.scale.peak_load_kw

        # --- Select and size sources ---
        available_sources = []
        weights = []
        for src_type in tier_def["sources"]:
            spec = SOURCE_SPECS.get(src_type, {}).get(self.scale_key)
            if spec:
                available_sources.append((src_type, spec))
                weights.append(spec["w"])

        if not available_sources:
            available_sources = [("solar_pv", SOURCE_SPECS["solar_pv"][self.scale_key])]
            weights = [1.0]

        weights = np.array(weights) / sum(weights)
        n_src = min(self.rng.integers(self.min_sources, self.max_sources + 1),
                    len(available_sources))
        chosen_idx = self.rng.choice(len(available_sources), size=n_src,
                                     replace=False, p=weights)

        sources = []
        source_types = []
        total_gen_kw = 0
        for idx in chosen_idx:
            src_type, spec = available_sources[idx]
            kw = self.rng.uniform(*spec["kw"])
            sources.append(_make_source(src_type, kw, self.rng))
            source_types.append(src_type)
            total_gen_kw += kw

        # Rescale to fit generation margin
        target_gen = peak * self.rng.uniform(*self.generation_margin)
        if total_gen_kw > 0:
            scale_factor = target_gen / total_gen_kw
            # Apply to largest source only to preserve diversity
            largest_idx = max(range(len(sources)), key=lambda i: sources[i].rated_kw)
            new_kw = sources[largest_idx].rated_kw * scale_factor
            new_kw = max(new_kw, 1.0)
            sources[largest_idx] = _make_source(source_types[largest_idx],
                                                new_kw, self.rng)

        # --- Select and size storage ---
        available_storage = []
        for st_type in tier_def["storage"]:
            spec = STORAGE_SPECS.get(st_type, {}).get(self.scale_key)
            if spec:
                available_storage.append((st_type, spec))

        n_stor = min(self.rng.integers(self.min_storage, self.max_storage + 1),
                     max(len(available_storage), 1))

        storage_units = []
        storage_types = []
        total_stor_kwh = 0
        total_stor_kw = 0

        if available_storage:
            stor_idx = self.rng.choice(len(available_storage), size=n_stor,
                                       replace=False)
            for idx in stor_idx:
                st_type, spec = available_storage[idx]
                kwh = self.rng.uniform(*spec["kwh"])
                kw = kwh * spec["pr"]
                storage_units.append(_make_storage(st_type, kwh, kw, self.rng))
                storage_types.append(st_type)
                total_stor_kwh += kwh
                total_stor_kw += kw

        # Validate backup hours
        target_backup = peak * self.rng.uniform(*self.backup_hours)
        if total_stor_kwh > 0 and total_stor_kwh < target_backup * 0.5:
            # Scale up largest storage
            if storage_units:
                factor = target_backup / total_stor_kwh
                li = max(range(len(storage_units)),
                         key=lambda i: storage_units[i].nominal_capacity_kwh)
                new_kwh = storage_units[li].nominal_capacity_kwh * factor
                spec = STORAGE_SPECS[storage_types[li]][self.scale_key]
                new_kw = new_kwh * spec["pr"]
                storage_units[li] = _make_storage(storage_types[li],
                                                   new_kwh, new_kw, self.rng)
                total_stor_kw = sum(u.max_discharge_kw for u in storage_units)

        # --- Controllers ---
        src_kw = sum(s.rated_kw for s in sources)
        stor_kw = total_stor_kw if total_stor_kw > 0 else sum(
            u.max_discharge_kw for u in storage_units)
        controllers = _make_controllers(source_types, storage_types,
                                        src_kw, stor_kw)

        # --- Grid interconnect ---
        grid_kw = 0.0
        if self.grid_interconnect:
            grid_kw = peak * self.rng.uniform(0.2, 0.8)

        # --- Assemble ---
        load_seed = int(self.rng.integers(0, 2**31))
        config = GridConfig(
            name=f"Scenario-{self._config_count}",
            scale=self.scale,
            sources=sources,
            storage_units=storage_units,
            controllers=controllers,
            load_profile=LoadProfile(self.scale, seed=load_seed),
            grid_interconnect_kw=grid_kw,
        )
        return config

    def generate_batch(self, n: int) -> list[GridConfig]:
        return [self.generate() for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────
# Failure Injection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FailureEvent:
    """A single failure event specification."""
    type: str
    start_hour: float
    duration_hours: float
    severity: float = 1.0
    target: str | None = None
    metadata: dict = field(default_factory=dict)


class FailedSource(EnergySource):
    """Wraps a source to inject output reduction during failure windows."""

    def __init__(self, original: EnergySource,
                 failure_windows: list[tuple[float, float, float]]):
        # Don't call super().__init__ — we proxy everything to original
        self._original = original
        self._failure_windows = failure_windows
        # Copy key attributes so the dispatcher can read them
        self.name = original.name
        self.rated_kw = original.rated_kw
        self.units = original.units
        self.latitude = original.latitude
        self.cumulative_kwh = original.cumulative_kwh

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        for start, end, severity in self._failure_windows:
            if start <= hour_of_year <= end:
                return self._original.output_kw(hour_of_year, weather_factor) * (1 - severity)
        return self._original.output_kw(hour_of_year, weather_factor)

    def step(self, hour_of_year: float, dt_hours: float,
             weather_factor: float = 1.0) -> SourceOutput:
        for start, end, severity in self._failure_windows:
            if start <= hour_of_year <= end:
                out = self._original.step(hour_of_year, dt_hours, weather_factor)
                reduced_power = out.power_kw * (1 - severity)
                return SourceOutput(power_kw=reduced_power,
                                    capacity_factor=reduced_power / max(self.rated_kw, 1),
                                    available=reduced_power > 0)
        return self._original.step(hour_of_year, dt_hours, weather_factor)

    def capacity_factor_annual(self) -> float:
        return self._original.capacity_factor_annual()

    def capital_cost_per_kw(self) -> float:
        return self._original.capital_cost_per_kw()

    def summary(self) -> dict:
        return self._original.summary()


class FailedStorage(StorageUnit):
    """Wraps a storage unit to inject failures during specific windows."""

    def __init__(self, original: StorageUnit,
                 failure_windows: list[tuple[float, float, float]]):
        self._original = original
        self._failure_windows = failure_windows
        self._current_hour = 0.0
        # Copy attributes the dispatcher reads
        self.name = original.name
        self.nominal_capacity_kwh = original.nominal_capacity_kwh
        self.max_charge_kw = original.max_charge_kw
        self.max_discharge_kw = original.max_discharge_kw
        self.units = original.units
        self.soc = original.soc
        self.temperature_c = original.temperature_c
        self.ambient_temp_c = original.ambient_temp_c
        self.cycle_count = original.cycle_count
        self.health = original.health
        self.cumulative_throughput_kwh = original.cumulative_throughput_kwh

    @property
    def effective_capacity_kwh(self) -> float:
        return self._original.effective_capacity_kwh

    @property
    def stored_energy_kwh(self) -> float:
        return self._original.stored_energy_kwh

    def step(self, power_kw: float, dt_hours: float) -> float:
        self._current_hour += dt_hours
        for start, end, severity in self._failure_windows:
            if start <= self._current_hour <= end:
                result = self._original.step(power_kw * (1 - severity), dt_hours)
                self._sync_state()
                return result
        result = self._original.step(power_kw, dt_hours)
        self._sync_state()
        return result

    def _sync_state(self):
        """Keep proxy attributes in sync with original."""
        self.soc = self._original.soc
        self.health = self._original.health
        self.temperature_c = self._original.temperature_c
        self.cycle_count = self._original.cycle_count

    def get_state(self):
        return self._original.get_state()

    def charge_efficiency(self, power_kw, soc):
        return self._original.charge_efficiency(power_kw, soc)

    def discharge_efficiency(self, power_kw, soc):
        return self._original.discharge_efficiency(power_kw, soc)

    def self_discharge_rate(self):
        return self._original.self_discharge_rate()

    def degradation_per_cycle(self):
        return self._original.degradation_per_cycle()

    def capital_cost_per_kwh(self):
        return self._original.capital_cost_per_kwh()

    def thermal_model(self, power_kw, dt_hours):
        return self._original.thermal_model(power_kw, dt_hours)

    def summary(self) -> dict:
        return self._original.summary()


class FailureScenario:
    """Composable failure injection for resilience testing."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.events: list[FailureEvent] = []

    def add_source_trip(self, start_hour: float, duration_hours: float,
                        target: str | None = None,
                        severity: float = 1.0) -> "FailureScenario":
        self.events.append(FailureEvent("source_trip", start_hour,
                                        duration_hours, severity, target))
        return self

    def add_storage_fault(self, start_hour: float, duration_hours: float,
                          target: str | None = None,
                          severity: float = 1.0) -> "FailureScenario":
        self.events.append(FailureEvent("storage_fault", start_hour,
                                        duration_hours, severity, target))
        return self

    def add_weather_crisis(self, start_hour: float, duration_hours: float,
                           severity: float = 0.1) -> "FailureScenario":
        self.events.append(FailureEvent("weather_crisis", start_hour,
                                        duration_hours, severity))
        return self

    def add_grid_disconnect(self, start_hour: float,
                            duration_hours: float) -> "FailureScenario":
        self.events.append(FailureEvent("grid_disconnect", start_hour,
                                        duration_hours, 1.0))
        return self

    def add_demand_surge(self, start_hour: float, duration_hours: float,
                         multiplier: float = 1.5) -> "FailureScenario":
        self.events.append(FailureEvent("demand_surge", start_hour,
                                        duration_hours, metadata={"multiplier": multiplier}))
        return self

    def add_simultaneous(self, start_hour: float, num_components: int = 2,
                         duration_hours: float = 48.0) -> "FailureScenario":
        self.events.append(FailureEvent("simultaneous", start_hour,
                                        duration_hours,
                                        metadata={"num_components": num_components}))
        return self

    def randomize(self, hours: int = 8760, min_events: int = 1,
                  max_events: int = 5) -> "FailureScenario":
        """Generate a random mix of failure events."""
        self.events = []
        n = self.rng.integers(min_events, max_events + 1)
        event_types = ["source_trip", "weather_crisis", "grid_disconnect",
                       "demand_surge", "storage_fault", "simultaneous"]
        type_weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]

        used_starts = []
        for _ in range(n):
            etype = self.rng.choice(event_types, p=type_weights)
            # Ensure events are spaced apart
            for attempt in range(20):
                start = self.rng.uniform(24, hours - 168)
                if all(abs(start - s) > 48 for s in used_starts):
                    break
            used_starts.append(start)

            if etype == "source_trip":
                dur = self.rng.uniform(6, 168)
                self.add_source_trip(start, dur, severity=self.rng.uniform(0.5, 1.0))
            elif etype == "storage_fault":
                dur = self.rng.uniform(12, 120)
                self.add_storage_fault(start, dur, severity=self.rng.uniform(0.5, 1.0))
            elif etype == "weather_crisis":
                dur = self.rng.uniform(48, 336)
                self.add_weather_crisis(start, dur, severity=self.rng.uniform(0.05, 0.25))
            elif etype == "grid_disconnect":
                dur = self.rng.uniform(6, 72)
                self.add_grid_disconnect(start, dur)
            elif etype == "demand_surge":
                dur = self.rng.uniform(24, 168)
                self.add_demand_surge(start, dur, multiplier=self.rng.uniform(1.2, 1.8))
            elif etype == "simultaneous":
                dur = self.rng.uniform(12, 72)
                nc = self.rng.integers(2, 4)
                self.add_simultaneous(start, num_components=nc, duration_hours=dur)
        return self

    def apply(self, config: GridConfig, hours: int = 8760,
              dt_hours: float = 1.0) -> tuple[GridConfig, np.ndarray, dict]:
        """
        Apply failure events to a configuration.

        Returns (modified_config, weather_factors, failure_timeline).
        The failure_timeline dict is keyed by (start, end) tuples with
        modification dicts for FailureAwareDispatcher.
        """
        config = copy.deepcopy(config)
        n_steps = int(hours / dt_hours)

        # Generate base weather
        rng_weather = np.random.default_rng(self.rng.integers(0, 2**31))
        weather = np.clip(
            0.7 + 0.3 * rng_weather.random(n_steps) +
            0.1 * np.sin(np.arange(n_steps) * 2 * np.pi / (24 / dt_hours * 7)),
            0.1, 1.0
        )

        failure_timeline = {}

        for event in self.events:
            start = event.start_hour
            end = start + event.duration_hours

            if event.type == "weather_crisis":
                # Modify weather array with ramp
                ramp = 6.0  # hours
                for i in range(n_steps):
                    h = i * dt_hours
                    if start - ramp <= h <= end + ramp:
                        if h < start:
                            factor = 1.0 - (1.0 - event.severity) * (h - (start - ramp)) / ramp
                        elif h > end:
                            factor = event.severity + (1.0 - event.severity) * (h - end) / ramp
                        else:
                            factor = event.severity
                        weather[i] = min(weather[i], max(0.05, factor))

            elif event.type == "source_trip":
                target_name = event.target
                for j, src in enumerate(config.sources):
                    if target_name is None or src.name == target_name:
                        if target_name is None:
                            # Pick one random source
                            idx = self.rng.integers(0, len(config.sources))
                            config.sources[idx] = FailedSource(
                                config.sources[idx],
                                [(start, end, event.severity)]
                            )
                            break
                        else:
                            config.sources[j] = FailedSource(
                                src, [(start, end, event.severity)]
                            )

            elif event.type == "storage_fault":
                if config.storage_units:
                    idx = self.rng.integers(0, len(config.storage_units))
                    config.storage_units[idx] = FailedStorage(
                        config.storage_units[idx],
                        [(start, end, event.severity)]
                    )

            elif event.type == "grid_disconnect":
                failure_timeline[(start, end)] = {"grid_interconnect_kw": 0}

            elif event.type == "demand_surge":
                mult = event.metadata.get("multiplier", 1.5)
                failure_timeline[(start, end)] = {"demand_multiplier": mult}

            elif event.type == "simultaneous":
                nc = event.metadata.get("num_components", 2)
                # Trip random sources and storage
                n_src = min(nc, len(config.sources))
                if n_src > 0:
                    idxs = self.rng.choice(len(config.sources), size=n_src,
                                           replace=False)
                    for idx in idxs:
                        config.sources[idx] = FailedSource(
                            config.sources[idx],
                            [(start, end, event.severity)]
                        )
                n_remain = nc - n_src
                if n_remain > 0 and config.storage_units:
                    n_stor = min(n_remain, len(config.storage_units))
                    idxs = self.rng.choice(len(config.storage_units),
                                           size=n_stor, replace=False)
                    for idx in idxs:
                        config.storage_units[idx] = FailedStorage(
                            config.storage_units[idx],
                            [(start, end, 0.8)]
                        )

        return config, weather, failure_timeline

    def describe(self) -> str:
        """Human-readable summary of failure events."""
        if not self.events:
            return "No failures (baseline)"
        parts = []
        for e in self.events:
            parts.append(f"{e.type} @ h{e.start_hour:.0f} "
                        f"({e.duration_hours:.0f}h, sev={e.severity:.1f})")
        return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Failure-Aware Dispatcher
# ──────────────────────────────────────────────────────────────────────

class FailureAwareDispatcher(EnergyDispatcher):
    """Dispatcher with time-varying grid and demand modifications."""

    def __init__(self, config: GridConfig,
                 failure_timeline: dict | None = None):
        super().__init__(config)
        self.failure_timeline = failure_timeline or {}

    def dispatch_step(self, hour_of_year: float, dt_hours: float,
                      weather_factor: float = 1.0,
                      temperature_c: float = 20.0) -> DispatchResult:
        original_grid = self.config.grid_interconnect_kw
        demand_mult = 1.0

        for (start, end), mods in self.failure_timeline.items():
            if start <= hour_of_year < end:
                if "grid_interconnect_kw" in mods:
                    self.config.grid_interconnect_kw = mods["grid_interconnect_kw"]
                if "demand_multiplier" in mods:
                    demand_mult = mods["demand_multiplier"]

        # Wrap demand if needed
        original_demand_fn = None
        if demand_mult != 1.0:
            original_demand_fn = self.load_profile.demand_kw
            mult = demand_mult
            self.load_profile.demand_kw = lambda h, t, _fn=original_demand_fn, _m=mult: _fn(h, t) * _m

        result = super().dispatch_step(hour_of_year, dt_hours,
                                       weather_factor, temperature_c)

        # Restore
        self.config.grid_interconnect_kw = original_grid
        if original_demand_fn is not None:
            self.load_profile.demand_kw = original_demand_fn

        return result


# ──────────────────────────────────────────────────────────────────────
# Resilience Metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ResilienceResult:
    """Extended resilience metrics for a simulation run."""
    lolp: float
    lole_hours: float
    ens_kwh: float
    system_reserve_margin: float
    min_reserve_margin: float
    max_deficit_kw: float
    max_deficit_hour: float
    max_deficit_duration_hours: float
    recovery_time_avg: float
    recovery_time_max: float
    min_aggregate_soc: float
    storage_depletion_hours: float
    base_metrics: dict


class ResilienceMetrics:
    """Compute extended resilience metrics from simulation results."""

    @staticmethod
    def compute(dispatcher: EnergyDispatcher) -> ResilienceResult:
        results = dispatcher.results
        if not results:
            return ResilienceResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {})

        dt = results[1].hour - results[0].hour if len(results) > 1 else 1.0
        n = len(results)

        unmet = np.array([r.unmet_demand_kw for r in results])
        demand = np.array([r.demand_kw for r in results])
        generation = np.array([r.total_generation_kw for r in results])

        # LOLP and LOLE
        deficit_mask = unmet > 0.1  # Small threshold to avoid float noise
        lolp = np.sum(deficit_mask) / n
        lole = lolp * n * dt  # Hours of lost load

        # ENS
        ens = np.sum(unmet) * dt

        # Reserve margins
        total_rated = sum(s.rated_kw for s in dispatcher.config.sources)
        peak_demand = np.max(demand)
        system_reserve = (total_rated - peak_demand) / peak_demand if peak_demand > 0 else 0

        hourly_supply = generation + np.array([r.total_storage_discharge_kw for r in results]) + \
                       np.array([r.grid_import_kw for r in results])
        hourly_margin = (hourly_supply - demand) / np.maximum(demand, 1)
        min_reserve = float(np.min(hourly_margin))

        # Deficit analysis
        max_deficit = float(np.max(unmet))
        max_deficit_hour = float(results[int(np.argmax(unmet))].hour) if max_deficit > 0 else 0

        # Contiguous deficit windows
        deficit_windows = []
        in_deficit = False
        start_idx = 0
        for i in range(n):
            if deficit_mask[i] and not in_deficit:
                in_deficit = True
                start_idx = i
            elif not deficit_mask[i] and in_deficit:
                in_deficit = False
                deficit_windows.append((start_idx, i))
        if in_deficit:
            deficit_windows.append((start_idx, n))

        max_deficit_dur = max((e - s) * dt for s, e in deficit_windows) if deficit_windows else 0

        # Recovery time: hours from end of deficit until aggregate SOC > 20%
        recovery_times = []
        for _, end_idx in deficit_windows:
            recovered = False
            for j in range(end_idx, min(end_idx + int(168 / dt), n)):
                states = results[j].storage_states
                if states:
                    avg_soc = np.mean([s.soc for s in states.values()])
                    if avg_soc > 0.2:
                        recovery_times.append((j - end_idx) * dt)
                        recovered = True
                        break
            if not recovered:
                recovery_times.append(168.0)  # Cap at 1 week

        recovery_avg = float(np.mean(recovery_times)) if recovery_times else 0
        recovery_max = float(np.max(recovery_times)) if recovery_times else 0

        # Storage depletion
        agg_socs = []
        for r in results:
            if r.storage_states:
                agg_socs.append(np.mean([s.soc for s in r.storage_states.values()]))
            else:
                agg_socs.append(1.0)
        agg_socs = np.array(agg_socs)
        min_soc = float(np.min(agg_socs))
        depletion_hours = float(np.sum(agg_socs < 0.05) * dt)

        base = dispatcher.compute_metrics()

        return ResilienceResult(
            lolp=lolp,
            lole_hours=lole,
            ens_kwh=ens,
            system_reserve_margin=system_reserve,
            min_reserve_margin=min_reserve,
            max_deficit_kw=max_deficit,
            max_deficit_hour=max_deficit_hour,
            max_deficit_duration_hours=max_deficit_dur,
            recovery_time_avg=recovery_avg,
            recovery_time_max=recovery_max,
            min_aggregate_soc=min_soc,
            storage_depletion_hours=depletion_hours,
            base_metrics=base,
        )


# ──────────────────────────────────────────────────────────────────────
# Scenario Runner
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioRunResult:
    """Result of a single scenario run."""
    config_id: int
    failure_id: int
    config_name: str
    source_types: list[str]
    storage_types: list[str]
    failure_desc: str
    resilience: ResilienceResult
    elapsed_seconds: float


class ScenarioRunner:
    """Batch execution: N configs x M failure scenarios with analysis."""

    def __init__(
        self,
        scale: str = "community",
        tier: str = "conventional",
        seed: int = 42,
        hours: int = 8760,
        dt_hours: float = 1.0,
    ):
        self.generator = ConfigGenerator(scale=scale, tier=tier, seed=seed)
        self.scale = scale
        self.tier = tier
        self.hours = hours
        self.dt_hours = dt_hours
        self.seed = seed
        self.results: list[ScenarioRunResult] = []

    def _run_one(self, config: GridConfig, failure: FailureScenario | None,
                 config_id: int, failure_id: int) -> ScenarioRunResult:
        """Run a single config+failure combination."""
        config_copy = copy.deepcopy(config)
        t0 = time.time()

        if failure and failure.events:
            mod_config, weather, timeline = failure.apply(config_copy,
                                                          self.hours, self.dt_hours)
            dispatcher = FailureAwareDispatcher(mod_config, timeline)
            dispatcher.simulate(hours=self.hours, dt_hours=self.dt_hours,
                               weather_factors=weather)
        else:
            dispatcher = EnergyDispatcher(config_copy)
            dispatcher.simulate(hours=self.hours, dt_hours=self.dt_hours)

        resilience = ResilienceMetrics.compute(dispatcher)
        elapsed = time.time() - t0

        src_types = [type(s).__name__ for s in config.sources]
        stor_types = [type(s).__name__ for s in config.storage_units]
        fail_desc = failure.describe() if failure else "Baseline"

        return ScenarioRunResult(
            config_id=config_id,
            failure_id=failure_id,
            config_name=config.name,
            source_types=src_types,
            storage_types=stor_types,
            failure_desc=fail_desc,
            resilience=resilience,
            elapsed_seconds=elapsed,
        )

    def run(self, n_configs: int = 10, n_failures: int = 3,
            include_baseline: bool = True) -> list[ScenarioRunResult]:
        """Run N configs x M failure scenarios."""
        self.results = []
        configs = self.generator.generate_batch(n_configs)
        failure_rng = np.random.default_rng(self.seed + 1000)

        total = n_configs * (n_failures + (1 if include_baseline else 0))
        done = 0

        for ci, config in enumerate(configs):
            # Baseline run
            if include_baseline:
                result = self._run_one(config, None, ci, -1)
                self.results.append(result)
                done += 1
                print(f"\r  Progress: {done}/{total} runs", end="", flush=True)

            # Failure scenario runs
            for fi in range(n_failures):
                failure = FailureScenario(
                    seed=int(failure_rng.integers(0, 2**31))
                ).randomize(hours=self.hours)
                result = self._run_one(config, failure, ci, fi)
                self.results.append(result)
                done += 1
                print(f"\r  Progress: {done}/{total} runs", end="", flush=True)

        print()
        return self.results

    def analyze(self) -> dict:
        """Compute aggregate statistics across all runs."""
        if not self.results:
            return {}

        baseline = [r for r in self.results if r.failure_id == -1]
        stressed = [r for r in self.results if r.failure_id >= 0]

        def _stats(values):
            arr = np.array(values) if values else np.array([0])
            return {
                "mean": float(np.mean(arr)), "median": float(np.median(arr)),
                "std": float(np.std(arr)), "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }

        # Distribution stats
        all_lolp = [r.resilience.lolp for r in self.results]
        all_ens = [r.resilience.ens_kwh for r in self.results]
        all_recovery = [r.resilience.recovery_time_max for r in self.results]
        all_deficit = [r.resilience.max_deficit_kw for r in self.results]

        # Component impact analysis
        component_impact = {}
        all_source_types = set()
        all_storage_types = set()
        for r in self.results:
            all_source_types.update(r.source_types)
            all_storage_types.update(r.storage_types)

        for comp in all_source_types | all_storage_types:
            with_comp = [r.resilience.lolp for r in stressed
                         if comp in r.source_types or comp in r.storage_types]
            without_comp = [r.resilience.lolp for r in stressed
                           if comp not in r.source_types and comp not in r.storage_types]
            if with_comp and without_comp:
                impact = np.mean(without_comp) - np.mean(with_comp)
            else:
                impact = 0.0
            component_impact[comp] = {
                "avg_lolp_with": float(np.mean(with_comp)) if with_comp else 0,
                "avg_lolp_without": float(np.mean(without_comp)) if without_comp else 0,
                "resilience_impact": float(impact),
                "count": len(with_comp),
            }

        # Failure type impact
        failure_type_impact = {}
        for etype in ["source_trip", "weather_crisis", "grid_disconnect",
                       "demand_surge", "storage_fault", "simultaneous"]:
            matching = [r for r in stressed
                       if etype in r.failure_desc]
            if matching:
                failure_type_impact[etype] = {
                    "avg_ens": float(np.mean([r.resilience.ens_kwh for r in matching])),
                    "avg_lolp": float(np.mean([r.resilience.lolp for r in matching])),
                    "avg_recovery": float(np.mean([r.resilience.recovery_time_max for r in matching])),
                    "count": len(matching),
                }

        # Best/worst configs
        config_avg_lolp = {}
        for r in stressed:
            config_avg_lolp.setdefault(r.config_id, []).append(r.resilience.lolp)
        config_scores = {k: np.mean(v) for k, v in config_avg_lolp.items()}
        sorted_configs = sorted(config_scores.items(), key=lambda x: x[1])

        most_resilient = []
        for cid, score in sorted_configs[:3]:
            rep = next(r for r in self.results if r.config_id == cid)
            most_resilient.append({
                "config_name": rep.config_name,
                "avg_lolp": float(score),
                "sources": rep.source_types,
                "storage": rep.storage_types,
            })

        least_resilient = []
        for cid, score in sorted_configs[-3:]:
            rep = next(r for r in self.results if r.config_id == cid)
            least_resilient.append({
                "config_name": rep.config_name,
                "avg_lolp": float(score),
                "sources": rep.source_types,
                "storage": rep.storage_types,
            })

        return {
            "n_configs": len(set(r.config_id for r in self.results)),
            "n_total_runs": len(self.results),
            "lolp": _stats(all_lolp),
            "ens_kwh": _stats(all_ens),
            "recovery_time": _stats(all_recovery),
            "max_deficit_kw": _stats(all_deficit),
            "component_impact": component_impact,
            "failure_type_impact": failure_type_impact,
            "most_resilient": most_resilient,
            "least_resilient": least_resilient,
            "baseline_avg_lolp": float(np.mean([r.resilience.lolp for r in baseline])) if baseline else 0,
            "stressed_avg_lolp": float(np.mean([r.resilience.lolp for r in stressed])) if stressed else 0,
        }

    def print_summary(self):
        """Print formatted analysis summary."""
        a = self.analyze()
        if not a:
            print("No results to analyze.")
            return

        print(f"\n{'='*70}")
        print(f"  RESILIENCE ANALYSIS — {a['n_configs']} configs, "
              f"{a['n_total_runs']} total runs")
        print(f"  Scale: {self.scale} | Tier: {self.tier}")
        print(f"{'='*70}")

        print(f"\n  ── Reliability Indices ──")
        lolp = a["lolp"]
        print(f"  LOLP (Loss of Load Prob):   mean={lolp['mean']:.4f}  "
              f"median={lolp['median']:.4f}  p95={lolp['p95']:.4f}")
        ens = a["ens_kwh"]
        print(f"  ENS  (Energy Not Served):   mean={ens['mean']:>10,.0f} kWh  "
              f"max={ens['max']:>10,.0f} kWh")
        rec = a["recovery_time"]
        print(f"  Recovery Time (max):        mean={rec['mean']:>6.1f}h  "
              f"worst={rec['max']:>6.1f}h")
        deficit = a["max_deficit_kw"]
        print(f"  Max Deficit:                mean={deficit['mean']:>10,.0f} kW  "
              f"worst={deficit['max']:>10,.0f} kW")

        print(f"\n  ── Baseline vs Stressed ──")
        print(f"  Baseline avg LOLP:  {a['baseline_avg_lolp']:.4f}")
        print(f"  Stressed avg LOLP:  {a['stressed_avg_lolp']:.4f}")

        if a["component_impact"]:
            print(f"\n  ── Component Resilience Impact ──")
            print(f"  {'Component':30s} {'LOLP with':>10s} {'LOLP w/o':>10s} {'Impact':>10s}")
            sorted_impact = sorted(a["component_impact"].items(),
                                   key=lambda x: x[1]["resilience_impact"], reverse=True)
            for comp, info in sorted_impact:
                if info["count"] > 0:
                    print(f"  {comp:30s} {info['avg_lolp_with']:>10.4f} "
                          f"{info['avg_lolp_without']:>10.4f} "
                          f"{info['resilience_impact']:>+10.4f}")

        if a["failure_type_impact"]:
            print(f"\n  ── Failure Type Impact ──")
            print(f"  {'Failure Type':25s} {'Avg LOLP':>10s} {'Avg ENS (kWh)':>15s} {'Avg Recovery':>12s}")
            for ftype, info in sorted(a["failure_type_impact"].items(),
                                       key=lambda x: x[1]["avg_ens"], reverse=True):
                print(f"  {ftype:25s} {info['avg_lolp']:>10.4f} "
                      f"{info['avg_ens']:>15,.0f} {info['avg_recovery']:>10.1f}h")

        if a["most_resilient"]:
            print(f"\n  ── Most Resilient Configurations ──")
            for cfg in a["most_resilient"]:
                print(f"  {cfg['config_name']:20s}  LOLP={cfg['avg_lolp']:.4f}  "
                      f"Src={cfg['sources']}  Stor={cfg['storage']}")

        if a["least_resilient"]:
            print(f"\n  ── Least Resilient Configurations ──")
            for cfg in a["least_resilient"]:
                print(f"  {cfg['config_name']:20s}  LOLP={cfg['avg_lolp']:.4f}  "
                      f"Src={cfg['sources']}  Stor={cfg['storage']}")

    def plot_results(self, save_path: str = "resilience_analysis.png"):
        """Generate 6-panel resilience analysis figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .styles import PALETTE as C, FONTS, apply_style, styled_legend

        a = self.analyze()
        if not a:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
        fig.suptitle(f"Resilience Analysis \u2014 {self.scale.title()} / {self.tier.title()} "
                     f"({a['n_configs']} configs, {a['n_total_runs']} runs)",
                     **FONTS["suptitle"])

        # 1. LOLP distribution
        ax = axes[0, 0]
        apply_style(ax)
        lolps = [r.resilience.lolp for r in self.results]
        ax.hist(lolps, bins=20, color=C["discharge"], alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(lolps), color=C["demand"], linestyle="--", label=f"Mean={np.mean(lolps):.3f}")
        ax.set_xlabel("LOLP"); ax.set_ylabel("Count")
        ax.set_title("Loss of Load Probability", **FONTS["title"])
        styled_legend(ax)

        # 2. ENS distribution
        ax = axes[0, 1]
        apply_style(ax)
        ens_vals = [r.resilience.ens_kwh for r in self.results]
        ax.hist(ens_vals, bins=20, color=C["demand"], alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(ens_vals), color=C["discharge"], linestyle="--", label=f"Mean={np.mean(ens_vals):,.0f}")
        ax.set_xlabel("ENS (kWh)"); ax.set_ylabel("Count")
        ax.set_title("Energy Not Served", **FONTS["title"])
        styled_legend(ax)

        # 3. Recovery time distribution
        ax = axes[0, 2]
        apply_style(ax)
        rec_vals = [r.resilience.recovery_time_max for r in self.results]
        ax.hist(rec_vals, bins=20, color=C["charge"], alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(rec_vals), color=C["demand"], linestyle="--")
        ax.set_xlabel("Max Recovery Time (hours)"); ax.set_ylabel("Count")
        ax.set_title("Recovery Time Distribution", **FONTS["title"])

        # 4. Component impact
        ax = axes[1, 0]
        apply_style(ax)
        impact = a.get("component_impact", {})
        if impact:
            comps = sorted(impact.keys(), key=lambda c: impact[c]["resilience_impact"],
                          reverse=True)[:8]
            impacts = [impact[c]["resilience_impact"] for c in comps]
            colors = [C["charge"] if v > 0 else C["demand"] for v in impacts]
            short_names = [c.replace("Battery", "").replace("Storage", "")[:15] for c in comps]
            ax.barh(short_names, impacts, color=colors, alpha=0.8)
            ax.set_xlabel("LOLP Reduction (positive = more resilient)")
            ax.set_title("Component Resilience Impact", **FONTS["title"])

        # 5. Failure type impact
        ax = axes[1, 1]
        apply_style(ax)
        ft = a.get("failure_type_impact", {})
        if ft:
            types = list(ft.keys())
            ens_by_type = [ft[t]["avg_ens"] for t in types]
            short_types = [t.replace("_", " ").title()[:12] for t in types]
            ax.bar(short_types, ens_by_type, color=C["grid_import"], alpha=0.8)
            ax.set_ylabel("Avg ENS (kWh)")
            ax.set_title("Impact by Failure Type", **FONTS["title"])
            ax.tick_params(axis='x', rotation=30)

        # 6. Baseline vs stressed per config
        ax = axes[1, 2]
        apply_style(ax)
        baseline = [r for r in self.results if r.failure_id == -1]
        if baseline:
            config_ids = sorted(set(r.config_id for r in baseline))[:10]
            x = np.arange(len(config_ids))
            bl_lolp = []
            st_lolp = []
            for cid in config_ids:
                bl = [r.resilience.lolp for r in self.results
                      if r.config_id == cid and r.failure_id == -1]
                st = [r.resilience.lolp for r in self.results
                      if r.config_id == cid and r.failure_id >= 0]
                bl_lolp.append(np.mean(bl) if bl else 0)
                st_lolp.append(np.mean(st) if st else 0)
            w = 0.35
            ax.bar(x - w/2, bl_lolp, w, label="Baseline", color=C["discharge"], alpha=0.8)
            ax.bar(x + w/2, st_lolp, w, label="Stressed", color=C["demand"], alpha=0.8)
            ax.set_xlabel("Config #"); ax.set_ylabel("Avg LOLP")
            ax.set_title("Baseline vs Stressed", **FONTS["title"])
            styled_legend(ax)
            ax.set_xticks(x)
            ax.set_xticklabels([f"C{i}" for i in config_ids])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved resilience analysis to {save_path}")

    def plot_heatmap(self, save_path: str = "component_heatmap.png"):
        """Source x Storage interaction heatmap showing avg LOLP."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .styles import FONTS

        stressed = [r for r in self.results if r.failure_id >= 0]
        if not stressed:
            return

        all_src = sorted(set(s for r in stressed for s in r.source_types))
        all_stor = sorted(set(s for r in stressed for s in r.storage_types))

        if len(all_src) < 2 or len(all_stor) < 2:
            return

        matrix = np.full((len(all_src), len(all_stor)), np.nan)
        for i, src in enumerate(all_src):
            for j, stor in enumerate(all_stor):
                matching = [r.resilience.lolp for r in stressed
                           if src in r.source_types and stor in r.storage_types]
                if matching:
                    matrix[i, j] = np.mean(matching)

        fig, ax = plt.subplots(figsize=(max(8, len(all_stor)),
                                        max(5, len(all_src) * 0.6)),
                               facecolor="white")
        im = ax.imshow(matrix, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(all_stor)))
        ax.set_xticklabels([s.replace("Battery", "").replace("Storage", "")[:12]
                           for s in all_stor], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_src)))
        ax.set_yticklabels([s[:15] for s in all_src], fontsize=8)
        ax.set_title(f"Avg LOLP by Source x Storage Combination\n"
                     f"({self.scale.title()} / {self.tier.title()})",
                     **FONTS["suptitle"])

        for i in range(len(all_src)):
            for j in range(len(all_stor)):
                if not np.isnan(matrix[i, j]):
                    text_color = "white" if matrix[i, j] > 0.3 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                           fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, label="Avg LOLP", shrink=0.8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved component heatmap to {save_path}")
