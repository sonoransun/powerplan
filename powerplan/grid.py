"""
Composite grid model and dispatch optimizer.

Manages the interaction between energy sources, storage units, power
controllers, and loads. Implements dispatch strategies to maximize
overall system efficiency.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .storage import StorageUnit
from .sources import EnergySource
from .controllers import PowerController, ControllerState
from .profiles import LoadProfile, DeploymentScale


@dataclass
class GridConfig:
    """Configuration for a composite power grid."""
    name: str
    scale: DeploymentScale
    sources: list[EnergySource] = field(default_factory=list)
    storage_units: list[StorageUnit] = field(default_factory=list)
    controllers: list[PowerController] = field(default_factory=list)
    load_profile: Optional[LoadProfile] = None
    grid_interconnect_kw: float = 0.0  # 0 = island mode


@dataclass
class DispatchResult:
    """Result of a single dispatch time step."""
    hour: float
    demand_kw: float
    total_generation_kw: float
    total_storage_discharge_kw: float
    total_storage_charge_kw: float
    grid_import_kw: float
    grid_export_kw: float
    curtailment_kw: float
    unmet_demand_kw: float
    controller_losses_kw: float
    storage_losses_kw: float
    system_efficiency: float
    renewable_fraction: float
    source_outputs: dict  # name -> kW
    storage_states: dict  # name -> StorageState
    controller_states: dict  # name -> ControllerState


class EnergyDispatcher:
    """
    Optimal dispatch engine for heterogeneous power systems.

    Dispatch priority:
    1. Use renewable generation directly for load
    2. Charge storage with excess renewables (prioritize by efficiency)
    3. Discharge storage to meet remaining load (prioritize by efficiency)
    4. Import from grid if interconnected
    5. Curtail excess generation
    """

    def __init__(self, config: GridConfig):
        self.config = config
        if config.load_profile is None:
            config.load_profile = LoadProfile(config.scale)
        self.load_profile = config.load_profile
        self.results: list[DispatchResult] = []

    def _sort_storage_for_charge(self) -> list[StorageUnit]:
        """Sort storage units by charge efficiency (highest first)."""
        return sorted(
            self.config.storage_units,
            key=lambda s: s.charge_efficiency(0, s.soc) * (1 - s.soc),
            reverse=True,
        )

    def _sort_storage_for_discharge(self) -> list[StorageUnit]:
        """Sort storage units by discharge efficiency and available energy."""
        return sorted(
            self.config.storage_units,
            key=lambda s: s.discharge_efficiency(0, s.soc) * s.soc,
            reverse=True,
        )

    def _get_controller_efficiency(self, power_kw: float) -> float:
        """Weighted average controller efficiency across active controllers."""
        if not self.config.controllers or power_kw <= 0:
            return 0.95  # Default assumption
        remaining = power_kw
        weighted_eff = 0.0
        active_capacity = 0.0
        for ctrl in self.config.controllers:
            if remaining <= 0:
                break
            allocated = min(remaining, ctrl.rated_kw)
            frac = allocated / ctrl.rated_kw if ctrl.rated_kw > 0 else 0
            eff = ctrl.efficiency_at_load(frac)
            weighted_eff += eff * allocated
            active_capacity += allocated
            remaining -= allocated
        return weighted_eff / active_capacity if active_capacity > 0 else 0.95

    def dispatch_step(self, hour_of_year: float, dt_hours: float,
                      weather_factor: float = 1.0,
                      temperature_c: float = 20.0) -> DispatchResult:
        """Execute one dispatch time step."""

        # 1. Get demand
        demand = self.load_profile.demand_kw(hour_of_year, temperature_c)

        # 2. Get generation (tracking renewable vs fossil)
        source_outputs = {}
        total_generation = 0.0
        renewable_generation = 0.0
        for src in self.config.sources:
            out = src.step(hour_of_year, dt_hours, weather_factor)
            source_outputs[src.name] = out.power_kw
            total_generation += out.power_kw
            if src.is_renewable:
                renewable_generation += out.power_kw

        # 3. Apply controller efficiency to generation
        ctrl_eff = self._get_controller_efficiency(total_generation)
        usable_generation = total_generation * ctrl_eff
        controller_losses = total_generation * (1 - ctrl_eff)

        # Update controller states
        controller_states = {}
        remaining_gen = total_generation
        for ctrl in self.config.controllers:
            cs = ctrl.convert(min(remaining_gen, ctrl.rated_kw), dt_hours, mode="regulate")
            controller_states[ctrl.name] = cs
            remaining_gen = max(0, remaining_gen - ctrl.rated_kw)

        # 4. Direct supply from renewables
        direct_supply = min(usable_generation, demand)
        remaining_demand = demand - direct_supply
        excess_generation = usable_generation - direct_supply

        # 5. Charge storage with excess generation
        total_charge = 0.0
        storage_losses = 0.0
        if excess_generation > 0:
            for unit in self._sort_storage_for_charge():
                if excess_generation <= 0:
                    break
                charge_power = min(excess_generation, unit.max_charge_kw)
                actual = unit.step(-charge_power, dt_hours)  # Returns negative
                absorbed = abs(actual)
                eff = unit.charge_efficiency(absorbed, unit.soc)
                storage_losses += absorbed * (1 - eff)
                total_charge += absorbed
                excess_generation -= charge_power

        # 6. Discharge storage to meet remaining demand
        total_discharge = 0.0
        if remaining_demand > 0:
            for unit in self._sort_storage_for_discharge():
                if remaining_demand <= 0:
                    break
                discharge_power = min(remaining_demand, unit.max_discharge_kw)
                actual = unit.step(discharge_power, dt_hours)
                delivered = abs(actual)
                total_discharge += delivered
                remaining_demand -= delivered

        # 7. Grid import/export
        grid_import = 0.0
        grid_export = 0.0
        if self.config.grid_interconnect_kw > 0:
            if remaining_demand > 0:
                grid_import = min(remaining_demand, self.config.grid_interconnect_kw)
                remaining_demand -= grid_import
            if excess_generation > 0:
                grid_export = min(excess_generation, self.config.grid_interconnect_kw)
                excess_generation -= grid_export

        # 8. Curtailment and unmet demand
        curtailment = max(0, excess_generation)
        unmet_demand = max(0, remaining_demand)

        # 9. System efficiency: fraction of generation that reaches useful load
        #    (direct use + stored) vs total generated
        useful_from_gen = direct_supply + total_charge  # Generation that wasn't curtailed/lost
        system_efficiency = useful_from_gen / total_generation if total_generation > 0 else 1.0

        # 10. Renewable fraction of load served
        total_served = direct_supply + total_discharge + grid_import
        renew_gen_share = renewable_generation / total_generation if total_generation > 0 else 1.0
        renewable_direct = direct_supply * renew_gen_share
        renewable_fraction = (renewable_direct + total_discharge) / total_served if total_served > 0 else 0

        storage_states = {u.name: u.get_state() for u in self.config.storage_units}

        result = DispatchResult(
            hour=hour_of_year,
            demand_kw=demand,
            total_generation_kw=total_generation,
            total_storage_discharge_kw=total_discharge,
            total_storage_charge_kw=total_charge,
            grid_import_kw=grid_import,
            grid_export_kw=grid_export,
            curtailment_kw=curtailment,
            unmet_demand_kw=unmet_demand,
            controller_losses_kw=controller_losses,
            storage_losses_kw=storage_losses,
            system_efficiency=system_efficiency,
            renewable_fraction=renewable_fraction,
            source_outputs=source_outputs,
            storage_states=storage_states,
            controller_states=controller_states,
        )
        self.results.append(result)
        return result

    def simulate(self, hours: int = 8760, dt_hours: float = 1.0,
                 weather_factors: Optional[np.ndarray] = None) -> list[DispatchResult]:
        """Run full simulation over specified hours."""
        self.results = []
        n_steps = int(hours / dt_hours)

        if weather_factors is None:
            # Synthetic weather: mostly clear, some cloudy/stormy periods
            rng = np.random.default_rng(123)
            weather_factors = np.clip(
                0.7 + 0.3 * rng.random(n_steps) +
                0.1 * np.sin(np.arange(n_steps) * 2 * np.pi / (24 / dt_hours * 7)),
                0.1, 1.0
            )

        temps = self.load_profile._annual_temperature(
            np.arange(0, hours, dt_hours)
        )

        for i in range(n_steps):
            hour = i * dt_hours
            self.dispatch_step(
                hour_of_year=hour,
                dt_hours=dt_hours,
                weather_factor=weather_factors[i],
                temperature_c=temps[i],
            )

        return self.results

    def compute_metrics(self) -> dict:
        """Compute aggregate performance metrics from simulation results."""
        if not self.results:
            return {}

        hours = np.array([r.hour for r in self.results])
        demand = np.array([r.demand_kw for r in self.results])
        generation = np.array([r.total_generation_kw for r in self.results])
        discharge = np.array([r.total_storage_discharge_kw for r in self.results])
        charge = np.array([r.total_storage_charge_kw for r in self.results])
        grid_imp = np.array([r.grid_import_kw for r in self.results])
        grid_exp = np.array([r.grid_export_kw for r in self.results])
        curtail = np.array([r.curtailment_kw for r in self.results])
        ctrl_loss = np.array([r.controller_losses_kw for r in self.results])
        eff = np.array([r.system_efficiency for r in self.results])
        renew = np.array([r.renewable_fraction for r in self.results])

        dt = hours[1] - hours[0] if len(hours) > 1 else 1.0

        total_demand_kwh = np.sum(demand) * dt
        total_gen_kwh = np.sum(generation) * dt
        total_curtail_kwh = np.sum(curtail) * dt
        total_import_kwh = np.sum(grid_imp) * dt
        total_export_kwh = np.sum(grid_exp) * dt
        total_ctrl_loss_kwh = np.sum(ctrl_loss) * dt

        # Capital costs
        source_capex = sum(s.capital_cost_per_kw() * s.rated_kw for s in self.config.sources)
        storage_capex = sum(u.capital_cost_per_kwh() * u.nominal_capacity_kwh
                           for u in self.config.storage_units)
        total_capex = source_capex + storage_capex

        # LCOE estimate (simplified — 20 year, 5% discount)
        discount_factor = sum(1 / (1.05 ** y) for y in range(1, 21))
        annual_gen = total_gen_kwh  # Already annual if 8760 hours
        lcoe = total_capex / (annual_gen * discount_factor) if annual_gen > 0 else float('inf')

        return {
            "simulation_hours": len(hours) * dt,
            "total_demand_kwh": total_demand_kwh,
            "total_generation_kwh": total_gen_kwh,
            "total_curtailment_kwh": total_curtail_kwh,
            "total_grid_import_kwh": total_import_kwh,
            "total_grid_export_kwh": total_export_kwh,
            "total_controller_losses_kwh": total_ctrl_loss_kwh,
            "generation_to_demand_ratio": total_gen_kwh / total_demand_kwh if total_demand_kwh > 0 else 0,
            "curtailment_fraction": total_curtail_kwh / total_gen_kwh if total_gen_kwh > 0 else 0,
            "avg_system_efficiency": float(np.mean(eff[eff > 0])) if np.any(eff > 0) else 0,
            "avg_renewable_fraction": float(np.mean(renew)),
            "peak_demand_kw": float(np.max(demand)),
            "peak_generation_kw": float(np.max(generation)),
            "self_sufficiency": 1 - (total_import_kwh / total_demand_kwh) if total_demand_kwh > 0 else 0,
            "total_capex_usd": total_capex,
            "estimated_lcoe_usd_kwh": lcoe,
            "source_details": [s.summary() for s in self.config.sources],
            "storage_details": [u.summary() for u in self.config.storage_units],
            "controller_details": [c.summary() for c in self.config.controllers],
        }
