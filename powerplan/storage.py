"""
Energy storage technology models with realistic electrochemical and mechanical parameters.

Each storage class models:
- Capacity (kWh), power rating (kW)
- Round-trip efficiency with state-dependent curves
- Self-discharge, degradation, thermal behavior
- Charge/discharge constraints
- Capital and operational costs
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class StorageState:
    """Instantaneous state of a storage unit."""
    soc: float              # State of charge [0, 1]
    temperature_c: float    # Cell/unit temperature °C
    cycle_count: float      # Equivalent full cycles
    health: float           # State of health [0, 1]
    power_kw: float         # Current power (+ discharge, - charge)
    energy_kwh: float       # Current stored energy


class StorageUnit(ABC):
    """Base class for all energy storage technologies."""

    def __init__(
        self,
        name: str,
        capacity_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        initial_soc: float = 0.5,
        ambient_temp_c: float = 25.0,
        units: int = 1,
    ):
        self.name = name
        self.nominal_capacity_kwh = capacity_kwh * units
        self.max_charge_kw = max_charge_kw * units
        self.max_discharge_kw = max_discharge_kw * units
        self.units = units

        self.soc = initial_soc
        self.temperature_c = ambient_temp_c
        self.ambient_temp_c = ambient_temp_c
        self.cycle_count = 0.0
        self.health = 1.0
        self.cumulative_throughput_kwh = 0.0

    @property
    def effective_capacity_kwh(self) -> float:
        return self.nominal_capacity_kwh * self.health

    @property
    def stored_energy_kwh(self) -> float:
        return self.soc * self.effective_capacity_kwh

    @abstractmethod
    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        """Charging efficiency at given power and SOC [0, 1]."""

    @abstractmethod
    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        """Discharging efficiency at given power and SOC [0, 1]."""

    @abstractmethod
    def self_discharge_rate(self) -> float:
        """Self-discharge rate per hour as fraction of stored energy."""

    @abstractmethod
    def degradation_per_cycle(self) -> float:
        """Health degradation per equivalent full cycle."""

    @abstractmethod
    def capital_cost_per_kwh(self) -> float:
        """Capital cost in $/kWh of capacity."""

    @abstractmethod
    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        """Update and return temperature after operating at power for dt."""

    def clamp_charge_power(self, requested_kw: float) -> float:
        """Clamp charging power respecting limits and SOC."""
        if self.soc >= 0.98:
            # Taper near full
            max_now = self.max_charge_kw * max(0, (1.0 - self.soc) / 0.02)
        else:
            max_now = self.max_charge_kw
        return min(requested_kw, max_now)

    def clamp_discharge_power(self, requested_kw: float) -> float:
        """Clamp discharging power respecting limits and SOC."""
        if self.soc <= 0.05:
            max_now = self.max_discharge_kw * max(0, self.soc / 0.05)
        else:
            max_now = self.max_discharge_kw
        return min(requested_kw, max_now)

    def step(self, power_kw: float, dt_hours: float) -> float:
        """
        Advance one time step. power_kw > 0 means discharge, < 0 means charge.
        Returns actual power delivered (+) or absorbed (-) in kW.
        """
        # Self-discharge
        self_discharge = self.self_discharge_rate() * dt_hours
        self.soc = max(0, self.soc - self_discharge)

        if power_kw > 0:
            # Discharge: deliver `clamped` kW, pull clamped/eff from store
            clamped = self.clamp_discharge_power(power_kw)
            eff = self.discharge_efficiency(clamped, self.soc)
            energy_from_store = clamped * dt_hours / eff if eff > 0 else 0
            available = self.stored_energy_kwh
            if energy_from_store > available:
                energy_from_store = available
                clamped = energy_from_store * eff / dt_hours if dt_hours > 0 else 0
            self.soc -= energy_from_store / self.effective_capacity_kwh if self.effective_capacity_kwh > 0 else 0
            actual_power = clamped  # Power delivered to load
        elif power_kw < 0:
            # Charge
            charge_kw = abs(power_kw)
            clamped = self.clamp_charge_power(charge_kw)
            eff = self.charge_efficiency(clamped, self.soc)
            energy_to_store = clamped * eff * dt_hours
            headroom = self.effective_capacity_kwh - self.stored_energy_kwh
            if energy_to_store > headroom:
                energy_to_store = headroom
                clamped = energy_to_store / (eff * dt_hours) if (eff * dt_hours) > 0 else 0
            self.soc += energy_to_store / self.effective_capacity_kwh if self.effective_capacity_kwh > 0 else 0
            actual_power = -clamped
        else:
            actual_power = 0.0

        # Degradation tracking
        throughput = abs(actual_power) * dt_hours
        self.cumulative_throughput_kwh += throughput
        equiv_cycles = throughput / (2 * self.nominal_capacity_kwh) if self.nominal_capacity_kwh > 0 else 0
        self.cycle_count += equiv_cycles
        self.health = max(0.0, self.health - equiv_cycles * self.degradation_per_cycle())

        # Thermal
        self.temperature_c = self.thermal_model(actual_power, dt_hours)

        self.soc = np.clip(self.soc, 0.0, 1.0)
        return actual_power

    def get_state(self) -> StorageState:
        return StorageState(
            soc=self.soc,
            temperature_c=self.temperature_c,
            cycle_count=self.cycle_count,
            health=self.health,
            power_kw=0.0,
            energy_kwh=self.stored_energy_kwh,
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "nominal_capacity_kwh": self.nominal_capacity_kwh,
            "effective_capacity_kwh": self.effective_capacity_kwh,
            "soc": self.soc,
            "health": self.health,
            "cycles": self.cycle_count,
            "temperature_c": self.temperature_c,
            "capital_cost": self.capital_cost_per_kwh() * self.nominal_capacity_kwh,
        }


class LithiumIonBattery(StorageUnit):
    """
    NMC/LFP lithium-ion battery model.
    Realistic parameters for grid-scale or residential Li-ion.
    """

    def __init__(self, capacity_kwh=13.5, max_power_kw=5.0, chemistry="nmc",
                 units=1, **kwargs):
        super().__init__(
            name=f"Li-ion ({chemistry.upper()})",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )
        self.chemistry = chemistry
        # NMC vs LFP differences
        self._base_efficiency = 0.96 if chemistry == "nmc" else 0.94
        self._cycle_life = 3000 if chemistry == "nmc" else 6000
        self._cost_per_kwh = 180 if chemistry == "nmc" else 150  # $/kWh 2025

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # Efficiency drops at high SOC and high C-rate
        c_rate = power_kw / self.nominal_capacity_kwh if self.nominal_capacity_kwh > 0 else 0
        base = self._base_efficiency
        soc_penalty = 0.03 * max(0, soc - 0.8) / 0.2
        crate_penalty = 0.02 * min(c_rate, 2.0)
        temp_penalty = 0.01 * max(0, (self.temperature_c - 40) / 20)
        return max(0.75, base - soc_penalty - crate_penalty - temp_penalty)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        c_rate = power_kw / self.nominal_capacity_kwh if self.nominal_capacity_kwh > 0 else 0
        base = self._base_efficiency
        soc_penalty = 0.04 * max(0, (0.2 - soc) / 0.2)
        crate_penalty = 0.02 * min(c_rate, 2.0)
        return max(0.75, base - soc_penalty - crate_penalty)

    def self_discharge_rate(self) -> float:
        return 0.0002  # ~0.5% per day

    def degradation_per_cycle(self) -> float:
        return 0.8 / self._cycle_life  # 80% EOL

    def capital_cost_per_kwh(self) -> float:
        return self._cost_per_kwh

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        heat_gen = abs(power_kw) * (1 - self._base_efficiency) * 0.5  # kW thermal
        thermal_mass = 0.5 * self.nominal_capacity_kwh  # kJ/°C approximation
        cooling_rate = 0.1  # natural convection coefficient
        dT = (heat_gen * 3600 * dt_hours - cooling_rate * (self.temperature_c - self.ambient_temp_c) * 3600 * dt_hours) / max(thermal_mass, 1)
        return self.temperature_c + dT * 0.001  # scaled


class SodiumSolidStateBattery(StorageUnit):
    """
    Sodium solid-state battery — emerging technology with high safety,
    wide temperature tolerance, and low-cost materials.
    """

    def __init__(self, capacity_kwh=10.0, max_power_kw=3.0, units=1, **kwargs):
        super().__init__(
            name="Na Solid-State",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # Solid-state electrolyte: lower internal resistance, flatter efficiency
        base = 0.92
        soc_penalty = 0.02 * max(0, soc - 0.85) / 0.15
        return max(0.80, base - soc_penalty)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        base = 0.91
        soc_penalty = 0.03 * max(0, (0.15 - soc) / 0.15)
        return max(0.78, base - soc_penalty)

    def self_discharge_rate(self) -> float:
        return 0.0001  # Very low — solid electrolyte

    def degradation_per_cycle(self) -> float:
        return 0.8 / 8000  # Long cycle life expected

    def capital_cost_per_kwh(self) -> float:
        return 120  # Sodium is abundant — projected cost advantage

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # Wide operating temperature range, less thermal sensitivity
        heat = abs(power_kw) * 0.08 * dt_hours
        cooling = 0.05 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        return self.temperature_c + (heat - cooling) * 0.1


class LiquidElectrolyteBattery(StorageUnit):
    """
    Flow battery with liquid electrolyte (vanadium redox or zinc-bromine).
    Decoupled power/energy, long duration, deep cycling.
    """

    def __init__(self, capacity_kwh=50.0, max_power_kw=10.0,
                 chemistry="vanadium", units=1, **kwargs):
        super().__init__(
            name=f"Flow Battery ({chemistry.title()})",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )
        self.chemistry = chemistry

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # Flow batteries: ~70-80% round-trip, pump losses included
        base = 0.85
        pump_loss = 0.03  # parasitic pump power
        return max(0.70, base - pump_loss)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        base = 0.84
        pump_loss = 0.03
        soc_penalty = 0.02 * max(0, (0.1 - soc) / 0.1)
        return max(0.68, base - pump_loss - soc_penalty)

    def self_discharge_rate(self) -> float:
        return 0.00005  # Negligible — electrolyte in tanks

    def degradation_per_cycle(self) -> float:
        return 0.8 / 20000  # Extremely long cycle life

    def capital_cost_per_kwh(self) -> float:
        return 250  # Higher upfront, but energy scales cheaply

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # Liquid thermal mass buffers temperature
        heat = abs(power_kw) * 0.15 * dt_hours
        cooling = 0.08 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        return self.temperature_c + (heat - cooling) * 0.05


class FlywheelStorage(StorageUnit):
    """
    Kinetic energy storage via high-speed flywheel.
    Very high power density, fast response, limited energy duration.
    Ideal for frequency regulation and power quality.
    """

    def __init__(self, capacity_kwh=5.0, max_power_kw=100.0, units=1, **kwargs):
        super().__init__(
            name="Flywheel",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )
        self.rpm = 20000 * np.sqrt(self.soc)  # Proportional to sqrt(KE)

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # Motor/generator efficiency ~90-95%, bearing losses
        base = 0.93
        # Higher SOC = higher rpm = slightly more windage loss
        windage = 0.01 * soc
        return max(0.85, base - windage)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        base = 0.93
        windage = 0.01 * soc
        return max(0.85, base - windage)

    def self_discharge_rate(self) -> float:
        # Significant standby losses — magnetic bearings, windage
        return 0.01  # ~24% per day — flywheels bleed energy fast

    def degradation_per_cycle(self) -> float:
        return 0.8 / 500000  # Mechanical — nearly unlimited cycles

    def capital_cost_per_kwh(self) -> float:
        return 5000  # High cost per kWh, but power is the value

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # Vacuum housing, minimal thermal interaction
        heat = abs(power_kw) * 0.07 * dt_hours
        cooling = 0.02 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        return self.temperature_c + (heat - cooling) * 0.02

    def step(self, power_kw: float, dt_hours: float) -> float:
        result = super().step(power_kw, dt_hours)
        self.rpm = 20000 * np.sqrt(max(0, self.soc))
        return result


class HydrogenFuelCell(StorageUnit):
    """
    Hydrogen storage + PEM fuel cell for power delivery.
    Electrolyzer for charging (H2 production), fuel cell for discharging.
    Long-duration, seasonal storage capable.
    """

    def __init__(self, h2_tank_kg=50.0, electrolyzer_kw=25.0,
                 fuel_cell_kw=20.0, units=1, **kwargs):
        # H2 energy density: ~33.3 kWh/kg (LHV)
        capacity_kwh = h2_tank_kg * 33.3
        super().__init__(
            name="H2 Fuel Cell",
            capacity_kwh=capacity_kwh,
            max_charge_kw=electrolyzer_kw,
            max_discharge_kw=fuel_cell_kw,
            units=units,
            **kwargs,
        )
        self.h2_tank_kg = h2_tank_kg * units
        self.electrolyzer_kw = electrolyzer_kw * units
        self.fuel_cell_kw = fuel_cell_kw * units

    @property
    def h2_stored_kg(self) -> float:
        return self.soc * self.h2_tank_kg

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # PEM electrolyzer: 60-75% efficiency (electricity → H2)
        base = 0.70
        # Part-load penalty
        load_frac = power_kw / self.electrolyzer_kw if self.electrolyzer_kw > 0 else 0
        part_load = 0.05 * max(0, 0.3 - load_frac)  # Penalty below 30% load
        return max(0.55, base - part_load)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        # PEM fuel cell: 45-60% electrical efficiency
        base = 0.55
        load_frac = power_kw / self.fuel_cell_kw if self.fuel_cell_kw > 0 else 0
        # Fuel cells are most efficient at partial load
        partial_bonus = 0.05 * max(0, 1 - load_frac)
        return min(0.60, max(0.40, base + partial_bonus))

    def self_discharge_rate(self) -> float:
        return 0.000001  # Compressed H2 — essentially zero loss

    def degradation_per_cycle(self) -> float:
        # Fuel cell membrane degrades; ~40,000 hours lifetime
        return 0.8 / 10000

    def capital_cost_per_kwh(self) -> float:
        # Tank is cheap per kWh; electrolyzer + fuel cell are expensive per kW
        return 35  # $/kWh for tank storage; power equipment costs tracked separately

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # Fuel cell generates significant waste heat
        heat = abs(power_kw) * 0.4 * dt_hours  # ~40% waste heat
        cooling = 0.15 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        return self.temperature_c + (heat - cooling) * 0.03


class GrapheneSupercapacitor(StorageUnit):
    """
    Extremely large graphene-based supercapacitor construct.

    Physics basis:
    - Energy stored electrostatically in electric double-layer at graphene
      electrode surfaces (no faradaic reactions → near-unlimited cycle life)
    - Graphene theoretical surface area: ~2630 m²/g → specific capacitance
      up to ~550 F/g in lab, ~200 F/g at scale
    - Cell voltage: 3.0-4.0V per cell (ionic liquid electrolyte)
    - Energy: E = ½CV²; Power: P = V²/(4·ESR)
    - Extremely low ESR from graphene conductivity → MW-class power

    Construct architecture:
    - Modular stacks of graphene-ionic-liquid cells in series/parallel
    - Active thermal management (graphene's 5000 W/m·K helps)
    - Voltage balancing BMS across series strings
    - Energy density ~30-85 Wh/kg (stack level), well below batteries
      but power density 10-50 kW/kg dominates for grid stabilization

    Key advantages over conventional ultracapacitors:
    - 5-10x energy density vs activated carbon EDLCs
    - No dendrite formation, no phase change → millions of cycles
    - Sub-millisecond response time
    - Wide temperature range (-40°C to +70°C)
    """

    def __init__(self, capacity_kwh=50.0, max_power_kw=10_000.0,
                 cell_voltage=3.8, esr_mohm=0.5, units=1, **kwargs):
        super().__init__(
            name="Graphene Supercap",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )
        self.cell_voltage = cell_voltage
        self.esr_mohm = esr_mohm  # Equivalent series resistance per cell
        # Derive capacitance from energy: E = ½CV² → C = 2E/V²
        energy_joules = capacity_kwh * units * 3.6e6
        self.total_capacitance_f = 2 * energy_joules / (cell_voltage ** 2)
        self.voltage = cell_voltage * np.sqrt(self.soc)  # V ∝ √SOC for supercap

    @property
    def current_voltage(self) -> float:
        """Supercap voltage is proportional to √SOC (E = ½CV²)."""
        return self.cell_voltage * np.sqrt(max(0.01, self.soc))

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # I²R losses in ESR dominate — efficiency depends on current
        # P = IV, I = P/V, loss = I²R = P²R/V²
        v = self.cell_voltage * np.sqrt(max(0.01, soc))
        current_a = (power_kw * 1000) / max(v, 1) if power_kw > 0 else 0
        esr_ohm = self.esr_mohm / 1000
        i2r_loss_w = current_a ** 2 * esr_ohm
        loss_fraction = i2r_loss_w / max(power_kw * 1000, 1) if power_kw > 0 else 0
        return max(0.90, min(0.995, 1.0 - loss_fraction))

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        # Symmetric — same I²R physics
        return self.charge_efficiency(power_kw, soc)

    def self_discharge_rate(self) -> float:
        # Leakage current through dielectric — better than EDLC but nonzero
        # ~5% per day for large constructs with ionic liquid electrolyte
        return 0.002

    def degradation_per_cycle(self) -> float:
        # Electrostatic storage — no chemical degradation pathway
        # Failure mode: ionic liquid decomposition, electrode delamination
        return 0.8 / 1_000_000  # Effectively unlimited cycling

    def capital_cost_per_kwh(self) -> float:
        # Graphene production cost dominates — CVD or reduced graphene oxide
        # High $/kWh but justified by power density and cycle life
        return 8000  # $/kWh — expensive per energy, cheap per power cycle

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # I²R heating, but graphene's extreme thermal conductivity helps
        v = self.current_voltage
        current_a = abs(power_kw * 1000) / max(v, 1) if power_kw != 0 else 0
        esr_ohm = self.esr_mohm / 1000
        heat_w = current_a ** 2 * esr_ohm
        heat_kw = heat_w / 1000
        # Graphene thermal conductivity aids heat spreading
        cooling = 0.3 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        dT = (heat_kw * dt_hours - cooling) * 0.05
        return self.temperature_c + dT

    def step(self, power_kw: float, dt_hours: float) -> float:
        result = super().step(power_kw, dt_hours)
        self.voltage = self.cell_voltage * np.sqrt(max(0.01, self.soc))
        return result


class SMES(StorageUnit):
    """
    Superconducting Magnetic Energy Storage.

    Physics basis:
    - Energy stored in magnetic field of a superconducting coil:
      E = ½LI², where L = inductance, I = persistent current
    - Superconducting wire (YBCO or Bi-2223 HTS tape) carries current
      with zero DC resistance below critical temperature Tc
    - Power conversion via voltage-source converter (VSC) at the DC bus
    - Energy density ~1-10 Wh/kg (coil level), but power density
      is extreme: 1-100 MW instantaneous

    Cryogenic system:
    - HTS tapes operate at 20-77K (liquid nitrogen to cryo-cooler)
    - Parasitic cryo-cooling power: ~1-5% of rated power continuously
    - Quench protection: dump resistors, active quench detection
    - Persistent current switch for zero-loss standby

    Grid applications:
    - Sub-cycle (< 16ms) response for frequency regulation
    - Fault current limiting
    - Power quality and voltage support
    - Spinning reserve with zero emissions
    """

    def __init__(self, capacity_kwh=20.0, max_power_kw=50_000.0,
                 inductance_h=10.0, operating_temp_k=30.0,
                 cryo_power_fraction=0.03, units=1, **kwargs):
        super().__init__(
            name="SMES",
            capacity_kwh=capacity_kwh,
            max_charge_kw=max_power_kw,
            max_discharge_kw=max_power_kw,
            units=units,
            **kwargs,
        )
        self.inductance_h = inductance_h * units
        self.operating_temp_k = operating_temp_k
        self.cryo_power_fraction = cryo_power_fraction
        # Derive max persistent current from E = ½LI²
        energy_j = capacity_kwh * units * 3.6e6
        self.max_current_a = np.sqrt(2 * energy_j / self.inductance_h)
        self.current_a = self.max_current_a * np.sqrt(self.soc)

    def charge_efficiency(self, power_kw: float, soc: float) -> float:
        # VSC (voltage source converter) losses + cryo parasitic
        # Superconducting coil itself: zero resistive loss
        vsc_eff = 0.98  # SiC-based power converter
        cryo_overhead = self.cryo_power_fraction
        return max(0.90, vsc_eff - cryo_overhead)

    def discharge_efficiency(self, power_kw: float, soc: float) -> float:
        vsc_eff = 0.98
        cryo_overhead = self.cryo_power_fraction
        return max(0.90, vsc_eff - cryo_overhead)

    def self_discharge_rate(self) -> float:
        # Persistent current → zero DC loss in superconductor
        # Only loss is cryo-cooler electricity (modeled separately)
        # Flux creep in HTS is ~0.01%/hour at operating conditions
        return 0.0001

    def degradation_per_cycle(self) -> float:
        # No electrochemistry — mechanical fatigue from Lorentz forces
        # HTS tapes have >100,000 cycle fatigue life for strain <0.4%
        return 0.8 / 500_000

    def capital_cost_per_kwh(self) -> float:
        # HTS wire: ~$100-400/kA·m, coil fabrication, cryostat
        # Extremely expensive per kWh, competitive per kW for short duration
        return 50_000  # $/kWh — the most expensive storage per energy unit

    def thermal_model(self, power_kw: float, dt_hours: float) -> float:
        # Cryogenic system maintains operating temperature
        # AC losses during charge/discharge cause heating
        ac_loss = abs(power_kw) * 0.005  # Hysteretic AC loss in HTS
        cryo_cooling = 0.5 * (self.temperature_c - self.ambient_temp_c) * dt_hours
        # Temperature refers to the coil cold mass — should stay near operating_temp_k
        # Map to Celsius for compatibility
        coil_temp_c = self.operating_temp_k - 273.15  # ~-243°C for 30K
        # Any heating above operating temp is immediately concerning
        dt_coil = (ac_loss * dt_hours - cryo_cooling) * 0.001
        return coil_temp_c + dt_coil

    def step(self, power_kw: float, dt_hours: float) -> float:
        result = super().step(power_kw, dt_hours)
        self.current_a = self.max_current_a * np.sqrt(max(0, self.soc))
        return result


# Factory for creating storage from config dicts
STORAGE_REGISTRY = {
    "lithium_ion": LithiumIonBattery,
    "sodium_solid_state": SodiumSolidStateBattery,
    "liquid_electrolyte": LiquidElectrolyteBattery,
    "flywheel": FlywheelStorage,
    "hydrogen_fuel_cell": HydrogenFuelCell,
    "graphene_supercap": GrapheneSupercapacitor,
    "smes": SMES,
}


def create_storage(type_key: str, **kwargs) -> StorageUnit:
    """Factory function to create storage units by type key."""
    if type_key not in STORAGE_REGISTRY:
        raise ValueError(f"Unknown storage type '{type_key}'. Available: {list(STORAGE_REGISTRY.keys())}")
    return STORAGE_REGISTRY[type_key](**kwargs)
