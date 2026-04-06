"""
Renewable energy source models with time-varying output.

Each source models:
- Rated capacity (kW)
- Time-dependent capacity factor (weather, solar angle, wind speed)
- Availability and intermittency
- Capital cost per kW installed
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class SourceOutput:
    """Instantaneous output of an energy source."""
    power_kw: float
    capacity_factor: float
    available: bool


class EnergySource(ABC):
    """Base class for energy sources."""

    def __init__(self, name: str, rated_kw: float, latitude: float = 35.0,
                 units: int = 1):
        self.name = name
        self.rated_kw = rated_kw * units
        self.units = units
        self.latitude = latitude
        self.cumulative_kwh = 0.0

    @property
    def is_renewable(self) -> bool:
        """Whether this source counts as renewable for grid metrics."""
        return True

    @abstractmethod
    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        """Power output at given hour of year [0-8760] and weather factor [0-1]."""

    @abstractmethod
    def capacity_factor_annual(self) -> float:
        """Expected annual average capacity factor."""

    @abstractmethod
    def capital_cost_per_kw(self) -> float:
        """Installed cost in $/kW."""

    def step(self, hour_of_year: float, dt_hours: float,
             weather_factor: float = 1.0) -> SourceOutput:
        power = self.output_kw(hour_of_year, weather_factor)
        self.cumulative_kwh += power * dt_hours
        return SourceOutput(
            power_kw=power,
            capacity_factor=power / self.rated_kw if self.rated_kw > 0 else 0,
            available=power > 0,
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "rated_kw": self.rated_kw,
            "annual_cf": self.capacity_factor_annual(),
            "capital_cost": self.capital_cost_per_kw() * self.rated_kw,
            "cumulative_kwh": self.cumulative_kwh,
        }


class SolarPV(EnergySource):
    """
    Photovoltaic solar array model.
    Models solar irradiance based on hour of day, day of year, latitude,
    panel tilt, and weather/cloud cover.
    """

    def __init__(self, rated_kw=10.0, tilt_deg: Optional[float] = None,
                 tracking: bool = False, latitude: float = 35.0, **kwargs):
        super().__init__(name="Solar PV", rated_kw=rated_kw, latitude=latitude, **kwargs)
        self.tilt_deg = tilt_deg if tilt_deg is not None else abs(latitude)
        self.tracking = tracking
        self.temp_coeff = -0.004  # Power temperature coefficient per °C above 25

    def _solar_geometry(self, hour_of_year: float) -> tuple[float, float]:
        """Returns (solar_elevation_deg, air_mass_factor)."""
        day = hour_of_year / 24.0
        hour_of_day = hour_of_year % 24

        # Declination angle
        declination = 23.45 * np.sin(np.radians(360 / 365 * (day - 81)))

        # Hour angle
        hour_angle = 15.0 * (hour_of_day - 12.0)

        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        # Solar elevation
        sin_elev = (np.sin(lat_rad) * np.sin(dec_rad) +
                    np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
        elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))

        # Air mass (Kasten & Young 1989)
        if elevation > 0:
            air_mass = 1.0 / (np.sin(np.radians(elevation)) +
                              0.50572 * (elevation + 6.07995) ** -1.6364)
        else:
            air_mass = 40.0  # Below horizon

        return elevation, air_mass

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        elevation, air_mass = self._solar_geometry(hour_of_year)

        if elevation <= 0:
            return 0.0

        # Clear-sky irradiance on tilted surface (simplified)
        # Direct normal irradiance through atmosphere
        dni = 1361 * 0.7 ** (air_mass ** 0.678)  # W/m² Meinel model

        # Angle of incidence on tilted surface (simplified)
        tilt_rad = np.radians(self.tilt_deg)
        elev_rad = np.radians(elevation)

        if self.tracking:
            cos_aoi = 1.0  # Perfect tracking
        else:
            cos_aoi = max(0, np.sin(elev_rad) * np.cos(tilt_rad) +
                         np.cos(elev_rad) * np.sin(tilt_rad))

        irradiance = dni * cos_aoi

        # Diffuse component (isotropic sky model)
        diffuse = 0.1 * dni * (1 + np.cos(tilt_rad)) / 2

        total_irradiance = (irradiance + diffuse) * weather_factor

        # Convert to power: rated at 1000 W/m² STC
        power_fraction = total_irradiance / 1000.0

        # Temperature derating (assume module temp = ambient + 25°C in sun)
        temp_derate = 1.0 + self.temp_coeff * 25.0  # Simplified

        output = self.rated_kw * power_fraction * temp_derate
        return max(0.0, min(output, self.rated_kw))

    def capacity_factor_annual(self) -> float:
        # Typical range 15-25% depending on latitude
        lat_factor = 1.0 - abs(self.latitude) / 90 * 0.4
        base = 0.18 * lat_factor
        if self.tracking:
            base *= 1.25
        return base

    def capital_cost_per_kw(self) -> float:
        base = 1000  # $/kW residential scale
        if self.rated_kw > 100:
            base = 750  # Utility scale discount
        if self.tracking:
            base += 200
        return base


class WindTurbine(EnergySource):
    """
    Wind turbine model with realistic power curve.
    Models wind speed variation, cut-in/cut-out, Betz limit.
    """

    def __init__(self, rated_kw=5.0, hub_height_m=30.0,
                 cut_in_ms=3.0, rated_wind_ms=12.0, cut_out_ms=25.0,
                 mean_wind_ms: float = 6.0,
                 latitude: float = 35.0, **kwargs):
        super().__init__(name="Wind Turbine", rated_kw=rated_kw,
                        latitude=latitude, **kwargs)
        self.hub_height_m = hub_height_m
        self.cut_in_ms = cut_in_ms
        self.rated_wind_ms = rated_wind_ms
        self.cut_out_ms = cut_out_ms
        self._mean_wind = mean_wind_ms

    def _wind_speed(self, hour_of_year: float, weather_factor: float) -> float:
        """Synthetic wind speed model with diurnal and seasonal variation."""
        day = hour_of_year / 24.0
        hour = hour_of_year % 24

        # Seasonal variation (windier in winter at mid-latitudes)
        seasonal = 1.0 + 0.2 * np.cos(2 * np.pi * (day - 15) / 365)

        # Diurnal variation (windier in afternoon)
        diurnal = 1.0 + 0.15 * np.sin(2 * np.pi * (hour - 6) / 24)

        # Turbulence (deterministic pseudo-random based on hour)
        turbulence = 1.0 + 0.3 * np.sin(hour_of_year * 2.7 + 1.3) * np.cos(hour_of_year * 0.8)

        # Height correction (wind shear power law, alpha=0.14 open terrain)
        height_factor = (self.hub_height_m / 10.0) ** 0.14

        wind = self._mean_wind * seasonal * diurnal * turbulence * height_factor * weather_factor
        return max(0, wind)

    def _power_curve(self, wind_speed: float) -> float:
        """Turbine power curve: cubic below rated, flat at rated, zero above cut-out."""
        if wind_speed < self.cut_in_ms or wind_speed > self.cut_out_ms:
            return 0.0
        if wind_speed >= self.rated_wind_ms:
            return self.rated_kw
        # Cubic region
        fraction = ((wind_speed - self.cut_in_ms) /
                    (self.rated_wind_ms - self.cut_in_ms)) ** 3
        return self.rated_kw * fraction

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        wind = self._wind_speed(hour_of_year, weather_factor)
        return self._power_curve(wind)

    def capacity_factor_annual(self) -> float:
        return 0.30  # Typical onshore

    def capital_cost_per_kw(self) -> float:
        if self.rated_kw < 20:
            return 3000  # Small/residential
        return 1300  # Utility scale


class MicroHydro(EnergySource):
    """
    Run-of-river micro-hydro generator.
    Seasonal flow variation, high capacity factor.
    """

    def __init__(self, rated_kw=15.0, head_m=10.0, design_flow_m3s=0.2,
                 latitude: float = 35.0, **kwargs):
        super().__init__(name="Micro Hydro", rated_kw=rated_kw,
                        latitude=latitude, **kwargs)
        self.head_m = head_m
        self.design_flow_m3s = design_flow_m3s
        self.turbine_efficiency = 0.85

    def _flow_factor(self, hour_of_year: float, weather_factor: float) -> float:
        """Seasonal streamflow variation."""
        day = hour_of_year / 24.0
        # Peak flow in spring (snowmelt), low in late summer
        seasonal = 0.7 + 0.6 * np.sin(2 * np.pi * (day - 60) / 365)
        return max(0.1, min(1.5, seasonal * weather_factor))

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        flow_frac = self._flow_factor(hour_of_year, weather_factor)
        # P = ρ * g * h * Q * η
        actual_flow = min(self.design_flow_m3s * flow_frac, self.design_flow_m3s)
        power = 9.81 * self.head_m * actual_flow * self.turbine_efficiency
        return min(power, self.rated_kw)

    def capacity_factor_annual(self) -> float:
        return 0.55

    def capital_cost_per_kw(self) -> float:
        return 4000


class Geothermal(EnergySource):
    """
    Geothermal power source — baseload with high capacity factor.
    Small-scale binary cycle or direct-use heat pump.
    """

    def __init__(self, rated_kw=50.0, well_temp_c=150.0,
                 latitude: float = 35.0, **kwargs):
        super().__init__(name="Geothermal", rated_kw=rated_kw,
                        latitude=latitude, **kwargs)
        self.well_temp_c = well_temp_c
        # Binary cycle efficiency depends on well temperature
        self.cycle_efficiency = 0.10 + 0.05 * (well_temp_c - 100) / 100

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        # Nearly constant output — minimal weather dependence
        # Small maintenance downtime modeled as slight derate
        maintenance = 0.02 * np.sin(2 * np.pi * hour_of_year / 8760)
        return self.rated_kw * (0.95 - abs(maintenance))

    def capacity_factor_annual(self) -> float:
        return 0.92

    def capital_cost_per_kw(self) -> float:
        return 6000  # High upfront, but 30+ year asset


class NaturalGasTurbine(EnergySource):
    """
    Natural gas turbine power source — dispatchable fossil fuel generation.

    Two plant types:
    1. Combined Cycle Gas Turbine (CCGT): High-efficiency baseload.
       Heat rate ~6800 BTU/kWh, capacity factor ~55%, emissions ~0.40 kg CO2/kWh.
       Uses waste heat recovery via steam turbine for higher efficiency.

    2. Combustion Turbine / Peaker: Fast-start, lower efficiency.
       Heat rate ~9500 BTU/kWh, capacity factor ~12%, emissions ~0.55 kg CO2/kWh.
       Dispatched during peak demand hours (typically 2-9 PM).

    Fuel cost derived from: heat_rate × gas_price / 1,000,000.
    """

    def __init__(self, rated_kw=100_000.0, plant_type="ccgt",
                 heat_rate_btu_kwh=None, gas_price_per_mmbtu=3.50,
                 latitude: float = 35.0, **kwargs):
        name = f"Gas {'CCGT' if plant_type == 'ccgt' else 'Peaker'}"
        super().__init__(name=name, rated_kw=rated_kw, latitude=latitude, **kwargs)
        self.plant_type = plant_type

        if heat_rate_btu_kwh is None:
            self.heat_rate = 6800 if plant_type == "ccgt" else 9500
        else:
            self.heat_rate = heat_rate_btu_kwh

        self.gas_price_per_mmbtu = gas_price_per_mmbtu
        # Fuel cost: heat_rate (BTU/kWh) × price ($/MMBTU) / 1,000,000
        self.fuel_cost_per_kwh = self.heat_rate * gas_price_per_mmbtu / 1_000_000

        # Emissions
        if plant_type == "ccgt":
            self.emission_rate_kg_co2_kwh = 0.40
            self._base_cf = 0.55
        else:
            self.emission_rate_kg_co2_kwh = 0.55
            self._base_cf = 0.12

        self.cumulative_emissions_kg = 0.0

    @property
    def is_renewable(self) -> bool:
        return False

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        # Maintenance derate (~5% of time)
        maintenance = 0.03 * np.sin(2 * np.pi * hour_of_year / 8760)

        if self.plant_type == "ccgt":
            # Baseload: runs near-constant with slight seasonal variation
            seasonal = 0.02 * np.cos(2 * np.pi * hour_of_year / 8760)
            output = self.rated_kw * (0.88 - abs(maintenance) + seasonal)
        else:
            # Peaker: dispatched during peak hours (14:00-21:00)
            hour_of_day = hour_of_year % 24
            if 14 <= hour_of_day <= 21:
                # Ramp shape: peak at 17:00-18:00
                peak_factor = np.exp(-0.5 * ((hour_of_day - 17.5) / 2.0) ** 2)
                output = self.rated_kw * peak_factor * (0.9 - abs(maintenance))
            elif 10 <= hour_of_day < 14:
                # Light shoulder load
                output = self.rated_kw * 0.15
            else:
                # Off-peak: minimal or standby
                output = self.rated_kw * 0.03
        return max(0.0, min(output, self.rated_kw))

    def step(self, hour_of_year: float, dt_hours: float,
             weather_factor: float = 1.0) -> SourceOutput:
        result = super().step(hour_of_year, dt_hours, weather_factor)
        self.cumulative_emissions_kg += result.power_kw * dt_hours * self.emission_rate_kg_co2_kwh
        return result

    def capacity_factor_annual(self) -> float:
        return self._base_cf

    def capital_cost_per_kw(self) -> float:
        return 1100 if self.plant_type == "ccgt" else 700

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "plant_type": self.plant_type,
            "heat_rate_btu_kwh": self.heat_rate,
            "fuel_cost_per_kwh": self.fuel_cost_per_kwh,
            "emission_rate_kg_co2_kwh": self.emission_rate_kg_co2_kwh,
            "cumulative_emissions_kg": self.cumulative_emissions_kg,
            "cumulative_emissions_tonnes": self.cumulative_emissions_kg / 1000,
        })
        return base


class MicroFusionReactor(EnergySource):
    """
    Compact fusion reactor power source.

    Models two fusion fuel cycles with distinct physics:

    1. Deuterium-Tritium (D-T):
       - Reaction: D + T → He-4 (3.5 MeV) + n (14.1 MeV)
       - Lowest ignition temperature: ~150 million K (13 keV)
       - Highest cross-section of any fusion reaction
       - 80% of energy in neutrons → requires blanket + thermal conversion
       - Tritium breeding via Li-6 blanket: n + Li-6 → T + He-4
       - Engineering Q ~5-15 for compact tokamaks / stellarators
       - Thermal conversion via supercritical CO2 Brayton cycle: η ≈ 40-45%

    2. Aneutronic proton-Boron-11 (p-B11):
       - Reaction: p + B-11 → 3 He-4 (8.7 MeV total)
       - No neutron production → direct energy conversion possible
       - Much higher ignition temperature: ~3 billion K (300 keV)
       - Lower cross-section, harder confinement
       - Direct electrostatic/magnetic conversion: η ≈ 70-85%
       - No radioactive fuel or waste
       - Engineering Q ~2-5 (much harder to achieve)

    Plasma confinement modeled via simplified Lawson criterion:
        nτE > threshold (density × confinement time)
    Confinement approaches: compact tokamak, field-reversed configuration,
    magnetized target fusion, or inertial electrostatic confinement.

    Power balance:
        P_electric = P_fusion × η_conversion - P_sustain
        P_sustain = P_magnets + P_heating + P_cryo + P_aux
        P_fusion = P_plasma × Q_engineering

    Startup sequence:
        - Magnet ramp-up (superconducting → minutes, resistive → seconds)
        - Plasma initiation (gas breakdown, current ramp)
        - Heating to ignition (NBI, ICRH, ECRH)
        - Burn phase (self-sustaining or driven)
    """

    def __init__(self, rated_kw=10_000.0, fuel_cycle="dt",
                 q_engineering=10.0, plasma_temp_kev=15.0,
                 confinement="compact_tokamak",
                 thermal_efficiency=None,
                 latitude: float = 35.0, **kwargs):
        name = f"Micro Fusion ({fuel_cycle.upper()})"
        super().__init__(name=name, rated_kw=rated_kw, latitude=latitude, **kwargs)

        self.fuel_cycle = fuel_cycle.lower()
        self.q_engineering = q_engineering
        self.confinement = confinement

        # Fuel-cycle-dependent parameters
        if self.fuel_cycle == "dt":
            self.plasma_temp_kev = plasma_temp_kev if plasma_temp_kev else 15.0
            self.thermal_efficiency = thermal_efficiency or 0.42  # sCO2 Brayton
            self.neutron_fraction = 0.80  # Fraction of fusion energy in neutrons
            self.tritium_breeding_ratio = 1.08  # Must be >1 for fuel self-sufficiency
            self.ignition_temp_kev = 13.0
            self._sustain_fraction = 0.08  # Fraction of gross power for sustainment
        elif self.fuel_cycle == "pb11":
            self.plasma_temp_kev = plasma_temp_kev if plasma_temp_kev else 300.0
            self.thermal_efficiency = thermal_efficiency or 0.75  # Direct conversion
            self.neutron_fraction = 0.01  # Side reactions only
            self.tritium_breeding_ratio = 0.0  # No tritium needed
            self.ignition_temp_kev = 300.0
            self._sustain_fraction = 0.20  # Higher sustain power — harder plasma
        else:
            raise ValueError(f"Unknown fuel cycle '{fuel_cycle}'. Use 'dt' or 'pb11'.")

        # Derived parameters
        # Gross fusion power needed to deliver rated_kw after conversion & sustain
        # P_net = P_fusion * η_conv - P_sustain = P_fusion * η_conv - P_fusion * f_sustain
        # P_net = P_fusion * (η_conv - f_sustain)
        self.gross_fusion_kw = rated_kw / max(0.01,
            self.thermal_efficiency - self._sustain_fraction)
        self.sustain_power_kw = self.gross_fusion_kw * self._sustain_fraction

        # Plasma state
        self.plasma_on = True
        self.uptime_hours = 0.0
        self.total_burns = 0
        self._hours_since_maintenance = 0.0
        # Maintenance interval: D-T needs more (tritium handling, activation)
        self._maintenance_interval_hours = 6000 if self.fuel_cycle == "dt" else 8000
        self._maintenance_duration_hours = 168  # 1 week
        self._in_maintenance = False
        self._maintenance_remaining = 0.0

    def _plasma_stability(self, hour_of_year: float) -> float:
        """
        Plasma stability factor [0, 1].
        Models disruption risk, ELM events, and confinement degradation.
        Compact reactors have occasional transient confinement losses.
        """
        # Base stability improves with operating experience (burn hours)
        maturity = min(1.0, self.uptime_hours / 5000)
        base_stability = 0.92 + 0.06 * maturity  # 92% → 98%

        # Periodic minor disruptions (ELM-like events)
        elm_cycle = 0.02 * np.sin(hour_of_year * 17.3) * np.cos(hour_of_year * 3.1)

        # Rare major disruption (probability-based, deterministic proxy)
        major_disruption = 0.0
        disruption_phase = np.sin(hour_of_year * 0.0073 + 2.1)
        if disruption_phase > 0.998:
            major_disruption = 0.5  # 50% power loss during major event

        return max(0.0, min(1.0, base_stability - abs(elm_cycle) - major_disruption))

    def _maintenance_check(self, dt_hours: float) -> bool:
        """Check if reactor is available or in scheduled maintenance."""
        if self._in_maintenance:
            self._maintenance_remaining -= dt_hours
            if self._maintenance_remaining <= 0:
                self._in_maintenance = False
                self._hours_since_maintenance = 0.0
                self.total_burns += 1
            return False  # Still in maintenance

        self._hours_since_maintenance += dt_hours
        if self._hours_since_maintenance >= self._maintenance_interval_hours:
            self._in_maintenance = True
            self._maintenance_remaining = self._maintenance_duration_hours
            return False  # Just entered maintenance

        return True  # Available

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        # Fusion is weather-independent, but we check maintenance & stability
        if not self.plasma_on or self._in_maintenance:
            return 0.0

        stability = self._plasma_stability(hour_of_year)

        # Gross fusion power modulated by plasma stability
        gross = self.gross_fusion_kw * stability

        # Thermal/direct conversion
        converted = gross * self.thermal_efficiency

        # Subtract sustainment power (magnets, heating, cryo, aux)
        net = converted - self.sustain_power_kw

        return max(0.0, min(net, self.rated_kw))

    def step(self, hour_of_year: float, dt_hours: float,
             weather_factor: float = 1.0) -> 'SourceOutput':
        """Override to include maintenance and uptime tracking."""
        available = self._maintenance_check(dt_hours)
        self.plasma_on = available

        if available:
            self.uptime_hours += dt_hours

        return super().step(hour_of_year, dt_hours, weather_factor)

    def capacity_factor_annual(self) -> float:
        # Accounts for maintenance downtime and plasma stability
        maintenance_frac = (self._maintenance_duration_hours /
                           (self._maintenance_interval_hours + self._maintenance_duration_hours))
        availability = 1.0 - maintenance_frac
        avg_stability = 0.95  # Expected long-run stability
        return availability * avg_stability

    def capital_cost_per_kw(self) -> float:
        # Fusion capital costs — dominated by magnets, vacuum vessel, blanket
        base = 15_000  # $/kW for first-of-a-kind compact fusion
        # Learning curve discount for larger installations
        if self.rated_kw > 50_000:
            base *= 0.6  # Economy of scale
        elif self.rated_kw > 10_000:
            base *= 0.8
        # p-B11 is harder — higher field magnets, more exotic materials
        if self.fuel_cycle == "pb11":
            base *= 1.8
        return base

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "fuel_cycle": self.fuel_cycle.upper(),
            "q_engineering": self.q_engineering,
            "thermal_efficiency": self.thermal_efficiency,
            "confinement": self.confinement,
            "uptime_hours": self.uptime_hours,
            "burns": self.total_burns,
            "in_maintenance": self._in_maintenance,
            "gross_fusion_kw": self.gross_fusion_kw,
            "sustain_power_kw": self.sustain_power_kw,
        })
        return base


class AntimatterReactor(EnergySource):
    """
    Antimatter power reactor with graphene Penning trap containment
    and optional target-atom electron production.

    Physics basis — proton-antiproton annihilation:
        p + p̄ → pions (π⁺, π⁻, π⁰), then:
        - π⁰ → 2γ (135 MeV each) — ~30% of annihilation energy
        - π± → μ± → e± — charged particles — ~70% of energy

    Total energy per pair: E = 2mc² = 1876.6 MeV
    Energy density: 1 μg antiprotons ≈ 180 MJ = 50 kWh
    (vs 0.6 MJ/g gasoline, 180 MJ/g fission, ~3.4 MJ/g D-T fusion)

    Graphene Penning trap containment:
    - Graphene electrodes for electrostatic axial confinement:
      * 2D carbon lattice: radiation-transparent for most particles
      * 5000 W/m·K thermal conductivity handles localized heating
      * Self-healing vacancy migration under irradiation
      * Configurable multi-layer stacks for radiation damage tolerance
    - HTS superconducting solenoid (6-8 T) for radial magnetic confinement
    - Ultra-high vacuum < 10⁻¹³ torr (better than interstellar space)
    - Antiproton cloud density limited by space-charge (Brillouin limit)

    Three power conversion pathways:

    1. Charged pion MHD: π± steered by magnetic nozzle into
       magnetohydrodynamic channel. High-velocity charged particle flow
       induces current in MHD electrodes. η ≈ 60-80%.

    2. Gamma-ray thermal: π⁰→2γ photons absorbed in high-Z shielding
       (tungsten/lead composite). Thermal energy extracted via
       supercritical CO₂ Brayton cycle. η ≈ 30-40%.

    3. Target-atom electron production (when target_atom is set):
       Antiprotons directed at target atoms form exotic antiprotonic
       atoms. The antiproton replaces an orbital electron and cascades
       to lower energy levels, producing excess electrons via:

       a) Auger electron emission: At each cascade transition, the
          antiproton ejects orbital electrons from the exotic atom.
          For high-Z targets, 20-45 Auger electrons per capture event
          at keV-scale energies. Graphene collection electrodes sweep
          electrons via applied electric field.

       b) Ionization electron harvest: Charged annihilation products
          (pions, nuclear fragments) stopping in the target medium
          create dense ionization trails. Each MeV of charged particle
          energy produces ~30 electron-ion pairs in gas targets.
          Collected as direct current on graphene Faraday cup electrodes.

       c) Antimatter-catalyzed fission (uranium-238 target):
          When the antiproton reaches the nucleus and annihilates with
          a nucleon, ~1.88 GeV is deposited into the nuclear potential.
          For actinide nuclei this exceeds the fission barrier, causing
          fission with ~200 MeV additional energy release per event.
          Fission fragments produce massive secondary ionization → more
          collectible electrons. Energy amplification factor: ~1.107.

       Target atoms and their properties:
       - Xenon-131 (Z=54): Gaseous target, clean electron extraction,
         ~10% of annihilation energy convertible to electron current.
         No fission. Moderate Auger yield.
       - Lead-208 (Z=82): Dense target, higher Auger yield (~15% of
         energy to electrons), more secondary ionization. No fission.
       - Uranium-238 (Z=92): Antimatter-catalyzed fission amplifies
         total energy by ~10.7%. Fission fragments produce copious
         ionization. ~22% of total energy to electron current. But
         produces radioactive fission products (tradeoff vs clean Xe).

       The electron current is collected on graphene electrodes and
       conditioned via DC-DC power electronics — no thermal cycle
       needed, yielding η_electron ≈ 35-55%.

    Containment failure modes (unlike fusion — loss of containment
    releases stored antimatter energy):
    - Vacuum degradation → residual gas annihilation
    - Magnetic field quench → radial deconfinement
    - Graphene electrode spallation from cumulative radiation damage
    - Space-charge instability at high antiproton density
    Emergency shutdown: controlled magnetic dump to annihilation target.

    Power balance (with target atom):
        P_gross = P_annihilation × fission_amplification
        P_mhd   = P_gross × f_pion_eff × η_MHD
        P_therm = P_gross × f_gamma × η_thermal
        P_elec  = P_gross × f_electron × η_electron_collection
        P_net   = P_mhd + P_therm + P_elec - P_containment
    """

    # Target atom properties for antiproton-atom interaction modeling
    TARGET_ATOMS = {
        "none": {
            "Z": 0,
            "name": "None (free annihilation)",
            "auger_yield_fraction": 0.0,
            "ionization_fraction": 0.0,
            "fission_amplification": 1.0,
            "pion_capture_fraction": 0.0,
        },
        "xenon": {
            "Z": 54,
            "name": "Xenon-131",
            "auger_yield_fraction": 0.02,
            "ionization_fraction": 0.08,
            "fission_amplification": 1.0,
            "pion_capture_fraction": 0.10,
        },
        "lead": {
            "Z": 82,
            "name": "Lead-208",
            "auger_yield_fraction": 0.03,
            "ionization_fraction": 0.12,
            "fission_amplification": 1.0,
            "pion_capture_fraction": 0.15,
        },
        "uranium": {
            "Z": 92,
            "name": "Uranium-238",
            "auger_yield_fraction": 0.04,
            "ionization_fraction": 0.18,
            "fission_amplification": 1.107,
            "pion_capture_fraction": 0.20,
        },
    }

    def __init__(
        self,
        rated_kw=5_000.0,
        containment="graphene_penning",
        annihilation_mode="proton_antiproton",
        target_atom="none",
        electron_collection_efficiency=0.45,
        magnetic_field_tesla=6.0,
        graphene_electrode_layers=100,
        trap_vacuum_torr=1e-13,
        fuel_reservoir_ug=5_500_000.0,
        mhd_efficiency=0.70,
        gamma_thermal_efficiency=0.35,
        charged_pion_fraction=0.70,
        containment_magnet_kw=None,
        latitude: float = 35.0,
        **kwargs,
    ):
        # Resolve target atom properties
        if target_atom not in self.TARGET_ATOMS:
            raise ValueError(
                f"Unknown target atom '{target_atom}'. "
                f"Available: {list(self.TARGET_ATOMS.keys())}"
            )
        target = self.TARGET_ATOMS[target_atom]
        self.target_atom = target_atom
        self.target_atom_Z = target["Z"]
        self.target_atom_name = target["name"]

        # Target-atom interaction fractions
        self.auger_yield_fraction = target["auger_yield_fraction"]
        self.ionization_fraction = target["ionization_fraction"]
        self.fission_amplification = target["fission_amplification"]
        self.pion_capture_fraction = target["pion_capture_fraction"]
        self.electron_collection_efficiency = electron_collection_efficiency

        # Total electron channel: Auger + ionization electrons
        self.electron_fraction = self.auger_yield_fraction + self.ionization_fraction

        has_target = target_atom != "none"
        name = (f"Antimatter [{target_atom.title()}]" if has_target
                else "Antimatter Reactor")
        super().__init__(name=name, rated_kw=rated_kw, latitude=latitude, **kwargs)

        self.containment = containment
        self.annihilation_mode = annihilation_mode
        self.magnetic_field_tesla = magnetic_field_tesla
        self.graphene_electrode_layers = graphene_electrode_layers
        self.trap_vacuum_torr = trap_vacuum_torr
        self.mhd_efficiency = mhd_efficiency
        self.gamma_thermal_efficiency = gamma_thermal_efficiency
        self.charged_pion_fraction = charged_pion_fraction

        # Three-pathway combined conversion efficiency:
        #
        # With a target atom, pion energy is partially captured by the
        # target medium (producing ionization) rather than reaching the
        # MHD channel. The effective pion fraction for MHD is reduced.
        # Additionally, fission amplification increases total gross power.
        #
        # Pathway 1 (MHD): remaining charged pions after target capture
        # Pathway 2 (Thermal): gamma rays (unchanged by target)
        # Pathway 3 (Electron): Auger + ionization electrons from target
        self.effective_pion_fraction = (
            self.charged_pion_fraction * (1 - self.pion_capture_fraction)
        )
        self.gamma_fraction = 1 - self.charged_pion_fraction  # Unchanged

        self.combined_efficiency = (
            self.effective_pion_fraction * self.mhd_efficiency +
            self.gamma_fraction * self.gamma_thermal_efficiency +
            self.electron_fraction * self.electron_collection_efficiency
        )

        # Containment parasitic power (magnets + vacuum pumps + cryo-coolers)
        # Target atom systems need slightly more parasitic for target delivery
        parasitic_fraction = 0.05 if not has_target else 0.06
        self.containment_magnet_kw = containment_magnet_kw or (rated_kw * parasitic_fraction)

        # Gross annihilation power (before fission amplification) to deliver rated_kw
        # P_net = P_annihilation * fission_amp * η_combined - P_containment
        # P_annihilation = (P_net + P_containment) / (fission_amp * η_combined)
        effective_efficiency = self.fission_amplification * self.combined_efficiency
        self.gross_annihilation_kw = (
            (rated_kw + self.containment_magnet_kw) /
            max(0.01, effective_efficiency)
        )

        # Fuel consumption: E = 2mc² per proton-antiproton pair
        # 1 μg antiprotons → 180 MJ = 180,000 kJ = 50 kWh
        # Fuel rate based on pre-amplification annihilation power
        self.energy_per_ug_kj = 180_000.0
        self.energy_per_ug_kwh = 50.0
        self.fuel_rate_ug_per_hour = (
            self.gross_annihilation_kw * 3600.0 / self.energy_per_ug_kj
        )

        # Fuel state — scale by units
        total_units = self.rated_kw / rated_kw if rated_kw > 0 else 1
        self._initial_fuel_ug = fuel_reservoir_ug * total_units
        self.fuel_remaining_ug = self._initial_fuel_ug
        self.fuel_consumed_ug = 0.0

        # Operating state
        self.containment_on = True
        self.uptime_hours = 0.0
        self.total_cycles = 0
        self._containment_failures = 0
        self._hours_since_maintenance = 0.0

        # Maintenance: graphene electrode replacement
        # Each layer provides ~40 hours of radiation tolerance
        # Target atoms cause more radiation damage → shorter interval
        damage_multiplier = 1.0 + 0.3 * (self.target_atom_Z / 92.0) if has_target else 1.0
        self._maintenance_interval_hours = graphene_electrode_layers * 40.0 / damage_multiplier
        self._maintenance_duration_hours = 336.0  # 2 weeks
        self._in_maintenance = False
        self._maintenance_remaining = 0.0

        # Electrode degradation tracking
        self._electrode_health = 1.0
        self._electrode_degradation_rate = 1.0 / max(self._maintenance_interval_hours, 1)

    def _containment_stability(self, hour_of_year: float) -> float:
        """
        Graphene Penning trap containment stability [0, 1].

        Models vacuum quality, magnetic field stability, electrode
        degradation, and space-charge effects on antiproton cloud.
        """
        # Base stability improves with operating maturity
        maturity = min(1.0, self.uptime_hours / 3000)
        base = 0.94 + 0.04 * maturity  # 94% → 98%

        # Electrode health — sqrt gives graceful degradation curve
        electrode_factor = np.sqrt(max(0.01, self._electrode_health))

        # Vacuum quality oscillation (outgassing, pump cycling)
        vacuum_noise = 0.01 * np.sin(hour_of_year * 11.7) * np.cos(hour_of_year * 2.3)

        # Magnetic field micro-instabilities
        field_noise = 0.005 * np.sin(hour_of_year * 23.1 + 0.7)

        # Space-charge penalty: more fuel in trap = harder to contain
        fill_fraction = self.fuel_remaining_ug / max(self._initial_fuel_ug, 1e-6)
        space_charge_penalty = 0.02 * fill_fraction ** 2

        # Rare severe containment perturbation
        perturbation = 0.0
        pert_phase = np.sin(hour_of_year * 0.0051 + 1.7)
        if pert_phase > 0.997:
            perturbation = 0.4
            self._containment_failures += 1

        stability = (base * electrode_factor
                     - abs(vacuum_noise)
                     - abs(field_noise)
                     - space_charge_penalty
                     - perturbation)
        return max(0.0, min(1.0, stability))

    def _containment_failure_check(self, hour_of_year: float) -> bool:
        """
        Check for catastrophic containment failure requiring emergency shutdown.

        Unlike fusion disruptions, antimatter containment loss risks
        uncontrolled annihilation. Failure triggers emergency magnetic dump
        to annihilation target, losing a small fraction of fuel.

        Returns True if containment nominal, False if failure occurred.
        """
        # Deterministic proxy for failure probability
        # Risk increases exponentially as electrode health degrades
        electrode_risk = 1e-6 * np.exp(3.0 * (1.0 - self._electrode_health))
        failure_signal = (np.sin(hour_of_year * 0.00137 + 3.14) *
                         np.cos(hour_of_year * 0.00291))
        threshold = 1.0 - electrode_risk * 1e4

        if failure_signal > max(threshold, 0.9998):
            # Emergency shutdown: lose 0.1% of remaining fuel
            fuel_lost = self.fuel_remaining_ug * 0.001
            self.fuel_remaining_ug -= fuel_lost
            self.fuel_consumed_ug += fuel_lost
            self._in_maintenance = True
            self._maintenance_remaining = self._maintenance_duration_hours * 1.5
            self._containment_failures += 1
            return False

        return True

    def _maintenance_check(self, dt_hours: float) -> bool:
        """Scheduled maintenance for graphene electrode replacement."""
        if self._in_maintenance:
            self._maintenance_remaining -= dt_hours
            if self._maintenance_remaining <= 0:
                self._in_maintenance = False
                self._hours_since_maintenance = 0.0
                self._electrode_health = 1.0  # Fresh graphene electrodes
                self.total_cycles += 1
            return False

        self._hours_since_maintenance += dt_hours

        # Degrade electrodes from radiation damage
        self._electrode_health = max(
            0.0,
            self._electrode_health - self._electrode_degradation_rate * dt_hours
        )

        if self._hours_since_maintenance >= self._maintenance_interval_hours:
            self._in_maintenance = True
            self._maintenance_remaining = self._maintenance_duration_hours
            return False

        return True

    def output_kw(self, hour_of_year: float, weather_factor: float = 1.0) -> float:
        # Antimatter is weather-independent
        if not self.containment_on or self._in_maintenance:
            return 0.0
        if self.fuel_remaining_ug <= 0:
            return 0.0

        stability = self._containment_stability(hour_of_year)

        # Gross annihilation power with fission amplification
        gross = self.gross_annihilation_kw * stability * self.fission_amplification

        # Three-pathway conversion:
        # 1. MHD: charged pions not captured by target medium
        mhd_power = gross * self.effective_pion_fraction * self.mhd_efficiency

        # 2. Thermal: gamma rays from π⁰ decay
        gamma_thermal_power = gross * self.gamma_fraction * self.gamma_thermal_efficiency

        # 3. Electron: Auger + ionization electrons collected on graphene electrodes
        electron_power = gross * self.electron_fraction * self.electron_collection_efficiency

        total_converted = mhd_power + gamma_thermal_power + electron_power

        # Subtract containment parasitic power
        net = total_converted - self.containment_magnet_kw

        return max(0.0, min(net, self.rated_kw))

    def step(self, hour_of_year: float, dt_hours: float,
             weather_factor: float = 1.0) -> SourceOutput:
        """Override to track maintenance, containment failures, and fuel consumption."""
        available = self._maintenance_check(dt_hours)

        if available:
            available = self._containment_failure_check(hour_of_year)

        self.containment_on = available

        if available:
            self.uptime_hours += dt_hours
            # Consume fuel proportional to actual stability
            stability = self._containment_stability(hour_of_year)
            fuel_used = self.fuel_rate_ug_per_hour * dt_hours * stability
            fuel_used = min(fuel_used, self.fuel_remaining_ug)
            self.fuel_remaining_ug -= fuel_used
            self.fuel_consumed_ug += fuel_used

        return super().step(hour_of_year, dt_hours, weather_factor)

    def capacity_factor_annual(self) -> float:
        maintenance_frac = (
            self._maintenance_duration_hours /
            (self._maintenance_interval_hours + self._maintenance_duration_hours)
        )
        availability = 1.0 - maintenance_frac
        avg_stability = 0.96
        # Fuel depletion: if fuel won't last a year, CF drops proportionally
        fuel_hours = self.fuel_remaining_ug / max(self.fuel_rate_ug_per_hour, 1e-15)
        fuel_factor = min(1.0, fuel_hours / 8760.0)
        return availability * avg_stability * fuel_factor

    def capital_cost_per_kw(self) -> float:
        # Most expensive technology: Penning trap + graphene electrodes +
        # HTS solenoid + MHD converter + gamma shielding + UHV system
        base = 100_000  # $/kW first-of-a-kind
        if self.rated_kw > 20_000:
            base *= 0.5
        elif self.rated_kw > 5_000:
            base *= 0.7
        return base

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "containment": self.containment,
            "annihilation_mode": self.annihilation_mode,
            "target_atom": self.target_atom,
            "target_atom_name": self.target_atom_name,
            "fission_amplification": self.fission_amplification,
            "combined_efficiency": self.combined_efficiency,
            "mhd_efficiency": self.mhd_efficiency,
            "gamma_thermal_efficiency": self.gamma_thermal_efficiency,
            "electron_collection_efficiency": self.electron_collection_efficiency,
            "effective_pion_fraction": self.effective_pion_fraction,
            "gamma_fraction": self.gamma_fraction,
            "electron_fraction": self.electron_fraction,
            "fuel_remaining_ug": self.fuel_remaining_ug,
            "fuel_consumed_ug": self.fuel_consumed_ug,
            "fuel_rate_ug_per_hour": self.fuel_rate_ug_per_hour,
            "uptime_hours": self.uptime_hours,
            "cycles": self.total_cycles,
            "in_maintenance": self._in_maintenance,
            "electrode_health": self._electrode_health,
            "containment_failures": self._containment_failures,
            "gross_annihilation_kw": self.gross_annihilation_kw,
            "containment_magnet_kw": self.containment_magnet_kw,
        })
        return base


SOURCE_REGISTRY = {
    "solar_pv": SolarPV,
    "wind": WindTurbine,
    "micro_hydro": MicroHydro,
    "geothermal": Geothermal,
    "natural_gas": NaturalGasTurbine,
    "micro_fusion": MicroFusionReactor,
    "antimatter": AntimatterReactor,
}


def create_source(type_key: str, **kwargs) -> EnergySource:
    """Factory function to create energy sources by type key."""
    if type_key not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown source type '{type_key}'. Available: {list(SOURCE_REGISTRY.keys())}")
    return SOURCE_REGISTRY[type_key](**kwargs)
