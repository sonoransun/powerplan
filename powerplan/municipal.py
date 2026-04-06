"""
Real-world municipal power infrastructure models with growth projections.

Models actual city power systems with fossil/renewable mixes, climate-specific
demand profiles, population growth, EV adoption, data center expansion, and
mandated renewable energy transition schedules.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .profiles import DeploymentScale, LoadProfile, SCALES
from .sources import (
    SolarPV, WindTurbine, MicroHydro, Geothermal, NaturalGasTurbine,
)
from .storage import LithiumIonBattery, LiquidElectrolyteBattery
from .controllers import (
    MPPTController, SiCConverter, BidirectionalInverter,
)
from .grid import GridConfig, EnergyDispatcher


# ──────────────────────────────────────────────────────────────────────
# Climate Zones
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ClimateZone:
    """Climate profile driving temperature, solar irradiance, and wind patterns."""
    name: str
    latitude: float
    summer_peak_temp_c: float
    winter_low_temp_c: float
    annual_mean_temp_c: float
    summer_peak_day: int          # Day of year for peak temp (200 = mid-July)
    diurnal_range_c: float        # Daily temperature swing amplitude
    solar_factor: float           # 0-1 relative to ideal clear-sky (1.0 = Phoenix)
    mean_wind_ms: float           # Mean wind speed at 10m reference height
    heating_degree_days: float    # HDD base 18C
    cooling_degree_days: float    # CDD base 18C

    def annual_temperature(self, hours: np.ndarray) -> np.ndarray:
        """Generate climate-specific annual temperature profile."""
        days = hours / 24.0
        # Seasonal: asymmetric sine centered on summer_peak_day
        seasonal_amp = (self.summer_peak_temp_c - self.winter_low_temp_c) / 2
        seasonal = self.annual_mean_temp_c + seasonal_amp * np.sin(
            2 * np.pi * (days - self.summer_peak_day + 91) / 365
        )
        # Diurnal: daily swing, min at 6 AM, max at 3 PM
        diurnal = self.diurnal_range_c * np.sin(
            2 * np.pi * (hours % 24 - 6) / 24
        )
        return seasonal + diurnal


CLIMATE_ZONES = {
    "hot_arid": ClimateZone(
        name="Hot Arid (Phoenix/Las Vegas)",
        latitude=33.4, summer_peak_temp_c=46, winter_low_temp_c=2,
        annual_mean_temp_c=24, summer_peak_day=200, diurnal_range_c=8,
        solar_factor=1.00, mean_wind_ms=4.5,
        heating_degree_days=600, cooling_degree_days=3800,
    ),
    "hot_humid": ClimateZone(
        name="Hot Humid (Houston/Miami)",
        latitude=30.0, summer_peak_temp_c=38, winter_low_temp_c=5,
        annual_mean_temp_c=22, summer_peak_day=205, diurnal_range_c=5,
        solar_factor=0.85, mean_wind_ms=5.0,
        heating_degree_days=800, cooling_degree_days=3200,
    ),
    "temperate": ClimateZone(
        name="Temperate (Denver/Austin)",
        latitude=39.7, summer_peak_temp_c=36, winter_low_temp_c=-8,
        annual_mean_temp_c=14, summer_peak_day=200, diurnal_range_c=7,
        solar_factor=0.90, mean_wind_ms=5.5,
        heating_degree_days=3200, cooling_degree_days=1200,
    ),
    "cold_continental": ClimateZone(
        name="Cold Continental (Minneapolis/Chicago)",
        latitude=45.0, summer_peak_temp_c=34, winter_low_temp_c=-28,
        annual_mean_temp_c=7, summer_peak_day=200, diurnal_range_c=6,
        solar_factor=0.70, mean_wind_ms=6.5,
        heating_degree_days=4500, cooling_degree_days=700,
    ),
    "marine_mild": ClimateZone(
        name="Marine Mild (Seattle/Portland)",
        latitude=47.5, summer_peak_temp_c=30, winter_low_temp_c=0,
        annual_mean_temp_c=11, summer_peak_day=205, diurnal_range_c=5,
        solar_factor=0.55, mean_wind_ms=5.0,
        heating_degree_days=2800, cooling_degree_days=200,
    ),
    "industrial_north": ClimateZone(
        name="Industrial North (Cleveland/Detroit)",
        latitude=41.5, summer_peak_temp_c=33, winter_low_temp_c=-18,
        annual_mean_temp_c=10, summer_peak_day=200, diurnal_range_c=6,
        solar_factor=0.65, mean_wind_ms=6.0,
        heating_degree_days=3800, cooling_degree_days=600,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Municipal Profile
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MunicipalProfile:
    """Real-world municipal power infrastructure specification."""
    name: str
    population: int
    scale: DeploymentScale
    climate: ClimateZone

    # Sector mix
    residential_fraction: float
    commercial_fraction: float
    industrial_fraction: float

    # Growth
    annual_load_growth_pct: float
    annual_pop_growth_pct: float

    # Current generation mix (kW)
    fossil_capacity_kw: float
    fossil_type: str             # "ccgt" or "peaker" (primary fossil plant type)
    solar_capacity_kw: float
    wind_capacity_kw: float
    hydro_capacity_kw: float
    geothermal_capacity_kw: float
    storage_kwh: float

    # Transition targets
    renewable_target_pct: float  # e.g. 0.50 for 50%
    renewable_target_year: int   # e.g. 2030
    net_zero_year: int           # e.g. 2045 (year to reach ~100% renewable)

    # Grid
    grid_interconnect_kw: float

    # Emerging demand drivers
    ev_adoption_rate_pct: float         # Annual growth in EV charging load
    data_center_load_kw: float          # Current data center baseload
    data_center_growth_pct: float       # Annual data center growth
    heat_pump_adoption_rate_pct: float  # Annual heat pump adoption growth


# ──────────────────────────────────────────────────────────────────────
# Municipal Load Profile
# ──────────────────────────────────────────────────────────────────────

class MunicipalLoadProfile(LoadProfile):
    """Load profile with climate-specific temperature, EV, data center, and heat pump loads."""

    def __init__(self, profile: MunicipalProfile, year_offset: int = 0,
                 seed: int = 42):
        super().__init__(profile.scale, seed)
        self.profile = profile
        self.year_offset = year_offset

        # Override sector fractions from municipal profile
        self._residential_frac = profile.residential_fraction
        self._commercial_frac = profile.commercial_fraction
        self._industrial_frac = profile.industrial_fraction

        # Scale peak for load growth
        growth = (1 + profile.annual_load_growth_pct / 100) ** year_offset
        self.peak_kw = profile.scale.peak_load_kw * growth

        # EV charging: grows as fraction of residential peak
        self._ev_peak_kw = (
            profile.scale.peak_load_kw * 0.02 *  # ~2% of peak per year of adoption
            profile.ev_adoption_rate_pct / 100 *
            year_offset
        )

        # Data center baseload
        self._dc_load_kw = (
            profile.data_center_load_kw *
            (1 + profile.data_center_growth_pct / 100) ** year_offset
        )

        # Heat pump winter boost factor
        self._hp_factor = (
            0.005 * profile.heat_pump_adoption_rate_pct / 100 * year_offset
        )

    def _annual_temperature(self, hours: np.ndarray) -> np.ndarray:
        """Use climate zone temperature profile."""
        return self.profile.climate.annual_temperature(hours)

    def demand_kw(self, hour_of_year: float, temperature_c: float = 20.0) -> float:
        base = super().demand_kw(hour_of_year, temperature_c)

        # EV charging: evening peak 6-11 PM, centered at 9 PM
        hour = hour_of_year % 24
        ev_load = self._ev_peak_kw * np.exp(-0.5 * ((hour - 21) / 2.0) ** 2)

        # Data center: constant baseload 24/7
        dc_load = self._dc_load_kw

        # Heat pump: increases winter demand (below 10C), slight summer reduction
        hp_load = 0.0
        if temperature_c < 10 and self._hp_factor > 0:
            hp_load = self.peak_kw * self._hp_factor * (10 - temperature_c) / 20

        return max(0.1, base + ev_load + dc_load + hp_load)


# ──────────────────────────────────────────────────────────────────────
# Config Builder — Year-Projected Municipal Configuration
# ──────────────────────────────────────────────────────────────────────

# Technology cost learning curves (annual decline rates)
COST_CURVES = {
    "solar": 0.05,    # 5%/year decline
    "wind": 0.03,     # 3%/year decline
    "battery": 0.08,  # 8%/year decline
    "gas": 0.00,      # Stable
}


def build_municipal_config(
    profile: MunicipalProfile,
    year_offset: int = 0,
    base_year: int = 2025,
) -> GridConfig:
    """
    Build a GridConfig from a MunicipalProfile projected forward by year_offset.

    Growth model:
    - Fossil capacity retires linearly toward net_zero_year
    - Renewable capacity added to meet interpolated RPS targets
    - Storage scales with renewable penetration (4 MWh per MW above 40%)
    - Technology costs decline per COST_CURVES
    - Load grows with compounded annual_load_growth_pct
    """
    year = base_year + year_offset
    years_to_net_zero = max(1, profile.net_zero_year - base_year)

    # --- Demand projection ---
    load_growth = (1 + profile.annual_load_growth_pct / 100) ** year_offset
    projected_peak_kw = profile.scale.peak_load_kw * load_growth

    # --- Fossil retirement ---
    retirement_progress = min(1.0, max(0, year_offset / years_to_net_zero))
    fossil_remaining_kw = profile.fossil_capacity_kw * (1 - retirement_progress)

    # Split fossil into CCGT and peaker proportionally
    if profile.fossil_type == "ccgt":
        ccgt_kw = fossil_remaining_kw * 0.75
        peaker_kw = fossil_remaining_kw * 0.25
    else:
        ccgt_kw = fossil_remaining_kw * 0.30
        peaker_kw = fossil_remaining_kw * 0.70

    # --- Renewable target interpolation ---
    years_to_target = max(1, profile.renewable_target_year - base_year)
    if year_offset <= years_to_target:
        # Interpolate toward renewable_target_pct
        current_renewable_pct = profile.renewable_target_pct * (year_offset / years_to_target)
    else:
        # Past first target, interpolate toward 100% at net_zero_year
        remaining = max(1, years_to_net_zero - years_to_target)
        progress = min(1.0, (year_offset - years_to_target) / remaining)
        current_renewable_pct = (profile.renewable_target_pct +
                                 (1.0 - profile.renewable_target_pct) * progress)

    # --- Size renewable capacity to meet target ---
    # Total generation needed: projected_peak × avg CF × margin
    # Renewable must provide current_renewable_pct of generation
    total_gen_needed_kw = projected_peak_kw * 1.15  # 15% margin
    renewable_target_kw = total_gen_needed_kw * current_renewable_pct

    # Start from existing renewable, add more as needed
    existing_renewable_kw = (profile.solar_capacity_kw +
                             profile.wind_capacity_kw +
                             profile.hydro_capacity_kw +
                             profile.geothermal_capacity_kw)
    additional_renewable_kw = max(0, renewable_target_kw - existing_renewable_kw)

    # Split additional into solar (60%) and wind (40%) by default
    solar_kw = profile.solar_capacity_kw + additional_renewable_kw * 0.60
    wind_kw = profile.wind_capacity_kw + additional_renewable_kw * 0.40
    hydro_kw = profile.hydro_capacity_kw
    geothermal_kw = profile.geothermal_capacity_kw

    # --- Storage scaling ---
    renewable_fraction = (solar_kw + wind_kw + hydro_kw + geothermal_kw) / max(total_gen_needed_kw, 1)
    if renewable_fraction > 0.40:
        # 4 MWh per MW of renewable above 40% threshold
        excess_renewable_mw = (renewable_fraction - 0.40) * total_gen_needed_kw / 1000
        storage_kwh = max(profile.storage_kwh,
                          profile.storage_kwh + excess_renewable_mw * 4000)
    else:
        # Modest growth from base
        storage_kwh = profile.storage_kwh * (1.15 ** year_offset)

    # --- Build sources ---
    sources = []
    climate = profile.climate

    if ccgt_kw > 1:
        sources.append(NaturalGasTurbine(rated_kw=ccgt_kw, plant_type="ccgt"))
    if peaker_kw > 1:
        sources.append(NaturalGasTurbine(rated_kw=peaker_kw, plant_type="peaker"))
    if solar_kw > 1:
        sources.append(SolarPV(
            rated_kw=solar_kw,
            latitude=climate.latitude,
            tracking=solar_kw > 5000,
        ))
    if wind_kw > 1:
        sources.append(WindTurbine(
            rated_kw=wind_kw,
            hub_height_m=min(100, 30 + 70 * min(1, wind_kw / 50000)),
            mean_wind_ms=climate.mean_wind_ms,
            latitude=climate.latitude,
        ))
    if hydro_kw > 1:
        sources.append(MicroHydro(rated_kw=hydro_kw, latitude=climate.latitude))
    if geothermal_kw > 1:
        sources.append(Geothermal(rated_kw=geothermal_kw, latitude=climate.latitude))

    # --- Build storage ---
    storage_units = []
    if storage_kwh > 0:
        # Split: 80% Li-ion LFP, 20% flow battery (if large enough)
        li_kwh = storage_kwh * 0.80
        flow_kwh = storage_kwh * 0.20

        if li_kwh > 1:
            storage_units.append(LithiumIonBattery(
                capacity_kwh=li_kwh,
                max_power_kw=li_kwh * 0.25,  # 4-hour duration
                chemistry="lfp",
            ))
        if flow_kwh > 100:
            storage_units.append(LiquidElectrolyteBattery(
                capacity_kwh=flow_kwh,
                max_power_kw=flow_kwh * 0.125,  # 8-hour duration
            ))

    # --- Build controllers ---
    total_src_kw = sum(s.rated_kw for s in sources)
    total_stor_kw = sum(u.max_discharge_kw for u in storage_units) if storage_units else 0
    controllers = [
        MPPTController(rated_kw=max(solar_kw + wind_kw, 1)),
        SiCConverter(rated_kw=max(total_src_kw * 0.5, 1)),
        BidirectionalInverter(rated_kw=max(total_stor_kw * 1.1, 1), phases=3),
    ]

    # --- Grid interconnect (grows slightly with demand) ---
    grid_kw = profile.grid_interconnect_kw * load_growth

    # --- Load profile ---
    load_profile = MunicipalLoadProfile(profile, year_offset=year_offset)

    # --- Assemble ---
    year_label = f" ({base_year + year_offset})" if year_offset > 0 else ""
    config = GridConfig(
        name=f"{profile.name}{year_label}",
        scale=profile.scale,
        sources=sources,
        storage_units=storage_units,
        controllers=controllers,
        load_profile=load_profile,
        grid_interconnect_kw=grid_kw,
    )
    return config


# ──────────────────────────────────────────────────────────────────────
# Growth Projection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ProjectionYearResult:
    """Result of a single projection year simulation."""
    year: int
    year_offset: int
    metrics: dict
    fossil_capacity_kw: float
    renewable_capacity_kw: float
    storage_capacity_kwh: float
    peak_demand_kw: float
    renewable_target_pct: float
    renewable_actual_pct: float
    total_capex: float
    lcoe: float
    emissions_tonnes: float


class GrowthProjection:
    """Run multi-year growth projection for a municipal profile."""

    def __init__(self, profile: MunicipalProfile, base_year: int = 2025):
        self.profile = profile
        self.base_year = base_year
        self.results: list[ProjectionYearResult] = []

    def run(self, years: list[int] | None = None,
            sim_hours: int = 8760, dt_hours: float = 1.0) -> list[ProjectionYearResult]:
        """Run simulation for each projection year offset."""
        if years is None:
            years = [0, 5, 10, 15, 20, 25]

        self.results = []
        for y in years:
            config = build_municipal_config(self.profile, year_offset=y,
                                            base_year=self.base_year)
            dispatcher = EnergyDispatcher(config)

            # Use climate-adjusted weather factors
            n_steps = int(sim_hours / dt_hours)
            rng = np.random.default_rng(42 + y)
            solar_factor = self.profile.climate.solar_factor
            weather = np.clip(
                solar_factor * (0.6 + 0.4 * rng.random(n_steps)) +
                0.1 * np.sin(np.arange(n_steps) * 2 * np.pi / (24 / dt_hours * 7)),
                0.05, 1.0
            )

            dispatcher.simulate(hours=sim_hours, dt_hours=dt_hours,
                               weather_factors=weather)
            metrics = dispatcher.compute_metrics()

            # Compute generation mix breakdown
            fossil_kw = sum(s.rated_kw for s in config.sources
                           if not s.is_renewable)
            renewable_kw = sum(s.rated_kw for s in config.sources
                              if s.is_renewable)
            storage_kwh = sum(u.nominal_capacity_kwh for u in config.storage_units)

            # Compute emissions from fossil sources
            emissions_kg = sum(
                getattr(s, 'cumulative_emissions_kg', 0)
                for s in config.sources
            )

            # Projected peak demand
            growth = (1 + self.profile.annual_load_growth_pct / 100) ** y
            peak = self.profile.scale.peak_load_kw * growth

            # Renewable target for this year
            years_to_target = max(1, self.profile.renewable_target_year - self.base_year)
            years_to_nz = max(1, self.profile.net_zero_year - self.base_year)
            if y <= years_to_target:
                target_pct = self.profile.renewable_target_pct * (y / years_to_target)
            else:
                remaining = max(1, years_to_nz - years_to_target)
                progress = min(1.0, (y - years_to_target) / remaining)
                target_pct = (self.profile.renewable_target_pct +
                             (1.0 - self.profile.renewable_target_pct) * progress)

            result = ProjectionYearResult(
                year=self.base_year + y,
                year_offset=y,
                metrics=metrics,
                fossil_capacity_kw=fossil_kw,
                renewable_capacity_kw=renewable_kw,
                storage_capacity_kwh=storage_kwh,
                peak_demand_kw=peak,
                renewable_target_pct=target_pct,
                renewable_actual_pct=metrics.get("avg_renewable_fraction", 0),
                total_capex=metrics.get("total_capex_usd", 0),
                lcoe=metrics.get("estimated_lcoe_usd_kwh", 0),
                emissions_tonnes=emissions_kg / 1000,
            )
            self.results.append(result)
            print(f"  Year {self.base_year + y}: "
                  f"Renew={result.renewable_actual_pct*100:.1f}% "
                  f"(target {target_pct*100:.0f}%)  "
                  f"Fossil={fossil_kw/1000:,.0f} MW  "
                  f"Storage={storage_kwh/1000:,.0f} MWh  "
                  f"LCOE=${result.lcoe:.4f}/kWh  "
                  f"CO2={result.emissions_tonnes:,.0f}t")

        return self.results

    def print_summary(self):
        """Print formatted projection summary."""
        if not self.results:
            print("No projection results.")
            return

        p = self.profile
        print(f"\n{'='*75}")
        print(f"  GROWTH PROJECTION — {p.name}")
        print(f"  Population: {p.population:,} | Climate: {p.climate.name}")
        print(f"  Load growth: {p.annual_load_growth_pct}%/yr | "
              f"Target: {p.renewable_target_pct*100:.0f}% by {p.renewable_target_year}, "
              f"net-zero by {p.net_zero_year}")
        print(f"{'='*75}")

        print(f"\n  {'Year':>6s}  {'Peak MW':>9s}  {'Fossil MW':>10s}  "
              f"{'Renew MW':>10s}  {'Stor MWh':>10s}  "
              f"{'Renew %':>8s}  {'Target':>7s}  "
              f"{'LCOE':>8s}  {'CO2 t':>10s}")
        print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*10}  "
              f"{'─'*8}  {'─'*7}  {'─'*8}  {'─'*10}")
        for r in self.results:
            print(f"  {r.year:>6d}  {r.peak_demand_kw/1000:>9,.1f}  "
                  f"{r.fossil_capacity_kw/1000:>10,.1f}  "
                  f"{r.renewable_capacity_kw/1000:>10,.1f}  "
                  f"{r.storage_capacity_kwh/1000:>10,.1f}  "
                  f"{r.renewable_actual_pct*100:>7.1f}%  "
                  f"{r.renewable_target_pct*100:>6.0f}%  "
                  f"${r.lcoe:>7.4f}  "
                  f"{r.emissions_tonnes:>10,.0f}")


# ──────────────────────────────────────────────────────────────────────
# Preset Municipal Profiles
# ──────────────────────────────────────────────────────────────────────

MUNICIPAL_PROFILES = {
    "small_town": MunicipalProfile(
        name="Rural Midwest Town",
        population=5_000,
        scale=DeploymentScale(
            name="Small Town",
            peak_load_kw=15_000,
            annual_consumption_kwh=45_000_000,
            num_endpoints=2_100,
            description="Small rural town, ~5,000 population",
        ),
        climate=CLIMATE_ZONES["cold_continental"],
        residential_fraction=0.60,
        commercial_fraction=0.25,
        industrial_fraction=0.15,
        annual_load_growth_pct=0.7,
        annual_pop_growth_pct=0.3,
        fossil_capacity_kw=20_000,
        fossil_type="peaker",
        solar_capacity_kw=2_000,
        wind_capacity_kw=500,
        hydro_capacity_kw=0,
        geothermal_capacity_kw=0,
        storage_kwh=500,
        renewable_target_pct=0.30,
        renewable_target_year=2030,
        net_zero_year=2045,
        grid_interconnect_kw=20_000,
        ev_adoption_rate_pct=1.5,
        data_center_load_kw=0,
        data_center_growth_pct=0,
        heat_pump_adoption_rate_pct=1.0,
    ),

    "college_town": MunicipalProfile(
        name="Mid-Size College Town",
        population=100_000,
        scale=DeploymentScale(
            name="College Town",
            peak_load_kw=300_000,
            annual_consumption_kwh=1_300_000_000,
            num_endpoints=42_000,
            description="Mid-size college town, ~100,000 population",
        ),
        climate=CLIMATE_ZONES["temperate"],
        residential_fraction=0.50,
        commercial_fraction=0.35,
        industrial_fraction=0.15,
        annual_load_growth_pct=1.5,
        annual_pop_growth_pct=1.2,
        fossil_capacity_kw=250_000,
        fossil_type="ccgt",
        solar_capacity_kw=80_000,
        wind_capacity_kw=40_000,
        hydro_capacity_kw=0,
        geothermal_capacity_kw=0,
        storage_kwh=20_000,
        renewable_target_pct=0.50,
        renewable_target_year=2030,
        net_zero_year=2040,
        grid_interconnect_kw=200_000,
        ev_adoption_rate_pct=3.0,
        data_center_load_kw=5_000,
        data_center_growth_pct=5.0,
        heat_pump_adoption_rate_pct=2.0,
    ),

    "major_metro": MunicipalProfile(
        name="Major Metropolitan Area",
        population=1_000_000,
        scale=DeploymentScale(
            name="Major Metro",
            peak_load_kw=3_000_000,
            annual_consumption_kwh=15_000_000_000,
            num_endpoints=420_000,
            description="Major metro area, ~1M population",
        ),
        climate=CLIMATE_ZONES["temperate"],
        residential_fraction=0.45,
        commercial_fraction=0.35,
        industrial_fraction=0.20,
        annual_load_growth_pct=2.0,
        annual_pop_growth_pct=1.5,
        fossil_capacity_kw=2_000_000,
        fossil_type="ccgt",
        solar_capacity_kw=800_000,
        wind_capacity_kw=400_000,
        hydro_capacity_kw=100_000,
        geothermal_capacity_kw=0,
        storage_kwh=200_000,
        renewable_target_pct=0.60,
        renewable_target_year=2030,
        net_zero_year=2040,
        grid_interconnect_kw=2_000_000,
        ev_adoption_rate_pct=5.0,
        data_center_load_kw=100_000,
        data_center_growth_pct=8.0,
        heat_pump_adoption_rate_pct=3.0,
    ),

    "sunbelt_boom": MunicipalProfile(
        name="Sunbelt Boomtown",
        population=250_000,
        scale=DeploymentScale(
            name="Sunbelt Boom",
            peak_load_kw=800_000,
            annual_consumption_kwh=4_000_000_000,
            num_endpoints=105_000,
            description="Fast-growing sunbelt city, ~250K population",
        ),
        climate=CLIMATE_ZONES["hot_arid"],
        residential_fraction=0.55,
        commercial_fraction=0.35,
        industrial_fraction=0.10,
        annual_load_growth_pct=3.5,
        annual_pop_growth_pct=4.0,
        fossil_capacity_kw=600_000,
        fossil_type="ccgt",
        solar_capacity_kw=300_000,
        wind_capacity_kw=0,
        hydro_capacity_kw=0,
        geothermal_capacity_kw=0,
        storage_kwh=80_000,
        renewable_target_pct=0.50,
        renewable_target_year=2030,
        net_zero_year=2045,
        grid_interconnect_kw=500_000,
        ev_adoption_rate_pct=4.0,
        data_center_load_kw=20_000,
        data_center_growth_pct=10.0,
        heat_pump_adoption_rate_pct=1.0,
    ),

    "rust_belt": MunicipalProfile(
        name="Northern Industrial City",
        population=150_000,
        scale=DeploymentScale(
            name="Rust Belt",
            peak_load_kw=400_000,
            annual_consumption_kwh=2_000_000_000,
            num_endpoints=62_000,
            description="Northern industrial city, ~150K population",
        ),
        climate=CLIMATE_ZONES["industrial_north"],
        residential_fraction=0.40,
        commercial_fraction=0.25,
        industrial_fraction=0.35,
        annual_load_growth_pct=0.5,
        annual_pop_growth_pct=-0.2,
        fossil_capacity_kw=400_000,
        fossil_type="ccgt",
        solar_capacity_kw=20_000,
        wind_capacity_kw=50_000,
        hydro_capacity_kw=0,
        geothermal_capacity_kw=0,
        storage_kwh=10_000,
        renewable_target_pct=0.40,
        renewable_target_year=2035,
        net_zero_year=2050,
        grid_interconnect_kw=300_000,
        ev_adoption_rate_pct=2.0,
        data_center_load_kw=30_000,
        data_center_growth_pct=10.0,
        heat_pump_adoption_rate_pct=1.5,
    ),
}
