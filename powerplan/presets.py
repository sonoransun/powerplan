"""
Preset grid configurations — from a single home to metropolitan scale.

Ten tuned baseline configs (5 conventional + 5 exotic), extracted from
run_simulation.py so they are importable without pulling in argparse or
matplotlib (e.g. from the browser/Pyodide runtime and the prebake script).
Each factory returns a fresh GridConfig with fresh component instances.
"""

from __future__ import annotations

from .storage import (
    LithiumIonBattery, SodiumSolidStateBattery, LiquidElectrolyteBattery,
    FlywheelStorage, HydrogenFuelCell,
    GrapheneSupercapacitor, SMES,
)
from .sources import (
    SolarPV, WindTurbine, MicroHydro, Geothermal,
    MicroFusionReactor, AntimatterReactor,
)
from .controllers import (
    SiCConverter, GaNConverter, MPPTController,
    BidirectionalInverter, HydrogenPowerController,
    FusionPowerController, CryogenicPowerSupply,
    AntimatterPowerController,
)
from .profiles import LoadProfile, SCALES
from .grid import GridConfig

# ──────────────────────────────────────────────────────────────────────
# Preset Configurations — from home to metro scale
# ──────────────────────────────────────────────────────────────────────

def config_home() -> GridConfig:
    """Single home: rooftop solar + Li-ion + small wind."""
    scale = SCALES["home"]
    return GridConfig(
        name="Home",
        scale=scale,
        sources=[
            SolarPV(rated_kw=8.0, latitude=35.0),
            WindTurbine(rated_kw=1.5, hub_height_m=15.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=13.5, max_power_kw=5.0, chemistry="lfp"),
        ],
        controllers=[
            MPPTController(rated_kw=8.0),
            GaNConverter(rated_kw=5.0),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=10.0,  # Grid-tied
    )


def config_neighborhood() -> GridConfig:
    """25-home neighborhood microgrid with diverse storage."""
    scale = SCALES["neighborhood"]
    return GridConfig(
        name="Neighborhood",
        scale=scale,
        sources=[
            SolarPV(rated_kw=100.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=25.0, hub_height_m=30.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=50.0, max_power_kw=25.0, chemistry="lfp", units=2),
            SodiumSolidStateBattery(capacity_kwh=30.0, max_power_kw=10.0),
            FlywheelStorage(capacity_kwh=3.0, max_power_kw=50.0),  # Power quality
        ],
        controllers=[
            MPPTController(rated_kw=100.0),
            SiCConverter(rated_kw=50.0),
            BidirectionalInverter(rated_kw=75.0),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=100.0,
    )


def config_community() -> GridConfig:
    """500-home community with all storage types."""
    scale = SCALES["community"]
    return GridConfig(
        name="Community",
        scale=scale,
        sources=[
            SolarPV(rated_kw=800.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=500.0, hub_height_m=60.0, units=3),
            MicroHydro(rated_kw=50.0, head_m=15.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=200.0, max_power_kw=100.0, chemistry="nmc", units=4),
            SodiumSolidStateBattery(capacity_kwh=150.0, max_power_kw=50.0, units=3),
            LiquidElectrolyteBattery(capacity_kwh=500.0, max_power_kw=100.0, chemistry="vanadium"),
            FlywheelStorage(capacity_kwh=5.0, max_power_kw=200.0, units=2),
            HydrogenFuelCell(h2_tank_kg=100.0, electrolyzer_kw=80.0, fuel_cell_kw=60.0),
        ],
        controllers=[
            MPPTController(rated_kw=800.0),
            SiCConverter(rated_kw=500.0),
            BidirectionalInverter(rated_kw=400.0, phases=3),
            HydrogenPowerController(rated_kw=80.0),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=500.0,
    )


def config_district() -> GridConfig:
    """Urban district — 8,000 endpoints, heterogeneous demand."""
    scale = SCALES["district"]
    return GridConfig(
        name="District",
        scale=scale,
        sources=[
            SolarPV(rated_kw=10_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=5_000.0, hub_height_m=80.0, units=10),
            MicroHydro(rated_kw=500.0, head_m=25.0),
            Geothermal(rated_kw=2_000.0, well_temp_c=160.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=1000.0, max_power_kw=500.0, chemistry="lfp", units=10),
            SodiumSolidStateBattery(capacity_kwh=500.0, max_power_kw=200.0, units=8),
            LiquidElectrolyteBattery(capacity_kwh=5000.0, max_power_kw=1000.0, chemistry="vanadium", units=3),
            FlywheelStorage(capacity_kwh=10.0, max_power_kw=500.0, units=5),
            HydrogenFuelCell(h2_tank_kg=2000.0, electrolyzer_kw=1000.0, fuel_cell_kw=800.0, units=2),
        ],
        controllers=[
            MPPTController(rated_kw=10_000.0),
            SiCConverter(rated_kw=8_000.0),
            BidirectionalInverter(rated_kw=5_000.0, phases=3),
            HydrogenPowerController(rated_kw=2_000.0),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=10_000.0,
    )


def config_metropolitan() -> GridConfig:
    """Full metropolitan deployment — 200,000 endpoints."""
    scale = SCALES["metropolitan"]
    return GridConfig(
        name="Metropolitan",
        scale=scale,
        sources=[
            SolarPV(rated_kw=250_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=15_000.0, hub_height_m=100.0, units=15),
            MicroHydro(rated_kw=5_000.0, head_m=30.0, units=3),
            Geothermal(rated_kw=50_000.0, well_temp_c=180.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=5000.0, max_power_kw=2500.0, chemistry="lfp", units=50),
            SodiumSolidStateBattery(capacity_kwh=2000.0, max_power_kw=800.0, units=30),
            LiquidElectrolyteBattery(capacity_kwh=20000.0, max_power_kw=5000.0, chemistry="vanadium", units=10),
            FlywheelStorage(capacity_kwh=20.0, max_power_kw=2000.0, units=10),
            HydrogenFuelCell(h2_tank_kg=50000.0, electrolyzer_kw=20000.0, fuel_cell_kw=15000.0, units=5),
        ],
        controllers=[
            MPPTController(rated_kw=250_000.0),
            SiCConverter(rated_kw=150_000.0),
            BidirectionalInverter(rated_kw=100_000.0, phases=3),
            HydrogenPowerController(rated_kw=40_000.0),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=200_000.0,
    )


def config_community_fusion() -> GridConfig:
    """Community scale with D-T micro-fusion baseload + graphene supercap stabilization."""
    scale = SCALES["community"]
    return GridConfig(
        name="Community+Fusion",
        scale=scale,
        sources=[
            MicroFusionReactor(
                rated_kw=1_500.0,
                fuel_cycle="dt",
                q_engineering=10.0,
                confinement="compact_tokamak",
            ),
            SolarPV(rated_kw=300.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=200.0, hub_height_m=60.0),
        ],
        storage_units=[
            GrapheneSupercapacitor(
                capacity_kwh=20.0,
                max_power_kw=5_000.0,
            ),
            LithiumIonBattery(capacity_kwh=200.0, max_power_kw=100.0, chemistry="lfp", units=2),
            HydrogenFuelCell(h2_tank_kg=200.0, electrolyzer_kw=100.0, fuel_cell_kw=80.0),
        ],
        controllers=[
            FusionPowerController(rated_kw=1_500.0, conversion="brayton"),
            MPPTController(rated_kw=300.0),
            SiCConverter(rated_kw=500.0),
            BidirectionalInverter(rated_kw=500.0, phases=3),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=500.0,
    )


def config_district_exotic() -> GridConfig:
    """District with fusion baseload, graphene supercap for power quality, SMES for grid stability."""
    scale = SCALES["district"]
    return GridConfig(
        name="District+Exotic",
        scale=scale,
        sources=[
            MicroFusionReactor(
                rated_kw=15_000.0,
                fuel_cycle="dt",
                q_engineering=12.0,
                confinement="compact_tokamak",
            ),
            SolarPV(rated_kw=5_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=3_000.0, hub_height_m=80.0, units=5),
            Geothermal(rated_kw=2_000.0, well_temp_c=160.0),
        ],
        storage_units=[
            GrapheneSupercapacitor(
                capacity_kwh=100.0,
                max_power_kw=50_000.0,
            ),
            SMES(
                capacity_kwh=30.0,
                max_power_kw=80_000.0,
                inductance_h=20.0,
                operating_temp_k=30.0,
            ),
            LithiumIonBattery(capacity_kwh=1000.0, max_power_kw=500.0, chemistry="lfp", units=5),
            SodiumSolidStateBattery(capacity_kwh=500.0, max_power_kw=200.0, units=4),
            LiquidElectrolyteBattery(capacity_kwh=5000.0, max_power_kw=1000.0, chemistry="vanadium", units=2),
            HydrogenFuelCell(h2_tank_kg=5000.0, electrolyzer_kw=2000.0, fuel_cell_kw=1500.0),
        ],
        controllers=[
            FusionPowerController(rated_kw=15_000.0, conversion="brayton"),
            CryogenicPowerSupply(rated_kw=500.0, cooling_stage_k=30.0),
            MPPTController(rated_kw=5_000.0),
            SiCConverter(rated_kw=10_000.0),
            BidirectionalInverter(rated_kw=8_000.0, phases=3),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=5_000.0,
    )


def config_metro_exotic() -> GridConfig:
    """
    Metropolitan scale with full exotic technology mix.
    Multiple fusion reactors (D-T + p-B11), massive graphene supercap arrays,
    SMES for grid stabilization, full conventional storage complement.
    """
    scale = SCALES["metropolitan"]
    return GridConfig(
        name="Metro+Exotic",
        scale=scale,
        sources=[
            MicroFusionReactor(
                rated_kw=100_000.0,
                fuel_cycle="dt",
                q_engineering=15.0,
                confinement="compact_tokamak",
                units=3,
            ),
            MicroFusionReactor(
                rated_kw=50_000.0,
                fuel_cycle="pb11",
                q_engineering=4.0,
                confinement="field_reversed",
            ),
            SolarPV(rated_kw=100_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=10_000.0, hub_height_m=100.0, units=8),
            Geothermal(rated_kw=30_000.0, well_temp_c=180.0),
        ],
        storage_units=[
            GrapheneSupercapacitor(
                capacity_kwh=500.0,
                max_power_kw=200_000.0,
                units=5,
            ),
            SMES(
                capacity_kwh=50.0,
                max_power_kw=150_000.0,
                inductance_h=50.0,
                operating_temp_k=20.0,
                units=3,
            ),
            LithiumIonBattery(capacity_kwh=5000.0, max_power_kw=2500.0, chemistry="lfp", units=20),
            SodiumSolidStateBattery(capacity_kwh=2000.0, max_power_kw=800.0, units=15),
            LiquidElectrolyteBattery(capacity_kwh=20000.0, max_power_kw=5000.0,
                                     chemistry="vanadium", units=5),
            HydrogenFuelCell(h2_tank_kg=50000.0, electrolyzer_kw=20000.0,
                            fuel_cell_kw=15000.0, units=3),
        ],
        controllers=[
            FusionPowerController(rated_kw=300_000.0, conversion="brayton"),
            FusionPowerController(rated_kw=50_000.0, conversion="direct"),
            CryogenicPowerSupply(rated_kw=2_000.0, cooling_stage_k=20.0),
            MPPTController(rated_kw=100_000.0),
            SiCConverter(rated_kw=100_000.0),
            BidirectionalInverter(rated_kw=80_000.0, phases=3),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=100_000.0,
    )


PRESETS = {
    "home": config_home,
    "neighborhood": config_neighborhood,
    "community": config_community,
    "district": config_district,
    "metro": config_metropolitan,
}

def config_district_antimatter() -> GridConfig:
    """
    District scale with antimatter baseload using uranium-238 target.
    Antiproton-uranium interaction produces Auger and ionization electrons
    collected on graphene electrodes, plus antimatter-catalyzed fission
    amplification (~10.7% energy bonus).
    """
    scale = SCALES["district"]
    return GridConfig(
        name="District+Antimatter",
        scale=scale,
        sources=[
            AntimatterReactor(
                rated_kw=20_000.0,
                containment="graphene_penning",
                target_atom="uranium",
                electron_collection_efficiency=0.48,
                magnetic_field_tesla=6.0,
                graphene_electrode_layers=150,
                fuel_reservoir_ug=5_000_000.0,
                mhd_efficiency=0.72,
                gamma_thermal_efficiency=0.36,
            ),
            SolarPV(rated_kw=5_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=2_000.0, hub_height_m=80.0, units=3),
        ],
        storage_units=[
            GrapheneSupercapacitor(
                capacity_kwh=150.0,
                max_power_kw=80_000.0,
            ),
            SMES(
                capacity_kwh=40.0,
                max_power_kw=100_000.0,
                inductance_h=25.0,
                operating_temp_k=25.0,
            ),
            LithiumIonBattery(capacity_kwh=1000.0, max_power_kw=500.0,
                             chemistry="lfp", units=5),
            LiquidElectrolyteBattery(capacity_kwh=5000.0, max_power_kw=1000.0,
                                     chemistry="vanadium", units=2),
        ],
        controllers=[
            AntimatterPowerController(rated_kw=20_000.0, mhd_fraction=0.56,
                                      electron_fraction=0.22),
            CryogenicPowerSupply(rated_kw=800.0, cooling_stage_k=25.0),
            MPPTController(rated_kw=5_000.0),
            SiCConverter(rated_kw=8_000.0),
            BidirectionalInverter(rated_kw=6_000.0, phases=3),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=5_000.0,
    )


def config_metro_antimatter() -> GridConfig:
    """
    Metropolitan scale with multiple antimatter reactors + fusion backup.
    Mixed target atoms: uranium for high-power baseload with fission
    amplification, xenon for clean electron-harvesting peaker units.
    """
    scale = SCALES["metropolitan"]
    return GridConfig(
        name="Metro+Antimatter",
        scale=scale,
        sources=[
            AntimatterReactor(
                rated_kw=100_000.0,
                containment="graphene_penning",
                target_atom="uranium",
                electron_collection_efficiency=0.50,
                magnetic_field_tesla=8.0,
                graphene_electrode_layers=200,
                fuel_reservoir_ug=70_000_000.0,
                mhd_efficiency=0.75,
                gamma_thermal_efficiency=0.38,
                units=2,
            ),
            AntimatterReactor(
                rated_kw=50_000.0,
                containment="graphene_penning",
                target_atom="xenon",
                electron_collection_efficiency=0.52,
                magnetic_field_tesla=6.0,
                graphene_electrode_layers=120,
                fuel_reservoir_ug=20_000_000.0,
                mhd_efficiency=0.72,
                gamma_thermal_efficiency=0.35,
            ),
            MicroFusionReactor(
                rated_kw=50_000.0,
                fuel_cycle="dt",
                q_engineering=12.0,
                confinement="compact_tokamak",
                units=2,
            ),
            SolarPV(rated_kw=80_000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=8_000.0, hub_height_m=100.0, units=6),
            Geothermal(rated_kw=20_000.0, well_temp_c=180.0),
        ],
        storage_units=[
            GrapheneSupercapacitor(
                capacity_kwh=800.0,
                max_power_kw=300_000.0,
                units=8,
            ),
            SMES(
                capacity_kwh=80.0,
                max_power_kw=200_000.0,
                inductance_h=60.0,
                operating_temp_k=20.0,
                units=4,
            ),
            LithiumIonBattery(capacity_kwh=5000.0, max_power_kw=2500.0,
                             chemistry="lfp", units=30),
            LiquidElectrolyteBattery(capacity_kwh=20000.0, max_power_kw=5000.0,
                                     chemistry="vanadium", units=8),
            HydrogenFuelCell(h2_tank_kg=50000.0, electrolyzer_kw=20000.0,
                            fuel_cell_kw=15000.0, units=3),
        ],
        controllers=[
            AntimatterPowerController(rated_kw=200_000.0, mhd_fraction=0.56,
                                      electron_fraction=0.22),
            AntimatterPowerController(rated_kw=50_000.0, mhd_fraction=0.63,
                                      electron_fraction=0.10),
            FusionPowerController(rated_kw=100_000.0, conversion="brayton"),
            CryogenicPowerSupply(rated_kw=3_000.0, cooling_stage_k=20.0),
            MPPTController(rated_kw=80_000.0),
            SiCConverter(rated_kw=120_000.0),
            BidirectionalInverter(rated_kw=100_000.0, phases=3),
        ],
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=100_000.0,
    )


EXOTIC_PRESETS = {
    "community_fusion": config_community_fusion,
    "district_exotic": config_district_exotic,
    "metro_exotic": config_metro_exotic,
    "district_antimatter": config_district_antimatter,
    "metro_antimatter": config_metro_antimatter,
}


ALL_PRESETS = {**PRESETS, **EXOTIC_PRESETS}
