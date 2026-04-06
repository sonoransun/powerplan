#!/usr/bin/env python3
"""
PowerPlan Example Scenarios -- From Home Solar to Municipal Green Conversion

Runs 8 progressive case studies demonstrating renewable energy scaling
from a single dwelling with rooftop solar to a major metropolitan area
pursuing full decarbonization.

Usage:
    python run_examples.py                  # All 8 scenarios (30-day sim)
    python run_examples.py --scenario 2     # Just scenario 2
    python run_examples.py --days 365       # Full-year simulations
    python run_examples.py --no-plot        # Skip plot generation
"""

import argparse
import copy
import time

import numpy as np

from powerplan.storage import (
    LithiumIonBattery, SodiumSolidStateBattery, LiquidElectrolyteBattery,
)
from powerplan.sources import SolarPV, WindTurbine, MicroHydro
from powerplan.controllers import (
    MPPTController, GaNConverter, SiCConverter, BidirectionalInverter,
)
from powerplan.profiles import LoadProfile, SCALES, DeploymentScale
from powerplan.grid import GridConfig, EnergyDispatcher
from powerplan.visualize import (
    plot_simulation, plot_system_diagram, plot_energy_flow, plot_projection,
)
from powerplan.municipal import (
    MUNICIPAL_PROFILES, build_municipal_config, GrowthProjection,
)
from powerplan.scenarios import ResilienceMetrics, FailureScenario, FailureAwareDispatcher


# ──────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────

HOURS = 720  # default 30 days; overridden by --days
PLOT = True


def _run(config, hours=None, plot_prefix=None, days_to_plot=None):
    """Run a simulation and return metrics. Thin wrapper around the engine."""
    h = hours or HOURS
    dispatcher = EnergyDispatcher(config)
    t0 = time.time()
    dispatcher.simulate(hours=h, dt_hours=1.0)
    elapsed = time.time() - t0
    metrics = dispatcher.compute_metrics()

    if PLOT and plot_prefix:
        dp = days_to_plot or (h // 24)
        plot_simulation(dispatcher, save_path=f"{plot_prefix}.png", days=dp)
        plot_system_diagram(dispatcher, save_path=f"{plot_prefix}_diagram.png")
        plot_energy_flow(dispatcher, save_path=f"{plot_prefix}_flow.png")

    return {"dispatcher": dispatcher, "metrics": metrics, "elapsed": elapsed}


def _header(num, title, story):
    print(f"\n{'='*75}")
    print(f"  SCENARIO {num}: {title}")
    print(f"{'='*75}")
    for line in story.strip().split("\n"):
        print(f"  {line.strip()}")
    print()


def _findings(metrics, extra=None, prev_metrics=None, prev_label=None):
    m = metrics
    print(f"  Key Findings:")
    print(f"    Self-sufficiency:     {m['self_sufficiency']*100:>7.1f}%")
    print(f"    Renewable fraction:   {m['avg_renewable_fraction']*100:>7.1f}%")
    print(f"    System efficiency:    {m['avg_system_efficiency']*100:>7.1f}%")
    print(f"    Curtailment:          {m['curtailment_fraction']*100:>7.1f}%")
    print(f"    Total CAPEX:          ${m['total_capex_usd']:>12,.0f}")
    print(f"    LCOE:                 ${m['estimated_lcoe_usd_kwh']:>12.4f}/kWh")
    print(f"    Peak demand:          {m['peak_demand_kw']:>10,.0f} kW")
    print(f"    Total generation:     {m['total_generation_kwh']:>10,.0f} kWh")

    if extra:
        for label, value in extra:
            print(f"    {label:22s}  {value}")

    if prev_metrics and prev_label:
        delta_ss = (m["self_sufficiency"] - prev_metrics["self_sufficiency"]) * 100
        delta_re = (m["avg_renewable_fraction"] - prev_metrics["avg_renewable_fraction"]) * 100
        sign_ss = "+" if delta_ss >= 0 else ""
        sign_re = "+" if delta_re >= 0 else ""
        print(f"\n    vs {prev_label}:")
        print(f"      Self-sufficiency:   {sign_ss}{delta_ss:.1f} pp")
        print(f"      Renewable fraction: {sign_re}{delta_re:.1f} pp")


# ──────────────────────────────────────────────────────────────────────
# Scenario 1: Starter Home Solar
# ──────────────────────────────────────────────────────────────────────

def scenario_1():
    _header(1, "Starter Home Solar",
        """A homeowner installs a 6 kW rooftop solar array on their south-facing
        roof. No battery storage -- just solar panels connected to the grid.
        The home has a 12 kW peak demand with typical residential load patterns.
        How much of their electricity can solar offset?""")

    config = GridConfig(
        name="Home Solar Only",
        scale=SCALES["home"],
        sources=[SolarPV(rated_kw=6.0, latitude=35.0)],
        storage_units=[],
        controllers=[MPPTController(rated_kw=6.0)],
        load_profile=LoadProfile(SCALES["home"]),
        grid_interconnect_kw=10.0,
    )

    result = _run(config, plot_prefix="example_1_home_solar")
    m = result["metrics"]
    _findings(m, extra=[
        ("Grid import", f"{m['total_grid_import_kwh']:,.0f} kWh"),
        ("Grid export", f"{m['total_grid_export_kwh']:,.0f} kWh"),
    ])
    return result


# ──────────────────────────────────────────────────────────────────────
# Scenario 2: Home Solar + Battery
# ──────────────────────────────────────────────────────────────────────

def scenario_2(prev=None):
    _header(2, "Home Solar + Battery",
        """The same homeowner adds a 10 kWh lithium-iron-phosphate (LFP) battery
        with 5 kW charge/discharge capability. Solar energy captured during the
        day can now power the home through the evening peak. How much does
        self-sufficiency improve? How hard does the battery work?""")

    config = GridConfig(
        name="Home Solar+Battery",
        scale=SCALES["home"],
        sources=[SolarPV(rated_kw=6.0, latitude=35.0)],
        storage_units=[
            LithiumIonBattery(capacity_kwh=10.0, max_power_kw=5.0, chemistry="lfp"),
        ],
        controllers=[
            MPPTController(rated_kw=6.0),
            GaNConverter(rated_kw=5.0),
        ],
        load_profile=LoadProfile(SCALES["home"]),
        grid_interconnect_kw=10.0,
    )

    result = _run(config, plot_prefix="example_2_home_battery")
    m = result["metrics"]
    stor = m["storage_details"][0] if m.get("storage_details") else {}
    _findings(m, extra=[
        ("Battery cycles", f"{stor.get('cycles', 0):.0f}"),
        ("Battery health", f"{stor.get('health', 1)*100:.1f}%"),
        ("Grid import", f"{m['total_grid_import_kwh']:,.0f} kWh"),
    ], prev_metrics=prev["metrics"] if prev else None, prev_label="Scenario 1")
    return result


# ──────────────────────────────────────────────────────────────────────
# Scenario 3: Neighborhood Microgrid
# ──────────────────────────────────────────────────────────────────────

def scenario_3(prev=None):
    _header(3, "Neighborhood Microgrid",
        """25 homes pool resources into a shared microgrid with a community
        solar array (150 kW, tracking), a small wind turbine (15 kW), and
        shared battery storage (200 kWh Li-ion + 50 kWh sodium solid-state).
        Aggregation smooths demand peaks and improves economics.""")

    config = GridConfig(
        name="Neighborhood Microgrid",
        scale=SCALES["neighborhood"],
        sources=[
            SolarPV(rated_kw=150.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=15.0, hub_height_m=25.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=200.0, max_power_kw=100.0, chemistry="lfp"),
            SodiumSolidStateBattery(capacity_kwh=50.0, max_power_kw=15.0),
        ],
        controllers=[
            MPPTController(rated_kw=150.0),
            SiCConverter(rated_kw=100.0),
            BidirectionalInverter(rated_kw=100.0),
        ],
        load_profile=LoadProfile(SCALES["neighborhood"]),
        grid_interconnect_kw=150.0,
    )

    result = _run(config, plot_prefix="example_3_neighborhood")
    m = result["metrics"]
    _findings(m, extra=[
        ("Homes served", "25"),
        ("CAPEX per home", f"${m['total_capex_usd']/25:,.0f}"),
    ], prev_metrics=prev["metrics"] if prev else None, prev_label="Scenario 2")
    return result


# ──────────────────────────────────────────────────────────────────────
# Scenario 4: Community Renewable Hub
# ──────────────────────────────────────────────────────────────────────

def scenario_4(prev=None):
    _header(4, "Community Renewable Hub",
        """A 500-home community builds a diverse renewable energy hub:
        1 MW solar farm (tracking), 500 kW wind (3 turbines), 30 kW micro-hydro,
        with 500 kWh Li-ion, 200 kWh vanadium flow battery, and 100 kWh
        sodium solid-state storage. Source diversity reduces intermittency.""")

    config = GridConfig(
        name="Community Hub",
        scale=SCALES["community"],
        sources=[
            SolarPV(rated_kw=1000.0, tracking=True, latitude=35.0),
            WindTurbine(rated_kw=500.0, hub_height_m=50.0, units=3),
            MicroHydro(rated_kw=30.0, head_m=12.0),
        ],
        storage_units=[
            LithiumIonBattery(capacity_kwh=500.0, max_power_kw=250.0, chemistry="lfp"),
            LiquidElectrolyteBattery(capacity_kwh=200.0, max_power_kw=50.0),
            SodiumSolidStateBattery(capacity_kwh=100.0, max_power_kw=30.0),
        ],
        controllers=[
            MPPTController(rated_kw=1000.0),
            SiCConverter(rated_kw=500.0),
            BidirectionalInverter(rated_kw=300.0, phases=3),
        ],
        load_profile=LoadProfile(SCALES["community"]),
        grid_interconnect_kw=500.0,
    )

    result = _run(config, plot_prefix="example_4_community_hub")
    m = result["metrics"]
    n_sources = len(config.sources)
    n_storage = len(config.storage_units)
    _findings(m, extra=[
        ("Source types", f"{n_sources} (solar + wind + hydro)"),
        ("Storage types", f"{n_storage} (Li-ion + flow + Na-SS)"),
        ("CAPEX per home", f"${m['total_capex_usd']/500:,.0f}"),
    ], prev_metrics=prev["metrics"] if prev else None, prev_label="Scenario 3")
    return result


# ──────────────────────────────────────────────────────────────────────
# Scenario 5: Small Town Baseline (Municipal)
# ──────────────────────────────────────────────────────────────────────

def scenario_5(prev=None):
    _header(5, "Small Town Current State",
        """A rural midwest town of 5,000 people today: 15 MW peak demand
        served primarily by a 20 MW natural gas peaker plant, with a small
        2 MW community solar installation and a 500 kWh battery pilot.
        Cold continental climate with harsh winters.
        What is the current renewable fraction and carbon footprint?""")

    profile = MUNICIPAL_PROFILES["small_town"]
    config = build_municipal_config(profile, year_offset=0)

    result = _run(config, plot_prefix="example_5_small_town_baseline")
    m = result["metrics"]

    # Extract emissions from source details
    emissions = sum(s.get("cumulative_emissions_tonnes", s.get("cumulative_emissions_kg", 0) / 1000)
                    for s in m.get("source_details", []))

    _findings(m, extra=[
        ("Population", f"{profile.population:,}"),
        ("Climate", profile.climate.name),
        ("Fossil capacity", f"{profile.fossil_capacity_kw/1000:,.0f} MW"),
        ("CO2 emissions", f"{emissions:,.0f} tonnes"),
    ], prev_metrics=prev["metrics"] if prev else None, prev_label="Scenario 4")
    return result


# ──────────────────────────────────────────────────────────────────────
# Scenario 6: Small Town 10-Year Transition
# ──────────────────────────────────────────────────────────────────────

def scenario_6(prev=None):
    _header(6, "Small Town 10-Year Transition",
        """The same rural town commits to 30% renewable by 2030 and passes
        a clean energy ordinance. Over 10 years, the gas peaker is gradually
        retired while solar and wind capacity grow. Battery storage scales
        to handle the intermittency. How does the grid evolve?""")

    profile = MUNICIPAL_PROFILES["small_town"]
    projection = GrowthProjection(profile, base_year=2025)
    sim_hours = HOURS
    results = projection.run(years=[0, 5, 10], sim_hours=sim_hours)
    projection.print_summary()

    if PLOT:
        plot_projection(projection,
                       save_path="example_6_small_town_transition.png")

    # Use final year metrics for comparison
    final = results[-1]
    first = results[0]
    print(f"\n  Transition Summary (2025 -> 2035):")
    print(f"    Renewable fraction:  {first.renewable_actual_pct*100:.1f}% -> {final.renewable_actual_pct*100:.1f}%")
    print(f"    Fossil capacity:     {first.fossil_capacity_kw/1000:,.0f} MW -> {final.fossil_capacity_kw/1000:,.0f} MW")
    print(f"    Storage:             {first.storage_capacity_kwh/1000:,.1f} MWh -> {final.storage_capacity_kwh/1000:,.1f} MWh")
    print(f"    CO2 emissions:       {first.emissions_tonnes:,.0f}t -> {final.emissions_tonnes:,.0f}t")
    print(f"    LCOE:                ${first.lcoe:.4f} -> ${final.lcoe:.4f}/kWh")

    return {"metrics": final.metrics, "projection": projection}


# ──────────────────────────────────────────────────────────────────────
# Scenario 7: College Town Full Green Conversion
# ──────────────────────────────────────────────────────────────────────

def scenario_7(prev=None):
    _header(7, "College Town Full Green Conversion",
        """A mid-size university city of 100,000 people pursues an aggressive
        clean energy plan: 50% renewable by 2030, 100% by 2040. The 300 MW
        peak demand grows 1.5%/year. EV adoption adds 3%/year to evening
        demand. Data centers bring 5 MW of new baseload. Can they hit
        net-zero by 2040?""")

    profile = MUNICIPAL_PROFILES["college_town"]
    projection = GrowthProjection(profile, base_year=2025)
    sim_hours = HOURS
    results = projection.run(years=[0, 5, 10, 15], sim_hours=sim_hours)
    projection.print_summary()

    if PLOT:
        plot_projection(projection,
                       save_path="example_7_college_town_conversion.png")

    final = results[-1]
    first = results[0]
    print(f"\n  Full Conversion Summary (2025 -> 2040):")
    print(f"    Renewable fraction:  {first.renewable_actual_pct*100:.1f}% -> {final.renewable_actual_pct*100:.1f}%")
    print(f"    Fossil capacity:     {first.fossil_capacity_kw/1000:,.0f} MW -> {final.fossil_capacity_kw/1000:,.0f} MW")
    print(f"    Renewable capacity:  {first.renewable_capacity_kw/1000:,.0f} MW -> {final.renewable_capacity_kw/1000:,.0f} MW")
    print(f"    Storage:             {first.storage_capacity_kwh/1000:,.1f} MWh -> {final.storage_capacity_kwh/1000:,.1f} MWh")
    print(f"    Peak demand growth:  {first.peak_demand_kw/1000:,.0f} MW -> {final.peak_demand_kw/1000:,.0f} MW")
    print(f"    CO2 emissions:       {first.emissions_tonnes:,.0f}t -> {final.emissions_tonnes:,.0f}t")

    return {"metrics": final.metrics, "projection": projection}


# ──────────────────────────────────────────────────────────────────────
# Scenario 8: Major Metro Green Conversion + Resilience
# ──────────────────────────────────────────────────────────────────────

def scenario_8(prev=None):
    _header(8, "Major Metro Green Conversion + Resilience Testing",
        """A major metropolitan area of 1 million people (3 GW peak) pursues
        full decarbonization: 60% renewable by 2030, 100% by 2040. At year
        15 (2040), we stress-test the fully-renewable grid with:
        - A 2-week weather crisis (low solar + wind)
        - A simultaneous source trip + grid disconnect
        - A demand surge (heat wave)
        Can a GW-scale renewable grid survive real-world failures?""")

    profile = MUNICIPAL_PROFILES["major_metro"]
    projection = GrowthProjection(profile, base_year=2025)
    sim_hours = HOURS
    results = projection.run(years=[0, 10, 15], sim_hours=sim_hours)
    projection.print_summary()

    if PLOT:
        plot_projection(projection,
                       save_path="example_8_metro_conversion.png")

    # Resilience testing on the final (year 15) configuration
    print(f"\n  ── Resilience Testing (Year 2040 Grid) ──")
    final_config = build_municipal_config(profile, year_offset=15, base_year=2025)

    failure_scenarios = [
        ("Weather Crisis (14 days low output)",
         FailureScenario(seed=100).add_weather_crisis(168, 336, severity=0.12)),
        ("Source Trip + Grid Disconnect",
         FailureScenario(seed=200)
             .add_source_trip(200, 72, severity=0.8)
             .add_grid_disconnect(200, 48)),
        ("Heat Wave Demand Surge (7 days +60%)",
         FailureScenario(seed=300).add_demand_surge(400, 168, multiplier=1.6)),
    ]

    for desc, failure in failure_scenarios:
        config_copy = copy.deepcopy(final_config)
        mod_config, weather, timeline = failure.apply(config_copy, sim_hours, 1.0)
        dispatcher = FailureAwareDispatcher(mod_config, timeline)
        dispatcher.simulate(hours=sim_hours, dt_hours=1.0, weather_factors=weather)
        res = ResilienceMetrics.compute(dispatcher)

        print(f"\n  Failure: {desc}")
        print(f"    LOLP:               {res.lolp:.4f} ({res.lolp*100:.1f}% of hours)")
        print(f"    Energy Not Served:  {res.ens_kwh:,.0f} kWh")
        print(f"    Max Deficit:        {res.max_deficit_kw:,.0f} kW")
        print(f"    Max Deficit Dur:    {res.max_deficit_duration_hours:.1f} hours")
        print(f"    Recovery Time:      {res.recovery_time_max:.1f} hours")

    final = results[-1]
    return {"metrics": final.metrics, "projection": projection}


# ──────────────────────────────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results):
    """Print a summary table comparing all completed scenarios."""
    print(f"\n\n{'='*95}")
    print(f"  PROGRESSIVE ANALYSIS -- Home Solar to Municipal Green Conversion")
    print(f"{'='*95}")
    print(f"  {'#':>2s}  {'Scenario':28s}  {'Peak kW':>10s}  {'Gen kW':>10s}  "
          f"{'Storage':>10s}  {'Self-Suff':>9s}  {'Renew%':>7s}  {'LCOE':>9s}")
    print(f"  {'─'*2}  {'─'*28}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*9}  {'─'*7}  {'─'*9}")

    for i, (name, res) in enumerate(all_results, 1):
        if res is None:
            continue
        m = res["metrics"]
        peak = m.get("peak_demand_kw", 0)
        gen = m.get("peak_generation_kw", 0)
        stor = sum(s.get("nominal_capacity_kwh", 0) for s in m.get("storage_details", []))

        ss = m.get("self_sufficiency", 0) * 100
        rf = m.get("avg_renewable_fraction", 0) * 100
        lcoe = m.get("estimated_lcoe_usd_kwh", 0)

        # Format storage
        if stor >= 1_000_000:
            stor_str = f"{stor/1e6:,.1f} GWh"
        elif stor >= 1000:
            stor_str = f"{stor/1000:,.0f} MWh"
        elif stor > 0:
            stor_str = f"{stor:,.0f} kWh"
        else:
            stor_str = "none"

        print(f"  {i:>2d}  {name:28s}  {peak:>10,.0f}  {gen:>10,.0f}  "
              f"{stor_str:>10s}  {ss:>8.1f}%  {rf:>6.1f}%  ${lcoe:>8.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    global HOURS, PLOT

    parser = argparse.ArgumentParser(
        description="PowerPlan Examples -- Home Solar to Municipal Green Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py                  Run all 8 scenarios (30-day sim)
  python run_examples.py --scenario 2     Just scenario 2
  python run_examples.py --days 365       Full-year simulations
  python run_examples.py --no-plot        Skip plot generation
        """,
    )
    parser.add_argument("--scenario", type=int, default=None,
                       help="Run a specific scenario (1-8)")
    parser.add_argument("--days", type=int, default=30,
                       help="Simulation duration in days (default: 30)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plot generation")

    args = parser.parse_args()
    HOURS = args.days * 24
    PLOT = not args.no_plot

    print(f"\n{'#'*75}")
    print(f"#{'':73s}#")
    print(f"#  {'PowerPlan -- From Home Solar to Municipal Green Conversion':^71s}#")
    print(f"#  {'8 Progressive Case Studies':^71s}#")
    print(f"#  {f'Simulation: {args.days} days per scenario':^71s}#")
    print(f"#{'':73s}#")
    print(f"{'#'*75}")

    scenarios = {
        1: ("Home Solar Only", scenario_1),
        2: ("Home Solar+Battery", scenario_2),
        3: ("Neighborhood Microgrid", scenario_3),
        4: ("Community Hub", scenario_4),
        5: ("Small Town (baseline)", scenario_5),
        6: ("Small Town (10yr transition)", scenario_6),
        7: ("College Town (green conversion)", scenario_7),
        8: ("Major Metro (green + resilience)", scenario_8),
    }

    if args.scenario:
        if args.scenario not in scenarios:
            print(f"Invalid scenario {args.scenario}. Choose 1-8.")
            return
        name, fn = scenarios[args.scenario]
        result = fn()
        return

    # Run all scenarios progressively
    all_results = []
    prev = None
    for num, (name, fn) in scenarios.items():
        if num <= 4:
            result = fn(prev) if num > 1 else fn()
        elif num <= 5:
            result = fn(prev)
        else:
            result = fn()
        all_results.append((name, result))
        if result and "metrics" in result:
            prev = result

    # Print comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
