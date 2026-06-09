#!/usr/bin/env python3
"""
PowerPlan — Heterogeneous Power System Simulator

Simulates arbitrary configurations of energy sources, storage technologies,
and solid-state power controllers from home scale to metropolitan deployments.

Usage:
    python run_simulation.py                    # Run all preset scenarios
    python run_simulation.py --scale home       # Single scale
    python run_simulation.py --scale metro      # Metropolitan scale
    python run_simulation.py --days 30          # Simulate 30 days
    python run_simulation.py --compare          # Compare all scales
    python run_simulation.py --custom           # Interactive custom config
"""

import argparse
import sys
import time
import json

import numpy as np

from powerplan.storage import (
    LithiumIonBattery, SodiumSolidStateBattery, LiquidElectrolyteBattery,
    FlywheelStorage, HydrogenFuelCell,
    GrapheneSupercapacitor, SMES,
    create_storage,
)
from powerplan.sources import (
    SolarPV, WindTurbine, MicroHydro, Geothermal,
    MicroFusionReactor, AntimatterReactor,
    create_source,
)
from powerplan.controllers import (
    SiCConverter, GaNConverter, MPPTController,
    BidirectionalInverter, HydrogenPowerController,
    FusionPowerController, CryogenicPowerSupply,
    AntimatterPowerController,
    create_controller,
)
from powerplan.profiles import LoadProfile, SCALES
from powerplan.grid import GridConfig, EnergyDispatcher
from powerplan.visualize import (
    plot_simulation, plot_comparison, plot_projection,
    plot_system_diagram, plot_energy_flow,
)
from powerplan.scenarios import (
    ConfigGenerator, FailureScenario, ResilienceMetrics, ScenarioRunner,
)
from powerplan.municipal import (
    MUNICIPAL_PROFILES, CLIMATE_ZONES, build_municipal_config, GrowthProjection,
)
from powerplan.presets import PRESETS, EXOTIC_PRESETS



# ──────────────────────────────────────────────────────────────────────
# Simulation runner
# ──────────────────────────────────────────────────────────────────────

def run_single(config: GridConfig, hours: int = 8760, dt: float = 1.0,
               plot: bool = True, days_to_plot: int | None = None) -> dict:
    """Run simulation for a single configuration and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Simulating: {config.name} ({config.scale.name})")
    print(f"  Scale: {config.scale.description}")
    print(f"  Sources: {len(config.sources)} | Storage: {len(config.storage_units)} | Controllers: {len(config.controllers)}")
    print(f"  Simulation: {hours}h at {dt}h steps")
    print(f"{'='*60}")

    dispatcher = EnergyDispatcher(config)
    t0 = time.time()
    dispatcher.simulate(hours=hours, dt_hours=dt)
    elapsed = time.time() - t0
    print(f"  Simulation completed in {elapsed:.1f}s")

    metrics = dispatcher.compute_metrics()
    print_metrics(metrics)

    if plot:
        safe_name = config.name.lower().replace(" ", "_").replace("+", "_")
        plot_simulation(dispatcher, save_path=f"powerplan_{safe_name}.png",
                       days=days_to_plot)
        plot_system_diagram(dispatcher,
                           save_path=f"powerplan_{safe_name}_diagram.png")
        plot_energy_flow(dispatcher,
                        save_path=f"powerplan_{safe_name}_flow.png")

    return {"dispatcher": dispatcher, "metrics": metrics}


def print_metrics(m: dict):
    """Pretty-print simulation metrics."""
    print(f"\n  ── Performance Metrics ──")
    print(f"  Total Demand:         {m['total_demand_kwh']:>14,.0f} kWh")
    print(f"  Total Generation:     {m['total_generation_kwh']:>14,.0f} kWh")
    print(f"  Gen/Demand Ratio:     {m['generation_to_demand_ratio']:>14.2f}")
    print(f"  Self-Sufficiency:     {m['self_sufficiency']*100:>13.1f}%")
    print(f"  Avg Renewable Frac:   {m['avg_renewable_fraction']*100:>13.1f}%")
    print(f"  Avg System Efficiency:{m['avg_system_efficiency']*100:>13.1f}%")
    print(f"  Curtailment:          {m['curtailment_fraction']*100:>13.1f}%")
    print(f"  Controller Losses:    {m['total_controller_losses_kwh']:>14,.0f} kWh")

    print(f"\n  ── Economics ──")
    print(f"  Total CAPEX:          ${m['total_capex_usd']:>13,.0f}")
    print(f"  Estimated LCOE:       ${m['estimated_lcoe_usd_kwh']:>13.4f}/kWh")

    print(f"\n  ── Sources ──")
    for s in m.get("source_details", []):
        print(f"    {s['name']:20s}  {s['rated_kw']:>8,.0f} kW  CF={s['annual_cf']:.2f}  ${s['capital_cost']:>12,.0f}")
        if "fuel_remaining_ug" in s:
            consumed_mg = s['fuel_consumed_ug'] / 1000
            remaining_mg = s['fuel_remaining_ug'] / 1000
            target = s.get('target_atom', 'none')
            fission = s.get('fission_amplification', 1.0)
            target_str = f"  Target: {s.get('target_atom_name', 'none')}" if target != "none" else ""
            fission_str = f"  Fission amp: {fission:.1%}" if fission > 1.0 else ""
            print(f"      Fuel: {consumed_mg:,.1f} mg consumed, "
                  f"{remaining_mg:,.1f} mg remaining  "
                  f"(rate={s['fuel_rate_ug_per_hour']:,.1f} μg/h)")
            e_frac = s.get('electron_fraction', 0)
            if e_frac > 0:
                print(f"      Pathways: MHD={s['effective_pion_fraction']:.0%}  "
                      f"Thermal={s['gamma_fraction']:.0%}  "
                      f"Electron={e_frac:.0%}  "
                      f"(η_collect={s['electron_collection_efficiency']:.0%})")
            print(f"      Electrode health: {s['electrode_health']:.1%}  "
                  f"Failures: {s['containment_failures']}"
                  f"{target_str}{fission_str}")

    print(f"\n  ── Storage ──")
    for s in m.get("storage_details", []):
        print(f"    {s['name']:20s}  {s['nominal_capacity_kwh']:>8,.0f} kWh  "
              f"SOC={s['soc']:.2f}  Health={s['health']:.3f}  "
              f"Cycles={s['cycles']:.0f}  ${s['capital_cost']:>12,.0f}")

    print(f"\n  ── Controllers ──")
    for c in m.get("controller_details", []):
        print(f"    {c['name']:20s}  {c['rated_kw']:>8,.0f} kW  "
              f"Losses={c['cumulative_loss_kwh']:>8,.0f} kWh  "
              f"Hours={c['operating_hours']:>6,.0f}")


def run_comparison(hours: int = 8760, dt: float = 1.0, exotic: bool = False):
    """Run presets and generate comparison plots."""
    dispatchers = []
    presets = {**PRESETS, **EXOTIC_PRESETS} if exotic else PRESETS
    for key, factory in presets.items():
        config = factory()
        result = run_single(config, hours=hours, dt=dt, plot=True, days_to_plot=14)
        dispatchers.append(result["dispatcher"])

    if len(dispatchers) > 1:
        suffix = "_exotic" if exotic else ""
        plot_comparison(dispatchers, save_path=f"powerplan_comparison{suffix}.png")
        print(f"\nComparison plot saved to powerplan_comparison{suffix}.png")


def interactive_custom():
    """Interactive configuration builder."""
    print("\n" + "="*60)
    print("  PowerPlan — Custom Configuration Builder")
    print("="*60)

    print("\nAvailable scales:")
    for key, scale in SCALES.items():
        print(f"  {key:15s} — {scale.description}")
    scale_key = input("\nSelect scale [home]: ").strip() or "home"
    scale = SCALES.get(scale_key, SCALES["home"])

    print(f"\nConfiguring {scale.name} ({scale.peak_load_kw:.0f} kW peak)")

    # Sources
    print("\nAvailable sources: solar_pv, wind, micro_hydro, geothermal, micro_fusion, antimatter")
    sources = []
    while True:
        src = input("Add source (or 'done'): ").strip()
        if src == "done" or src == "":
            break
        try:
            kw = float(input(f"  Rated kW for {src}: "))
            sources.append(create_source(src, rated_kw=kw))
            print(f"  Added {src} at {kw} kW")
        except (ValueError, KeyError) as e:
            print(f"  Error: {e}")

    # Storage
    print("\nAvailable storage: lithium_ion, sodium_solid_state, liquid_electrolyte,")
    print("  flywheel, hydrogen_fuel_cell, graphene_supercap, smes")
    storage = []
    while True:
        st = input("Add storage (or 'done'): ").strip()
        if st == "done" or st == "":
            break
        try:
            if st == "hydrogen_fuel_cell":
                kg = float(input(f"  H2 tank kg: "))
                ekw = float(input(f"  Electrolyzer kW: "))
                fkw = float(input(f"  Fuel cell kW: "))
                storage.append(create_storage(st, h2_tank_kg=kg, electrolyzer_kw=ekw, fuel_cell_kw=fkw))
            elif st == "flywheel":
                kwh = float(input(f"  Capacity kWh: "))
                kw = float(input(f"  Max power kW: "))
                storage.append(create_storage(st, capacity_kwh=kwh, max_power_kw=kw))
            else:
                kwh = float(input(f"  Capacity kWh: "))
                kw = float(input(f"  Max power kW: "))
                storage.append(create_storage(st, capacity_kwh=kwh, max_power_kw=kw))
            print(f"  Added {st}")
        except (ValueError, KeyError) as e:
            print(f"  Error: {e}")

    # Use sensible default controllers
    total_src_kw = sum(s.rated_kw for s in sources)
    total_stor_kw = sum(s.max_discharge_kw for s in storage)
    controllers = [
        MPPTController(rated_kw=max(total_src_kw, 1)),
        SiCConverter(rated_kw=max(total_stor_kw, 1)),
        BidirectionalInverter(rated_kw=max(total_stor_kw, 1)),
    ]

    grid_kw = float(input(f"\nGrid interconnect kW (0 for island) [{scale.peak_load_kw:.0f}]: ").strip()
                     or str(scale.peak_load_kw))

    config = GridConfig(
        name="Custom",
        scale=scale,
        sources=sources,
        storage_units=storage,
        controllers=controllers,
        load_profile=LoadProfile(scale),
        grid_interconnect_kw=grid_kw,
    )

    hours = int(input("Simulation hours [8760]: ").strip() or "8760")
    run_single(config, hours=hours, plot=True, days_to_plot=30)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    all_presets = {**PRESETS, **EXOTIC_PRESETS}
    parser = argparse.ArgumentParser(
        description="PowerPlan — Heterogeneous Power System Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                          Run all conventional presets
  python run_simulation.py --scale home             Single home scenario
  python run_simulation.py --scale metro_exotic     Metro with fusion + exotic storage
  python run_simulation.py --compare --exotic       Compare all (conventional + exotic)
  python run_simulation.py --scale district --resilience  Resilience metrics on preset
  python run_simulation.py --scenarios 10 --failures 3    10 random configs, 3 failures each
  python run_simulation.py --scenarios 20 --tier exotic --scale district  Exotic at district
  python run_simulation.py --scenarios 5 --scale metropolitan --tier antimatter  Metro stress
  python run_simulation.py --custom                 Interactive builder
        """,
    )
    parser.add_argument("--scale", choices=list(all_presets.keys()),
                       help="Run a specific preset scale")
    parser.add_argument("--days", type=int, default=None,
                       help="Number of days to simulate (default: 365)")
    parser.add_argument("--plot-days", type=int, default=None,
                       help="Number of days to show in plots (default: all)")
    parser.add_argument("--dt", type=float, default=1.0,
                       help="Time step in hours (default: 1.0)")
    parser.add_argument("--compare", action="store_true",
                       help="Run all presets and compare")
    parser.add_argument("--exotic", action="store_true",
                       help="Include exotic technologies (fusion, graphene supercap, SMES)")
    parser.add_argument("--custom", action="store_true",
                       help="Interactive custom configuration")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--scenarios", type=int, default=None, metavar="N",
                       help="Generate and test N random configurations")
    parser.add_argument("--failures", type=int, default=3, metavar="M",
                       help="Failure scenarios per config (default: 3)")
    parser.add_argument("--resilience", action="store_true",
                       help="Compute extended resilience metrics")
    parser.add_argument("--tier", choices=["conventional", "exotic", "antimatter"],
                       default="conventional",
                       help="Technology tier for scenario generation")
    parser.add_argument("--scenario-seed", type=int, default=42,
                       help="Random seed for scenario generation")
    parser.add_argument("--municipal", choices=list(MUNICIPAL_PROFILES.keys()),
                       help="Run a municipal infrastructure preset")
    parser.add_argument("--project-years", type=int, nargs="+", default=None,
                       metavar="Y", help="Projection year offsets (e.g. 0 5 10 15 20 25)")
    parser.add_argument("--climate", choices=list(CLIMATE_ZONES.keys()),
                       help="Override climate zone for municipal preset")
    parser.add_argument("--base-year", type=int, default=2025,
                       help="Base year for municipal projection (default: 2025)")
    parser.add_argument("--list-municipal", action="store_true",
                       help="List available municipal presets")

    args = parser.parse_args()
    hours = (args.days * 24) if args.days else 8760

    if args.list_municipal:
        print("\nAvailable municipal presets:")
        print(f"  {'Key':15s}  {'Name':30s}  {'Pop':>10s}  {'Peak MW':>10s}  {'Climate':>25s}")
        for key, p in MUNICIPAL_PROFILES.items():
            print(f"  {key:15s}  {p.name:30s}  {p.population:>10,}  "
                  f"{p.scale.peak_load_kw/1000:>10,.0f}  {p.climate.name:>25s}")
        print(f"\nAvailable climate zones: {', '.join(CLIMATE_ZONES.keys())}")
        return
    elif args.municipal:
        import dataclasses
        profile = MUNICIPAL_PROFILES[args.municipal]
        if args.climate:
            profile = dataclasses.replace(profile, climate=CLIMATE_ZONES[args.climate])
        if args.project_years:
            print(f"\n{'='*75}")
            print(f"  MUNICIPAL GROWTH PROJECTION — {profile.name}")
            print(f"  Climate: {profile.climate.name} | Base year: {args.base_year}")
            print(f"  Projection years: {[args.base_year + y for y in args.project_years]}")
            print(f"{'='*75}")
            projection = GrowthProjection(profile, base_year=args.base_year)
            projection.run(years=args.project_years, sim_hours=hours,
                          dt_hours=args.dt)
            projection.print_summary()
            if not args.no_plot:
                safe = args.municipal.replace(" ", "_")
                plot_projection(projection,
                               save_path=f"powerplan_municipal_{safe}_projection.png")
        else:
            config = build_municipal_config(profile, year_offset=0,
                                            base_year=args.base_year)
            result = run_single(config, hours=hours, dt=args.dt,
                      plot=not args.no_plot, days_to_plot=args.plot_days)
            if args.resilience:
                res = ResilienceMetrics.compute(result["dispatcher"])
                print(f"\n  ── Resilience Metrics ──")
                print(f"  LOLP:                 {res.lolp:>13.4f}")
                print(f"  LOLE:                 {res.lole_hours:>13.1f} hours")
                print(f"  Energy Not Served:    {res.ens_kwh:>13,.0f} kWh")
                print(f"  Reserve Margin:       {res.system_reserve_margin:>13.1%}")
    elif args.custom:
        interactive_custom()
    elif args.scenarios:
        scale_key = args.scale or "community"
        # Accept preset names that aren't raw scale keys
        if scale_key not in SCALES:
            # Map preset names to their underlying scale
            scale_map = {"metro": "metropolitan", "metro_exotic": "metropolitan",
                        "metro_antimatter": "metropolitan",
                        "district_exotic": "district",
                        "district_antimatter": "district",
                        "community_fusion": "community"}
            scale_key = scale_map.get(scale_key, "community")
        print(f"\n{'='*70}")
        print(f"  SCENARIO GENERATION — {args.scenarios} configs, "
              f"{args.failures} failures each")
        print(f"  Scale: {scale_key} | Tier: {args.tier} | Seed: {args.scenario_seed}")
        print(f"  Simulation: {hours}h at {args.dt}h steps")
        print(f"{'='*70}")
        runner = ScenarioRunner(
            scale=scale_key, tier=args.tier, seed=args.scenario_seed,
            hours=hours, dt_hours=args.dt,
        )
        runner.run(n_configs=args.scenarios, n_failures=args.failures,
                   include_baseline=True)
        runner.print_summary()
        runner.plot_results()
        if args.scenarios >= 3:
            runner.plot_heatmap()
    elif args.compare:
        run_comparison(hours=hours, dt=args.dt, exotic=args.exotic)
    elif args.scale:
        config = all_presets[args.scale]()
        result = run_single(config, hours=hours, dt=args.dt,
                  plot=not args.no_plot, days_to_plot=args.plot_days)
        if args.resilience:
            res = ResilienceMetrics.compute(result["dispatcher"])
            print(f"\n  ── Resilience Metrics ──")
            print(f"  LOLP:                 {res.lolp:>13.4f}")
            print(f"  LOLE:                 {res.lole_hours:>13.1f} hours")
            print(f"  Energy Not Served:    {res.ens_kwh:>13,.0f} kWh")
            print(f"  Reserve Margin:       {res.system_reserve_margin:>13.1%}")
            print(f"  Min Reserve Margin:   {res.min_reserve_margin:>13.1%}")
            print(f"  Max Deficit:          {res.max_deficit_kw:>13,.0f} kW")
            print(f"  Max Deficit Duration: {res.max_deficit_duration_hours:>13.1f} hours")
            print(f"  Avg Recovery Time:    {res.recovery_time_avg:>13.1f} hours")
            print(f"  Max Recovery Time:    {res.recovery_time_max:>13.1f} hours")
            print(f"  Min Aggregate SOC:    {res.min_aggregate_soc:>13.1%}")
            print(f"  Storage Depletion:    {res.storage_depletion_hours:>13.1f} hours")
    elif args.exotic:
        for key in EXOTIC_PRESETS:
            config = EXOTIC_PRESETS[key]()
            run_single(config, hours=hours, dt=args.dt,
                      plot=not args.no_plot, days_to_plot=args.plot_days)
    else:
        for key in PRESETS:
            config = PRESETS[key]()
            run_single(config, hours=hours, dt=args.dt,
                      plot=not args.no_plot, days_to_plot=args.plot_days)


if __name__ == "__main__":
    main()
