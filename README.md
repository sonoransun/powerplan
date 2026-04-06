# PowerPlan -- Heterogeneous Power System Simulator

Simulates arbitrary configurations of energy sources, storage technologies, and solid-state power controllers from home scale to metropolitan deployments. Includes speculative technologies (micro-fusion, antimatter), real-world municipal infrastructure models with growth projections, programmatic scenario generation, and resilience testing with failure injection.

## Features

- **7 energy source models**: Solar PV, Wind, Micro-Hydro, Geothermal, Natural Gas (CCGT/Peaker), Micro-Fusion (D-T/p-B11), Antimatter Reactor (with target atom electron production)
- **7 storage technologies**: Li-ion (NMC/LFP), Sodium Solid-State, Flow Battery, Flywheel, Hydrogen Fuel Cell, Graphene Supercapacitor, SMES
- **8 power controllers**: SiC, GaN, MPPT, Bidirectional Inverter, H2 Controller, Fusion, Cryogenic, Antimatter (triple-pathway)
- **5 deployment scales**: Home (12 kW) to Metropolitan (500 MW)
- **5 municipal archetypes**: Small Town, College Town, Major Metro, Sunbelt Boomtown, Rust Belt Industrial
- **6 climate zones**: Hot Arid, Hot Humid, Temperate, Cold Continental, Marine Mild, Industrial North
- **Growth projections**: Multi-year fossil-to-renewable transition modeling with technology cost curves
- **Scenario generation**: Programmatic random configuration builder with failure injection and resilience metrics (LOLP, LOLE, ENS)
- **Rich visualization**: 9-panel simulation dashboard, system architecture diagram, Sankey-style energy flow, radar comparison charts, projection trend plots, resilience heatmaps
- **Colorblind-safe palette**: All visualizations use Wong 2011 adapted colors

## Installation

```bash
git clone <repository-url>
cd powerplan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10

## Quick Start

```bash
# Run a single scale preset
python run_simulation.py --scale community --days 30

# Compare all conventional presets
python run_simulation.py --compare --days 14

# Run exotic tech (fusion + antimatter)
python run_simulation.py --scale district_antimatter --days 30

# Municipal infrastructure with 25-year projection
python run_simulation.py --municipal college_town --project-years 0 5 10 15 20 25

# Random scenario generation with failure injection
python run_simulation.py --scenarios 10 --failures 3 --scale district --tier exotic
```

## Architecture

```
powerplan/
  __init__.py          Package root
  profiles.py          Load profiles and deployment scales
  sources.py           Energy source models (7 types)
  storage.py           Storage technology models (7 types)
  controllers.py       Solid-state power controller models (8 types)
  grid.py              Dispatch engine and grid configuration
  visualize.py         Visualization (dashboard, diagram, flow, comparison, projection)
  styles.py            Visual design constants (colorblind-safe palette)
  scenarios.py         Scenario generation, failure injection, resilience metrics
  municipal.py         Municipal infrastructure and growth projections
run_simulation.py      CLI entry point
```

### Dispatch Engine

The `EnergyDispatcher` processes each hourly time step with a 5-priority dispatch:

1. Renewable/fossil generation supplies load directly
2. Excess generation charges storage (prioritized by efficiency)
3. Remaining demand discharged from storage (prioritized by efficiency)
4. Grid import/export if interconnected
5. Curtailment of excess / unmet demand tracking

## Technology Reference

### Energy Sources

| Source | Type | Capacity Factor | Cost ($/kW) | Key Feature |
|--------|------|----------------|-------------|-------------|
| Solar PV | Renewable | 15-25% | 750-1,200 | Latitude/tracking/tilt modeling |
| Wind Turbine | Renewable | ~30% | 1,300-3,000 | Cubic power curve, hub height |
| Micro-Hydro | Renewable | ~55% | 4,000 | Seasonal flow, run-of-river |
| Geothermal | Renewable | ~92% | 6,000 | Baseload, binary cycle |
| Natural Gas | Fossil | 12-55% | 700-1,100 | CCGT (baseload) or Peaker (peak hours) |
| Micro-Fusion | Speculative | ~88-92% | 9,000-27,000 | D-T or p-B11, plasma stability |
| Antimatter | Speculative | ~76-92% | 50,000-100,000 | 3 conversion pathways, target atoms |

### Storage Technologies

| Technology | Round-Trip Eff | Cycle Life | Cost ($/kWh) | Key Feature |
|------------|---------------|------------|-------------|-------------|
| Li-ion NMC | ~92% | 3,000 | 180 | General purpose |
| Li-ion LFP | ~88% | 6,000 | 150 | Long cycle life |
| Na Solid-State | ~84% | 8,000 | 120 | Low cost, wide temp |
| Flow Battery | ~70% | 20,000 | 250 | Decoupled power/energy |
| Flywheel | ~86% | 500,000 | 5,000 | MW-class power, fast response |
| H2 Fuel Cell | ~38% | 10,000 | 35 | Seasonal storage |
| Graphene Supercap | ~95% | 1,000,000 | 8,000 | Sub-ms response, unlimited cycles |
| SMES | ~95% | 500,000 | 50,000 | Sub-ms, superconducting coil |

### Power Controllers

| Controller | Semiconductor | Peak Efficiency | Topology |
|------------|--------------|-----------------|----------|
| SiC Converter | Silicon Carbide | ~98.5% | Full bridge |
| GaN Converter | Gallium Nitride | ~99% | Half bridge |
| MPPT | -- | ~97.5% | Boost (solar tracking) |
| Bidirectional | -- | ~97.5% | H-bridge (grid-tied) |
| H2 Controller | -- | ~96% | Interleaved boost |
| Fusion | -- | ~97% | Multi-stage (Brayton/direct) |
| Cryogenic | -- | ~95% | Resonant (HTS magnet) |
| Antimatter | -- | ~96% | Triple pathway (MHD/thermal/electron) |

## Deployment Scales

| Scale | Endpoints | Peak Load | Annual Consumption |
|-------|-----------|-----------|-------------------|
| Home | 1 | 12 kW | 10.5 MWh |
| Neighborhood | 25 | 250 kW | 220 MWh |
| Community | 500 | 2 MW | 5 GWh |
| District | 8,000 | 25 MW | 80 GWh |
| Metropolitan | 200,000 | 500 MW | 2 TWh |

## Municipal Infrastructure Models

Real-world municipal archetypes with fossil/renewable generation mixes, climate-specific load profiles, and energy transition modeling:

| Preset | Pop | Peak | Climate | Current Mix | Target |
|--------|-----|------|---------|-------------|--------|
| `small_town` | 5K | 15 MW | Cold Continental | 20 MW gas + 2.5 MW renewable | 80% by 2045 |
| `college_town` | 100K | 300 MW | Temperate | 250 MW gas + 120 MW renewable | 100% by 2040 |
| `major_metro` | 1M | 3 GW | Temperate | 2 GW gas + 1.3 GW renewable | 100% by 2040 |
| `sunbelt_boom` | 250K | 800 MW | Hot Arid | 600 MW gas + 300 MW solar | 100% by 2045 |
| `rust_belt` | 150K | 400 MW | Industrial North | 400 MW gas + 70 MW renewable | 80% by 2050 |

Growth projections model: fossil retirement, renewable buildout, storage scaling, EV adoption, data center growth, heat pump electrification, technology cost curves.

## Scenario Generation & Resilience Testing

```bash
# Generate 10 random configs, test each with 3 failure scenarios
python run_simulation.py --scenarios 10 --failures 3 --scale district --tier exotic
```

**Failure types**: source trip, storage fault, weather crisis, grid disconnect, demand surge, simultaneous multi-component failure.

**Resilience metrics**: LOLP (Loss of Load Probability), LOLE (Loss of Load Expectation), ENS (Energy Not Served), recovery time, storage depletion, reserve margins.

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--scale PRESET` | Run a preset (home, community, district, metro, ...) | -- |
| `--days N` | Simulation duration in days | 365 |
| `--dt HOURS` | Time step in hours | 1.0 |
| `--compare` | Compare all conventional presets | -- |
| `--exotic` | Include exotic technologies | -- |
| `--custom` | Interactive configuration builder | -- |
| `--municipal PRESET` | Run a municipal infrastructure preset | -- |
| `--project-years Y...` | Growth projection year offsets | -- |
| `--climate ZONE` | Override climate zone | -- |
| `--base-year YEAR` | Base year for projection | 2025 |
| `--list-municipal` | List available municipal presets | -- |
| `--scenarios N` | Generate N random configurations | -- |
| `--failures M` | Failure scenarios per config | 3 |
| `--resilience` | Compute extended resilience metrics | -- |
| `--tier TIER` | Technology tier (conventional/exotic/antimatter) | conventional |
| `--scenario-seed S` | Random seed for reproducibility | 42 |
| `--no-plot` | Skip plot generation | -- |

## Visualization Outputs

Each simulation run generates three visualizations:

- **Simulation Dashboard** (`powerplan_<name>.png`): 9-panel view with power balance, source breakdown, storage SOC, efficiency trends, energy flow waterfall, storage health, and visual metric gauges
- **System Architecture** (`powerplan_<name>_diagram.png`): Schematic showing physical components (sources, controllers, bus, storage, load) with flow arrows proportional to power
- **Energy Flow** (`powerplan_<name>_flow.png`): Sankey-style diagram showing kWh flowing from sources to destinations (direct use, storage, curtailment, losses)

Additional outputs from specific modes:
- **Scale Comparison** (`powerplan_comparison.png`): Radar chart + 6 bar chart panels
- **Growth Projection** (`powerplan_municipal_<name>_projection.png`): 6-panel trend showing capacity evolution, LCOE, renewable penetration, emissions
- **Resilience Analysis** (`resilience_analysis.png`): LOLP/ENS distributions, component impact, failure type impact
- **Component Heatmap** (`component_heatmap.png`): Source x Storage LOLP interaction matrix
