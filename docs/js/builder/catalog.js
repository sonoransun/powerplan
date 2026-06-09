/**
 * catalog.js — the builder's technology catalog.
 *
 * Hand-ported from the Python source of truth:
 *   powerplan/profiles.py     → SCALES_INFO
 *   powerplan/sources.py      → source constructor params/defaults
 *   powerplan/storage.py      → storage constructor params/defaults
 *   powerplan/controllers.py  → controller constructor params/defaults
 *   powerplan/scenarios.py    → TECH_TIERS, SOURCE_SPECS, STORAGE_SPECS,
 *                               CONTROLLER_REQUIREMENTS (ranges ported EXACTLY)
 *
 * Slider bounds = the per-scale spec range extended ±50%; the un-extended
 * spec range is surfaced as the "typical" band.
 */

// ── deployment scales (powerplan/profiles.py SCALES) ──────────────────

export const SCALE_ORDER = ["home", "neighborhood", "community", "district", "metropolitan"];

export const SCALES_INFO = {
  home: {
    key: "home", label: "Single Home", peak_load_kw: 12.0,
    annual_consumption_kwh: 10_500, num_endpoints: 1,
    description: "Single residential dwelling, 3-4 bedroom",
  },
  neighborhood: {
    key: "neighborhood", label: "Neighborhood", peak_load_kw: 250.0,
    annual_consumption_kwh: 220_000, num_endpoints: 25,
    description: "25-home residential neighborhood microgrid",
  },
  community: {
    key: "community", label: "Community", peak_load_kw: 2_000.0,
    annual_consumption_kwh: 5_000_000, num_endpoints: 500,
    description: "Community of ~500 homes + small commercial",
  },
  district: {
    key: "district", label: "District", peak_load_kw: 25_000.0,
    annual_consumption_kwh: 80_000_000, num_endpoints: 8_000,
    description: "Urban district with mixed residential/commercial/light industrial",
  },
  metropolitan: {
    key: "metropolitan", label: "Metropolitan", peak_load_kw: 500_000.0,
    annual_consumption_kwh: 2_000_000_000, num_endpoints: 200_000,
    description: "Large metro area with full heterogeneous demand mix",
  },
};

// ── tech tiers (scenarios.py TECH_TIERS, exact) ───────────────────────

export const TECH_TIERS = {
  conventional: {
    sources: ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas"],
    storage: ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
      "flywheel", "hydrogen_fuel_cell"],
  },
  exotic: {
    sources: ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas",
      "micro_fusion"],
    storage: ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
      "flywheel", "hydrogen_fuel_cell", "graphene_supercap", "smes"],
  },
  antimatter: {
    sources: ["solar_pv", "wind", "micro_hydro", "geothermal", "natural_gas",
      "micro_fusion", "antimatter"],
    storage: ["lithium_ion", "sodium_solid_state", "liquid_electrolyte",
      "flywheel", "hydrogen_fuel_cell", "graphene_supercap", "smes"],
  },
};

// ── per-scale spec tables (scenarios.py, exact) ───────────────────────
// Sources: kw range + selection weight. Absent scale = not modeled there.

const SOURCE_SPECS = {
  solar_pv: {
    home: { kw: [3, 12], w: 0.8 },
    neighborhood: { kw: [50, 200], w: 0.7 },
    community: { kw: [300, 1500], w: 0.6 },
    district: { kw: [3000, 20000], w: 0.5 },
    metropolitan: { kw: [50000, 300000], w: 0.4 },
  },
  wind: {
    home: { kw: [0.5, 3], w: 0.4 },
    neighborhood: { kw: [10, 50], w: 0.5 },
    community: { kw: [200, 800], w: 0.5 },
    district: { kw: [2000, 15000], w: 0.5 },
    metropolitan: { kw: [8000, 80000], w: 0.4 },
  },
  micro_hydro: {
    community: { kw: [20, 100], w: 0.3 },
    district: { kw: [200, 1000], w: 0.3 },
    metropolitan: { kw: [2000, 15000], w: 0.2 },
  },
  geothermal: {
    district: { kw: [1000, 5000], w: 0.25 },
    metropolitan: { kw: [20000, 80000], w: 0.3 },
  },
  micro_fusion: {
    community: { kw: [500, 2000], w: 0.2 },
    district: { kw: [5000, 20000], w: 0.3 },
    metropolitan: { kw: [30000, 150000], w: 0.25 },
  },
  natural_gas: {
    community: { kw: [500, 3000], w: 0.3 },
    district: { kw: [5000, 50000], w: 0.3 },
    metropolitan: { kw: [50000, 500000], w: 0.25 },
  },
  antimatter: {
    district: { kw: [10000, 30000], w: 0.15 },
    metropolitan: { kw: [50000, 200000], w: 0.15 },
  },
};

// Storage: kwh range + power-to-energy ratio "pr" (max_power = kwh × pr).

const STORAGE_SPECS = {
  lithium_ion: {
    home: { kwh: [5, 20], pr: 0.4 },
    neighborhood: { kwh: [30, 150], pr: 0.4 },
    community: { kwh: [200, 1000], pr: 0.5 },
    district: { kwh: [2000, 15000], pr: 0.5 },
    metropolitan: { kwh: [50000, 300000], pr: 0.5 },
  },
  sodium_solid_state: {
    home: { kwh: [5, 15], pr: 0.3 },
    neighborhood: { kwh: [20, 80], pr: 0.3 },
    community: { kwh: [100, 600], pr: 0.3 },
    district: { kwh: [1000, 8000], pr: 0.3 },
    metropolitan: { kwh: [20000, 100000], pr: 0.3 },
  },
  liquid_electrolyte: {
    community: { kwh: [200, 1000], pr: 0.2 },
    district: { kwh: [2000, 20000], pr: 0.2 },
    metropolitan: { kwh: [50000, 400000], pr: 0.2 },
  },
  flywheel: {
    neighborhood: { kwh: [1, 5], pr: 15.0 },
    community: { kwh: [3, 15], pr: 15.0 },
    district: { kwh: [20, 100], pr: 10.0 },
    metropolitan: { kwh: [50, 500], pr: 10.0 },
  },
  hydrogen_fuel_cell: {
    community: { kwh: [1000, 5000], pr: 0.02 },
    district: { kwh: [10000, 200000], pr: 0.02 },
    metropolitan: { kwh: [100000, 5000000], pr: 0.015 },
  },
  graphene_supercap: {
    community: { kwh: [10, 50], pr: 150.0 },
    district: { kwh: [50, 300], pr: 200.0 },
    metropolitan: { kwh: [200, 5000], pr: 200.0 },
  },
  smes: {
    district: { kwh: [10, 80], pr: 2000.0 },
    metropolitan: { kwh: [30, 500], pr: 2000.0 },
  },
};

export { SOURCE_SPECS, STORAGE_SPECS };

// Which controllers each component type requires (scenarios.py, exact).
export const CONTROLLER_REQUIREMENTS = {
  solar_pv: ["mppt"],
  natural_gas: [],
  micro_fusion: ["fusion"],
  antimatter: ["antimatter"],
  hydrogen_fuel_cell: ["hydrogen"],
  smes: ["cryogenic"],
};

export const H2_KWH_PER_KG = 33.3; // LHV — storage.py HydrogenFuelCell

// ── helpers ───────────────────────────────────────────────────────────

function availability(specTable, key) {
  const out = {};
  for (const sk of SCALE_ORDER) out[sk] = specTable[key]?.[sk] ?? null;
  return out;
}

/** Round to 3 significant digits — keeps defaults readable. */
export function nice(x) {
  if (!isFinite(x) || x === 0) return 0;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(x))) - 2);
  return Math.round(x / mag) * mag;
}

function niceStep(range) {
  if (!isFinite(range) || range <= 0) return 1;
  const raw = range / 200;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
}

// ── parameter shorthands ──────────────────────────────────────────────

const UNITS = { name: "units", label: "Units", kind: "int", default: 1, min: 1, max: 60 };
const r = (name, label, def, min, max, opts = {}) =>
  ({ name, label, kind: "range", default: def, min, max,
     step: opts.step ?? niceStep(max - min), ...opts });

// ── SOURCE_TYPES (constructors in powerplan/sources.py) ───────────────

export const SOURCE_TYPES = [
  {
    key: "solar_pv", label: "Solar PV", className: "SolarPV",
    speculative: false, renewable: true,
    availability: availability(SOURCE_SPECS, "solar_pv"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      { name: "tilt_deg", label: "Panel tilt", kind: "range", unit: "°",
        default: null, nullText: "auto ≡ latitude", min: 0, max: 60, step: 1 },
      { name: "tracking", label: "Sun tracking", kind: "bool", default: false },
      r("latitude", "Latitude", 35, 0, 60, { unit: "°", step: 1 }),
      UNITS,
    ],
  },
  {
    key: "wind", label: "Wind Turbine", className: "WindTurbine",
    speculative: false, renewable: true,
    availability: availability(SOURCE_SPECS, "wind"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      r("hub_height_m", "Hub height", 30, 10, 140, { unit: "m", step: 5 }),
      r("mean_wind_ms", "Mean wind speed", 6, 3, 12, { unit: "m/s", step: 0.5 }),
      r("cut_in_ms", "Cut-in speed", 3, 1, 6, { unit: "m/s", step: 0.5, advanced: true }),
      r("rated_wind_ms", "Rated wind speed", 12, 8, 18, { unit: "m/s", step: 0.5, advanced: true }),
      r("cut_out_ms", "Cut-out speed", 25, 18, 35, { unit: "m/s", step: 0.5, advanced: true }),
      UNITS,
    ],
  },
  {
    key: "micro_hydro", label: "Micro Hydro", className: "MicroHydro",
    speculative: false, renewable: true,
    availability: availability(SOURCE_SPECS, "micro_hydro"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      r("head_m", "Head", 10, 2, 100, { unit: "m", step: 1 }),
      r("design_flow_m3s", "Design flow", 0.2, 0.01, 10, { unit: "m³/s", step: 0.01, advanced: true }),
      UNITS,
    ],
  },
  {
    key: "geothermal", label: "Geothermal", className: "Geothermal",
    speculative: false, renewable: true,
    availability: availability(SOURCE_SPECS, "geothermal"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      r("well_temp_c", "Well temperature", 150, 100, 250, { unit: "°C", step: 5 }),
      UNITS,
    ],
  },
  {
    key: "natural_gas", label: "Gas Turbine", className: "NaturalGasTurbine",
    speculative: false, renewable: false,
    availability: availability(SOURCE_SPECS, "natural_gas"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      { name: "plant_type", label: "Plant type", kind: "enum", default: "ccgt",
        options: [{ value: "ccgt", label: "CCGT (baseload)" },
                  { value: "peaker", label: "Peaker (14:00–21:00)" }] },
      r("gas_price_per_mmbtu", "Gas price", 3.5, 1, 15, { unit: "$/MMBTU", step: 0.25, advanced: true }),
      { name: "heat_rate_btu_kwh", label: "Heat rate", kind: "range", unit: "BTU/kWh",
        default: null, nullText: "auto: 6 800 CCGT · 9 500 peaker",
        min: 5500, max: 12000, step: 100, advanced: true },
      UNITS,
    ],
  },
  {
    key: "micro_fusion", label: "Micro Fusion", className: "MicroFusionReactor",
    speculative: true, renewable: true,
    availability: availability(SOURCE_SPECS, "micro_fusion"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      { name: "fuel_cycle", label: "Fuel cycle", kind: "enum", default: "dt",
        options: [{ value: "dt", label: "D-T (thermal, η≈0.42)" },
                  { value: "pb11", label: "p-B11 (direct, η≈0.75)" }] },
      r("q_engineering", "Q engineering", 10, 1, 20,
        { step: 0.5, inert: "reported but unused in the power balance" }),
      { name: "thermal_efficiency", label: "Conversion η", kind: "range",
        default: null, nullText: "auto: 0.42 D-T · 0.75 p-B11",
        min: 0.2, max: 0.9, step: 0.01, advanced: true },
      { name: "confinement", label: "Confinement", kind: "enum", default: "compact_tokamak",
        advanced: true,
        options: [{ value: "compact_tokamak", label: "Compact tokamak" },
                  { value: "field_reversed", label: "Field-reversed config" },
                  { value: "magnetized_target", label: "Magnetized target" },
                  { value: "inertial_electrostatic", label: "Inertial electrostatic" }] },
      UNITS,
    ],
  },
  {
    key: "antimatter", label: "Antimatter Reactor", className: "AntimatterReactor",
    speculative: true, renewable: true,
    availability: availability(SOURCE_SPECS, "antimatter"),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", scaleSpec: true },
      { name: "target_atom", label: "Target atom", kind: "enum", default: "none",
        options: [{ value: "none", label: "None (free annihilation)" },
                  { value: "xenon", label: "Xenon-131 (clean e⁻ harvest)" },
                  { value: "lead", label: "Lead-208 (dense, high Auger)" },
                  { value: "uranium", label: "Uranium-238 (catalyzed fission ×1.107)" }] },
      r("electron_collection_efficiency", "e⁻ collection η", 0.45, 0.2, 0.6, { step: 0.01, advanced: true }),
      r("mhd_efficiency", "MHD η", 0.70, 0.4, 0.85, { step: 0.01, advanced: true }),
      r("gamma_thermal_efficiency", "γ-thermal η", 0.35, 0.2, 0.45, { step: 0.01, advanced: true }),
      r("magnetic_field_tesla", "Trap field", 6, 2, 10, { unit: "T", step: 0.5, advanced: true }),
      { name: "graphene_electrode_layers", label: "Graphene layers", kind: "int",
        default: 100, min: 20, max: 400, advanced: true },
      r("fuel_reservoir_ug", "Antiproton reservoir", 5_500_000, 100_000, 100_000_000,
        { unit: "µg", step: 100_000, advanced: true }),
      UNITS,
    ],
  },
];

// ── STORAGE_TYPES (constructors in powerplan/storage.py) ──────────────

export const STORAGE_TYPES = [
  {
    key: "lithium_ion", label: "Li-ion Battery", className: "LithiumIonBattery",
    speculative: false,
    availability: availability(STORAGE_SPECS, "lithium_ion"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      { name: "chemistry", label: "Chemistry", kind: "enum", default: "nmc",
        options: [{ value: "nmc", label: "NMC (96% η, 3 000 cyc)" },
                  { value: "lfp", label: "LFP (94% η, 6 000 cyc)" }] },
      UNITS,
    ],
  },
  {
    key: "sodium_solid_state", label: "Na Solid-State", className: "SodiumSolidStateBattery",
    speculative: false,
    availability: availability(STORAGE_SPECS, "sodium_solid_state"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      UNITS,
    ],
  },
  {
    key: "liquid_electrolyte", label: "Flow Battery", className: "LiquidElectrolyteBattery",
    speculative: false,
    availability: availability(STORAGE_SPECS, "liquid_electrolyte"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      { name: "chemistry", label: "Chemistry", kind: "enum", default: "vanadium",
        options: [{ value: "vanadium", label: "Vanadium redox" },
                  { value: "zinc_bromine", label: "Zinc-bromine" }] },
      UNITS,
    ],
  },
  {
    key: "flywheel", label: "Flywheel", className: "FlywheelStorage",
    speculative: false,
    availability: availability(STORAGE_SPECS, "flywheel"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      UNITS,
    ],
  },
  {
    // SPECIAL: parametrized by tank/electrolyzer/fuel-cell, not capacity_kwh.
    key: "hydrogen_fuel_cell", label: "H₂ Fuel Cell", className: "HydrogenFuelCell",
    speculative: false, h2: true,
    availability: availability(STORAGE_SPECS, "hydrogen_fuel_cell"),
    params: [
      { name: "h2_tank_kg", label: "H₂ tank", kind: "range", unit: "kg", scaleSpec: true },
      { name: "electrolyzer_kw", label: "Electrolyzer (charge)", kind: "range", unit: "kW", prDerived: true },
      { name: "fuel_cell_kw", label: "Fuel cell (discharge)", kind: "range", unit: "kW", prDerived: true },
      UNITS,
    ],
  },
  {
    key: "graphene_supercap", label: "Graphene Supercap", className: "GrapheneSupercapacitor",
    speculative: false,
    availability: availability(STORAGE_SPECS, "graphene_supercap"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      r("cell_voltage", "Cell voltage", 3.8, 3.0, 4.0, { unit: "V", step: 0.1, advanced: true }),
      r("esr_mohm", "ESR", 0.5, 0.1, 2.0, { unit: "mΩ", step: 0.1, advanced: true }),
      UNITS,
    ],
  },
  {
    key: "smes", label: "SMES", className: "SMES",
    speculative: false,
    availability: availability(STORAGE_SPECS, "smes"),
    params: [
      { name: "capacity_kwh", label: "Capacity", kind: "range", unit: "kWh", scaleSpec: true },
      { name: "max_power_kw", label: "Max power", kind: "range", unit: "kW", prDerived: true },
      r("inductance_h", "Coil inductance", 10, 1, 100, { unit: "H", step: 1, advanced: true }),
      r("operating_temp_k", "Operating temp", 30, 4, 77, { unit: "K", step: 1, advanced: true }),
      r("cryo_power_fraction", "Cryo parasitic", 0.03, 0.01, 0.05, { step: 0.005, advanced: true }),
      UNITS,
    ],
  },
];

// ── CONTROLLER_TYPES (constructors in powerplan/controllers.py) ───────
// No per-scale spec table; rated_kw bounds derive from the scale peak.

const allScales = () => availability({}, "_none_"); // all null → available, no band

export const CONTROLLER_TYPES = [
  {
    key: "sic", label: "SiC Converter", className: "SiCConverter", speculative: false,
    availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      r("switching_freq_khz", "Switching freq", 100, 20, 500, { unit: "kHz", step: 10, advanced: true }),
    ],
  },
  {
    key: "gan", label: "GaN Converter", className: "GaNConverter", speculative: false,
    availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      r("switching_freq_khz", "Switching freq", 500, 100, 2000, { unit: "kHz", step: 50, advanced: true }),
    ],
  },
  {
    key: "mppt", label: "MPPT Controller", className: "MPPTController", speculative: false,
    availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      { name: "algorithm", label: "Algorithm", kind: "enum", default: "perturb_observe",
        advanced: true,
        options: [{ value: "perturb_observe", label: "Perturb & observe" },
                  { value: "incremental_conductance", label: "Incremental conductance" }] },
    ],
  },
  {
    key: "bidirectional", label: "Bidir Inverter", className: "BidirectionalInverter",
    speculative: false, availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      { name: "phases", label: "Phases", kind: "enum", default: 1, numeric: true,
        options: [{ value: 1, label: "Single-phase" }, { value: 3, label: "Three-phase" }] },
      { name: "grid_voltage", label: "Grid voltage", kind: "enum", default: 240, numeric: true,
        advanced: true,
        options: [{ value: 240, label: "240 V" }, { value: 480, label: "480 V" }] },
    ],
  },
  {
    key: "hydrogen", label: "H₂ Controller", className: "HydrogenPowerController",
    speculative: false, availability: allScales(),
    params: [{ name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true }],
  },
  {
    key: "fusion", label: "Fusion Controller", className: "FusionPowerController",
    speculative: true, availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      { name: "conversion", label: "Conversion", kind: "enum", default: "brayton",
        options: [{ value: "brayton", label: "sCO₂ Brayton (thermal)" },
                  { value: "direct", label: "Direct (charged particles)" }] },
    ],
  },
  {
    key: "cryogenic", label: "Cryo Power Supply", className: "CryogenicPowerSupply",
    speculative: false, availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      r("cooling_stage_k", "Cooling stage", 30, 4, 77, { unit: "K", step: 1, advanced: true }),
    ],
  },
  {
    key: "antimatter", label: "Antimatter Controller", className: "AntimatterPowerController",
    speculative: true, availability: allScales(),
    params: [
      { name: "rated_kw", label: "Rated power", kind: "range", unit: "kW", ctrlSized: true },
      r("mhd_fraction", "MHD pathway share", 0.70, 0.3, 0.8, { step: 0.01 }),
      r("electron_fraction", "e⁻ pathway share", 0.0, 0, 0.3, { step: 0.01, advanced: true }),
      r("containment_parasitic_fraction", "Containment parasitic", 0.05, 0.02, 0.1,
        { step: 0.005, advanced: true }),
    ],
  },
];

// ── lookups ───────────────────────────────────────────────────────────

const LISTS = { sources: SOURCE_TYPES, storage: STORAGE_TYPES, controllers: CONTROLLER_TYPES };

export function typeDef(category, key) {
  return LISTS[category]?.find((t) => t.key === key) ?? null;
}
export function typeList(category) { return LISTS[category] ?? []; }

/** Python class name (as found in the data files) → registry key. */
export const CLASSNAME_TO_KEY = {};
for (const cat of Object.keys(LISTS)) {
  CLASSNAME_TO_KEY[cat] = {};
  for (const t of LISTS[cat]) CLASSNAME_TO_KEY[cat][t.className] = t.key;
}

// ── slider bounds & defaults ──────────────────────────────────────────

function specFor(category, typeKey, scaleKey) {
  const table = category === "sources" ? SOURCE_SPECS : STORAGE_SPECS;
  return table[typeKey]?.[scaleKey] ?? null;
}

/**
 * Bounds for one parameter at one scale.
 * Returns {min, max, step, typical:[lo,hi]|null} — typical is the exact
 * scenarios.py spec band before the ±50% extension.
 */
export function paramBounds(category, typeKey, param, scaleKey) {
  if (param.scaleSpec || param.prDerived) {
    const spec = specFor(category, typeKey, scaleKey);
    if (!spec) return null; // not modeled at this scale
    let [lo, hi] = category === "sources" ? spec.kw : spec.kwh;
    if (category === "storage") {
      if (typeKey === "hydrogen_fuel_cell") {
        if (param.name === "h2_tank_kg") { lo /= H2_KWH_PER_KG; hi /= H2_KWH_PER_KG; }
        else if (param.name === "electrolyzer_kw") { lo *= spec.pr * 1.2; hi *= spec.pr * 1.2; }
        else if (param.name === "fuel_cell_kw") { lo *= spec.pr; hi *= spec.pr; }
      } else if (param.name === "max_power_kw") {
        lo *= spec.pr; hi *= spec.pr;
      }
    }
    const min = nice(lo * 0.5), max = nice(hi * 1.5);
    return { min, max, step: niceStep(max - min), typical: [nice(lo), nice(hi)] };
  }
  if (param.ctrlSized) {
    const peak = SCALES_INFO[scaleKey].peak_load_kw;
    const min = nice(Math.max(peak * 0.005, 0.5)), max = nice(peak * 4);
    return { min, max, step: niceStep(max - min), typical: null };
  }
  if (param.kind === "range" || param.kind === "int") {
    return { min: param.min, max: param.max, step: param.step ?? 1, typical: null };
  }
  return null;
}

/** Concrete default params for a freshly added component at a scale. */
export function defaultParams(category, typeKey, scaleKey) {
  const def = typeDef(category, typeKey);
  const out = {};
  for (const p of def.params) {
    if (p.scaleSpec || p.prDerived || p.ctrlSized) {
      const b = paramBounds(category, typeKey, p, scaleKey);
      if (!b) { out[p.name] = 1; continue; }
      out[p.name] = b.typical
        ? nice(Math.sqrt(b.typical[0] * b.typical[1]))    // geometric mid of spec band
        : nice(SCALES_INFO[scaleKey].peak_load_kw * 0.25); // controllers: ¼ peak
    } else {
      out[p.name] = p.default ?? null;
    }
  }
  return out;
}

// ── headline capacities ───────────────────────────────────────────────

const u = (c) => Number(c.params?.units) || 1;

export function sourceKw(comp) { return (Number(comp.params?.rated_kw) || 0) * u(comp); }

export function storageEnergyKwh(comp) {
  if (comp.type === "hydrogen_fuel_cell")
    return (Number(comp.params?.h2_tank_kg) || 0) * H2_KWH_PER_KG * u(comp);
  return (Number(comp.params?.capacity_kwh) || 0) * u(comp);
}

export function storagePowerKw(comp) {
  if (comp.type === "hydrogen_fuel_cell")
    return (Number(comp.params?.fuel_cell_kw) || 0) * u(comp);
  return (Number(comp.params?.max_power_kw) || 0) * u(comp);
}

export function totalGenerationKw(state) {
  return (state.sources ?? []).reduce((s, c) => s + sourceKw(c), 0);
}
export function totalStoragePowerKw(state) {
  return (state.storage ?? []).reduce((s, c) => s + storagePowerKw(c), 0);
}

// ── controller suggestions (sized like scenarios._make_controllers) ───

/**
 * Suggest missing required controllers for the current build.
 * Sizing mirrors scenarios.py: mppt→Σ solar kW · fusion→0.5×Σ fusion kW ·
 * antimatter→0.5×Σ antimatter kW · cryogenic→0.02×(Σ gen + Σ storage power) ·
 * hydrogen→0.2×Σ storage power.
 */
export function suggestController(state) {
  const have = new Set((state.controllers ?? []).map((c) => c.type));
  const needed = new Map(); // ctrlKey → [component labels]
  for (const cat of ["sources", "storage"]) {
    for (const comp of state[cat] ?? []) {
      for (const ck of CONTROLLER_REQUIREMENTS[comp.type] ?? []) {
        if (!needed.has(ck)) needed.set(ck, []);
        needed.get(ck).push(typeDef(cat, comp.type)?.label ?? comp.type);
      }
    }
  }
  const sumKw = (types) => (state.sources ?? [])
    .filter((c) => types.includes(c.type)).reduce((s, c) => s + sourceKw(c), 0);
  const storPower = totalStoragePowerKw(state);
  const total = totalGenerationKw(state) + storPower;

  const suggestions = [];
  for (const [ck, requiredBy] of needed) {
    if (have.has(ck)) continue;
    let kw = 1;
    if (ck === "mppt") kw = sumKw(["solar_pv"]);
    else if (ck === "fusion") kw = 0.5 * sumKw(["micro_fusion"]);
    else if (ck === "antimatter") kw = 0.5 * sumKw(["antimatter"]);
    else if (ck === "cryogenic") kw = 0.02 * total;
    else if (ck === "hydrogen") kw = 0.2 * storPower;
    kw = Math.max(nice(kw), 1);
    suggestions.push({
      type: ck, label: typeDef("controllers", ck).label,
      rated_kw: kw, requiredBy: [...new Set(requiredBy)],
    });
  }
  return suggestions;
}
