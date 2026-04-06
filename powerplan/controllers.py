"""
Solid-state power controller models.

Models high-efficiency power electronics that manage energy flow between
sources, storage, loads, and grid interconnection points.

Includes:
- DC-DC converters (buck/boost)
- DC-AC inverters (grid-tied and island mode)
- MPPT controllers for renewables
- Bidirectional converters for storage
- Load-dependent efficiency curves based on SiC/GaN semiconductors
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ControllerState:
    """Snapshot of controller operating state."""
    input_kw: float
    output_kw: float
    efficiency: float
    temperature_c: float
    loading_pct: float
    mode: str  # "mppt", "charge", "discharge", "bypass", "idle"


class PowerController(ABC):
    """Base class for solid-state power controllers."""

    def __init__(self, name: str, rated_kw: float, topology: str = "full_bridge"):
        self.name = name
        self.rated_kw = rated_kw
        self.topology = topology
        self.temperature_c = 25.0
        self.cumulative_loss_kwh = 0.0
        self.operating_hours = 0.0

    @abstractmethod
    def efficiency_at_load(self, load_fraction: float) -> float:
        """Efficiency as function of loading [0, 1]."""

    def convert(self, input_kw: float, dt_hours: float,
                mode: str = "regulate") -> ControllerState:
        """Process power through the controller."""
        clamped_input = min(abs(input_kw), self.rated_kw)
        load_frac = clamped_input / self.rated_kw if self.rated_kw > 0 else 0

        eff = self.efficiency_at_load(load_frac)

        # Thermal derating above 80°C junction
        if self.temperature_c > 80:
            thermal_derate = max(0.5, 1.0 - (self.temperature_c - 80) / 40)
            clamped_input *= thermal_derate

        output_kw = clamped_input * eff
        loss_kw = clamped_input * (1 - eff)

        # Thermal model — junction temperature
        thermal_resistance = 0.5  # °C/W simplified
        ambient = 25.0
        self.temperature_c = ambient + loss_kw * 1000 * thermal_resistance * 0.001

        self.cumulative_loss_kwh += loss_kw * dt_hours
        self.operating_hours += dt_hours if clamped_input > 0 else 0

        return ControllerState(
            input_kw=clamped_input,
            output_kw=output_kw,
            efficiency=eff,
            temperature_c=self.temperature_c,
            loading_pct=load_frac * 100,
            mode=mode,
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "rated_kw": self.rated_kw,
            "topology": self.topology,
            "cumulative_loss_kwh": self.cumulative_loss_kwh,
            "operating_hours": self.operating_hours,
        }


class SiCConverter(PowerController):
    """
    Silicon Carbide (SiC) MOSFET based converter.
    State-of-the-art efficiency for medium-high power applications.
    Lower switching losses, higher temperature operation.
    """

    def __init__(self, rated_kw=50.0, switching_freq_khz=100.0, **kwargs):
        super().__init__(name="SiC Converter", rated_kw=rated_kw,
                        topology="full_bridge", **kwargs)
        self.switching_freq_khz = switching_freq_khz

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.01:
            return 0.0
        # SiC efficiency curve: peak ~98.5% at 40-60% load
        # Standby losses dominate at low load, conduction at high load
        standby = 0.005 / max(load_fraction, 0.01)  # Fixed losses
        conduction = 0.005 * load_fraction  # I²R losses
        switching = 0.003 * (self.switching_freq_khz / 100)  # Switching losses
        total_loss = standby + conduction + switching
        return max(0.80, min(0.985, 1.0 - total_loss))


class GaNConverter(PowerController):
    """
    Gallium Nitride (GaN) HEMT based converter.
    Highest efficiency at high frequency, compact form factor.
    Ideal for residential and small commercial.
    """

    def __init__(self, rated_kw=10.0, switching_freq_khz=500.0, **kwargs):
        super().__init__(name="GaN Converter", rated_kw=rated_kw,
                        topology="half_bridge", **kwargs)
        self.switching_freq_khz = switching_freq_khz

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.01:
            return 0.0
        # GaN: even lower switching losses, peak ~99% at optimal load
        standby = 0.003 / max(load_fraction, 0.01)
        conduction = 0.004 * load_fraction
        # GaN excels at high frequency — switching losses scale less
        switching = 0.001 * (self.switching_freq_khz / 500)
        total_loss = standby + conduction + switching
        return max(0.82, min(0.99, 1.0 - total_loss))


class MPPTController(PowerController):
    """
    Maximum Power Point Tracking controller for solar PV.
    Continuously adjusts operating voltage to extract maximum power.
    """

    def __init__(self, rated_kw=10.0, algorithm="perturb_observe", **kwargs):
        super().__init__(name="MPPT Controller", rated_kw=rated_kw,
                        topology="boost", **kwargs)
        self.algorithm = algorithm
        self.tracking_efficiency = 0.995  # MPPT tracking accuracy

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.02:
            return 0.0
        # MPPT converters optimized for wide input range
        base = 0.975 * self.tracking_efficiency
        standby = 0.004 / max(load_fraction, 0.02)
        conduction = 0.003 * load_fraction
        return max(0.85, min(base, base - standby - conduction))


class BidirectionalInverter(PowerController):
    """
    Grid-tied bidirectional inverter for storage systems.
    Handles both DC-AC (discharge) and AC-DC (charge) conversion.
    """

    def __init__(self, rated_kw=25.0, grid_voltage=240, phases=1, **kwargs):
        super().__init__(name="Bidir Inverter", rated_kw=rated_kw,
                        topology="h_bridge", **kwargs)
        self.grid_voltage = grid_voltage
        self.phases = phases

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.02:
            return 0.0
        # CEC weighted efficiency style curve
        standby = 0.008 / max(load_fraction, 0.02)
        conduction = 0.006 * load_fraction
        filter_loss = 0.003  # Output filter losses
        total_loss = standby + conduction + filter_loss
        return max(0.80, min(0.975, 1.0 - total_loss))


class HydrogenPowerController(PowerController):
    """
    Specialized controller for electrolyzer + fuel cell systems.
    Manages DC bus, ramp rates, and mode transitions.
    """

    def __init__(self, rated_kw=25.0, **kwargs):
        super().__init__(name="H2 Controller", rated_kw=rated_kw,
                        topology="interleaved_boost", **kwargs)
        self.ramp_rate_kw_per_sec = rated_kw * 0.1  # 10%/sec ramp limit
        self._current_output = 0.0

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.05:
            return 0.0
        # Balance-of-plant parasitic loads reduce system efficiency
        bop = 0.02  # Pumps, fans, controls
        standby = 0.006 / max(load_fraction, 0.05)
        conduction = 0.005 * load_fraction
        return max(0.78, min(0.96, 1.0 - standby - conduction - bop))


class FusionPowerController(PowerController):
    """
    Integrated power controller for compact fusion reactor systems.

    Manages the complete power conversion chain:
    1. Thermal → Electric: sCO2 Brayton turbine-generator (D-T)
       or direct electrostatic/magnetic conversion (p-B11)
    2. Magnet Power Supply: HTS superconducting magnet energization,
       persistent current switch, quench protection dump circuits
    3. Plasma Heating: RF systems (ICRH at ~50 MHz, ECRH at ~170 GHz),
       neutral beam injection power supplies
    4. Grid Interface: synchronous condenser mode, reactive power support

    Efficiency chain:
    - Brayton cycle: 42-45% thermal → electric
    - Direct conversion: 70-85% for charged particle beams
    - Internal bus losses: 1-2% (SiC-based DC bus)
    - Transformer/switchgear: 99.2% (HV transmission class)
    """

    def __init__(self, rated_kw=10_000.0, conversion="brayton", **kwargs):
        super().__init__(name="Fusion Controller", rated_kw=rated_kw,
                        topology="multi_stage", **kwargs)
        self.conversion = conversion
        # Brayton cycle has fixed parasitic loads (pumps, bearings, seals)
        self._parasitic_fraction = 0.03 if conversion == "brayton" else 0.02

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.05:
            return 0.0
        # Turbine-generator efficiency peaks at 70-90% load
        if self.conversion == "brayton":
            # sCO2 Brayton partial-load characteristic
            peak_eff = 0.97  # Electrical conversion at generator terminals
            part_load = 1.0 - 0.08 * (1.0 - load_fraction) ** 2  # Gradual dropoff
            parasitic = self._parasitic_fraction / max(load_fraction, 0.1)
        else:
            # Direct conversion — more linear response
            peak_eff = 0.98
            part_load = 1.0 - 0.04 * (1.0 - load_fraction) ** 2
            parasitic = self._parasitic_fraction / max(load_fraction, 0.1)
        return max(0.80, min(peak_eff, peak_eff * part_load - parasitic))


class CryogenicPowerSupply(PowerController):
    """
    Cryogenic power supply for HTS superconducting systems.

    Serves both SMES coils and fusion reactor magnets.
    Manages:
    - Cryo-cooler compressor drive (Gifford-McMahon or pulse tube)
    - Persistent current switch heater
    - Quench detection and dump resistor switching
    - Current lead thermal intercept stages
    """

    def __init__(self, rated_kw=500.0, cooling_stage_k=30.0, **kwargs):
        super().__init__(name="Cryo Power Supply", rated_kw=rated_kw,
                        topology="resonant_converter", **kwargs)
        self.cooling_stage_k = cooling_stage_k
        # Carnot efficiency penalty: lower temp = more work
        # Practical cryo-coolers achieve 10-30% of Carnot COP
        self.carnot_fraction = 0.15

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.02:
            return 0.0
        # Cryo-cooler has minimum power draw even at partial load
        min_fraction = 0.3  # Compressor doesn't scale linearly
        effective_load = max(load_fraction, min_fraction)
        # Power supply converter efficiency
        converter_eff = 0.96 - 0.02 / max(effective_load, 0.1)
        # Carnot penalty for cryogenic work
        cryo_overhead = 0.05 * (300 / max(self.cooling_stage_k, 4))  # Scales with 1/T
        cryo_overhead = min(cryo_overhead, 0.15)  # Cap at 15%
        return max(0.75, min(0.95, converter_eff - cryo_overhead))


class AntimatterPowerController(PowerController):
    """
    Integrated power controller for antimatter annihilation reactor systems.

    Manages three power conversion pathways downstream of the reactor:

    1. MHD Direct Conversion: Charged pion streams (pi+, pi-) steered by
       magnetic nozzle into magnetohydrodynamic channel. High-velocity
       charged particle flow induces current in MHD electrodes.
       Peak electrical efficiency: ~96% at MHD electrode terminals.

    2. Thermal Conversion: Gamma rays from pi0 decay (2 x 135 MeV photons)
       absorbed in high-Z shielding (tungsten/lead). Thermal energy
       extracted via sCO2 Brayton cycle or thermionic emitters.
       Peak electrical efficiency: ~93% at generator terminals.

    3. Electron Direct Conversion: Auger and ionization electrons from
       antiproton-target-atom interactions collected on graphene Faraday
       cup electrodes. DC electron current conditioned via SiC buck/boost
       converter — no thermal cycle, no moving parts.
       Peak electrical efficiency: ~98% (DC-DC power conditioning only).

    4. Containment overhead: Penning trap magnets, ultra-high vacuum
       pumps, cryo-cooler compressors, trap electrode bias supplies.
       Typically 5-7% of rated power consumed parasitically.

    Note: The AntimatterReactor source handles physics-level conversion
    (annihilation → MHD/thermal/electron). This controller handles the
    electrical conversion chain: power conditioning, bus losses, and
    parasitic loads for each pathway.
    """

    def __init__(self, rated_kw=5_000.0, mhd_fraction=0.70,
                 electron_fraction=0.0,
                 containment_parasitic_fraction=0.05, **kwargs):
        super().__init__(name="Antimatter Controller", rated_kw=rated_kw,
                        topology="triple_pathway", **kwargs)
        self.mhd_fraction = mhd_fraction
        self.electron_fraction = electron_fraction
        self.thermal_fraction = max(0, 1.0 - mhd_fraction - electron_fraction)
        self.containment_parasitic_fraction = containment_parasitic_fraction

    def efficiency_at_load(self, load_fraction: float) -> float:
        if load_fraction < 0.05:
            return 0.0

        # Pathway 1 — MHD: peaks at moderate load
        mhd_peak = 0.96
        mhd_part_load = 1.0 - 0.06 * (1.0 - load_fraction) ** 2
        mhd_eff = mhd_peak * mhd_part_load

        # Pathway 2 — Thermal: Brayton-like characteristic
        thermal_peak = 0.93
        thermal_part_load = 1.0 - 0.10 * (1.0 - load_fraction) ** 2
        thermal_eff = thermal_peak * thermal_part_load

        # Pathway 3 — Electron direct: DC-DC conversion, very flat curve
        # No thermal cycle → near-constant high efficiency across load range
        electron_peak = 0.98
        electron_part_load = 1.0 - 0.02 * (1.0 - load_fraction) ** 2
        electron_eff = electron_peak * electron_part_load

        # Weighted combination across all three pathways
        combined = (self.mhd_fraction * mhd_eff +
                   self.thermal_fraction * thermal_eff +
                   self.electron_fraction * electron_eff)

        # Containment parasitic overhead (fixed fraction, worse at low load)
        parasitic = self.containment_parasitic_fraction / max(load_fraction, 0.1)

        return max(0.75, min(combined, combined - parasitic))


CONTROLLER_REGISTRY = {
    "sic": SiCConverter,
    "gan": GaNConverter,
    "mppt": MPPTController,
    "bidirectional": BidirectionalInverter,
    "hydrogen": HydrogenPowerController,
    "fusion": FusionPowerController,
    "cryogenic": CryogenicPowerSupply,
    "antimatter": AntimatterPowerController,
}


def create_controller(type_key: str, **kwargs) -> PowerController:
    if type_key not in CONTROLLER_REGISTRY:
        raise ValueError(f"Unknown controller type '{type_key}'. Available: {list(CONTROLLER_REGISTRY.keys())}")
    return CONTROLLER_REGISTRY[type_key](**kwargs)
