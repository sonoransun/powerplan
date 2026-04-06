"""
Load profiles and deployment scale definitions.

Defines realistic electrical demand patterns from single-home to
metropolitan scale, with heterogeneous sub-population mixing.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeploymentScale:
    """Defines a deployment scale with its characteristics."""
    name: str
    peak_load_kw: float
    annual_consumption_kwh: float
    num_endpoints: int  # Number of homes/buildings/facilities
    description: str


# Predefined deployment scales
SCALES = {
    "home": DeploymentScale(
        name="Single Home",
        peak_load_kw=12.0,
        annual_consumption_kwh=10_500,
        num_endpoints=1,
        description="Single residential dwelling, 3-4 bedroom",
    ),
    "neighborhood": DeploymentScale(
        name="Neighborhood",
        peak_load_kw=250.0,
        annual_consumption_kwh=220_000,
        num_endpoints=25,
        description="25-home residential neighborhood microgrid",
    ),
    "community": DeploymentScale(
        name="Community",
        peak_load_kw=2_000.0,
        annual_consumption_kwh=5_000_000,
        num_endpoints=500,
        description="Community of ~500 homes + small commercial",
    ),
    "district": DeploymentScale(
        name="District",
        peak_load_kw=25_000.0,
        annual_consumption_kwh=80_000_000,
        num_endpoints=8_000,
        description="Urban district with mixed residential/commercial/light industrial",
    ),
    "metropolitan": DeploymentScale(
        name="Metropolitan",
        peak_load_kw=500_000.0,
        annual_consumption_kwh=2_000_000_000,
        num_endpoints=200_000,
        description="Large metro area with full heterogeneous demand mix",
    ),
}


class LoadProfile:
    """
    Generates realistic time-varying electrical load profiles.

    Combines base load, weather-responsive load, and stochastic demand
    across heterogeneous consumer populations.
    """

    def __init__(self, scale: DeploymentScale, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self.peak_kw = scale.peak_load_kw

        # Population mix fractions (heterogeneous distribution)
        self._residential_frac = self._get_residential_fraction()
        self._commercial_frac = self._get_commercial_fraction()
        self._industrial_frac = 1.0 - self._residential_frac - self._commercial_frac

    def _get_residential_fraction(self) -> float:
        n = self.scale.num_endpoints
        if n <= 25:
            return 0.95
        elif n <= 500:
            return 0.80
        elif n <= 8000:
            return 0.55
        else:
            return 0.45

    def _get_commercial_fraction(self) -> float:
        n = self.scale.num_endpoints
        if n <= 25:
            return 0.05
        elif n <= 500:
            return 0.15
        elif n <= 8000:
            return 0.30
        else:
            return 0.35

    def _residential_profile(self, hour_of_day: float, day_of_year: float) -> float:
        """Residential load: morning/evening peaks, low overnight."""
        # Morning peak 7-9 AM
        morning = 0.3 * np.exp(-0.5 * ((hour_of_day - 7.5) / 1.2) ** 2)
        # Evening peak 6-9 PM (larger)
        evening = 0.5 * np.exp(-0.5 * ((hour_of_day - 19.5) / 1.8) ** 2)
        # Base load (always-on appliances)
        base = 0.25
        # Seasonal — more AC in summer, more heating in winter
        seasonal = 0.15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
        return base + morning + evening + abs(seasonal)

    def _commercial_profile(self, hour_of_day: float, day_of_year: float) -> float:
        """Commercial load: business hours dominated."""
        # Ramp up 7-9 AM, plateau 9-5, ramp down 5-7 PM
        if hour_of_day < 6:
            load = 0.15
        elif hour_of_day < 9:
            load = 0.15 + 0.7 * (hour_of_day - 6) / 3
        elif hour_of_day < 17:
            load = 0.85
        elif hour_of_day < 20:
            load = 0.85 - 0.6 * (hour_of_day - 17) / 3
        else:
            load = 0.20
        # Weekend reduction
        day_of_week = day_of_year % 7
        if day_of_week >= 5:
            load *= 0.4
        return load

    def _industrial_profile(self, hour_of_day: float, day_of_year: float) -> float:
        """Industrial load: relatively flat, slight day/night variation."""
        base = 0.70
        day_boost = 0.2 * np.exp(-0.5 * ((hour_of_day - 13) / 4) ** 2)
        return base + day_boost

    def demand_kw(self, hour_of_year: float, temperature_c: float = 20.0) -> float:
        """
        Total electrical demand at given hour of year.

        Args:
            hour_of_year: Hour [0, 8760)
            temperature_c: Ambient temperature for weather-responsive load
        """
        day = hour_of_year / 24.0
        hour = hour_of_year % 24

        # Weighted mix of consumer types
        res = self._residential_frac * self._residential_profile(hour, day)
        com = self._commercial_frac * self._commercial_profile(hour, day)
        ind = self._industrial_frac * self._industrial_profile(hour, day)
        base_load = (res + com + ind) * self.peak_kw

        # Weather-responsive component (HVAC)
        # Cooling above 24°C, heating below 10°C
        if temperature_c > 24:
            hvac = 0.15 * (temperature_c - 24) / 15 * self.peak_kw
        elif temperature_c < 10:
            hvac = 0.10 * (10 - temperature_c) / 20 * self.peak_kw
        else:
            hvac = 0.0

        # Stochastic component (random demand fluctuations)
        noise = self.rng.normal(0, 0.02 * self.peak_kw)

        total = base_load + hvac + noise
        return max(0.1, min(total, self.peak_kw * 1.1))

    def generate_year(self, dt_hours: float = 1.0) -> np.ndarray:
        """Generate full-year load profile at given time step."""
        hours = np.arange(0, 8760, dt_hours)
        # Synthetic temperature profile
        temps = self._annual_temperature(hours)
        loads = np.array([self.demand_kw(h, t) for h, t in zip(hours, temps)])
        return loads

    def _annual_temperature(self, hours: np.ndarray) -> np.ndarray:
        """Synthetic annual temperature profile."""
        days = hours / 24.0
        # Seasonal baseline
        seasonal = 15 + 15 * np.sin(2 * np.pi * (days - 80) / 365)
        # Diurnal variation
        diurnal = 5 * np.sin(2 * np.pi * (hours % 24 - 6) / 24)
        return seasonal + diurnal

    def summary(self) -> dict:
        return {
            "scale": self.scale.name,
            "peak_kw": self.peak_kw,
            "residential_pct": self._residential_frac * 100,
            "commercial_pct": self._commercial_frac * 100,
            "industrial_pct": self._industrial_frac * 100,
            "num_endpoints": self.scale.num_endpoints,
        }
