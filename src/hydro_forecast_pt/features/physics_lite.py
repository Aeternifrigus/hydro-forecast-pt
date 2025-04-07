"""
Physics-lite hydrological bucket model for hydro-forecast-ch.

Implements a conceptual rainfall-runoff model with:
- Two-layer soil moisture bucket (fast + slow)
- Snow accumulation and melt (degree-day method)
- Runoff generation (saturation excess + baseflow)
- State persistence for sequential prediction

Used as a "physics-informed" feature for ML models.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class BucketState:
    """State of the bucket model at a given time."""
    fast_store: float = 0.0      # mm - surface/upper soil
    slow_store: float = 0.0      # mm - lower soil/groundwater
    snow_store: float = 0.0      # mm - snow water equivalent
    fast_capacity: float = 100.0
    slow_capacity: float = 500.0
    snow_capacity: float = 1000.0

    def total_storage(self) -> float:
        return self.fast_store + self.slow_store + self.snow_store

    def to_dict(self) -> Dict[str, float]:
        return {
            "fast_store": self.fast_store,
            "slow_store": self.slow_store,
            "snow_store": self.snow_store,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BucketState":
        return cls(
            fast_store=d.get("fast_store", 0.0),
            slow_store=d.get("slow_store", 0.0),
            snow_store=d.get("snow_store", 0.0),
        )


@dataclass
class BucketParams:
    """Parameters for the bucket model."""
    # Capacities (mm)
    fast_capacity: float = 100.0
    slow_capacity: float = 500.0
    snow_capacity: float = 1000.0

    # Rate parameters (per week)
    k_fast: float = 0.3          # Fast runoff coefficient
    k_slow: float = 0.02         # Slow runoff / baseflow coefficient
    k_perc: float = 0.1          # Percolation fast -> slow

    # Snow parameters
    snow_temp_threshold: float = 0.0   # °C - temp below which precip is snow
    snow_melt_factor: float = 3.5      # mm/°C/day - degree-day melt factor
    refreeze_factor: float = 0.1       # Fraction of melt that refreezes

    # ET parameters
    pet_factor: float = 0.5      # mm/°C/week - potential ET factor
    pet_temp_base: float = 5.0   # °C - base temp for ET

    # Runoff partitioning
    fast_runoff_frac: float = 0.7  # Fraction of excess going to fast runoff

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "BucketParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class BucketOutput:
    """Outputs from a single bucket step."""
    # Fluxes (mm/week)
    precipitation: float = 0.0
    snowfall: float = 0.0
    rainfall: float = 0.0
    snowmelt: float = 0.0
    refreeze: float = 0.0
    pet: float = 0.0
    aet: float = 0.0           # Actual ET
    fast_runoff: float = 0.0
    slow_runoff: float = 0.0
    percolation: float = 0.0
    total_runoff: float = 0.0

    # State after step
    state: BucketState = field(default_factory=BucketState)

    def to_dict(self) -> Dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "state"}
        d.update(self.state.to_dict())
        return d


class BucketModel:
    """
    Physics-lite two-layer bucket model with snow.

    Structure:
    ┌─────────────────────────────────────┐
    │           PRECIPITATION             │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    SNOW (if T<0°C)      RAIN (if T≥0°C)
         │                   │
         ▼                   ▼
    Snow Store        Fast Store
         │                   │
    Melt (T>0°C)      ┌──────┴──────┐
         │            ▼             ▼
         └──────►  Percolation   Fast Runoff
                      │
                      ▼
                   Slow Store
                      │
                      ▼
                 Slow Runoff
                      │
                      ▼
               TOTAL RUNOFF

    The model runs at weekly timestep.
    """

    def __init__(
        self,
        params: Optional[BucketParams] = None,
        initial_state: Optional[BucketState] = None,
    ):
        """
        Initialize bucket model.

        Args:
            params: Model parameters (default from config)
            initial_state: Initial state (default: 50% filled)
        """
        config = get_config()

        if params is None:
            p = config.features.physics_lite.parameters
            params = BucketParams(
                fast_capacity=p.get("fast_capacity_mm", 100),
                slow_capacity=p.get("slow_capacity_mm", 500),
                snow_capacity=p.get("snow_capacity_mm", 1000),
                k_fast=p.get("k_fast", 0.3),
                k_slow=p.get("k_slow", 0.02),
                k_perc=p.get("k_snow_melt", 0.1),  # Using snow melt param for percolation
                snow_temp_threshold=p.get("snow_temp_threshold", 0.0),
                snow_melt_factor=p.get("snow_melt_factor", 3.5),
            )

        self.params = params

        if initial_state is None:
            # Start at 50% capacity
            initial_state = BucketState(
                fast_store=params.fast_capacity * 0.5,
                slow_store=params.slow_capacity * 0.5,
                snow_store=0.0,
                fast_capacity=params.fast_capacity,
                slow_capacity=params.slow_capacity,
                snow_capacity=params.snow_capacity,
            )

        self.state = initial_state
        self.history: List[BucketOutput] = []

    def step(
        self,
        precipitation_mm: float,
        temperature_c: float,
        snowfall_mm: Optional[float] = None,
    ) -> BucketOutput:
        """
        Run one weekly time step.

        Args:
            precipitation_mm: Total precipitation (mm/week)
            temperature_c: Mean weekly temperature (°C)
            snowfall_mm: Snowfall (mm/week), if None estimated from temp

        Returns:
            BucketOutput with fluxes and updated state
        """
        p = self.params
        s = self.state

        out = BucketOutput(precipitation=precipitation_mm)

        # 1. Partition precipitation into snow/rain
        if snowfall_mm is not None:
            out.snowfall = min(snowfall_mm, precipitation_mm)
            out.rainfall = precipitation_mm - out.snowfall
        else:
            # Temperature-based partitioning
            if temperature_c < p.snow_temp_threshold:
                out.snowfall = precipitation_mm
                out.rainfall = 0.0
            else:
                out.snowfall = 0.0
                out.rainfall = precipitation_mm

        # 2. Snow processes
        # Accumulation
        s.snow_store = min(s.snow_store + out.snowfall, p.snow_capacity)

        # Melt (degree-day method)
        # Melt = melt_factor * max(0, T - threshold) * 7 days
        melt_potential = p.snow_melt_factor * max(0, temperature_c - p.snow_temp_threshold) * 7
        out.snowmelt = min(melt_potential, s.snow_store)
        s.snow_store -= out.snowmelt

        # Refreeze (if temp drops below freezing)
        if temperature_c < p.snow_temp_threshold:
            out.refreeze = min(p.refreeze_factor * out.snowmelt, out.snowmelt)
            s.snow_store += out.refreeze
            out.snowmelt -= out.refreeze
        else:
            out.refreeze = 0.0

        # Water input to soil = rain + snowmelt
        water_input = out.rainfall + out.snowmelt

        # 3. Potential ET
        out.pet = max(0, (temperature_c - p.pet_temp_base) * p.pet_factor * 7)

        # 4. Fast store receives water input
        s.fast_store = min(s.fast_store + water_input, p.fast_capacity)

        # 5. ET from fast store first
        fast_et = min(out.pet, s.fast_store)
        s.fast_store -= fast_et
        et_remaining = out.pet - fast_et

        # Remaining ET from slow store
        slow_et = min(et_remaining, s.slow_store)
        s.slow_store -= slow_et
        out.aet = fast_et + slow_et

        # 6. Percolation fast -> slow
        # Excess above field capacity percolates
        field_capacity_fast = p.fast_capacity * 0.7  # 70% = field capacity
        excess_fast = max(0, s.fast_store - field_capacity_fast)
        out.percolation = min(excess_fast * p.k_perc, excess_fast)
        s.fast_store -= out.percolation
        s.slow_store = min(s.slow_store + out.percolation, p.slow_capacity)

        # 7. Fast runoff (saturation excess + direct)
        fast_runoff_potential = s.fast_store * p.k_fast
        out.fast_runoff = min(fast_runoff_potential, s.fast_store)
        s.fast_store -= out.fast_runoff

        # 8. Slow runoff (baseflow)
        out.slow_runoff = min(s.slow_store * p.k_slow, s.slow_store)
        s.slow_runoff = min(out.slow_runoff, s.slow_store)  # Fixed: was using out.slow_runoff twice
        s.slow_store -= out.slow_runoff

        # 9. Total runoff
        out.total_runoff = out.fast_runoff + out.slow_runoff

        # Save state snapshot
        out.state = BucketState(
            fast_store=s.fast_store,
            slow_store=s.slow_store,
            snow_store=s.snow_store,
            fast_capacity=p.fast_capacity,
            slow_capacity=p.slow_capacity,
            snow_capacity=p.snow_capacity,
        )

        self.history.append(out)
        return out

    def simulate(
        self,
        precipitation: np.ndarray,
        temperature: np.ndarray,
        snowfall: Optional[np.ndarray] = None,
        reset_state: bool = True,
    ) -> pd.DataFrame:
        """
        Run simulation for a time series.

        Args:
            precipitation: Array of weekly precipitation (mm)
            temperature: Array of weekly mean temperature (°C)
            snowfall: Array of weekly snowfall (mm), optional
            reset_state: Whether to reset to initial state

        Returns:
            DataFrame with outputs for each time step
        """
        if reset_state:
            self.state = BucketState(
                fast_store=self.params.fast_capacity * 0.5,
                slow_store=self.params.slow_capacity * 0.5,
                snow_store=0.0,
                fast_capacity=self.params.fast_capacity,
                slow_capacity=self.params.slow_capacity,
                snow_capacity=self.params.snow_capacity,
            )
            self.history = []

        n = len(precipitation)
        if snowfall is None:
            snowfall = [None] * n

        for i in range(n):
            self.step(precipitation[i], temperature[i], snowfall[i])

        # Convert to DataFrame
        df = pd.DataFrame([out.to_dict() for out in self.history])
        return df

    def get_state(self) -> BucketState:
        """Get current state."""
        return self.state

    def set_state(self, state: BucketState) -> None:
        """Set current state."""
        self.state = state

    def get_features(self) -> Dict[str, float]:
        """Get current state as feature vector."""
        s = self.state
        p = self.params

        return {
            "bucket_fast_store": s.fast_store,
            "bucket_slow_store": s.slow_store,
            "bucket_snow_store": s.snow_store,
            "bucket_total_storage": s.total_storage(),
            "bucket_fast_frac": s.fast_store / p.fast_capacity if p.fast_capacity > 0 else 0,
            "bucket_slow_frac": s.slow_store / p.slow_capacity if p.slow_capacity > 0 else 0,
            "bucket_snow_frac": s.snow_store / p.snow_capacity if p.snow_capacity > 0 else 0,
            "bucket_saturation": (s.fast_store + s.slow_store) / (p.fast_capacity + p.slow_capacity),
        }

    def save_state(self, path: str) -> None:
        """Save state to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump({
                "state": self.state.to_dict(),
                "params": self.params.to_dict(),
            }, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load state from JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.state = BucketState.from_dict(data["state"])
        self.params = BucketParams.from_dict(data["params"])


class MultiBasinBucketModel:
    """
    Bucket model for multiple basins with separate states.
    """

    def __init__(
        self,
        basin_ids: List[str],
        params: Optional[Dict[str, BucketParams]] = None,
    ):
        """
        Initialize multi-basin model.

        Args:
            basin_ids: List of basin identifiers
            params: Per-basin parameters (default: same for all)
        """
        self.basin_ids = basin_ids
        self.models: Dict[str, BucketModel] = {}

        default_params = BucketParams()

        for basin_id in basin_ids:
            basin_params = params.get(basin_id) if params else default_params
            self.models[basin_id] = BucketModel(params=basin_params)

    def step(
        self,
        precipitation: Dict[str, float],
        temperature: Dict[str, float],
        snowfall: Optional[Dict[str, float]] = None,
    ) -> Dict[str, BucketOutput]:
        """Run one step for all basins."""
        outputs = {}
        for basin_id in self.basin_ids:
            p = precipitation.get(basin_id, 0.0)
            t = temperature.get(basin_id, 0.0)
            s = snowfall.get(basin_id) if snowfall else None
            outputs[basin_id] = self.models[basin_id].step(p, t, s)
        return outputs

    def get_all_features(self) -> pd.DataFrame:
        """Get features for all basins as DataFrame."""
        rows = []
        for basin_id, model in self.models.items():
            feats = model.get_features()
            feats["basin_id"] = basin_id
            rows.append(feats)
        return pd.DataFrame(rows)

    def save_all_states(self, path: str) -> None:
        """Save all basin states."""
        import json
        data = {}
        for basin_id, model in self.models.items():
            data[basin_id] = {
                "state": model.state.to_dict(),
                "params": model.params.to_dict(),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_all_states(self, path: str) -> None:
        """Load all basin states."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        for basin_id, d in data.items():
            if basin_id in self.models:
                self.models[basin_id].state = BucketState.from_dict(d["state"])
                self.models[basin_id].params = BucketParams.from_dict(d["params"])


def add_bucket_features(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    temp_col: str = "temperature_c",
    snowfall_col: Optional[str] = "snowfall_mm",
    basin_col: str = "basin_id",
    date_col: str = "week_end",
) -> pd.DataFrame:
    """
    Add bucket model features to DataFrame.

    Runs the bucket model sequentially through the data and adds
    state variables as features.

    Args:
        df: DataFrame with weather data
        precip_col: Precipitation column
        temp_col: Temperature column
        snowfall_col: Snowfall column (optional)
        basin_col: Basin identifier column
        date_col: Date column

    Returns:
        DataFrame with bucket features added
    """
    result = df.copy()
    result = result.sort_values([basin_col, date_col])

    # Initialize multi-basin model
    basins = result[basin_col].unique()
    model = MultiBasinBucketModel(basins)

    # Run simulation for each basin
    all_features = []

    for basin_id in basins:
        basin_data = result[result[basin_col] == basin_id].copy()

        precip = basin_data[precip_col].values
        temp = basin_data[temp_col].values
        snow = basin_data[snowfall_col].values if snowfall_col and snowfall_col in basin_data.columns else None

        # Simulate
        sim_df = model.models[basin_id].simulate(precip, temp, snow, reset_state=True)

        # Add features to basin data
        for col in sim_df.columns:
            basin_data[f"bucket_{col}"] = sim_df[col].values

        all_features.append(basin_data)

    result = pd.concat(all_features, ignore_index=True)
    logger.info(f"Added bucket features for {len(basins)} basins")

    return result


def create_bucket_model(
    basin_id: Optional[str] = None,
    params: Optional[BucketParams] = None,
) -> BucketModel:
    """Factory function to create bucket model."""
    return BucketModel(params=params)


def create_multi_basin_model(
    basin_ids: List[str],
    params: Optional[Dict[str, BucketParams]] = None,
) -> MultiBasinBucketModel:
    """Factory function to create multi-basin model."""
    return MultiBasinBucketModel(basin_ids, params)