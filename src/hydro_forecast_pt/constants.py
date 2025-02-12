"""
Constants for hydro-forecast-ch.

Contains Swiss basin definitions, ENTSO-E codes, and other project constants.
"""
from enum import Enum
from typing import Dict, List


class EntsoeAreaCode(str, Enum):
    """ENTSO-E bidding zone / area codes."""
    SWITZERLAND = "CH"
    GERMANY = "DE"
    FRANCE = "FR"
    ITALY_NORD = "IT-NORD"
    AUSTRIA = "AT"


class EntsoeProductionType(str, Enum):
    """ENTSO-E production type codes (PSR type)."""
    HYDRO_RUN_OF_RIVER = "B16"
    HYDRO_RESERVOIR = "B15"
    HYDRO_PUMPED_STORAGE = "B17"
    NUCLEAR = "B03"
    SOLAR = "B09"
    WIND_ONSHORE = "B10"
    WIND_OFFSHORE = "B11"


class EntsoeDocumentType(str, Enum):
    """ENTSO-E document type codes."""
    GENERATION = "A75"
    INSTALLED_CAPACITY = "A68"
    ACTUAL_TOTAL_LOAD = "A65"
    DAY_AHEAD_PRICES = "A44"


class EntsoeProcessType(str, Enum):
    """ENTSO-E process type codes."""
    REALISED = "A16"
    DAY_AHEAD = "A01"
    WEEK_AHEAD = "A31"
    MONTH_AHEAD = "A32"


# Swiss basin definitions
SWISS_BASINS: List[Dict] = [
    {
        "id": "rhine",
        "name": "Rhine",
        "countries": ["CH", "DE", "FR", "AT", "LI"],
        "major_sub_basins": ["Aare", "Reuss", "Limmat", "Thur", "Bir"],
        "gauge_stations": ["Reu240", "Aar230", "Lim200", "Thur200"],
        "area_km2": 37500,
        "swiss_area_km2": 34500,
    },
    {
        "id": "rhone",
        "name": "Rhône",
        "countries": ["CH", "FR"],
        "major_sub_basins": ["Vispa", "Navizence", "Borgne", "Drance", "Trient"],
        "gauge_stations": ["Rho100", "Vis100", "Nav100"],
        "area_km2": 15000,
        "swiss_area_km2": 5200,
    },
    {
        "id": "inn",
        "name": "Inn",
        "countries": ["CH", "AT", "DE", "IT"],
        "major_sub_basins": ["Landquart", "Albula", "Surses", "Spöl"],
        "gauge_stations": ["Inn100", "Alb100"],
        "area_km2": 26000,
        "swiss_area_km2": 2200,
    },
    {
        "id": "ticino",
        "name": "Ticino",
        "countries": ["CH", "IT"],
        "major_sub_basins": ["Moesa", "Calancasca", "Verzasca", "Maggia"],
        "gauge_stations": ["Tic100", "Mag100"],
        "area_km2": 3800,
        "swiss_area_km2": 1700,
    },
]

# Basin outlets (for routing)
BASIN_OUTLETS: Dict[str, Dict] = {
    "rhine": {"lat": 47.56, "lon": 7.59, "name": "Basel"},
    "rhone": {"lat": 46.20, "lon": 6.14, "name": "Geneva"},
    "inn": {"lat": 47.00, "lon": 10.50, "name": "Pradella"},
    "ticino": {"lat": 46.15, "lon": 8.80, "name": "Magadino"},
}

# Major Swiss hydro plants (from WASTA / BFE)
MAJOR_SWISS_HYDRO_PLANTS: List[Dict] = [
    # Rhine basin
    {"name": "Albbruck-Dogern", "basin": "rhine", "type": "run_of_river", "capacity_mw": 72, "lat": 47.56, "lon": 8.13},
    {"name": "Rheinfelden", "basin": "rhine", "type": "run_of_river", "capacity_mw": 120, "lat": 47.55, "lon": 7.79},
    {"name": "Laufenburg", "basin": "rhine", "type": "run_of_river", "capacity_mw": 120, "lat": 47.55, "lon": 8.05},
    # Rhône basin
    {"name": "Chancy-Pougny", "basin": "rhone", "type": "run_of_river", "capacity_mw": 85, "lat": 46.15, "lon": 5.95},
    {"name": "Verbois", "basin": "rhone", "type": "run_of_river", "capacity_mw": 110, "lat": 46.19, "lon": 6.04},
    {"name": "Seujet", "basin": "rhone", "type": "run_of_river", "capacity_mw": 30, "lat": 46.20, "lon": 6.15},
    # Inn basin
    {"name": "Pradella", "basin": "inn", "type": "run_of_river", "capacity_mw": 110, "lat": 46.78, "lon": 10.07},
    {"name": "Martina", "basin": "inn", "type": "run_of_river", "capacity_mw": 40, "lat": 46.83, "lon": 10.30},
    # Ticino basin
    {"name": "Piottino", "basin": "ticino", "type": "run_of_river", "capacity_mw": 65, "lat": 46.55, "lon": 8.78},
    {"name": "Biaschina", "basin": "ticino", "type": "run_of_river", "capacity_mw": 180, "lat": 46.40, "lon": 8.75},
]

# BFE reservoir regions
BFE_RESERVOIR_REGIONS: List[str] = [
    "all",
    "region_1",  # Western Alps
    "region_2",  # Eastern Alps
    "region_3",  # Central Alps
    "region_4",  # Ticino
]

# Default forecast horizons
DEFAULT_HORIZONS_WEEKS: List[int] = [1, 2, 3, 4, 5, 6]

# Time constants
SECONDS_PER_WEEK: int = 7 * 24 * 60 * 60  # 604800
SECONDS_PER_DAY: int = 24 * 60 * 60  # 86400
SECONDS_PER_HOUR: int = 60 * 60  # 3600

# Unit conversion functions
def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def mm_to_mcm(mm: float, area_km2: float) -> float:
    """Convert mm over area to million cubic meters.

    1 mm over 1 km² = 1000 m³ = 0.001 MCM
    """
    return mm * area_km2 * 0.001

# Data version for cache invalidation
DATA_VERSION: str = "2026.10"

# Version
__version__: str = "0.1.0"