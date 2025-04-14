"""
Basin aggregation for hydro-forecast-ch.

Aggregates gridded weather data (ERA5, ECMWF) to Swiss basin averages
using basin masks derived from hydro-atlas or similar datasets.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping

from ..config import get_config
from ..constants import SWISS_BASINS, BASIN_OUTLETS

logger = logging.getLogger(__name__)


class BasinAggregator:
    """
    Aggregates gridded weather data to basin averages.

    Supports:
    - Weighted averaging using basin masks
    - Point extraction at basin outlets
    - Multiple basin geometries
    """

    def __init__(
        self,
        basin_shapes_path: Optional[Union[str, Path]] = None,
        grid_resolution: float = 0.1,  # degrees
    ):
        """
        Initialize basin aggregator.

        Args:
            basin_shapes_path: Path to basin shapefile/GeoJSON
            grid_resolution: Grid resolution in degrees (ERA5-Land = 0.1)
        """
        self.grid_resolution = grid_resolution
        self.basin_shapes_path = Path(basin_shapes_path) if basin_shapes_path else None
        self.basin_geoms = self._load_basin_geometries()
        self.basin_masks = {}

    def _load_basin_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load basin geometries from file or create from known coordinates."""
        basins = {}

        if self.basin_shapes_path and self.basin_shapes_path.exists():
            try:
                gdf = gpd.read_file(self.basin_shapes_path)
                # Expect columns: basin_id, geometry
                for basin_id in gdf["basin_id"].unique():
                    basins[basin_id] = gdf[gdf["basin_id"] == basin_id]
                logger.info(f"Loaded {len(basins)} basin geometries from {self.basin_shapes_path}")
            except Exception as e:
                logger.warning(f"Failed to load basin shapes: {e}")

        # Fallback: create approximate basins from bounding boxes
        if not basins:
            logger.info("Creating approximate basin geometries from bounding boxes")
            basins = self._create_approximate_basins()

        return basins

    def _create_approximate_basins(self) -> Dict[str, gpd.GeoDataFrame]:
        """Create approximate basin geometries from known bounding boxes."""
        # Approximate bounding boxes for major Swiss basins
        # These are rough approximations - real implementation should use HydroATLAS or similar
        basin_bboxes = {
            "rhine": (5.9, 46.5, 10.5, 47.8),   # Most of northern/eastern CH
            "rhone": (5.9, 45.8, 7.5, 46.5),    # Western CH (Valais, Vaud, Geneva)
            "inn": (10.0, 46.5, 10.5, 47.0),    # Engadine (small)
            "ticino": (8.5, 46.0, 9.5, 46.6),   # Southern CH (Ticino)
        }

        basins = {}
        for basin_id, (minx, miny, maxx, maxy) in basin_bboxes.items():
            geom = box(minx, miny, maxx, maxy)
            gdf = gpd.GeoDataFrame(
                {"basin_id": [basin_id], "name": [SWISS_BASINS[basin_id]["name"]]},
                geometry=[geom],
                crs="EPSG:4326"
            )
            basins[basin_id] = gdf

        return basins

    def create_basin_mask(
        self,
        basin_id: str,
        ds: xr.Dataset,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> xr.DataArray:
        """
        Create boolean mask for a basin on the dataset grid.

        Args:
            basin_id: Basin identifier
            ds: xarray Dataset with lat/lon coordinates
            lat_dim: Latitude dimension name
            lon_dim: Longitude dimension name

        Returns:
            Boolean DataArray (True = inside basin)
        """
        if basin_id in self.basin_masks:
            # Check if mask matches current grid
            existing = self.basin_masks[basin_id]
            if (existing.sizes.get(lat_dim) == ds.sizes.get(lat_dim) and
                existing.sizes.get(lon_dim) == ds.sizes.get(lon_dim)):
                return existing

        # Get basin geometry
        if basin_id not in self.basin_geoms:
            raise ValueError(f"Unknown basin: {basin_id}")

        basin_gdf = self.basin_geoms[basin_id]

        # Get grid coordinates
        lats = ds[lat_dim].values
        lons = ds[lon_dim].values

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create mask using rasterio
        # Transform from lat/lon to pixel coordinates
        transform = rasterio.transform.from_bounds(
            lons.min(), lats.min(), lons.max(), lats.max(),
            len(lons), len(lats)
        )

        # Combine all geometries for this basin
        shapes = [mapping(geom) for geom in basin_gdf.geometry]

        # Create mask (True = outside, so invert)
        mask = geometry_mask(
            shapes,
            out_shape=(len(lats), len(lons)),
            transform=transform,
            invert=True,
        )

        # Create DataArray
        mask_da = xr.DataArray(
            mask,
            dims=[lat_dim, lon_dim],
            coords={lat_dim: lats, lon_dim: lons},
            name=f"{basin_id}_mask",
        )

        self.basin_masks[basin_id] = mask_da
        logger.debug(f"Created mask for {basin_id}: {mask.sum()} grid cells")

        return mask_da

    def aggregate_to_basin(
        self,
        ds: xr.Dataset,
        basin_id: str,
        variables: Optional[List[str]] = None,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
        weight_by_area: bool = True,
    ) -> xr.Dataset:
        """
        Aggregate gridded data to basin average.

        Args:
            ds: xarray Dataset with weather variables
            basin_id: Basin identifier
            variables: Variables to aggregate (None = all)
            lat_dim: Latitude dimension
            lon_dim: Longitude dimension
            weight_by_area: Weight by grid cell area (cos(lat))

        Returns:
            Dataset with basin-averaged variables
        """
        mask = self.create_basin_mask(basin_id, ds, lat_dim, lon_dim)

        # Select variables
        if variables is None:
            variables = [v for v in ds.data_vars if lat_dim in ds[v].dims and lon_dim in ds[v].dims]

        result_vars = {}

        for var in variables:
            data = ds[var]

            # Apply mask
            masked = data.where(mask)

            if weight_by_area:
                # Weight by cosine of latitude (grid cell area)
                weights = np.cos(np.deg2rad(mask[lat_dim]))
                weights_da = xr.DataArray(weights, dims=[lat_dim], coords={lat_dim: mask[lat_dim]})
                weights_2d = weights_da.broadcast_like(masked)
                weights_2d = weights_2d.where(mask)

                # Weighted mean
                weighted_sum = (masked * weights_2d).sum(dim=[lat_dim, lon_dim], skipna=True)
                weight_sum = weights_2d.sum(dim=[lat_dim, lon_dim], skipna=True)
                basin_mean = weighted_sum / weight_sum
            else:
                # Simple mean
                basin_mean = masked.mean(dim=[lat_dim, lon_dim], skipna=True)

            result_vars[var] = basin_mean

        return xr.Dataset(result_vars)

    def aggregate_all_basins(
        self,
        ds: xr.Dataset,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, xr.Dataset]:
        """Aggregate to all known basins."""
        results = {}
        for basin_id in self.basin_geoms.keys():
            try:
                results[basin_id] = self.aggregate_to_basin(ds, basin_id, variables)
            except Exception as e:
                logger.warning(f"Failed to aggregate {basin_id}: {e}")
        return results

    def extract_point(
        self,
        ds: xr.Dataset,
        basin_id: str,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> xr.Dataset:
        """
        Extract data at basin outlet point.

        Args:
            ds: xarray Dataset
            basin_id: Basin identifier
            lat_dim: Latitude dimension
            lon_dim: Longitude dimension

        Returns:
            Dataset with point values
        """
        if basin_id not in BASIN_OUTLETS:
            raise ValueError(f"No outlet defined for basin: {basin_id}")

        outlet_lon, outlet_lat = BASIN_OUTLETS[basin_id]

        # Select nearest grid point
        result = ds.sel({lon_dim: outlet_lon, lat_dim: outlet_lat}, method="nearest")

        return result

    def get_basin_area_km2(self, basin_id: str, swiss_only: bool = True) -> float:
        """Get basin area in km²."""
        return SWISS_BASINS[basin_id]["swiss_area_km2"] if swiss_only else SWISS_BASINS[basin_id]["total_area_km2"]


def create_basin_masks_from_hydroatlas(
    hydroatlas_path: Union[str, Path],
    output_path: Union[str, Path],
    grid_resolution: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Create basin masks from HydroATLAS data.

    HydroATLAS provides high-resolution basin delineations worldwide.
    This function creates masks for the ERA5-Land grid.

    Args:
        hydroatlas_path: Path to HydroATLAS basin shapefile
        output_path: Output path for mask files
        grid_resolution: Target grid resolution

    Returns:
        Dictionary of basin_id -> mask array
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load HydroATLAS
    basins_gdf = gpd.read_file(hydroatlas_path)

    # Filter to Swiss basins (HYBAS_ID for major European basins)
    # This is a simplified version - real implementation would match
    # HydroATLAS basin IDs to our Swiss basins
    swiss_hybas_ids = {
        "rhine": [2120017310, 2120017320],  # Example HYBAS IDs
        "rhone": [2120017400],
        "inn": [2120017500],
        "ticino": [2120017600],
    }

    # Create target grid (ERA5-Land bounds for Switzerland)
    lons = np.arange(5.9, 10.5, grid_resolution)
    lats = np.arange(45.8, 47.8, grid_resolution)

    masks = {}

    for basin_id, hybas_ids in swiss_hybas_ids.items():
        basin_geoms = basins_gdf[basins_gdf["HYBAS_ID"].isin(hybas_ids)]

        if basin_geoms.empty:
            logger.warning(f"No HydroATLAS geometry found for {basin_id}")
            continue

        # Create mask
        transform = rasterio.transform.from_bounds(
            lons.min(), lats.min(), lons.max(), lats.max(),
            len(lons), len(lats)
        )

        shapes = [mapping(geom) for geom in basin_geoms.geometry]
        mask = geometry_mask(
            shapes,
            out_shape=(len(lats), len(lons)),
            transform=transform,
            invert=True,
        )

        masks[basin_id] = mask

        # Save mask
        np.save(output_path / f"{basin_id}_mask.npy", mask)

        logger.info(f"Created HydroATLAS mask for {basin_id}: {mask.sum()} cells")

    return masks


# Convenience function
def get_basin_aggregator(
    basin_shapes_path: Optional[Union[str, Path]] = None,
) -> BasinAggregator:
    """Get configured BasinAggregator instance."""
    return BasinAggregator(basin_shapes_path=basin_shapes_path)