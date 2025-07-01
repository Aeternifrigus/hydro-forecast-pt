# Feature Catalog

Complete specification of all 217 engineered features used in the Swiss hydro-forecasting model.

---

## Overview

| Category | Count | Description |
|----------|-------|-------------|
| **Temporal** | 42 | Cyclical time encoding, water year, holidays |
| **Hydrological** | 112 | Process-based: AWI, bucket model, snowmelt, degree days |
| **Lag** | 32 | Target CF history (1-8w, 12w, 26w, 52w) |
| **Rolling** | 98 | Multi-scale window statistics (precip, temp) |
| **Physics/Interaction** | 18 | Cross-variable physics interactions + basin statics |
| **Total** | **217** | |

---

## 1. Temporal Features (42)

### Cyclical Encoding (8)
| Feature | Formula | Period | Purpose |
|---------|---------|--------|---------|
| `sin_week` | sin(2π × week / 52) | 52 weeks | Seasonal cycle |
| `cos_week` | cos(2π × week / 52) | 52 weeks | Seasonal cycle |
| `sin_month` | sin(2π × month / 12) | 12 months | Monthly cycle |
| `cos_month` | cos(2π × month / 12) | 12 months | Monthly cycle |
| `sin_doy` | sin(2π × day_of_year / 365.25) | 365.25 days | Sub-weekly precision |
| `cos_doy` | cos(2π × day_of_year / 365.25) | 365.25 days | Sub-weekly precision |
| `sin_week_year` | sin(2π × week / 52.18) | 52.18 weeks | Leap-year aware |
| `cos_week_year` | cos(2π × week / 52.18) | 52.18 weeks | Leap-year aware |

### Water Year Features (4)
| Feature | Description |
|---------|-------------|
| `water_year` | Hydrological year (Oct 1 - Sep 30) |
| `week_of_water_year` | 1-52 within water year |
| `month_of_water_year` | 1-12 within water year (Oct=1) |
| `days_since_water_year_start` | Cumulative days since Oct 1 |

### Calendar Features (6)
| Feature | Description |
|---------|-------------|
| `is_weekend` | Saturday or Sunday (1/0) |
| `is_holiday` | Swiss national holidays (1/0) |
| `quarter` | Q1-Q4 |
| `season` | DJF/MAM/JJA/SON (0-3) |
| `days_in_month` | 28-31 |
| `is_leap_year` | 1 if leap year else 0 |

### Trend Features (24)
| Feature | Description |
|---------|-------------|
| `year` | Calendar year (e.g., 2024) |
| `year_normalized` | (year - 2024) / 10 |
| `week_since_start` | Monotonic week counter from training start |
| `month_since_start` | Monotonic month counter |
| + 20 polynomial features: week^1..5, month^1..5, interactions |

---

## 2. Hydrological Process Features (112)

### Antecedent Wetness Index - AWI (7)
| Feature | Formula | Parameters |
|---------|---------|------------|
| `awi` | AWIₜ = α·AWIₜ₋₁ + (1-α)·Pₜ | α=0.85 (calibrated) |
| `awi_norm` | awi / max(awi) | Normalized [0,1] |
| `awi_roll_2w` | Rolling mean 2w | Smoothing |
| `awi_roll_4w` | Rolling mean 4w | Monthly scale |
| `awi_roll_12w` | Rolling mean 12w | Seasonal scale |
| `awi_diff` | awi - awi.shift(1) | Rate of change |
| `awi_saturated` | 1 if awi > 0.9 quantile | Saturation indicator |

**Physical basis**: Recursive soil moisture accounting. High AWI → catchment wet → more runoff per mm rain.

### Two-Layer Bucket Model (16)
| Feature | Reservoir | Time Constant | Formula |
|---------|-----------|---------------|---------|
| `bucket_fast_storage` | Fast (surface) | τ_fast = 2 weeks | S_fastₜ = S_fastₜ₋₁ + Pₜ - Q_fastₜ |
| `bucket_slow_storage` | Slow (baseflow) | τ_slow = 8 weeks | S_slowₜ = S_slowₜ₋₁ + Q_fastₜ - Q_slowₜ |
| `bucket_fast_runoff` | Fast | - | Q_fast = S_fast / τ_fast |
| `bucket_slow_runoff` | Slow | - | Q_slow = S_slow / τ_slow |
| `bucket_total_runoff` | Combined | - | Q_fast + Q_slow |
| `bucket_fast_storage_norm` | Fast | - | Normalized storage |
| `bucket_slow_storage_norm` | Slow | - | Normalized storage |
| `bucket_storage_ratio` | Ratio | - | S_fast / (S_fast + S_slow) |
| + 8 rolling versions (2w, 4w, 12w means of above) | | | |

**Physical basis**: Linear reservoir cascade. Fast = surface runoff (hours-days), Slow = baseflow (weeks-months).

### Snowmelt Modeling (12)
| Feature | Formula | Parameters |
|---------|---------|------------|
| `snowmelt_mm` | max(0, T_mean - 0°C) × DDF | DDF = 3.5 mm/°C/day |
| `snowmelt_mm_weekly` | Σ daily melt | Weekly aggregate |
| `refreezing_mm` | max(0, -T_min) × 0.5 × DDF | Cold content replenishment |
| `snowpack_mm` | Previous + precip_snow - melt + refreeze | Snow water equivalent |
| `snow_cover_fraction` | 1 if snow_depth > 0 else 0 | Binary cover |
| `snow_depth_cm` | From ERA5-Land (max weekly) | Direct measurement |
| `snow_accumulation_mm` | precip × 1_{T<0°C} | Snowfall water equiv. |
| `melt_refreeze_ratio` | melt / (refreeze + ε) | Melt dominance |
| `snowmelt_roll_2w` | Rolling mean 2w | Short-term melt trend |
| `snowmelt_roll_4w` | Rolling mean 4w | Monthly melt volume |
| `snowmelt_roll_12w` | Rolling mean 12w | Seasonal melt volume |
| `snow_depletion_date` | Week when snowpack → 0 | Seasonal timing |

**Physical basis**: Temperature-index (degree-day) model. Calibrated for Swiss Alps.

### Degree Days (14)
| Feature | Formula | Base |
|---------|---------|------|
| `cdd_week` | Σ max(0, T_daily - 15) | 15°C |
| `hdd_week` | Σ max(0, 15 - T_daily) | 15°C |
| `cdd_week_max` | Max daily CDD | Extreme heat |
| `hdd_week_max` | Max daily HDD | Extreme cold |
| `cdd_roll_2w` | Rolling sum 2w | Short-term evaporative demand |
| `cdd_roll_4w` | Rolling sum 4w | Monthly |
| `cdd_roll_12w` | Rolling sum 12w | Seasonal |
| `cdd_roll_26w` | Rolling sum 26w | Half-year |
| `hdd_roll_2w` | Rolling sum 2w | Heating demand |
| `hdd_roll_4w` | Rolling sum 4w | Monthly |
| `hdd_roll_12w` | Rolling sum 12w | Seasonal |
| `hdd_roll_26w` | Rolling sum 26w | Half-year |
| `gdd_week` | Σ max(0, T_daily - 5) | 5°C (growing) |
| `gdd_roll_12w` | Rolling sum 12w | Seasonal GDD |

**Physical basis**: CDD = cooling degree days (evapotranspiration proxy), HDD = heating degree days (winter energy demand proxy).

### Soil Moisture & Runoff Proxies (15)
| Feature | Description |
|---------|-------------|
| `soil_moisture_proxy` | ERA5-Land 0-7cm layer mean (weekly) |
| `soil_moisture_anomaly` | Current - climatology |
| `runoff_proxy` | bucket_total_runoff (from bucket model) |
| `runoff_anomaly` | Current - climatology |
| `baseflow_index` | slow_runoff / total_runoff |
| `quickflow_ratio` | fast_runoff / total_runoff |
| `runoff_roll_2w` | Rolling mean 2w |
| `runoff_roll_4w` | Rolling mean 4w |
| `runoff_roll_12w` | Rolling mean 12w |
| `runoff_roll_26w` | Rolling mean 26w |
| `soil_moisture_roll_2w` | Rolling mean 2w |
| `soil_moisture_roll_4w` | Rolling mean 4w |
| `soil_moisture_roll_12w` | Rolling mean 12w |
| `soil_moisture_roll_26w` | Rolling mean 26w |
| `wetness_index` | Combined AWI + soil_moisture + bucket_fast |

### Precipitation Phase Partitioning (8)
| Feature | Formula |
|---------|---------|
| `precip_rain_mm` | precip × 1_{T_mean > 1°C} |
| `precip_snow_mm` | precip × 1_{T_mean ≤ 1°C} |
| `rain_fraction` | precip_rain / precip |
| `snow_fraction` | precip_snow / precip |
| `precip_intensity_mm_day` | precip / 7 |
| `rain_intensity` | precip_rain / 7 |
| `snow_intensity` | precip_snow / 7 |
| `phase_transition` | 1 if rain_fraction crossed 0.5 | 

---

## 3. Lag Features (32)

### Target Capacity Factor Lags
| Horizon | Features | Count |
|---------|----------|-------|
| Short-term | `target_cf_lag_1w` ... `target_cf_lag_8w` | 8 |
| Medium-term | `target_cf_lag_12w`, `target_cf_lag_26w` | 2 |
| Seasonal | `target_cf_lag_52w` | 1 |
| **Per-basin** | Above × 4 basins (Rhine, Rhône, Inn, Ticino) | **44** |

*Note: In practice we use 32 most informative lags selected via MI (Mutual Information)*

### Additional Lag Features (8)
| Feature | Description |
|---------|-------------|
| `precip_lag_1w` ... `precip_lag_4w` | Precipitation lags |
| `temp_lag_1w` ... `temp_lag_4w` | Temperature lags |

---

## 4. Rolling Window Features (98)

### Precipitation Rolling Sums (40)
| Windows | Features per window | Total |
|---------|---------------------|-------|
| 1, 2, 3, 4, 8, 12, 26, 52 weeks | sum, mean, std, min, max | 8 × 5 = 40 |

**Generated features** (for each window `w`):
- `precip_roll_{w}w_sum` — Total precipitation
- `precip_roll_{w}w_mean` — Weekly average
- `precip_roll_{w}w_std` — Variability
- `precip_roll_{w}w_min` — Dry spell indicator
- `precip_roll_{w}w_max` — Extreme event indicator

### Temperature Rolling Stats (40)
| Windows | Features per window | Total |
|---------|---------------------|-------|
| 1, 2, 3, 4, 8, 12, 26, 52 weeks | mean, std, min, max, range | 8 × 5 = 40 |

**Generated features** (for each window `w`):
- `temp_roll_{w}w_mean` — Average temperature
- `temp_roll_{w}w_std` — Temperature variability
- `temp_roll_{w}w_min` — Cold spell indicator
- `temp_roll_{w}w_max` — Heat wave indicator
- `temp_roll_{w}w_range` | max - min | Diurnal/weekly range |

### Radiation & Snow Rolling (18)
| Feature | Windows | Count |
|---------|---------|-------|
| `radiation_roll_{w}w_sum` | 1, 4, 12, 26, 52 | 5 |
| `radiation_roll_{w}w_mean` | 1, 4, 12, 26, 52 | 5 |
| `snow_depth_roll_{w}w_max` | 1, 4, 12, 26, 52 | 5 |
| `snow_depth_roll_{w}w_mean` | 1, 12 | 2 |
| `snow_depth_roll_{w}w_std` | 1, 12 | 2 |

---

## 5. Physics/Interaction Features (18)

### Cross-Variable Interactions (12)
| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `precip_x_temp` | precip × temp_mean | Rain vs. snow partitioning |
| `precip_x_awi` | precip × AWI | Saturation-excess runoff |
| `precip_x_snowmelt` | precip × snowmelt | Rain-on-snow events |
| `awi_x_snowmelt` | AWI × snowmelt | Wet catchment + melt = flood risk |
| `temp_x_snow_depth` | temp_mean × snow_depth | Melt rate modulation |
| `cdd_x_awi` | CDD × AWI | Evaporative demand on wet catchment |
| `hdd_x_snow_depth` | HDD × snow_depth | Cold content |
| `runoff_x_temp` | runoff_proxy × temp | Temperature-modulated runoff |
| `radiation_x_temp` | radiation × temp | Energy balance |
| `precip_intensity_x_awi` | (precip/7) × AWI | High-intensity rain on wet soil |
| `snowmelt_x_precip_rain` | snowmelt × precip_rain | Rain-on-snow (major flood driver) |
| `bucket_fast_x_snowmelt` | fast_runoff × snowmelt | Quick response + melt |

### Basin Static Attributes (6)
| Feature | Source | Description |
|---------|--------|-------------|
| `basin_elevation_mean_m` | DEM | Mean catchment elevation |
| `basin_elevation_std_m` | DEM | Elevation variability (glacier proxy) |
| `basin_glacier_cover_pct` | GLIMS/SGI | Glacier area % |
| `basin_lake_area_pct` | SwissTopo | Lake area % (regulation) |
| `basin_soil_permeability` | Soil map | Mean permeability class |
| `basin_area_km2` | FOEN | Catchment area |

---

## Feature Selection Pipeline

```python
# 1. Remove constant/near-constant
selector = VarianceThreshold(threshold=1e-5)

# 2. Mutual Information (regression)
mi_scores = mutual_info_regression(X, y, n_neighbors=3, random_state=42)
top_k = mi_scores.nlargest(150).index  # k=150

# 3. Variance Inflation Factor (multicollinearity)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
selected = vif_data[vif_data["VIF"] < 10]["feature"].tolist()

# 4. Domain knowledge keep-list (always retain)
always_keep = ["basin_id", "sin_week", "cos_week", "awi", "snowmelt_mm", 
               "bucket_total_runoff", "cdd_week", "hdd_week"]
```

**Result**: 217 → ~150 features after selection.

---

## Feature Importance (Typical Top-20 for RF)

| Rank | Feature | Category | Mean Importance |
|------|---------|----------|-----------------|
| 1 | `target_cf_lag_1w` | Lag | 0.18 |
| 2 | `awi` | Hydro | 0.12 |
| 3 | `snowmelt_mm` | Hydro | 0.10 |
| 4 | `precip_roll_4w_sum` | Rolling | 0.09 |
| 5 | `bucket_total_runoff` | Hydro | 0.08 |
| 6 | `sin_week` | Temporal | 0.07 |
| 7 | `temp_roll_12w_mean` | Rolling | 0.06 |
| 8 | `precip_roll_12w_sum` | Rolling | 0.05 |
| 9 | `snow_depth_cm` | Hydro | 0.04 |
| 10 | `cdd_week` | Hydro | 0.04 |
| 11 | `target_cf_lag_52w` | Lag | 0.03 |
| 12 | `radiation_roll_12w_sum` | Rolling | 0.03 |
| 13 | `precip_x_awi` | Interaction | 0.02 |
| 14 | `bucket_fast_storage` | Hydro | 0.02 |
| 15 | `soil_moisture_proxy` | Hydro | 0.02 |
| ... | ... | ... | ... |

*Importance varies by basin and horizon. Per-horizon importance available via `model.get_feature_importance(horizon=h)`.*