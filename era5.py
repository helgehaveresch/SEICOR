"""ERA5 model-level profile above the ship corridor for the plume files (files from download_era_5.py)."""
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import xarray as xr

RD = 287.06            # J/(kg K), gas constant of dry air (IFS)
RV_OVER_RD_M1 = 0.609133  # virtual temperature: Tv = T * (1 + 0.609133 * q)
G0 = 9.80665           # m/s2


def _read_grib(path, start, end):
    """Fields from start to end (time-sorted file): {(shortName, level, time): 2D array} and the grid (lat, lon, pv)."""
    fields, grid = {}, None
    first, last = int(start.strftime("%Y%m%d%H%M")), int(end.strftime("%Y%m%d%H%M"))
    with open(path, "rb") as file:
        while (msg := eccodes.codes_grib_new_from_file(file)) is not None:
            try:
                stamp = eccodes.codes_get(msg, "dataDate") * 10000 + eccodes.codes_get(msg, "dataTime")
                if stamp > last:
                    break
                if stamp < first:
                    continue
                if grid is None:
                    n_lat, n_lon = eccodes.codes_get(msg, "Nj"), eccodes.codes_get(msg, "Ni")
                    lat = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees") - eccodes.codes_get(msg, "jDirectionIncrementInDegrees") * np.arange(n_lat)
                    lon = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees") + eccodes.codes_get(msg, "iDirectionIncrementInDegrees") * np.arange(n_lon)
                    grid = (lat, lon, eccodes.codes_get_array(msg, "pv"))
                time = pd.to_datetime(str(stamp), format="%Y%m%d%H%M")
                key = (eccodes.codes_get(msg, "shortName"), eccodes.codes_get(msg, "level"), time)
                fields[key] = eccodes.codes_get_values(msg).reshape(len(grid[0]), len(grid[1]))
            finally:
                eccodes.codes_release(msg)
    return fields, grid


def read_era5_corridor(era5_dir, date, location):
    """Hourly ERA5 profile above location = (lat, lon) from 00 UTC of date (YYMMDD) to 00 UTC of the next day.

    The height above ground of each model level is computed per grid column from T, q, lnsp and the
    level coefficients stored in the GRIB files (IFS hypsometric equation), then all fields are
    interpolated bilinearly along the model level. Returns a dataset with dimensions (time, level).
    """
    era5_dir = Path(era5_dir)
    day = pd.to_datetime(date, format="%y%m%d")
    next_day = day + pd.Timedelta(days=1)
    times = pd.date_range(day, next_day, freq="1h")

    fields, grid = {}, None
    for month in dict.fromkeys((f"{day:%Y%m}", f"{next_day:%Y%m}")):
        for prefix in ("era5_ml", "era5_lnsp_z"):
            path = era5_dir / f"{prefix}_{month}.grib"
            if month != f"{day:%Y%m}" and not path.exists():
                continue  # last downloaded month: passes in the last hour of the day get no ERA5 profile
            new_fields, grid = _read_grib(path, day, next_day)
            fields.update(new_fields)
    times = times[[("lnsp", 1, t) in fields for t in times]]

    lat, lon, pv = grid
    half_a, half_b = pv[:len(pv) // 2], pv[len(pv) // 2:]
    levels = np.array(sorted({level for name, level, _ in fields if name == "t"}))  # index 0 = highest level

    def stack(name):
        return np.stack([np.stack([fields[(name, level, t)] for level in levels]) for t in times])

    temperature, humidity = stack("t"), stack("q")
    surface_pressure = np.stack([np.exp(fields[("lnsp", 1, t)]) for t in times])[:, None]
    surface_geopotential = np.stack([fields[("z", 1, t)] for t in times])

    p_lower = half_a[levels][None, :, None, None] + half_b[levels][None, :, None, None] * surface_pressure
    p_upper = half_a[levels - 1][None, :, None, None] + half_b[levels - 1][None, :, None, None] * surface_pressure
    virtual_temperature = temperature * (1 + RV_OVER_RD_M1 * humidity)
    log_ratio = np.log(p_lower / p_upper)
    layer_geopotential = RD * virtual_temperature * log_ratio
    below = np.cumsum(layer_geopotential[:, ::-1], axis=1)[:, ::-1] - layer_geopotential  # up to the lower half level
    alpha = 1 - p_upper / (p_lower - p_upper) * log_ratio
    height = (below + alpha * RD * virtual_temperature) / G0

    dims = ("time", "level", "latitude", "longitude")
    profile = xr.Dataset(
        {
            "height": (dims, height, {"units": "m", "long_name": "height above the model surface"}),
            "pressure": (dims, 0.5 * (p_lower + p_upper), {"units": "Pa", "long_name": "air pressure"}),
            "t": (dims, temperature, {"units": "K", "long_name": "air temperature"}),
            "q": (dims, humidity, {"units": "kg kg-1", "long_name": "specific humidity"}),
            "u": (dims, stack("u"), {"units": "m s-1", "long_name": "eastward wind"}),
            "v": (dims, stack("v"), {"units": "m s-1", "long_name": "northward wind"}),
            "surface_altitude": (("time", "latitude", "longitude"), surface_geopotential / G0, {"units": "m", "long_name": "model surface altitude"}),
        },
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )
    profile = profile.sortby("latitude").interp(latitude=location[0], longitude=location[1]).drop_vars(["latitude", "longitude"])
    return add_wind_uncertainty(profile, era5_dir, day, next_day, location)


def add_wind_uncertainty(profile, era5_dir, start, end, location):
    """Add u_uncertainty, v_uncertainty: the ensemble spread of the ERA5 EDA (era5_ml_spread_YYYYMM.grib).

    The spread is 3-hourly on a coarser grid; it is interpolated bilinearly to location and linearly
    to the hourly times of the profile, on the same model levels. Without spread file the profile is
    returned unchanged.
    """
    fields, grid = {}, None
    for month in dict.fromkeys((f"{start:%Y%m}", f"{end:%Y%m}")):
        path = Path(era5_dir) / f"era5_ml_spread_{month}.grib"
        if path.exists():
            new_fields, new_grid = _read_grib(path, start, end)
            fields.update(new_fields)
            grid = grid or new_grid
    if not fields:
        return profile
    lat, lon, _ = grid
    times = sorted({t for name, _, t in fields if name == "u"})
    levels = profile["level"].values
    if not all((name, level, t) in fields for name in ("u", "v") for level in levels for t in times):
        print(f"ERA5 spread incomplete for {start:%Y-%m-%d}: no ERA5 wind uncertainty")
        return profile
    spread = xr.Dataset(
        {f"{name}_uncertainty": (("time", "level", "latitude", "longitude"),
                                 np.stack([np.stack([fields[(name, level, t)] for level in levels]) for t in times]),
                                 {"units": "m s-1", "long_name": f"uncertainty of {name}: ensemble spread (standard deviation) of the ERA5 EDA"})
         for name in ("u", "v")},
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )
    spread = spread.sortby("latitude").interp(latitude=location[0], longitude=location[1]).drop_vars(["latitude", "longitude"])
    return profile.merge(spread.interp(time=profile["time"]))


def add_era5_to_plume_ds(ds_plume, era5_profile, max_height_m):
    """Add the ERA5 profile at the time of the ship pass (linear in time), levels up to max_height_m above ground."""
    if era5_profile is None:
        ds_plume.attrs["era5_available"] = 0
        return ds_plume
    pass_time = pd.to_datetime(ds_plume.attrs["t"], utc=True).tz_localize(None)
    profile = era5_profile.interp(time=pass_time).drop_vars("time")
    profile = profile.isel(level=np.flatnonzero(profile["height"].values <= max_height_m))
    if profile.sizes["level"] == 0:  # outside the available hours
        ds_plume.attrs["era5_available"] = 0
        return ds_plume
    profile = profile.rename(level="era5_level")
    profile = profile.rename({name: f"era5_{name}" for name in profile.data_vars})
    ds_plume = ds_plume.assign(profile.data_vars)
    ds_plume.attrs["era5_available"] = 1
    return ds_plume
