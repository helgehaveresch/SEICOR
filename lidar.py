"""Flip-corrected LIDAR wind for the plume files (file from wind_model_meteomast_lidar_comparison.py)."""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_STATION = "Wind_1082"  # unaveraged LIDAR profiles (one every ~16 s); "Wind10_1082" are the 10-min averages


def read_lidar_corrected(path):
    """Flip-corrected LIDAR written by wind_model_meteomast_lidar_comparison.py; UTC time index (centre of the 10-min periods)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def read_lidar_raw(lidar_dir, date, station=RAW_STATION):
    """Unaveraged LIDAR profiles of date (YYMMDD) and the days before and after (windows across midnight).

    Returns a DataFrame indexed by UTC time with u_wind_{h}, v_wind_{h} for every range gate (same
    convention as the corrected file: u = FF sin(DD), v = FF cos(DD)), or None without files.
    The 180 degree direction flips are not corrected: a flip turns the sign of all samples in a
    10-min window, which does not change their standard deviation.
    """
    day = pd.to_datetime(date, format="%y%m%d")
    frames = []
    for d in (day - pd.Timedelta(days=1), day, day + pd.Timedelta(days=1)):
        path = Path(lidar_dir) / f"{station}@Y{d:%Y}_M{d:%m}_D{d:%d}.CSV.zip"
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as zf, zf.open(path.name[:-4]) as f:
            frames.append(pd.read_csv(f, skiprows=1))
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True).replace([9998.0, 9999.0], np.nan)
    time = pd.to_datetime(raw["Time and Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    heights = sorted(int(col.split(" at ")[1][:-1]) for col in raw.columns if col.startswith("Horizontal Wind Speed (m/s) at "))
    columns = {}
    for h in heights:
        speed = raw[f"Horizontal Wind Speed (m/s) at {h}m"].to_numpy(dtype=float)
        direction = np.deg2rad(raw[f"Wind Direction (deg) at {h}m"].to_numpy(dtype=float))
        columns[f"u_wind_{h}"] = speed * np.sin(direction)
        columns[f"v_wind_{h}"] = speed * np.cos(direction)
    df = pd.DataFrame(columns, index=pd.DatetimeIndex(time, name="time"))
    df = df[df.index.notna()].sort_index()
    return df[~df.index.duplicated(keep="first")]


def uv_std_in_windows(df_raw, centres, heights, averaging_minutes=10, min_samples=3):
    """Standard deviation of the unaveraged u, v in the averaging period around each centre time.

    Returns two arrays (height, time); NaN where fewer than min_samples valid profiles are in the period.
    """
    half_period = pd.Timedelta(minutes=averaging_minutes) / 2
    u_std = np.full((len(heights), len(centres)), np.nan)
    v_std = np.full_like(u_std, np.nan)
    for j, centre in enumerate(centres):
        window = df_raw[(df_raw.index >= centre - half_period) & (df_raw.index < centre + half_period)]
        for i, h in enumerate(heights):
            for out, comp in ((u_std, "u_wind"), (v_std, "v_wind")):
                name = f"{comp}_{h}"
                values = window[name].dropna().to_numpy() if name in window else np.array([])
                if len(values) >= min_samples:
                    out[i, j] = np.std(values, ddof=1)
    return u_std, v_std


def add_lidar_to_plume_ds(ds_plume, df_lidar, averaging_minutes=10, df_lidar_raw=None):
    """Add the LIDAR records whose averaging period overlaps the plume window; lidar_available = 0 if there are none.

    With the unaveraged profiles (df_lidar_raw from read_lidar_raw) the uncertainty of u and v is added
    as the standard deviation of the profiles in each 10-min averaging period."""
    ds_plume.attrs["lidar_available"] = 0
    if df_lidar is None:
        return ds_plume
    times = pd.to_datetime(ds_plume["times_plume"].values)
    half_period = pd.Timedelta(minutes=averaging_minutes) / 2
    selected = df_lidar[(df_lidar.index + half_period >= times.min()) & (df_lidar.index - half_period <= times.max())]
    if selected.empty:
        return ds_plume
    heights = sorted(int(col.rsplit("_", 1)[1]) for col in selected.columns if col.startswith("wind_speed_") and "std" not in col)
    stack = lambda name: np.stack([selected[f"{name}_{h}"].to_numpy() for h in heights])
    ds_plume = ds_plume.assign_coords(lidar_times=selected.index.to_numpy(dtype="datetime64[ns]"),
                                      lidar_height=("lidar_height", heights, {"units": "m"}))
    ds_plume = ds_plume.assign(
        wind_speed_lidar=(["lidar_height", "lidar_times"], stack("wind_speed"), {"units": "m s-1"}),
        wind_speed_std_lidar=(["lidar_height", "lidar_times"], stack("wind_speed_std"), {"units": "m s-1"}),
        wind_dir_lidar=(["lidar_height", "lidar_times"], stack("wind_dir"), {"units": "degree", "long_name": "wind direction, 180 degree flips corrected"}),
        lidar_direction_flipped=(["lidar_times"], selected["direction_flipped"].to_numpy(dtype=bool)),
    )
    if df_lidar_raw is not None:
        u_std, v_std = uv_std_in_windows(df_lidar_raw, selected.index, heights, averaging_minutes)
        ds_plume = ds_plume.assign(
            u_wind_lidar=(["lidar_height", "lidar_times"], stack("u_wind"), {"units": "m s-1", "long_name": "u = FF sin(DD), flips corrected"}),
            v_wind_lidar=(["lidar_height", "lidar_times"], stack("v_wind"), {"units": "m s-1", "long_name": "v = FF cos(DD), flips corrected"}),
            u_wind_uncertainty_lidar=(["lidar_height", "lidar_times"], u_std,
                                      {"units": "m s-1", "long_name": "uncertainty of u: standard deviation of the unaveraged profiles in the 10-min period"}),
            v_wind_uncertainty_lidar=(["lidar_height", "lidar_times"], v_std,
                                      {"units": "m s-1", "long_name": "uncertainty of v: standard deviation of the unaveraged profiles in the 10-min period"}),
        )
    ds_plume["lidar_times"].attrs["long_name"] = "centre of the 10-min averaging period"
    ds_plume.attrs["lidar_available"] = 1
    return ds_plume
