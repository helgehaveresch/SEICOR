"""Flip-corrected LIDAR wind for the plume files (file from wind_model_meteomast_lidar_comparison.py)."""
import numpy as np
import pandas as pd


def read_lidar_corrected(path):
    """Flip-corrected LIDAR written by wind_model_meteomast_lidar_comparison.py; UTC time index (centre of the 10-min periods)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def add_lidar_to_plume_ds(ds_plume, df_lidar, averaging_minutes=10):
    """Add the LIDAR records whose averaging period overlaps the plume window; lidar_available = 0 if there are none."""
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
    ds_plume["lidar_times"].attrs["long_name"] = "centre of the 10-min averaging period"
    ds_plume.attrs["lidar_available"] = 1
    return ds_plume
