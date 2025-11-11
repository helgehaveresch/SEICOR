#%%
import numpy as np
import xarray as xr
import pandas as pd
import os

# %%
def add_ship_trajectory_to_plume_ds(ds_plume, filtered_ship_groups):
    group = filtered_ship_groups.get(int(ds_plume.mmsi))
    # determine plume time (prefer attr 't', fall back to times_plume median)
    t_attr = ds_plume.attrs.get("t", None)
    t_plume = pd.to_datetime(t_attr)
    # Build trajectory mask for ±15 minutes around plume time
    ship_idx_times = pd.to_datetime(group.index)
    traj_mask = (ship_idx_times >= (t_plume - pd.Timedelta(minutes=15))) & (ship_idx_times <= (t_plume + pd.Timedelta(minutes=15)))

    # Extract times, lons, lats (ensure dtype compatibility)
    ship_ais_times = ship_idx_times[traj_mask].to_numpy(dtype="datetime64[ns]")
    ship_ais_lons = group.loc[traj_mask, "Longitude"].astype(float).to_numpy()
    ship_ais_lats = group.loc[traj_mask, "Latitude"].astype(float).to_numpy()

    # Assign coordinate and variables to dataset
    ds_plume = ds_plume.assign_coords(ship_ais_times = ship_ais_times)
    ds_plume = ds_plume.assign(
        ship_ais_lons = (["ship_ais_times"], ship_ais_lons),
        ship_ais_lats = (["ship_ais_times"], ship_ais_lats),
    )

    return ds_plume

def add_plume_file_paths_to_ship_passes(ship_passes, plumes_out_dir):
    for idx, passing_ship in ship_passes.iterrows():
        plumes_out_path = os.path.join(plumes_out_dir, f"plume_{passing_ship['Plume_number']:03d}_t_{idx.strftime('%Y%m%d_%H%M%S')}_mmsi_{passing_ship['MMSI']}.nc")
        ship_passes.at[idx, 'plume_file'] = plumes_out_path
    return ship_passes

def add_insitu_to_plume_ds(ds_plume, df_insitu):
    """
    Add in-situ data to plume dataset based on time alignment.

    Args:
        ds_plume (xr.Dataset): The plume dataset with 'datetime' coordinate.
        df_insitu (pd.DataFrame): In-situ dataframe with 'time' index.

    Returns:
        xr.Dataset: The plume dataset with added in-situ data variables.
    """
    try:
        times_dt = pd.to_datetime(ds_plume["times_plume"].values)
    except Exception:
        times_dt = pd.to_datetime(ds_plume["times_plume"])
    times_index = pd.to_datetime(times_dt)
    try:
        insitu_tz = df_insitu.index.tz
    except Exception:
        insitu_tz = None
    if insitu_tz is not None:
        if getattr(times_index, 'tz', None) is None:
            try:
                times_index = times_index.tz_localize('UTC').tz_convert(insitu_tz)
            except Exception:
                times_index = times_index.tz_localize(insitu_tz)
        else:
            times_index = times_index.tz_convert(insitu_tz)
    else:
        try:
            times_index = times_index.tz_convert(None)
        except Exception:
            try:
                times_index = times_index.tz_localize(None)
            except Exception:
                pass
    if len(times_index) > 0 and times_index.isnull().any():
        first_valid = times_index[times_index.notnull()].tolist()[0]
        times_index = times_index.fillna(first_valid)
    times = times_index.to_pydatetime()
    
    in_situ_mask = (df_insitu.index >= times[0]) & (df_insitu.index <= times[-1])    
    in_situ_times = df_insitu.index[in_situ_mask]
    in_situ_no2 = df_insitu['c_no2'][in_situ_mask]
    

    L = 870  # meters

    # Interpolate n_air to times_window
    n_air = df_insitu["p_0"][in_situ_mask] * 1e2 * 6.02214076e23 / (df_insitu["T_in"][in_situ_mask] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(in_situ_times))
    # Use the pandas DatetimeIndex for reindexing (preserves tz-awareness)
    try:
        n_air_aligned = n_air_interp.reindex(times_index, method='nearest').values
    except Exception:
        # Fallback: reindex using the python datetime array
        n_air_aligned = n_air_interp.reindex(pd.DatetimeIndex(times), method='nearest').values

    in_situ_times = in_situ_times.to_numpy(dtype="datetime64[ns]")
    ds_plume = ds_plume.assign_coords(insitu_times = in_situ_times)
    ds_plume = ds_plume.assign(
        no2_insitu = (["insitu_times"], in_situ_no2),
        no_insitu = (["insitu_times"], df_insitu["n_no"][in_situ_mask]),
        nox_insitu = (["insitu_times"], df_insitu["c_nox"][in_situ_mask]),
        co2_insitu = (["insitu_times"], df_insitu["c_co2"][in_situ_mask]),
        o3_insitu = (["insitu_times"], df_insitu["c_o3"][in_situ_mask]),
        so2_insitu = (["insitu_times"], df_insitu["c_so2"][in_situ_mask]),
        n_air_insitu = (["insitu_times"], n_air),
        n_air_insitu_interp = (["window_plume"], n_air_aligned),        
        p_0_insitu = (["insitu_times"], df_insitu["p_0"][in_situ_mask]),
        T_in_insitu = (["insitu_times"], df_insitu["T_in"][in_situ_mask]),
        T_out_insitu = (["insitu_times"], df_insitu["T_out"][in_situ_mask]),
        wind_dir_insitu = (["insitu_times"], df_insitu["wind_dir"][in_situ_mask]),
        wind_speed_insitu = (["insitu_times"], df_insitu["wind_speed"][in_situ_mask]),
    )

    return ds_plume