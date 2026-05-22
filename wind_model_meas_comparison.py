#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from SEICOR.in_situ import read_in_situ
import re
import matplotlib.dates as mdates
from pathlib import Path
import sys
import logging
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import xarray as xr
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from typing import Sequence
finkenwerder_path = r"D:\weatherstations\Finkenwerder_Airport\weatherdata_hourly.csv"
mittelnkirchen_path = r"D:\weatherstations\Mittelnkirchen-Hohenfelde\weatherdata_hourly.csv"
york_path = r"D:\weatherstations\York-Moorende\weatherdata_hourly.csv"
billwerder_dir = r"D:\weatherstations\Billwerder"
horiba_dir = r"D:\InSitu"
airpointer_dir = r"D:\InSitu\Messdaten"
billwerder_path = r"D:\weatherstations\Billwerder\weatherdata_hourly.csv"
rissen_path = r"D:\weatherstations\Rissen\weatherdata_hourly.csv"


def find_closest_grid_indices(ds: xr.Dataset, lat_pt: float, lon_pt: float, z_levels: Sequence[float]):
    """Find indices (bottom_top, south_north, west_east) closest to a point.

    Parameters
    - ds: xarray Dataset containing `latitude(south_north, west_east)`,
          `longitude(south_north, west_east)`, and `height(bottom_top, south_north, west_east)`.
    - lat_pt, lon_pt: target geographic coordinates in degrees.
    - z_levels: sequence of target heights (m) for which to find closest bottom_top index.

    Returns a dict with keys:
    - 'south_north', 'west_east': integer grid indices of the horizontal cell closest to lat/lon
    - 'bottom_top_indices': list of integers (one per requested z in `z_levels`)
    - 'grid_lat', 'grid_lon': the lat/lon at the chosen cell
    - 'grid_heights': 1D array of heights at that (bottom_top,) column
    """
    import numpy as np

    # extract 2D lat/lon arrays
    lat_grid = ds['latitude'].values  # shape (south_north, west_east)
    lon_grid = ds['longitude'].values

    # meters-per-degree approx at target latitude
    lat_rad = np.deg2rad(float(lat_pt))
    mperlon = 111320.0 * np.cos(lat_rad)
    mperlat = 111132.0

    # compute horizontal distance in meters to every grid cell
    dy = (lat_grid - float(lat_pt)) * mperlat
    dx = (lon_grid - float(lon_pt)) * mperlon
    dist = np.hypot(dy, dx)

    # find nearest horizontal cell (flat) and also compute surrounding indices along each axis
    flat_idx = int(np.argmin(dist))
    iy, ix = np.unravel_index(flat_idx, lat_grid.shape)

    # derive 1D slices along grid axes at the chosen column/row
    lat_along_sn = lat_grid[:, ix]  # varying with south_north
    lon_along_we = lon_grid[iy, :]  # varying with west_east

    # find surrounding south_north indices that bracket lat_pt
    sn_pos = int(np.searchsorted(lat_along_sn, float(lat_pt)))
    sn_lo = max(0, sn_pos - 1)
    sn_hi = min(lat_along_sn.size - 1, sn_pos)

    # find surrounding west_east indices that bracket lon_pt
    we_pos = int(np.searchsorted(lon_along_we, float(lon_pt)))
    we_lo = max(0, we_pos - 1)
    we_hi = min(lon_along_we.size - 1, we_pos)

    # extract vertical column of heights at that cell
    heights = ds['height'].values[:, iy, ix]  # shape (bottom_top,)

    # for each requested z, find the lower/upper bottom_top indices that bracket z
    bottom_pairs = []
    for zt in z_levels:
        pos = int(np.searchsorted(heights, float(zt)))
        lo = max(0, pos - 1)
        hi = min(heights.size - 1, pos)
        bottom_pairs.append((int(lo), int(hi)))

    return {
        'south_north': int(iy),
        'west_east': int(ix),
        'south_north_pair': (int(sn_lo), int(sn_hi)),
        'west_east_pair': (int(we_lo), int(we_hi)),
        'bottom_top_pairs': bottom_pairs,
        'grid_lat': float(lat_grid[iy, ix]),
        'grid_lon': float(lon_grid[iy, ix]),
        'grid_heights': heights,
    }

def load_and_stack_csvs(dir_path, prefix, recursive=False, ):
    """
    Read all CSV files in dir_path starting with `prefix` and stack them into a single DataFrame.
    - dir_path: folder to search
    - prefix: filename prefix to match (case-sensitive)
    - recursive: search subfolders if True
    - parse_time: parse a 'time' column as datetime if present
    - read_csv_kwargs: extra kwargs passed to pd.read_csv
    Returns a pandas.DataFrame (empty if no files found or all fail).
    """

    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = f"{prefix}*.csv"
    files = sorted(path.rglob(pattern) if recursive else path.glob(pattern))
    files = [p for p in files if p.is_file()]

    logging.info("Found %d file(s) with prefix %r in %s", len(files), prefix, dir_path)
    dfs = []
    for fp in files:
        try:
            read_kwargs = {"sep": ";", "decimal": ","}
            # optionally set encoding if needed, e.g. encoding="latin-1"
            df = pd.read_csv(fp, **read_kwargs)
            dfs.append(df)
        except Exception as e:
            logging.warning("Failed to read %s: %s", fp, e)

    if not dfs:
        logging.info("No dataframes loaded; returning empty DataFrame")
        return pd.DataFrame()

    stacked = pd.concat(dfs, ignore_index=True, sort=False)
    logging.info("Stacked dataframe shape: %s", stacked.shape)
    return stacked

def load_weather_data_csv(file_path):
    """
    Load weather data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing weather data.

    Returns:
    pd.DataFrame: A DataFrame containing the weather data.
    """
    try:
        weather_data = pd.read_csv(file_path)
        return weather_data
    except Exception as e:
        print(f"An error occurred while loading the weather data: {e}")
        return None

def calc_u_v_wind(df, variable_speed='wspd', variable_direction='wdir', variable_speed_list = None, variable_dir_list = None, output_list_u=None, output_list_v = None, convert_speed_to_mps=False):
    
    if variable_speed_list is None:
        if convert_speed_to_mps:
            df[variable_speed] = df[variable_speed]/3.6
        u = df[variable_speed] * np.sin(np.deg2rad(df[variable_direction]))
        v = df[variable_speed] * np.cos(np.deg2rad(df[variable_direction]))
        df['u_wind'] = u
        df['v_wind'] = v
    
    else:
        for i, var_speed in enumerate(variable_speed_list):
            var_dir = variable_dir_list[i]
            out_u = output_list_u[i]
            out_v = output_list_v[i]
            if convert_speed_to_mps:
                df[var_speed] = df[var_speed]/3.6
            u = df[var_speed] * np.sin(np.deg2rad(df[var_dir]))
            v = df[var_speed] * np.cos(np.deg2rad(df[var_dir]))
            df[out_u] = u
            df[out_v] = v
    return df

def calc_speed_dir_wind(df, variable_u='u_wind', variable_v='v_wind', variable_u_list = None, variable_v_list = None, output_list_speed=None, output_list_dir = None ):
    if variable_u_list is None:
        speed = np.sqrt(df[variable_u]**2 + df[variable_v]**2)
        direction = (np.rad2deg(np.arctan2(df[variable_u], df[variable_v])) + 360) % 360
        df['wind_speed'] = speed
        df['wind_dir'] = direction
    else:
        for i, var_u in enumerate(variable_u_list):
            var_v = variable_v_list[i]
            out_speed = output_list_speed[i]
            out_dir = output_list_dir[i]
            speed = np.sqrt(df[var_u]**2 + df[var_v]**2)
            direction = (np.rad2deg(np.arctan2(df[var_u], df[var_v])) + 360) % 360
            df[out_speed] = speed
            df[out_dir] = direction
    return df

def calculate_rmse(a, b):
    """Return RMSE between arrays `a` and `b`, ignoring non-finite values."""
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((a[mask] - b[mask])**2)))

def calculate_rmse_circular(a, b):
    """Return RMSE between arrays `a` and `b`, ignoring non-finite values."""
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(((a[mask] - b[mask]+180)%360-180)**2)))

def read_all_horiba_(dir):
    df_list = []
    for date in pd.date_range(start="2025-06-01", end="2025-06-30"):
        date_str = date.strftime("%y%m%d")
        df = read_in_situ(dir, date_str)
        if not df.empty:
            df_list.append(df)
    df_final = pd.concat(df_list)
    return df_final

def ensure_utc_time_column(df, col="time"):
    if col in df.columns:
        ts = pd.to_datetime(df[col], errors="coerce")
        # if tz-naive -> localize to UTC, else convert to UTC
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df[col] = ts
    return df

def read_uni_hamburg_wind_data_single_txt(file_path):
    from pathlib import Path
    path = Path(file_path)
    vals = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            vals.append(line)
    return np.array(vals, dtype=float)

def read_all_uni_hamburg_wind_data(dir_path, station_name, time, 
                                   variab_list = ["wind_speed", "wind_dir", "wind_max", "solar_rad"],
                                   variab_name = {  "wind_speed": "FF",
                                                    "wind_dir": "DD",
                                                    "wind_max": "FB",
                                                    "solar_rad": "G",}):

    for var in variab_list:
        file_path = f"{dir_path}/{station_name}{variab_name[var]}_202503010000-202510312359.txt"
        data = read_uni_hamburg_wind_data_single_txt(file_path)
        df_temp = pd.DataFrame({var: data}, index=time)
        if var == variab_list[0]:
            df_final = df_temp
        else:
            df_final = pd.merge(df_final, df_temp, left_index=True, right_index=True)
    return df_final

def read_sequential_wrfwind_files(
    folder_path: str = r"Q:\BREDOM\SEICOR\wind_model_v1_1",
    prefix: str = "seicor_wrfmeteo_v1-1_d05",
    var_name: str | None = None,
    agg_func=None,
    file_pattern: tuple = ('.nc',),
    target_lat: float | None = None,
    target_lon: float | None = None,
    target_zs: Sequence[float] | None = None,
    z_profile = np.arange(0, 500, 10)
):
    """Read files in `folder_path` whose basenames start with `prefix` one-by-one.

    For each file the reader opens the file, extracts `var_name` (or the first variable
    if `var_name` is None), applies `agg_func` if provided, appends the result(s) to a
    1-D numpy array, and closes the file before moving to the next.

    Parameters
    - folder_path: directory containing files
    - prefix: filename prefix to match
    - var_name: variable name to extract from netCDF files; if None the first variable is used
    - agg_func: optional callable that takes a numpy array and returns a scalar to append
                If None, the full flattened variable data are appended elementwise.
    - file_pattern: tuple of file extensions to consider (default '.nc')

    Returns
    - numpy.ndarray 1D of concatenated values

    Example
    >>> arr = read_sequential_wrfwind_files(var_name='U10')
    """
    import os
    import numpy as np
    import xarray as _xr

    folder = os.path.expanduser(folder_path)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    names = sorted([n for n in os.listdir(folder) if n.startswith(prefix)])
    
    times_list = []
    u1_list = []
    u2_list = []
    u3_list = []
    u4_list = []
    v1_list = []
    v2_list = []
    v3_list = []
    v4_list = []
    wspd_1_list = []
    wspd_2_list = []
    wspd_3_list = []
    wspd_4_list = []
    wdir_1_list = []
    wdir_2_list = []
    wdir_3_list = []
    wdir_4_list = []
    u_prof_list = []
    v_prof_list = []
    wdir_prof_list = []
    wspd_prof_list = []
    u10m_list = []
    v10m_list = []
    pblh_list = []
    p_list = []
    sw_down_list = []
    t_list = []
    t_pot_list = []
    tke_list = []

    sn_we_indices = None

    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in file_pattern:
            ds = None
            ds = _xr.open_dataset(path)
            
            tvals = ds['time'].values

            # determine surrounding horizontal and vertical indices using helper
            info = find_closest_grid_indices(ds, target_lat, target_lon, target_zs)
            
            sn_lo, sn_hi = info['south_north_pair']
            we_lo, we_hi = info['west_east_pair']
            bt_pairs = info['bottom_top_pairs']
            
            heights = info['grid_heights']
            
            info_prof = find_closest_grid_indices(ds, target_lat, target_lon, z_profile)
            bt_pairs_prof = info_prof['bottom_top_pairs']


            # prepare arrays for interpolated time series for this file
            # determine time length
            u_var = -ds['u']
            v_var = -ds['v']
            u_10m_var = -ds['u10']
            v_10m_var = -ds['v10']
            pblh_var = ds['pblh']
            p_var = ds['pressure']
            sw_down_var = ds['swdown']
            t_var = ds['temperature']
            t_pot_var = ds['theta']
            tke_var = ds['tke']

            nt = int(u_var.sizes['time'])

            u_ts_per_z = [np.zeros(nt, dtype=float) for _ in target_zs]
            v_ts_per_z = [np.zeros(nt, dtype=float) for _ in target_zs]
            u_prof_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            v_prof_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            p_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            t_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            t_pot_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            tke_ts_per_z = np.zeros((nt, len(z_profile)), dtype=float)
            u10m_ts = np.zeros(nt, dtype=float)
            v10m_ts = np.zeros(nt, dtype=float)
            sw_down_ts = np.zeros(nt, dtype=float)
            pblh_ts = np.zeros(nt, dtype=float)

            # helper to fetch value safely by positional indexing
            def _fetch(arr, t_idx, bt_idx, sn_idx_local, we_idx_local, dims):
                a = arr
                # collect axes to take with their original axis indices
                ops = []
                if 'time' in dims:
                    ops.append((dims.index('time'), t_idx))
                if 'bottom_top' in dims:
                    ops.append((dims.index('bottom_top'), bt_idx))
                if 'south_north' in dims:
                    ops.append((dims.index('south_north'), sn_idx_local))
                if 'west_east' in dims:
                    ops.append((dims.index('west_east'), we_idx_local))
                # perform takes in descending axis order so earlier takes do not shift later axis indices
                ops.sort(key=lambda x: x[0], reverse=True)
                for axis_idx, take_idx in ops:
                    a = np.take(a, int(take_idx), axis=axis_idx)
                return float(a)

            u_arr = u_var.values
            v_arr = v_var.values
            u10m_arr = u_10m_var.values
            v10m_arr = v_10m_var.values
            pblh_arr = pblh_var.values
            p_arr = p_var.values
            sw_down_arr = sw_down_var.values
            t_arr = t_var.values
            t_pot_arr = t_pot_var.values
            tke_arr = tke_var.values
            u_dims = list(u_var.dims)
            v_dims = list(v_var.dims)
            u10m_dims = list(u_10m_var.dims)
            v10m_dims = list(v_10m_var.dims)
            pblh_dims = list(pblh_var.dims)
            p_dims = list(p_var.dims)
            sw_down_dims = list(sw_down_var.dims)
            t_dims = list(t_var.dims)
            t_pot_dims = list(t_pot_var.dims)
            tke_dims = list(tke_var.dims)


            # compute fractional weights in horizontal local coords
            lat1 = ds['latitude'].values[sn_lo, we_lo]
            lat2 = ds['latitude'].values[sn_hi, we_lo]
            lon1 = ds['longitude'].values[sn_lo, we_lo]
            lon2 = ds['longitude'].values[sn_lo, we_hi]
            dlat = lat2 - lat1 if (lat2 - lat1) != 0 else 1.0
            dlon = lon2 - lon1 if (lon2 - lon1) != 0 else 1.0
            t_lat = (float(target_lat) - lat1) / dlat
            t_lon = (float(target_lon) - lon1) / dlon
            t_lat = np.clip(t_lat, 0.0, 1.0)
            t_lon = np.clip(t_lon, 0.0, 1.0)

            # loop over times
            for ti in range(nt):
                for zi, (bt_lo, bt_hi) in enumerate(bt_pairs):
                    # for each corner get u/v at bottom_lo and bottom_hi
                    def _get_corner(arr_list, dims_list, bt_idx_local):
                        var_list = []
                        for arr, dims in zip(arr_list, dims_list):
                            v00_var = _fetch(arr, ti, bt_idx_local, sn_lo, we_lo, dims)
                            v10_var = _fetch(arr, ti, bt_idx_local, sn_hi, we_lo, dims)
                            v01_var = _fetch(arr, ti, bt_idx_local, sn_lo, we_hi, dims)
                            v11_var = _fetch(arr, ti, bt_idx_local, sn_hi, we_hi, dims)
                            val_var = (1 - t_lon) * (1 - t_lat) * v00_var + t_lon * (1 - t_lat) * v01_var + (1 - t_lon) * t_lat * v10_var + t_lon * t_lat * v11_var
                            var_list.append(val_var)
                        return var_list


                    if bt_lo == bt_hi:
                        val_u, val_v = _get_corner([u_arr, v_arr], [u_dims, v_dims], bt_lo)
                    else:
                        val_lo_u, val_lo_v = _get_corner([u_arr, v_arr], [u_dims, v_dims], bt_lo)
                        val_hi_u, val_hi_v = _get_corner([u_arr, v_arr], [u_dims, v_dims], bt_hi)
                        h_lo = heights[bt_lo]
                        h_hi = heights[bt_hi]
                        frac = (target_zs[zi] - h_lo) / (h_hi - h_lo)
                        frac = np.clip(frac, 0.0, 1.0)
                        val_u = (1 - frac) * val_lo_u + frac * val_hi_u
                        val_v = (1 - frac) * val_lo_v + frac * val_hi_v

                    u_ts_per_z[zi][ti] = val_u
                    v_ts_per_z[zi][ti] = val_v

                for zi, (bt_lo, bt_hi) in enumerate(bt_pairs_prof):
                    if bt_lo == bt_hi:
                        val_u_prof, val_v_prof, val_p, val_t, val_t_pot, val_tke = _get_corner([u_arr, v_arr, p_arr, t_arr, t_pot_arr, tke_arr], [u_dims, v_dims, p_dims, t_dims, t_pot_dims, tke_dims], bt_lo)
                    else:
                        val_lo_u_prof, val_lo_v_prof, val_lo_p, val_lo_t, val_lo_t_pot, val_lo_tke = _get_corner([u_arr, v_arr, p_arr, t_arr, t_pot_arr, tke_arr], [u_dims, v_dims, p_dims, t_dims, t_pot_dims, tke_dims], bt_lo)
                        val_hi_u_prof, val_hi_v_prof, val_hi_p, val_hi_t, val_hi_t_pot, val_hi_tke = _get_corner([u_arr, v_arr, p_arr, t_arr, t_pot_arr, tke_arr], [u_dims, v_dims, p_dims, t_dims, t_pot_dims, tke_dims], bt_hi)
                        h_lo = heights[bt_lo]
                        h_hi = heights[bt_hi]
                        frac = (z_profile[zi] - h_lo) / (h_hi - h_lo)
                        frac = np.clip(frac, 0.0, 1.0)
                        val_u_prof = (1 - frac) * val_lo_u_prof + frac * val_hi_u_prof
                        val_v_prof = (1 - frac) * val_lo_v_prof + frac * val_hi_v_prof
                        val_p = (1 - frac) * val_lo_p + frac * val_hi_p
                        val_t = (1 - frac) * val_lo_t + frac * val_hi_t
                        val_t_pot = (1 - frac) * val_lo_t_pot + frac * val_hi_t_pot
                        val_tke = (1 - frac) * val_lo_tke + frac * val_hi_tke

                    u_prof_ts_per_z[ti, zi] = val_u_prof
                    v_prof_ts_per_z[ti, zi] = val_v_prof
                    p_ts_per_z[ti, zi] = val_p
                    t_ts_per_z[ti, zi] = val_t
                    t_pot_ts_per_z[ti, zi] = val_t_pot
                    tke_ts_per_z[ti, zi] = val_tke

                val_u10m, val_v10m, val_sw_down, val_pblh  = _get_corner([u10m_arr, v10m_arr, sw_down_arr, pblh_arr], [u10m_dims, v10m_dims, sw_down_dims, pblh_dims], 0)
                u10m_ts[ti] = val_u10m
                v10m_ts[ti] = val_v10m
                sw_down_ts[ti] = val_sw_down
                pblh_ts[ti] = val_pblh

            wind_speed_prof_ts_per_z = np.sqrt(u_prof_ts_per_z**2 + v_prof_ts_per_z**2)
            wind_dir_ts_per_z = (np.rad2deg(np.arctan2(u_prof_ts_per_z, v_prof_ts_per_z)) + 360) % 360
            # append arrays for this file
            times_list.append(np.asarray(tvals).ravel() if tvals is not None else np.arange(nt))
            u_prof_list.append(u_prof_ts_per_z)
            v_prof_list.append(v_prof_ts_per_z)
            wdir_prof_list.append(wind_dir_ts_per_z)
            wspd_prof_list.append(wind_speed_prof_ts_per_z)
            u10m_list.append(u10m_ts)
            v10m_list.append(v10m_ts)
            pblh_list.append(pblh_ts)
            p_list.append(p_ts_per_z)
            t_list.append(t_ts_per_z)
            t_pot_list.append(t_pot_ts_per_z)
            tke_list.append(tke_ts_per_z)
            sw_down_list.append(sw_down_ts)
            for zi in range(len(target_zs)):
                u1_list.append(u_ts_per_z[zi]) if zi == 0 else None
                u2_list.append(u_ts_per_z[zi]) if zi == 1 else None
                u3_list.append(u_ts_per_z[zi]) if zi == 2 else None
                u4_list.append(u_ts_per_z[zi]) if zi == 3 else None
                v1_list.append(v_ts_per_z[zi]) if zi == 0 else None
                v2_list.append(v_ts_per_z[zi]) if zi == 1 else None
                v3_list.append(v_ts_per_z[zi]) if zi == 2 else None
                v4_list.append(v_ts_per_z[zi]) if zi == 3 else None

                wdir_1_list.append( (np.rad2deg(np.arctan2(u_ts_per_z[0], v_ts_per_z[0])) + 360) % 360 ) if zi == 0 else None
                wdir_2_list.append( (np.rad2deg(np.arctan2(u_ts_per_z[1], v_ts_per_z[1])) + 360) % 360 ) if zi == 1 else None
                wdir_3_list.append( (np.rad2deg(np.arctan2(u_ts_per_z[2], v_ts_per_z[2])) + 360) % 360 ) if zi == 2 else None
                wdir_4_list.append( (np.rad2deg(np.arctan2(u_ts_per_z[3], v_ts_per_z[3])) + 360) % 360 ) if zi == 3 else None
                wspd_1_list.append( np.sqrt(u_ts_per_z[0]**2 + v_ts_per_z[0]**2) ) if zi == 0 else None
                wspd_2_list.append( np.sqrt(u_ts_per_z[1]**2 + v_ts_per_z[1]**2) ) if zi == 1 else None
                wspd_3_list.append( np.sqrt(u_ts_per_z[2]**2 + v_ts_per_z[2]**2) ) if zi == 2 else None
                wspd_4_list.append( np.sqrt(u_ts_per_z[3]**2 + v_ts_per_z[3]**2) ) if zi == 3 else None


                if ds is not None:
                    ds.close()

    # flatten and concatenate into 1D arrays
    def _concat(lists):
        if not lists:
            return np.array([], dtype=float)
        return np.concatenate([np.asarray(x).ravel() for x in lists])
    
    def _concat_2d(lists):
        if not lists:
            return np.empty((0, len(z_profile)), dtype=float)
        return np.concatenate([np.asarray(x) for x in lists], axis=0)
    
    times_all = _concat(times_list)

    # helper: ensure an array has the same length as times_all; if empty, fill with NaN
    def _ensure_length_or_nan(arr):
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return np.full(times_all.size, np.nan)
        # if lengths differ but not zero, try to trim or pad with NaN to match times
        if arr.size != times_all.size:
            if arr.size > times_all.size:
                return arr[: times_all.size]
            else:
                pad = np.full(times_all.size - arr.size, np.nan)
                return np.concatenate([arr, pad])
        return arr
    def _ensure_length_or_nan_2d(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.full((times_all.size, arr.shape[1]), np.nan)
        if arr.shape[0] != times_all.size:
            if arr.shape[0] > times_all.size:
                return arr[: times_all.size, :]
            else:
                pad = np.full((times_all.size - arr.shape[0], arr.shape[1]), np.nan)
                return np.concatenate([arr, pad], axis=0)
        return arr


    u1_arr = _ensure_length_or_nan(_concat(u1_list))
    u2_arr = _ensure_length_or_nan(_concat(u2_list))
    u3_arr = _ensure_length_or_nan(_concat(u3_list))
    u4_arr = _ensure_length_or_nan(_concat(u4_list))
    v1_arr = _ensure_length_or_nan(_concat(v1_list))
    v2_arr = _ensure_length_or_nan(_concat(v2_list))
    v3_arr = _ensure_length_or_nan(_concat(v3_list))
    v4_arr = _ensure_length_or_nan(_concat(v4_list))
    wdir_1_arr = _ensure_length_or_nan(_concat(wdir_1_list))
    wdir_2_arr = _ensure_length_or_nan(_concat(wdir_2_list))
    wdir_3_arr = _ensure_length_or_nan(_concat(wdir_3_list))
    wdir_4_arr = _ensure_length_or_nan(_concat(wdir_4_list))
    wspd_1_arr = _ensure_length_or_nan(_concat(wspd_1_list))
    wspd_2_arr = _ensure_length_or_nan(_concat(wspd_2_list))
    wspd_3_arr = _ensure_length_or_nan(_concat(wspd_3_list))
    wspd_4_arr = _ensure_length_or_nan(_concat(wspd_4_list))
    u_prof_arr = _ensure_length_or_nan_2d(_concat_2d(u_prof_list))
    v_prof_arr = _ensure_length_or_nan_2d(_concat_2d(v_prof_list))
    wdir_prof_arr = _ensure_length_or_nan_2d(_concat_2d(wdir_prof_list))
    wspd_prof_arr = _ensure_length_or_nan_2d(_concat_2d(wspd_prof_list))
    u10m_arr = _ensure_length_or_nan(_concat(u10m_list))
    v10m_arr = _ensure_length_or_nan(_concat(v10m_list))
    pblh_arr = _ensure_length_or_nan(_concat(pblh_list))
    p_arr = _ensure_length_or_nan_2d(_concat_2d(p_list))
    t_arr = _ensure_length_or_nan_2d(_concat_2d(t_list))
    t_pot_arr = _ensure_length_or_nan_2d(_concat_2d(t_pot_list))
    tke_arr = _ensure_length_or_nan_2d(_concat_2d(tke_list))
    sw_down_arr = _ensure_length_or_nan(_concat(sw_down_list))

    out_ds = _xr.Dataset(
        data_vars={
            'u1': (('time',), u1_arr),
            'u2': (('time',), u2_arr),
            'u3': (('time',), u3_arr),
            'u4': (('time',), u4_arr),
            'v1': (('time',), v1_arr),
            'v2': (('time',), v2_arr),
            'v3': (('time',), v3_arr),
            'v4': (('time',), v4_arr),
            'wspd_1': (('time',), wspd_1_arr),
            'wspd_2': (('time',), wspd_2_arr),  
            'wspd_3': (('time',), wspd_3_arr),
            'wspd_4': (('time',), wspd_4_arr),  
            'wdir_1': (('time',), wdir_1_arr),
            'wdir_2': (('time',), wdir_2_arr),
            'wdir_3': (('time',), wdir_3_arr),
            'wdir_4': (('time',), wdir_4_arr),
            'u_prof': (('time','z'), u_prof_arr),
            'v_prof': (('time','z'), v_prof_arr),
            'wdir_prof': (('time','z'), wdir_prof_arr),
            'wspd_prof': (('time','z'), wspd_prof_arr),
            'u10m': (('time',), u10m_arr),
            'v10m': (('time',), v10m_arr),
            'pblh': (('time',), pblh_arr),
            'p': (('time','z'), p_arr),
            't': (('time','z'), t_arr),
            't_pot': (('time','z'), t_pot_arr),
            'tke': (('time','z'), tke_arr),
            'sw_down': (('time',), sw_down_arr),
        },
        coords={
            'time': times_all,
            'z': z_profile,
        }
    )

    return out_ds

def plot_with_orthogonal_regression(x, y, xlabel, ylabel, max_1_1, one_to_one=True, xlim=None, ylim=None,
                                    color=None, color_label=None, cmap='viridis'):
    """
    Scatter plot with orthogonal (both-sided) linear regression (total least squares via PCA).
    x, y: pandas Series or numpy arrays
    color: optional array/Series used to color the markers (shows a colorbar if provided)
    color_label: label for the colorbar
    one_to_one: if True plot 1:1 dashed red line
    xlim, ylim: optional tuple to set axis limits
    """
    """
    Parameters (additional notes):
    - xlabel: string label for the x-axis (describe physical quantity and units, e.g. 'station wind speed (m/s)')
    - ylabel: string label for the y-axis (describe physical quantity and units, e.g. 'model speed (m/s)')
    """
    plt.figure(figsize=(6, 6))

    # prepare numeric arrays and mask invalid values (also mask color)
    xarr = np.asarray(x)
    yarr = np.asarray(y)
    mask = np.isfinite(xarr) & np.isfinite(yarr)

    colorarr = None
    if color is not None:
        colorarr = np.asarray(color)
        # ensure colorarr has same shape and mask it
        if colorarr.shape != xarr.shape:
            # try to align via pandas index/values if Series passed
            try:
                colorarr = np.asarray(pd.Series(color).reindex_like(pd.Series(x)).values)
            except Exception:
                colorarr = None
        if colorarr is not None:
            colorarr = np.where(mask, colorarr, np.nan)

    # scatter only masked points (avoid plotting NaNs)
    if colorarr is None:
        sc = plt.scatter(xarr[mask], yarr[mask], s=20, alpha=0.8)
    else:
        sc = plt.scatter(xarr[mask], yarr[mask], c=colorarr[mask], cmap=cmap, s=28, edgecolor='k', linewidth=0.1)

    slope = intercept = r = np.nan
    if mask.sum() >= 2:
        xm = xarr[mask]
        ym = yarr[mask]
        # correlation
        try:
            r = np.corrcoef(xm, ym)[0, 1]
        except Exception:
            r = np.nan

        X = np.vstack([xm, ym]).T
        Xc = X - X.mean(axis=0)
        # principal component -> direction of maximal variance
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        direction = Vt[0]
        # slope (dy/dx) and intercept of orthogonal regression line
        if abs(direction[0]) > 1e-12:
            slope = float(direction[1] / direction[0])
            intercept = float(X.mean(axis=0)[1] - slope * X.mean(axis=0)[0])
            xmin = np.nanmin(xm)
            xmax = np.nanmax(xm)
            xvals = np.linspace(xmin, xmax, 200)
            yvals = slope * xvals + intercept
            # compute RMSE for masked points
            try:
                rmse = calculate_rmse(xm, ym)
            except Exception:
                rmse = np.nan
            try:
                mean_bias = float(np.nanmean(ym - xm))
            except Exception:
                mean_bias = np.nan
            label_fit = f"Orth. fit: slope={slope:.3f}, offset={intercept:.3f}, r={r:.3f}, RMSE={rmse:.3f}, Bias={mean_bias:.3f}"
            plt.plot(xvals, yvals, color='k', linewidth=1.5, label=label_fit)

    if one_to_one:
        # determine sensible range for 1:1 line from data or provided limits
        if xlim is not None:
            xmin, xmax = xlim
        else:
            valid_x = xarr[np.isfinite(xarr)]
            if valid_x.size == 0:
                xmin, xmax = 0.0, 1.0
            else:
                xmin = np.nanmin(valid_x)
                xmax = np.nanmax(valid_x)
                # extend a bit for visibility
                pad = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
                xmin -= pad; xmax += pad
        plt.plot([0, max_1_1], [0, max_1_1], 'r--', label='1:1 Line')

    # add colorbar if color provided
    if colorarr is not None and np.isfinite(colorarr).any():
        try:
            cb = plt.colorbar(sc)
            if color_label:
                cb.set_label(color_label)
        except Exception:
            pass

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def match_times_and_plot(ds_model, measurement_df, labels: dict = None, labels_legend: dict = None, labels_axes: dict = None, return_mask: bool = False):
    """Match billwerder minutely times to ds_model times and plot orthogonal regressions.

    Expects `measurement_df` to contain columns `u_wind_50`, `u_wind_110`, `u_wind_175`, `u_wind_280`.
    Expects `ds_model` to contain variables `u1,u2,u3,u4` and coordinate `time`.

    Parameters:
    - labels: optional dict mapping measurement or model variable names to display labels.
        Example: {'u_wind_50': 'Station u @50m', 'u1': 'Model u @50m', 'model_speed': 'Model speed (m/s)'}
    """
    # extract times
    ds_times = ds_model['time'].values
    try:
        ds_times_int = ds_times.astype('datetime64[ns]').astype('int64')
    except Exception:
        ds_times_int = np.arange(ds_times.size)

    # billwerder times: prefer index, else 'time' column
    if 'time' in measurement_df.columns:
        bw_times = measurement_df['time'].values
    else:
        bw_times = measurement_df.index.values
    try:
        bw_times_int = bw_times.astype('datetime64[ns]').astype('int64')
    except Exception:
        bw_times_int = np.arange(bw_times.size)

    # helper to find nearest indices in bw_times for each ds_time
    def nearest_indices(target_ints, ref_ints):
        idxs = np.searchsorted(ref_ints, target_ints)
        idxs_closest = []
        for i, pos in enumerate(idxs):
            if pos == 0:
                idxs_closest.append(0)
            elif pos >= ref_ints.size:
                idxs_closest.append(ref_ints.size - 1)
            else:
                left = ref_ints[pos - 1]
                right = ref_ints[pos]
                if abs(target_ints[i] - left) <= abs(right - target_ints[i]):
                    idxs_closest.append(pos - 1)
                else:
                    idxs_closest.append(pos)
        return np.array(idxs_closest, dtype=int)

    matched_idx = nearest_indices(ds_times_int, bw_times_int)
    # helper to obtain display labels: legend/title vs axis labels
    def _lbl_legend(key, default):
        if labels_legend is not None:
            return labels_legend.get(key, default)
        if labels is not None:
            return labels.get(key, default)
        return default

    def _lbl_axis(key, default):
        if labels_axes is not None:
            return labels_axes.get(key, default)
        if labels is not None:
            return labels.get(key, default)
        return default

    # --- build per-height time masks where both measured and model speeds > threshold ---
    speed_threshold = 0
    height_mask = {}
    # mapping of model index -> height label
    index_to_height = {'1': '50', '2': '110', '3': '175', '4': '280'}
    # candidate measurement speed column names per height
    height_speed_map = {
        '50': ('wind_speed_50', 'wspd_1'),
        '110': ('wind_speed_110', 'wspd_2'),
        '175': ('wind_speed_175', 'wspd_3'),
        '280': ('wind_speed_280', 'wspd_4'),
    }
    # also allow generic 'wind_speed' -> level 50
    if 'wind_speed' in measurement_df.columns:
        height_speed_map['50'] = ('wind_speed', 'wspd_1')

    # compute masks aligned to ds_times (same length as matched_idx)
    for h, (meas_col_speed, model_var_speed) in height_speed_map.items():
        if meas_col_speed in measurement_df.columns and model_var_speed in ds_model:
            try:
                meas_arr = np.asarray(measurement_df[meas_col_speed].values)[matched_idx]
            except Exception:
                # fallback: try align by index
                try:
                    meas_arr = np.asarray(measurement_df[meas_col_speed].reindex(pd.to_datetime(ds_times)).values)
                except Exception:
                    meas_arr = np.full(len(ds_times), np.nan)
            model_arr = np.asarray(ds_model[model_var_speed].values)
            mask_both = (np.isfinite(meas_arr) & np.isfinite(model_arr) & (meas_arr > speed_threshold) & (model_arr > speed_threshold))
        else:
            mask_both = np.zeros(len(ds_times), dtype=bool)
        height_mask[h] = mask_both
    # mapping of billwerder cols to ds_model vars
    # default: multi-height columns


    pairs = [
        ('u_wind_50', 'u1'),
        ('u_wind_110', 'u2'),
        ('u_wind_175', 'u3'),
        ('u_wind_280', 'u4'),
        ('v_wind_50', 'v1'),
        ('v_wind_110', 'v2'),
        ('v_wind_175', 'v3'),
        ('v_wind_280', 'v4'),
        ('wind_speed_50', 'wspd_1'),
        ('wind_speed_110', 'wspd_2'),
        ('wind_speed_175', 'wspd_3'),
        ('wind_speed_280', 'wspd_4'),
        ('wind_dir_50', 'wdir_1'),
        ('wind_dir_110', 'wdir_2'),
        ('wind_dir_175', 'wdir_3'),
        ('wind_dir_280', 'wdir_4'),
    ]

    # If the station DataFrame provides only a single-height `u_wind`/`v_wind`,
    # prefer the simple mapping to model level 1 (`u1`/`v1`).
    
    
    mask = np.ones(len(measurement_df['u_wind_110'].values[matched_idx]), dtype=bool)
    for meas_col, ds_var in pairs:
        if meas_col in ['u_wind_110', 'v_wind_110']:
            
            x = measurement_df[meas_col].values
            x_matched = x[matched_idx]
            y = ds_model[ds_var].values

            # base finite mask
            #mask = mask & np.isfinite(x_matched) & np.isfinite(y)
            mask = mask & (np.abs(x_matched - y) < 3)

    n_total = np.count_nonzero(measurement_df['u_wind_110'].values[matched_idx])
    n_kept = np.count_nonzero(mask)
    n_discarded = n_total - n_kept


    for meas_col, ds_var in pairs:

        if meas_col not in measurement_df.columns or ds_var not in ds_model:
            continue
        x = measurement_df[meas_col].values
        x_matched = x[matched_idx]
        y = ds_model[ds_var].values

        # determine height key (e.g. '50','110',...) for this pair and apply speed-based time mask if available
        level = None
        m = re.search(r'_(\d+)$', meas_col)
        if m:
            level = m.group(1)
        else:
            m2 = re.search(r'(\d+)', ds_var)
            if m2:
                dig = m2.group(1)
                # map model index digits 1->50 etc if present
                level = index_to_height.get(dig, dig)


        mask_pair = np.isfinite(x_matched) & np.isfinite(y)
        if level is not None and level in height_mask:
            mask_pair = mask_pair & height_mask[level]
        # Apply additional mask for outliers
        mask_pair = mask_pair & mask

        if mask_pair.sum() < 3:
            continue

        xm = x_matched[mask_pair].astype(float)
        ym = y[mask_pair].astype(float)

        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xm, ym, s=12, alpha=0.6)
        try:
            r = np.corrcoef(xm, ym)[0, 1]
        except Exception:
            r = np.nan

        X = np.vstack([xm, ym]).T
        Xc = X - X.mean(axis=0)
        # principal component -> direction of maximal variance
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        direction = Vt[0]
        # slope (dy/dx) and intercept of orthogonal regression line
        if abs(direction[0]) > 1e-12:
            slope = float(direction[1] / direction[0])
            intercept = float(X.mean(axis=0)[1] - slope * X.mean(axis=0)[0])
            xmin = np.nanmin(xm)
            xmax = np.nanmax(xm)
            xvals = np.linspace(xmin, xmax, 200)
            yvals = slope * xvals + intercept
            try:
                if meas_col in ['wind_dir_50', 'wind_dir_110', 'wind_dir_175', 'wind_dir_280']:
                    rmse = calculate_rmse_circular(xm, ym)
                else:
                    rmse = calculate_rmse(xm, ym)
            except Exception:
                rmse = np.nan
            try:
                if meas_col in ['wind_dir_50', 'wind_dir_110', 'wind_dir_175', 'wind_dir_280']:
                    mean_bias = float(np.nanmean((ym - xm+180)%360-180))
                else:
                    mean_bias = float(np.nanmean(ym - xm))
            except Exception:
                mean_bias = np.nan
            label_fit = f"Orth. fit: slope={slope:.3f}, offset={intercept:.3f}, r={r:.3f}"
            ax.plot(xvals, yvals, color='k', linewidth=1.5, label=label_fit)
        #1:1 line
        ax.plot([min(xm.min(), ym.min()), max(xm.max(), ym.max())], [min(xm.min(), ym.min()), max(xm.max(), ym.max())], 'r--', label='1:1 Line')
        x_label = _lbl_axis(meas_col, meas_col)
        y_label = _lbl_axis(ds_var, ds_var)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # include RMSE and mean bias in the title if available
        try:
            title_rmse = f", RMSE={rmse:.2f}"
        except Exception:
            title_rmse = ""
        try:
            title_bias = f", Bias={mean_bias:.2f}"
        except Exception:
            title_bias = ""
        ax.set_title(f'{_lbl_legend(meas_col, x_label)} vs {_lbl_legend(ds_var, y_label)} (kept {mask_pair.sum()}, discarded {len(mask_pair)-mask_pair.sum()}{title_rmse}{title_bias})')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # timeseries overlay (model vs station) — show only masked times
        fig, ax = plt.subplots(figsize=(12, 4))
        ds_times_pd = pd.to_datetime(ds_times)
        ax.plot(ds_times_pd[mask_pair], y[mask_pair], label=_lbl_legend(ds_var, f'model {ds_var}'), linewidth=1)
        ax.plot(ds_times_pd[mask_pair], x_matched[mask_pair], label=_lbl_legend(meas_col, f'station {meas_col}'), linewidth=1, alpha=0.8)
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel(_lbl_axis(ds_var, ds_var))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if return_mask:
        return mask_pair

def match_times_and_plot_10m(ds_model, measurement_df, labels: dict = None, labels_legend: dict = None, labels_axes: dict = None):
    """Match billwerder minutely times to ds_model times and plot orthogonal regressions.

    Expects `measurement_df` to contain columns `u_wind_50`, `u_wind_110`, `u_wind_175`, `u_wind_280`.
    Expects `ds_model` to contain variables `u1,u2,u3,u4` and coordinate `time`.

    Parameters:
    - labels: optional dict mapping measurement or model variable names to display labels.
        Example: {'u_wind_50': 'Station u @50m', 'u1': 'Model u @50m', 'model_speed': 'Model speed (m/s)'}
    """
    # extract times
    ds_times = ds_model['time'].values
    try:
        ds_times_int = ds_times.astype('datetime64[ns]').astype('int64')
    except Exception:
        ds_times_int = np.arange(ds_times.size)

    # billwerder times: prefer index, else 'time' column
    if 'time' in measurement_df.columns:
        bw_times = measurement_df['time'].values
    else:
        bw_times = measurement_df.index.values
    try:
        bw_times_int = bw_times.astype('datetime64[ns]').astype('int64')
    except Exception:
        bw_times_int = np.arange(bw_times.size)

    # helper to find nearest indices in bw_times for each ds_time
    def nearest_indices(target_ints, ref_ints):
        idxs = np.searchsorted(ref_ints, target_ints)
        idxs_closest = []
        for i, pos in enumerate(idxs):
            if pos == 0:
                idxs_closest.append(0)
            elif pos >= ref_ints.size:
                idxs_closest.append(ref_ints.size - 1)
            else:
                left = ref_ints[pos - 1]
                right = ref_ints[pos]
                if abs(target_ints[i] - left) <= abs(right - target_ints[i]):
                    idxs_closest.append(pos - 1)
                else:
                    idxs_closest.append(pos)
        return np.array(idxs_closest, dtype=int)

    matched_idx = nearest_indices(ds_times_int, bw_times_int)
    # helper to obtain display labels: legend/title vs axis labels
    def _lbl_legend(key, default):
        if labels_legend is not None:
            return labels_legend.get(key, default)
        if labels is not None:
            return labels.get(key, default)
        return default

    def _lbl_axis(key, default):
        if labels_axes is not None:
            return labels_axes.get(key, default)
        if labels is not None:
            return labels.get(key, default)
        return default

    # --- build per-height time masks where both measured and model speeds > threshold ---
    speed_threshold = 1.5


    # If the station DataFrame provides only a single-height `u_wind`/`v_wind`,
    # prefer the simple mapping to model level 1 (`u1`/`v1`).
    if 'u_wind' in measurement_df.columns and 'u10m' in ds_model:
        pairs = [
            ('u_wind', 'u10m'),
            ('v_wind', 'v10m'),
            ('wind_speed', 'wspd_10m'),
            ('wind_dir', 'wdir_10m'),
        ]

    mask = np.isfinite(measurement_df['u_wind'].values[matched_idx])
    for meas_col, ds_var in pairs:
        if meas_col in ['u_wind', 'v_wind']:
            x = measurement_df[meas_col].values
            x_matched = x[matched_idx]
            y = ds_model[ds_var].values

            mask = mask & np.isfinite(x_matched) & np.isfinite(y)
            # filter for |model - measurement| < 1 m/s
            mask = mask & (np.abs(x_matched - y) < 2)

    n_total = np.count_nonzero(measurement_df['u_wind'].values[matched_idx])
    n_kept = np.count_nonzero(mask)
    n_discarded = n_total - n_kept

    if n_kept < 3:
        return print(f"Not enough valid points after masking for 10m comparison (kept {n_kept}, discarded {n_discarded}). Skipping plot.")    

    for meas_col, ds_var in pairs:
        if meas_col not in measurement_df.columns or ds_var not in ds_model:
            continue
        x = measurement_df[meas_col].values
        x_matched = x[matched_idx]
        y = ds_model[ds_var].values


        xm = x_matched[mask].astype(float)
        ym = y[mask].astype(float)

        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xm, ym, s=12, alpha=0.6)
        try:
            r = np.corrcoef(xm, ym)[0, 1]
        except Exception:
            r = np.nan

        X = np.vstack([xm, ym]).T
        Xc = X - X.mean(axis=0)
        # principal component -> direction of maximal variance
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        direction = Vt[0]
        # slope (dy/dx) and intercept of orthogonal regression line
        if abs(direction[0]) > 1e-12:
            slope = float(direction[1] / direction[0])
            intercept = float(X.mean(axis=0)[1] - slope * X.mean(axis=0)[0])
            xmin = np.nanmin(xm)
            xmax = np.nanmax(xm)
            xvals = np.linspace(xmin, xmax, 200)
            yvals = slope * xvals + intercept
            try:
                if meas_col in ['wind_dir', 'wind_dir_50', 'wind_dir_110', 'wind_dir_175', 'wind_dir_280']:
                    rmse = calculate_rmse_circular(xm, ym)
                else:
                    rmse = calculate_rmse(xm, ym)
            except Exception:
                rmse = np.nan
            try:
                if meas_col in ['wind_dir', 'wind_dir_50', 'wind_dir_110', 'wind_dir_175', 'wind_dir_280']:
                    mean_bias = float(np.nanmean((ym - xm+180)%360-180))
                else:
                    mean_bias = float(np.nanmean(ym - xm))
            except Exception:
                mean_bias = np.nan
            label_fit = f"Orth. fit: slope={slope:.3f}, offset={intercept:.3f}, r={r:.3f}"
            ax.plot(xvals, yvals, color='k', linewidth=1.5, label=label_fit)
        #1:1 line
        ax.plot([min(xm.min(), ym.min()), max(xm.max(), ym.max())], [min(xm.min(), ym.min()), max(xm.max(), ym.max())], 'r--', label='1:1 Line')
        x_label = _lbl_axis(meas_col, meas_col)
        y_label = _lbl_axis(ds_var, ds_var)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # include RMSE and mean bias in the title if available
        try:
            title_rmse = f", RMSE={rmse:.2f}"
        except Exception:
            title_rmse = ""
        try:
            title_bias = f", Bias={mean_bias:.2f}"
        except Exception:
            title_bias = ""
        ax.set_title(f'{_lbl_legend(meas_col, x_label)} vs {_lbl_legend(ds_var, y_label)} (n={n_kept}, discarded={n_discarded}{title_rmse}{title_bias})')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # timeseries overlay (model vs station) — show only masked times
        fig, ax = plt.subplots(figsize=(12, 4))
        ds_times_pd = pd.to_datetime(ds_times)
        ax.plot(ds_times_pd[mask], y[mask], label=_lbl_legend(ds_var, f'model {ds_var}'), linewidth=1)
        ax.plot(ds_times_pd[mask], x_matched[mask], label=_lbl_legend(meas_col, f'station {meas_col}'), linewidth=1, alpha=0.8)
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel(_lbl_axis(ds_var, ds_var))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def prepare_station_data(times, speeds=None, directions=None,
                                time_tz: str = 'UTC',
                                speed_col_name: str = 'wind_speed',
                                dir_col_name: str = 'wind_dir',
                                convert_speed_to_mps: bool = False):
    """
    Create a station DataFrame from hard-coded arrays (no CSV).

    times: sequence of datetimes (numpy datetime64 or pandas-compatible)
    speeds: sequence of wind speeds (same length as times)
    directions: sequence of wind directions (degrees)

    Returns DataFrame with UTC `time`, and `u_wind`/`v_wind` computed.
    """
    times = pd.to_datetime(times, errors='coerce')
    if hasattr(times, 'dt'):
        if times.dt.tz is None:
            try:
                times = times.dt.tz_localize(time_tz)
            except Exception:
                times = times.dt.tz_localize('UTC')
        else:
            times = times.dt.tz_convert('UTC')
    else:
        times = pd.to_datetime(times).tz_localize('UTC')

    n = len(times)
    if speeds is None:
        speeds = np.full(n, np.nan)
    if directions is None:
        directions = np.full(n, np.nan)

    df = pd.DataFrame({
        'time': times,
        speed_col_name: np.asarray(speeds),
        dir_col_name: np.asarray(directions),
    })
    df = df.reset_index(drop=True)
    df = calc_u_v_wind(df, variable_speed=speed_col_name, variable_direction=dir_col_name, convert_speed_to_mps=convert_speed_to_mps)
    return df


#%%
finkenwerder_hourly = load_weather_data_csv(finkenwerder_path)
mittelnkirchen_hourly = load_weather_data_csv(mittelnkirchen_path)
york_hourly = load_weather_data_csv(york_path)
horiba_insitu = read_all_horiba_(horiba_dir)
airpointer_insitu = load_and_stack_csvs(airpointer_dir, prefix="202506", recursive=True).rename(columns={"Time": "time"})
#%%

minutely_time_mez = pd.date_range(start="2025-03-01 00:00", end="2025-10-31 23:59", freq="min", tz="Europe/Berlin")
minutely_time_utc = minutely_time_mez.tz_convert("UTC")

variab_list_billwerder = ["wind_speed_50", "wind_dir_50", "wind_max_50", "wind_speed_110", "wind_dir_110", "wind_max_110", "wind_speed_175", "wind_dir_175", "wind_max_175", "wind_speed_280", "wind_dir_280", "wind_max_280"]
variab_name_billwerder = {
        "wind_speed_50": "FF050",
        "wind_dir_50": "DD050",
        "wind_max_50": "FB050",
        "wind_speed_110": "FF110",
        "wind_dir_110": "DD110",
        "wind_max_110": "FB110",
        "wind_speed_175": "FF175",
        "wind_dir_175": "DD175",
        "wind_max_175": "FB175",
        "wind_speed_280": "FF280",
        "wind_dir_280": "DD280",
        "wind_max_280": "FB280",
    } 
billwerder_minutely = read_all_uni_hamburg_wind_data(billwerder_dir, station_name="",  time=minutely_time_utc, variab_list=variab_list_billwerder, variab_name=variab_name_billwerder)
billwerder_hourly = (billwerder_minutely.replace(99999, np.nan).resample("h").mean(numeric_only=True).rename_axis("time").reset_index())
finkenwerder_hourly = ensure_utc_time_column(finkenwerder_hourly, col="time")
mittelnkirchen_hourly = ensure_utc_time_column(mittelnkirchen_hourly, col="time")
york_hourly = ensure_utc_time_column(york_hourly, col="time")
airpointer_insitu = ensure_utc_time_column(airpointer_insitu, col="time")
time_col = "time" 
airpointer_insitu[time_col] = pd.to_datetime(airpointer_insitu[time_col], errors="coerce")
airpointer_insitu = airpointer_insitu.dropna(subset=[time_col])
airpointer_insitu = airpointer_insitu.set_index(time_col)
airpointer_insitu = airpointer_insitu.replace(-9999, np.nan)

finkenwerder_hourly = calc_u_v_wind(finkenwerder_hourly, convert_speed_to_mps=True)
mittelnkirchen_hourly = calc_u_v_wind(mittelnkirchen_hourly, convert_speed_to_mps=True)
york_hourly = calc_u_v_wind(york_hourly, convert_speed_to_mps=True)
airpointer_insitu = calc_u_v_wind(airpointer_insitu, variable_speed='wind_speed', variable_direction='wind_direction_corr', convert_speed_to_mps=False)
horiba_insitu = calc_u_v_wind(horiba_insitu, variable_speed='wind_speed', variable_direction='wind_dir')
billwerder_hourly = calc_u_v_wind(billwerder_hourly, 
                                  variable_speed_list= ["wind_speed_50", "wind_speed_110", "wind_speed_175", "wind_speed_280"], 
                                  variable_dir_list= ["wind_dir_50", "wind_dir_110", "wind_dir_175", "wind_dir_280"], 
                                  output_list_u= ["u_wind_50", "u_wind_110", "u_wind_175", "u_wind_280"], 
                                  output_list_v= ["v_wind_50", "v_wind_110", "v_wind_175", "v_wind_280"])
billwerder_minutely = calc_u_v_wind(billwerder_minutely, 
                                  variable_speed_list= ["wind_speed_50", "wind_speed_110", "wind_speed_175", "wind_speed_280"], 
                                  variable_dir_list= ["wind_dir_50", "wind_dir_110", "wind_dir_175", "wind_dir_280"], 
                                  output_list_u= ["u_wind_50", "u_wind_110", "u_wind_175", "u_wind_280"], 
                                  output_list_v= ["v_wind_50", "v_wind_110", "v_wind_175", "v_wind_280"])

airpointer_hourly = (airpointer_insitu.resample("h").mean(numeric_only=True).reset_index())


horiba_insitu.index = pd.to_datetime(horiba_insitu.index, errors='coerce')
horiba_insitu = horiba_insitu[~horiba_insitu.index.isna()]
horiba_insitu = horiba_insitu.replace(-9999, np.nan)
horiba_hourly = (horiba_insitu.resample('h').mean(numeric_only=True).reset_index())

airpointer_hourly = calc_speed_dir_wind(airpointer_hourly, variable_u='u_wind', variable_v='v_wind')
horiba_hourly = calc_speed_dir_wind(horiba_hourly, variable_u='u_wind', variable_v='v_wind')
billwerder_hourly = calc_speed_dir_wind(billwerder_hourly, 
                                        variable_u_list= ["u_wind_50", "u_wind_110", "u_wind_175", "u_wind_280"], 
                                        variable_v_list= ["v_wind_50", "v_wind_110", "v_wind_175", "v_wind_280"], 
                                        output_list_speed= ["wind_speed_50", "wind_speed_110", "wind_speed_175", "wind_speed_280"], 
                                        output_list_dir= ["wind_dir_50", "wind_dir_110", "wind_dir_175", "wind_dir_280"])
#%%
cols = {
    "finkenwerder_hourly": "wspd",
    "mittelnkirchen_hourly": "wspd",
    "york_hourly": "wspd",
    "airpointer_hourly": "wind_speed",
    "median_hourly": "median_wspd",
    "horiba_hourly": "wind_speed",
    "billwerder_hourly": "wind_speed"
}

ref_names = list(cols.keys())  
aligned = {}
for name in ref_names:
    if name not in globals():
        continue
    df = globals().get(name)
    df2 = df.copy()
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")
    df2 = df2.set_index("time")
    aligned[name] = df2

u_series = []
v_series = []
speed_series = []
dir_series = []

for name, df2 in aligned.items():
    if "u_wind" in df2.columns:
        u_series.append(df2["u_wind"].rename(name))
    if "v_wind" in df2.columns:
        v_series.append(df2["v_wind"].rename(name))

    # speed: prefer 'wind_speed' then 'wspd' then station-specific speed cols
    if "wind_speed" in df2.columns:
        speed_series.append(df2["wind_speed"].rename(name))
    elif "wspd" in df2.columns:
        speed_series.append(df2["wspd"].rename(name))

    # direction: prefer 'wind_dir' then 'wdir' then station-specific direction cols
    if "wind_dir" in df2.columns:
        dir_series.append(df2["wind_dir"].rename(name))
    elif "wdir" in df2.columns:
        dir_series.append(df2["wdir"].rename(name))

# concat to aligned DataFrames (will align by time index)
u_concat = pd.concat(u_series, axis=1) 
v_concat = pd.concat(v_series, axis=1) 
speed_concat = pd.concat(speed_series, axis=1) 
dir_concat = pd.concat(dir_series, axis=1) 

median_u = u_concat.median(axis=1, skipna=True).rename("median_u")
median_v = v_concat.median(axis=1, skipna=True).rename("median_v") 
median_wspd = speed_concat.median(axis=1, skipna=True).rename("median_wspd")

dir_rad = np.deg2rad(dir_concat)
sin_df = np.sin(dir_rad)
cos_df = np.cos(dir_rad)
median_sin = sin_df.median(axis=1, skipna=True)
median_cos = cos_df.median(axis=1, skipna=True)
median_wdir_direct = (np.rad2deg(np.arctan2(median_sin, median_cos)) + 360) % 360
median_wdir_direct = median_wdir_direct.rename("median_wdir")

# assemble median DataFrame using the station medians (direct medians, not computed from u/v unless available)
median_df = pd.concat(
    [   median_u,                 # median of reported u_wind if available
        median_v,                 # median of reported v_wind if available
        median_wspd,     # median of reported speeds across stations
        median_wdir_direct        # median direction computed from reported directions (circular)
    ],
    axis=1
)
median_df.index.name = "time"

#median_df.to_csv(r"Q:\BREDOM\SEICOR\weatherstations\median_winddata_hourly.csv")
median_hourly = median_df.reset_index()
valid_times = median_df[median_df["median_wspd"] >= 2].index

hourly_names = [
    "finkenwerder_hourly", "mittelnkirchen_hourly",
    "york_hourly", "airpointer_hourly", "horiba_hourly",  "median_hourly"
]

#for name in hourly_names:
#    df = globals().get(name)
#    valid_times =df[df[cols[name]] >= 2].index
#    df["time"] = pd.to_datetime(df["time"], errors="coerce")
#    globals()[name] = df[df["time"].isin(valid_times)].copy().reset_index(drop=True)

ds = xr.open_dataset(r"D:\wrf_meteo\mittelnkirchen_model_ds.nc")
#%%

mask1 = (mittelnkirchen_hourly['time'].dt.month == 6) & (mittelnkirchen_hourly['time'].dt.day < 15) & (mittelnkirchen_hourly["time"].dt.day > 2)
plt.plot(mittelnkirchen_hourly["time"][mask1], mittelnkirchen_hourly["u_wind"][mask1], label="Measured u @10m")
mask2 = (ds["time"].dt.month == 6) & (ds["time"].dt.day < 15) & (ds["time"].dt.day > 2) & (np.abs(ds["u10m"]) > 1.5)
plt.plot(ds["time"][mask2], ds["u10m"][mask2], label="Model u @10m")
#%%
uprof_arr = ds["u_prof"].isel(time=mask2) / ds["u10m"][mask2].values[:, None]
uprof_arr.T.plot()
uprof_arr.mean(dim="time").plot()
uprof_arr.std(dim="time").plot()
prof_scale = np.log(ds.z.values/ 0.05)/np.log(10/0.05)
#%%
z0 = 0.32
prof_scale = np.log(ds.z.values/ z0)/np.log(10/z0)
plt.plot(ds.z.values, prof_scale, label="logarithmic profile scaling")
plt.plot(ds.z.values, uprof_arr.mean(dim="time"), label="mean u profile normalized by u10m")
#%%
z0 = 0.055
ds2 = xr.open_dataset(r"D:\wrf_meteo\ship_corridor_model_ds.nc")
uprof_arr2 = ds2["u_prof"].isel(time=mask2) / ds2["u10m"][mask2].values[:, None]
prof_scale2 = np.log(ds.z.values/z0)/np.log(10/z0)
plt.plot(ds.z.values, prof_scale2, label = "log-profile with z0=0.055m")
plt.plot(ds.z.values, uprof_arr2.mean(dim="time"), label="model profile")
plt.xlabel("Height / m")
plt.ylabel("u profile factor")
plt.legend()
#%%
billwerder_lat = 53.51922389077999 
billwerder_lon = 10.10283134530118
z1, z2, z3, z4 = 50, 110, 175, 280
z_profile = np.arange(0, 500, 10)

#billwerder_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#    prefix = "seicor_wrfmeteo_v1-1_d05",
#    file_pattern = ('.nc',),
#    target_zs=[z1, z2, z3, z4],
#    target_lat=billwerder_lat,
#    target_lon=billwerder_lon
#)
#
#billwerder_model_ds.to_netcdf(r"D:\wrf_meteo\billwerder_model_ds.nc")
billwerder_model_ds = xr.open_dataset(r"D:\wrf_meteo\billwerder_model_ds.nc")
labels_axes = {'u_wind_50': 'Meteomast u @50m / m/s',
        'u_wind_110': 'Meteomast u @110m / m/s',
        'u_wind_175': 'Meteomast u @175m / m/s',
        'u_wind_280': 'Meteomast u @280m / m/s',
        'v_wind_50': 'Meteomast v @50m / m/s',
        'v_wind_110': 'Meteomast v @110m / m/s',
        'v_wind_175': 'Meteomast v @175m / m/s',
        'v_wind_280': 'Meteomast v @280m / m/s',
        'u1': 'Model u @50m / m/s',
        'u2': 'Model u @110m / m/s',
        'u3': 'Model u @175m / m/s',
        'u4': 'Model u @280m / m/s',
        'v1': 'Model v @50m / m/s',
        'v2': 'Model v @110m / m/s',
        'v3': 'Model v @175m / m/s',
        'v4': 'Model v @280m / m/s',
        'wind_speed_50': 'Meteomast wind speed @50m / m/s',
        'wind_speed_110': 'Meteomast wind speed @110m / m/s',
        'wind_speed_175': 'Meteomast wind speed @175m / m/s',
        'wind_speed_280': 'Meteomast wind speed @280m / m/s',
        'wspd_1': 'Model wind speed @50m / m/s',
        'wspd_2': 'Model wind speed @110m / m/s',
        'wspd_3': 'Model wind speed @175m / m/s',
        'wspd_4': 'Model wind speed @280m / m/s',
        'wind_dir_50': 'Meteomast wind dir @50m / °',
        'wind_dir_110': 'Meteomast wind dir @110m / °',
        'wind_dir_175': 'Meteomast wind dir @175m / °',
        'wind_dir_280': 'Meteomast wind dir @280m / °',
        'wdir_1': 'Model wind dir @50m / °',
        'wdir_2': 'Model wind dir @110m / °',
        'wdir_3': 'Model wind dir @175m / °',
        'wdir_4': 'Model wind dir @280m / °',
          }

labels_legend = {'u_wind_50': 'Meteomast',
        'u_wind_110': 'Meteomast',
        'u_wind_175': 'Meteomast',
        'u_wind_280': 'Meteomast',
        'v_wind_50': 'Meteomast',
        'v_wind_110': 'Meteomast',
        'v_wind_175': 'Meteomast',
        'v_wind_280': 'Meteomast',
        'u1': 'Model ',
        'u2': 'Model ',
        'u3': 'Model ',
        'u4': 'Model ',
        'v1': 'Model ',
        'v2': 'Model ',
        'v3': 'Model ',
        'v4': 'Model ',
        'wind_speed_50': 'Meteomast ',
        'wind_speed_110': 'Meteomast ',
        'wind_speed_175': 'Meteomast ',
        'wind_speed_280': 'Meteomast ',
        'wspd_1': 'Model ',
        'wspd_2': 'Model ',
        'wspd_3': 'Model ',
        'wspd_4': 'Model ',
        'wind_dir_50': 'Meteomast',
        'wind_dir_110': 'Meteomast',
        'wind_dir_175': 'Meteomast',
        'wind_dir_280': 'Meteomast',
        'wdir_1': 'Model',
        'wdir_2': 'Model',
        'wdir_3': 'Model',
        'wdir_4': 'Model',
          }
mask = match_times_and_plot(billwerder_model_ds, billwerder_minutely, labels=labels_legend, labels_legend=labels_legend, labels_axes=labels_axes, return_mask=True)
york_times = york_hourly['time'].values
york_station_df = prepare_station_data(york_times, speeds=york_hourly.wspd, directions=york_hourly.wdir,)
york_lat = 53.50831564247247
york_lon = 9.737605460273095
york_z = 10.0

#york_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#                                                prefix = "seicor_wrfmeteo_v1-1_d05",
#                                                file_pattern=('.nc',),
#                                                target_lat=york_lat,
#                                                target_lon=york_lon,
#                                                target_zs=[york_z])
#
#york_model_ds.to_netcdf(r"D:\wrf_meteo\york_model_ds.nc")
#%%
york_model_ds = xr.open_dataset(r"D:\wrf_meteo\york_model_ds.nc")
#calcuclate wdir10m and wspd10m if not already present 
york_model_ds = calc_speed_dir_wind(york_model_ds, variable_u_list=['u10m'], variable_v_list=['v10m'], output_list_speed=['wspd_10m'], output_list_dir=['wdir_10m'])
labels_axes = {'u_wind': 'York u / m/s', 'u1': 'Model u @10m / m/s', 'v_wind': 'York v / m/s', 'v1': 'Model v @10m / m/s', 'wind_speed': 'York wind speed / m/s', 'wspd_1': 'Model wind speed / m/s', 'wind_dir': 'York wind dir / °', 'wdir_1': 'Model wind dir / °'}
labels_legend = {'u_wind': 'York', 'u1': 'Model', 'v_wind': 'York', 'v1': 'Model', 'wind_speed': 'York', 'wspd_1': 'Model', 'wind_dir': 'York', 'wdir_1': 'Model'}
match_times_and_plot_10m(york_model_ds, york_station_df, labels=labels_legend, labels_legend=labels_legend, labels_axes=labels_axes)
mittelnkirchen_station_df = prepare_station_data(mittelnkirchen_hourly['time'].values, speeds=mittelnkirchen_hourly.wspd, directions=mittelnkirchen_hourly.wdir,)
mittelnkirchen_lat = 53.5534
mittelnkirchen_lon = 9.6097
mittelnkirchen_z = 10.0

#mittelnkirchen_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#                                                prefix="seicor_wrfmeteo_v1-1_d05",
#                                                file_pattern=('.nc',),
#                                                target_lat=mittelnkirchen_lat,
#                                                target_lon=mittelnkirchen_lon,
#                                                target_zs=[mittelnkirchen_z])
#mittelnkirchen_model_ds.to_netcdf(r"D:\wrf_meteo\mittelnkirchen_model_ds.nc")
mittelnkirchen_model_ds = xr.open_dataset(r"D:\wrf_meteo\mittelnkirchen_model_ds.nc")
mittelnkirchen_model_ds = calc_speed_dir_wind(mittelnkirchen_model_ds, variable_u_list=['u10m'], variable_v_list=['v10m'], output_list_speed=['wspd_10m'], output_list_dir=['wdir_10m'])
labels_axes = {'u_wind': 'Mittelnkirchen u / m/s', 'u1': 'Model u @10m / m/s', 'v_wind': 'Mittelnkirchen v / m/s', 'v1': 'Model v @10m / m/s', 'wind_speed': 'Mittelnkirchen wind speed / m/s', 'wspd_1': 'Model wind speed / m/s', 'wind_dir': 'Mittelnkirchen wind dir / °', 'wdir_1': 'Model wind dir / °'}
labels_legend = {'u_wind': 'Mittelnkirchen', 'u1': 'Model', 'v_wind': 'Mittelnkirchen', 'v1': 'Model', 'wind_speed': 'Mittelnkirchen', 'wspd_1': 'Model', 'wind_dir': 'Mittelnkirchen', 'wdir_1': 'Model'}
match_times_and_plot_10m(mittelnkirchen_model_ds, mittelnkirchen_station_df, labels=labels_legend, labels_legend=labels_legend, labels_axes=labels_axes)


impact_lat, impact_lon = 53.56959003899151, 9.691754344758492
impact_z = 10.0

#impact_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#                                                prefix="seicor_wrfmeteo_v1-1_d05",
#                                                file_pattern=('.nc',),
#                                                target_lat=impact_lat,
#                                                target_lon=impact_lon,
#                                                target_zs=[impact_z])
#impact_model_ds.to_netcdf(r"D:\wrf_meteo\impact_model_ds.nc")
impact_model_ds = xr.open_dataset(r"D:\wrf_meteo\impact_model_ds.nc")
ship_corridor_lat, ship_corridor_lon = 53.56634576422346, 9.690192228678288
ship_corridor_z = 70.0
#ship_corridor_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#                                                prefix="seicor_wrfmeteo_v1-1_d05",
#                                                file_pattern=('.nc',),
#                                                target_lat=ship_corridor_lat,
#                                                target_lon=ship_corridor_lon,
#                                                target_zs=[ship_corridor_z])
#ship_corridor_model_ds.to_netcdf(r"D:\wrf_meteo\ship_corridor_model_ds.nc")
ship_corridor_model_ds = xr.open_dataset(r"D:\wrf_meteo\ship_corridor_model_ds.nc")

# %%
ship_corridor_lat, ship_corridor_lon = 53.56634576422346, 9.690192228678288
z1, z2, z3, z4 = 50, 110, 175, 280
z_profile = np.arange(0, 500, 10)

#ship_corridor_profile_model_ds = read_sequential_wrfwind_files(r"D:\wrf_meteo",
#    prefix = "seicor_wrfmeteo_v1-1_d05",
#    file_pattern = ('.nc',),
#    target_zs=[z1, z2, z3, z4],
#    target_lat=ship_corridor_lat,
#    target_lon=ship_corridor_lon
#)
#
#ship_corridor_profile_model_ds.to_netcdf(r"D:\wrf_meteo\ship_corridor_profile_model_ds.nc")
# %%
ship_corridor_profile_model_ds = xr.open_dataset(r"D:\wrf_meteo\ship_corridor_profile_model_ds.nc")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.u1, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.u1, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
##%%
#plt.plot(ship_corridor_profile_model_ds.u1, billwerder_model_ds.u1, linestyle='none', marker='o', alpha=0.5)
#
##%%
#plt.plot(ship_corridor_profile_model_ds.v1, billwerder_model_ds.v1, linestyle='none', marker='o', alpha=0.5)
#
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.v1, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.v1, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.u2, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.u2, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.v2, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.v2, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.u3, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.u3, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.v3, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.v3, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.u4, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.u4, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(billwerder_model_ds.time, billwerder_model_ds.v4, label="model mean u profile")
#plt.plot(ship_corridor_profile_model_ds.time, ship_corridor_profile_model_ds.v4, label="model mean v profile")
#plt.xlabel("Time")
#plt.ylabel("Wind component / m/s")
## %%
#plt.plot(ship_corridor_profile_model_ds.u1, billwerder_model_ds.u1, linestyle='none', marker='.', color='blue', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.u2, billwerder_model_ds.u2, linestyle='none', marker='.', color='blue', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.u3, billwerder_model_ds.u3, linestyle='none', marker='.', color='blue', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.u4, billwerder_model_ds.u4, linestyle='none', marker='.', color='blue', alpha=0.2)
#
#plt.plot(ship_corridor_profile_model_ds.v1, billwerder_model_ds.v1, linestyle='none', marker='.', color='red', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.v2, billwerder_model_ds.v2, linestyle='none', marker='.', color='red', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.v3, billwerder_model_ds.v3, linestyle='none', marker='.', color='red', alpha=0.2)
#plt.plot(ship_corridor_profile_model_ds.v4, billwerder_model_ds.v4, linestyle='none', marker='.', color='red', alpha=0.2)
#%%


# Calculate and plot correlation, RMSE, bias, and a single linear fit for all u and all v points
from scipy.stats import linregress

# Concatenate all u and v points
u_model = np.concatenate([
    ship_corridor_profile_model_ds.u1.values[mask],
    ship_corridor_profile_model_ds.u2.values[mask],
    ship_corridor_profile_model_ds.u3.values[mask],
    ship_corridor_profile_model_ds.u4.values[mask]
])
u_meas = np.concatenate([
    billwerder_model_ds.u1.values[mask],
    billwerder_model_ds.u2.values[mask],
    billwerder_model_ds.u3.values[mask],
    billwerder_model_ds.u4.values[mask]
])

v_model = np.concatenate([
    ship_corridor_profile_model_ds.v1.values[mask],
    ship_corridor_profile_model_ds.v2.values[mask],
    ship_corridor_profile_model_ds.v3.values[mask],
    ship_corridor_profile_model_ds.v4.values[mask]
])
v_meas = np.concatenate([
    billwerder_model_ds.v1.values[mask],
    billwerder_model_ds.v2.values[mask],
    billwerder_model_ds.v3.values[mask],
    billwerder_model_ds.v4.values[mask]
])
plt.plot(u_model, u_meas, linestyle='none', marker='.', color='blue', alpha=0.2)
plt.plot(v_model, v_meas, linestyle='none', marker='.', color='red', alpha=0.2)

# u fit and stats
mask_u = np.isfinite(u_model) & np.isfinite(u_meas)
if np.sum(mask_u) > 1:
    corr_u = np.corrcoef(u_model[mask_u], u_meas[mask_u])[0, 1]
    rmse_u = calculate_rmse(u_model[mask_u], u_meas[mask_u])
    bias_u = np.nanmean(u_meas[mask_u] - u_model[mask_u])
    slope_u, intercept_u, r_value_u, p_value_u, std_err_u = linregress(u_model[mask_u], u_meas[mask_u])
    xfit_u = np.linspace(np.nanmin(u_model[mask_u]), np.nanmax(u_model[mask_u]), 100)
    yfit_u = slope_u * xfit_u + intercept_u
    plt.plot(xfit_u, yfit_u, color='blue', linestyle='-', alpha=0.7, label='u fit')
    stats_text_u = f"u: r={corr_u:.2f}, RMSE={rmse_u:.2f}, bias={bias_u:.2f}, slope={slope_u:.2f}"
    plt.text(0.05, 0.95, stats_text_u, transform=plt.gca().transAxes, fontsize=9, color='blue', va='top')

# v fit and stats
mask_v = np.isfinite(v_model) & np.isfinite(v_meas)
if np.sum(mask_v) > 1:
    corr_v = np.corrcoef(v_model[mask_v], v_meas[mask_v])[0, 1]
    rmse_v = calculate_rmse(v_model[mask_v], v_meas[mask_v])
    bias_v = np.nanmean(v_meas[mask_v] - v_model[mask_v])
    slope_v, intercept_v, r_value_v, p_value_v, std_err_v = linregress(v_model[mask_v], v_meas[mask_v])
    xfit_v = np.linspace(np.nanmin(v_model[mask_v]), np.nanmax(v_model[mask_v]), 100)
    yfit_v = slope_v * xfit_v + intercept_v
    plt.plot(xfit_v, yfit_v, color='red', linestyle='-', alpha=0.7, label='v fit')
    stats_text_v = f"v: r={corr_v:.2f}, RMSE={rmse_v:.2f}, bias={bias_v:.2f}, slope={slope_v:.2f}"
    plt.text(0.05, 0.89, stats_text_v, transform=plt.gca().transAxes, fontsize=9, color='red', va='top')
plt.grid()
plt.xlabel("Model at shipping lane / m/s")
plt.ylabel("Model at weathermast / m/s")
plt.legend()
len(v_meas)
# %%

times_all = billwerder_model_ds.time.values
# Histogram of hours where mask is False
import matplotlib.pyplot as plt
import pandas as pd
mask_false_times = pd.to_datetime(times_all[~mask])
hours = mask_false_times.hour
plt.figure()
plt.hist(hours, bins=range(25), align='left', rwidth=0.8)
plt.xlabel('Hour of Day')
plt.ylabel('Count (mask == False)')
plt.title('Histogram of Hours Where mask == False')
plt.xticks(range(24))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
times = billwerder_model_ds.time[mask]
# %%
import pandas as pd
import numpy as np
from datetime import timedelta

# Read the CSV file
plume = pd.read_csv(r"Q:\BREDOM\SEICOR\plume_timestamps.csv")

# Ensure UTC_Time is in datetime format
plume["UTC_Time"] = pd.to_datetime(plume["UTC_Time"], errors="coerce")

# Example: times should be a pandas Series or numpy array of datetime64[ns] (replace with your actual times)
times = billwerder_model_ds.time[mask]

# Only consider rows where plume_useful is True
mask_useful = plume["plume_useful"] == True
plume_useful = plume[mask_useful]

# --- Histogram: useful plume counts per day (UTC) ---
plume_useful = plume_useful.dropna(subset=["UTC_Time"]).copy()
plume_useful = ensure_utc_time_column(plume_useful, col="UTC_Time")

# Mean time between successive useful plumes
_plume_times = (
    plume_useful["UTC_Time"]
    .dropna()
    .sort_values()
    .drop_duplicates()
)
if _plume_times.size < 2:
    print("Mean time between useful plumes: n/a (<2 useful plumes)")
else:
    _dt = _plume_times.diff().dropna()
    mean_dt = _dt.mean()
    mean_hours = float(mean_dt / pd.Timedelta(hours=1))
    print(f"Mean time between useful plumes: {mean_dt} ({mean_hours:.2f} hours)")

plume_useful["day"] = plume_useful["UTC_Time"].dt.floor("D")
daily_counts = plume_useful.groupby("day").size().sort_index()

# Count days with/without useful plumes from `daily_counts` (include zero-count days).
if daily_counts.size:
    _day0 = daily_counts.index.min()
    _day1 = daily_counts.index.max()
else:
    # fallback: use the model time window (so counts are still well-defined)
    _day0 = pd.to_datetime(np.min(times).values, errors="coerce")
    _day1 = pd.to_datetime(np.max(times).values, errors="coerce")
    _day0 = _day0.floor("D")
    _day1 = _day1.floor("D")

_tz = getattr(_day0, "tz", None)
_all_days = pd.date_range(_day0, _day1, freq="D", tz=_tz)
daily_counts_complete = daily_counts.reindex(_all_days, fill_value=0)

days_with_plume = int((daily_counts_complete > 0).sum())
days_without_plume = int((daily_counts_complete == 0).sum())
print(f"Days with ≥1 useful plume: {days_with_plume}")
print(f"Days with 0 useful plumes: {days_without_plume}")

_plot_days = daily_counts.index
try:
    if getattr(_plot_days, "tz", None) is not None:
        _plot_days = _plot_days.tz_convert(None)
except Exception:
    pass

plt.figure(figsize=(14, 4))
plt.bar(_plot_days, daily_counts.values, width=0.9, align='center')
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Day')
plt.ylabel('Useful plumes per day')
plt.grid(axis='y', alpha=0.3)
#plt.tight_layout()
plt.show()

# --- Histogram: useful plume counts by hour-of-day (UTC) ---
plume_useful["hour"] = plume_useful["UTC_Time"].dt.hour
hourly_counts = plume_useful.groupby("hour").size().reindex(range(24), fill_value=0)

plt.figure(figsize=(10, 4))
plt.bar(hourly_counts.index, hourly_counts.values, width=0.9, align='center')
plt.xticks(range(24))
plt.xlabel('Hour of day (UTC)')
plt.ylabel('Useful plumes per hour')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Count number of rows where UTC_Time is within 30 min of any value in times
def within_30min(plume_time, times):
    # Returns True if any time in times is within 30 min of plume_time
    return np.any(np.abs(times - plume_time) <= np.timedelta64(30, 'm'))


# Count number of useful plumes within the first and last time in times

t_start = np.min(times).values
t_end = np.max(times).values
mask_in_range = (plume_useful["UTC_Time"].values >= t_start) & (plume_useful["UTC_Time"].values <= t_end)
count_in_range = mask_in_range.sum()


count = sum(within_30min(time, times) for time in plume_useful["UTC_Time"].values)
print(f"Number of plume_useful rows within 30 min of times: {count}")
print(f"Number of plume_useful rows within time range: {count_in_range}")
# %%
