import numpy as np
import pandas as pd
import SEICOR.wind as swind
import os
import re
from datetime import timezone
from io import StringIO
from glob import glob

MAST_HEIGHTS = (50, 110, 175, 280)  # Billwerder meteomast measurement heights in m

def read_airpointer(folder_path, date):
    """
    Reads the Airpointer files for a given date into a DataFrame.

    Each Airpointer file contains the hour *before* the timestamp in its filename,
    so the first file of the next day is read as well and the result is cut to the date.
    Airpointer and HORIBA timestamps share the same clock (checked via NO2 cross-correlation).

    Parameters
    ----------
    folder_path : str
        Path to the Airpointer folder (containing year subfolders).
    date : str
        Date string in the format 'YYMMDD' (e.g., '250511').

    Returns
    -------
    df : pd.DataFrame
        Airpointer data with UTC 'time' index. Wind columns are renamed to 'wind_dir' and 'wind_speed'.
    """
    day = pd.to_datetime(date, format="%y%m%d")
    next_day = day + pd.Timedelta(days=1)
    file_paths = sorted(glob(os.path.join(folder_path, day.strftime("%Y"), f"{day:%Y%m%d}_*.csv")))
    file_paths += sorted(glob(os.path.join(folder_path, next_day.strftime("%Y"), f"{next_day:%Y%m%d}_00*.csv")))
    if not file_paths:
        raise FileNotFoundError(f"No Airpointer files found for {date} in {folder_path}")

    df = pd.concat([pd.read_csv(f, sep=";", decimal=",", parse_dates=["Time"]) for f in file_paths])
    df = df.replace(-9999, np.nan)  # Airpointer fill value
    df = df.rename(columns={"Time": "time", "wind_direction_corr": "wind_dir"})
    df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df.index >= day.tz_localize("UTC")) & (df.index < next_day.tz_localize("UTC"))]
    return df

def read_meteomast(folder_path, date, heights=MAST_HEIGHTS):
    """
    Reads the wind of the Uni Hamburg meteomast (Billwerder) for a given date.

    The files hold one 1-min value per line (e.g. FF110_202503010000-202608312359.txt, FF = speed,
    DD = direction), the period is taken from the file name. Time stamps are MEZ = UTC+1 all year
    (no summer time). Only the lines of the requested day are read.

    Parameters
    ----------
    folder_path : str
        Path to the meteomast folder.
    date : str
        Date string in the format 'YYMMDD' (e.g., '250511').
    heights : iterable of int
        Mast heights in m.

    Returns
    -------
    df : pd.DataFrame
        UTC 'time' index with columns 'wind_speed_mast_{h}' and 'wind_dir_mast_{h}'.
    """
    mez = timezone(pd.Timedelta(hours=1))
    day = pd.to_datetime(date, format="%y%m%d").tz_localize("UTC")
    next_day = day + pd.Timedelta(days=1)
    columns = {}
    for var, code in (("wind_speed", "FF"), ("wind_dir", "DD")):
        for h in heights:
            pieces = []
            for fp in sorted(glob(os.path.join(folder_path, f"{code}{h:03d}_*.txt"))):
                m = re.search(r"_(\d{12})-(\d{12})\.txt$", fp)
                if m is None:
                    continue
                t0, t1 = (pd.to_datetime(s, format="%Y%m%d%H%M").tz_localize(mez).tz_convert("UTC") for s in m.groups())
                if t1 < day or t0 >= next_day:
                    continue
                start = max(0, int((day - t0) / pd.Timedelta(minutes=1)))
                vals = pd.read_csv(fp, header=None, skiprows=start, nrows=24 * 60).iloc[:, 0].to_numpy(dtype=float)
                times = t0 + pd.to_timedelta(start + np.arange(vals.size), unit="min")
                pieces.append(pd.Series(vals, index=times))
            if pieces:
                columns[f"{var}_mast_{h}"] = pd.concat(pieces).sort_index()
    if not columns:
        raise FileNotFoundError(f"No meteomast files found for {date} in {folder_path}")
    df = pd.DataFrame(columns)
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace(99999, np.nan)  # meteomast fill value
    df.index.name = "time"
    return df[(df.index >= day) & (df.index < next_day)]

def read_in_situ(folder_path, date, airpointer_path=None, meteomast_path=None, median_wind_file=None):
    """
    Reads and combines in-situ measurement files for a given date into an xarray Dataset.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing in-situ data files.
    date : str
        Date string in the format 'YYMMDD' (e.g., '250511').
    airpointer_path : str, optional
        Path to the Airpointer folder. If given, 'wind_dir' and 'wind_speed' are taken from the
        Airpointer (nearest sample within 10 s); the HORIBA wind is kept as 'wind_dir_horiba'
        and 'wind_speed_horiba'.
    meteomast_path : str, optional
        Path to the meteomast folder. If given, the mast wind of all heights is added as
        'wind_speed_mast_{h}' and 'wind_dir_mast_{h}' (nearest 1-min value within 1 min).
    median_wind_file : str, optional
        Path to the hourly median wind of the weather stations (median_winddata_hourly.csv).
        If given, it is added as 'wind_speed_median' and 'wind_dir_median' (nearest hour within 1 h).

    Returns
    -------
    ds : xarray.Dataset
        Combined in-situ data as an xarray Dataset with metadata.
    """

    
    year_folder = f"20{date[:2]}"
    filename = os.path.join(year_folder, f"av0_20{date}_*.txt")
    file_paths = sorted(glob(os.path.join(folder_path, filename)))

    dfs = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
        lines = [line.replace(',', '.') for line in lines]
        columns_line = lines[0].strip().split('\t')
        units_line = lines[1].strip().split('\t')
        one_letter_vars = [
            "time", "p_0", "c_co2", "dewpoint", "gust_of_wind", "c_h2o", "T_in", "Precip_type", "n_no",
            "c_no2", "c_nox", "c_o3", "quality", "rain_intens", "rainfall", "rel_humid", "c_so2",
            "T_out", "wind_dir", "wind_speed", "wind_chill"
        ]
        var_map = dict(zip(columns_line, one_letter_vars))
        data_lines = lines[2:]
        df = pd.read_csv(
            StringIO(''.join(data_lines)),
            sep='\t',
            names=[var_map[name] for name in columns_line],
            parse_dates=[var_map[columns_line[0]]],
            dayfirst=True
        )
        for col in df.columns:
            if col != var_map[columns_line[0]]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        dfs.append(df)

    full_df = pd.concat(dfs)
    if full_df['time'].dt.tz is None:
        full_df['time'] = full_df['time'].dt.tz_localize('UTC')
    else:
        full_df['time'] = full_df['time'].dt.tz_convert('UTC')
    full_df.set_index("time", inplace=True)
    full_df.sort_index(inplace=True)

    if airpointer_path is not None:
        df_ap = read_airpointer(airpointer_path, date)
        full_df = full_df.rename(columns={"wind_dir": "wind_dir_horiba", "wind_speed": "wind_speed_horiba"})
        # nearest (not interpolated) to avoid averaging wind directions across 0/360 deg
        full_df = pd.merge_asof(
            full_df, df_ap[["wind_dir", "wind_speed"]],
            left_index=True, right_index=True,
            direction="nearest", tolerance=pd.Timedelta("10s"),
        )
    if meteomast_path is not None:
        df_mast = read_meteomast(meteomast_path, date)
        full_df = pd.merge_asof(
            full_df, df_mast,
            left_index=True, right_index=True,
            direction="nearest", tolerance=pd.Timedelta("1min"),
        )
    if median_wind_file is not None:
        df_median = pd.read_csv(median_wind_file, parse_dates=["time"])
        df_median["time"] = pd.to_datetime(df_median["time"], utc=True)
        df_median = (df_median.set_index("time").sort_index()[["median_wspd", "median_wdir"]]
                     .rename(columns={"median_wspd": "wind_speed_median", "median_wdir": "wind_dir_median"}))
        full_df = pd.merge_asof(
            full_df, df_median,
            left_index=True, right_index=True,
            direction="nearest", tolerance=pd.Timedelta("1h"),
        )
    #for orig_name, unit in zip(columns_line[1:], units_line[1:]):
    #    short_name = var_map[orig_name]
    #    ds[short_name].attrs['long_name'] = orig_name
    #    ds[short_name].attrs['units'] = unit

    return full_df

def apply_time_mask_to_insitu(df, start_time, end_time):
    """
    Apply a time mask to the in-situ DataFrame, keeping only the data within the specified time range.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be masked.
    start_time : pd.Timestamp
        The start time of the mask.
    end_time : pd.Timestamp
        The end time of the mask.

    Returns
    -------
    pd.DataFrame
        The masked DataFrame.
    """
    return df[(df.index >= start_time) & (df.index <= end_time)]

def calc_mean_wind(df, t, window_seconds=60, speed_col='wind_speed', dir_col='wind_dir'):
    wind_start = t - pd.Timedelta(seconds=window_seconds/2)
    wind_end = t + pd.Timedelta(seconds=window_seconds/2)
    wind_sel = df[(df.index >= wind_start) & (df.index <= wind_end)]

    #calculate u/v components of the wind
    u = wind_sel[speed_col] * np.sin(np.deg2rad(wind_sel[dir_col]))
    v = wind_sel[speed_col] * np.cos(np.deg2rad(wind_sel[dir_col]))
    #calculate mean u/v components
    u_mean = float(u.mean()) if u.size > 0 else np.nan
    v_mean = float(v.mean()) if v.size > 0 else np.nan
    #calculate mean wind direction and speed
    wind_speed_mean = np.sqrt(u_mean**2 + v_mean**2)
    wind_dir_mean = np.arctan2(u_mean, v_mean)
    # Convert mean wind direction from radians to degrees
    wind_dir_mean = np.rad2deg(wind_dir_mean)
    #adjust wind direction to be within [0, 360] degrees
    wind_dir_mean = wind_dir_mean % 360
    return wind_speed_mean, wind_dir_mean

def add_wind_to_ship_passes(df_closest, ds):
    if ds is None or ds.empty:
        for col in ["wind_speed", "wind_dir", "rel_wind_speed", "rel_wind_dir", "rel_wind_u", "rel_wind_v"]:
            df_closest[col] = np.nan
        return df_closest
    # values are collected in row order and assigned as whole columns, not by index label:
    # two ships passing in the same second share a time index
    mast_heights = [h for h in MAST_HEIGHTS if f'wind_speed_mast_{h}' in ds.columns]
    columns = ['wind_speed', 'wind_dir']
    columns += [f'wind_{v}_mast_{h}' for h in mast_heights for v in ('speed', 'dir')]
    columns += ['rel_wind_speed', 'rel_wind_dir', 'rel_wind_u', 'rel_wind_v']
    values = {col: [] for col in columns}
    for t, row in df_closest.iterrows():
        wind_speed, wind_dir = calc_mean_wind(ds, t)
        values['wind_speed'].append(wind_speed)
        values['wind_dir'].append(wind_dir)
        # meteomast wind at all heights (if read in by read_in_situ)
        for h in mast_heights:
            mast_speed, mast_dir = calc_mean_wind(ds, t, speed_col=f'wind_speed_mast_{h}', dir_col=f'wind_dir_mast_{h}')
            values[f'wind_speed_mast_{h}'].append(mast_speed)
            values[f'wind_dir_mast_{h}'].append(mast_dir)
        # compute relative wind seen from the moving ship using Mean_Speed and Mean_Course
        ship_speed = row.get('Mean_Speed', np.nan)
        ship_course = row.get('Mean_Course', np.nan)
        if np.isnan(ship_speed) or np.isnan(ship_course) or np.isnan(wind_speed) or np.isnan(wind_dir):
            rel = (np.nan, np.nan, np.nan, np.nan)
        else:
            rel_speed, rel_dir = swind.compute_relative_wind(ship_speed, ship_course, wind_speed, wind_dir)
            # compute vector components (meteorological FROM -> to vector: u = -s*sin(dir), v = -s*cos(dir))
            dir_rad = np.deg2rad(rel_dir)
            u_rel = -rel_speed * np.sin(dir_rad)
            v_rel = -rel_speed * np.cos(dir_rad)
            rel = (float(rel_speed), float(rel_dir), float(u_rel), float(v_rel))
        for col, val in zip(['rel_wind_speed', 'rel_wind_dir', 'rel_wind_u', 'rel_wind_v'], rel):
            values[col].append(val)
    for col, vals in values.items():
        df_closest[col] = vals
    return df_closest