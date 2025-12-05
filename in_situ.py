import numpy as np
import pandas as pd
import os
from io import StringIO
from glob import glob

def read_in_situ(folder_path, date, filename=None):
    """
    Reads and combines in-situ measurement files for a given date into an xarray Dataset.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing in-situ data files.
    date : str
        Date string in the format 'YYMMDD' (e.g., '250511').
    filename : str, optional
        Glob filename for files. If None, uses default filename.

    Returns
    -------
    ds : xarray.Dataset
        Combined in-situ data as an xarray Dataset with metadata.
    """

    if filename is None:
        filename = f"av0_20{date}_*.txt"
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

def calc_mean_wind(df, t, window_seconds=60):
    wind_start = t - pd.Timedelta(seconds=window_seconds/2)
    wind_end = t + pd.Timedelta(seconds=window_seconds/2)
    wind_sel = df[(df.index >= wind_start) & (df.index <= wind_end)]

    #calculate u/v components of the wind
    u = wind_sel['wind_speed'] * np.sin(np.deg2rad(wind_sel['wind_dir']))
    v = wind_sel['wind_speed'] * np.cos(np.deg2rad(wind_sel['wind_dir']))
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
    for i, row in df_closest.iterrows():
        t = row.name
        wind_speed, wind_dir = calc_mean_wind(ds, t)
        df_closest.at[i, 'wind_speed'] = wind_speed
        df_closest.at[i, 'wind_dir'] = wind_dir
    return df_closest