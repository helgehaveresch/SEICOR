#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from SEICOR.in_situ import read_in_situ
import re
import matplotlib.dates as mdates
from pathlib import Path
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#finkenwerder_path = r"Q:\BREDOM\SEICOR\weatherstations\Finkenwerder_Airport\weatherdata_hourly.csv"
#fuhlsbuettel_path = r"Q:\BREDOM\SEICOR\weatherstations\Fuhlsbüttel_Airport\weatherdata_hourly.csv"
#mittelnkirchen_path = r"Q:\BREDOM\SEICOR\weatherstations\Mittelnkirchen-Hohenfelde\weatherdata_hourly.csv"
#york_path = r"Q:\BREDOM\SEICOR\weatherstations\York-Moorende\weatherdata_hourly.csv"
rissen_dir = r"Q:\BREDOM\SEICOR\weatherstations\Rissen"
#billwerder_dir = r"Q:\BREDOM\SEICOR\weatherstations\Billwerder"
#horiba_dir = r"Q:\BREDOM\SEICOR\InSitu"
#airpointer_dir = r"Q:\BREDOM\SEICOR\InSitu\Messdaten"
#billwerder_path = r"Q:\BREDOM\SEICOR\weatherstations\Billwerder\weatherdata_hourly.csv"
rissen_path = r"Q:\BREDOM\SEICOR\weatherstations\Rissen\weatherdata_hourly.csv"





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
#calculate u/v components of the wind
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



#%%
"""
finkenwerder_hourly = load_weather_data_csv(finkenwerder_path)
fuhlsbuettel_hourly = load_weather_data_csv(fuhlsbuettel_path)
mittelnkirchen_hourly = load_weather_data_csv(mittelnkirchen_path)
york_hourly = load_weather_data_csv(york_path)
horiba_insitu = read_all_horiba_(horiba_dir)
airpointer_insitu = load_and_stack_csvs(airpointer_dir, prefix="202506", recursive=True)
"""
#%%
#create a minutely time column in MEZ from 202503010000 to 202510312359
minutely_time_mez = pd.date_range(start="2025-03-01 00:00", end="2025-10-31 23:59", freq="min", tz="Europe/Berlin")
#convert to UTC
minutely_time_utc = minutely_time_mez.tz_convert("UTC")
rissen_minutely = read_all_uni_hamburg_wind_data(rissen_dir, station_name="RIM_", time=minutely_time_utc)
"""
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
"""
#average to hourly and replave 99999 with NaN
rissen_hourly = (
rissen_minutely
.replace(99999, np.nan)
.resample("h")
.mean(numeric_only=True)
.rename_axis("time")
.reset_index()
)
