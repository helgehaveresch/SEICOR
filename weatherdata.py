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

finkenwerder_path = r"Q:\BREDOM\SEICOR\weatherstations\Finkenwerder_Airport\weatherdata_hourly.csv"
fuhlsbuettel_path = r"Q:\BREDOM\SEICOR\weatherstations\Fuhlsbüttel_Airport\weatherdata_hourly.csv"
mittelnkirchen_path = r"Q:\BREDOM\SEICOR\weatherstations\Mittelnkirchen-Hohenfelde\weatherdata_hourly.csv"
york_path = r"Q:\BREDOM\SEICOR\weatherstations\York-Moorende\weatherdata_hourly.csv"
rissen_dir = r"Q:\BREDOM\SEICOR\weatherstations\Rissen"
billwerder_dir = r"Q:\BREDOM\SEICOR\weatherstations\Billwerder"
horiba_dir = r"Q:\BREDOM\SEICOR\InSitu"
airpointer_dir = r"Q:\BREDOM\SEICOR\InSitu\Messdaten"
billwerder_path = r"Q:\BREDOM\SEICOR\weatherstations\Billwerder\weatherdata_hourly.csv"
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

def read_all_uni_hamburg_wind_data(dir_path, station_name, time, variab_list = ["wind_speed", "wind_dir", "wind_max"],     variab_name = {
        "wind_speed": "FF",
        "wind_dir": "DD",
        "wind_max": "FB",
    }):

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

finkenwerder_hourly = load_weather_data_csv(finkenwerder_path)
fuhlsbuettel_hourly = load_weather_data_csv(fuhlsbuettel_path)
mittelnkirchen_hourly = load_weather_data_csv(mittelnkirchen_path)
york_hourly = load_weather_data_csv(york_path)
horiba_insitu = read_all_horiba_(horiba_dir)
airpointer_insitu = load_and_stack_csvs(airpointer_dir, prefix="202506", recursive=True)
#%%
#create a minutely time column in MEZ from 202503010000 to 202510312359
minutely_time_mez = pd.date_range(start="2025-03-01 00:00", end="2025-10-31 23:59", freq="min", tz="Europe/Berlin")
#convert to UTC
minutely_time_utc = minutely_time_mez.tz_convert("UTC")
rissen_minutely = read_all_uni_hamburg_wind_data(rissen_dir, station_name="RIM_", time=minutely_time_utc)

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

#average to hourly and replave 99999 with NaN
rissen_hourly = (
rissen_minutely
.replace(99999, np.nan)
.resample("h")
.mean(numeric_only=True)
.rename_axis("time")
.reset_index()
)
billwerder_hourly = (
billwerder_minutely
.replace(99999, np.nan)
.resample("h")
.mean(numeric_only=True)
.rename_axis("time")
.reset_index()
)


finkenwerder_hourly = ensure_utc_time_column(finkenwerder_hourly, col="time")
fuhlsbuettel_hourly = ensure_utc_time_column(fuhlsbuettel_hourly, col="time")
mittelnkirchen_hourly = ensure_utc_time_column(mittelnkirchen_hourly, col="time")
york_hourly = ensure_utc_time_column(york_hourly, col="time")
airpointer_insitu = ensure_utc_time_column(airpointer_insitu, col="Time")

#%%
time_col = "Time" if "Time" in airpointer_insitu.columns else "time"
airpointer_insitu[time_col] = pd.to_datetime(airpointer_insitu[time_col], errors="coerce")
airpointer_insitu = airpointer_insitu.dropna(subset=[time_col])
airpointer_insitu = airpointer_insitu.set_index(time_col)
airpointer_insitu = airpointer_insitu.replace(-9999, np.nan)

finkenwerder_hourly = calc_u_v_wind(finkenwerder_hourly, convert_speed_to_mps=True)
fuhlsbuettel_hourly = calc_u_v_wind(fuhlsbuettel_hourly, convert_speed_to_mps=True)
mittelnkirchen_hourly = calc_u_v_wind(mittelnkirchen_hourly, convert_speed_to_mps=True)
york_hourly = calc_u_v_wind(york_hourly, convert_speed_to_mps=True)
airpointer_insitu = calc_u_v_wind(airpointer_insitu, variable_speed='wind_speed', variable_direction='wind_direction_corr', convert_speed_to_mps=False)
horiba_insitu = calc_u_v_wind(horiba_insitu, variable_speed='wind_speed', variable_direction='wind_dir')
rissen_hourly = calc_u_v_wind(rissen_hourly, variable_speed='wind_speed', variable_direction='wind_dir')
billwerder_hourly = calc_u_v_wind(billwerder_hourly, 
                                  variable_speed_list= ["wind_speed_50", "wind_speed_110", "wind_speed_175", "wind_speed_280"], 
                                  variable_dir_list= ["wind_dir_50", "wind_dir_110", "wind_dir_175", "wind_dir_280"], 
                                  output_list_u= ["u_wind_50", "u_wind_110", "u_wind_175", "u_wind_280"], 
                                  output_list_v= ["v_wind_50", "v_wind_110", "v_wind_175", "v_wind_280"])


airpointer_hourly = (
    airpointer_insitu
    .resample("h")
    .mean(numeric_only=True)
    .reset_index()
)

# if you prefer a uniform column name:
if time_col == "Time":
    airpointer_hourly = airpointer_hourly.rename(columns={"Time": "time"})

horiba_insitu.index = pd.to_datetime(horiba_insitu.index, errors='coerce')
horiba_insitu = horiba_insitu[~horiba_insitu.index.isna()]
horiba_insitu = horiba_insitu.replace(-9999, np.nan)

horiba_hourly = (
    horiba_insitu
    .resample('h')
    .mean(numeric_only=True)
    .reset_index()
)

airpointer_hourly = calc_speed_dir_wind(airpointer_hourly, variable_u='u_wind', variable_v='v_wind')
horiba_hourly = calc_speed_dir_wind(horiba_hourly, variable_u='u_wind', variable_v='v_wind')
rissen_hourly = calc_speed_dir_wind(rissen_hourly, variable_u='u_wind', variable_v='v_wind')
billwerder_hourly = calc_speed_dir_wind(billwerder_hourly, 
                                        variable_u_list= ["u_wind_50", "u_wind_110", "u_wind_175", "u_wind_280"], 
                                        variable_v_list= ["v_wind_50", "v_wind_110", "v_wind_175", "v_wind_280"], 
                                        output_list_speed= ["wind_speed_50", "wind_speed_110", "wind_speed_175", "wind_speed_280"], 
                                        output_list_dir= ["wind_dir_50", "wind_dir_110", "wind_dir_175", "wind_dir_280"])
#%%

#select June only 
finkenwerder_june = finkenwerder_hourly[finkenwerder_hourly['time'].dt.month == 6]
fuhlsbuettel_june = fuhlsbuettel_hourly[fuhlsbuettel_hourly['time'].dt.month == 6]
mittelnkirchen_june = mittelnkirchen_hourly[mittelnkirchen_hourly['time'].dt.month == 6]
york_june = york_hourly[york_hourly['time'].dt.month == 6]
airpointer_june = airpointer_hourly[airpointer_hourly['time'].dt.month == 6]
horiba_june = horiba_hourly[horiba_hourly['time'].dt.month == 6]
rissen_june = rissen_hourly[rissen_hourly['time'].dt.month == 6]
billwerder_june = billwerder_hourly[billwerder_hourly['time'].dt.month == 6]
#%%
plt.figure(figsize=(12, 6))
plt.plot(finkenwerder_june['time'], finkenwerder_june['wspd'], label='Finkenwerder Airport')
#plt.plot(fuhlsbuettel_june['time'], fuhlsbuettel_june['wspd'], label='Fuhlsbüttel Airport')
#plt.plot(mittelnkirchen_june['time'], mittelnkirchen_june['wspd'], label='Mittelnkirchen-Hohenfelde')
plt.plot(york_june['time'], york_june['wspd'], label='York-Moorende')
plt.plot(airpointer_june['time'], airpointer_june['wind_speed'], label='Airpointer')
#plt.plot(horiba_june['time'], horiba_june['wind_speed'], label='Horiba')
#plt.plot(billwerder_june['time'], billwerder_june['wind_speed_50'], label='Billwerder 50m')
#plt.plot(rissen_june['time'], rissen_june['wind_speed'], label='Rissen')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Hourly Wind Speed Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(finkenwerder_june['time'], finkenwerder_june['wdir'], label='Finkenwerder Airport')
#plt.plot(fuhlsbuettel_june['time'], fuhlsbuettel_june['wdir'], label='Fuhlsbüttel Airport')
#plt.plot(mittelnkirchen_june['time'], mittelnkirchen_june['wdir'], label='Mittelnkirchen-Hohenfelde')
plt.plot(york_june['time'], york_june['wdir'], label='York-Moorende')
plt.plot(airpointer_june['time'], airpointer_june['wind_dir'], label='Airpointer')
#plt.plot(horiba_june['time'], horiba_june['wind_dir'], label='Horiba')
#plt.plot(billwerder_june['time'], billwerder_june['wind_dir_50'], label='Billwerder 50m')
#plt.plot(rissen_june['time'], rissen_june['wind_dir'], label='Rissen')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('Time')
plt.ylabel('Wind Dir (°)')
plt.title('Hourly Wind Direction Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(finkenwerder_june['time'], finkenwerder_june['u_wind'], label='Finkenwerder Airport')
#plt.plot(fuhlsbuettel_june['time'], fuhlsbuettel_june['u_wind'], label='Fuhlsbüttel Airport')
#plt.plot(mittelnkirchen_june['time'], mittelnkirchen_june['u_wind'], label='Mittelnkirchen-Hohenfelde')
plt.plot(york_june['time'], york_june['u_wind'], label='York-Moorende')
plt.plot(airpointer_june['time'], airpointer_june['u_wind'], label='Airpointer')
#plt.plot(horiba_june['time'], horiba_june['u_wind'], label='Horiba')
#plt.plot(rissen_june['time'], rissen_june['u_wind'], label='Rissen')
#plt.plot(billwerder_june['time'], billwerder_june['u_wind_50'], label='Billwerder 50m')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('Time')
plt.ylabel('U Wind (m/s)')
plt.title('Hourly U Wind Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(finkenwerder_june['time'], finkenwerder_june['v_wind'], label='Finkenwerder Airport')
#plt.plot(fuhlsbuettel_june['time'], fuhlsbuettel_june['v_wind'], label='Fuhlsbüttel Airport')
#plt.plot(mittelnkirchen_june['time'], mittelnkirchen_june['v_wind'], label='Mittelnkirchen-Hohenfelde')
plt.plot(york_june['time'], york_june['v_wind'], label='York-Moorende')
plt.plot(airpointer_june['time'], airpointer_june['v_wind'], label='Airpointer')
#plt.plot(horiba_june['time'], horiba_june['v_wind'], label='Horiba')
#plt.plot(rissen_june['time'], rissen_june['v_wind'], label='Rissen')
#plt.plot(billwerder_june['time'], billwerder_june['v_wind_50'], label='Billwerder 50m')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('Time')
plt.ylabel('V Wind (m/s)')
plt.title('Hourly V Wind Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

def plot_with_orthogonal_regression(x, y, xlabel, ylabel, max_1_1, one_to_one=True, xlim=None, ylim=None):
    """
    Scatter plot with orthogonal (both-sided) linear regression (total least squares via PCA).
    x, y: pandas Series or numpy arrays
    one_to_one: if True plot 1:1 dashed red line
    xlim, ylim: optional tuple to set axis limits
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=20, alpha=0.8)

    # prepare numeric arrays and mask invalid values
    xarr = np.asarray(x)
    yarr = np.asarray(y)
    mask = np.isfinite(xarr) & np.isfinite(yarr)

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
            label_fit = f"Orth. fit: slope={slope:.3f}, offset={intercept:.3f}, r={r:.3f}"
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

# replaced scatter blocks with calls to the helper:

finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    airpointer_june_main['wind_speed'],
    finkenwerder_june_main['wspd'],
    'Wind Speed (m/s) Airpointer',
    'Wind Speed (m/s) Finkenwerder Airport',
    max_1_1=11
)

# %%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    airpointer_june_main['wind_speed'],
    york_june_main['wspd'],
    'Wind Speed (m/s) Airpointer',
    'Wind Speed (m/s) York-Moorende',
    max_1_1=11
)
# %%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    york_june_main['wspd'],
    finkenwerder_june_main['wspd'],
    'Wind Speed (m/s) York-Moorende',
    'Wind Speed (m/s) Finkenwerder Airport',
    max_1_1=11
)

# %%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    airpointer_june_main['wind_dir'],
    finkenwerder_june_main['wdir'],
    'Wind Direction (°) Airpointer',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)
# %%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    airpointer_june_main['wind_dir'],
    york_june_main['wdir'],
    'Wind Direction (°) Airpointer',
    'Wind Direction (°) York-Moorende',
    max_1_1=360
)
# %%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    york_june_main['wdir'],
    finkenwerder_june_main['wdir'],
    'Wind Direction (°) York-Moorende',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)
#%%
finkenwerder_june_main = finkenwerder_june[finkenwerder_june['time'].dt.day < 25]
york_june_main = york_june[york_june['time'].dt.day < 25]
airpointer_june_main = airpointer_june[airpointer_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    billwerder_june['wind_dir_110'],
    finkenwerder_june['wdir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)

# %%
billwerder_june_main = billwerder_june[billwerder_june['time'].dt.day < 25]
plot_with_orthogonal_regression(
    billwerder_june_main['wind_dir_110'],
    airpointer_june_main['wind_dir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) Airpointer',
    max_1_1=360
)

# %%
plot_with_orthogonal_regression(
    billwerder_june['wind_dir_110'],
    york_june['wdir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) York-Moorende',
    max_1_1=360
)
# %%

# %%
ranges = [(0.0, 10.0), (93.0, 273.0), (321.0, 360.0)]

wd = billwerder_june_main['wind_dir_110']

# mask for values inside the specified ranges
mask_in_ranges = np.zeros(len(wd), dtype=bool)
for lo, hi in ranges:
    mask_in_ranges |= (wd >= lo) & (wd <= hi)

billwerder_dir_mask = wd.notna() & mask_in_ranges
billwerder_valid_times = pd.to_datetime(billwerder_june_main.loc[billwerder_dir_mask, 'time'])

finkenwerder_june_main_filtered = finkenwerder_june_main[finkenwerder_june_main['time'].isin(billwerder_valid_times)].copy()
york_june_main_filtered = york_june_main[york_june_main['time'].isin(billwerder_valid_times)].copy()
airpointer_june_main_filtered = airpointer_june_main[airpointer_june_main['time'].isin(billwerder_valid_times)].copy()
billwerder_june_main_filtered = billwerder_june_main.loc[billwerder_dir_mask].copy()

# %%
plot_with_orthogonal_regression(
    airpointer_june_main_filtered['wind_speed'],
    finkenwerder_june_main_filtered['wspd'],
    'Wind Speed (m/s) Airpointer',
    'Wind Speed (m/s) Finkenwerder Airport',
    max_1_1=11
)

# %%
plot_with_orthogonal_regression(
    airpointer_june_main_filtered['wind_speed'],
    york_june_main_filtered['wspd'],
    'Wind Speed (m/s) Airpointer',
    'Wind Speed (m/s) York-Moorende',
    max_1_1=11
)
# %%
plot_with_orthogonal_regression(
    york_june_main_filtered['wspd'],
    finkenwerder_june_main_filtered['wspd'],
    'Wind Speed (m/s) York-Moorende',
    'Wind Speed (m/s) Finkenwerder Airport',
    max_1_1=11
)

# %%
plot_with_orthogonal_regression(
    airpointer_june_main_filtered['wind_dir'],
    finkenwerder_june_main_filtered['wdir'],
    'Wind Direction (°) Airpointer',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)
# %%
plot_with_orthogonal_regression(
    airpointer_june_main_filtered['wind_dir'],
    york_june_main_filtered['wdir'],
    'Wind Direction (°) Airpointer',
    'Wind Direction (°) York-Moorende',
    max_1_1=360
)
# %%
plot_with_orthogonal_regression(
    york_june_main_filtered['wdir'],
    finkenwerder_june_main_filtered['wdir'],
    'Wind Direction (°) York-Moorende',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)
#%%
plot_with_orthogonal_regression(
    billwerder_june_main_filtered['wind_dir_110'],
    finkenwerder_june_main_filtered['wdir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) Finkenwerder Airport',
    max_1_1=360
)

# %%
plot_with_orthogonal_regression(
    billwerder_june_main_filtered['wind_dir_110'],
    airpointer_june_main_filtered['wind_dir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) Airpointer',
    max_1_1=360
)

# %%
plot_with_orthogonal_regression(
    billwerder_june_main_filtered['wind_dir_110'],
    york_june_main_filtered['wdir'],
    'Wind Direction (°) Billwerder 110m',
    'Wind Direction (°) York-Moorende',
    max_1_1=360
)
# %%
