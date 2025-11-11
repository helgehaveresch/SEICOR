#%%
import numpy as np
import pandas as pd
import re
import xarray as xr
import shutil
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import process_SC_img_data
from imaging_tools.process_Individual_SC_files import mask_zenith
from SEICOR.enhancements import rolling_background_enh
import os
from matplotlib.gridspec import GridSpec
from PIL import Image
import pytz
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from io import StringIO
from glob import glob
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from numpy.fft import fft, ifft, fftfreq


def read_ais(file):
    """
    Input is file and path of ais datafile.
    """
    KNOTS_TO_MS = 1/1.944

    # open file line by line
    with open(file, "r", encoding="cp1252") as f:
        content = [line.split(";") for line in f]

    # sort content by length, different message types have different length
    # and different information content
    content5 = pd.DataFrame([line for line in content if len(line) == 5],
                            columns=["MMSI",
                                     "Interogration",
                                     "Message",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content6 = pd.DataFrame([line for line in content if len(line) == 6],
                            columns=["MMSI",
                                     "Class",
                                     "Part",
                                     "Name",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content7 = pd.DataFrame([line for line in content if len(line) == 7],
                            columns=["MMSI",
                                     "Voyage data",
                                     "Unkown",
                                     "Dimensions",
                                     "ERI",
                                     "Draught load UTC Time",
                                     "Serial"],
                            dtype=str)

    content7["len"] = ""
    content7["beam"] = ""
    content7["draught"] = ""
    content7["load"] = ""
    content7["UTC Time"] = ""

    for i, index in enumerate(content7.index):
        content7.loc[index, "len"] = content7.loc[index, "Dimensions"].split(" ")[1]
        content7.loc[index, "beam"] = content7.loc[index, "Dimensions"].split(" ")[3]
        content7.loc[index, "draught"] = content7.loc[index, "Draught load UTC Time"].split(" ")[1]
        content7.loc[index, "load"] = content7.loc[index, "Draught load UTC Time"].split(" ")[3]
        content7.loc[index, "UTC Time"] = content7.loc[index, "Draught load UTC Time"].split(" ")[4]+" "+content7.loc[index, "Draught load UTC Time"].split(" ")[5]

    content9 = pd.DataFrame([line for line in content if len(line) == 9],
                            columns=["MMSI",
                                     "Class",
                                     "Unknown",
                                     "Name",
                                     "Type",
                                     "Destination",
                                     "Dimensions",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content9["Length_in_m"] = ""
    content9["Width_in_m"] = ""
    content9["Draught_in_m"] = ""

    for i, index in enumerate(content9.index):
        content9.loc[index, "Length_in_m"] = content9.loc[index, "Dimensions"].split(" ")[0]
        content9.loc[index, "Width_in_m"] = content9.loc[index, "Dimensions"].split(" ")[1]
        content9.loc[index, "Draught_in_m"] = content9.loc[index, "Dimensions"].split(" ")[2]

    # content 11 is most common message and includes positional data
    content11 = pd.DataFrame([line for line in content if len(line) == 11],
                             columns=["MMSI",
                                      "Navigation_status",
                                      "Rate of turn",
                                      "Speed in knots",
                                      "Latitude",
                                      "Longitude",
                                      "Course_in_deg",
                                      "True_heading_in_deg",
                                      "UTC seconds",
                                      "UTC Time",
                                      "Serial"],
                             dtype=str)

    content11 = content11.assign(Name="", Type="", Length_in_m="", Width_in_m="",
                                 Draught_in_m="", Destination="", ERI="")

    content11["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content11["UTC Time"]])
    content5["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content5["UTC Time"]])
    content6["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content6["UTC Time"]])
    content7["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content7["UTC Time"]])
    content9["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content9["UTC Time"]])

    # combine all information from different message types
    for i, mmsi in enumerate(content11["MMSI"].unique()):
        index11 = np.where(mmsi == content11["MMSI"])[0]

        if len(content7) > 0:
            try:
                index7 = np.where(mmsi == content7["MMSI"])[0]
                content11.loc[index11, "ERI"] = str(np.unique(content7.loc[index7, "ERI"])[0])
            except:
                pass

        if len(content6) > 0:
            index6 = np.where(mmsi == content6["MMSI"])[0]
        if len(content9) > 0:
            index9 = np.where(mmsi == content9["MMSI"])[0]
            if len(index9) > 0:
                content11.loc[index11, "Name"] = str(np.unique(content9.loc[index9, "Name"])[0])
                content11.loc[index11, "Type"] = str(np.unique(content9.loc[index9, "Type"])[0])
                content11.loc[index11, "Destination"] = str(np.unique(content9.loc[index9, "Destination"])[0])
                content11.loc[index11, "Length_in_m"] = str(np.unique(content9.loc[index9, "Length_in_m"])[0])
                content11.loc[index11, "Width_in_m"] = str(np.unique(content9.loc[index9, "Width_in_m"])[0])
                content11.loc[index11, "Draught_in_m"] = str(np.unique(content9.loc[index9, "Draught_in_m"])[0])

        if all(content11.loc[index11, "Name"] == "Part B"):
            try:
                content11.loc[index11, "Name"] = str(np.unique(content6.loc[index6, "Name"])[0])
            except:
                content11.loc[index11, "Name"] = str(" ")

    # convert to appropiate data types and change some units
    content11 = content11.loc[["kt" in c for c in content11.loc[:, "Speed in knots"]], :]
    content11.loc[:, "Speed in knots"] = [float(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Speed in knots"]]
    content11.loc[:, "Latitude"] = [np.float64(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Latitude"]]
    content11.loc[:, "Longitude"] = [np.float64(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Longitude"]]
    content11.loc[:, "Name"] = [str(str(c).rstrip()) for c in content11.loc[:, "Name"]]
    content11.loc[:, "Navigation_status"] = [ str(c.rstrip()) for c in content11.loc[:, "Navigation_status"]]
    content11.loc[:, "Type"] = [str(str(c).rstrip()) for c in content11.loc[:, "Type"]]
    content11.drop_duplicates(inplace=True)

    content11.loc[:, "True_heading_in_deg"] = [str(i) for i in content11.loc[:, "True_heading_in_deg"]]
    content11.loc[:, "True_heading_in_deg"] = [float(i.replace("°", "")) if i is not isinstance(i, float) else np.nan for i in content11.loc[:, "True_heading_in_deg"]]
    content11.loc[:, "True_heading_in_deg"] = [float(i) if float(i) <= 359 else np.nan for i in content11.loc[:, "True_heading_in_deg"]]

    content11.loc[:, "Speed_in_m/s"] = [i * KNOTS_TO_MS for i in content11.loc[:, "Speed in knots"]]

    content11.loc[:, "Course_in_deg"] = [str(i) for i in content11.loc[:, "Course_in_deg"]]
    content11.loc[content11.loc[:, "Course_in_deg"] == " unk. ", "Course_in_deg"] = 9999
    content11.loc[:, "Course_in_deg"] = pd.to_numeric(content11.loc[:, "Course_in_deg"], errors='coerce')

    content11.loc[:, "Length_in_m"] = pd.to_numeric(content11.loc[:, "Length_in_m"].values, errors="coerce")
    content11.loc[:, "Width_in_m"] = pd.to_numeric(content11.loc[:, "Width_in_m"].values, errors="coerce")
    content11.loc[:, "Draught_in_m"] = pd.to_numeric(content11.loc[:, "Draught_in_m"].values, errors="coerce")

    content11.set_index("UTC Time", inplace=True)
    content11.index.name = "UTC_Time"
    # subset only important information
    content11 = content11[["MMSI", "ERI", "Navigation_status", "Speed_in_m/s", "Latitude", "Longitude",
                           "Course_in_deg", "True_heading_in_deg", "Serial", "Name", "Type",
                           "Length_in_m", "Width_in_m", "Draught_in_m", "Destination"]]
    # prepare output with appropiate data types
    content11 = content11.astype({"MMSI": int,
                                  "ERI": str,
                                  "Navigation_status": str,
                                  "Speed_in_m/s": float,
                                  "Latitude": float,
                                  "Longitude": float,
                                  "Course_in_deg": float,
                                  "True_heading_in_deg": float,
                                  "Serial": str,
                                  "Name": str,
                                  "Type": str,
                                  "Length_in_m": float,
                                  "Width_in_m": float,
                                  "Draught_in_m": float,
                                  "Destination": str})

    content11 = content11.replace({np.nan: None})

    return content11

def convert_timestamp(time):
    """
    Convert timestamps to pandas.datetime objects.
    """
    time = np.asarray([str(t.replace(" ", "")) for t in time])

    y = [str("20"+y[:2]) for y in time]
    m = [str(m[2:4]) for m in time]
    d = [str(d[4:6]) for d in time]
    H = [str(H[6:8]) for H in time]
    M = [str(M[8:10]) for M in time]
    S = [str(S[10:12]) for S in time]

    time = [pd.to_datetime(y+"-"+m+"-"+d+" "+H+":"+M+":"+S, yearfirst=True, utc=True) for y, m, d, H, M, S
            in zip(y, m, d, H, M, S)]
    return time

def interpolate_ais(ais, limit=60, resample="1S"):
    """


    Parameters
    ----------
    ais : dataframe
        Interpolates the AIS signals to 1 second resolution.
    limit: float
        Limits the amount of time that can be interpolated. Here a maximum of
        "limit" seconds can be interpolated.
    resample: str
        resample to which time resolution. Default is "1S".
    Returns
    -------
    output : dataframe
        Interpolated AIS signals.

    """

    ais_list = [it[1] for it in list(ais.groupby("MMSI"))]

    for i, ais_dat in enumerate(ais_list):
        ais_dat = ais_dat.loc[~ais_dat.index.duplicated(keep="first")]
        tmp = ais_dat.resample(resample).interpolate(method="linear", limit=limit)
        tmp["MMSI"] = tmp["MMSI"].bfill(limit=limit)
        tmp = tmp.dropna(subset=["MMSI"])
        tmp = tmp.astype({"MMSI": int})
        ais_list[i] = tmp

    output = pd.concat(ais_list)

    return output

def filter_ais(ds, lat_lon_window, start_time, end_time, measurement_times, endpoints_los, instrument_location, length=20, distance_threshold=0.002):

    ds = restrict_lat_lon(ds, lat_lon_window)

    sailing_mask = ds["Type"].str.lower() == "vessel:sail"
    anchored_mask = ds["Navigation_status"].str.lower() == "anchored"
    length_mask = (ds["Length_in_m"] < length) #filter out ships that are too small or too large

    ds["filtermask"] = 0
    ds.loc[anchored_mask, "filtermask"] = 2
    ds.loc[sailing_mask, "filtermask"] = 3
    ds.loc[length_mask, "filtermask"] = 4 # todo: all filtering just by masks?

    # 1. Filter out all ships that have Type 'Vessel:sail' or are anchored
    sailing_mmsi = set(ds[ds["Type"].str.lower() == "vessel:sail"]["MMSI"].unique())
    anchored_mmsi = set(ds[ds["Navigation_status"].str.lower() == "anchored"]["MMSI"].unique())
    small_mmsi = set(ds[ds["Length_in_m"] < length]["MMSI"].unique())
    sailing_mmsi = sailing_mmsi  | small_mmsi | anchored_mmsi

    # Start from grouping ships by MMSI
    ship_groups = ds.groupby("MMSI")
    # Only keep ship data within the measurement time interval
    filtered_ship_groups = []
    for mmsi, group in ship_groups:
        if mmsi not in sailing_mmsi:
            mask = (group.index >= start_time) & (group.index <= end_time) #filter out ships outside the IMPACT operational time window
            #filtered_group = group.loc[mask]
            if not mask.any():
                ds.loc[group.index, "filtermask"] = 1  # time window not matching
            else:
                pos = np.sqrt(group["Longitude"]**2 + group["Latitude"]**2)
                if pos.diff().any() > 0.001:
                    filtered_ship_groups.append((mmsi, group))
                else:
                    ds.loc[group.index, "filtermask"] = 5  # no movement

    df_closest = filter_close_positions(filtered_ship_groups, start_time, end_time, measurement_times, endpoints_los, instrument_location, ds, distance_threshold)

    close_mmsi = df_closest["MMSI"].unique().tolist()
    filtered_ship_groups = {mmsi: group for mmsi, group in filtered_ship_groups if mmsi in close_mmsi}

    maskedout_mmsi = set(ds["MMSI"].unique()) - set(close_mmsi)
    maskedout_groups = {mmsi: group for mmsi, group in ship_groups if mmsi in maskedout_mmsi}
    filtered_ship_groups = dict(filtered_ship_groups)

    return ship_groups, filtered_ship_groups, maskedout_groups, df_closest

def restrict_lat_lon(ds, lat_lon_window):
    lat_mask = (ds["Latitude"] < lat_lon_window[0]) | (ds["Latitude"] > lat_lon_window[1])
    lon_mask = (ds["Longitude"] < lat_lon_window[2]) | (ds["Longitude"] > lat_lon_window[3])
    ds.loc[lon_mask, "Longitude"] = np.nan
    ds.loc[lat_mask, "Latitude"] = np.nan
    return ds

def filter_close_positions(filtered_ship_groups, start_time, end_time, measurement_times, endpoints_los, instrument_location, ds,distance_threshold):
    closest_points = []
    for mmsi, group in filtered_ship_groups:
        
        ship_lats = group["Latitude"].values
        ship_lons = group["Longitude"].values
        ais_times = group.index
        ship_positions = np.stack([ship_lats, ship_lons], axis=1)
        idxs = np.abs(measurement_times.values[:, None] - ais_times.values).argmin(axis=0)

        # Vectorized distance calculation
        ab = endpoints_los[idxs] - np.array(instrument_location)      #np.stack([lat2_s - instrument_location[0], lon2_s - instrument_location[1]], axis=1)
        ab_norm = ab / np.linalg.norm(ab, axis=1)[:, None]
        ap = ship_positions - np.array(instrument_location)
        proj = np.sum(ap * ab_norm, axis=1)[:, None] * ab_norm
        dists = np.linalg.norm(ap - proj, axis=1)

        close_positions_mask = dists <= distance_threshold
        
        if np.any(close_positions_mask):
            
            mask = close_positions_mask.astype(int)
            diff = np.diff(mask, prepend=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if mask[-1] == 1:
                ends = np.append(ends, len(mask))
            for start, end in zip(starts, ends):
                pass_idxs = np.arange(start, end)
                if len(pass_idxs) == 0:
                    continue
                min_idx = pass_idxs[np.argmin(dists[pass_idxs])]
                t_ship = pd.to_datetime(group.index[min_idx])
                if (t_ship >= start_time) and (t_ship <= end_time):
                    measurement_diffs = np.abs(measurement_times - t_ship)
                    closest_meas_idx = measurement_diffs.argmin()
                    closest_time_diff = measurement_diffs[closest_meas_idx]
                    closest_meas_time = measurement_times[closest_meas_idx]
                    mean_speed, mean_course = calculate_mean_speed_course(group, t_ship)
                    ship_category = sort_ship_sizes(group)
                    closest_points.append({
                        "MMSI": mmsi,
                        "UTC_Time": group.index[min_idx],
                        "Closest_Impact_Measurement_Index": closest_meas_idx,
                        "Closest_Impact_Measurement_Time": closest_meas_time,
                        "Closest_Impact_Measurement_Time_Diff": closest_time_diff,
                        "Mean_Speed": mean_speed,
                        "Mean_Course": mean_course,
                        "Latitude": ship_lats[min_idx],
                        "Longitude": ship_lons[min_idx],
                        "Distance": dists[min_idx],
                        "Draught_in_m": group["Draught_in_m"].iloc[min_idx],
                        "Length_in_m": group["Length_in_m"].iloc[min_idx],
                        "ship_category": ship_category
                    })
        else:
            ds.loc[group.index, "filtermask"] = 6  # no close positions
    closest_points = sorted(closest_points, key=lambda x: pd.to_datetime(x["UTC_Time"]))
    df_closest = pd.DataFrame(closest_points)
    df_closest.set_index("UTC_Time", inplace=True)
    return df_closest

def sort_ship_sizes(ship_group):
    draught = ship_group["Draught_in_m"].iloc[0]
    if draught is not None and not np.isnan(draught):
        if draught <= 6:
            ship_category = 'small'
        elif draught > 9:
            ship_category = 'large'
        else:
            ship_category = 'middle'
    else:
        ship_category = 'unknown'
    return ship_category

def prepare_ais(file, interpolation_limit):
    data = read_ais(file)
    #data = interpolate_ais(data, interpolation_limit)
    data.index = pd.to_datetime(data.index)
    return data

def calculate_mean_speed_course(ship_group, t):
    """
    Calculate mean speed (m/s) and mean course (degrees) for a ship around time t.

    Args:
        filtered_ships (dict): MMSI -> DataFrame with index as timestamps, columns 'Latitude', 'Longitude'.
        mmsi (int or str): Ship MMSI.
        t (pd.Timestamp): Time of interest (timezone-aware or naive).

    Returns:
        mean_speed (float): Mean speed in meters/second.
        mean_course (float): Mean course in degrees from North (0-360).
        None, None if not enough data.
    """
    if ship_group is None:
        return None, None

    # Ensure t is UTC
    if t.tzinfo:
        t_utc = t.tz_convert('UTC')
    else:
        t_utc = t.tz_localize('UTC')

    # Select AIS positions within ±60 seconds of t
    ais_window = ship_group.loc[
        (ship_group.index >= t_utc - pd.Timedelta(seconds=60)) &
        (ship_group.index <= t_utc + pd.Timedelta(seconds=60))
    ]

    if ais_window.empty or len(ais_window) < 2:
        return None, None

    
    lat = ais_window["Latitude"].astype(float).values
    lon = ais_window["Longitude"].astype(float).values
    times = pd.to_datetime(ais_window.index).values.astype('datetime64[s]').astype(float)

    # Use first and last positions
    lat1, lon1, t1 = lat[0], lon[0], times[0]
    lat2, lon2, t2 = lat[-1], lon[-1], times[-1]

    # Haversine formula for distance (meters)
    R = 6371000.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    dt = t2 - t1  # seconds
    mean_speed = distance / dt if dt > 0 else np.nan

    # Bearing calculation
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    mean_course = (np.degrees(bearing) + 360) % 360

    if np.isnan(mean_course) or np.isnan(mean_speed):
        return None, None

    return mean_speed, mean_course

def read_in_situ(folder_path, date, pattern=None):
    """
    Reads and combines in-situ measurement files for a given date into an xarray Dataset.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing in-situ data files.
    date : str
        Date string in the format 'YYMMDD' (e.g., '250511').
    pattern : str, optional
        Glob pattern for files. If None, uses default pattern.

    Returns
    -------
    ds : xarray.Dataset
        Combined in-situ data as an xarray Dataset with metadata.
    """

    if pattern is None:
        pattern = f"av0_20{date}_*.txt"
    file_paths = sorted(glob(os.path.join(folder_path, pattern)))

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

    print(f"Read and combined {len(file_paths)} files for {date}")
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

def coarsen_impact_measurements(ds, factor):
    """
    Coarsens the IMPACT measurements by a specified factor.

    Args:
        ds (xr.Dataset): The IMPACT dataset to coarsen.
        factor (int): The factor by which to coarsen the dataset.

    Returns:
        xr.Dataset: The coarsened IMPACT dataset.
    """
    return ds.coarsen(dim_0=factor, boundary="trim").mean()

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

def read_lpdoas(file_path, date, mode= "ppb"):
    """
    Reads LP-DOAS .dat file for a given date, skipping bad lines.

    Parameters
    ----------
    file_path : str
        The path to the LP-DOAS .dat file.
    date : str
        The date for which to read the data (format: 'YYMMDD').
    mode : str
        The mode for reading the data ('ppb' or 'SC') (default: 'ppb').

    Returns
    -------
    df : pd.DataFrame
        DataFrame with parsed LP-DOAS data.
    """

    file_path = r"P:\data\data_tmp\Open_Path\20{}_365_Eval_{}.dat".format(date, mode)
    df = pd.read_csv(
        file_path,
        sep='\t',
        engine='python',
        on_bad_lines='skip'  # skips lines with wrong number of columns
    )
    # Convert 'StartDateAndTime' to datetime and set timezone to UTC
    df['StartDateAndTime'] = pd.to_datetime(df['StartDateAndTime'])
    if df['StartDateAndTime'].dt.tz is None:
        df['StartDateAndTime'] = df['StartDateAndTime'].dt.tz_localize('UTC')
    else:
        df['StartDateAndTime'] = df['StartDateAndTime'].dt.tz_convert('UTC')
    df.set_index("StartDateAndTime", inplace=True)
    df.sort_index(inplace=True)
    return df

def mask_lp_doas_file(df_lp_doas, start_time, end_time, rms_threshold = 0.0005):
    """
    Masks the LP-DOAS DataFrame for a specific time range and RMS threshold.

    Args:
        df_lp_doas (pd.DataFrame): The LP-DOAS DataFrame to mask.
        start_time (pd.Timestamp): The start time for the mask.
        end_time (pd.Timestamp): The end time for the mask.

    Returns:
        pd.DataFrame: The masked LP-DOAS DataFrame.
    """
    mask = (df_lp_doas.index >= start_time) & (df_lp_doas.index <= end_time) & (df_lp_doas['RMS'] < rms_threshold)
    return df_lp_doas[mask]

def point_line_distance(lat, lon, lat1, lon1, lat2, lon2):
    """
    Calculate the perpendicular distance from (lat, lon) to the line defined by (lat1, lon1) and (lat2, lon2).
    All coordinates in degrees.
    """
    # Convert to numpy arrays for vector math
    p = np.array([lat, lon])
    a = np.array([lat1, lon1])
    b = np.array([lat2, lon2])
    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a
    # Project ap onto ab, then find the perpendicular vector
    ab_norm = ab / np.linalg.norm(ab)
    proj = np.dot(ap, ab_norm) * ab_norm
    perp = ap - proj
    return np.linalg.norm(perp)

def plot_LOS(lon1, lon2, lat1, lat2, ax=None, label=None):
    """
    Plot lines from (lon1, lat1) to each unique (lon2, lat2) pair.
    """
    if ax is None:
        ax = plt.gca()
    # Get unique pairs of (lon2, lat2)
    unique_points = set(zip(np.ravel(lon2), np.ravel(lat2)))
    print(unique_points)
    for i, (x2, y2) in enumerate(unique_points):
        if label and i == 0:
            ax.plot([lon1, x2], [lat1, y2], color='red', linewidth=2, linestyle='--', label=label)
        else:
            ax.plot([lon1, x2], [lat1, y2], color='red', linewidth=2, linestyle='--')
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')

def plot_trajectories(filtered_ships, maskedout_ships, df_closest, lon1, lon2, lat1, lat2, window_small):
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.3, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ships not filtered out")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()

    # Plot the masked out ships (either sailing vessels or distance > 0.005)

    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.3, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships")
    plot_LOS(lon1, lon2, lat1, lat2)
    #plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()

    # Plot the ships that are left after filtering (filtered_ships)
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Non-sailing ships with minimum distance ≤ 0.005 deg to LOS")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()


    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships (closeup)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    #plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()

    multi_pass_mmsi = df_closest["MMSI"].value_counts()
    multi_pass_mmsi = multi_pass_mmsi[multi_pass_mmsi > 1].index.tolist()

    plt.figure(figsize=(10, 8))
    for mmsi in multi_pass_mmsi:
        group = filtered_ships.get(mmsi)
        if group is not None:
            plt.plot(group["Longitude"], group["Latitude"], label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ships that pass multiple times (filtered ships)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()

def plot_maskedout_ships_details(maskedout_ships, window_small, lon1, lon2, lat1, lat2):
    
    plt.figure(figsize=(12, 8))
    filtermask_labels = {
        1: "Outside operation time",
        2: "Anchored",
        3: "Sailing vessel",
        4: "Too small",
        5: "No movement",
        6: "No close positions"
    }
    colors = {1: "red", 2: "green", 3: "blue", 4: "orange", 5: "purple", 6: "gray"}

    for mmsi, group in maskedout_ships.items():
        for mask_val in group["filtermask"].unique():
            mask = group["filtermask"] == mask_val
            if mask.any():
                label = filtermask_labels.get(mask_val, f"filtermask={mask_val}")
                plt.plot(
                    group.loc[mask, "Longitude"],
                    group.loc[mask, "Latitude"],
                    color=colors.get(mask_val, "black"),
                    alpha=0.7,
                    label=label if plt.gca().get_legend_handles_labels()[1].count(label) == 0 else None
                )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships by filtermask")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.show()

def plot_ship_stats(filtered_ships): 
    # lentgh histogram
    lengths = pd.Series(
        [group["Length_in_m"].dropna().iloc[0] for group in filtered_ships.values() if not group["Length_in_m"].dropna().empty]
    )
    plt.figure()
    plt.hist(lengths, bins=30, alpha=0.7)
    plt.xlabel("Length (m)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Lengths (filtered ships, one per MMSI)")
    plt.show()

    # Draught histogram
    draughts = pd.Series(
        [group["Draught_in_m"].dropna().iloc[0] for group in filtered_ships.values() if not group["Draught_in_m"].dropna().empty]
    )
    plt.figure()
    plt.hist(draughts, bins=30, alpha=0.7)
    plt.xlabel("Draught (m)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Draughts (filtered ships)")
    plt.show()

    # Velocity histogram
    velocities = pd.Series(
        [group["Speed_in_m/s"].dropna().iloc[0] for group in filtered_ships.values() if not group["Speed_in_m/s"].dropna().empty]
    )
    plt.figure()
    plt.hist(velocities, bins=30, alpha=0.7)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Velocities (filtered ships)")
    plt.show()

    # Correlation between length and draught
    length_draught_df = pd.DataFrame({
        "Length_in_m": lengths,
        "Draught_in_m": draughts
    }).dropna()
    plt.figure()
    plt.scatter(length_draught_df["Length_in_m"], length_draught_df["Draught_in_m"], alpha=0.5)
    plt.xlabel("Length (m)")
    plt.ylabel("Draught (m)")
    plt.title("Correlation between Length and Draught (filtered ships)")
    plt.show()

    corr = length_draught_df["Length_in_m"].corr(length_draught_df["Draught_in_m"])
    print(f"Correlation between length and draught: {corr:.2f}")

def calculate_LOS(ds_measurements, instrument_location, line_length=0.05):
    """
    Calculate the line of sight (LOS) endpoints based on the instrument location and viewing direction.
    Parameters:
        ds_measurements (xarray.Dataset): Dataset containing the measurements.
        instrument_location (list): [latitude, longitude] of the instrument.
        line_length (float): Length of the line segment to plot.
    Returns:
        endpoints_los (np.ndarray): Array of shape (N, 2) with the endpoints of the LOS.
    """
    lat1, lon1 = instrument_location
    viewing_direction = ds_measurements["viewing-azimuth-angle"].isel(viewing_direction=0).values
    bearing_rad = np.deg2rad(viewing_direction)
    delta_lat = line_length * np.cos(bearing_rad)
    delta_lon = line_length * np.sin(bearing_rad) / np.cos(np.deg2rad(lat1))
    lat2 = lat1 + delta_lat
    lon2 = lon1 + delta_lon
    endpoints_los = np.column_stack([lat2, lon2])
    return endpoints_los

def calc_start_end_times(ds_measurements):
    """
    Calculate the start and end times of the measurement period.
    """
    measurement_times = pd.to_datetime(ds_measurements["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")
    start_time = measurement_times.min()
    end_time = measurement_times.max()
    return start_time, end_time, measurement_times

def styles_from_ship_category(parameter, category):
    if parameter == "linestyle":
        linestyle = {'small': ':', 'middle': '--', 'large': '-', 'unknown': '-.'}
        return linestyle.get(category, '-')
    elif parameter == "transparency":
        transparency = {'small': 0.3, 'middle': 0.5, 'large': 0.7, 'unknown': 0.5}
        return transparency.get(category, 0.5)
    elif parameter == "label":
        label = {'small': 'Small ship', 'middle': 'Medium ship', 'large': 'Large ship', 'unknown': 'Unknown'}
        return label.get(category, 'Unknown')
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

def rms_mask(ds, threshold=0.01, instrument = "IMPACT"):
    if instrument == "IMPACT":
        rms_mask = ds["rms"].mean(dim="viewing_direction") < 0.01
    ds_masked = ds.where(rms_mask, drop=True)
    return ds_masked

def plot_no2_timeseries(
    ds_masked,
    df_closest,
    start_time,
    end_time,
    add_ship_lines=True,
    legend_location="upper right",
    separate_legend=False,
):
    """
    Plot NO2 timeseries with optional vertical ship lines.

    Parameters
    ----------
    ds_masked : xarray.Dataset
        Masked NO2 dataset.
    ship_times : list of datetime, optional
        Times of ship passes.
    start_time, end_time : datetime, optional
        Time window for plotting ship lines.
    add_ship_lines : bool, default True
        Whether to add vertical ship lines.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    legend_fontsize : int
        Font size for legend.
    date_str : str
        Date string for title.
    """
    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    if add_ship_lines and df_closest is not None and start_time is not None and end_time is not None:
        for idx, row in df_closest.iterrows():
            t_ship = pd.to_datetime(row["UTC_Time"]) if "UTC_Time" in row else idx
            category = row.get("ship_category")
            if (t_ship >= start_time) and (t_ship <= end_time):
                label = styles_from_ship_category("label", category)
                # Only add label if not already present
                if label not in ax.get_legend_handles_labels()[1]:
                    ax.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=label
                    )
                else:
                    ax.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=""
                    )

    # Plot NO2 enhancement with RMS mask applied
    ax.plot(
        ds_masked.datetime.isel(viewing_direction=0),
        ds_masked["a[NO2]"].mean(dim="viewing_direction"),
        label="NO$_2$ (FOV-averaged)",
        linewidth=2
    )

    ax.set_xlabel("Time (UTC)", fontsize=18)
    ax.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=16)
    date_str = start_time.strftime("%d.%m.%Y")
    
    ax.set_title(f"NO$_2$ dSCD timeseries on {date_str}", fontsize=20)
    ax.grid()
    # Set x-axis to show only time (HH:MM) and restrict number of ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.tick_params(axis='both', labelsize=16)
    #ax.legend(loc=legend_location, fontsize=14)
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    if separate_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(3, 2))
        ax_leg = fig_legend.add_subplot(111, frameon=False)
        ax_leg.axis('off')  # Hide the axis
        fig_legend.legend(handles, labels, loc='center', fontsize=14)
        plt.show()
        plt.close(fig_legend)
    else:
        ax.legend(loc=legend_location, fontsize=14)
        plt.show()

def plot_single_ship(
    ds_measurements=None,
    ds_enhancements=None,
    dir_enhancements=None,
    t_after_start_h=None,
    interval_h=0.1,
    vmin=None,
    vmax=None,
    include_mmsi=False,
    include_timestamp=False,
    y_axis_ticks_interval=7,
    x_axis_tick_interval=2,
    mode="dSCD",
    save_fig=False,
    save_dir=None
    ):
    """ 
    Create 2d Plot of NO2 dSCD or enhancements

    Parameters
    ----------
    ds_measurements : xarray.Dataset
        NO2 measurements dataset.
    ds_enhancements : xarray.Dataset
        NO2 enhancements dataset.
    dir_enhancements : xarray.Dataset
        Directional NO2 enhancements dataset.
    t_after_start_h : float
        Time (in hours) after start time to begin plotting.
    interval_h : float
        Time interval (in hours) for the plot.
    vmin, vmax : float
        Minimum and maximum values for the color scale.
    include_mmsi : bool
        Whether to include MMSI in the plot title.
    include_timestamp : bool
        Whether to include timestamp in the plot title.
    y_axis_ticks_interval : int
        Interval for y-axis ticks.
    x_axis_tick_interval : int
        Interval for x-axis ticks.
    mode : str
        Plotting mode ("dSCD", "enhancement", "integrated", or "combined").
    """

    if ds_measurements is not None:
        no2_full = ds_measurements["a[NO2]"].values  # shape: (time, viewing_direction)
        times_full = pd.to_datetime(ds_measurements["datetime"].isel(viewing_direction=0).values)
        viewing_dirs = ds_measurements["los"].values
    if ds_enhancements is not None:
        no2_full = ds_enhancements["no2_enhancement"].values
        times_full = pd.to_datetime(ds_enhancements["datetime"].isel(viewing_direction=0).values)
        viewing_dirs = ds_enhancements["los"].values
    if dir_enhancements is not None:
        no2_full = dir_enhancements["no2_enhancement"].values
        times_full = dir_enhancements["times_window"].values
        viewing_dirs = dir_enhancements["los"].values
        mmsi = dir_enhancements["mmsi"]
        timestamp = dir_enhancements["t"]

    if t_after_start_h is not None:
        mask = (
            (times_full >= pd.Timestamp(times_full[0].date()) + pd.Timedelta(hours=t_after_start_h)) &
            (times_full < pd.Timestamp(times_full[0].date()) + pd.Timedelta(hours=t_after_start_h + interval_h))
        )
        times_sel = times_full[mask]
        no2_sel = no2_full[:, mask]
    else:
        times_sel = times_full
        no2_sel = no2_full

    X, Y = np.meshgrid(times_sel, np.arange(no2_sel.shape[0]))

    if mode == "dSCD":
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pcm = ax.pcolormesh(
            X, Y, no2_sel,
            shading='auto',
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(pcm)
        cbar.set_label(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=22)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)
        ax.set_xlabel("Time (UTC)", fontsize=22)
        ax.set_ylabel("Viewing direction", fontsize=22)
        ax.set_title("NO$_2$ dSCD Single Ship", fontsize=24)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{viewing_dirs[i,0]-90:.1f}°" for i in yticks], fontsize=18)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        plt.show()

    elif mode == "enhancement":
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pcm = ax.pcolormesh(
            X, Y, no2_full,
            shading='auto',
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(pcm)
        cbar.set_label(r"NO$_2$ Enh. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=22)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)
        ax.set_xlabel("Time (UTC)", fontsize=22)
        ax.set_ylabel("Viewing direction", fontsize=22)
        title = "NO$_2$ dSCD Enhancements"
        if include_mmsi:
            title += f", MMSI: {mmsi}"
        if include_timestamp:
            title += f", Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ax.set_title(title, fontsize=14)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{viewing_dirs[i,0]-90:.1f}°" for i in yticks], fontsize=18)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        if save_fig: 
            plt.savefig(os.path.join(save_dir, f"NO2_{timestamp.strftime('%Y%m%d_%H%M%S')}_{mmsi}.png"))
            plt.close()
        else:
            plt.show()

    elif mode == "integrated":
        no2_integrated = no2_full.sum(axis=0)  # shape: (selected_times,)
        plt.figure(figsize=(10, 4))
        plt.plot(times_sel, no2_integrated, color='tab:blue', linewidth=2)
        plt.xlabel("Time (UTC)", fontsize=18)
        plt.ylabel(r"NO$_2$ dSCD int. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=18)
        plt.title("Vertically Integrated NO$_2$ Enhancement", fontsize=20)
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        plt.tight_layout()
        plt.show()

    elif mode == "combined":
        no2_integrated = no2_full.sum(axis=0)  # shape: (selected_times,)
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        # --- 2D NO2 enhancement plot ---
        ax = axs[0]
        pcm = ax.pcolormesh(
            X, Y, no2_full,
            shading='auto',
            cmap='viridis', vmin=vmin if vmin is not None else -3e16, vmax=vmax
        )
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(r"NO$_2$ Enh. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)

        ax.set_ylabel("Viewing direction", fontsize=22)
        ax.set_title("NO$_2$ dSCD Enhancements", fontsize=24)

        # Set y-ticks to actual viewing direction angles (every Nth for clarity)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{viewing_dirs[i,0]-90:.1f}°" for i in yticks], fontsize=18)

        # Format x-axis: only every 2 min a tick
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # --- Vertically integrated NO2 subplot ---
        ax2 = axs[1]
        ax2.plot(times_sel, no2_integrated, color='tab:blue', label="Vertically Integrated NO$_2$", linewidth=2)
        ax2.set_ylabel(r"NO$_2$ dSCD int. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=18)
        ax2.set_xlabel("Time (UTC)", fontsize=22)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.grid(True)
        ax2.legend(fontsize=20)
        ax2.yaxis.get_offset_text().set_fontsize(16)

        plt.tight_layout()
        # Make ax2 narrower in x-direction (e.g., 60% width, aligned left)
        pos = ax2.get_position()
        ax2.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

        plt.show()

    else:
        raise ValueError("mode must be one of: 'dSCD', 'enhancement', 'integrated', 'combined'")

def parse_img_time(fname):
    # Handles formats:
    # 1. yymmdd_hhmmssms.JPG
    # 2. yymmdd_hhmmssmms-??_???-??.JPG

    base = os.path.splitext(fname)[0]
    cet = pytz.timezone("Europe/Berlin")
    # Try old format first
    try:
        dt = pd.to_datetime(base, format="%y%m%d_%H%M%S%f").tz_localize("CET")
        return dt
    except Exception:
        pass
    # Try extended format: yymmdd_hhmmssmms-??_???-??
    match = re.match(r"(\d{6}_\d{8,9})", base)
    if match:
        try:
            dt = pd.to_datetime(match.group(1), format="%y%m%d_%H%M%S%f").tz_localize("CET")
            return dt
        except Exception:
            pass
    return None

def get_closest_image(img_files, img_times, target_time):
    """
    Returns the filename and time difference (in seconds) of the image closest to target_time.

    Parameters
    ----------
    img_files : list of str
        List of image filenames.
    img_times : list of datetime
        List of datetimes corresponding to img_files.
    target_time : datetime
        The time to which the closest image is sought.

    Returns
    -------
    closest_file : str
        Filename of the closest image.
    time_diff : float
        Absolute time difference in seconds between image and target_time.
    """
    closest_file, closest_time = min(zip(img_files, img_times), key=lambda x: abs(x[1] - target_time))
    time_diff = abs((closest_time - target_time).total_seconds())
    return closest_file, time_diff

def assign_video_images_to_ship_pass(df_closest, img_dir, utc):
    """
    For each row in df_closest, find and assign the closest image file and time difference.
    Adds columns 'closest_image_file' and 'image_time_diff'.
    """
    img_files, img_times = zip(*[
        (f, parse_img_time(f).astimezone(utc))
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") and parse_img_time(f) is not None
    ])
    closest_files = []
    time_diffs = []
    for t in df_closest.index:
        closest_file, time_diff = get_closest_image(img_files, img_times, t)
        closest_files.append(os.path.join(img_dir, closest_file))
        time_diffs.append(time_diff)
    df_closest = df_closest.copy()
    df_closest["closest_image_file"] = closest_files
    df_closest["image_time_diff"] = time_diffs
    return df_closest

def copy_ship_images(df_closest, img_dir, img_out_dir, time_threshold=30):
    """
    Copies images for each ship in df_closest to img_out_dir, renaming them to include MMSI.
    Only copies if image_time_diff <= time_threshold (seconds).
    """

    os.makedirs(img_out_dir, exist_ok=True)

    for ship_time, ship in df_closest.iterrows():
        mmsi = ship["MMSI"]
        closest_file = ship["closest_image_file"]
        time_diff = ship["image_time_diff"]
        if time_diff <= time_threshold:
            src = closest_file if os.path.isabs(closest_file) else os.path.join(img_dir, closest_file)
            base, ext = os.path.splitext(os.path.basename(closest_file))
            new_name = f"{base}_{mmsi}.jpg"
            dst = os.path.join(img_out_dir, new_name)
            shutil.copy2(src, dst)
        else:
            print(f"No suitable image found for MMSI {mmsi} at time {ship_time}, closest image was {closest_file} with time difference {time_diff:.2f} seconds")
    print("Copied Images of ships")

def upwind_constant_background_enh(row, ds_measurements, measurement_times, df_closest, window_minutes=(1, 3), ref_search_minutes=60, ref_window_minutes=1,  do_lp=False, df_lp = None):
    """
    Subtracts background for a single ship pass (row from df_closest).
    Returns: dict with keys: mmsi, t, no2_data, times_window, window, window_ref, ref_found
    """
    mmsi = row["MMSI"]
    t = pd.to_datetime(row.name)#.tz_localize("UTC")
    time_diff = row["Closest_Impact_Measurement_Time_Diff"].total_seconds()
    if time_diff > 60:
        return None
    window = ((measurement_times >= t - pd.Timedelta(minutes=window_minutes[0])) & (measurement_times < t + pd.Timedelta(minutes=window_minutes[1])))
    # Find reference window
    ref_found = False
    ref_offset = 3
    while not ref_found and ref_offset < ref_search_minutes:
        ref_start = t - pd.Timedelta(minutes=ref_offset)
        ref_end = t - pd.Timedelta(minutes=ref_offset - ref_window_minutes)
        window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
        ref_times = measurement_times[window_ref]
        other_ships_in_window = False
        for idx2, row2 in df_closest.iterrows():
            if row2["MMSI"] == mmsi:
                continue
            other_t = pd.to_datetime(idx2).tz_localize("UTC") if pd.to_datetime(idx2).tzinfo is None else pd.to_datetime(idx2)
            if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                other_ships_in_window = True
                break
        if not other_ships_in_window and window_ref.sum() > 0:
            ref_found = True
        else:
            ref_offset += 1
    if not ref_found:
        print(f"No clean reference window found for MMSI {mmsi} at {t}")
        return None
    no2_enhancement = ds_measurements["a[NO2]"].isel(dim_0=window) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
    vertically_integrated_no2 = no2_enhancement.sum(dim="viewing_direction")
    o4_enhancement = ds_measurements["a[O4]"].isel(dim_0=window) - ds_measurements["a[O4]"].isel(dim_0=window_ref).mean(dim="dim_0")
    times_window = measurement_times[window]

    if do_lp: 
        lp_window = ((df_lp.index >= t - pd.Timedelta(minutes=window_minutes[0])) & (df_lp.index < t + pd.Timedelta(minutes=window_minutes[1])))
        lp_window_ref = ((df_lp.index >= ref_start) & (df_lp.index < ref_end))
        lp_no2_enhancement = df_lp['Fit Coefficient (NO2)'][lp_window] - df_lp['Fit Coefficient (NO2)'][lp_window_ref].mean()
        lp_times_window = df_lp.index[lp_window]

        return {
            "mmsi": mmsi,
            "t": t,
            "no2_enhancement": no2_enhancement,
            "o4_enhancement": o4_enhancement,
            "vertically_integrated_no2_enhancement": vertically_integrated_no2,
            "times_window": times_window,
            "window": window,
            "window_ref": window_ref,
            "ref_found": ref_found,
            "los": ds_measurements["los"],
            "lp_no2": df_lp['Fit Coefficient (NO2)'][lp_window],
            "lp_no2_enhancement": lp_no2_enhancement,
            "lp_times_window": lp_times_window,
        }
    
    return {
        "mmsi": mmsi,
        "t": t,
        "no2_enhancement": no2_enhancement,
        "o4_enhancement": o4_enhancement,
        "vertically_integrated_no2_enhancement": vertically_integrated_no2,
        "times_window": times_window,
        "window": window,
        "window_ref": window_ref,
        "ref_found": ref_found,
        "los": ds_measurements["los"],
    }

def plot_no2_enhancements_for_all_ships(df_closest, ds_measurements, measurement_times, out_dir):
    """
    Loops over df_closest, subtracts background, and saves NO2 enhancement images.
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in df_closest.iterrows():
        result = upwind_constant_background_enh(row, ds_measurements, measurement_times, df_closest)
        if result is None:
            continue
        plot_single_ship(
            dir_enhancements=result,
            t_after_start_h=None,
            interval_h=None,
            mode="enhancement",
            vmin=-3e16,
            include_mmsi=True,
            include_timestamp=True,
            x_axis_tick_interval=1,
            save_fig=True,
            save_dir=out_dir,
        )

def plot_ship_pass_subplot_v1(
    result,
    row,
    ds_measurements,
    df_insitu,
    no2_out_dir,
    save=True
):
    """
    Plot NO2 enhancement, integrated NO2, video image, wind/ship polar, and in-situ NO2 for a ship pass.
    """

    # Try to find the corresponding image
    img_file = None
    if "closest_image_file" in row and pd.notnull(row["closest_image_file"]):
        img_file = row["closest_image_file"]

    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])

    # NO2 enhancement 2D plot
    ax0 = fig.add_subplot(gs[0, 0])
    X, Y = np.meshgrid(result["times_window"], np.arange(result["no2_enhancement"].shape[0]))
    pcm = ax0.pcolormesh(X, Y, result["no2_enhancement"], shading='auto')
    fig.colorbar(pcm, ax=ax0, label=r"NO$_2$ Enhancement / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax0.set_title(f"MMSI {result['mmsi']} NO2 around {result['t'].strftime('%Y-%m-%d %H:%M:%S')}")
    ax0.set_xlabel("Time")
    N = 3  # Show every 3rd LOS value
    los_vals = result["los"].isel(dim_0=0).values - 90
    yticks = np.arange(0, len(los_vals), N)
    ax0.set_yticks(yticks)
    ax0.set_yticklabels([f"{float(los_vals[i]):.1f}°" for i in yticks])
    ax0.set_ylabel("LOS / °")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax0.get_xticklabels(), rotation=45)

    # Integrated NO2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(result["times_window"], result["vertically_integrated_no2_enhancement"], color='tab:blue')
    ax1.set_title("Vertically integrated NO$_2$ enhancement")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Integrated NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Video image
    ax2 = fig.add_subplot(gs[0, 2])
    if img_file and os.path.exists(img_file):
        img = Image.open(img_file)
        ax2.imshow(img)
        ax2.set_title("Video image")
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No image", ha="center", va="center")
        ax2.axis("off")

    # Wind/ship polar plot
    wind_dir_rad = np.deg2rad(row['wind_dir'])
    wind_speed_mean = row['wind_speed']
    ax3 = fig.add_subplot(gs[0, 3], polar=True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    viewing_dir_deg = ds_measurements["viewing-azimuth-angle"].isel(viewing_direction=0).values[row["Closest_Impact_Measurement_Index"]]
    viewing_dir_rad = np.deg2rad(viewing_dir_deg)
    ax3.plot([viewing_dir_rad, viewing_dir_rad], [0, ax3.get_rmax()], color='red', linewidth=2, label='Viewing direction', linestyle='--')
    wind_arrow_length = 0.7
    ax3.arrow(
        wind_dir_rad, wind_speed_mean, 0, -wind_arrow_length,
        width=0.03, head_width=0.15, head_length=0.3, color='tab:blue', alpha=0.8
    )
    ax3.text(
        wind_dir_rad+np.deg2rad(10), wind_speed_mean + 1.0,
        f"{wind_speed_mean:.2f} m/s\n{wind_dir_rad:.0f}°",
        ha='center', va='bottom', fontsize=10, color='black'
    )
    ship_arrow_length = 0.7
    ax3.arrow(
        np.deg2rad(row["Mean_Course"]), row["Mean_Speed"], 0, ship_arrow_length,
        width=0.03, head_width=0.15, head_length=0.3, color='green', alpha=0.8
    )
    ax3.text(
        np.deg2rad(row["Mean_Course"])+np.deg2rad(10), row["Mean_Speed"] + 1,
        f"{row['Mean_Speed']:.2f} m/s\n{row['Mean_Course']:.0f}°",
        ha='center', va='bottom', fontsize=10, color='black'
    )
    view_line = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Viewing direction')
    wind_arrow = FancyArrowPatch((0, 0), (0.5, 0.5), color='tab:blue', arrowstyle='->', mutation_scale=15, label='Wind arrow')
    ship_arrow = FancyArrowPatch((0, 0), (0.5, 0.5), color='green', arrowstyle='->', mutation_scale=15, label='Ship velocity')
    ax3.legend(handles=[view_line, wind_arrow, ship_arrow], loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # In-situ NO2
    ax4 = fig.add_subplot(gs[0, 4])
    in_situ_mask = (df_insitu.index >= result["times_window"][0]) & (df_insitu.index <= result["times_window"][-1])
    in_situ_times = df_insitu.index[in_situ_mask]
    in_situ_no2 = df_insitu['c_no2'][in_situ_mask]
    if len(in_situ_times) > 0:
        ax4.plot(in_situ_times, in_situ_no2, color='tab:red')
        ax4.set_xlim(in_situ_times[0], in_situ_times[-1])
    else:
        ax4.text(0.5, 0.5, "No in-situ data", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("In-situ NO$_2$")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("NO$_2$ [ppb]")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=45)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(no2_out_dir, f"NO2_subplot_{result['t'].strftime('%Y%m%d_%H%M%S')}_{result['mmsi']}.png"))
        plt.close()
    else:
        plt.show()

def plot_ship_pass_subplot_v2(
        result, row, ds_measurements, df_insitu, df_lp_doas, no2_out_dir, filtered_ships, lat1, lon1, lat2, lon2, save=False
    ):

    img_file = None
    if "closest_image_file" in row and pd.notnull(row["closest_image_file"]):
        img_file = row["closest_image_file"]
    
    # Create subplot
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])

    # NO2 enhancement 2D plot
    ax0 = fig.add_subplot(gs[0, 0])
    X, Y = np.meshgrid(result["times_window"], np.arange(result["no2_enhancement"].shape[0]))
    pcm = ax0.pcolormesh(X, Y, result["no2_enhancement"], shading='auto')
    fig.colorbar(pcm, ax=ax0, label=r"NO$_2$ Enhancement / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax0.set_title(f"MMSI {result['mmsi']} NO2 around {result['t'].strftime('%Y-%m-%d %H:%M:%S')}")
    ax0.set_xlabel("Time")
    N = 3  # Show every 3rd LOS value
    los_vals = result["los"].isel(dim_0=0).values - 90
    yticks = np.arange(0, len(los_vals), N)
    ax0.set_yticks(yticks)
    ax0.set_yticklabels([f"{float(los_vals[i]):.1f}°" for i in yticks])
    ax0.set_ylabel("LOS / °")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax0.get_xticklabels(), rotation=45)

    # Integrated NO2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(result["times_window"], result["vertically_integrated_no2_enhancement"], color='tab:blue')
    ax1.set_title("Vertically integrated NO$_2$ enhancement")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Integrated NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Video image
    ax2 = fig.add_subplot(gs[0, 2])
    if img_file and os.path.exists(img_file):
        img = Image.open(img_file)
        ax2.imshow(img)
        ax2.set_title("Video image")
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No image", ha="center", va="center")
        ax2.axis("off")

    min_lon, max_lon = lon1 - 0.05, lon1 + 0.05
    min_lat, max_lat = lat1 - 0.04, lat1 + 0.02

    tiler = cimgt.OSM()
    mercator = tiler.crs
    ax3 = fig.add_subplot(gs[0, 3], projection=mercator)
    ax3.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax3.add_image(tiler, 13, interpolation='bilinear', alpha=1.0)
    ax3.set_facecolor('white')  

    # Plot ship trajectory (last and next 10 minutes, no current position marker)
    group = filtered_ships.get(result["mmsi"])
    if group is not None:
        traj_mask = (pd.to_datetime(group.index) >= result["t"] - pd.Timedelta(minutes=15)) & (pd.to_datetime(group.index) <= result["t"] + pd.Timedelta(minutes=15))
        traj_lons = group.loc[traj_mask, "Longitude"].astype(float).values
        traj_lats = group.loc[traj_mask, "Latitude"].astype(float).values
        if len(traj_lons) > 1:
            ax3.plot(traj_lons, traj_lats, color='green', linewidth=3, label='Ship trajectory', transform=ccrs.PlateCarree())

    # Plot instrument location
    ax3.plot(lon1, lat1, marker='x', color='black', markersize=12, label='Instrument', transform=ccrs.PlateCarree())

    # Plot viewing direction as a line from instrument
    view_lat2 = lat2[row["Closest_Impact_Measurement_Index"]]
    view_lon2 = lon2[row["Closest_Impact_Measurement_Index"]]
    ax3.plot([lon1, view_lon2], [lat1, view_lat2], color='red', linewidth=3, linestyle='--', label='Viewing direction', transform=ccrs.PlateCarree())

    # Add wind arrow (bottom left, bigger)
    lon_margin = 0.007
    lat_margin = 0.007


    # Wind arrow in upper right
    wind_dir_rad = np.deg2rad(row['wind_dir'])
    wind_speed_mean = row['wind_speed']
    wind_arrow_length = 0.01
    wind_start_lat = max_lat - lat_margin - wind_arrow_length * np.cos(wind_dir_rad) / 2
    wind_start_lon = max_lon - lon_margin - 0.035 - wind_arrow_length * np.sin(wind_dir_rad) / np.cos(np.deg2rad(wind_start_lat))/2
    wind_end_lon = wind_start_lon + wind_arrow_length * np.sin(wind_dir_rad) / np.cos(np.deg2rad(wind_start_lat))
    wind_end_lat = wind_start_lat + wind_arrow_length * np.cos(wind_dir_rad)

    # White box for wind arrow
    box_width = 0.025
    box_height = 0.005
    box = mpatches.FancyBboxPatch(
        (max_lon - 2*lon_margin-0.03, max_lat - lat_margin + 0.001),
        box_width, box_height,
        boxstyle="round,pad=0.01",
        linewidth=0,
        facecolor='white',
        alpha=0.8, 
        transform=ccrs.PlateCarree(),
        zorder=9
    )
    ax3.add_patch(box)

    # Draw wind arrow
    ax3.arrow(
        wind_start_lon, wind_start_lat,
        wind_end_lon - wind_start_lon, wind_end_lat - wind_start_lat,
        color='tab:blue', width=0.001, head_width=0.006, head_length=0.006,
        length_includes_head=True, transform=ccrs.PlateCarree(), zorder=10
    )
    ax3.text(
        max_lon - lon_margin - 0.03 +0.006, max_lat - lat_margin - 0.005,
        f"Wind\n{wind_speed_mean:.1f} m/s\n{row['wind_dir']:.0f}°",
        color='tab:blue', fontsize=10, ha='left', va='bottom',
        transform=ccrs.PlateCarree(), zorder=11,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
    )

    ship_dir_rad = np.deg2rad(row["Mean_Course"])
    ship_arrow_length = 0.01
    ship_start_lat = min_lat + lat_margin + 0.025 - ship_arrow_length * np.cos(ship_dir_rad) / 2
    ship_start_lon = min_lon + lon_margin + 0.005 - ship_arrow_length * np.sin(ship_dir_rad) / np.cos(np.deg2rad(ship_start_lat)) / 2
    ship_end_lon = ship_start_lon + ship_arrow_length * np.sin(ship_dir_rad) / np.cos(np.deg2rad(ship_start_lat))
    ship_end_lat = ship_start_lat + ship_arrow_length * np.cos(ship_dir_rad)
    # White box for ship arrow

    ship_box = mpatches.FancyBboxPatch(
        (min_lon , min_lat +0.025 ),
        2*lon_margin, box_height,
        boxstyle="round,pad=0.01",
        linewidth=0,
        facecolor='white',
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        zorder=9
    )
    ax3.add_patch(ship_box)
    # Draw ship arrow
    ax3.arrow(
        ship_start_lon, ship_start_lat,
        ship_end_lon - ship_start_lon, ship_end_lat - ship_start_lat,
        color='green', width=0.001, head_width=0.006, head_length=0.006,
        length_includes_head=True, transform=ccrs.PlateCarree(), zorder=10
    )    
    ax3.text(
        min_lon + lon_margin - 0.005, min_lat + lat_margin + 0.013,
        f"Ship\n{row["Mean_Speed"]:.1f} m/s",
        color='green', fontsize=10, ha='left', va='bottom',
        transform=ccrs.PlateCarree(), zorder=11,
        )
    ax3.set_facecolor('white')  # White background

    # Add latitude/longitude gridlines
    gl = ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                    linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax3.legend(loc='lower right', fontsize=10)
    ax3.set_title("Map: Ship, Wind, Instrument, Viewing Dir.")


    ax4 = fig.add_subplot(gs[0, 4])

    # Ensure times_window is tz-naive for comparison with ds.time
    in_situ_mask = (df_insitu.index >= result["times_window"][0]) & (df_insitu.index <= result["times_window"][-1])
    in_situ_times = df_insitu.index[in_situ_mask]
    in_situ_no2 = df_insitu['c_no2'][in_situ_mask]
    

    L = 870  # meters

    # Interpolate n_air to times_window
    n_air = df_insitu["p_0"][in_situ_mask] * 1e2 * 6.02214076e23 / (df_insitu["T_in"][in_situ_mask] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(in_situ_times))
    n_air_aligned = n_air_interp.reindex(pd.to_datetime(result["times_window"]), method='nearest').values

    # IMPACT: n_NO2 = no2_data / (L * n_air) * 1e9
    # Apply rolling mean over 20 dim_0 (time) for IMPACT
    impact_enh_rolling = result["no2_enhancement"].isel(viewing_direction=slice(5,9)).mean(dim="viewing_direction").rolling(dim_0=20, center=True).mean()
    n_NO2_impact = impact_enh_rolling * 1e4 / (L * n_air_aligned) * 1e9

    ax4.plot(result["times_window"], n_NO2_impact, color='orange', label="IMPACT n$_{NO_2}$ [ppb] (rolling mean)")
    ax4.plot(result["lp_times_window"], result["lp_no2_enhancement"], color='tab:blue', label="LP-DOAS n$_{NO_2}$ [ppb]")
    
    #In-situ enhancement: subtract mean in reference window (for plotting, not converted)
    #if len(in_situ_times) > 0:
    #    in_situ_ref_mask = (ds_times >= times_window_naive[0] - pd.Timedelta(minutes=ref_offset)) & \
    #                       (ds_times < times_window_naive[0] - pd.Timedelta(minutes=ref_offset - 1.0))
    #    in_situ_ref = ds['c_no2'].values[in_situ_ref_mask]
    #    in_situ_enh = in_situ_no2 - np.nanmean(in_situ_ref) if len(in_situ_ref) > 0 else in_situ_no2
    #    ax4.plot(in_situ_times, in_situ_enh, color='tab:red', label="In-situ NO$_2$ enhancement [ppb]")
    #else:
    #    ax4.text(0.5, 0.5, "No in-situ data", ha="center", va="center", transform=ax4.transAxes)

    ax4.set_title("NO$_2$ (IMPACT, In-situ, LP-DOAS, all as ppb)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("NO$_2$ [ppb]")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=45)
    ax4.legend(loc="upper left")
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(no2_out_dir, f"NO2_enhancement_subplot_{result['t'].strftime('%Y%m%d_%H%M%S')}_{result['mmsi']}.png"))
        plt.close()
    else:
        plt.show()

def plot_wind_polar(ds, wind_dir_var='wind_dir', wind_speed_var='wind_speed', title='Wind Speed and Direction (Polar Plot)'):
    """
    Plots wind speed as a function of wind direction in a polar plot.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing wind direction and wind speed.
    wind_dir_var : str
        Name of the wind direction variable in degrees.
    wind_speed_var : str
        Name of the wind speed variable.
    title : str
        Title for the plot.
    """

    wind_dir_rad = np.deg2rad(ds[wind_dir_var].values)
    wind_speed = ds[wind_speed_var].values

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    sc = ax.scatter(wind_dir_rad, wind_speed, c=wind_speed, cmap='viridis', alpha=0.75)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    plt.colorbar(sc, ax=ax, label='Wind Speed (m/s)')
    ax.set_title(title)
    ax.set_rlabel_position(225)

    plt.show()

def plot_no2_enhancement_and_insitu(
    ds_measurements,
    df_insitu,
    df_closest,
    title="NO2 time series and ship passes according to AIS",
    save_path=None
):
    """
    Plot NO2 enhancement (satellite) and in-situ NO2 with ship pass vertical lines.

    Parameters
    ----------
    ds_measurements : xarray.Dataset
        Dataset containing NO2 enhancement and datetime.
    df_insitu : pd.DataFrame
        DataFrame with in-situ NO2 data (index: datetime, column: 'c_no2').
    df_closest : pd.DataFrame
        DataFrame with ship pass times (index: datetime).
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    # Plot vertical lines for closest approach times within window
    for t in df_closest.index:
        ax1.axvline(pd.to_datetime(t), color='red', alpha=0.5)

    # Plot NO2 enhancement
    line1, = ax1.plot(
        ds_measurements["datetime"].isel(viewing_direction=0),
        ds_measurements["NO2_enhancement"].mean(dim="viewing_direction"),
        label="NO2 Enhancement",
        color='blue'
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Satellite NO2 Enhancement")
    ax1.set_title(title)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Plot in situ NO2 on a secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        df_insitu.index,
        df_insitu['c_no2'],
        label="In Situ NO2",
        color='green'
    )
    ax2.set_ylabel("In Situ NO2")

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_path_averaged_vmr_no2(df_insitu, ds):
    """
    Calculate NO2 volume mixing ratio (VMR) from coarsened IMPACT and in-situ data.
    !!! Note: This approach is not accurate and implements systematic errors 

    Args:
        df_insitu (pandas.DataFrame): In-situ dataframe with 'p_0', 'T_in', and 'time'.
        coarsened_ds (xr.Dataset): Coarsened IMPACT dataset with 'a[O4]', 'a[NO2]', and 'datetime'.

    Returns:
        xr.DataArray: VMR_NO2 aligned to coarsened_ds times [ppb].
    """
    # Calculate air number density [molec/m^3]
    n_air = df_insitu["p_0"] * 1e2 * 6.02214076e23 / (df_insitu["T_in"] + 273.15) / 8.314
    # Calculate O4 number density [molec^2/m^6]
    n_O4 = (0.20942 * n_air) ** 2

    # Interpolate n_O4 and n_air to coarsened_ds times
    n_O4_interp = pd.Series(n_O4.values, index=pd.to_datetime(df_insitu.index))
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(df_insitu.index))
    measurement_times = pd.to_datetime(ds["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")
    n_O4_aligned = n_O4_interp.reindex(measurement_times, method='nearest').values
    n_air_aligned = n_air_interp.reindex(measurement_times, method='nearest').values

    # Calculate O4 effective path length [m]
    L_O4 = ds["a[O4]"] * 1e40 * 1e10 / n_O4_aligned

    # Broadcast n_air_aligned to match coarsened_ds["a[NO2]"] shape
    n_air_aligned = xr.ones_like(ds["a[NO2]"]) * np.array(n_air_aligned)

    # Calculate NO2 VMR [ppb]
    VMR_NO2 = ds["a[NO2]"] * 1e4 / L_O4 / n_air_aligned * 1e9

    return VMR_NO2

def plot_all_instruments_timeseries_VMR(
    df_lp_doas,
    df_insitu,
    coarsened_ds,
    VMR_NO2,
    df_closest=None,
    title="NO$_2$ measurements from all instruments",
    save_path=None
):
    """
    Plots LP-DOAS, In Situ, and IMPACT NO2 timeseries with ship pass vertical lines.

    Parameters
    ----------
    df_lp_doas : pd.DataFrame
        LP-DOAS data (index: datetime, column: 'Fit Coefficient (NO2)').
    df_insitu : pd.DataFrame
        In-situ data (index: datetime, column: 'c_no2').
    coarsened_ds : xr.Dataset
        Coarsened IMPACT dataset with 'datetime' and NO2 VMR.
    VMR_NO2 : xr.DataArray
        Path-averaged NO2 VMR from IMPACT.
    df_closest : pd.DataFrame, optional
        DataFrame with ship pass times and categories.
    start_time, end_time : datetime, optional
        Time window for plotting.
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()


    # Plot ship pass vertical lines using df_closest and styles_from_ship_category
    if df_closest is not None:
        for idx, row in df_closest.iterrows():
            category = row.get("ship_category")
            label = styles_from_ship_category("label", category)
            # Only add label if not already present
            if label not in ax1.get_legend_handles_labels()[1]:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=label
                )
            else:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=""
                )

    # Plot LP-DOAS
    ax1.plot(
        df_lp_doas.index,
        df_lp_doas['Fit Coefficient (NO2)'],
        label="LP-DOAS"
    )

    # Plot In Situ NO2
    ax1.plot(
        df_insitu.index,
        df_insitu['c_no2'],
        label="In Situ", color="green"
    )

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel("In-Situ & LP-DOAS VMR / ppb", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    #ax1.set_ylim(0, 35)

    # Second y-axis: IMPACT
    ax2 = ax1.twinx()
    ax2.plot(
        coarsened_ds["datetime"].isel(viewing_direction=0),
        VMR_NO2.isel(viewing_direction=slice(4,8)).mean(dim="viewing_direction"),
        label="IMPACT", color='C1', zorder=1
    )
    ax2.set_ylabel("IMPACT VMR / ppb", fontsize=14, color='C1')
    ax2.tick_params(axis='both', labelsize=12)
    #ax2.set_ylim(0, 3.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_all_instruments_timeseries_SC(
    df_lp_doas_SC,
    df_insitu,
    coarsened_ds,
    df_closest=None,
    ylim_left=None,
    ylim_right=None,
    title="NO$_2$ measurements from all instruments",
    save_path=None
):
    """
    Plots LP-DOAS, In Situ, and IMPACT NO2 timeseries with ship pass vertical lines.

    """
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()


    # Plot ship pass vertical lines using df_closest and styles_from_ship_category
    if df_closest is not None:
        for idx, row in df_closest.iterrows():
            category = row.get("ship_category")
            label = styles_from_ship_category("label", category)
            # Only add label if not already present
            if label not in ax1.get_legend_handles_labels()[1]:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=label
                )
            else:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=""
                )

    # Plot LP-DOAS
    ax1.plot(
        df_lp_doas_SC.index,
        df_lp_doas_SC['Fit Coefficient (NO2)'],
        label="LP-DOAS"
    )

    # Plot In Situ NO2
    ax1.plot(
        coarsened_ds["datetime"].isel(viewing_direction=0),
        coarsened_ds["a[NO2]"].isel(viewing_direction=slice(5,9)).mean(dim="viewing_direction"),
        label="IMPACT"
    )

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_ylim(0, ylim_left)

    # Second y-axis: IMPACT
    ax2 = ax1.twinx()
    ax2.plot(df_insitu.index, df_insitu["c_no2"], color='green',zorder=1, alpha=0.6, label="In Situ")
    ax2.set_ylabel(r"In Situ NO$_2$ / ppb", color="green", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, ylim_right) 

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def mask_rms_and_reduce_impact(ds_measurements, threshold=0.01):
    rms_mean = ds_measurements["rms"].isel(viewing_direction=slice(5, 9)).mean(dim="viewing_direction")
    mask = rms_mean < threshold
    time_variable = ds_measurements["datetime"].isel(viewing_direction = 0) 
    ds_impact_reduced = ds_measurements.isel(viewing_direction=slice(5, 9)).mean(dim="viewing_direction")
    ds_impact_reduced["datetime"] = time_variable
    ds_impact_masked = ds_impact_reduced.where(mask, drop=True)
    return mask, ds_impact_masked

def polynomial_background_enh(ds_impact_masked, degree=8): 
    time_diff = pd.to_datetime(ds_impact_masked["datetime"]) - pd.to_datetime(ds_impact_masked["datetime"])[0]
    x = time_diff.total_seconds().astype(float)
    y = ds_impact_masked["a[NO2]"]

    # Fit 6th degree polynomial
    coeffs = np.polyfit(x, y, degree)
    poly_fit = np.polyval(coeffs, x)

    ds_impact_masked["NO2_polynomial"] = xr.DataArray(
    poly_fit,
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )
    ds_impact_masked["NO2_enhancement_polynomial"] = xr.DataArray(
    y - poly_fit,
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )

    return ds_impact_masked

def plot_timeseries_and_enhancement(ds_impact_masked, time, orig , fit, enhancement, title= f"Correction for NO$_2$-Background"):

    plt.figure(figsize=(15, 5))
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.plot(pd.to_datetime(time), ds_impact_masked[orig], label='NO$_2$ dSCDs', alpha=0.6, linewidth=2)
    ax1.plot(pd.to_datetime(time), ds_impact_masked[fit], label='Fit', color='red', linewidth=2)
    ax1.plot(pd.to_datetime(time), ds_impact_masked[enhancement], label='Enhancements', color='red', linewidth=2)

    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

def polynomial_background_enh_lp_doas(df_lp_doas, degree = 8):

    x_lp = (df_lp_doas.index - df_lp_doas.index[0]).total_seconds().astype(float)
    y_lp = df_lp_doas['Fit Coefficient (NO2)']

    coeffs_lp = np.polyfit(x_lp, y_lp, degree)
    poly_fit_lp = np.polyval(coeffs_lp, x_lp)

    # Calculate enhancement (detrended)
    lpdoas_enhancement = y_lp - poly_fit_lp
    df_lp_doas['NO2_polynomial'] = poly_fit_lp
    df_lp_doas['NO2_enhancement_polynomial'] = lpdoas_enhancement

    return df_lp_doas

def plot_timeseries_and_enhancement_lpdoas(df_lp_doas):
    plt.figure(figsize=(12, 5))
    plt.plot(df_lp_doas.index, df_lp_doas['Fit Coefficient (NO2)'], label='Original Data', alpha=0.6, linewidth=2)
    plt.plot(df_lp_doas.index, df_lp_doas['NO2_polynomial'], label='Polynomial Fit (deg 8)', color='red', linewidth=2)
    plt.plot(df_lp_doas.index, df_lp_doas['NO2_enhancement_polynomial'], label='Enhancement', color='red', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('NO$_2$ SC')
    plt.title('LP-DOAS Original Data and Polynomial Fit')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def fft_background_enh(ds_impact_masked):

    time_diff = pd.to_datetime(ds_impact_masked["datetime"]) - pd.to_datetime(ds_impact_masked["datetime"])[0]
    x = time_diff.total_seconds().astype(float)
    dt = np.median(np.diff(x)) #!!! todo: this fft assumes evenly spaced data, creating it by interpolation should be valid
    N = len(x)

    # FFT
    Y = fft(ds_impact_masked["a[NO2]"])
    freqs = fftfreq(N, d=dt)  # in Hz

    # 10 min period = 600 s, so frequency = 1/600 Hz
    f_cut = 1/3000  # Hz

    # Zero out frequencies with |f| < f_cut (i.e., periods > 10 min)
    Y_filtered = Y.copy()
    Y_filtered[np.abs(freqs) < f_cut] = 0

    # Inverse FFT to get filtered signal
    ds_impact_masked["NO2_fft_filter"] = xr.DataArray(
    np.real(ifft(Y_filtered)),
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )
    ds_impact_masked["NO2_enhancements_fft_filter"] = xr.DataArray(
    ds_impact_masked["a[NO2]"] - ds_impact_masked["NO2_fft_filter"],
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )

    return ds_impact_masked

def fft_background_enh_lp_doas(df_lp_doas):
    x_lp = (df_lp_doas.index - df_lp_doas.index[0]).total_seconds().astype(float)
    y_lp = df_lp_doas['Fit Coefficient (NO2)']
    # Sampling interval in seconds for LP-DOAS (assumes x_lp is in seconds and evenly spaced)
    dt_lp = np.median(np.diff(x_lp))
    N_lp = len(y_lp)

    # FFT
    Y_lp = fft(y_lp)
    freqs_lp = fftfreq(N_lp, d=dt_lp)  # in Hz

    # 10 min period = 600 s, so frequency = 1/600 Hz
    f_cut_lp = 1/3000  # Hz

    # Zero out frequencies with |f| < f_cut_lp (i.e., periods > 10 min)
    Y_lp_filtered = Y_lp.copy()
    Y_lp_filtered[np.abs(freqs_lp) < f_cut_lp] = 0

    # Inverse FFT to get filtered signal
    df_lp_doas["NO2_fft_filter"] = np.real(ifft(Y_lp_filtered))

    # Plot original and filtered signals
    df_lp_doas["NO2_enhancements_fft_filter"] = y_lp - df_lp_doas["NO2_fft_filter"]

    return df_lp_doas

def plot_enhancement_impact_and_insitu(
    ds_impact_masked,
    df_lp_doas_SC,
    df_insitu,
    save_path=None
    ):
    L_impact = 870  # meters (IMPACT path length)
    L_lp = 870      # meters (LP-DOAS path length, adjust if different)

    # Calculate n_air (air number density) at each time (in molec/m^3)
    n_air = df_insitu["p_0"] * 1e2 * 6.02214076e23 / (df_insitu["T_in"] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(df_insitu.index))

    # Align n_air to IMPACT and LP-DOAS times
    n_air_impact = n_air_interp.reindex(pd.to_datetime(ds_impact_masked["datetime"]).tz_localize('UTC'), method='nearest').values
    n_air_lp = n_air_interp.reindex(df_lp_doas_SC.index, method='nearest')

    # Convert enhancements to VMR [ppb]
    # IMPACT: enhancement in 1/cm^2, convert to 1/m^2, then to VMR
    ds_impact_masked["Enhancement_VMR_lpdoas"] = ds_impact_masked["NO2_enhancement_polynomial"] * 1e4 / (L_impact * n_air_impact) * 1e9  # [ppb]

    # LP-DOAS: enhancement in 1/cm^2, convert to 1/m^2, then to VMR
    lpdoas_vmr = df_lp_doas_SC["NO2_enhancement_polynomial"] * 1e4 / (2 * L_lp * n_air_lp) * 1e9  # [ppb], factor 2 for roundtrip

    # --- Plot with ship passes colored by ship height ---
    plt.figure(figsize=(15, 5))
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"$\Delta$VMR NO$_2$ / ppb", fontsize=14)
    ax1.set_title(f"NO$_2$ Enhancements for IMPACT and LP-DOAS", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    add_ship_lines = True
    if add_ship_lines and df_closest is not None and start_time is not None and end_time is not None:
        for idx, row in df_closest.iterrows():
            t_ship = pd.to_datetime(row["UTC_Time"]) if "UTC_Time" in row else idx
            category = row.get("ship_category")
            if (t_ship >= start_time) and (t_ship <= end_time):
                label = styles_from_ship_category("label", category)
                # Only add label if not already present
                if label not in ax1.get_legend_handles_labels()[1]:
                    ax1.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=label
                    )
                else:
                    ax1.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=""
                    )


    ax1.plot(df_lp_doas.index, lpdoas_vmr, label='LP-DOAS', alpha=0.8)
    ax1.plot(ds_impact_masked["datetime"], ds_impact_masked["Enhancement_VMR_lpdoas"], label='IMPACT')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(no2_out_dir, f"NO2_enhancement_vmr_{date}.png"))

date=f"250511"
path= r"P:\data\data_tmp\SLANT_25\MAI2025"
in_situ_path = r"P:\data\data_tmp\InSitu"
img_dir = r"P:\data\data_tmp\video\{}_video".format(date)
img_out_dir = r"P:\data\data_tmp\video\{}_video_ships".format(date)
no2_out_dir = r"P:\data\data_tmp\analysis\{}_no2plots".format(date)
ais_dir = r"P:\data\data_tmp\AIS\shipplotter{}.log".format(date)
lp_doas_dir = r"P:\data\data_tmp\Open_Path\20{}_365_Eval_ppb.dat".format(date)


instrument_location = [53.56958522848946, 9.69174249821205]
lat_lon_window = [53.4, 53.8, 9.3, 10.0]  # [lat_min, lat_max, lon_min, lon_max] area of interest for ais data 
lat_lon_window_small = [53.55, 53.60, 9.6, 9.8]  # [lat_min, lat_max, lon_min, lon_max] area of interest for ais data
#%% Initialize IMPACT measurements
ds_measurements = read_SC_file_imaging(path, date, f"ID.NO2_VIS_zenith_sync")

ds_measurements = mask_zenith(ds_measurements)
ds_measurements = rolling_background_enh(ds_measurements, window_size=500)
endpoints_los = calculate_LOS(ds_measurements, instrument_location)
lat1, lon1 = instrument_location
lat2, lon2 = endpoints_los[:, 0], endpoints_los[:, 1] #todo: to be replaced 
start_time, end_time, measurement_times = calc_start_end_times(ds_measurements)
#%% Initialize ais data
ds = prepare_ais(ais_dir, 60)
#%% Filter AIS data
ship_groups, filtered_ships, maskedout_ships, df_closest = filter_ais(ds, 
    lat_lon_window, start_time, end_time, measurement_times, endpoints_los, instrument_location, length=20, distance_threshold=0.002)


#%%
plot_trajectories(filtered_ships, maskedout_ships, df_closest, lon1, lon2, lat1, lat2, lat_lon_window_small)
plot_maskedout_ships_details(maskedout_ships, lat_lon_window_small, lon1, lon2, lat1, lat2)
plot_ship_stats(filtered_ships)
#%%
ds_masked = rms_mask(ds_measurements, threshold=0.01, instrument="IMPACT")
#%%
plot_no2_timeseries(ds_masked, df_closest, start_time, end_time, separate_legend=True)
#%%
plot_single_ship(
    ds_measurements, 
    t_after_start_h=9.7,
    interval_h=0.1, mode="dSCD"
)
# %%
df_closest = assign_video_images_to_ship_pass(df_closest, img_dir, utc=pytz.utc)
copy_ship_images(df_closest, img_dir, img_out_dir, time_threshold=30)
#%%
plot_no2_enhancements_for_all_ships(df_closest, ds_measurements, measurement_times, no2_out_dir)

#%%

df_insitu = read_in_situ(in_situ_path, date)
df_insitu = apply_time_mask_to_insitu(df_insitu, start_time, end_time)
df_closest = add_wind_to_ship_passes(df_closest, df_insitu)

# %%

plot_wind_polar(df_insitu)

#%%

for idx, row in df_closest.iterrows():
    result = upwind_constant_background_enh(row, ds_measurements, measurement_times, df_closest)
    if result is None:
        continue
    plot_ship_pass_subplot_v1(result, row, ds_measurements, df_insitu, no2_out_dir, save=True)

#%%
plot_no2_enhancement_and_insitu(
    ds_measurements,
    df_insitu,
    df_closest,
    save_path=None
)

# %%
df_lp_doas = read_lpdoas(lp_doas_dir, date)
df_lp_doas = mask_lp_doas_file(df_lp_doas, start_time, end_time)
# %%
coarsened_ds = coarsen_impact_measurements(ds_measurements, 40)
# Restrict to min/max times of ds_measurements["datetime"]

# %%
VMR_NO2 = calculate_path_averaged_vmr_no2(df_insitu, coarsened_ds)

#%%
plot_all_instruments_timeseries_VMR(
    df_lp_doas,
    df_insitu,
    coarsened_ds,
    VMR_NO2,
    df_closest=df_closest,
    title=f"NO$_2$ measurements on {date}",
    save_path=None
)

# %%
df_lp_doas_SC = read_lpdoas(lp_doas_dir, date, mode="SC")
df_lp_doas_SC = mask_lp_doas_file(df_lp_doas_SC, start_time, end_time)

#%%
plot_all_instruments_timeseries_SC(
    df_lp_doas_SC,
    df_insitu,
    coarsened_ds,
    df_closest=df_closest,
    title="NO$_2$ measurements on {date}"
)


# %%

for idx, row in df_closest.iterrows():
    result = upwind_constant_background_enh(row, ds_measurements, measurement_times, df_closest, do_lp= True, df_lp = df_lp_doas)
    if result is None:
        continue
    plot_ship_pass_subplot_v2(result, row, ds_measurements, df_insitu, df_lp_doas, no2_out_dir, filtered_ships, lat1, lon1, lat2, lon2, save=True)

# %%

mask, ds_impact_masked = mask_rms_and_reduce_impact(ds_measurements)
ds_impact_masked = polynomial_background_enh(ds_impact_masked)
plot_timeseries_and_enhancement(ds_impact_masked, ds_impact_masked["datetime"], "a[NO2]", "NO2_polynomial", "NO2_enhancement_polynomial")

df_lp_doas_SC = polynomial_background_enh_lp_doas(df_lp_doas_SC, degree=8)
plot_timeseries_and_enhancement(df_lp_doas_SC, df_lp_doas_SC.index, "Fit Coefficient (NO2)", "NO2_polynomial", "NO2_enhancement_polynomial")


ds_impact_masked = fft_background_enh(ds_impact_masked)
plot_timeseries_and_enhancement(ds_impact_masked, ds_impact_masked["datetime"], "a[NO2]", "NO2_fft_filter", "NO2_enhancements_fft_filter")


df_lp_doas = fft_background_enh_lp_doas(df_lp_doas)
plot_timeseries_and_enhancement(df_lp_doas, df_lp_doas_SC.index, "Fit Coefficient (NO2)", "NO2_fft_filter", "NO2_enhancements_fft_filter")


# %%
