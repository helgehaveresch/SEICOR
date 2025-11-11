#%%
import numpy as np
import pandas as pd
import re
import xarray as xr
import shutil
import matplotlib.pyplot as plt
import sys
from itertools import groupby
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import process_SC_img_data
from scipy.signal import find_peaks
import os
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from PIL import Image
import pytz

from io import StringIO
from glob import glob
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator



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

def prepare_ais(file, interpolation_limit):
    data = read_ais(file)
    #data = interpolate_ais(data, interpolation_limit)
    
    #data = data.to_xarray()
    return data

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


date=f"250531"

for i in range(15,31):
    if i < 10:
        date = f"2505{str(i).zfill(2)}"
    else:
        date = f"2505{i}"
    


    #%%
    path= r"P:\data\data_tmp\SLANT_25\MAI2025"
    ds_measurements = read_SC_file_imaging(path, date, f"ID.NO2_VIS_zenith")
    #%%
    mask = ds_measurements["los"] == 180
    ds_measurements = ds_measurements.where(~mask, drop=True)

    ds_measurements["NO2_enhancement"] = ds_measurements["a[NO2]"] - ds_measurements["a[NO2]"].rolling(dim_0=500).mean()
    ds_measurements["O4_enhancement"] = ds_measurements["a[O4]"] - ds_measurements["a[O4]"].rolling(dim_0=500).mean()

    #%%
    ds = prepare_ais(r"P:\data\data_tmp\AIS\shipplotter{}.log".format(date), 60)
    ds.index = pd.to_datetime(ds.index)
    ds["filtermask"] = 0

    lon_mask = (ds["Longitude"] < 9.3) | (ds["Longitude"] > 10.0)
    lat_mask = (ds["Latitude"] < 53.4) | (ds["Latitude"] > 53.8)

    ds.loc[lon_mask, "Longitude"] = np.nan
    ds.loc[lat_mask, "Latitude"] = np.nan

    """
    xr_ds = ds.to_xarray()

    xr_ds = xr_ds.assign_coords(
        UTC_Time=pd.to_datetime(xr_ds.coords['UTC_Time'].values)
    )

    xr_ds = xr_ds.astype({
        "MMSI": "int64",
        "ERI": "str",
        "Navigation_status": "str",
        "Speed_in_m/s": "float64",
        "Latitude": "float64",
        "Longitude": "float64",
        "Course_in_deg": "float64",
        "True_heading_in_deg": "float64",
        "Serial": "str",
        "Name": "str",
        "Type": "str",
        "Length_in_m": "float64",
        "Width_in_m": "float64",
        "Draught_in_m": "float64",
        "Destination": "str"
    })
    """
    # %%

    # %%
    instrument_location = [53.56958522848946, 9.69174249821205]
    lat1, lon1 = instrument_location
    line_length = 0.05  

    viewing_direction = ds_measurements["viewing-azimuth-angle"].isel(viewing_direction=0).values

    bearing_rad = np.deg2rad(viewing_direction)
    delta_lat = line_length * np.cos(bearing_rad)
    delta_lon = line_length * np.sin(bearing_rad) / np.cos(np.deg2rad(lat1))
    lat2 = lat1 + delta_lat
    lon2 = lon1 + delta_lon
    endpoints_los = np.column_stack([lat2, lon2])

    measurement_times = pd.to_datetime(ds_measurements["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")
    start_time = pd.to_datetime(measurement_times.min())
    end_time = pd.to_datetime(measurement_times.max())


    # %%
    sailing_mask = ds["Type"].str.lower() == "vessel:sail"
    anchored_mask = ds["Navigation_status"].str.lower().isin(["anchored"])
    timemask = (ds.index >= start_time) & (ds.index <= end_time)

    ds["filtermask"] = 0
    ds.loc[~timemask, "filtermask"] = 1
    ds.loc[anchored_mask, "filtermask"] = 2
    ds.loc[sailing_mask, "filtermask"] = 3 #%% todo: all filtering just by masks?

    # 1. Filter out all ships that have Type 'Vessel:sail' or are anchored
    sailing_mmsi = set(ds[ds["Type"].str.lower() == "vessel:sail"]["MMSI"].unique())



    anchored_mask = ds["Navigation_status"].str.lower().isin(["anchored"])
    anchored_mmsi = set(ds[anchored_mask]["MMSI"].unique())
    sailing_mmsi = sailing_mmsi | anchored_mmsi

    # Start from grouping ships by MMSI
    ship_groups = ds.groupby("MMSI")
    # Only keep ship data within the measurement time interval
    filtered_ship_groups = []
    for mmsi, group in ship_groups:
        if mmsi not in sailing_mmsi:
            group_times = pd.to_datetime(group.index)
            mask = (group_times >= start_time) & (group_times <= end_time)
            filtered_group = group.loc[mask]
            if not filtered_group.empty:
                filtered_ship_groups.append((mmsi, filtered_group))
    
    # Now, for each ship position, find the closest measurement time and use the corresponding endpoint
    distances = []
    closest_points = []
    for mmsi, group in filtered_ship_groups:
        lats = group["Latitude"].values
        lons = group["Longitude"].values
        ais_times = pd.to_datetime(group.index)
        positions = np.stack([lats, lons], axis=1)
        #print("length of lats before" +str(len(lats)))
        # Only consider positions within 0.02 degrees of the instrument
        #close_to_instr = (np.abs(lats - lat1) <= 0.05) & (np.abs(lons - lon1) <= 0.02)
        #if not np.any(close_to_instr):
        #    continue  # Skip if no positions are close
    #
        #lats = lats[close_to_instr]
        #lons = lons[close_to_instr]
        #ais_times = ais_times[close_to_instr]
        #positions = positions[close_to_instr]
        #print("length of lats after" +str(len(lats)))
        # Vectorized: find closest measurement index for all times
        idxs = np.abs(measurement_times.values[:, None] - ais_times.values).argmin(axis=0)
        lat2_ship = endpoints_los[idxs, 0]
        lon2_ship = endpoints_los[idxs, 1]

        # Vectorized distance calculation
        ab = np.stack([lat2_ship - lat1, lon2_ship - lon1], axis=1)
        ab_norm = ab / np.linalg.norm(ab, axis=1)[:, None]
        ap = positions - np.array([lat1, lon1])
        proj = np.sum(ap * ab_norm, axis=1)[:, None] * ab_norm
        perp = ap - proj
        dists = np.linalg.norm(perp, axis=1)

        distances.append((mmsi, dists))

        close_positions_mask = dists <= 0.002
        # Find contiguous segments (passes) using numpy
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
            closest_points.append({
                "MMSI": mmsi,
                "UTC_Time": group.index[min_idx],
                "Latitude": lats[min_idx],
                "Longitude": lons[min_idx],
                "Distance": dists[min_idx]
            })
    filtered_mmsi = [point["MMSI"] for point in closest_points]
    filtered_ship_groups = {mmsi: group for mmsi, group in filtered_ship_groups if mmsi in filtered_mmsi}

    maskedout_mmsi = set(ds["MMSI"].unique()) - set(filtered_mmsi)
    maskedout_groups = {mmsi: group for mmsi, group in ship_groups if mmsi in maskedout_mmsi}
    filtered_ship_groups = dict(filtered_ship_groups)

    # %%
    # Plot the ships that are left after filtering (filtered_ship_groups)
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ship_groups.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Non-sailing ships with minimum distance ≤ 0.005 deg to LOS")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    plt.legend(loc="best", fontsize="small", ncol=2)
    #plt.show()

    # Plot the masked out ships (either sailing vessels or distance > 0.005)

    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_groups.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    #plt.legend(loc="best", fontsize="small", ncol=2)
    #plt.show()

    #%%
    # Plot the ships that are left after filtering (filtered_ship_groups)
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ship_groups.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Non-sailing ships with minimum distance ≤ 0.005 deg to LOS")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    plt.xlim(9.6, 9.8)
    plt.ylim(53.55, 53.60)
    plt.legend(loc="best", fontsize="small", ncol=2)
    #plt.show()


    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_groups.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships (closeup)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    plt.xlim(9.6, 9.8)
    plt.ylim(53.55, 53.60)
    #plt.legend(loc="best", fontsize="small", ncol=2)
    #plt.show()

    #%%
    # Plot ships that pass the LOS multiple times
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ship_groups.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ships passing LOS multiple times (distance ≤ 0.005 deg)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    plt.legend(loc="best", fontsize="small", ncol=2)
    #plt.show()
    # %%
    # Length histogram
    lengths = pd.concat([group["Length_in_m"] for group in filtered_ship_groups.values()]).dropna()
    plt.figure()
    plt.hist(lengths, bins=30, alpha=0.7)
    plt.xlabel("Length (m)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Lengths (filtered ships)")
    #plt.show()

    # Draught histogram
    draughts = pd.concat([group["Draught_in_m"] for group in filtered_ship_groups.values()]).dropna()
    plt.figure()
    plt.hist(draughts, bins=30, alpha=0.7)
    plt.xlabel("Draught (m)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Draughts (filtered ships)")
    #plt.show()

    # Velocity histogram
    velocities = pd.concat([group["Speed_in_m/s"] for group in filtered_ship_groups.values()]).dropna()
    plt.figure()
    plt.hist(velocities, bins=30, alpha=0.7)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Velocities (filtered ships)")
    #plt.show()

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
    #plt.show()

    corr = length_draught_df["Length_in_m"].corr(length_draught_df["Draught_in_m"])
    print(f"Correlation between length and draught: {corr:.2f}")
    # %%

    # %%

    closest_times = [point["UTC_Time"] for point in closest_points]

    plt.figure(figsize=(12, 4))
    for t in closest_times:
        plt.axvline(t, color='red', alpha=0.5)
    plt.plot(ds_measurements["datetime"].isel(viewing_direction=0), ds_measurements["NO2_enhancement"].mean(dim="viewing_direction"), label="NO2 Enhancement")
    plt.xlabel("Time")
    plt.title("Closest approach times of filtered ships (vertical lines)")
    plt.tight_layout()
    #plt.show()

    # %%
    # Plot the line
    plot_LOS(lon1, lon2, lat1, lat2)
    for mmsi, group in ship_groups:
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.2)
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')
    plt.legend()


    # %%


    # Directory with images and output directory
    img_dir = r"P:\data\data_tmp\video\{}_video".format(date)
    out_dir = r"P:\data\data_tmp\video\{}_video_ships".format(date)
    os.makedirs(out_dir, exist_ok=True)

    # Helper to parse image filename to datetime
    def parse_img_time(fname):
        # Handles formats:
        # 1. yymmdd_hhmmssms.JPG
        # 2. yymmdd_hhmmssmms-??_???-??.JPG
        import re
        base = os.path.splitext(fname)[0]
        # Try standard format first
        try:
            return datetime.strptime(base, "%y%m%d_%H%M%S%f")
        except Exception:
            pass
        # Try extended format: yymmdd_hhmmssmms-??_???-??
        # Extract the first part before the first dash
        match = re.match(r"(\d{6}_\d{8,9})", base)
        #print(match)
        if match:
            try:
                return datetime.strptime(match.group(1), "%y%m%d_%H%M%S%f")
            except Exception:
                pass
        return None

    # List all image files and their datetimes
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    img_times = [(f, parse_img_time(f)) for f in img_files]
    img_times = [(f, t) for f, t in img_times if t is not None]

    for point in closest_points:
        t = point["UTC_Time"]
        mmsi = point["MMSI"]
        # Remove timezone info for comparison if present
        if hasattr(t, 'tz_localize') or hasattr(t, 'tzinfo'):
            t_naive = t.replace(tzinfo=None)
        else:
            t_naive = t
        
        utc = pytz.timezone("UTC")
        cet = pytz.timezone("Europe/Berlin")
        t_utc = utc.localize(t_naive)
        t_cet = t_utc.astimezone(cet)

        # Calculate the offset between UTC and CET in hours
        offset_hours = int((t_cet.utcoffset().total_seconds()) // 3600)
        t_plusoffset = t_naive + timedelta(hours=offset_hours)

        # Find closest image
        closest_img = min(img_times, key=lambda x: abs(x[1] - t_plusoffset))
        time_diff = abs((closest_img[1] - t_plusoffset).total_seconds())
        if time_diff <= 30:
            src = os.path.join(img_dir, closest_img[0])
            # Change name to originalname+mmsi+.jpg
            base, ext = os.path.splitext(closest_img[0])
            new_name = f"{base}_{mmsi}.jpg"
            dst = os.path.join(out_dir, new_name)
            shutil.copy2(src, dst)
            #print(f"Copied {closest_img[0]} as {new_name} for time {t} (+2h: {t_plus2h})")
        else:
            print(f"No suitable image found for MMSI {mmsi} at time {t} (+2h: {t_plusoffset}), closest image was {closest_img[0]} with time difference {time_diff:.2f} seconds")
    print(f"Copied Images of ships")
    #%%


    # Output directory for NO2 plots
    no2_out_dir = r"P:\data\data_tmp\analysis\{}_no2plots".format(date)
    os.makedirs(no2_out_dir, exist_ok=True)

    # Convert measurement times to pandas DatetimeIndex for fast lookup
    measurement_times = pd.to_datetime(ds_measurements["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")

    for point in closest_points:
        mmsi = point["MMSI"]
        # Check ship size (length > 20 m)
        group = filtered_ship_groups.get(mmsi)
        if group is not None:
            length = group["Length_in_m"].iloc[0]
            if length is None or length <= 20:
                continue  # Skip if ship is too small or length is missing
        else:
            continue  # Skip if group not found
        t = pd.to_datetime(point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(point["UTC_Time"]).tzinfo is None else pd.to_datetime(point["UTC_Time"])
        # Find index of closest measurement time
        idx = (abs(measurement_times - t)).argmin()
        # Only plot if closest measurement is within 30 seconds
        time_diff = abs((measurement_times[idx] - t).total_seconds())
        if time_diff > 60:
            print(time_diff)
            print(f"No measurement within 30s for MMSI {mmsi} at {t} (closest diff: {time_diff:.2f}s)")
            continue
        #timewindow of 5 minutes around t
        window = (abs(measurement_times - t) <= pd.Timedelta(minutes=2))
        window = ((measurement_times >= t - pd.Timedelta(minutes=1)) & (measurement_times < t + pd.Timedelta(minutes=3)))
        # Find a reference window with no other ship within ±5 min
        ref_found = False
        ref_offset = 3  # start 3 min before t
        while not ref_found and ref_offset < 60:  # don't go back more than 60 min
            ref_start = t - pd.Timedelta(minutes=ref_offset)
            ref_end = t - pd.Timedelta(minutes=ref_offset - 1.0)
            window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
            # Check if any closest_points are within ±5 min of this window
            ref_times = measurement_times[window_ref]
            other_ships_in_window = False
            for other_point in closest_points:
                if other_point["MMSI"] == mmsi:
                    continue
                other_t = pd.to_datetime(other_point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(other_point["UTC_Time"]).tzinfo is None else pd.to_datetime(other_point["UTC_Time"])
                if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                    other_ships_in_window = True
                    break
            if not other_ships_in_window and window_ref.sum() > 0:
                ref_found = True
            else:
                ref_offset += 1  # go further back
        if not ref_found:
            print(f"No clean reference window found for MMSI {mmsi} at {t}")
            continue
        # Select data in window
        no2_data = ds_measurements["a[NO2]"].isel(dim_0=window) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
        times_window = measurement_times[window]
        # Plot
        plt.figure(figsize=(8, 4))
        X, Y = np.meshgrid(times_window, np.arange(no2_data.shape[0]))
        plt.pcolormesh(X, Y, no2_data, shading='auto')
        plt.colorbar(label="a[NO2]")
        plt.title(f"MMSI {mmsi} NO2 around {t}")
        plt.xlabel("Viewing direction index")
        plt.ylabel("Time")
        plt.tight_layout()
        plt.savefig(os.path.join(no2_out_dir, f"NO2_{t.strftime('%Y%m%d_%H%M%S')}_{mmsi}.png"))
        plt.close('all')


    #%%
    # ...existing code...



    for point in closest_points:
        mmsi = point["MMSI"]
        group = filtered_ship_groups.get(mmsi)
        if group is not None:
            length = group["Length_in_m"].iloc[0]
            if length is None or length <= 20:
                continue
        else:
            continue
        t = pd.to_datetime(point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(point["UTC_Time"]).tzinfo is None else pd.to_datetime(point["UTC_Time"])
        idx = (abs(measurement_times - t)).argmin()
        time_diff = abs((measurement_times[idx] - t).total_seconds())
        if time_diff > 60:
            print(time_diff)
            print(f"No measurement within 30s for MMSI {mmsi} at {t} (closest diff: {time_diff:.2f}s)")
            continue
        window = (abs(measurement_times - t) <= pd.Timedelta(minutes=5))
        ref_found = False
        ref_offset = 3
        while not ref_found and ref_offset < 60:
            ref_start = t - pd.Timedelta(minutes=ref_offset)
            ref_end = t - pd.Timedelta(minutes=ref_offset - 1.0)
            window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
            ref_times = measurement_times[window_ref]
            other_ships_in_window = False
            for other_point in closest_points:
                if other_point["MMSI"] == mmsi:
                    continue
                other_t = pd.to_datetime(other_point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(other_point["UTC_Time"]).tzinfo is None else pd.to_datetime(other_point["UTC_Time"])
                if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                    other_ships_in_window = True
                    break
            if not other_ships_in_window and window_ref.sum() > 0:
                ref_found = True
            else:
                ref_offset += 1
        if not ref_found:
            print(f"No clean reference window found for MMSI {mmsi} at {t}")
            continue

        # Data for plotting
        no2_data = ds_measurements["a[NO2]"].isel(dim_0=window) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
        times_window = measurement_times[window]
        ref_no2 = ds_measurements["a[NO2]"].isel(dim_0=window_ref) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")

        # Try to find the corresponding image
        img_file = None
        img_dir = r"P:\data\data_tmp\video\{}_video".format(date)
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        img_times = [(f, parse_img_time(f)) for f in img_files]
        img_times = [(f, t_img) for f, t_img in img_times if t_img is not None]
        t_naive = t.replace(tzinfo=None)
        utc = pytz.timezone("UTC")
        cet = pytz.timezone("Europe/Berlin")
        t_utc = utc.localize(t_naive)
        t_cet = t_utc.astimezone(cet)

        # Calculate the offset between UTC and CET in hours
        offset_hours = int((t_cet.utcoffset().total_seconds()) // 3600)
        t_plusoffset = t_naive + timedelta(hours=offset_hours)
        #t_plus2h = t_naive + timedelta(hours=2)
        if img_times:
            closest_img = min(img_times, key=lambda x: abs(x[1] - t_plusoffset))
            time_diff_img = abs((closest_img[1] - t_plusoffset).total_seconds())
            if time_diff_img <= 30:
                img_file = os.path.join(img_dir, closest_img[0])

        # Create subplot
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, width_ratios=[2, 1, 1])

        # NO2 enhancement plot
        ax0 = fig.add_subplot(gs[0, 0])
        X, Y = np.meshgrid(times_window, np.arange(no2_data.shape[0]))
        pcm = ax0.pcolormesh(X, Y, no2_data, shading='auto')
        fig.colorbar(pcm, ax=ax0, label="a[NO2]")
        ax0.set_title(f"MMSI {mmsi} NO2 around {t.strftime('%Y-%m-%d %H:%M:%S')}")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Viewing direction")

        # Reference region plot
        ax1 = fig.add_subplot(gs[0, 1])
        X, Y = np.meshgrid(measurement_times[window_ref], np.arange(ref_no2.shape[0]))
        pcm = ax1.pcolormesh(X, Y, ref_no2, shading='auto')
        fig.colorbar(pcm, ax=ax1, label="a[NO2]")
        ax1.set_title("Reference region mean NO2")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Viewing direction")

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

        plt.tight_layout()
        plt.savefig(os.path.join(no2_out_dir, f"NO2_subplot_{t.strftime('%Y%m%d_%H%M%S')}_{mmsi}.png"))
        plt.close()
    # %%

    #%%



    # Folder and file pattern
    folder_path = r"P:\data\data_tmp\InSitu"
    #date_str = "20250426"
    pattern = f"av0_20{date}_*.txt"
    file_paths = sorted(glob(os.path.join(folder_path, pattern)))

    # Initialize an empty list to hold DataFrames
    dfs = []

    # Loop through each matching file
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()

        # Normalize decimal separators
        lines = [line.replace(',', '.') for line in lines]

        # Extract columns and units
        columns_line = lines[0].strip().split('\t')
        units_line = lines[1].strip().split('\t')

        # One-letter/short variable names
        one_letter_vars = ["time", "p_0", "c_co2", "dewpoint", "gust_of_wind", "c_h2o", "T_in", "Precip_type", "n_no", 
                        "c_no2", "c_nox", "c_o3", "quality", "rain_intens", "rainfall", "rel_humid", "c_so2", 
                        "T_out", "wind_dir", "wind_speed", "wind_chill"]

        var_map = dict(zip(columns_line, one_letter_vars))

        # Read data lines only (skip header + unit lines)
        data_lines = lines[2:]

        df = pd.read_csv(
            StringIO(''.join(data_lines)),
            sep='\t',
            names=[var_map[name] for name in columns_line],
            parse_dates=[var_map[columns_line[0]]],
            dayfirst=True
        )

        for col in df.columns:
            if col != var_map[columns_line[0]]:  # skip 'time'
                df[col] = pd.to_numeric(df[col], errors='coerce')
        dfs.append(df)

    # Concatenate all dataframes
    full_df = pd.concat(dfs)
    full_df.set_index("time", inplace=True)
    full_df.sort_index(inplace=True)

    # Convert to xarray Dataset
    ds = full_df.to_xarray()

    # Attach metadata to variables
    for orig_name, unit in zip(columns_line[1:], units_line[1:]):
        short_name = var_map[orig_name]
        ds[short_name].attrs['long_name'] = orig_name
        ds[short_name].attrs['units'] = unit

    # Optional: Print info
    print(f"Read and combined {len(file_paths)} files for {date}")
    print(ds)


    # %%
    wind_dir_rad = np.deg2rad(ds['wind_dir'].values)
    wind_speed = ds['wind_speed'].values

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plot wind speed as a function of wind direction
    sc = ax.scatter(wind_dir_rad, wind_speed, c=wind_speed, cmap='viridis', alpha=0.75)

    ax.set_theta_zero_location('N')  # 0° at the top (North)
    ax.set_theta_direction(-1)       # Clockwise

    plt.colorbar(sc, ax=ax, label='Wind Speed (m/s)')
    ax.set_title('Wind Speed and Direction (Polar Plot)')
    ax.set_rlabel_position(225)  # Move radial labels away from overlapping
    plt.savefig(os.path.join(no2_out_dir, f"wind_{date}.png"))

    #plt.show()
    # %%
    for point in closest_points:
        #print(point)
        mmsi = point["MMSI"]
        group = filtered_ship_groups.get(mmsi)
        if group is not None:
            length = group["Length_in_m"].iloc[0]
            if length is None or length <= 20:
                continue
        else:
            continue
        t = pd.to_datetime(point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(point["UTC_Time"]).tzinfo is None else pd.to_datetime(point["UTC_Time"])
        idx = (abs(measurement_times - t)).argmin()
        time_diff = abs((measurement_times[idx] - t).total_seconds())
        if time_diff > 60:
            print(time_diff)
            print(f"No measurement within 30s for MMSI {mmsi} at {t} (closest diff: {time_diff:.2f}s)")
            continue
        window = (abs(measurement_times - t) <= pd.Timedelta(minutes=5))
        ref_found = False
        ref_offset = 3
        while not ref_found and ref_offset < 60:
            ref_start = t - pd.Timedelta(minutes=ref_offset)
            ref_end = t - pd.Timedelta(minutes=ref_offset - 1.0)
            window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
            ref_times = measurement_times[window_ref]
            other_ships_in_window = False
            for other_point in closest_points:
                if other_point["MMSI"] == mmsi:
                    continue
                other_t = pd.to_datetime(other_point["UTC_Time"]).tz_localize("UTC") if pd.to_datetime(other_point["UTC_Time"]).tzinfo is None else pd.to_datetime(other_point["UTC_Time"])
                if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                    other_ships_in_window = True
                    break
            if not other_ships_in_window and window_ref.sum() > 0:
                ref_found = True
            else:
                ref_offset += 1
        if not ref_found:
            print(f"No clean reference window found for MMSI {mmsi} at {t}")
            continue

        # Data for plotting
        no2_data = ds_measurements["a[NO2]"].isel(dim_0=window) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
        times_window = measurement_times[window]
        ref_no2 = ds_measurements["a[NO2]"].isel(dim_0=window_ref) - ds_measurements["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")

        # Try to find the corresponding image
        img_file = None
        img_dir = r"P:\data\data_tmp\video\{}_video".format(date)
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        img_times = [(f, parse_img_time(f)) for f in img_files]
        img_times = [(f, t_img) for f, t_img in img_times if t_img is not None]
        t_naive = t.replace(tzinfo=None)
        utc = pytz.timezone("UTC")
        cet = pytz.timezone("Europe/Berlin")
        t_utc = utc.localize(t_naive)
        t_cet = t_utc.astimezone(cet)

        # Calculate the offset between UTC and CET in hours
        offset_hours = int((t_cet.utcoffset().total_seconds()) // 3600)
        t_plusoffset = t_naive + timedelta(hours=offset_hours)
        if img_times:
            closest_img = min(img_times, key=lambda x: abs(x[1] - t_plusoffset))
            time_diff_img = abs((closest_img[1] - t_plusoffset).total_seconds())
            if time_diff_img <= 30:
                img_file = os.path.join(img_dir, closest_img[0])

        # --- Wind subplot: 1 minute mean around ship pass ---
        # Assume ds is your in-situ xarray Dataset with wind_dir and wind_speed
        # Find 1 minute window around t
        wind_start = t - pd.Timedelta(seconds=30)
        wind_end = t + pd.Timedelta(seconds=30)

        # Make sure wind_start and wind_end are tz-naive to match ds.time
        wind_start = wind_start.replace(tzinfo=None)
        wind_end = wind_end.replace(tzinfo=None)

        wind_sel = ds.sel(time=slice(wind_start, wind_end))
        wind_dir_mean = float(wind_sel['wind_dir'].mean().values) if wind_sel['wind_dir'].size > 0 else np.nan
        wind_speed_mean = float(wind_sel['wind_speed'].mean().values) if wind_sel['wind_speed'].size > 0 else np.nan

        # Create subplot
        fig = plt.figure(figsize=(22, 5))
        gs = GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])
        import matplotlib.dates as mdates  # Add this import at the top if not present

        ax0 = fig.add_subplot(gs[0, 0])
        X, Y = np.meshgrid(times_window, np.arange(no2_data.shape[0]))
        pcm = ax0.pcolormesh(X, Y, no2_data, shading='auto')
        fig.colorbar(pcm, ax=ax0, label=r"NO2 - NO2$_{upwind}$ SC / 1/cm$^2$")
        ax0.set_title(f"MMSI {mmsi} NO2 around {t.strftime('%Y-%m-%d %H:%M:%S')}")
        ax0.set_xlabel("Time")
        
        N = 3  # Show every 3rd LOS value
        los_vals = ds_measurements["los"].isel(dim_0=window[0]).values - 90
        yticks = np.arange(0, len(los_vals), N)
        ax0.set_yticks(yticks)
        ax0.set_yticklabels([f"{float(los_vals[i]):.1f}°" for i in yticks])
        ax0.set_ylabel("LOS / °")
        # Tilt x-axis ticks
        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax0.get_xticklabels(), rotation=45)

        ax1 = fig.add_subplot(gs[0, 1])

        # Vertically integrate (sum) over viewing directions (axis 0)
        no2_integrated = no2_data.sum(axis=0)  # shape: (time,)
        # If you prefer mean, use .mean(axis=0) instead

        ax1.plot(times_window, no2_integrated, color='tab:blue')
        ax1.set_title("Vertically integrated NO$_2$ enhancement")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Integrated NO$_2$ [a.u.]")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
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

        ax3 = fig.add_subplot(gs[0, 3], polar=True)
        ax3.set_theta_zero_location('N')  # 0° at the top (North)
        ax3.set_theta_direction(-1)  
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.lines import Line2D

        # ...existing code...

        if not np.isnan(wind_dir_mean) and not np.isnan(wind_speed_mean):
            wind_dir_rad = np.deg2rad(wind_dir_mean)
            # Plot wind as a point
            #ax3.scatter(wind_dir_rad, wind_speed_mean, color='tab:blue', s=100, label='Wind')
            # Plot wind arrow at the datapoint, pointing in wind direction
            arrow_length = 0.7  # length of the arrow (in m/s units, adjust as needed)
            ax3.arrow(
                wind_dir_rad, wind_speed_mean, 0, -arrow_length,
                width=0.03, head_width=0.15, head_length=0.3, color='tab:blue', alpha=0.8
            )
            # Plot a line in the viewing direction
            viewing_dir_deg = ds_measurements["viewing-azimuth-angle"].isel(viewing_direction=0).values[idx]
            viewing_dir_rad = np.deg2rad(viewing_dir_deg)
            
            # Annotate wind
            ax3.text(
                wind_dir_rad+np.deg2rad(10), wind_speed_mean + 1.0,
                f"{wind_speed_mean:.2f} m/s\n{wind_dir_mean:.0f}°",
                ha='center', va='bottom', fontsize=10, color='black'
            )

        # --- Add ship velocity arrow from AIS data ---
        ship_group = filtered_ship_groups.get(mmsi)
        if ship_group is not None:
            t_utc = t.tz_convert('UTC') if t.tzinfo else t.tz_localize('UTC')
            ais_window = ship_group.loc[
                (ship_group.index >= t_utc - pd.Timedelta(seconds=60)) &
                (ship_group.index <= t_utc + pd.Timedelta(seconds=60))
            ]
            if not ais_window.empty:
                lat = ais_window["Latitude"].astype(float).values
                lon = ais_window["Longitude"].astype(float).values
                times = pd.to_datetime(ais_window.index).values.astype('datetime64[s]').astype(float)
                if len(lat) >= 2 and len(lon) >= 2:
                    from numpy import radians, sin, cos, arctan2, sqrt, degrees
                    R = 6371000.0
                    lat1, lon1 = radians(lat[0]), radians(lon[0])
                    lat2, lon2 = radians(lat[-1]), radians(lon[-1])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
                    distance = R * c
                    dt = times[-1] - times[0]
                    mean_speed = distance / dt if dt > 0 else np.nan
                    x = sin(dlon) * cos(lat2)
                    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                    bearing = arctan2(x, y)
                    mean_course = (degrees(bearing) + 360) % 360
                if not np.isnan(mean_course) and not np.isnan(mean_speed):
                    mean_course_rad = np.deg2rad(mean_course)
                    # Plot ship as a point
                    #ax3.scatter(mean_course_rad, mean_speed, color='green', s=100, label='Ship')
                    # Plot ship velocity arrow at the datapoint, pointing in course direction
                    ship_arrow_length = 0.7  # length of the arrow (in m/s units, adjust as needed)
                    ax3.arrow(
                        mean_course_rad, mean_speed, 0, ship_arrow_length,
                        width=0.03, head_width=0.15, head_length=0.3, color='green', alpha=0.8
                    )
                    # Annotate ship
                    ax3.text(
                        mean_course_rad+np.deg2rad(10), mean_speed + 1,
                        f"{mean_speed:.2f} m/s\n{mean_course:.0f}°",
                        ha='center', va='bottom', fontsize=10, color='black'
                    )
        ax3.plot([viewing_dir_rad, viewing_dir_rad], [0, ax3.get_rmax()], color='red', linewidth=2, label='Viewing direction', linestyle='--')
        # Custom legend handles
        wind_point = Line2D([0], [0], marker='o', color='w', label='Wind',
                            markerfacecolor='tab:blue', markersize=10)
        view_line = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Viewing direction')
        wind_arrow = FancyArrowPatch((0, 0), (0.5, 0.5), color='tab:blue', arrowstyle='->', mutation_scale=15, label='Wind arrow')
        ship_arrow = FancyArrowPatch((0, 0), (0.5, 0.5), color='green', arrowstyle='->', mutation_scale=15, label='Ship velocity')
        ax3.legend(handles=[view_line, wind_arrow, ship_arrow], loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax4 = fig.add_subplot(gs[0, 4])

        # Ensure times_window is tz-naive for comparison with ds.time
        times_window_naive = pd.to_datetime(times_window).tz_localize(None)
        # ds['time'] is the index of the in-situ dataset (already tz-naive)
        ds_times = pd.to_datetime(ds['time'].values)
        # Mask for in-situ data within the NO2 measurement window
        in_situ_mask = (ds_times >= times_window_naive[0]) & (ds_times <= times_window_naive[-1])
        in_situ_times = ds_times[in_situ_mask]
        in_situ_no2 = ds['c_no2'].values[in_situ_mask]

        # Plot, but only if there is data in the window
        if len(in_situ_times) > 0:
            ax4.plot(in_situ_times, in_situ_no2, color='tab:red')
        else:
            ax4.text(0.5, 0.5, "No in-situ data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("In-situ NO$_2$")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("NO$_2$ [ppb]")
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax4.get_xticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(no2_out_dir, f"NO2_subplot_{t.strftime('%Y%m%d_%H%M%S')}_{mmsi}.png"))
        plt.close()

    # %%
    closest_times = [point["UTC_Time"] for point in closest_points]

    # Define start and end time from impact measurements
    start_time = start_time.tz_localize(None)
    end_time = end_time.tz_localize(None)
    closest_times_filtered = [
        t for t in closest_times
        if (pd.to_datetime(t).tz_localize(None) >= start_time) and (pd.to_datetime(t).tz_localize(None) <= end_time)
    ]
    # Filter in situ NO2 data to the measurement window
    ds_times = pd.to_datetime(ds['time'].values)
    in_window_mask = (ds_times >= start_time) & (ds_times <= end_time)
    ds_times_window = ds_times[in_window_mask]
    c_no2_window = ds['c_no2'].values[in_window_mask]

    plt.figure(figsize=(12, 4))

    # Plot vertical lines for closest approach times within window
    for t in closest_times_filtered:
        plt.axvline(pd.to_datetime(t), color='red', alpha=0.5)

    # Plot satellite NO2 enhancement
    ax1 = plt.gca()
    line1, = ax1.plot(
        ds_measurements["datetime"].isel(viewing_direction=0),
        ds_measurements["NO2_enhancement"].mean(dim="viewing_direction"),
        label="NO2 Enhancement",
        color='blue'
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Satellite NO2 Enhancement")
    ax1.set_title("NO2 time series and ship passes according to AIS")

    # Plot in situ NO2 on a secondary y-axis (only within window)
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        ds_times_window,
        c_no2_window,
        label="In Situ NO2",
        color='green'
    )
    ax2.set_ylabel("In Situ NO2")

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    #plt.show()

    # %%
    file_path = r"Q:\BREDOM\SEICOR\Open_Path\SpectralAnalysis\results\20{}_365_Eval_SC.dat".format(date)

    df = pd.read_csv(
        file_path,
        sep='\t',
        engine='python',
        #on_bad_lines='skip'  # skips lines with wrong number of columns
    )



    # %%
    mask = df['RMS'] < 0.0005

    # %%
    mask2 = ds_measurements["rms"] < 0.01
    ds_measurements = ds_measurements.where(mask2, drop=True)

    coarsened_ds = ds_measurements.coarsen(dim_0=20, boundary="trim").mean()
    # Restrict to min/max times of ds_measurements["datetime"]
    dtimes = coarsened_ds["datetime"].isel(viewing_direction=0).values
    dtimes_pd = pd.to_datetime(dtimes)
    tmin = dtimes_pd[~pd.isnull(dtimes_pd)].min()
    tmax = dtimes_pd[~pd.isnull(dtimes_pd)].max()

    # Mask for df times within the range
    df_times = pd.to_datetime(df.loc[mask, 'StartDateAndTime'])
    df_mask = (df_times >= tmin) & (df_times <= tmax)

    # %%
    plt.figure(figsize=(15, 5))
    # Ensure tmin and tmax are tz-naive
    tmin_naive = pd.to_datetime(tmin).tz_localize(None)
    tmax_naive = pd.to_datetime(tmax).tz_localize(None)
    # First y-axis: satellite NO2
    ax1 = plt.gca()    
    # Add vertical lines for ship passes
    for point in closest_points:
        t_ship = pd.to_datetime(point["UTC_Time"]).tz_localize(None)
        if tmin_naive <= t_ship <= tmax_naive:
            ax1.axvline(t_ship, color='red', alpha=0.5, linestyle='--', label="Ship pass" if 'Ship pass' not in ax1.get_legend_handles_labels()[1] else "")

    ax1.plot(
        df_times[df_mask],
        df.loc[mask, 'Fit Coefficient (NO2)'][df_mask],
        marker=".", linestyle='', label="LP-DOAS NO2 SC"
    )
    ax1.plot(
        coarsened_ds["datetime"].isel(viewing_direction=0),
        coarsened_ds["a[NO2]"].isel(viewing_direction=slice(4,8)).mean(dim="viewing_direction"),
        label="IMPACT (viewing direction 5-7 average) SC", marker=".", linestyle=''
    )



    ax1.set_xlabel("Time")
    ax1.set_ylabel("NO2 1/ cm$^2$")
    ax1.set_title(f"NO2 {date}")

    # Second y-axis: in situ NO2
    ax2 = ax1.twinx()
    # Find the time axis for in situ data (assumes ds['c_no2'] exists and ds['time'] is the time coordinate)
    in_situ_times = pd.to_datetime(ds['time'].values)
    ds_mask = (in_situ_times >= tmin) & (in_situ_times <= tmax)
    in_situ_no2 = ds['c_no2'].values
    ax2.plot(in_situ_times[ds_mask], in_situ_no2[ds_mask], color='green', alpha=0.7, label="In Situ NO2")
    ax2.set_ylabel("In Situ NO2 [ppb]")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(no2_out_dir, f"NO2_{date}_all_instruments.png"))

# %%
