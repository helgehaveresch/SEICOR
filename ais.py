from datetime import date
import numpy as np
import pandas as pd
import re
import os
from geopy.distance import geodesic
#import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)
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

def interpolate_ais(ais, interpolation_limit=60, resample="1s"):
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
        # Resample to desired frequency
        tmp = ais_dat.resample(resample).asfreq()
        # Interpolate only float columns
        float_cols = tmp.select_dtypes(include=["float", "float32", "float64"]).columns
        tmp[float_cols] = tmp[float_cols].interpolate(method="linear", limit=interpolation_limit)
        # Forward-fill and backward-fill int and string columns, but only up to the gap limit
        non_float_cols = tmp.columns.difference(float_cols)
        for col in non_float_cols:
            # Only fill gaps up to the limit, leave longer gaps as NaN
            tmp[col] = tmp[col].ffill(limit=interpolation_limit).bfill(limit=interpolation_limit).infer_objects(copy=False)
        # Backward-fill MMSI to ensure no NaNs at start, but only up to the gap limit
        if "MMSI" in tmp.columns:
            tmp["MMSI"] = tmp["MMSI"].bfill(limit=interpolation_limit)
        tmp = tmp.dropna(subset=["MMSI"])
        tmp = tmp.astype({"MMSI": int})
        ais_list[i] = tmp

    output = pd.concat(ais_list)

    return output

def interpolate_ais_group(ais_group, interpolation_limit=60, resample="1s"):
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

    ais_group = ais_group.loc[~ais_group.index.duplicated(keep="first")]
    # Resample to desired frequency
    tmp = ais_group.resample(resample).asfreq()
    # Interpolate only float columns
    float_cols = tmp.select_dtypes(include=["float", "float32", "float64"]).columns
    tmp[float_cols] = tmp[float_cols].interpolate(method="linear", limit=interpolation_limit)
    # Forward-fill and backward-fill int and string columns, but only up to the gap limit
    non_float_cols = tmp.columns.difference(float_cols)
    for col in non_float_cols:
        # Only fill gaps up to the limit, leave longer gaps as NaN
        tmp[col] = tmp[col].ffill(limit=interpolation_limit).bfill(limit=interpolation_limit).infer_objects(copy=False)
    # Backward-fill MMSI to ensure no NaNs at start, but only up to the gap limit
    if "MMSI" in tmp.columns:
        tmp["MMSI"] = tmp["MMSI"].bfill(limit=interpolation_limit)
        tmp = tmp.dropna(subset=["MMSI"])
        tmp = tmp.astype({"MMSI": int})
    ais_group = tmp


    return ais_group


def restrict_lat_lon_ais(df_ais, lat_lon_window):
    """
    Restrict the latitude and longitude of the AIS data to a specified window.
    """
    lat_mask = (df_ais["Latitude"] < lat_lon_window[0]) | (df_ais["Latitude"] > lat_lon_window[1])
    lon_mask = (df_ais["Longitude"] < lat_lon_window[2]) | (df_ais["Longitude"] > lat_lon_window[3])
    df_ais.loc[lon_mask, "Longitude"] = np.nan
    df_ais.loc[lat_mask, "Latitude"] = np.nan
    return df_ais

def pre_filter_ais(df_ais, lat_lon_window, start_time, end_time, length=20):
    """
    Filter the AIS data based on various criteria such as location, time, and ship characteristics.
    """

    df_ais = restrict_lat_lon_ais(df_ais, lat_lon_window)

    sailing_mask = df_ais["Type"].str.lower() == "vessel:sail"
    anchored_mask = df_ais["Navigation_status"].str.lower() == "anchored"
    length_mask = (df_ais["Length_in_m"] < length) #filter out ships that are too small or too large

    df_ais["filtermask"] = 0
    df_ais.loc[anchored_mask, "filtermask"] = 2
    df_ais.loc[sailing_mask, "filtermask"] = 3
    df_ais.loc[length_mask, "filtermask"] = 4 # todo: all filtering just by masks?

    # 1. Filter out all ships that have Type 'Vessel:sail' or are anchored
    sailing_mmsi = set(df_ais[df_ais["Type"].str.lower() == "vessel:sail"]["MMSI"].unique())
    anchored_mmsi = set(df_ais[df_ais["Navigation_status"].str.lower() == "anchored"]["MMSI"].unique())
    small_mmsi = set(df_ais[df_ais["Length_in_m"] < length]["MMSI"].unique())
    sailing_mmsi = sailing_mmsi  | small_mmsi | anchored_mmsi

    # Start from grouping ships by MMSI
    ship_groups = df_ais.groupby("MMSI")
    # Only keep ship data within the measurement time interval
    filtered_ship_groups = []
    for mmsi, group in ship_groups:
        if mmsi not in sailing_mmsi:
            mask = (group.index >= start_time) & (group.index <= end_time) #filter out ships outside the IMPACT operational time window
            #filtered_group = group.loc[mask]
            if not mask.any():
                df_ais.loc[group.index, "filtermask"] = 1  # time window not matching
            else:
                pos = np.sqrt(group["Longitude"]**2 + group["Latitude"]**2)
                if pos.diff().any() > 0.001:
                    filtered_ship_groups.append((mmsi, group))
                else:
                    df_ais.loc[group.index, "filtermask"] = 5  # no movement

    return df_ais, ship_groups, filtered_ship_groups

def filter_ship_passes(df_ais, ship_groups, filtered_ship_groups, start_time, end_time, measurement_times, endpoints_los, instrument_location, distance_threshold=0.002):
    """
    Filter ship passes based on various criteria such as time, location, and distance and create
    a DataFrame with the results.
    """

    ship_passes = []
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

        ship_pass_mask = dists <= distance_threshold
        
        if np.any(ship_pass_mask):
            
            mask = ship_pass_mask.astype(int)
            diff = np.diff(mask, prepend=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if mask[-1] == 1:
                ends = np.append(ends, len(mask))
            for start, end in zip(starts, ends):
                pass_idxs = np.arange(start, end)
                if len(pass_idxs) == 0:
                    continue
                pass_df = group.iloc[pass_idxs][["Latitude", "Longitude"]].copy()
                pass_df.index = group.index[pass_idxs]
                pass_df = interpolate_ais_group(pass_df, interpolation_limit=60, resample="1s")
                pass_idxs = np.arange(len(pass_df))
                pass_lats = pass_df["Latitude"]
                pass_lons = pass_df["Longitude"]
                pass_times = pass_df.index
                pass_ship_positions = np.stack([pass_lats, pass_lons], axis=1)
                pass_impact_idxs = np.abs(measurement_times.values[:, None] - pass_times.values).argmin(axis=0)

                # Vectorized distance calculation
                pass_ab = endpoints_los[pass_impact_idxs] - np.array(instrument_location)      #np.stack([lat2_s - instrument_location[0], lon2_s - instrument_location[1]], axis=1)
                pass_ab_norm = pass_ab / np.linalg.norm(pass_ab, axis=1)[:, None]
                pass_ap = pass_ship_positions - np.array(instrument_location)
                pass_proj = np.sum(pass_ap * pass_ab_norm, axis=1)[:, None] * pass_ab_norm
                pass_dists = np.linalg.norm(pass_ap - pass_proj, axis=1)

                min_idx = pass_idxs[np.argmin(pass_dists)]
                t_ship = pd.to_datetime(pass_times[min_idx])
                if (t_ship >= start_time) and (t_ship <= end_time):
                    measurement_diffs = np.abs(measurement_times - t_ship)
                    closest_meas_idx = measurement_diffs.argmin()
                    closest_time_diff = measurement_diffs[closest_meas_idx]
                    closest_meas_time = measurement_times[closest_meas_idx]
                    mean_speed, mean_course = calculate_mean_speed_course(group, t_ship)
                    ship_category = sort_ship_sizes(group)
                    try:
                        if min_idx in group.index:
                            draught = group["Draught_in_m"].loc[min_idx]
                            length_m = group["Length_in_m"].loc[min_idx]
                        else:
                            draught = group["Draught_in_m"].iloc[int(min_idx)]
                            length_m = group["Length_in_m"].iloc[int(min_idx)]
                    except (IndexError, KeyError, ValueError):
                        draught = np.nan
                        length_m = np.nan
                        print(f"Warning: min_idx={min_idx} out-of-range for MMSI {mmsi}; using NaN")

                    ship_passes.append({
                        "MMSI": mmsi,
                        "UTC_Time": pass_times[min_idx],
                        "Closest_Impact_Measurement_Index": closest_meas_idx,
                        "Closest_Impact_Measurement_Time": closest_meas_time,
                        "Closest_Impact_Measurement_Time_Diff": closest_time_diff,
                        "Mean_Speed": mean_speed,
                        "Mean_Course": mean_course,
                        "Latitude": pass_lats.iloc[min_idx],
                        "Longitude": pass_lons.iloc[min_idx],
                        "Distance": pass_dists[min_idx],
                        "Draught_in_m": draught,
                        "Length_in_m": length_m,
                        "ship_category": ship_category
                    })
        else:
            df_ais.loc[group.index, "filtermask"] = 6  # no close positions
    ship_passes = sorted(ship_passes, key=lambda x: pd.to_datetime(x["UTC_Time"]))
    df_ship_passes = pd.DataFrame(ship_passes)
    df_ship_passes["Plume_number"] = df_ship_passes.index
    df_ship_passes.set_index("UTC_Time", inplace=True)


    close_ship_mmsi = df_ship_passes["MMSI"].unique().tolist()
    filtered_ship_groups = {mmsi: group for mmsi, group in filtered_ship_groups if mmsi in close_ship_mmsi}

    maskedout_mmsi = set(df_ais["MMSI"].unique()) - set(close_ship_mmsi)
    maskedout_groups = {mmsi: group for mmsi, group in ship_groups if mmsi in maskedout_mmsi}
    filtered_ship_groups = dict(filtered_ship_groups)

    return df_ais, filtered_ship_groups, maskedout_groups, df_ship_passes

def sort_ship_sizes(ship_group):
    """
    Sort ships into categories based on their size.
    """
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

def prepare_ais(dir, date, interpolation_limit=60):
    file = os.path.join(dir, "shipplotter{}.log".format(date))
    data = read_ais(file)
    #data = interpolate_ais(data, interpolation_limit) #interpolation is now done at a later step to reduce comp. effort
    data.index = pd.to_datetime(data.index)
    return data

def calculate_mean_speed_course(ship_group, t):
    """
    Calculate the mean speed and course of a ship based on its AIS data around a specific time t.

    Args:
        ship_group (pd.DataFrame): AIS data for a specific ship.
        t (pd.Timestamp): Time of interest (UTC).

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

    # Calculate distance using geopy
    coord1 = (lat1, lon1)
    coord2 = (lat2, lon2)
    distance = geodesic(coord1, coord2).meters

    dt = t2 - t1  # seconds
    mean_speed = distance / dt if dt > 0 else np.nan

    # Calculate bearing (direction) using geopy
    mean_course = initial_bearing(coord1, coord2)

    if np.isnan(mean_course) or np.isnan(mean_speed):
        return None, None

    return mean_speed, mean_course

def initial_bearing(coord1, coord2):
    """
    Calculate the initial bearing between two lat/lon points.
    Returns bearing in degrees (0° = North, clockwise).
    """
    lat1 = np.radians(coord1[0])
    lon1 = np.radians(coord1[1])
    lat2 = np.radians(coord2[0])
    lon2 = np.radians(coord2[1])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (
        np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360