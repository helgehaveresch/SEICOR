# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:13:54 2023

@author: kakrau
"""

import pandas as pd
import numpy as np
import re
from gauss_plume.utilities import distance_in_m, trajectories, calc_wind_components


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


def combine_peaks_ais(peaks, ais, forward, backward, lat_lightpath, lon_lightpath,
                      max_distance=500, min_speed=0.5):
    """


    Parameters
    ----------
    peaks : dataframe
        peaks: maximum as index (datetime).
        left_bases: left border of peak (datetime).
        right_bases: right border of peak (datetime).
    ais : dataframe
        Dataframe with AIS signals.
    forward : float or int
        Time in seconds after each peak occurence that can be evaluated for
        finding source ships.
    backward : float or int
        Time in seconds before each peak occurence that can be evaluated for
        finding source ships.
    lat_lightpath : array of floats.
        Latitude points of the lightpath used for calculating the distance
        between lightpath and ship position. Should be high resolution, e.g. 1 m
        distance between points.
    lon_lightpath : array of floats.
        Longitude points of the lightpath used for calculating the distance
        between lightpath and ship position. Should be high resolution, e.g. 1 m
        distance between points.
    max_distance : float, optional
        Maximum distance in m to be considered between AIS location and lightpath.
        The default is 500. If the distance between AIS location and lightpath is lower,
        the location can be the respective source of the peak.
    min_speed : float, optional
        Minimum speed for a ship to be assigned. The default is 0.5 m/s.

    Returns
    -------
    output : tuple
        List of the respective AIS source positions for each peak. For each peak,
        there should only be a single ship with all necessary AIS messages to track
        the ship before and after it passes through the lightpath.

    """

    # empty dataframe for with same columns as ais
    no_ship_found = pd.DataFrame(columns=ais.columns)
    asigned_peaks = [None]*len(peaks)
    # filter for moving ships
    ais = ais.loc[ais.loc[:, "Speed_in_m/s"] > min_speed]

    for i in range(peaks.shape[0]):
        peak = peaks.iloc[i]
        # calculate time windows for assignment
        peak_time_backward = peak.name-pd.Timedelta(backward, "s")
        peak_time_forward = peak.name+pd.Timedelta(forward, "s")
        # subset ais to assignment window
        ais_subset = ais.iloc[ais.index > peak_time_backward, :]
        ais_subset = ais_subset.iloc[ais_subset.index < peak_time_forward, :]

        # empty list for distance between ais and lightpath
        tmp_distances = [0]*len(ais_subset)
        if len(ais_subset) > 0:
            # calculate distance for every ais postion
            for j in range(len(ais_subset)):
                tmp_distances[j] = np.nanmin(distance_in_m(lon_lightpath,
                                                           lat_lightpath,
                                                           ais_subset.iloc[j, ais_subset.columns.get_loc("Longitude")],
                                                           ais_subset.iloc[j, ais_subset.columns.get_loc("Latitude")]))
            # check for ships within max_distance
            if np.nanmin(tmp_distances) < max_distance:
                if len(pd.unique(ais_subset.loc[np.array(tmp_distances) < max_distance, "MMSI"].values)) == 1:
                    # set ship position, at closest position in time to peak maximum
                    ais_subset = ais_subset.loc[np.array(tmp_distances) < max_distance]
                    asigned_peaks[i] = ais_subset.iloc[ais_subset.index.get_indexer([peak.name], method="nearest")]
                else:
                    asigned_peaks[i] = no_ship_found
            else:
                asigned_peaks[i] = no_ship_found
        else:
            asigned_peaks[i] = no_ship_found

    # for all found ships get all ais positions between peak-backward to peak+forward
    output = [None]*len(asigned_peaks)
    for i in range(len(asigned_peaks)):
        if not asigned_peaks[i].empty:
            ais_subset = ais.loc[ais.loc[:, "MMSI"] == asigned_peaks[i].iloc[0, asigned_peaks[i].columns.get_loc("MMSI")]]
            lower_limit = pd.to_datetime(peaks.iloc[i, peaks.columns.get_loc("left_bases")]-pd.Timedelta(backward, "s"), utc=True)
            upper_limit = pd.to_datetime(peaks.iloc[i, peaks.columns.get_loc("right_bases")]+pd.Timedelta(forward, "s"), utc=True)
            ais_subset = ais_subset.iloc[ais_subset.index > lower_limit, :]
            ais_subset = ais_subset.iloc[ais_subset.index < upper_limit, :]

            # just for sanity checks add distance to lightpath
            ais_subset["Distance_in_m"] = 0.
            for j in range(len(ais_subset)):
                ais_subset.iloc[j, ais_subset.columns.get_loc("Distance_in_m")] = np.nanmin(distance_in_m(lon_lightpath,
                                                                                                           lat_lightpath,
                                                                                                           ais_subset.iloc[j, ais_subset.columns.get_loc("Longitude")],
                                                                                                           ais_subset.iloc[j, ais_subset.columns.get_loc("Latitude")]))
            output[i] = ais_subset
    # clean the output
    index = [True if i is not None else False for i in output]
    peaks = peaks.loc[index]
    output = [i for i in output if i is not None]
    output = [i for i in output if not i.empty]

    return peaks, output


def combine_peaks_trajectories(peaks, ais, insitu, lon, lat, ws_key, wd_key,
                               lon_key, lat_key, speed_key, max_traveltime,
                               max_distance, forward, backward, extend_values=True, single_ship=True):
    """

    Combine ships and peaks. For each peak a plume trajectory is calculated and
    it is checked whether the plume ends close to the measurement site. Afterwards,
    it is checked whether there is only a single possible source ship, if that
    is the case, the ship is assigned as the source of the peak. For each assigned peak,
    all ship positions of the respective source ship 180 seconds before and 180 seconds
    after the peak maximum are returned.

    Returns all peaks with a respective source ship and a list of positions for that peak.

    Parameters
    ----------
    peaks : pandas Dataframe
        dataframe containing information about the peaks. Each row is a single peak.
        columns: [index, left_bases, right_bases]
        index: datetime64, time of the peak maximum in UTC
        left_bases: datetime64, left base of the peak (start of the peak) in UTC
        right_bases: datetime64, right base of the peak (end of the peak) in UTC
    ais : pandas Dataframe
        dataframe containing the AIS information. Each row represents a single AIS signal.
        columns: [MMSI, Speed_in_m/s, Latitude, Longitude, Name, Type,
                  Length_in_m, Width_in_m, Draught_in_m, Destination]
        index: datetime64 in UTC
        MMSI: MMSI idenfication number of the ship
        Speed_in_m/s: Speed over ground in meter per second. OPTIONAL
        Latitude: Latitude in degree.
        Longitude: Longitude in degree.
        Name: Name of the ship. OPTIONAL
        Type: Type of the ship. OPTIONAL
        Length_in_m: Length of the ship in m. OPTIONAL
        Width_in_m: Width of the ship in m. OPTIONAL
        Draught_in_m: Draft of the ship in m. OPTIONAL
        Destination: Transmitted destination of the ship. OPTIONAL
    insitu : pandas Dataframe
        dataframe containing the wind information.
        columns: [wind speed, wind direction]
        index: datetime64 in UTC
        wind speed should be in m per second.
        wind direction should be in degree.
    lat : float
        Latitude in degreeof the measurement site.
    lon : float
        Longitude in degree  of the measurment site.
    ws_key : str
        column name for wind speed in "insitu".
    wd_key : str
        column name for wind direction in "insitu".
    lon_key : str
        column name for Longitude in "ais".
    lat_key : str
        column name for Latitude in "ais".
    speed_key: str
        column name for the ship speed in "ais".
    max_traveltime : int
        maximum plume travel time in seconds.
    max_distance : float
        maximum distance between plume end position and measurement site.
    extend_values: bool
        Should all AIS positions be collected, or only the matching one?
        Default is True.
    Returns
    -------
    peaks : pandas Dataframe
        returns the peaks dataframe, but only for peaks where a source ship has
        been identified.
    output : list
        returns a list of AIS positions for each peak. Has the same length as "peaks"
        and the i-th position in the list corresponds to the i-th peak (e.g., the
        item at position 0 corresponds to the peak in row 0).

    """

    # calculate the wind components
    insitu["u"], insitu["v"] = calc_wind_components(insitu[ws_key], insitu[wd_key])
    # empty dataframe to add if there was no source ship found
    no_ship_found = pd.DataFrame(columns=ais.columns)
    # filter ais data for moving ships
    ais = ais.loc[ais.loc[:, speed_key] > 0.1]
    # empty list of assigned peaks
    asigned_peaks = [None]*len(peaks)

    # first we iterate over all peaks to find a ship
    for i in range(peaks.shape[0]):
        peak = peaks.iloc[i]
        # subset all ais data to the time some seconds before the peak
        ais_subset = ais.iloc[ais.index >= peak.name-pd.Timedelta(max_traveltime, "s"), :]
        ais_subset = ais_subset.iloc[ais_subset.index <= peak.name, :]
        # find the closest wind data for the peak
        insitu_subset = insitu.iloc[insitu.index.get_indexer([peak.name], method="nearest")]
        # calculate the time between peak maximum and each ais position
        ais_subset.loc[:, "timedelta"] = np.array(np.array(peak.name-ais_subset.index, dtype="timedelta64[s]"),
                                                  dtype=float)
        # calculate a trajectory starting at each ais position
        ais_subset.loc[:, "end_lon"], ais_subset.loc[:, "end_lat"] = trajectories(ais_subset.loc[:, lon_key].values,
                                                                                  ais_subset.loc[:, lat_key].values,
                                                                                  insitu_subset.loc[insitu_subset.index, "u"].values,
                                                                                  insitu_subset.loc[insitu_subset.index, "v"].values,
                                                                                  ais_subset.loc[:, "timedelta"].values)
        # calculate the distance between the measurement station and each trajectory end point
        ais_subset.loc[:, "distance_in_m"] = distance_in_m(lon,
                                                           lat,
                                                           ais_subset.loc[:, "end_lon"],
                                                           ais_subset.loc[:, "end_lat"])

        # check if a trajectory ended close to the measurement station
        if any(ais_subset.loc[:, "distance_in_m"]):
            if np.nanmin(ais_subset.loc[:, "distance_in_m"]) < max_distance:
                # subset the ais again to only include positions with trajectories ending at the measurement station
                ais_subset = ais_subset.loc[ais_subset.loc[:, "distance_in_m"] < max_distance]
                # ais_subset = ais_subset.iloc[np.logical_and(ais_subset.index >= peak.left_bases,
                #                                             ais_subset.index <= peak.right_bases)]
                # check if there was only single ship with trajectories close to the measurement station
                if single_ship:
                    if len(pd.unique(ais_subset.loc[:, "MMSI"].values)) == 1:
                        # of all possible ais positions use the one closest in time to the peak maximum
                        asigned_peaks[i] = ais_subset.iloc[ais_subset.index.get_indexer([peak.name], method="nearest")]
                    else:
                        asigned_peaks[i] = no_ship_found
                else:
                    asigned_peaks[i] = ais_subset
            else:
                asigned_peaks[i] = no_ship_found
        else:
            asigned_peaks[i] = no_ship_found

    # here we iterate over all peaks with a source ship and assign all ais positions 180 seconds before
    # and after the peak
    if extend_values:
        output = [None]*len(asigned_peaks)
        for i in range(len(asigned_peaks)):
            # if there was a ship we get all AIS positions +- 180 seconds before and after peak
            if not asigned_peaks[i].empty:
                ais_subset = ais.loc[ais.loc[:, "MMSI"] == asigned_peaks[i].iloc[0, asigned_peaks[i].columns.get_loc("MMSI")]]
                lower_limit = pd.to_datetime(peaks.iloc[i, peaks.columns.get_loc("left_bases")]-pd.Timedelta(max_traveltime+backward, "s"),
                                             utc=True)
                upper_limit = pd.to_datetime(peaks.iloc[i, peaks.columns.get_loc("right_bases")]+pd.Timedelta(forward, "s"),
                                             utc=True)
                ais_subset = ais_subset.iloc[ais_subset.index > lower_limit, :]
                ais_subset = ais_subset.iloc[ais_subset.index < upper_limit, :]

                # just to check we also add the information about the distance to the measurement
                # station for all ais positions
                ais_subset["Distance_to_station_in_m"] = distance_in_m(lon,
                                                                       lat,
                                                                       ais_subset.iloc[:, ais_subset.columns.get_loc("Longitude")],
                                                                       ais_subset.iloc[:, ais_subset.columns.get_loc("Latitude")])
                output[i] = ais_subset
    else:
        output = asigned_peaks

    # Here we subset the peaks we put into the function
    # if there was a ship assigned the peak is kept, otherwise it will be removed
    index = [True if i is not None else False for i in output]
    peaks = peaks.loc[index]
    output = [i for i in output if i is not None]

    return peaks, output


def prepare_ais(file, interpolation_limit):
    data = read_ais(file)
    data = interpolate_ais(data, interpolation_limit)
    return data


def check_for_uncertain_sources(peaks, ais, forward, backward, lat_lightpath, lon_lightpath,
                                max_distance=500):
    """
    Function to check peaks for ship passages, outputs if no ship, one ship or
    more than one ship could be assigned to a peak.

    Parameters
    ----------
    peaks : dataframe
        peaks: maximum as index (datetime).
        left_bases: left border of peak (datetime).
        right_bases: right border of peak (datetime).
    ais : dataframe
        Dataframe with AIS signals.
    forward : float or int
        Time in seconds after each peak occurence that can be evaluated for
        finding source ships.
    backward : float or int
        Time in seconds before each peak occurence that can be evaluated for
        finding source ships.
    lat_lightpath : array of floats.
        Latitude points of the lightpath used for calculating the distance
        between lightpath and ship position. Should be high resolution, e.g. 1 m
        distance between points.
    lon_lightpath : array of floats.
        Longitude points of the lightpath used for calculating the distance
        between lightpath and ship position. Should be high resolution, e.g. 1 m
        distance between points.
    max_distance : float, optional
        Maximum distance in m to be considered between AIS location and lightpath.
        The default is 500. If the distance between AIS location and lightpath is lower,
        the location can be the respective source of the peak.

    Returns
    -------
    output :
        List of the of the number of possible sources.
        0 = no ship
        1 = one ship
        2 = more than one ship

    """


    no_ship_found = 0
    one_ship_found = 1
    more_ships_found = 2
    asigned_peaks = [None]*len(peaks)
    ais = ais.loc[ais.loc[:, "Speed_in_m/s"] > 0.1]

    for i in range(peaks.shape[0]):
        peak = peaks.iloc[i]
        peak_time_backward = peak.name-pd.Timedelta(backward, "s")
        peak_time_forward = peak.name+pd.Timedelta(forward, "s")
        ais_subset = ais.iloc[ais.index > peak_time_backward, :]
        ais_subset = ais_subset.iloc[ais_subset.index < peak_time_forward, :]

        tmp_distances = [0]*len(ais_subset)

        for j in range(len(ais_subset)):
            tmp_distances[j] = np.nanmin(distance_in_m(lon_lightpath,
                                                       lat_lightpath,
                                                       ais_subset.iloc[j, ais_subset.columns.get_loc("Longitude")],
                                                       ais_subset.iloc[j, ais_subset.columns.get_loc("Latitude")]))
            # evtl ändern auf Schiffslänge*3 oder ähnliches
        if np.nanmin(tmp_distances) < max_distance:
            if len(pd.unique(ais_subset.loc[np.array(tmp_distances) < max_distance, "MMSI"].values)) == 1:
                asigned_peaks[i] = one_ship_found
            else:
                asigned_peaks[i] = more_ships_found
        else:
            asigned_peaks[i] = no_ship_found

    return asigned_peaks


def summarize_sources(ais_sources, peaks):
    """


    Parameters
    ----------
    ais_sources : list of ais sources
    peaks : dataframe of peaks

    Returns
    -------
    output : summarized data frame of ais sources.
        numerical values are mean of values found in ais_sources.
        non-numerical values are the values of the first row in ais_sources.

    """

    n = len(ais_sources)
    output = [None]*n

    for i in range(n):
        tmp = pd.DataFrame(columns=ais_sources[i].columns)
        col_types = ais_sources[i].dtypes

        for j in tmp.columns:
            if col_types[j] == float:
                tmp.loc[0, j] = ais_sources[i].loc[:, j].mean()
            else:
                tmp.loc[0, j] = ais_sources[i].bfill(axis=0).iloc[0, tmp.columns.get_loc(j)]

        output[i] = tmp

    output = pd.concat(output)
    output.set_index(peaks.index, inplace=True)
    output = pd.merge(output, peaks, left_index=True, right_index=True)

    return output
