# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:04:23 2023

@author: kakrau
"""

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

def read_hdf(file, key=None, **kwargs):
    """


    Parameters
    ----------
    file : string
        hdf path + filename.
    key : string, optional
        Key to desired dataset in file. The default is None.
    **kwargs : TYPE
        Extra arguments used in pandas.read_hdf().

    Returns
    -------
    output : ---
        Dataset stored under "key" in "file".

    """
    output = pd.read_hdf(file, key, mode='r', **kwargs)
    return output


def colorbar(mappable, **kwargs):
    """
    Function to create colorbar with same size as plot. Mappable is something
    like ax1.plot(...).
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)


def colorbar_lineplot(mappable, cmap, values, **kwargs):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([min(values), max(values)])
    return fig.colorbar(mappable, cax=cax, **kwargs)


def log_wind_profile(U_r, z, z_r=10, z0=0.01):
    """


    Parameters
    ----------
    U_r : float
        Wind speed in m/s at reference height.
    z : float
        Desired height.
    z_r : float, optional
        Reference height. The default is 10.
    z0 : float, optional
        Roughness height. The default is 0.01 (mowed meadow).

    Returns
    -------
    U_z : float
        Wind speed in m/s at desired height.

    """

    U_z = U_r * np.log(z/z0)/np.log(z_r/z0)

    return U_z


def calc_wind_speed_dir(u, v):
    """


    Parameters
    ----------
    u : float or np.array of float
        u compnent of wind.
    v : float or np.array of float
        v component of wind.

    Returns
    -------
    U : float or np.array of float
        Wind speed in m/s.
    U_dir : float or np.array of float
        Wind direction in °.

    """

    U = np.sqrt(u**2.+v**2.)
    U_dir = (np.rad2deg(np.arctan2(u, v))+180) % 360

    return U, U_dir


def calc_wind_components(U, U_dir):
    """

    Wind speed and wind direction are converted into components of the horizonal
    wind vector. These components are used in the "trajectories"-function.

    Parameters
    ----------
    U : float or np.array of float
        Wind speed in m/s.
    U_dir : float or np.array of float
        Wind direction in degree.

    Returns
    -------
    u : float or np.array of float
        u component of wind in m/s (WEST-EAST).
        positive towards EAST.
    v : float or np.array of float
        v component of wind in m/s (SOUTH-NORTH).
        positive towards NORTH.

    """

    u = -U * np.sin(np.deg2rad(U_dir))
    v = -U * np.cos(np.deg2rad(U_dir))

    return u, v


def add_distance_in_m(lon, lat, dx, dy):
    """

    This function is used to add a distance in x and / or y direction to a starting
    position at Longitude "lon" and Latitude "lat".
    Is only valid if we stay away from the poles and only add or subtract small
    distances (some kilometers are okay).

    Parameters
    ----------
    lon : float or array of floats
        longitude in degree.
    lat : float or array of floats
        latitutde in degree.
    dx : float
        added distance in x direction in m.
    dy : float
        added distance in y direction in m.

    Returns
    -------
    lon_new : float or array of floats
        lon + dx.
    lat_new : float or array of floats
        lat + dy.

    """
    # rough estimate for everything, should be okay if we stay away from poles
    # and only add/subtract small distances
    EARTH_RADIUS = 6371000

    lat_new = lat + np.rad2deg(dy/EARTH_RADIUS)
    lon_new = lon + np.rad2deg(dx/EARTH_RADIUS/np.cos(np.deg2rad(lat)))

    return lon_new, lat_new


def bearing_func(lon1, lat1, lon2, lat2):
    """
    Calculates azimuth angle from one point to next one

    Parameters
    ----------
    lon1 : float
    lat1 : float
    lon2 : float
    lat2 : float

    Returns
    -------
    float
        Azimuth angle from Point1 (lon1, lat1) to Point2 (lon2, lat2)

    """
    #θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    
    y = np.sin(lon2-lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
    
    phi = np.arctan2(y, x)
    
    return (np.rad2deg(phi)+360) % 360


def calc_bearing(lon, lat):
    """
    Calculates azimuth angle of a list of points.
    Calculation starts at first point and ends at last.


    Parameters
    ----------
    lon : float
        Array of Longitudes
    lat : float
        Array of Latitudes

    Returns
    -------
    phi : float
        Angle from Point 1 (Lon0, Lat0) to Point2 (Lon1, Lat1)
        [bearing_func(lon[i], lat[i], lon[i+1], lat[i+1]) for i in range(n)]
        
        len(phi) == n-1

    """
    n = len(lat)-1
    phi = np.array([bearing_func(lon[i], lat[i], lon[i+1], lat[i+1]) for i in range(n)])
    return phi


def calc_destination_point(lon, lat, distance, bearing):
    """
    

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
    distance : float
        Distance between the start and endpoint.
    bearing : float
        Angle from start point towards endpoint.

    Returns
    -------
    dest_lon : float
        Longitude of the calculated endpoint.
    dest_lat : float
        Latitude of the calculated endpoint.

    """
    EARTH_RADIUS = 6371000  # Radius of the Earth in kilometers

    distance = distance / EARTH_RADIUS

    # Convert bearing to radians
    bearing = np.deg2rad(bearing)

    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate destination point latitude
    dest_lat = np.arcsin(np.sin(lat_rad) * np.cos(distance) +
                         np.cos(lat_rad) * np.sin(distance) * np.cos(bearing))

    # Calculate destination point longitude
    dest_lon = lon_rad + np.arctan2(np.sin(bearing) * np.sin(distance) * np.cos(lat_rad),
                                    np.cos(distance) - np.sin(lat_rad) * np.sin(dest_lat))

    # Convert latitude and longitude back to degrees
    dest_lat = np.degrees(dest_lat)
    dest_lon = np.degrees(dest_lon)

    return dest_lon, dest_lat


def distance_in_m(lon1, lat1, lon2, lat2):
    """

    Calculate distance in m between two geographic coordinates. Uses Haversine
    formula.Only a rough estimate

    Parameters
    ----------
    lon1 : float
        longitude 1 in degree.
    lat1 : float
        latitude 1 in degree.
    lon2 : float
        longitude 2 in degree.
    lat2 : float
        latitude 2 in degree.

    Returns
    -------
    float
        distance in m between the two coordinates.

    """
    EARTH_RADIUS = 6371000

    dlat = np.deg2rad(lat2-lat1)
    dlon = np.deg2rad(lon2-lon1)

    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    a = (np.sin(dlat/2) * np.sin(dlat/2) +
         np.sin(dlon/2) * np.sin(dlon/2) * np.cos(lat1) * np.cos(lat2))

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return c*EARTH_RADIUS


def convert_geographic_grid(geo1, geo2, grid1, grid2, p):
    """
    Convert coordinates from one grid to another grid, by two known grid points.

    Parameters
    ----------
    geo1 : iterable of len 2
        lower left corner in grid 1
    geo2 : iterable of len 2
        upper right corner in grid 1
    grid1 : iterable of len 2
        lower left corner in grid 2
    grid2 : iterable of len 2
        upper right corner in grid 2
    p : tuple with 2 arrays
        x and y positions of points to be converted from grid 1 to grid 2

    Returns
    -------
    x : iterable
        x positions of p in grid 2
    y : iterable
        y positions of p in grid 2

    """

    lon1, lat1 = geo1
    lon2, lat2 = geo2
    x1, y1 = grid1
    x2, y2 = grid2
    lon, lat = p

    x = np.interp(lon, [lon1, lon2], [x1, x2])
    y = np.interp(lat, [lat1, lat2], [y1, y2])

    return x, y


def get_light_path(x0, y0, x1, y1):
    """
    Returns x and y coordinates of a line approximated along a sparse grid.

    "https://en.wikipedia.org/wiki/Bresenham's_line_algorithm"
    """

    dx = abs(x1-x0)
    sx = 1 if x1 > x0 else -1
    dy = -abs(y1-y0)
    sy = 1 if y1 > y0 else -1

    err = dx+dy

    if abs(x1-x0) > abs(y1-y0):
        n = abs(x1-x0)+1
    else:
        n = abs(y1-y0)+1


    x_eval = np.array([np.nan]*n)
    y_eval = np.array([np.nan]*n)

    i = 0
    while True:
        x_eval[i] = x0
        y_eval[i] = y0
        if x0 == x1 and y0==y1:
            break
        err2 = 2*err

        if err2 >= dy:
            err = err+dy
            x0 = x0+sx

        if err2 <= dx:
            err = err+dx
            y0 = y0+sy

        i = i+1

    return np.array(x_eval, dtype=int), np.array(y_eval, dtype=int)


def statistics(data):
    """


    Parameters
    ----------
    data : iterable
        array of values.

    Returns
    -------
    output : tuple
        mean, median, min, max and sd of data.

    """

    output = (np.nanmean(data), np.nanmedian(data), np.nanmin(data), np.nanmax(data),
              np.nanstd(data))

    return output


def trajectories_reverse(lon, lat, u, v, time):
    """

    This function is used to calculate the plume trajectory. Starting at Longitude (lon),
    and Latitude (lat) an air parcel will be transported for an amount of time in seconds.

    Parameters
    ----------
    lon : float
        starting longitude of trajectory.
    lat : float
        starting latitude of trajectory.
    u : float
        u-component of the wind (WEST-EAST) in m/s.
        positive towards EAST.
    v : float
        v-component of the wind (SOUTH-NORTH) in m/s.
        positive towards NORTH.
    time : float
        time in seconds between start and end of trajectory.

    Returns
    -------
    lon_new : float
        longitude end position of trajectory.
    lat_new : float
        latitude end position of trajectory.

    """

    lon_new, lat_new = add_distance_in_m(lon, lat, -u*time, -v*time)

    return lon_new, lat_new


def trajectories(lon, lat, u, v, time):
    """

    This function is used to calculate the plume trajectory. Starting at Longitude (lon),
    and Latitude (lat) an air parcel will be transported for an amount of time in seconds.

    Parameters
    ----------
    lon : float
        starting longitude of trajectory.
    lat : float
        starting latitude of trajectory.
    u : float
        u-component of the wind (WEST-EAST) in m/s.
        positive towards EAST.
    v : float
        v-component of the wind (SOUTH-NORTH) in m/s.
        positive towards NORTH.
    time : float
        time in seconds between start and end of trajectory.

    Returns
    -------
    lon_new : float
        longitude end position of trajectory.
    lat_new : float
        latitude end position of trajectory.

    """

    lon_new, lat_new = add_distance_in_m(lon, lat, u*time, v*time)

    return lon_new, lat_new


def calc_sunrise_sunset(lon, lat, start_date, end_date):
    """


    Parameters
    ----------
    lon : float
        Longitude of measurement device.
    lat : float
        Latitude of measurement device.
    start_date : str
        startdate as "yyyy-mm-dd".
    end_date : str
        enddate as "yyyy-mm-dd".

    Returns
    -------
    sun : TYPE
        DESCRIPTION.

    """
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    if not str(start_date.tz) == "UTC" and str(end_date.tz) == "UTC":
        dates = dates.tz_localize(tz="UTC")

    sun = pvlib.solarposition.sun_rise_set_transit_ephem(dates,
                                                         lat,
                                                         lon,
                                                         next_or_previous="next",
                                                         altitude=0,
                                                         pressure=101325,
                                                         temperature=12,
                                                         horizon='0:00')


    sun["Date"] = pd.to_datetime(sun.index).date
    sun.set_index("Date", inplace=True)

    return sun


def prepare_weather_mean(file):
    data_weather = pd.read_csv(file)
    data_weather.drop("Unnamed: 0", axis=1, inplace=True)
    data_weather["Time"] = pd.to_datetime(data_weather["Time"], utc=True)
    data_weather.set_index("Time", inplace=True)
    data_weather["u"], data_weather["v"] = calc_wind_components(data_weather["Wind Spd."], data_weather["Wind Dir."])
    data_weather = data_weather.resample("30min").mean()
    data_weather["Wind Spd."], data_weather["Wind Dir."] = calc_wind_speed_dir(data_weather["u"], data_weather["v"])
    return data_weather


def prepare_weather_std(file):
    data_weather = pd.read_csv(file)
    data_weather.drop("Unnamed: 0", axis=1, inplace=True)
    data_weather["Time"] = pd.to_datetime(data_weather["Time"], utc=True)
    data_weather.set_index("Time", inplace=True)
    data_weather["u"], data_weather["v"] = calc_wind_components(data_weather["Wind Spd."], data_weather["Wind Dir."])
    data_weather = data_weather.resample("30min").std()
    data_weather["Wind Spd."], data_weather["Wind Dir."] = calc_wind_speed_dir(data_weather["u"], data_weather["v"])
    return data_weather


def prepare_radiation(file):
    data_strahlung = pd.read_csv(file)
    data_strahlung["time"] = pd.to_datetime(data_strahlung["time"], utc=True)
    data_strahlung.set_index("time", inplace=True)
    return data_strahlung


def prepare_cloudcover(file):
    data_bedeckung = pd.read_csv(file)
    data_bedeckung["time"] = pd.to_datetime(data_bedeckung["time"], utc=True)
    data_bedeckung.set_index("time", inplace=True)
    return data_bedeckung


def prepare_sun_data(file):
    data_sonne = pd.read_csv(file)
    data_sonne["Date"] = pd.to_datetime(data_sonne["Date"], utc=True)
    # we want to have some extra time after sunrise and after sunset
    # because the situation is not suddenly changed but rather transitions to
    # night/daytime conditions
    data_sonne["sunrise"] = pd.to_datetime(data_sonne["sunrise"], utc=True)+pd.Timedelta(1, "h")
    data_sonne["sunset"] = pd.to_datetime(data_sonne["sunset"], utc=True)+pd.Timedelta(1, "h")
    data_sonne.set_index("Date", inplace=True)
    return data_sonne

def convertToConcentration(tc, pressure, temperature, conversion_factor,
                           au):
    """
    Parameters
    ----------
    tc : array
        Measured trace gas mixing ratio.
    pressure : array
        Measured pressure in Pa.
    temperature : array
        measured temperatue in K.
    conversion_factor : float, optional
        Conversion factor to be used to get from mixing ratio to concentration.
        For example 1 ppb is 1 molecue per 1e9 molecules.
        For example 1 ppm is 1 molecule per 1e6 molecules.
        The default is 1e-9.
    au : float
        Atomic mass of the trace gas molecule.

    Returns
    -------
    concentration : array
        Concentration of the measured trace gas in g/m³.

    """

    R = 8.314459848  # J/(mol*K)

    mixing_ratio = tc*conversion_factor # ppb is actually 1/100 000 000 so 1e-9
    concentration  = (au*pressure*mixing_ratio)/(R*temperature)

    return concentration