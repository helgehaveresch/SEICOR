
#%%
"""
Simple 3D Gaussian puff model with moving source.

Produces a 3D concentration distribution (time, lat, lon, z) as an xarray.Dataset.

Inputs (examples / accepted types):
- source_lats, source_lons: 1D arrays of length T giving source location per timestep (degrees)
- times: 1D array-like of length T (can be datetimes or numeric)
- u, v: arrays with shape (T, ny, nx) or (T,) scalars; wind fields in m/s (u east, v north)
- sigma_h, sigma_z: horizontal and vertical dispersion (meters) (scalars or length-T arrays)
- emission_rate: mass per puff (Q) in arbitrary units (scalar or length-T)
- plume_rise_params: dict with keys 'A' and 'L' where rise = A*(1 - exp(-wind_speed / L))
- grid: optional dict specifying grid extents and resolution:
    {
      'lon_min': , 'lon_max': , 'lat_min': , 'lat_max': , 'nx': , 'ny': , 'z_levels': array_like (m)
    }
  If not provided, grid is built around source positions with default extents.

Returns: xarray.Dataset with variable `C` (concentration) dims (time, lat, lon, z)

Notes / assumptions:
- Uses simple equirectangular approximation to convert degrees <-> meters.
- Treats each timestep independently (instantaneous puff at that timestep centered at source location).
- Plume rise is computed from local wind speed (spatial mean of u/v at time step) and applied as a vertical offset to the Gaussian center.

"""

from typing import Optional, Sequence, Union, Dict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
import matplotlib.dates as mdates
import pandas as pd
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
import SEICOR.plumes

def _meters_per_deg(lat_deg: float):
    """Return approximate meters per degree longitude and latitude at given latitude."""
    # meters per degree latitude ~ 111132 m
    m_per_deg_lat = 111132.0
    # meters per degree longitude varies with latitude
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_deg))
    return m_per_deg_lon, m_per_deg_lat


def puff_model_3D(
    source_lats: Sequence[float],
    source_lons: Sequence[float],
    times: Sequence,
    u: Union[np.ndarray, float],
    v: Union[np.ndarray, float],
    ship_lats: Optional[Sequence[float]] = None,
    ship_lons: Optional[Sequence[float]] = None,
    sigma_h: Union[float, Sequence[float]] = 200.0,
    sigma_z: Union[float, Sequence[float]] = 50.0,
    emission_rate: Union[float, Sequence[float]] = 1.0,
    emission_height: float = 0.0,
    plume_rise_model: str = "briggs",
    plume_rise_params: Optional[Dict[str, float]] = None,
    grid: Optional[Dict] = None,
    # new parameters controlling puff spread growth (m^2/s)
    Kh: float = 10.0,
    Kz: float = 3.0,
    ) -> xr.Dataset:
    """Compute 3D Gaussian puff fields for each timestep and return as xr.Dataset.

    The function treats each timestep independently: a puff is created at the source position for
    that timestep with mass `emission_rate` and spread `sigma_h`/`sigma_z`.

    """
    source_lats = np.asarray(source_lats)
    source_lons = np.asarray(source_lons)
    times = np.asarray(times)
    nt = len(times)

    if source_lats.shape[0] != nt or source_lons.shape[0] != nt:
        raise ValueError("source_lats/source_lons must have same length as times")

    # normalize sigma and emission_rate to arrays length nt
    if np.isscalar(sigma_h):
        sigma_h_arr = np.full(nt, float(sigma_h))
    else:
        sigma_h_arr = np.asarray(sigma_h, dtype=float)
    if np.isscalar(sigma_z):
        sigma_z_arr = np.full(nt, float(sigma_z))
    else:
        sigma_z_arr = np.asarray(sigma_z, dtype=float)
    if np.isscalar(emission_rate):
        Q_arr = np.full(nt, float(emission_rate))
    else:
        Q_arr = np.asarray(emission_rate, dtype=float)

    # handle wind arrays
    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    # acceptable shapes: (nt, ny, nx) or (nt,) or (nt,ny,nx) etc. We'll accept (nt, ny, nx) or (nt,)

    # build grid
    if grid is None:
        # build a grid centered on the center source position ±500 m by default
        center_idx = int(len(source_lats) / 2)
        slat0 = float(source_lats[center_idx])
        slon0 = float(source_lons[center_idx])
        mperlon, mperlat = _meters_per_deg(slat0)
        # default extents: ±500 m around center source
        buff_m = 500.0
        # convert meters to degrees
        dlat = buff_m / mperlat
        dlon = buff_m / mperlon
        lat_min = slat0 - dlat
        lat_max = slat0 + dlat
        lon_min = slon0 - dlon
        lon_max = slon0 + dlon
        nx = 242
        ny = 242
        # vertical levels default
        z_levels = np.arange(0.0, 410.0, 10.0)
    else:
        lon_min = grid['lon_min']
        lon_max = grid['lon_max']
        lat_min = grid['lat_min']
        lat_max = grid['lat_max']
        nx = grid.get('nx', 242)
        ny = grid.get('ny', 242)
        z_levels = np.asarray(grid.get('z_levels', np.arange(0.0, 410.0, 10.0)))

    lon = np.linspace(lon_min, lon_max, nx)
    lat = np.linspace(lat_min, lat_max, ny)
    z = np.asarray(z_levels)

    # precompute meters per degree using lat_center for lon scaling
    lat_center = np.mean([lat_min, lat_max])
    mperlon, mperlat = _meters_per_deg(lat_center)

    # create 2D grids of lon/lat and convert to meters relative to source
    lon2d, lat2d = np.meshgrid(lon, lat)
    # precompute meter-scaled grids to avoid repeated multiplication
    lon2d_m_base = lon2d * mperlon
    lat2d_m_base = lat2d * mperlat

    # prepare output array: (time, z, lat, lon)
    C = np.zeros((nt, z.size, ny, nx), dtype=float)

    # Prepare wind time-series as scalar means per timestep (m/s)
    def _time_means(arr):
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == nt:
            return a.astype(float)
        if a.ndim == 3 and a.shape[0] == nt:
            return np.array([float(np.nanmean(a[t])) for t in range(nt)])
        # fallback: try to coerce
        try:
            return np.asarray([float(a[t]) for t in range(nt)])
        except Exception:
            return np.zeros(nt)

    u_mean = _time_means(u_arr)
    v_mean = _time_means(v_arr)

    # compute cumulative time (seconds) from the first time entry
    times_arr = np.asarray(times)
    if np.issubdtype(times_arr.dtype, np.datetime64):
        ts_seconds = (times_arr.astype('datetime64[s]') - times_arr[0].astype('datetime64[s]'))
        times_seconds = ts_seconds.astype('timedelta64[s]').astype(float)
    else:
        # assume numeric times; interpret as seconds if not otherwise
        times_seconds = (times_arr.astype(float) - float(times_arr[0])).astype(float)

    z_rise_arr, dz_dF, dz_dS = calc_plume_rise_array(nt, times_seconds, 
                                       u_mean, v_mean, 
                                       plume_rise_params, 
                                       mode=plume_rise_model)
    
    #precompute the chemical evolution of the puff per timestep
    Q_arr, dQdQ0, dQdR, dQdf, dQdt, dQdtau = calc_NO_NO2_conversion(nt, times_seconds, Q_arr, NO_NO2_conversion_params=None)

    # timestep durations for each index (seconds); for last step reuse previous dt if needed
    if nt > 1:
        dt_raw = np.diff(times_seconds)
        dt_steps = np.empty(nt, dtype=float)
        dt_steps[:-1] = dt_raw
        dt_steps[-1] = dt_raw[-1]
    else:
        dt_steps = np.array([1.0], dtype=float)

    # cumulative displacement from time 0 to each index in meters
    cum_dx = np.zeros(nt, dtype=float)
    cum_dy = np.zeros(nt, dtype=float)
    for i in range(1, nt):
        cum_dx[i] = cum_dx[i - 1] + u_mean[i - 1] * dt_steps[i - 1]
        cum_dy[i] = cum_dy[i - 1] + v_mean[i - 1] * dt_steps[i - 1]

    # For each time (t0), accumulate contributions from all previous emissions (t1 <= t0)

    slat0 = float(source_lats[0])
    slon0 = float(source_lons[0])
    sigma_h0 = float(sigma_h_arr[0])
    sigma_z0 = float(sigma_z_arr[0])
    z0 = float(emission_height)    

    for t0 in range(nt):

        for t1 in range(0, t0+1):

            # elapsed time since emission (seconds)
            tau = float(times_seconds[t0] - times_seconds[t1])
            #last timestep
            dt = float(dt_steps[t1])
            
            # advected center: displacement between t0 and t1 (meters)
            dx_m = cum_dx[t1] - cum_dx[t0]
            dy_m = cum_dy[t1] - cum_dy[t0]
            # convert meters displacement to degrees using meters-per-degree at lat_center
            dlon = dx_m / mperlon
            dlat = dy_m / mperlat

            slon_adv = source_lons[t1] + dlon
            slat_adv = source_lats[t1] + dlat

            # use precomputed time-dependent plume rise for this timestep
            z0 = float(emission_height) + float(z_rise_arr[t0-t1])
            sigma_h0 = np.sqrt(2 * Kh * tau + 1e-12)
            sigma_z0 = np.sqrt(2 * Kz * tau + 1e-12)

            # compute emission-specific Q and use vectorized calculations across z
            Q_t1 = float(Q_arr[t0-t1])


            C_0 = Q_t1 * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12)* 10**-6  # convert from molecules/m^3 to molecules/cm^3 for columns in molecules/cm^2

            # advected mesh (in meters) relative to instrument/source
            lon_mesh_m = lon2d_m_base - slon_adv * mperlon
            lat_mesh_m = lat2d_m_base - slat_adv * mperlat

            lon_exp = np.exp(-(lon_mesh_m ** 2) / (2.0 * sigma_h0 ** 2))
            lat_exp = np.exp(-(lat_mesh_m ** 2) / (2.0 * sigma_h0 ** 2))

            # vertical exponent for all z levels (vectorized)
            vert_exp = np.exp(-0.5 * ((z - z0) ** 2) / (sigma_z0 ** 2)) + np.exp(-0.5 * ((z + z0) ** 2) / (sigma_z0 ** 2))

            # broadcast multiply: (nz, ny, nx) = (nz,1,1) * (1,ny,nx) * (1,ny,nx)
            contrib = C_0 * (vert_exp[:, None, None] * lon_exp[None, :, :] * lat_exp[None, :, :])
            C[t0, :, :, :] += contrib

            center_points = (slat_adv, slon_adv, z0)
            

    # build xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'C': (('time', 'z', 'lat', 'lon'), C)
        },
        coords={
            'time': times,
            'z': z,
            'lat': lat,
            'lon': lon,
            # store source trajectory as time coordinates
            'source_lat': (('time',), source_lats),
            'source_lon': (('time',), source_lons),
        },
        attrs={
            'description': '3D Gaussian puff model concentrations',
            'units': 'concentration'
        }
    )

    # --- puff centerpoints per observation time (time x emission) ---
    # create arrays shaped (time, emission_index) where emission_index corresponds to source/emission times
    try:
        puff_time_since = np.full((nt, nt), np.nan, dtype=float)
        puff_center_lat = np.full((nt, nt), np.nan, dtype=float)
        puff_center_lon = np.full((nt, nt), np.nan, dtype=float)
        puff_center_z = np.full((nt, nt), np.nan, dtype=float)

        for t0 in range(nt):
            for t1 in range(0, t0 + 1):
                # time since emission (observation t0 minus emission t1)
                tau = float(times_seconds[t0] - times_seconds[t1])
                # advected center using same convention as model loop: dx from cum_dx
                dx_m = cum_dx[t1] - cum_dx[t0]
                dy_m = cum_dy[t1] - cum_dy[t0]
                dlon = dx_m / mperlon
                dlat = dy_m / mperlat
                slon_adv = source_lons[t1] + dlon
                slat_adv = source_lats[t1] + dlat
                # vertical center used in model (emission height + rise for elapsed time index)
                try:
                    z0_local = float(emission_height) + float(z_rise_arr[t0 - t1])
                except Exception:
                    z0_local = float(emission_height)

                puff_time_since[t0, t1] = tau
                puff_center_lat[t0, t1] = slat_adv
                puff_center_lon[t0, t1] = slon_adv
                puff_center_z[t0, t1] = z0_local

        # attach as data variables (dimensions: time, emission)
        ds = ds.assign(
            puff_time_since=(('time', 'emission'), puff_time_since),
            puff_center_lat=(('time', 'emission'), puff_center_lat),
            puff_center_lon=(('time', 'emission'), puff_center_lon),
            puff_center_z=(('time', 'emission'), puff_center_z),
        )
        # add emission coordinate as the source/emission times for the second axis
        ds = ds.assign_coords({'emission': (('emission',), times)})
    except Exception:
        # if anything fails, return dataset without puff arrays
        pass

    return ds


def puff_model_2D(
    source_lats: Sequence[float],
    source_lons: Sequence[float],
    times: Sequence,
    u: Union[np.ndarray, float],
    v: Union[np.ndarray, float],
    instrument_lat: float,
    instrument_lon: float,
    instrument_altitude: float = 0.0,
    sigma_h: Union[float, Sequence[float]] = 200.0,
    sigma_z: Union[float, Sequence[float]] = 50.0,
    emission_rate: Union[float, Sequence[float]] = 1.0,
    emission_height: float = 0.0,
    plume_rise_model: str = "briggs",
    plume_rise_params: Optional[Dict[str, float]] = None,
    NO_NO2_conversion_params: Optional[Dict[str, float]] = None,
    grid: Optional[Dict] = None,
    Kh: float = 10.0,
    Kz: float = 3.0,
    azimuth_deg: float = 180.0,
    viewing_elevations: Sequence[float] = (0.0,),
    distance_max: Optional[float] = None,
    spacing_m: float = 5.0,
) -> xr.Dataset:

    """
    Compute 2D Gaussian puff fields for each timestep along the instrument's lines of sight (LOS).
    Each LOS is defined by the instrument (lat, lon, z), azimuth, and a viewing elevation angle.
    For each LOS, sample points every spacing_m (default 5 m) along the LOS, and compute the concentration at each point.
    Returns xarray.Dataset with dims (time, elevation, los_distance) and variable 'C'.
    """

    source_lats = np.asarray(source_lats)
    source_lons = np.asarray(source_lons)
    times = np.asarray(times)
    nt = len(times)
    az_rad = np.deg2rad(azimuth_deg)
    mperlon, mperlat = _meters_per_deg(float(instrument_lat))
    delta_east = (np.asarray(source_lons) - instrument_lon) * mperlon
    delta_north = (np.asarray(source_lats) - instrument_lat) * mperlat
    # Project onto azimuth direction (0=north, 90=east)
    source_distances = delta_east * np.sin(az_rad) + delta_north * np.cos(az_rad)
    # Perpendicular (cross-azimuth) distance from instrument to source
    # (azimuth + 90 deg is the perpendicular direction)
    az_perp = az_rad + np.pi/2
    source_cross = delta_east * np.sin(az_perp) + delta_north * np.cos(az_perp)

    # Prepare LOS grid
    viewing_elevations = np.asarray(viewing_elevations)
    n_elev = len(viewing_elevations)
    if distance_max is None:
        distance_max = 2000.0  # meters
    los_distances = np.arange(0.0, distance_max + spacing_m, spacing_m)
    n_los = len(los_distances)
    # For each LOS, compute (x, y, z) coordinates for each point along the LOS
    # Instrument is at (instrument_lat, instrument_lon, instrument_altitude)
    # For each elevation: x = d * sin(az) * cos(el), y = d * cos(az) * cos(el), z = instrument_altitude + d * sin(el)
    los_coords = np.zeros((n_elev, n_los, 3), dtype=float)  # (elev, los, xyz)
    for i, elev_deg in enumerate(viewing_elevations):
        elev_rad = np.deg2rad(elev_deg)
        # x (east), y (north), z (up)
        los_coords[i, :, 0] = los_distances * np.sin(az_rad) * np.cos(elev_rad)
        los_coords[i, :, 1] = los_distances * np.cos(az_rad) * np.cos(elev_rad)
        los_coords[i, :, 2] = instrument_altitude + los_distances * np.sin(elev_rad)
    if np.isscalar(sigma_h):
        sigma_h_arr = np.full(nt, float(sigma_h))
    else:
        sigma_h_arr = np.asarray(sigma_h, dtype=float)
    if np.isscalar(sigma_z):
        sigma_z_arr = np.full(nt, float(sigma_z))
    else:
        sigma_z_arr = np.asarray(sigma_z, dtype=float)
    if np.isscalar(emission_rate):
        Q_arr = np.full(nt, float(emission_rate))
    else:
        Q_arr = np.asarray(emission_rate, dtype=float)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)

    # For 2D model, we assume the plume is centered at the source_cross=0 (i.e., instrument azimuth), and integrate over the perpendicular direction analytically (Gaussian integral)

    # Prepare output array: (time, elevation, los_distance)
    C = np.zeros((nt, n_elev, n_los), dtype=float)

    # Prepare wind time-series as scalar means per timestep (m/s)
    def _time_means(arr):
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == nt:
            return a.astype(float)
        if a.ndim == 3 and a.shape[0] == nt:
            return np.array([float(np.nanmean(a[t])) for t in range(nt)])
        try:
            return np.asarray([float(a[t]) for t in range(nt)])
        except Exception:
            return np.zeros(nt)

    u_mean = _time_means(u_arr)
    v_mean = _time_means(v_arr)

    # compute cumulative time (seconds) from the first time entry
    times_arr = np.asarray(times)
    if np.issubdtype(times_arr.dtype, np.datetime64):
        ts_seconds = (times_arr.astype('datetime64[s]') - times_arr[0].astype('datetime64[s]'))
        times_seconds = ts_seconds.astype('timedelta64[s]').astype(float)
    else:
        times_seconds = (times_arr.astype(float) - float(times_arr[0])).astype(float)

    # precompute time-dependent plume rise per timestep 
    z_rise_arr, dz_dF, dz_dS = calc_plume_rise_array(nt, times_seconds, 
                                       u_mean, v_mean, 
                                       plume_rise_params, 
                                       mode=plume_rise_model)


    #precompute the chemical evolution of the puff per timestep
    Q_arr, dQdQ0, dQdR, dQdf, dQdt, dQdtau = calc_NO_NO2_conversion(nt, times_seconds, Q_arr, NO_NO2_conversion_params=NO_NO2_conversion_params)

    # timestep durations for each index (seconds); for last step reuse previous dt if needed
    if nt > 1:
        dt_raw = np.diff(times_seconds)
        dt_steps = np.empty(nt, dtype=float)
        dt_steps[:-1] = dt_raw
        dt_steps[-1] = dt_raw[-1]
    else:
        dt_steps = np.array([1.0], dtype=float)

    # cumulative displacement from time 0 to each index in meters (projected along azimuth)
    az_rad = np.deg2rad(azimuth_deg)
    cum_disp_along = np.zeros(nt, dtype=float)
    cum_disp_cross = np.zeros(nt, dtype=float)
    for i in range(1, nt):
        # project wind onto azimuth direction
        wind_proj = u_mean[i - 1] * np.sin(az_rad) + v_mean[i - 1] * np.cos(az_rad)
        cum_disp_along[i] = cum_disp_along[i - 1] + wind_proj * dt_steps[i - 1]
        # for cross-azimuth, project onto perpendicular direction
        az_perp = az_rad + np.pi/2
        wind_proj_perp = u_mean[i - 1] * np.sin(az_perp) + v_mean[i - 1] * np.cos(az_perp)
        cum_disp_cross[i] = cum_disp_cross[i - 1] + wind_proj_perp * dt_steps[i - 1]



    # Vectorized version: eliminate for-loops over LOS and elevation
    # Precompute LOS grid (shape: n_elev, n_los, 3)
    x = los_coords[..., 0]  # (n_elev, n_los)
    y = los_coords[..., 1]
    z_pt = los_coords[..., 2]
    # Projected distances for all LOS points
    along = x * np.sin(az_rad) + y * np.cos(az_rad)  # (n_elev, n_los)
    az_perp = az_rad + np.pi/2
    cross = x * np.sin(az_perp) + y * np.cos(az_perp)

    for t0 in range(nt):
        # For each t0, sum contributions from all t1 <= t0
        contrib_sum = np.zeros((n_elev, n_los), dtype=float)
        for t1 in range(0, t0 + 1):
            tau = float(times_seconds[t0] - times_seconds[t1])
            dt = float(dt_steps[t1])
            dx = cum_disp_along[t1] - cum_disp_along[t0]
            dy = cum_disp_cross[t1] - cum_disp_cross[t0]
            center_dist = dx + source_distances[t1]
            cross_dist = dy + source_cross[t1]
            z0 = float(emission_height) + float(z_rise_arr[t0 - t1])
            sigma_h0 = np.sqrt(2 * Kh * tau + 1e-12)
            sigma_z0 = np.sqrt(2 * Kz * tau + 1e-12)
            Q_t1 = float(Q_arr[t0 - t1])
            norm = Q_t1 * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12)* 10**-6  # convert from molecules/m^3 to molecules/cm^3 for columns in molecules/cm^2
            # Vectorized Gaussian evaluation
            dist_exp = np.exp(-((along - center_dist) ** 2) / (2.0 * sigma_h0 ** 2))
            cross_exp = np.exp(-(cross_dist ** 2) / (2.0 * sigma_h0 ** 2))
            vert_exp = np.exp(-0.5 * ((z_pt - z0) ** 2) / (sigma_z0 ** 2)) + np.exp(-0.5 * ((z_pt + z0) ** 2) / (sigma_z0 ** 2))
            contrib = norm * dist_exp * cross_exp * vert_exp
            contrib_sum += contrib
        C[t0, :, :] = contrib_sum

    # --- Calculate puff centerline in instrument coordinates ---
    # For each emission (t1) and observation time (t0), compute puff center z and distance to LOS plane
    puff_center_z = np.full((nt, nt), np.nan, dtype=float)
    puff_center_cross_distance = np.full((nt, nt), np.nan, dtype=float)
    puff_center_along_distance = np.full((nt, nt), np.nan, dtype=float)
    for t0 in range(nt):
        for t1 in range(0, t0 + 1):
            tau = float(times_seconds[t0] - times_seconds[t1])
            dx = cum_disp_along[t1] - cum_disp_along[t0]
            dy = cum_disp_cross[t1] - cum_disp_cross[t0]
            center_dist = dx + source_distances[t1]
            cross_dist = dy + source_cross[t1]
            z0 = float(emission_height) + float(z_rise_arr[t0 - t1])
            puff_center_z[t0, t1] = z0
            puff_center_cross_distance[t0, t1] = cross_dist
            puff_center_along_distance[t0, t1] = center_dist

    ds = xr.Dataset(
        data_vars={
            'C': (('time', 'elevation', 'los_distance'), C),
            'puff_center_z': (('time', 'emission'), puff_center_z),
            'puff_center_cross_distance': (('time', 'emission'), puff_center_cross_distance),
            'puff_center_along_distance': (('time', 'emission'), puff_center_along_distance),
        },
        coords={
            'time': times,
            'elevation': viewing_elevations,
            'los_distance': los_distances,
            'emission': (('emission',), times),
        },
        attrs={
            'description': '2D Gaussian puff model concentrations along instrument lines of sight',
            'units': 'concentration'
        }
    )
    return ds


def puff_model_2D_with_derivatives(
    source_lats: Sequence[float],
    source_lons: Sequence[float],
    times: Sequence,
    u: Union[np.ndarray, float],
    v: Union[np.ndarray, float],
    instrument_lat: float,
    instrument_lon: float,
    instrument_altitude: float = 0.0,
    sigma_h: Union[float, Sequence[float]] = 200.0,
    sigma_z: Union[float, Sequence[float]] = 50.0,
    emission_rate: Union[float, Sequence[float]] = 1.0,
    emission_height: float = 0.0,
    plume_rise_model: str = "briggs",
    plume_rise_params: Optional[Dict[str, float]] = None,
    NO_NO2_conversion_params: Optional[Dict[str, float]] = None,
    grid: Optional[Dict] = None,
    Kh: float = 10.0,
    Kz: float = 3.0,
    azimuth_deg: float = 180.0,
    viewing_elevations: Sequence[float] = (0.0,),
    distance_max: Optional[float] = None,
    spacing_m: float = 5.0,
    background_columns: float = None
) -> xr.Dataset:

    """
    Compute 2D Gaussian puff fields for each timestep along the instrument's lines of sight (LOS).
    Each LOS is defined by the instrument (lat, lon, z), azimuth, and a viewing elevation angle.
    For each LOS, sample points every spacing_m (default 5 m) along the LOS, and compute the concentration at each point.
    Returns xarray.Dataset with dims (time, elevation, los_distance) and variable 'C'.
    """

    source_lats = np.asarray(source_lats)
    source_lons = np.asarray(source_lons)
    times = np.asarray(times)
    nt = len(times)
    az_rad = np.deg2rad(azimuth_deg)
    mperlon, mperlat = _meters_per_deg(float(instrument_lat))
    delta_east = (np.asarray(source_lons) - instrument_lon) * mperlon
    delta_north = (np.asarray(source_lats) - instrument_lat) * mperlat
    # Project onto azimuth direction (0=north, 90=east)
    source_distances = delta_east * np.sin(az_rad) + delta_north * np.cos(az_rad)
    # Perpendicular (cross-azimuth) distance from instrument to source
    # (azimuth + 90 deg is the perpendicular direction)
    az_perp = az_rad + np.pi/2
    source_cross = delta_east * np.sin(az_perp) + delta_north * np.cos(az_perp)

    # Prepare LOS grid
    viewing_elevations = np.asarray(viewing_elevations)
    n_elev = len(viewing_elevations)
    if distance_max is None:
        distance_max = 2000.0  # meters
    los_distances = np.arange(0.0, distance_max + spacing_m, spacing_m)
    n_los = len(los_distances)
    # For each LOS, compute (x, y, z) coordinates for each point along the LOS
    # Instrument is at (instrument_lat, instrument_lon, instrument_altitude)
    # For each elevation: x = d * sin(az) * cos(el), y = d * cos(az) * cos(el), z = instrument_altitude + d * sin(el)
    los_coords = np.zeros((n_elev, n_los, 3), dtype=float)  # (elev, los, xyz)
    for i, elev_deg in enumerate(viewing_elevations):
        elev_rad = np.deg2rad(elev_deg)
        # x (east), y (north), z (up)
        los_coords[i, :, 0] = los_distances * np.sin(az_rad) * np.cos(elev_rad)
        los_coords[i, :, 1] = los_distances * np.cos(az_rad) * np.cos(elev_rad)
        los_coords[i, :, 2] = instrument_altitude + los_distances * np.sin(elev_rad)

    if np.isscalar(emission_rate):
        Q_arr = np.full(nt, float(emission_rate))
    else:
        Q_arr = np.asarray(emission_rate, dtype=float)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)

    # For 2D model, we assume the plume is centered at the source_cross=0 (i.e., instrument azimuth), and integrate over the perpendicular direction analytically (Gaussian integral)

    # Prepare output array: (time, elevation, los_distance)

    columns = np.zeros((nt, n_elev), dtype=float)  # placeholder for potential additional derivatives
    dcolumns_dQ0 = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dR = np.zeros((nt, n_elev), dtype=float)
    dcolumns_df0 = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dt = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dtau = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dF = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dS = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dsigma_h0 = np.zeros((nt, n_elev), dtype=float)
    dcolumns_dsigma_z0 = np.zeros((nt, n_elev), dtype=float )
    dcolumns_dbackground = np.zeros((nt, n_elev), dtype=float)
    jacobian = np.zeros((nt, n_elev, 10), dtype=float)  # 10 parameters np.zeros((nt, n_elev, n_los), dtype=float)  # placeholder for potential Jacobian of LOS integration

    # Prepare wind time-series as scalar means per timestep (m/s)
    def _time_means(arr):
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == nt:
            return a.astype(float)
        if a.ndim == 3 and a.shape[0] == nt:
            return np.array([float(np.nanmean(a[t])) for t in range(nt)])
        try:
            return np.asarray([float(a[t]) for t in range(nt)])
        except Exception:
            return np.zeros(nt)

    u_mean = _time_means(u_arr)
    v_mean = _time_means(v_arr)

    # compute cumulative time (seconds) from the first time entry
    times_arr = np.asarray(times)
    if np.issubdtype(times_arr.dtype, np.datetime64):
        ts_seconds = (times_arr.astype('datetime64[s]') - times_arr[0].astype('datetime64[s]'))
        times_seconds = ts_seconds.astype('timedelta64[s]').astype(float)
    else:
        times_seconds = (times_arr.astype(float) - float(times_arr[0])).astype(float)

    # precompute time-dependent plume rise per timestep 
    z_rise_arr, dz_dF, dz_dS = calc_plume_rise_array(nt, times_seconds, 
                                       u_mean, v_mean, 
                                       plume_rise_params, 
                                       mode=plume_rise_model)


    #precompute the chemical evolution of the puff per timestep
    Q_arr, dQdQ0, dQdR, dQdf0, dQdt, dQdtau = calc_NO_NO2_conversion(nt, times_seconds, Q_arr, NO_NO2_conversion_params=NO_NO2_conversion_params)

    # timestep durations for each index (seconds); for last step reuse previous dt if needed
    if nt > 1:
        dt_raw = np.diff(times_seconds)
        dt_steps = np.empty(nt, dtype=float)
        dt_steps[:-1] = dt_raw
        dt_steps[-1] = dt_raw[-1]
    else:
        dt_steps = np.array([1.0], dtype=float)

    # cumulative displacement from time 0 to each index in meters (projected along azimuth)
    az_rad = np.deg2rad(azimuth_deg)
    cum_disp_along = np.zeros(nt, dtype=float)
    cum_disp_cross = np.zeros(nt, dtype=float)
    for i in range(1, nt):
        # project wind onto azimuth direction
        wind_proj = u_mean[i - 1] * np.sin(az_rad) + v_mean[i - 1] * np.cos(az_rad)
        cum_disp_along[i] = cum_disp_along[i - 1] + wind_proj * dt_steps[i - 1]
        # for cross-azimuth, project onto perpendicular direction
        az_perp = az_rad + np.pi/2
        wind_proj_perp = u_mean[i - 1] * np.sin(az_perp) + v_mean[i - 1] * np.cos(az_perp)
        cum_disp_cross[i] = cum_disp_cross[i - 1] + wind_proj_perp * dt_steps[i - 1]



    # Vectorized version: eliminate for-loops over LOS and elevation
    # Precompute LOS grid (shape: n_elev, n_los, 3)
    x = los_coords[..., 0]  # (n_elev, n_los)
    y = los_coords[..., 1]
    z_pt = los_coords[..., 2]
    # Projected distances for all LOS points
    along = x * np.sin(az_rad) + y * np.cos(az_rad)  # (n_elev, n_los)
    az_perp = az_rad + np.pi/2
    cross = x * np.sin(az_perp) + y * np.cos(az_perp)

    for t0 in range(nt):
        # For each t0, sum contributions from all t1 <= t0
        contrib_sum = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dQ0 = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dR = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dtau = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_df0 = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dt = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dsigma_h0 = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dsigma_z0 = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dF = np.zeros((n_elev, n_los), dtype=float)
        dcontrib_sum_dS = np.zeros((n_elev, n_los), dtype=float)
        for t1 in range(0, t0 + 1):
            tau = float(times_seconds[t0] - times_seconds[t1])
            dt = float(dt_steps[t1])
            dx = cum_disp_along[t1] - cum_disp_along[t0]
            dy = cum_disp_cross[t1] - cum_disp_cross[t0]
            center_dist = dx + source_distances[t1]
            cross_dist = dy + source_cross[t1]
            apparent_displacement = np.sqrt((dx+source_distances[t1]-source_distances[t0])**2 + (dy+source_cross[t1]-source_cross[t0])**2)
            z0 = float(emission_height) + float(z_rise_arr[t0 - t1])
            #traditional
            #sigma_h0 = np.sqrt(2 * Kh * tau + 1e-12)
            #sigma_z0 = np.sqrt(2 * Kz * tau + 1e-12)
            #dsigma_h0dKdh = tau / np.sqrt(2 * Kh * tau + 1e-12)
            #dsigma_z0dKdz = tau / np.sqrt(2 * Kz * tau + 1e-12)
            #according to briggs
            #x= meters from source wind*time
            x = apparent_displacement
            sigma_h0 = Kh * x / np.sqrt(1+0.0001*x)+ 5
            sigma_z0 = Kz * x / np.sqrt(1+0.0001*x)+ 5
            dsigma_h0dKdh = x / np.sqrt(1+0.0001*x)
            dsigma_z0dKdz = x / np.sqrt(1+0.0001*x)
            
            Q_t1 = float(Q_arr[t0 - t1])
            norm = Q_t1 * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12)* 10**-6  # convert from molecules/m^3 to molecules/cm^3 for columns in molecules/cm^2
            # Vectorized Gaussian evaluation
            dist_exp = np.exp(-((along - center_dist) ** 2) / (2.0 * sigma_h0 ** 2))
            cross_exp = np.exp(-(cross_dist ** 2) / (2.0 * sigma_h0 ** 2))
            vert_exp = np.exp(-0.5 * ((z_pt - z0) ** 2) / (sigma_z0 ** 2)) + np.exp(-0.5 * ((z_pt + z0) ** 2) / (sigma_z0 ** 2))
            contrib = norm * dist_exp * cross_exp * vert_exp
            contrib_sum += contrib

            #calculate derivatives using the same vectorized components
            dcontrib_sum_dQ0 += dQdQ0[t0 - t1] * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12) * dist_exp * cross_exp * vert_exp* 10**-6
            dcontrib_sum_dR += dQdR[t0 - t1] * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12) * dist_exp * cross_exp * vert_exp* 10**-6
            dcontrib_sum_dtau += dQdtau[t0 - t1] * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12) * dist_exp * cross_exp * vert_exp* 10**-6
            dcontrib_sum_df0 += dQdf0[t0 - t1] * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12) * dist_exp * cross_exp * vert_exp* 10**-6
            dcontrib_sum_dt += dQdt[t0 - t1] * dt / (sigma_h0**2 * sigma_z0 * (2.0 * np.pi)**1.5 + 1e-12) * dist_exp * cross_exp * vert_exp* 10**-6
            dcontrib_sum_dsigma_h0 += contrib * dsigma_h0dKdh * ((along - center_dist) ** 2 / sigma_h0 ** 3 + (cross_dist ** 2) / sigma_h0 ** 3 - 2 / sigma_h0) # neglecting the factor for numerical stability
            dcontrib_sum_dsigma_z0 += dsigma_z0dKdz * norm * dist_exp * cross_exp * (((z_pt - z0) ** 2) / (sigma_z0 ** 3) *np.exp(-0.5 * ((z_pt - z0) ** 2) / (sigma_z0 ** 2)) + ((z_pt + z0) ** 2) / (sigma_z0 ** 3) * np.exp(-0.5 * ((z_pt + z0) ** 2) / (sigma_z0 ** 2))) - contrib * dsigma_z0dKdz /sigma_z0 # neglecting the factor for numerical stability
            dcontrib_sum_dF += norm * dist_exp * cross_exp * ( dz_dF[t0 - t1] * (z_pt - z0) / (sigma_z0 ** 2) * np.exp(-0.5 * ((z_pt - z0) ** 2) / (sigma_z0 ** 2)) - dz_dF[t0 - t1] * (z_pt + z0) / (sigma_z0 ** 2) * np.exp(-0.5 * ((z_pt + z0) ** 2) / (sigma_z0 ** 2)))
            dcontrib_sum_dS += norm * dist_exp * cross_exp * ( dz_dS[t0 - t1] * (z_pt - z0) / (sigma_z0 ** 2) * np.exp(-0.5 * ((z_pt - z0) ** 2) / (sigma_z0 ** 2)) - dz_dS[t0 - t1] * (z_pt + z0) / (sigma_z0 ** 2) * np.exp(-0.5 * ((z_pt + z0) ** 2) / (sigma_z0 ** 2)))

        if background_columns is None:
            background_columns = 0.0

        columns[t0, :] = np.trapezoid(contrib_sum, los_distances, axis=1) * 10**2 + background_columns # convert from molecules*m/cm^3 to molecules/cm^2 for columns, and store the integrated column for this time and elevation
        dcolumns_dQ0[t0, :] = np.trapezoid(dcontrib_sum_dQ0, los_distances, axis=1)* 10**2# sensitivity to NO2 emisson rate
        dcolumns_dR[t0, :] = np.trapezoid(dcontrib_sum_dR, los_distances, axis=1)* 10**2# sensitivity to initial NO2/NOx ratio
        dcolumns_df0[t0, :] = np.trapezoid(dcontrib_sum_df0, los_distances, axis=1)* 10**2# sensitivity to final NO2/NOx ratio
        dcolumns_dt[t0, :] = np.trapezoid(dcontrib_sum_dt, los_distances, axis=1)* 10**2# sensitivity to NO2/NOx conversion rate
        dcolumns_dtau[t0, :] = np.trapezoid(dcontrib_sum_dtau, los_distances, axis=1)* 10**2# sensitivity to NOX lifetime
        dcolumns_dsigma_h0[t0, :] = np.trapezoid(dcontrib_sum_dsigma_h0, los_distances, axis=1)* 10**2# sensitivity to horizontal spread
        dcolumns_dsigma_z0[t0, :] = np.trapezoid(dcontrib_sum_dsigma_z0, los_distances, axis=1)* 10**2# sensitivity to vertical spread
        dcolumns_dF[t0, :] = np.trapezoid(dcontrib_sum_dF, los_distances, axis=1)* 10**2# sensitivity to plume rise buoyancy flux
        dcolumns_dS[t0, :] = np.trapezoid(dcontrib_sum_dS, los_distances, axis=1)* 10**2# sensitivity to plume rise stability parameter
        dcolumns_dbackground[t0, :] = 1 # placeholder for sensitivity to background columns if needed
        # Build a jacobian matrix of the derivatives with respect to parameters for each time and elevation, using the integrated column sensitivities
        jacobian[t0, :, :] = np.stack([dcolumns_dQ0[t0, :], dcolumns_dR[t0, :], dcolumns_df0[t0, :], dcolumns_dt[t0, :], dcolumns_dtau[t0, :], dcolumns_dsigma_h0[t0, :], dcolumns_dsigma_z0[t0, :], dcolumns_dF[t0, :], dcolumns_dS[t0, :], dcolumns_dbackground[t0, :]], axis=1)
        

    # --- Calculate puff centerline in instrument coordinates ---
    # For each emission (t1) and observation time (t0), compute puff center z and distance to LOS plane
    puff_center_z = np.full((nt, nt), np.nan, dtype=float)
    puff_center_cross_distance = np.full((nt, nt), np.nan, dtype=float)
    puff_center_along_distance = np.full((nt, nt), np.nan, dtype=float)
    for t0 in range(nt):
        for t1 in range(0, t0 + 1):
            tau = float(times_seconds[t0] - times_seconds[t1])
            dx = cum_disp_along[t1] - cum_disp_along[t0]
            dy = cum_disp_cross[t1] - cum_disp_cross[t0]
            center_dist = dx + source_distances[t1]
            cross_dist = dy + source_cross[t1]
            z0 = float(emission_height) + float(z_rise_arr[t0 - t1])
            puff_center_z[t0, t1] = z0
            puff_center_cross_distance[t0, t1] = cross_dist
            puff_center_along_distance[t0, t1] = center_dist

    ds = xr.Dataset(
        data_vars={
            'columns': (('time', 'elevation', ), columns),
            'jacobian': (('time', 'elevation', 'parameter'), jacobian),
        },
        coords={
            'time': times,
            'elevation': viewing_elevations,
            'los_distance': los_distances,
            'parameter': ('parameter', ['Q0', 'R', 'f0', 't', 'tau', 'sigma_h0', 'sigma_z0', 'F', 'S', 'background']),
        },
        attrs={
            'description': '2D Gaussian puff model columns along instrument lines of sight',
            'units': 'molecules/cm^2 for columns',
        }
    )
    return ds


def calc_plume_rise_array( nt, times_seconds, u_mean, v_mean, plume_rise_params=None, mode="briggs"):
    if plume_rise_params is None:
        z_rise_arr = np.zeros(nt, dtype=float)
    else:
        if mode == "linear":
            A = float(plume_rise_params.get('A', 0.0))
            T = float(plume_rise_params.get('T', 5.0))
            wind_speed = np.hypot(u_mean, v_mean)
            if T <= 0.0:
                # instantaneous rise to A
                z_rise_arr = np.full(nt , A, dtype=float)
            else:
                # linear ramp from 0 to A over T seconds, then constant A
                frac = np.asarray(times_seconds, dtype=float) / float(T)
                frac_clipped = np.minimum(np.maximum(frac, 0.0), 1.0)
                z_rise_arr = A * frac_clipped
                dz_dA = frac_clipped
                #dz_dT is piecewise: calc numerically
                epsilon = 1e-4
                z_rise_arr_eps_up = A * np.minimum(np.maximum(np.asarray(times_seconds, dtype=float) / float(T+epsilon), 0.0), 1.0)
                z_rise_arr_eps_down = A * np.minimum(np.maximum(np.asarray(times_seconds, dtype=float) / float(T-epsilon), 0.0), 1.0)
                dz_dT = (z_rise_arr_eps_up - z_rise_arr_eps_down) / (2 * epsilon)
                return z_rise_arr, dz_dA, dz_dT
        if mode == "exponential":
            A = float(plume_rise_params.get('A', 0.0))
            T = float(plume_rise_params.get('T', 1.0))
            wind_speed = np.hypot(u_mean, v_mean)
            if T <= 0.0:
                z_rise_arr = np.full(nt, A, dtype=float)
            else:
                # elementwise computation across time
                z_rise_arr = A * (1.0 - np.exp(- times_seconds / T))
                dz_dA = 1.0 - np.exp(- times_seconds / T)
                dz_dT = np.where((times_seconds > 0) & (times_seconds < T), A * times_seconds / T**2 * np.exp(- times_seconds / T), 0.0)
                return z_rise_arr, dz_dA, dz_dT
        if mode == "briggs":
            # Briggs plume rise formula for buoyant plumes in crossflow
            F = plume_rise_params.get('F', 120.0)  # buoyancy flux parameter (m^4/s^3)
            S = plume_rise_params.get('S', 0.0)  # stability parameter (1/s)
            u = np.maximum(np.hypot(u_mean, v_mean), 1e-3)  # wind speed (m/s)
            z_rise_arr = 2.6 * ((F*times_seconds**2/u)/(S*times_seconds**2+4.3))**(1/3)
            dz_dF = z_rise_arr / (3*F)
            dz_dS = - z_rise_arr * times_seconds**2 / (3*(S*times_seconds**2+4.3))
            return z_rise_arr, dz_dF, dz_dS


def calc_NO_NO2_conversion(nt, time_since_emission, Q_NO2_arr, NO_NO2_conversion_params=None):
    if NO_NO2_conversion_params is None:
        return Q_NO2_arr
    else:
        Q0 = float(NO_NO2_conversion_params.get('Q0', 2e22)) # initial NO2 emission rate        
        R = float(NO_NO2_conversion_params.get('R', 1/0.13)) # initial NOx/NO2 ratio
        f = float(NO_NO2_conversion_params.get('f0', 1.32)) # target NOx/NO2 ratio
        t = float(NO_NO2_conversion_params.get('t', 600.0)) # NO2/NOx conversion time in seconds
        tau = float(NO_NO2_conversion_params.get('tau', 3600)) # exponential decay of total NOx emissions

        Q_NO2_arr = Q0 * R * np.exp(-time_since_emission/tau) / (f + np.exp(-time_since_emission/t) * (R -f))
        if Q0 != 0:
            dQdQ0 = Q_NO2_arr / Q0
        else:
            Q_NO2_arr = R * np.exp(-time_since_emission/tau) / (f + np.exp(-time_since_emission/t) * (R -f))
        dQdR = Q_NO2_arr * ( 1 / R - np.exp(-time_since_emission/t) / (f + np.exp(-time_since_emission/t) * (R -f)))
        dQdf = - Q_NO2_arr * ( 1 - np.exp(-time_since_emission/t) ) / (f + np.exp(-time_since_emission/t) * (R -f))
        dQdt = - Q_NO2_arr * ( np.exp(-time_since_emission/t) * (R -f) * time_since_emission/t**2 ) / (f + np.exp(-time_since_emission/t) * (R -f))
        dQdtau = Q_NO2_arr * time_since_emission / tau**2
    return Q_NO2_arr, dQdQ0, dQdR, dQdf, dQdt, dQdtau


def plot_last_timestep(
    ds: xr.Dataset,
    z_level: Optional[float] = None,
    integrate: bool = True,
    cmap: str = 'binary',
    figsize: tuple = (9, 6),
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    instrument_lat: Optional[float] = None,
    instrument_lon: Optional[float] = None,
    instrument_azimuth_deg: float = 180.0,
    instrument_line_len_km: Optional[float] = None,
):
    """Plot the last timestep from the puff-model dataset.

    - If `integrate` is True, plot the column-integrated concentration (sum over `z`).
    - If `integrate` is False, plot the concentration at the nearest `z_level`.

    Returns the (fig, ax) tuple.
    """
    if 'C' not in ds:
        raise ValueError("Dataset must contain variable 'C'")

    C_last = ds['C'].isel(time=-1)  # dims: z, lat, lon

    if integrate:
        arr = C_last.sum(dim='z').values
        title = 'Integrated column concentration (last timestep)'
    else:
        if z_level is None:
            z_level = float(ds['z'].values[0])
        iz = int(np.argmin(np.abs(ds['z'].values - z_level)))
        arr = C_last.isel(z=iz).values
        title = f'Concentration at z={float(ds.z.values[iz]):.1f} m (last timestep)'

    lon = ds['lon'].values
    lat = ds['lat'].values

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(lon, lat, arr, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(title)
    cbar = fig.colorbar(pcm, ax=ax, label=ds.attrs.get('units', ''))
    # plot source and ship markers if available in dataset coords
    try:
        if 'source_lat' in ds.coords and 'source_lon' in ds.coords:
            sx = float(ds['source_lon'].isel(time=-1).values)
            sy = float(ds['source_lat'].isel(time=-1).values)
            ax.plot(sx, sy, 'ro', markersize=6, label='source')
    except Exception:
        pass
    try:
        if 'ship_lat' in ds.coords and 'ship_lon' in ds.coords:
            shx = float(ds['ship_lon'].isel(time=-1).values)
            shy = float(ds['ship_lat'].isel(time=-1).values)
            ax.plot(shx, shy, marker='s', color='blue', markersize=6, label='ship')
    except Exception:
        pass
    # plot instrument marker and azimuth line if provided
    try:
        if instrument_lat is not None and instrument_lon is not None:
            ax.plot(instrument_lon, instrument_lat, marker='^', color='orange', markersize=7, label='instrument')
            # compute line length from grid extents if not provided
            mperlon_inst, mperlat_inst = _meters_per_deg(float(instrument_lat))
            lon_min = float(ds['lon'].values[0])
            lon_max = float(ds['lon'].values[-1])
            lat_min = float(ds['lat'].values[0])
            lat_max = float(ds['lat'].values[-1])
            dx_e = max(abs((lon_max - instrument_lon) * mperlon_inst), abs((lon_min - instrument_lon) * mperlon_inst))
            dy_n = max(abs((lat_max - instrument_lat) * mperlat_inst), abs((lat_min - instrument_lat) * mperlat_inst))
            default_len = max(dx_e, dy_n) * 1.1
            length_m = (instrument_line_len_km * 1000.0) if (instrument_line_len_km is not None) else default_len
            az = np.deg2rad(instrument_azimuth_deg)
            east_m = np.sin(az) * length_m
            north_m = np.cos(az) * length_m
            lon_end = instrument_lon + east_m / mperlon_inst
            lat_end = instrument_lat + north_m / mperlat_inst
            ax.plot([instrument_lon, lon_end], [instrument_lat, lat_end], color='orange', linewidth=1.2, linestyle='-')
    except Exception:
        pass
    # show legend if markers were added
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    # plot puff centerpoints (last timestep) if available
    ax.plot(ds['puff_center_lon'].isel(time = -1), ds['puff_center_lat'].isel(time = -1), color='magenta', linestyle='-', linewidth=1.2, label='plume centerline')

    # add grid lines for reference
    ax.grid(which='both', linestyle='--', linewidth=0.5, color='k', alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    return fig, ax


def plot_lon_integrated(
    ds: xr.Dataset,
    integrate_z: bool = True,
    figsize: tuple = (8, 4),
    cmap: str = 'binary',
    save_path: Optional[str] = None,
    instrument_lat: Optional[float] = None,
    instrument_lon: Optional[float] = None,
    instrument_azimuth_deg: float = 180.0,
):
    """Plot concentration from the last timestep integrated over latitude.

    If `integrate_z` is True, integrates over z as well and produces a 1D
    line (lon vs integrated concentration). If False, produces a pcolormesh
    of lon (x) vs z (y) showing the lat-integrated concentration per height.
    Returns (fig, ax).
    """
    if 'C' not in ds:
        raise ValueError("Dataset must contain variable 'C'")

    C_last = ds['C'].isel(time=-1)  # z, lat, lon
    lon = ds['lon'].values
    z = ds['z'].values

    # integrate over latitude
    arr_z_lon = C_last.sum(dim='lat').values  # shape (z, lon)

    if integrate_z:
        arr_lon = np.nansum(arr_z_lon, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lon, arr_lon, '-o')
        ax.set_xlabel('lon')
        ax.set_ylabel(f'integrated ({ds.attrs.get("units","")})')
        ax.plot(ds['lon'].values, arr_lon, '-o')
        # plot puff centerpoints (last timestep) projected onto lon axis
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
        pcm = ax.pcolormesh(lon, z, arr_z_lon, shading='auto', cmap=cmap)
        ax.plot(ds['puff_center_lon'].isel(time = -1), ds['puff_center_z'].isel(time = -1), color='magenta', linestyle='-', linewidth=1.2, label='plume centerline')
        ax.set_xlabel('lon')
        ax.set_ylabel('z (m)')
        ax.set_title('Last timestep: lat-integrated concentration (lon vs z)')
        cbar = fig.colorbar(pcm, ax=ax, label=ds.attrs.get('units', ''))
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)
        return fig, ax


def plot_lat_integrated(
    ds: xr.Dataset,
    integrate_z: bool = True,
    figsize: tuple = (8, 4),
    cmap: str = 'binary',
    save_path: Optional[str] = None,
    instrument_lat: Optional[float] = None,
    instrument_lon: Optional[float] = None,
    instrument_azimuth_deg: float = 180.0,
):
    """Plot concentration from the last timestep integrated over longitude.

    If `integrate_z` is True, integrates over z as well and produces a 1D
    line (lat vs integrated concentration). If False, produces a pcolormesh
    of lat (x) vs z (y) showing the lon-integrated concentration per height.
    Returns (fig, ax).
    """
    if 'C' not in ds:
        raise ValueError("Dataset must contain variable 'C'")

    C_last = ds['C'].isel(time=-1)  # z, lat, lon
    lat = ds['lat'].values
    z = ds['z'].values

    # integrate over longitude
    arr_z_lat = C_last.sum(dim='lon').values  # shape (z, lat)

    if integrate_z:
        arr_lat = np.nansum(arr_z_lat, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lat, arr_lat, '-o')
        ax.set_xlabel('lat')
        ax.set_ylabel(f'integrated ({ds.attrs.get("units","")})')
        ax.set_title('Last timestep: concentration integrated over longitude and height')
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
        pcm = ax.pcolormesh(lat, z, arr_z_lat, shading='auto', cmap=cmap)
        ax.plot(ds['puff_center_lat'].isel(time = -1), ds['puff_center_z'].isel(time = -1), color='magenta', linestyle='-', linewidth=1.2, label='plume centerline')
        ax.set_xlabel('lat')
        ax.set_ylabel('z (m)')
        ax.set_title('Last timestep: lon-integrated concentration (lat vs z)')
        cbar = fig.colorbar(pcm, ax=ax, label=ds.attrs.get('units', ''))
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)
        return fig, ax


def create_video(
    ds: xr.Dataset,
    out_path: str = 'puff_animation.mp4',
    integrate: bool = True,
    z_level: Optional[float] = None,
    fps: int = 4,
    cmap: str = 'binary',
    figsize: tuple = (8, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_source_marker: bool = True,
    source_lats: Optional[Sequence[float]] = None,
    source_lons: Optional[Sequence[float]] = None,
    instrument_lat: Optional[float] = None,
    instrument_lon: Optional[float] = None,
    instrument_azimuth_deg: float = 180.0,
    instrument_line_len_km: Optional[float] = None,
):
    """Create a video (mp4) from all timesteps in `ds`.

    Notes:
    - Requires `imageio` and a working ffmpeg installation for mp4 encoding.
    - This implementation renders each frame with matplotlib and appends to the writer.
    """
    if 'C' not in ds:
        raise ValueError("Dataset must contain variable 'C'")

    lon = ds['lon'].values
    lat = ds['lat'].values
    nt = ds.sizes['time']

    # determine vmin/vmax if not provided
    if vmin is None or vmax is None:
        if integrate:
            arr_max = float(ds['C'].sum(dim='z').max())
        else:
            if z_level is None:
                zl = ds['z'].values[0]
            else:
                zl = z_level
            iz = int(np.argmin(np.abs(ds['z'].values - zl)))
            arr_max = float(ds['C'].isel(z=iz).max())
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = arr_max/2

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264')

    for t in range(nt):
        if integrate:
            arr = ds['C'].isel(time=t).sum(dim='z').values
            title = f'Integrated column concentration (t={ds.time.values[t]})'
        else:
            if z_level is None:
                z_level = float(ds['z'].values[0])
            iz = int(np.argmin(np.abs(ds['z'].values - z_level)))
            arr = ds['C'].isel(time=t, z=iz).values
            title = f'Concentration at z={float(ds.z.values[iz]):.1f} m (t={ds.time.values[t]})'

        # ensure output image height is divisible by 16 to avoid ffmpeg macro-block resizing
        dpi = float(plt.rcParams.get('figure.dpi', 100))
        # adjust figsize height so that height_pixels = figsize[1] * dpi is multiple of 16
        desired_h_px = int(np.ceil((figsize[1] * dpi) / 16.0) * 16)
        adj_figsize = (figsize[0], float(desired_h_px) / dpi)
        fig, ax = plt.subplots(figsize=adj_figsize)
        pcm = ax.pcolormesh(lon, lat, arr, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title(title)
        cbar = fig.colorbar(pcm, ax=ax, label=ds.attrs.get('units', ''))

        sx = float(source_lons[t])
        sy = float(source_lats[t])
        ax.plot(sx, sy, 'ro', markersize=6, label='source')
        plotted_any = True

        # plot instrument marker and azimuth line 
        ax.plot(instrument_lon, instrument_lat, marker='^', color='orange', markersize=6, label='instrument')
        mperlon_inst, mperlat_inst = _meters_per_deg(float(instrument_lat))
        lon_min = float(lon[0])
        lon_max = float(lon[-1])
        lat_min = float(lat[0])
        lat_max = float(lat[-1])
        dx_e = max(abs((lon_max - instrument_lon) * mperlon_inst), abs((lon_min - instrument_lon) * mperlon_inst))
        dy_n = max(abs((lat_max - instrument_lat) * mperlat_inst), abs((lat_min - instrument_lat) * mperlat_inst))
        default_len = max(dx_e, dy_n) * 1.1
        length_m = (instrument_line_len_km * 1000.0) if (instrument_line_len_km is not None) else default_len
        az = np.deg2rad(instrument_azimuth_deg)
        east_m = np.sin(az) * length_m
        north_m = np.cos(az) * length_m
        lon_end = instrument_lon + east_m / mperlon_inst
        lat_end = instrument_lat + north_m / mperlat_inst
        ax.plot([instrument_lon, lon_end], [instrument_lat, lat_end], color='orange', linewidth=1.0, linestyle='-', label='instrument line of sight')


        ax.plot(ds['puff_center_lon'].isel(time = t), ds['puff_center_lat'].isel(time = t), color='magenta', linestyle='-', linewidth=1.2, label='plume centerline')
        if plotted_any:
            ax.legend()

        # add grid lines for reference
        ax.grid(which='both', linestyle='--', linewidth=0.5, color='k', alpha=0.4)
        plt.tight_layout()

        # render to PNG in-memory and read with imageio (robust across backends)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=int(dpi))
        buf.seek(0)
        try:
            img = imageio.imread(buf)
        finally:
            buf.close()
        writer.append_data(img)
        plt.close(fig)

    writer.close()
    return out_path


def compute_instrument_columns(
    ds: xr.Dataset,
    inst_lat: float,
    inst_lon: float,
    inst_height: float = 10.0,
    elev_angles: Optional[Sequence[float]] = None,
    azimuth_deg: float = 180.0,
    sample_step_m: float = 5.0,
    return_center_positions: bool = False,
    azimuth_tol_deg: float = 5.0,
) -> Union[np.ndarray, tuple]:
    """Compute line-of-sight integrated columns for an instrument at a fixed location.

    Returns an array of shape (time, n_angles) with integrated concentration along each
    viewing elevation angle (degrees above horizon). Azimuth is degrees clockwise from north
    (so 180 -> south).

    Notes:
    - Uses simple equirectangular conversion and trilinear interpolation on the model grid.
    - Sampling stops when the ray leaves the model horizontal extents or drops below z=0.
    """
    if elev_angles is None:
        elev_angles = np.linspace(-2.8, 34.0, 37)
    elev_angles = np.asarray(elev_angles, dtype=float)

    lon = ds['lon'].values
    lat = ds['lat'].values
    z = ds['z'].values
    times = ds['time'].values
    C = ds['C'].values  # shape (time, z, lat, lon)

    nt = ds.sizes['time']
    nz = z.size
    nlat = lat.size
    nlon = lon.size
    nang = elev_angles.size

    # meters per degree at instrument latitude
    mperlon, mperlat = _meters_per_deg(float(inst_lat))

    # estimate maximum range to sample: distance to farthest grid corner
    lon_min = float(lon[0])
    lon_max = float(lon[-1])
    lat_min = float(lat[0])
    lat_max = float(lat[-1])
    dx_east = max(abs((lon_max - inst_lon) * mperlon), abs((lon_min - inst_lon) * mperlon))
    dy_north = max(abs((lat_max - inst_lat) * mperlat), abs((lat_min - inst_lat) * mperlat))
    max_range = np.hypot(dx_east, dy_north)

    s_samples = np.arange(0.0, max_range + sample_step_m, sample_step_m)

    # precompute grid spacings for interpolation
    lon_spacing = float(lon[1] - lon[0])
    lat_spacing = float(lat[1] - lat[0])
    z_spacing = float(z[1] - z[0])

    # prepare output
    columns = np.full((nt, nang), np.nan, dtype=float)

    az_rad = np.deg2rad(azimuth_deg)

    for ti in range(nt):
        for ai, elev in enumerate(elev_angles):
            elev_rad = np.deg2rad(elev)
            # compute ray points in ENU meters
            cos_e = np.cos(elev_rad)
            sin_e = np.sin(elev_rad)

            east_m = s_samples * cos_e * np.sin(az_rad)
            north_m = s_samples * cos_e * np.cos(az_rad)
            z_m = inst_height + s_samples * sin_e

            # convert to lat/lon
            lon_pts = inst_lon + east_m / mperlon
            lat_pts = inst_lat + north_m / mperlat

            vals = []
            valid_s = []
            for s_idx in range(s_samples.size):
                lp = lon_pts[s_idx]
                la = lat_pts[s_idx]
                zz = z_m[s_idx]

                # stop if below ground
                if zz < 0:
                    break
                # stop if outside horizontal bounds
                if lp < lon_min or lp > lon_max or la < lat_min or la > lat_max:
                    break

                # find surrounding indices
                ix = int(np.searchsorted(lon, lp) - 1)
                iy = int(np.searchsorted(lat, la) - 1)
                iz = int(np.searchsorted(z, zz) - 1)

                if ix < 0:
                    ix = 0
                if iy < 0:
                    iy = 0
                if iz < 0:
                    iz = 0
                if ix >= nlon - 1 or iy >= nlat - 1 or iz >= nz - 1:
                    # if on the edge, use nearest-neighbor
                    try:
                        v = C[ti, min(max(iz, 0), nz - 1), min(max(iy, 0), nlat - 1), min(max(ix, 0), nlon - 1)]
                    except Exception:
                        v = 0.0
                    vals.append(v)
                    valid_s.append(s_samples[s_idx])
                    continue

                # fractional distances
                x1 = lon[ix]
                y1 = lat[iy]
                z1 = z[iz]
                xd = (lp - x1) / lon_spacing
                yd = (la - y1) / lat_spacing
                zd = (zz - z1) / z_spacing

                # corner values
                c000 = C[ti, iz, iy, ix]
                c001 = C[ti, iz, iy, ix + 1]
                c010 = C[ti, iz, iy + 1, ix]
                c011 = C[ti, iz, iy + 1, ix + 1]
                c100 = C[ti, iz + 1, iy, ix]
                c101 = C[ti, iz + 1, iy, ix + 1]
                c110 = C[ti, iz + 1, iy + 1, ix]
                c111 = C[ti, iz + 1, iy + 1, ix + 1]

                c00 = c000 * (1 - xd) + c001 * xd
                c01 = c010 * (1 - xd) + c011 * xd
                c10 = c100 * (1 - xd) + c101 * xd
                c11 = c110 * (1 - xd) + c111 * xd

                c0 = c00 * (1 - yd) + c01 * yd
                c1 = c10 * (1 - yd) + c11 * yd

                c = c0 * (1 - zd) + c1 * zd
                vals.append(float(c))
                valid_s.append(s_samples[s_idx])

            if len(vals) == 0:
                columns[ti, ai] = 0.0
            else:
                # integrate along s in meters
                col = float(np.trapezoid(np.asarray(vals), np.asarray(valid_s)))* 10**2
                columns[ti, ai] = col

    # --- compute plume-center viewing geometry relative to instrument ---
    center_info = None


    pclon_all = np.asarray(ds['puff_center_lon'].values)  # shape (time, emission)
    pclat_all = np.asarray(ds['puff_center_lat'].values)
    pclz_all = np.asarray(ds['puff_center_z'].values)

    # intersection results (point where segment between two puff centers crosses instrument azimuth ray)
    intersect_lon = np.full(nt, np.nan, dtype=float)
    intersect_lat = np.full(nt, np.nan, dtype=float)
    intersect_z = np.full(nt, np.nan, dtype=float)
    intersect_range_m = np.full(nt, np.nan, dtype=float)
    intersect_elev_deg = np.full(nt, np.nan, dtype=float)
    intersect_flag = np.full(nt, False, dtype=bool)
    az_rad = np.deg2rad(azimuth_deg)
    for ti in range(nt):
        pclon = pclon_all[ti, :]
        pclat = pclat_all[ti, :]
        valid = np.isfinite(pclon) & np.isfinite(pclat)

        # per-puff east/north and bearings
        east_j = (pclon - inst_lon) * mperlon
        north_j = (pclat - inst_lat) * mperlat
        bearings_j = (np.degrees(np.arctan2(east_j, north_j)) % 360.0)
        ang_diff_j = ((bearings_j - azimuth_deg + 180.0) % 360.0) - 180.0
        # select two puff centers that bracket the instrument azimuth if possible
        idx_pos = np.where((valid) & (ang_diff_j >= 0))[0]
        idx_neg = np.where((valid) & (ang_diff_j <= 0))[0]
        sel_idx = None
        if idx_pos.size > 0 and idx_neg.size > 0:
            # closest on each side
            i_pos = idx_pos[np.argmin(ang_diff_j[idx_pos])]
            i_neg = idx_neg[np.argmax(ang_diff_j[idx_neg])]
            sel_idx = (i_neg, i_pos)
        else:
            # fallback: skip and set nans for this time index
            sel_idx = None
        if sel_idx is not None:
            ia, ib = sel_idx
            # positions in meters relative to instrument
            Ax = east_j[ia]
            Ay = north_j[ia]
            Az = pclz_all[ti, ia] if (pclz_all is not None and np.isfinite(pclz_all[ti, ia])) else np.nan
            Bx = east_j[ib]
            By = north_j[ib]
            Bz = pclz_all[ti, ib] if (pclz_all is not None and np.isfinite(pclz_all[ti, ib])) else np.nan

            # instrument ray unit vector in horizontal plane
            ux = np.sin(az_rad)
            uy = np.cos(az_rad)

            # solve 2x2 linear system: A + t*(B-A) = s * u
            M = np.array([[Bx - Ax, -ux], [By - Ay, -uy]], dtype=float)
            rhs = np.array([-Ax, -Ay], dtype=float)
            det = np.linalg.det(M)
            if abs(det) > 1e-8:
                sol = np.linalg.solve(M, rhs)
                t_sol = float(sol[0])
                s_sol = float(sol[1])
                if (t_sol >= 0.0) and (t_sol <= 1.0) and (s_sol >= 0.0):
                    east_int = s_sol * ux
                    north_int = s_sol * uy
                    lon_int = inst_lon + east_int / mperlon
                    lat_int = inst_lat + north_int / mperlat
                    # interpolate z along segment
                    if np.isfinite(Az) and np.isfinite(Bz):
                        z_int = float(Az + t_sol * (Bz - Az))
                    else:
                        z_int = np.nan
                    intersect_lon[ti] = lon_int
                    intersect_lat[ti] = lat_int
                    intersect_z[ti] = z_int
                    intersect_range_m[ti] = s_sol
                    if np.isfinite(z_int):
                        intersect_elev_deg[ti] = float(np.degrees(np.arctan2((z_int - inst_height), s_sol)))
                    else:
                        intersect_elev_deg[ti] = np.nan
                    intersect_flag[ti] = True

    center_info = {
        'intersect_lon': intersect_lon,
        'intersect_lat': intersect_lat,
        'intersect_z': intersect_z,
        'intersect_range_m': intersect_range_m,
        'intersect_elev_deg': intersect_elev_deg,
        'intersect_flag': intersect_flag,
    }

    if return_center_positions:
        return columns, center_info
    return columns


def compute_instrument_columns_2D(
    ds: xr.Dataset,
    inst_height: float = 10.0,
    elev_angles: Optional[Sequence[float]] = None,
    sample_step_m: float = 5.0,
) -> np.ndarray:
    """
    Compute line-of-sight integrated columns for an instrument at a fixed height for the 2D puff model output.
    Returns an array of shape (time, n_angles) with integrated concentration along each viewing elevation angle (degrees above horizon).
    Also returns center_info dict with the intersection of the plume centerline and instrument line of sight (distance, z, elev) for each time.
    - ds: output from puff_model_2D (dims: time, z, distance)
    - inst_height: instrument height above ground (m)
    - elev_angles: sequence of elevation angles (deg)
    - sample_step_m: step size along the line of sight (m)
    """
    if elev_angles is None:
        elev_angles = np.linspace(-2.8, 34.0, 37)
    elev_angles = np.asarray(elev_angles, dtype=float)


    # 2D model output: integrate along los_distance for each elevation
    # ds dims: (time, elevation, los_distance)
    C = ds['C'].values  # shape (time, elevation, los_distance)
    nt, nang, nlos = C.shape
    columns = np.full((nt, nang), np.nan, dtype=float)
    los_distance = ds['los_distance'].values
    for ti in range(nt):
        for ai in range(nang):
            # integrate along los_distance (axis=-1)
            vals = C[ti, ai, :]
            columns[ti, ai] = np.trapezoid(vals, los_distance)* 10**2

    # --- compute plume centerline intersection with instrument line of sight (distance, z, elev) ---
    # Use puff_center_z and puff_center_distance if present
    center_info = None
    if 'puff_center_z' in ds and 'puff_center_cross_distance' in ds and 'puff_center_along_distance' in ds:
        puff_center_z = np.asarray(ds['puff_center_z'].values)  # (time, emission)
        puff_center_cross_distance = np.asarray(ds['puff_center_cross_distance'].values)
        puff_center_along_distance = np.asarray(ds['puff_center_along_distance'].values)
        intersect_distance = np.full(nt, np.nan, dtype=float)
        intersect_z = np.full(nt, np.nan, dtype=float)
        intersect_elev_deg = np.full(nt, np.nan, dtype=float)
        for ti in range(nt):
            dists = puff_center_cross_distance[ti, :]
            zs = puff_center_z[ti, :]
            valid = np.isfinite(dists) & np.isfinite(zs)
            # Find two points that bracket distance=0 (instrument location)
            idx_pos = np.where((valid) & (dists >= 0))[0]
            idx_neg = np.where((valid) & (dists <= 0))[0]
            sel_idx = None
            if idx_pos.size > 0 and idx_neg.size > 0:
                i_pos = idx_pos[np.argmin(dists[idx_pos])]
                i_neg = idx_neg[np.argmax(dists[idx_neg])]
                sel_idx = (i_neg, i_pos)
            if sel_idx is not None:
                ia, ib = sel_idx
                # Linear interpolation between (dists[ia], zs[ia], dists_along[ia]) and (dists[ib], zs[ib], dists_along[ib])
                dA, zA, dA_along = dists[ia], zs[ia], puff_center_along_distance[ti, ia]
                dB, zB, dB_along = dists[ib], zs[ib], puff_center_along_distance[ti, ib]
                if dB != dA:
                    t = -dA / (dB - dA)
                    z_int = zA + t * (zB - zA)
                    dist_along_int = dA_along + t * (dB_along - dA_along)
                    intersect_distance[ti] = dist_along_int
                    intersect_z[ti] = z_int
                    if np.isfinite(z_int):
                        intersect_elev_deg[ti] = float(np.degrees(np.arctan2((z_int - inst_height), dist_along_int)))
                    else:
                        intersect_elev_deg[ti] = np.nan

        center_info = {
            'intersect_distance': intersect_distance,
            'intersect_z': intersect_z,
            'intersect_elev_deg': intersect_elev_deg,
        }
    return columns, center_info


def create_instrument_video(
    ds: xr.Dataset,
    inst_lat: float,
    inst_lon: float,
    inst_height: float = 10.0,
    elev_angles: Optional[Sequence[float]] = None,
    azimuth_deg: float = 180.0,
    sample_step_m: float = 50.0,
    out_path: str = 'inst_columns.mp4',
    fps: int = 4,
    figsize: tuple = (6, 4),
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Create a video showing instrument column vs elevation for each timestep.

    Each frame is a simple line plot of column (y) vs elevation angle (x).
    """
    cols_res = compute_instrument_columns(ds, inst_lat, inst_lon, inst_height, elev_angles, azimuth_deg, sample_step_m, return_center_positions=True)
    if isinstance(cols_res, tuple):
        cols, center_info = cols_res
    else:
        cols = cols_res
        center_info = None
    nt, nang = cols.shape
    if elev_angles is None:
        elevs = np.linspace(-2.8, 34.0, nang)
    else:
        elevs = np.asarray(elev_angles)

    if vmin is None or vmax is None:
        vmin_calc = float(np.nanmin(cols)) if np.isfinite(np.nanmin(cols)) else 0.0
        vmax_calc = float(np.nanmax(cols)) if np.isfinite(np.nanmax(cols)) else 1.0
        if vmin is None:
            vmin = vmin_calc
        if vmax is None:
            vmax = vmax_calc

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264')

    dpi = float(plt.rcParams.get('figure.dpi', 100))
    desired_h_px = int(np.ceil((figsize[1] * dpi) / 16.0) * 16)
    adj_figsize = (figsize[0], float(desired_h_px) / dpi)

    for t in range(nt):
        fig, ax = plt.subplots(figsize=adj_figsize)
        y = cols[t, :]
        ax.plot(elevs, y, '-o', color='tab:orange')
        ax.fill_between(elevs, y, 0.0, color='tab:orange', alpha=0.3)
        ax.set_xlabel('elevation (deg)')
        ax.set_ylabel('column (concentration·m)')
        ax.set_ylim(vmin, vmax)
        ax.set_title(f'Instrument columns (t={ds.time.values[t]})')
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=int(dpi))
        buf.seek(0)
        try:
            img = imageio.imread(buf)
        finally:
            buf.close()
        writer.append_data(img)
        plt.close(fig)

    writer.close()
    return out_path


def plot_instrument_colormap(
    ds: xr.Dataset,
    inst_lat: float,
    inst_lon: float,
    inst_height: float = 10.0,
    elev_angles: Optional[Sequence[float]] = None,
    azimuth_deg: float = 180.0,
    sample_step_m: float = 5.0,
    out_path: Optional[str] = 'inst_columns.png',
    cmap: str = 'binary',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Create and save a colormap plot (time x elevation) of instrument columns.

    - x axis: time (uses datetimes if present)
    - y axis: elevation angles
    """

    # Handle output from puff_model_2D_with_derivatives (columns, jacobian)
    if 'columns' in ds:
        # Use the columns variable directly
        cols = ds['columns'].values  # shape (time, elevation)
        times = ds['time'].values
        elevs = ds['elevation'].values
        nt, nang = cols.shape
        # No center_info for this output
        center_info = None
    elif set(ds['C'].dims) == {'time', 'elevation', 'los_distance'}:
        # 2D model output: dims (time, elevation, los_distance)
        cols, center_info = compute_instrument_columns_2D(ds, inst_height=inst_height, elev_angles=elev_angles, sample_step_m=sample_step_m)
        times = ds['time'].values
        elevs = ds['elevation'].values if elev_angles is None else np.asarray(elev_angles)
        nt, nang = cols.shape
    else:
        # 3D model output
        cols, center_info = compute_instrument_columns(ds, inst_lat, inst_lon, inst_height, elev_angles, azimuth_deg, sample_step_m, return_center_positions=True)
        times = ds['time'].values
        elevs = np.linspace(-2.8, 34.0, cols.shape[1]) if elev_angles is None else np.asarray(elev_angles)
        nt, nang = cols.shape

    # prepare x values for pcolormesh
    if np.issubdtype(times.dtype, np.datetime64):
        x_vals = mdates.date2num(times.astype('datetime64[ms]').astype('O'))
    else:
        x_vals = np.arange(nt)

    # compute vmin/vmax if needed
    if vmin is None:
        vmin = float(np.nanmin(cols)) if np.isfinite(np.nanmin(cols)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(cols)) if np.isfinite(np.nanmax(cols)) else 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(x_vals, elevs, cols.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, label=f"column ({ds.attrs.get('units','')})")
    ax.set_ylabel('VEA / °')

    # overlay plume center viewing elevation (instrument perspective) if available (only for 3D)
    if center_info is not None:
        c_elev = center_info.get('intersect_elev_deg', None)
        if c_elev is not None:
            mask = np.where(c_elev <= max(elevs))[0]
            ax.plot(x_vals[mask], c_elev[mask], color='red', linewidth=1.2, label='plume center elev')
    if np.issubdtype(times.dtype, np.datetime64):
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('time (UTC)')
    else:
        ax.set_xlabel('time index')
    ax.grid(False)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
    plt.show()
    plt.close(fig)






#%%

if __name__ == '__main__':
    # consolidated example: create dataset and plot last timestep

    import time

    ds_plume = xr.open_dataset(r"P:\data\SEICOR\plumes_2\plumes_250412\plume_032_t_20250412_164820_mmsi_247389200.nc")
    funnel_height = float(ds_plume.attrs['funnel_height_m'])
    lats = ds_plume['ship_ais_lats'].values
    lons = ds_plume['ship_ais_lons'].values
    ais_times = ds_plume['ship_ais_times'].values
    ais_times = ds_plume['ship_ais_times'].values
    t_funnel = pd.to_datetime(ds_plume.attrs['t_funnel'])
    t_ais = pd.to_datetime(ds_plume.attrs['t']).tz_localize(None)
    idx_closest_ais = np.argmin(np.abs(ais_times - np.datetime64(t_ais)))
    idx_closest_funnel = np.argmin(np.abs(ais_times - np.datetime64(t_funnel)))
    idx_offset = idx_closest_ais - idx_closest_funnel
    #select ais times +/-5min around t_funnel and interpolate time and lats and lons to 1s resolution
    time_mask = (ais_times >= (t_ais - np.timedelta64(1, 'm'))) & (ais_times <= (t_ais + np.timedelta64(3, 'm')))
    # produce a second mask shifted by idx_offset so selections can be aligned/compared
    time_mask_shifted = np.zeros_like(time_mask, dtype=bool)
    try:
        if idx_offset >= 0:
            time_mask_shifted[idx_offset:] = time_mask[: time_mask.size - idx_offset]
        else:
            off = int(abs(idx_offset))
            time_mask_shifted[: time_mask.size - off] = time_mask[off:]
    except Exception:
        # fallback: if idx_offset not usable, keep shifted mask identical
        time_mask_shifted = time_mask.copy()

    ais_times_sel = ais_times[time_mask]
    # corrected ship positions
    ais_lats_sel_shifted = lats[time_mask_shifted]
    ais_lons_sel_shifted = lons[time_mask_shifted]
    ais_times_interp = np.arange(ais_times_sel[0], ais_times_sel[-1] + np.timedelta64(2, 's'), np.timedelta64(2, 's'))
    ais_lats_interp = np.interp(ais_times_interp.astype('int64'), ais_times_sel.astype('int64'), ais_lats_sel_shifted)
    ais_lons_interp = np.interp(ais_times_interp.astype('int64'), ais_times_sel.astype('int64'), ais_lons_sel_shifted)
    #calculate u, v wind from ds_plume["wind_dir_insitu"] and ds_plume["wind_speed_insitu"]
    wind_dir = ds_plume["wind_dir_insitu"].values
    wind_speed = ds_plume["wind_speed_insitu"].values
    #convert wind_dir from degrees to radians
    wind_dir_rad = np.deg2rad(wind_dir)
    u_wind = wind_speed * np.sin(wind_dir_rad)
    v_wind = wind_speed * np.cos(wind_dir_rad)
    #interpolate wind 
    times_insitu = ds_plume['insitu_times'].values
    u_wind_interp = np.interp(ais_times_interp.astype('int64'), times_insitu.astype('int64'), u_wind)
    v_wind_interp = np.interp(ais_times_interp.astype('int64'), times_insitu.astype('int64'), v_wind)
    inst_lat = 53.56958522848946
    inst_lon = 9.69174249821205
    inst_height = 8.0
    elevs = ds_plume.vea.values
    azi = ds_plume.vaa.values[0]

    #print('Running 3D puff model...')
    ##start timer
    #start_time = time.time()
    #ds = puff_model_3D(
    #    source_lats=ais_lats_interp,
    #    source_lons=ais_lons_interp,
    #    times=ais_times_interp,
    #    u=u_wind_interp,
    #    v=v_wind_interp,
    #    sigma_h=300.0,
    #    sigma_z=80.0,
    #    emission_rate=1.0,
    #    emission_height=funnel_height,
    #    plume_rise_model='briggs',
    #    plume_rise_params={'A': 250.0, 'T': 120.0, 'F': 240.0, 'S': 0.0},
    #)
    #ds.to_netcdf('puff_example.nc')
    #end_time = time.time()
    #print(f'3D puff model completed in {end_time - start_time:.2f} seconds')
    #start_time = time.time()
    #plot_instrument_colormap(ds, inst_lat, inst_lon, inst_height, elevs, out_path='inst_columns.png', azimuth_deg=azi, sample_step_m=2.0)
    #end_time = time.time()
    #print(f'Instrument column colormap completed in {end_time - start_time:.2f} seconds')

    print('Running 2D puff model...')
    start_time = time.time()
    ds_2D = puff_model_2D_with_derivatives(
        source_lats=ais_lats_interp,
        source_lons=ais_lons_interp,
        times=ais_times_interp,
        u=u_wind_interp,
        v=v_wind_interp,
        instrument_lat=inst_lat,
        instrument_lon=inst_lon,
        instrument_altitude=inst_height,
        sigma_h=300.0,
        emission_rate=1.0,
        emission_height=funnel_height,
        plume_rise_model='briggs',
        plume_rise_params={'A': 250.0, 'T': 120.0, 'F': 240.0, 'S': 10**(-6)},
        NO_NO2_conversion_params={'R': 0.13, 't': 400.0, 'Q0': 3*10**22},
        azimuth_deg=azi,
        viewing_elevations=elevs,
    )

    ds_2D.to_netcdf('puff_example_2D.nc')
    end_time = time.time()
    print(f'2D puff model completed in {end_time - start_time:.2f} seconds')
    print('Plotting instrument colormap...')
    start_time = time.time()
    plot_instrument_colormap(ds_2D, inst_lat, inst_lon, inst_height, elevs, out_path='inst_columns.png', azimuth_deg=azi, sample_step_m=2.0)
    end_time = time.time()
    print(f'Instrument column colormap for 2D model completed in {end_time - start_time:.2f} seconds')
    
    ds_model_interp = ds_2D.interp(time=ds_plume['times_plume'])

    mask = SEICOR.plumes.detect_plume_ztest(
    ds_plume["no2_enhancement_interp"].values,
    p_threshold=0.20,
    min_cluster_size=5,
    connectivity=1,
    kernel_arm=1,
    require_connection=True,
    ds_plume=ds_plume,
    keep_second_largest=False,
    second_size_threshold=100,)

    # example plot of the last timestep
    #try:
    #    plot_last_timestep(ds, integrate=True, instrument_lat=inst_lat, instrument_lon=inst_lon, instrument_azimuth_deg=azi, save_path='last_timestep.png')
    #    plot_lon_integrated(ds=ds, integrate_z=False, save_path='lon_integrated.png')
    #    plot_lat_integrated(ds=ds, integrate_z=False, save_path='lat_integrated.png')
    #    create_video(
    #        ds,
    #        out_path='puff_animation.mp4', 
    #        source_lats=ais_lats_interp,
    #        source_lons=ais_lons_interp,
    #        instrument_lat=inst_lat, instrument_lon=inst_lon, instrument_azimuth_deg=azi)
    #
    #    plot_instrument_colormap(ds, inst_lat, inst_lon, inst_height, elevs, out_path='inst_columns.png', azimuth_deg=azi, sample_step_m=2.0)
    #
    #except Exception as e:
    #    print('Plot example failed:', e)



## %%
#ds_plume = xr.open_dataset(r"P:\data\SEICOR\plumes_2\plumes_250412\plume_032_t_20250412_164820_mmsi_247389200.nc")
#ds_sel = ds_plume.isel(image_row=slice(15, None))
#a = ds_sel.image_row * ds_sel["no2_enhancement_interp"].where(ds_sel["no2_enhancement_interp"] > 0)
#a.plot()
#centerline_elev = a.sum(dim='image_row')/ds_sel["no2_enhancement_interp"].where(ds_sel["no2_enhancement_interp"] > 0).sum(dim='image_row')
## %%
#plt.imshow(ds_plume["no2_enhancement_interp"], aspect='auto', origin='lower')
#plt.plot(centerline_elev, color='red', linewidth=2.0)
## %%

# %%
