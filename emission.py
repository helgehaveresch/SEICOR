#%%
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.append(str(Path(r"C:\Users\hhave\Documents\Promotion\scripts")))
import SEICOR.plumes
from SEICOR.wind import compute_relative_wind

from scipy.signal import savgol_filter
from SEICOR.puff_model import _meters_per_deg

INST_LAT = 53.56958522848946
INST_LON = 9.69174249821205


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lon1r, lat2r, lon2r = np.deg2rad([lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    return R * (2.0 * np.arcsin(np.sqrt(a)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    # bearing (towards) degrees clockwise from North
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlonr = np.deg2rad(lon2 - lon1)
    x = np.sin(dlonr) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlonr)
    brad = np.arctan2(x, y)
    bdeg = (np.degrees(brad) + 360.0) % 360.0
    return bdeg


def compute_plume_centerline(ds_plume, times, inst_lat=INST_LAT, inst_lon=INST_LON,
                             dt_model_s=2.0, smooth_window_s=60.0, smooth_polyorder=2):
    """
    Plume centerline in the x,y-plane (east/north in m relative to the instrument) from the smoothed AIS track
    and the in-situ wind (as in puff_model_2D), and the horizontal distance of the plume along the line of sight.

    The AIS track is shifted in time by (t_funnel - t_ais) so that the ship position at t_ais is placed at t_funnel.
    Returns a dict with the model time grid, smoothed track, centerline[t_obs, t_emission] and the plume
    distance/age on the model and on the measurement time grid (NaN where the centerline does not cross the LOS).
    """
    dt_model = np.timedelta64(int(dt_model_s * 1e9), 'ns')
    azi = float(ds_plume.vaa[ds_plume.window_plume.values[0]].values)
    az_rad = np.deg2rad(azi)
    az_perp = az_rad + np.pi / 2
    mperlon, mperlat = _meters_per_deg(inst_lat)

    # AIS selection around t_ais, shifted in time by (t_funnel - t_ais) so that the ship position at t_ais
    # is placed exactly at t_funnel (the ship position at time t is the AIS position at t - dt_shift)
    ais_lats = np.asarray(ds_plume['ship_ais_lats'].values, dtype=float)
    ais_lons = np.asarray(ds_plume['ship_ais_lons'].values, dtype=float)
    ais_times = ds_plume['ship_ais_times'].values
    t_ais = pd.to_datetime(ds_plume.attrs['t']).tz_localize(None)
    t_funnel_attr = pd.to_datetime(ds_plume.attrs['t_funnel'])
    dt_shift = np.timedelta64(t_funnel_attr - t_ais)
    ais_times_shifted = ais_times + dt_shift
    time_mask = (ais_times_shifted >= (t_ais - np.timedelta64(2, 'm'))) & (ais_times_shifted <= (t_ais + np.timedelta64(4, 'm')))
    ais_times_sel = ais_times_shifted[time_mask]
    ais_x_sel = (ais_lons[time_mask] - inst_lon) * mperlon  # east (m) relative to instrument
    ais_y_sel = (ais_lats[time_mask] - inst_lat) * mperlat  # north (m) relative to instrument
    # AIS contains duplicate timestamps -> drop them before interpolation
    ais_times_sel, idx_unique = np.unique(ais_times_sel, return_index=True)
    ais_x_sel = ais_x_sel[idx_unique]
    ais_y_sel = ais_y_sel[idx_unique]

    model_times = np.arange(ais_times_sel[0], ais_times_sel[-1] + dt_model, dt_model)
    nt_model = len(model_times)
    src_x = np.interp(model_times.astype('int64'), ais_times_sel.astype('int64'), ais_x_sel)
    src_y = np.interp(model_times.astype('int64'), ais_times_sel.astype('int64'), ais_y_sel)

    # smooth the interpolated track (AIS positions are jumpy at ~10 s reporting interval)
    win = int(round(smooth_window_s / dt_model_s))
    win = win + 1 if win % 2 == 0 else win
    win = min(win, nt_model - (1 - nt_model % 2))
    if win > smooth_polyorder:
        src_x = savgol_filter(src_x, win, smooth_polyorder, mode='interp')
        src_y = savgol_filter(src_y, win, smooth_polyorder, mode='interp')

    # wind (same convention as puff_model.py: u = ws*sin(dir), v = ws*cos(dir), displacement applied as cum[t1]-cum[t0])
    wind_dir_rad = np.deg2rad(ds_plume['wind_dir_insitu'].values)
    wind_speed_ins = ds_plume['wind_speed_insitu'].values
    times_insitu = ds_plume['insitu_times'].values
    u_wind = np.interp(model_times.astype('int64'), times_insitu.astype('int64'), wind_speed_ins * np.sin(wind_dir_rad))
    v_wind = np.interp(model_times.astype('int64'), times_insitu.astype('int64'), wind_speed_ins * np.cos(wind_dir_rad))

    model_seconds = (model_times - model_times[0]) / np.timedelta64(1, 's')
    dt_steps = np.empty(nt_model, dtype=float)
    dt_steps[:-1] = np.diff(model_seconds)
    dt_steps[-1] = dt_steps[-2]
    cum_x = np.concatenate(([0.0], np.cumsum(u_wind[:-1] * dt_steps[:-1])))
    cum_y = np.concatenate(([0.0], np.cumsum(v_wind[:-1] * dt_steps[:-1])))

    # centerline[t0, t1]: position at observation time t0 of the puff emitted at t1 (<= t0)
    centerline_x = src_x[None, :] + (cum_x[None, :] - cum_x[:, None])
    centerline_y = src_y[None, :] + (cum_y[None, :] - cum_y[:, None])
    not_emitted = np.triu(np.ones((nt_model, nt_model), dtype=bool), k=1)
    centerline_x[not_emitted] = np.nan
    centerline_y[not_emitted] = np.nan

    # project onto the viewing azimuth (along) and its perpendicular (cross)
    centerline_along = centerline_x * np.sin(az_rad) + centerline_y * np.cos(az_rad)
    centerline_cross = centerline_x * np.sin(az_perp) + centerline_y * np.cos(az_perp)

    # horizontal distance of the plume: where the centerline crosses the viewing plane (cross = 0).
    # If several crossings exist, the one of the most recently emitted puff is used.
    plume_dist_model = np.full(nt_model, np.nan)
    plume_age_model = np.full(nt_model, np.nan)
    for t0 in range(1, nt_model):
        c = centerline_cross[t0, :t0 + 1]
        a = centerline_along[t0, :t0 + 1]
        idx_cross = np.where(np.sign(c[:-1]) * np.sign(c[1:]) <= 0)[0]
        if idx_cross.size == 0:
            continue
        i = idx_cross[-1]
        frac = c[i] / (c[i] - c[i + 1]) if c[i] != c[i + 1] else 0.0
        plume_dist_model[t0] = a[i] + frac * (a[i + 1] - a[i])
        t_emit = model_seconds[i] + frac * (model_seconds[i + 1] - model_seconds[i])
        plume_age_model[t0] = model_seconds[t0] - t_emit

    # interpolate to the measurement time grid
    times_ns = pd.DatetimeIndex(times).values.astype('int64')
    valid_model = np.isfinite(plume_dist_model)
    plume_dist_meas = np.full(len(times_ns), np.nan)
    plume_age_meas = np.full(len(times_ns), np.nan)
    if np.any(valid_model):
        plume_dist_meas = np.interp(times_ns, model_times.astype('int64')[valid_model], plume_dist_model[valid_model],
                                    left=np.nan, right=np.nan)
        plume_age_meas = np.interp(times_ns, model_times.astype('int64')[valid_model], plume_age_model[valid_model],
                                   left=np.nan, right=np.nan)

    return {
        'azi': azi, 'az_rad': az_rad, 'az_perp': az_perp,
        't_ais': t_ais, 't_funnel_attr': t_funnel_attr, 'dt_shift': dt_shift,
        'ais_times_sel': ais_times_sel, 'ais_x_sel': ais_x_sel, 'ais_y_sel': ais_y_sel,
        'model_times': model_times, 'src_x': src_x, 'src_y': src_y, 'u_wind': u_wind, 'v_wind': v_wind,
        'centerline_x': centerline_x, 'centerline_y': centerline_y,
        'plume_dist_model': plume_dist_model, 'plume_age_model': plume_age_model,
        'plume_dist_meas': plume_dist_meas, 'plume_age_meas': plume_age_meas,
    }


def compute_cross_sectional_flux(ds_plume, arr, mask_arr, times, vea, t_funnel, use_plume_distance=True,
                                 z0=1e-4, **centerline_kwargs):
    """
    Cross-sectional NO2 flux through the vertical plane of the line of sight.

    1. plume centerline and horizontal plume distance d(t) along the LOS (compute_plume_centerline)
    2. length element per VEA perpendicular to the LOS at the plume: L = d / cos(alpha),
       dl = L * (tan(dalpha_up) + tan(dalpha_low))
    3. logarithmic wind profile at the VEA heights, corrected to the apparent wind seen from the ship
       (ship speed/course at t_funnel), of which only the component perpendicular to the LOS is used
    4. flux grid = SC * dl * |u_perp| * 1e4 (cm^-2 -> m^-2), summed over the plume mask -> flux time series

    arr, mask_arr: (vea x time) slant columns [molec cm^-2] and plume mask.
    use_plume_distance=False uses the constant ship distance attribute instead of the plume distance.
    centerline_kwargs are passed to compute_plume_centerline.
    Returns (ds_plume with the intermediate variables added, result dict).
    """
    times = pd.DatetimeIndex(times)
    vea = np.asarray(vea, dtype=float)

    # --- 1. plume centerline and horizontal plume distance ---
    cl = compute_plume_centerline(ds_plume, times, **centerline_kwargs)
    azi = cl['azi']
    plume_dist_meas = cl['plume_dist_meas']
    ds_plume = ds_plume.assign(plume_horizontal_distance_m=(['window_plume'], plume_dist_meas))
    ds_plume = ds_plume.assign(plume_age_s=(['window_plume'], cl['plume_age_meas']))

    try:
        ship_dist_m = float(ds_plume.attrs.get('ship_distance_to_instrument_m', None))
    except Exception:
        ship_dist_m = np.nan

    # distance per measurement time; where the centerline does not cross the LOS, hold the nearest valid
    # value (fallback: ship distance attribute)
    if use_plume_distance and np.any(np.isfinite(plume_dist_meas)):
        plume_dist_flux = pd.Series(plume_dist_meas).interpolate(limit_direction='both').values
        plume_dist_flux[~np.isfinite(plume_dist_flux)] = ship_dist_m
    else:
        plume_dist_flux = np.full(len(times), ship_dist_m)
    ds_plume = ds_plume.assign(plume_distance_flux_m=(['window_plume'], plume_dist_flux))

    # --- 2. heights and length elements (vea x time) ---
    vea_rad = np.deg2rad(vea)
    vea_height_m = np.tan(vea_rad)[:, None] * plume_dist_flux[None, :]

    # length element for each VEA, perpendicular to the LOS at the plume distance:
    # L = d / cos(alpha) is the slant path to the plume; the element spans from the midpoint to the
    # lower neighbouring row to the midpoint to the upper one: dl = L * (tan(dalpha_up) + tan(dalpha_low))
    # (= 2 L tan(dalpha/2) for equally spaced rows); edge rows use the spacing to their single neighbour
    vea_dh_m = np.full_like(vea_height_m, np.nan, dtype=float)
    if len(vea) >= 2:
        dvea = np.abs(np.diff(vea_rad))
        dalpha_up = 0.5 * np.concatenate((dvea, dvea[-1:]))
        dalpha_low = 0.5 * np.concatenate((dvea[:1], dvea))
        los_path_m = plume_dist_flux[None, :] / np.cos(vea_rad)[:, None]
        vea_dh_m = los_path_m * (np.tan(dalpha_up) + np.tan(dalpha_low))[:, None]

    ds_plume = ds_plume.assign(vea_height_m=(['image_row', 'window_plume'], vea_height_m))
    ds_plume = ds_plume.assign(vea_dh_m=(['image_row', 'window_plume'], vea_dh_m))

    # --- 3. wind profile ---
    # reference sensor height (m) if present in attrs, otherwise default to 10 m
    z_ref = float(ds_plume.attrs.get('wind_sensor_height_m', 10.0))

    # reference wind speed aligned to `times` (nearest in-situ value)
    u_ref_ts = None
    if 'wind_speed_insitu' in ds_plume:
        uvar = ds_plume['wind_speed_insitu']
        try:
            if 'insitu_times' in uvar.dims:
                ins_times = pd.to_datetime(ds_plume['insitu_times'].values)
                u_series = pd.Series(uvar.values, index=ins_times)
                u_ref_ts = u_series.reindex(times, method='nearest').values.astype(float)
            else:
                vals = np.asarray(uvar.values, dtype=float)
                u_ref_ts = vals if vals.size == len(times) else np.full(len(times), float(np.nanmean(vals)))
        except Exception:
            u_ref_ts = None
    if u_ref_ts is None:
        try:
            u_ref_ts = np.full(len(times), float(ds_plume.attrs.get('wind_speed_insitu', np.nan)))
        except Exception:
            u_ref_ts = np.full(len(times), np.nan)

    # u(z,t) = u_ref(t) * ln(z/z0)/ln(z_ref/z0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_denom = np.log(np.maximum(z_ref, z0) / z0)
        ratio = np.log(np.maximum(vea_height_m, z0) / z0) / ln_denom
        wind_profile = ratio * u_ref_ts[None, :]
    # mask unrealistic values where vea_h <= z0
    wind_profile[np.isnan(ratio)] = np.nan

    # ship speed at funnel from consecutive AIS positions, interpolated to the measurement times
    t_funnel = pd.to_datetime(t_funnel)
    idx_near = int(np.nanargmin(np.abs((times - t_funnel).total_seconds())))
    ship_times = pd.to_datetime(ds_plume['ship_ais_times'].values)
    ship_lats = np.asarray(ds_plume['ship_ais_lats'].values, dtype=float)
    ship_lons = np.asarray(ds_plume['ship_ais_lons'].values, dtype=float)
    dists = np.array([_haversine_m(ship_lats[i], ship_lons[i], ship_lats[i+1], ship_lons[i+1]) for i in range(len(ship_times)-1)])
    dt = np.diff(ship_times.values.astype('datetime64[s]')).astype(float)
    dt[dt == 0] = np.nan  # avoid division by zero
    seg_speeds = dists / dt
    speeds = np.append(seg_speeds, seg_speeds[-1])
    ship_secs = ship_times.astype('int64') / 1e9
    order = np.argsort(ship_secs)
    ship_speed_ts = np.interp(times.astype('int64') / 1e9, ship_secs[order], speeds[order])
    ship_speed_at_funnel = float(ship_speed_ts[idx_near])

    # ship course at funnel from the AIS segment at the nearest AIS index
    idx_ais = int(np.nanargmin(np.abs((ship_times - t_funnel).total_seconds())))
    if idx_ais < len(ship_times) - 1:
        ship_course_at_funnel = _bearing_deg(ship_lats[idx_ais], ship_lons[idx_ais], ship_lats[idx_ais+1], ship_lons[idx_ais+1])
    else:
        ship_course_at_funnel = _bearing_deg(ship_lats[idx_ais-1], ship_lons[idx_ais-1], ship_lats[idx_ais], ship_lons[idx_ais])

    # wind direction at funnel from the nearest in-situ value
    ins_times = pd.to_datetime(ds_plume['insitu_times'].values)
    idx_ins = int(np.nanargmin(np.abs((ins_times - t_funnel).total_seconds())))
    wind_dir_at_funnel = float(ds_plume['wind_dir_insitu'].values[idx_ins])

    # apparent wind seen from the ship; only the component perpendicular to the line of sight
    # transports NO2 through the measurement plane
    apparent_speed, apparent_dir = compute_relative_wind(ship_speed_at_funnel, ship_course_at_funnel, wind_profile,
                                                         wind_dir_at_funnel * np.ones_like(wind_profile))
    wind_perp = np.abs(apparent_speed * np.sin(np.deg2rad(apparent_dir - azi)))
    perp_fraction = float(np.abs(np.sin(np.deg2rad(np.nanmedian(apparent_dir) - azi))))
    print(f'apparent wind at funnel from {np.nanmedian(apparent_dir):.1f} deg, LOS azimuth {azi:.1f} deg '
          f'-> perpendicular fraction {perp_fraction:.2f}')

    ds_plume = ds_plume.assign(apparent_wind_speed_m_s=(['image_row', 'window_plume'], apparent_speed))
    ds_plume = ds_plume.assign(apparent_wind_dir_deg=(['image_row', 'window_plume'], apparent_dir))
    ds_plume = ds_plume.assign(wind_profile_m_s=(['image_row', 'window_plume'], wind_perp))

    # --- 4. flux: slant column * length element * perpendicular wind, summed over the plume mask ---
    # convert slant columns from 1/cm^2 to 1/m^2 by multiplying 1e4
    flux_grid = arr * vea_dh_m * wind_perp * 1e4
    flux_grid_masked = np.where(mask_arr, flux_grid, np.nan)
    flux_ts = np.nansum(np.where(mask_arr, flux_grid, 0.0), axis=0)

    ds_plume = ds_plume.assign(flux_grid=(['image_row', 'window_plume'], flux_grid_masked))
    ds_plume = ds_plume.assign(flux_transport=(['window_plume'], flux_ts))

    result = dict(cl)
    result.update({
        'ship_dist_m': ship_dist_m, 'plume_dist_flux': plume_dist_flux,
        'vea_height_m': vea_height_m, 'vea_dh_m': vea_dh_m,
        'ship_speed_at_funnel': ship_speed_at_funnel, 'ship_course_at_funnel': ship_course_at_funnel,
        'wind_dir_at_funnel': wind_dir_at_funnel,
        'apparent_wind_speed': apparent_speed, 'apparent_wind_dir': apparent_dir,
        'wind_perp': wind_perp, 'perp_fraction': perp_fraction,
        'flux_grid': flux_grid_masked, 'flux_ts': flux_ts,
    })
    return ds_plume, result


def fit_flux_and_convert_nox(times, flux_ts, t_funnel, t_fit_end=None, deg=3, no2_nox_ratio=0.138):
    """
    Fit a polynomial (default cubic) to the NO2 flux time series for t_funnel < t < t_fit_end, extrapolate it
    to t_funnel (1-sigma uncertainty from the fit covariance) and convert the NO2 flux to NOx
    (NO2/NOx = no2_nox_ratio at the funnel) in molec/s and g/s (NO part with the NO molar mass).
    Returns a dict, or None if there are not enough points to fit.
    """
    times = pd.DatetimeIndex(times)
    flux_ts = np.asarray(flux_ts, dtype=float)
    t_funnel = pd.to_datetime(t_funnel)
    if t_fit_end is not None:
        t_fit_end = pd.to_datetime(t_fit_end)
        if not (times.min() < t_fit_end <= times.max()):
            print(f'Warning: t_fit_end {t_fit_end} is outside the measurement time range; fitting until the end')
            t_fit_end = None

    mask_time = np.asarray(times > t_funnel)
    if t_fit_end is not None:
        mask_time &= np.asarray(times < t_fit_end)
    mask_time &= np.isfinite(flux_ts)
    if mask_time.sum() <= deg + 1:
        print('Not enough points after t_funnel to fit flux polynomial')
        return None

    t0 = times.min()
    x_seconds = np.asarray((times - t0).total_seconds())
    coeffs, cov = np.polyfit(x_seconds[mask_time], flux_ts[mask_time], deg=deg, cov=True)
    poly = np.poly1d(coeffs)

    x_funnel = (t_funnel - t0).total_seconds()
    J = x_funnel ** np.arange(deg, -1, -1)
    var_funnel = float(J @ cov @ J)
    no2_at_funnel = float(poly(x_funnel))
    no2_at_funnel_std = float(np.sqrt(var_funnel)) if var_funnel >= 0 else np.nan

    # NO2 -> NOx (molec/s) and grams/s
    NA = 6.02214076e23
    M_NO2 = 46.0055
    M_NO = 30.0061
    nox_molec = no2_at_funnel / no2_nox_ratio
    no_molec = nox_molec - no2_at_funnel
    g_per_no2_molec = (M_NO2 + (1.0 / no2_nox_ratio - 1.0) * M_NO) / NA  # NOx grams per NO2 molecule
    nox_g = no2_at_funnel * g_per_no2_molec
    nox_g_std = no2_at_funnel_std * g_per_no2_molec
    print(f'NO2 flux @funnel = {no2_at_funnel:.6g} ± {no2_at_funnel_std:.6g} molec/s; '
          f'NOx = {nox_molec:.6g} molec/s -> {nox_g:.6g} ± {nox_g_std:.3g} g/s')

    return {
        'mask_time': mask_time, 't_fit_end': t_fit_end, 'coeffs': coeffs, 'cov': cov,
        'fit_ts': poly(x_seconds),
        'no2_at_funnel': no2_at_funnel, 'no2_at_funnel_std': no2_at_funnel_std,
        'nox_at_funnel_molec': nox_molec, 'no_at_funnel_molec': no_molec,
        'nox_at_funnel_g': nox_g, 'nox_at_funnel_g_std': nox_g_std,
    }

#%%
#timedelta funnel 
dt_funnel = pd.Timedelta(seconds=2)
t_fit_end = pd.to_datetime("2025-06-15T05:16:15")#None #
# Font sizes
LABEL_FS = 16
TICK_FS = 17
TITLE_FS = 18
CB_FS = 17
#file_path = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_004_t_20250615_051423_mmsi_211676580.nc"
file_path = r"p:\data\SEICOR\plumes_2\plumes_250428\plume_024_t_20250428_112317_mmsi_563068900.nc"
ds_plume = xr.open_dataset(file_path)
# %%
a = ds_plume.no2_ref-ds_plume.no2_ref.mean(dim="window_ref")

mask = SEICOR.plumes.detect_plume_ztest(
        ds_plume["no2_enhancement_interp"].values,
        bg_std=a.std(),
        bg_mean=a.mean(),
        p_threshold=0.001,
        min_cluster_size=5,
        connectivity=1,
        kernel_arm=1,
        require_connection=True,
        ds_plume=ds_plume,
        keep_second_largest=False,
        second_size_threshold=100,
    )          

a = ds_plume.no2_ref-ds_plume.no2_ref.mean(dim="window_ref")

#mask = SEICOR.plumes.detect_plume_ztest(
#        ds_plume["no2_enhancement_c_back"].values,
#        bg_std=a.std(),
#        bg_mean=a.mean(),
#        p_threshold=0.001,
#        min_cluster_size=5,
#        connectivity=1,
#        kernel_arm=1,
#        require_connection=True,
#        ds_plume=ds_plume,
#        keep_second_largest=False,
#        second_size_threshold=100,
#    )    
ship_mask = SEICOR.plumes.detect_plume_ztest_left(
        ds_plume["no2_enhancement_c_back"].values, 
        p_threshold=0.15, 
        min_cluster_size=20)

reference_image_row_ship = np.array(ds_plume.image_row[37:42])
reference_image_row_plume = np.array(ds_plume.image_row[37:42])

# --- Column-wise reference subtraction (x-dimension correction) ---
# For each x-column where plume OR ship is detected, compute the mean over the
# reference VEA/image_row band and subtract it from the full column.
arr = np.asarray(ds_plume["no2_enhancement_c_back"].values, dtype=float)
plume_mask = np.asarray(mask, dtype=bool)
ship_mask = np.asarray(ship_mask, dtype=bool)

# Align masks to array shape
if plume_mask.shape != arr.shape and plume_mask.T.shape == arr.shape:
    plume_mask = plume_mask.T
if ship_mask.shape != arr.shape and ship_mask.T.shape == arr.shape:
    ship_mask = ship_mask.T

mask_union = plume_mask | ship_mask if (plume_mask.shape == arr.shape and ship_mask.shape == arr.shape) else None

image_row_vals = np.asarray(ds_plume["image_row"].values)
ref_idx_ship = np.where(np.isin(image_row_vals, reference_image_row_ship))[0]
ref_idx_plume = np.where(np.isin(image_row_vals, reference_image_row_plume))[0]

arr_refcorr = arr.copy()
if mask_union is not None and arr_refcorr.ndim == 2:
    for j in range(arr_refcorr.shape[1]):
        has_plume = bool(np.any(plume_mask[:, j])) if plume_mask.shape == arr_refcorr.shape else False
        has_ship = bool(np.any(ship_mask[:, j])) if ship_mask.shape == arr_refcorr.shape else False
        if not (has_plume or has_ship):
            continue

        # Reference selection:
        # - If both plume and ship are present in this column, use BOTH reference bands.
        # - Otherwise, use the reference band corresponding to what is present.
        if has_plume and has_ship:
            ref_idx = np.union1d(ref_idx_plume, ref_idx_ship)
        else:
            ref_idx = ref_idx_plume if has_plume else ref_idx_ship
        if ref_idx.size == 0:
            continue
        ref_vals = arr_refcorr[ref_idx, j]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if ref_vals.size == 0:
            continue
        offset = float(np.nanmean(ref_vals))
        if np.isfinite(offset):
            arr_refcorr[:, j] -= offset

    ds_plume["no2_enhancement_c_back_refcorr"] = (ds_plume["no2_enhancement_c_back"].dims, arr_refcorr)
else:
    print("Reference subtraction skipped: mask/array shapes did not align")

ds_plume = ds_plume.assign(plume_mask=(["image_row", "window_plume"], mask.astype(bool)))



arr =  ds_plume['no2_enhancement_c_back_refcorr'].values  # limit to central VEA range for better visualization
times = pd.to_datetime(ds_plume['times_plume'].values)
vea = np.asarray(ds_plume['vea'].values)

if arr.ndim != 2:
    raise ValueError('expected 2D array for NO2')
if arr.shape[0] == len(times) and arr.shape[1] == len(vea):
    arr = arr.T

# --- plume mask in the same orientation as `arr` (vea x time) ---
mask_arr = np.asarray(ds_plume['plume_mask'].values, dtype=bool)
if mask_arr.shape != arr.shape:
    if mask_arr.T.shape == arr.shape:
        mask_arr = mask_arr.T
    else:
        raise ValueError('plume mask and data array shapes do not match: '
                         f'{mask_arr.shape} vs {arr.shape}')

t_funnel = pd.to_datetime(ds_plume.attrs['t_funnel']) + dt_funnel

# --- cross-sectional flux: plume centerline -> length elements -> apparent wind -> flux ---
ds_plume, flux_res = compute_cross_sectional_flux(ds_plume, arr, mask_arr, times, vea, t_funnel,
                                                  use_plume_distance=True)
flux_ts = flux_res['flux_ts']

# edges for pcolormesh
xnum = mdates.date2num(times.to_pydatetime())
if len(xnum) >= 2:
    xedges = np.empty(xnum.size + 1, dtype=float)
    xedges[0] = xnum[0]
    xedges[1:-1] = 0.5 * (xnum[:-1] + xnum[1:])
    xedges[-1] = xnum[-1]
else:
    xedges = np.array([xnum[0], xnum[0]], dtype=float)
dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
yedges = np.concatenate((vea - dy/2.0, [vea[-1] + dy/2.0]))

# Combined figure: NO2 image (top) + flux time series (bottom), shared time axis
fig, (ax, axf) = plt.subplots(
    2,
    1,
    figsize=(15, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [1.5, 1.0]}
)
# set color limits from the underlying ds_plume['no2_extended'] (same slice used above)
try:
    no2_slice = ds_plume['no2_enhancement_c_back_refcorr'].values
    vmin = float(np.nanmin(no2_slice))
    vmax = float(np.nanmax(no2_slice))
except Exception:
    vmin, vmax = None, None

norm = None
if vmin is not None and vmax is not None and np.isfinite(vmin) and np.isfinite(vmax) and (vmin < 0.0) and (vmax > 0.0):
    vlim = float(max(abs(vmin), abs(vmax)))
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

if norm is None:
    pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', vmin=vmin, vmax=vmax)
else:
    pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', norm=norm)
xcent = (xedges[:-1] + xedges[1:]) / 2.0
ycent = (yedges[:-1] + yedges[1:]) / 2.0
Xc, Yc = np.meshgrid(xcent, ycent)
ax.contour(Xc, Yc, mask.astype(int), levels=[0.5], colors="red", linewidths=1.5)

# ship mask contour (blue)
#try:
#    ship_mask_plot = np.asarray(ship_mask, dtype=bool)
#    if ship_mask_plot.shape != arr.shape and ship_mask_plot.T.shape == arr.shape:
#        ship_mask_plot = ship_mask_plot.T
#    if ship_mask_plot.shape == arr.shape:
#        ax.contour(Xc, Yc, ship_mask_plot.astype(int), levels=[0.5], colors="blue", linewidths=1.3, linestyles=':')
#except Exception:
#    pass
# fewer x-ticks (AutoDateLocator with limited ticks)
ax.xaxis_date()
duration = times.max() - times.min()
hours = duration.total_seconds() / 3600.0
locator = mdates.MinuteLocator(interval=2)
locator = mdates.MinuteLocator(interval=1)
time_fmt = '%H:%M'
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax.tick_params(axis='x', labelbottom=False)


ax.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, )
t_fit_end = pd.to_datetime(t_fit_end)
if t_fit_end is not None and t_fit_end > times.min() and t_fit_end < times.max():
    ax.axvline(t_fit_end, color='C3', linestyle='--', linewidth=1.5,)


# VEA ticks: choose up to 6 ticks and format with one decimal
n_yticks = min(6, len(vea))
idxs = np.unique(np.round(np.linspace(0, len(vea)-1, n_yticks)).astype(int))
yticks = vea[idxs]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{v:.1f}°" for v in yticks])
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.grid(True, which='major', linewidth=1.1, alpha=0.8)

#ax.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
ax.set_ylabel('VEA / °', fontsize=LABEL_FS)
# set title using filename and optional timestamp/mmsi from attributes
try:
    tattr = ds_plume.attrs.get('t', None)
    mmsi = ds_plume.attrs.get('mmsi', None)
    if tattr is not None:
        tstr = pd.to_datetime(tattr).strftime('%Y-%m-%d %H:%M:%S')
        title = f"{tstr}, mmsi: {mmsi}"
except Exception:
    title = Path(file_path).name
ax.set_title('', fontsize=TITLE_FS)

# Centered figure title
#fig.suptitle("NO$_2$ Flux Measurement of a Cargo Ship", fontsize=TITLE_FS, x=0.5, y=0.98, ha='center')
fig.subplots_adjust(top=0.94, hspace=0.06)
fig.canvas.draw()

# Vertical colorbar next to the color plot.
# Shrink both subplots so their plotting-box widths match.
cbar_fmt = ScalarFormatter(useMathText=True)
cbar_fmt.set_powerlimits((-3, 3))

ax_pos = ax.get_position()
axf_pos = axf.get_position()
cbar_w = 0.018
# Space we reserve on the right side so the plots and colorbar fit.
cbar_gap = 0.01
# Extra shift to the right (in figure coordinates) to move the colorbar further right
# without shrinking the plots further.
cbar_shift_right = 0.01

delta = cbar_w + cbar_gap

# Shrink both subplots equally so their widths match
ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width - delta, ax_pos.height])
axf.set_position([axf_pos.x0, axf_pos.y0, axf_pos.width - delta, axf_pos.height])

# Place the colorbar based on the *original* axis right edge, then shift right.
# (Using the post-shrink position would cancel out the padding effect.)
cax_x0 = ax_pos.x1 - cbar_w + cbar_shift_right
# Keep colorbar inside the figure canvas
cax_x0 = min(cax_x0, 0.99 - cbar_w)
cax = fig.add_axes([cax_x0, ax_pos.y0, cbar_w, ax_pos.height])
cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', format=cbar_fmt)

plt.draw()
offset_text = ''
try:
    offset_text = cbar.ax.yaxis.get_offset_text().get_text()
except Exception:
    offset_text = ''

label = 'NO$_2$ Enhancement'
if offset_text:
    offset_clean = offset_text.replace('\\times', '').replace('\u00d7', '').strip()
    cbar.set_label(f"{label} /\n{offset_clean} " + f"#molec.$\,$cm$^{{-2}}$", fontsize=CB_FS)
    cbar.ax.yaxis.get_offset_text().set_visible(False)
else:
    cbar.set_label(label, fontsize=CB_FS)

cbar.ax.tick_params(labelsize=TICK_FS)

# Scale for plotting: show order-of-magnitude in the y-label
flux_ts_finite = np.asarray(flux_ts, dtype=float)
flux_ts_finite = flux_ts_finite[np.isfinite(flux_ts_finite)]
if flux_ts_finite.size == 0:
    flux_exp = 0
else:
    maxabs = float(np.max(np.abs(flux_ts_finite)))
    flux_exp = int(np.floor(np.log10(maxabs))) if maxabs > 0 else 0
flux_scale = 10.0 ** flux_exp
if not np.isfinite(flux_scale) or flux_scale == 0:
    flux_scale = 1.0
    flux_exp = 0

flux_ts_plot = flux_ts / flux_scale

# Plot flux time series on the shared-x subplot
axf.plot(times, flux_ts_plot, marker='o', label='vertically integrated flux')

# Fit cubic to flux_ts for t_funnel < t < t_fit_end, extrapolate to t_funnel and convert to NOx
fit_res = fit_flux_and_convert_nox(times, flux_ts, t_funnel, t_fit_end=t_fit_end, deg=3, no2_nox_ratio=0.138)
if fit_res is not None:
    mask_time = fit_res['mask_time']
    axf.plot(times[mask_time], fit_res['fit_ts'][mask_time] / flux_scale, color='red', label='poly3 fit')
    axf.errorbar(
        [pd.to_datetime(t_funnel)],
        [fit_res['no2_at_funnel'] / flux_scale],
        yerr=[fit_res['no2_at_funnel_std'] / flux_scale],
        fmt='s',
        color='C2',
        label='NO$_2$ flux @funnel'
    )

    # store fit and NOx results
    ds_plume.attrs['flux_poly3_coeffs'] = [float(c) for c in fit_res['coeffs']]
    ds_plume.attrs['flux_poly3_coeffs_cov'] = fit_res['cov'].tolist()
    ds_plume.attrs['flux_poly3_fit_at_funnel'] = fit_res['no2_at_funnel']
    ds_plume.attrs['flux_poly3_fit_at_funnel_std'] = fit_res['no2_at_funnel_std']
    ds_plume.attrs['nox_at_funnel_molec'] = fit_res['nox_at_funnel_molec']
    ds_plume.attrs['no_at_funnel_molec'] = fit_res['no_at_funnel_molec']
    ds_plume.attrs['nox_at_funnel_g'] = fit_res['nox_at_funnel_g']
    ds_plume.attrs['nox_at_funnel_g_std'] = fit_res['nox_at_funnel_g_std']
    # fitted flux as variable, zero in columns without plume
    flux_poly3_fit = np.where(mask_arr.sum(axis=0) > 0, fit_res['fit_ts'], 0.0)
    ds_plume = ds_plume.assign(flux_poly3_fit=(['window_plume'], flux_poly3_fit))

axf.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, label='t_funnel')
if t_fit_end is not None and t_fit_end > times.min() and t_fit_end < times.max():
    axf.axvline(t_fit_end, color='C3', linestyle='--', linewidth=1.5, label='t_fit_end')
axf.set_ylabel(f"NO$_2$ Flux /\n$10^{{{flux_exp}}}$ #molec.$\,$s$^{{-1}}$", fontsize=LABEL_FS)
axf.tick_params(axis='both', labelsize=TICK_FS)
axf.grid(True, which='major', linewidth=1.1, alpha=0.8)
# Shared time axis formatting + remove padding so the first/last points touch plot edges
axf.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
axf.xaxis.set_major_locator(locator)
axf.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax.set_xlim(pd.to_datetime(times.min()), pd.to_datetime(times.max()))
ax.margins(x=0)
axf.margins(x=0)
axf.legend(fontsize=14)
#rotate x-ticks of axf 
#axf.xaxis.set_tick_params(rotation=45)
fig.savefig(r"C:\Users\hhave\Nextcloud_neu\Promotion\Other\IMPACT_cargo_ship_cross_sectional_flux.pdf")

#plt.show()


"""
# Create a simple time series DataArray and plot it
ts = xr.DataArray(integrated_plume, coords={"time": times}, dims=["time"])

fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(times, integrated_plume, marker='o')
ax2.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
ax2.set_ylabel('Vertically integrated NO$_2$ (integral over VEA)', fontsize=LABEL_FS)
ax2.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
fig2.autofmt_xdate(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Fit 2nd-order polynomial to integrated series for t > t_funnel ---
try:


    mask_time = (times > t_funnel ) & (times < t_fit_end)
    if np.any(mask_time):
        # convert times to seconds since first time for numerical stability
        t0 = pd.to_datetime(times.min())
        x_seconds = (pd.to_datetime(times[mask_time]) - t0).total_seconds()
        y = integrated_plume[mask_time]
        if len(x_seconds) < 3:
            print('Not enough points after t_funnel to fit a 2nd-order polynomial')
        else:
            # obtain covariance matrix for coefficient uncertainties
            coeffs, cov = np.polyfit(x_seconds, y, deg=3, cov=True)
            p = np.poly1d(coeffs)

            # evaluate fit only on times >= t_funnel for plotting
            times_mask = times[mask_time]
            x_mask = (pd.to_datetime(times_mask) - t0).total_seconds()
            fitted_mask = p(x_mask)

            # compute fitted value and uncertainty at t_funnel
            x_funnel = (pd.to_datetime(t_funnel) - t0).total_seconds()
            J = np.array([x_funnel**3, x_funnel**2, x_funnel, 1.0])
            try:
                var_funnel = float(J @ cov @ J.T)
                std_funnel = float(np.sqrt(var_funnel)) if var_funnel >= 0 else np.nan
            except Exception:
                var_funnel = np.nan
                std_funnel = np.nan
            fitted_at_funnel = float(p(x_funnel))

            # plot overlay on the existing figure (create new if closed)
            fig2, ax2 = plt.subplots(figsize=(12,4))
            ax2.plot(times, integrated_plume, marker='o', label='integrated')
            ax2.plot(times_mask, fitted_mask, label='poly3 fit', color='red')
            # show fitted value at funnel with errorbar
            ax2.errorbar([pd.to_datetime(t_funnel)], [fitted_at_funnel], yerr=[std_funnel], fmt='s', color='C2', label='fit@funnel')
            ax2.axvline(t_funnel, color='k', linestyle='--', label='t_funnel')
            ax2.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
            ax2.set_ylabel('Vertically integrated NO$_2$', fontsize=LABEL_FS)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
            ax2.legend()
            fig2.autofmt_xdate(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            # store coefficients, funnel fit and uncertainties in dataset attributes
            try:
                ds_plume.attrs['poly3_coeffs'] = [float(c) for c in coeffs]
                ds_plume.attrs['poly3_coeffs_cov'] = cov.tolist()
                ds_plume.attrs['t_funnel_used'] = str(pd.to_datetime(t_funnel))
                ds_plume.attrs['poly3_fit_at_funnel'] = float(fitted_at_funnel)
                ds_plume.attrs['poly3_fit_at_funnel_std'] = float(std_funnel)
            except Exception:
                pass
            # print summary
            print(f'poly3 fit at funnel: {fitted_at_funnel:.6g} ± {std_funnel:.6g}')
    else:
        print('No times greater than t_funnel; skipping polynomial fit')
except Exception as e:
    print('Polynomial fit failed:', e)

# Optionally add the integrated result back into the dataset for later use
try:
    ds_plume = ds_plume.assign(integrated_plume=(['window_plume'], integrated_plume))
except Exception:
    # If dimension names don't match, skip assigning but keep `ts` available
    pass
"""

# %%
# --- plot: centerline map (left) + horizontal plume distance over time (right) ---
model_times = flux_res['model_times']
centerline_x, centerline_y = flux_res['centerline_x'], flux_res['centerline_y']
src_x, src_y = flux_res['src_x'], flux_res['src_y']
ais_x_sel, ais_y_sel, ais_times_sel = flux_res['ais_x_sel'], flux_res['ais_y_sel'], flux_res['ais_times_sel']
plume_dist_model, plume_dist_meas = flux_res['plume_dist_model'], flux_res['plume_dist_meas']
plume_dist_flux, ship_dist_m = flux_res['plume_dist_flux'], flux_res['ship_dist_m']
azi, az_rad = flux_res['azi'], flux_res['az_rad']
valid_model = np.isfinite(plume_dist_model)

fig_c, (ax_map, ax_d) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.6]})

meas_start, meas_end = np.datetime64(times.min()), np.datetime64(times.max())
t0_snapshots = np.where((model_times >= meas_start) & (model_times <= meas_end))[0]
t0_snapshots = t0_snapshots[np.linspace(0, t0_snapshots.size - 1, 5).astype(int)]
cmap_snap = plt.get_cmap('viridis')
for k, t0 in enumerate(t0_snapshots):
    ax_map.plot(centerline_x[t0, :t0 + 1], centerline_y[t0, :t0 + 1], color=cmap_snap(k / max(len(t0_snapshots) - 1, 1)),
                label=pd.to_datetime(model_times[t0]).strftime('%H:%M:%S'))
ax_map.plot(ais_x_sel, ais_y_sel, 'k.', ms=4, label='AIS (shifted)')
ax_map.plot(src_x, src_y, 'k-', lw=1, alpha=0.6, label='AIS smoothed')
los_len = 1.5 * np.nanmax(np.abs(plume_dist_model)) if np.any(valid_model) else 1000.0
ax_map.plot([0, los_len * np.sin(az_rad)], [0, los_len * np.cos(az_rad)], 'r--', lw=1.5, label=f'LOS (az={azi:.1f}°)')
ax_map.plot(0, 0, 'r^', ms=10, label='instrument')
ax_map.set_aspect('equal', adjustable='datalim')
ax_map.set_xlabel('East / m', fontsize=LABEL_FS)
ax_map.set_ylabel('North / m', fontsize=LABEL_FS)
ax_map.set_title('Plume centerline', fontsize=TITLE_FS)
ax_map.tick_params(axis='both', labelsize=TICK_FS - 3)
ax_map.grid(True, alpha=0.6)
ax_map.legend(fontsize=10)

ax_d.plot(times, plume_dist_flux, '-', color='0.6', lw=3, label='distance used for flux')
ax_d.plot(times, plume_dist_meas, 'o-', ms=3, label='plume centerline in LOS')
ax_d.axhline(ship_dist_m, color='C1', linestyle=':', linewidth=1.5, label='ship distance (attr)')
ax_d.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, label='t_funnel')
ax_d.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
ax_d.set_ylabel('Horizontal distance / m', fontsize=LABEL_FS)
ax_d.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
ax_d.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
ax_d.set_xlim(times.min(), times.max())
ax_d.tick_params(axis='both', labelsize=TICK_FS - 3)
ax_d.grid(True, alpha=0.6)
ax_d.legend(fontsize=12)
fig_c.tight_layout()
plt.show()

# %%
# --- AIS track in the x,y-plane with instrument and line of sight ---
inst_lat, inst_lon = INST_LAT, INST_LON
mperlon, mperlat = _meters_per_deg(inst_lat)
ais_lats = np.asarray(ds_plume['ship_ais_lats'].values, dtype=float)
ais_lons = np.asarray(ds_plume['ship_ais_lons'].values, dtype=float)
ais_times = ds_plume['ship_ais_times'].values
t_ais, t_funnel_attr_dt = flux_res['t_ais'], flux_res['t_funnel_attr']
u_wind, v_wind = flux_res['u_wind'], flux_res['v_wind']

ais_x_all = (ais_lons - inst_lon) * mperlon
ais_y_all = (ais_lats - inst_lat) * mperlat
ais_secs_sel = (ais_times_sel - np.datetime64(t_ais)) / np.timedelta64(1, 's')

fig_t, ax_t = plt.subplots(figsize=(10, 8))
ax_t.plot(ais_x_all, ais_y_all, '.-', color='0.7', ms=3, lw=0.8, label='AIS full (raw)')
sc = ax_t.scatter(ais_x_sel, ais_y_sel, c=ais_secs_sel, cmap='viridis', s=18, zorder=3, label='AIS selected (shifted)')
ax_t.plot(src_x, src_y, 'k-', lw=1.2, label='AIS smoothed')
# ship at t_ais (raw AIS) and at t_funnel (smoothed, shifted track)
ais_times_u, idx_u = np.unique(ais_times, return_index=True)
t_ais_ns = np.datetime64(t_ais).astype('datetime64[ns]').astype('int64')
x_a = np.interp(t_ais_ns, ais_times_u.astype('int64'), ais_x_all[idx_u])
y_a = np.interp(t_ais_ns, ais_times_u.astype('int64'), ais_y_all[idx_u])
ax_t.plot(x_a, y_a, 'o', mfc='none', mec='C1', ms=12, mew=2, label='ship @ t_ais (raw)')
x_f = np.interp(np.datetime64(t_funnel_attr_dt).astype('datetime64[ns]').astype('int64'), model_times.astype('int64'), src_x)
y_f = np.interp(np.datetime64(t_funnel_attr_dt).astype('datetime64[ns]').astype('int64'), model_times.astype('int64'), src_y)
ax_t.plot(x_f, y_f, 's', mfc='none', mec='C3', ms=12, mew=2, label='ship @ t_funnel (smoothed)')
# instrument and line of sight
los_len_t = 2.0 * ship_dist_m if np.isfinite(ship_dist_m) else 1000.0
ax_t.plot([0, los_len_t * np.sin(az_rad)], [0, los_len_t * np.cos(az_rad)], 'r--', lw=1.5, label=f'LOS (az={azi:.1f}°)')
ax_t.plot(0, 0, 'r^', ms=12, label='instrument')
# wind arrow (direction the air moves to, same convention as the centerline calculation)
u_to, v_to = -np.nanmean(u_wind), -np.nanmean(v_wind)
ws_mean = np.hypot(u_to, v_to)
if ws_mean > 0:
    arrow_len = 0.25 * los_len_t
    ax_t.annotate('', xy=(arrow_len * u_to / ws_mean, arrow_len * v_to / ws_mean), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color='C0', lw=2))
    ax_t.text(arrow_len * u_to / ws_mean, arrow_len * v_to / ws_mean, f' wind {ws_mean:.1f} m/s', color='C0', fontsize=12)

cb_t = fig_t.colorbar(sc, ax=ax_t)
cb_t.set_label('Time relative to t_ais / s', fontsize=LABEL_FS - 2)
ax_t.set_aspect('equal', adjustable='datalim')
ax_t.set_xlim(np.nanmin(ais_x_sel) - 200, np.nanmax(ais_x_sel) + 200)
ax_t.set_ylim(min(np.nanmin(ais_y_sel), los_len_t * np.cos(az_rad)) - 200, max(np.nanmax(ais_y_sel), 0) + 200)
ax_t.set_xlabel('East / m', fontsize=LABEL_FS)
ax_t.set_ylabel('North / m', fontsize=LABEL_FS)
ax_t.set_title(f'AIS track, mmsi {ds_plume.attrs.get("mmsi", "")}', fontsize=TITLE_FS)
ax_t.grid(True, alpha=0.6)
ax_t.legend(fontsize=10, loc='best')
fig_t.tight_layout()
plt.show()

ds_plume.close()

# %%
