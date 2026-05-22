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
#%%
#timedelta funnel 
dt_funnel = pd.Timedelta(seconds=2)
t_fit_end = pd.to_datetime("2025-06-15T05:16:15")#None #
# Font sizes
LABEL_FS = 16
TICK_FS = 17
TITLE_FS = 18
CB_FS = 17
file_path = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_004_t_20250615_051423_mmsi_211676580.nc"
#file_path = r"p:\data\SEICOR\plumes_2\plumes_250428\plume_024_t_20250428_112317_mmsi_563068900.nc"
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
reference_image_row_plume = np.array(ds_plume.image_row[8:15])

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

# --- compute heights for each VEA using ship distance in dataset attributes ---
ship_dist_attr = ds_plume.attrs.get('ship_distance_to_instrument_m', None)
try:
    ship_dist_m = float(ship_dist_attr)
except Exception:
    ship_dist_m = np.nan

vea_rad = np.deg2rad(vea)
# heights for each VEA at the reported ship distance (1D array, meters)
vea_height_m = np.tan(vea_rad) * ship_dist_m

# length element for each VEA: centered difference of heights
vea_dh_m = np.full_like(vea_height_m, np.nan, dtype=float)
if len(vea_height_m) >= 2:
    if len(vea_height_m) > 2:
        vea_dh_m[1:-1] = 0.5 * (vea_height_m[2:] - vea_height_m[:-2])
    vea_dh_m[0] = vea_height_m[1] - vea_height_m[0]
    vea_dh_m[-1] = vea_height_m[-1] - vea_height_m[-2]

# try to add to dataset as 1D variables on the 'vea' coordinate
try:
    ds_plume = ds_plume.assign(vea_height_m=(['vea'], vea_height_m))
    ds_plume = ds_plume.assign(vea_dh_m=(['vea'], vea_dh_m))
except Exception:
    # fallback: store shapes and scalar distance as attrs
    ds_plume.attrs['vea_height_shape'] = getattr(vea_height_m, 'shape', None)
    ds_plume.attrs['vea_dh_shape'] = getattr(vea_dh_m, 'shape', None)


# --- scale wind to all heights using logarithmic profile (z0=1e-4) ---
z0 = 1e-4
# reference sensor height (m) if present in attrs, otherwise default to 10 m
z_ref = float(ds_plume.attrs.get('wind_sensor_height_m', 10.0))

# obtain reference wind speed as time series aligned to `times`
u_ref_ts = None
if 'wind_speed_insitu' in ds_plume:
    uvar = ds_plume['wind_speed_insitu']
    try:
        # if wind is on insitu_times, interpolate/reindex to `times`
        if 'insitu_times' in uvar.dims:
            ins_times = pd.to_datetime(ds_plume['insitu_times'].values)
            u_series = pd.Series(uvar.values, index=ins_times)
            try:
                u_aligned = u_series.reindex(pd.DatetimeIndex(times), method='nearest')
            except Exception:
                # fallback to nearest using numpy interp on numeric seconds
                u_aligned = u_series.reindex(pd.DatetimeIndex(times), method='nearest')
            u_ref_ts = u_aligned.values.astype(float)
        else:
            # if already aligned to times/window_plume
            vals = np.asarray(uvar.values)
            if vals.size == len(times):
                u_ref_ts = vals.astype(float)
            else:
                # reduce or expand as needed via nearest
                try:
                    u_ref_ts = np.full(len(times), float(np.nanmean(vals)))
                except Exception:
                    u_ref_ts = None
    except Exception:
        u_ref_ts = None

if u_ref_ts is None:
    # fallback: try attrs
    try:
        u_ref_ts = np.full(len(times), float(ds_plume.attrs.get('wind_speed_insitu', np.nan)))
    except Exception:
        u_ref_ts = np.full(len(times), np.nan)

# compute wind profile: for each vea height (1D) and time, u(z,t) = u_ref(t) * ln(z/z0)/ln(z_ref/z0)
vea_h = vea_height_m
with np.errstate(divide='ignore', invalid='ignore'):
    ln_denom = np.log(np.maximum(z_ref, z0) / z0)
    ratio = np.log(np.maximum(vea_h, z0) / z0) / ln_denom

    # If u_ref_ts is 1D length times, broadcast
    try:
        u_ref_ts = np.asarray(u_ref_ts, dtype=float)
        if u_ref_ts.ndim == 1 and u_ref_ts.size == len(times):
            wind_profile = ratio[:, None] * u_ref_ts[None, :]
        else:
            # scalar or unmatched shape
            u_scalar = float(np.nanmean(u_ref_ts))
            wind_profile = ratio[:, None] * u_scalar
    except Exception:
        wind_profile = np.full((len(vea_h), len(times)), np.nan)

# mask unrealistic values where vea_h <= z0
wind_profile[np.isnan(ratio), :] = np.nan

ds_plume = ds_plume.assign(wind_profile_m_s=(['image_row','window_plume'], wind_profile))



# --- correct wind profile by ship speed computed from AIS positions ---
def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lon1r, lat2r, lon2r = np.deg2rad([lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    return R * (2.0 * np.arcsin(np.sqrt(a)))


if 'ship_ais_times' in ds_plume and 'ship_ais_lats' in ds_plume and 'ship_ais_lons' in ds_plume:
    ship_times = pd.to_datetime(ds_plume['ship_ais_times'].values)
    ship_lats = np.asarray(ds_plume['ship_ais_lats'].values, dtype=float)
    ship_lons = np.asarray(ds_plume['ship_ais_lons'].values, dtype=float)

    if len(ship_times) >= 2:
        # compute speeds between consecutive AIS points
        dists = np.array([_haversine_m(ship_lats[i], ship_lons[i], ship_lats[i+1], ship_lons[i+1]) for i in range(len(ship_times)-1)])
        dt = np.diff(ship_times.astype('datetime64[s]')).astype(float)
        # avoid division by zero
        dt[dt == 0] = np.nan
        seg_speeds = dists / dt
        # assign speed per AIS timestamp (use forward/backward fill)
        speeds = np.empty(len(ship_times), dtype=float)
        speeds[:-1] = seg_speeds
        speeds[-1] = seg_speeds[-1]
        speed_series = pd.Series(speeds, index=pd.to_datetime(ship_times))
        ship_secs = speed_series.index.astype('int64') / 1e9
        times_secs = pd.DatetimeIndex(times).astype('int64') / 1e9
        ship_vals = speed_series.values.astype(float)
        # ensure monotonic times for np.interp
        order = np.argsort(ship_secs)
        ship_speed_ts = np.interp(times_secs, ship_secs[order], ship_vals[order])





# if we have AIS info, compute ship speed/course at t_funnel and compute relative wind there
def _bearing_deg(lat1, lon1, lat2, lon2):
    # bearing (towards) degrees clockwise from North
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlonr = np.deg2rad(lon2 - lon1)
    x = np.sin(dlonr) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlonr)
    brad = np.arctan2(x, y)
    bdeg = (np.degrees(brad) + 360.0) % 360.0
    return bdeg

t_funnel_attr = ds_plume.attrs.get('t_funnel', None)

t_funnel = pd.to_datetime(t_funnel_attr)+dt_funnel
times_dt = pd.to_datetime(times)
idx_near = int(np.nanargmin(np.abs((times_dt - t_funnel).total_seconds())))

# ship speed at funnel: prefer interpolated series, else try to compute from AIS
ship_speed_at_funnel = np.nan
if ship_speed_ts is not None:
    sarr = np.asarray(ship_speed_ts, dtype=float)
    if sarr.size == len(times):
        ship_speed_at_funnel = float(sarr[idx_near])


# try AIS-derived course at nearest AIS index
ship_course_at_funnel = None

if 'ship_ais_times' in ds_plume and 'ship_ais_lats' in ds_plume and 'ship_ais_lons' in ds_plume:
    ship_times = pd.to_datetime(ds_plume['ship_ais_times'].values)
    ship_lats = np.asarray(ds_plume['ship_ais_lats'].values, dtype=float)
    ship_lons = np.asarray(ds_plume['ship_ais_lons'].values, dtype=float)
    # find nearest AIS index
    if len(ship_times) >= 2:
        diffs = np.abs((ship_times - np.datetime64(t_funnel)) / np.timedelta64(1, 's'))
        idx_ais = int(np.nanargmin(diffs))
        if idx_ais < len(ship_times)-1:
            ship_course_at_funnel = _bearing_deg(ship_lats[idx_ais], ship_lons[idx_ais], ship_lats[idx_ais+1], ship_lons[idx_ais+1])
        else:
            # use previous segment
            ship_course_at_funnel = _bearing_deg(ship_lats[idx_ais-1], ship_lons[idx_ais-1], ship_lats[idx_ais], ship_lons[idx_ais])


# wind speed and direction at funnel from insitu (if present)
wind_speed_at_funnel = np.nan
wind_dir_at_funnel = np.nan
try:
    if 'wind_speed_insitu' in ds_plume and 'insitu_times' in ds_plume.coords:
        ins_times = pd.to_datetime(ds_plume['insitu_times'].values)
        ws = np.asarray(ds_plume['wind_speed_insitu'].values, dtype=float)
        wd = np.asarray(ds_plume['wind_dir_insitu'].values, dtype=float) if 'wind_dir_insitu' in ds_plume else None
        # find nearest insitu index
        diffs2 = np.abs((ins_times - np.datetime64(t_funnel)) / np.timedelta64(1, 's'))
        if len(diffs2) > 0:
            idx_ins = int(np.nanargmin(diffs2))
            wind_speed_at_funnel = float(ws[idx_ins])
            if wd is not None:
                wind_dir_at_funnel = float(wd[idx_ins])
except Exception:
    pass



# compute relative wind at funnel using helper

corrected_speed, corrected_dir = compute_relative_wind(ship_speed_at_funnel, ship_course_at_funnel, wind_profile, wind_dir_at_funnel*np.ones_like(wind_profile)) 
wind_profile = corrected_speed
# store corrected profile
assigned = False

ds_plume = ds_plume.assign(wind_profile_m_s=(['image_row', 'window_plume'], wind_profile))

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

# --- Vertical integration of values inside the plume mask ---
try:
    mask_arr = ds_plume['plume_mask'].values.astype(bool)
except Exception:
    mask_arr = mask.astype(bool)

# Ensure mask has same orientation as `arr` (vea x time)
if mask_arr.shape != arr.shape:
    if mask_arr.T.shape == arr.shape:
        mask_arr = mask_arr.T
    else:
        raise ValueError('plume mask and data array shapes do not match: '
                         f'{mask_arr.shape} vs {arr.shape}')

# Integrate along the VEA axis using trapezoidal rule, only including mask True
# `vea` provides the coordinate values for the vertical axis
integrated_plume = np.trapezoid(np.where(mask_arr, arr, 0.0), x=vea, axis=0)

# --- compute flux: slant column * dh * wind, then vertically integrate to time series ---

dh = np.asarray(vea_dh_m)

wp = np.asarray(ds_plume['wind_profile_m_s'].values)
wp2 = wp

dh2 =  dh * np.ones((len(times), 1))
dh2 = dh2.T


# compute flux grid (same shape as arr)
# use  slant columns and convert from 1/cm^2 to 1/m^2 by multiplying 1e4
flux_grid = arr * dh2 * np.abs(wp2) * 1e4
# apply plume mask
flux_grid_masked = np.where(mask_arr, flux_grid, np.nan)

# vertically integrate (sum) to get transport-like time series (units depend on arr units)
flux_ts = np.nansum(np.where(mask_arr, flux_grid, 0.0), axis=0)

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

# store grid and timeseries in dataset when possible
try:
    ds_plume = ds_plume.assign(flux_grid=(['vea','window_plume'], flux_grid_masked))
except Exception:
    try:
        ds_plume = ds_plume.assign(flux_grid=(['vea','time'], flux_grid_masked))
    except Exception:
        ds_plume.attrs['flux_grid_shape'] = flux_grid_masked.shape

try:
    ds_plume = ds_plume.assign(flux_transport=(['window_plume'], flux_ts))
except Exception:
    try:
        ds_plume = ds_plume.assign(flux_transport=(['time'], flux_ts))
    except Exception:
        ds_plume.attrs['flux_transport_shape'] = flux_ts.shape


# Plot flux time series on the shared-x subplot
axf.plot(times, flux_ts_plot, marker='o', label='vertically integrated flux')


# Fit polynomial (2nd order) to flux_ts for times > t_funnel and plot
if t_fit_end is not None:
    mask_time = (times > t_funnel ) & (times < t_fit_end)
else: 
    mask_time = (times > t_funnel)
if np.any(mask_time):
    t0 = pd.to_datetime(times.min())
    x_seconds = (pd.to_datetime(times[mask_time]) - t0).total_seconds()
    y = flux_ts[mask_time]
    if len(x_seconds) >= 3:
        coeffs_f, cov_f = np.polyfit(x_seconds, y, deg=3, cov=True)
        p_f = np.poly1d(coeffs_f)
        x_mask = (pd.to_datetime(times[mask_time]) - t0).total_seconds()
        fitted_mask = p_f(x_mask)

        x_funnel = (pd.to_datetime(t_funnel) - t0).total_seconds()
        J = np.array([x_funnel**3, x_funnel**2, x_funnel, 1.0])
        try:
            var_funnel = float(J @ cov_f @ J.T)
            std_funnel = float(np.sqrt(var_funnel)) if var_funnel >= 0 else np.nan
        except Exception:
            var_funnel = np.nan
            std_funnel = np.nan
        fitted_at_funnel = float(p_f(x_funnel))

        # overlay fit on the existing flux subplot
        axf.plot(times[mask_time], fitted_mask / flux_scale, color='red', label='poly3 fit')
        axf.errorbar(
            [pd.to_datetime(t_funnel)],
            [fitted_at_funnel / flux_scale],
            yerr=[std_funnel / flux_scale],
            fmt='s',
            color='C2',
            label='NO$_2$ flux @funnel'
        )

        # store fit results
        try:
            ds_plume.attrs['flux_poly3_coeffs'] = [float(c) for c in coeffs_f]
            ds_plume.attrs['flux_poly3_coeffs_cov'] = cov_f.tolist()
            ds_plume.attrs['flux_poly3_fit_at_funnel'] = float(fitted_at_funnel)
            ds_plume.attrs['flux_poly3_fit_at_funnel_std'] = float(std_funnel)
            #assign the fitted flux at funnel as a new variable (same dimension as window_plume/time) but set all values outside the maksk to zero
            
            flux_poly3_fit = p_f((pd.to_datetime(times) - t0).total_seconds())
            flux_poly3_fit = np.where(mask_arr.sum(axis=0) > 0, flux_poly3_fit, 0.0)
            ds_plume = ds_plume.assign(flux_poly3_fit=(['window_plume'], flux_poly3_fit))
            
        except Exception:
            pass
        print(f'flux poly3 at funnel: {fitted_at_funnel:.6g} ± {std_funnel:.6g}')
    else:
        print('Not enough points after t_funnel to fit flux polynomial')
else:
    print('No times greater than t_funnel; skipping flux polynomial fit')

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

# --- convert flux at t_funnel to NOx and grams (use flux_transport if available) ---
try:

    times_dt = pd.to_datetime(times)
    diffs = np.abs((times_dt - t_funnel).total_seconds())
    idx_near = int(np.nanargmin(diffs))

    # prefer flux_transport (flux = slant*dh*wind) at funnel; fall back to integrated_plume

    flux_at_funnel = float(ds_plume.attrs['flux_poly3_fit_at_funnel'])

    # convert to NOx by dividing by provided factor
    nox_molec = flux_at_funnel / 0.138
    no_molec = nox_molec - flux_at_funnel
    ds_plume.attrs['nox_at_funnel_molec'] = float(nox_molec)
    ds_plume.attrs['no_at_funnel_molec'] = float(no_molec)

    # convert molecules to grams
    NA = 6.02214076e23
    M_NO2 = 46.0055
    M_NO = 30.0061
    no2_moles = flux_at_funnel / NA
    no2_grams = no2_moles * M_NO2
    no_moles = no_molec / NA
    no_grams = no_moles * M_NO
    nox_grams = no_grams + no2_grams
    ds_plume.attrs['nox_at_funnel_g'] = float(nox_grams)
    print(f'flux@funnel = {flux_at_funnel:.6g} molecules; NOx = {nox_molec:.6g} molecules -> {nox_grams:.6g} g/s')
except Exception as e:
    print('Failed flux->NOx conversion:', e)

ds_plume.close()

# %%
