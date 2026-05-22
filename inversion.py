
#%%
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from typing import Optional, Sequence, Union, Dict
from SEICOR.puff_model import puff_model_2D_with_derivatives
from SEICOR import plumes
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import lmfit
from scipy.optimize import least_squares




def initialize_setting_and_measurement_vector(plume_measurement_file: str, plume_model_parameters: Dict[str, Union[float, np.ndarray]], shift_funnel_height = 0):
    import numpy as np
    ds_plume = xr.open_dataset(plume_measurement_file)
    funnel_height = float(ds_plume.attrs['funnel_height_m']) + shift_funnel_height
    lats = ds_plume['ship_ais_lats'].values
    lons = ds_plume['ship_ais_lons'].values
    ais_times = ds_plume['ship_ais_times'].values
    t_funnel = pd.to_datetime(ds_plume.attrs['t_funnel'])
    t_ais = pd.to_datetime(ds_plume.attrs['t']).tz_localize(None)
    idx_closest_ais = np.argmin(np.abs(ais_times - np.datetime64(t_ais)))
    idx_closest_funnel = np.argmin(np.abs(ais_times - np.datetime64(t_funnel)))
    idx_offset = idx_closest_ais - idx_closest_funnel
    print(f"t_funnel: {t_funnel}, t_ais: {t_ais}")
    print(f"index_offset between t_ais and t_funnel: {idx_offset} (positive means ais is later than funnel)")
    #select ais times +/-5min around t_funnel and interpolate time and lats and lons to 1s resolution
    time_mask = (ais_times >= (t_ais - np.timedelta64(2, 'm'))) & (ais_times <= (t_ais + np.timedelta64(4, 'm')))
    # corrected funnel positions 
    lats_diff, lons_diff = lats[idx_closest_funnel] - lats[idx_closest_ais], lons[idx_closest_funnel] - lons[idx_closest_ais]
    ais_lats_sel_shifted = lats[time_mask] - lats_diff
    ais_lons_sel_shifted = lons[time_mask] - lons_diff    

    #time_mask_shifted = np.zeros_like(time_mask, dtype=bool) # this one is outdated. use the above lines instead
    #try:
    #    if idx_offset >= 0:
    #        time_mask_shifted[idx_offset:] = time_mask[: time_mask.size - idx_offset]
    #    else:
    #        off = int(abs(idx_offset))
    #        time_mask_shifted[: time_mask.size - off] = time_mask[off:]
    #except Exception:
    #    # fallback: if idx_offset not usable, keep shifted mask identical
    #    time_mask_shifted = time_mask.copy()
    #ais_lats_sel_shifted = lats[time_mask_shifted]
    #ais_lons_sel_shifted = lons[time_mask_shifted]

    ais_times_sel = ais_times[time_mask]


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
    azi = ds_plume.vaa[ds_plume.window_plume.values[0]].values

    meas_time_grid = ds_plume['times_plume']
    ref_image = ds_plume["no2_ref"] - ds_plume["no2_ref"].mean(dim="window_ref")
    std=ref_image.std()

    plume_mask = plumes.detect_plume_ztest(
        ds_plume["no2_enhancement_interp"].values,
        bg_std=ref_image.std(),
        bg_mean=ref_image.mean(),
        p_threshold=0.001,
        min_cluster_size=5,
        connectivity=1,
        kernel_arm=1,
        require_connection=True,
        ds_plume=ds_plume,
        keep_second_largest=False,
        second_size_threshold=100,
    ) 
    
    ship_mask = plumes.detect_plume_ztest_left(
        ds_plume["no2_enhancement_c_back"].values, 
        p_threshold=0.15, 
        min_cluster_size=20)
    

    plume_mask = np.asarray(plume_mask, dtype=bool)
    ship_mask = np.asarray(ship_mask, dtype=bool)
    #ds_plume["no2_enhancement_c_back"] -= ds_plume["no2_enhancement_c_back"].isel(image_row=slice(37, 42)).mean(dim="image_row")

    #    # Add random True points: 20% of original True count
    #import numpy as np
    #rng = np.random.default_rng()
    #n_true = np.count_nonzero(plume_mask)
    #n_add = max(1, int(round(0.2 * n_true)))
    #ship_mask_use = np.asarray(ship_mask, dtype=bool)
    #if ship_mask_use.shape != plume_mask.shape and ship_mask_use.T.shape == plume_mask.shape:
    #    ship_mask_use = ship_mask_use.T
    #if ship_mask_use.shape != plume_mask.shape:
    #    # If we cannot align, fall back to not excluding anything (but keep code robust)
    #    ship_mask_use = np.zeros_like(plume_mask, dtype=bool)
    #
    #eligible = (~plume_mask) & (~ship_mask_use)
    #false_indices = np.argwhere(eligible)
    #if false_indices.shape[0] > 0:
    #    chosen = rng.choice(false_indices.shape[0], size=min(n_add, false_indices.shape[0]), replace=False)
    #    add_points = false_indices[chosen]
    #    for idx in add_points:
    #        plume_mask[tuple(idx)] = True

    # Grow a "tube" around the plume mask (dilation) until the mask contains ~2x the original True pixels.
    # Avoid adding pixels that fall within the ship mask.
    ship_mask_use = np.asarray(ship_mask, dtype=bool)
    if ship_mask_use.shape != plume_mask.shape and ship_mask_use.T.shape == plume_mask.shape:
        ship_mask_use = ship_mask_use.T
    if ship_mask_use.shape != plume_mask.shape:
        # If we cannot align, fall back to not excluding anything (but keep code robust)
        ship_mask_use = np.zeros_like(plume_mask, dtype=bool)

    plume_mask = np.asarray(plume_mask, dtype=bool)
    if plume_mask.ndim != 2:
        raise ValueError(f"plume_mask must be 2D, got shape {plume_mask.shape}")

    # Mask out the ±10 seconds window around t_funnel
    times_plume = pd.to_datetime(np.asarray(ds_plume['times_plume'].values))
    try:
        times_plume = times_plume.tz_localize(None)
    except Exception:
        pass
    t_funnel_use = pd.to_datetime(t_funnel)
    try:
        t_funnel_use = t_funnel_use.tz_localize(None)
    except Exception:
        pass



    # Preserve full measurement (before any zeroing) and the *initial* (pre-expansion) plume mask.
    # We will use these for plotting/diagnostics.
    plume_mask_initial = plume_mask.copy()
    no2_enhancement_full = np.asarray(ds_plume["no2_enhancement_c_back"].values)

    # Zero out the measurement outside the *initial* plume mask.
    # This way, when we expand the mask, newly added pixels contribute zeros.
    #no2_enhancement = np.asarray(no2_enhancement_full)
    #plume_mask_initial_use = plume_mask_initial
    #if plume_mask_initial_use.shape != no2_enhancement.shape and plume_mask_initial_use.T.shape == no2_enhancement.shape:
    #    plume_mask_initial_use = plume_mask_initial_use.T
    #if plume_mask_initial_use.shape == no2_enhancement.shape:
    #    no2_enhancement_zeroed = no2_enhancement.copy()
    #    no2_enhancement_zeroed[~plume_mask_initial_use] = 0.0
    #    ds_plume = ds_plume.copy()
    #    ds_plume["no2_enhancement_interp"] = (ds_plume["no2_enhancement_interp"].dims, no2_enhancement_zeroed)



    n_true_initial = int(np.count_nonzero(plume_mask))
    target_true = 3 * n_true_initial

    def _shift2d(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        shifted = np.zeros_like(mask, dtype=bool)
        ny, nx = mask.shape

        y_src_start = max(0, -dy)
        y_src_end = ny - max(0, dy)
        x_src_start = max(0, -dx)
        x_src_end = nx - max(0, dx)

        y_dst_start = max(0, dy)
        y_dst_end = ny - max(0, -dy)
        x_dst_start = max(0, dx)
        x_dst_end = nx - max(0, -dx)

        if y_src_end <= y_src_start or x_src_end <= x_src_start:
            return shifted

        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = mask[y_src_start:y_src_end, x_src_start:x_src_end]
        return shifted

    def _dilate8(mask: np.ndarray) -> np.ndarray:
        dil = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                dil |= _shift2d(mask, dy, dx)
        return dil

    if n_true_initial > 0:
        # Add rings around the plume. If the final ring would overshoot, take an evenly spaced subset.
        current_true = n_true_initial
        max_iters = int(max(plume_mask.shape) + 2)
        for _ in range(max_iters):
            if current_true >= target_true:
                break

            ring = _dilate8(plume_mask) & (~plume_mask) & (~ship_mask_use)
            n_ring = int(np.count_nonzero(ring))
            if n_ring == 0:
                break

            n_need = target_true - current_true
            if n_ring <= n_need:
                plume_mask |= ring
                current_true += n_ring
            else:
                ring_idx = np.argwhere(ring)
                # Choose an evenly spaced subset across the ring to keep the expansion spatially balanced.
                step = max(1, int(ring_idx.shape[0] // n_need))
                chosen_idx = ring_idx[::step][:n_need]
                plume_mask[chosen_idx[:, 0], chosen_idx[:, 1]] = True
                current_true += int(chosen_idx.shape[0])
                break
            
    # Mask out the 7 lowermost viewing directions (lowest VEA angles)
    vea_vals = np.asarray(ds_plume['vea'].values)
    try:
        vea_vals = vea_vals.astype(float)
    except Exception:
        pass
    n_low = 7
    if vea_vals.size >= n_low:
        low_idx = np.argsort(vea_vals)[:n_low]
        if plume_mask.shape[0] == vea_vals.size:
            plume_mask[low_idx, :] = False
            ship_mask_use[low_idx, :] = False
        elif plume_mask.shape[1] == vea_vals.size:
            plume_mask[:, low_idx] = False
            ship_mask_use[:, low_idx] = False

    time_exclude = (times_plume >= (t_funnel_use - pd.Timedelta(seconds=30))) & (times_plume <= (t_funnel_use + pd.Timedelta(seconds=30)))
    if time_exclude.size > 0:
        if plume_mask.shape[1] == time_exclude.size:
            plume_mask[:, time_exclude] = False
            ship_mask_use[:, time_exclude] = False
        elif plume_mask.shape[0] == time_exclude.size:
            plume_mask[time_exclude, :] = False
            ship_mask_use[time_exclude, :] = False

    #calc mean and std of ref_image where plume_mask is False and ship_mask_use is False, and print the values
    ref_image_masked = ds_plume["no2_enhancement_c_back"].where((~plume_mask) & (~ship_mask_use))
    print(f"Mean of ref_image (plume_mask=False, ship_mask_use=False): {ref_image_masked.mean().values}")
    print(f"Std of ref_image (plume_mask=False, ship_mask_use=False): {ref_image_masked.std().values}")


    # Plot NO2 enhancements with plume contour + t_funnel marker (similar to emission.py)
    arr = np.asarray(ds_plume['no2_enhancement_c_back'].values)
    times = pd.to_datetime(ds_plume['times_plume'].values)
    try:
        times = times.tz_localize(None)
    except Exception:
        pass
    vea = np.asarray(ds_plume['vea'].values)

    if arr.ndim != 2:
        raise ValueError(f"expected 2D array for no2_enhancement_c_back, got shape {arr.shape}")
    # Ensure arr is (vea, time)
    if arr.shape[0] == len(times) and arr.shape[1] == len(vea):
        arr = arr.T

    # Also prepare the full (un-zeroed) measurement in the same orientation for later plots.
    arr_full = np.asarray(no2_enhancement_full)
    if arr_full.ndim == 2 and arr_full.shape[0] == len(times) and arr_full.shape[1] == len(vea):
        arr_full = arr_full.T

    # Build edges for pcolormesh
    xnum = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
    dx = np.median(np.diff(xnum)) if len(xnum) > 1 else 1.0
    xedges = np.concatenate((xnum - dx / 2.0, [xnum[-1] + dx / 2.0]))
    dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
    yedges = np.concatenate((vea - dy / 2.0, [vea[-1] + dy / 2.0]))

    fig, ax = plt.subplots(figsize=(14, 5))

    vmax_meas = float(np.nanmax(np.abs(arr)))
    vmin_meas = -vmax_meas
    norm_meas = None
    if np.isfinite(vmin_meas) and np.isfinite(vmax_meas) and (vmin_meas < 0.0) and (vmax_meas > 0.0):
        norm_meas = mcolors.TwoSlopeNorm(vmin=vmin_meas, vcenter=0.0, vmax=vmax_meas)
    if norm_meas is None:
        pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', vmin=vmin_meas, vmax=vmax_meas)
    else:
        pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', norm=norm_meas)

    # Contour of plume mask boundary
    mask_arr = np.asarray(plume_mask, dtype=int)
    if mask_arr.shape != arr.shape and mask_arr.T.shape == arr.shape:
        mask_arr = mask_arr.T
    if mask_arr.shape == arr.shape:
        xcent = (xedges[:-1] + xedges[1:]) / 2.0
        ycent = (yedges[:-1] + yedges[1:]) / 2.0
        Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, mask_arr, levels=[0.5], colors='red', linewidths=1.5)

    # Contour of ship mask boundary (blue)
    ship_arr = np.asarray(ship_mask_use, dtype=int)
    if ship_arr.shape != arr.shape and ship_arr.T.shape == arr.shape:
        ship_arr = ship_arr.T
    if ship_arr.shape == arr.shape:
        try:
            Xc
            Yc
        except NameError:
            xcent = (xedges[:-1] + xedges[1:]) / 2.0
            ycent = (yedges[:-1] + yedges[1:]) / 2.0
            Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, ship_arr, levels=[0.5], colors='blue', linewidths=1.5)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    ax.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, label='t_funnel')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('VEA (°)')
    ax.set_title('NO$_2$ enhancement with plume mask contour')
    fig.colorbar(pcm, ax=ax, label='NO$_2$ enhancement')
    plt.tight_layout()
    plt.show()

    # Apply mask to ds_plume['no2_enhancement_interp'] (image_row, window_plume)
    # Use only the rows that correspond to the elevations in the mask
    # Assume image_row aligns with elevation (if not, user should clarify)
    no2_enh = ds_plume['no2_enhancement_c_back'].values  # (image_row, window_plume)
    # If image_row != elevation, try to select the correct rows

    #make sure the mask, model and no2_enh have compatible shapes
    no2_transponed = no2_enh.T
    no2_masked = no2_transponed[plume_mask.T]


    # Flatten to 1D
    no2_masked_vec = no2_masked.flatten()
    print(f"Shape of masked NO2 enhancement array: {no2_enh.shape}")
    # Create mask_flat for covariance size
    mask_flat = plume_mask.T.flatten()

    print(f"Shape of flattened mask: {plume_mask.shape}")

    # Diagonal covariance matrix from std
    cov_matrix = np.eye(mask_flat.sum()) * std.values**2

    # Store plotting context so we can re-use it after running the forward model
    plume_mask_plot = np.asarray(plume_mask, dtype=bool)
    ship_mask_plot = np.asarray(ship_mask_use, dtype=bool)
    if plume_mask_plot.shape != arr.shape and plume_mask_plot.T.shape == arr.shape:
        plume_mask_plot = plume_mask_plot.T
    if ship_mask_plot.shape != arr.shape and ship_mask_plot.T.shape == arr.shape:
        ship_mask_plot = ship_mask_plot.T

    plume_mask_initial_plot = np.asarray(plume_mask_initial, dtype=bool)
    if plume_mask_initial_plot.shape != arr_full.shape and plume_mask_initial_plot.T.shape == arr_full.shape:
        plume_mask_initial_plot = plume_mask_initial_plot.T

    return {
        'ais_lats_interp': ais_lats_interp,
        'ais_lons_interp': ais_lons_interp,
        'ais_times_interp': ais_times_interp,
        'u_wind_interp': u_wind_interp,
        'v_wind_interp': v_wind_interp,
        'inst_lat': inst_lat,
        'inst_lon': inst_lon,
        'inst_height': inst_height,
        'elevs': elevs,
        'azi': azi,
        'funnel_height': funnel_height,
        'meas_time_grid': meas_time_grid,
        'plume_mask': plume_mask,
        'ship_mask': ship_mask_use,
        't_funnel': t_funnel,
        'meas_times': times,
        'meas_vea': vea,
        'meas_no2_enhancement_2d': arr,
        'meas_no2_enhancement_2d_full': arr_full,
        'plume_mask_plot': plume_mask_plot,
        'plume_mask_initial_plot': plume_mask_initial_plot,
        'ship_mask_plot': ship_mask_plot,
    }, no2_masked_vec, cov_matrix

#get rid of the following function
def initialize_setting_and_measurement_vector_hacky(plume_measurement_file: str,plume_measurement_file_2: str , plume_model_parameters: Dict[str, Union[float, np.ndarray]], shift_funnel_height = 0):
    import numpy as np
    ds_plume = xr.open_dataset(plume_measurement_file)
    ds_plume_2 = xr.open_dataset(plume_measurement_file_2)
    funnel_height = float(ds_plume.attrs['funnel_height_m']) + shift_funnel_height
    lats = ds_plume_2['ship_ais_lats'].values
    lons = ds_plume_2['ship_ais_lons'].values
    ais_times = ds_plume_2['ship_ais_times'].values
    t_funnel = pd.to_datetime(ds_plume.attrs['t_funnel']) + pd.Timedelta(seconds=2)
    t_ais = pd.to_datetime(ds_plume_2.attrs['t']).tz_localize(None)
    idx_closest_ais = np.argmin(np.abs(ais_times - np.datetime64(t_ais)))
    idx_closest_funnel = np.argmin(np.abs(ais_times - np.datetime64(t_funnel)))
    idx_offset = idx_closest_ais - idx_closest_funnel
    print(f"t_funnel: {t_funnel}, t_ais: {t_ais}")
    print(f"index_offset between t_ais and t_funnel: {idx_offset} (positive means ais is later than funnel)")
    #select ais times +/-5min around t_funnel and interpolate time and lats and lons to 1s resolution
    time_mask = (ais_times >= (t_ais - np.timedelta64(2, 'm'))) & (ais_times <= (t_ais + np.timedelta64(4, 'm')))
    # corrected funnel positions 
    lats_diff, lons_diff = lats[idx_closest_funnel] - lats[idx_closest_ais], lons[idx_closest_funnel] - lons[idx_closest_ais]
    ais_lats_sel_shifted = lats[time_mask] - lats_diff
    ais_lons_sel_shifted = lons[time_mask] - lons_diff    

    #time_mask_shifted = np.zeros_like(time_mask, dtype=bool) # this one is outdated. use the above lines instead
    #try:
    #    if idx_offset >= 0:
    #        time_mask_shifted[idx_offset:] = time_mask[: time_mask.size - idx_offset]
    #    else:
    #        off = int(abs(idx_offset))
    #        time_mask_shifted[: time_mask.size - off] = time_mask[off:]
    #except Exception:
    #    # fallback: if idx_offset not usable, keep shifted mask identical
    #    time_mask_shifted = time_mask.copy()
    #ais_lats_sel_shifted = lats[time_mask_shifted]
    #ais_lons_sel_shifted = lons[time_mask_shifted]

    ais_times_sel = ais_times[time_mask]


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
    azi = ds_plume.vaa[ds_plume.window_plume.values[0]].values

    meas_time_grid = ds_plume['times_plume']
    ref_image = ds_plume["no2_ref"] - ds_plume["no2_ref"].mean(dim="window_ref")
    std=ref_image.std()

    plume_mask = plumes.detect_plume_ztest(
        ds_plume["no2_enhancement_interp"].values,
        bg_std=ref_image.std(),
        bg_mean=ref_image.mean(),
        p_threshold=0.001,
        min_cluster_size=5,
        connectivity=1,
        kernel_arm=1,
        require_connection=True,
        ds_plume=ds_plume,
        keep_second_largest=False,
        second_size_threshold=100,
    ) 
    
    ship_mask = plumes.detect_plume_ztest_left(
        ds_plume["no2_enhancement_c_back"].values, 
        p_threshold=0.10, 
        min_cluster_size=20)
    

    plume_mask = np.asarray(plume_mask, dtype=bool)
    ship_mask = np.asarray(ship_mask, dtype=bool)

    reference_image_row_ship = np.array(ds_plume.image_row[37:42])
    reference_image_row_plume = np.array(ds_plume.image_row[8:15])

    # --- Column-wise reference subtraction (x-dimension correction) ---
    # For each x-column where plume OR ship is detected, compute the mean over the
    # reference VEA/image_row band and subtract it from the full column.
    arr = np.asarray(ds_plume["no2_enhancement_c_back"].values, dtype=float)
    plume_mask = np.asarray(plume_mask, dtype=bool)
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

        ds_plume["no2_enhancement_c_back"] = (ds_plume["no2_enhancement_c_back"].dims, arr_refcorr)


    #ds_plume["no2_enhancement_c_back"] -= ds_plume["no2_enhancement_c_back"].isel(image_row=slice(37, 42)).mean(dim="image_row")

    #    # Add random True points: 20% of original True count
    #import numpy as np
    #rng = np.random.default_rng()
    #n_true = np.count_nonzero(plume_mask)
    #n_add = max(1, int(round(0.2 * n_true)))
    #ship_mask_use = np.asarray(ship_mask, dtype=bool)
    #if ship_mask_use.shape != plume_mask.shape and ship_mask_use.T.shape == plume_mask.shape:
    #    ship_mask_use = ship_mask_use.T
    #if ship_mask_use.shape != plume_mask.shape:
    #    # If we cannot align, fall back to not excluding anything (but keep code robust)
    #    ship_mask_use = np.zeros_like(plume_mask, dtype=bool)
    #
    #eligible = (~plume_mask) & (~ship_mask_use)
    #false_indices = np.argwhere(eligible)
    #if false_indices.shape[0] > 0:
    #    chosen = rng.choice(false_indices.shape[0], size=min(n_add, false_indices.shape[0]), replace=False)
    #    add_points = false_indices[chosen]
    #    for idx in add_points:
    #        plume_mask[tuple(idx)] = True

    # Grow a "tube" around the plume mask (dilation) until the mask contains ~2x the original True pixels.
    # Avoid adding pixels that fall within the ship mask.
    ship_mask_use = np.asarray(ship_mask, dtype=bool)
    if ship_mask_use.shape != plume_mask.shape and ship_mask_use.T.shape == plume_mask.shape:
        ship_mask_use = ship_mask_use.T
    if ship_mask_use.shape != plume_mask.shape:
        # If we cannot align, fall back to not excluding anything (but keep code robust)
        ship_mask_use = np.zeros_like(plume_mask, dtype=bool)

    plume_mask = np.asarray(plume_mask, dtype=bool)
    if plume_mask.ndim != 2:
        raise ValueError(f"plume_mask must be 2D, got shape {plume_mask.shape}")

    # Mask out the ±10 seconds window around t_funnel
    times_plume = pd.to_datetime(np.asarray(ds_plume['times_plume'].values))
    try:
        times_plume = times_plume.tz_localize(None)
    except Exception:
        pass
    t_funnel_use = pd.to_datetime(t_funnel)
    try:
        t_funnel_use = t_funnel_use.tz_localize(None)
    except Exception:
        pass



    # Preserve full measurement (before any zeroing) and the *initial* (pre-expansion) plume mask.
    # We will use these for plotting/diagnostics.
    plume_mask_initial = plume_mask.copy()
    no2_enhancement_full = np.asarray(ds_plume["no2_enhancement_c_back"].values)

    # Zero out the measurement outside the *initial* plume mask.
    # This way, when we expand the mask, newly added pixels contribute zeros.
    #no2_enhancement = np.asarray(no2_enhancement_full)
    #plume_mask_initial_use = plume_mask_initial
    #if plume_mask_initial_use.shape != no2_enhancement.shape and plume_mask_initial_use.T.shape == no2_enhancement.shape:
    #    plume_mask_initial_use = plume_mask_initial_use.T
    #if plume_mask_initial_use.shape == no2_enhancement.shape:
    #    no2_enhancement_zeroed = no2_enhancement.copy()
    #    no2_enhancement_zeroed[~plume_mask_initial_use] = 0.0
    #    ds_plume = ds_plume.copy()
    #    ds_plume["no2_enhancement_interp"] = (ds_plume["no2_enhancement_interp"].dims, no2_enhancement_zeroed)



    n_true_initial = int(np.count_nonzero(plume_mask))
    target_true = 2 * n_true_initial

    def _shift2d(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        shifted = np.zeros_like(mask, dtype=bool)
        ny, nx = mask.shape

        y_src_start = max(0, -dy)
        y_src_end = ny - max(0, dy)
        x_src_start = max(0, -dx)
        x_src_end = nx - max(0, dx)

        y_dst_start = max(0, dy)
        y_dst_end = ny - max(0, -dy)
        x_dst_start = max(0, dx)
        x_dst_end = nx - max(0, -dx)

        if y_src_end <= y_src_start or x_src_end <= x_src_start:
            return shifted

        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = mask[y_src_start:y_src_end, x_src_start:x_src_end]
        return shifted

    def _dilate8(mask: np.ndarray) -> np.ndarray:
        dil = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                dil |= _shift2d(mask, dy, dx)
        return dil

    if n_true_initial > 0:
        # Add rings around the plume. If the final ring would overshoot, take an evenly spaced subset.
        current_true = n_true_initial
        max_iters = int(max(plume_mask.shape) + 2)
        for _ in range(max_iters):
            if current_true >= target_true:
                break

            ring = _dilate8(plume_mask) & (~plume_mask) & (~ship_mask_use)
            n_ring = int(np.count_nonzero(ring))
            if n_ring == 0:
                break

            n_need = target_true - current_true
            if n_ring <= n_need:
                plume_mask |= ring
                current_true += n_ring
            else:
                ring_idx = np.argwhere(ring)
                # Choose an evenly spaced subset across the ring to keep the expansion spatially balanced.
                step = max(1, int(ring_idx.shape[0] // n_need))
                chosen_idx = ring_idx[::step][:n_need]
                plume_mask[chosen_idx[:, 0], chosen_idx[:, 1]] = True
                current_true += int(chosen_idx.shape[0])
                break
            
    # Mask out the 7 lowermost viewing directions (lowest VEA angles)
    vea_vals = np.asarray(ds_plume['vea'].values)
    try:
        vea_vals = vea_vals.astype(float)
    except Exception:
        pass
    n_low = 7
    if vea_vals.size >= n_low:
        low_idx = np.argsort(vea_vals)[:n_low]
        if plume_mask.shape[0] == vea_vals.size:
            plume_mask[low_idx, :] = False
            ship_mask_use[low_idx, :] = False
        elif plume_mask.shape[1] == vea_vals.size:
            plume_mask[:, low_idx] = False
            ship_mask_use[:, low_idx] = False

    #time_exclude = (times_plume >= (t_funnel_use - pd.Timedelta(seconds=30))) & (times_plume <= (t_funnel_use + pd.Timedelta(seconds=30)))
    #if time_exclude.size > 0:
    #    if plume_mask.shape[1] == time_exclude.size:
    #        plume_mask[:, time_exclude] = False
    #        ship_mask_use[:, time_exclude] = False
    #    elif plume_mask.shape[0] == time_exclude.size:
    #        plume_mask[time_exclude, :] = False
    #        ship_mask_use[time_exclude, :] = False

    #calc mean and std of ref_image where plume_mask is False and ship_mask_use is False, and print the values
    ref_image_masked = ds_plume["no2_enhancement_c_back"].where((~plume_mask) & (~ship_mask_use))
    print(f"Mean of ref_image (plume_mask=False, ship_mask_use=False): {ref_image_masked.mean().values}")
    print(f"Std of ref_image (plume_mask=False, ship_mask_use=False): {ref_image_masked.std().values}")


    # Plot NO2 enhancements with plume contour + t_funnel marker (similar to emission.py)
    arr = np.asarray(ds_plume['no2_enhancement_c_back'].values)
    times = pd.to_datetime(ds_plume['times_plume'].values)
    try:
        times = times.tz_localize(None)
    except Exception:
        pass
    vea = np.asarray(ds_plume['vea'].values)

    if arr.ndim != 2:
        raise ValueError(f"expected 2D array for no2_enhancement_c_back, got shape {arr.shape}")
    # Ensure arr is (vea, time)
    if arr.shape[0] == len(times) and arr.shape[1] == len(vea):
        arr = arr.T

    # Also prepare the full (un-zeroed) measurement in the same orientation for later plots.
    arr_full = np.asarray(no2_enhancement_full)
    if arr_full.ndim == 2 and arr_full.shape[0] == len(times) and arr_full.shape[1] == len(vea):
        arr_full = arr_full.T

    # Build edges for pcolormesh
    xnum = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
    dx = np.median(np.diff(xnum)) if len(xnum) > 1 else 1.0
    xedges = np.concatenate((xnum - dx / 2.0, [xnum[-1] + dx / 2.0]))
    dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
    yedges = np.concatenate((vea - dy / 2.0, [vea[-1] + dy / 2.0]))

    fig, ax = plt.subplots(figsize=(14, 5))

    vmax_meas = float(np.nanmax(np.abs(arr)))
    vmin_meas = -vmax_meas
    norm_meas = None
    if np.isfinite(vmin_meas) and np.isfinite(vmax_meas) and (vmin_meas < 0.0) and (vmax_meas > 0.0):
        norm_meas = mcolors.TwoSlopeNorm(vmin=vmin_meas, vcenter=0.0, vmax=vmax_meas)
    if norm_meas is None:
        pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', vmin=vmin_meas, vmax=vmax_meas)
    else:
        pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='bwr', norm=norm_meas)

    # Contour of plume mask boundary
    mask_arr = np.asarray(plume_mask, dtype=int)
    if mask_arr.shape != arr.shape and mask_arr.T.shape == arr.shape:
        mask_arr = mask_arr.T
    if mask_arr.shape == arr.shape:
        xcent = (xedges[:-1] + xedges[1:]) / 2.0
        ycent = (yedges[:-1] + yedges[1:]) / 2.0
        Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, mask_arr, levels=[0.5], colors='red', linewidths=1.5)

    # Contour of ship mask boundary (blue)
    ship_arr = np.asarray(ship_mask_use, dtype=int)
    if ship_arr.shape != arr.shape and ship_arr.T.shape == arr.shape:
        ship_arr = ship_arr.T
    if ship_arr.shape == arr.shape:
        try:
            Xc
            Yc
        except NameError:
            xcent = (xedges[:-1] + xedges[1:]) / 2.0
            ycent = (yedges[:-1] + yedges[1:]) / 2.0
            Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, ship_arr, levels=[0.5], colors='blue', linewidths=1.5)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    ax.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, label='t_funnel')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('VEA (°)')
    ax.set_title('NO$_2$ enhancement with plume mask contour')
    fig.colorbar(pcm, ax=ax, label='NO$_2$ enhancement')
    plt.tight_layout()
    plt.show()

    # Apply mask to ds_plume['no2_enhancement_interp'] (image_row, window_plume)
    # Use only the rows that correspond to the elevations in the mask
    # Assume image_row aligns with elevation (if not, user should clarify)
    no2_enh = ds_plume['no2_enhancement_c_back'].values  # (image_row, window_plume)
    # If image_row != elevation, try to select the correct rows

    #make sure the mask, model and no2_enh have compatible shapes
    no2_transponed = no2_enh.T
    no2_masked = no2_transponed[plume_mask.T]


    # Flatten to 1D
    no2_masked_vec = no2_masked.flatten()
    print(f"Shape of masked NO2 enhancement array: {no2_enh.shape}")
    # Create mask_flat for covariance size
    mask_flat = plume_mask.T.flatten()

    print(f"Shape of flattened mask: {plume_mask.shape}")

    # Diagonal covariance matrix from std
    cov_matrix = np.eye(mask_flat.sum()) * std.values**2

    # Store plotting context so we can re-use it after running the forward model
    plume_mask_plot = np.asarray(plume_mask, dtype=bool)
    ship_mask_plot = np.asarray(ship_mask_use, dtype=bool)
    if plume_mask_plot.shape != arr.shape and plume_mask_plot.T.shape == arr.shape:
        plume_mask_plot = plume_mask_plot.T
    if ship_mask_plot.shape != arr.shape and ship_mask_plot.T.shape == arr.shape:
        ship_mask_plot = ship_mask_plot.T

    plume_mask_initial_plot = np.asarray(plume_mask_initial, dtype=bool)
    if plume_mask_initial_plot.shape != arr_full.shape and plume_mask_initial_plot.T.shape == arr_full.shape:
        plume_mask_initial_plot = plume_mask_initial_plot.T

    return {
        'ais_lats_interp': ais_lats_interp,
        'ais_lons_interp': ais_lons_interp,
        'ais_times_interp': ais_times_interp,
        'u_wind_interp': u_wind_interp,
        'v_wind_interp': v_wind_interp,
        'inst_lat': inst_lat,
        'inst_lon': inst_lon,
        'inst_height': inst_height,
        'elevs': elevs,
        'azi': azi,
        'funnel_height': funnel_height,
        'meas_time_grid': meas_time_grid,
        'plume_mask': plume_mask,
        'ship_mask': ship_mask_use,
        't_funnel': t_funnel,
        'meas_times': times,
        'meas_vea': vea,
        'meas_no2_enhancement_2d': arr,
        'meas_no2_enhancement_2d_full': arr_full,
        'plume_mask_plot': plume_mask_plot,
        'plume_mask_initial_plot': plume_mask_initial_plot,
        'ship_mask_plot': ship_mask_plot,
    }, no2_masked_vec, cov_matrix


def plot_measurement_model_difference(model_parameters: Dict[str, object], model_columns_2d: np.ndarray):
    """Three-panel plot: (fit) measurement, model, difference."""

    meas_arr = np.asarray(model_parameters['meas_no2_enhancement_2d'])
    meas_full = np.asarray(model_parameters.get('meas_no2_enhancement_2d_full', meas_arr))
    times = pd.to_datetime(np.asarray(model_parameters['meas_times']))
    try:
        times = times.tz_localize(None)
    except Exception:
        pass
    vea = np.asarray(model_parameters['meas_vea'], dtype=float)
    t_funnel = pd.to_datetime(model_parameters['t_funnel'])

    plume_mask_plot = np.asarray(model_parameters.get('plume_mask_plot', None))
    plume_mask_initial_plot = np.asarray(model_parameters.get('plume_mask_initial_plot', None))
    ship_mask_plot = np.asarray(model_parameters.get('ship_mask_plot', None))

    model_arr = np.asarray(model_columns_2d)
    if model_arr.ndim != 2:
        raise ValueError(f"model_columns_2d must be 2D, got shape {model_arr.shape}")
    # Expect model_arr in (time, elevation); transpose to (elevation, time)
    if model_arr.shape[0] == len(times) and model_arr.shape[1] == len(vea):
        model_arr = model_arr.T

    if meas_arr.shape != model_arr.shape:
        raise ValueError(f"measurement/model shapes do not match: {meas_arr.shape} vs {model_arr.shape}")

    # Shared color scaling for fit measurement + model
    stack = np.array([
        np.nanmax(np.abs(meas_arr)),
        np.nanmax(np.abs(model_arr)),
    ], dtype=float)
    vmax = float(np.nanmax(stack))
    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = 1.0
    vmin = -vmax
    # Edges for pcolormesh
    xnum = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
    dx = np.median(np.diff(xnum)) if len(xnum) > 1 else 1.0
    xedges = np.concatenate((xnum - dx / 2.0, [xnum[-1] + dx / 2.0]))
    dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
    yedges = np.concatenate((vea - dy / 2.0, [vea[-1] + dy / 2.0]))

    # Layout: three panels directly next to each other with minimal spacing,
    # shared y-axis, and a single colorbar for all three.
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.03],
        wspace=0.04,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharex=ax0, sharey=ax0)
    cax = fig.add_subplot(gs[0, 3])

    axes = (ax0, ax1, ax2)

    # Fit/used measurement panel (may be zeroed outside initial mask)
    pcm0 = ax0.pcolormesh(xedges, yedges, meas_arr, shading='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    pcm0.set_zorder(0)


    # Overlay masks + t_funnel on panels
    xcent = (xedges[:-1] + xedges[1:]) / 2.0
    ycent = (yedges[:-1] + yedges[1:]) / 2.0
    Xc, Yc = np.meshgrid(xcent, ycent)

    # Fit mask (expanded plume mask used for inversion) on measurement/model/diff panels
    if plume_mask_plot is not None and plume_mask_plot.shape == meas_arr.shape:
        ax0.contour(Xc, Yc, plume_mask_plot.astype(int), levels=[0.5], colors='orange', linewidths=1.5, linestyles='--',)
        ax1.contour(Xc, Yc, plume_mask_plot.astype(int), levels=[0.5], colors='orange', linewidths=1.5, linestyles='--')
        ax2.contour(Xc, Yc, plume_mask_plot.astype(int), levels=[0.5], colors='orange', linewidths=1.5, linestyles='--')

        # Also show the initial mask on the fit-measurement panel for comparison
        if plume_mask_initial_plot is not None and plume_mask_initial_plot.shape == meas_arr.shape:
            ax0.contour(Xc,Yc,plume_mask_initial_plot.astype(int),levels=[0.5],colors='red',linewidths=1.2,)
    #if ship_mask_plot is not None and ship_mask_plot.shape == meas_arr.shape:
    #    ax0.contour(Xc, Yc, ship_mask_plot.astype(int), levels=[0.5], colors='blue', linewidths=1.5)
    tline0 = ax0.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5, label='t_funnel')
    ax1.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5)
    ax2.axvline(t_funnel, color='k', linestyle='--', linewidth=1.5)

    
    pcm1 = ax1.pcolormesh(xedges, yedges, model_arr, shading='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    pcm1.set_zorder(0)


    diff = model_arr - meas_arr
    pcm2 = ax2.pcolormesh(xedges, yedges, diff, shading='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    pcm2.set_zorder(0)

    # Typography
    title_fs = 18
    label_fs = 18
    tick_fs = 14
    cbar_label_fs = 16
    cbar_tick_fs = 13
    legend_fs = 12

    for ax in axes:
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Draw grid on top of the image so it stays visible.
        ax.set_axisbelow(False)
        ax.grid(True, which='major', linewidth=1.2, alpha=0.6, zorder=5)
        ax.tick_params(axis='both', labelsize=tick_fs)

    # Show only every 6th y tick (based on VEA values)
    if vea.size > 0:
        yticks = vea[::8]
        for ax in axes:
            ax.set_yticks(yticks)

    ax0.set_title('Measurement', fontsize=title_fs)
    ax1.set_title('Model', fontsize=title_fs)
    ax2.set_title('Residual (Model - Measurement)', fontsize=title_fs)
    ax0.set_ylabel('VEA / °', fontsize=label_fs)
    ax0.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax1.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax2.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax1.tick_params(axis='y', labelleft=False)
    ax2.tick_params(axis='y', labelleft=False)
    fig.autofmt_xdate()

    # Legend in the first panel
    from matplotlib.lines import Line2D

    legend_handles = [tline0]
    if plume_mask_plot is not None and plume_mask_plot.shape == meas_arr.shape:
        legend_handles.append(Line2D([0], [0], color='orange', lw=1.5, ls='--', label='Fit mask'))
        if plume_mask_initial_plot is not None and plume_mask_initial_plot.shape == meas_arr.shape:
            legend_handles.append(Line2D([0], [0], color='red', lw=1.2, label='Plume mask'))
    ax0.legend(handles=legend_handles, loc='upper left', fontsize=legend_fs)

    # Single colorbar for all three panels
    cb01 = fig.colorbar(pcm0, cax=cax)
    cb01.set_label('NO$_2$ column / #molec. cm$^{-2}$', fontsize=cbar_label_fs)
    cb01.ax.tick_params(labelsize=cbar_tick_fs)

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.88)
    plt.show()


def forward_model(state_vector, model_parameters):
    #'Q0', 'R', 'f0', 't', 'tau', 'sigma_h0', 'sigma_z0', 'F', 'S', 'background']
    # if any of the state_vector elements are below 0 set them to zero to avoid numerical issues in the forward model (e.g. negative emission rate or negative dispersion)
    #state_vector = np.maximum(state_vector, 0)


    print("Running forward model with state vector:", state_vector)
    ds_2D = puff_model_2D_with_derivatives(
        source_lats=model_parameters['ais_lats_interp'],
        source_lons=model_parameters['ais_lons_interp'],
        times=model_parameters['ais_times_interp'],
        u=model_parameters['u_wind_interp'],
        v=model_parameters['v_wind_interp'],
        instrument_lat=model_parameters['inst_lat'],
        instrument_lon=model_parameters['inst_lon'],
        instrument_altitude=model_parameters['inst_height'],
        Kh=state_vector[5],
        Kz=state_vector[6],
        emission_height=model_parameters['funnel_height'],
        plume_rise_model='linear',
        plume_rise_params={'A': state_vector[7], 'T': state_vector[8], 'F': state_vector[7], 'S': state_vector[8]},
        NO_NO2_conversion_params={'Q0': state_vector[0], 'R': state_vector[1], 'f0': state_vector[2], 't': state_vector[3], 'tau': state_vector[4]},
        azimuth_deg=model_parameters['azi'],
        viewing_elevations=model_parameters['elevs'],
        background_columns=state_vector[9]
    )

    ds_model_interp = ds_2D.interp(time=model_parameters['meas_time_grid'])
    plume_mask = model_parameters['plume_mask']

    # ds_model_interp['columns']: dims (time, elevation)
    # plume_mask: shape (elevation, window_plume)
    # Align mask to model output: assume window_plume aligns with time (after interp)
    # Mask shape: (elevation, time)
    # Transpose mask to (time, elevation) to match ds_model_interp['columns']
    mask = plume_mask.T  # Now shape (time, elevation)

    # Flatten columns and jacobian where mask is True
    columns = ds_model_interp['columns'].values  # (time, elevation)
    jacobian = ds_model_interp['jacobian'].values  # (time, elevation, parameter)

    # Flatten mask, columns, jacobian
    mask_flat = mask.flatten()
    columns_flat = columns.flatten()
    jacobian_flat = jacobian.reshape(-1, jacobian.shape[-1])  # (time*elevation, parameter)

    # Select only masked points
    model_vec = columns_flat[mask_flat]
    jacobian_vec = jacobian_flat[mask_flat, :]

    # model_vec: (m,), jacobian_vec: (m, n_param)
    return model_vec,   jacobian_vec


def levenberg_marquardt_inversion(
    measurement_vector,
    measurement_covariance_matrix,
    state_vector_apr,
    model_parameters,
    forward_model,
    max_nfev=100,
    verbose=2,
    regularize_idx=None,
    regularize_prior=None,
    regularize_strength=1e6
):
    """
    Solve the nonlinear inverse problem using the Levenberg-Marquardt algorithm with Tikhonov regularization.
    regularize_idx: indices of state vector to regularize (keep close to prior)
    regularize_prior: prior values for those indices
    regularize_strength: regularization weight (large = more constant)
    Returns the fitted state vector and the posterior covariance matrix.
    """
    from scipy.linalg import cholesky
    L = cholesky(measurement_covariance_matrix, lower=True)
    Linv = np.linalg.inv(L)

    # Setup regularization
    if regularize_idx is not None and regularize_prior is not None:
        reg_idx = np.array(regularize_idx)
        reg_prior = np.array(regularize_prior)
        n_reg = len(reg_idx)
        def reg_resid(state_vec):
            return np.sqrt(regularize_strength) * (state_vec[reg_idx] - reg_prior)
        def reg_jac(state_vec):
            J = np.zeros((n_reg, state_vec.size))
            for i, idx in enumerate(reg_idx):
                J[i, idx] = np.sqrt(regularize_strength)
            return J
    else:
        reg_resid = lambda state_vec: np.array([])
        reg_jac = lambda state_vec: np.zeros((0, state_vec.size))

    def residuals(state_vec):
        model_vec, _ = forward_model(state_vec, model_parameters)
        res = measurement_vector - model_vec
        # Whiten residuals
        res_data = Linv @ res
        res_reg = reg_resid(state_vec)
        return np.concatenate([res_data, res_reg])

    def jac(state_vec):
        _, jacobian = forward_model(state_vec, model_parameters)
        # Whiten jacobian
        jac_data = Linv @ (-jacobian)
        jac_reg = reg_jac(state_vec)
        return np.vstack([jac_data, jac_reg])

    result = least_squares(
        residuals,
        state_vector_apr,
        jac=jac,
        method='lm',
        max_nfev=max_nfev,
        verbose=verbose
    )

    # Posterior covariance: (J^T C^-1 J + R)^-1
    _, jacobian = forward_model(result.x, model_parameters)
    JT_Cinv_J = jacobian.T @ np.linalg.inv(measurement_covariance_matrix) @ jacobian
    if regularize_idx is not None and regularize_prior is not None:
        R = np.zeros((jacobian.shape[1], jacobian.shape[1]))
        for i, idx in enumerate(reg_idx):
            R[idx, idx] = regularize_strength
        posterior_cov = np.linalg.inv(JT_Cinv_J + R)
    else:
        posterior_cov = np.linalg.inv(JT_Cinv_J)

    return result.x, posterior_cov, result


def lmfit_inversion(
    measurement_vector,
    measurement_covariance_matrix,
    state_vector_apr,
    model_parameters,
    forward_model,
    lower_bounds=None,
    upper_bounds=None,
    prior_mean=None,
    prior_sigma=None,
    prior_cov=None,
    regularize_idx=None,
    regularize_prior=None,
    regularize_strength=1e6,
    verbose=True,
    drop_invalid=True
):
    """
    Inversion using lmfit.Minimizer with bounds and Tikhonov regularization.
    Args:
        measurement_vector: (m,)
        measurement_covariance_matrix: (m, m) diagonal
        state_vector_apr: (n,)
        model_parameters: dict for forward_model
        forward_model: function(state_vector, model_parameters) -> (model_vector, jacobian)
        lower_bounds, upper_bounds: arrays of shape (n,) or None
        prior_mean: (n,) prior mean. If None and prior_sigma/prior_cov is provided, defaults to state_vector_apr.
        prior_sigma: (n,) prior 1-sigma uncertainties per parameter (diagonal prior covariance).
        prior_cov: (n, n) full prior covariance matrix.
        regularize_idx: indices of state vector to regularize (keep close to prior)
        regularize_prior: prior values for those indices
        regularize_strength: regularization weight (large = more constant)
        verbose: print diagnostics
    Returns:
        fitted_state: (n,)
        posterior_cov: (n, n)
        result: lmfit.MinimizerResult
    """

    import lmfit
    from scipy.linalg import cholesky
    import numpy as np

    n = state_vector_apr.size

    measurement_vector = np.asarray(measurement_vector)
    measurement_covariance_matrix = np.asarray(measurement_covariance_matrix)

    if measurement_vector.ndim != 1:
        raise ValueError(f"measurement_vector must be 1D, got shape {measurement_vector.shape}")
    if measurement_covariance_matrix.ndim != 2:
        raise ValueError(
            f"measurement_covariance_matrix must be 2D, got shape {measurement_covariance_matrix.shape}"
        )
    if measurement_covariance_matrix.shape[0] != measurement_vector.size or measurement_covariance_matrix.shape[1] != measurement_vector.size:
        raise ValueError(
            "measurement_covariance_matrix shape must match measurement_vector length: "
            f"{measurement_covariance_matrix.shape} vs {measurement_vector.size}"
        )

    forward_model_use = forward_model
    cov_use = measurement_covariance_matrix
    meas_use = measurement_vector

    if drop_invalid:
        finite_meas = np.isfinite(meas_use)
        if not np.all(finite_meas):
            if verbose:
                print(f"Dropping {np.count_nonzero(~finite_meas)} non-finite measurement points")

        # Also drop points where the forward model is non-finite at the a-priori state.
        # This keeps the residual size fixed across iterations.
        model_vec0, _ = forward_model(state_vector_apr, model_parameters)
        model_vec0 = np.asarray(model_vec0)
        if model_vec0.shape != meas_use.shape:
            raise ValueError(
                "forward_model returned model_vec with incompatible shape: "
                f"{model_vec0.shape} vs measurement_vector {meas_use.shape}"
            )
        finite_model0 = np.isfinite(model_vec0)

        valid = finite_meas & finite_model0
        n_dropped = int(valid.size - np.count_nonzero(valid))
        if n_dropped > 0:
            if verbose:
                print(f"Dropping {n_dropped} points due to NaN/inf in measurements/model (at prior)")
            meas_use = meas_use[valid]
            cov_use = cov_use[np.ix_(valid, valid)]

            def forward_model_use(state_vector, model_parameters):
                model_vec, jacobian = forward_model(state_vector, model_parameters)
                model_vec = np.asarray(model_vec)
                jacobian = np.asarray(jacobian)
                return model_vec[valid], jacobian[valid, :]

    # Sanity check covariance before whitening
    if not np.all(np.isfinite(cov_use)):
        raise ValueError("measurement_covariance_matrix contains NaN/inf")
    if np.any(np.diag(cov_use) <= 0):
        raise ValueError("measurement_covariance_matrix must have strictly positive diagonal")

    m = meas_use.size
    # Whitening
    L = cholesky(cov_use, lower=True)
    Linv = np.linalg.inv(L)

    # Setup lmfit Parameters with bounds
    params = lmfit.Parameters()
    for i in range(n):
        name = f'x{i}'
        val = state_vector_apr[i]
        lb = float(lower_bounds[i]) if lower_bounds is not None else -np.inf
        ub = float(upper_bounds[i]) if upper_bounds is not None else np.inf
        params.add(name, value=val, min=lb, max=ub)

    # Prior / regularization setup
    # Two mutually exclusive ways to add a Gaussian prior:
    # 1) prior_sigma/prior_cov (preferred; supports per-parameter or full covariance)
    # 2) legacy regularize_idx/regularize_strength (diagonal subset, scalar strength)
    if (prior_sigma is not None) or (prior_cov is not None):
        if (regularize_idx is not None) or (regularize_prior is not None):
            raise ValueError("Use either prior_sigma/prior_cov OR regularize_idx/regularize_prior, not both")
        if prior_mean is None:
            prior_mean = state_vector_apr
        prior_mean = np.asarray(prior_mean, dtype=float)
        if prior_mean.shape != (n,):
            raise ValueError(f"prior_mean must have shape ({n},), got {prior_mean.shape}")

        if prior_sigma is not None:
            prior_sigma = np.asarray(prior_sigma, dtype=float)
            if prior_sigma.shape != (n,):
                raise ValueError(f"prior_sigma must have shape ({n},), got {prior_sigma.shape}")
            if np.any(~np.isfinite(prior_sigma)) or np.any(prior_sigma <= 0):
                raise ValueError("prior_sigma must be finite and strictly positive")
            w = 1.0 / prior_sigma

            def reg_resid(state_vec):
                return w * (state_vec - prior_mean)

            def reg_jac(state_vec):
                return np.diag(w)

            prior_precision = np.diag(w * w)
        else:
            prior_cov = np.asarray(prior_cov, dtype=float)
            if prior_cov.shape != (n, n):
                raise ValueError(f"prior_cov must have shape ({n},{n}), got {prior_cov.shape}")
            if not np.all(np.isfinite(prior_cov)):
                raise ValueError("prior_cov contains NaN/inf")
            # Cholesky may raise if not SPD; we let that surface with context.
            L_prior = np.linalg.cholesky(prior_cov)
            Linv_prior = np.linalg.inv(L_prior)

            def reg_resid(state_vec):
                return Linv_prior @ (state_vec - prior_mean)

            def reg_jac(state_vec):
                return Linv_prior

            prior_precision = Linv_prior.T @ Linv_prior

    elif regularize_idx is not None and regularize_prior is not None:
        reg_idx = np.array(regularize_idx)
        reg_prior = np.array(regularize_prior)
        n_reg = len(reg_idx)

        def reg_resid(state_vec):
            return np.sqrt(regularize_strength) * (state_vec[reg_idx] - reg_prior)

        def reg_jac(state_vec):
            J = np.zeros((n_reg, n))
            for i, idx in enumerate(reg_idx):
                J[i, idx] = np.sqrt(regularize_strength)
            return J

        prior_precision = np.zeros((n, n))
        for idx in reg_idx:
            prior_precision[int(idx), int(idx)] = regularize_strength
    else:
        reg_resid = lambda state_vec: np.array([])
        reg_jac = lambda state_vec: np.zeros((0, n))
        prior_precision = np.zeros((n, n))

    # Cache the most recent forward-model evaluation.
    # lmfit will often call residual and jacobian at the same parameter vector; caching avoids
    # calling forward_model_use twice in those cases.
    _fm_cache = {
        'x': None,
        'model_vec': None,
        'jacobian': None,
    }

    def _eval_forward_model_cached(state_vec: np.ndarray):
        x_cached = _fm_cache['x']
        if x_cached is not None and np.array_equal(state_vec, x_cached):
            return _fm_cache['model_vec'], _fm_cache['jacobian']

        model_vec, jacobian = forward_model_use(state_vec, model_parameters)
        model_vec = np.asarray(model_vec)
        jacobian = np.asarray(jacobian)

        _fm_cache['x'] = state_vec.copy()
        _fm_cache['model_vec'] = model_vec
        _fm_cache['jacobian'] = jacobian
        return model_vec, jacobian

    def residuals_lmfit(params):
        state_vec = np.array([params[f'x{i}'].value for i in range(n)])
        model_vec, _ = _eval_forward_model_cached(state_vec)
        res = meas_use - model_vec
        res_data = Linv @ res
        res_reg = reg_resid(state_vec)
        return np.concatenate([res_data, res_reg])

    def jacobian_lmfit(params):
        state_vec = np.array([params[f'x{i}'].value for i in range(n)])
        _, jacobian = _eval_forward_model_cached(state_vec)
        jac_data = Linv @ (-jacobian)
        jac_reg = reg_jac(state_vec)
        jac_full = np.vstack([jac_data, jac_reg])
        return jac_full

    # NOTE: SciPy's legacy leastsq does not accept a Jacobian keyword ('jac').
    # When providing an analytic Jacobian, we must use the 'least_squares' method.
    minner = lmfit.Minimizer(residuals_lmfit, params, jac=jacobian_lmfit)
    result = minner.minimize(method='least_squares')

    # Diagnostics
    fitted_state = np.array([result.params[f'x{i}'].value for i in range(n)])
    # Posterior covariance: (J^T C^-1 J + R)^-1
    _, jacobian = forward_model_use(fitted_state, model_parameters)
    JT_Cinv_J = jacobian.T @ np.linalg.inv(cov_use) @ jacobian
    posterior_cov = np.linalg.inv(JT_Cinv_J + prior_precision)

    if verbose:
        print("lmfit diagnostics:")
        print(result.message)
        print("nfev:", result.nfev)
        print("chisqr:", result.chisqr)
        print("redchi:", result.redchi)
        print("aic:", result.aic)
        print("bic:", result.bic)

    return fitted_state, posterior_cov, result
#%%

# Example usage for lmfit_inversion with bounds and regularization
#plume_measurement_file = r"P:\data\SEICOR\plumes_2\plumes_250428\plume_024_t_20250428_112317_mmsi_563068900.nc" #"D:\SEICOR\plumes_2\plumes_250412\plume_028_t_20250412_150343_mmsi_255806434.nc"
#plume_measurement_file = r"D:\SEICOR\plumes_2\plumes_250412\plume_028_t_20250412_150343_mmsi_255806434.nc"
#plume_measurement_file = r"P:\data\SEICOR\plumes_2\plumes_250412\plume_032_t_20250412_164820_mmsi_247389200.nc"
plume_measurement_file = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_004_t_20250615_051423_mmsi_211676580.nc"
plume_measurement_file_2 = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_003_t_20250615_051307_mmsi_228397600.nc"
#plume_measurement_file = r"P:\data\SEICOR\plumes_2\plumes_250502\plume_016_t_20250502_081152_mmsi_564398000.nc"
model_parameters, measurement_vector, measurement_covariance_matrix = initialize_setting_and_measurement_vector_hacky(plume_measurement_file, plume_measurement_file_2, plume_model_parameters={}, shift_funnel_height = 13)
state_vector_apr = np.array([3*10**22, 1/0.13, 1.32, 600.0, 3600, 0.1, 0.1, 120, 120, 0])
#'Q0', 'R', 'f0', 't', 'tau', 'sigma_h0', 'sigma_z0', 'F', 'S', 'background']
# Set bounds (example: all parameters >= 0, upper bounds can be set as needed)
lower_bounds = np.array([   0,    1,  1,     1e-6,      1e-6,  1e-6,  1e-6, -200, 0, -5*10**16])
upper_bounds = np.array([5e24, 1000, 10,     1e36,      1e36, 1.0, 1.0,  5000.0, 5000.0,  5*10**16])

# Regularize the last two parameters (F, S) to their a priori values
regularize_idx = [1, 2,  9]
regularize_prior =  [state_vector_apr[1], state_vector_apr[2], state_vector_apr[9]]
regularize_strength = 1e6
#%%
fitted_state_lmfit, posterior_cov_lmfit, result_lmfit = lmfit_inversion(
    measurement_vector,
    measurement_covariance_matrix,
    state_vector_apr,
    model_parameters,
    forward_model,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    prior_sigma=np.array([1*10**23, 1/0.13/100, 1.32/100, 600.0, 3600/10, 1, 1, 120*2, 120*2, 1]),
    verbose=True
)
#%%

ds_2D = puff_model_2D_with_derivatives(
    source_lats=model_parameters['ais_lats_interp'],
    source_lons=model_parameters['ais_lons_interp'],
    times=model_parameters['ais_times_interp'],
    u=model_parameters['u_wind_interp'],
    v=model_parameters['v_wind_interp'],
    instrument_lat=model_parameters['inst_lat'],
    instrument_lon=model_parameters['inst_lon'],
    instrument_altitude=model_parameters['inst_height'],
    Kh=fitted_state_lmfit[5],
    Kz=fitted_state_lmfit[6],
    emission_height=model_parameters['funnel_height'],
    plume_rise_model='linear',
    plume_rise_params={'A': fitted_state_lmfit[7], 'T': fitted_state_lmfit[8], 'F': fitted_state_lmfit[7], 'S': fitted_state_lmfit[8]},
    NO_NO2_conversion_params={'Q0': fitted_state_lmfit[0], 'R': fitted_state_lmfit[1], 'f0': fitted_state_lmfit[2], 't': fitted_state_lmfit[3], 'tau': fitted_state_lmfit[4]},
    azimuth_deg=model_parameters['azi'],
    viewing_elevations=model_parameters['elevs'],
    background_columns=fitted_state_lmfit[9]
)
#%%
ds_model_interp = ds_2D.interp(time=model_parameters['meas_time_grid'])#
#apply the plume mask to ds_model_interp['columns'] before calculating chi-squared, and only keep the values where the mask is True
plume_mask = model_parameters['plume_mask'].T
ds_model_interp = ds_2D.interp(time=model_parameters['meas_time_grid'])
model_vec = ds_model_interp['columns'].values[plume_mask].flatten()
#ignore nans in the chi-squared calculation
model_vec_use = model_vec[~np.isnan(model_vec)]
measurement_vector_use = measurement_vector[~np.isnan(model_vec)]
cov_use = measurement_covariance_matrix[~np.isnan(model_vec), :][:, ~np.isnan(model_vec)]
chi2 = np.sum((measurement_vector_use - model_vec_use)**2 / np.diag(cov_use).flatten())
chi2_red = chi2 / (len(measurement_vector_use) - len(state_vector_apr))

print(f"Chi-squared for fitted state: {chi2}")
print(f"Reduced chi-squared for fitted state: {chi2_red}")
plot_measurement_model_difference(model_parameters, ds_model_interp['columns'].values)

plt.plot(model_parameters['meas_times'],model_parameters['meas_no2_enhancement_2d'].sum(axis=0))
plt.plot(model_parameters['meas_times'],ds_model_interp['columns'].sum(axis=1))
plt.legend(['Measurement', 'Model'])
plt.xlabel('Time')
plt.ylabel('Total NO2 enhancement')
plt.title('Total NO2 enhancement over time')
#rotate x-axis labels
plt.xticks(rotation=45)
plt.show()
#%%

ds_2D = puff_model_2D_with_derivatives(
    source_lats=model_parameters['ais_lats_interp'],
    source_lons=model_parameters['ais_lons_interp'],
    times=model_parameters['ais_times_interp'],
    u=model_parameters['u_wind_interp'],
    v=model_parameters['v_wind_interp'],
    instrument_lat=model_parameters['inst_lat'],
    instrument_lon=model_parameters['inst_lon'],
    instrument_altitude=model_parameters['inst_height'],
    Kh=fitted_state_lmfit[5],
    Kz=fitted_state_lmfit[6]/3.7,
    emission_height=model_parameters['funnel_height'],
    plume_rise_model='linear',
    plume_rise_params={'A': 20, 'T': fitted_state_lmfit[8], 'F': fitted_state_lmfit[7], 'S': fitted_state_lmfit[8]},
    NO_NO2_conversion_params={'Q0': fitted_state_lmfit[0]/2.3, 'R': fitted_state_lmfit[1], 'f0': fitted_state_lmfit[2], 't': fitted_state_lmfit[3]*1000, 'tau': fitted_state_lmfit[4]},
    azimuth_deg=model_parameters['azi'],
    viewing_elevations=model_parameters['elevs'],
    background_columns=fitted_state_lmfit[9]
)

ds_model_interp = ds_2D.interp(time=model_parameters['meas_time_grid'])
model_vec = ds_model_interp['columns'].values[plume_mask].flatten()
#ignore nans in the chi-squared calculation
model_vec_use = model_vec[~np.isnan(model_vec)]
measurement_vector_use = measurement_vector[~np.isnan(model_vec)]
cov_use = measurement_covariance_matrix[~np.isnan(model_vec), :][:, ~np.isnan(model_vec)]
chi2 = np.sum((measurement_vector_use - model_vec_use)**2 / np.diag(cov_use).flatten())
chi2_red = chi2 / (len(measurement_vector_use) - len(state_vector_apr))

plt.plot(model_vec_use - measurement_vector_use)
plt.show()
print(f"Chi-squared for fitted state: {chi2}")
print(f"Reduced chi-squared for fitted state: {chi2_red}")
plot_measurement_model_difference(model_parameters, ds_model_interp['columns'].values)


plt.plot(model_parameters['meas_times'],model_parameters['meas_no2_enhancement_2d'].sum(axis=0))
plt.plot(model_parameters['meas_times'],ds_model_interp['columns'].sum(axis=1))
plt.legend(['Measurement', 'Model'])
plt.xlabel('Time')
plt.ylabel('Total NO2 enhancement')
plt.title('Total NO2 enhancement over time')
#rotate x-axis labels
plt.xticks(rotation=45)
plt.show()

#%%

#%%
fitted_state, posterior_cov, result = levenberg_marquardt_inversion(
    measurement_vector,
    measurement_covariance_matrix,
    state_vector_apr,
    model_parameters,
    forward_model
)


# %%
"""
ds_2D = puff_model_2D_with_derivatives(
    source_lats=model_parameters['ais_lats_interp'],
    source_lons=model_parameters['ais_lons_interp'],
    times=model_parameters['ais_times_interp'],
    u=model_parameters['u_wind_interp'],
    v=model_parameters['v_wind_interp'],
    instrument_lat=model_parameters['inst_lat'],
    instrument_lon=model_parameters['inst_lon'],
    instrument_altitude=model_parameters['inst_height'],
    Kh=fitted_state[3],
    Kz=fitted_state[4],
    emission_rate=1.0,
    emission_height=model_parameters['funnel_height'],
    plume_rise_model='briggs',
    plume_rise_params={'A': 250.0, 'T': 120.0, 'F': fitted_state[5], 'S': fitted_state[6]},
    NO_NO2_conversion_params={'R': fitted_state[1], 't': fitted_state[2], 'Q0': fitted_state[0]},
    azimuth_deg=model_parameters['azi'],
    viewing_elevations=model_parameters['elevs'],
)
# %%
ds_model_interp = ds_2D.interp(time=model_parameters['meas_time_grid'])
plot_measurement_model_difference(model_parameters, ds_model_interp['columns'].values)


# %%

#finite difference test for jacobian
epsilon = 1e-6
state_vector_eps_up = state_vector_apr.copy()
state_vector_eps_down = state_vector_apr.copy()
state_vector_eps = state_vector_apr.copy()
state_vector_eps[-1] = 1e15 # Avoid zero for background to prevent division issues in finite difference
model_vec_base, jacobian = forward_model(state_vector_apr, model_parameters)
finite_diff_jacobian = np.zeros_like(jacobian)
for i in range(len(state_vector_apr)):
    state_vector_eps_up[i] += state_vector_eps[i] * epsilon
    state_vector_eps_down[i] -= state_vector_eps[i] * epsilon
    model_vec_eps_up, _ = forward_model(state_vector_eps_up, model_parameters)
    model_vec_eps_down, _ = forward_model(state_vector_eps_down, model_parameters)
    finite_diff_jacobian[:, i] = (model_vec_eps_up - model_vec_eps_down) / (2 * state_vector_eps[i] * epsilon)
    state_vector_eps_up[i] = state_vector_apr[i]
    state_vector_eps_down[i] = state_vector_apr[i]


#%%
# plot the relative difference between the analytical jacobian and the finite difference jacobian
relative_diff = np.abs(jacobian - finite_diff_jacobian) / (np.abs(jacobian) + 1e-20)
#plot for each parameter
param_names = [r'$Q_0$', 'R', r'$f_0$', r'$1/r$', 'tau', r'$\sigma_{h0}$', r'$\sigma_{z0}$', 'A', 'T', r'$B_0$']
fig, axes = plt.subplots(1, len(param_names), figsize=(20, 5))
for i, ax in enumerate(axes):
    im = ax.plot(relative_diff[:, i])
    ax.set_title(f'{param_names[i]}')
    #logscale
    if i < 9:
        ax.set_yscale('log')
#title for the whole figure
fig.suptitle(r'Relative difference analytical and finite difference Jacobian ($\frac{|J_{analytical} - J_{finite\_diff}|}{|J_{analytical}|}$)')
plt.tight_layout()

"""
# %%
