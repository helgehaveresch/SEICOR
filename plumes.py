#%%
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import os
from scipy import ndimage
from scipy.stats import norm
# %%
def add_ship_trajectory_to_plume_ds(ds_plume, filtered_ship_groups):
    group = filtered_ship_groups.get(int(ds_plume.mmsi))
    # determine plume time (prefer attr 't', fall back to times_plume median)
    t_attr = ds_plume.attrs.get("t", None)
    t_plume = pd.to_datetime(t_attr)
    # Build trajectory mask for ±15 minutes around plume time
    ship_idx_times = pd.to_datetime(group.index)
    traj_mask = (ship_idx_times >= (t_plume - pd.Timedelta(minutes=15))) & (ship_idx_times <= (t_plume + pd.Timedelta(minutes=15)))

    # Extract times, lons, lats (ensure dtype compatibility)
    ship_ais_times = ship_idx_times[traj_mask].to_numpy(dtype="datetime64[ns]")
    ship_ais_lons = group.loc[traj_mask, "Longitude"].astype(float).to_numpy()
    ship_ais_lats = group.loc[traj_mask, "Latitude"].astype(float).to_numpy()

    # Assign coordinate and variables to dataset
    ds_plume = ds_plume.assign_coords(ship_ais_times = ship_ais_times)
    ds_plume = ds_plume.assign(
        ship_ais_lons = (["ship_ais_times"], ship_ais_lons),
        ship_ais_lats = (["ship_ais_times"], ship_ais_lats),
    )

    return ds_plume

def add_plume_file_paths_to_ship_passes(ship_passes, plumes_out_dir):
    for idx, passing_ship in ship_passes.iterrows():
        plumes_out_path = os.path.join(plumes_out_dir, f"plume_{passing_ship['Plume_number']:03d}_t_{idx.strftime('%Y%m%d_%H%M%S')}_mmsi_{passing_ship['MMSI']}.nc")
        ship_passes.at[idx, 'plume_file'] = plumes_out_path
    return ship_passes

def add_insitu_to_plume_ds(ds_plume, df_insitu):
    """
    Add in-situ data to plume dataset based on time alignment.

    Args:
        ds_plume (xr.Dataset): The plume dataset with 'datetime' coordinate.
        df_insitu (pd.DataFrame): In-situ dataframe with 'time' index.

    Returns:
        xr.Dataset: The plume dataset with added in-situ data variables.
    """
    try:
        times_dt = pd.to_datetime(ds_plume["times_plume"].values)
    except Exception:
        times_dt = pd.to_datetime(ds_plume["times_plume"])
    times_index = pd.to_datetime(times_dt)
    try:
        insitu_tz = df_insitu.index.tz
    except Exception:
        insitu_tz = None
    if insitu_tz is not None:
        if getattr(times_index, 'tz', None) is None:
            try:
                times_index = times_index.tz_localize('UTC').tz_convert(insitu_tz)
            except Exception:
                times_index = times_index.tz_localize(insitu_tz)
        else:
            times_index = times_index.tz_convert(insitu_tz)
    else:
        try:
            times_index = times_index.tz_convert(None)
        except Exception:
            try:
                times_index = times_index.tz_localize(None)
            except Exception:
                pass
    if len(times_index) > 0 and times_index.isnull().any():
        first_valid = times_index[times_index.notnull()].tolist()[0]
        times_index = times_index.fillna(first_valid)
    times = times_index.to_pydatetime()
    
    in_situ_mask = (df_insitu.index >= times[0]) & (df_insitu.index <= times[-1])    
    in_situ_times = df_insitu.index[in_situ_mask]
    in_situ_no2 = df_insitu['c_no2'][in_situ_mask]
    

    L = 870  # meters

    # Interpolate n_air to times_window
    n_air = df_insitu["p_0"][in_situ_mask] * 1e2 * 6.02214076e23 / (df_insitu["T_in"][in_situ_mask] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(in_situ_times))
    # Use the pandas DatetimeIndex for reindexing (preserves tz-awareness)
    try:
        n_air_aligned = n_air_interp.reindex(times_index, method='nearest').values
    except Exception:
        # Fallback: reindex using the python datetime array
        n_air_aligned = n_air_interp.reindex(pd.DatetimeIndex(times), method='nearest').values

    in_situ_times = in_situ_times.to_numpy(dtype="datetime64[ns]")
    ds_plume = ds_plume.assign_coords(insitu_times = in_situ_times)
    ds_plume = ds_plume.assign(
        no2_insitu = (["insitu_times"], in_situ_no2),
        no_insitu = (["insitu_times"], df_insitu["n_no"][in_situ_mask]),
        nox_insitu = (["insitu_times"], df_insitu["c_nox"][in_situ_mask]),
        co2_insitu = (["insitu_times"], df_insitu["c_co2"][in_situ_mask]),
        o3_insitu = (["insitu_times"], df_insitu["c_o3"][in_situ_mask]),
        so2_insitu = (["insitu_times"], df_insitu["c_so2"][in_situ_mask]),
        n_air_insitu = (["insitu_times"], n_air),
        n_air_insitu_interp = (["window_plume"], n_air_aligned),        
        p_0_insitu = (["insitu_times"], df_insitu["p_0"][in_situ_mask]),
        T_in_insitu = (["insitu_times"], df_insitu["T_in"][in_situ_mask]),
        T_out_insitu = (["insitu_times"], df_insitu["T_out"][in_situ_mask]),
        wind_dir_insitu = (["insitu_times"], df_insitu["wind_dir"][in_situ_mask]),
        wind_speed_insitu = (["insitu_times"], df_insitu["wind_speed"][in_situ_mask]),
    )

    return ds_plume


def detect_plume_ztest(
    image,
    bg_mean=None,
    bg_std=None,
    p_threshold=0.15,
    min_cluster_size=20,
    kernel_arm=1,
    median_kernel_arm=None,
    connectivity=1,
    # new options for source-connection checking
    require_connection=False,
    ds_plume=None,
    time_tol_seconds=30,
    viewdir_min=8,
    viewdir_max=18,
    # option to keep second largest cluster
    keep_second_largest=False,
    second_size_threshold=None,
):
    """
    Detect plume pixels using the neighborhood Z-test described (Kuhlmann et al. 2019).

    New parameters:
    - require_connection: if True, at least one kept cluster must intersect the source region.
    - t0: central timestamp of source event (datetime or numeric same units as time_grid).
    - time_grid: 2D array (same shape as image) with timestamp per pixel (datetime64 or numeric seconds).
    - viewdir_grid: 2D array (same shape as image) with viewing direction per pixel (numeric).
    - time_tol_seconds: tolerance around t0 (seconds) to define source-region in time.
    - viewdir_min/viewdir_max: viewing direction window to define source-region.
    - keep_second_largest: if True, also include second largest cluster when it exceeds second_size_threshold.
    - second_size_threshold: required size (pixels) for second largest cluster to be kept. If None defaults to min_cluster_size.
    """
    kernel_arm=1
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    if bg_mean is None:
        bg_mean = np.nanmean(image)
    if bg_std is None:
        bg_std = np.nanstd(image, ddof=0)
    bg_mean = np.asarray(bg_mean)
    bg_std = np.asarray(bg_std)

    # build cross-shaped footprint for neighborhood mean
    size = 2 * kernel_arm + 1
    footprint = np.zeros((size, size), dtype=bool)
    c = kernel_arm
    footprint[c, c] = True
    for a in range(1, kernel_arm + 1):
        footprint[c + a, c] = True
        footprint[c - a, c] = True
        footprint[c, c + a] = True
        footprint[c, c - a] = True

    # compute neighborhood mean while ignoring NaNs
    def local_mean(arr, footprint):
        def _func(values):
            vals = values[~np.isnan(values)]
            return vals.mean() if vals.size else np.nan
        return ndimage.generic_filter(arr, _func, footprint=footprint, mode="constant", cval=np.nan)

    neigh_mean = local_mean(image, footprint)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (neigh_mean - bg_mean) / bg_std
    pvals = norm.cdf(-z)

    mask = (pvals <= p_threshold) & (~np.isnan(pvals))

    if median_kernel_arm is None:
        median_kernel_arm = kernel_arm + 1
    size_m = 2 * median_kernel_arm + 1
    footprint_m = np.zeros((size_m, size_m), dtype=bool)
    cm = median_kernel_arm
    footprint_m[cm, cm] = True
    for a in range(1, median_kernel_arm + 1):
        footprint_m[cm + a, cm] = True
        footprint_m[cm - a, cm] = True
        footprint_m[cm, cm + a] = True
        footprint_m[cm, cm - a] = True

    #mask_smoothed = ndimage.median_filter(mask.astype(np.uint8), footprint=footprint_m, mode="constant", cval=0)
    #mask = mask_smoothed.astype(bool)

    labeled, ncomp = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity))
    if ncomp == 0:
        return mask  # empty
    
    #sort out all clusters smaller than min_cluster_size
    sizes = ndimage.sum(mask, labeled, range(1, ncomp + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]
    if not keep_labels:
        return np.zeros_like(mask, dtype=bool)

    if require_connection:
            # helper: build source_mask if required
        source_mask = None
        # assume times_plume is 1D (columns) and image_row is 1D (rows)
        times_1d = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
        t0_dt = pd.to_datetime(ds_plume.attrs.get("t"), utc=True)

        # 1D boolean masks
        time_ok_1d = np.abs(times_1d - t0_dt) <= pd.Timedelta(seconds=time_tol_seconds)
        vd_1d = np.asarray(ds_plume.image_row)
        view_ok_1d = (vd_1d >= viewdir_min) & (vd_1d <= viewdir_max)

        # broadcast to image shape: time -> columns, view -> rows
        try:
            time_ok = np.broadcast_to(time_ok_1d[None, :], image.shape)   # shape (rows, cols)
            view_ok = np.broadcast_to(view_ok_1d[:, None], image.shape)   # shape (rows, cols)
        except Exception:
            # fallback if broadcast_to fails (shouldn't), use tile
            time_ok = np.tile(time_ok_1d[None, :], (image.shape[0], 1))
            view_ok = np.tile(view_ok_1d[:, None], (1, image.shape[1]))

        source_mask = time_ok & view_ok

        if source_mask.shape != image.shape:
            raise ValueError("time_grid and viewdir_grid must have same shape as image")
    # filter clusters by min size first
    mask_filtered = np.isin(labeled, keep_labels)

    kept_sizes = [(lab, s) for lab, s in zip(range(1, ncomp + 1), sizes) if lab in keep_labels]
    if not kept_sizes:
        return np.zeros_like(mask, dtype=bool)

    # sort kept clusters by size descending
    kept_sizes_sorted = sorted(kept_sizes, key=lambda x: x[1], reverse=True)
    largest_label = kept_sizes_sorted[0][0]
    largest_size = kept_sizes_sorted[0][1]

    # default final mask = largest cluster
    final_mask = (labeled == largest_label)

    # if second largest requested, note its label/size
    sec_label = sec_size = None
    if len(kept_sizes_sorted) > 1:
        sec_label, sec_size = kept_sizes_sorted[1]
    second_threshold = second_size_threshold if second_size_threshold is not None else min_cluster_size

    # if connection to source is required, enforce rules described:
    if require_connection:
        # list of kept labels that intersect the source region
        intersecting_labels = [lab for lab, _ in kept_sizes_sorted if np.any((labeled == lab) & source_mask)]

        # if no kept cluster intersects source region -> no contour
        if not intersecting_labels:
            return np.zeros_like(mask, dtype=bool)

        # if largest cluster intersects source -> plot it (and optionally second largest)
        if largest_label in intersecting_labels:
            final_mask = (labeled == largest_label)
            if keep_second_largest and sec_label is not None and sec_size >= second_threshold:
                final_mask = final_mask | (labeled == sec_label)
            return final_mask

        # largest cluster does NOT intersect source region, but at least one cluster does
        # pick the intersecting cluster with largest size (closest/most significant in source region)
        # build dict for sizes to pick the largest intersecting cluster
        sizes_dict = {lab: s for lab, s in kept_sizes_sorted}
        best_intersect_label = max(intersecting_labels, key=lambda L: sizes_dict.get(L, 0))

        # plot the intersecting cluster instead (discard largest)
        final_mask = (labeled == best_intersect_label)

        # if keep_second_largest requested, also include the original largest cluster
        if keep_second_largest:
            final_mask = final_mask | (labeled == largest_label)

        return final_mask

    # if require_connection is False: keep previous behavior (largest, optionally second largest)
    if keep_second_largest and sec_label is not None and sec_size >= second_threshold:
        final_mask = final_mask | (labeled == sec_label)

    return final_mask

def detect_plume_ztest_left(
    image,
    bg_mean=None,
    bg_std=None,
    p_threshold=0.3,
    min_cluster_size=10,
    kernel_arm=0,
    median_kernel_arm=None,
    connectivity=1,
):
    """
    Left-sided neighborhood Z-test to detect negative enhancements.
    H0: neigh_mean >= bg_mean
    H1: neigh_mean < bg_mean
    Reject H0 when p <= p_threshold (p = CDF(z)).
    Returns: final_mask, ncomp, labeled, sizes, mask_filtered, pvals, mask_ini
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    if bg_mean is None:
        bg_mean = np.nanmean(image)
    if bg_std is None:
        bg_std = np.nanstd(image, ddof=0)
    bg_mean = np.asarray(bg_mean)
    bg_std = np.asarray(bg_std)
    
    size = 2 * kernel_arm + 1
    footprint = np.zeros((size, size), dtype=bool)
    c = kernel_arm
    footprint[c, c] = True
    for a in range(1, kernel_arm + 1):
        footprint[c + a, c] = True
        footprint[c - a, c] = True
        footprint[c, c + a] = True
        footprint[c, c - a] = True

    def local_mean(arr, footprint):
        def _func(values):
            vals = values[~np.isnan(values)]
            return vals.mean() if vals.size else np.nan
        return ndimage.generic_filter(arr, _func, footprint=footprint, mode="constant", cval=np.nan)

    neigh_mean = local_mean(image, footprint)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (neigh_mean - bg_mean) / bg_std

    # left-sided p-value: p = CDF(z)
    pvals = norm.cdf(z)

    mask_ini = (pvals <= p_threshold) & (~np.isnan(pvals))
    mask = mask_ini

    if median_kernel_arm is None:
        median_kernel_arm = kernel_arm + 1
    size_m = 2 * median_kernel_arm + 1
    footprint_m = np.zeros((size_m, size_m), dtype=bool)
    cm = median_kernel_arm
    footprint_m[cm, cm] = True
    for a in range(1, median_kernel_arm + 1):
        footprint_m[cm + a, cm] = True
        footprint_m[cm - a, cm] = True
        footprint_m[cm, cm + a] = True
        footprint_m[cm, cm - a] = True

    mask_smoothed = ndimage.median_filter(mask.astype(np.uint8), footprint=footprint_m, mode="constant", cval=0)


    labeled, ncomp = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity))
    if ncomp == 0:
        #print("no ship pixels detected after filtering")
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, ncomp + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]
    if not keep_labels:
        #print("no ship pixels detected after filtering")
        return np.zeros_like(mask, dtype=bool)

    mask_filtered = np.isin(labeled, keep_labels)
    sizes_keep = [(lab, s) for lab, s in zip(range(1, ncomp + 1), sizes) if (lab in keep_labels)]
    largest_label = max(sizes_keep, key=lambda x: x[1])[0]
    final_mask = (labeled == largest_label)
    if final_mask.any() == None:
        #print("no ship pixels detected after filtering")
        return np.zeros(image.shape, dtype=bool)
    return final_mask

import matplotlib.pyplot as plt

def sort_plumes(ds_plume, out_dir, date, p_threshold_plume=0.15, p_threshold_ship=0.3):
    if "no2_enhancement_interp" not in ds_plume:
        print("no NO2_enhancement_interp in dataset")
        return ds_plume
    slice_rows = slice(8,20)
    tol = pd.Timedelta("50s")
    times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
    t0 = pd.to_datetime(ds_plume.attrs.get("t"), utc=True)
    win_mask = (times >= (t0 - tol)) & (times <= (t0 + tol))
    idx = np.nonzero(win_mask)[0]
    image_cut_plume = ds_plume["no2_enhancement_interp"].isel(image_row = slice_rows, window_plume = idx).values
    mask = detect_plume_ztest(image_cut_plume, bg_mean = ds_plume["no2_enhancement_interp"].mean().values, bg_std= ds_plume["no2_enhancement_interp"].std().values, p_threshold=p_threshold_plume)

    image_full = ds_plume["no2_enhancement_interp"].values

    # create full-size mask and put the detected mask into the correct location
    mask_full = np.zeros_like(image_full, dtype=bool)
    # slice_rows is a slice object, idx is the 1D array of time/column indices
    mask_full[slice_rows, idx] = mask

    # plot image with plume boundary overlay using time on x-axis and VEA on y-axis
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(8, 6))
    # time coordinate (columns) and VEA (rows)
    times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
    ny, nx = image_full.shape
    # build xedges from times
    xnum = mdates.date2num(times.to_pydatetime())
    dx = np.diff(xnum)
    if len(dx) > 0:
        left = xnum[0] - dx[0] / 2.0
        right = xnum[-1] + dx[-1] / 2.0
        xedges = np.concatenate(([left], xnum[:-1] + dx / 2.0, [right]))
    else:
        xedges = np.array([xnum[0] - 0.5, xnum[0] + 0.5])
    yedges = np.arange(ny + 1)

    mesh = ax.pcolormesh(xedges, yedges, image_full, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="NO$_2$ enhancement")

    # draw boundary around the mask region using mesh centres
    xcent = (xedges[:-1] + xedges[1:]) / 2.0
    ycent = (yedges[:-1] + yedges[1:]) / 2.0
    Xc, Yc = np.meshgrid(xcent, ycent)
    ax.contour(Xc, Yc, mask_full.astype(int), levels=[0.5], colors="red", linewidths=1.5)

    ax.set_title("NO$_2$ enhancement with plume mask boundary (placed at correct location)")
    ax.set_xlabel("Time (UTC)")
    vea_vals = ds_plume["vea"].values 
    yticks = np.arange(0, len(vea_vals), 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
    ax.set_ylabel("VEA / °")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("VEA / °")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    plume_found = False
    ship_found = False
    if mask_full.sum() > 0:
        plume_found = True
        savepath = out_dir / f"plume_detection" / f"plumes_{date}" / f"ship_yes" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_plume.png"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    plt.close('all')
    if plume_found == False:
        slice_rows = slice(4,10)
        tol = pd.Timedelta("50s")
        times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
        t0 = pd.to_datetime(ds_plume.attrs.get("t"), utc=True)
        win_mask = (times >= (t0 - tol)) & (times <= (t0 + tol))
        idx = np.nonzero(win_mask)[0]
        image_cut_ship = ds_plume["no2_enhancement_interp"].isel(image_row = slice_rows, window_plume = idx).values
        mask = detect_plume_ztest_left(image_cut_ship, bg_mean = ds_plume["no2_enhancement_interp"].mean().values, bg_std= ds_plume["no2_enhancement_interp"].std().values, p_threshold=p_threshold_ship, min_cluster_size=20)

        image_full = ds_plume["no2_enhancement_interp"].values

        # create full-size mask and put the detected mask into the correct location
        mask_full = np.zeros_like(image_full, dtype=bool)
        # slice_rows is a slice object, idx is the 1D array of time/column indices
        mask_full[slice_rows, idx] = mask

        # plot image with plume boundary overlay using time on x-axis and VEA on y-axis
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(8, 6))
        times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
        ny, nx = image_full.shape
        xnum = mdates.date2num(times.to_pydatetime())
        dx = np.diff(xnum)
        if len(dx) > 0:
            left = xnum[0] - dx[0] / 2.0
            right = xnum[-1] + dx[-1] / 2.0
            xedges = np.concatenate(([left], xnum[:-1] + dx / 2.0, [right]))
        else:
            xedges = np.array([xnum[0] - 0.5, xnum[0] + 0.5])
        yedges = np.arange(ny + 1)

        mesh = ax.pcolormesh(xedges, yedges, image_full, cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="NO$_2$ enhancement")

        xcent = (xedges[:-1] + xedges[1:]) / 2.0
        ycent = (yedges[:-1] + yedges[1:]) / 2.0
        Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, mask_full.astype(int), levels=[0.5], colors="blue", linewidths=1.5)

        ax.set_title("NO$_2$ enhancement with plume mask boundary (placed at correct location)")
        ax.set_xlabel("Time (UTC)")
        vea_vals = ds_plume["vea"].values 
        yticks = np.arange(0, len(vea_vals), 3)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
        ax.set_ylabel("VEA / °")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("VEA / °")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        if mask_full.sum() == 0:
            savepath = out_dir / f"plume_detection" / f"plumes_{date}" / f"ship_no" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_out.png"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
        else:
            ship_found = True
            savepath = out_dir / f"plume_detection" / f"plumes_{date}" / f"ship_yes" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_ship.png"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
    plt.close('all')
    ds_plume.attrs["plume_or_ship_found"] = str(bool(plume_found or ship_found))
    return ds_plume