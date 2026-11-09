#%%
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import os
import warnings
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
import SEICOR.in_situ
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
    # built in row order, not by index label: two ships passing in the same second share a time index
    ship_passes['plume_file'] = [
        os.path.join(plumes_out_dir, f"plume_{passing_ship['Plume_number']:03d}_t_{idx.strftime('%Y%m%d_%H%M%S')}_mmsi_{passing_ship['MMSI']}.nc")
        for idx, passing_ship in ship_passes.iterrows()
    ]
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
    if df_insitu is None or df_insitu.empty:
        ds_plume.attrs["insitu_available"] = 0
        return ds_plume
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
    if not in_situ_mask.any():
        ds_plume.attrs["insitu_available"] = 0
        return ds_plume
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

    # meteomast wind at all heights (if read in by read_in_situ)
    mast_heights = [h for h in SEICOR.in_situ.MAST_HEIGHTS if f"wind_speed_mast_{h}" in df_insitu.columns]
    if mast_heights:
        ds_plume = ds_plume.assign_coords(mast_height=("mast_height", mast_heights, {"units": "m"}))
        ds_plume = ds_plume.assign(
            wind_speed_mast=(["mast_height", "insitu_times"], np.stack([df_insitu[f"wind_speed_mast_{h}"][in_situ_mask].to_numpy() for h in mast_heights])),
            wind_dir_mast=(["mast_height", "insitu_times"], np.stack([df_insitu[f"wind_dir_mast_{h}"][in_situ_mask].to_numpy() for h in mast_heights])),
        )

    ds_plume.attrs["insitu_available"] = 1
    return ds_plume


def _local_nanmean(arr, footprint):
    """Mean over the footprint around each pixel, ignoring NaNs and pixels outside the image;
    NaN where the footprint contains no valid pixel."""
    weights = footprint.astype(float)
    valid = ~np.isnan(arr)
    total = ndimage.convolve(np.where(valid, arr, 0.0), weights, mode="constant", cval=0.0)
    count = ndimage.convolve(valid.astype(float), weights, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def _merge_nearby_clusters(labeled, keep_labels, seed_label, seconds, max_gap_seconds, max_row_distance, max_extension_seconds=None):
    """Give seed_label to the clusters connected to it: on rows at most max_row_distance apart, they are
    at most max_gap_seconds apart in time (seconds: time of each column), also as a chain over other clusters.
    A cluster reaching more than max_extension_seconds before or after the seed cluster is not merged,
    and the chain does not continue over it."""
    keep_labels = np.asarray(keep_labels)
    n_rows = labeled.shape[0]
    first = np.full((keep_labels.size, n_rows), np.inf)
    last = np.full((keep_labels.size, n_rows), -np.inf)
    for i, lab in enumerate(keep_labels):
        rows, cols = np.nonzero(labeled == lab)
        np.minimum.at(first[i], rows, seconds[cols])
        np.maximum.at(last[i], rows, seconds[cols])

    # time distance of each pair of clusters: smallest over all pairs of similar rows, <= 0 if they overlap in time
    close = np.zeros((keep_labels.size, keep_labels.size), dtype=bool)
    for shift in range(-max_row_distance, max_row_distance + 1):
        rows_a = np.arange(max(0, -shift), min(n_rows, n_rows - shift))
        rows_b = rows_a + shift
        distance = np.maximum(first[:, None, rows_b] - last[None, :, rows_a],
                              first[None, :, rows_a] - last[:, None, rows_b])
        close |= (distance <= max_gap_seconds).any(axis=2)

    seed = int(np.flatnonzero(keep_labels == seed_label)[0])
    if max_extension_seconds is not None:
        within = (first.min(axis=1) >= first[seed].min() - max_extension_seconds) & (last.max(axis=1) <= last[seed].max() + max_extension_seconds)
        close &= within[:, None] & within[None, :]

    _, groups = connected_components(csr_matrix(close), directed=False)
    merged = labeled.copy()
    merged[np.isin(labeled, keep_labels[groups == groups[seed]])] = seed_label
    return merged


def _detect_plume_ztest_core(
    image,
    bg_mean,
    bg_std,
    p_threshold=0.15,
    min_cluster_size=20,
    kernel_arm=1,
    median_kernel_arm=None,
    connectivity=1,
    # new options for source-connection checking
    require_connection=False,
    ds_plume=None,
    time_tol_seconds=60,
    viewdir_min=8,
    viewdir_max=18,
    keep_second_largest=False,
    second_size_threshold=None,
    # option to merge clusters close in time on similar rows (needs ds_plume for the times)
    merge_max_gap_seconds=None,
    merge_max_row_distance=2,
    merge_min_cluster_size=40,
    merge_max_cluster_size=None,
    merge_max_extension_minutes=None,
    exclude_mask=None,
):
    """Right-sided neighborhood Z-test (plume pixels) with cluster selection.

    Pixels in exclude_mask (e.g. the ship mask) do not enter the neighborhood means and
    cannot become plume pixels.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    bg_mean = np.asarray(bg_mean)
    bg_std = np.asarray(bg_std)
    if exclude_mask is not None:
        image = np.where(np.asarray(exclude_mask, dtype=bool), np.nan, np.asarray(image, dtype=float))

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

    neigh_mean = _local_nanmean(image, footprint)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (neigh_mean - bg_mean) / bg_std
    pvals = norm.cdf(-z)

    mask = (pvals <= p_threshold) & (~np.isnan(pvals))
    if exclude_mask is not None:
        mask &= ~np.asarray(exclude_mask, dtype=bool)  # the neighborhood mean of an excluded pixel is still defined

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

    # merge the neighbouring clusters into the cluster that the selection below would pick without merging
    if merge_max_gap_seconds is not None:
        seed_labels = [lab for lab in keep_labels if not require_connection or np.any((labeled == lab) & source_mask)]
        max_size = np.inf if merge_max_cluster_size is None else merge_max_cluster_size
        merge_labels = [lab for lab in keep_labels if merge_min_cluster_size <= sizes[lab - 1] <= max_size]
        seed_label = max(seed_labels, key=lambda lab: sizes[lab - 1], default=None)
        if seed_label in merge_labels and len(merge_labels) > 1:
            times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
            seconds = np.asarray((times - times[0]).total_seconds(), dtype=float)
            max_extension_seconds = None if merge_max_extension_minutes is None else 60.0 * merge_max_extension_minutes
            labeled = _merge_nearby_clusters(labeled, merge_labels, seed_label, seconds, merge_max_gap_seconds,
                                             merge_max_row_distance, max_extension_seconds)
            sizes = ndimage.sum(mask, labeled, range(1, ncomp + 1))
            keep_labels = np.unique(labeled[np.isin(labeled, keep_labels)]).tolist()

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
    time_tol_seconds=60,
    viewdir_min=8,
    viewdir_max=18,
    # option to keep second largest cluster
    keep_second_largest=False,
    second_size_threshold=None,
):
    """
    Detect plume pixels using the neighborhood Z-test described (Kuhlmann et al. 2019).

    This is the original single-pass routine. If bg_mean/bg_std are not provided,
    they are computed once from the full image.
    """
    if bg_mean is None:
        bg_mean = np.nanmean(image)
    if bg_std is None:
        bg_std = np.nanstd(image, ddof=0)

    return _detect_plume_ztest_core(
        image,
        bg_mean=bg_mean,
        bg_std=bg_std,
        p_threshold=p_threshold,
        min_cluster_size=min_cluster_size,
        kernel_arm=kernel_arm,
        median_kernel_arm=median_kernel_arm,
        connectivity=connectivity,
        require_connection=require_connection,
        ds_plume=ds_plume,
        time_tol_seconds=time_tol_seconds,
        viewdir_min=viewdir_min,
        viewdir_max=viewdir_max,
        keep_second_largest=keep_second_largest,
        second_size_threshold=second_size_threshold,
    )


def detect_plume_ztest_iterative(
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
    time_tol_seconds=60,
    viewdir_min=8,
    viewdir_max=18,
    # option to keep second largest cluster
    keep_second_largest=False,
    second_size_threshold=None,
    ship_p_threshold=0.15,
    ship_min_cluster_size=20,
    ship_kernel_arm=0,
    ship_median_kernel_arm=None,
    ship_connectivity=1,
    max_iter=10,
    stop_pixels=15,
    bg_extra=None,
    verbose=False,
    merge_max_gap_seconds=None,
    merge_max_row_distance=2,
    merge_min_cluster_size=40,
    merge_max_cluster_size=None,
    merge_max_extension_minutes=None,
    bg_offset_from_window=False,
):
    """
    Iterative neighborhood Z-test.

    Each iteration first detects ship pixels (left-sided test, detect_plume_ztest_left) and
    then plume pixels (right-sided test, _detect_plume_ztest_core) on the original image,
    with the ship pixels left out of the neighborhood means of the plume test. The
    background mean/std of the next iteration are then estimated from the image with the
    union of plume and ship pixels masked out, plus the pixels in bg_extra (e.g. the
    reference images). So the ship pixels never enter the background statistics or the
    plume test. The loop stops when the union mask changes by at most stop_pixels,
    nothing is detected, or max_iter is reached.

    If bg_mean/bg_std are given, they are used for the first iteration only (e.g. the
    statistics of the reference images); otherwise the first iteration estimates them from
    the full image (plus bg_extra), finds the ship with them and re-estimates them without
    the ship pixels before the plume test.

    bg_offset_from_window: the background level is taken from the image itself, so that an
    offset of the whole image against the references (bg_extra) is background, not plume.
    The first plume test then uses the median of the image without the ship pixels as
    background mean (bg_std as given). Later iterations take the mean only from the image
    pixels outside plume and ship, and the std pooled from these pixels and bg_extra, each
    relative to its own mean; the bg_mean > bg_std guard is not applied.

    Returns:
        plume_mask, ship_mask, plume_masks, ship_masks
        - plume_mask: plume mask from the last iteration
        - ship_mask: ship mask from the last iteration
        - plume_masks: list of plume masks from each iteration
        - ship_masks: list of ship masks from each iteration
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    image = np.asarray(image, dtype=float)
    extra = np.array([]) if bg_extra is None else np.asarray(bg_extra, dtype=float).ravel()

    def _bg_stats(excluded):
        if bg_offset_from_window:
            window = image[~excluded].ravel()
            if not np.isfinite(window).any():
                return current_bg_mean, current_bg_std  # everything masked: keep the statistics
            mean = np.nanmean(window)
            residuals = np.concatenate([window - mean, extra - np.nanmean(extra) if extra.size else extra])
            return mean, np.nanstd(residuals, ddof=0)
        pixels = np.concatenate([image[~excluded].ravel(), extra])
        mean, std = np.nanmean(pixels), np.nanstd(pixels, ddof=0)
        # guard: a background mean above the noise level means the plume is not masked well
        # enough yet; use 0 to avoid skewing the Z-test
        if np.isfinite(mean) and np.isfinite(std) and mean > std:
            if verbose:
                print(f"Iterative Z-test: bg_mean ({mean:.2e}) > bg_std ({std:.2e}), using bg_mean = 0")
            mean = 0.0
        return mean, std

    no_mask = np.zeros(image.shape, dtype=bool)

    def _initial_stats(excluded):
        est_mean, est_std = _bg_stats(excluded)
        return (est_mean if bg_mean is None else bg_mean), (est_std if bg_std is None else bg_std)

    estimate_initial = bg_mean is None or bg_std is None
    if estimate_initial:
        current_bg_mean, current_bg_std = _initial_stats(no_mask)
    else:
        current_bg_mean, current_bg_std = bg_mean, bg_std

    previous_mask = None
    plume_mask = ship_mask = no_mask
    plume_masks = []
    ship_masks = []

    for i in range(max(int(max_iter), 1)):
        # ship first, so that its pixels can be left out of the plume test
        ship_mask = np.asarray(detect_plume_ztest_left(
            image,
            bg_mean=current_bg_mean,
            bg_std=current_bg_std,
            p_threshold=ship_p_threshold,
            min_cluster_size=ship_min_cluster_size,
            kernel_arm=ship_kernel_arm,
            median_kernel_arm=ship_median_kernel_arm,
            connectivity=ship_connectivity,
        ), dtype=bool)
        if i == 0 and bg_offset_from_window:
            current_bg_mean = np.nanmedian(image[~ship_mask])  # background level of this image, robust to the plume
        elif i == 0 and estimate_initial and ship_mask.any():
            current_bg_mean, current_bg_std = _initial_stats(ship_mask)  # statistics without the ship pixels
        plume_mask = np.asarray(_detect_plume_ztest_core(
            image,
            bg_mean=current_bg_mean,
            bg_std=current_bg_std,
            p_threshold=p_threshold,
            min_cluster_size=min_cluster_size,
            kernel_arm=kernel_arm,
            median_kernel_arm=median_kernel_arm,
            connectivity=connectivity,
            require_connection=require_connection,
            ds_plume=ds_plume,
            time_tol_seconds=time_tol_seconds,
            viewdir_min=viewdir_min,
            viewdir_max=viewdir_max,
            keep_second_largest=keep_second_largest,
            second_size_threshold=second_size_threshold,
            merge_max_gap_seconds=merge_max_gap_seconds,
            merge_max_row_distance=merge_max_row_distance,
            merge_min_cluster_size=merge_min_cluster_size,
            merge_max_cluster_size=merge_max_cluster_size,
            merge_max_extension_minutes=merge_max_extension_minutes,
            exclude_mask=ship_mask,
        ), dtype=bool)
        final_mask = plume_mask | ship_mask
        plume_masks.append(plume_mask.copy())
        ship_masks.append(ship_mask.copy())

        if verbose:
            print(f"Iterative Z-test: iteration {i}, bg_mean {float(current_bg_mean):.2e}, "
                  f"bg_std {float(current_bg_std):.2e}, plume px {plume_mask.sum()}, ship px {ship_mask.sum()}")

        if previous_mask is not None and np.count_nonzero(final_mask != previous_mask) <= stop_pixels:
            break  # converged (also when nothing is found twice in a row)

        previous_mask = final_mask.copy()
        current_bg_mean, current_bg_std = _bg_stats(final_mask)

    return plume_mask, ship_mask, plume_masks, ship_masks


def plume_mask_of(ds, plume_mask_settings):
    """Iterative z-test plume mask (and ship mask) of the enhancement image of ds: first pass with
    the statistics of the upwind (and downwind) reference images, then the background is
    re-estimated from the reference pixels plus the plume-window pixels outside plume and ship.
    plume_mask_settings is the processing/plume_mask section of the preprocessor settings."""
    enh_var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
    ref_image = ds["no2_ref"] - ds["no2_ref"].mean(dim="window_ref")
    if "no2_ref_down" in ds:
        ref_image_down = ds["no2_ref_down"] - ds["no2_ref_down"].mean(dim="window_ref_down")
        ref_image_down = ref_image_down.rename({"window_ref_down": "window_ref"})  # unify dimension name
        ref_image = xr.concat([ref_image, ref_image_down], dim="window_ref", join="outer")
    mask, ship_mask, _, _ = detect_plume_ztest_iterative(
        ds[enh_var].values,
        bg_mean=ref_image.mean().values,
        bg_std=ref_image.std().values,
        bg_extra=ref_image.values if plume_mask_settings["bg_include_references"] else None,
        p_threshold=plume_mask_settings["p_threshold"],
        min_cluster_size=plume_mask_settings["min_cluster_size"],
        connectivity=plume_mask_settings["connectivity"],
        kernel_arm=plume_mask_settings["kernel_arm"],
        require_connection=plume_mask_settings["require_connection"],
        ds_plume=ds,
        keep_second_largest=False,
        second_size_threshold=100,
        ship_p_threshold=plume_mask_settings["ship_p_threshold"],
        ship_min_cluster_size=plume_mask_settings["ship_min_cluster_size"],
        max_iter=plume_mask_settings["max_iter"],
        stop_pixels=plume_mask_settings["stop_pixels"],
        merge_max_gap_seconds=plume_mask_settings["merge_max_gap_seconds"],
        merge_max_row_distance=plume_mask_settings["merge_max_row_distance"],
        merge_min_cluster_size=plume_mask_settings["merge_min_cluster_size"],
        merge_max_cluster_size=plume_mask_settings["merge_max_cluster_size"],
        merge_max_extension_minutes=plume_mask_settings["merge_max_extension_minutes"],
        bg_offset_from_window=plume_mask_settings.get("bg_offset_from_window", False),
    )
    return np.asarray(mask, dtype=bool), np.asarray(ship_mask, dtype=bool)


def background_of(ds, mask, ship_mask):
    """Mean and std of the enhancement image of ds (as used by plume_mask_of) outside the plume and ship masks."""
    enh_var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
    outside = ~(np.asarray(mask, dtype=bool) | np.asarray(ship_mask, dtype=bool))
    pixels = ds[enh_var].values[outside]
    if not np.isfinite(pixels).any():
        return np.nan, np.nan
    return float(np.nanmean(pixels)), float(np.nanstd(pixels))


def subtract_background(ds, mask, ship_mask, min_background_fraction=0.3):
    """Correct the enhancement image of ds (as used by plume_mask_of) for the background left in it.

    The background is the mean of the image outside the plume and ship masks. A warning is
    raised if it is larger than its std (the reference does not match the background of the
    plume window). The mean is subtracted from the image, and the vertical integral is
    recomputed, only if at least min_background_fraction of the pixels lie outside plume and
    ship; otherwise the image stays as it is.

    The uncorrected image and integral are kept as <name>_uncorrected; attrs background_mean,
    background_std, background_fraction and background_corrected (1/0) record the correction.
    """
    enh_var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
    int_var = "vertically_integrated_" + enh_var
    outside = ~(np.asarray(mask, dtype=bool) | np.asarray(ship_mask, dtype=bool))
    fraction = float(outside.mean())
    mean, std = background_of(ds, mask, ship_mask)
    ds.attrs.update(background_mean=mean, background_std=std, background_fraction=fraction, background_corrected=0)
    if mean > std:
        warnings.warn(f"MMSI {ds.attrs.get('mmsi')} at {ds.attrs.get('t')}: background mean {mean:.2e} > std {std:.2e} "
                      f"outside plume and ship mask ({fraction:.0%} of the pixels)")
    if fraction < min_background_fraction or not np.isfinite(mean):
        return ds
    ds[enh_var + "_uncorrected"] = ds[enh_var].copy()
    ds[enh_var] = ds[enh_var] - mean
    if int_var in ds:
        ds[int_var + "_uncorrected"] = ds[int_var].copy()
        ds[int_var] = ds[enh_var].sum(dim="image_row")
    ds.attrs["background_corrected"] = 1
    return ds


def reference_hits_plume(ds, mask):
    """True if a reference window overlaps the time span of the masked plume."""
    if not mask.any():
        return False
    plume_times = pd.to_datetime(ds["times_plume"].values)[mask.any(axis=0)]
    t_min, t_max = plume_times.min(), plume_times.max()
    for time_var in ("times_ref", "times_ref_down"):
        if time_var in ds:
            ref_times = pd.to_datetime(ds[time_var].values)
            if ((ref_times >= t_min) & (ref_times <= t_max)).any():
                return True
    return False


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

    neigh_mean = _local_nanmean(image, footprint)

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

def sort_plumes(ds_plume, out_dir, date, p_threshold_plume=0.15, p_threshold_ship=0.3,
                time_tolerance_seconds=50, plume_rows=(8, 20), ship_rows=(4, 10),
                plume_min_cluster_size=20, ship_min_cluster_size=20):
    """Flags whether a plume or, if not, the ship itself is found close to the pass time.

    The plume is searched in image rows plume_rows[0]:plume_rows[1], the ship in
    ship_rows[0]:ship_rows[1], both within +-time_tolerance_seconds of the pass time, as
    clusters of at least plume_min_cluster_size / ship_min_cluster_size pixels.
    Rows are the original IMPACT viewing direction numbers (image_row coordinate), so they
    keep pointing at the same elevations when the lowest viewing directions are dropped.
    Sets ds_plume.attrs["plume_or_ship_found"] to "True"/"False".
    """
    from SEICOR.plotting import set_elevation_yticks, positive_color_limit  # local: plotting imports enhancements, which imports this module

    row_labels = np.asarray(ds_plume["image_row"].values)

    def _row_positions(rows):
        return np.nonzero((row_labels >= rows[0]) & (row_labels < rows[1]))[0]

    # prefer interpolated enhancement, fall back to c_back variant
    if 'no2_enhancement_interp' in ds_plume:
        varname = 'no2_enhancement_interp'
    elif 'no2_enhancement_c_back' in ds_plume:
        varname = 'no2_enhancement_c_back'
    else:
        print('no NO2 enhancement variable (interp or c_back) in dataset')
        return ds_plume
    
    # record which enhancement variable will be used and create an alias
    ds_plume.attrs["enhancement_var_used"] = varname
    if 'no2_enhancement_interp' not in ds_plume:
        try:
            ds_plume = ds_plume.assign(no2_enhancement_interp=ds_plume['no2_enhancement_c_back'])
        except Exception:
            pass

    slice_rows = _row_positions(plume_rows)
    tol = pd.Timedelta(seconds=time_tolerance_seconds)
    times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
    t0 = pd.to_datetime(ds_plume.attrs.get("t"), utc=True)
    win_mask = (times >= (t0 - tol)) & (times <= (t0 + tol))
    idx = np.nonzero(win_mask)[0]
    image_cut_plume = ds_plume[varname].isel(image_row = slice_rows, window_plume = idx).values
    bg_mean_val = ds_plume[varname].mean().values
    bg_std_val = ds_plume[varname].std().values
    mask = detect_plume_ztest(image_cut_plume, bg_mean = bg_mean_val, bg_std= bg_std_val, p_threshold=p_threshold_plume,
                              min_cluster_size=plume_min_cluster_size)

    image_full = ds_plume[varname].values

    # create full-size mask and put the detected mask into the correct location
    mask_full = np.zeros_like(image_full, dtype=bool)
    # slice_rows (row positions) and idx (time/column indices) are 1D index arrays
    mask_full[np.ix_(slice_rows, idx)] = mask

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

    mesh = ax.pcolormesh(xedges, yedges, image_full, cmap="bwr", shading="auto",
                         vmin=-positive_color_limit(image_full), vmax=positive_color_limit(image_full))
    fig.colorbar(mesh, ax=ax, label="NO$_2$ enhancement")

    # draw boundary around the mask region using mesh centres
    xcent = (xedges[:-1] + xedges[1:]) / 2.0
    ycent = (yedges[:-1] + yedges[1:]) / 2.0
    Xc, Yc = np.meshgrid(xcent, ycent)
    ax.contour(Xc, Yc, mask_full.astype(int), levels=[0.5], colors="red", linewidths=1.5)

    set_elevation_yticks(ax, ds_plume["vea"].values, row_offset=0.5)
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
        savepath = out_dir / "plume_or_ship_detection" / f"plume_or_ship_{date}" / "plume_or_ship_yes" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_plume.png"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    plt.close('all')
    if plume_found == False:
        slice_rows = _row_positions(ship_rows)
        tol = pd.Timedelta(seconds=time_tolerance_seconds)
        times = pd.to_datetime(ds_plume["times_plume"].values, utc=True)
        t0 = pd.to_datetime(ds_plume.attrs.get("t"), utc=True)
        win_mask = (times >= (t0 - tol)) & (times <= (t0 + tol))
        idx = np.nonzero(win_mask)[0]
        image_cut_ship = ds_plume[varname].isel(image_row = slice_rows, window_plume = idx).values
        mask = detect_plume_ztest_left(image_cut_ship, bg_mean = bg_mean_val, bg_std= bg_std_val, p_threshold=p_threshold_ship, min_cluster_size=ship_min_cluster_size)

        image_full = ds_plume[varname].values

        # create full-size mask and put the detected mask into the correct location
        mask_full = np.zeros_like(image_full, dtype=bool)
        # slice_rows (row positions) and idx (time/column indices) are 1D index arrays
        mask_full[np.ix_(slice_rows, idx)] = mask

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

        mesh = ax.pcolormesh(xedges, yedges, image_full, cmap="bwr", shading="auto",
                         vmin=-positive_color_limit(image_full), vmax=positive_color_limit(image_full))
        fig.colorbar(mesh, ax=ax, label="NO$_2$ enhancement")

        xcent = (xedges[:-1] + xedges[1:]) / 2.0
        ycent = (yedges[:-1] + yedges[1:]) / 2.0
        Xc, Yc = np.meshgrid(xcent, ycent)
        ax.contour(Xc, Yc, mask_full.astype(int), levels=[0.5], colors="blue", linewidths=1.5)

        set_elevation_yticks(ax, ds_plume["vea"].values, row_offset=0.5)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("VEA / °")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        if mask_full.sum() == 0:
            savepath = out_dir / "plume_or_ship_detection" / f"plume_or_ship_{date}" / "plume_or_ship_no" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_out.png"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
        else:
            ship_found = True
            savepath = out_dir / "plume_or_ship_detection" / f"plume_or_ship_{date}" / "plume_or_ship_yes" / f"plume_{t0.strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_ship.png"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
    plt.close('all')
    ds_plume.attrs["plume_or_ship_found"] = str(bool(plume_found or ship_found))
    return ds_plume
# %%
