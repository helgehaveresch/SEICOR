#%%
#this script processes the input data from IMPACT, LP-DOAS, IN-Situ, AIS and the video camera, to create unified datasets and sanity_check_plots for further analysis.
import shutil
import yaml
import os
import sys
import platform
import traceback
from pathlib import Path

_platform = platform.system().lower()  
if _platform.startswith("win"):
    _default_scripts_dir = Path(r"C:\Users\hhave\Documents\Promotion\scripts")
    _default_settings_path = _default_scripts_dir / "SEICOR" / "plume_preprocessor_settings.yaml"
    _plat_key = "windows"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #enables concurrent reading of hdf5/netcdf files on windows network drives
elif _platform.startswith("linux"):
    _default_scripts_dir = Path("/raid/home2/hhaveresch/scripts")
    _default_settings_path = _default_scripts_dir / "SEICOR" / "plume_preprocessor_settings.yaml"
    _plat_key = "linux"

SCRIPTS_DIR = Path("SCRIPTS_DIR", _default_scripts_dir)
SETTINGS_PATH = Path("PLUME_SETTINGS_PATH", _default_settings_path)

import netCDF4 #oder
import xarray as xr
sys.path.append(str(SCRIPTS_DIR))
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import mask_zenith, correct_destriped_for_noon_reference
from imaging_tools.utilities import get_month_folder
import SEICOR.ais
import SEICOR.in_situ
import SEICOR.lp_doas
import SEICOR.video_camera
import SEICOR.impact
import SEICOR.enhancements
import SEICOR.plotting
import SEICOR.plumes

with open(SETTINGS_PATH, "r") as file:
    settings = yaml.safe_load(file)

BASEPATH                = Path(settings.get("base_paths", {}).get(_plat_key))
date                    = settings["date"]
#if an argument is passed to the script, use that as date
if len(sys.argv) > 1:
    date = sys.argv[1]
instrument_settings     = settings["Instruments"]
processing_settings     = settings["processing"]
IMPACT_path             = (BASEPATH / instrument_settings["IMPACT_SC_path"] / get_month_folder(date))
IMPACT_SC_file_ending   = instrument_settings["IMPACT_SC_file_ending"]
do_ref_correction       = instrument_settings["do_reference_correction"]
SC_ref_offset_corr      = instrument_settings["SC_reference_offset_correction_files"]
instrument_location     = instrument_settings["instrument_location"]
lat1, lon1              = instrument_settings["instrument_location"]
lp_doas_dir             = (BASEPATH / instrument_settings["lp_doas_dir"])
in_situ_path            = (BASEPATH / instrument_settings["in_situ_path"])
ais_dir                 = (BASEPATH / instrument_settings["ais_dir"])
img_dir                 = (BASEPATH / instrument_settings["img_dir"])
weather_stations_dir    = (BASEPATH / instrument_settings["weather_stations_dir"])
ais_settings            = processing_settings["ais_filter"]
lat_lon_window          = ais_settings["lat_lon_window"]  # [lat_min, lat_max, lon_min, lon_max] area of interest for ais data
lat_lon_window_small    = ais_settings["lat_lon_window_small"]
enhancement_settings    = processing_settings["enhancement"]

img_out_dir             = (BASEPATH / settings["Output"]["img_out_dir"] / f"{date}_video_ships")
plumes_out_dir          = (BASEPATH / settings["Output"]["plume_out_dir"] / f"plumes_{date}")
ship_passes_out_dir     = (BASEPATH / settings["Output"]["ship_passes_out_dir"])
out_dir                 = (BASEPATH / settings["Output"]["plots_out_dir"])
#%% Initialize IMPACT measurements
print("Starting plume preprocessing...")
ds_impact = read_SC_file_imaging(IMPACT_path, date, IMPACT_SC_file_ending)
if do_ref_correction:
    ds_ref_corr = read_SC_file_imaging(IMPACT_path, date, SC_ref_offset_corr)
    ds_impact = correct_destriped_for_noon_reference(ds_impact, ds_ref_corr)
ds_impact = mask_zenith(ds_impact)
ds_impact = SEICOR.enhancements.rolling_background_enh(ds_impact, window_size=enhancement_settings["rolling_background_window"])
ds_impact = SEICOR.impact.rolling_impact(ds_impact, window=processing_settings["quality"]["rolling_mean_window"])
endpoints_los = SEICOR.impact.calculate_LOS(ds_impact, instrument_location)
lat2, lon2 = endpoints_los[:, 0], endpoints_los[:, 1] #todo: to be replaced 
start_time, end_time, measurement_times = SEICOR.impact.calc_start_end_times(ds_impact)
#%% Initialize in-situ data
try:
    df_insitu = SEICOR.in_situ.read_in_situ(in_situ_path, date)
    df_insitu = SEICOR.in_situ.apply_time_mask_to_insitu(df_insitu, start_time, end_time)
    ds_impact = SEICOR.impact.calculate_path_averaged_vmr_no2(df_insitu, ds_impact)
except Exception as e:
    print("Error in in-situ processing block:", e)
    traceback.print_exc()
    df_insitu = None
#%% Initialize lp-doas data
try:
    df_lp_doas = SEICOR.lp_doas.read_lpdoas(lp_doas_dir, date)
    df_lp_doas = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])
    df_lp_doas_SC = SEICOR.lp_doas.read_lpdoas(lp_doas_dir, date, mode="SC")
    df_lp_doas_SC = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_SC, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])

    mask, ds_impact_masked = SEICOR.impact.mask_rms_and_reduce_impact(ds_impact, rms_threshold=processing_settings["quality"]["min_rms_IMPACT"])
    ds_impact_masked = SEICOR.enhancements.polynomial_background_enh(ds_impact_masked, degree=enhancement_settings["polynomial_background_degree"])
    ds_impact_masked = SEICOR.enhancements.fft_background_enh(ds_impact_masked, t_cut=enhancement_settings["high_pass_filter_time_period"])
    df_lp_doas_SC = SEICOR.enhancements.polynomial_background_enh_lp_doas(df_lp_doas_SC, degree=enhancement_settings["polynomial_background_degree"])
    df_lp_doas_SC = SEICOR.enhancements.fft_background_enh_lp_doas(df_lp_doas_SC, t_cut=enhancement_settings["high_pass_filter_time_period"])
except Exception as e:
    print("Error in LP-DOAS / IMPACT processing block:", e)
    traceback.print_exc()
    # Fallbacks so downstream cells can still run. Adjust these as needed.
    df_lp_doas = None
    df_lp_doas_SC = None
    mask = None
    # keep ds_impact available if it exists, otherwise None
    ds_impact_masked = ds_impact if 'ds_impact' in locals() else None
#%% Initialize ais data
df_ais = SEICOR.ais.prepare_ais(ais_dir, date, interpolation_limit = ais_settings["interpolation_limit"])
df_ais, ship_groups, filtered_ship_groups = SEICOR.ais.pre_filter_ais(df_ais, lat_lon_window, start_time, end_time, length=ais_settings["max_ship_length"])
#%% Filter ship passes
df_ais, filtered_ship_groups, maskedout_groups, ship_passes = SEICOR.ais.filter_ship_passes(df_ais, 
    ship_groups, filtered_ship_groups, start_time, end_time, measurement_times, endpoints_los, instrument_location, distance_threshold=ais_settings["distance_threshold"])
#%%
ship_passes = SEICOR.in_situ.add_wind_to_ship_passes(ship_passes, df_insitu)
ship_passes = SEICOR.video_camera.assign_video_images_to_ship_pass(ship_passes, img_dir, date)
ship_passes = SEICOR.plumes.add_plume_file_paths_to_ship_passes(ship_passes, plumes_out_dir)
Path(ship_passes_out_dir).mkdir(parents=True, exist_ok=True)
ship_passes.to_csv(ship_passes_out_dir / f"ship_passes_{date}.csv")

#%%
import matplotlib.pyplot as plt
import pandas as pd


#%%
def plot_reference_image_with_plume_mask(ds_plume, out_dir, date, p_threshold=0.2, min_cluster_size = 0.2):
    ref_image = ds_plume["no2_ref"] - ds_plume["no2_ref"].mean(dim="window_ref")
    mask = SEICOR.plumes.detect_plume_ztest(ref_image.values, p_threshold=p_threshold, min_cluster_size=min_cluster_size)
    plt.imshow(ref_image.values, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
    plt.title("Reference image with detected plume pixels")
    if mask.sum() > 0:
        os.makedirs(out_dir / f"reference_quality" / f"plumes_{date}_upwind" / f"yes_plume", exist_ok=True)
        plt.savefig(out_dir / f"reference_quality" / f"plumes_{date}_upwind" / f"yes_plume" / f"reference_image_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
        plt.close('all')
    else:
        os.makedirs(out_dir / f"reference_quality" / f"plumes_{date}_upwind" / f"no_plume", exist_ok=True)
        plt.savefig(out_dir / f"reference_quality" / f"plumes_{date}_upwind" / f"no_plume" / f"reference_image_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
        plt.close('all')

#%%
for idx, ship_pass_single in ship_passes.iterrows():
    ds_plume = SEICOR.enhancements.upwind_constant_background_enh(ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
    if ds_plume is not None:
        plot_reference_image_with_plume_mask(ds_plume, out_dir, date, p_threshold=0.15, min_cluster_size=30)
        ds_plume = SEICOR.plumes.add_ship_trajectory_to_plume_ds(ds_plume, filtered_ship_groups)
        ds_plume = SEICOR.plumes.add_insitu_to_plume_ds(ds_plume, df_insitu)
        #ds_plume = SEICOR.impact.call_nlin_c_for_offaxis_ref_and_add_to_plume_ds(ds_plume, ship_pass_single, settings["processing"]["enhancement"]["nlin_param_file"], IMPACT_path)
        ds_plume = SEICOR.enhancements.upwind_downwind_interp_background_enh(ds_plume, ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
        ds_plume = SEICOR.plumes.sort_plumes(ds_plume, out_dir, p_threshold_plume=0.02, p_threshold_ship=0.02, date=date)
        Path(plumes_out_dir).mkdir(parents=True, exist_ok=True)
        ds_plume.to_netcdf(Path(ship_pass_single['plume_file']), mode="w")

# %%
import numpy as np
from scipy import ndimage
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

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


import xarray as xr
for idx, row in ship_passes.iterrows():
    plume_file = row['plume_file']
    try:
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
            plt.figure(figsize=(10, 6))
            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
            plt.title("No2 enhancement with detected plume pixels")
            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}0", exist_ok=True)
            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}0" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
            plt.close('all')
    except Exception as e:
        print(f"Could not open plume file {plume_file}: {e}")
#
#import xarray as xr
#for idx, row in ship_passes.iterrows():
#    plume_file = row['plume_file']
#    try:
#        ds_plume = xr.open_dataset(plume_file)
#        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
#        if plume_found:
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}0", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}0" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}1", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}1" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}2", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}2" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}3", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}3" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=1, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}3b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}3b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=2, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}4", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}4" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=2, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}5", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}5" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=2, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}6", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}6" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=2, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}7", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}7" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=2, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}7b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}7b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=3, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}8", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}8" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=3, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}9", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}9" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=3, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}10", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}10" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=3, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}11", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}11" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=3, kernel_arm = 1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}11b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}11b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=1, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}12", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}12" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=1, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}13", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}13" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=1, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}14", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}14" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=1, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}15", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}15" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=1, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}15b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}15b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}16", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}16" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}17", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}17" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}18", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}18" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}19", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}19" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}19b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}19b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=3, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}20", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}20" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=3, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}21", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}21" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=3, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}22", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}22" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=3, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}23", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}23" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=3, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}23b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}23b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.15, min_cluster_size=5, connectivity=4, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}24", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}24" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=5, connectivity=4, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}25", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}25" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.25, min_cluster_size=5, connectivity=4, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}26", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}26" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.3, min_cluster_size=5, connectivity=4, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.colorbar()
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}27", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}27" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#            mask = detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.4, min_cluster_size=5, connectivity=4, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#            plt.figure(figsize=(10, 6))
#            plt.imshow(ds_plume["no2_enhancement_interp"].values, origin="lower", aspect="auto", cmap="viridis")
#            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#            plt.title("No2 enhancement with detected plume pixels")
#            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}27b", exist_ok=True)
#            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}27b" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#            plt.close('all')
#        #ds_plume = xr.open_dataset(plume_file)
#        #plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
#        #if plume_found:
#        #    mask = detect_plume_ztest(ds_plume["no2_enhancement_c_back"].values, p_threshold=0.35, min_cluster_size=20, connectivity=2, kernel_arm = 2, require_connection=True, keep_second_largest=False, second_size_threshold=100)
#        #    plt.figure(figsize=(10, 6))
#        #    plt.imshow(ds_plume["no2_enhancement_c_back"].values, origin="lower", aspect="auto", cmap="viridis")
#        #    plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
#        #    plt.title("No2 enhancement with detected plume pixels")
#        #    os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}_c_back", exist_ok=True)
#        #    plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}_c_back" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
#        #    plt.close('all')
#
#    except Exception as e:
#        print(f"Could not open plume file {plume_file}: {e}")

#%%
if settings["Plotting"]["generate_plots"]:
    SEICOR.plotting.plot_trajectories(
        filtered_ship_groups, 
        maskedout_groups, 
        ship_passes, 
        lon1, 
        lon2, 
        lat1, 
        lat2, 
        lat_lon_window_small, 
        save=True, 
        out_dir=out_dir)
    
    SEICOR.plotting.plot_maskedout_ships_details(
        maskedout_groups, 
        lat_lon_window_small, 
        lon1, 
        lon2, 
        lat1, 
        lat2, 
        save=True, 
        out_dir=out_dir)
    
    SEICOR.plotting.plot_ship_stats(
        filtered_ship_groups, 
        save=True, 
        out_dir=out_dir)

    SEICOR.plotting.plot_no2_timeseries(
        ds_impact_masked, 
        ship_passes, 
        start_time, 
        end_time, 
        separate_legend=True, 
        save=True, 
        out_dir=out_dir)

    SEICOR.plotting.plot_no2_enhancements_for_all_ships(
        (ship_passes_out_dir / f"ship_passes_{date}.csv"),
        (out_dir / f"no2plots" / "{}_no2plots".format(date)))
    
    SEICOR.plotting.plot_no2_enhancements_for_all_ships_full_overview(
        (ship_passes_out_dir / f"ship_passes_{date}.csv"), 
        (out_dir / f"no2plots_full_overview" / "{}_no2plots".format(date)), 
        lat1, lon1, lat2, lon2, save=True)
    
    SEICOR.plotting.plot_wind_polar(
        df_insitu,
        save=True, 
        out_dir=out_dir)
    
    SEICOR.plotting.plot_no2_enhancement_and_insitu(
        ds_impact,
        df_insitu,
        ship_passes,
        save=True, 
        out_dir=out_dir)

    SEICOR.plotting.plot_all_instruments_timeseries_VMR(
        df_lp_doas,
        df_insitu,
        ds_impact,
        ds_impact["VMR_NO2"],
        df_closest=ship_passes,
        title=f"NO$_2$ measurements on {date}",
        save=True, 
        out_dir=out_dir)
    
    SEICOR.plotting.plot_all_instruments_timeseries_SC(
        df_lp_doas_SC,
        df_insitu,
        ds_impact,
        df_closest=ship_passes,
        title=f"NO$_2$ measurements on {date}",
        save=True, 
        out_dir=out_dir)

# %%
#import xarray as xr 
#import pandas as pd
#import shutil
##load median wind dataset
#median_wind_file = weather_stations_dir / f"Median_winddata" / f"median_winddata_hourly.csv"
#df_median_wind = pd.read_csv(median_wind_file, parse_dates=["time"])
#
## find best matching column names for direction and speed
#dir_candidates = ["median_wdir", "median_wind_dir", "wind_dir", "wdir", "direction"]
#speed_candidates = ["median_wspd", "median_wspd_reported", "median_speed_uv", "median_wind_speed", "wind_speed", "wspd"]
#
## define angular ranges (degrees)
#r1_lo, r1_hi = 273.2, 293.2
#r2_lo, r2_hi =  93.2, 113.2
#
## build masks: direction in either range AND speed > 2 m/s
#dir_vals = df_median_wind["median_wdir"]
#speed_vals = df_median_wind["median_wspd"]
#
#mask_dir = dir_vals.between(r1_lo, r1_hi) | dir_vals.between(r2_lo, r2_hi)
#mask_speed = speed_vals > 2.0
#
#median_keep_mask = mask_dir & mask_speed & dir_vals.notna() & speed_vals.notna()
#
## filtered dataframe and list of valid times
#df_median_wind_filtered = df_median_wind.loc[median_keep_mask].copy().reset_index(drop=True)
#valid_median_times = pd.to_datetime(df_median_wind_filtered["time"]) if "time" in df_median_wind_filtered.columns else df_median_wind_filtered.index
#
## expose filtered objects for downstream use
#df_median_wind = df_median_wind  # keep original
#df_median_wind_kept = df_median_wind_filtered
#valid_median_times = valid_median_times
#
#df_all_ship_passes = pd.read_csv(ship_passes_out_dir / f"all_ship_passes.csv", parse_dates=["UTC_Time", "Closest_Impact_Measurement_Time" ]).set_index("UTC_Time")
## %%
#valid_hour_keys = set(valid_median_times.dt.strftime("%Y-%m-%d %H"))
#hour_keys = pd.Series(df_all_ship_passes.index.strftime("%Y-%m-%d %H"), index=df_all_ship_passes.index)
#df_all_ship_passes_mask = hour_keys.isin(valid_hour_keys)
#df_all_ship_passes_filtered = df_all_ship_passes.loc[df_all_ship_passes_mask].copy()
#df_all_ship_passes_filtered.sort_index(inplace=True)
# %%
#for idx, ship_pass in df_all_ship_passes_filtered.iterrows():
#    try:
#        stored = ship_pass.get("plume_file", None)
#        fn = Path(stored).name
#        date = ship_pass['Closest_Impact_Measurement_Time'].strftime('%y%m%d')
#        out_dir = Path(f"E:\\plumes\\plumes_{date}")
#        plume_file = out_dir / fn
#        ds_plume = xr.open_dataset(plume_file)
#        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
#        if plume_found:
#            #copy the respective plume_detection_file to a separate folder for further analysis
#            image_path = Path(f"Q:\\BREDOM\\SEICOR\\analysis\\plume_detection\\plumes_{date}\\ship_yes")
#            fn = f"plume_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}_mask_ship.png"
#            original_image_path = image_path / fn
#            dst_path = Path(r"Q:\BREDOM\SEICOR\analysis\wind_filtered_plots")
#            shutil.copy(original_image_path, dst_path / fn)
#            pass
#        elif not plume_found:
#            print("no plume found")
#    except Exception as e:
#        pass
#        print(f"Could not open plume file {plume_file}: {e}")
# %%

