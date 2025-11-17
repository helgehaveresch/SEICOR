#%%
#this script processes the input data from IMPACT, LP-DOAS, IN-Situ, AIS and the video camera, to create unified datasets and sanity_check_plots for further analysis.
import yaml
import os
import sys
import platform
from pathlib import Path

_platform = platform.system().lower()  
if _platform.startswith("win"):
    _default_scripts_dir = Path(r"C:\Users\hhave\Documents\Promotion\scripts")
    _default_settings_path = _default_scripts_dir / "SEICOR" / "plume_preprocessor_settings.yaml"
    _plat_key = "windows"
elif _platform.startswith("linux"):
    _default_scripts_dir = Path("/raid/home2/hhaveresch/scripts")
    _default_settings_path = _default_scripts_dir / "SEICOR" / "plume_preprocessor_settings.yaml"
    _plat_key = "linux"

SCRIPTS_DIR = Path("SCRIPTS_DIR", _default_scripts_dir)
SETTINGS_PATH = Path("PLUME_SETTINGS_PATH", _default_settings_path)

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
ais_settings            = processing_settings["ais_filter"]
lat_lon_window          = ais_settings["lat_lon_window"]  # [lat_min, lat_max, lon_min, lon_max] area of interest for ais data
lat_lon_window_small    = ais_settings["lat_lon_window_small"]
enhancement_settings    = processing_settings["enhancement"]

img_out_dir             = (BASEPATH / settings["Output"]["img_out_dir"] / f"{date}_video_ships")
plumes_out_dir          = (BASEPATH / settings["Output"]["plume_out_dir"] / f"plumes_{date}")
ship_passes_out_dir     = (BASEPATH / settings["Output"]["ship_passes_out_dir"])
out_dir                 = (BASEPATH / settings["Output"]["plots_out_dir"])
#%% Initialize IMPACT measurements
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
df_insitu = SEICOR.in_situ.read_in_situ(in_situ_path, date)
df_insitu = SEICOR.in_situ.apply_time_mask_to_insitu(df_insitu, start_time, end_time)
ds_impact = SEICOR.impact.calculate_path_averaged_vmr_no2(df_insitu, ds_impact)
#%% Initialize lp-doas data
df_lp_doas = SEICOR.lp_doas.read_lpdoas(lp_doas_dir, date)
df_lp_doas = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])
df_lp_doas_SC = SEICOR.lp_doas.read_lpdoas(lp_doas_dir, date, mode="SC")
df_lp_doas_SC = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_SC, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])

mask, ds_impact_masked = SEICOR.impact.mask_rms_and_reduce_impact(ds_impact, rms_threshold=processing_settings["quality"]["min_rms_IMPACT"])
ds_impact_masked = SEICOR.enhancements.polynomial_background_enh(ds_impact_masked, degree=enhancement_settings["polynomial_background_degree"])
ds_impact_masked = SEICOR.enhancements.fft_background_enh(ds_impact_masked, t_cut=enhancement_settings["high_pass_filter_time_period"])
df_lp_doas_SC = SEICOR.enhancements.polynomial_background_enh_lp_doas(df_lp_doas_SC, degree=enhancement_settings["polynomial_background_degree"])
df_lp_doas_SC = SEICOR.enhancements.fft_background_enh_lp_doas(df_lp_doas_SC, t_cut=enhancement_settings["high_pass_filter_time_period"])

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
import numpy as np
from scipy import ndimage
from scipy.stats import norm
import matplotlib.pyplot as plt
import xarray as xr
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
):
    """
    Detect plume pixels using the neighborhood Z-test described (Kuhlmann et al. 2019).
    - image: 2D numpy array (X_pix values, e.g. XCO2 or enhancement)
    - bg_mean: scalar or 2D array with background mean (if None compute global mean)
    - bg_std: scalar or 2D array with background std (if None compute global std)
    - p_threshold: p-value cutoff (right-sided test). Default 0.20 as in text.
    - min_cluster_size: remove clusters smaller than this (pixels).
    - kernel_arm: arm length for the cross used for neighborhood mean (1 => center + 4 neighbors)
    - median_kernel_arm: arm length for median smoothing kernel (if None uses kernel_arm+1)
    - connectivity: connectivity for labeling (1 => 4-connectivity on 2D)
    Returns binary mask (bool) of plume pixels.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    if bg_mean is None:
        bg_mean = np.nanmean(image)
    if bg_std is None:
        bg_std = np.nanstd(image, ddof=0)
    # allow bg_mean/bg_std to be scalars or arrays
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
        # ndimage.generic_filter with nan handling
        def _func(values):
            vals = values[~np.isnan(values)]
            return vals.mean() if vals.size else np.nan
        return ndimage.generic_filter(arr, _func, footprint=footprint, mode="constant", cval=np.nan)

    neigh_mean = local_mean(image, footprint)

    # Z-score (right-sided test)
    # if bg_mean/bg_std are scalars they broadcast, if arrays must match image shape
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (neigh_mean - bg_mean) / bg_std
    # p = P(sample >= observed) under H0 where H0 mean==bg_mean -> right-sided p = 1 - CDF(z)
    # but text defines p = CDF(-z) which equals 1 - CDF(z) for normal
    pvals = norm.cdf(-z)

    # initial binary mask: reject H0 when p <= p_threshold (plume pixel)
    mask_ini = (pvals <= p_threshold) & (~np.isnan(pvals))
    mask = mask_ini
    # median filter smoothing with larger cross kernel (arms +1 by default)
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

    # median filter (apply to integer mask)
    mask_smoothed = ndimage.median_filter(mask.astype(np.uint8), footprint=footprint_m, mode="constant", cval=0)
    mask = mask_smoothed.astype(bool)

    # remove small clusters and keep the largest cluster per description
    labeled, ncomp = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity))
    if ncomp == 0:
        return mask  # empty
    #print(f"mask has {ncomp} initial clusters labeled {labeled}")

    sizes = ndimage.sum(mask, labeled, range(1, ncomp + 1))
    # remove clusters smaller than min_cluster_size
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]
    if not keep_labels:
        return np.zeros_like(mask, dtype=bool)

    # build mask containing only clusters >= threshold
    mask_filtered = np.isin(labeled, keep_labels)

    # choose largest cluster as final plume mask
    sizes_keep = [(lab, s) for lab, s in zip(range(1, ncomp + 1), sizes) if (lab in keep_labels)]
    largest_label = max(sizes_keep, key=lambda x: x[1])[0]
    final_mask = (labeled == largest_label)

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

    # plot image with plume boundary overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(image_full, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="NO$_2$ enhancement")

    # semi-transparent overlay for the mask (correct location)
    #ax.imshow(np.ma.masked_where(~mask_full, mask_full), origin="lower", cmap="Reds", alpha=0.25, aspect="auto")

    # draw boundary around the mask region
    ax.contour(mask_full.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")

    ax.set_title("NO$_2$ enhancement with plume mask boundary (placed at correct location)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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

        # plot image with plume boundary overlay
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image_full, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="NO$_2$ enhancement")

        # draw boundary around the mask region
        ax.contour(mask_full.astype(int), levels=[0.5], colors="blue", linewidths=1.5, origin="lower")

        ax.set_title("NO$_2$ enhancement with plume mask boundary (placed at correct location)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
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
    ds_plume["plume_or_ship_found"] = plume_found or ship_found
    return ds_plume
#%%

for idx, ship_pass_single in ship_passes.iterrows():
    ds_plume = SEICOR.enhancements.upwind_constant_background_enh(ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
    if ds_plume is not None:
        ds_plume = SEICOR.plumes.add_ship_trajectory_to_plume_ds(ds_plume, filtered_ship_groups)
        ds_plume = SEICOR.plumes.add_insitu_to_plume_ds(ds_plume, df_insitu)
        #ds_plume = SEICOR.impact.call_nlin_c_for_offaxis_ref_and_add_to_plume_ds(ds_plume, ship_pass_single, settings["processing"]["enhancement"]["nlin_param_file"], IMPACT_path)
        ds_plume = SEICOR.enhancements.upwind_downwind_interp_background_enh(ds_plume, ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
        ds_plume = sort_plumes(ds_plume, out_dir, p_threshold_plume=0.02, p_threshold_ship=0.02, date=date)
        Path(plumes_out_dir).mkdir(parents=True, exist_ok=True)
        ds_plume.to_netcdf(ship_pass_single['plume_file'])

# %%
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
