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
import SEICOR.wind  

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



for idx, row in ship_passes.iterrows():
    plume_file = row['plume_file']
    try:
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            # prefer interpolated enhancement, fall back to c_back if missing
            if 'no2_enhancement_interp' in ds_plume.variables:
                varname = 'no2_enhancement_interp'
            elif 'no2_enhancement_c_back' in ds_plume.variables:
                varname = 'no2_enhancement_c_back'
            else:
                print(f"{plume_file}: no enhancement variable found; skipping")
                ds_plume.close()
                continue

            arr = ds_plume[varname].values
            mask = SEICOR.plumes.detect_plume_ztest(arr, p_threshold=0.15, min_cluster_size=5, connectivity=1, kernel_arm=1, require_connection=True, keep_second_largest=False, second_size_threshold=100)
            plt.figure(figsize=(10, 6))
            plt.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
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

    def _safe_run(name, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"Error running {name}: {e}")
            traceback.print_exc()

    _safe_run(
        "plot_trajectories",
        SEICOR.plotting.plot_trajectories,
        filtered_ship_groups,
        maskedout_groups,
        ship_passes,
        lon1,
        lon2,
        lat1,
        lat2,
        lat_lon_window_small,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_maskedout_ships_details",
        SEICOR.plotting.plot_maskedout_ships_details,
        maskedout_groups,
        lat_lon_window_small,
        lon1,
        lon2,
        lat1,
        lat2,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_ship_stats",
        SEICOR.plotting.plot_ship_stats,
        filtered_ship_groups,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_no2_timeseries",
        SEICOR.plotting.plot_no2_timeseries,
        ds_impact_masked,
        ship_passes,
        start_time,
        end_time,
        separate_legend=True,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_no2_enhancements_for_all_ships",
        SEICOR.plotting.plot_no2_enhancements_for_all_ships,
        (ship_passes_out_dir / f"ship_passes_{date}.csv"),
        (out_dir / f"no2plots" / "{}_no2plots".format(date)),
    )

    _safe_run(
        "plot_no2_enhancements_for_all_ships_full_overview",
        SEICOR.plotting.plot_no2_enhancements_for_all_ships_full_overview,
        (ship_passes_out_dir / f"ship_passes_{date}.csv"),
        (out_dir / f"no2plots_full_overview" / "{}_no2plots".format(date)),
        lat1,
        lon1,
        lat2,
        lon2,
        save=True,
    )

    _safe_run(
        "plot_wind_polar",
        SEICOR.plotting.plot_wind_polar,
        df_insitu,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_no2_enhancement_and_insitu",
        SEICOR.plotting.plot_no2_enhancement_and_insitu,
        ds_impact,
        df_insitu,
        ship_passes,
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_all_instruments_timeseries_VMR",
        SEICOR.plotting.plot_all_instruments_timeseries_VMR,
        df_lp_doas,
        df_insitu,
        ds_impact,
        ds_impact["VMR_NO2"],
        df_closest=ship_passes,
        title=f"NO$_2$ measurements on {date}",
        save=True,
        out_dir=out_dir,
    )

    _safe_run(
        "plot_all_instruments_timeseries_SC",
        SEICOR.plotting.plot_all_instruments_timeseries_SC,
        df_lp_doas_SC,
        df_insitu,
        ds_impact,
        df_closest=ship_passes,
        title=f"NO$_2$ measurements on {date}",
        save=True,
        out_dir=out_dir,
    )

# %%
import xarray as xr 
import pandas as pd
import shutil
import numpy as np
#load median wind dataset
median_wind_file = weather_stations_dir / f"Median_winddata" / f"median_winddata_hourly.csv"
df_median_wind = pd.read_csv(median_wind_file, parse_dates=["time"])

# find best matching column names for direction and speed
dir_candidates = ["median_wdir", "median_wind_dir", "wind_dir", "wdir", "direction"]
speed_candidates = ["median_wspd", "median_wspd_reported", "median_speed_uv", "median_wind_speed", "wind_speed", "wspd"]

# define angular ranges (degrees)
r1_lo, r1_hi = 273.2, 293.2
r2_lo, r2_hi =  93.2, 113.2

# build masks: direction in either range AND speed > 2 m/s
dir_vals = df_median_wind["median_wdir"]
speed_vals = df_median_wind["median_wspd"]

mask_dir = dir_vals.between(r1_lo, r1_hi) | dir_vals.between(r2_lo, r2_hi)
mask_speed = speed_vals > 2.0

median_keep_mask = mask_dir & mask_speed & dir_vals.notna() & speed_vals.notna()

# filtered dataframe and list of valid times
df_median_wind_filtered = df_median_wind.loc[median_keep_mask].copy().reset_index(drop=True)
valid_median_times = pd.to_datetime(df_median_wind_filtered["time"]) if "time" in df_median_wind_filtered.columns else df_median_wind_filtered.index

# expose filtered objects for downstream use
df_median_wind = df_median_wind  # keep original
df_median_wind_kept = df_median_wind_filtered
valid_median_times = valid_median_times

df_all_ship_passes = pd.read_csv(ship_passes_out_dir / f"all_ship_passes.csv", parse_dates=["UTC_Time", "Closest_Impact_Measurement_Time" ]).set_index("UTC_Time")
# %%
valid_hour_keys = set(valid_median_times.dt.strftime("%Y-%m-%d %H"))
hour_keys = pd.Series(df_all_ship_passes.index.strftime("%Y-%m-%d %H"), index=df_all_ship_passes.index)
df_all_ship_passes_mask = hour_keys.isin(valid_hour_keys)
df_all_ship_passes_filtered = df_all_ship_passes.loc[df_all_ship_passes_mask].copy()
df_all_ship_passes_filtered.sort_index(inplace=True)
# Map median wind values (hourly) to each ship pass by matching the same hour
hour_map = pd.to_datetime(df_median_wind_kept["time"]).dt.strftime("%Y-%m-%d %H")
# create a lookup Series for speed and direction
median_speed_lookup = pd.Series(df_median_wind_kept["median_wspd"].values, index=hour_map.values)
median_dir_lookup = pd.Series(df_median_wind_kept["median_wdir"].values, index=hour_map.values)

# compute hour keys for ship passes (we already used this earlier as `hour_keys`)
ship_hour_keys = df_all_ship_passes_filtered.index.strftime("%Y-%m-%d %H")

# Map into new columns; missing values will remain NaN
df_all_ship_passes_filtered = df_all_ship_passes_filtered.assign(
    median_wind_speed=ship_hour_keys.map(median_speed_lookup),
    median_wind_dir=ship_hour_keys.map(median_dir_lookup),
)
# %%
for idx, ship_pass in df_all_ship_passes_filtered.iterrows():
    try:
        stored = ship_pass.get("plume_file", None)
        fn = Path(stored).name
        date = ship_pass['Closest_Impact_Measurement_Time'].strftime('%y%m%d')
        out_dir = Path(f"Q:\\BREDOM\\SEICOR\\plumes\\plumes_{date}")
        plume_file = out_dir / fn
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            rel_wind_speed, rel_wind_dir = SEICOR.wind.compute_relative_wind(ship_pass['Mean_Speed'], ship_pass['Mean_Course'], ship_pass['median_wind_speed'], ship_pass['median_wind_dir'])

            #copy the respective plume_detection_file to a separate folder for further analysis
            image_path = Path(f"Q:\\BREDOM\\SEICOR\\analysis\\plume_detection\\plumes_{date}\\ship_yes")
            dst_path = Path(r"Q:\BREDOM\SEICOR\analysis\wind_filtered_plots")

            # Prepare candidate filename patterns to handle variations in how images were saved
            t = pd.to_datetime(ds_plume.attrs.get('t'))
            time_nounder = t.strftime('%Y%m%d_%H%M%S')
            mmsi = ds_plume.attrs.get('mmsi')

            candidates = [
                f"plume_{time_nounder}_{mmsi}_mask_plume.png",
                f"plume_{time_nounder}_{mmsi}_mask_ship.png",
            ]

            found = False
            for candidate in candidates:
                original_image_path = image_path / candidate
                if original_image_path.exists():
                    try:
                        shutil.copy(original_image_path, dst_path / candidate)
                        print(f"Copied to {dst_path / candidate}")
                        # also copy into relative-wind-classified subfolders
                        rel_gt_dir = dst_path / "rel_wind_gt2"
                        rel_lt_dir = dst_path / "rel_wind_lt2"
                        rel_gt_dir.mkdir(parents=True, exist_ok=True)
                        rel_lt_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            rws = float(rel_wind_speed)
                        except Exception:
                            rws = None

                        if rws is None or np.isnan(rws):
                            # unknown relative wind: leave only in main dst
                            pass
                        elif rws > 2.0:
                            try:
                                shutil.copy(original_image_path, rel_gt_dir / candidate)
                                print(f"Copied to {rel_gt_dir}")
                            except Exception as e:
                                print(f"Failed to copy to {rel_gt_dir}: {e}")
                        else:
                            try:
                                shutil.copy(original_image_path, rel_lt_dir / candidate)
                                print(f"Copied to {rel_lt_dir}")
                            except Exception as e:
                                print(f"Failed to copy to {rel_lt_dir}: {e}")

                        # Also classify by wind direction relative to ship course
                        try:
                            wind_from = float(ship_pass.get('median_wind_dir'))
                            ship_course = float(ship_pass.get('Mean_Course'))
                        except Exception:
                            wind_from = None
                            ship_course = None

                        if wind_from is not None and ship_course is not None:
                            # wind_to is the direction the wind is going TO (meteorological from + 180)
                            wind_to = (wind_from + 180.0) % 360.0

                            def angdiff(a, b):
                                d = (a - b + 180.0) % 360.0 - 180.0
                                return abs(d)

                            thresh_deg = 15.0
                            same_dir = angdiff(wind_to, ship_course) <= thresh_deg
                            against_dir = angdiff(wind_from, ship_course) <= thresh_deg

                            same_dir_dir = dst_path / "wind_same_dir"
                            against_dir_dir = dst_path / "wind_against_dir"
                            against_dir_dir_low = dst_path / "wind_against_dir_low_speed"
                            same_dir_dir.mkdir(parents=True, exist_ok=True)
                            against_dir_dir.mkdir(parents=True, exist_ok=True)
                            against_dir_dir_low.mkdir(parents=True, exist_ok=True)

                            if same_dir:
                                try:
                                    shutil.copy(original_image_path, same_dir_dir / candidate)
                                    print(f"Copied to {same_dir_dir}")
                                except Exception as e:
                                    print(f"Failed to copy to {same_dir_dir}: {e}")
                            elif against_dir:
                                # Only classify as 'against' when relative wind speed is sufficiently large
                                try:
                                    rws_check = float(rel_wind_speed)
                                except Exception:
                                    rws_check = None

                                if rws_check is None or np.isnan(rws_check):
                                    print(f"Against-wind detected but relative wind unknown; skipping against_dir copy for {candidate}")
                                elif rws_check <= 2.0:
                                    try:
                                        shutil.copy(original_image_path, against_dir_dir_low / candidate)
                                        print(f"Copied to {against_dir_dir_low}")
                                    except Exception as e:
                                        print(f"Failed to copy to {against_dir_dir_low}: {e}")
                                else:
                                    print(f"Against-wind detected but relative wind {rws_check:.2f} < 2.0 m/s; skipping against_dir copy for {candidate}")

                        found = True
                        break
                    except Exception as e:
                        print(f"Failed to copy {original_image_path} -> {dst_path / candidate}: {e}")
                        found = True
                        break

            if not found:
                print(f"No plume image found for plume file {plume_file}. Tried: {candidates}")
            pass
        elif not plume_found:
            print("no plume found")
    except Exception as e:
        pass
        print(f"Could not open plume file {plume_file}: {e}")


# %%