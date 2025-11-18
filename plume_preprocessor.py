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
import matplotlib.pyplot as plt
import pandas as pd


#%%
def plot_reference_image_with_plume_mask(ds_plume, out_dir, date, p_threshold=0.2, min_cluster_size = 0.2):
    ref_image = ds_plume["no2_ref"] - ds_plume["no2_ref"].mean(dim="window_ref")
    mask = SEICOR.plumes.detect_plume_ztest(ref_image.values, p_threshold=p_threshold, min_cluster_size=min_cluster_size)
    plt.imshow(ref_image.values, origin="lower", cmap="viridis")
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
        ds_plume.to_netcdf(ship_pass_single['plume_file'])

# %%
import xarray as xr
for idx, row in ship_passes.iterrows():
    plume_file = row['plume_file']
    try:
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            mask = SEICOR.plumes.detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=20)
            plt.figure(figsize=(10, 6))
            plt.imshow(ds_plume["no2_enhancement_c_back"].values, origin="lower", aspect="auto", cmap="viridis")
            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
            plt.title("No2 enhancement with detected plume pixels")
            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}", exist_ok=True)
            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
            plt.close('all')

        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            mask = SEICOR.plumes.detect_plume_ztest(ds_plume["no2_enhancement_interp"].values, p_threshold=0.2, min_cluster_size=20)
            plt.figure(figsize=(10, 6))
            plt.imshow(ds_plume["no2_enhancement_c_back"].values, origin="lower", aspect="auto", cmap="viridis")
            plt.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=1.5, origin="lower")
            plt.title("No2 enhancement with detected plume pixels")
            os.makedirs(out_dir / f"plume_mask" / f"plumes_{date}_c_back", exist_ok=True)
            plt.savefig(out_dir / f"plume_mask" / f"plumes_{date}_c_back" / f"no2_enhancement_with_plume_mask_{date}_{pd.to_datetime(ds_plume.attrs.get('t')).strftime('%Y%m%d_%H%M%S')}_{ds_plume.attrs.get('mmsi')}.png")
            plt.close('all')

    except Exception as e:
        print(f"Could not open plume file {plume_file}: {e}")

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
