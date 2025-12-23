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
import pandas as pd
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
OUTPATH = Path(r"/raid/home2/hhaveresch")
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
#%% read median wind data set from weather stations
median_wind_file = weather_stations_dir / f"Median_winddata" / f"median_winddata_hourly.csv"
df_median_wind = pd.read_csv(median_wind_file, parse_dates=["time"])
#%% Initialize in-situ data
try:
    df_insitu = SEICOR.in_situ.read_in_situ(in_situ_path, date)
    
    # replace in-situ wind values with nearest-time median wind observations
    # some sources provide time as the index — ensure a `time` column exists
    
    df_insitu = df_insitu.reset_index()
    df_insitu.rename(columns={df_insitu.columns[0]: 'time'}, inplace=True)
    df_insitu['time'] = pd.to_datetime(df_insitu['time'], utc=True)
    df_median_wind['time'] = pd.to_datetime(df_median_wind['time'], utc=True)
    left = df_insitu.sort_values('time').reset_index(drop=True)
    right = df_median_wind.sort_values('time').reset_index(drop=True)
    merged = pd.merge_asof(left, right[['time', 'median_wspd', 'median_wdir']], on='time', direction='nearest')
    if 'median_wspd' in merged.columns:
        merged['wind_speed'] = merged['median_wspd']
    if 'median_wdir' in merged.columns:
        merged['wind_dir'] = merged['median_wdir']
    # adopt merged dataframe as df_insitu (keeps other columns)
    df_insitu = merged
    # restore time as the index (original data often used time as index)
    df_insitu = df_insitu.set_index('time')
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
# ensure a column exists to record whether a plume or ship was found (default 'False')
if 'plume_or_ship_found' not in ship_passes.columns:
    ship_passes['plume_or_ship_found'] = 'False'


#%% Process each ship pass
for idx, ship_pass_single in ship_passes.iterrows():
    ds_plume = SEICOR.enhancements.upwind_constant_background_enh(ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
    if ds_plume is not None:
        ds_plume = SEICOR.plumes.add_ship_trajectory_to_plume_ds(ds_plume, filtered_ship_groups)
        ds_plume = SEICOR.plumes.add_insitu_to_plume_ds(ds_plume, df_insitu)
        #ds_plume = SEICOR.impact.call_nlin_c_for_offaxis_ref_and_add_to_plume_ds(ds_plume, ship_pass_single, settings["processing"]["enhancement"]["nlin_param_file"], IMPACT_path)
        ds_plume = SEICOR.enhancements.upwind_downwind_interp_background_enh(ds_plume, ship_pass_single, ds_impact, measurement_times, ship_passes, df_lp=df_lp_doas)
        Path(plumes_out_dir).mkdir(parents=True, exist_ok=True)
        ds_plume = SEICOR.plumes.sort_plumes(ds_plume, out_dir, p_threshold_plume=0.03, p_threshold_ship=0.02, date=date)
        if settings["Plotting"]["generate_plots"] and ds_plume.attrs.get("plume_or_ship_found", "False") == "True":
            ref_image = ds_plume["no2_ref"] - ds_plume["no2_ref"].mean(dim="window_ref")
            mask = SEICOR.plumes.detect_plume_ztest(ref_image.values, p_threshold=0.15, min_cluster_size=30, ds_plume=ds_plume)
            SEICOR.plotting.plot_reference_image_with_plume_mask(ref_image, mask, ds_plume, out_dir, date)
        ds_plume.to_netcdf(Path(ship_pass_single['plume_file']), mode="w")
        # record result back into ship_passes dataframe so it can be saved
        try:
            ship_passes.at[idx, 'plume_or_ship_found'] = ds_plume.attrs.get('plume_or_ship_found', 'False')
        except Exception:
            pass
Path(ship_passes_out_dir).mkdir(parents=True, exist_ok=True)
# write ship_passes ensuring datetime64 columns are formatted and not writing the index
ship_passes.to_csv(
    ship_passes_out_dir / f"ship_passes_{date}.csv",
    index=False,
    date_format="%Y-%m-%dT%H:%M:%S%z",
)
# %%
for idx, row in ship_passes.iterrows():
    plume_file = row['plume_file']
    try:
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if plume_found:
            # Prefer interpolated enhancement if available, else fall back to c_back.
            mask = SEICOR.plumes.detect_plume_ztest(
                ds_plume["no2_enhancement_interp"].values,
                p_threshold=0.20,
                min_cluster_size=5,
                connectivity=1,
                kernel_arm=1,
                require_connection=True,
                ds_plume=ds_plume,
                keep_second_largest=False,
                second_size_threshold=100,
            )
            SEICOR.plotting.plot_no2_enhancement_with_plume_mask(ds_plume, mask, out_dir, date)
    except Exception as e:
        print(f"Could not open plume file {plume_file}: {e}")
#%%
# after checking/saving plume figures, persist updated ship_passes (with plume flag)
Path(ship_passes_out_dir).mkdir(parents=True, exist_ok=True)
# final persist: format datetimes as ISO-like strings and do not write the index
ship_passes.to_csv(
    ship_passes_out_dir / f"ship_passes_{date}.csv",
    index=True,
)

# %%

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
        (img_dir / f"{date}_video"),
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
