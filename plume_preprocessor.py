#%%
#this script processes the input data from IMPACT, LP-DOAS, IN-Situ, AIS and the video camera, to create unified datasets and sanity_check_plots for further analysis.
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

import numpy as np
sys.path.append(str(SCRIPTS_DIR))
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import mask_zenith, mask_short_azimuth_blocks, correct_destriped_for_noon_reference
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
import SEICOR.era5
import SEICOR.lidar

with open(SETTINGS_PATH, "r") as file:
    settings = yaml.safe_load(file)

BASEPATH                = Path(settings.get("base_paths", {}).get(_plat_key))
date                    = settings["date"]
#if an argument is passed to the script, use that as date
if len(sys.argv) > 1:
    date = sys.argv[1]
instrument_settings     = settings["Instruments"]
time_resolution         = settings["processing"]["coarsen"]["time_resolution_seconds"]  # None/0: no coarsening
IMPACT_path             = (BASEPATH / instrument_settings["IMPACT_SC_path"].format(yy=date[:2]) / get_month_folder(date))
IMPACT_SC_file_ending   = instrument_settings["IMPACT_SC_file_ending"]
do_ref_correction       = instrument_settings["do_reference_correction"]
SC_ref_offset_corr      = instrument_settings["SC_reference_offset_correction_files"]
instrument_location     = instrument_settings["instrument_location"]
lat1, lon1              = instrument_settings["instrument_location"]
lp_doas_dir             = (BASEPATH / instrument_settings["lp_doas_dir"])
in_situ_path            = (BASEPATH / instrument_settings["in_situ_path"])
airpointer_path         = (BASEPATH / instrument_settings["airpointer_path"])
meteomast_path          = (BASEPATH / instrument_settings["meteomast_path"])
median_wind_file        = (BASEPATH / instrument_settings["median_wind_file"])
ais_dir                 = (BASEPATH / instrument_settings["ais_dir"])
img_dir                 = (BASEPATH / instrument_settings["img_dir"])
weather_stations_dir    = (BASEPATH / instrument_settings["weather_stations_dir"])
img_out_dir             = (BASEPATH / settings["Output"]["img_out_dir"] / f"{date}_video_ships")
plumes_out_dir          = (BASEPATH / settings["Output"]["plume_out_dir"] / f"plumes_{date}")
ship_passes_out_dir     = (BASEPATH / settings["Output"]["ship_passes_out_dir"])
out_dir                 = (BASEPATH / settings["Output"]["plots_out_dir"])
impact_out_dir          = (BASEPATH / settings["Output"]["impact_coarsened_out_dir"])
#%% Skip days without SD measurements
# a day is only processed if the SD scans were measured: the SD file of the first viewing direction
# must exist and contain at least one measurement
sd_file = IMPACT_path / f"{date}01{instrument_settings.get('required_SD_file_ending', 'SD.NO2_VIS_daily_ref')}"
n_sd = 0
if sd_file.exists():
    with open(sd_file, errors="ignore") as f:
        n_sd = sum(1 for line in f if line.strip()[:1].isdigit())
if n_sd == 0:
    print(f"SKIPPED: no SD measurements on {date} ({sd_file.name} {'is empty' if sd_file.exists() else 'does not exist'})")
    sys.exit(0)
#%% Initialize IMPACT measurements
print("Starting plume preprocessing...")
ds_impact = read_SC_file_imaging(IMPACT_path, date, IMPACT_SC_file_ending)
if do_ref_correction:
    ds_ref_corr = read_SC_file_imaging(IMPACT_path, date, SC_ref_offset_corr)
    ds_impact = correct_destriped_for_noon_reference(ds_impact, ds_ref_corr)
ds_impact = mask_zenith(ds_impact)
# zenith measurements labelled as off-axis: short blocks at another viewing azimuth
ds_impact = mask_short_azimuth_blocks(ds_impact, max_minutes=instrument_settings.get("zenith_block_max_minutes", 5))
# drop the lowest viewing directions (index 0 = lowest VEA); the viewing_direction coordinate keeps the
# original numbers, so image_row in the plume files and the row settings still refer to the same elevations
ds_impact = ds_impact.isel(viewing_direction=slice(instrument_settings.get("drop_lowest_viewing_directions", 0), None))
if time_resolution:
    # average in fixed time bins (gaps stay gaps) and save the coarsened day
    ds_impact = SEICOR.impact.coarsen_impact_measurements(ds_impact, time_resolution)
    impact_out_dir.mkdir(parents=True, exist_ok=True)
    ds_impact.to_netcdf(impact_out_dir / f"IMPACT_{date}_{time_resolution:g}s.nc", mode="w")
# sample/pixel counts in the settings are given in raw measurements: convert them with the number of
# raw measurements per coarsened one (1 without coarsening)
coarsen_factor          = ds_impact.attrs.get("coarsen_factor", 1)
processing_settings     = SEICOR.impact.scale_sample_counts(settings["processing"], coarsen_factor)
ais_settings            = processing_settings["ais_filter"]
lat_lon_window          = ais_settings["lat_lon_window"]  # [lat_min, lat_max, lon_min, lon_max] area of interest for ais data
lat_lon_window_small    = ais_settings["lat_lon_window_small"]
enhancement_settings    = processing_settings["enhancement"]
window_minutes          = (processing_settings["plume_window"]["before_minutes"], processing_settings["plume_window"]["after_minutes"])
ref_settings            = processing_settings["reference_search"]
reference_search_kwargs = dict(                     # passed to both enhancement functions
    ref_start_offset=ref_settings["start_offset_minutes"],
    ref_search_minutes=ref_settings["search_minutes"],
    ref_window_minutes=ref_settings["window_minutes"],
    ref_min_count=ref_settings["min_count"],
    ref_min_span_seconds=ref_settings["min_span_seconds"],
    other_ship_minutes=ref_settings["other_ship_minutes"],
    other_ship_min_length_m=ref_settings.get("other_ship_min_length_m"),
    ref_ztest_p_threshold=ref_settings["ztest_p_threshold"],
    ref_ztest_min_cluster_size=ref_settings["ztest_min_cluster_size"],
)
detection_settings      = processing_settings["detection"]
plume_mask_settings     = processing_settings["plume_mask"]
if enhancement_settings["calculate_rolling_background"]:
    ds_impact = SEICOR.enhancements.rolling_background_enh(ds_impact, window_size=enhancement_settings["rolling_background_window"])
ds_impact = SEICOR.impact.rolling_impact(ds_impact, window=processing_settings["quality"]["rolling_mean_window"])
endpoints_los = SEICOR.impact.calculate_LOS(ds_impact, instrument_location)
lat2, lon2 = endpoints_los[:, 0], endpoints_los[:, 1] #todo: to be replaced 
start_time, end_time, measurement_times = SEICOR.impact.calc_start_end_times(ds_impact)
#%% Initialize in-situ data (HORIBA gases, Airpointer wind, meteomast wind at all heights)
try:
    # median wind of the weather stations is not used, add median_wind_file=median_wind_file to include it
    df_insitu = SEICOR.in_situ.read_in_situ(in_situ_path, date, airpointer_path=airpointer_path, meteomast_path=meteomast_path)
    df_insitu = SEICOR.in_situ.apply_time_mask_to_insitu(df_insitu, start_time, end_time)
    ds_impact = SEICOR.impact.calculate_path_averaged_vmr_no2(df_insitu, ds_impact)
except Exception as e:
    print("Error in in-situ processing block:", e)
    traceback.print_exc()
    df_insitu = None
#%% Initialize ERA5 profile above the ship corridor
try:
    era5_profile = SEICOR.era5.read_era5_corridor(BASEPATH / instrument_settings["era5_dir"], date, instrument_settings["ship_corridor_location"])
except Exception as e:
    print("Error reading ERA5:", e)
    traceback.print_exc()
    era5_profile = None
#%% Initialize flip-corrected LIDAR wind (written by wind_model_meteomast_lidar_comparison.py, 2026 only)
try:
    df_lidar = SEICOR.lidar.read_lidar_corrected(Path(settings["bredom_paths"][_plat_key]) / instrument_settings["lidar_file"])
except Exception as e:
    print("Error reading LIDAR:", e)
    traceback.print_exc()
    df_lidar = None
#%% Initialize lp-doas data
try:
    df_lp_doas_310 = SEICOR.lp_doas.read_lpdoas_310(lp_doas_dir, date)
    df_lp_doas_365 = SEICOR.lp_doas.read_lpdoas_365(lp_doas_dir, date)
    df_lp_doas_310 = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_310, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])
    df_lp_doas_365 = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_365, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])

    #df_lp_doas_SC_310 = SEICOR.lp_doas.read_lpdoas_310(lp_doas_dir, date, mode="SC")
    #df_lp_doas_SC_365 = SEICOR.lp_doas.read_lpdoas_365(lp_doas_dir, date, mode="SC")
    #df_lp_doas_SC_310 = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_SC_310, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])
    #df_lp_doas_SC_365 = SEICOR.lp_doas.mask_lp_doas_file(df_lp_doas_SC_365, start_time, end_time, rms_threshold=processing_settings["quality"]["min_rms_lp_doas"])

    mask, ds_impact_masked = SEICOR.impact.mask_rms_and_reduce_impact(ds_impact, rms_threshold=processing_settings["quality"]["min_rms_IMPACT"])
    if enhancement_settings["calculate_polynomial_background"]:
        ds_impact_masked = SEICOR.enhancements.polynomial_background_enh(ds_impact_masked, degree=enhancement_settings["polynomial_background_degree"])
    if enhancement_settings["calculate_high_pass_fft_background"]:
        ds_impact_masked = SEICOR.enhancements.fft_background_enh(ds_impact_masked, t_cut=enhancement_settings["high_pass_filter_time_period"])
    #df_lp_doas_SC_310 = SEICOR.enhancements.polynomial_background_enh_lp_doas(df_lp_doas_SC_310, degree=enhancement_settings["polynomial_background_degree"])
    #df_lp_doas_SC_310 = SEICOR.enhancements.fft_background_enh_lp_doas(df_lp_doas_SC_310, t_cut=enhancement_settings["high_pass_filter_time_period"])
    #df_lp_doas_SC_365 = SEICOR.enhancements.fft_background_enh_lp_doas(df_lp_doas_SC_365, t_cut=enhancement_settings["high_pass_filter_time_period"])
    #df_lp_doas_SC_365 = SEICOR.enhancements.polynomial_background_enh_lp_doas(df_lp_doas_SC_365, degree=enhancement_settings["polynomial_background_degree"])

except Exception as e:
    print("Error in LP-DOAS / IMPACT processing block:", e)
    traceback.print_exc()
    # Fallbacks so downstream cells can still run. Adjust these as needed.
    df_lp_doas = None
    df_lp_doas_SC_310 = None
    df_lp_doas_SC_365 = None
    df_lp_doas_310 = None
    df_lp_doas_365 = None
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
# two ships passing within a few minutes show the same images: drop the pass of the smaller ship if it is small
ship_passes, dropped_passes = SEICOR.ais.drop_small_close_passes(
    ship_passes, max_gap_minutes=ais_settings["close_pass_minutes"], small_length_m=ais_settings["close_pass_small_length_m"])
for t_drop, row in dropped_passes.iterrows():
    print(f"Pass of MMSI {row['MMSI']} ({row['Length_in_m']} m) at {t_drop} dropped: a larger ship passes within {ais_settings['close_pass_minutes']} min")
ship_passes = SEICOR.in_situ.add_wind_to_ship_passes(ship_passes, df_insitu)
ship_passes = SEICOR.video_camera.assign_video_images_to_ship_pass(ship_passes, img_dir, date)
ship_passes = SEICOR.plumes.add_plume_file_paths_to_ship_passes(ship_passes, plumes_out_dir)
# ensure a column exists to record whether a plume or ship was found (default 'False')
if 'plume_or_ship_found' not in ship_passes.columns:
    ship_passes['plume_or_ship_found'] = 'False'


#%% Process each ship pass
# save the settings of this run next to the plume files (with the processed date)
Path(plumes_out_dir).mkdir(parents=True, exist_ok=True)
with open(plumes_out_dir / "plume_preprocessor_settings_used.yaml", "w") as file:
    yaml.safe_dump({**settings, "date": date}, file, sort_keys=False)

# inputs of the plume dataset that are the same for every ship pass and plume window
plume_ds_kwargs = dict(
    ds_impact=ds_impact,
    measurement_times=measurement_times,
    ship_passes=ship_passes,
    filtered_ship_groups=filtered_ship_groups,
    df_insitu=df_insitu,
    df_lp_310=df_lp_doas_310,
    df_lp_365=df_lp_doas_365,
    nlin_param_file=enhancement_settings["nlin_param_file"] if enhancement_settings["run_upwind_doas_fit"] else None,
    impact_path=IMPACT_path,
    **reference_search_kwargs,
)

extend_step = processing_settings["plume_window"]["extend_step_minutes"]
max_extensions = processing_settings["plume_window"]["max_extensions"]

def detect_plume(ds_plume):
    """Plume/ship flag of the pass (sort_plumes)."""
    return SEICOR.plumes.sort_plumes(
        ds_plume, out_dir, date=date,
        p_threshold_plume=detection_settings["p_threshold_plume"],
        p_threshold_ship=detection_settings["p_threshold_ship"],
        time_tolerance_seconds=detection_settings["time_tolerance_seconds"],
        plume_rows=tuple(detection_settings["plume_rows"]),
        ship_rows=tuple(detection_settings["ship_rows"]),
        plume_min_cluster_size=detection_settings["plume_min_cluster_size"],
        ship_min_cluster_size=detection_settings["ship_min_cluster_size"],
        # the plume is searched in the same time window as the source region of the plume mask
        plume_seconds_before=plume_mask_settings.get("source_seconds_before", 60),
        plume_seconds_after=plume_mask_settings.get("source_seconds_after", 60),
    )


# rows are addressed by position (pos), not by the time index: two ships passing in the same second share a time index
for pos, (idx, ship_pass_single) in enumerate(ship_passes.iterrows()):
    ds_plume = SEICOR.enhancements.build_plume_ds(ship_pass_single, window_minutes, **plume_ds_kwargs)
    if ds_plume is None:
        continue
    ds_plume = detect_plume(ds_plume)

    if ds_plume.attrs.get("plume_or_ship_found", "False") == "True":
        # plume mask; while it touches the left or right edge of the plume window, extend the window
        # on that side by extend_step minutes (the references move along) and mask again. The window
        # is extended in one direction only, set at the first extension: "after" if the mask touches
        # the right edge, "before" only if it touches the left edge but not the right one.
        # The step with the biggest plume cluster whose references do not overlap it is saved.
        detection_attrs = dict(ds_plume.attrs)
        before, after = window_minutes
        best = None
        direction = None  # "before" or "after", set at the first extension
        for n_ext in range(max_extensions + 1):
            if n_ext > 0:
                ds_new = SEICOR.enhancements.build_plume_ds(ship_pass_single, (before, after), **plume_ds_kwargs)
                if ds_new is None or "no2_ref" not in ds_new:
                    break  # no clean upwind reference for the extended window
                ds_new.attrs = {**detection_attrs, **ds_new.attrs}
                ds_plume = ds_new
            mask, ship_mask = SEICOR.plumes.plume_mask_of(ds_plume, plume_mask_settings)
            if SEICOR.plumes.reference_hits_plume(ds_plume, mask):
                print(f"Reference overlaps the masked plume for MMSI {ship_pass_single['MMSI']} at {idx}, "
                      f"window ({before}, {after}) min not used")
            elif best is None or mask.sum() > best[1].sum():
                best = (ds_plume, mask, ship_mask, n_ext)
            left_px, right_px = int(mask[:, 0].sum()), int(mask[:, -1].sum())
            if direction is None and (left_px or right_px):
                direction = "after" if right_px else "before"
            if direction is None or (direction == "before" and not left_px) or (direction == "after" and not right_px):
                break  # the mask does not touch the edge of the extension direction
            if direction == "before":
                before += extend_step
            else:
                after += extend_step
        if best is None:  # every step overlapped a reference: keep the last window without a plume mask
            best = (ds_plume, np.zeros_like(mask), ship_mask, n_ext)
        ds_plume, mask, ship_mask, n_used = best
        ds_plume.attrs["plume_window_extensions"] = n_used
        ds_plume.attrs["plume_window_extend_direction"] = direction if n_used else "none"
        # store enhancement plume mask (and the ship mask excluded from the background) in the plume dataset
        ds_plume = ds_plume.assign(plume_mask=(["image_row", "window_plume"], mask),
                                   ship_mask=(["image_row", "window_plume"], ship_mask))
        # background left in the enhancement image (mean outside plume and ship): warn if mean > std,
        # subtract it if enough of the window is background
        ds_plume = SEICOR.plumes.subtract_background(
            ds_plume, mask, ship_mask, min_background_fraction=plume_mask_settings["min_background_fraction"])

        if settings["Plotting"]["generate_plots"]:
            try:  # a failing plot (e.g. a file that cannot be written) must not stop the processing
                SEICOR.plotting.plot_no2_enhancement_with_plume_mask(ds_plume, mask, out_dir, date)
                SEICOR.plotting.plot_reference_image_with_plume_mask(
                    ds_plume, out_dir, date, min_cluster_size=SEICOR.impact.scale_count(30, coarsen_factor))
            except Exception as e:
                print(f"Error plotting the plume mask of MMSI {ship_pass_single['MMSI']} at {idx}: {e}")
                traceback.print_exc()
    ds_plume = SEICOR.era5.add_era5_to_plume_ds(ds_plume, era5_profile, instrument_settings["era5_max_height_m"])
    ds_plume = SEICOR.lidar.add_lidar_to_plume_ds(ds_plume, df_lidar)
    ds_plume.to_netcdf(Path(ship_pass_single['plume_file']), mode="w")
    # record result back into ship_passes dataframe so it can be saved
    ship_passes.iat[pos, ship_passes.columns.get_loc('plume_or_ship_found')] = ds_plume.attrs.get('plume_or_ship_found', 'False')
Path(ship_passes_out_dir).mkdir(parents=True, exist_ok=True)
# write ship_passes with the UTC_Time index (read back with index_col=0) and full time precision
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

    # in-situ plots are skipped if the in-situ block failed (df_insitu is None, VMR_NO2 not calculated)
    if df_insitu is not None:
        _safe_run(
            "plot_wind_polar",
            SEICOR.plotting.plot_wind_polar,
            df_insitu,
            save=True,
            out_dir=out_dir,
        )

        # this plot shows the rolling background enhancement
        if enhancement_settings["calculate_rolling_background"]:
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
            df_lp_doas_365,
            df_insitu,
            ds_impact,
            ds_impact["VMR_NO2"],
            df_closest=ship_passes,
            title=f"NO$_2$ measurements on {date}",
            save=True,
            out_dir=out_dir,
        )

    #_safe_run(
    #    "plot_all_instruments_timeseries_SC",
    #    SEICOR.plotting.plot_all_instruments_timeseries_SC,
    #    df_lp_doas_SC_365,
    #    df_insitu,
    #    ds_impact,
    #    df_closest=ship_passes,
    #    title=f"NO$_2$ measurements on {date}",
    #    save=True,
    #    out_dir=out_dir,
    #)

# %%
