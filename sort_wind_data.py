import shutil
import yaml
import os
import sys
import platform
import numpy as np
import xarray as xr 
import pandas as pd
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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

sys.path.append(str(SCRIPTS_DIR))
from imaging_tools.utilities import get_month_folder
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
#load median wind dataset
median_wind_file = weather_stations_dir / f"Median_winddata" / f"median_winddata_hourly.csv"
df_median_wind = pd.read_csv(median_wind_file, parse_dates=["time"])

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
def _process_ship_pass_task(task):
    """Worker function to process a single ship_pass task.

    task: tuple (idx, row_dict)
    Returns: dict with status info
    """

    idx, row = task
    try:
        ship_pass = row
        stored = ship_pass.get("plume_file", None)
        if stored is None:
            return {"idx": idx, "status": "no_plume_file"}
        fn = Path(stored).name
        date = pd.to_datetime(ship_pass['Closest_Impact_Measurement_Time']).strftime('%y%m%d')
        out_dir = Path(f"/misc/dodecagon/BREDOM/SEICOR/plumes/plumes_{date}")
        plume_file = out_dir / fn
        ds_plume = xr.open_dataset(plume_file)
        plume_found = ds_plume.attrs.get("plume_or_ship_found", "False") == "True"
        if not plume_found:
            return {"idx": idx, "status": "no_plume_found"}

        rel_wind_speed, rel_wind_dir = SEICOR.wind.compute_relative_wind(ship_pass.get('Mean_Speed'), ship_pass.get('Mean_Course'), ship_pass.get('median_wind_speed'), ship_pass.get('median_wind_dir'))

        image_path = Path(f"/misc/dodecagon/BREDOM/SEICOR/analysis/plume_mask/plumes_{date}")
        dst_path = Path(r"/raid/home2/hhaveresch/wind_filtered_plots")

        t = pd.to_datetime(ds_plume.attrs.get('t'))
        time_nounder = t.strftime('%Y%m%d_%H%M%S')
        mmsi = ds_plume.attrs.get('mmsi')

        candidates = [
            f"no2_enhancement_with_plume_mask_{date}_{time_nounder}_{mmsi}.png"
        ]

        found = False
        for candidate in candidates:
            original_image_path = image_path / candidate
            #print(original_image_path)
            if original_image_path.exists():
                try:
                    dst_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(original_image_path, dst_path / candidate)

                    # classify by relative wind speed
                    rel_gt_dir = dst_path / "rel_wind_gt2"
                    rel_lt_dir = dst_path / "rel_wind_lt2"
                    rel_gt_dir.mkdir(parents=True, exist_ok=True)
                    rel_lt_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        rws = float(rel_wind_speed)
                    except Exception:
                        rws = None

                    if rws is not None and (not np.isnan(rws)):
                        if rws > 2.0:
                            shutil.copy(original_image_path, rel_gt_dir / candidate)
                        else:
                            shutil.copy(original_image_path, rel_lt_dir / candidate)

                    # classify by wind direction relative to ship
                    try:
                        wind_from = float(ship_pass.get('median_wind_dir'))
                        ship_course = float(ship_pass.get('Mean_Course'))
                    except Exception:
                        wind_from = None
                        ship_course = None

                    if wind_from is not None and ship_course is not None:
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
                        against_dir_high = dst_path / "wind_against_dir_high_speed"
                        same_dir_dir.mkdir(parents=True, exist_ok=True)
                        against_dir_dir.mkdir(parents=True, exist_ok=True)
                        against_dir_dir_low.mkdir(parents=True, exist_ok=True)
                        against_dir_high.mkdir(parents=True, exist_ok=True)

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
                            elif rws_check >2.0:
                                try:
                                    shutil.copy(original_image_path, against_dir_high / candidate)
                                    print(f"Copied to {against_dir_high}")
                                except Exception as e:
                                    print(f"Failed to copy to {against_dir_high}: {e}")
                            else:
                                print(f"Against-wind detected but relative wind {rws_check:.2f} < 2.0 m/s; skipping against_dir copy for {candidate}")

                    found = True
                    break
                except Exception as e:
                    return {"idx": idx, "status": "copy_failed", "error": str(e)}

        if not found:
            return {"idx": idx, "status": "no_image_found", "tried": candidates}

        return {"idx": idx, "status": "ok", "image": str(plume_file)}

    except Exception as e:
        return {"idx": idx, "status": "error", "error": str(e)}


# Run processing in parallel using a process pool
tasks = [(idx, row.to_dict()) for idx, row in df_all_ship_passes_filtered.iterrows()]

def run_ship_passes_parallel(tasks_list, max_workers=10):
    """Run the ship-pass tasks in parallel and return results list."""
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = list(exe.map(_process_ship_pass_task, tasks_list))
    return results


if __name__ == '__main__':
    if tasks:
        results = run_ship_passes_parallel(tasks, max_workers=10)
        for r in results:
            if r.get("status") not in ("ok", "no_plume_found"):
                print(f"Task result: {r}")

