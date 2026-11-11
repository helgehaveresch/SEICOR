import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import sys
import os
import copy
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import mask_zenith

def read_reference_intervals(log_file, max_minutes=5):
    """Time intervals (UT) of the reference ("zenith") measurements from an IMPACT _D.LOG file.

    The log lists every telescope move ("New telescope position: <elevation> elevation, <azimuth>
    azimuth") and the measurement programme (MULTIAXIS_PROGRAMMED_n = <elevation> <azimuth> <S|_> ...,
    S = scan). Every position whose elevation is not a scan elevation is a reference measurement
    (90/85.5 deg, 80 deg on 2026-05-17..29, 70 deg, test scans); it lasts until the next telescope
    move, at most max_minutes. The labels of these measurements in the spectra are not reliable
    (2026-05-17..29 they are labelled as off-axis), so they are taken from the log.
    Returns a DataFrame with columns start, end, elevation, azimuth (empty if nothing is found).
    """
    import re
    move = re.compile(r"^(\d\d)\.(\d\d)\.(\d{4}) (\d\d):(\d\d):(\d\d):(\d\d) New telescope position: "
                      r"([-\d.]+) elevation, ([-\d.]+) azimuth")
    programme = re.compile(r"^MULTIAXIS_PROGRAMMED_\d+ = +([-\d.]+) +([-\d.]+) +(\S)")
    moves, scan_elevations = [], set()
    with open(log_file, encoding="latin-1") as f:
        for line in f:
            m = move.match(line)
            if m:
                d, mo, y, hh, mm, ss, cc, elev, az = m.groups()
                t = pd.Timestamp(int(y), int(mo), int(d), int(hh), int(mm), int(ss), int(cc) * 10000)
                moves.append((t, float(elev), float(az)))
                continue
            m = programme.match(line)
            if m and m.group(3) == "S":
                scan_elevations.add(round(float(m.group(1)), 1))
    columns = ["start", "end", "elevation", "azimuth"]
    if not moves or not scan_elevations:
        return pd.DataFrame(columns=columns)
    max_duration = pd.Timedelta(minutes=max_minutes)
    rows = []
    for i, (t, elev, az) in enumerate(moves):
        if round(elev, 1) in scan_elevations:
            continue
        end = t + max_duration
        if i + 1 < len(moves):
            end = min(end, moves[i + 1][0])
        rows.append((t, end, elev, az))
    return pd.DataFrame(rows, columns=columns)


def mask_reference_measurements(ds, log_file, max_minutes=5, time_var="datetime"):
    """Drop the measurements taken during the reference ("zenith") measurements listed in log_file
    (see read_reference_intervals). Returns (ds without these measurements, the intervals)."""
    intervals = read_reference_intervals(log_file, max_minutes=max_minutes)
    if intervals.empty:
        return ds, intervals
    times_da = ds[time_var]
    if "viewing_direction" in times_da.dims:  # after mask_zenith the time variables carry viewing_direction
        times_da = times_da.max("viewing_direction")
    dim = times_da.dims[0]
    times = pd.to_datetime(times_da.values)
    drop = np.zeros(len(times), dtype=bool)
    for start, end in zip(intervals["start"], intervals["end"]):
        drop |= (times >= start) & (times < end)
    return ds.isel({dim: ~drop}), intervals


def calculate_LOS(ds_impact, instrument_location, line_length=0.05):
    """
    Calculate the line of sight (LOS) endpoints based on the instrument location and viewing direction.
    Parameters:
        ds_impact (xarray.Dataset): Dataset containing the measurements.
        instrument_location (list): [latitude, longitude] of the instrument.
        line_length (float): Length of the line segment to plot.
    Returns:
        endpoints_los (np.ndarray): Array of shape (N, 2) with the endpoints of the LOS.
    """
    lat1, lon1 = instrument_location
    viewing_direction = ds_impact["viewing-azimuth-angle"].isel(viewing_direction=0).values
    bearing_rad = np.deg2rad(viewing_direction)
    delta_lat = line_length * np.cos(bearing_rad)
    delta_lon = line_length * np.sin(bearing_rad) / np.cos(np.deg2rad(lat1))
    lat2 = lat1 + delta_lat
    lon2 = lon1 + delta_lon
    endpoints_los = np.column_stack([lat2, lon2])
    return endpoints_los

def calc_start_end_times(ds_impact):
    """
    Calculate the start and end times of the measurement period.
    """
    measurement_times = pd.to_datetime(ds_impact["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")
    start_time = measurement_times.min()
    end_time = measurement_times.max()
    return start_time, end_time, measurement_times

def rolling_impact(ds, window):
    """
    Apply rolling window to the IMPACT dataset.
    """
    ds["NO2_rolling"] = ds["a[NO2]"].rolling(dim_0=window).mean()
    ds["O4_rolling"] = ds["a[O4]"].rolling(dim_0=window).mean()
    return ds

def _measurement_times(ds):
    """Time of each measurement (dim_0), also if datetime was broadcast over viewing_direction."""
    times = ds["datetime"]
    if "viewing_direction" in times.dims:
        times = times.max(dim="viewing_direction")  # skips NaT of masked viewing directions
    return times.values


def coarsen_impact_measurements(ds, time_resolution_seconds):
    """Average the IMPACT measurements (dim_0) in fixed time bins of time_resolution_seconds.

    The bins are aligned to midnight UTC of the first measurement, so other datasets of the same
    day (e.g. the upwind DOAS fit) get the same bins. All measurements in a bin are averaged
    ignoring NaN, the fit errors sig[...] are propagated as sqrt(sum sig^2) / n, and datetime is
    the bin centre, i.e. a regular time grid. Bins without measurements are dropped, so gaps stay
    gaps. dim_0 keeps the index of the first raw measurement in each bin.

    attrs: coarsen_time_resolution_seconds, and coarsen_factor = bin length / median raw time
    step (about the number of raw measurements per bin), used to scale settings that count
    measurements or pixels. Returns ds unchanged if time_resolution_seconds is None or 0.
    """
    if not time_resolution_seconds:
        return ds
    times = _measurement_times(ds).astype("datetime64[ns]")
    if np.any(np.diff(times) < np.timedelta64(0, "ns")):
        order = np.argsort(times, kind="stable")
        ds, times = ds.isel(dim_0=order), times[order]
    res_ns = int(round(time_resolution_seconds * 1e9))
    t_ns = times.astype(np.int64)
    day0_ns = times[0].astype("datetime64[D]").astype("datetime64[ns]").astype(np.int64)
    bins = (t_ns - day0_ns) // res_ns
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])  # first raw measurement of each bin
    centres = (day0_ns + bins[starts] * res_ns + res_ns // 2).astype("datetime64[ns]")

    out = ds.isel(dim_0=starts)  # coordinates and variables without dim_0
    with np.errstate(invalid="ignore", divide="ignore"):  # all-NaN bins give NaN
        for name, var in ds.data_vars.items():
            if "dim_0" not in var.dims:
                continue
            dims = [d for d in var.dims if d != "dim_0"] + ["dim_0"]
            if np.issubdtype(var.dtype, np.datetime64):
                reduced = np.broadcast_to(centres, [var.sizes[d] for d in dims[:-1]] + [len(starts)]).copy()
            else:
                values = var.transpose(*dims).values.astype(float)
                valid = np.isfinite(values)
                count = np.add.reduceat(valid, starts, axis=-1)
                if name.startswith("sig["):
                    reduced = np.sqrt(np.add.reduceat(np.where(valid, values ** 2, 0.0), starts, axis=-1)) / count
                else:
                    reduced = np.add.reduceat(np.where(valid, values, 0.0), starts, axis=-1) / count
            out[name] = (dims, reduced, var.attrs)
            out[name] = out[name].transpose(*var.dims)
    out.attrs["coarsen_time_resolution_seconds"] = float(time_resolution_seconds)
    out.attrs["coarsen_factor"] = float(time_resolution_seconds / (np.median(np.diff(t_ns)) / 1e9))
    return out


# settings that count raw measurements or image pixels, as (section, key) of the processing settings;
# with coarsened IMPACT data, scale_sample_counts divides them by the coarsening factor
SAMPLE_COUNT_SETTINGS = [
    ("quality", "rolling_mean_window"),
    ("enhancement", "rolling_background_window"),
    ("reference_search", "min_count"),
    ("reference_search", "ztest_min_cluster_size"),
    ("detection", "plume_min_cluster_size"),
    ("detection", "ship_min_cluster_size"),
    ("plume_mask", "merge_min_cluster_size"),
    ("plume_mask", "merge_max_cluster_size"),
    ("plume_mask", "ship_min_cluster_size"),
    ("plume_mask", "stop_pixels"),
]


def scale_count(count, factor):
    """A number of raw measurements/pixels expressed in coarsened ones (at least 1; None stays None)."""
    if count is None:
        return None
    return max(1, int(round(count / factor)))


def scale_sample_counts(processing_settings, factor):
    """Copy of the processing settings with the SAMPLE_COUNT_SETTINGS divided by factor, so they keep
    their meaning in raw measurements when the IMPACT data is coarsened by factor."""
    scaled = copy.deepcopy(processing_settings)
    for section, key in SAMPLE_COUNT_SETTINGS:
        if key in scaled.get(section, {}):
            scaled[section][key] = scale_count(scaled[section][key], factor)
    return scaled

def rms_mask(ds, threshold=0.01, instrument = "IMPACT"):
    if instrument == "IMPACT":
        rms_mask = ds["rms"].mean(dim="viewing_direction") < threshold
    ds_masked = ds.where(rms_mask, drop=True)
    return ds_masked

def calculate_path_averaged_vmr_no2(df_insitu, ds):
    """
    Calculate NO2 volume mixing ratio (VMR) from coarsened IMPACT and in-situ data.
    !!! Note: This approach is not accurate and implements systematic errors 

    Args:
        df_insitu (pandas.DataFrame): In-situ dataframe with 'p_0', 'T_in', and 'time'.
        coarsened_ds (xr.Dataset): Coarsened IMPACT dataset with 'a[O4]', 'a[NO2]', and 'datetime'.

    Returns:
        xr.DataArray: VMR_NO2 aligned to coarsened_ds times [ppb].
    """
    # Calculate air number density [molec/m^3]
    n_air = df_insitu["p_0"] * 1e2 * 6.02214076e23 / (df_insitu["T_in"] + 273.15) / 8.314
    # Calculate O4 number density [molec^2/m^6]
    n_O4 = (0.20942 * n_air) ** 2

    # Interpolate n_O4 and n_air to coarsened_ds times
    n_O4_interp = pd.Series(n_O4.values, index=pd.to_datetime(df_insitu.index))
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(df_insitu.index))
    measurement_times = pd.to_datetime(ds["datetime"].isel(viewing_direction=0).values).tz_localize("UTC")
    n_O4_aligned = n_O4_interp.reindex(measurement_times, method='nearest').values
    n_air_aligned = n_air_interp.reindex(measurement_times, method='nearest').values

    # Calculate O4 effective path length [m]
    L_O4 = ds["O4_rolling"] * 1e40 * 1e10 / n_O4_aligned

    # Broadcast n_air_aligned to match coarsened_ds["a[NO2]"] shape
    n_air_aligned = xr.ones_like(ds["NO2_rolling"]) * np.array(n_air_aligned)

    # Calculate NO2 VMR [ppb]
    ds["VMR_NO2"] = ds["NO2_rolling"] * 1e4 / L_O4 / n_air_aligned * 1e9

    return ds

def mask_rms_and_reduce_impact(ds_impact, rms_threshold=0.01):
    rms_mean = ds_impact["rms"].sel(viewing_direction=slice(5, 8)).mean(dim="viewing_direction")
    mask = rms_mean < rms_threshold
    time_variable = ds_impact["datetime"].isel(viewing_direction = 0) 
    ds_impact_reduced = ds_impact.sel(viewing_direction=slice(5, 8)).mean(dim="viewing_direction")
    ds_impact_reduced["datetime"] = time_variable
    ds_impact_masked = ds_impact_reduced.where(mask, drop=True)
    return mask, ds_impact_masked

def _to_decimal_hours(ts):
    return (
            ts.hour
            + ts.minute / 60.0
            + ts.second / 3600.0
            + ts.microsecond / 3_600_000_000.0
        )

def add_offaxis_ref_fit_to_plume_ds(ds_plume, IMPACT_path, SC_file_name, coarsen_time_resolution_seconds=None):
    """
    Add off-axis reference fit information to the plume dataset.

    Args:
        ds_plume (xr.Dataset): The plume dataset.
        SC_file_name (str): The path to the off-axis reference fit dataset.
        coarsen_time_resolution_seconds: time bins of the IMPACT data the plume dataset was made
            from (None = not coarsened); the fit is coarsened the same way before it is matched in time.
    """
    date = pd.to_datetime(ds_plume["times_plume"].values)[0].strftime("%y%m%d")
    ds_off = read_SC_file_imaging(IMPACT_path, date, mode= SC_file_name)
    ds_off = mask_zenith(ds_off)
    # same viewing directions as the plume (the lowest ones may have been dropped)
    ds_off = ds_off.sel(viewing_direction=ds_plume["image_row"].values)
    ds_off = coarsen_impact_measurements(ds_off, coarsen_time_resolution_seconds)

    # small tolerance for edge matching
    t_tol = pd.Timedelta(milliseconds=1)
    mask = (
        (ds_off["datetime"].isel(viewing_direction=0) >= ds_plume["times_plume"][0] - t_tol)
        & (ds_off["datetime"].isel(viewing_direction=0) <= ds_plume["times_plume"][-1] + t_tol)
    )
    ds_off = ds_off.sel(dim_0=mask.values)

    # target shape from ds_plume
    n_rows = ds_plume.dims.get("image_row")
    n_cols = ds_plume.dims.get("window_plume")
    target_shape = (n_rows, n_cols)

    def _pad_or_trim(arr, target_shape):
        # produce float array filled with NaN and copy arr into top-left corner / broadcast where sensible
        out = np.full(target_shape, np.nan, dtype=float)
        if arr is None:
            return out
        arr = np.asarray(arr)
        if arr.size == 0:
            return out

        if arr.ndim == 1:
            # assume arr is a time series -> broadcast to rows if length matches columns
            if arr.shape[0] == target_shape[1]:
                out[:] = arr[np.newaxis, :].astype(float)
            elif arr.shape[0] == target_shape[0]:
                out[: arr.shape[0], :] = arr[:, np.newaxis].astype(float)
            else:
                minc = min(arr.shape[0], target_shape[1])
                out[0, :minc] = arr[:minc].astype(float)
        elif arr.ndim == 2:
            r, c = arr.shape
            # prefer orientation where first dimension matches image_row
            if r == target_shape[0]:
                minc = min(c, target_shape[1])
                out[:, :minc] = arr[:, :minc].astype(float)
            elif c == target_shape[0]:
                # transpose case
                arrT = arr.T
                minc = min(arrT.shape[1], target_shape[1])
                out[:, :minc] = arrT[:, :minc].astype(float)
            else:
                minr = min(r, target_shape[0])
                minc = min(c, target_shape[1])
                out[:minr, :minc] = arr[:minr, :minc].astype(float)
        else:
            # higher-dim -> try to squeeze to 2D if possible
            arr2 = arr.reshape((arr.shape[0], -1)) if arr.size >= 2 else arr.ravel()
            return _pad_or_trim(arr2, target_shape)
        return out

    no2_arr = ds_off["a[NO2]"].values if "a[NO2]" in ds_off else None
    o4_arr = ds_off["a[O4]"].values if "a[O4]" in ds_off else None
    rms_arr = ds_off["rms"].values if "rms" in ds_off else None

    no2_padded = _pad_or_trim(no2_arr, target_shape)
    o4_padded = _pad_or_trim(o4_arr, target_shape)
    rms_padded = _pad_or_trim(rms_arr, target_shape)

    ds_plume = ds_plume.assign(
        no2_upwind_ret=(["image_row", "window_plume"], no2_padded),
        o4_upwind_ret=(["image_row", "window_plume"], o4_padded),
        rms_upwind_ret=(["image_row", "window_plume"], rms_padded),
    )

    return ds_plume


def call_nlin_c_for_offaxis_ref_and_add_to_plume_ds(ds_plume, ship_pass_single, param_file, IMPACT_path,
                                                    coarsen_time_resolution_seconds=None):
    """
    Calls the nlin-c plume retrieval for each ship pass and saves the resulting plume datasets.
    """
    fit_window = pd.to_datetime(ds_plume["times_plume"].values)
    ref_window = pd.to_datetime(ds_plume["times_ref"].values)
    date_str = fit_window[0].strftime("%d.%m.%Y")
    print(fit_window[0])
    print(fit_window[-1])
    time_start = _to_decimal_hours(fit_window[0]) - 0.002
    time_end = _to_decimal_hours(fit_window[-1]) + 0.002
    ref_time_start = ref_window[0].strftime("%H:%M:%S")
    ref_time_end = ref_window[-1].strftime("%H:%M:%S")
    plume_name_str = os.path.basename(ship_pass_single['plume_file'])[0:-4]
    print(f'Fitting plume at {ds_plume.t}')
    param_file_modified = f"{param_file}_tmp_plume_{plume_name_str}"
    try: 
        with open(param_file, "r") as f:
            lines = f.readlines()
        
        with open(param_file_modified, "w") as f:
            for line in lines:
                if line.strip().startswith("START_DATE"):
                    f.write(f"START_DATE     = {date_str}\n")
                elif line.strip().startswith("END_DATE"):
                    f.write(f"END_DATE       = {date_str}\n")
                elif line.startswith("START_TIME"):
                    f.write(f"START_TIME     = {time_start}\n")
                elif line.startswith("END_TIME"):
                    f.write(f"END_TIME       = {time_end}\n")
                elif line.startswith("BACKGROUND_AVERAGING_START_TIME"):
                    f.write(f"BACKGROUND_AVERAGING_START_TIME = {ref_time_start}\n")
                elif line.startswith("BACKGROUND_AVERAGING_END_TIME"):
                    f.write(f"BACKGROUND_AVERAGING_END_TIME   = {ref_time_end}\n")
                elif line.strip().startswith("SLANT_EXTENSION"):
                    parts = line.split("=", 1)
                    val = parts[1].strip() if len(parts) > 1 else ""
                    new_val = f"{val}_{plume_name_str}"
                    f.write(f"SLANT_EXTENSION      = {new_val}\n")
                else:
                    f.write(line)
        print(f"Running nlin_c.exe on: {param_file_modified}")
        subprocess.run([r"P:\exe_64\nlin_c.exe", param_file_modified])
        SC_file_ending = "ID.NO2_VIS_upwind_" + plume_name_str
        ds_plume = add_offaxis_ref_fit_to_plume_ds(ds_plume, IMPACT_path, SC_file_name=SC_file_ending,
                                                   coarsen_time_resolution_seconds=coarsen_time_resolution_seconds)

    finally:
        try:
            os.remove(param_file_modified)
            print(f"Deleted temporary file: {param_file_modified}")
        except Exception as e:
            print(f"Error deleting {param_file_modified}: {e}")
    return ds_plume