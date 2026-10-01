#%%
"""
Compare the WRF wind model (v1_2, LIDAR test period May/June 2026) with

  (a) the Uni Hamburg meteomast at Billwerder (FF/DD at 50/110/175/280 m) and
  (b) the ZX wind LIDAR at Wedel (10-min averages at 10 ... 279 m).

Workflow
  1. Read the Billwerder minutely data and extract the model column above the mast.
  2. Build a time mask of model hours where model and mast agree within MASK_TOL m/s
     in u and v (same idea as the ``mask`` in wind_model_meas_comparison.py).
  3. Extract the model column above the ship corridor at the LIDAR heights and apply
     the mask.
  4. Read the LIDAR data, average it onto the model times and compare per height
     (speed, direction, u, v) with scatter plots, time series and a stats summary.
  5. (section 10) Compare ERA5 10 m / 100 m wind at the same point with WRF and the LIDAR.

Conventions (kept from wind_model_meas_comparison.py)
  u = FF * sin(DD), v = FF * cos(DD) for the measurements, i.e. the vector points
  towards the direction the wind comes *from*.  Model u/v are therefore negated, so
  that wdir = atan2(u, v) yields the meteorological "from" direction for both.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"   # needed for NetCDF files on the Q: network drive
import logging
import re
import sys
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
try:                                            # labels below contain Δ, ° and — ; a cp1252 console would choke
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------- paths
BILLWERDER_DIR = Path(r"D:\SEICOR\weatherstations\Billwerder")
MODEL_DIR = Path(r"D:\SEICOR\wind_model_v1_2")
MODEL_PREFIX = "seicor_wrfmeteo_lidartest_v1_2_d05"
MODEL_PREFIX_COARSE = "seicor_wrfmeteo_lidartest_v1_2_d04"   # next coarser domain, only used in section 10
LIDAR_DIR = Path(r"D:\SEICOR\wind_LIDAR")
LIDAR_STATION = "Wind10_1082"            # 10-min averages; "Wind_1082" would be the 1 s data
LIDAR_YEAR = 2026                        # only LIDAR files of this year are read; None -> all years
MAST_PAD = pd.Timedelta("1h")            # the meteomast is read for the LIDAR period +- this margin
                                         # (covers the centred AVG_WINDOW averages at the period edges)
ERA5_GRIB = Path(r"D:\SEICOR\ERA5\b0977d2507d43a7a4511f7726e749f39.grib")  # 10u/10v/100u/100v, hourly
CACHE_DIR = Path(r"D:\wrf_meteo\v1_2")   # extracted model columns are cached here as NetCDF
FORCE_REEXTRACT = False                  # True -> ignore cached model columns
SAVE_FIG_DIR = None                      # e.g. Path(r"D:\plots\lidar_model"); None -> only plt.show()

# ----------------------------------------------------------------------------- locations / heights
BILLWERDER_LAT, BILLWERDER_LON = 53.51922389077999, 10.10283134530118
BILLWERDER_HEIGHTS = [50, 110, 175, 280]
BILLWERDER_TZ = "+01:00"                 # mast files are 1-min averages stamped in MEZ = UTC+1 all year
                                         # (NOT Europe/Berlin: the files do not follow summer time)

SHIP_CORRIDOR_LAT, SHIP_CORRIDOR_LON = 53.56634576422346, 9.690192228678288
LIDAR_LAT, LIDAR_LON = 53.56957, 9.69191  # from the GPS column of the LIDAR files (~350 m N of the corridor)
MODEL_TARGET_NAME = "lidar"               # "ship_corridor" or "lidar" -> which column of the model is compared to the LIDAR
MODEL_TARGETS = {"ship_corridor": (SHIP_CORRIDOR_LAT, SHIP_CORRIDOR_LON), "lidar": (LIDAR_LAT, LIDAR_LON)}
# LIDAR range gates (38 m is a fixed reference gate and is skipped)
LIDAR_HEIGHTS = [10, 24, 39, 59, 79, 109, 139, 174, 219, 279]
# both model columns are extracted on the same heights, so that the two points in the domain can
# be compared against each other as well as against their respective instrument
MODEL_COLUMN_HEIGHTS = sorted(set(BILLWERDER_HEIGHTS) | set(LIDAR_HEIGHTS))
MODEL_POINT_HEIGHTS = BILLWERDER_HEIGHTS   # heights for the point-to-point model comparison
BILLWERDER_FOCUS_HEIGHT = 110              # height of the combined scatter + time series overview figure

# ----------------------------------------------------------------------------- comparison settings
MASK_HEIGHTS = [110]              # meteomast heights used for the model/mast agreement mask
MASK_HEIGHTS_LIDAR = [109]        # LIDAR gate closest to MASK_HEIGHTS, for the alternative mask
MASK_SOURCE = "meteomast"         # "meteomast" or "lidar": which of the two masks drives the comparison.
                                  # "lidar" is circular for the LIDAR-vs-model statistics below (it selects
                                  # exactly the hours on which the two agree) - use it as a diagnostic only.
MASK_TOL = 3.0            # m/s; keep model hours with |u_meas - u_model| < tol and |v_meas - v_model| < tol
# (meteomast height, LIDAR gate) pairs for the direct instrument-to-instrument comparison;
# the two sites are ~30 km apart, so this compares two locations, not two collocated sensors
MAST_LIDAR_PAIRS = [(50, 59), (110, 109), (175, 174), (280, 279)]
# both instruments are averaged into this common window for that comparison; the LIDAR record is
# natively 10-min, so "10min" passes it through unchanged and anything coarser aggregates it
MEAS_COMPARE_WINDOW = "20min"

# --------------------------------------------------- LIDAR 180 deg direction-flip correction
# The LIDAR reports the whole column ~180 deg off for stretches at a time (all range gates flip
# together, so the artefact cannot be found from the profile itself).  The meteomast is used as
# an independent polarity reference: it only decides *whether* a stretch is flipped, the
# correction itself is a fixed +180 deg, so no mast values enter the corrected LIDAR data.
# NOTE this makes the LIDAR-vs-meteomast *direction* statistics in section 8 partly circular
# (flipped samples are removed by construction); speeds and the model comparison are unaffected.
LIDAR_FLIP_CORRECT = True
LIDAR_FLIP_REF_MAST_HEIGHT = 110  # meteomast height used as the polarity reference
LIDAR_FLIP_TOL = 110.0            # deg; flag a sample when it sits further than this from the reference
LIDAR_FLIP_MIN_SPEED = 0        # m/s; below this the direction is too noisy on both sides to judge
LIDAR_FLIP_BRIDGE = 3             # samples; close holes of up to this many unjudgeable samples in a run
LIDAR_FLIP_MAX_DURATION = None    # e.g. pd.Timedelta("2h") to correct only short stretches; None = all
AVG_WINDOW = "1h"         # measurements are averaged in a window centred on each model time; None -> nearest sample
MIN_SPEED_DIR = 1.0       # m/s; wind direction is only compared when both model and measurement exceed this
# the LIDAR is compared to the model as a 20-min average centred on each model time (the two 10-min
# records at t-5 and t+5 min).  {"10min": None} would use the single 10-min sample nearest to each
# model time instead; add e.g. "1h": AVG_WINDOW to also compare the LIDAR averaged over a full model
# hour.  AVG_WINDOW itself still governs the meteomast.
LIDAR_AVERAGES = {"20min": "20min"}
PRIMARY_AVG = "20min"     # the averaging that gets the scatter and time series figures
PLOT_ALL_AVERAGES = False # True -> those figures are produced for every entry of LIDAR_AVERAGES

# --------------------------------------------------- ERA5 comparison (section 10)
ERA5_HEIGHTS = [10, 100]  # ERA5 levels; LIDAR and WRF are interpolated linearly in z where they lack the level
ERA5_USE_MASK = False     # True -> restrict the ERA5 comparison to the hours kept by the WRF agreement mask
# ERA5 agreement mask (section 11): ERA5 above the meteomast is compared with the mast exactly like the
# WRF mask (MASK_TOL, AVG_WINDOW); ERA5 has no 110 m level, so each mast height is paired with an ERA5 level
ERA5_MASK_HEIGHT_PAIRS = [(110, 100)]    # (meteomast height, ERA5 height)
ERA5_MASK_TOL = 3                    # m/s; tolerance of the ERA5 mask (the WRF mask uses MASK_TOL)


# ============================================================================= helpers: wind vectors / stats
def uv_from_speed_dir(speed, direction):
    """u = FF*sin(DD), v = FF*cos(DD) (vector pointing towards where the wind comes from)."""
    u = speed * np.sin(np.deg2rad(direction))
    v = speed * np.cos(np.deg2rad(direction))
    return u, v


def speed_dir_from_uv(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = (np.rad2deg(np.arctan2(u, v)) + 360) % 360
    return speed, direction


def add_uv_columns(df, heights):
    """Add u_wind_{h}/v_wind_{h} from wind_speed_{h}/wind_dir_{h} for every height."""
    for h in heights:
        df[f"u_wind_{h}"], df[f"v_wind_{h}"] = uv_from_speed_dir(df[f"wind_speed_{h}"], df[f"wind_dir_{h}"])
    return df


def wrap_180(d):
    return (np.asarray(d, float) + 180) % 360 - 180


def orthogonal_fit_stats(x, y, circular=False):
    """n, r, RMSE, bias (y - x), and an orthogonal (total least squares) fit y = slope*x + intercept.

    For circular=True the difference is wrapped to [-180, 180) and y is unwrapped
    relative to x before computing r and the fit.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    out = dict(n=int(m.sum()), r=np.nan, rmse=np.nan, bias=np.nan, slope=np.nan, intercept=np.nan)
    if out["n"] < 3:
        return out
    x, y = x[m], y[m]
    if circular:
        d = wrap_180(y - x)
        y = x + d
    else:
        d = y - x
    out["rmse"] = float(np.sqrt(np.mean(d**2)))
    out["bias"] = float(np.mean(d))
    out["r"] = float(np.corrcoef(x, y)[0, 1])
    X = np.column_stack([x, y])
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    dx, dy = Vt[0]
    if abs(dx) > 1e-12:
        out["slope"] = float(dy / dx)
        out["intercept"] = float(X.mean(axis=0)[1] - out["slope"] * X.mean(axis=0)[0])
    return out


def scatter_with_fit(ax, x, y, xlabel, ylabel, title="", circular=False, n_available=None, lims=None):
    """Scatter x vs y with 1:1 line, orthogonal fit and stats in the title. Returns the stats dict.

    n_available: number of finite pairs before the mask was applied; when given, the title
    reports "kept N, discarded M" instead of just the number of plotted points.
    lims: (lo, hi) used for both axes instead of the range of x and y, e.g. to give a masked and
    an unmasked panel the same scale (ignored for circular, which is always 0-360).

    circular=True treats x and y as directions in degrees: the fit, r, RMSE and bias are all
    computed on y unwrapped relative to x (see orthogonal_fit_stats), so a pair such as
    (350 deg, 10 deg) counts as a 20 deg difference instead of 340 deg.  Fitting the raw 0-360
    values instead - as the older wind_model_meas_comparison.py does - puts the wrapped points
    in the far corners of the panel and drags the slope towards 2.
    """
    st = orthogonal_fit_stats(x, y, circular=circular)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=12, alpha=0.6)
    if circular:
        lo, hi = 0.0, 360.0
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    elif lims is not None:
        lo, hi = lims
    elif m.any():
        lo = min(np.nanmin(x[m]), np.nanmin(y[m]))
        hi = max(np.nanmax(x[m]), np.nanmax(y[m]))
    else:
        lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="1:1")
    if np.isfinite(st["slope"]):
        label_fit = f"orth. fit: slope={st['slope']:.2f}, offset={st['intercept']:.2f}"
        if circular:
            # the fit was made on y unwrapped relative to x, so the line has to be wrapped back
            # into [0, 360) for drawing and broken wherever it crosses the wrap
            xv = np.linspace(lo, hi, 721)
            yv = (st["slope"] * xv + st["intercept"]) % 360.0
            yv = np.where(np.abs(np.diff(yv, prepend=yv[0])) > 180.0, np.nan, yv)
            ax.plot(xv, yv, "k-", linewidth=1.2, label=label_fit)
        else:
            xv = np.linspace(lo, hi, 50)
            ax.plot(xv, st["slope"] * xv + st["intercept"], "k-", linewidth=1.2, label=label_fit)
            if lims is None:
                pad = 0.05 * (hi - lo) if hi > lo else 1.0   # keep the axes on the data, not on a steep fit line
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
    if lims is not None and not circular:
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
    n_txt = f"n={st['n']}" if n_available is None else f"kept {st['n']}, discarded {n_available - st['n']}"
    stats_txt = f"r={st['r']:.2f}, RMSE={st['rmse']:.2f}, bias={st['bias']:.2f}"
    ax.set_title(f"{title}\n{stats_txt}", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return st


def finish_figure(fig, name):
    if SAVE_FIG_DIR is not None:
        SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================================= helpers: WRF model column
def find_closest_grid_indices(ds: xr.Dataset, lat_pt: float, lon_pt: float):
    """Indices of the four grid cells surrounding (lat_pt, lon_pt) and the bilinear fractions.

    Returns (sn_lo, sn_hi, we_lo, we_hi, t_lat, t_lon) with t_* in [0, 1].
    """
    lat_grid = ds["latitude"].values
    lon_grid = ds["longitude"].values

    mperlon = 111320.0 * np.cos(np.deg2rad(lat_pt))
    mperlat = 111132.0
    dist = np.hypot((lat_grid - lat_pt) * mperlat, (lon_grid - lon_pt) * mperlon)
    iy, ix = np.unravel_index(int(np.argmin(dist)), lat_grid.shape)

    lat_along_sn = lat_grid[:, ix]
    lon_along_we = lon_grid[iy, :]
    sn_pos = int(np.searchsorted(lat_along_sn, lat_pt))
    sn_lo, sn_hi = max(0, sn_pos - 1), min(lat_along_sn.size - 1, sn_pos)
    we_pos = int(np.searchsorted(lon_along_we, lon_pt))
    we_lo, we_hi = max(0, we_pos - 1), min(lon_along_we.size - 1, we_pos)

    lat1, lat2 = lat_grid[sn_lo, we_lo], lat_grid[sn_hi, we_lo]
    lon1, lon2 = lon_grid[sn_lo, we_lo], lon_grid[sn_lo, we_hi]
    dlat = (lat2 - lat1) if lat2 != lat1 else 1.0
    dlon = (lon2 - lon1) if lon2 != lon1 else 1.0
    t_lat = float(np.clip((lat_pt - lat1) / dlat, 0.0, 1.0))
    t_lon = float(np.clip((lon_pt - lon1) / dlon, 0.0, 1.0))
    return sn_lo, sn_hi, we_lo, we_hi, t_lat, t_lon


def _interp_column(values, heights, target_zs):
    """Linear vertical interpolation of values (nt, nz) on heights (nz,) to target_zs -> (nt, len(target_zs))."""
    order = np.argsort(heights)
    h = heights[order]
    out = np.empty((values.shape[0], len(target_zs)), dtype=float)
    for ti in range(values.shape[0]):
        out[ti] = np.interp(target_zs, h, values[ti, order])
    return out


def _with_time(da):
    """Files with a single time step store their variables without the time dimension."""
    return da if "time" in da.dims else da.expand_dims("time")


def read_wrf_point_timeseries(folder, prefix, targets: dict):
    """Bilinearly interpolated model columns above several points, in one pass through the files.

    targets: {name: (lat, lon, heights)}.  Returns {name: Dataset} with u, v, wspd, wdir on
    (time, z) and u10m, v10m, wspd10m, wdir10m, pblh, sw_down on (time,).

    The 10 m diagnostic wind (u10, v10) is prepended to the column at z = 10 m so that
    heights below the lowest model level (~28 m) are interpolated instead of clipped.
    Files are stored uncompressed and contiguous, so reading a 2x2 slab costs the same as a
    single column (~20 s per 3-D variable and file over the network).
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.name.startswith(prefix) and p.suffix.lower() == ".nc")
    if not files:
        raise FileNotFoundError(f"No files starting with {prefix!r} in {folder}")

    parts = {name: [] for name in targets}
    grid_sw = {}
    for path in files:
        logging.info("Reading model columns from %s", path.name)
        with xr.open_dataset(path) as ds:
            times = np.atleast_1d(ds["time"].values)
            nt = times.size
            for name, (target_lat, target_lon, heights) in targets.items():
                target_zs = np.asarray(heights, float)
                sn_lo, sn_hi, we_lo, we_hi, t_lat, t_lon = find_closest_grid_indices(ds, target_lat, target_lon)
                slab = ds[["u", "v", "u10", "v10", "pblh", "swdown", "height"]].isel(
                    south_north=slice(sn_lo, sn_hi + 1), west_east=slice(we_lo, we_hi + 1)).load()
                grid_sw.setdefault(name, (float(slab["latitude"].values[0, 0]), float(slab["longitude"].values[0, 0])))
                corners = [
                    (sn_lo, we_lo, (1 - t_lon) * (1 - t_lat)),
                    (sn_lo, we_hi, t_lon * (1 - t_lat)),
                    (sn_hi, we_lo, (1 - t_lon) * t_lat),
                    (sn_hi, we_hi, t_lon * t_lat),
                ]
                u = np.zeros((nt, target_zs.size))
                v = np.zeros((nt, target_zs.size))
                surf = {k: np.zeros(nt) for k in ("u10m", "v10m", "pblh", "sw_down")}
                for sn, we, w in corners:
                    if w == 0.0:
                        continue
                    col = slab.isel(south_north=sn - sn_lo, west_east=we - we_lo)
                    h = np.concatenate([[10.0], col["height"].values])
                    u10 = _with_time(col["u10"]).values
                    v10 = _with_time(col["v10"]).values
                    # sign convention: see module docstring
                    u_col = np.column_stack([-u10, -_with_time(col["u"]).transpose("time", "bottom_top").values])
                    v_col = np.column_stack([-v10, -_with_time(col["v"]).transpose("time", "bottom_top").values])
                    u += w * _interp_column(u_col, h, target_zs)
                    v += w * _interp_column(v_col, h, target_zs)
                    surf["u10m"] += w * -u10
                    surf["v10m"] += w * -v10
                    surf["pblh"] += w * _with_time(col["pblh"]).values
                    surf["sw_down"] += w * _with_time(col["swdown"]).values

                wspd, wdir = speed_dir_from_uv(u, v)
                wspd10, wdir10 = speed_dir_from_uv(surf["u10m"], surf["v10m"])
                parts[name].append(xr.Dataset(
                    data_vars={
                        "u": (("time", "z"), u),
                        "v": (("time", "z"), v),
                        "wspd": (("time", "z"), wspd),
                        "wdir": (("time", "z"), wdir),
                        "u10m": (("time",), surf["u10m"]),
                        "v10m": (("time",), surf["v10m"]),
                        "wspd10m": (("time",), wspd10),
                        "wdir10m": (("time",), wdir10),
                        "pblh": (("time",), surf["pblh"]),
                        "sw_down": (("time",), surf["sw_down"]),
                    },
                    coords={"time": times, "z": target_zs},
                ))

    out = {}
    for name, (target_lat, target_lon, _) in targets.items():
        ds_out = xr.concat(parts[name], dim="time")
        _, keep = np.unique(ds_out["time"].values, return_index=True)   # drop duplicated hours between files
        ds_out = ds_out.isel(time=np.sort(keep))
        ds_out.attrs.update(target_lat=target_lat, target_lon=target_lon,
                            grid_lat_sw=grid_sw[name][0], grid_lon_sw=grid_sw[name][1],
                            source=str(folder / f"{prefix}*"))
        out[name] = ds_out
    return out


def load_model_columns(targets: dict, prefix=MODEL_PREFIX):
    """Cached wrapper around read_wrf_point_timeseries; only targets without a valid cache file are extracted."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    missing = {}
    for name, (lat, lon, heights) in targets.items():
        cache = CACHE_DIR / f"{name}_{prefix}_model_ds.nc"
        heights = np.asarray(heights, float)
        if cache.exists() and not FORCE_REEXTRACT:
            ds = xr.open_dataset(cache).load()
            ds.close()
            if (ds["z"].size == heights.size and np.allclose(ds["z"].values, heights)
                    and np.isclose(ds.attrs.get("target_lat", np.nan), lat)
                    and np.isclose(ds.attrs.get("target_lon", np.nan), lon)):
                logging.info("Loaded cached model column %s (%d times)", cache.name, ds.sizes["time"])
                out[name] = ds
                continue
            logging.info("Cache %s does not match requested heights/location -> re-extracting", cache.name)
        missing[name] = (lat, lon, heights)
    if missing:
        for name, ds in read_wrf_point_timeseries(MODEL_DIR, prefix, missing).items():
            cache = CACHE_DIR / f"{name}_{prefix}_model_ds.nc"
            ds.to_netcdf(cache)
            logging.info("Extracted model column %s: %s .. %s (%d times), cached to %s",
                         name, ds["time"].values[0], ds["time"].values[-1], ds.sizes["time"], cache)
            out[name] = ds
    return out


# ============================================================================= helpers: measurements
def read_billwerder(dir_path, heights=BILLWERDER_HEIGHTS, tz=BILLWERDER_TZ, start=None, end=None):
    """Minutely Uni Hamburg mast files (one value per line), e.g. FF110_202503010000-202510312359.txt.

    The period is taken from the file name; all files found per variable/height are stacked.
    start/end (tz-aware timestamps, optional) restrict the read to that period: only the lines
    inside it are parsed, so a short period out of a multi-year file loads quickly.
    Returns a DataFrame indexed by UTC time with wind_speed_{h}, wind_dir_{h}, wind_max_{h},
    u_wind_{h}, v_wind_{h}.

    Time stamps: the files hold one value per minute in MEZ = UTC+1 with no summer-time jump,
    so `tz` must be the fixed offset "+01:00".  With "Europe/Berlin" the generated index would
    end 60 values short of the file whenever the period spans a spring-forward (pandas steps a
    minute frequency in absolute time but stops at the local end label), and it would put every
    value one hour too early if a file were to start inside summer time.
    """
    dir_path = Path(dir_path)
    pattern = re.compile(r"_(\d{12})-(\d{12})\.txt$")
    columns = {}
    for var, code in (("wind_speed", "FF"), ("wind_dir", "DD"), ("wind_max", "FB")):
        for h in heights:
            pieces = []
            for fp in sorted(dir_path.glob(f"{code}{h:03d}_*.txt")):
                m = pattern.search(fp.name)
                if m is None:
                    logging.warning("Cannot parse period from %s, skipped", fp.name)
                    continue
                t0, t1 = (pd.to_datetime(s, format="%Y%m%d%H%M") for s in m.groups())
                times = pd.date_range(t0, t1, freq="min", tz=tz).tz_convert("UTC")
                if start is None and end is None:
                    vals = np.loadtxt(fp, dtype=float)
                    n = min(vals.size, times.size)
                    if vals.size != times.size:
                        logging.warning("%s: %d values but %d minutes in the file name, truncating to %d "
                                        "(check the time zone / the file period)", fp.name, vals.size, times.size, n)
                    pieces.append(pd.Series(vals[:n], index=times[:n]))
                    continue
                # line i of the file belongs to times[i] -> read only the lines of the requested period
                i0 = 0 if start is None else int(times.searchsorted(start, side="left"))
                i1 = times.size if end is None else int(times.searchsorted(end, side="right"))
                if i1 <= i0:
                    continue                                  # file does not overlap the period
                vals = pd.read_csv(fp, header=None, skiprows=i0, nrows=i1 - i0, dtype=float).iloc[:, 0].values
                if vals.size != i1 - i0:
                    logging.warning("%s: only %d of the %d requested values present (check the file period)",
                                    fp.name, vals.size, i1 - i0)
                pieces.append(pd.Series(vals, index=times[i0:i0 + vals.size]))
            if pieces:
                columns[f"{var}_{h}"] = pd.concat(pieces).sort_index()
    if not columns:
        raise FileNotFoundError(f"No Billwerder files found in {dir_path}")
    df = pd.DataFrame(columns)
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace(99999, np.nan)
    df.index.name = "time"
    logging.info("Billwerder mast: %s .. %s (%d minutes)", df.index[0], df.index[-1], len(df))
    return add_uv_columns(df, heights)


def read_csv_from_zip(zip_path, inner_csv_name, **read_csv_kwargs):
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        with zf.open(inner_csv_name, mode="r") as f:
            return pd.read_csv(f, **read_csv_kwargs)


def read_lidar(dir_path, station=LIDAR_STATION, heights=LIDAR_HEIGHTS, year=LIDAR_YEAR):
    """All daily {station}@Y2026_M05_D18.CSV.zip files in dir_path -> DataFrame indexed by UTC time.

    year: only files of this year are read (None -> all).

    The file header states "Time sync: UTC +0 hrs" and that time stamps mark the *beginning*
    of the 10-min averaging period; the index is therefore shifted to the centre of the period.
    Columns: wind_speed_{h}, wind_dir_{h}, wind_speed_std_{h}, packets_{h}, u_wind_{h}, v_wind_{h},
    met_wind_speed, met_wind_dir.
    """
    dir_path = Path(dir_path)
    zips = sorted(dir_path.glob(f"{station}@Y{'*' if year is None else year}_M*_D*.CSV.zip"))
    if not zips:
        raise FileNotFoundError(f"No {station} CSV zips{'' if year is None else f' for {year}'} in {dir_path}")
    frames = []
    for zp in zips:
        try:
            frames.append(read_csv_from_zip(zp, zp.name[:-4], sep=",", skiprows=1))
        except Exception as e:
            logging.warning("Failed to read %s: %s", zp.name, e)
    raw = pd.concat(frames, ignore_index=True).replace([9998.0, 9999.0], np.nan)

    time = pd.to_datetime(raw["Time and Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    time = time.dt.tz_localize("UTC")
    averaging_period = pd.Timedelta("10min") if station.startswith("Wind10") else pd.Timedelta("1s")
    df = pd.DataFrame(index=pd.DatetimeIndex(time + averaging_period / 2, name="time"))
    for h in heights:
        df[f"wind_speed_{h}"] = raw[f"Horizontal Wind Speed (m/s) at {h}m"].values
        df[f"wind_dir_{h}"] = raw[f"Wind Direction (deg) at {h}m"].values
        df[f"wind_speed_std_{h}"] = raw[f"Horizontal Wind Speed Std. Dev. (m/s) at {h}m"].values
        df[f"packets_{h}"] = raw[f"Packets in Average at {h}m"].values
    df["met_wind_speed"] = raw["Met Wind Speed (m/s)"].values
    df["met_wind_dir"] = raw["Met Wind Direction (deg)"].values
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logging.info("LIDAR %s: %s .. %s (%d records from %d files)", station, df.index[0], df.index[-1], len(df), len(zips))
    return add_uv_columns(df, heights)


def lidar_column_direction(lidar_df, heights=None):
    """Circular mean of the wind direction over all range gates (the flip moves the whole column)."""
    heights = LIDAR_HEIGHTS if heights is None else heights
    ang = np.deg2rad(np.column_stack([lidar_df[f"wind_dir_{h}"].values for h in heights]))
    with np.errstate(invalid="ignore"):
        mean_dir = np.rad2deg(np.arctan2(np.nanmean(np.sin(ang), axis=1),
                                         np.nanmean(np.cos(ang), axis=1)))
    return (mean_dir + 360) % 360


def detect_direction_flips(lidar_df, mast_df, heights=None, ref_mast_height=None, tol=None,
                           min_speed=None, bridge=None, max_duration=None):
    """Find the stretches in which the LIDAR direction is ~180 deg off, using the mast as reference.

    Returns (flip_mask, runs) with flip_mask a boolean array over lidar_df.index and runs a
    DataFrame of the detected stretches (start, end, duration, samples, mean offset, mean speed,
    and whether the stretch was corrected or skipped because of max_duration).
    """
    heights = LIDAR_HEIGHTS if heights is None else heights
    ref_mast_height = LIDAR_FLIP_REF_MAST_HEIGHT if ref_mast_height is None else ref_mast_height
    tol = LIDAR_FLIP_TOL if tol is None else tol
    min_speed = LIDAR_FLIP_MIN_SPEED if min_speed is None else min_speed
    bridge = LIDAR_FLIP_BRIDGE if bridge is None else bridge
    max_duration = LIDAR_FLIP_MAX_DURATION if max_duration is None else max_duration

    t = lidar_df.index
    col_dir = lidar_column_direction(lidar_df, heights)
    col_speed = np.nanmean(np.column_stack([lidar_df[f"wind_speed_{h}"].values for h in heights]), axis=1)
    ref = resample_measurements(mast_df, BILLWERDER_HEIGHTS,
                                window=pd.infer_freq(t) or "10min").reindex(t)
    ref_dir = ref[f"wind_dir_{ref_mast_height}"].values
    ref_speed = ref[f"wind_speed_{ref_mast_height}"].values

    offset = wrap_180(col_dir - ref_dir)
    judgeable = np.isfinite(offset) & (col_speed > min_speed) & (ref_speed > min_speed)
    flagged = judgeable & (np.abs(offset) > tol)

    # group flagged samples into runs, closing holes of up to `bridge` unjudgeable samples
    flip_mask = np.zeros(len(t), dtype=bool)
    rows = []
    i = 0
    while i < len(flagged):
        if not flagged[i]:
            i += 1
            continue
        a = b = i
        j = i + 1
        while j < len(flagged):
            if flagged[j]:
                b, j = j, j + 1
            elif (not judgeable[j]) and (j - b) <= bridge:
                j += 1
            else:
                break
        seg = slice(a, b + 1)
        duration = t[b] - t[a] + (t[1] - t[0] if len(t) > 1 else pd.Timedelta(0))
        corrected = max_duration is None or duration <= max_duration
        if corrected:
            flip_mask[seg] = True
        rows.append(dict(start=t[a], end=t[b], duration=duration, samples=b - a + 1,
                         flagged=int(flagged[seg].sum()), mean_offset=float(np.nanmean(np.abs(offset[seg]))),
                         mean_speed=float(np.nanmean(col_speed[seg])), corrected=corrected))
        i = b + 1
    return flip_mask, pd.DataFrame(rows)


def apply_direction_flip(lidar_df, flip_mask, heights=None):
    """Add 180 deg to the direction of the flagged samples (and flip u/v accordingly)."""
    heights = LIDAR_HEIGHTS if heights is None else heights
    out = lidar_df.copy()
    for h in heights:
        d = out[f"wind_dir_{h}"].values.copy()
        d[flip_mask] = (d[flip_mask] + 180.0) % 360.0
        out[f"wind_dir_{h}"] = d
        for comp in ("u_wind", "v_wind"):
            c = out[f"{comp}_{h}"].values.copy()
            c[flip_mask] = -c[flip_mask]
            out[f"{comp}_{h}"] = c
    return out


def resample_measurements(meas_df, heights, window="10min", label_centre=True):
    """Average a measurement DataFrame into fixed windows (default the LIDAR's own 10 min).

    Speeds are averaged as scalars, directions are recomputed from the vector-averaged u/v.
    label_centre shifts the bin label from the start to the centre of the window, which is where
    the LIDAR stamps already sit after read_lidar(), so both instruments end up on one grid.
    """
    out = meas_df.select_dtypes("number").resample(window).mean()
    for h in heights:
        out[f"wind_dir_{h}"] = speed_dir_from_uv(out[f"u_wind_{h}"], out[f"v_wind_{h}"])[1]
    if label_centre:
        out.index = out.index + pd.Timedelta(window) / 2
    return out


def average_to_model_times(meas_df, model_times, window=AVG_WINDOW):
    """Bring a measurement DataFrame onto the (hourly, UTC-naive) model times.

    window=None : nearest sample within 30 min
    otherwise   : mean over [t - window/2, t + window/2) for every model time t.
    Speeds are averaged as scalars; directions are recomputed from the averaged u/v.
    """
    model_times = pd.DatetimeIndex(model_times)
    df = meas_df.select_dtypes("number").copy()
    df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
    dir_cols = [c for c in df.columns if c.startswith("wind_dir_")]

    if window is None:
        pos = df.index.get_indexer(model_times, method="nearest", tolerance=pd.Timedelta("30min"))
        out = df.iloc[np.where(pos < 0, 0, pos)].copy()
        out.index = model_times
        out[pos < 0] = np.nan
        return out

    half = pd.Timedelta(window) / 2
    shifted = df.drop(columns=dir_cols)
    shifted.index = shifted.index + half            # bins [t-half, t+half) are then labelled t
    out = shifted.resample(window).mean().reindex(model_times)
    for c in dir_cols:
        h = c.split("_")[-1]
        out[c] = speed_dir_from_uv(out[f"u_wind_{h}"], out[f"v_wind_{h}"])[1]
    return out


def build_agreement_mask(model_ds, meas_at_model_times, heights, tol=MASK_TOL, source="measurement",
                         model_heights=None):
    """True for model times where model and measurement u/v agree within tol at all given heights.

    model_heights: model level compared with each entry of `heights` (default: the same heights),
    for models without a level at the measurement height (ERA5: 100 m vs the mast's 110 m).

    Returns (mask, evaluable, available)
      mask      - agreement within tol; False wherever the comparison cannot be made
      evaluable - model and measurement both finite at every height, i.e. where `mask` carries
                  information (needed to compare two masks over a common set of hours)
      available - False if the measurement does not overlap the model period at all; the mask
                  returned is then all-True so the rest of the script still runs unmasked
    """
    nt = model_ds.sizes["time"]
    mask = np.ones(nt, dtype=bool)
    evaluable = np.ones(nt, dtype=bool)
    model_heights = heights if model_heights is None else model_heights
    for h, hm in zip(heights, model_heights):
        um = model_ds["u"].sel(z=hm).values
        vm = model_ds["v"].sel(z=hm).values
        uo = meas_at_model_times[f"u_wind_{h}"].values
        vo = meas_at_model_times[f"v_wind_{h}"].values
        finite = np.isfinite(uo) & np.isfinite(vo) & np.isfinite(um) & np.isfinite(vm)
        evaluable &= finite
        mask &= finite & (np.abs(uo - um) < tol) & (np.abs(vo - vm) < tol)
    if not evaluable.any():
        logging.warning("%s has no data in the model period (%s .. %s) -> mask disabled, all %d model hours kept",
                        source, model_ds["time"].values[0], model_ds["time"].values[-1], nt)
        return np.ones(nt, dtype=bool), np.zeros(nt, dtype=bool), False
    logging.info("%s mask (%s m, tol %.1f m/s): kept %d of %d evaluable hours (%d model hours in total)",
                 source, heights, tol, int(mask.sum()), int(evaluable.sum()), nt)
    return mask, evaluable, True


# ============================================================================= helpers: ERA5
ERA5_SHORT_NAMES = {"10u": ("u", 10), "10v": ("v", 10), "100u": ("u", 100), "100v": ("v", 100)}


def _era5_bilinear_weights(h, lat_pt, lon_pt):
    """[(flat index, weight), ...] of the four regular_ll grid points surrounding (lat_pt, lon_pt)."""
    import eccodes
    ni = eccodes.codes_get(h, "Ni")
    lat1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
    lon1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
    dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
    dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
    if not eccodes.codes_get(h, "jScansPositively"):
        dlat = -dlat
    fj = (lat_pt - lat1) / dlat
    fi = ((lon_pt - lon1) % 360.0) / dlon
    j0, i0 = int(np.floor(fj)), int(np.floor(fi))
    tj, ti = fj - j0, fi - i0
    return [(j0 * ni + i0, (1 - tj) * (1 - ti)), (j0 * ni + i0 + 1, (1 - tj) * ti),
            ((j0 + 1) * ni + i0, tj * (1 - ti)), ((j0 + 1) * ni + i0 + 1, tj * ti)]


def read_era5_points(grib_path, targets: dict):
    """Bilinearly interpolated ERA5 10 m / 100 m wind above several points, in one pass through the GRIB.

    targets: {name: (lat, lon)}.  Returns {name: Dataset} with u, v, wspd, wdir on (time, z),
    z = [10, 100], time = validity time (UTC-naive).  u/v are negated like the WRF columns
    (see module docstring), so wdir is the meteorological "from" direction.
    """
    import eccodes
    values = {name: {} for name in targets}     # name -> {(time, comp, z): value}
    weights = None
    n_msg = 0
    with open(grib_path, "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                short = eccodes.codes_get(h, "shortName")
                if short not in ERA5_SHORT_NAMES:
                    continue
                if weights is None:
                    weights = {name: _era5_bilinear_weights(h, lat, lon) for name, (lat, lon) in targets.items()}
                t = pd.Timestamp(f"{eccodes.codes_get(h, 'validityDate'):08d}"
                                 f"{eccodes.codes_get(h, 'validityTime'):04d}")
                field = eccodes.codes_get_values(h)
                comp, z = ERA5_SHORT_NAMES[short]
                for name, w in weights.items():
                    values[name][(t, comp, z)] = sum(field[k] * wk for k, wk in w)
                n_msg += 1
                if n_msg % 200 == 0:
                    logging.info("ERA5: %d messages read (at %s)", n_msg, t)
            finally:
                eccodes.codes_release(h)
    if weights is None:
        raise ValueError(f"No 10u/10v/100u/100v messages in {grib_path}")

    out = {}
    zs = sorted({z for _, z in ERA5_SHORT_NAMES.values()})
    for name, (lat, lon) in targets.items():
        s = pd.Series(values[name])
        times = pd.DatetimeIndex(sorted(s.index.get_level_values(0).unique()))
        u = np.column_stack([-s.xs(("u", z), level=(1, 2)).reindex(times).values for z in zs])
        v = np.column_stack([-s.xs(("v", z), level=(1, 2)).reindex(times).values for z in zs])
        wspd, wdir = speed_dir_from_uv(u, v)
        out[name] = xr.Dataset(
            data_vars={"u": (("time", "z"), u), "v": (("time", "z"), v),
                       "wspd": (("time", "z"), wspd), "wdir": (("time", "z"), wdir)},
            coords={"time": times.values, "z": np.asarray(zs, float)},
            attrs=dict(target_lat=lat, target_lon=lon, source=str(grib_path)))
    return out


def load_era5_points(targets: dict):
    """Cached wrapper around read_era5_points (the GRIB is ~2 GB on the network drive)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out, missing = {}, {}
    for name, (lat, lon) in targets.items():
        cache = CACHE_DIR / f"{name}_era5_{ERA5_GRIB.stem}.nc"
        if cache.exists() and not FORCE_REEXTRACT:
            ds = xr.open_dataset(cache).load()
            ds.close()
            if (np.isclose(ds.attrs.get("target_lat", np.nan), lat)
                    and np.isclose(ds.attrs.get("target_lon", np.nan), lon)):
                logging.info("Loaded cached ERA5 point %s (%d times)", cache.name, ds.sizes["time"])
                out[name] = ds
                continue
        missing[name] = (lat, lon)
    if missing:
        for name, ds in read_era5_points(ERA5_GRIB, missing).items():
            cache = CACHE_DIR / f"{name}_era5_{ERA5_GRIB.stem}.nc"
            ds.to_netcdf(cache)
            logging.info("Extracted ERA5 point %s: %s .. %s (%d times), cached to %s",
                         name, ds["time"].values[0], ds["time"].values[-1], ds.sizes["time"], cache)
            out[name] = ds
    return out


def lidar_at_level(lidar_df, h):
    """LIDAR at height h (linear between the neighbouring gates) -> DataFrame with wspd, wdir, u, v."""
    gates = np.asarray(LIDAR_HEIGHTS, float)
    if h in LIDAR_HEIGHTS:
        lo = hi = h
        w = 0.0
    else:
        i = int(np.clip(np.searchsorted(gates, h), 1, gates.size - 1))
        lo, hi = LIDAR_HEIGHTS[i - 1], LIDAR_HEIGHTS[i]
        w = (h - lo) / (hi - lo)
    col = {var: (1 - w) * lidar_df[f"{src}_{lo}"].values + w * lidar_df[f"{src}_{hi}"].values
           for var, src in (("wspd", "wind_speed"), ("u", "u_wind"), ("v", "v_wind"))}
    col["wdir"] = speed_dir_from_uv(col["u"], col["v"])[1]
    return pd.DataFrame(col, index=lidar_df.index)


def column_at_level(ds, h, times):
    """WRF/ERA5 column Dataset at height h (linear in z) on the given times -> DataFrame with wspd, wdir, u, v."""
    lev = ds[["wspd", "u", "v"]].interp(z=float(h))
    df = pd.DataFrame({var: lev[var].values for var in ("wspd", "u", "v")},
                      index=pd.DatetimeIndex(ds["time"].values)).reindex(times)
    df["wdir"] = speed_dir_from_uv(df["u"].values, df["v"].values)[1]
    return df


# ============================================================================= 1. Billwerder mast and model column above it
#%%
# the LIDAR is read first: its period decides which part of the (multi-year) mast record is read
lidar_raw = read_lidar(LIDAR_DIR)
billwerder = read_billwerder(BILLWERDER_DIR, start=lidar_raw.index[0] - MAST_PAD, end=lidar_raw.index[-1] + MAST_PAD)
target_lat, target_lon = MODEL_TARGETS[MODEL_TARGET_NAME]
model_columns = load_model_columns({
    "billwerder": (BILLWERDER_LAT, BILLWERDER_LON, MODEL_COLUMN_HEIGHTS),
    MODEL_TARGET_NAME: (target_lat, target_lon, MODEL_COLUMN_HEIGHTS),
})
billwerder_model = model_columns["billwerder"]
model_times = pd.DatetimeIndex(billwerder_model["time"].values)
billwerder_at_model = average_to_model_times(billwerder, model_times)

corridor_model = model_columns[MODEL_TARGET_NAME]
if not np.array_equal(corridor_model["time"].values, billwerder_model["time"].values):
    raise RuntimeError("Model times at ship corridor and Billwerder differ - delete the cache in CACHE_DIR and re-run")

if LIDAR_FLIP_CORRECT:
    flip_mask, flip_runs = detect_direction_flips(lidar_raw, billwerder)
    lidar = apply_direction_flip(lidar_raw, flip_mask)
    n_corr = int(flip_mask.sum())
    print(f"\nLIDAR 180 deg direction-flip correction "
          f"(reference: meteomast @{LIDAR_FLIP_REF_MAST_HEIGHT} m, tol {LIDAR_FLIP_TOL:g} deg, "
          f"min speed {LIDAR_FLIP_MIN_SPEED:g} m/s, max duration {LIDAR_FLIP_MAX_DURATION})")
    if flip_runs.empty:
        print("  no flipped stretches found")
    else:
        show = flip_runs.copy()
        show["start"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
        show["end"] = show["end"].dt.strftime("%m-%d %H:%M")
        show["duration"] = (show["duration"] / pd.Timedelta("1h")).round(2)
        print(show.to_string(index=False,
                             columns=["start", "end", "duration", "samples", "flagged",
                                      "mean_offset", "mean_speed", "corrected"],
                             float_format=lambda v: f"{v:.1f}"))
        print(f"  corrected {n_corr} of {len(lidar_raw)} samples "
              f"({100 * n_corr / len(lidar_raw):.1f} %) in {int(flip_runs['corrected'].sum())} stretches")
        if not flip_runs["corrected"].all():
            skipped = flip_runs.loc[~flip_runs["corrected"], "samples"].sum()
            print(f"  LEFT UNCORRECTED: {int((~flip_runs['corrected']).sum())} stretches "
                  f"({int(skipped)} samples) longer than LIDAR_FLIP_MAX_DURATION")
else:
    lidar = lidar_raw
    flip_mask, flip_runs = np.zeros(len(lidar_raw), dtype=bool), pd.DataFrame()

lidar_variants = {label: average_to_model_times(lidar, model_times, window=win)
                  for label, win in LIDAR_AVERAGES.items()}
lidar_at_model = lidar_variants[PRIMARY_AVG]
# continuous LIDAR series at the PRIMARY_AVG averaging, for the time series figures
lidar_ts = resample_measurements(lidar, LIDAR_HEIGHTS, window=LIDAR_AVERAGES[PRIMARY_AVG] or "10min")
for label, df_var in lidar_variants.items():
    n_ok = int(np.isfinite(df_var[[f"wind_speed_{h}" for h in LIDAR_HEIGHTS]].values).any(axis=1).sum())
    logging.info("LIDAR averaging %-5s: %d of %d model times covered", label, n_ok, len(model_times))

if LIDAR_FLIP_CORRECT and not flip_runs.empty:
    fig, ax = plt.subplots(figsize=(16, 5))
    tt = lidar_raw.index.tz_convert(None)
    ref10 = resample_measurements(billwerder, BILLWERDER_HEIGHTS, window="10min").reindex(lidar_raw.index)
    ax.plot(tt, ref10[f"wind_dir_{LIDAR_FLIP_REF_MAST_HEIGHT}"], ".", ms=2, color="tab:green",
            alpha=0.5, label=f"meteomast @{LIDAR_FLIP_REF_MAST_HEIGHT} m (reference)")
    ax.plot(tt, lidar_column_direction(lidar_raw), ".", ms=3, color="lightgrey", label="LIDAR column, raw")
    ax.plot(tt, lidar_column_direction(lidar), ".", ms=3, color="tab:blue", label="LIDAR column, corrected")
    for _, r in flip_runs.iterrows():
        ax.axvspan(r["start"].tz_convert(None), r["end"].tz_convert(None),
                   color="tab:orange" if r["corrected"] else "tab:purple", alpha=0.18)
    ax.set_ylim(0, 360)
    ax.set_ylabel("wind direction / °")
    ax.set_xlabel("time (UTC)")
    ax.set_title(f"LIDAR 180° direction flips: {int(flip_mask.sum())} of {len(lidar_raw)} samples "
                 f"corrected in {int(flip_runs['corrected'].sum())} stretches (shaded)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    finish_figure(fig, "lidar_direction_flip_correction")

# two independent masks on the same model hours: the meteomast judges the model column above
# Billwerder, the LIDAR judges the model column above the ship corridor
mask_mast, eval_mast, mast_available = build_agreement_mask(
    billwerder_model, billwerder_at_model, MASK_HEIGHTS, source="Meteomast")
mask_lidar, eval_lidar, lidar_mask_available = build_agreement_mask(
    corridor_model, lidar_at_model, MASK_HEIGHTS_LIDAR, source="LIDAR")

MASKS = {"meteomast": (mask_mast, eval_mast, mast_available, MASK_HEIGHTS, "meteomast"),
         "lidar": (mask_lidar, eval_lidar, lidar_mask_available, MASK_HEIGHTS_LIDAR, "LIDAR")}
mask, mask_evaluable, mask_available, mask_heights_used, mask_source_name = MASKS[MASK_SOURCE]
mask_label = (f"masked by {mask_source_name} (|Δu|,|Δv| < {MASK_TOL:g} m/s at {mask_heights_used} m)"
              if mask_available else f"unmasked (no {mask_source_name} data in the model period)")

# ============================================================================= 2. diagnostic: mast vs model at the mask heights
#%%
if mask_available:
    fig, axes = plt.subplots(len(BILLWERDER_HEIGHTS), 4, figsize=(18, 4 * len(BILLWERDER_HEIGHTS)))
    axes = np.atleast_2d(axes)
    for row, h in enumerate(BILLWERDER_HEIGHTS):
        mdl = billwerder_model.sel(z=h)
        for col, (meas_col, model_var, label, circular) in enumerate([
                (f"wind_speed_{h}", "wspd", "wind speed / m/s", False),
                (f"wind_dir_{h}", "wdir", "wind dir / °", True),
                (f"u_wind_{h}", "u", "u / m/s", False),
                (f"v_wind_{h}", "v", "v / m/s", False)]):
            x = billwerder_at_model[meas_col].values
            y = mdl[model_var].values
            sel = np.ones_like(mask) if not circular else (
                (billwerder_at_model[f"wind_speed_{h}"].values > MIN_SPEED_DIR) & (mdl["wspd"].values > MIN_SPEED_DIR))
            ax = axes[row, col]
            n_avail = int((np.isfinite(x) & np.isfinite(y) & sel).sum())
            drop = ~mask & sel & np.isfinite(x) & np.isfinite(y)
            if drop.any():
                ax.scatter(x[drop], y[drop], s=10, color="lightgrey", label=f"discarded by mask (n={int(drop.sum())})")
            scatter_with_fit(ax, x[mask & sel], y[mask & sel], f"Meteomast {label}", f"Model {label}",
                             n_available=n_avail,
                             title=f"Billwerder @{h} m", circular=circular)
    fig.suptitle(f"Meteomast Billwerder vs model column above the mast — {mask_label}")
    fig.tight_layout()
    finish_figure(fig, "billwerder_vs_model")

    # one combined overview at a single height: scatter (top) and time series (bottom)
    h = BILLWERDER_FOCUS_HEIGHT
    mdl = billwerder_model.sel(z=h)
    panels = [(f"wind_speed_{h}", "wspd", "wind speed / m/s", False),
              (f"wind_dir_{h}", "wdir", "wind dir / °", True),
              (f"u_wind_{h}", "u", "u / m/s", False),
              (f"v_wind_{h}", "v", "v / m/s", False)]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for col, (meas_col, model_var, label, circular) in enumerate(panels):
        x = billwerder_at_model[meas_col].values
        y = mdl[model_var].values
        sel = ((billwerder_at_model[f"wind_speed_{h}"].values > MIN_SPEED_DIR)
               & (mdl["wspd"].values > MIN_SPEED_DIR)) if circular else np.ones_like(mask)
        finite = np.isfinite(x) & np.isfinite(y)

        ax = axes[0, col]
        drop = ~mask & sel & finite
        if drop.any():
            ax.scatter(x[drop], y[drop], s=10, color="lightgrey", label=f"discarded by mask (n={int(drop.sum())})")
        scatter_with_fit(ax, x[mask & sel], y[mask & sel], f"Meteomast {label}", f"Model {label}",
                         n_available=int((finite & sel).sum()), title=f"Billwerder @{h} m", circular=circular)

        ax = axes[1, col]
        ax.plot(model_times, x, ".", ms=3, color="tab:blue", alpha=0.8, label=f"meteomast @{h} m")
        ax.plot(model_times, y, ".", ms=3, color="tab:red", label="model")
        ax.plot(model_times[~mask], y[~mask], ".", ms=3, color="lightgrey", label="model (discarded)")
        if circular:
            ax.set_ylim(0, 360)
        ax.set_ylabel(label)
        ax.set_xlabel("time (UTC)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle(f"Meteomast Billwerder vs model above the mast @{h} m — {mask_label}")
    fig.tight_layout()
    finish_figure(fig, f"billwerder_vs_model_overview_{BILLWERDER_FOCUS_HEIGHT:03d}m")

    # the same height as time series only: wind speed (top) and direction (bottom)
    # meteomast as 10-min averages (small dots) and as the AVG_WINDOW means that enter the mask (large dots)
    mast_10 = resample_measurements(billwerder, [h], window="10min")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, meas_col, model_var, label in ((ax1, f"wind_speed_{h}", "wspd", "wind speed / m/s"),
                                           (ax2, f"wind_dir_{h}", "wdir", "wind dir / °")):
        y = mdl[model_var].values
        ax.plot(mast_10.index.tz_convert(None), mast_10[meas_col].values, ".", ms=1.5,
                color="tab:blue", alpha=0.5, label=f"meteomast 10-min @{h} m")
        #ax.plot(model_times, billwerder_at_model[meas_col].values, ".", ms=4, color="tab:blue",
        #        label=f"meteomast {AVG_WINDOW} mean @{h} m")
        ax.plot(model_times[mask], y[mask], ".", ms=4, color="tab:red", label="model (kept)")
        ax.plot(model_times[~mask], y[~mask], ".", ms=4, color="lightgrey", label="model (discarded by mask)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 360)
    ax2.set_xlim(model_times[0], model_times[-1])     # the mast record reaches beyond the model period
    ax2.set_xlabel("time (UTC)")
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    #ax1.set_title(f"Meteomast Billwerder vs model above the mast @{h} m — {mask_label}")
    fig.tight_layout()
    finish_figure(fig, f"billwerder_vs_model_timeseries_{BILLWERDER_FOCUS_HEIGHT:03d}m")

# ============================================================================= 3. do the two masks agree?
#%%
corridor_model_masked = corridor_model.isel(time=mask)
lidar_at_model_masked = lidar_at_model[mask]

both_eval = eval_mast & eval_lidar          # hours where both masks carry information
n_both = int(both_eval.sum())
if n_both == 0:
    raise RuntimeError("Meteomast and LIDAR never cover the same model hour - cannot compare the masks")

cat_names = ["kept by both", "discarded by both", "kept by meteomast only", "kept by LIDAR only"]
cat_counts = {
    cat_names[0]: int((mask_mast & mask_lidar & both_eval).sum()),
    cat_names[1]: int((~mask_mast & ~mask_lidar & both_eval).sum()),
    cat_names[2]: int((mask_mast & ~mask_lidar & both_eval).sum()),
    cat_names[3]: int((~mask_mast & mask_lidar & both_eval).sum()),
}
n_same = cat_counts[cat_names[0]] + cat_counts[cat_names[1]]
n_diff = cat_counts[cat_names[2]] + cat_counts[cat_names[3]]
agree_frac = n_same / n_both
print(f"\nMask agreement over {n_both} model hours that meteomast ({MASK_HEIGHTS} m) and "
      f"LIDAR ({MASK_HEIGHTS_LIDAR} m) can both judge:")
for name in cat_names:
    print(f"  {name:<24s} {cat_counts[name]:5d}   {100 * cat_counts[name] / n_both:5.1f} %")
print(f"  {'-' * 24} {'-' * 5}   {'-' * 7}")
print(f"  {'same decision':<24s} {n_same:5d}   {100 * agree_frac:5.1f} %")
print(f"  {'different decision':<24s} {n_diff:5d}   {100 * (1 - agree_frac):5.1f} %")

# confusion matrix + timeline of the four categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
conf = np.array([[cat_counts[cat_names[0]], cat_counts[cat_names[3]]],
                 [cat_counts[cat_names[2]], cat_counts[cat_names[1]]]])
ax1.imshow(conf / n_both, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{conf[i, j]}\n{100 * conf[i, j] / n_both:.1f} %", ha="center", va="center",
                 color="white" if conf[i, j] / n_both > 0.5 else "black")
ax1.set_xticks([0, 1], ["kept", "discarded"])
ax1.set_yticks([0, 1], ["kept", "discarded"])
ax1.set_xlabel(f"meteomast mask @{MASK_HEIGHTS} m")
ax1.set_ylabel(f"LIDAR mask @{MASK_HEIGHTS_LIDAR} m")
ax1.set_title(f"n = {n_both} hours, same decision in {100 * agree_frac:.1f} %")

cat_of_hour = np.where(mask_mast & mask_lidar, 0,
                       np.where(~mask_mast & ~mask_lidar, 1, np.where(mask_mast, 2, 3)))
for code, (name, color) in enumerate(zip(cat_names, ["tab:green", "lightgrey", "tab:orange", "tab:purple"])):
    pick = both_eval & (cat_of_hour == code)
    ax2.plot(model_times[pick], np.full(int(pick.sum()), code), "o", ms=4, color=color,
             label=f"{name} ({cat_counts[name]})")
ax2.set_yticks(range(4), cat_names)
ax2.set_xlabel("time (UTC)")
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
ax2.set_title("when the two masks disagree")
fig.tight_layout()
finish_figure(fig, "mask_agreement")

# ============================================================================= 4. overlap window of model and LIDAR
#%%
# times where model and LIDAR both have data (mask-independent) -> x-range of the time series plots
has_lidar = np.isfinite(lidar_at_model[[f"wind_speed_{h}" for h in LIDAR_HEIGHTS]].values).any(axis=1)
if not has_lidar.any():
    raise RuntimeError("LIDAR record and model period do not overlap at all")
overlap_times = model_times[has_lidar]
pad = 0.02 * (overlap_times.max() - overlap_times.min())
overlap_xlim = (overlap_times.min() - pad, overlap_times.max() + pad)
logging.info("LIDAR and model overlap: %d hours between %s and %s",
             int(has_lidar.sum()), overlap_times.min(), overlap_times.max())

overlap = mask & has_lidar
logging.info("...of which the mask keeps %d hours between %s and %s",
             int(overlap.sum()), model_times[overlap].min() if overlap.any() else None,
             model_times[overlap].max() if overlap.any() else None)

# ============================================================================= 5. per-height comparison LIDAR vs model
#%%
stats_rows = []
for avg_label, lidar_df in lidar_variants.items():
    make_figures = PLOT_ALL_AVERAGES or avg_label == PRIMARY_AVG
    for h in LIDAR_HEIGHTS:
        mdl = corridor_model.sel(z=h)
        meas_speed = lidar_df[f"wind_speed_{h}"].values
        dir_ok = (meas_speed > MIN_SPEED_DIR) & (mdl["wspd"].values > MIN_SPEED_DIR)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (meas_col, model_var, label, circular) in zip(axes, [
                (f"wind_speed_{h}", "wspd", "wind speed / m/s", False),
                (f"wind_dir_{h}", "wdir", "wind dir / °", True),
                (f"u_wind_{h}", "u", "u / m/s", False),
                (f"v_wind_{h}", "v", "v / m/s", False)]):
            x = lidar_df[meas_col].values
            y = mdl[model_var].values
            sel = dir_ok if circular else np.ones_like(mask)
            st_all = orthogonal_fit_stats(x[sel], y[sel], circular=circular)   # every usable hour, mask ignored
            st = orthogonal_fit_stats(x[mask & sel], y[mask & sel], circular=circular)
            stats_rows.append(dict(height=h, variable=model_var, averaging=avg_label,
                                   n_discarded=st_all["n"] - st["n"],
                                   **{f"{k}_masked": v for k, v in st.items()},
                                   **{f"{k}_all": v for k, v in st_all.items()}))
            if not make_figures:
                continue
            drop = ~mask & sel & np.isfinite(x) & np.isfinite(y)
            if drop.any():
                ax.scatter(x[drop], y[drop], s=10, color="lightgrey", label=f"discarded by mask (n={int(drop.sum())})")
            scatter_with_fit(ax, x[mask & sel], y[mask & sel], f"LIDAR {label}", f"Model {label}",
                             title=f"@{h} m", circular=circular, n_available=st_all["n"])
        if not make_figures:
            plt.close(fig)
            continue
        fig.suptitle(f"LIDAR ({avg_label} average) vs model column above {MODEL_TARGET_NAME} @{h} m — {mask_label}")
        fig.tight_layout()
        finish_figure(fig, f"lidar_vs_model_scatter_{avg_label}_{h:03d}m")

        # time series over the overlap window: LIDAR at PRIMARY_AVG as dots, model hours as markers
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        lidar_times = lidar_ts.index.tz_convert(None)
        ax1.plot(lidar_times, lidar_ts[f"wind_speed_{h}"], ".", ms=2, color="tab:blue", alpha=0.7, label=f"LIDAR {PRIMARY_AVG}")
        ax1.plot(model_times[mask], mdl["wspd"].values[mask], "o", ms=3, color="tab:red", label="model (kept)")
        ax1.plot(model_times[~mask], mdl["wspd"].values[~mask], "o", ms=3, color="lightgrey", label="model (discarded by mask)")
        ax1.set_ylabel("wind speed / m/s")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(alpha=0.3)
        ax2.plot(lidar_times, lidar_ts[f"wind_dir_{h}"], ".", ms=2, color="tab:blue", alpha=0.7, label=f"LIDAR {PRIMARY_AVG}")
        ax2.plot(model_times[mask], mdl["wdir"].values[mask], "o", ms=3, color="tab:red", label="model (kept)")
        ax2.plot(model_times[~mask], mdl["wdir"].values[~mask], "o", ms=3, color="lightgrey", label="model (discarded by mask)")
        ax2.set_ylabel("wind dir / °")
        ax2.set_ylim(0, 360)
        ax2.set_xlabel("time (UTC)")
        ax2.grid(alpha=0.3)
        ax2.set_xlim(*overlap_xlim)          # sharex -> both panels
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
        avail_h = np.isfinite(lidar_df[f"wind_speed_{h}"].values) & np.isfinite(mdl["wspd"].values)
        ax1.set_title(f"LIDAR vs model above {MODEL_TARGET_NAME} @{h} m ({avg_label} average) — "
                      f"kept {int((mask & avail_h).sum())}, discarded {int((~mask & avail_h).sum())} model hours")
        fig.tight_layout()
        finish_figure(fig, f"lidar_vs_model_timeseries_{avg_label}_{h:03d}m")

stats = pd.DataFrame(stats_rows).set_index(["averaging", "variable", "height"]).sort_index()
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
show = ["n_masked", "n_discarded", "r_masked", "rmse_masked", "bias_masked", "slope_masked",
        "n_all", "r_all", "rmse_all", "bias_all", "slope_all"]
for avg_label in LIDAR_AVERAGES:
    print(f"\nLIDAR ({avg_label} average) vs model above {MODEL_TARGET_NAME} — {mask_label}")
    print(stats.loc[avg_label, show].round(2))
if SAVE_FIG_DIR is not None:
    stats.to_csv(SAVE_FIG_DIR / "lidar_vs_model_stats.csv")

# ============================================================================= 6. summary profiles of the statistics
#%%
# line style distinguishes the LIDAR averaging, colour the variable
avg_styles = dict(zip(LIDAR_AVERAGES, ["o-", "s--", "^:"]))
# (column key, axis label, carries the unit of the variable, reference value or None)
STAT_COLUMNS = [("r", "correlation r", False, None),
                ("rmse", "RMSE", True, None),
                ("bias", "bias (model − LIDAR)", True, 0.0),
                ("slope", "orth. fit slope", False, 1.0),
                ("intercept", "orth. fit offset", True, 0.0)]
fig, axes = plt.subplots(2, len(STAT_COLUMNS), figsize=(4.6 * len(STAT_COLUMNS), 9), sharey=True)
for row, (variables, unit) in enumerate([((("wspd", "tab:blue"), ("u", "tab:green"), ("v", "tab:red")), "m/s"),
                                         ((("wdir", "tab:orange"),), "°")]):
    masked_values = {stat: [] for stat, *_ in STAT_COLUMNS}
    for var, color in variables:
        for avg_label, style in avg_styles.items():
            sv = stats.loc[(avg_label, var)]
            for col, (stat, _, _, _) in enumerate(STAT_COLUMNS):
                axes[row, col].plot(sv[f"{stat}_masked"], sv.index, style, color=color, ms=4,
                                    label=f"{var} {avg_label}")
                masked_values[stat].append(sv[f"{stat}_masked"].values)
                if mask_available and avg_label == PRIMARY_AVG:
                    axes[row, col].plot(sv[f"{stat}_all"], sv.index, "o-", color=color, alpha=0.25, ms=3,
                                        label=f"{var} {avg_label} all hours")
    for col, (stat, label, has_unit, ref) in enumerate(STAT_COLUMNS):
        ax = axes[row, col]
        ax.set_xlabel(f"{label} / {unit}" if has_unit else label)
        if ref is not None:
            ax.axvline(ref, color="k", linewidth=0.8)
        if stat in ("slope", "intercept"):
            # the unmasked fits run away wherever r is near zero (an orthogonal fit is unstable
            # on an isotropic cloud), so scale these axes on the masked fits only
            v = np.concatenate(masked_values[stat])
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = min(v.min(), ref), max(v.max(), ref)
                pad = 0.15 * (hi - lo) if hi > lo else max(0.5, abs(lo) * 0.1)
                ax.set_xlim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[row, 0].set_ylabel("height / m")
fig.suptitle(f"LIDAR vs model above {MODEL_TARGET_NAME}: statistics per height — {mask_label}")
fig.tight_layout()
finish_figure(fig, "lidar_vs_model_stats_profile")

# ============================================================================= 7. mean profiles LIDAR vs model (kept hours)
#%%
fig, ax = plt.subplots(figsize=(5, 6))
n_kept = {}
for avg_label, style in avg_styles.items():
    lidar_df = lidar_variants[avg_label]
    kept = mask & np.isfinite(lidar_df[f"wind_speed_{LIDAR_HEIGHTS[-1]}"].values)
    n_kept[avg_label] = int(kept.sum())
    lidar_prof = np.array([np.nanmean(lidar_df[f"wind_speed_{h}"].values[kept]) for h in LIDAR_HEIGHTS])
    ax.plot(lidar_prof, LIDAR_HEIGHTS, style, color="tab:blue", label=f"LIDAR ({avg_label})")
kept = mask & np.isfinite(lidar_at_model[f"wind_speed_{LIDAR_HEIGHTS[-1]}"].values)
model_prof = np.nanmean(corridor_model["wspd"].sel(z=LIDAR_HEIGHTS).values[kept], axis=0)
ax.plot(model_prof, LIDAR_HEIGHTS, "s-", color="tab:red", label=f"model above {MODEL_TARGET_NAME}")
ax.set_xlabel("mean wind speed / m/s")
ax.set_ylabel("height / m")
ax.set_title("Mean profile over kept hours" + "\n" + ", ".join(f"{k}: {v}" for k, v in n_kept.items()))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
finish_figure(fig, "lidar_vs_model_mean_profile")

# ============================================================================= 8. LIDAR vs meteomast, MEAS_COMPARE_WINDOW averages
#%%
# Direct instrument-to-instrument comparison, independent of the model: both instruments are
# averaged into the same windows (centre-labelled) and matched on that common grid.
mast_res = resample_measurements(billwerder, BILLWERDER_HEIGHTS, window=MEAS_COMPARE_WINDOW)
lidar_res = resample_measurements(lidar, LIDAR_HEIGHTS, window=MEAS_COMPARE_WINDOW)
common_times = mast_res.index.intersection(lidar_res.index)
mast_res = mast_res.loc[common_times]
lidar_res = lidar_res.loc[common_times]
logging.info("LIDAR vs meteomast: %d common %s intervals between %s and %s",
             len(common_times), MEAS_COMPARE_WINDOW, common_times.min(), common_times.max())

meas_stats_rows = []
for h_mast, h_lidar in MAST_LIDAR_PAIRS:
    speed_ok = ((mast_res[f"wind_speed_{h_mast}"].values > MIN_SPEED_DIR)
                & (lidar_res[f"wind_speed_{h_lidar}"].values > MIN_SPEED_DIR))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (col_tpl, label, circular) in zip(axes, [
            ("wind_speed_{h}", "wind speed / m/s", False),
            ("wind_dir_{h}", "wind dir / °", True),
            ("u_wind_{h}", "u / m/s", False),
            ("v_wind_{h}", "v / m/s", False)]):
        x = lidar_res[col_tpl.format(h=h_lidar)].values
        y = mast_res[col_tpl.format(h=h_mast)].values
        sel = speed_ok if circular else np.ones(len(common_times), dtype=bool)
        st = scatter_with_fit(ax, x[sel], y[sel], f"LIDAR Wedel {label}", f"Meteomast Billwerder {label}",
                              title=f"LIDAR @{h_lidar} m vs meteomast @{h_mast} m", circular=circular)
        meas_stats_rows.append(dict(mast_height=h_mast, lidar_height=h_lidar,
                                    variable=col_tpl.format(h="").strip("_"), **st))
    fig.suptitle(f"LIDAR (Wedel) vs meteomast (Billwerder), {MEAS_COMPARE_WINDOW} averages, "
                 f"{len(common_times)} common intervals")
    fig.tight_layout()
    finish_figure(fig, f"lidar_vs_meteomast_{MEAS_COMPARE_WINDOW}_scatter_{h_mast:03d}m")

    # time series of the same pair
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    t = common_times.tz_convert(None)
    ax1.plot(t, lidar_res[f"wind_speed_{h_lidar}"], ".", ms=3, color="tab:blue",
             label=f"LIDAR @{h_lidar} m")
    ax1.plot(t, mast_res[f"wind_speed_{h_mast}"], ".", ms=3, color="tab:green",
             label=f"meteomast @{h_mast} m")
    ax1.set_ylabel("wind speed / m/s")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    ax2.plot(t, lidar_res[f"wind_dir_{h_lidar}"], ".", ms=3, color="tab:blue", label=f"LIDAR @{h_lidar} m")
    ax2.plot(t, mast_res[f"wind_dir_{h_mast}"], ".", ms=3, color="tab:green", label=f"meteomast @{h_mast} m")
    ax2.set_ylabel("wind dir / °")
    ax2.set_ylim(0, 360)
    ax2.set_xlabel("time (UTC)")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax1.set_title(f"LIDAR @{h_lidar} m (Wedel) vs meteomast @{h_mast} m (Billwerder), "
                  f"{MEAS_COMPARE_WINDOW} averages")
    fig.tight_layout()
    finish_figure(fig, f"lidar_vs_meteomast_{MEAS_COMPARE_WINDOW}_timeseries_{h_mast:03d}m")

meas_stats = pd.DataFrame(meas_stats_rows).set_index(["variable", "mast_height"]).sort_index()
print(f"\nLIDAR (Wedel) vs meteomast (Billwerder), {MEAS_COMPARE_WINDOW} averages "
      f"- {len(common_times)} common intervals, no model mask applied")
print(meas_stats[["lidar_height", "n", "r", "rmse", "bias", "slope", "intercept"]].round(2))
if SAVE_FIG_DIR is not None:
    meas_stats.to_csv(SAVE_FIG_DIR / f"lidar_vs_meteomast_{MEAS_COMPARE_WINDOW}_stats.csv")

# statistics vs height: one row for the speeds (m/s), one for the direction (deg)
# (column key, axis label, carries the unit of the variable, reference value or None)
MEAS_STAT_COLUMNS = [("r", "correlation r", False, None),
                     ("rmse", "RMSE", True, None),
                     ("bias", "bias (meteomast − LIDAR)", True, 0.0),
                     ("slope", "orth. fit slope", False, 1.0),
                     ("intercept", "orth. fit offset", True, 0.0)]
fig, axes = plt.subplots(2, len(MEAS_STAT_COLUMNS), figsize=(4.6 * len(MEAS_STAT_COLUMNS), 9), sharey=True)
for row, (variables, unit) in enumerate([((("wind_speed", "tab:blue"), ("u_wind", "tab:green"),
                                           ("v_wind", "tab:red")), "m/s"),
                                         ((("wind_dir", "tab:orange"),), "°")]):
    for var, color in variables:
        sv = meas_stats.loc[var]
        for col, (stat, _, _, _) in enumerate(MEAS_STAT_COLUMNS):
            axes[row, col].plot(sv[stat], sv.index, "o-", color=color, label=var)
    for col, (stat, label, has_unit, ref) in enumerate(MEAS_STAT_COLUMNS):
        ax = axes[row, col]
        ax.set_xlabel(f"{label} / {unit}" if has_unit else label)
        if ref is not None:
            ax.axvline(ref, color="k", linewidth=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[row, 0].set_ylabel("meteomast height / m")
fig.suptitle(f"LIDAR vs meteomast, {MEAS_COMPARE_WINDOW} averages: statistics per height")
fig.tight_layout()
finish_figure(fig, f"lidar_vs_meteomast_{MEAS_COMPARE_WINDOW}_stats_profile")

# ============================================================================= 9. the model at its two points
#%%
# How different is the model itself between the meteomast and the ship corridor?  This sets the
# floor for transferring the meteomast-based mask to the corridor: whatever the model does
# differently between the two points cannot be judged by the mast.
mperlat, mperlon = 111132.0, 111320.0 * np.cos(np.deg2rad(BILLWERDER_LAT))
point_distance = np.hypot((BILLWERDER_LAT - target_lat) * mperlat,
                          (BILLWERDER_LON - target_lon) * mperlon) / 1000.0
print(f"\nModel at Billwerder vs model above {MODEL_TARGET_NAME} "
      f"({point_distance:.1f} km apart, {billwerder_model.sizes['time']} hours)")

point_stats_rows = []
for h in MODEL_POINT_HEIGHTS:
    a = billwerder_model.sel(z=h)     # model above the meteomast
    b = corridor_model.sel(z=h)       # model above the ship corridor
    dir_ok = (a["wspd"].values > MIN_SPEED_DIR) & (b["wspd"].values > MIN_SPEED_DIR)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (var, label, circular) in zip(axes, [("wspd", "wind speed / m/s", False),
                                                 ("wdir", "wind dir / °", True),
                                                 ("u", "u / m/s", False),
                                                 ("v", "v / m/s", False)]):
        x, y = a[var].values, b[var].values
        sel = dir_ok if circular else np.ones(len(x), dtype=bool)
        st = scatter_with_fit(ax, x[sel], y[sel], f"Model @Billwerder {label}",
                              f"Model @{MODEL_TARGET_NAME} {label}",
                              title=f"@{h} m", circular=circular)
        point_stats_rows.append(dict(height=h, variable=var, **st))
    fig.suptitle(f"Model at its two points in the domain @{h} m "
                 f"(Billwerder vs {MODEL_TARGET_NAME}, {point_distance:.1f} km apart)")
    fig.tight_layout()
    finish_figure(fig, f"model_point_to_point_scatter_{h:03d}m")

point_stats = pd.DataFrame(point_stats_rows).set_index(["variable", "height"]).sort_index()
print(point_stats[["n", "r", "rmse", "bias", "slope", "intercept"]].round(2))
if SAVE_FIG_DIR is not None:
    point_stats.to_csv(SAVE_FIG_DIR / "model_point_to_point_stats.csv")

# time series of both points at the mask height
h_ref = MASK_HEIGHTS[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
ax1.plot(model_times, billwerder_model["wspd"].sel(z=h_ref), ".", ms=3, color="tab:purple",
         label="model @Billwerder")
ax1.plot(model_times, corridor_model["wspd"].sel(z=h_ref), ".", ms=3, color="tab:red",
         label=f"model @{MODEL_TARGET_NAME}")
ax1.set_ylabel("wind speed / m/s")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(alpha=0.3)
ax2.plot(model_times, billwerder_model["wdir"].sel(z=h_ref), ".", ms=3, color="tab:purple",
         label="model @Billwerder")
ax2.plot(model_times, corridor_model["wdir"].sel(z=h_ref), ".", ms=3, color="tab:red",
         label=f"model @{MODEL_TARGET_NAME}")
ax2.set_ylabel("wind dir / °")
ax2.set_ylim(0, 360)
ax2.set_xlabel("time (UTC)")
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
ax1.set_title(f"Model time series at both points @{h_ref} m")
fig.tight_layout()
finish_figure(fig, "model_point_to_point_timeseries")

# how much of the mask tolerance is used up by the model's own point-to-point difference?
du = corridor_model["u"].sel(z=h_ref).values - billwerder_model["u"].sel(z=h_ref).values
dv = corridor_model["v"].sel(z=h_ref).values - billwerder_model["v"].sel(z=h_ref).values
within = np.isfinite(du) & np.isfinite(dv) & (np.abs(du) < MASK_TOL) & (np.abs(dv) < MASK_TOL)
finite = np.isfinite(du) & np.isfinite(dv)
print(f"\nPoint-to-point difference of the model itself @{h_ref} m:")
print(f"  |du| mean {np.nanmean(np.abs(du)):.2f} m/s, p95 {np.nanpercentile(np.abs(du), 95):.2f} m/s")
print(f"  |dv| mean {np.nanmean(np.abs(dv)):.2f} m/s, p95 {np.nanpercentile(np.abs(dv), 95):.2f} m/s")
print(f"  within the mask tolerance of {MASK_TOL:g} m/s in both components: "
      f"{int(within.sum())} of {int(finite.sum())} hours ({100 * within.sum() / max(finite.sum(), 1):.1f} %)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(du[finite], bins=40, alpha=0.6, label=f"$\\Delta$u (mean {np.nanmean(du):+.2f} m/s)")
ax.hist(dv[finite], bins=40, alpha=0.6, label=f"$\\Delta$v (mean {np.nanmean(dv):+.2f} m/s)")
for x in (-MASK_TOL, MASK_TOL):
    ax.axvline(x, color="k", linestyle="--", linewidth=1)
ax.set_xlabel(f"model @{MODEL_TARGET_NAME} − model @Billwerder / m/s")
ax.set_ylabel("hours")
ax.set_title(f"Model point-to-point difference @{h_ref} m (dashed: mask tolerance)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
finish_figure(fig, "model_point_to_point_difference_hist")

# ============================================================================= 10. ERA5 vs WRF vs LIDAR
#%%
# ERA5 (0.25 deg, hourly instantaneous) at the same point as the WRF column compared to the LIDAR.
# Everything is put on the ERA5 hours; the LIDAR uses the PRIMARY_AVG averaging around each hour.
# At 100 m the LIDAR (79/109 m gates) and WRF are interpolated linearly in z.
era5 = load_era5_points({MODEL_TARGET_NAME: (target_lat, target_lon)})[MODEL_TARGET_NAME]
era5_times = pd.DatetimeIndex(era5["time"].values)
coarse_model = load_model_columns({MODEL_TARGET_NAME: (target_lat, target_lon, MODEL_COLUMN_HEIGHTS)},
                                  prefix=MODEL_PREFIX_COARSE)[MODEL_TARGET_NAME]
dom_fine, dom_coarse = MODEL_PREFIX.split("_")[-1], MODEL_PREFIX_COARSE.split("_")[-1]
wrf_fine, wrf_coarse = f"WRF {dom_fine}", f"WRF {dom_coarse}"
lidar_at_era5 = average_to_model_times(lidar, era5_times, window=LIDAR_AVERAGES[PRIMARY_AVG])
era5_mask = pd.Series(mask, index=model_times).reindex(era5_times, fill_value=False).values
era5_mask_label = mask_label if ERA5_USE_MASK else "all hours (WRF mask not applied)"
logging.info("ERA5 at %s: %s .. %s (%d hours)", MODEL_TARGET_NAME, era5_times[0], era5_times[-1], len(era5_times))

ERA5_PAIRS = [("LIDAR", "ERA5"), ("LIDAR", wrf_fine), ("LIDAR", wrf_coarse),
              (wrf_fine, "ERA5"), (wrf_coarse, "ERA5")]            # (x, y) of each scatter row
era5_stats_rows = []
for h in ERA5_HEIGHTS:
    src = {"LIDAR": lidar_at_level(lidar_at_era5, h),
           wrf_fine: column_at_level(corridor_model, h, era5_times),
           wrf_coarse: column_at_level(coarse_model, h, era5_times),
           "ERA5": column_at_level(era5, h, era5_times)}
    # a fair comparison: only hours where all sources have a speed
    common = np.logical_and.reduce([np.isfinite(df["wspd"].values) for df in src.values()])
    if ERA5_USE_MASK:
        common &= era5_mask
    n_common = int(common.sum())
    if n_common == 0:
        logging.warning("ERA5 @%d m: no hours common to LIDAR, WRF %s/%s and ERA5 - skipped", h, dom_fine, dom_coarse)
        continue

    fig, axes = plt.subplots(len(ERA5_PAIRS), 4, figsize=(20, 4.6 * len(ERA5_PAIRS)))
    for row, (nx, ny) in enumerate(ERA5_PAIRS):
        dir_ok = (src[nx]["wspd"].values > MIN_SPEED_DIR) & (src[ny]["wspd"].values > MIN_SPEED_DIR)
        for col, (var, label, circular) in enumerate([("wspd", "wind speed / m/s", False),
                                                      ("wdir", "wind dir / °", True),
                                                      ("u", "u / m/s", False),
                                                      ("v", "v / m/s", False)]):
            sel = common & dir_ok if circular else common
            st = scatter_with_fit(axes[row, col], src[nx][var].values[sel], src[ny][var].values[sel],
                                  f"{nx} {label}", f"{ny} {label}", title=f"{ny} vs {nx} @{h} m",
                                  circular=circular)
            era5_stats_rows.append(dict(height=h, pair=f"{ny} vs {nx}", variable=var, **st))
    fig.suptitle(f"ERA5 vs WRF ({dom_fine}, {dom_coarse}) vs LIDAR above {MODEL_TARGET_NAME} @{h} m — "
                 f"{n_common} common hours, {era5_mask_label}")
    fig.tight_layout()
    finish_figure(fig, f"era5_wrf_lidar_scatter_{h:03d}m")

    # time series: LIDAR at PRIMARY_AVG as dots, hourly WRF and ERA5 as lines
    lidar_raw_h = lidar_at_level(lidar_ts, h)
    # meteomast (Billwerder, ~30 km east) at its level closest to h, as 10-min averages like the LIDAR
    h_mast = min(BILLWERDER_HEIGHTS, key=lambda hm: abs(hm - h))
    mast_10 = resample_measurements(billwerder, [h_mast], window="10min")
    mast_10 = mast_10[(mast_10.index.tz_convert(None) >= era5_times[0])
                      & (mast_10.index.tz_convert(None) <= era5_times[-1])]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, var, mast_col in ((ax1, "wspd", f"wind_speed_{h_mast}"), (ax2, "wdir", f"wind_dir_{h_mast}")):
        ax.plot(lidar_raw_h.index.tz_convert(None), lidar_raw_h[var], ".", ms=2, color="tab:blue",
                alpha=0.6, label=f"LIDAR {PRIMARY_AVG}")
        ax.plot(mast_10.index.tz_convert(None), mast_10[mast_col], ".", ms=2, color="tab:green",
                alpha=0.6, label=f"meteomast 10-min @{h_mast} m")
        style = "." if var == "wdir" else "-"
        ax.plot(era5_times, src[wrf_fine][var], style, ms=3, lw=1, color="tab:red",
                label=f"{wrf_fine}")
        ax.plot(era5_times, src[wrf_coarse][var], style, ms=3, lw=1, color="tab:orange",
                label=f"{wrf_coarse}")
        ax.plot(era5_times, src["ERA5"][var], style, ms=3, lw=1, color="k", label="ERA5")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("wind speed / m/s")
    ax2.set_ylabel("wind dir / °")
    ax2.set_ylim(0, 360)
    ax2.set_xlabel("date")
    ax2.set_xlim(era5_times[0], era5_times[-1])
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20))
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax1.set_title(f"LIDAR, WRF {dom_fine}/{dom_coarse} and ERA5 @{h} m, meteomast Billwerder @{h_mast} m")
    fig.tight_layout()
    finish_figure(fig, f"era5_wrf_lidar_timeseries_{h:03d}m")

if era5_stats_rows:
    era5_stats = pd.DataFrame(era5_stats_rows).set_index(["pair", "variable", "height"]).sort_index()
    print(f"\nERA5 vs WRF {dom_fine}/{dom_coarse} vs LIDAR above {MODEL_TARGET_NAME} (bias = y - x, pair = 'y vs x') — {era5_mask_label}")
    print(era5_stats[["n", "r", "rmse", "bias", "slope", "intercept"]].round(2))
    if SAVE_FIG_DIR is not None:
        era5_stats.to_csv(SAVE_FIG_DIR / "era5_wrf_lidar_stats.csv")

# ============================================================================= 11. ERA5 masked by the meteomast
#%%
# Same recipe as the WRF mask: ERA5 above the meteomast vs the mast (AVG_WINDOW means centred on
# each hour), keep the hours with |du| and |dv| < ERA5_MASK_TOL.  The mask is then applied to ERA5 vs
# LIDAR.  The WRF mask is shown next to it as a reference for how much of any improvement is
# just the selection of "easy" weather rather than ERA5 matching the mast.
era5_billwerder = load_era5_points({"billwerder": (BILLWERDER_LAT, BILLWERDER_LON)})["billwerder"]
if not np.array_equal(era5_billwerder["time"].values, era5["time"].values):
    raise RuntimeError("ERA5 times at Billwerder and at the LIDAR differ - delete the ERA5 caches in CACHE_DIR")
mast_at_era5 = average_to_model_times(billwerder, era5_times)
mask_era5, eval_era5, era5_mask_available = build_agreement_mask(
    era5_billwerder, mast_at_era5, [hm for hm, _ in ERA5_MASK_HEIGHT_PAIRS],
    model_heights=[he for _, he in ERA5_MASK_HEIGHT_PAIRS], tol=ERA5_MASK_TOL, source="ERA5 meteomast")
era5_mask_desc = ", ".join(f"mast {hm} m / ERA5 {he} m" for hm, he in ERA5_MASK_HEIGHT_PAIRS)
wrf_mask_at_era5 = era5_mask          # WRF mask (MASK_SOURCE) on the ERA5 hours, from section 10

# how do the two masks relate on the hours both can judge?
wrf_eval_at_era5 = pd.Series(mask_evaluable, index=model_times).reindex(era5_times, fill_value=False).values
both = eval_era5 & wrf_eval_at_era5
print(f"\nERA5 mask ({era5_mask_desc}, tol {ERA5_MASK_TOL:g} m/s) vs WRF mask (tol {MASK_TOL:g} m/s) on {int(both.sum())} hours both can judge:")
for name, sel in [("kept by both", mask_era5 & wrf_mask_at_era5), ("kept by ERA5 mask only", mask_era5 & ~wrf_mask_at_era5),
                  ("kept by WRF mask only", ~mask_era5 & wrf_mask_at_era5), ("discarded by both", ~mask_era5 & ~wrf_mask_at_era5)]:
    n = int((sel & both).sum())
    print(f"  {name:<24s} {n:5d}   {100 * n / max(int(both.sum()), 1):5.1f} %")

# both masks are False wherever they cannot judge an hour, so "no mask" is every hour with LIDAR and ERA5
MASK_VARIANTS = {"no mask": np.ones(len(era5_times), dtype=bool),
                 "ERA5 mask": mask_era5,
                 "WRF mask": wrf_mask_at_era5}
ERA5_PANELS = [("wspd", "wind speed / m/s", False), ("wdir", "wind dir / °", True),
               ("u", "u / m/s", False), ("v", "v / m/s", False)]
era5_masked_rows = []
for h in ERA5_HEIGHTS:
    lid = lidar_at_level(lidar_at_era5, h)
    e5 = column_at_level(era5, h, era5_times)
    base = np.isfinite(lid["wspd"].values) & np.isfinite(e5["wspd"].values)
    dir_ok = (lid["wspd"].values > MIN_SPEED_DIR) & (e5["wspd"].values > MIN_SPEED_DIR)

    # one axis range per panel from all points, shared by the unmasked and the masked figure
    panel_lims = {}
    for var, _, circular in ERA5_PANELS:
        if not circular:
            v = np.concatenate([lid[var].values[base], e5[var].values[base]])
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            panel_lims[var] = (lo - pad, hi + pad)

    # without any mask
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (var, label, circular) in zip(axes, ERA5_PANELS):
        x, y = lid[var].values, e5[var].values
        sel = base & dir_ok if circular else base
        scatter_with_fit(ax, x[sel], y[sel], f"LIDAR {label}", f"ERA5 {label}", title=f"@{h} m", circular=circular,
                         lims=panel_lims.get(var))
    fig.suptitle(f"ERA5 vs LIDAR above {MODEL_TARGET_NAME} @{h} m — no mask")
    fig.tight_layout()
    finish_figure(fig, f"era5_unmasked_vs_lidar_scatter_{h:03d}m")

    # with the ERA5 mask
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (var, label, circular) in zip(axes, ERA5_PANELS):
        x, y = lid[var].values, e5[var].values
        sel = base & dir_ok if circular else base
        for mname, m in MASK_VARIANTS.items():
            st = orthogonal_fit_stats(x[sel & m], y[sel & m], circular=circular)
            era5_masked_rows.append(dict(height=h, variable=var, mask=mname, **st))
        drop = sel & ~mask_era5
        if drop.any():
            ax.scatter(x[drop], y[drop], s=10, color="lightgrey", label=f"discarded by ERA5 mask (n={int(drop.sum())})")
        scatter_with_fit(ax, x[sel & mask_era5], y[sel & mask_era5], f"LIDAR {label}", f"ERA5 {label}",
                         title=f"@{h} m", circular=circular, n_available=int(sel.sum()), lims=panel_lims.get(var))
    fig.suptitle(f"ERA5 vs LIDAR above {MODEL_TARGET_NAME} @{h} m — masked by meteomast vs ERA5 "
                 f"(|Δu|,|Δv| < {ERA5_MASK_TOL:g} m/s, {era5_mask_desc})")
    fig.tight_layout()
    finish_figure(fig, f"era5_masked_vs_lidar_scatter_{h:03d}m")

era5_masked_stats = (pd.DataFrame(era5_masked_rows)
                     .pivot_table(index=["variable", "height"], columns="mask",
                                  values=["n", "r", "rmse", "bias"], sort=False)
                     .reindex(columns=list(MASK_VARIANTS), level=1))
print(f"\nERA5 vs LIDAR above {MODEL_TARGET_NAME}: effect of the masks (bias = ERA5 - LIDAR)")
print(era5_masked_stats.round(2).to_string())
if SAVE_FIG_DIR is not None:
    era5_masked_stats.to_csv(SAVE_FIG_DIR / "era5_masked_vs_lidar_stats.csv")

# ----------------------------------------------------------------------------- LIDAR averaging for ERA5
# ERA5 is an instantaneous hourly value of a ~30 km grid box, so a longer LIDAR average may be the
# fairer counterpart.  Compare PRIMARY_AVG with ERA5_LIDAR_AVG_CHECK (both centred on the ERA5
# time stamp) on the same hours, so that only the averaging differs.
ERA5_LIDAR_AVG_CHECK = "1h"
lidar_at_era5_check = average_to_model_times(lidar, era5_times, window=ERA5_LIDAR_AVG_CHECK)
avg_rows = []
for h in ERA5_HEIGHTS:
    e5 = column_at_level(era5, h, era5_times)
    lids = {PRIMARY_AVG: lidar_at_level(lidar_at_era5, h), ERA5_LIDAR_AVG_CHECK: lidar_at_level(lidar_at_era5_check, h)}
    common = np.isfinite(e5["wspd"].values) & np.logical_and.reduce([np.isfinite(l["wspd"].values) for l in lids.values()])
    for var, _, circular in ERA5_PANELS:
        for avg, lid in lids.items():
            sel = common & ((lid["wspd"].values > MIN_SPEED_DIR) & (e5["wspd"].values > MIN_SPEED_DIR) if circular else True)
            for mname in ("no mask", "ERA5 mask"):
                m = sel & MASK_VARIANTS[mname]
                st = orthogonal_fit_stats(lid[var].values[m], e5[var].values[m], circular=circular)
                avg_rows.append(dict(height=h, variable=var, averaging=f"LIDAR {avg}", mask=mname, **st))
era5_avg_stats = (pd.DataFrame(avg_rows)
                  .pivot_table(index=["variable", "height"], columns=["mask", "averaging"],
                               values=["n", "r", "rmse", "bias"], sort=False))
print(f"\nERA5 vs LIDAR above {MODEL_TARGET_NAME}: LIDAR averaged over {PRIMARY_AVG} vs {ERA5_LIDAR_AVG_CHECK}, "
      f"centred on the ERA5 time stamp (same hours for both; bias = ERA5 - LIDAR)")
print(era5_avg_stats.round(2).to_string())
if SAVE_FIG_DIR is not None:
    era5_avg_stats.to_csv(SAVE_FIG_DIR / "era5_vs_lidar_averaging_stats.csv")

# %%
