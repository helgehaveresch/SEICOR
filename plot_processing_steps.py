#%%
"""Publication figure of the processing steps of one ship pass, from the measured NO2 dSCDs to the
plume and ship masks:

  (a) measured NO2 dSCDs (elevation x time) with the reference window(s) and the plume window
  (b) background: dSCD time series at three elevations, the reference means and the background
      (interpolated linearly in time between the upwind and downwind reference, or the constant
      upwind reference if there is no downwind one)
  (c) enhancement (dSCD minus background, before the background correction) with the plume and ship
      search regions of the detection (plumes.sort_plumes) and the detected plume pixels
  (d) final enhancement (after the background correction) with the iterative plume and ship masks
      (plumes.plume_mask_of) and the source region

Everything is read from the plume file written by plume_preprocessor.py. The detection in (c) is
recomputed as in plumes.sort_plumes (median of the initial plume window as background level, std of
the reference images as noise level), because the preprocessor does not store it.

Run the cells in the interactive window: set the pass in the settings cell below, then run all.
"""
import glob
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if platform.system().lower().startswith("win"):
    SCRIPTS_DIR = Path(r"C:\Users\hhave\Documents\Promotion\scripts")
    PLAT_KEY = "windows"
else:
    SCRIPTS_DIR = Path("/raid/home2/hhaveresch/scripts")
    PLAT_KEY = "linux"
sys.path.insert(0, str(SCRIPTS_DIR))
from SEICOR import impact, plumes  # noqa: E402
from SEICOR.plotting import NO2_ENH, O4_ENH  # noqa: E402  labels of the SCD enhancements, as in plotting.py

#%% settings: the ship pass and the files
PASS_KEY = "20250520_063128"   # pass time as in the plume file names, YYYYMMDD_HHMMSS (MMSI 356582000)
# shortlist:
# PASS_KEY = "20260217_155938"  # 17.02.26 15:59, MMSI 356582000, big ship, plume 8 -> 30 deg, white background
# PASS_KEY = "20250520_063128"  # 20.05.25 06:31, MMSI 477928600, biggest ship, plume 8 -> 30 deg, white background
# PASS_KEY = "20250416_054306"  # 16.04.25 05:43, MMSI 228401800, big ship, plume from +50 s, rises fast
# PASS_KEY = "20250531_053923"  # 31.05.25 05:39, MMSI 477321000, clean diagonal 5 -> 30 deg, ship before the pass
# PASS_KEY = "20250830_175439"  # 30.08.25 17:54, MMSI 563049900, big ship, short rising plume
# PASS_KEY = "20250527_070652"  # 27.05.25 07:06, MMSI 352822000, ship and plume side by side at the pass, broad plume
# PASS_KEY = "20260402_153214"  # 02.04.26 15:32, MMSI 219630000, strong ship at the pass, wide plume behind it
# further candidates:
# PASS_KEY = "20260309_162459"  # 09.03.26 16:24, MMSI 538002734, flat at 8 deg for 30 s, then up to 30 deg
# PASS_KEY = "20250707_073235"  # 07.07.25 07:32, MMSI 563050600, flat at 6 deg, then up to 23 deg
# PASS_KEY = "20250903_054603"  # 03.09.25 05:46, MMSI 477148700, compact ship block, vertical plume behind it
# PASS_KEY = "20250923_144738"  # 23.09.25 14:47, MMSI 247452100, plume stays low (4 -> 10 deg)
# plumes rising high (from near the ship to >= 25 deg):
# PASS_KEY = "20250328_134906"  # 28.03.25 13:49, MMSI 220379000, steep compact rise 10 -> 30 deg within 60 s, strong ship
# PASS_KEY = "20260418_151314"  # 18.04.26 15:13, MMSI 563274400, compact band 8 -> 30 deg over 2 min, strong plume
# PASS_KEY = "20250520_065507"  # 20.05.25 06:55, MMSI 563176600, big ship block, broad plume 8 -> 28 deg over 4 min
# PASS_KEY = "20260415_153439"  # 15.04.26 15:34, MMSI 538002961, band 7 -> 30 deg over 2 min, noisier background
# PASS_KEY = "20260529_050910"  # 29.05.26 05:09, MMSI 413065000, short steep rise 8 -> 30 deg, compact
# cruise ships with plumes rising high:
# PASS_KEY = "20250622_182919"  # 22.06.25 18:29, COSTA FAVOLOSA (331 m, MMSI 247311100), clean diagonal 8 -> 30 deg, weak ship
# PASS_KEY = "20250813_163410"  # 13.08.25 16:34, MEIN SCHIFF 4 (331 m, MMSI 229678000), strong plume 10 -> 30 deg, ship clear; background structure above 15 deg
# PASS_KEY = "20250906_160911"  # 06.09.25 16:09, AIDAPERLA (340 m, MMSI 247385300), plume from the ship top 13 -> 27 deg, clear ship
# PASS_KEY = "20250831_162135"  # 31.08.25 16:21, AIDASOL (299 m, MMSI 247302900), steep rise 9 -> 30 deg, clear ship; background slightly negative
# (also 31.05.25 03:32 in the shortlist above is a cruise ship: EMERALD PRINCESS, 325 m)
# cruise ships (AIS type 60-69, >= 150 m, from AIS/shipinfo.txt):
# PASS_KEY = "20250511_183927"  # 11.05.25 18:39, AIDAPRIMA (340 m, MMSI 247353800), plume leaves the ship top at 12 deg and sinks, clean background
# PASS_KEY = "20250424_163154"  # 24.04.25 16:31, AIDABELLA (295 m, MMSI 247229700), ship at the pass, broad long plume 0-12 deg, very clean background
# PASS_KEY = "20250405_173026"  # 05.04.25 17:30, AIDANOVA (370 m, MMSI 247389200), biggest ship block, strongest plume; data gap at +100..+175 s
# PASS_KEY = "20250518_162806"  # 18.05.25 16:28, AIDASOL (299 m, MMSI 247302900), ship at the pass, long plume at 3-10 deg
# PASS_KEY = "20250511_184437"  # 11.05.25 18:44, AMADEA (247 m, MMSI 308445000), small ship signal, compact plume 0-10 deg
# PASS_KEY = "20250501_162854"  # 01.05.25 16:28, AIDALUNA (292 m, MMSI 247255400), strong vertical plume, weak ship

MMSI = None                     # only needed if several ships pass in the same second
# MMSI = "356582000"

SETTINGS_FILE = SCRIPTS_DIR / "SEICOR" / "plume_preprocessor_settings.yaml"
# SETTINGS_FILE = Path("/path/to/other_settings.yaml")  # e.g. a copy whose Output points to a test directory

PLUME_FILE = None               # None: found via the settings (<plume_out_dir>/plumes_<date>/); or a path to a plume file
# PLUME_FILE = "/misc/dodecagon/BREDOM/SEICOR/plumes/plumes_260217/plume_021_t_20260217_155938_mmsi_356582000.nc"
# PLUME_FILE = "/misc/dodecagon/BREDOM/SEICOR/plumes_test/plumes_260217/plume_021_t_20260217_155938_mmsi_356582000.nc"

OUT_FILE = None                 # None: <plots_out_dir>/processing_steps/processing_steps_<PASS_KEY>.pdf (+ .png preview)
# OUT_FILE = False              # only show the figure, do not save it
# OUT_FILE = Path.home() / "figures" / f"processing_steps_{PASS_KEY}.pdf"
# OUT_FILE = Path.home() / "figures" / f"processing_steps_{PASS_KEY}.png"  # png only

ELEVATIONS = (2.7, 10.3, 20.6)  # elevations (deg) of the time series in (b); the nearest rows are used
# ELEVATIONS = (3.5, 12.0, 25.0)
# ELEVATIONS = (1.9, 7.7, 15.5)

CMAP_DSCD = "magma_r"           # colour map of the measured dSCDs: light background, dark plume (lines drawn in LINE_DSCD)
# CMAP_DSCD = "magma"             # dark maps: set LINE_DSCD = "white"
# CMAP_DSCD = "viridis"
# CMAP_DSCD = "cividis"
# CMAP_DSCD = "Blues"
LINE_DSCD = "#222222"           # plume window, default windows and pass time on the dSCD image ("white" for dark maps like viridis)
CMAP_ENH = "RdBu_r"             # colour map of the enhancements
# CMAP_ENH = "bwr"

PLOT_IMAGE = True               # last row, next to the O4 panel: camera image closest to the pass (<img_dir>/<date>_video.zip or folder)
# PLOT_IMAGE = False
IMAGE_OFFSET_S = 0              # image closest to pass time + this many seconds
# IMAGE_OFFSET_S = 30           # e.g. half a minute after the pass
IMAGE_CROP = None               # None: full image; or (left, top, right, bottom) as fractions of the image size
# IMAGE_CROP = (0.0, 0.45, 1.0, 0.85)  # strip with the ship
PLOT_BACKGROUND = False         # panel with the interpolated background as an image (between (b) and the enhancements)
# PLOT_BACKGROUND = True
PLOT_O4 = True                  # last panel: O4 enhancement with the plume and ship masks
# PLOT_O4 = False
PLOT_OLD = True                 # panel next to (b): NO2 enhancement of the ship from the old retrieval (SLANT_old)
# PLOT_OLD = False
OLD_SC_PATH = "IMPACT/SLANT_old/SLANT_{yy}"  # relative to the base path, month folder is added (as IMPACT_SC_path)
OLD_SC_FILE_ENDING = "ID.NO2_VIS_zenith"     # old product (zenith reference)
# OLD_SC_FILE_ENDING = "ID.NO2_VIS_fix_destriped_ref_250907"
OLD_EXTRA_DROP = 1              # viewing directions dropped in addition to those below image_row.min() of the plume file
# the old panel shows the time range of the plume window and labels the rows with the elevations of the plume file
O4_STORED = 1e40                # unit of the O4 columns as stored (fit output; check against the cross section scaling)
O4_E = 1e43                     # display unit of the O4 columns
O4_UNIT = "10$^{43}$ molec.$^2$ cm$^{-5}$"
# O4_UNIT = "a.u."

E = 1e16  # display unit of the columns
INK, MUTED = "#222222", "#666666"  # text and outlines
C_UP, C_DN = "#0072B2", "#D55E00"  # Okabe-Ito (colour-blind safe)
C_SHIP = "#56B4E9"  # ship mask and ship region: Okabe-Ito sky blue, light enough to stay visible on the saturated (dark) blue ship pixels
ROW_COLOURS = ["#0072B2", "#D55E00", "#009E73"]


#%% functions
def find_plume_file(pass_key, settings, mmsi=None):
    base = Path(settings["base_paths"][PLAT_KEY])
    pattern = f"plume_*_t_{pass_key}_mmsi_{mmsi or '*'}.nc"
    hits = sorted(glob.glob(str(base / settings["Output"]["plume_out_dir"] / f"plumes_{pass_key[2:8]}" / pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} plume files match {pattern} (set MMSI or PLUME_FILE): {hits}")
    return Path(hits[0])


def load_camera_image(settings, t, crop=None):
    """Camera image closest to time t (UTC) from <img_dir>/<yymmdd>_video.zip or the folder <yymmdd>_video,
    as in video_camera.assign_video_images_to_ship_pass. Returns (image array, image time) or (None, None)."""
    import io
    import zipfile
    from PIL import Image
    from SEICOR.video_camera import parse_img_time
    date = t.strftime("%y%m%d")
    img_dir = Path(settings["base_paths"][PLAT_KEY]) / settings["Instruments"]["img_dir"]
    zip_path, dir_path = img_dir / f"{date}_video.zip", img_dir / f"{date}_video"
    if zip_path.is_file():
        zf = zipfile.ZipFile(zip_path)
        names = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
        read = zf.read
    elif dir_path.is_dir():
        zf = None
        names = [str(p) for p in dir_path.iterdir() if p.suffix.lower() == ".jpg"]
        read = lambda n: Path(n).read_bytes()  # noqa: E731
    else:
        warnings.warn(f"no camera images for {date} in {img_dir}")
        return None, None
    timed = [(n, parse_img_time(Path(n).name)) for n in names]
    timed = [(n, ti) for n, ti in timed if ti is not None]
    if not timed:
        return None, None
    name, t_img = min(timed, key=lambda x: abs(x[1] - t))
    img = Image.open(io.BytesIO(read(name))).convert("RGB")
    if zf is not None:
        zf.close()
    if crop is not None:
        w, h = img.size
        img = img.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return np.asarray(img), t_img.tz_convert("UTC")


def load_old_slant(settings, t_start, t_end):
    """NO2 dSCDs of the old retrieval (OLD_SC_PATH, OLD_SC_FILE_ENDING) between t_start and t_end (UTC) for all
    rows (viewing directions, 0-based: file 01 -> row 0); zenith spectra (LOS 180) are dropped. vea is LOS - 90 of
    the old files (old calibration). Returns (times as DatetimeIndex, vea per row, no2 [row, time]) or None.
    Reading the 47 MB files is slow, so the result is cached as .npz in <plots_out_dir>/processing_steps/."""
    from imaging_tools.utilities import get_month_folder
    date = t_start.strftime("%y%m%d")
    base = Path(settings["base_paths"][PLAT_KEY])
    cache = (base / settings["Output"]["plots_out_dir"] / "processing_steps"
             / f"old_{OLD_SC_FILE_ENDING}_{t_start:%Y%m%d_%H%M%S}_{t_end:%H%M%S}_all.npz")
    if cache.is_file():
        c = np.load(cache)
        return pd.DatetimeIndex(c["times"]).tz_localize("UTC"), c["vea"], c["no2"]
    folder = base / OLD_SC_PATH.format(yy=date[:2]) / get_month_folder(date)
    day0 = pd.Timestamp("1992-12-31", tz="UTC")  # day 0 of the NLIN time column (doas_tools.file_handling)
    times, vea, no2 = None, [], []
    for i in range(1, 100):
        f = folder / f"{date}{i:02d}{OLD_SC_FILE_ENDING}"
        if not f.is_file():
            break
        # columns: 0 day of year since 1993, 5 LOS, 9 NO2 SCD
        d = pd.read_csv(f, comment="*", header=None, usecols=[0, 5, 9], names=["doy", "los", "no2"], sep=r"\s+", encoding="cp850")
        d = d[d.los != 180.0]
        t = day0 + pd.to_timedelta(d.doy.values, unit="D")
        sel = (t >= t_start) & (t <= t_end)
        d, t = d[sel], t[sel]
        if not len(d):
            break
        v = float(np.median(d.los)) - 90
        if times is None:
            times = t
        # rows of one image share the time stamp; align to the first row in case single spectra are missing
        s_ = pd.Series(d.no2.values, index=t)
        s_ = s_[~s_.index.duplicated()]
        no2.append(s_.reindex(times, method="nearest", tolerance=pd.Timedelta(seconds=1)).values)
        vea.append(v)
    if times is None:
        warnings.warn(f"no old slant columns {OLD_SC_FILE_ENDING} in {folder} for {t_start}..{t_end}")
        return None
    vea, no2 = np.array(vea), np.array(no2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, times=times.tz_convert(None).values, vea=vea, no2=no2)
    return times, vea, no2


def bin_in_time(times, values, res_s):
    """Mean of values [row, time] in fixed time bins of res_s seconds aligned to midnight UTC, as the preprocessor
    coarsens the IMPACT data (empty bins dropped); returns (bin centres as DatetimeIndex, binned values)."""
    midnight = times[0].normalize()
    k = np.floor((times - midnight).total_seconds().values / res_s).astype(int)
    df = pd.DataFrame(values.T).groupby(k).mean()
    return midnight + pd.to_timedelta((df.index.values + 0.5) * res_s, unit="s"), df.values.T


def coarsen_factor(settings, date):
    """Raw measurements per coarsened one, from the coarsened IMPACT file of the day (scales the
    cluster sizes of the detection like in the preprocessor); 1 without coarsening or file."""
    res = settings["processing"]["coarsen"]["time_resolution_seconds"]
    if not res:
        return 1.0
    f = Path(settings["base_paths"][PLAT_KEY]) / settings["Output"]["impact_coarsened_out_dir"] / f"IMPACT_{date}_{res:g}s.nc"
    try:
        with xr.open_dataset(f) as d:
            factor = float(d.attrs.get("coarsen_factor", 1.0))
    except FileNotFoundError:
        warnings.warn(f"{f} not found: cluster sizes of the detection are not scaled")
        return 1.0
    if not np.isfinite(factor) or factor <= 0:
        warnings.warn(f"invalid coarsen_factor {factor} in {f}: cluster sizes of the detection are not scaled")
        return 1.0
    return factor


def make_figure(ds, settings, factor, elevations=ELEVATIONS):
    """The figure (a)-(d) of the plume dataset ds; returns the matplotlib figure."""
    proc = settings["processing"]
    det, pms = proc["detection"], proc["plume_mask"]
    if "no2_ref" not in ds:
        raise ValueError("the plume file has no upwind reference (no2_ref): nothing to show")
    t0 = pd.to_datetime(ds.attrs["t"], utc=True)

    def minutes(t):
        return (pd.to_datetime(t, utc=True) - t0).total_seconds().values / 60

    vea, rows = ds.vea.values, ds.image_row.values
    dv = np.gradient(vea)
    y_lim = (vea[0] - dv[0] / 2, vea[-1] + dv[-1] / 2)

    def vea_edges(r0, r1):
        """VEA edges of the image rows r0..r1 (original viewing direction numbers, inclusive), clipped to the image."""
        sel = np.nonzero((rows >= r0) & (rows <= r1))[0]
        return vea[sel[0]] - dv[sel[0]] / 2, vea[sel[-1]] + dv[sel[-1]] / 2

    # measured dSCDs: the extended window (10 min before to 15 min after the pass) if stored
    if "no2_extended" in ds:
        m_raw, raw = minutes(ds.times_plume_extended.values), ds.no2_extended.values / E
    else:
        m_raw, raw = minutes(ds.times_plume.values), ds.no2.values / E
    m_pl = minutes(ds.times_plume.values)
    m_up = minutes(ds.times_ref.values)
    has_down = "no2_ref_down" in ds
    m_dn = minutes(ds.times_ref_down.values) if has_down else None

    # enhancement as used by the preprocessor, before and after the background correction
    enh_var = ds.attrs.get("enhancement_var_used") or ("no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back")
    enh = ds[enh_var].values
    enh_raw = ds[f"{enh_var}_uncorrected"].values if f"{enh_var}_uncorrected" in ds else enh
    plume_mask = ds.plume_mask.values.astype(bool) if "plume_mask" in ds else np.zeros(enh.shape, bool)
    ship_mask = ds.ship_mask.values.astype(bool) if "ship_mask" in ds else np.zeros(enh.shape, bool)

    # background: per-row mean of the reference windows, linear in time between their centres
    up = ds.no2_ref.mean("window_ref").values
    dn = ds.no2_ref_down.mean("window_ref_down").values if has_down else up
    c_up = m_up.mean()
    c_dn = m_dn.mean() if has_down else c_up + 1.0

    # plume detection as in plumes.sort_plumes, on the initial plume window (before any extension)
    before, after = proc["plume_window"]["before_minutes"], proc["plume_window"]["after_minutes"]
    init = (m_pl >= -before) & (m_pl < after)
    bg_level = float(np.nanmedian(enh_raw[:, init]))
    bg_std = float(plumes.reference_image(ds).std())
    s_before, s_after = pms["source_seconds_before"], pms["source_seconds_after"]
    r_p = np.nonzero((rows >= det["plume_rows"][0]) & (rows < det["plume_rows"][1]))[0]
    c_p = np.nonzero(init & (m_pl * 60 >= -s_before) & (m_pl * 60 <= s_after))[0]
    detected = np.zeros(enh.shape, bool)
    if len(r_p) and len(c_p):
        detected[np.ix_(r_p, c_p)] = plumes.detect_plume_ztest(
            enh_raw[np.ix_(r_p, c_p)], bg_mean=bg_level, bg_std=bg_std, p_threshold=det["p_threshold_plume"],
            min_cluster_size=impact.scale_count(det["plume_min_cluster_size"], factor))
    print(f"MMSI {ds.attrs.get('mmsi')} at {t0}: {enh_var}, {'up- and downwind' if has_down else 'upwind'} reference, "
          f"{ds.attrs.get('plume_window_extensions', 0)} window extension(s)\n"
          f"detection: {int(detected.sum())} plume pixels (level {bg_level:.2e}, reference std {bg_std:.2e}); "
          f"masks: plume {int(plume_mask.sum())}, ship {int(ship_mask.sum())} pixels")

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    o4_var = enh_var.replace("no2", "o4")
    camera, t_cam = load_camera_image(settings, t0 + pd.Timedelta(seconds=IMAGE_OFFSET_S), IMAGE_CROP) if PLOT_IMAGE else (None, None)
    old = None
    if PLOT_OLD:
        old = load_old_slant(settings, pd.to_datetime(ds.times_ref.values[0], utc=True) - pd.Timedelta(seconds=5),
                             pd.to_datetime(ds.times_plume.values[-1], utc=True) + pd.Timedelta(seconds=5))

    # layout of the double-column figure of Copernicus (AMT), 17 cm wide:
    #   camera            | measured dSCDs
    #   legend (contours) | background time series   (same widths, so the time axes line up)
    #   [interpolated background, optional, full width]
    #   enhancement       | masks
    #   old retrieval     | O4                       (same size as the panels above)
    heights = [1, 1] + ([1] if PLOT_BACKGROUND else []) + [1.05, 1.05]
    fig = plt.figure(figsize=(6.7, 0.35 + 1.8 * len(heights)))
    gs = fig.add_gridspec(len(heights), 2, height_ratios=heights, hspace=0.3, wspace=0.12,
                          left=0.085, right=0.885, top=1 - 0.1 / len(heights), bottom=0.2 / len(heights))
    gs_top = gs[0:2, :].subgridspec(2, 2, width_ratios=[1, 2.4], wspace=0.3, hspace=0.12)
    ax_a = fig.add_subplot(gs_top[0, 1])
    ax_b = fig.add_subplot(gs_top[1, 1], sharex=ax_a)
    ax_cam = fig.add_subplot(gs_top[0, 0]) if camera is not None else None
    ax_leg = fig.add_subplot(gs_top[1, 0])
    ax_leg.axis("off")
    row = iter(range(2, len(heights)))
    ax_bg = fig.add_subplot(gs[next(row), :]) if PLOT_BACKGROUND else None
    row_enh = next(row)
    # enhancement panels: no time labels between the two rows, so they sit closer together
    gs_bot = gs[row_enh:row_enh + 2, :].subgridspec(2, 2, hspace=0.1, wspace=0.12)
    ax_c, ax_d = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])
    plot_o4 = PLOT_O4 and o4_var in ds
    ax_old = fig.add_subplot(gs_bot[1, 0], sharex=ax_c) if old is not None else None
    ax_o = fig.add_subplot(gs_bot[1, 1], sharex=ax_d) if plot_o4 else None

    def cbar(im, ax, label):
        """Colour bar right of the plot area (x = 0.905), as high as ax."""
        pos = ax.get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([0.905, pos.y0, 0.015, pos.height]))
        cb.set_label(label)

    windows = [((m_up[0], m_up[-1]), C_UP, "-", "pre-pass back."), ((m_pl[0], m_pl[-1]), INK, "--", "plume window")]
    if has_down:
        windows.append(((m_dn[0], m_dn[-1]), C_DN, "-", "post-pass back."))
    x_lim = (min(m_raw[0], m_up[0]), max(m_raw[-1], m_dn[-1] if has_down else m_raw[-1]))

    # default windows (settings), drawn dotted where they differ from the ones used: the initial plume
    # window before any extension and the first downwind reference the search tries (enhancements.find_reference_window);
    # explained in the caption, not labelled in the plot
    ref_set = proc["reference_search"]
    dn0 = after + ref_set["start_offset_minutes"]
    default_windows = []
    if abs(m_pl[0] + before) > 0.1 or abs(m_pl[-1] - after) > 0.1:
        default_windows.append(((-before, after), LINE_DSCD))
    if has_down and abs(m_dn[0] - dn0) > 0.1:
        default_windows.append(((dn0, dn0 + ref_set["window_minutes"]), C_DN))

    def draw_default_windows(ax):
        for (lo, hi), colour in default_windows:
            ax.add_patch(Rectangle((lo, y_lim[0]), hi - lo, y_lim[1] - y_lim[0], fill=False, ec=colour, lw=1.0, ls=":"))

    # camera image of the ship (the axes shrink to the image aspect)
    if camera is not None:
        ax_cam.imshow(camera, aspect="equal")
        ax_cam.set_anchor("E")
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        print(f"camera image {t_cam:%H:%M:%S} UTC ({(t_cam - t0).total_seconds():+.0f} s from the pass)")

    # measured dSCDs
    # colour range from the rows above the ship (the lowest rows have much larger dSCDs and saturate)
    vmin, vmax = np.nanpercentile(raw[rows >= det["plume_rows"][0]], [1, 99.5])
    im_a = ax_a.pcolormesh(m_raw, vea, raw, cmap=CMAP_DSCD, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
    for (lo, hi), colour, ls, name in windows:
        # grey shades as in the line plot below: background windows darker, plume window light (keeps the plume visible)
        if name == "plume window":
            ax_a.axvspan(lo, hi, color="0.96", alpha=0.35, lw=0, zorder=2)
        else:
            ax_a.axvspan(lo, hi, color="0.88", alpha=0.6, lw=0, zorder=2)
        # labels above the boxes: upwind right-aligned to its box end, downwind left-aligned, so they do not overlap
        x_txt, ha = {"pre-pass back.": (hi, "right"), "post-pass back.": (lo, "left")}.get(name, ((lo + hi) / 2, "center"))
        ax_a.text(x_txt, y_lim[1] + 0.5, name, color=INK, ha=ha, va="bottom", fontsize=6.5)
    draw_default_windows(ax_a)
    ax_a.axvline(0, color=LINE_DSCD, lw=0.8, ls=":")
    ax_a.set_xlim(*x_lim)
    ax_a.set_ylim(*y_lim)
    ax_a.set_ylabel("elevation / °")
    ax_a.tick_params(labelbottom=False)  # same time axis as the background panel below
    cbar(im_a, ax_a, "NO$_2$ dSCD / 10$^{16}$ molec. cm$^{-2}$")

    # background from the references at three elevations
    tt = np.linspace(*x_lim, 400)
    alpha = np.clip((tt - c_up) / (c_dn - c_up), 0, 1) if has_down else np.zeros_like(tt)
    end_labels = []  # [y, text] of the elevation labels at the right end of the background lines
    for v, colour in zip(elevations, ROW_COLOURS):
        i = int(np.argmin(np.abs(vea - v)))
        ax_b.plot(m_raw, pd.Series(raw[i]).rolling(7, center=True, min_periods=1).mean(), color=colour, lw=0.9)
        bg = ((1 - alpha) * up[i] + alpha * dn[i]) / E
        ax_b.plot(tt, bg, color=colour, lw=1.2, ls="--")
        ax_b.plot([m_up[0], m_up[-1]], [up[i] / E] * 2, color=colour, lw=3, solid_capstyle="butt")
        if has_down:
            ax_b.plot([m_dn[0], m_dn[-1]], [dn[i] / E] * 2, color=colour, lw=3, solid_capstyle="butt")
        end_labels.append([bg[-1], f"{vea[i]:.1f}°"])
    for (lo, hi), _, _, name in windows:
        ax_b.axvspan(lo, hi, color="0.96" if name == "plume window" else "0.88", lw=0, zorder=0)
    # elevation labels at the line ends, pushed apart to at least 7 % of the y range so they do not overlap
    y_lo, y_hi = ax_b.get_ylim()
    y_hi += 0.15 * (y_hi - y_lo)  # headroom for the legend above the highest line
    ax_b.set_ylim(y_lo, y_hi)
    gap = 0.07 * (y_hi - y_lo)
    end_labels.sort()
    for k in range(1, len(end_labels)):
        end_labels[k][0] = max(end_labels[k][0], end_labels[k - 1][0] + gap)
    for y, text in end_labels:
        ax_b.text(x_lim[1] + 0.01 * (x_lim[1] - x_lim[0]), y, text, color=INK, va="center", fontsize=7)
    ax_b.axvline(0, color=INK, lw=0.8, ls=":")
    ax_b.set_xlim(*x_lim)
    ax_b.set_xlabel("Time / min")
    ax_b.set_ylabel("NO$_2$ dSCD / 10$^{16}$ molec. cm$^{-2}$")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(handles=[Line2D([], [], color=MUTED, lw=0.9, label="measured (10 s mean)"),
                         Line2D([], [], color=MUTED, lw=3, label="ref. mean"),
                         Line2D([], [], color=MUTED, lw=1.2, ls="--",
                                label="interp. background" if has_down else "background (upwind ref.)")],
                loc="upper left", bbox_to_anchor=(0.06, 1.0), frameon=False, ncol=3, fontsize=6.5,
                handlelength=1.6, columnspacing=1.0, handletextpad=0.5)  # right of the panel letter

    # enhancement panels: one colour scale (also for the old retrieval), source region, contours
    lim = 2 * np.nanmedian(enh[plume_mask]) / E if plume_mask.any() else 4 * bg_std / E
    kw = dict(cmap=CMAP_ENH, vmin=-lim, vmax=lim, shading="nearest", rasterized=True)
    src_y0, src_y1 = vea_edges(*pms["source_rows"])  # source_rows: inclusive (plumes._detect_plume_ztest_core)
    src_rect = dict(xy=(-s_before / 60, src_y0), width=(s_before + s_after) / 60, height=src_y1 - src_y0,
                    fill=False, ec=INK, lw=1.2, ls="-.", zorder=3)  # above the image (added before pcolormesh)

    def draw_masks(ax):
        if plume_mask.any():
            ax.contour(m_pl, vea, plume_mask.astype(int), levels=[0.5], colors=INK, linewidths=1.0)
        if ship_mask.any():
            ax.contour(m_pl, vea, ship_mask.astype(int), levels=[0.5], colors=C_SHIP, linewidths=1.2)

    # ship and plume in the old retrieval: dSCD minus the mean over the upwind reference window, per row
    if old is not None:
        t_old, vea_old, no2_old = old
        # same time bins as the current data (the old files have one value per spectrum, ~0.4 s)
        res = settings["processing"]["coarsen"]["time_resolution_seconds"]
        if res:
            t_old, no2_old = bin_in_time(t_old, no2_old, res)
        # drop the lowest viewing directions as in the plume file plus OLD_EXTRA_DROP more; the rows are labelled
        # with the elevations of the plume file (current calibration), so they match the other panels
        row_old = np.arange(len(vea_old))
        keep = (row_old >= rows.min() + OLD_EXTRA_DROP) & np.isin(row_old, rows)
        no2_old = no2_old[keep]
        vea_old = np.array([vea[rows == r][0] for r in row_old[keep]])
        m_old = minutes(t_old)
        in_ref = (m_old >= m_up[0] - 1e-3) & (m_old <= m_up[-1] + 1e-3)
        enh_old = (no2_old - np.nanmean(no2_old[:, in_ref], axis=1, keepdims=True)) / E
        show = (m_old >= m_pl[0]) & (m_old <= m_pl[-1])
        ax_old.pcolormesh(m_old[show], vea_old, enh_old[:, show], **kw)
        draw_masks(ax_old)
        ax_old.axvline(0, color=INK, lw=0.8, ls=":")
        ax_old.set_xlim(m_pl[0], m_pl[-1])
        dv_old = np.gradient(vea_old)
        ax_old.set_ylim(vea_old[0] - dv_old[0] / 2, vea_old[-1] + dv_old[-1] / 2)
        ax_old.set_xlabel("Time / min")
        ax_old.set_ylabel("elevation / °")

    # optional: interpolated background as an image, same colour scale as the measured dSCDs
    if PLOT_BACKGROUND:
        a_raw = np.clip((m_raw - c_up) / (c_dn - c_up), 0, 1) if has_down else np.zeros_like(m_raw)
        background = ((1 - a_raw)[None, :] * up[:, None] + a_raw[None, :] * dn[:, None]) / E
        im_bg = ax_bg.pcolormesh(m_raw, vea, background, cmap=CMAP_DSCD, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
        for (lo, hi), colour, ls, name in windows:
            ax_bg.add_patch(Rectangle((lo, y_lim[0]), hi - lo, y_lim[1] - y_lim[0], fill=False, ec=colour if colour != INK else LINE_DSCD,
                                      lw=1.4, ls=ls))
        draw_default_windows(ax_bg)
        ax_bg.axvline(0, color=LINE_DSCD, lw=0.8, ls=":")
        ax_bg.set_xlim(*x_lim)
        ax_bg.set_ylim(*y_lim)
        ax_bg.set_ylabel("elevation / °")
        ax_bg.set_xlabel("Time / min")
        cbar(im_bg, ax_bg, "NO$_2$ dSCD / 10$^{16}$ molec. cm$^{-2}$")

    # enhancement and detection, masks
    for ax in (ax_c, ax_d):
        ax.axvline(0, color=INK, lw=0.8, ls=":")
        ax.set_xlim(m_pl[0], m_pl[-1])
        ax.set_ylim(*y_lim)
        ax.add_patch(Rectangle(**src_rect))
    ax_c.pcolormesh(m_pl, vea, enh_raw / E, **kw)
    tol = det["time_tolerance_seconds"] / 60
    y0, y1 = vea_edges(det["ship_rows"][0], det["ship_rows"][1] - 1)  # ship_rows: python slice [start, stop)
    ax_c.add_patch(Rectangle((-tol, y0), 2 * tol, y1 - y0, fill=False, ec=C_SHIP, lw=1.0, ls="--"))
    if detected.any():
        ax_c.contour(m_pl, vea, detected.astype(int), levels=[0.5], colors=INK, linewidths=0.9, linestyles="--")
    ax_c.set_ylabel("elevation / °")
    im_d = ax_d.pcolormesh(m_pl, vea, enh / E, **kw)
    draw_masks(ax_d)
    ax_d.tick_params(labelleft=False)
    # time labels only on the lowest panel of each column
    if old is not None:
        ax_c.tick_params(labelbottom=False)
    else:
        ax_c.set_xlabel("Time / min")
    if plot_o4:
        ax_d.tick_params(labelbottom=False)
    else:
        ax_d.set_xlabel("Time / min")
    cbar(im_d, ax_d, NO2_ENH + " / 10$^{16}$ molec. cm$^{-2}$")

    # O4 enhancement (same background method as NO2, no background correction) with the NO2 masks
    if plot_o4:
        o4 = ds[o4_var].values * O4_STORED / O4_E
        o4_lim = np.nanpercentile(np.abs(o4), 99)
        im_o = ax_o.pcolormesh(m_pl, vea, o4, cmap=CMAP_ENH, vmin=-o4_lim, vmax=o4_lim, shading="nearest", rasterized=True)
        draw_masks(ax_o)
        ax_o.axvline(0, color=INK, lw=0.8, ls=":")
        ax_o.set_xlim(m_pl[0], m_pl[-1])
        ax_o.set_ylim(*y_lim)
        ax_o.tick_params(labelleft=False)  # same elevations as the old retrieval left of it
        ax_o.set_xlabel("Time / min")
        cbar(im_o, ax_o, O4_ENH + " / " + O4_UNIT)

    # one legend for the contours and regions of all enhancement panels, in the free slot under the camera
    ax_leg.legend(handles=[Line2D([], [], color=INK, lw=1.2, ls="-.", label="source region"),
                           Line2D([], [], color=C_SHIP, lw=1.0, ls="--", label="ship region"),
                           Line2D([], [], color=INK, lw=0.9, ls="--", label="detected plume"),
                           Line2D([], [], color=INK, lw=1.0, label="plume mask (NO$_2$)"),
                           Line2D([], [], color=C_SHIP, lw=1.2, label="ship mask (NO$_2$)")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7.5, handlelength=2.2)

    # panel letters inside the axes, top left, in reading order; short names of the enhancement panels next to them
    names = {ax_c: "Detection", ax_d: "Identification", ax_old: "No smear correction", ax_o: O4_ENH}
    box = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85)
    letters = iter("abcdefghij")
    for ax in (ax_cam, ax_a, ax_b, ax_bg, ax_c, ax_d, ax_old, ax_o):
        if ax is not None:
            x = 0.012 if ax is not ax_cam else 0.03
            ax.text(x, 0.965, f"({next(letters)})", transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", fontsize=9, color=INK, zorder=10, bbox=box)
            if ax in names:
                ax.annotate(names[ax], xy=(x, 0.965), xycoords="axes fraction", xytext=(19, 0), textcoords="offset points",
                            ha="left", va="top", fontsize=8, color=INK, zorder=10, bbox=box)
    return fig


#%% run
settings = yaml.safe_load(open(SETTINGS_FILE))
plume_file = Path(PLUME_FILE) if PLUME_FILE else find_plume_file(PASS_KEY, settings, MMSI)
print(f"plume file: {plume_file}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    ds = xr.load_dataset(plume_file)
    fig = make_figure(ds, settings, coarsen_factor(settings, PASS_KEY[2:8]))

if OUT_FILE is not False:
    out = Path(OUT_FILE) if OUT_FILE else (Path(settings["base_paths"][PLAT_KEY]) / settings["Output"]["plots_out_dir"]
                                           / "processing_steps" / f"processing_steps_{PASS_KEY}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.suffix.lower() != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=300)  # preview
    print(f"saved {out}")
plt.show()

# %%
