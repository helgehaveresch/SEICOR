"""Evaluation of the detection parameter study (run_detection_study.py): which combination of p_threshold_plume,
plume_min_cluster_size and p_threshold_ship (with its ship_min_cluster_size) flags actual plumes and ships best,
while it sorts out noise and signals far below the detection limit.

There is no ground truth, so the evaluation uses references that do not depend on the detection parameters:

1. Noise only (null windows). The extended enhancement image of every pass (no2_enhancement_interp_extended,
   -10..+15 min) has time windows before the pre-pass background window with no ship of this pass. Windows that are
   also free of all other ship passes of the day (AIS, ship_passes csv: no pass from NULL_CLEAR_BEFORE_S before to
   NULL_CLEAR_AFTER_S after the window) contain only noise and background structure. Each such window is treated like
   a pass with its pass time source_seconds_before after the window start: plume region and ship region as in
   plumes.sort_plumes. Flags there are false alarms.
2. Signal strength. Matched-filter scores of the z-score image (enhancement - background level) / noise level, with
   the background and noise level of the detection, averaged over boxes of BOX rows x columns; score = the largest
   box mean x sqrt(box size):
     S_plume  plume region (plume_rows, source time window), positive side (plume)
     S_ship   ship region (ship_rows, +-time_tolerance_seconds), negative side (the ship blocks the light)
     S_full   whole initial plume window from plume_rows[0] up, positive side: plumes outside the plume region
              (higher or later), which the detection cannot see by design
   The same scores of the null windows give their noise distributions. A score above the S_HI_QUANTILE quantile of
   the null scores is a clear signal, below the S_LO_QUANTILE quantile it is below the detection limit. S_full is
   compared with the null quantile corrected for its larger area (max of k independent plume regions).

The plume and the ship test of every combination are recomputed on exactly the input the preprocessor used (the plume
window before any extension and background correction, background level and noise level of plumes.sort_plumes;
min cluster sizes scaled with the coarsening factor of the day), with the parameters of the combination's
settings.yaml. The flag of a pass is plume test or (only if no plume) ship test, as in sort_plumes. The recomputed
flags are compared with the plume_or_ship_found flags of the study runs (consistency check).

Per combination:
    recall_plume   fraction of the clear plumes (S_plume) found by the plume test
    recall_ship    fraction of the clear ship signatures (S_ship) flagged
    FN             passes with a clear signal (plume, ship or S_full) that are not flagged (false negatives)
    FN_outside     of these: clear plume only outside the plume region (limit of the region, not of the parameters)
    below_flagged  fraction of the passes with all scores below the detection limit that are flagged
    far_plume / far_ship / far_flag   false alarm rates of plume test / ship test / flag in the null windows
Recommended: far_flag <= MAX_FAR and below_flagged <= MAX_BELOW, then the fewest false negatives (FN), then the
lowest far_flag; if no combination meets the limits, the best recall_plume + recall_ship - 2 far_flag - below_flagged.

Output in <base>/<STUDY_DIR>/evaluation/: passes.csv, null_windows.csv, combinations.csv, report.md and figures:
heatmaps.png, score_distribution.png, false_negatives.png (missed passes of the recommended combination),
null_windows_top.png (null windows with the highest scores: check that they are noise), disputed_passes.png.

Usage: python3 evaluate_detection_study.py [dates]
"""
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from SEICOR import plumes, impact, plotting  # noqa: E402
from run_detection_study import STUDY_DIR, BASE  # noqa: E402
from run_plume_preprocessor_test import TEST_DATES  # noqa: E402

BOX = (3, 7)                 # rows x columns (1.5 s) of the matched filter: ~3 elevations x 10 s
S_HI_QUANTILE = 0.995        # clear signal: score above this quantile of the null scores
S_LO_QUANTILE = 0.90         # below the detection limit: score below this quantile of the null scores
NULL_CLEAR_BEFORE_S = 420    # null window: no ship pass from this long before its start ...
NULL_CLEAR_AFTER_S = 60      # ... to this long after its end (plumes of earlier ships drift through for minutes)
NULL_GAP_TO_REF_S = 10       # null windows end at least this long before the pre-pass background window
NULL_REF_S = 60              # background of a null window: per-row means over this long ...
NULL_REF_GAP_S = 60          # ... this long before and after the window, interpolated in time (as for a pass)
MAX_FAR = 0.05
MAX_BELOW = 0.05
N_EXAMPLES = 24
DATES = sys.argv[1:] or TEST_DATES  # e.g. python3 evaluate_detection_study.py 250520 (quick test on one day)

STUDY = BASE / STUDY_DIR
OUT = STUDY / "evaluation"


def minutes(t, t0):
    return (pd.to_datetime(t, utc=True) - t0).total_seconds().values / 60


def seconds(t, t0):
    return (pd.to_datetime(t, utc=True) - t0).total_seconds().values


def score(z):
    """Matched-filter score of a z-score image: largest BOX mean x sqrt(box size) (NaN counts as 0)."""
    z = np.nan_to_num(np.asarray(z, float), nan=0.0)
    if z.shape[0] < BOX[0] or z.shape[1] < BOX[1]:
        return np.nan
    return float(uniform_filter(z, size=BOX, mode="constant").max() * np.sqrt(BOX[0] * BOX[1]))


def find_combos():
    """Combinations with results for all test days, with the detection parameters of their settings.yaml."""
    rows = []
    for f in sorted(STUDY.glob("*/settings.yaml")):
        name = f.parent.name
        n_days = sum((f.parent / "ship_passes" / f"ship_passes_{d}.csv").is_file() for d in DATES)
        if n_days < len(DATES):
            print(f"skip {name}: {n_days}/{len(DATES)} days")
            continue
        det = yaml.safe_load(f.read_text())["processing"]["detection"]
        rows.append(dict(combo=name, p_plume=det["p_threshold_plume"], mcs_plume=det["plume_min_cluster_size"],
                         p_ship=det["p_threshold_ship"], mcs_ship=det["ship_min_cluster_size"]))
    return pd.DataFrame(rows)


def coarsen_factor(date, settings):
    res = settings["processing"]["coarsen"]["time_resolution_seconds"]
    with xr.open_dataset(BASE / settings["Output"]["impact_coarsened_out_dir"] / f"IMPACT_{date}_{res:g}s.nc") as d:
        return float(d.attrs.get("coarsen_factor", 1.0))


def tests(plume_region, ship_region, bg_mean, bg_std, factor, C):
    """Plume test for every (p_plume, mcs_plume) and ship test for every (p_ship, mcs_ship) of the combinations;
    returns {combo: (plume found, ship found, flag)}; the ship test counts only without plume, as in sort_plumes."""
    plume = {(p, n): bool(plumes.detect_plume_ztest(plume_region, bg_mean=bg_mean, bg_std=bg_std, p_threshold=p,
                                                    min_cluster_size=impact.scale_count(n, factor)).any())
             for p, n in set(zip(C.p_plume, C.mcs_plume))}
    ship = {}
    if ship_region.size:
        ship = {(p, n): bool(plumes.detect_plume_ztest_left(ship_region, bg_mean=bg_mean, bg_std=bg_std, p_threshold=p,
                                                           min_cluster_size=impact.scale_count(n, factor)).any())
                for p, n in set(zip(C.p_ship, C.mcs_ship))}
    out = {}
    for r in C.itertuples():
        pl = plume[(r.p_plume, r.mcs_plume)]
        sh = (not pl) and ship.get((r.p_ship, r.mcs_ship), False)
        out[r.combo] = (pl, sh, pl or sh)
    return out


def regions(img, s, rows, det, pms):
    """Plume region, ship region and full window (from plume_rows[0] up) of image img (rows x time); s: seconds of
    the columns relative to the pass time."""
    r_pl = (rows >= det["plume_rows"][0]) & (rows < det["plume_rows"][1])
    r_sh = (rows >= det["ship_rows"][0]) & (rows < det["ship_rows"][1])
    c_pl = (s >= -pms["source_seconds_before"]) & (s <= pms["source_seconds_after"])
    c_sh = np.abs(s) <= det["time_tolerance_seconds"]
    return img[np.ix_(r_pl, c_pl)], img[np.ix_(r_sh, c_sh)], img[rows >= det["plume_rows"][0]]


def null_background(t_ext, img, ws, we):
    """Background of a null window [ws, we] as for a real pass: per-row means of a NULL_REF_S window ending
    NULL_REF_GAP_S before it and one starting NULL_REF_GAP_S after it, interpolated linearly in time between their
    centres. Returns (columns of the window, background) or None if a background window has too few columns."""
    gap, length = pd.Timedelta(seconds=NULL_REF_GAP_S), pd.Timedelta(seconds=NULL_REF_S)
    b = (t_ext >= ws - gap - length) & (t_ext < ws - gap)
    a = (t_ext > we + gap) & (t_ext <= we + gap + length)
    if b.sum() < 10 or a.sum() < 10:
        return None
    x = (t_ext - ws).total_seconds().values
    xb, xa = x[b].mean(), x[a].mean()
    mb, ma = np.nanmean(img[:, b], axis=1, keepdims=True), np.nanmean(img[:, a], axis=1, keepdims=True)
    cols = (t_ext >= ws) & (t_ext <= we)
    w = (x[cols] - xb) / (xa - xb)
    return cols, (1 - w) * mb + w * ma


def null_windows(ds, n_cols, pass_times):
    """Null windows of the extended image (all rows, n_cols columns) before the pre-pass background window and free of
    all ship passes of the day: list of (start, times, image)."""
    var = "no2_enhancement_interp_extended" if "no2_enhancement_interp_extended" in ds else "no2_enhancement_c_back_extended"
    if var not in ds or "times_plume_extended" not in ds:
        return []
    t_ext = pd.to_datetime(ds.times_plume_extended.values, utc=True)
    img = ds[var].values
    ref_start = pd.to_datetime(ds.times_ref.values[0], utc=True)
    # like a real pass, every null window gets its own background from a background window before and one after it
    # (null_background): the extended image far from the background windows of the pass has a background drift,
    # different in every row, that the real detection never sees
    out = []
    i = int(np.searchsorted(t_ext, t_ext[0] + pd.Timedelta(seconds=NULL_REF_S + NULL_REF_GAP_S)))
    last = ref_start - pd.Timedelta(seconds=NULL_GAP_TO_REF_S + NULL_REF_GAP_S + NULL_REF_S)  # end of the window
    while i + n_cols <= len(t_ext):
        ws, we = t_ext[i], t_ext[i + n_cols - 1]
        if we > last:
            break
        clear = not ((pass_times >= ws - pd.Timedelta(seconds=NULL_CLEAR_BEFORE_S))
                     & (pass_times <= we + pd.Timedelta(seconds=NULL_CLEAR_AFTER_S + NULL_REF_GAP_S + NULL_REF_S))).any()
        bg = null_background(t_ext, img, ws, we) if clear else None
        if bg is not None:
            cols, background = bg
            win = img[:, cols] - background
            if np.isfinite(win).mean() > 0.9:
                out.append((ws, t_ext[cols], win))
        i += n_cols  # non-overlapping
    return out


def load_flags(C):
    """plume_or_ship_found of the study runs: DataFrame (UTC_Time, MMSI) x combination."""
    flags = {}
    for name in C.combo:
        d = pd.concat(pd.read_csv(f, index_col=0, usecols=lambda c: c in ("UTC_Time", "MMSI", "plume_or_ship_found"))
                      for f in [STUDY / name / "ship_passes" / f"ship_passes_{d}.csv" for d in DATES])
        d.index = pd.to_datetime(d.index, utc=True)
        d = d.set_index("MMSI", append=True)
        flags[name] = d.loc[~d.index.duplicated(), "plume_or_ship_found"].astype(str) == "True"
    return pd.DataFrame(flags)


def plume_file_of(t, mmsi, date, flags_row, C):
    """Plume file of a pass, preferably from a combination that did not flag it (stored image = detection input:
    no window extension, no background correction)."""
    for name in sorted(C.combo, key=lambda nm: (bool(flags_row.get(nm, True)), nm)):
        hits = glob.glob(str(STUDY / name / "plumes" / f"plumes_{date}" / f"plume_*_t_{t:%Y%m%d_%H%M%S}_mmsi_{mmsi}.nc"))
        if hits:
            return Path(hits[0]), name
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = find_combos()
    print(f"{len(C)} complete combinations", flush=True)
    settings = yaml.safe_load((STUDY / C.combo.iloc[0] / "settings.yaml").read_text())
    proc = settings["processing"]
    det, pms = proc["detection"], proc["plume_mask"]
    before, after = proc["plume_window"]["before_minutes"], proc["plume_window"]["after_minutes"]
    flags = load_flags(C)
    print(f"{len(flags)} passes", flush=True)

    passes, nulls = [], []
    for date in DATES:
        day = flags[flags.index.get_level_values(0).strftime("%y%m%d") == date]
        if day.empty:
            continue
        factor = coarsen_factor(date, settings)
        pass_times = day.index.get_level_values(0)
        for (t, mmsi), frow in day.iterrows():
            f, src = plume_file_of(t, mmsi, date, frow, C)
            if f is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ds = xr.load_dataset(f)
                if "no2_ref" not in ds:
                    continue
                t0 = pd.to_datetime(ds.attrs["t"], utc=True)
                var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"  # as sort_plumes
                raw = ds[f"{var}_uncorrected"].values if f"{var}_uncorrected" in ds else ds[var].values
                s = seconds(ds.times_plume.values, t0)
                init = (s >= -before * 60 - 1e-3) & (s < after * 60)  # initial plume window (before any extension)
                img, s = raw[:, init], s[init]
                rows = ds.image_row.values
                bg_mean, bg_std = float(np.nanmedian(img)), float(plumes.reference_image(ds).std())
                if not np.isfinite(bg_std) or bg_std <= 0:
                    continue
                r_pl, r_sh, r_full = regions(img, s, rows, det, pms)
                if r_pl.size == 0:
                    continue
                row = dict(date=date, t=t, mmsi=mmsi, file=str(f), source_combo=src, bg_std=bg_std,
                           S_plume=score((r_pl - bg_mean) / bg_std), S_ship=score(-(r_sh - bg_mean) / bg_std),
                           S_full=score((r_full - bg_mean) / bg_std), n_full=r_full.size, n_plume=r_pl.size)
                for nm, (pl, sh, fl) in tests(r_pl, r_sh, bg_mean, bg_std, factor, C).items():
                    row[f"plume_{nm}"], row[f"ship_{nm}"], row[f"flag_{nm}"] = pl, sh, fl
                    row[f"study_{nm}"] = bool(frow[nm])
                passes.append(row)

                # null windows, as long as the plume region (which is longer than the ship region)
                n_cols = r_pl.shape[1]
                for ws, tw, nimg in null_windows(ds, n_cols, pass_times):
                    ns = seconds(tw, ws) - pms["source_seconds_before"]  # pass time source_seconds_before after the start
                    n_pl, n_sh, _ = regions(nimg, ns, rows, det, pms)
                    nbg = float(np.nanmedian(nimg))
                    # direct: realistic background (also plumes of ships without AIS, clouds); mirrored around the
                    # background level: noise only (positive plumes cannot make a plume alarm, dark structures no
                    # ship alarm; symmetric noise is unchanged)
                    nrow = dict(date=date, start=ws, file=str(f), bg_std=bg_std,
                                S_plume=score((n_pl - nbg) / bg_std), S_ship=score(-(n_sh - nbg) / bg_std),
                                S_plume_m=score(-(n_pl - nbg) / bg_std), S_ship_m=score((n_sh - nbg) / bg_std))
                    for nm, (pl, sh, fl) in tests(n_pl, n_sh, nbg, bg_std, factor, C).items():
                        nrow[f"plume_{nm}"], nrow[f"ship_{nm}"], nrow[f"flag_{nm}"] = pl, sh, fl
                    for nm, (pl, sh, fl) in tests(2 * nbg - n_pl, 2 * nbg - n_sh, nbg, bg_std, factor, C).items():
                        nrow[f"plume_m_{nm}"], nrow[f"ship_m_{nm}"], nrow[f"flag_m_{nm}"] = pl, sh, fl
                    nulls.append(nrow)
        print(f"{date}: {len(passes)} passes, {len(nulls)} null windows so far", flush=True)

    P = pd.DataFrame(passes)
    N = pd.DataFrame(nulls).drop_duplicates(subset=["date", "start"]).reset_index(drop=True)
    P.to_csv(OUT / "passes.csv", index=False)
    N.to_csv(OUT / "null_windows.csv", index=False)

    # thresholds from the mirrored null scores (noise only)
    th = dict(plume_hi=N.S_plume_m.quantile(S_HI_QUANTILE), plume_lo=N.S_plume_m.quantile(S_LO_QUANTILE),
              ship_hi=N.S_ship_m.quantile(S_HI_QUANTILE), ship_lo=N.S_ship_m.quantile(S_LO_QUANTILE))
    # S_full: max over k ~ n_full / n_plume independent plume regions -> quantile q^(1/k) of the null S_plume
    k = (P.n_full / P.n_plume).clip(lower=1)
    P["full_hi"] = [N.S_plume_m.quantile(S_HI_QUANTILE ** (1 / kk)) for kk in k]
    P["clear_plume"] = P.S_plume >= th["plume_hi"]
    P["clear_ship"] = P.S_ship >= th["ship_hi"]
    P["clear_outside"] = (P.S_full >= P.full_hi) & ~P.clear_plume
    P["clear_any"] = P.clear_plume | P.clear_ship | P.clear_outside
    P["below_all"] = (P.S_plume < th["plume_lo"]) & (P.S_ship < th["ship_lo"]) & (P.S_full < P.full_hi)
    P.to_csv(OUT / "passes.csv", index=False)

    rows = []
    for r in C.itertuples():
        nm = r.combo
        fl = P[f"flag_{nm}"]
        missed = P.clear_any & ~fl
        rows.append(dict(
            combo=nm, p_plume=r.p_plume, mcs_plume=r.mcs_plume, p_ship=r.p_ship, mcs_ship=r.mcs_ship,
            recall_plume=P.loc[P.clear_plume, f"plume_{nm}"].mean(),
            recall_ship=fl[P.clear_ship].mean(),
            FN=int(missed.sum()), FN_rate=missed.sum() / max(P.clear_any.sum(), 1),
            FN_outside=int((missed & P.clear_outside & ~P.clear_ship).sum()),
            below_flagged=fl[P.below_all].mean(),
            far_plume=N[f"plume_m_{nm}"].mean(), far_ship=N[f"ship_m_{nm}"].mean(), far_flag=N[f"flag_m_{nm}"].mean(),
            far_flag_realistic=N[f"flag_{nm}"].mean(), far_plume_realistic=N[f"plume_{nm}"].mean(),
            flagged=int(fl.sum()), mismatch_study=int((fl != P[f"study_{nm}"]).sum())))
    R = pd.DataFrame(rows)
    R["score"] = R.recall_plume + R.recall_ship - 2 * R.far_flag - R.below_flagged
    ok = R[(R.far_flag <= MAX_FAR) & (R.below_flagged <= MAX_BELOW)]
    if len(ok):
        best = ok.sort_values(["FN", "far_flag", "below_flagged"]).iloc[0]
        rule = f"far_flag <= {MAX_FAR}, below_flagged <= {MAX_BELOW}, then fewest false negatives"
    else:
        best = R.sort_values("score", ascending=False).iloc[0]
        rule = "no combination meets the limits: best recall_plume + recall_ship - 2 far_flag - below_flagged"
    R.to_csv(OUT / "combinations.csv", index=False)
    report(P, N, R, th, best, rule)
    figures(P, N, R, th, best)
    print(f"recommended: {best.combo}  (written to {OUT})", flush=True)


NULL_SHOW_S = 160  # length of a null window in the figures (the plume region: source_seconds_before + after)


def show(ax, file, bg_std, t_start=None, title=""):
    """Enhancement image of a plume file: plume window, or the null window from t_start with its background."""
    ds = xr.load_dataset(file)
    t0 = pd.to_datetime(ds.attrs["t"], utc=True)
    if t_start is None:
        var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
        img = ds[f"{var}_uncorrected"].values if f"{var}_uncorrected" in ds else ds[var].values
        x = minutes(ds.times_plume.values, t0)
    else:
        var = "no2_enhancement_interp_extended" if "no2_enhancement_interp_extended" in ds else "no2_enhancement_c_back_extended"
        t = pd.to_datetime(ds.times_plume_extended.values, utc=True)
        t_end = t_start + pd.Timedelta(seconds=NULL_SHOW_S)
        cols, background = null_background(t, ds[var].values, t_start, t_end)
        img, x = ds[var].values[:, cols] - background, minutes(t[cols], t0)
    lim = 4 * bg_std
    ax.pcolormesh(x, ds.vea.values, img, cmap=plotting.CMAP_ENH, vmin=-lim, vmax=lim, shading="nearest", rasterized=True)
    if t_start is None:
        ax.axvline(0, color="#222222", lw=0.6, ls=":")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)


def sheet(path, items, ncol=4):
    """Figure with one panel per item (callable drawing into an axes)."""
    if not items:
        return
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow), constrained_layout=True, squeeze=False)
    for ax, draw in zip(axes.ravel(), items):
        draw(ax)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figures(P, N, R, th, best):
    # heatmaps p_plume x mcs_plume, one row per (p_ship, mcs_ship)
    metrics = [("FN", "false negatives (passes)", "magma_r", None), ("recall_plume", "recall clear plumes", "magma", (0, 1)),
               ("recall_ship", "recall clear ships", "magma", (0, 1)), ("far_flag", "false alarm rate (noise)", "magma_r", (0, 0.5)),
               ("below_flagged", "flagged below the limit", "magma_r", (0, 0.5))]
    ships = R[["p_ship", "mcs_ship"]].drop_duplicates().sort_values(["p_ship", "mcs_ship"]).values.tolist()
    fig, axes = plt.subplots(len(ships), len(metrics), figsize=(3.2 * len(metrics), 2.7 * len(ships)),
                             constrained_layout=True, squeeze=False)
    for i, (ps, ms) in enumerate(ships):
        sub = R[(R.p_ship == ps) & (R.mcs_ship == ms)]
        for ax, (m, title, cmap, lim) in zip(axes[i], metrics):
            grid = sub.pivot(index="p_plume", columns="mcs_plume", values=m)
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim else dict(vmin=0, vmax=max(R[m].max(), 1))
            im = ax.imshow(grid.values, cmap=cmap, origin="lower", aspect="auto", **kw)
            for (a, b), v in np.ndenumerate(grid.values):
                if np.isfinite(v):
                    ax.text(b, a, f"{v:.0f}" if m == "FN" else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if im.norm(v) > 0.6 else "black")
            if best.p_ship == ps and best.mcs_ship == ms:
                bi, bj = list(grid.index).index(best.p_plume), list(grid.columns).index(best.mcs_plume)
                ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#56B4E9", lw=2.5))
            ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=7)
            ax.set_yticks(range(len(grid.index)), grid.index, fontsize=7)
            ax.set_xlabel("plume_min_cluster_size", fontsize=7)
            ax.set_ylabel("p_threshold_plume", fontsize=7)
            ax.set_title(f"{title}\np_ship {ps}, ship mcs {ms}", fontsize=8)
    fig.savefig(OUT / "heatmaps.png", dpi=170)
    plt.close(fig)

    # score distributions: passes vs noise
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), constrained_layout=True)
    for ax, s, lo, hi, name in ((axes[0], "S_plume", th["plume_lo"], th["plume_hi"], "plume region (positive)"),
                                (axes[1], "S_ship", th["ship_lo"], th["ship_hi"], "ship region (negative)")):
        bins = np.linspace(min(N[s + "_m"].min(), P[s].min()), np.nanpercentile(P[s], 98), 60)
        ax.hist(N[s + "_m"], bins=bins, density=True, alpha=0.6, color="0.5", label=f"null windows, mirrored (n={len(N)})")
        ax.hist(N[s], bins=bins, density=True, histtype="step", color="#222222", label="null windows, as they are")
        ax.hist(P[s], bins=bins, density=True, alpha=0.6, color="#D55E00", label=f"ship passes (n={len(P)})")
        for v, y, lab in ((lo, 0.95, f"below < {lo:.1f}"), (hi, 0.80, f"clear ≥ {hi:.1f}")):
            ax.axvline(v, color="#222222", ls="--", lw=1)
            ax.text(v, ax.get_ylim()[1] * y, " " + lab, fontsize=7, va="top")
        ax.set_xlabel(f"matched-filter score, {name}")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "score_distribution.png", dpi=200)
    plt.close(fig)

    nm = best.combo
    # false negatives of the recommended combination
    fn = P[P.clear_any & ~P[f"flag_{nm}"]].sort_values("S_plume", ascending=False).head(N_EXAMPLES)
    sheet(OUT / "false_negatives.png", [
        (lambda ax, r=r: show(ax, r.file, r.bg_std, title=f"{r.t:%y%m%d %H:%M} missed by {nm}: S_plume {r.S_plume:.1f}, "
                                                          f"S_ship {r.S_ship:.1f}, S_full {r.S_full:.1f}"
                                                          + (" (outside region)" if r.clear_outside else "")))
        for r in fn.itertuples()])

    # null windows with the highest scores: are they really noise?
    top = N.assign(S=N[["S_plume", "S_ship"]].max(axis=1)).sort_values("S", ascending=False).head(12)
    sheet(OUT / "null_windows_top.png", [
        (lambda ax, r=r: show(ax, r.file, r.bg_std, t_start=r.start,
                              title=f"null {r.start:%y%m%d %H:%M:%S}: S_plume {r.S_plume:.1f}, S_ship {r.S_ship:.1f}, "
                                    f"flag {nm}: {'yes' if r[f'flag_{nm}'] else 'no'}"))
        for _, r in top.iterrows()])

    # passes on which the combinations disagree, sorted by S_plume
    fcols = [f"flag_{c}" for c in R.combo]
    frac = P[fcols].mean(axis=1)
    disputed = P[(frac > 0) & (frac < 1)].sort_values("S_plume")
    if len(disputed):
        pick = disputed.iloc[np.unique(np.linspace(0, len(disputed) - 1, min(N_EXAMPLES, len(disputed))).astype(int))]
        sheet(OUT / "disputed_passes.png", [
            (lambda ax, i=i, r=r: show(ax, r.file, r.bg_std,
                                       title=f"{r.t:%y%m%d %H:%M} S_pl {r.S_plume:.1f} S_sh {r.S_ship:.1f}: flagged by "
                                             f"{int(frac[i] * len(fcols))}/{len(fcols)}, rec. {'yes' if r[f'flag_{nm}'] else 'no'}"))
            for i, r in pick.iterrows()])


def report(P, N, R, th, best, rule):
    lines = [
        "# Detection parameter study: evaluation", "",
        f"{len(P)} ship passes on {P.date.nunique()} test days, {len(N)} null windows (noise only, free of all ship passes), "
        f"{len(R)} parameter combinations.", "",
        f"Matched-filter scores (box {BOX[0]} rows x {BOX[1]} columns). Clear signal: above the {S_HI_QUANTILE:.1%} quantile "
        f"of the mirrored null scores (noise only); below the detection limit: below the {S_LO_QUANTILE:.0%} quantile.",
        "far_*: false alarm rates in the mirrored null windows (noise only: each null window mirrored around its "
        "background level, so positive plumes of ships without AIS cannot cause a plume alarm and dark structures no "
        "ship alarm). far_flag_realistic: in the null windows as they are (also plumes of ships without AIS, clouds, "
        "data gaps; see null_windows_top.png).",
        f"- plume region: clear S_plume >= {th['plume_hi']:.2f}: {int(P.clear_plume.sum())} passes",
        f"- ship region: clear S_ship >= {th['ship_hi']:.2f}: {int(P.clear_ship.sum())} passes",
        f"- clear plume only outside the plume region (S_full, area-corrected threshold): {int(P.clear_outside.sum())} passes",
        f"- any clear signal: {int(P.clear_any.sum())} passes; all scores below the limit: {int(P.below_all.sum())} passes", "",
        f"**Recommended: p_threshold_plume = {best.p_plume}, plume_min_cluster_size = {best.mcs_plume}, "
        f"p_threshold_ship = {best.p_ship} (ship_min_cluster_size {best.mcs_ship})** ({rule}):",
        f"false negatives {best.FN} of {int(P.clear_any.sum())} ({best.FN_outside} only outside the plume region), "
        f"recall clear plumes {best.recall_plume:.2f}, clear ships {best.recall_ship:.2f}, "
        f"false alarm rate {best.far_flag:.3f} (plume test {best.far_plume:.3f}, ship test {best.far_ship:.3f}), "
        f"flagged below the limit {best.below_flagged:.2f}.", "",
    ]
    cols = ["combo", "FN", "FN_outside", "recall_plume", "recall_ship", "far_flag", "far_plume", "far_ship",
            "far_flag_realistic", "below_flagged", "flagged", "mismatch_study"]
    t = R.sort_values(["FN", "far_flag"])[cols].copy()
    for c in ("recall_plume", "recall_ship", "far_flag", "far_plume", "far_ship", "far_flag_realistic", "below_flagged"):
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    lines += ["",
              "FN: passes with a clear plume, clear ship signature or clear plume outside the plume region that the "
              "combination does not flag (false negatives). FN_outside: of these, only a plume outside the plume "
              "region: a limit of plume_rows / the source time window, not of the thresholds.",
              "mismatch_study: passes where the recomputed flag differs from the flag of the study run (should be 0 or "
              "close to it; differences come from plume files whose window was extended in every combination).", "",
              "Figures: heatmaps.png, score_distribution.png, false_negatives.png (missed passes of the recommended "
              "combination), null_windows_top.png (null windows with the highest scores: if these show plumes, ships "
              "missing in the AIS data contaminate the noise sample and the false alarm rates are too high), "
              "disputed_passes.png (passes on which the combinations disagree)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
