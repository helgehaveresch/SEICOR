"""Evaluation of the detection parameter study (run_detection_study.py): which combination of p_threshold_plume and
plume_min_cluster_size detects actual plumes best, while it sorts out noise and plumes far below the detection limit.

There is no ground truth, so the evaluation uses two references that do not depend on the detection parameters:

1. Noise only (null windows). The extended enhancement image of every pass (no2_enhancement_interp_extended,
   -10..+15 min) has windows before the pre-pass background window with no ship of this pass. Windows of the size
   of the detection window that are also free of all other ship passes of the day (AIS, ship_passes csv: no pass from
   NULL_CLEAR_BEFORE_S before to NULL_CLEAR_AFTER_S after the window) contain only noise (and background structure).
   Detections there are false alarms.
2. Signal strength S of every pass, a matched-filter score of the detection window: the z-score image
   (enhancement - background level) / noise level, as in the z-test of the detection, averaged over boxes of
   BOX rows x columns; S = the largest box mean x sqrt(box size) (the significance of the strongest compact blob).
   The same score of the null windows gives its noise distribution. Passes with S above the S_HI_QUANTILE quantile of
   the null scores are "clear plumes" (true signal), passes with S below the S_LO_QUANTILE quantile are "below the
   detection limit" (indistinguishable from noise); passes in between are "marginal".

The plume detection of every combination is recomputed on exactly the input the preprocessor used (the plume window
before any extension and background correction, the background level and noise level of plumes.sort_plumes), so the
ship test (independent of these parameters) does not mask it. A check against the plume_or_ship_found flags of the
study runs is part of the report.

Per combination:
    recall_clear   fraction of clear plumes detected                              (high is good)
    rate_marginal  fraction of marginal passes detected                           (informative)
    rate_below     fraction of passes below the detection limit detected          (low is good)
    far_null       false alarm rate in the null windows                           (low is good)
The recommended combination has far_null <= MAX_FAR and rate_below <= MAX_RATE_BELOW and the highest recall_clear
(ties: lower far_null, then lower rate_below); if none passes, the best score recall_clear - far_null - rate_below.

Output in <base>/<STUDY_DIR>/evaluation/: passes.csv (one row per pass, S and the detection of every combination),
null_windows.csv, combinations.csv, report.md, figures (heatmaps, S distributions, images of the passes on which
the combinations disagree, for a visual check).

Usage: python3 evaluate_detection_study.py
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
from run_detection_study import STUDY_DIR, BASE, P_THRESHOLDS, MIN_CLUSTER_SIZES, combo_name  # noqa: E402
from run_plume_preprocessor_test import TEST_DATES  # noqa: E402

BOX = (3, 7)                 # rows x columns (1.5 s) of the matched filter: ~3 elevations x 10 s
S_HI_QUANTILE = 0.995        # clear plume: S above this quantile of the null scores
S_LO_QUANTILE = 0.90         # below the detection limit: S below this quantile of the null scores
NULL_CLEAR_BEFORE_S = 420    # null window: no ship pass from this long before its start ...
NULL_CLEAR_AFTER_S = 60      # ... to this long after its end (plumes of earlier ships drift through for minutes)
NULL_GAP_TO_REF_S = 10       # null windows end at least this long before the pre-pass background window
MAX_FAR = 0.05
MAX_RATE_BELOW = 0.05
N_EXAMPLES = 24              # passes shown in the figure of disputed passes

OUT = BASE / STUDY_DIR / "evaluation"
COMBOS = [(p, n, combo_name(p, n)) for p in P_THRESHOLDS for n in MIN_CLUSTER_SIZES]


def minutes(t, t0):
    return (pd.to_datetime(t, utc=True) - t0).total_seconds().values / 60


def coarsen_factor(date, settings):
    res = settings["processing"]["coarsen"]["time_resolution_seconds"]
    f = BASE / settings["Output"]["impact_coarsened_out_dir"] / f"IMPACT_{date}_{res:g}s.nc"
    with xr.open_dataset(f) as d:
        return float(d.attrs.get("coarsen_factor", 1.0))


def score(z):
    """Matched-filter score of a z-score image: largest BOX mean x sqrt(box size) (NaN counts as 0)."""
    z = np.nan_to_num(z, nan=0.0)
    if z.shape[0] < BOX[0] or z.shape[1] < BOX[1]:
        return np.nan
    return float(uniform_filter(z, size=BOX, mode="constant").max() * np.sqrt(BOX[0] * BOX[1]))


def detect_all(region, bg_mean, bg_std, factor):
    """Plume test of plumes.sort_plumes on region for every combination: {combo name: detected}."""
    return {name: bool(plumes.detect_plume_ztest(region, bg_mean=bg_mean, bg_std=bg_std, p_threshold=p,
                                                 min_cluster_size=impact.scale_count(n, factor)).any())
            for p, n, name in COMBOS}


def pass_input(ds, det, pms, plume_window):
    """Detection input of plumes.sort_plumes: plume-window image before extension and background correction,
    background level (median of that image), noise level (std of the reference images), rows and columns of the
    detection region. Returns None if the pass has no reference."""
    if "no2_ref" not in ds:
        return None
    t0 = pd.to_datetime(ds.attrs["t"], utc=True)
    var = ds.attrs.get("enhancement_var_used") or "no2_enhancement_interp"
    raw = ds[f"{var}_uncorrected"].values if f"{var}_uncorrected" in ds else ds[var].values
    m = minutes(ds.times_plume.values, t0)
    init = (m >= -plume_window[0] - 1e-6) & (m < plume_window[1])  # initial window (before any extension)
    rows = ds.image_row.values
    r = (rows >= det["plume_rows"][0]) & (rows < det["plume_rows"][1])
    c = init & (m * 60 >= -pms["source_seconds_before"]) & (m * 60 <= pms["source_seconds_after"])
    return dict(t0=t0, image=raw[:, init], region=raw[np.ix_(r, c)], rows=r, n_cols=int(c.sum()),
                bg_mean=float(np.nanmedian(raw[:, init])), bg_std=float(plumes.reference_image(ds).std()))


def null_windows(ds, inp, pass_times):
    """Null windows (detection-region sized) of the extended image before the pre-pass background window, free of
    all ship passes of the day; returns a list of (start time, region)."""
    var = "no2_enhancement_interp_extended" if "no2_enhancement_interp_extended" in ds else "no2_enhancement_c_back_extended"
    if var not in ds:
        return []
    t_ext = pd.to_datetime(ds.times_plume_extended.values, utc=True)
    img = ds[var].values[inp["rows"]]
    ref_start = pd.to_datetime(ds.times_ref.values[0], utc=True)
    n = inp["n_cols"]
    out, i = [], 0
    while i + n <= len(t_ext):
        ws, we = t_ext[i], t_ext[i + n - 1]
        if we > ref_start - pd.Timedelta(seconds=NULL_GAP_TO_REF_S):
            break
        clear = not ((pass_times >= ws - pd.Timedelta(seconds=NULL_CLEAR_BEFORE_S))
                     & (pass_times <= we + pd.Timedelta(seconds=NULL_CLEAR_AFTER_S))).any()
        if clear and np.isfinite(img[:, i:i + n]).mean() > 0.9:
            out.append((ws, img[:, i:i + n]))
        i += n  # non-overlapping
    return out


def load_flags():
    """plume_or_ship_found of every pass in every combination (study runs): DataFrame index (UTC_Time, MMSI)."""
    flags = {}
    for _, _, name in COMBOS:
        dfs = []
        for f in sorted(glob.glob(str(BASE / STUDY_DIR / name / "ship_passes" / "ship_passes_*.csv"))):
            d = pd.read_csv(f, index_col=0, usecols=lambda c: c in ("UTC_Time", "MMSI", "plume_file", "plume_or_ship_found"))
            dfs.append(d)
        if dfs:
            d = pd.concat(dfs)
            d.index = pd.to_datetime(d.index, utc=True)
            flags[name] = d.set_index("MMSI", append=True)
    return flags


def plume_file_of(t, mmsi, date, flags):
    """Plume file of a pass, preferably from a combination that did not flag it (stored image = detection input,
    no extension, no background correction); else from the strictest combination."""
    def was_flagged(nm):
        try:
            v = flags[nm].loc[(t, mmsi), "plume_or_ship_found"]
        except KeyError:
            return True  # day not run yet in this combination
        return str(v.iloc[0] if isinstance(v, pd.Series) else v) == "True"
    order = sorted(flags, key=lambda nm: (was_flagged(nm), nm))
    for name in order:
        hits = glob.glob(str(BASE / STUDY_DIR / name / "plumes" / f"plumes_{date}" / f"plume_*_t_{t:%Y%m%d_%H%M%S}_mmsi_{mmsi}.nc"))
        if hits:
            return Path(hits[0]), name
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load((BASE / STUDY_DIR / COMBOS[0][2] / "settings.yaml").read_text())
    proc = settings["processing"]
    det, pms = proc["detection"], proc["plume_mask"]
    plume_window = (proc["plume_window"]["before_minutes"], proc["plume_window"]["after_minutes"])
    flags = load_flags()
    missing = [nm for _, _, nm in COMBOS if nm not in flags]
    if missing:
        print(f"WARNING: no results yet for {len(missing)} combinations: {missing}")
    flagged = pd.DataFrame({nm: f.loc[~f.index.duplicated(), "plume_or_ship_found"].astype(str) == "True"
                            for nm, f in flags.items()})
    done = pd.DataFrame({nm: pd.Series(True, index=f.index[~f.index.duplicated()]) for nm, f in flags.items()})
    flagged = flagged.where(done.notna(), other=np.nan)  # NaN: day not run yet in this combination
    print(f"{len(flagged)} passes, {len(flags)} combinations with results")

    passes, nulls = [], []
    for date in TEST_DATES:
        day = flagged[flagged.index.get_level_values(0).strftime("%y%m%d") == date]
        if day.empty:
            continue
        factor = coarsen_factor(date, settings)
        pass_times = day.index.get_level_values(0)
        for (t, mmsi) in day.index:
            f, src = plume_file_of(t, mmsi, date, flags)
            if f is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ds = xr.load_dataset(f)
                inp = pass_input(ds, det, pms, plume_window)
                if inp is None or inp["region"].size == 0 or not np.isfinite(inp["bg_std"]) or inp["bg_std"] <= 0:
                    continue
                z = (inp["region"] - inp["bg_mean"]) / inp["bg_std"]
                row = dict(date=date, t=t, mmsi=mmsi, file=str(f), source_combo=src, S=score(z),
                           bg_std=inp["bg_std"], **{f"det_{k}": v for k, v in detect_all(inp["region"], inp["bg_mean"], inp["bg_std"], factor).items()},
                           **{f"flag_{nm}": day.loc[(t, mmsi), nm] for nm in day.columns})
                passes.append(row)
                for ws, region in null_windows(ds, inp, pass_times):
                    bg = float(np.nanmedian(region))
                    nulls.append(dict(date=date, pass_t=t, start=ws, S=score((region - bg) / inp["bg_std"]),
                                      **{f"det_{k}": v for k, v in detect_all(region, bg, inp["bg_std"], factor).items()}))
        print(f"{date}: {len(passes)} passes, {len(nulls)} null windows so far", flush=True)

    P, N = pd.DataFrame(passes), pd.DataFrame(nulls).drop_duplicates(subset=["date", "start"])
    P.to_csv(OUT / "passes.csv", index=False)
    N.to_csv(OUT / "null_windows.csv", index=False)
    if N.empty or P.empty:
        print("no passes or null windows: nothing to evaluate")
        return

    s_hi, s_lo = N.S.quantile(S_HI_QUANTILE), N.S.quantile(S_LO_QUANTILE)
    P["class"] = np.where(P.S >= s_hi, "clear", np.where(P.S < s_lo, "below", "marginal"))

    # consistency: a recomputed plume detection must be flagged in the study run of the same combination
    inconsistent = {nm: int((P[f"det_{nm}"] & (P[f"flag_{nm}"] == False)).sum()) for _, _, nm in COMBOS if f"flag_{nm}" in P}  # noqa: E712

    rows = []
    for p, n, nm in COMBOS:
        d, cls = P[f"det_{nm}"], P["class"]
        rows.append(dict(combo=nm, p_threshold_plume=p, plume_min_cluster_size=n,
                         recall_clear=d[cls == "clear"].mean(), rate_marginal=d[cls == "marginal"].mean(),
                         rate_below=d[cls == "below"].mean(), far_null=N[f"det_{nm}"].mean(),
                         detected=int(d.sum()), flagged_in_study=int((P[f"flag_{nm}"] == True).sum()) if f"flag_{nm}" in P else None,  # noqa: E712
                         inconsistent_with_study=inconsistent.get(nm)))
    C = pd.DataFrame(rows)
    C["score"] = C.recall_clear - C.far_null - C.rate_below
    ok = C[(C.far_null <= MAX_FAR) & (C.rate_below <= MAX_RATE_BELOW)]
    if len(ok):
        best = ok.sort_values(["recall_clear", "far_null", "rate_below"], ascending=[False, True, True]).iloc[0]
        rule = f"far_null <= {MAX_FAR} and rate_below <= {MAX_RATE_BELOW}, highest recall_clear"
    else:
        best = C.sort_values("score", ascending=False).iloc[0]
        rule = "no combination meets the limits: best recall_clear - far_null - rate_below"
    C.to_csv(OUT / "combinations.csv", index=False)

    figures(P, N, C, s_hi, s_lo, best)
    report(P, N, C, s_hi, s_lo, best, rule, missing)
    print(f"recommended: {best.combo}  (written to {OUT})")


def figures(P, N, C, s_hi, s_lo, best):
    # heatmaps over the parameter grid
    metrics = [("recall_clear", "recall of clear plumes", "magma_r"), ("far_null", "false alarm rate (noise)", "magma_r"),
               ("rate_below", "detected below the limit", "magma_r"), ("score", "recall − FAR − below", plotting.CMAP_ENH)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for ax, (m, title, cmap) in zip(axes, metrics):
        grid = C.pivot(index="p_threshold_plume", columns="plume_min_cluster_size", values=m)
        kw = dict(vmin=-1, vmax=1) if m == "score" else dict(vmin=0, vmax=1)
        im = ax.imshow(grid.values, cmap=cmap, origin="lower", aspect="auto", **kw)
        for (i, j), v in np.ndenumerate(grid.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (cmap == "magma_r" and v > 0.6) else "black")
        bi, bj = list(grid.index).index(best.p_threshold_plume), list(grid.columns).index(best.plume_min_cluster_size)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#56B4E9", lw=2.5))
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel("plume_min_cluster_size")
        ax.set_ylabel("p_threshold_plume")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "heatmaps.png", dpi=200)
    plt.close(fig)

    # distribution of S: passes vs noise
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    bins = np.linspace(min(N.S.min(), P.S.min()), np.nanpercentile(P.S, 99), 60)
    ax.hist(N.S, bins=bins, density=True, alpha=0.6, color="0.5", label=f"null windows (n={len(N)})")
    ax.hist(P.S, bins=bins, density=True, alpha=0.6, color="#D55E00", label=f"ship passes (n={len(P)})")
    for v, lab, y in ((s_lo, f"below limit < {s_lo:.1f}", 0.95), (s_hi, f"clear ≥ {s_hi:.1f}", 0.82)):
        ax.axvline(v, color="#222222", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1] * y, " " + lab, fontsize=7, va="top")
    ax.set_xlabel("matched-filter score S")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "score_distribution.png", dpi=200)
    plt.close(fig)

    # detection rate vs S for a few combinations (the strictest, the loosest, the recommended one)
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    edges = np.nanpercentile(P.S, np.linspace(0, 100, 13))
    mid = (edges[:-1] + edges[1:]) / 2
    for nm in dict.fromkeys([C.combo.iloc[0], C.combo.iloc[-1], best.combo, "p0.020_mcs10"]):
        rate = [P[f"det_{nm}"][(P.S >= a) & (P.S < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(mid, rate, marker="o", ms=3, lw=1.5 if nm == best.combo else 0.9, label=nm)
    ax.axvspan(ax.get_xlim()[0], s_lo, color="0.92", zorder=0)
    ax.axvline(s_hi, color="#222222", ls="--", lw=1)
    ax.set_xlabel("matched-filter score S")
    ax.set_ylabel("fraction detected")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "detection_vs_score.png", dpi=200)
    plt.close(fig)

    # images of disputed passes (detected by some combinations, not by all), sorted by S, for a visual check
    det_cols = [f"det_{nm}" for nm in C.combo]
    frac = P[det_cols].mean(axis=1)
    disputed = P[(frac > 0) & (frac < 1)].sort_values("S")
    if len(disputed):
        pick = disputed.iloc[np.unique(np.linspace(0, len(disputed) - 1, min(N_EXAMPLES, len(disputed))).astype(int))]
        ncol = 4
        nrow = int(np.ceil(len(pick) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow), constrained_layout=True, squeeze=False)
        for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
            ds = xr.load_dataset(r.file)
            var = ds.attrs.get("enhancement_var_used") or "no2_enhancement_interp"
            img = ds[f"{var}_uncorrected"].values if f"{var}_uncorrected" in ds else ds[var].values
            t0 = pd.to_datetime(ds.attrs["t"], utc=True)
            lim = 4 * r.bg_std
            ax.pcolormesh(minutes(ds.times_plume.values, t0), ds.vea.values, img, cmap=plotting.CMAP_ENH,
                          vmin=-lim, vmax=lim, shading="nearest", rasterized=True)
            ax.axvline(0, color="#222222", lw=0.6, ls=":")
            ax.set_title(f"{r.t:%y%m%d %H:%M} S={r.S:.1f} ({r['class']}), detected by {int(frac[r.name] * len(det_cols))}/{len(det_cols)}, "
                         f"rec.: {'yes' if r[f'det_{best.combo}'] else 'no'}", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.savefig(OUT / "disputed_passes.png", dpi=150)
        plt.close(fig)


def report(P, N, C, s_hi, s_lo, best, rule, missing):
    cls = P["class"].value_counts()
    lines = [
        "# Detection parameter study: evaluation", "",
        f"{len(P)} ship passes on {P.date.nunique()} test days, {len(N)} null windows (noise only, free of all ship passes).",
        f"Matched-filter score S (box {BOX[0]} rows x {BOX[1]} columns): clear plume S >= {s_hi:.2f} "
        f"({S_HI_QUANTILE:.1%} quantile of the null scores), below the detection limit S < {s_lo:.2f} "
        f"({S_LO_QUANTILE:.0%} quantile): {cls.get('clear', 0)} clear, {cls.get('marginal', 0)} marginal, "
        f"{cls.get('below', 0)} below the limit.", "",
        f"**Recommended: p_threshold_plume = {best.p_threshold_plume}, plume_min_cluster_size = {best.plume_min_cluster_size}** "
        f"({rule}): recall of clear plumes {best.recall_clear:.2f}, false alarm rate {best.far_null:.3f}, "
        f"detected below the limit {best.rate_below:.2f}, marginal {best.rate_marginal:.2f}.", "",
    ]
    if missing:
        lines += [f"WARNING: no results for {len(missing)} combinations: {', '.join(missing)}", ""]
    cols = ["combo", "recall_clear", "rate_marginal", "rate_below", "far_null", "score", "detected", "flagged_in_study", "inconsistent_with_study"]
    t = C[cols].copy()
    for c in ("recall_clear", "rate_marginal", "rate_below", "far_null", "score"):
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    lines += ["", "inconsistent_with_study: passes the recomputed plume test detects but the study run did not flag "
              "(should be 0; the study flags also contain ship detections, so flagged_in_study >= detected).", "",
              "Figures: heatmaps.png, score_distribution.png, detection_vs_score.png, disputed_passes.png "
              "(passes on which the combinations disagree: check visually whether they are plumes)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
