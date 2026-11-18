"""Parameter study of the plume masking (plumes.plume_mask_of): recompute the plume and ship mask of every flagged
pass of a base run with every combination of

    plume_mask/p_threshold:            0.16, 0.18, 0.2, 0.22
    plume_mask/merge_max_gap_seconds:  10, 15
    plume_mask/merge_min_cluster_size: 30, 70, 110      (raw measurements, scaled with the coarsening)
    plume_mask/merge_max_cluster_size: 200, 400         (raw measurements, scaled with the coarsening)
    plume_mask/ship_p_threshold:       0.005, 0.01, 0.02, 0.05

(192 combinations; all other settings from the settings.yaml of the base run) and evaluate the masks.

Offline: the masks are recomputed on the plume files of the base run (BASE_COMBO of the detection study, i.e. the
detection settings of that run), on the enhancement image before the background correction (as the preprocessor
masks it). Not redone per combination: the window extension (the stored window is the one the base run chose) and
the background correction.

No ground truth, so the masks are compared with reference masks that do not depend on the masking parameters, from
a matched-filter significance image: the z-score image (enhancement - median of the window) / noise level (std of
the reference images, as in the masking), averaged over boxes of BOX rows x columns, times sqrt(box size):
    reference plume: significance >= Z_REF, connected components that touch the source region
    reference ship:  significance <= -Z_REF in the ship rows within +-time_tolerance_seconds of the pass
Per pass and combination: IoU, F1, recall (fraction of the reference in the mask), precision (fraction of the mask
on the reference grown by DILATE pixels), captured signal (sum of the enhancement in mask and reference / in the
reference), noise pixels (fraction of the mask with significance < 1), ship pixels in the plume mask (fraction of
the reference ship); for the ship mask IoU and F1. Passes with an empty reference plume count for the false mask
rate (non-empty plume mask without significant plume).

Output in <base>/<STUDY_DIR>/mask_study/: per_pass.csv, combinations.csv, report.md, parameter_effects.png,
examples.png.

Usage (background, detached; argument: output folder name in the study folder, default mask_study):
    setsid nohup python3 mask_study.py mask_study_ship_window > mask_study_$(date +%y%m%d_%H%M).log 2>&1 &
Plots with plume and ship mask of every pass for PLOT_COMBO: <output folder>/plots/plume_mask/plumes_<date>/; with
the option --plots-all for every combination: <output folder>/plots_all/<combination>/plume_mask/plumes_<date>/.
"""
import copy
import glob
import itertools
import multiprocessing
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
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from SEICOR import plumes, impact, plotting  # noqa: E402
from run_detection_study import STUDY_DIR, BASE, ROUNDS  # noqa: E402

BASE_COMBO = "p0.020_mcs80_ps0.005_smcs60"  # detection settings = plume_preprocessor_settings.yaml of 261117
DATES = ROUNDS["round3"]["dates"]
GRID = dict(p_threshold=[0.16, 0.18, 0.2, 0.22], merge_max_gap_seconds=[10, 15],
            merge_min_cluster_size=[30, 70, 110], merge_max_cluster_size=[200, 400], ship_p_threshold=[0.005, 0.01, 0.02, 0.05])
BOX = (3, 7)        # rows x columns (1.5 s) of the significance image
Z_REF = 4.0         # reference masks: |significance| >= Z_REF
DILATE = 1          # precision: mask pixels within this many pixels of the reference count as correct
PROCESSES = 20

STUDY = BASE / STUDY_DIR
OUT = STUDY / ([a for a in sys.argv[1:] if not a.startswith("--")] or ["mask_study"])[0]  # output folder name (first argument)
PLOT_ALL = "--plots-all" in sys.argv  # plots of every pass for every combination in OUT/plots_all/<combination>/
# combinations already plotted by an earlier --plots-all run are not plotted again (delete their folder to replot)
PLOTTED = {d.name for d in (OUT / "plots_all").glob("*") if d.is_dir()} if PLOT_ALL else set()
# plots with plume and ship mask of every pass for these settings (current plume_mask settings), in OUT/plots
PLOT_COMBO = dict(p_threshold=0.18, merge_max_gap_seconds=15, merge_min_cluster_size=70, merge_max_cluster_size=200,
                  ship_p_threshold=0.05)
COMBOS = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def combo_name(c):
    return (f"p{c['p_threshold']:g}_gap{c['merge_max_gap_seconds']}_mmin{c['merge_min_cluster_size']}"
            f"_mmax{c['merge_max_cluster_size']}_ps{c['ship_p_threshold']:g}")


def significance(img, bg_mean, bg_std):
    z = np.nan_to_num((img - bg_mean) / bg_std, nan=0.0)
    return ndimage.uniform_filter(z, size=BOX, mode="nearest") * np.sqrt(BOX[0] * BOX[1])


def reference_masks(ds, img, sig, settings):
    """Reference plume (significance >= Z_REF, components touching the source region) and reference ship."""
    pms, det = settings["processing"]["plume_mask"], settings["processing"]["detection"]
    t0 = pd.to_datetime(ds.attrs["t"], utc=True)
    s = (pd.to_datetime(ds.times_plume.values, utc=True) - t0).total_seconds().values
    rows = ds.image_row.values
    src = (((rows >= pms["source_rows"][0]) & (rows <= pms["source_rows"][1]))[:, None]
           & ((s >= -pms["source_seconds_before"]) & (s <= pms["source_seconds_after"]))[None, :])
    lab, n = ndimage.label(sig >= Z_REF)
    keep = np.unique(lab[src & (lab > 0)])
    ref_plume = np.isin(lab, keep) & (lab > 0)
    ship_region = (((rows >= det["ship_rows"][0]) & (rows < det["ship_rows"][1]))[:, None]
                   & (np.abs(s) <= det["time_tolerance_seconds"])[None, :])
    ref_ship = (sig <= -Z_REF) & ship_region
    return ref_plume, ref_ship


def scores(mask, ref, grown):
    inter = (mask & ref).sum()
    union = (mask | ref).sum()
    recall = inter / ref.sum() if ref.sum() else np.nan
    precision = (mask & grown).sum() / mask.sum() if mask.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return inter / union if union else np.nan, f1, recall, precision


def evaluate_pass(job):
    """All combinations for one plume file: list of dicts (one per combination)."""
    f, settings, factor = job
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.load_dataset(f)
        if "no2_ref" not in ds:
            return []
        var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
        if f"{var}_uncorrected" in ds:  # mask the image before the background correction, as the preprocessor does
            ds[var] = ds[f"{var}_uncorrected"]
        img = ds[var].values
        bg_mean, bg_std = float(np.nanmedian(img)), float(plumes.reference_image(ds).std())
        sig = significance(img, bg_mean, bg_std)
        ref_plume, ref_ship = reference_masks(ds, img, sig, settings)
        grown_plume = ndimage.binary_dilation(ref_plume, iterations=DILATE) if ref_plume.any() else ref_plume
        grown_ship = ndimage.binary_dilation(ref_ship, iterations=DILATE) if ref_ship.any() else ref_ship
        ref_signal = np.nansum(np.where(ref_plume, img - bg_mean, 0.0))
        base = dict(file=str(f), t=ds.attrs["t"], mmsi=ds.attrs.get("mmsi"), ref_plume_px=int(ref_plume.sum()),
                    ref_ship_px=int(ref_ship.sum()))
        out = []
        for c in COMBOS:
            raw = copy.deepcopy(settings["processing"])
            raw["plume_mask"].update(c)
            pms = impact.scale_sample_counts(raw, factor)["plume_mask"]
            try:
                mask, ship = plumes.plume_mask_of(ds, pms)
            except Exception as e:  # a failing combination must not stop the study
                out.append(dict(base, combo=combo_name(c), **c, error=str(e)[:200]))
                continue
            if c == PLOT_COMBO or (PLOT_ALL and combo_name(c) not in PLOTTED):
                try:
                    plotting.plot_no2_enhancement_with_plume_mask(
                        ds, mask, OUT / "plots_all" / combo_name(c) if PLOT_ALL else OUT / "plots",
                        Path(f).parent.name[-6:], ship_mask=ship)
                except Exception as e:
                    print(f"plot of {f} failed: {e}", flush=True)
            iou, f1, rec, prec = scores(mask, ref_plume, grown_plume)
            s_iou, s_f1, s_rec, s_prec = scores(ship, ref_ship, grown_ship)
            out.append(dict(
                base, combo=combo_name(c), **c, mask_px=int(mask.sum()), ship_px=int(ship.sum()),
                iou=iou, f1=f1, recall=rec, precision=prec,
                signal=np.nansum(np.where(mask & ref_plume, img - bg_mean, 0.0)) / ref_signal if ref_signal > 0 else np.nan,
                noise_px=float((sig[mask] < 1).mean()) if mask.any() else np.nan,
                ship_in_plume=float((mask & ref_ship).sum() / ref_ship.sum()) if ref_ship.any() else np.nan,
                ship_iou=s_iou, ship_f1=s_f1, ship_recall=s_rec, ship_precision=s_prec,
                false_mask=bool(mask.any()) if not ref_plume.any() else np.nan))
        return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load((STUDY / BASE_COMBO / "settings.yaml").read_text())
    res = settings["processing"]["coarsen"]["time_resolution_seconds"]
    jobs = []
    for date in DATES:
        with xr.open_dataset(STUDY / "IMPACT_coarsened" / f"IMPACT_{date}_{res:g}s.nc") as d:
            factor = float(d.attrs.get("coarsen_factor", 1.0))
        passes = pd.read_csv(STUDY / BASE_COMBO / "ship_passes" / f"ship_passes_{date}.csv", index_col=0)
        for t, mmsi in passes.loc[passes.plume_or_ship_found.astype(str) == "True", "MMSI"].items():
            t = pd.to_datetime(t, utc=True)
            hits = glob.glob(str(STUDY / BASE_COMBO / "plumes" / f"plumes_{date}" / f"plume_*_t_{t:%Y%m%d_%H%M%S}_mmsi_{mmsi}.nc"))
            if hits:
                jobs.append((hits[0], settings, factor))
    print(f"{len(jobs)} flagged passes on {len(DATES)} days, {len(COMBOS)} combinations", flush=True)
    rows = []
    with multiprocessing.Pool(PROCESSES) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate_pass, jobs, chunksize=2)):
            rows += r
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} passes", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "per_pass.csv", index=False)
    if "error" in P and P.error.notna().any():
        print(f"WARNING: {P.error.notna().sum()} failed mask computations", flush=True)
        P = P[P.error.isna()]

    has_plume, has_ship = P.ref_plume_px > 0, P.ref_ship_px > 0
    agg = lambda d, col: d[col].mean()  # noqa: E731
    C = []
    for name, d in P.groupby("combo"):
        dp, ds_ = d[d.ref_plume_px > 0], d[d.ref_ship_px > 0]
        C.append(dict(combo=name, **{k: d[k].iloc[0] for k in GRID},
                      plume_f1=agg(dp, "f1"), plume_iou=agg(dp, "iou"), recall=agg(dp, "recall"),
                      precision=dp.precision.mean(), signal=agg(dp, "signal"), noise_px=d.noise_px.mean(),
                      ship_in_plume=ds_.ship_in_plume.mean(), ship_f1=agg(ds_, "ship_f1"),
                      false_mask=d.false_mask.mean(), mask_px=d.mask_px.mean()))
    C = pd.DataFrame(C)
    C["score"] = C.plume_f1 + 0.5 * C.ship_f1 - C.false_mask.fillna(0) - C.ship_in_plume.fillna(0)
    C = C.sort_values("score", ascending=False)
    C.to_csv(OUT / "combinations.csv", index=False)
    best = C.iloc[0]
    print(f"best: {best.combo}", flush=True)
    figures(P, C, best)
    report(P, C, best, int(has_plume.groupby(P.file).first().sum()), int(has_ship.groupby(P.file).first().sum()))


def figures(P, C, best):
    # effect of every parameter: distribution of the combination scores per value
    metrics = [("plume_f1", "F1 plume mask"), ("ship_f1", "F1 ship mask"), ("false_mask", "false mask rate"),
               ("ship_in_plume", "ship pixels in plume mask"), ("noise_px", "mask pixels without signal")]
    fig, axes = plt.subplots(len(metrics), len(GRID), figsize=(3 * len(GRID), 2.2 * len(metrics)),
                             constrained_layout=True, squeeze=False)
    for i, (m, title) in enumerate(metrics):
        for j, k in enumerate(GRID):
            vals = sorted(C[k].unique())
            axes[i, j].boxplot([C.loc[C[k] == v, m].dropna() for v in vals], tick_labels=[f"{v:g}" for v in vals])
            axes[i, j].axhline(best[m], color=plotting.C_SHIP, lw=1, ls="--")
            if i == len(metrics) - 1:
                axes[i, j].set_xlabel(k, fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(title, fontsize=8)
            axes[i, j].tick_params(labelsize=7)
    fig.suptitle(f"each box: the combinations with that value; dashed: best ({best.combo})", fontsize=9)
    fig.savefig(OUT / "parameter_effects.png", dpi=160)
    plt.close(fig)

    # examples: passes where the best and the current settings (0.18, 15, 70, 200, 0.05) differ most
    cur = combo_name(dict(p_threshold=0.18, merge_max_gap_seconds=15, merge_min_cluster_size=70,
                          merge_max_cluster_size=200, ship_p_threshold=0.05))
    a = P[P.combo == best.combo].set_index("file")
    b = P[P.combo == cur].set_index("file")
    diff = (a.mask_px - b.mask_px).abs().sort_values(ascending=False).head(8)
    if len(diff) == 0:
        return
    fig, axes = plt.subplots(len(diff), 2, figsize=(11, 2.4 * len(diff)), constrained_layout=True, squeeze=False)
    settings = yaml.safe_load((STUDY / BASE_COMBO / "settings.yaml").read_text())
    for row, f in zip(axes, diff.index):
        ds = xr.load_dataset(f)
        var = "no2_enhancement_interp" if "no2_enhancement_interp" in ds else "no2_enhancement_c_back"
        if f"{var}_uncorrected" in ds:
            ds[var] = ds[f"{var}_uncorrected"]
        res = settings["processing"]["coarsen"]["time_resolution_seconds"]
        date = Path(f).parent.name[-6:]
        with xr.open_dataset(STUDY / "IMPACT_coarsened" / f"IMPACT_{date}_{res:g}s.nc") as d:
            factor = float(d.attrs.get("coarsen_factor", 1.0))
        t0 = pd.to_datetime(ds.attrs["t"], utc=True)
        x = (pd.to_datetime(ds.times_plume.values, utc=True) - t0).total_seconds().values / 60
        img = ds[var].values
        lim = 4 * float(plumes.reference_image(ds).std())
        for ax, name in zip(row, (cur, best.combo)):
            c = next(c for c in COMBOS if combo_name(c) == name)
            raw = copy.deepcopy(settings["processing"])
            raw["plume_mask"].update(c)
            mask, ship = plumes.plume_mask_of(ds, impact.scale_sample_counts(raw, factor)["plume_mask"])
            ax.pcolormesh(x, ds.vea.values, img, cmap=plotting.CMAP_ENH, vmin=-lim, vmax=lim, shading="nearest", rasterized=True)
            for m, col in ((mask, plotting.C_PLUME), (ship, plotting.C_SHIP)):
                if m.any():
                    ax.contour(x, ds.vea.values, m.astype(int), levels=[0.5], colors=col, linewidths=1.0)
            ax.set_title(f"{t0:%y%m%d %H:%M} {'current' if name == cur else 'best'}: {name}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.savefig(OUT / "examples.png", dpi=140)
    plt.close(fig)


def report(P, C, best, n_plume, n_ship):
    n = P.file.nunique()
    lines = ["# Plume masking study", "",
             f"{n} flagged passes of {BASE_COMBO} on {len(DATES)} days, {len(COMBOS)} combinations. Reference "
             f"masks (significance of {BOX[0]} rows x {BOX[1]} columns boxes, |z| >= {Z_REF}): {n_plume} passes with a "
             f"reference plume, {n_ship} with a reference ship.", "",
             f"**Best: {best.combo}** (score = F1 plume + 0.5 F1 ship - false mask rate - ship pixels in plume mask): "
             f"F1 plume {best.plume_f1:.3f} (recall {best.recall:.3f}, precision {best.precision:.3f}, signal "
             f"{best.signal:.3f}), F1 ship {best.ship_f1:.3f}, false mask rate {best.false_mask:.3f}, ship pixels in "
             f"plume mask {best.ship_in_plume:.3f}.", ""]
    cols = ["combo", "score", "plume_f1", "recall", "precision", "signal", "noise_px", "ship_f1", "ship_in_plume",
            "false_mask", "mask_px"]
    t = C[cols].head(25).copy()
    for c in cols[1:]:
        t[c] = t[c].map(lambda v: f"{v:.3f}" if c != "mask_px" else f"{v:.0f}")
    lines += ["Top 25 of 144 (all in combinations.csv):", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    lines += ["", "Offline study: masks recomputed on the plume files of the base run; window extension and background "
              "correction not redone per combination. Figures: parameter_effects.png, examples.png (current settings vs "
              "best, passes with the largest difference)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
