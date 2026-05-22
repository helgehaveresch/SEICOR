#!/usr/bin/env python3
"""Plot all plumes marked useful in a plume_timestamps CSV.

Reads `plume_timestamps.csv`, filters rows where `plume_useful` is True,
opens each `plume_file` with xarray and creates a NO2 time-vs-VEA plot
following the style in `random_plots.py`. PNGs are saved to the output
directory (created if missing).

Usage:
  python SEICOR/plot_useful_plumes.py \
    --csv "Q:/BREDOM/SEICOR/plume_timestamps.csv" \
    --out "/misc/dodecagon/BREDOM/SEICOR/useful_plumes_plots" \
    --max 0

Options:
  --max N    : limit to first N plumes (0 = all)
  --dry-run  : don't open .nc files, just print what would be done
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import argparse
import os


# disable HDF5 file locking on network drives (Windows)
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')


def plot_no2_from_ds(ds, out_png, label_fs=14, tick_fs=12, title_fs=16, cb_fs=12):
    # replicate essential plotting choices from random_plots.py
    arr = ds['no2_extended'].values
    # choose central range if too wide
    if arr.shape[1] > 2000:
        arr = arr[:, 500:1500]
        times = pd.to_datetime(ds['times_plume_extended'].values[500:1500])
    else:
        times = pd.to_datetime(ds['times_plume_extended'].values)

    vea = np.asarray(ds['vea'].values)

    if arr.ndim != 2:
        raise ValueError('expected 2D array for NO2')
    if arr.shape[0] == len(times) and arr.shape[1] == len(vea):
        arr = arr.T

    xnum = mdates.date2num(times.to_pydatetime())
    dx = np.median(np.diff(xnum)) if len(xnum) > 1 else 1.0
    xedges = np.concatenate((xnum - dx/2.0, [xnum[-1] + dx/2.0]))
    dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
    yedges = np.concatenate((vea - dy/2.0, [vea[-1] + dy/2.0]))

    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='viridis')

    # x ticks
    locator = mdates.AutoDateLocator()
    fmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    # VEA ticks: up to 6 ticks, one decimal
    n_yticks = min(6, len(vea))
    idxs = np.unique(np.round(np.linspace(0, len(vea)-1, n_yticks)).astype(int))
    yticks = vea[idxs]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}°" for v in yticks])
    ax.tick_params(axis='both', labelsize=tick_fs)

    ax.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax.set_ylabel('VEA (°)', fontsize=label_fs)

    # Title from attrs when available
    title = Path(out_png).stem
    try:
        tattr = ds.attrs.get('t', None)
        mmsi = ds.attrs.get('mmsi', None)
        if tattr is not None:
            import pandas as _pd
            tstr = _pd.to_datetime(tattr).strftime('%Y-%m-%d %H:%M:%S')
            title = f"{tstr}, mmsi: {mmsi}"
    except Exception:
        pass
    ax.set_title(title, fontsize=title_fs)

    # colorbar with scalar formatter and move offset into label
    fmt_cb = ScalarFormatter(useMathText=True)
    fmt_cb.set_powerlimits((-3, 3))
    cbar = fig.colorbar(pcm, ax=ax, format=fmt_cb)
    offset_text = ''
    try:
        offset_text = cbar.ax.yaxis.get_offset_text().get_text()
    except Exception:
        offset_text = ''
    label = 'NO$_2$ dSCD'
    if offset_text:
        offset_clean = offset_text.replace('\\times', '').replace('\u00d7', '').strip()
        cbar.set_label(f"{label} / {offset_clean} " + r"$\mathrm{molec}\,\mathrm{cm}^{-2}$", fontsize=cb_fs)
        try:
            cbar.ax.yaxis.get_offset_text().set_visible(False)
        except Exception:
            pass
    else:
        cbar.set_label(label, fontsize=cb_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    plt.tight_layout()
    fig.autofmt_xdate()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='plume_timestamps.csv path')
    p.add_argument('--out', required=True, help='output directory for PNGs')
    p.add_argument('--max', type=int, default=0, help='limit to first N plumes (0 = all)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    csvp = Path(args.csv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csvp, parse_dates=['UTC_Time'])
    df_useful = df[df['plume_useful'] == True]
    if args.max > 0:
        df_useful = df_useful.iloc[:args.max]

    if df_useful.empty:
        print('No useful plumes found in', csvp)
        return

    for i, row in df_useful.iterrows():
        plume_file = Path(row['plume_file'])
        out_png = outdir / (plume_file.stem + '.png')
        if args.dry_run:
            print('[DRY]', plume_file, '->', out_png)
            continue

        if not plume_file.exists():
            print('File not found, skipping:', plume_file)
            continue

        try:
            ds = xr.open_dataset(plume_file)
            plot_no2_from_ds(ds, str(out_png))
            ds.close()
            print('Wrote', out_png)
        except Exception as e:
            print('Failed to plot', plume_file, e)


if __name__ == '__main__':
    main()
