#%%
#!/usr/bin/env python3
"""Interactive plume labeling tool that reads plume .nc files recursively.

This script searches the given root directory (recursively) for NetCDF files.
For each file with attribute `plume_or_ship_found` == "True" it plots
`no2_enhancement_interp` and displays a small GUI allowing the user to
click a point on the image and press YES/NO. YES sets `plume_useful`="True",
NO sets it to "False". The clicked coordinate is stored as
`plume_point_x` and `plume_point_y` attributes on the netCDF file.
"""
from pathlib import Path, PurePath
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import netCDF4
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import xarray as xr
import sys
#%%

class LabelGUI:
    def __init__(self, data_array: np.ndarray, plume_file: Path, title=None, cmap='viridis'):
        self.plume_file = Path(plume_file)
        self.point = None
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.15)

        # Display the 2D data array; origin='lower' matches common plotting for these datasets
        self.img = self.ax.imshow(data_array, origin='lower', aspect='auto', cmap=cmap)
        self.ax.set_title(title or f"{self.plume_file.name}")
        # instruction above the image
        try:
            self.fig.suptitle('mark the plume source in the image', fontsize=12, y=0.98)
        except Exception:
            pass
        self.fig.colorbar(self.img, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)

        # Buttons
        ax_yes = plt.axes([0.7, 0.03, 0.1, 0.06])
        ax_no = plt.axes([0.82, 0.03, 0.1, 0.06])
        self.btn_yes = Button(ax_yes, 'YES')
        self.btn_no = Button(ax_no, 'NO')
        # place a small question label to the left of the buttons
        self.fig.text(0.68, 0.06, 'Plume useful?', fontsize=10, ha='right', va='center')
        self.btn_yes.on_clicked(self.on_yes)
        self.btn_no.on_clicked(self.on_no)

        # connect click
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        # ignore clicks outside the main axes
        if event.inaxes != self.ax:
            return
        # xdata corresponds to column index, ydata to row index because of imshow
        self.point = (float(event.xdata), float(event.ydata))
        # draw marker
        self.ax.plot(self.point[0], self.point[1], 'ro')
        self.fig.canvas.draw()

    def write_attributes(self, useful: bool):
        # keep result in-memory; main() will persist using xarray
        self._last_choice = bool(useful)
        # point is already stored in self.point (or None)
        return

    def on_yes(self, event):
        self.write_attributes(True)
        plt.close(self.fig)

    def on_no(self, event):
        self.write_attributes(False)
        plt.close(self.fig)

    def get_result(self):
        # returns None if user closed window without choosing
        if not hasattr(self, '_last_choice'):
            return None
        return {'useful': self._last_choice, 'point': self.point}


def find_plume_nc_files(root_dir: Path):
    # recursively find .nc files under root_dir
    files = [p for p in sorted(Path(root_dir).rglob('*.nc'))]
    return files


def update_ship_pass_csvs(ship_pass_dir: Path, plume_root: Path, propagate_nc_to_csv: bool = False):
    """Ensure all ship-pass CSVs have a 'plume_useful' column (False by default)
    and set it to True when the corresponding .nc file has plume_useful == 'True'.
    """
    ship_pass_dir = Path(ship_pass_dir)
    plume_root = Path(plume_root)
    if not ship_pass_dir.exists():
        print('ship_pass_dir does not exist, skipping CSV sync:', ship_pass_dir)
        return

    csvs = sorted(ship_pass_dir.glob('*.csv'))
    for csvf in csvs:
        try:
            df = pd.read_csv(csvf)
        except Exception as e:
            print(f'Failed to read CSV {csvf}: {e}')
            continue

        # Add column if missing (default False)
        if 'plume_useful' not in df.columns:
            df['plume_useful'] = False

        # Iterate rows and update based on .nc attr
        for idx, row in df.iterrows():
            date= pd.to_datetime(row.UTC_Time).strftime('%y%m%d')
            stored = row.get('plume_file', None)
            if stored is None or (isinstance(stored, float) and np.isnan(stored)):
                continue

            # resolve candidate path: if absolute use it, else try plume_root subfolders
            p = Path(stored).name
            # try direct under plume_root, else search for matching filename
            candidate = plume_root / f"plumes_{date}" / p
            if not candidate.exists():
                # search recursively for file name
                matches = list(plume_root.rglob(p))
                candidate = matches[0] if matches else None


            if candidate is None:
                continue

            try:
                ds = xr.open_dataset(candidate)
                # read current dataset attribute (treat strings/booleans)
                val = ds.attrs.get('plume_useful', 'False')
                ds.close()
                is_useful_ds = (str(val) == 'True') or (val is True)

                # read CSV value and normalise to boolean-like
                csv_val = row.get('plume_useful', False)
                try:
                    csv_is_useful = bool(csv_val) if isinstance(csv_val, (bool, np.bool_)) else str(csv_val).strip().lower() in ('true', '1', 't', 'yes')
                except Exception:
                    csv_is_useful = False

                # Propagation rules:
                # - If `propagate_nc_to_csv` is True, propagate True from .nc -> CSV.
                # - If the .nc says False but CSV says True, promote the CSV decision and set the .nc to True.
                if is_useful_ds and not csv_is_useful:
                    if propagate_nc_to_csv:
                        df.at[idx, 'plume_useful'] = True
                elif (not is_useful_ds) and csv_is_useful:
                    # CSV indicates useful but .nc does not — update the .nc to match CSV
                    try:
                        ds2_orig = xr.open_dataset(candidate)
                        ds2 = ds2_orig.load()
                        ds2_orig.close()
                        ds2.attrs['plume_useful'] = 'True'
                        ds2.to_netcdf(candidate, mode='w')
                        ds2.close()
                        df.at[idx, 'plume_useful'] = True
                        print(f'Updated {candidate}: set plume_useful=True from CSV')
                    except Exception:
                        print(f'Failed to write plume_useful=True to {candidate}')
            except Exception:
                # ignore files we cannot open
                continue

        # write CSV back (preserve index false)
        try:
            df.to_csv(csvf, index = False)
            print(f'Updated CSV: {csvf}')
        except Exception as e:
            print(f'Failed to write CSV {csvf}: {e}')


def count_ship_passes(ship_pass_dir: Path):
    """Count total ship_pass rows and number of useful plumes (plume_useful==True) across CSVs."""
    ship_pass_dir = Path(ship_pass_dir)
    if not ship_pass_dir.exists():
        print('ship_pass_dir does not exist, cannot count:', ship_pass_dir)
        return

    total_rows = 0
    total_useful = 0
    total_found = 0
    csvs = sorted(ship_pass_dir.glob('*.csv'))
    for csvf in csvs:
        try:
            df = pd.read_csv(csvf)
        except Exception as e:
            print(f'Failed to read CSV {csvf}: {e}')
            continue
        n = len(df)
        total_rows += n
        if 'plume_useful' in df.columns:
            col = df['plume_useful']
            if pd.api.types.is_bool_dtype(col) or col.dtype == bool:
                total_useful += int(col.sum())
            else:
                total_useful += int(col.astype(str).str.lower().isin(['true', '1', 't', 'yes']).sum())
        # also count plume_or_ship_found entries
        if 'plume_or_ship_found' in df.columns:
            colf = df['plume_or_ship_found']
            try:
                if pd.api.types.is_bool_dtype(colf) or colf.dtype == bool:
                    total_found += int(colf.sum())
                else:
                    total_found += int(colf.astype(str).str.strip().str.lower().isin(['true', '1', 't', 'yes']).sum())
            except Exception:
                # fallback: treat non-parseable as 0
                pass

    print(f'Total ship_pass rows (CSV rows): {total_rows}')
    print(f'Total plume_useful == True: {total_useful}')
    print(f'Total plume_or_ship_found == True: {total_found}')


def export_timestamps(ship_pass_dir: Path, out_csv: Path):
    """Collect `UTC_Time`, `plume_useful` and `mmsi` from all ship-pass CSVs and write to out_csv.

    The output CSV will have columns: `UTC_Time`, `plume_useful`, `plume_file`, `mmsi`, `source_csv`.
    """
    ship_pass_dir = Path(ship_pass_dir)
    out_csv = Path(out_csv)
    if not ship_pass_dir.exists():
        print('ship_pass_dir does not exist, cannot export timestamps:', ship_pass_dir)
        return

    rows = []
    csvs = sorted(ship_pass_dir.glob('*.csv'))
    for csvf in csvs:
        try:
            df = pd.read_csv(csvf)
        except Exception as e:
            print(f'Failed to read CSV {csvf}: {e}')
            continue

        if 'UTC_Time' not in df.columns:
            # nothing to export from this CSV
            continue

        for _, r in df.iterrows():
            utc = r.get('UTC_Time', None)
            if pd.isna(utc):
                continue
            pu = r.get('plume_useful', False)
            # normalize boolean-like
            try:
                if isinstance(pu, bool):
                    pu_val = pu
                else:
                    pu_val = str(pu).strip().lower() in ('true', '1', 't', 'yes')
            except Exception:
                pu_val = False

            pf = r.get('plume_file', None)
            mmsi = r.get('MMSI', None)
            rows.append({'UTC_Time': utc, 'plume_useful': pu_val, 'plume_file': pf, 'mmsi': mmsi, 'source_csv': str(csvf)})

    try:
        out_df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f'Wrote {len(rows)} timestamp rows to {out_csv}')
    except Exception as e:
        print(f'Failed to write export CSV {out_csv}: {e}')


def count_duplicate_mmsi(ship_pass_dir: Path):
    """Count MMSI occurrences among rows where plume_useful == True and print duplicates."""
    ship_pass_dir = Path(ship_pass_dir)
    if not ship_pass_dir.exists():
        print('ship_pass_dir does not exist, cannot count duplicates:', ship_pass_dir)
        return

    mmsi_counts = {}
    csvs = sorted(ship_pass_dir.glob('*.csv'))
    for csvf in csvs:
        try:
            df = pd.read_csv(csvf)
        except Exception as e:
            print(f'Failed to read CSV {csvf}: {e}')
            continue

        if 'plume_useful' not in df.columns:
            continue

        # select rows marked useful
        useful = df[df['plume_useful'].astype(str).str.lower().isin(['true','1','t','yes'])]
        if useful.empty:
            continue

        if 'MMSI' in useful.columns:
            for val in useful['MMSI'].dropna():
                try:
                    key = str(int(val))
                except Exception:
                    key = str(val)
                mmsi_counts[key] = mmsi_counts.get(key, 0) + 1
    if not mmsi_counts:
        print('No MMSI entries found in plume_useful==True rows.')
        return

    # find duplicates (count > 1)
    duplicates = {k: v for k, v in mmsi_counts.items() if v > 1}
    total_mmsi = len(mmsi_counts)
    total_duplicates = len(duplicates)
    print(f'Total distinct MMSI in plume_useful==True: {total_mmsi}')
    print(f'Number of MMSI appearing more than once: {total_duplicates}')
    if total_duplicates:
        print('Top duplicated MMSI (count):')
        for k, v in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:50]:
            print(f'  {k}: {v}')


def resolve_conflicts_within_window(ship_pass_dir: Path, plume_root: Path, window_seconds: int = 120):
    """Find plume_useful==True entries across CSVs that are within `window_seconds` and resolve interactively.

    For each close pair, plot both .nc images side-by-side, highlight the plume timestamp (from attrs `t`/times_plume),
    and let the user click on the plot of the plume to keep. The other entry's `plume_useful` will be set to False
    in its CSV file.
    """
    ship_pass_dir = Path(ship_pass_dir)
    plume_root = Path(plume_root)
    if not ship_pass_dir.exists():
        print('ship_pass_dir does not exist, cannot resolve conflicts:', ship_pass_dir)
        return

    entries = []
    csvs = sorted(ship_pass_dir.glob('*.csv'))
    for csvf in csvs:
        try:
            df = pd.read_csv(csvf)
        except Exception as e:
            print(f'Failed to read CSV {csvf}: {e}')
            continue

        for idx, row in df.iterrows():
            pu = row.get('plume_useful', False)
            try:
                is_useful = bool(pu) if isinstance(pu, (bool, np.bool_)) else str(pu).strip().lower() in ('true', '1', 't', 'yes')
            except Exception:
                is_useful = False
            if not is_useful:
                continue

            utc = row.get('UTC_Time', None)
            try:
                t = pd.to_datetime(utc, utc=True)
            except Exception:
                # try to skip if cannot parse
                continue

            stored = row.get('plume_file', None)
            if stored is None or (isinstance(stored, float) and np.isnan(stored)):
                continue

            entries.append({'time': t, 'plume_file': stored, 'csv': csvf, 'row_index': idx})

    if not entries:
        print('No useful plume entries found in CSVs.')
        return
    # sort by time
    entries.sort(key=lambda x: x['time'])
    # iterate and find adjacent pairs within window_seconds
    i = 0
    while i < len(entries) - 1:
        a = entries[i]
        b = entries[i+1]
        if (b['time'] - a['time']).total_seconds() <= window_seconds:
            print ('Found close pair:', a['time'], b['time'])
            # confirm both still marked useful in their CSVs (reload file)

            try:
                df_a = pd.read_csv(a['csv'])
                val_a = df_a.at[a['row_index'], 'plume_useful'] if 'plume_useful' in df_a.columns else False
                is_a = bool(val_a) if isinstance(val_a, (bool, np.bool_)) else str(val_a).strip().lower() in ('true','1','t','yes')
            except Exception:
                is_a = False
            try:
                df_b = pd.read_csv(b['csv'])
                val_b = df_b.at[b['row_index'], 'plume_useful'] if 'plume_useful' in df_b.columns else False
                is_b = bool(val_b) if isinstance(val_b, (bool, np.bool_)) else str(val_b).strip().lower() in ('true','1','t','yes')
            except Exception:
                is_b = False

            if not (is_a and is_b):
                i += 1
                continue

            # resolve this pair interactively
            print(f'Conflict: {a["plume_file"]} ({a["time"].isoformat()}) and {b["plume_file"]} ({b["time"].isoformat()})')
            # resolve paths to .nc files
            def resolve_path(stored):
                # only use the filename and combine with plume_root and date folder
                filename = Path(stored).name
                try:
                    date = pd.to_datetime(a['time']).strftime('%y%m%d')
                    cand = plume_root / f'plumes_{date}' / filename
                    if cand.exists():
                        return cand
                except Exception:
                    pass
                return None

            p_a = resolve_path(a['plume_file'])
            p_b = resolve_path(b['plume_file'])
            if p_a is None or p_b is None:
                print('Could not locate one or both .nc files for the pair; skipping.')
                i += 1
                continue

            keep = interactive_choose_between_ncs(p_a, p_b)
            if keep is None:
                print('User cancelled selection; leaving both entries unchanged.')
                i += 1
                continue

            # handle 'both' choice (keep both) or keep one path
            if keep == 'both':
                print('User chose to keep both plumes; leaving both marked useful.')
                i += 2
                continue

            # chosen to keep one path; set the other to False in its CSV
            if Path(keep) == Path(p_a):
                other = b
                other_csv = b['csv']
                other_idx = b['row_index']
            else:
                other = a
                other_csv = a['csv']
                other_idx = a['row_index']

            try:
                df_other = pd.read_csv(other_csv)
                if 'plume_useful' not in df_other.columns:
                    df_other['plume_useful'] = False
                df_other.at[other_idx, 'plume_useful'] = False
                df_other.to_csv(other_csv, index=False)
                print(f'Set plume_useful=False for {other["plume_file"]} in {other_csv}')
            except Exception as e:
                print(f'Failed to update CSV {other_csv}: {e}')

            # advance past the pair
            i += 2
        else:
            i += 1
            print('No conflict between:', a['time'], b['time'])


def interactive_choose_between_ncs(nc1: Path, nc2: Path):
    """Plot two .nc files side-by-side, highlight their plume timestamp column, and allow click to choose which to keep.
    Returns path of kept .nc (string) or None if cancelled.
    """
    try:
        ds1 = xr.open_dataset(nc1)
        ds2 = xr.open_dataset(nc2)
        ds1l = ds1.load(); ds2l = ds2.load()
    except Exception as e:
        print(f'Failed to open datasets: {e}')
        try:
            ds1.close()
            ds2.close()
        except Exception:
            pass
        return None

    try:
        arr1 = ds1l['no2_enhancement_interp'].values if 'no2_enhancement_interp' in ds1l.variables else None
        arr2 = ds2l['no2_enhancement_interp'].values if 'no2_enhancement_interp' in ds2l.variables else None
    except Exception:
        arr1 = None; arr2 = None

    if arr1 is None or arr2 is None:
        print('One of the datasets lacks no2_enhancement_interp; cannot compare interactively.')
        ds1.close(); ds2.close()
        return None

    # compute timestamp column indices
    def find_time_index(ds):
        try:
            times = pd.to_datetime(ds['times_plume'].values, utc=True)
            t_attr = ds.attrs.get('t', None)
            if t_attr is None:
                return None
            t0 = pd.to_datetime(t_attr, utc=True)
            diffs = np.abs((times - t0) / np.timedelta64(1, 's'))
            return int(np.nanargmin(diffs))
        except Exception:
            return None

    idx1 = find_time_index(ds1l)
    idx2 = find_time_index(ds2l)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(arr1, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title(nc1.name)
    if idx1 is not None:
        ax1.axvline(idx1, color='red')
    # existing click marker
    pxx = ds1l.attrs.get('plume_point_x', None)
    pyy = ds1l.attrs.get('plume_point_y', None)
    if pxx is not None and not (isinstance(pxx, float) and np.isnan(pxx)) and pyy is not None and not (isinstance(pyy, float) and np.isnan(pyy)):
        ax1.plot(int(pxx), int(pyy), 'ro')

    ax2.imshow(arr2, origin='lower', aspect='auto', cmap='viridis')
    ax2.set_title(nc2.name)
    if idx2 is not None:
        ax2.axvline(idx2, color='red')
    pxx = ds2l.attrs.get('plume_point_x', None)
    pyy = ds2l.attrs.get('plume_point_y', None)
    if pxx is not None and not (isinstance(pxx, float) and np.isnan(pxx)) and pyy is not None and not (isinstance(pyy, float) and np.isnan(pyy)):
        ax2.plot(int(pxx), int(pyy), 'ro')

    plt.suptitle('Click on the plot of the plume to KEEP (close window to cancel)')

    choice = {'keep': None}

    def onclick(event):
        if event.inaxes == ax1:
            choice['keep'] = str(nc1)
            plt.close(fig)
        elif event.inaxes == ax2:
            choice['keep'] = str(nc2)
            plt.close(fig)

    # Keep Both button
    ax_both = plt.axes([0.44, 0.01, 0.12, 0.05])
    btn_both = Button(ax_both, 'Keep Both')

    def on_keep_both(event):
        choice['keep'] = 'both'
        plt.close(fig)

    btn_both.on_clicked(on_keep_both)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    try:
        ds1.close(); ds2.close()
    except Exception:
        pass
    return choice['keep']


def reconcile_plume_or_ship_found(dir_a: Path, dir_b: Path, plume_root: Path):
    """Compare ship-pass CSVs in two directories and interactively label differing plume_or_ship_found entries.

    For rows that appear in both directories (matched by `plume_file` filename and `UTC_Time`) but have
    different `plume_or_ship_found` boolean values, resolve the corresponding .nc under `plume_root/plumes_<YYMMDD>`
    and open it with the `LabelGUI` so the user can set `plume_useful`. The chosen value is written into the .nc file.
    """
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)
    plume_root = Path(plume_root)
    if not dir_a.exists() or not dir_b.exists():
        print('One of the provided ship-pass dirs does not exist:', dir_a, dir_b)
        return

    def read_all_csvs(d):
        rows = []
        for csvf in sorted(Path(d).glob('*.csv')):
            try:
                df = pd.read_csv(csvf, dtype=str)
            except Exception:
                continue
            if 'plume_file' not in df.columns or 'UTC_Time' not in df.columns or 'plume_or_ship_found' not in df.columns:
                continue
            for _, r in df.iterrows():
                pf = r.get('plume_file')
                if pd.isna(pf) or pf is None:
                    continue
                rows.append({'plume_file': Path(str(pf)).name, 'UTC_Time': r.get('UTC_Time'), 'plume_or_ship_found': r.get('plume_or_ship_found'), 'source_csv': str(csvf)})
        return pd.DataFrame(rows)

    df_a = read_all_csvs(dir_a)
    df_b = read_all_csvs(dir_b)
    if df_a.empty or df_b.empty:
        print('No comparable rows found in one or both dirs')
        return

    # normalize booleans
    def norm_bool_series(s):
        return s.fillna('').astype(str).str.strip().str.lower().isin(['true','1','t','yes'])

    df_a['found_bool'] = norm_bool_series(df_a['plume_or_ship_found'])
    df_b['found_bool'] = norm_bool_series(df_b['plume_or_ship_found'])

    # merge on plume_file and UTC_Time
    merged = pd.merge(df_a, df_b, on=['plume_file', 'UTC_Time'], suffixes=('_a', '_b'))
    # select rows where booleans differ
    diffs = merged[merged['found_bool_a'] != merged['found_bool_b']]
    if diffs.empty:
        print('No differing plume_or_ship_found values between the two directories.')
        return

    print(f'Found {len(diffs)} entries with differing plume_or_ship_found; opening each for interactive labeling...')

    for _, row in diffs.iterrows():
        pf = row['plume_file']
        utc = row['UTC_Time']
        print('-' * 60)
        print('Plume file:', pf)
        print('UTC_Time:', utc)
        print(f"dir A ({dir_a}): {row.get('plume_or_ship_found_a')}  -- {row.get('source_csv_a')}")
        print(f"dir B ({dir_b}): {row.get('plume_or_ship_found_b')}  -- {row.get('source_csv_b')}")

        # attempt to resolve path under plume_root using date from UTC_Time
        cand = None
        try:
            date = pd.to_datetime(utc, utc=True).strftime('%y%m%d')
            cand = plume_root / f'plumes_{date}' / pf
            if not cand.exists():
                # fallback: try recursive search for filename
                matches = list(plume_root.rglob(pf))
                cand = matches[0] if matches else None
        except Exception:
            # fallback to recursive search
            matches = list(plume_root.rglob(pf))
            cand = matches[0] if matches else None

        if cand is None or not Path(cand).exists():
            print('Could not locate .nc for', pf, '; skipping')
            continue

        # open dataset and show LabelGUI
        try:
            ds_orig = xr.open_dataset(cand)
            ds = ds_orig.load()  # load into memory to avoid locking issues
            ds_orig.close()
            if 'no2_enhancement_interp' not in ds.variables:
                print('Dataset lacks no2_enhancement_interp; skipping', cand)
                ds.close()
                continue
            arr = ds['no2_enhancement_interp'].values
            if arr.ndim != 2:
                print('Dataset variable not 2D; skipping', cand)
                ds.close()
                continue
            title = f"{Path(cand).parent.name}/{Path(cand).name}\n(from {row.get('source_csv_a')} vs {row.get('source_csv_b')})"
            gui = LabelGUI(arr, cand, title=title)
            plt.show()
            res = gui.get_result()
            if res is None:
                print('No decision made for', cand, '; leaving unchanged')
                ds.close()
                continue
            useful = res.get('useful', False)
            point = res.get('point', None)
            ds.attrs['plume_useful'] = 'True' if useful else 'False'
            if point is not None:
                ds.attrs['plume_point_x'] = int(np.round(point[0]))
                ds.attrs['plume_point_y'] = int(np.round(point[1]))
            else:
                ds.attrs['plume_point_x'] = np.nan
                ds.attrs['plume_point_y'] = np.nan

            # Additional computed metrics: ship position at ds.attrs['t'], distance to IMPACT,
            # funnel time (t_funnel) from plume_point_x, funnel_top_vea from plume_point_y,
            # and funnel_height_m = tan(vea_rad) * distance.
            inst_lat, inst_lon = 53.56958522848946, 9.69174249821205
            # ship position nearest to ds.attrs['t']
            t_attr = ds.attrs.get('t', None)
            ship_dist_m = np.nan
            try:
                ship_times = pd.to_datetime(ds['ship_ais_times'].values, utc=True)
                t0 = pd.to_datetime(t_attr, utc=True)
                diffs = np.abs((ship_times - t0) / np.timedelta64(1, 's'))
                if len(diffs) > 0:
                    idx_near = int(np.nanargmin(diffs))
                    ship_lat = float(ds['ship_ais_lats'].values[idx_near])
                    ship_lon = float(ds['ship_ais_lons'].values[idx_near])
                    ds.attrs['ship_lat_at_t'] = ship_lat
                    ds.attrs['ship_lon_at_t'] = ship_lon
                    try:
                        ship_dist_m = float(geodesic((ship_lat, ship_lon), (inst_lat, inst_lon)).meters)
                    except Exception:
                        ship_dist_m = np.nan
                    ds.attrs['ship_distance_to_instrument_m'] = ship_dist_m
            except Exception:
                ds.attrs['ship_lat_at_t'] = np.nan
                ds.attrs['ship_lon_at_t'] = np.nan
                ds.attrs['ship_distance_to_instrument_m'] = np.nan

            # funnel time and funnel_top_vea from plume_point indices
            try:
                px = ds.attrs.get('plume_point_x', None)
                py = ds.attrs.get('plume_point_y', None)
                if px is not None and not (isinstance(px, float) and np.isnan(px)):
                    ix = int(round(float(px)))
                    times_plume = pd.to_datetime(ds['times_plume'].values)
                    ix = max(0, min(ix, len(times_plume)-1))
                    t_funnel = times_plume[ix]
                    ds.attrs['t_funnel'] = pd.to_datetime(t_funnel).isoformat()
                else:
                    ds.attrs['t_funnel'] = ''
                if py is not None and not (isinstance(py, float) and np.isnan(py)):
                    iy = int(round(float(py)))
                    vea_vals = np.asarray(ds['vea'].values)
                    iy = max(0, min(iy, len(vea_vals)-1))
                    funnel_top_vea = float(vea_vals[iy])
                    ds.attrs['funnel_top_vea'] = funnel_top_vea
                else:
                    ds.attrs['funnel_top_vea'] = np.nan
            except Exception:
                ds.attrs['t_funnel'] = ''
                ds.attrs['funnel_top_vea'] = np.nan

            # compute funnel height above IMPACT
            try:
                vea_deg = ds.attrs.get('funnel_top_vea', None)
                if (vea_deg is not None) and not (isinstance(vea_deg, float) and np.isnan(vea_deg)) and not (np.isnan(ship_dist_m)):
                    vea_rad = np.deg2rad(float(vea_deg))
                    funnel_height = np.tan(vea_rad) * float(ship_dist_m)
                    ds.attrs['funnel_height_m'] = float(funnel_height)
                else:
                    ds.attrs['funnel_height_m'] = np.nan
            except Exception:
                ds.attrs['funnel_height_m'] = np.nan

            ds.to_netcdf(cand, mode='w')
            print('Wrote plume_useful=', ds.attrs.get('plume_useful'), 'to', cand)
            ds.close()
        except Exception as e:
            print('Failed to open or write dataset', cand, e)
            try:
                ds.close()
            except Exception:
                pass


def merge_ship_passes_from_orig(orig_dir: Path, current_dir: Path):
    """Copy CSVs from `orig_dir`, merge `plume_useful` values from matching files in `current_dir`,
    and write the merged CSVs into `current_dir` (overwrite existing files).

    Matching is done by filename. Within each file, rows are matched using both `plume_file` (filename)
    and `UTC_Time` when available; otherwise matching falls back to `plume_file` only.
    """
    orig_dir = Path(orig_dir)
    current_dir = Path(current_dir)
    if not orig_dir.exists():
        print('Original directory does not exist:', orig_dir)
        return
    if not current_dir.exists():
        print('Current directory does not exist, creating:', current_dir)
        try:
            current_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('Failed to create current_dir:', e)
            return

    summary = {'copied': 0, 'merged': 0, 'skipped': 0}
    for orig_csv in sorted(orig_dir.glob('*.csv')):
        try:
            orig_df = pd.read_csv(orig_csv, dtype=str)
        except Exception as e:
            print('Failed to read original CSV', orig_csv, e)
            summary['skipped'] += 1
            continue

        target_csv = current_dir / orig_csv.name
        if not target_csv.exists():
            # no current file: simply write the original into current_dir
            try:
                orig_df.to_csv(target_csv, index=False)
                print('Copied (no existing):', orig_csv.name)
                summary['copied'] += 1
            except Exception as e:
                print('Failed to copy', orig_csv, e)
                summary['skipped'] += 1
            continue

        # there is a current file; try to merge plume_useful
        try:
            cur_df = pd.read_csv(target_csv, dtype=str)
        except Exception as e:
            print('Failed to read current CSV', target_csv, e)
            # fallback: overwrite with orig
            try:
                orig_df.to_csv(target_csv, index=False)
                summary['copied'] += 1
            except Exception:
                summary['skipped'] += 1
            continue

        if 'plume_useful' not in cur_df.columns:
            # nothing to merge; overwrite with original
            try:
                orig_df.to_csv(target_csv, index=False)
                print('Overwrote (no plume_useful in current):', orig_csv.name)
                summary['copied'] += 1
            except Exception as e:
                print('Failed to overwrite', target_csv, e)
                summary['skipped'] += 1
            continue

        # prepare matching keys
        def norm(x):
            return '' if pd.isna(x) else str(x).strip()

        orig_df['_pf'] = orig_df.get('plume_file', '').apply(lambda s: Path(norm(s)).name if norm(s) else '')
        cur_df['_pf'] = cur_df.get('plume_file', '').apply(lambda s: Path(norm(s)).name if norm(s) else '')
        have_utc = ('UTC_Time' in orig_df.columns) and ('UTC_Time' in cur_df.columns)
        if have_utc:
            orig_df['_ut'] = orig_df.get('UTC_Time', '').apply(norm)
            cur_df['_ut'] = cur_df.get('UTC_Time', '').apply(norm)
            cur_df['_key'] = cur_df['_pf'].fillna('') + '|' + cur_df['_ut'].fillna('')
            mapping = dict(zip(cur_df['_key'], cur_df['plume_useful']))
            def map_row(r):
                k = (r['_pf'] or '') + '|' + (r['_ut'] or '')
                return mapping.get(k)
        else:
            cur_df['_key'] = cur_df['_pf']
            mapping = dict(zip(cur_df['_key'], cur_df['plume_useful']))
            def map_row(r):
                k = r['_pf']
                return mapping.get(k)

        # apply mapping: set plume_useful in orig_df where mapping exists
        merged_count = 0
        if 'plume_useful' not in orig_df.columns:
            orig_df['plume_useful'] = False

        for idx, r in orig_df.iterrows():
            val = map_row(r)
            if val is not None and not (pd.isna(val)):
                try:
                    is_useful = True if str(val).strip().lower() in ('true','1','t','yes') else False
                except Exception:
                    is_useful = False
                orig_df.at[idx, 'plume_useful'] = is_useful
                merged_count += 1

        # drop helper cols
        for c in ['_pf', '_ut', '_key']:
            if c in orig_df.columns:
                orig_df.drop(columns=[c], inplace=True)
            if c in cur_df.columns:
                cur_df.drop(columns=[c], inplace=True)

        try:
            orig_df.to_csv(target_csv, index=False)
            print(f'Merged {merged_count} plume_useful values into {target_csv.name}')
            summary['merged'] += 1
        except Exception as e:
            print('Failed to write merged CSV', target_csv, e)
            summary['skipped'] += 1

    print('Done. Summary:', summary)

def review_found_and_relabel(plume_root: Path, min_date):
    """Iterate .nc files under `plume_root` and present files where
    `plume_or_ship_found == True` but `plume_useful` is False (or missing)
    for interactive relabeling. Writes `plume_useful`, `plume_point_x`,
    `plume_point_y` and the ship/funnel metrics into each .nc.
    `min_date` should be a timezone-aware pandas.Timestamp or comparable.
    """
    plume_root = Path(plume_root)
    if not plume_root.exists():
        print('plume_root does not exist:', plume_root)
        return

    files = find_plume_nc_files(plume_root)
    if not files:
        print('No .nc files found under', plume_root)
        return

    updated = 0
    for nc in files:
        # quick open to inspect attrs without loading full variables
        try:
            ds_probe = xr.open_dataset(nc)
        except Exception:
            continue

        try:
            t_attr = ds_probe.attrs.get('t', None)
            found = str(ds_probe.attrs.get('plume_or_ship_found', 'False')).strip() == 'True'
            useful_attr = ds_probe.attrs.get('plume_useful', 'False')
            is_useful = (str(useful_attr) == 'True') or (useful_attr is True)
            has_var = 'no2_enhancement_interp' in ds_probe.variables
            ds_probe.close()
        except Exception:
            try:
                ds_probe.close()
            except Exception:
                pass
            continue

        # only consider detected plumes that are currently NOT marked useful and have the variable
        if (not found) or is_useful or (not has_var):
            continue

        # enforce minimum date
        try:
            if t_attr is None:
                continue
            t_ds = pd.to_datetime(t_attr, utc=True)
            if t_ds < min_date:
                continue
        except Exception:
            continue

        # load dataset into memory for safe reading/writing and plotting
        try:
            ds_orig = xr.open_dataset(nc).load()
            ds = ds_orig.copy()
            ds_orig.close()
        except Exception as e:
            print('Failed to open/load', nc, e)
            continue

        try:
            arr = ds['no2_enhancement_interp'].values
            if arr is None or arr.ndim != 2:
                ds.close()
                continue
        except Exception:
            ds.close()
            continue

        title = f"{Path(nc).parent.name}/{Path(nc).name} (review detected-but-not-useful)"
        gui = LabelGUI(arr, nc, title=title, cmap='viridis')
        plt.show()
        res = gui.get_result()
        if res is None:
            print('No decision made for', nc, '; leaving unchanged')
            ds.close()
            continue

        useful = res.get('useful', False)
        point = res.get('point', None)

        # write choice and point
        ds.attrs['plume_useful'] = 'True' if useful else 'False'
        if point is not None and useful:
            ds.attrs['plume_point_x'] = int(np.round(point[0]))
            ds.attrs['plume_point_y'] = int(np.round(point[1]))
        else:
            ds.attrs['plume_point_x'] = np.nan
            ds.attrs['plume_point_y'] = np.nan

        # compute ship/funnel metrics (same logic used elsewhere in this script)
        inst_lat, inst_lon = 53.56958522848946, 9.69174249821205
        ship_dist_m = np.nan
        try:
            ship_times = pd.to_datetime(ds['ship_ais_times'].values, utc=True)
            t0 = pd.to_datetime(ds.attrs.get('t', None), utc=True)
            diffs = np.abs((ship_times - t0) / np.timedelta64(1, 's'))
            if len(diffs) > 0:
                idx_near = int(np.nanargmin(diffs))
                ship_lat = float(ds['ship_ais_lats'].values[idx_near])
                ship_lon = float(ds['ship_ais_lons'].values[idx_near])
                ds.attrs['ship_lat_at_t'] = ship_lat
                ds.attrs['ship_lon_at_t'] = ship_lon
                try:
                    ship_dist_m = float(geodesic((ship_lat, ship_lon), (inst_lat, inst_lon)).meters)
                except Exception:
                    ship_dist_m = np.nan
                ds.attrs['ship_distance_to_instrument_m'] = ship_dist_m
            else:
                ds.attrs['ship_lat_at_t'] = np.nan
                ds.attrs['ship_lon_at_t'] = np.nan
                ds.attrs['ship_distance_to_instrument_m'] = np.nan
        except Exception:
            ds.attrs['ship_lat_at_t'] = np.nan
            ds.attrs['ship_lon_at_t'] = np.nan
            ds.attrs['ship_distance_to_instrument_m'] = np.nan

        # funnel time and funnel_top_vea from plume_point indices
        try:
            px = ds.attrs.get('plume_point_x', None)
            py = ds.attrs.get('plume_point_y', None)
            if px is not None and not (isinstance(px, float) and np.isnan(px)):
                ix = int(round(float(px)))
                times_plume = pd.to_datetime(ds['times_plume'].values)
                ix = max(0, min(ix, len(times_plume)-1))
                t_funnel = times_plume[ix]
                ds.attrs['t_funnel'] = pd.to_datetime(t_funnel).isoformat()
            else:
                ds.attrs['t_funnel'] = ''
            if py is not None and not (isinstance(py, float) and np.isnan(py)):
                iy = int(round(float(py)))
                vea_vals = np.asarray(ds['vea'].values)
                iy = max(0, min(iy, len(vea_vals)-1))
                funnel_top_vea = float(vea_vals[iy])
                ds.attrs['funnel_top_vea'] = funnel_top_vea
            else:
                ds.attrs['funnel_top_vea'] = np.nan
        except Exception:
            ds.attrs['t_funnel'] = ''
            ds.attrs['funnel_top_vea'] = np.nan

        # compute funnel height above IMPACT
        try:
            vea_deg = ds.attrs.get('funnel_top_vea', None)
            if (vea_deg is not None) and not (isinstance(vea_deg, float) and np.isnan(vea_deg)) and not (np.isnan(ship_dist_m)):
                vea_rad = np.deg2rad(float(vea_deg))
                funnel_height = np.tan(vea_rad) * float(ship_dist_m)
                ds.attrs['funnel_height_m'] = float(funnel_height)
            else:
                ds.attrs['funnel_height_m'] = np.nan
        except Exception:
            ds.attrs['funnel_height_m'] = np.nan

        # persist changes
        try:
            ds.to_netcdf(nc, mode='w')
            updated += 1
            print('Wrote review result plume_useful=', ds.attrs.get('plume_useful'), 'to', nc)
        except Exception as e:
            print('Failed to write dataset', nc, e)
        finally:
            ds.close()

    print('Review complete. Updated files:', updated)
    return updated

def main():
    p = argparse.ArgumentParser(description='Label plume datasets interactively (plotting no2_enhancement_interp)')
    p.add_argument('--plume-root', required=False, default=r"P:\data\\SEICOR\plumes_2", help='Root directory containing plume subfolders (recursive search). Defaults to Q:\\BREDOM\\SEICOR\\plumes_2')
    p.add_argument('--filter', required=False, default=None, help='Optional substring filter for file paths')
    p.add_argument('--ship-pass-dir', required=False, default=r"P:\data\\SEICOR\ship_passes_2", help='Directory containing ship_passes CSV files')
    p.add_argument('--sync-csvs', action='store_true', help='Only sync plume_useful into ship-pass CSVs and exit')
    p.add_argument('--count-csvs', action='store_true', help='Count total ship_pass rows and useful plumes in CSVs and exit')
    p.add_argument('--export-timestamps', required=False, default=None, help='Path to CSV to write UTC_Time and plume_useful from all ship-pass CSVs')
    p.add_argument('--count-duplicate-mmsi', action='store_true', help='Count MMSI occurrences when plume_useful==True and report duplicates')
    p.add_argument('--sync-and-report', action='store_true', help='Sync CSVs, export timestamps (to --export-timestamps or default), and print counts')
    p.add_argument('--resolve-conflicts', action='store_true', help='Find close-in-time useful plumes and resolve interactively')
    p.add_argument('--resolve-window', type=int, default=120, help='Time window in seconds for conflict detection')
    p.add_argument('--reconcile-found', nargs=2, metavar=('DIR_A','DIR_B'), help='Compare plume_or_ship_found between two ship_pass dirs and open differing plumes from plume_root for interactive plume_useful labeling')
    p.add_argument('--merge-ship-passes', nargs=2, metavar=('ORIG_DIR','CURRENT_DIR'), help='Copy CSVs from ORIG_DIR, merge plume_useful from CURRENT_DIR, and write merged files into CURRENT_DIR (overwrite)')
    p.add_argument('--review-found', action='store_true', help='Review .nc files with plume_or_ship_found==True but plume_useful==False and relabel interactively')
    args = p.parse_args()

    plume_root = Path(args.plume_root)
    if not plume_root.exists():
        print('plume-root does not exist:', plume_root)
        sys.exit(1)
    import pandas as pd
    # Only consider plumes at or after this date (YYMMDD)
    min_date_str = '250326'
    try:
        min_date = pd.to_datetime(min_date_str, format='%y%m%d', utc=True)
    except Exception:
        min_date = pd.to_datetime(min_date_str, utc=True)

    # build the analysis plumes out_dir based on provided date (server location)


    # first, try to read ship_pass CSVs and collect plume files listed there
    valid_files = []
    ship_pass_dir = Path(args.ship_pass_dir)
    # If requested, only perform CSV sync and/or counting and exit
    if args.sync_csvs:
        update_ship_pass_csvs(ship_pass_dir, plume_root)
        # after syncing, optionally also print counts
        if args.count_csvs:
            count_ship_passes(ship_pass_dir)
        if args.export_timestamps:
            export_timestamps(ship_pass_dir, Path(args.export_timestamps))
        return
    if args.sync_and_report:
        update_ship_pass_csvs(ship_pass_dir, plume_root, propagate_nc_to_csv=False)
        if args.export_timestamps:
            outp = Path(args.export_timestamps)
        else:
            outp = ship_pass_dir / 'plume_timestamps.csv'
        export_timestamps(ship_pass_dir, outp)
        count_ship_passes(ship_pass_dir)
        return
    if args.count_csvs:
        count_ship_passes(ship_pass_dir)
        return
    if args.count_duplicate_mmsi:
        count_duplicate_mmsi(ship_pass_dir)
        return
    if args.export_timestamps:
        export_timestamps(ship_pass_dir, Path(args.export_timestamps))
        return
    if args.resolve_conflicts:
        resolve_conflicts_within_window(ship_pass_dir, plume_root, window_seconds=args.resolve_window)
        return
    if args.reconcile_found:
        a, b = args.reconcile_found
        reconcile_plume_or_ship_found(Path(a), Path(b), plume_root)
        return
    if args.merge_ship_passes:
        orig, curr = args.merge_ship_passes
        merge_ship_passes_from_orig(Path(orig), Path(curr))
        return
    if args.review_found:
        review_found_and_relabel(plume_root, min_date)
        return
    if ship_pass_dir.exists():
        
        csvs = sorted(ship_pass_dir.glob('*.csv'))
        for csvf in csvs:
            print(f'Reading ship_pass CSV: {csvf}')
            try:
                df = pd.read_csv(csvf)
            except Exception as e:
                print(f'Failed to read CSV {csvf}: {e}')
                continue
            # iterate rows
            for _, ship_pass in df.iterrows():
                try:
                    found = ship_pass.get('plume_or_ship_found', False)
                except Exception:
                    found = False
                # accept booleans or string versions
                if str(found).lower() != 'true':
                    continue
                # enforce minimum date on ship_pass UTC_Time
                try:
                    ship_time = pd.to_datetime(ship_pass.UTC_Time, utc=True)
                    if ship_time < min_date:
                        continue
                except Exception:
                    # if we cannot parse time, skip the row
                    continue
                stored = ship_pass.get('plume_file', None)
                if stored is None or (isinstance(stored, float) and np.isnan(stored)):
                    continue
                fn = Path(stored).name
                date= pd.to_datetime(ship_pass.UTC_Time).strftime('%y%m%d')
                out_dir_server = Path(r"P:\data\SEICOR\plumes_2\plumes_" + date)
                candidate = out_dir_server / fn
                if candidate.exists():
                    valid_files.append(candidate)
                else:
                    print(f'Candidate plume file not found on server: {candidate}')
    else:
        print('ship-pass-dir does not exist, skipping ship_pass CSV reading:', ship_pass_dir)
    # remove duplicates while preserving order
    seen = set()
    valid_files = [x for x in valid_files if not (x in seen or seen.add(x))]
    
    detected_file = plume_root / 'plume_detected.txt'

    # If ship-pass CSVs produced files, skip recursive search and use them.
    if not valid_files:
        if detected_file.exists():
            # read the listing and use entries as the files to label
            print(f'Found {detected_file}; using listed entries')
            with detected_file.open('r', encoding='utf-8') as fh:
                entries = [ln.strip() for ln in fh if ln.strip()]

            for ent in entries:
                p = Path(ent)
                if not p.is_absolute():
                    p = plume_root / ent
                if not p.exists():
                    print(f'Listed file not found, skipping: {p}')
                    continue
                try:
                    ds = xr.open_dataset(p)
                    # skip files with dataset time before minimum date
                    t_attr = ds.attrs.get('t', None)
                    if t_attr is None:
                        ds.close()
                        print(f'Skipping listed file (missing t attr): {p}')
                        continue
                    try:
                        t_ds = pd.to_datetime(t_attr, utc=True)
                        if t_ds < min_date:
                            ds.close()
                            print(f'Skipping listed file (date before min): {p}')
                            continue
                    except Exception:
                        ds.close()
                        print(f'Skipping listed file (invalid t attr): {p}')
                        continue

                    found = str(ds.attrs.get('plume_or_ship_found', 'False')) == 'True'
                    has_var = 'no2_enhancement_interp' in ds.variables
                    ds.close()
                    if found and has_var:
                        valid_files.append(p)
                    else:
                        print(f'Skipping listed file (no plume found or missing var): {p}')
                except Exception as e:
                    print(f'Failed to open listed file {p}: {e}')

        else:
            # no listing present — search recursively and build the list, then save it
            nc_files = find_plume_nc_files(plume_root)
            if args.filter:
                nc_files = [p for p in nc_files if args.filter in str(p)]

            for nc in nc_files:
                try:
                    ds = xr.open_dataset(nc)
                    print(f'Checking file: {nc}')
                    # skip files where dataset time is before minimum date
                    t_attr = ds.attrs.get('t', None)
                    if t_attr is None:
                        ds.close()
                        print(f'  Skipping (missing t attr): {nc}')
                        continue
                    try:
                        t_ds = pd.to_datetime(t_attr, utc=True)
                        if t_ds < min_date:
                            ds.close()
                            print(f'  Skipping (date before min): {nc}')
                            continue
                    except Exception:
                        ds.close()
                        print(f'  Skipping (invalid t attr): {nc}')
                        continue

                    found = str(ds.attrs.get('plume_or_ship_found', 'False')) == 'True'
                    print(f'  plume_or_ship_found: {found}')
                    has_var = 'no2_enhancement_interp' in ds.variables
                    ds.close()
                    if found and has_var:
                        valid_files.append(nc)
                except Exception as e:
                    print(f'Failed to open {nc}: {e}')

            # write discovered valid files to plume_detected.txt for next runs
            try:
                with detected_file.open('w', encoding='utf-8') as fh:
                    for p in valid_files:
                        try:
                            rel = p.relative_to(plume_root)
                            fh.write(str(rel).replace('\\', '/') + '\n')
                        except Exception:
                            fh.write(str(p) + '\n')
                print(f'Wrote {len(valid_files)} entries to {detected_file}')
            except Exception as e:
                print(f'Could not write detected file {detected_file}: {e}')

    if not valid_files:
        print('No plume .nc files with plume_or_ship_found==True were found under', plume_root)
        sys.exit(0)

    print(f'Found {len(valid_files)} plume files to label; press YES/NO to record label and move on. Close window to quit.')

    for nc in valid_files:
        try:
            ds_orig = xr.open_dataset(nc)
            ds = ds_orig.load()  # load into memory to avoid locking issues
            ds_orig.close()

        except Exception as e:
            print(f'Could not open dataset {nc}: {e}')
            continue

        if 'no2_enhancement_interp' not in ds.variables:
            print(f"Dataset {nc} has no 'no2_enhancement_interp' variable; skipping")
            ds.close()
            continue

        arr = ds['no2_enhancement_interp'].values
        # ensure 2D
        if arr.ndim != 2:
            print(f"no2_enhancement_interp in {nc} is not 2D; skipping")
            ds.close()
            continue

        title = f"{PurePath(nc).parent.name}/{nc.name}"
        viewer = LabelGUI(arr, nc, title=title, cmap='viridis')
        plt.show()
        result = viewer.get_result()
        if result is None:
            print(f'No label selected for {nc}; skipping write')
            ds.close()
            continue
        useful = result.get('useful', False)
        point = result.get('point', None)
        # write basic attributes into the opened xarray Dataset
        try:
            ds.attrs['plume_useful'] = 'True' if useful else 'False'
            if point is not None:
                ds.attrs['plume_point_x'] = int(np.round(point[0]))
                ds.attrs['plume_point_y'] = int(np.round(point[1]))
            else:
                ds.attrs['plume_point_x'] = np.nan
                ds.attrs['plume_point_y'] = np.nan
            # Additional computed metrics: ship position at ds.attrs['t'], distance to IMPACT,
            # funnel time (t_funnel) from plume_point_x, funnel_top_vea from plume_point_y,
            # and funnel_height_m = tan(vea_rad) * distance.
            
            inst_lat, inst_lon = 53.56958522848946, 9.69174249821205

            # ship position nearest to ds.attrs['t']
            t_attr = ds.attrs.get('t', None)
            ship_dist_m = np.nan

            # ensure both are timezone-aware (UTC) to avoid tz-naive vs tz-aware subtraction
            ship_times = pd.to_datetime(ds['ship_ais_times'].values, utc=True)
            t0 = pd.to_datetime(t_attr, utc=True)
            # compute absolute differences in seconds robustly
            diffs = np.abs((ship_times - t0) / np.timedelta64(1, 's'))
            if len(diffs) > 0:
                idx_near = int(np.nanargmin(diffs))
                ship_lat = float(ds['ship_ais_lats'].values[idx_near])
                ship_lon = float(ds['ship_ais_lons'].values[idx_near])
                ds.attrs['ship_lat_at_t'] = ship_lat
                ds.attrs['ship_lon_at_t'] = ship_lon
                print(f'Ship position at t: lat={ship_lat}, lon={ship_lon}')
                try:
                    ship_dist_m = float(geodesic((ship_lat, ship_lon), (inst_lat, inst_lon)).meters)
                except Exception:
                    ship_dist_m = np.nan
                ds.attrs['ship_distance_to_instrument_m'] = ship_dist_m
            # funnel time and funnel_top_vea from plume_point indices
            try:
                px = ds.attrs.get('plume_point_x', None)
                py = ds.attrs.get('plume_point_y', None)
                if px is not None and not (isinstance(px, float) and np.isnan(px)):
                    ix = int(round(float(px)))
                    times_plume = pd.to_datetime(ds['times_plume'].values)
                    ix = max(0, min(ix, len(times_plume)-1))
                    t_funnel = times_plume[ix]
                    ds.attrs['t_funnel'] = pd.to_datetime(t_funnel).isoformat()
                else:
                    ds.attrs['t_funnel'] = ''
                if py is not None and not (isinstance(py, float) and np.isnan(py)):
                    iy = int(round(float(py)))
                    vea_vals = np.asarray(ds['vea'].values)
                    iy = max(0, min(iy, len(vea_vals)-1))
                    funnel_top_vea = float(vea_vals[iy])
                    ds.attrs['funnel_top_vea'] = funnel_top_vea
                else:
                    ds.attrs['funnel_top_vea'] = np.nan
            except Exception:
                ds.attrs['t_funnel'] = ''
                ds.attrs['funnel_top_vea'] = np.nan

            # compute funnel height above IMPACT
            try:
                vea_deg = ds.attrs.get('funnel_top_vea', None)
                if (vea_deg is not None) and not (isinstance(vea_deg, float) and np.isnan(vea_deg)) and not (np.isnan(ship_dist_m)):
                    vea_rad = np.deg2rad(float(vea_deg))
                    funnel_height = np.tan(vea_rad) * float(ship_dist_m)
                    ds.attrs['funnel_height_m'] = float(funnel_height)
                else:
                    ds.attrs['funnel_height_m'] = np.nan
            except Exception:
                ds.attrs['funnel_height_m'] = np.nan



            ds.to_netcdf(nc, mode='w')
            print(f'Wrote attributes to {nc}')
        except Exception as e:
            print(f'Failed to write attributes to {nc}: {e}')
        finally:
            ds.close()


if __name__ == '__main__':
    main()

# %%
