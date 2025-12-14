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
import numpy as np
import xarray as xr
import sys


class LabelGUI:
    def __init__(self, data_array: np.ndarray, plume_file: Path, title=None, cmap='viridis'):
        self.plume_file = Path(plume_file)
        self.point = None
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.15)

        # Display the 2D data array; origin='lower' matches common plotting for these datasets
        self.img = self.ax.imshow(data_array, origin='lower', aspect='auto', cmap=cmap)
        self.ax.set_title(title or f"{self.plume_file.name}")
        self.fig.colorbar(self.img, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)

        # Buttons
        ax_yes = plt.axes([0.7, 0.03, 0.1, 0.06])
        ax_no = plt.axes([0.82, 0.03, 0.1, 0.06])
        self.btn_yes = Button(ax_yes, 'YES')
        self.btn_no = Button(ax_no, 'NO')
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
        try:
            # open netCDF in append mode and set global attributes
            with netCDF4.Dataset(self.plume_file.as_posix(), 'r+') as nc:
                nc.setncattr('plume_useful', 'True' if useful else 'False')
                if self.point is not None:
                    nc.setncattr('plume_point_x', float(self.point[0]))
                    nc.setncattr('plume_point_y', float(self.point[1]))
                else:
                    nc.setncattr('plume_point_x', np.nan)
                    nc.setncattr('plume_point_y', np.nan)
            print(f"Wrote attributes to {self.plume_file}")
        except Exception as e:
            print(f"Failed to write attributes to {self.plume_file}: {e}")

    def on_yes(self, event):
        self.write_attributes(True)
        plt.close(self.fig)

    def on_no(self, event):
        self.write_attributes(False)
        plt.close(self.fig)


def find_plume_nc_files(root_dir: Path):
    # recursively find .nc files under root_dir
    files = [p for p in sorted(Path(root_dir).rglob('*.nc'))]
    return files


def main():
    p = argparse.ArgumentParser(description='Label plume datasets interactively (plotting no2_enhancement_interp)')
    p.add_argument('--plume-root', required=False, default=r"Q:\\BREDOM\\SEICOR\\plumes_2", help='Root directory containing plume subfolders (recursive search). Defaults to Q:\\BREDOM\\SEICOR\\plumes_2')
    p.add_argument('--filter', required=False, default=None, help='Optional substring filter for file paths')
    args = p.parse_args()

    plume_root = Path(args.plume_root)
    if not plume_root.exists():
        print('plume-root does not exist:', plume_root)
        sys.exit(1)

    detected_file = plume_root / 'plume_detected.txt'

    valid_files = []

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
            ds = xr.open_dataset(nc)
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
        ds.close()


if __name__ == '__main__':
    main()

# %%
