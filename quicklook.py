#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from cindi3.get_max_doas_data import read_SC_MAX_DOAS_simple
from doas_tools.file_handling import read_SC_file_imaging
from doas_tools.file_handling import read_img
from imaging_tools.process_IMG import process_img_data
from imaging_tools.process_SC import process_SC_img_data
from imaging_tools.plotting import flotter_plotter_cmap
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter
import glob
import numpy as np
import scipy.odr as odr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
#%%
path = r"P:\data\wedel\SLANT_25\MAE2025"
ds = read_SC_file_imaging(path, f"250328", f"ID.NO2_VIS_fix_highres")

ds = process_SC_img_data(ds, "time_series", start_idx=0, end_idx=None)

ds_img = read_img(r"P:\data\wedel\org\MAE2025\250328SD.IMG")
ds_img = process_img_data(ds_img, path_to_settings= r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\proc_settings.yaml")
# %%
los_mask = ds.los == 180
masked_indices = np.argwhere(los_mask.values)  # Get (i, j) indices where condition is met

# Create a boolean mask of the same shape as ds.los
expanded_mask = np.ones_like(ds.los, dtype=bool)

# Expand the mask to include neighbors (up, down, left, right)
for i, j in masked_indices:
    for di in [-2,-1, 0, 1,2]:  # Row shift (-1, 0, +1)
        for dj in [-2,-1, 0, 1,2]:  # Column shift (-1, 0, +1)
            ni, nj = i + di, j + dj
            if 0 <= ni < ds.los.shape[0] and 0 <= nj < ds.los.shape[1]:  # Bounds check
                expanded_mask[ni, nj] = False
# Apply the mask
#%%
samp_start=200
samp_end=7600
ds["a[NO2]"].isel(viewing_direction=slice(0,20), sample=slice(samp_start,samp_end)).plot()
# %%
ds["a[O4]"].isel(viewing_direction=slice(0,20), sample=slice(samp_start,samp_end)).plot()

# %%
ds["ints1"].isel(viewing_direction=slice(0,20), sample=slice(samp_start,samp_end)).plot()

#%%
ds["datetime"].isel( sample=slice(samp_start,samp_end)).plot()
# %%
plt.plot(ds["a[NO2]"].where(expanded_mask).isel(viewing_direction=slice(2,20), sample=slice(samp_start,samp_end)).T, label="mean")
#%%
time_values = np.array(pd.to_datetime(ds["datetime"].where(expanded_mask[0])
                             .isel(sample=slice(samp_start,samp_end)).values).strftime('%H:%M:%S'))

# Plot using only time values

#%%
plt.plot(ds["datetime"].where(expanded_mask[0]).isel(sample=slice(samp_start,samp_end)).values, 
         ds["a[NO2]"].where(expanded_mask)
         .isel(viewing_direction=slice(5,10), sample=slice(samp_start,samp_end)).T, 
         label="mean")

plt.xlabel("Time")
plt.legend()
plt.show()
# %%
plt.plot(ds["datetime"].where(expanded_mask[0]).isel(sample=slice(samp_start,samp_end)).values, 
         ds["a[O4]"].where(expanded_mask)
         .isel(viewing_direction=slice(5,10), sample=slice(samp_start,samp_end)).T, 
         label="mean")

plt.xlabel("Time")
plt.legend()
plt.show()# %% 
#%% 
plt.plot(ds["datetime"].where(expanded_mask[0]).isel(sample=slice(samp_start,samp_end)).values, 
         ds["expt"].where(expanded_mask[0]).where(ds["expt"]<20)
         .isel( sample=slice(samp_start,samp_end)).T, 
         label="mean")

plt.xlabel("Time")
plt.legend()
plt.show()# %% 
#%% 
plt.plot(ds["datetime"].where(expanded_mask[0]).isel(sample=slice(samp_start,samp_end)).values, 
         ds["viewing-azimuth-angle"].where(expanded_mask[0])
         .isel( sample=slice(samp_start,samp_end)).T, 
         label="mean")

plt.xlabel("Time")
plt.legend()
plt.show()# %% 
#%%
plt.plot(ds["expt"].isel(sample=slice(samp_start,samp_end)).T, label="mean")

#%%
plt.plot(ds["viewing-azimuth-angle"].isel(sample=slice(samp_start,samp_end)).T, label="mean")
# %%

base_date = "2025-03-28"

# File path to the time values
large_path = r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\large_size.txt"
mid_path = r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\mid_size.txt"
small_path = r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\small_size.txt"

with open(large_path, "r") as file:
    time_strings = [line.strip().zfill(6) for line in file]  # pad with zeros just in case
large_ships = np.array([
    np.datetime64(f"{base_date}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}")
    for time_str in time_strings]) - np.timedelta64(1, 'h')

with open(mid_path, "r") as file:
    time_strings = [line.strip().zfill(6) for line in file]  # pad with zeros just in case
mid_ships = np.array([
    np.datetime64(f"{base_date}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}")
    for time_str in time_strings]) - np.timedelta64(1, 'h')

with open(small_path, "r") as file:
    time_strings = [line.strip().zfill(6) for line in file]  # pad with zeros just in case

small_ships = np.array([
    np.datetime64(f"{base_date}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}")
    for time_str in time_strings]) - np.timedelta64(1, 'h')







samp_start = 200
samp_end = 7600
# %%
line_views = range(6, 22, 3)
plt.figure(figsize=(10,5))
for j in line_views:
    plt.plot(
        ds["datetime"].isel(sample=slice(samp_start,samp_end)).values,
        ds["a[NO2]"].isel(sample=slice(samp_start,samp_end), viewing_direction=j),
        label=f"VD {j}"
    )
for t in large_ships:
    plt.axvline(x=t.astype('datetime64[s]'), color='red', alpha=0.4, linewidth=2)
for t in mid_ships:
    plt.axvline(x=t.astype('datetime64[s]'), color='blue', alpha=0.25, linewidth=2)
for t in small_ships:
    plt.axvline(x=t.astype('datetime64[s]'), color='green', alpha=0.2, linewidth=2)
plt.xlabel("Time (UTC)")
plt.ylabel("a[NO2]")
plt.tight_layout()
plt.legend()

# %%
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

line_views = range(6, 22, 3)
fig = plt.figure(figsize=(10, 5))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

# Create parasite axes for second and third y-axis
par1 = host.twinx()
par2 = host.twinx()

# Offset the third axis to the right
par2.axis["right"] = par2.get_grid_helper().new_fixed_axis(loc="right", axes=par2, offset=(60, 0))
par2.axis["right"].toggle(all=True)

# --- Primary axis: a[O4] ---
for j in line_views:
    host.plot(
        ds["datetime"].isel(sample=slice(samp_start, samp_end)).values,
        ds["a[O4]"].isel(sample=slice(samp_start, samp_end), viewing_direction=j),
        label=f"VD {j}"
    )

# Vertical lines for ship detections
for t in large_ships:
    host.axvline(x=t.astype('datetime64[s]'), color='red', alpha=0.4, linewidth=2)
for t in mid_ships:
    host.axvline(x=t.astype('datetime64[s]'), color='blue', alpha=0.25, linewidth=2)
for t in small_ships:
    host.axvline(x=t.astype('datetime64[s]'), color='green', alpha=0.2, linewidth=2)

# --- Second axis: Exposure Time ---
par1.plot(
    ds_img["t_0"].values,
    ds_img["t_expos"],
    color='black', linestyle='--',
    label='Exposure Time'
)

# --- Third axis: vaa_c ---
par2.plot(
    ds_img["t_0"].values,
    ds_img["vaa_c"],
    color='purple', linestyle='--', alpha=0.7,
    label='VAA'
)

# Axis labels
host.set_xlabel("Time (UTC)")
host.set_ylabel("a[O4]")
par1.set_ylabel("Exposure Time (s)")
par2.set_ylabel("VAA [°]")

# Legends
host.legend(loc="upper left")
par1.axis["right"].label.set_color('black')
par2.axis["right"].label.set_color('purple')

plt.tight_layout()
plt.show()


# %%

# %%
path = r"P:\data\wedel\SLANT_25\APR2025"
ds = read_SC_file_imaging(path, f"250401", f"ID.NO2_VIS_fix")
ds = process_SC_img_data(ds, "time_series", start_idx=0, end_idx=None)

ds_img = read_img(r"P:\data\wedel\org\APR2025\250401ID.IMG_10")
ds_img = process_img_data(ds_img, path_to_settings= r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\proc_settings.yaml")
# %%
ds["a[O4]"].plot()
# %%
ds["datetime"].isel(sample=slice(3000,4500)).plot()
# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

window_samples = 400  # ~2 minutes window (adjust as needed)

view_inds = np.arange(0, 41)  # for pcolormesh
line_views = range(6, 22, 3)  # for line plot

all_ships = [(t, 'large size') for t in large_ships] + \
            [(t, 'medium size') for t in mid_ships] + \
            [(t, 'small size') for t in small_ships]

all_ships.sort(key=lambda x: x[0])
datetimes = ds["datetime"].values

for i, (ship_time, category) in enumerate(all_ships):
    # Find nearest sample index
    closest_idx = np.argmin(np.abs(datetimes - ship_time))
    start_idx = max(closest_idx - window_samples, 0)
    end_idx = min(closest_idx + window_samples, len(datetimes))
    sample_slice = slice(start_idx, end_idx)
    sample_slice_short = slice(start_idx, start_idx +12)
    time_window = datetimes[sample_slice]
    time_numeric = mdates.date2num(time_window)

    # Calculate bin edges
    time_edges = np.zeros(len(time_numeric) + 1)
    time_edges[1:-1] = (time_numeric[1:] + time_numeric[:-1]) / 2
    time_edges[0] = time_numeric[0] - (time_numeric[1] - time_numeric[0]) / 2
    time_edges[-1] = time_numeric[-1] + (time_numeric[-1] - time_numeric[-2]) / 2
    y = view_inds
    y_edges = np.append(y, y[-1] + 1)

    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))

    # --- Line Plot ---
    for j in line_views:
        axs[0].plot(
            time_window,
            ds["a[NO2]"].isel(sample=sample_slice, viewing_direction=j).where(ds.los.isel(sample=sample_slice, viewing_direction=j) != 180),
            label=f"VD {j}"
        )

    # Add vlines for *all* ships in the window
    for other_time, other_cat in all_ships:
        if time_window[0] <= other_time <= time_window[-1]:
            axs[0].axvline(
                x=other_time,
                color={'large size': 'red', 'medium size': 'blue', 'small size': 'green'}[other_cat],
                linestyle='--',
                linewidth=2,
                alpha=0.5 if other_time != ship_time else 0.9
            )

    axs[0].set_ylabel("a[NO2]")
    axs[0].set_title(f"{category.capitalize()} Ship @ {str(ship_time)}")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.3)

    # --- Heatmaps ---
    var_list = ["a[NO2]", "a[O4]"]  # You can expand this list
    for ax, var_name in zip(axs[1:], var_list):
        data = ds[var_name].isel(sample=sample_slice, viewing_direction=view_inds)
        data = data.where(ds.los.isel(sample=sample_slice) != 180)
        mesh = ax.pcolormesh(time_edges, y_edges, data.values, shading='auto', cmap='viridis')
        fig.colorbar(mesh, ax=ax, label=var_name, orientation="vertical")
        ax.set_ylabel("Viewing Dir.")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Final adjustments
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_xlabel("Time (UTC)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename
    ship_time_str = str(ship_time).replace(":", "-")
    safe_cat = category.replace(" ", "_")
    plt.savefig(f"P:\\data\\wedel\\quick_analysis\\ship_{ship_time_str}_{safe_cat}_{i}.png")
    plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Extract the data
  # Shape: (viewing_direction, sample)

time_slice = slice(1500, 5000)  # Adjust as needed
data = ds["a[RING]"].isel(sample= time_slice).where((ds["a[O4]"].isel(sample=time_slice) > 1390) & (ds.los.isel(sample=time_slice) != 180)
).values  # 2D array
time = ds["datetime"].isel(sample= time_slice).values  # 1D array
view_dirs = ds["viewing_direction"].values  # 1D array

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

# Plot using imshow
im = ax.imshow(
    data,
    aspect='auto',
    extent=[time[0], time[-1], view_dirs[0], view_dirs[-1]],
    origin='lower',
    cmap='viridis',  # Adjust as needed
)

# Format time axis
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
fig.autofmt_xdate()

# Labels and colorbar
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Viewing Direction")
ax.set_title("RMS Time Series")
fig.colorbar(im, ax=ax, label="a[NO2]")

plt.tight_layout()
plt.show()


# %%

plt.plot(
    ds["a[O4]"].isel(sample=slice(4000, 6000)).mean("sample"), ds.viewing_direction.values, label="t_expos = 0.1s")

plt.xlabel("a[O4]")
plt.ylabel("Viewing Direction")
plt.legend()
# %%
ds_img["t_expos"].plot()

# %%
# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

window_samples = 200  # ~2 minutes window (adjust as needed)

view_inds = np.arange(0, 41)  # for pcolormesh
line_views = range(6, 22, 3)  # for line plot

all_ships = [(t, 'large size') for t in large_ships] + \
            [(t, 'medium size') for t in mid_ships] + \
            [(t, 'small size') for t in small_ships]

all_ships.sort(key=lambda x: x[0])
datetimes = ds["datetime"].values

for i, (ship_time, category) in enumerate(all_ships):
    # Find nearest sample index
    closest_idx = np.argmin(np.abs(datetimes - ship_time))
    start_idx = max(closest_idx - window_samples, 0)
    end_idx = min(closest_idx + window_samples, len(datetimes))
    sample_slice = slice(start_idx, end_idx)
    sample_slice_short = slice(start_idx, start_idx +50)
    time_window = datetimes[sample_slice]
    time_numeric = mdates.date2num(time_window)

    # Calculate bin edges
    time_edges = np.zeros(len(time_numeric) + 1)
    time_edges[1:-1] = (time_numeric[1:] + time_numeric[:-1]) / 2
    time_edges[0] = time_numeric[0] - (time_numeric[1] - time_numeric[0]) / 2
    time_edges[-1] = time_numeric[-1] + (time_numeric[-1] - time_numeric[-2]) / 2
    y = view_inds
    y_edges = np.append(y, y[-1] + 1)

    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))

    # --- Line Plot ---
    for j in line_views:
        axs[0].plot(
            time_window,
            ds["a[NO2]"].isel(sample=sample_slice, viewing_direction=j).where(ds.los.isel(sample=sample_slice, viewing_direction=j) != 180)-ds["a[NO2]"].isel(sample=sample_slice_short, viewing_direction=j).where(ds.los.isel(sample=sample_slice_short, viewing_direction=j) != 180).mean(dim="sample"),
            label=f"VD {j}"
        )

    # Add vlines for *all* ships in the window
    for other_time, other_cat in all_ships:
        if time_window[0] <= other_time <= time_window[-1]:
            axs[0].axvline(
                x=other_time,
                color={'large size': 'red', 'medium size': 'blue', 'small size': 'green'}[other_cat],
                linestyle='--',
                linewidth=2,
                alpha=0.5 if other_time != ship_time else 0.9
            )
    axs[0].set_ylabel("a[NO2]")
    axs[0].set_title(f"{category.capitalize()} Ship @ {str(ship_time)}")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.3)

    for other_time, other_cat in all_ships:
        if time_window[0] <= other_time <= time_window[-1]:
            axs[1].axvline(
                x=other_time,
                color={'large size': 'red', 'medium size': 'blue', 'small size': 'green'}[other_cat],
                linestyle='--',
                linewidth=2,
                alpha=0.5 if other_time != ship_time else 0.9
            )
    axs[1].plot(time_window,
            ds["a[NO2]"].isel(sample=sample_slice).sum(dim="viewing_direction").where(ds.los.isel(sample=sample_slice, viewing_direction=0) != 180)-
            ds["a[NO2]"].isel(sample=sample_slice_short).sum(dim="viewing_direction").where(ds.los.isel(sample=sample_slice_short, viewing_direction=0) != 180).mean(dim="sample"),
            label=f"Vertical Integration")
    axs[1].set_ylabel("a[NO2]")
    axs[1].set_title(f"{category.capitalize()} Ship @ {str(ship_time)}")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.3)

    # --- Heatmaps ---
    var_list = ["a[NO2]"]  # You can expand this list
    for ax, var_name in zip(axs[2:], var_list):
        data = ds[var_name].isel(sample=sample_slice, viewing_direction=view_inds)-ds[var_name].isel(sample=sample_slice_short, viewing_direction=view_inds).mean(dim="sample")
        data = data.where(ds.los.isel(sample=sample_slice) != 180)
        mesh = ax.pcolormesh(time_edges, y_edges, data.values, shading='auto', cmap='viridis')
        fig.colorbar(mesh, ax=ax, label=var_name, orientation="vertical")
        ax.set_ylabel("Viewing Dir.")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Final adjustments
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_xlabel("Time (UTC)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename
    ship_time_str = str(ship_time).replace(":", "-")
    safe_cat = category.replace(" ", "_")
    plt.savefig(f"P:\\data\\wedel\\quick_analysis\\ship_{ship_time_str}_{safe_cat}_{i}.png")
    plt.show()
# %%
path = r"P:\data\wedel\SLANT_25\APR2025"
ds = read_SC_file_imaging(path, f"250401", f"ID.NO2_VIS_upwind_rest")

ds = process_SC_img_data(ds, "time_series", start_idx=0, end_idx=None)
los_mask = ds.los == 180
masked_indices = np.argwhere(los_mask.values)  # Get (i, j) indices where condition is met

# Create a boolean mask of the same shape as ds.los
expanded_mask = np.ones_like(ds.los, dtype=bool)

# Expand the mask to include neighbors (up, down, left, right)
for i, j in masked_indices:
    for di in [-2,-1, 0, 1,2]:  # Row shift (-1, 0, +1)
        for dj in [-2,-1, 0, 1,2]:  # Column shift (-1, 0, +1)
            ni, nj = i + di, j + dj
            if 0 <= ni < ds.los.shape[0] and 0 <= nj < ds.los.shape[1]:  # Bounds check
                expanded_mask[ni, nj] = False
# %%
plt.imshow(ds["a[O4]"].where(expanded_mask), aspect="auto", origin="lower")

# %%
plt.plot(ds["viewing-azimuth-angle"].isel(sample=slice(10,600)))
# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

window_samples = 280  # ~2 minutes window (adjust as needed)

view_inds = np.arange(0, 41)  # for pcolormesh
line_views = range(6, 22, 3)  # for line plot

all_ships = [(t, 'large size') for t in large_ships] + \
            [(t, 'medium size') for t in mid_ships] + \
            [(t, 'small size') for t in small_ships]

all_ships.sort(key=lambda x: x[0])
datetimes = ds["datetime"].values

for i, (ship_time, category) in enumerate(all_ships):
    # Find nearest sample index
    closest_idx = np.argmin(np.abs(datetimes - ship_time))
    start_idx = max(closest_idx - int(window_samples/2), 0)
    end_idx = min(closest_idx + window_samples, len(datetimes))
    sample_slice = slice(start_idx, end_idx)
    sample_slice_short = slice(start_idx, start_idx +50)
    time_window = datetimes[sample_slice]
    time_numeric = mdates.date2num(time_window)

    # Calculate bin edges
    time_edges = np.zeros(len(time_numeric) + 1)
    time_edges[1:-1] = (time_numeric[1:] + time_numeric[:-1]) / 2
    time_edges[0] = time_numeric[0] - (time_numeric[1] - time_numeric[0]) / 2
    time_edges[-1] = time_numeric[-1] + (time_numeric[-1] - time_numeric[-2]) / 2
    y = view_inds
    y_edges = np.append(y, y[-1] + 1)

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 7))
    axs = [
    fig.add_axes([0.1, 0.65, 0.68, 0.3]),   # Top subplot
    fig.add_axes([0.1, 0.35, 0.85, 0.25]),  # Middle subplot
    fig.add_axes([0.1, 0.05, 0.85, 0.25])   # New bottom subplot
    ]
    for other_time, other_cat in all_ships:
        if time_window[0] <= other_time <= time_window[-1]:
            axs[0].axvline(
                x=other_time,
                color={'large size': 'red', 'medium size': 'blue', 'small size': 'green'}[other_cat],
                linestyle='--',
                linewidth=2,
                alpha=0.5 if other_time != ship_time else 0.9
            )
    axs[0].plot(time_window,
            ds["a[NO2]"].isel(sample=sample_slice).sum(dim="viewing_direction").where(ds.los.isel(sample=sample_slice, viewing_direction=0) != 180)-
            ds["a[NO2]"].isel(sample=sample_slice_short).sum(dim="viewing_direction").where(ds.los.isel(sample=sample_slice_short, viewing_direction=0) != 180).mean(dim="sample"),
            label=f"Vertically Integrated NO2")
    axs[0].set_ylabel("NO2 dSCD / #/cm$^2$")
    axs[0].set_title(f"{category.capitalize()} Ship @ {str(ship_time)[0:-3]} UTC")
    axs[0].legend()
    axs[0].set_xlim(time_window[0], time_window[-1])

    axs[0].grid(True, linestyle="--", alpha=0.3)

    # --- Heatmaps ---
    var_list = ["a[NO2]","a[O4]"]  # You can expand this list
    for ax, var_name in zip(axs[1:], var_list):
        data = ds[var_name].isel(sample=sample_slice, viewing_direction=view_inds)-ds[var_name].isel(sample=sample_slice_short, viewing_direction=view_inds).mean(dim="sample")
        data = data.where(ds.los.isel(sample=sample_slice) != 180)
        if var_name == "a[NO2]":
            mesh = ax.pcolormesh(time_edges, y_edges, data.values, vmin=data.values.min()/4, shading='auto', cmap='viridis')
        elif var_name == "a[O4]":
            mesh = ax.pcolormesh(time_edges, y_edges, data.values, vmin=data.values.min()/4, shading='auto', cmap='viridis')
        else:
            mesh = ax.pcolormesh(time_edges, y_edges, data.values, shading='auto', cmap='viridis')
        fig.colorbar(mesh, ax=ax, label=r"{} dSCD / #/cm$^2$".format(var_name), orientation="vertical")
        ax.set_yticks(np.arange(0, len(ds["los"]), 5), np.round(ds["los"].isel(sample=0).values[::5]-91,1), rotation=45)

        ax.set_ylabel("Viewing Elevation / °")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Final adjustments
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time (UTC)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename
    ship_time_str = str(ship_time).replace(":", "-")
    safe_cat = category.replace(" ", "_")
    plt.savefig(f"P:\\data\\wedel\\quick_analysis\\ship_{ship_time_str}_{safe_cat}_{i}.png")
    plt.show()
# %%
