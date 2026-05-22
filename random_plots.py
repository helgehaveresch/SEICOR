#%%
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
#%%
# Font sizes
LABEL_FS = 14
TICK_FS = 12
TITLE_FS = 16
CB_FS = 12
file_path = r"D:\SEICOR\plumes_5\plumes_250328\plume_014_t_20250328_094448_mmsi_636021013.nc"
ds = xr.open_dataset(file_path)
# %%



LABEL_FS = 15
TICK_FS = 17
TITLE_FS = 17
CB_FS = 14





idx_min = 636
idx_max = 1175
arr = ds['no2_extended'].values[:,idx_min:idx_max] -  ds['no2_enhancement_interp_extended'].values[:,idx_min:idx_max]# limit to central VEA range for better visualization
times = pd.to_datetime(ds['times_plume_extended'].values[idx_min:idx_max] )
vea = np.asarray(ds['vea'].values)

if arr.ndim != 2:
    raise ValueError('expected 2D array for NO2')
if arr.shape[0] == len(times) and arr.shape[1] == len(vea):
    arr = arr.T

# edges for pcolormesh
xnum = mdates.date2num(times.to_pydatetime())
dx = np.median(np.diff(xnum)) if len(xnum) > 1 else 1.0
xedges = np.concatenate((xnum - dx/2.0, [xnum[-1] + dx/2.0]))
dy = np.median(np.diff(vea)) if len(vea) > 1 else 1.0
yedges = np.concatenate((vea - dy/2.0, [vea[-1] + dy/2.0]))

fig, ax = plt.subplots(figsize=(8,5))
# set color limits from the underlying ds['no2_extended'] (same slice used above)
try:
    no2_slice = ds['no2_extended'].values[:,idx_min:idx_max]
    vmin = float(np.nanmin(no2_slice))
    vmax = float(np.nanmax(no2_slice))
except Exception:
    vmin, vmax = None, None

pcm = ax.pcolormesh(xedges, yedges, arr, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)

# fewer x-ticks (AutoDateLocator with limited ticks)
ax.xaxis_date()
duration = times.max() - times.min()
hours = duration.total_seconds() / 3600.0
locator = mdates.MinuteLocator(interval=2)
fmt = '%H:%M'
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
fig.autofmt_xdate()

# VEA ticks: choose up to 6 ticks and format with one decimal
n_yticks = min(6, len(vea))
idxs = np.unique(np.round(np.linspace(0, len(vea)-1, n_yticks)).astype(int))
yticks = vea[idxs]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{v:.1f}°" for v in yticks])
ax.tick_params(axis='both', labelsize=TICK_FS)

ax.set_xlabel('Time (UTC)', fontsize=LABEL_FS)
ax.set_ylabel('VEA / °', fontsize=LABEL_FS)
# set title using filename and optional timestamp/mmsi from attributes
try:
    tattr = ds.attrs.get('t', None)
    mmsi = ds.attrs.get('mmsi', None)
    if tattr is not None:
        tstr = pd.to_datetime(tattr).strftime('%Y-%m-%d %H:%M:%S')
        title = f"NO$_2$ background"
except Exception:
    title = Path(file_path).name
ax.set_title(title, fontsize=TITLE_FS)
fig.autofmt_xdate()
# Use ScalarFormatter to obtain exponent offset and move it into the label
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((-3, 3))
cbar = fig.colorbar(pcm, ax=ax, format=fmt)
plt.draw()
offset_text = ''
try:
    offset_text = cbar.ax.yaxis.get_offset_text().get_text()
except Exception:
    offset_text = ''

label = r'NO$_{2}$ backg. dSCD'
if offset_text:
    # remove multiplication sign (×) from offset text before using
    offset_clean = offset_text.replace('\\times', '').replace('\u00d7', '').strip()
    # use proper mathtext for units: molec cm^-2
    cbar.set_label(f"{label} / {offset_clean} " + r"$\mathrm{molec.}\,\mathrm{cm}^{-2}$", fontsize=CB_FS)
    try:
        cbar.ax.yaxis.get_offset_text().set_visible(False)
        print("removed offset text")
    except Exception:
        pass
else:
    cbar.set_label(label, fontsize=CB_FS)

# ensure colorbar tick labels match figure tick font size
cbar.ax.tick_params(labelsize=TICK_FS)

# shift colorbar slightly to the left to improve spacing
try:
    pos = cbar.ax.get_position()
    shift = 0.6
    new_pos = [pos.x0 - shift, pos.y0, pos.width, pos.height]
    cbar.ax.set_position(new_pos)
except Exception:
    pass
plt.tight_layout()
plt.show()
ds.close()
# %%
