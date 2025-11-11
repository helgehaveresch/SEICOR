#%%
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import process_SC_img_data
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import pandas as pd

#%%
date_str = "251016"
variable = "NO2"

slant_folder= r"P:\data\data_tmp\SLANT_25\OKT2025"
azimuths = [106., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220.,
        230., 240., 250., 260., 270., 311]
ds_slant = read_SC_file_imaging(slant_folder, date_str, mode="AD.NO2_VIS")
ds_slant = process_SC_img_data(ds_slant, structure="image_predefined", vaa_reference=azimuths)
variable_str = f"a[{variable}]"
title = f"Hemispheric Scan {variable}"
image_num = 2

ds_image = ds_slant.isel(image=image_num)
s_freq = int(len(ds_slant["viewing-azimuth-angle"])/25)
if s_freq == 0:
    s_freq = 1

date = ds_image["datetime"].values[np.where(~pd.isna(ds_image.datetime.values))[0][0]].astype('datetime64[D]')
time = ds_image["time_str"].values[np.where(~pd.isna(ds_image.time_str.values))[0][0]]
los = ds_image["los"].values[np.where(~pd.isna(ds_image["los"].values))[0][0]] - 90

plt.figure(figsize=(10,5))
plt.imshow(ds_image[variable_str].T, aspect="auto", origin="lower")
plt.ylabel("Viewing Elevation Angle / °")
plt.xlabel("Viewing Azimuth Angle / °")
plt.xticks(np.arange(0, len(ds_image["viewing-azimuth-angle"]), s_freq), ds_image["viewing-azimuth-angle"].values[::s_freq], rotation=45)
plt.yticks(np.arange(0, len(los),2), [f"{v:.1f}" for v in los[::2]], rotation=45)
plt.title( "{} on {} at {} UTC".format(title,date, time))
plt.tight_layout()
plt.colorbar()# %%

# %%
save_dir = r"P:\data\data_tmp\SEICOR_Hemisphere"
os.makedirs(save_dir, exist_ok=True)

title = f"Hemispheric Scan {variable}"
n_images = ds_slant.sizes.get("image", 0)
s_freq = max(1, int(len(ds_slant["viewing-azimuth-angle"]) / 25))

for image_num in range(n_images):
    plt.figure(figsize=(10,5))
    ds_image = ds_slant.isel(image=image_num)
    date = ds_image["datetime"].values[np.where(~pd.isna(ds_image.datetime.values))[0][0]].astype('datetime64[D]')
    time = ds_image["time_str"].values[np.where(~pd.isna(ds_image.time_str.values))[0][0]]
    los = ds_image["los"].values[np.where(~pd.isna(ds_image["los"].values))[0][0]] - 90 

    plt.figure(figsize=(10,5))
    plt.imshow(ds_image[variable_str].T, aspect="auto", origin="lower")
    plt.ylabel("Viewing Elevation Angle / °")
    plt.xlabel("Viewing Azimuth Angle / °")
    plt.xticks(np.arange(0, len(ds_image["viewing-azimuth-angle"]), s_freq), ds_image["viewing-azimuth-angle"].values[::s_freq], rotation=45)
    plt.yticks(np.arange(0, len(los),2), [f"{v:.1f}" for v in los[::2]], rotation=45)
    plt.title( "{} on {} at {} UTC".format(title,date, time))
    plt.tight_layout()
    plt.colorbar()# %%
    time_str_safe = time.replace(":", "-").replace(" ", "_")
    variable_str_safe = variable_str.replace("[", "").replace("]", "").replace("a", "")

    fname = f"{date}_{time_str_safe}_image_{image_num}_{variable_str_safe}.png"
    save_path = os.path.join(save_dir,variable_str_safe , fname)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure directory exists
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {save_path}")
# %%
import imageio
from pathlib import Path

date_filter = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"  # example; change to desired date or None
# build list of PNGs ending with O4.png and create a GIF
save_dir = r"P:\data\data_tmp\SEICOR_Hemisphere\{}".format(variable)
out_gif = Path(save_dir) / "{}_{}_images.gif".format(variable, date_filter)

# Set date_filter to a string like "2025-10-21" (matches filenames that contain the date).
# Set to None to use all dates.

files = sorted(Path(save_dir).glob("*{}.png".format(variable)))

# filter by date string in filename if requested
if date_filter:
    # match either the literal date substring or date with no dashes (just in case)
    date_no_dash = date_filter.replace("-", "")
    files = [f for f in files if (date_filter in f.name) or (date_no_dash in f.name)]

if len(files) == 0:
    print(f"No files matching '*O4.png' (and date={date_filter}) found in {save_dir}")
else:
    images = []
    for f in files:
        try:
            images.append(imageio.imread(str(f)))
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    if images:
        # fps controls speed; change duration or fps as desired
        imageio.mimsave(str(out_gif), images, fps=2)
        print(f"Saved GIF: {out_gif}")


# %%

