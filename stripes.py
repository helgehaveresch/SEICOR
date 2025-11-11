#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_pc_file_imaging
from doas_tools.file_handling import read_img
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_IMG import process_img_data
from imaging_tools.process_SC import correct_destriped_for_noon_reference, find_min_zenith_dim0
import glob
import shutil
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta

# Paths
base_folder = r"Q:\BREDOM\SEICOR\IMPACT\pc"
month_folder = "JUL2025"
folder = os.path.join(base_folder, month_folder, "origs")
folder_dst = os.path.join(base_folder, month_folder)
reference_file = os.path.join(folder, "reference_selection.txt")

# Helper to get month folder from reference date
def get_month_folder(ref_str):
    # ref_str is like "250406"
    month_lookup = {
        "01": "JAN2025", "02": "FEB2025", "03": "MAE2025", "04": "APR2025",
        "05": "MAI2025", "06": "JUN2025", "07": "JUL2025", "08": "AUG2025",
        "09": "SEP2025", "10": "OKT2025", "11": "NOV2025", "12": "DEZ2025"
    }
    month_num = ref_str[2:4]
    return month_lookup.get(month_num, "")
#%%
plt.rc('axes', titlesize=22)      # Title font size
plt.rc('axes', labelsize=20)      # X and Y label font size
plt.rc('xtick', labelsize=18)     # X tick label font size
plt.rc('ytick', labelsize=18)     # Y tick label font size
plt.rc('legend', fontsize=18)     # Legend font size
#%%
date_str = r"250401"
file_str = r"_D.PC"
img_file_str = r"_D.IMG"
#ds_pc = read_pc_file_imaging(r"P:\data\data_tmp\pc\APR2025", date_str, file_str)
#intensity = ds_pc.counts_per_second.max(dim="column")
#ds_img = read_img(r"P:\data\data_tmp\org\APR2025\{}{}".format(date_str, img_file_str))
#ds_img_binned = process_img_data(ds_img, path_to_settings= r"C:\Users\hhave\Documents\Promotion\scripts\SEICOR\proc_settings.yaml")
#intensity_img = ds_img.counts_raw.max(dim="column")
#intensity_binned = ds_img_binned.counts_binned.max(dim="column")
ds_SC = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\APR2025", date_str, f"SD.NO2_VIS_fix_destriped")
ds_SC =ds_SC.isel(dim_0=slice(20, ds_SC.sizes['dim_0']-20))
#%%
#adjust size
plt.figure(figsize=(12,6))
plt.imshow(ds_SC["a[O4]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC.datetime.values]
ticks = [i for i, t in enumerate(ds_SC.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
plt.xlabel("Time UTC")
plt.ylabel("Image Row")
plt.colorbar(label= "O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
#%%
plt.figure(figsize=(12,6))
plt.imshow(ds_SC["rms"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC.datetime.values]
ticks = [i for i, t in enumerate(ds_SC.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
plt.xlabel("Time UTC")
plt.ylabel("Image Row")
plt.colorbar(label= "O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
# %%
integrated = ds_SC["a[O4]"].sum(dim="dim_0")

plt.figure()
plt.plot(integrated)
plt.xlabel("Image Row")
plt.ylabel("Integrated O4")
plt.title("O4 Integrated Over Time")
plt.show()
#%%
dates_list = ["250415", "250420", "250421"]
ds_SC_list = ["SD.NO2_VIS_zenith_switch_1504","SD.NO2_VIS_zenith_switch_2004", "SD.NO2_VIS_zenith_switch_2104"]
ref_dates = ["250415", "250420", "250421"]
linestyles = ['-', '--', ':', '-.']
colours = ['b', 'orange', 'c']
SC_list = []
SC_list_2 = []
SC_list_3 = []

for ds_SC_name in ds_SC_list:
    for date_str in dates_list:
        ds_SC = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\APR2025", date_str, ds_SC_name)
        ds_SC = ds_SC.sel(viewing_direction=slice(7, 42))
        ds_SC_2 = ds_SC.sel(viewing_direction=slice(7, 42), dim_0=slice(100, ds_SC.sizes['dim_0']-100)) 
        ds_SC_3 = ds_SC.sel(viewing_direction=slice(7, 42), dim_0=slice(250, ds_SC.sizes['dim_0']-250))
        SC_list.append(ds_SC)
        SC_list_2.append(ds_SC_2)
        SC_list_3.append(ds_SC_3)
#%%
variables=["a[O4]", "a[NO2]", "a[H2O]", "a[RING]", "a[Offset]", "rms"]
lower_limits_ax1 = [1800, 0.8e16, 0, -0.025, -0.012, 0.00065, -0.055]
upper_limits_ax1 = [5000, 6e16, 4e23, 0.0075, 0.0075, 0.0016, 0.045]
lower_limits_ax2 = [-400, -3e15, -7e22, -0.005, -0.01, -0.0001, -0.007]
upper_limits_ax2 = [400, 3e15, 7e22, 0.005, 0.01, 0.0001, 0.007]
l=0
for variab in variables:
    i=0
    k=0
    for ds_SC_name in ds_SC_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))

        j=0
        for date_str in dates_list:
            linestyle = linestyles[0]
            colour = colours[j]
            window = 10  # set your window size
            integrated = SC_list[i][variab].mean(dim="dim_0")
            std = SC_list[i][variab].std(dim="dim_0")/np.sqrt(SC_list[i].sizes['dim_0'])
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            variability = integrated - rolling_mean
            #integrated_2 = SC_list_2[i]["a[O4]"].mean(dim="dim_0")
            #rolling_mean_2 = integrated_2.rolling(viewing_direction=window, center=True).mean()
            #variability_2 = integrated_2 - rolling_mean_2
            #integrated_3 = SC_list_3[i]["a[O4]"].mean(dim="dim_0")
            #rolling_mean_3 = integrated_3.rolling(viewing_direction=window, center=True).mean()
            #variability_3 = integrated_3 - rolling_mean_3
            #plt.plot(variability, SC_list[i]["viewing_direction"], label=f"meas: {date_str} ref: {ref_dates[k]}", linestyle=linestyle, color=colour)
            #plt.fill_betweenx(SC_list[i]["viewing_direction"], variability - std, variability + std, color=colour, alpha=0.3,)
            #plt.plot(variability_2, SC_list_2[i]["viewing_direction"], label=f"meas: {date_str} ref: {ref_dates[k]}", linestyle=linestyles[1], color=colour)
            #plt.plot(variability_3, SC_list_3[i]["viewing_direction"], label=f"meas: {date_str} ref: {ref_dates[k]}", linestyle=linestyles[2], color=colour)
            ax1.plot(integrated, SC_list[i]["viewing_direction"], label=f"meas: {date_str} ref: {ref_dates[k]}", linestyle=linestyle, color=colour)
            ax1.fill_betweenx(SC_list[i]["viewing_direction"], integrated - std, integrated + std, color=colour, alpha=0.3)
            ax1.plot(rolling_mean, SC_list[i]["viewing_direction"], linestyle="--", color="k")
            ax1.set_xlabel(f"Mean {variab}")
            ax1.set_ylabel("Image Row")
            ax1.set_xlim(lower_limits_ax1[l], upper_limits_ax1[l])
            ax1.grid()
            ax1.set_title(f"{variab} Time Average , Rolling Mean")
            ax1.legend()
            ax2.plot(variability, SC_list[i]["viewing_direction"], linestyle=linestyle, color=colour)
            ax2.fill_betweenx(SC_list[i]["viewing_direction"], variability - std, variability + std, color=colour, alpha=0.3)
            ax2.set_xlabel(f"Variability {variab}")
            ax2.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
            ax2.grid()
            ax2.set_title(f"{variab} (Time Average - Rolling Mean)")
            #rotate xticks
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
            i+=1
            j+=1 
        #plt.ylabel("Image Row")
        #plt.xlabel(f"Mean {variab}")
        #plt.xlim(lower_limits[l], upper_limits[l])
        #plt.grid()
        #plt.title(f"{variab} Integrated Over Time")
        #plt.legend()
        #plt.show()
        plt.tight_layout()
        plt.show()
        k+=1

    l+=1

#%%

num_types = len(ds_SC_list)      # 3
num_vars = len(variables)        # 5

fig, axes = plt.subplots(num_types, num_vars, sharey='row', sharex='col', figsize=(num_vars*5, num_types*4))

for k, ds_SC_name in enumerate(ds_SC_list):        # rows: SC types
    for l, variab in enumerate(variables):          # columns: variables
        ax = axes[k, l]
        i = k * len(dates_list)  # Index for SC_list
        for j, date_str in enumerate(dates_list):
            linestyle = linestyles[0]
            colour = colours[j]
            window = 10
            integrated = SC_list[i][variab].mean(dim="dim_0")
            std = SC_list[i][variab].std(dim="dim_0") / np.sqrt(SC_list[i].sizes['dim_0'])
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            variability = integrated - rolling_mean

            # Plot integrated, std, rolling mean
            ax.plot(integrated, SC_list[i]["viewing_direction"], label=f"meas.: {date_str}", linestyle=linestyle, color=colour)
            ax.fill_betweenx(SC_list[i]["viewing_direction"], integrated - std, integrated + std, color=colour, alpha=0.3)
            ax.plot(rolling_mean, SC_list[i]["viewing_direction"], linestyle="--", color="k")
            ax.set_xlim(lower_limits_ax1[l], upper_limits_ax1[l])
            ax.grid()
            i += 1

        if k == num_types - 1:
            ax.set_xlabel(f"{variab}")
        if k == 0:  
            ax.set_title(f"{variab}")
        if l == 0:
            ax.set_ylabel("Image Row")
        #ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

handles, labels = axes[0, 0].get_legend_handles_labels()

fig_legend = plt.figure(figsize=(4, 2))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')
legend = ax_legend.legend(handles, labels, loc='center')
plt.show(block=True)
#%%

num_types = len(ds_SC_list)      # 3
num_vars = len(variables)        # 5

integrated_matrix_ref_list = []
fig, axes = plt.subplots(num_types, num_vars, sharey='row', sharex='col', figsize=(num_vars*5, num_types*4))

for k, ds_SC_name in enumerate(ds_SC_list):         # rows: SC types
    integrated_matrix = []
    for l, variab in enumerate(variables):          # columns: variables
        ax = axes[k, l]
        i = k * len(dates_list)  # Index for SC_list
        integrated_matrix__single_days = []
        for j, date_str in enumerate(dates_list):
            linestyle = linestyles[0]
            colour = colours[j]
            window = 10
            integrated = SC_list[i][variab].mean(dim="dim_0")
            std = SC_list[i][variab].std(dim="dim_0") / np.sqrt(SC_list[i].sizes['dim_0'])
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            variability = integrated - rolling_mean
            # Plot integrated, std, rolling mean
            ax.plot(variability, SC_list[i]["viewing_direction"], label=f"meas.: {date_str}", linestyle=linestyle, color=colour)
            #ax.fill_betweenx(SC_list[i]["viewing_direction"], variability - std, variability + std, color=colour, alpha=0.3)
            ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
            ax.grid()
            i += 1
            integrated_matrix__single_days.append(variability[~np.isnan(variability)])
        if k == num_types - 1:
            ax.set_xlabel(f"{variab}")
        if k == 0:  
            ax.set_title(f"{variab}")
        if l == 0:
            ax.set_ylabel("Image Row")
        #ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        integrated_matrix.append(np.concatenate(integrated_matrix__single_days, axis=0))
    integrated_matrix_ref_list.append(integrated_matrix)

plt.tight_layout()
plt.show()

handles, labels = axes[0, 0].get_legend_handles_labels()

fig_legend = plt.figure(figsize=(4, 2))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')
legend = ax_legend.legend(handles, labels, loc='center')
plt.show(block=True)
#%%


for i, date in enumerate(ref_dates):
    # Stack as 2D array: shape (num_vars, num_points)
    integrated_matrix = np.array(integrated_matrix_ref_list[i])
    # Calculate covariance matrix
    corr_matrix = np.corrcoef(integrated_matrix)

    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix,vmin=-1,vmax=1, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation')
    plt.xticks(np.arange(len(variables)), variables, rotation=45, fontsize=14)
    plt.yticks(np.arange(len(variables)), variables, fontsize=14)
    plt.title(f'Correlation Matrix, ref {date}')
    plt.tight_layout()
    plt.show()

#%%
#plt.plot(ds_pc.counts_per_second.mean(dim="viewing_direction").mean(dim="sample"))
#plt.show()
#plt.plot(ds_img.counts_raw.mean(dim="row").mean(dim="sample"))
#plt.show()
#diff = ds_img.counts_raw.mean(dim="row").mean(dim="sample") - ds_pc.counts_per_second.mean(dim="viewing_direction").mean(dim="sample") *0.3
#
#plt.plot(diff)
# %%


##%%
#plt.figure(figsize=(10,5))
#plt.imshow(intensity, origin="lower", aspect="auto")
#plt.xticks(ticks=np.arange(len(ds_pc.t_0)), labels = [str(t)[11:19] for t in ds_pc.t_0.values], rotation=45)
#plt.title("Intensity (counts/s) of reference")
#plt.colorbar()
#plt.show()
##%%
#plt.figure(figsize=(10,5))
#plt.imshow(intensity_binned.T, origin="lower", aspect="auto")
#plt.xticks(ticks=np.arange(len(ds_img_binned.t_0)), labels = [str(t)[11:19] for t in ds_img_binned.t_0.values], rotation=45)
#plt.title("Intensity (counts) of reference")
#plt.colorbar()
#plt.show()

# %%
date_list_orig = ['250401', '250402', '250403', '250417', '250418', '250421', '250423', '250430', '250520', '250522', '250524', '250527', '250604', '250712', '250812', '250824', '250905', '250915', '250930', '251001']

date_list = ['250712', '250812', '250824', '250905', '250915', '250930', '251001']

import shutil
folder = r"Q:\BREDOM\SEICOR\IMPACT\pc"  # Change to your folder path
folder_dst = r"E:\data\pc"  # Change to your destination folder path

for date in date_list:
    orig_file = f"{date}ID.PC"
    new_prefix = f"{date}ID.PC"
    month_str = get_month_folder(date)
    base_folder = os.path.join(folder, month_str)
    dst_folder = os.path.join(folder_dst, month_str)

    for filename in os.listdir(base_folder):
        if filename.startswith(orig_file):
            new_filename = filename.replace(orig_file, new_prefix, 1)
            os.makedirs(dst_folder, exist_ok=True)
            shutil.copy(os.path.join(base_folder, filename), os.path.join(dst_folder, new_filename))
            print(f"Renamed: {filename} → {new_filename}")
# %%

src_folder = r"Q:\BREDOM\SEICOR\IMPACT\pc\MAI2025"         # Source folder
dst_folder = os.path.join(src_folder, "origs")      # Destination folder

os.makedirs(dst_folder, exist_ok=True)

# Find all files ending with _D.PC?? (e.g., _D.PC01, _D.PC99, etc.)
pattern = os.path.join(src_folder, "*_D.PC??")
for filepath in glob.glob(pattern):
    filename = os.path.basename(filepath)
    shutil.copy2(filepath, os.path.join(dst_folder, filename))
    print(f"Copied: {filename} → {dst_folder}")
# %%
from datetime import datetime, timedelta

# Define the year and month
year = 2025
month = 7

# Reference dates in yymmdd format
reference_dates = ["250401", "250406", "250413", "250420", "250429", "250506", "250526",   
                   "250616", "250703", "250812", "250821", "250902", "250903", "250906", "250911", "250923", "251001"]

# Convert reference dates to datetime objects
ref_days = [datetime.strptime(d, "%y%m%d") for d in reference_dates]

# Get all days in the month
start_date = datetime(year, month, 1)
if month == 12:
    end_date = datetime(year + 1, 1, 1)
else:
    end_date = datetime(year, month + 1, 1)
delta = end_date - start_date

# Prepare output lines
lines = []
for i in range(delta.days):
    day = start_date + timedelta(days=i)
    day_str = day.strftime("%y%m%d")
    # Find the highest reference date that is <= current day
    ref_idx = 0
    for j, ref_day in enumerate(ref_days):
        if day >= ref_day:
            ref_idx = j
        else:
            break
    line = f"{day_str}\t{reference_dates[ref_idx]}"
    lines.append(line)

# Write to file
with open(r"Q:\BREDOM\SEICOR\IMPACT\pc\JUL2025\origs\reference_selection.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

#%%
# Read the reference selection file
with open(reference_file, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    day_str, ref_str = line.split('\t')
    orig_file = f"{ref_str}_D.PC"
    new_file = f"{day_str}_D.PC"
    # Only proceed if a file named like the new file already exists in the origs folder
    if any(f.startswith(new_file) for f in os.listdir(folder)):
        # Determine the correct origs folder for the reference file
        ref_month_folder = get_month_folder(ref_str)
        search_folder = os.path.join(base_folder, ref_month_folder, "origs")
        for filename in os.listdir(search_folder):
            #print(filename)
            if filename.startswith(orig_file):
                new_filename = filename.replace(orig_file, new_file, 1)
                shutil.copy(
                    os.path.join(search_folder, filename),
                    os.path.join(folder_dst, new_filename)
                )
                print(f"Copied and renamed: {filename} → {new_filename}")
# %%

import os
import shutil
import glob

src_folder = r"Q:\BREDOM\SEICOR\IMPACT\pc\SEP2025"         # Source folder
dst_folder = r"Q:\BREDOM\SEICOR\IMPACT\pc\SEP2025\origs"      # Destination folder
#dst_folder = r"E:\SEICOR_destriped\pc\SEP2025"   
os.makedirs(dst_folder, exist_ok=True)

# Find all files ending with _D.PC?? (e.g., _D.PC01, _D.PC99, etc.)
pattern = os.path.join(src_folder, "*_D.PC??")
for filepath in glob.glob(pattern):
    filename = os.path.basename(filepath)
    # Replace SD.PC with KD.PC in the filename
    new_filename = filename.replace("_D.PC", "_D.PC", 1)
    shutil.copy2(filepath, os.path.join(dst_folder, new_filename))
    print(f"Copied and renamed: {filename} → {new_filename} in {dst_folder}")
#%%

date_str = r"250907"

ds_SC = read_SC_file_imaging( r"E:\SEICOR_destriped\SLANT_25\SEP2025", date_str, f"SD.NO2_VIS_fix_destriping_test_102_2")

ds_SC["a[O4]"].plot()

# %%
base_folder = r"E:\SEICOR_destriped\SLANT_25\JUL2025"
file_suffix = "SD.NO2_VIS_fix_destriped"

# Get all files ending with SD.NO2_VIS_fix_destriped in the folder
all_files = os.listdir(base_folder)
date_str_list = [f[:6] for f in all_files if f.endswith(file_suffix)]

# Remove duplicates and sort
date_str_list = sorted(set(date_str_list))

# Read all datasets into a list
ds_list = []
for date_str in date_str_list:
    ds = read_SC_file_imaging(base_folder, date_str, file_suffix)
    ds_list.append(ds)
    #SC_list_3.append(ds_SC_3)

# %%
variables = ["a[O4]", "a[NO2]", "a[H2O]", "a[RING]", "a[Offset]", "rms"]
lower_limits_ax2 = [-400, -3e15, -7e22, -0.005, -0.01, -0.0001, -0.007]
upper_limits_ax2 = [400, 3e15, 7e22, 0.005, 0.01, 0.0001, 0.007]
window = 10  # Rolling mean window

save_base = r"E:\SEICOR_destriped\stripe images"
month_folder = os.path.basename(base_folder)
save_folder = os.path.join(save_base, month_folder)
os.makedirs(save_folder, exist_ok=True)

for day_idx, ds in enumerate(ds_list):
    # --- Save the imshow image for a[O4] ---
    plt.figure(figsize=(12,6))
    plt.imshow(ds["a[O4]"], origin="lower", aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label= "O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
    save_path_img = os.path.join(save_folder, f"{date_str}_O4_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()



    ds = ds.isel(viewing_direction=slice(7, 42))
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):
        ax = axes[l]
        integrated = ds[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = integrated - rolling_mean
        ax.plot(variability, ds["viewing_direction"], color="b")
        ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        ax.set_title(f"{variab}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for Day {day_idx+1}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save figure
    date_str = date_str_list[day_idx]
    save_path = os.path.join(save_folder, f"{date_str}_variability.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
# %%
date_str = r"250907"
ds_SC = read_SC_file_imaging( r"E:\SEICOR_destriped\SLANT_25\SEP2025", date_str, f"SD.NO2_VIS_fix_destriping_test_102")

# %%
base_folder = r"E:\SEICOR_destriped\SLANT_25\APR2025"
file_suffix = "SD.NO2_VIS_fix_destriped"

# Get all files ending with SD.NO2_VIS_fix_destriped in the folder
all_files = os.listdir(base_folder)
date_str_list = [f[:6] for f in all_files if f.endswith(file_suffix)]

# Remove duplicates and sort
date_str_list = sorted(set(date_str_list))

variables = ["a[O4]", "a[NO2]", "a[H2O]", "a[RING]", "a[Offset]", "rms"]
lower_limits_ax2 = [-400, -3e15, -7e22, -0.005, -0.01, -0.0001, -0.007]
upper_limits_ax2 = [400, 3e15, 7e22, 0.005, 0.01, 0.0001, 0.007]
window = 5  # Rolling mean window

save_base = r"E:\SEICOR_destriped\stripe images"
month_folder = os.path.basename(base_folder)
save_folder = os.path.join(save_base, month_folder)
os.makedirs(save_folder, exist_ok=True)

for date_str in date_str_list:
    ds = read_SC_file_imaging(base_folder, date_str, file_suffix)

    # --- Save the imshow image for a[O4] ---
    plt.figure(figsize=(12,6))
    plt.imshow(ds["a[O4]"], origin="lower", aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
    save_path_img = os.path.join(save_folder, f"{date_str}_O4_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    # --- Variability plots ---
    ds_proc = ds.isel(viewing_direction=slice(7, 42))
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):
        ax = axes[l]
        integrated = ds_proc[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = integrated - rolling_mean
        stripe_variation = variability.std().item()
        ax.plot(variability, ds_proc["viewing_direction"], color="b")
        ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        if np.isnan(stripe_variation) or stripe_variation == 0:
            stripe_label = "0"
        else:
            sci = f"{stripe_variation:.2e}"       # e.g. "1.23e+04"
            mantissa, exp = sci.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp = int(exp)
            stripe_label = rf"${mantissa}\times10^{{{exp}}}$"

        ax.set_title(f"{variab}\nstripe var: {stripe_label}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder, f"{date_str}_variability.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
# %%


dates_to_test = ["250415", "250501", "250516", "250601", "250602", "250615", "250701", "250715"]   # yymmdd (5 dates to test)
reference_dates = ["250413", "250429", "250526", "250616", "250703", "250906", "250907", "250908"]


base_folder = r"Q:\BREDOM\SEICOR\IMPACT\stripe_test\SLANT_25"
file_suffix = "SD.NO2_VIS_fix_destriping_test"

variables = ["a[O4]", "a[NO2]", "a[H2O]", "a[RING]", "a[Offset]", "rms", "sh[Bezug]"]
lower_limits_ax2 = [-400, -3e15, -7e22, -2e-17, -0.01, -0.0001, -0.002]
upper_limits_ax2 = [400, 3e15, 7e22, 2e-17, 0.01, 0.0001, 0.002]
window = 5  # Rolling mean window

variation=[]
mean=[]
variation_rel=[]

for date_str in dates_to_test:
    variation_tmp=[]
    mean_tmp=[]
    variation_rel_tmp=[]
    for ref_date in reference_dates:
        variation_variab=[]
        mean_variab=[]
        variation_rel_variab=[]
        month_folder = get_month_folder(date_str)
        save_base = r"Q:\BREDOM\SEICOR\IMPACT\stripe_test\stripe_images"
        save_folder = os.path.join(save_base, month_folder)
        os.makedirs(save_folder, exist_ok=True)

        save_folders = [os.path.join(save_folder, "O4"), os.path.join(save_folder, "iterations"), os.path.join(save_folder, "rms"), os.path.join(save_folder, "shift"), os.path.join(save_folder, "variability"), os.path.join(save_folder, "mean"), os.path.join(save_folder, "variability_rel")]
        for sf in save_folders:
            os.makedirs(sf, exist_ok=True)

        ds = read_SC_file_imaging(f"{base_folder}\\{month_folder}", date_str, f"{file_suffix}_ref_{ref_date}")

        # --- Save the imshow image for a[O4] ---
        plt.figure(figsize=(12,6))
        plt.imshow(ds["a[O4]"], origin="lower", aspect="auto", interpolation="none")
        labels = [str(t)[11:16] for t in ds.datetime.values]
        ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
        labels_halfhour = [labels[i] for i in ticks]
        plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
        plt.xlabel("Time UTC")
        plt.ylabel("Image Row")
        plt.colorbar(label="O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
        save_path_img = os.path.join(save_folder, f"O4\\{date_str}_ref_{ref_date}_O4_image.png")
        plt.tight_layout()
        plt.savefig(save_path_img, dpi=200)
        plt.close()

        plt.figure(figsize=(12,6))
        plt.imshow(ds["it"], origin="lower", vmin=0, vmax=25, aspect="auto", interpolation="none")
        labels = [str(t)[11:16] for t in ds.datetime.values]
        ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
        labels_halfhour = [labels[i] for i in ticks]
        plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
        plt.xlabel("Time UTC")
        plt.ylabel("Image Row")
        plt.colorbar(label="Iterations")
        save_path_img = os.path.join(save_folder, f"iterations\\{date_str}_ref_{ref_date}_it_image.png")
        plt.tight_layout()
        plt.savefig(save_path_img, dpi=200)
        plt.close()

        im = plt.imshow(ds["rms"].values, origin="lower", aspect="auto", interpolation="none",
                        norm=LogNorm(vmin=0.0001, vmax=0.007), cmap="viridis")

        labels = [str(t)[11:16] for t in ds.datetime.values]
        ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
        labels_halfhour = [labels[i] for i in ticks]
        plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
        plt.xlabel("Time UTC")
        plt.ylabel("Image Row")
        plt.colorbar(im, label="RMS (log scale)")
        save_path_img = os.path.join(save_folder, f"rms\\{date_str}_ref_{ref_date}_rms_image.png")
        plt.tight_layout()
        plt.savefig(save_path_img, dpi=200)
        plt.close()

        plt.figure(figsize=(12,6))
        plt.imshow(ds["sh[Bezug]"], origin="lower", vmin = -0.2, vmax=0.2, aspect="auto", interpolation="none")
        labels = [str(t)[11:16] for t in ds.datetime.values]
        ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
        labels_halfhour = [labels[i] for i in ticks]
        plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
        plt.xlabel("Time UTC")
        plt.ylabel("Image Row")
        plt.colorbar(label="Shift")
        save_path_img = os.path.join(save_folder, f"shift\\{date_str}_ref_{ref_date}_sh_image.png")
        plt.tight_layout()
        plt.savefig(save_path_img, dpi=200)
        plt.close()

        # --- Variability plots ---
        ds_proc = ds.isel(viewing_direction=slice(7, 42))
        fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
        if len(variables) == 1:
            axes = [axes]
        for l, variab in enumerate(variables):

            ax = axes[l]
            integrated = ds_proc[variab].mean(dim="dim_0")
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            variability = integrated - rolling_mean
            stripe_variation = variability.std().item()
            ax.plot(variability, ds_proc["viewing_direction"], color="b")
            ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
            ax.set_xlabel(f"Variability {variab}")
            ax.grid()
            if l == 0:
                ax.set_ylabel("Image Row")
            if np.isnan(stripe_variation) or stripe_variation == 0:
                stripe_label = "0"
            else:
                sci = f"{stripe_variation:.2e}"       # e.g. "1.23e+04"
                mantissa, exp = sci.split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                exp = int(exp)
                stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
            variation_variab.append(stripe_variation)
            ax.set_title(f"{variab}\nstripe var: {stripe_label}")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        variation_tmp.append(variation_variab)
        plt.suptitle(f"Variability for {date_str}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_folder, f"variability\\{date_str}_ref_{ref_date}_variability.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        # --- Variability plots ---
        fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
        if len(variables) == 1:
            axes = [axes]
        for l, variab in enumerate(variables):

            ax = axes[l]
            integrated = ds_proc[variab].mean(dim="dim_0")
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            variability = (integrated - rolling_mean) / rolling_mean
            stripe_variation = variability.std().item()
            ax.plot(variability, ds_proc["viewing_direction"], color="b")
            #ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
            ax.set_xlabel(f"Variability {variab}")
            ax.grid()
            if l == 0:
                ax.set_ylabel("Image Row")
            if np.isnan(stripe_variation) or stripe_variation == 0:
                stripe_label = "0"
            else:
                sci = f"{stripe_variation:.2e}"       
                mantissa, exp = sci.split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                exp = int(exp)
                stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
            variation_rel_variab.append(stripe_variation)
            ax.set_title(f"{variab}\nstripe var: {stripe_label}")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        variation_rel_tmp.append(variation_rel_variab)
        plt.suptitle(f"Variability for {date_str}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_folder, f"variability_rel\\{date_str}_ref_{ref_date}_variability_rel.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        # --- mean plots ---
        fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
        if len(variables) == 1:
            axes = [axes]
        for l, variab in enumerate(variables):

            ax = axes[l]
            integrated = ds_proc[variab].mean(dim="dim_0")
            rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
            integrated_mean = integrated.mean().item()
            ax.plot(integrated, ds_proc["viewing_direction"], color="b")
            #ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
            ax.set_xlabel(f"Variability {variab}")
            ax.grid()

            mean_variab.append(integrated_mean)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        mean_tmp.append(mean_variab)
        plt.suptitle(f"Variability for {date_str}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_folder, f"mean\\{date_str}_ref_{ref_date}_mean_rel.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    variation.append(variation_tmp)
    variation_rel.append(variation_rel_tmp)
    mean.append(mean_tmp)    
# %%



plt.figure(figsize=(10,6))
variation_arr = np.array(variation)  # shape (n_test_dates, n_reference_dates, n_variables)
x = np.arange(len(reference_dates))

for i, test_date in enumerate(dates_to_test):
    y = variation_arr[i, :, 0]  # change last index to choose another variable if needed
    plt.plot(x, y, marker='o', label=test_date)
plt.plot(x, variation_arr[:, :, 0].mean(axis=0), marker='o', label="Mean", linewidth=5)  # Mean across test dates
plt.xticks(ticks=x, labels=reference_dates, rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Variability O4")
#plt.title("Variability Comparison")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
variation_arr = np.array(variation)  # shape (n_test_dates, n_reference_dates, n_variables)
x = np.arange(len(reference_dates))

for i, test_date in enumerate(dates_to_test):
    y = variation_arr[i, :, 1]  # change last index to choose another variable if needed
    plt.plot(x, y, marker='o', label=test_date)
plt.plot(x, variation_arr[:, :, 1].mean(axis=0), marker='o', label="Mean", linewidth=5)  # Mean across test dates
plt.xticks(ticks=x, labels=reference_dates, rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Variability NO2")
#plt.title("Variability Comparison")
plt.legend()
plt.tight_layout()
plt.show()

mean_arr = np.array(mean)  # shape (n_test_dates, n_reference_dates, n_variables)

for i, test_date in enumerate(dates_to_test):
    y = mean_arr[i, :, 5]  # change last index to choose another variable if needed
    plt.plot(x, y, marker='o', label=test_date)
plt.plot(x, mean_arr[:, :, 5].mean(axis=0), marker='o', label="Mean", linewidth=5)  # Mean across test dates
plt.xticks(ticks=x, labels=reference_dates, rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Mean rms")
#plt.title("Variability Comparison")
plt.legend()
plt.tight_layout()
plt.show()

x = np.arange(len(dates_to_test))
for i, ref_date in enumerate(reference_dates):
    y = mean_arr[:, i, 6]  # change last index to choose another variable if needed
    plt.plot(x, y, marker='o', label=ref_date)
plt.plot(x, mean_arr[:, :, 6].mean(axis=1), marker='o', label="Mean", linewidth=5)  # Mean across test dates
plt.xticks(ticks=x, labels=dates_to_test, rotation=45)
plt.xlabel("Measurement date")
plt.ylabel("Mean shift")
#plt.title("Variability Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# %%

# only use this single reference date
ref_date = "250907"

# build list of existing measurement dates between 250326 and 250831
start = datetime.strptime("250326", "%y%m%d")
end = datetime.strptime("250831", "%y%m%d")
dates_to_test = []
d = start

base_folder = r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25"
file_suffix = "SD.NO2_VIS_fix_destriped"

variables = ["a[O4]", "a[NO2]", "a[H2O]", "a[RING]", "a[Offset]", "rms", "sh[Bezug]"]
lower_limits_ax2 = [-400, -3e15, -7e22, -2e-17, -0.01, -0.0001, -0.002]
upper_limits_ax2 = [400, 3e15, 7e22, 2e-17, 0.01, 0.0001, 0.002]
window = 5  # Rolling mean window

while d <= end:
    date_str = d.strftime("%y%m%d")
    month_folder = get_month_folder(date_str)
    search_folder = os.path.join(base_folder, month_folder)
    pattern = os.path.join(search_folder, f"{date_str}*{file_suffix}_ref_{ref_date}*")
    if os.path.isdir(search_folder) and glob.glob(pattern):
        dates_to_test.append(date_str)
    d += timedelta(days=1)

# now process only the found dates with the single reference date
variation = []
mean = []
variation_rel = []

for date_str in dates_to_test:
    variation_tmp = []
    mean_tmp = []
    variation_rel_tmp = []

    month_folder = get_month_folder(date_str)
    save_base = r"Q:\BREDOM\SEICOR\IMPACT\stripe_test\stripe_images"
    save_folder = os.path.join(save_base, month_folder)
    os.makedirs(save_folder, exist_ok=True)

    save_folders = [
        os.path.join(save_folder, "O4"),
        os.path.join(save_folder, "iterations"),
        os.path.join(save_folder, "rms"),
        os.path.join(save_folder, "shift"),
        os.path.join(save_folder, "variability"),
        os.path.join(save_folder, "mean"),
        os.path.join(save_folder, "variability_rel"),
    ]
    for sf in save_folders:
        os.makedirs(sf, exist_ok=True)

    # read dataset for this date and the single reference
    ds = read_SC_file_imaging(os.path.join(base_folder, month_folder), date_str, f"{file_suffix}_ref_{ref_date}")
    # --- Save the imshow image for a[O4] ---
    plt.figure(figsize=(12,6))
    plt.imshow(ds["a[O4]"], origin="lower", aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
    save_path_img = os.path.join(save_folder, f"O4\\{date_str}_ref_{ref_date}_O4_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    plt.figure(figsize=(12,6))
    plt.imshow(ds["it"], origin="lower", vmin=0, vmax=25, aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="Iterations")
    save_path_img = os.path.join(save_folder, f"iterations\\{date_str}_ref_{ref_date}_it_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    im = plt.imshow(ds["rms"].values, origin="lower", aspect="auto", interpolation="none",
                    norm=LogNorm(vmin=0.0001, vmax=0.007), cmap="viridis")

    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(im, label="RMS (log scale)")
    save_path_img = os.path.join(save_folder, f"rms\\{date_str}_ref_{ref_date}_rms_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    plt.figure(figsize=(12,6))
    plt.imshow(ds["sh[Bezug]"], origin="lower", vmin = -0.2, vmax=0.2, aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds.datetime.values]
    ticks = [i for i, t in enumerate(ds.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="Shift")
    save_path_img = os.path.join(save_folder, f"shift\\{date_str}_ref_{ref_date}_sh_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    # --- Variability plots ---
    ds_proc = ds.isel(viewing_direction=slice(7, 42))
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):

        ax = axes[l]
        integrated = ds_proc[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = integrated - rolling_mean
        stripe_variation = variability.std().item()
        ax.plot(variability, ds_proc["viewing_direction"], color="b")
        ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        if np.isnan(stripe_variation) or stripe_variation == 0:
            stripe_label = "0"
        else:
            sci = f"{stripe_variation:.2e}"       # e.g. "1.23e+04"
            mantissa, exp = sci.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp = int(exp)
            stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
        variation_tmp.append(stripe_variation)
        ax.set_title(f"{variab}\nstripe var: {stripe_label}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder, f"variability\\{date_str}_ref_{ref_date}_variability.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    # --- Variability plots ---
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):

        ax = axes[l]
        integrated = ds_proc[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = (integrated - rolling_mean) / rolling_mean
        stripe_variation = variability.std().item()
        ax.plot(variability, ds_proc["viewing_direction"], color="b")
        #ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        if np.isnan(stripe_variation) or stripe_variation == 0:
            stripe_label = "0"
        else:
            sci = f"{stripe_variation:.2e}"       
            mantissa, exp = sci.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp = int(exp)
            stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
        variation_rel_tmp.append(stripe_variation)
        ax.set_title(f"{variab}\nstripe var: {stripe_label}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder, f"variability_rel\\{date_str}_ref_{ref_date}_variability_rel.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    # --- mean plots ---
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):

        ax = axes[l]
        integrated = ds_proc[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        integrated_mean = integrated.mean().item()
        ax.plot(integrated, ds_proc["viewing_direction"], color="b")
        #ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()

        mean_tmp.append(integrated_mean)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder, f"mean\\{date_str}_ref_{ref_date}_mean_rel.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    variation.append(variation_tmp)
    variation_rel.append(variation_rel_tmp)
    mean.append(mean_tmp)
#%%
ref_date = "250703"

variation_2 = []
mean_2 = []
variation_rel_2 = []

for date_str in dates_to_test:
    variation_tmp = []
    mean_tmp = []
    variation_rel_tmp = []

    month_folder = get_month_folder(date_str)
    # use a different save base so images don't overwrite the first run
    save_base_2 = r"Q:\BREDOM\SEICOR\IMPACT\stripe_test\stripe_images_2"
    save_folder_2 = os.path.join(save_base_2, month_folder)
    os.makedirs(save_folder_2, exist_ok=True)

    save_folders_2 = [
        os.path.join(save_folder_2, "O4"),
        os.path.join(save_folder_2, "iterations"),
        os.path.join(save_folder_2, "rms"),
        os.path.join(save_folder_2, "shift"),
        os.path.join(save_folder_2, "variability"),
        os.path.join(save_folder_2, "mean"),
        os.path.join(save_folder_2, "variability_rel"),
    ]
    for sf in save_folders_2:
        os.makedirs(sf, exist_ok=True)

    # read dataset for this date and the single reference (keep same ref_date)
    ds_2 = read_SC_file_imaging(os.path.join(base_folder, month_folder), date_str, f"{file_suffix}_ref_{ref_date}")

    # --- Save the imshow image for a[O4] ---
    plt.figure(figsize=(12,6))
    plt.imshow(ds_2["a[O4]"], origin="lower", aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds_2.datetime.values]
    ticks = [i for i, t in enumerate(ds_2.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
    save_path_img = os.path.join(save_folder_2, f"O4\\{date_str}_ref_{ref_date}_O4_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    plt.figure(figsize=(12,6))
    plt.imshow(ds_2["it"], origin="lower", vmin=0, vmax=25, aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds_2.datetime.values]
    ticks = [i for i, t in enumerate(ds_2.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="Iterations")
    save_path_img = os.path.join(save_folder_2, f"iterations\\{date_str}_ref_{ref_date}_it_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    im = plt.imshow(ds_2["rms"].values, origin="lower", aspect="auto", interpolation="none",
                    norm=LogNorm(vmin=0.0001, vmax=0.007), cmap="viridis")

    labels = [str(t)[11:16] for t in ds_2.datetime.values]
    ticks = [i for i, t in enumerate(ds_2.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(im, label="RMS (log scale)")
    save_path_img = os.path.join(save_folder_2, f"rms\\{date_str}_ref_{ref_date}_rms_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    plt.figure(figsize=(12,6))
    plt.imshow(ds_2["sh[Bezug]"], origin="lower", vmin = -0.2, vmax=0.2, aspect="auto", interpolation="none")
    labels = [str(t)[11:16] for t in ds_2.datetime.values]
    ticks = [i for i, t in enumerate(ds_2.datetime.values) if str(t)[14:16] in ["00"]]
    labels_halfhour = [labels[i] for i in ticks]
    plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
    plt.xlabel("Time UTC")
    plt.ylabel("Image Row")
    plt.colorbar(label="Shift")
    save_path_img = os.path.join(save_folder_2, f"shift\\{date_str}_ref_{ref_date}_sh_image.png")
    plt.tight_layout()
    plt.savefig(save_path_img, dpi=200)
    plt.close()

    # --- Variability plots ---
    ds_proc_2 = ds_2.isel(viewing_direction=slice(7, 42))
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):
        ax = axes[l]
        integrated = ds_proc_2[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = integrated - rolling_mean
        stripe_variation = variability.std().item()
        ax.plot(variability, ds_proc_2["viewing_direction"], color="b")
        ax.set_xlim(lower_limits_ax2[l], upper_limits_ax2[l])
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        if np.isnan(stripe_variation) or stripe_variation == 0:
            stripe_label = "0"
        else:
            sci = f"{stripe_variation:.2e}"
            mantissa, exp = sci.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp = int(exp)
            stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
        variation_tmp.append(stripe_variation)
        ax.set_title(f"{variab}\nstripe var: {stripe_label}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder_2, f"variability\\{date_str}_ref_{ref_date}_variability.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    # --- Relative variability plots ---
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):
        ax = axes[l]
        integrated = ds_proc_2[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        variability = (integrated - rolling_mean) / rolling_mean
        stripe_variation = variability.std().item()
        ax.plot(variability, ds_proc_2["viewing_direction"], color="b")
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        if l == 0:
            ax.set_ylabel("Image Row")
        if np.isnan(stripe_variation) or stripe_variation == 0:
            stripe_label = "0"
        else:
            sci = f"{stripe_variation:.2e}"
            mantissa, exp = sci.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp = int(exp)
            stripe_label = rf"${mantissa}\times10^{{{exp}}}$"
        variation_rel_tmp.append(stripe_variation)
        ax.set_title(f"{variab}\nstripe var: {stripe_label}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder_2, f"variability_rel\\{date_str}_ref_{ref_date}_variability_rel.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    # --- mean plots ---
    fig, axes = plt.subplots(1, len(variables), sharey=True, figsize=(len(variables)*5, 5))
    if len(variables) == 1:
        axes = [axes]
    for l, variab in enumerate(variables):
        ax = axes[l]
        integrated = ds_proc_2[variab].mean(dim="dim_0")
        rolling_mean = integrated.rolling(viewing_direction=window, center=True).mean()
        integrated_mean = integrated.mean().item()
        ax.plot(integrated, ds_proc_2["viewing_direction"], color="b")
        ax.set_xlabel(f"Variability {variab}")
        ax.grid()
        mean_tmp.append(integrated_mean)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"Variability for {date_str}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_folder_2, f"mean\\{date_str}_ref_{ref_date}_mean_rel.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    variation_2.append(variation_tmp)
    variation_rel_2.append(variation_rel_tmp)
    mean_2.append(mean_tmp)

#%%
plt.figure(figsize=(10,6))
variation_arr = np.array(variation)  # shape (n_test_dates, n_reference_dates, n_variables)
variation_arr_2 = np.array(variation_2)
x = np.arange(len(dates_to_test))

# choose a reduced set of ticks (max 10 ticks)
def _select_ticks(labels, max_ticks=10):
    n = len(labels)
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        step = max(1, n // (max_ticks - 1))
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        idx = np.array(idx)
    return idx

tick_idx = _select_ticks(dates_to_test, max_ticks=10)
plt.plot(x, variation_arr_2[:, 0], marker='o', linewidth=2, label="ref 250703")  # variability O4 with different ref date
plt.plot(x, variation_arr[:, 0], marker='o', linewidth=2, label="ref 250907")  # variability O4
plt.xticks(ticks=tick_idx, labels=[dates_to_test[i] for i in tick_idx], rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Variability O4")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
tick_idx = _select_ticks(dates_to_test, max_ticks=10)
plt.plot(x, variation_arr_2[:, 1], marker='o', linewidth=2, label="ref 250703")  # variability NO2 with different ref date
plt.plot(x, variation_arr[:, 1], marker='o', linewidth=2, label="ref 250907")  # variability NO2
plt.xticks(ticks=tick_idx, labels=[dates_to_test[i] for i in tick_idx], rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Variability NO2")
plt.legend()
plt.tight_layout()
plt.show()

mean_arr = np.array(mean)  # shape (n_test_dates, n_reference_dates, n_variables)
mean_arr_2 = np.array(mean_2)
x = np.arange(len(dates_to_test))
plt.figure(figsize=(10,6))

tick_idx = _select_ticks(dates_to_test, max_ticks=10)

# prepare data for log scale (replace non-positive values with NaN to avoid log errors)
y_ref2 = np.array(mean_arr_2[:, 5], dtype=float)
y_ref1 = np.array(mean_arr[:, 5], dtype=float)
y_ref2 = np.where(y_ref2 > 0, y_ref2, np.nan)
y_ref1 = np.where(y_ref1 > 0, y_ref1, np.nan)

plt.figure(figsize=(10,6))
plt.semilogy(x, y_ref2, marker='o', linewidth=2, label="ref 250703")  # mean rms (log scale)
plt.semilogy(x, y_ref1, marker='o', linewidth=2, label="ref 250907")  # mean rms (log scale)
plt.xticks(ticks=tick_idx, labels=[dates_to_test[i] for i in tick_idx], rotation=45)
plt.xlabel("Reference date")
plt.ylabel("Mean rms (log scale)")
plt.legend()
plt.tight_layout()
plt.show()

x = np.arange(len(dates_to_test))
tick_idx = _select_ticks(dates_to_test, max_ticks=10)
plt.figure(figsize=(10,6))
plt.plot(x, mean_arr_2[:, 6], marker='o', linewidth=2, label="ref 250703")  # mean shift with different ref dates
plt.plot(x, mean_arr[:, 6], marker='o', linewidth=2, label="ref 250907")  # mean shift
plt.xticks(ticks=tick_idx, labels=[dates_to_test[i] for i in tick_idx], rotation=45)
plt.xlabel("Measurement date")
plt.ylabel("Mean shift")
plt.legend()
plt.tight_layout()
plt.show()
# %%
date_str = r"250908"
ds_SC = read_SC_file_imaging( r"E:\SEICOR_destriped\SLANT_25\SEP2025", date_str, f"SD.NO2_VIS_fix_destriping_test_192")


plt.figure(figsize=(12,6))
plt.imshow(ds_SC["a[O4]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC.datetime.values]
ticks = [i for i, t in enumerate(ds_SC.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45)
plt.xlabel("Time UTC")
plt.ylabel("Image Row")
plt.colorbar(label="O$_4$ / $10^{40}$ molec$^2$/cm$^5$")
plt.tight_layout()
plt.show()
# %%

date_str = r"250703"
ds_SC = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\JUL2025", date_str, f"ID.NO2_VIS_fix_destriped_ref_250907")
ds_SC_ref = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\JUL2025", date_str, f"_D.NO2_VIS_fix_stripe_correction_ref_250907")

ds_SC = correct_destriped_for_noon_reference(ds_SC, ds_SC_ref)
# %%
ds_date_1 = r"250906"
ds_SC_1 = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\SEP2025", ds_date_1, f"_D.NO2_VIS_ref_250907")
# %%
ds_date_2 = r"250907"
ds_SC_2 = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\SEP2025", ds_date_2, f"_D.NO2_VIS_ref_250907")
# %%
ds_date_3 = r"250908"
ds_SC_3 = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\SEP2025", ds_date_3, f"_D.NO2_VIS_ref_250907")
#%%
idx = find_min_zenith_dim0(ds_SC_3)
# %%
for i in ds_SC_1.dim_0:
    if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        print(i.values)
        plt.plot(i,ds_SC_1["a[O4]"].isel(dim_0=i).mean("viewing_direction").values)
plt.show()

# %%
plt.figure(figsize=(10,6))
i=0
while i < len(ds_SC_1.dim_0):
    if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        val_1 = ds_SC_1["a[O4]"].isel(dim_0=i-2).mean("viewing_direction").values
        val_2 = ds_SC_1["a[O4]"].isel(dim_0=i).mean("viewing_direction").values
        diff = val_2 - val_1
        plt.plot(ds_SC_1["time"].isel(dim_0=i), diff, 'o', color='blue')
        i += 1
    i += 1

i=0 
while i < len(ds_SC_2.dim_0):
    if ds_SC_2["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        val_1 = ds_SC_2["a[O4]"].isel(dim_0=i-2).mean("viewing_direction").values
        val_2 = ds_SC_2["a[O4]"].isel(dim_0=i).mean("viewing_direction").values
        diff = val_2 - val_1
        plt.plot(ds_SC_2["time"].isel(dim_0=i), diff, 'o', color='orange')
        i += 1
    i += 1

i=0
while i < len(ds_SC_3.dim_0):
    if ds_SC_3["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        val_1 = ds_SC_3["a[O4]"].isel(dim_0=i-2).mean("viewing_direction").values
        val_2 = ds_SC_3["a[O4]"].isel(dim_0=i).mean("viewing_direction").values
        diff = val_2 - val_1
        plt.plot(ds_SC_3["time"].isel(dim_0=i), diff, 'o', color='green')
        i += 1
    i += 1
plt.show()
#%%

import numpy as np

# scatter vs zenith with legend (vectorized, tolerant float compare)
plt.figure(figsize=(10,6))
for ds, color, label in (
    (ds_SC_1, 'blue', f"meas {ds_date_1}"),
    (ds_SC_2, 'orange', f"meas {ds_date_2}"),
    (ds_SC_3, 'green', f"meas {ds_date_3}"),
):
    idxs = np.where(np.isclose(ds["viewing-azimuth-angle"].values, 102.3, atol=1e-3))[0]
    zs = []
    diffs = []
    for i in idxs:
        if i - 2 >= 0:
            v_now = ds["a[O4]"].isel(dim_0=i).mean("viewing_direction").item()
            v_prev = ds["a[O4]"].isel(dim_0=i-2).mean("viewing_direction").item()
            zs.append(ds["zenith-angle"].isel(dim_0=i).item())
            diffs.append(v_now - v_prev)
    if zs:
        plt.plot(zs, diffs, 'o', color=color, label=label)
plt.xlabel("Zenith angle")
plt.ylabel("O4 diff (dim_0 i - i-2)")
plt.legend(title="Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()

# time-series plot for ds_SC_3 (uses ds_SC_3['time'] if present, otherwise 'datetime')
time_coord_name = "time" if "time" in ds_SC_3.coords or "time" in ds_SC_3 else "datetime"
idxs = np.where(np.isclose(ds_SC_3["viewing-azimuth-angle"].values, 102.3, atol=1e-3))[0]
times = []
diffs = []
for i in idxs:
    if i - 2 >= 0:
        times.append(ds_SC_3[time_coord_name].isel(dim_0=i).values)
        v_now = ds_SC_3["a[O4]"].isel(dim_0=i).mean("viewing_direction").item()
        v_prev = ds_SC_3["a[O4]"].isel(dim_0=i-2).mean("viewing_direction").item()
        diffs.append(v_now - v_prev)

plt.figure(figsize=(12,5))
if times:
    plt.plot(times, diffs, 'o-', color='green', label=f"O4 diff @ az=102.3 ({ds_date_3})")
    plt.gcf().autofmt_xdate()
plt.xlabel(f"{time_coord_name}")
plt.ylabel("O4 diff (dim_0 i - i-2)")
plt.title(f"O4 difference time series for azimuth ≈ 102.3 ({ds_date_3})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

date_str = r"250417"
ds_SC = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\APR2025", date_str, f"SD.NO2_VIS_fix_destriped_ref_250907")
ds_SC_ref = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\APR2025", date_str, f"_D.NO2_VIS_fix_destriped_ref_250907")
ds_orig = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\APR2025", date_str, f"SD.NO2_VIS")

# Increase sizes for labels/ticks/titles
_label_fs = 18
_tick_fs = 16
_title_fs = 20

# %% original image
plt.figure(figsize=(12,6))
plt.imshow(ds_orig["a[O4]"], origin="lower", vmin=-3000, vmax=5500, aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_orig.datetime.values]
ticks = [i for i, t in enumerate(ds_orig.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("O$_4$ / $10^{40}$ molec$^2$/cm$^5$", fontsize=_label_fs)
plt.title(f"O4 original", fontsize=_title_fs)
plt.tight_layout()
plt.show()
# %% second plot (before correction)
plt.figure(figsize=(12,6))
plt.imshow(ds_SC["a[O4]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC.datetime.values]
ticks = [i for i, t in enumerate(ds_SC.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("O$_4$ / $10^{40}$ molec$^2$/cm$^5$", fontsize=_label_fs)
plt.title(f"O4 ref 250907", fontsize=_title_fs)
plt.tight_layout()
plt.show()
# %%
ds_SC_corrected = correct_destriped_for_noon_reference(ds_SC, ds_SC_ref)

# find zenith index and compute mean of the reference correction (for title)
plt.figure(figsize=(12,6))
plt.imshow(ds_SC_corrected["a[O4]"], vmin=-3000, vmax=5500, origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC_corrected.datetime.values]
ticks = [i for i, t in enumerate(ds_SC_corrected.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("O$_4$ / $10^{40}$ molec$^2$/cm$^5$", fontsize=_label_fs)
plt.title(f"O4 ref 250907, offset corrected", fontsize=_title_fs)
plt.tight_layout()
plt.show()
# %%
idx = find_min_zenith_dim0(ds_SC_ref)
plt.figure(figsize=(6,5))
plt.plot(ds_SC_ref["a[O4]"].isel(dim_0=idx), ds_SC_ref["viewing_direction"], color='blue')
plt.ylabel("Image Row", fontsize=_label_fs)
plt.xlabel("O$_4$ / $10^{40}$ molec$^2$/cm$^5$", fontsize=_label_fs)
plt.title(f"Reference correction O4 at zenith \n --> offset: {ds_SC_ref['a[O4]'].isel(dim_0=idx).mean().item():.2f}", fontsize=_title_fs)
plt.grid()
plt.tight_layout()
plt.show()
mean_ref_corr = ds_SC_ref["a[O4]"].isel(dim_0=idx).mean().item()

# %% corrected image with mean sho
# %%
mean_ref_corr = ds_SC_ref["a[O4]"].isel(dim_0=idx).mean().item()

# --- NO2 versions of the O4 plots (same styling) ---
mean_ref_corr_no2 = ds_SC_ref["a[NO2]"].isel(dim_0=idx).mean().item()

# original NO2 image
plt.figure(figsize=(12,6))
plt.imshow(ds_orig["a[NO2]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_orig.datetime.values]
ticks = [i for i, t in enumerate(ds_orig.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("NO$_2$ / molec/cm$^2$", fontsize=_label_fs)
plt.title(f"NO2 original", fontsize=_title_fs)
plt.tight_layout()
plt.show()

# before-correction NO2 image
plt.figure(figsize=(12,6))
plt.imshow(ds_SC["a[NO2]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC.datetime.values]
ticks = [i for i, t in enumerate(ds_SC.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("NO$_2$ / molec/cm$^2$", fontsize=_label_fs)
plt.title(f"NO2 ref 250907", fontsize=_title_fs)
plt.tight_layout()
plt.show()

# corrected NO2 image with mean shown in title
plt.figure(figsize=(12,6))
plt.imshow(ds_SC_corrected["a[NO2]"], origin="lower", aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_SC_corrected.datetime.values]
ticks = [i for i, t in enumerate(ds_SC_corrected.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("NO$_2$  / molec/cm$^2$", fontsize=_label_fs)
plt.title(f"NO2 ref 250907, offset corrected", fontsize=_title_fs)
plt.tight_layout()
plt.show()

# reference correction at zenith for NO2
plt.figure(figsize=(6,5))
plt.plot(ds_SC_ref["a[NO2]"].isel(dim_0=idx), ds_SC_ref["viewing_direction"], color='tab:orange')
plt.ylabel("Image Row", fontsize=_label_fs)
plt.xlabel("NO$_2$  / molec/cm$^2$", fontsize=_label_fs)
plt.title(f"Reference correction NO2 at zenith\nmean: {mean_ref_corr_no2:.0f} / molec/cm$^2$", fontsize=_title_fs)
plt.grid()
plt.tight_layout()
plt.show()
# %%
mean_ref_corr = ds_SC_ref["a[O4]"].isel(dim_0=idx).mean().item()

# --- difference images: original minus corrected (O4 and NO2) ---
diff_o4 = ds_orig["a[O4]"] - ds_SC_corrected["a[O4]"]
diff_no2 = ds_orig["a[NO2]"] - ds_SC_corrected["a[NO2]"]

# robust symmetric limits (99th percentile)
vlim_o4 = np.nanpercentile(np.abs(diff_o4.values), 99)
vlim_no2 = np.nanpercentile(np.abs(diff_no2.values), 99)
vlim_o4 = vlim_o4 if vlim_o4 > 0 else 1.0
vlim_no2 = vlim_no2 if vlim_no2 > 0 else 1.0

plt.figure(figsize=(12,6))
plt.imshow(diff_o4, origin="lower", cmap="RdBu_r", vmin=-vlim_o4, vmax=vlim_o4, aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_orig.datetime.values]
ticks = [i for i, t in enumerate(ds_orig.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("O$_4$ (orig - corrected) / $10^{40}$ molec$^2$/cm$^5$", fontsize=_label_fs)
plt.title("O4: original - corrected", fontsize=_title_fs)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.imshow(diff_no2, origin="lower", cmap="RdBu_r", vmin=-vlim_no2, vmax=vlim_no2, aspect="auto", interpolation="none")
labels = [str(t)[11:16] for t in ds_orig.datetime.values]
ticks = [i for i, t in enumerate(ds_orig.datetime.values) if str(t)[14:16] in ["00"]]
labels_halfhour = [labels[i] for i in ticks]
plt.xticks(ticks=ticks, labels=labels_halfhour, rotation=45, fontsize=_tick_fs)
plt.yticks(fontsize=_tick_fs)
plt.xlabel("Time UTC", fontsize=_label_fs)
plt.ylabel("Image Row", fontsize=_label_fs)
cbar = plt.colorbar()
cbar.set_label("NO$_2$ (orig - corrected) / molec/cm$^2$", fontsize=_label_fs)
plt.title("NO2: original - corrected", fontsize=_title_fs)
plt.tight_layout()
plt.show()
# %%
ds_date_1 = r"250906"
ds_SC_1 = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\SEP2025", ds_date_1, f"_D.NO2_VIS_ref_250907")
# %%
i=0
plt.figure(figsize=(10,6))
while i < len(ds_SC_1.dim_0):
    if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        plt.plot(ds_SC_1["a[O4]"].isel(dim_0=i-2), ds_SC_1.viewing_direction, color='red')
        plt.plot(ds_SC_1["a[O4]"].isel(dim_0=i), ds_SC_1.viewing_direction, color='blue')
        plt.plot(ds_SC_1["a[O4]"].isel(dim_0=i+1), ds_SC_1.viewing_direction, color='green')
        i += 1
    i += 1

plt.show()
# %%
i=0
plt.figure(figsize=(10,6))
while i < len(ds_SC_1.dim_0):
    if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:
        plt.plot(ds_SC_1["a[O4]"].isel(dim_0=i-2)-ds_SC_1["a[O4]"].isel(dim_0=i), ds_SC_1.viewing_direction, color='red')

        i += 1
    i += 1

plt.show()
# %%
i=0
plt.figure(figsize=(10,6))
az_target = 102.3
# find all indices where azimuth == 102.3 (tolerant compare)
inds = np.where(np.isclose(ds_SC_1["viewing-azimuth-angle"].values, az_target, atol=1e-3))[0]

while i < len(ds_SC_1.dim_0):
    if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:

        vd = ds_SC_1["viewing_direction"].values.astype(float)           # y-axis
        x_minus = ds_SC_1["a[O4]"].isel(dim_0=i-2).values.astype(float)  # red
        x_now   = ds_SC_1["a[O4]"].isel(dim_0=i).values.astype(float)    # blue
        x_plus  = ds_SC_1["a[O4]"].isel(dim_0=i+1).values.astype(float)  # green

        # plot the three profiles
        plt.plot(x_minus, vd, color='red',   alpha=0.7, label='_nolegend_')
        plt.plot(x_now,   vd, color='blue',  alpha=0.7, label='_nolegend_')
        #plt.plot(x_plus,  vd, color='green', alpha=0.7, label='_nolegend_')

        # combine data and fit a single linear model y = m*x + b

        m_minus, b_minus = np.polyfit(vd, x_minus, 1)
        y_fit_minus = m_minus * vd + b_minus

        m_now, b_now = np.polyfit(vd, x_now, 1)
        y_fit_now = m_now * vd + b_now

        m_plus, b_plus = np.polyfit(vd, x_plus, 1)
        y_fit_plus = m_plus * vd + b_plus

        plt.plot(y_fit_minus, vd, 'r--', linewidth=1.5)
        plt.plot(y_fit_now,   vd, 'b--', linewidth=1.5)
        #plt.plot(y_fit_plus,  vd, 'g--', linewidth=1.5)

        # annotate the current index on the plot (in axes coords)
        plt.text(0.02, 0.95, f"i = {i}", transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        i += 1
    i += 1

# add legend explaining colours (single representative handles)
plt.plot([], [], color='red',   label='dim_0 = i-2')
plt.plot([], [], color='blue',  label='dim_0 = i')
plt.plot([], [], color='green', label='dim_0 = i+1')
plt.plot([], [], color='k',     linestyle='--', label='linear fit (combined)')

plt.xlabel("O4")
plt.ylabel("Image Row (viewing_direction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
i=0
plt.figure(figsize=(10,6))
az_target = 102.3
# find all indices where azimuth == 102.3 (tolerant compare)
inds = np.where(np.isclose(ds_SC_1["viewing-azimuth-angle"].values, az_target, atol=1e-3))[0]
i = find_min_zenith_dim0(ds_SC_1)
if ds_SC_1["viewing-azimuth-angle"].isel(dim_0=i)==102.3:

    vd = ds_SC_1["viewing_direction"].values.astype(float)           # y-axis
    x_minus = ds_SC_1["a[O4]"].isel(dim_0=i-2).values.astype(float)  # red
    x_now   = ds_SC_1["a[O4]"].isel(dim_0=i).values.astype(float)    # blue
    x_plus  = ds_SC_1["a[O4]"].isel(dim_0=i+1).values.astype(float)  # green

    # plot the three profiles
    plt.plot(x_minus, vd, color='red',   alpha=0.7, label='_nolegend_')
    plt.plot(x_now,   vd, color='blue',  alpha=0.7, label='_nolegend_')
    #plt.plot(x_plus,  vd, color='green', alpha=0.7, label='_nolegend_')

    # combine data and fit a single linear model y = m*x + b

    m_minus, b_minus = np.polyfit(vd, x_minus, 1)
    y_fit_minus = m_minus * vd + b_minus

    m_now, b_now = np.polyfit(vd, x_now, 1)
    y_fit_now = m_now * vd + b_now

    m_plus, b_plus = np.polyfit(vd, x_plus, 1)
    y_fit_plus = m_plus * vd + b_plus

    plt.plot(y_fit_minus, vd, 'r--', linewidth=1.5)
    plt.plot(y_fit_now,   vd, 'b--', linewidth=1.5)
    #plt.plot(y_fit_plus,  vd, 'g--', linewidth=1.5)

    # annotate the current index on the plot (in axes coords)
    plt.text(0.02, 0.95, f"i = {i}", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))


# add legend explaining colours (single representative handles)
plt.plot([], [], color='red',   label='dim_0 = i-2')
plt.plot([], [], color='blue',  label='dim_0 = i')
plt.plot([], [], color='green', label='dim_0 = i+1')
plt.plot([], [], color='k',     linestyle='--', label='linear fit (combined)')

plt.xlabel("O4")
plt.ylabel("Image Row (viewing_direction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
i=0
plt.figure(figsize=(10,6))
az_target = 102.3
ev_target = 180.0

# Plot profiles + linear fits for day1, day2, day3 at their zenith-min index
for ds, color, date_label in (
    (ds_SC_1, 'tab:blue', ds_date_1),
    (ds_SC_2, 'tab:orange', ds_date_2),
    (ds_SC_3, 'tab:green', ds_date_3),
):
    # find representative index (min zenith) and check azimuth
    az_vals = ds["viewing-azimuth-angle"].values
    dxs = np.where(np.isclose(az_vals, az_target, atol=1e-3))[0]
    ds_sel = ds.isel(dim_0=dxs)

    i_sel = ds_sel.dim_0.isel(dim_0=find_min_zenith_dim0(ds_sel)).item()
    if not np.isclose(ds["viewing-azimuth-angle"].isel(dim_0=i_sel).item(), az_target, atol=1e-3):
        continue

    vd = ds["viewing_direction"].values.astype(float)

    # plot i-2, i profiles if available
    for offset, col, alpha in [(-2, color, 0.6), (0, color, 1.0)]:
        idx = i_sel + offset
        if idx < 0 or idx >= ds.sizes['dim_0']:
            continue
        x = ds["a[O4]"].isel(dim_0=idx).values.astype(float)
        lw = 1.0 if offset != 0 else 1.5
        plt.plot(x, vd, color=col, alpha=alpha, linewidth=lw, label='_nolegend_')

    # fit linear model through both the central profile and the i-2 profile
    try:
        idx_minus = i_sel - 2
        idx_center = i_sel
        if 0 <= idx_minus < ds.sizes['dim_0'] and 0 <= idx_center < ds.sizes['dim_0']:
            x_minus = ds["a[O4]"].isel(dim_0=idx_minus).values.astype(float)
            x_center = ds["a[O4]"].isel(dim_0=idx_center).values.astype(float)

            # quadratic fits (degree=2)
            coeffs_minus = np.polyfit(vd, x_minus, 3)   # x = p2*vd^2 + p1*vd + p0
            x_fit = np.polyval(coeffs_minus, vd)

            coeffs_center = np.polyfit(vd, x_center, 3)
            vd_fit = np.polyval(coeffs_center, vd)

            plt.plot(x_fit, vd, color=color, linestyle='--', linewidth=1.8, label=f"{date_label} fit east (deg2)")
            plt.plot(vd_fit, vd, color=color, linestyle='-.', linewidth=1.8, label=f"{date_label} fit center (deg2)")
    except Exception:
        pass

# Legend and labels
plt.xlabel("O4")
plt.ylabel("Image Row (viewing_direction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
date_str = r"250417"
ds_SC = read_SC_file_imaging( r"Q:\BREDOM\SEICOR\IMPACT\SLANT_25\APR2025", date_str, f"SD.NO2_VIS_fix_destriped_ref_250907")
ds_SC_2 = read_SC_file_imaging( r"P:\data\data_tmp\SLANT_25\APR2025", date_str, f"SD.NO2_VIS_ref_250907")
a=ds_SC["a[NO2]"] - ds_SC_2["a[NO2]"]
a.plot()


# %%
import subprocess
param_file = r"P:\data\data_tmp\pars\nlin_d\SEICOR_D_offaxis.NO2_VIS_ref_250907"
exe_path = r'P:\exe_64\nlin_c.exe'
cmd = f'start /wait "" "{exe_path}" {param_file}'

subprocess.run(cmd, shell=True, check=True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

base_root = r"P:\data\data_tmp\SLANT_25"   # root containing month folders (APR2025, MAY2025, ...)
refs = ["250907", "250703"]
vars_to_subtract = ["a[O4]"]   # change / extend if you want other variables adjusted/analysed

# init results
results = {}
for r in refs:
    results[f"SD_ref_{r}"] = []
    results[f"_D_ref_{r}"] = []
results["SD.NO2_VIS_fix"] = []
dates = []

# iterate days across multiple months
start = datetime(2025, 3, 26)
end = datetime(2025, 10, 8)
d = start
while d <= end:
    date_str = d.strftime("%y%m%d")
    dates.append(d)
    print(date_str)
    # determine month-folder for this date (uses get_month_folder defined earlier)
    month_folder = get_month_folder(date_str)
    data_folder = os.path.join(base_root, month_folder)

    # read daily "SD.NO2_VIS_fix" once per day (may be used for plotting/compare)
    ds_fix = None
    try:
        ds_fix = read_SC_file_imaging(data_folder, date_str, "SD.NO2_VIS_fix")
    except Exception:
        ds_fix = None

    for ref in refs:
        key_sd = f"SD_ref_{ref}"
        key_d = f"_D_ref_{ref}"

        ds_sd = None
        ds__d = None
        try:
            ds_sd = read_SC_file_imaging(data_folder, date_str, f"SD.NO2_VIS_fix_ref_{ref}")
        except Exception:
            ds_sd = None
        try:
            ds__d = read_SC_file_imaging(data_folder, date_str, f"_D.NO2_VIS_fix_ref_{ref}")
        except Exception:
            ds__d = None

        var_sd_val = np.nan
        var_d_val = np.nan

        # proceed only if we have at least the reference _D (for offset) and the SD file
        if ds__d is not None and ds_sd is not None:
            try:
                idx_min = find_min_zenith_dim0(ds__d)
            except Exception:
                idx_min = 0

            # compute offset from the _D dataset at min zenith (fallback to 0)
            offset = 0.0
            for v in vars_to_subtract:
                if v in ds__d:
                    offset = float(ds__d[v].isel(dim_0=idx_min).mean().item())
                    break

            # subtract offset from SD
            ds_sd_adj = ds_sd.copy(deep=True)
            for v in vars_to_subtract:
                if v in ds_sd_adj:
                    ds_sd_adj[v] = ds_sd_adj[v] - offset

            # compute variability: mean over dim_0, rolling mean over viewing_direction slice(7,None), then std
            try:
                integrated = ds_sd_adj[vars_to_subtract[0]].mean(dim="dim_0")
                integ_slice = integrated.sel(viewing_direction=slice(7, None))
                rolling_mean = integ_slice.rolling(viewing_direction=7, center=True).mean()
                variability = integ_slice - rolling_mean
                var_sd_val = float(variability.std(dim="viewing_direction", skipna=True).item())
            except Exception:
                var_sd_val = float(np.nanstd(ds_sd_adj[vars_to_subtract[0]].mean(dim="dim_0").sel(viewing_direction=slice(7, None)).values))

            # also compute variability for the _D dataset after subtracting same offset
            try:
                ds_d_adj = ds__d.copy(deep=True)
                for v in vars_to_subtract:
                    if v in ds_d_adj:
                        ds_d_adj[v] = ds_d_adj[v] - offset
                integrated_d = ds_d_adj[vars_to_subtract[0]].mean(dim="dim_0")
                integ_slice_d = integrated_d.sel(viewing_direction=slice(7, None))
                rolling_mean_d = integ_slice_d.rolling(viewing_direction=7, center=True).mean()
                variability_d = integ_slice_d - rolling_mean_d
                var_d_val = float(variability_d.std(dim="viewing_direction", skipna=True).item())
            except Exception:
                var_d_val = float(np.nan)

        # append one value per day/ref (NaN if missing)
        results[key_sd].append(var_sd_val)
        results[key_d].append(var_d_val)

    # compute and append daily value for SD.NO2_VIS_fix (if present)
    if ds_fix is not None:
        try:
            integrated_fix = ds_fix[vars_to_subtract[0]].mean(dim="dim_0")
            integ_slice_fix = integrated_fix.sel(viewing_direction=slice(7, None))
            rolling_mean_fix = integ_slice_fix.rolling(viewing_direction=7, center=True).mean()
            variability_fix = integ_slice_fix - rolling_mean_fix
            var_fix_val = float(variability_fix.std(dim="viewing_direction", skipna=True).item())
        except Exception:
            var_fix_val = float(np.nan)
    else:
        var_fix_val = np.nan
    results["SD.NO2_VIS_fix"].append(var_fix_val)

    d += timedelta(days=1)

            
#%%
# convert to numpy arrays and plot
dates_np = np.array(dates)

plt.figure(figsize=(12,5))
for ref in refs:
    key_sd = f"SD_ref_{ref}"
    y_sd = np.array(results[key_sd], dtype=float)
    plt.plot(dates_np, y_sd, marker="o", label=f"fix ref {ref}")
y_fix = np.array(results[key_fix], dtype=float)
plt.plot(dates_np, y_fix, marker="o", label=f"daily ref")


plt.xlabel("Date")
plt.ylabel(f"Variability of {vars_to_subtract[0]}")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()


# %%
