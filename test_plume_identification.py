#%%
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.append(str(Path(r"C:\Users\hhave\Documents\Promotion\scripts")))
import SEICOR.plumes
import SEICOR.plotting

#file_path = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_004_t_20250615_051423_mmsi_211676580.nc"
#file_path = r"p:\data\SEICOR\plumes_2\plumes_250428\plume_024_t_20250428_112317_mmsi_563068900.nc"
#file_path = r"P:\data\SEICOR\plumes_2\plumes_250428\plume_024_t_20250428_112317_mmsi_563068900.nc" #"D:\SEICOR\plumes_2\plumes_250412\plume_028_t_20250412_150343_mmsi_255806434.nc"
#file_path = r"D:\SEICOR\plumes_2\plumes_250412\plume_028_t_20250412_150343_mmsi_255806434.nc"
file_path = r"P:\data\SEICOR\plumes_2\plumes_250412\plume_032_t_20250412_164820_mmsi_247389200.nc"
#file_path= r"P:\data\SEICOR\plumes_2\plumes_250615\plume_004_t_20250615_051423_mmsi_211676580.nc"
#file_path = r"P:\data\SEICOR\plumes_2\plumes_250615\plume_003_t_20250615_051307_mmsi_228397600.nc"
#file_path = r"P:\data\SEICOR\plumes_2\plumes_250502\plume_016_t_20250502_081152_mmsi_564398000.nc"
ds_plume = xr.open_dataset(file_path)
# %%
a = ds_plume.no2_ref-ds_plume.no2_ref.mean(dim="window_ref")

mask = SEICOR.plumes.detect_plume_ztest(
        ds_plume["no2_enhancement_interp"].values,
        bg_std=None,
        bg_mean=None,
        p_threshold=0.001,
        min_cluster_size=5,
        connectivity=1,
        kernel_arm=1,
        require_connection=True,
        ds_plume=ds_plume,
        keep_second_largest=False,
        second_size_threshold=100,
    )   
# %%
out_dir = Path(r"C:\Users\hhave\Downloads")
date = "250615"  #example date

# %%
a = ds_plume.no2_ref-ds_plume.no2_ref.mean(dim="window_ref")

plume_mask, ship_mask, plume_mask_iterations, ship_mask_iterations = SEICOR.plumes.detect_plume_ztest_iterative(
        ds_plume["no2_enhancement_interp"].values,
        bg_std=None,
        bg_mean=None,
        p_threshold=0.05,
        min_cluster_size=5,
        connectivity=1,
        kernel_arm=1,
        require_connection=True,
        ds_plume=ds_plume,
        keep_second_largest=True,
        second_size_threshold=100,
        ship_p_threshold=0.1,
        ship_min_cluster_size=20,
        ship_kernel_arm=0,
        ship_median_kernel_arm=None,
        ship_connectivity=1,
    )
# %%
out_dir = Path(r"C:\Users\hhave\Downloads")
date = "250615"  #example date

SEICOR.plotting.plot_no2_enhancement_with_plume_and_ship_mask(ds_plume, plume_mask, ship_mask, out_dir, "final_mask")

for iteration_idx, (plume_iteration, ship_iteration) in enumerate(zip(plume_mask_iterations, ship_mask_iterations)):
    iteration_out_dir = out_dir
    SEICOR.plotting.plot_no2_enhancement_with_plume_and_ship_mask(
        ds_plume,
        plume_iteration,
        ship_iteration,
        iteration_out_dir,
        str(iteration_idx),
    )
# %%
