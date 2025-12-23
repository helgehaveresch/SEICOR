import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from PIL import Image
sys.path.append(os.path.dirname(__file__))
from enhancements import upwind_constant_background_enh
import zipfile
from io import BytesIO


def plot_LOS(lon1, lon2, lat1, lat2, ax=None, label=None):
    """
    Plot lines from (lon1, lat1) to each unique (lon2, lat2) pair.
    """
    if ax is None:
        ax = plt.gca()
    # Get unique pairs of (lon2, lat2)
    unique_points = set(zip(np.ravel(lon2), np.ravel(lat2)))
    for i, (x2, y2) in enumerate(unique_points):
        if label and i == 0:
            ax.plot([lon1, x2], [lat1, y2], color='red', linewidth=2, linestyle='--', label=label)
        else:
            ax.plot([lon1, x2], [lat1, y2], color='red', linewidth=2, linestyle='--')
    plt.scatter([lon1], [lat1], color='black', marker='x', label='IMPACT')

def plot_trajectories(filtered_ships, maskedout_ships, df_closest, lon1, lon2, lat1, lat2, window_small, save=False, out_dir=None):
    first_key = next(iter(filtered_ships))
    first_value = filtered_ships[first_key]
    date_str = first_value.index[0].strftime("%y%m%d")
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.3, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ships not filtered out")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"filtered_ships_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    # Plot the masked out ships (either sailing vessels or distance > 0.005)

    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.3, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"maskedout_ships_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    # Plot the ships that are left after filtering (filtered_ships)
    plt.figure(figsize=(10, 8))
    for mmsi, group in filtered_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Non-sailing ships with minimum distance ≤ 0.005 deg to LOS")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"filtered_ships_small_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


    plt.figure(figsize=(10, 8))
    for mmsi, group in maskedout_ships.items():
        plt.plot(group["Longitude"], group["Latitude"], alpha=0.7, label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships (closeup)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    #plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"maskedout_ships_small_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    multi_pass_mmsi = df_closest["MMSI"].value_counts()
    multi_pass_mmsi = multi_pass_mmsi[multi_pass_mmsi > 1].index.tolist()

    plt.figure(figsize=(10, 8))
    for mmsi in multi_pass_mmsi:
        group = filtered_ships.get(mmsi)
        if group is not None:
            plt.plot(group["Longitude"], group["Latitude"], label=f"MMSI {mmsi}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ships that pass multiple times (filtered ships)")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"multipass_ships_small_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_maskedout_ships_details(maskedout_ships, window_small, lon1, lon2, lat1, lat2, save=False, out_dir=None):
    first_key = next(iter(maskedout_ships))
    first_value = maskedout_ships[first_key]
    date_str = first_value.index[0].strftime("%y%m%d")
    plt.figure(figsize=(12, 8))
    filtermask_labels = {
        1: "Outside operation time",
        2: "Anchored",
        3: "Sailing vessel",
        4: "Too small",
        5: "No movement",
        6: "No close positions"
    }
    colors = {1: "red", 2: "green", 3: "blue", 4: "orange", 5: "purple", 6: "gray"}

    for mmsi, group in maskedout_ships.items():
        for mask_val in group["filtermask"].unique():
            mask = group["filtermask"] == mask_val
            if mask.any():
                label = filtermask_labels.get(mask_val, f"filtermask={mask_val}")
                plt.plot(
                    group.loc[mask, "Longitude"],
                    group.loc[mask, "Latitude"],
                    color=colors.get(mask_val, "black"),
                    alpha=0.7,
                    label=label if plt.gca().get_legend_handles_labels()[1].count(label) == 0 else None
                )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Masked out ships by filtermask")
    plot_LOS(lon1, lon2, lat1, lat2)
    plt.xlim(window_small[2], window_small[3])
    plt.ylim(window_small[0], window_small[1])
    plt.legend(loc="best", fontsize="small", ncol=2)
    if save:
        savepath = os.path.join(out_dir, f"trajectories", f"maskedout_ships_details_{date_str}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_ship_stats(filtered_ships, save=False, out_dir=None):
    first_key = next(iter(filtered_ships))
    first_value = filtered_ships[first_key]
    date_str = first_value.index[0].strftime("%y%m%d")
    # lentgh histogram
    lengths = pd.Series(
        [group["Length_in_m"].dropna().iloc[0] for group in filtered_ships.values() if not group["Length_in_m"].dropna().empty]
    )
    plt.figure()
    plt.hist(lengths, bins=30, alpha=0.7)
    plt.xlabel("Length (m)")
    plt.ylabel("Count")
    plt.title(f"Distribution of Ship Lengths {date_str}")
    if save:
        savepath = os.path.join(out_dir, f"ship_stats", f"{date_str}_Length.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    # Draught histogram
    draughts = pd.Series(
        [group["Draught_in_m"].dropna().iloc[0] for group in filtered_ships.values() if not group["Draught_in_m"].dropna().empty]
    )
    plt.figure()
    plt.hist(draughts, bins=30, alpha=0.7)
    plt.xlabel("Draught (m)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Draughts (filtered ships)")
    if save:
        savepath = os.path.join(out_dir, f"ship_stats", f"{date_str}_Draught.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    # Velocity histogram
    velocities = pd.Series(
        [group["Speed_in_m/s"].dropna().iloc[0] for group in filtered_ships.values() if not group["Speed_in_m/s"].dropna().empty]
    )
    plt.figure()
    plt.hist(velocities, bins=30, alpha=0.7)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("Histogram of Ship Velocities (filtered ships)")
    if save:
        savepath = os.path.join(out_dir, f"ship_stats", f"{date_str}_Velocity.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    # Correlation between length and draught
    length_draught_df = pd.DataFrame({
        "Length_in_m": lengths,
        "Draught_in_m": draughts
    }).dropna()
    plt.figure()
    plt.scatter(length_draught_df["Length_in_m"], length_draught_df["Draught_in_m"], alpha=0.5)
    plt.xlabel("Length (m)")
    plt.ylabel("Draught (m)")
    corr = length_draught_df["Length_in_m"].corr(length_draught_df["Draught_in_m"])
    plt.title("Correlation between Length and Draught: {:.2f}".format(corr))
    if save:
        savepath = os.path.join(out_dir, f"ship_stats", f"{date_str}_corr_Length_Draught.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def styles_from_ship_category(parameter, category):
    if parameter == "linestyle":
        linestyle = {'small': ':', 'middle': '--', 'large': '-', 'unknown': '-.'}
        return linestyle.get(category, '-')
    elif parameter == "transparency":
        transparency = {'small': 0.3, 'middle': 0.5, 'large': 0.7, 'unknown': 0.5}
        return transparency.get(category, 0.5)
    elif parameter == "label":
        label = {'small': 'Small ship', 'middle': 'Medium ship', 'large': 'Large ship', 'unknown': 'Unknown'}
        return label.get(category, 'Unknown')
    else:
        raise ValueError(f"Unknown parameter: {parameter}")
    
def plot_no2_timeseries(
    ds_masked,
    df_closest,
    start_time,
    end_time,
    add_ship_lines=True,
    legend_location="upper right",
    separate_legend=False,
    save=False,
    out_dir=None
):
    """
    Plot NO2 timeseries with optional vertical ship lines.
    """
    # create explicit figure object so we can save/close it later
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    if add_ship_lines and df_closest is not None and start_time is not None and end_time is not None:
        for idx, passing_ship in df_closest.iterrows():
            t_ship = pd.to_datetime(passing_ship["UTC_Time"]) if "UTC_Time" in passing_ship else idx
            category = passing_ship.get("ship_category")
            if (t_ship >= start_time) and (t_ship <= end_time):
                label = styles_from_ship_category("label", category)
                # Only add label if not already present
                if label not in ax.get_legend_handles_labels()[1]:
                    ax.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=label
                    )
                else:
                    ax.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=""
                    )

    # Plot NO2 enhancement with RMS mask applied
    ax.plot(
        ds_masked.datetime,
        ds_masked["a[NO2]"],
        label="NO$_2$ (FOV-averaged)",
        linewidth=2
    )

    ax.set_xlabel("Time (UTC)", fontsize=18)
    ax.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=16)
    date_str = start_time.strftime("%d.%m.%Y")
    
    ax.set_title(f"NO$_2$ dSCD timeseries on {date_str}", fontsize=20)
    ax.grid()
    # Set x-axis to show only time (HH:MM) and restrict number of ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.tick_params(axis='both', labelsize=16)
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()

    if separate_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(3, 2))
        ax_leg = fig_legend.add_subplot(111, frameon=False)
        ax_leg.axis('off')  # Hide the axis
        fig_legend.legend(handles, labels, loc='center', fontsize=14)

        if save:
            first_dt = pd.to_datetime(ds_masked["datetime"].values[0])
            main_dir = os.path.join(out_dir, "no2_timeseries")
            os.makedirs(main_dir, exist_ok=True)
            main_savepath = os.path.join(main_dir, f"no2_timeseries_{first_dt.strftime('%y%m%d')}.png")
            legend_savepath = os.path.join(main_dir, f"no2_timeseries_legend.png")

            # save main figure and legend figure separately
            fig.savefig(main_savepath, bbox_inches='tight')
            fig_legend.savefig(legend_savepath, bbox_inches='tight')

            plt.close(fig)
            plt.close(fig_legend)
        else:
            plt.show()
            plt.close(fig)
            plt.close(fig_legend)
    else:
        ax.legend(loc=legend_location, fontsize=14)
        if save:
            first_dt = pd.to_datetime(ds_masked["datetime"].values[0])
            save_dir = os.path.join(out_dir, "no2_timeseries")
            os.makedirs(save_dir, exist_ok=True)
            main_savepath = os.path.join(save_dir, f"no2_timeseries_{first_dt.strftime('%y%m%d')}.png")
            fig.savefig(main_savepath, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        


def plot_single_ship(
    ds_impact=None,
    ds_enhancements=None,
    dir_enhancements=None,
    t_after_start_h=None,
    interval_h=0.1,
    vmin=None,
    vmax=None,
    include_mmsi=False,
    include_timestamp=False,
    y_axis_ticks_interval=7,
    x_axis_tick_interval=2,
    mode="dSCD",
    save_fig=False,
    save_dir=None
    ):
    """ 
    Create 2d Plot of NO2 dSCD or enhancements

    Parameters
    ----------
    ds_impact : xarray.Dataset
        NO2 measurements dataset.
    ds_enhancements : xarray.Dataset
        NO2 enhancements dataset.
    dir_enhancements : xarray.Dataset
        Directional NO2 enhancements dataset.
    t_after_start_h : float
        Time (in hours) after start time to begin plotting.
    interval_h : float
        Time interval (in hours) for the plot.
    vmin, vmax : float
        Minimum and maximum values for the color scale.
    include_mmsi : bool
        Whether to include MMSI in the plot title.
    include_timestamp : bool
        Whether to include timestamp in the plot title.
    y_axis_ticks_interval : int
        Interval for y-axis ticks.
    x_axis_tick_interval : int
        Interval for x-axis ticks.
    mode : str
        Plotting mode ("dSCD", "enhancement", "integrated", or "combined").
    """

    if ds_impact is not None:
        no2_full = ds_impact["a[NO2]"].values  # shape: (time, viewing_direction)
        times_full = pd.to_datetime(ds_impact["datetime"].isel(viewing_direction=0).values)
        viewing_dirs = ds_impact["los"].values
    if ds_enhancements is not None:
        no2_full = ds_enhancements["no2_enhancement"].values
        times_full = pd.to_datetime(ds_enhancements["datetime"].isel(viewing_direction=0).values)
        viewing_dirs = ds_enhancements["los"].values
    if dir_enhancements is not None:
        no2_full = dir_enhancements["no2_enhancement_c_back"].values
        times_full = dir_enhancements["times_plume"].values
        viewing_dirs = dir_enhancements["image_row"].values
        mmsi = dir_enhancements.mmsi
        timestamp = pd.to_datetime(dir_enhancements.t)
    if t_after_start_h is not None:
        mask = (
            (times_full >= pd.Timestamp(times_full[0].date()) + pd.Timedelta(hours=t_after_start_h)) &
            (times_full < pd.Timestamp(times_full[0].date()) + pd.Timedelta(hours=t_after_start_h + interval_h))
        )
        times_sel = times_full[mask]
        no2_sel = no2_full[:, mask]
    else:
        times_sel = times_full
        no2_sel = no2_full

    X, Y = np.meshgrid(times_sel, np.arange(no2_sel.shape[0]))

    if mode == "dSCD":
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pcm = ax.pcolormesh(
            X, Y, no2_sel,
            shading='auto',
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(pcm)
        cbar.set_label(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=22)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)
        ax.set_xlabel("Time (UTC)", fontsize=22)
        ax.set_ylabel("Viewing direction", fontsize=22)
        ax.set_title("NO$_2$ dSCD Single Ship", fontsize=24)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        # viewing_dirs can be 2D (e.g., Nx2) or 1D (N,). Handle both cases.
        vd = np.asarray(viewing_dirs)
        if vd.ndim == 2:
            labels = [f"{vd[i,0]-90:.1f}°" for i in yticks]
        else:
            labels = [f"{vd[i]-90:.1f}°" for i in yticks]
        ax.set_yticklabels(labels, fontsize=18)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        plt.show()

    elif mode == "enhancement":
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pcm = ax.pcolormesh(
            X, Y, no2_full,
            shading='auto',
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(pcm)
        cbar.set_label(r"NO$_2$ Enh. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=22)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)
        ax.set_xlabel("Time (UTC)", fontsize=22)
        ax.set_ylabel("Viewing direction", fontsize=22)
        title = "NO$_2$ dSCD Enhancements"
        if include_mmsi:
            title += f", MMSI: {mmsi}"
        if include_timestamp:
            title += f", Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ax.set_title(title, fontsize=14)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        # viewing_dirs can be 2D (e.g., Nx2) or 1D (N,). Handle both cases.
        vd = np.asarray(viewing_dirs)
        if vd.ndim == 2:
            labels = [f"{vd[i,0]-90:.1f}°" for i in yticks]
        else:
            labels = [f"{vd[i]:.1f}°" for i in yticks]
        ax.set_yticklabels(labels, fontsize=18)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        if save_fig: 
            savepath = os.path.join(save_dir, f"NO2_{timestamp.strftime('%Y%m%d_%H%M%S')}_{mmsi}.png")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()

    elif mode == "integrated":
        no2_integrated = no2_full.sum(axis=0)  # shape: (selected_times,)
        plt.figure(figsize=(10, 4))
        plt.plot(times_sel, no2_integrated, color='tab:blue', linewidth=2)
        plt.xlabel("Time (UTC)", fontsize=18)
        plt.ylabel(r"NO$_2$ dSCD int. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=18)
        plt.title("Vertically Integrated NO$_2$ Enhancement", fontsize=20)
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        plt.tight_layout()
        plt.show()

    elif mode == "combined":
        no2_integrated = no2_full.sum(axis=0)  # shape: (selected_times,)
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        # --- 2D NO2 enhancement plot ---
        ax = axs[0]
        pcm = ax.pcolormesh(
            X, Y, no2_full,
            shading='auto',
            cmap='viridis', vmin=vmin if vmin is not None else -3e16, vmax=vmax
        )
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(r"NO$_2$ Enh. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)

        ax.set_ylabel("Viewing direction", fontsize=22)
        ax.set_title("NO$_2$ dSCD Enhancements", fontsize=24)

        # Set y-ticks to actual viewing direction angles (every Nth for clarity)
        N = y_axis_ticks_interval
        yticks = np.arange(0, len(viewing_dirs), N)
        ax.set_yticks(yticks)
        # viewing_dirs can be 2D (e.g., Nx2) or 1D (N,). Handle both cases.
        vd = np.asarray(viewing_dirs)
        if vd.ndim == 2:
            labels = [f"{vd[i,0]-90:.1f}°" for i in yticks]
        else:
            labels = [f"{vd[i]-90:.1f}°" for i in yticks]
        ax.set_yticklabels(labels, fontsize=18)

        # Format x-axis: only every 2 min a tick
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # --- Vertically integrated NO2 subplot ---
        ax2 = axs[1]
        ax2.plot(times_sel, no2_integrated, color='tab:blue', label="Vertically Integrated NO$_2$", linewidth=2)
        ax2.set_ylabel(r"NO$_2$ dSCD int. / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=18)
        ax2.set_xlabel("Time (UTC)", fontsize=22)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_axis_tick_interval))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.grid(True)
        ax2.legend(fontsize=20)
        ax2.yaxis.get_offset_text().set_fontsize(16)

        plt.tight_layout()
        # Make ax2 narrower in x-direction (e.g., 60% width, aligned left)
        pos = ax2.get_position()
        ax2.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

        plt.show()

    else:
        raise ValueError("mode must be one of: 'dSCD', 'enhancement', 'integrated', 'combined'")
    
def plot_no2_enhancements_for_all_ships(path_ship_passes, plot_out_dir):
    """
    Loops over df_closest, subtracts background, and saves NO2 enhancement images.
    """
    ship_passes = pd.read_csv(path_ship_passes, index_col=0, parse_dates=True)
    for idx, ship_pass_single in ship_passes.iterrows():
        #check if plume_file exists
        if not os.path.isfile(ship_pass_single['plume_file']):
            print(f"Plume file {ship_pass_single['plume_file']} does not exist. Skipping.")
            continue
        ds_plume = xr.open_dataset(ship_pass_single['plume_file'])

        plot_single_ship(
            dir_enhancements=ds_plume,
            t_after_start_h=None,
            interval_h=None,
            mode="enhancement",
            vmin=-3e16,
            include_mmsi=True,
            include_timestamp=True,
            x_axis_tick_interval=1,
            save_fig=True,
            save_dir=plot_out_dir,
        )

def plot_ship_pass_subplot_v1(
    ds_plume,
    passing_ship,
    ds_impact,
    df_insitu,
    no2_out_dir,
    save=True
    ):
    """
    Plot NO2 enhancement, integrated NO2, video image, wind/ship polar, and in-situ NO2 for a ship pass.
    """

    # Try to find the corresponding image (treat explicit 'no image' as missing)
    img_file = None
    if (
        "closest_image_file" in passing_ship
        and pd.notnull(passing_ship["closest_image_file"])
        and str(passing_ship["closest_image_file"]).strip().lower() != "no image"
    ):
        img_file = passing_ship["closest_image_file"]
    # Normalize and convert plume times to matplotlib date numbers to avoid
    # overflow when matplotlib converts numpy datetime64 or tz-aware timestamps.
    try:
        times_dt = pd.to_datetime(ds_plume["times_plume"].values)
    except Exception:
        times_dt = pd.to_datetime(ds_plume["times_plume"])

    # Build a pandas DatetimeIndex for robust tz-handling and reindexing
    times_index = pd.to_datetime(times_dt)

    # If in-situ index is tz-aware, make times_index tz-aware in the same tz.
    try:
        insitu_tz = df_insitu.index.tz
    except Exception:
        insitu_tz = None

    if insitu_tz is not None:
        # times_index may be tz-naive or tz-aware; convert/localize to insitu_tz
        if getattr(times_index, 'tz', None) is None:
            # assume times are UTC if naive; localize then convert
            try:
                times_index = times_index.tz_localize('UTC').tz_convert(insitu_tz)
            except Exception:
                # fallback: directly localize to insitu_tz
                times_index = times_index.tz_localize(insitu_tz)
        else:
            times_index = times_index.tz_convert(insitu_tz)
    else:
        # ensure tz-naive
        try:
            times_index = times_index.tz_convert(None)
        except Exception:
            try:
                times_index = times_index.tz_localize(None)
            except Exception:
                pass

    # Replace NaT entries with the first valid time to avoid conversion issues
    if len(times_index) > 0 and times_index.isnull().any():
        first_valid = times_index[times_index.notnull()].tolist()[0]
        times_index = times_index.fillna(first_valid)

    # Python datetimes (may be tz-aware) and matplotlib float dates
    times = times_index.to_pydatetime()
    times_num = mdates.date2num(times)
    
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])

    # Build plume times DatetimeIndex and matplotlib date numbers (times_num)
    try:
        times_dt = pd.to_datetime(ds_plume["times_plume"].values)
    except Exception:
        times_dt = pd.to_datetime(ds_plume["times_plume"])
    times_index = pd.to_datetime(times_dt)
    try:
        insitu_tz = df_insitu.index.tz
    except Exception:
        insitu_tz = None
    if insitu_tz is not None:
        if getattr(times_index, 'tz', None) is None:
            try:
                times_index = times_index.tz_localize('UTC').tz_convert(insitu_tz)
            except Exception:
                times_index = times_index.tz_localize(insitu_tz)
        else:
            times_index = times_index.tz_convert(insitu_tz)
    else:
        try:
            times_index = times_index.tz_convert(None)
        except Exception:
            try:
                times_index = times_index.tz_localize(None)
            except Exception:
                pass
    if len(times_index) > 0 and times_index.isnull().any():
        first_valid = times_index[times_index.notnull()].tolist()[0]
        times_index = times_index.fillna(first_valid)
    times = times_index.to_pydatetime()
    times_num = mdates.date2num(times)

    # NO2 enhancement 2D plot
    ax0 = fig.add_subplot(gs[0, 0])
    X, Y = np.meshgrid(ds_plume["times_window"], np.arange(ds_plume["no2_enhancement_c_back"].shape[0]))
    pcm = ax0.pcolormesh(X, Y, ds_plume["no2_enhancement_c_back"], shading='auto')
    fig.colorbar(pcm, ax=ax0, label=r"NO$_2$ Enhancement / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    tstr_display = pd.to_datetime(ds_plume.t).strftime('%Y-%m-%d %H:%M:%S')
    tstr_fname = pd.to_datetime(ds_plume.t).strftime('%Y%m%d_%H%M%S')
    ax0.set_title(f"MMSI {ds_plume.mmsi} NO2 around {tstr_display}")
    ax0.set_xlabel("Time")
    N = 3  # Show every 3rd LOS value
    vea_vals = ds_plume["vea"].values 
    yticks = np.arange(0, len(vea_vals), N)
    ax0.set_yticks(yticks)
    ax0.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
    ax0.set_ylabel("VEA / °")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax0.get_xticklabels(), rotation=45)

    # Integrated NO2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(ds_plume["times_window"], ds_plume["vertically_integrated_no2_enhancement"], color='tab:blue')
    ax1.set_title("Vertically integrated NO$_2$ enhancement")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Integrated NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Video image
    ax2 = fig.add_subplot(gs[0, 2])
    if img_file:
        fname = os.path.basename(img_file)
        fdir = os.path.dirname(img_file)
        zip_candidate = fdir + ".zip"
        shown = False
        # First try zip (file_dir + ".zip")
        if os.path.isfile(zip_candidate):
            try:
                with zipfile.ZipFile(zip_candidate, "r") as zf:
                    member = None
                    if fname in zf.namelist():
                        member = fname
                    else:
                        for m in zf.namelist():
                            if m.endswith("/" + fname) or m.endswith("\\" + fname) or m.endswith(fname):
                                member = m
                                break
                    if member:
                        data = zf.read(member)
                        img = Image.open(BytesIO(data))
                        ax2.imshow(img)
                        ax2.set_title(f"Video image ({os.path.basename(zip_candidate)})")
                        ax2.axis("off")
                        shown = True
            except Exception:
                shown = False
        # Fallback: direct file on disk
        if not shown and os.path.exists(img_file):
            try:
                img = Image.open(img_file)
                ax2.imshow(img)
                ax2.set_title("Video image")
                ax2.axis("off")
                shown = True
            except Exception:
                shown = False
        if not shown:
            ax2.text(0.5, 0.5, "No image", ha="center", va="center")
            ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No image", ha="center", va="center")
        ax2.axis("off")

    # Wind/ship polar plot
    wind_dir_rad = np.deg2rad(passing_ship['wind_dir'])
    wind_speed_mean = passing_ship['wind_speed']
    ax3 = fig.add_subplot(gs[0, 3], polar=True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    viewing_dir_deg = ds_impact["viewing-azimuth-angle"].isel(viewing_direction=0).values[passing_ship["Closest_Impact_Measurement_Index"]]
    viewing_dir_rad = np.deg2rad(viewing_dir_deg)
    ax3.plot([viewing_dir_rad, viewing_dir_rad], [0, ax3.get_rmax()], color='red', linewidth=2, label='Viewing direction', linestyle='--')
    wind_arrow_length = 0.7
    ax3.arrow(
        wind_dir_rad, wind_speed_mean, 0, -wind_arrow_length,
        width=0.03, head_width=0.15, head_length=0.3, color='tab:blue', alpha=0.8
    )
    ax3.text(
        wind_dir_rad+np.deg2rad(10), wind_speed_mean + 1.0,
        f"{wind_speed_mean:.2f} m/s\n{wind_dir_rad:.0f}°",
        ha='center', va='bottom', fontsize=10, color='black'
    )
    ship_arrow_length = 0.7
    ax3.arrow(
        np.deg2rad(passing_ship["Mean_Course"]), passing_ship["Mean_Speed"], 0, ship_arrow_length,
        width=0.03, head_width=0.15, head_length=0.3, color='green', alpha=0.8
    )
    ax3.text(
        np.deg2rad(passing_ship["Mean_Course"])+np.deg2rad(10), passing_ship["Mean_Speed"] + 1,
        f"{passing_ship['Mean_Speed']:.2f} m/s\n{passing_ship['Mean_Course']:.0f}°",
        ha='center', va='bottom', fontsize=10, color='black'
    )
    view_line = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Viewing direction')
    wind_arrow = mpatches.FancyArrowPatch((0, 0), (0.5, 0.5), color='tab:blue', arrowstyle='->', mutation_scale=15, label='Wind arrow')
    ship_arrow = mpatches.FancyArrowPatch((0, 0), (0.5, 0.5), color='green', arrowstyle='->', mutation_scale=15, label='Ship velocity')
    ax3.legend(handles=[view_line, wind_arrow, ship_arrow], loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # In-situ NO2
    ax4 = fig.add_subplot(gs[0, 4])
    in_situ_mask = (df_insitu.index >= ds_plume["times_window"][0]) & (df_insitu.index <= ds_plume["times_window"][-1])
    in_situ_times = df_insitu.index[in_situ_mask]
    in_situ_no2 = df_insitu['c_no2'][in_situ_mask]
    if len(in_situ_times) > 0:
        ax4.plot(in_situ_times, in_situ_no2, color='tab:red')
        ax4.set_xlim(in_situ_times[0], in_situ_times[-1])
    else:
        ax4.text(0.5, 0.5, "No in-situ data", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("In-situ NO$_2$")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("NO$_2$ [ppb]")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=45)

    plt.tight_layout()
    if save:
        savepath = os.path.join(no2_out_dir, f"NO2_enhancement_subplot_{tstr_fname}_{ds_plume.mmsi}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_ship_pass_subplot_v2(
        ds_plume, passing_ship, img_dir, no2_out_dir, lat1, lon1, lat2, lon2, save=False
    ):

    # Try to find the corresponding image (treat explicit 'no image' as missing)
    img_file = None
    if (
        "closest_image_file" in passing_ship
        and pd.notnull(passing_ship["closest_image_file"])
        and str(passing_ship["closest_image_file"]).strip().lower() != "no image"
    ):
        img_file = passing_ship["closest_image_file"]
    
    # Create subplot
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 5, width_ratios=[2, 1, 1, 1, 1])

    # Build plume times DatetimeIndex and matplotlib date numbers (times_num)
    try:
        times_dt = pd.to_datetime(ds_plume["times_plume"].values)
    except Exception:
        times_dt = pd.to_datetime(ds_plume["times_plume"])
    times_index = pd.to_datetime(times_dt)

    times_insitu = pd.to_datetime(ds_plume.insitu_times.values)
    try:
        insitu_tz = times_insitu.tz
    except Exception:
        insitu_tz = None
    if insitu_tz is not None:
        if getattr(times_index, 'tz', None) is None:
            try:
                times_index = times_index.tz_localize('UTC').tz_convert(insitu_tz)
            except Exception:
                times_index = times_index.tz_localize(insitu_tz)
        else:
            times_index = times_index.tz_convert(insitu_tz)
    else:
        try:
            times_index = times_index.tz_convert(None)
        except Exception:
            try:
                times_index = times_index.tz_localize(None)
            except Exception:
                pass
    if len(times_index) > 0 and times_index.isnull().any():
        first_valid = times_index[times_index.notnull()].tolist()[0]
        times_index = times_index.fillna(first_valid)
    times = times_index.to_pydatetime()
    times_num = mdates.date2num(times)

    # NO2 enhancement 2D plot
    ax0 = fig.add_subplot(gs[0, 0])
    X, Y = np.meshgrid(times_num, np.arange(ds_plume["no2_enhancement_c_back"].shape[0]))
    pcm = ax0.pcolormesh(X, Y, ds_plume["no2_enhancement_c_back"], shading='auto')
    fig.colorbar(pcm, ax=ax0, label=r"NO$_2$ Enhancement / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    tstr_display = pd.to_datetime(ds_plume.t).strftime('%Y-%m-%d %H:%M:%S')
    tstr_fname = pd.to_datetime(ds_plume.t).strftime('%Y%m%d_%H%M%S')
    ax0.set_title(f"MMSI {ds_plume.mmsi} NO2 around {tstr_display}")
    ax0.set_xlabel("Time")
    N = 3  # Show every 3rd VEA value
    vea_vals = ds_plume["vea"].values
    yticks = np.arange(0, len(vea_vals), N)
    ax0.set_yticks(yticks)
    ax0.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
    ax0.set_ylabel("VEA / °")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax0.get_xticklabels(), rotation=45)

    # Integrated NO2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(times_num, ds_plume["vertically_integrated_no2_enhancement_c_back"], color='tab:blue')
    ax1.set_title("Vertically integrated NO$_2$ enhancement")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Integrated NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Video image
    ax2 = fig.add_subplot(gs[0, 2])
    img_file = (img_dir / img_file)
    if img_file:
        fname = os.path.basename(img_file)
        fdir = os.path.dirname(img_file)
        zip_candidate = fdir + ".zip"
        shown = False
        # First try zip (file_dir + ".zip")
        if os.path.isfile(zip_candidate):
            try:
                with zipfile.ZipFile(zip_candidate, "r") as zf:
                    member = None
                    if fname in zf.namelist():
                        member = fname
                    else:
                        for m in zf.namelist():
                            if m.endswith("/" + fname) or m.endswith("\\" + fname) or m.endswith(fname):
                                member = m
                                break
                    if member:
                        data = zf.read(member)
                        img = Image.open(BytesIO(data))
                        ax2.imshow(img)
                        ax2.set_title(f"Video image {fname}")
                        ax2.axis("off")
                        shown = True
            except Exception:
                shown = False
        # Fallback: direct file on disk
        if not shown and os.path.exists(img_file):
            try:
                img = Image.open(img_file)
                ax2.imshow(img)
                ax2.set_title("Video image")
                ax2.axis("off")
                shown = True
            except Exception:
                shown = False
        if not shown:
            ax2.text(0.5, 0.5, "No image", ha="center", va="center")
            ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No image", ha="center", va="center")
        ax2.axis("off")

    min_lon, max_lon = lon1 - 0.05, lon1 + 0.05
    min_lat, max_lat = lat1 - 0.04, lat1 + 0.02

    tiler = cimgt.OSM()
    mercator = tiler.crs
    ax3 = fig.add_subplot(gs[0, 3], projection=mercator)
    ax3.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax3.add_image(tiler, 13, interpolation='bilinear', alpha=1.0)
    ax3.set_facecolor('white')  

    # Plot ship trajectory (last and next 10 minutes, no current position marker)

    if len(ds_plume["ship_ais_lons"]) > 1:
        ax3.plot(ds_plume["ship_ais_lons"], ds_plume["ship_ais_lats"], color='green', linewidth=3, label='Ship trajectory', transform=ccrs.PlateCarree())

    # Plot instrument location
    ax3.plot(lon1, lat1, marker='x', color='black', markersize=12, label='Instrument', transform=ccrs.PlateCarree())

    # Plot viewing direction as a line from instrument
    view_lat2 = lat2[passing_ship["Closest_Impact_Measurement_Index"]]
    view_lon2 = lon2[passing_ship["Closest_Impact_Measurement_Index"]]
    ax3.plot([lon1, view_lon2], [lat1, view_lat2], color='red', linewidth=3, linestyle='--', label='Viewing direction', transform=ccrs.PlateCarree())

    # Add wind arrow (bottom left, bigger)
    lon_margin = 0.007
    lat_margin = 0.007


    # Wind arrow in upper right
    wind_dir_rad = np.deg2rad(passing_ship['wind_dir'])
    wind_speed_mean = passing_ship['wind_speed']
    wind_arrow_length = 0.01
    wind_start_lat = max_lat - lat_margin - wind_arrow_length * np.cos(wind_dir_rad) / 2
    wind_start_lon = max_lon - lon_margin - 0.035 - wind_arrow_length * np.sin(wind_dir_rad) / np.cos(np.deg2rad(wind_start_lat))/2
    wind_end_lon = wind_start_lon + wind_arrow_length * np.sin(wind_dir_rad) / np.cos(np.deg2rad(wind_start_lat))
    wind_end_lat = wind_start_lat + wind_arrow_length * np.cos(wind_dir_rad)

    # White box for wind arrow
    box_width = 0.025
    box_height = 0.005
    box = mpatches.FancyBboxPatch(
        (max_lon - 2*lon_margin-0.03, max_lat - lat_margin + 0.001),
        box_width, box_height,
        boxstyle="round,pad=0.01",
        linewidth=0,
        facecolor='white',
        alpha=0.8, 
        transform=ccrs.PlateCarree(),
        zorder=9
    )
    ax3.add_patch(box)

    # Draw wind arrow
    ax3.arrow(
        wind_start_lon, wind_start_lat,
        wind_end_lon - wind_start_lon, wind_end_lat - wind_start_lat,
        color='tab:blue', width=0.001, head_width=0.006, head_length=0.006,
        length_includes_head=True, transform=ccrs.PlateCarree(), zorder=10
    )
    ax3.text(
        max_lon - lon_margin - 0.03 +0.006, max_lat - lat_margin - 0.005,
        f"Wind\n{wind_speed_mean:.1f} m/s\n{passing_ship['wind_dir']:.0f}°",
        color='tab:blue', fontsize=10, ha='left', va='bottom',
        transform=ccrs.PlateCarree(), zorder=11,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
    )

    ship_dir_rad = np.deg2rad(passing_ship["Mean_Course"])
    ship_arrow_length = 0.01
    ship_start_lat = min_lat + lat_margin + 0.025 - ship_arrow_length * np.cos(ship_dir_rad) / 2
    ship_start_lon = min_lon + lon_margin + 0.005 - ship_arrow_length * np.sin(ship_dir_rad) / np.cos(np.deg2rad(ship_start_lat)) / 2
    ship_end_lon = ship_start_lon + ship_arrow_length * np.sin(ship_dir_rad) / np.cos(np.deg2rad(ship_start_lat))
    ship_end_lat = ship_start_lat + ship_arrow_length * np.cos(ship_dir_rad)
    # White box for ship arrow

    ship_box = mpatches.FancyBboxPatch(
        (min_lon , min_lat +0.025 ),
        2*lon_margin, box_height,
        boxstyle="round,pad=0.01",
        linewidth=0,
        facecolor='white',
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        zorder=9
    )
    ax3.add_patch(ship_box)
    # Draw ship arrow
    ax3.arrow(
        ship_start_lon, ship_start_lat,
        ship_end_lon - ship_start_lon, ship_end_lat - ship_start_lat,
        color='green', width=0.001, head_width=0.006, head_length=0.006,
        length_includes_head=True, transform=ccrs.PlateCarree(), zorder=10
    )    
    ax3.text(
        min_lon + lon_margin - 0.005, min_lat + lat_margin + 0.013,
        f"Ship\n{passing_ship['Mean_Speed']:.1f} m/s",
        color='green', fontsize=10, ha='left', va='bottom',
        transform=ccrs.PlateCarree(), zorder=11,
        )
    ax3.set_facecolor('white')  # White background

    # Add latitude/longitude gridlines
    gl = ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                    linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax3.legend(loc='lower right', fontsize=10)
    ax3.set_title("Map: Ship, Wind, Instrument, Viewing Dir.")


    ax4 = fig.add_subplot(gs[0, 4])

    # Ensure in-situ comparison uses the same (naive) datetime objects
    # times (python datetimes) was created above and normalized to tz-naive UTC
    in_situ_mask = (times_insitu >= times[0]) & (times_insitu <= times[-1])    
    in_situ_times = times_insitu[in_situ_mask]
    in_situ_no2 = ds_plume['no2_insitu'][in_situ_mask]
    

    L = 870  # meters

    # Interpolate n_air to times_window
    n_air = ds_plume["p_0_insitu"][in_situ_mask] * 1e2 * 6.02214076e23 / (ds_plume["T_out_insitu"][in_situ_mask] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(in_situ_times))
    # Use the pandas DatetimeIndex for reindexing (preserves tz-awareness)
    try:
        n_air_aligned = n_air_interp.reindex(times_index, method='nearest').values
    except Exception:
        # Fallback: reindex using the python datetime array
        n_air_aligned = n_air_interp.reindex(pd.DatetimeIndex(times), method='nearest').values

    # IMPACT: n_NO2 = no2_data / (L * n_air) * 1e9
    # Apply rolling mean over 20 dim_0 (time) for IMPACT
    impact_enh_rolling = ds_plume["no2_enhancement_c_back"].isel(image_row=slice(5,9)).mean(dim="image_row").rolling(window_plume=20, center=True).mean()
    n_NO2_impact = impact_enh_rolling * 1e4 / (L * n_air_aligned) * 1e9

    # Convert LP-DOAS times to matplotlib date numbers for consistent plotting
    lp_available = ("lp_times_window" in ds_plume) and ("lp_no2_enhancement" in ds_plume)
    lp_times_num = np.array([])

    if lp_available:
        try:
            lp_times_dt = pd.to_datetime(ds_plume["lp_times_window"].values)
        except Exception:
            lp_times_dt = pd.to_datetime(ds_plume.get("lp_times_window", []))

        # If pandas Series / tz-aware, normalize to UTC-naive
        try:
            if getattr(lp_times_dt, 'dt', None) is not None and getattr(lp_times_dt.dt, 'tz', None) is not None:
                lp_times_dt = lp_times_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass

        lp_times = pd.to_datetime(lp_times_dt).to_pydatetime()
        if len(lp_times) > 0:
            lp_times_num = mdates.date2num(lp_times)
        else:
            lp_times_num = np.array([])

    # Plot IMPACT (always)
    ax4.plot(times_num, n_NO2_impact, color='orange', label="IMPACT n$_{NO_2}$ [ppb] (rolling mean)")

    # Plot LP-DOAS only if available and there are times
    if lp_available and lp_times_num.size > 0:
        try:
            lp_vals = ds_plume["lp_no2_enhancement"].values
        except Exception:
            lp_vals = np.asarray(ds_plume.get("lp_no2_enhancement", []))

        # If lengths match, plot directly; otherwise align by nearest length and warn
        if lp_vals is None or len(lp_vals) == 0:
            # nothing to plot
            pass
        else:
            if len(lp_vals) == len(lp_times_num):
                ax4.plot(lp_times_num, lp_vals, color='tab:blue', label="LP-DOAS n$_{NO_2}$ [ppb]")
            else:
                # Try to align by truncating the longer array (safe fallback)
                L = min(len(lp_vals), len(lp_times_num))
                ax4.plot(lp_times_num[:L], lp_vals[:L], color='tab:blue', label="LP-DOAS n$_{NO_2}$ [ppb]")
                print(f"Warning: LP-DOAS times ({len(lp_times_num)}) and values ({len(lp_vals)}) lengths differ; plotting first {L} points.")
    else:
        # LP-DOAS not present — only IMPACT will be shown
        pass
    #In-situ enhancement: subtract mean in reference window (for plotting, not converted)
    #if len(in_situ_times) > 0:
    #    in_situ_ref_mask = (ds_times >= times_window_naive[0] - pd.Timedelta(minutes=ref_offset)) & \
    #                       (ds_times < times_window_naive[0] - pd.Timedelta(minutes=ref_offset - 1.0))
    #    in_situ_ref = ds['c_no2'].values[in_situ_ref_mask]
    #    in_situ_enh = in_situ_no2 - np.nanmean(in_situ_ref) if len(in_situ_ref) > 0 else in_situ_no2
    #    ax4.plot(in_situ_times, in_situ_enh, color='tab:red', label="In-situ NO$_2$ enhancement [ppb]")
    #else:
    #    ax4.text(0.5, 0.5, "No in-situ data", ha="center", va="center", transform=ax4.transAxes)

    ax4.set_title("NO$_2$ (IMPACT, In-situ, LP-DOAS, all as ppb)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("NO$_2$ [ppb]")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=45)
    ax4.legend(loc="upper left")
    
    plt.tight_layout()
    if save:
        savepath = os.path.join(no2_out_dir, f"NO2_enhancement_subplot_{tstr_fname}_{ds_plume.mmsi}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_no2_enhancements_for_all_ships_full_overview(path_ship_passes, img_dir, out_dir, lat1, lon1, lat2, lon2, save=False):
    ship_passes = pd.read_csv(path_ship_passes, index_col=0, parse_dates=True)
    for idx, ship_pass_single in ship_passes.iterrows():
        if not os.path.isfile(ship_pass_single['plume_file']):
            print(f"Plume file {ship_pass_single['plume_file']} does not exist. Skipping.")
            continue
        ds_plume = xr.open_dataset(ship_pass_single['plume_file'])
        plot_ship_pass_subplot_v2(ds_plume, ship_pass_single, img_dir, out_dir, lat1, lon1, lat2, lon2, save=True)



def plot_wind_polar(ds, wind_dir_var='wind_dir', wind_speed_var='wind_speed', title='Wind Speed and Direction (Polar Plot)', save=False, out_dir=''):
    """
    Plots wind speed as a function of wind direction in a polar plot.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing wind direction and wind speed.
    wind_dir_var : str
        Name of the wind direction variable in degrees.
    wind_speed_var : str
        Name of the wind speed variable.
    title : str
        Title for the plot.
    """

    wind_dir_rad = np.deg2rad(ds[wind_dir_var].values)
    wind_speed = ds[wind_speed_var].values

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    sc = ax.scatter(wind_dir_rad, wind_speed, c=wind_speed, cmap='viridis', alpha=0.75)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    plt.colorbar(sc, ax=ax, label='Wind Speed (m/s)')
    ax.set_title(title)
    ax.set_rlabel_position(225)

    if save:
        savepath = os.path.join(out_dir, f"wind_polar_plots", f"wind_polar_plot_{ds.index[0].strftime('%y%m%d')}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_no2_enhancement_and_insitu(
    ds_impact,
    df_insitu,
    df_closest,
    title="NO2 time series and ship passes according to AIS",
    save=False, 
    out_dir=''
):
    """
    Plot NO2 enhancement (satellite) and in-situ NO2 with ship pass vertical lines.

    Parameters
    ----------
    ds_impact : xarray.Dataset
        Dataset containing NO2 enhancement and datetime.
    df_insitu : pd.DataFrame
        DataFrame with in-situ NO2 data (index: datetime, column: 'c_no2').
    df_closest : pd.DataFrame
        DataFrame with ship pass times (index: datetime).
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    # Plot vertical lines for closest approach times within window
    for t in df_closest.index:
        ax1.axvline(pd.to_datetime(t), color='red', alpha=0.5)

    # Plot NO2 enhancement
    line1, = ax1.plot(
        ds_impact["datetime"].isel(viewing_direction=0),
        ds_impact["NO2_enhancement_rolling_back"].mean(dim="viewing_direction"),
        label="NO2 Enhancement",
        color='blue'
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Satellite NO2 Enhancement")
    ax1.set_title(title)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Plot in situ NO2 on a secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        df_insitu.index,
        df_insitu['c_no2'],
        label="In Situ NO2",
        color='green'
    )
    ax2.set_ylabel("In Situ NO2")

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    if save:
        first_dt = pd.to_datetime(ds_impact["datetime"].values[0][0])
        savepath = os.path.join(out_dir, f"no2_enhancement_and_insitu", f"no2_enhancement_and_insitu_{first_dt.strftime('%y%m%d')}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_all_instruments_timeseries_VMR(
    df_lp_doas,
    df_insitu,
    coarsened_ds,
    VMR_NO2,
    df_closest=None,
    title="NO$_2$ measurements from all instruments",
    save=False,
    out_dir=''
):
    """
    Plots LP-DOAS, In Situ, and IMPACT NO2 timeseries with ship pass vertical lines.

    Parameters
    ----------
    df_lp_doas : pd.DataFrame
        LP-DOAS data (index: datetime, column: 'Fit Coefficient (NO2)').
    df_insitu : pd.DataFrame
        In-situ data (index: datetime, column: 'c_no2').
    coarsened_ds : xr.Dataset
        Coarsened IMPACT dataset with 'datetime' and NO2 VMR.
    VMR_NO2 : xr.DataArray
        Path-averaged NO2 VMR from IMPACT.
    df_closest : pd.DataFrame, optional
        DataFrame with ship pass times and categories.
    start_time, end_time : datetime, optional
        Time window for plotting.
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()


    # Plot ship pass vertical lines using df_closest and styles_from_ship_category
    if df_closest is not None:
        for idx, passing_ship in df_closest.iterrows():
            category = passing_ship.get("ship_category")
            label = styles_from_ship_category("label", category)
            # Only add label if not already present
            if label not in ax1.get_legend_handles_labels()[1]:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=label
                )
            else:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=""
                )

    # Plot LP-DOAS
    ax1.plot(
        df_lp_doas.index,
        df_lp_doas['Fit Coefficient (NO2)'],
        label="LP-DOAS"
    )

    # Plot In Situ NO2
    ax1.plot(
        df_insitu.index,
        df_insitu['c_no2'],
        label="In Situ", color="green"
    )

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel("In-Situ & LP-DOAS VMR / ppb", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    #ax1.set_ylim(0, 35)

    # Second y-axis: IMPACT
    ax2 = ax1.twinx()
    ax2.plot(
        coarsened_ds["datetime"].isel(viewing_direction=0),
        VMR_NO2.isel(viewing_direction=slice(4,8)).mean(dim="viewing_direction"),
        label="IMPACT", color='C1', zorder=1
    )
    ax2.set_ylabel("IMPACT VMR / ppb", fontsize=14, color='C1')
    ax2.tick_params(axis='both', labelsize=12)
    #ax2.set_ylim(0, 3.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

    plt.tight_layout()
    if save:
        first_dt = pd.to_datetime(coarsened_ds["datetime"].values[0][0])
        savepath = os.path.join(out_dir, f"timeseries_VMR", f"timeseries_VMR_{first_dt.strftime('%y%m%d')}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def plot_all_instruments_timeseries_SC(
    df_lp_doas_SC,
    df_insitu,
    coarsened_ds,
    df_closest=None,
    ylim_left=None,
    ylim_right=None,
    title="NO$_2$ measurements from all instruments",
    save=False,
    out_dir=''
):
    """
    Plots LP-DOAS, In Situ, and IMPACT NO2 timeseries with ship pass vertical lines.

    """
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()


    # Plot ship pass vertical lines using df_closest and styles_from_ship_category
    if df_closest is not None:
        for idx, passing_ship in df_closest.iterrows():
            category = passing_ship.get("ship_category")
            label = styles_from_ship_category("label", category)
            # Only add label if not already present
            if label not in ax1.get_legend_handles_labels()[1]:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=label
                )
            else:
                ax1.axvline(
                    idx,
                    color='red',
                    alpha=styles_from_ship_category("transparency", category),
                    linestyle=styles_from_ship_category("linestyle", category),
                    linewidth=2,
                    label=""
                )

    # Plot LP-DOAS
    ax1.plot(
        df_lp_doas_SC.index,
        df_lp_doas_SC['Fit Coefficient (NO2)'],
        label="LP-DOAS"
    )

    # Plot In Situ NO2
    ax1.plot(
        coarsened_ds["datetime"].isel(viewing_direction=0),
        coarsened_ds["a[NO2]"].isel(viewing_direction=slice(5,9)).mean(dim="viewing_direction"),
        label="IMPACT"
    )

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_ylim(0, ylim_left)

    # Second y-axis: IMPACT
    ax2 = ax1.twinx()
    ax2.plot(df_insitu.index, df_insitu["c_no2"], color='green',zorder=1, alpha=0.6, label="In Situ")
    ax2.set_ylabel(r"In Situ NO$_2$ / ppb", color="green", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, ylim_right) 

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)


    plt.tight_layout()
    if save:
        first_dt = pd.to_datetime(coarsened_ds["datetime"].values[0][0])
        savepath = os.path.join(out_dir, f"timeseries_SC", f"timeseries_SC_{first_dt.strftime('%y%m%d')}.png")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_timeseries_and_enhancement(ds_impact_masked, time, orig , fit, enhancement, title= f"Correction for NO$_2$-Background"):

    plt.figure(figsize=(15, 5))
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.plot(pd.to_datetime(time), ds_impact_masked[orig], label='NO$_2$ dSCDs', alpha=0.6, linewidth=2)
    ax1.plot(pd.to_datetime(time), ds_impact_masked[fit], label='Fit', color='red', linewidth=2)
    ax1.plot(pd.to_datetime(time), ds_impact_masked[enhancement], label='Enhancements', color='red', linewidth=2)

    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"NO$_2$ dSCD / $\frac{\#\mathrm{molec.}}{\mathrm{cm}^2}$", fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

def plot_enhancement_impact_and_insitu(
    ds_impact_masked,
    df_lp_doas_SC,
    df_insitu,
    df_closest=None,
    start_time=None,
    end_time=None,
    no2_out_dir=None,
    date=None
    ):

    L_impact = 870  # meters (IMPACT path length)
    L_lp = 870      # meters (LP-DOAS path length, adjust if different)

    # Calculate n_air (air number density) at each time (in molec/m^3)
    n_air = df_insitu["p_0"] * 1e2 * 6.02214076e23 / (df_insitu["T_in"] + 273.15) / 8.314
    n_air_interp = pd.Series(n_air.values, index=pd.to_datetime(df_insitu.index))

    # Align n_air to IMPACT and LP-DOAS times
    n_air_impact = n_air_interp.reindex(pd.to_datetime(ds_impact_masked["datetime"]).tz_localize('UTC'), method='nearest').values
    n_air_lp = n_air_interp.reindex(df_lp_doas_SC.index, method='nearest')

    # Convert enhancements to VMR [ppb]
    # IMPACT: enhancement in 1/cm^2, convert to 1/m^2, then to VMR
    ds_impact_masked["Enhancement_VMR_lpdoas"] = ds_impact_masked["NO2_enhancement_polynomial"] * 1e4 / (L_impact * n_air_impact) * 1e9  # [ppb]

    # LP-DOAS: enhancement in 1/cm^2, convert to 1/m^2, then to VMR
    lpdoas_vmr = df_lp_doas_SC["NO2_enhancement_polynomial"] * 1e4 / (2 * L_lp * n_air_lp) * 1e9  # [ppb], factor 2 for roundtrip

    # --- Plot with ship passes colored by ship height ---
    plt.figure(figsize=(15, 5))
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel("Time (UTC)", fontsize=14)
    ax1.set_ylabel(r"$\Delta$VMR NO$_2$ / ppb", fontsize=14)
    ax1.set_title(f"NO$_2$ Enhancements for IMPACT and LP-DOAS", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    add_ship_lines = True
    if add_ship_lines and df_closest is not None and start_time is not None and end_time is not None:
        for idx, passing_ship in df_closest.iterrows():
            t_ship = pd.to_datetime(passing_ship["UTC_Time"]) if "UTC_Time" in passing_ship else idx
            category = passing_ship.get("ship_category")
            if (t_ship >= start_time) and (t_ship <= end_time):
                label = styles_from_ship_category("label", category)
                # Only add label if not already present
                if label not in ax1.get_legend_handles_labels()[1]:
                    ax1.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=label
                    )
                else:
                    ax1.axvline(
                        t_ship,
                        color='red',
                        alpha=styles_from_ship_category("transparency", category),
                        linestyle=styles_from_ship_category("linestyle", category),
                        linewidth=2,
                        label=""
                    )


    ax1.plot(df_lp_doas_SC.index, lpdoas_vmr, label='LP-DOAS', alpha=0.8)
    ax1.plot(ds_impact_masked["datetime"], ds_impact_masked["Enhancement_VMR_lpdoas"], label='IMPACT')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(no2_out_dir, f"NO2_enhancement_vmr_{date}.png"))


def plot_no2_enhancement_with_plume_mask(ds_plume, mask, out_dir, date,):
    """Plot NO2 enhancement using pcolormesh with time on the x-axis and save to disk.

    The function will try to find a time coordinate inside `ds_plume` by looking
    for names in `time_coord_names`. If a matching coordinate is found and its
    length matches the second dimension of the data array, the x-axis will show
    datetimes. Otherwise the data is plotted without a time axis.
    """


    data = ds_plume["no2_enhancement_interp"].values
    ny, nx = data.shape

    # find time coordinate
    times = pd.to_datetime(ds_plume["times_plume"].values)

    fig, ax = plt.subplots(figsize=(10, 6))
    # convert to matplotlib datenums and build edges for pcolormesh
    xnum = mdates.date2num(times.to_pydatetime())
    # compute edges by midpoints
    dx = np.diff(xnum)
    if len(dx) > 0:
        left = xnum[0] - dx[0] / 2.0
        right = xnum[-1] + dx[-1] / 2.0
        xedges = np.concatenate(([left], xnum[:-1] + dx / 2.0, [right]))
    else:
        # single column
        xedges = np.array([xnum[0] - 0.5, xnum[0] + 0.5])
    yedges = np.arange(ny + 1)

    mesh = ax.pcolormesh(xedges, yedges, data, cmap="viridis", shading="auto")
    # contour needs x coords for proper alignment; use mesh coordinates
    Xc, Yc = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0)
    ax.contour(Xc, Yc, mask.astype(int), levels=[0.5], colors="red", linewidths=1.5)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # show time as hours:minutes:seconds on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    vea_vals = ds_plume["vea"].values 
    yticks = np.arange(0, len(vea_vals), 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
    ax.set_ylabel("VEA / °")
    ax.set_xlabel("Time (UTC)")
    fig.autofmt_xdate()
    mmsi = ds_plume.attrs.get('mmsi', 'unknown_mmsi')
    ax.set_title(f"NO$_2$ enhancement with plume mask, date: {date}, mmsi: {mmsi}")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("NO2 enhancement")
    out_folder = out_dir / f"plume_mask" / f"plumes_{date}"
    out_folder.mkdir(parents=True, exist_ok=True)
    try:
        t_attr = ds_plume.attrs.get('t')
        tstr = pd.to_datetime(t_attr).strftime('%Y%m%d_%H%M%S') if t_attr is not None else "unknown_time"
    except Exception:
        tstr = "unknown_time"
    fname = out_folder / f"no2_enhancement_with_plume_mask_{date}_{tstr}_{mmsi}.png"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_reference_image_with_plume_mask(ref_image, mask, ds_plume, out_dir, date):    


    data = ref_image.values
    ny, nx = data.shape

    # try to obtain time coordinate named 'time_ref' (fall back to other names)
    
    times = pd.to_datetime(ds_plume["times_ref"].values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # build xedges depending on whether times are available
    xnum = mdates.date2num(times.to_pydatetime())
    dx = np.diff(xnum)
    if len(dx) > 0:
        left = xnum[0] - dx[0] / 2.0
        right = xnum[-1] + dx[-1] / 2.0
        xedges = np.concatenate(([left], xnum[:-1] + dx / 2.0, [right]))
    else:
        xedges = np.array([xnum[0] - 0.5, xnum[0] + 0.5])


    # y edges correspond to VEA dimension
    yedges = np.arange(ny + 1)

    mesh = ax.pcolormesh(xedges, yedges, data, cmap="viridis", shading="auto")

    # contour overlay (align to cell centers)
    Xc, Yc = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0)
    ax.contour(Xc, Yc, mask.astype(int), levels=[0.5], colors="red", linewidths=1.5)

    # y ticks from 'vea' coordinate if available
    if "vea" in ds_plume:
        vea_vals = ds_plume["vea"].values
        step = max(1, len(vea_vals) // 8)
        yticks = np.arange(0, len(vea_vals), step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{float(vea_vals[i]):.1f}°" for i in yticks])
    else:
        ax.set_yticks(np.arange(0, ny, max(1, ny // 8)))

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("VEA / °")
    ax.set_title("Reference NO$_2$ variability, date: {}".format(date))

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("dNO2")

    # build output folder and filename similar to other plotting function
    out_base = out_dir / f"reference_quality" / f"plumes_{date}_upwind"
    sub = "yes_plume" if mask.sum() > 0 else "no_plume"
    out_folder = out_base / sub
    out_folder.mkdir(parents=True, exist_ok=True)

    try:
        t_attr = ds_plume.attrs.get('t')
        tstr = pd.to_datetime(t_attr).strftime('%Y%m%d_%H%M%S') if t_attr is not None else "unknown_time"
    except Exception:
        tstr = "unknown_time"
    mmsi = ds_plume.attrs.get('mmsi', 'unknown_mmsi')
    fname = out_folder / f"reference_image_with_plume_mask_{date}_{tstr}_{mmsi}.png"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)