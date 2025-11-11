import numpy as np
import pandas as pd
import xarray as xr
from numpy.fft import fft, ifft, fftfreq


def rolling_background_enh(ds, window_size=500): 
    ds["NO2_enhancement_rolling_back"] = ds["a[NO2]"] - ds["a[NO2]"].rolling(dim_0=window_size).mean()
    ds["NO2_rolling_background"] = ds["a[NO2]"].rolling(dim_0=window_size).mean()
    ds["O4_enhancement_rolling_back"] = ds["a[O4]"] - ds["a[O4]"].rolling(dim_0=window_size).mean()
    ds["O4_rolling_background"] = ds["a[O4]"].rolling(dim_0=window_size).mean()
    return ds

def upwind_constant_background_enh(row, ds_impact, measurement_times, ship_passes, window_minutes=(1, 3), ref_search_minutes=60, ref_window_minutes=1,  do_lp=False, df_lp = None):
    """
    Subtracts background for a single ship pass (row from ship_passes).
    Returns: dict with keys: mmsi, t, no2_data, times_window, window, window_ref, ref_found
    """
    mmsi = row["MMSI"]
    t = pd.to_datetime(row.name)#.tz_localize("UTC")
    time_diff = row["Closest_Impact_Measurement_Time_Diff"].total_seconds()
    if time_diff > 60:
        return None
    window = ((measurement_times >= t - pd.Timedelta(minutes=window_minutes[0])) & (measurement_times < t + pd.Timedelta(minutes=window_minutes[1])))
    # Find reference window
    ref_found = False
    ref_offset = 3
    while not ref_found and ref_offset < ref_search_minutes:
        ref_start = t - pd.Timedelta(minutes=ref_offset)
        ref_end = t - pd.Timedelta(minutes=ref_offset - ref_window_minutes)
        window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
        ref_times = measurement_times[window_ref]
        other_ships_in_window = False
        for idx2, row2 in ship_passes.iterrows():
            if row2["MMSI"] == mmsi:
                continue
            other_t = pd.to_datetime(idx2).tz_localize("UTC") if pd.to_datetime(idx2).tzinfo is None else pd.to_datetime(idx2)
            if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                other_ships_in_window = True
                break
        if not other_ships_in_window and window_ref.sum() > 0:
            ref_found = True
        else:
            ref_offset += 1
    if not ref_found:
        print(f"No clean reference window found for MMSI {mmsi} at {t}")
        return None
    no2_enhancement = ds_impact["a[NO2]"].isel(dim_0=window) - ds_impact["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
    vertically_integrated_no2 = no2_enhancement.sum(dim="viewing_direction")
    o4_enhancement = ds_impact["a[O4]"].isel(dim_0=window) - ds_impact["a[O4]"].isel(dim_0=window_ref).mean(dim="dim_0")

    ds =xr.Dataset(
            data_vars=dict(
                no2=(["image_row", "window_plume"], ds_impact["a[NO2]"].isel(dim_0=window).values),
                o4=(["image_row", "window_plume"], ds_impact["a[O4]"].isel(dim_0=window).values),
                o3=(["image_row", "window_plume"], ds_impact["a[O3]"].isel(dim_0=window).values),
                h2o=(["image_row", "window_plume"], ds_impact["a[H2O]"].isel(dim_0=window).values),
                ring=(["image_row", "window_plume"], ds_impact["a[RING]"].isel(dim_0=window).values),
                rms=(["image_row", "window_plume"], ds_impact["rms"].isel(dim_0=window).values),
                no2_enhancement_c_back=(["image_row", "window_plume"], no2_enhancement.values),
                o4_enhancement_c_back=(["image_row", "window_plume"], o4_enhancement.values),
                vertically_integrated_no2_enhancement_c_back=(["window_plume"], vertically_integrated_no2.values),
                times_plume=(["window_plume"], np.array(pd.to_datetime(measurement_times[window]), dtype='datetime64[ns]')),
                no2_ref=(["image_row", "window_ref"], ds_impact["a[NO2]"].isel(dim_0=window_ref).values),
                o4_ref=(["image_row", "window_ref"], ds_impact["a[O4]"].isel(dim_0=window_ref).values),
                times_ref=(["window_ref"], np.array(pd.to_datetime(measurement_times[window_ref]), dtype='datetime64[ns]')),
                vea=(["image_row"], ds_impact["los"].isel(dim_0=window[0]).values),
                vaa=(["window"], ds_impact["viewing-azimuth-angle"].isel(viewing_direction=0).values),
                no2_enhancement_rolling_back=(["image_row", "window_plume"], ds_impact["NO2_enhancement_rolling_back"].isel(dim_0=window).values),
                o4_enhancement_rolling_back=(["image_row", "window_plume"], ds_impact["O4_enhancement_rolling_back"].isel(dim_0=window).values),
                no2_rolling_background=(["image_row", "window_plume"], ds_impact["NO2_rolling_background"].isel(dim_0=window).values),
                o4_rolling_background=(["image_row", "window_plume"], ds_impact["O4_rolling_background"].isel(dim_0=window).values),
                no2_rolling=(["image_row", "window_plume"], ds_impact["NO2_rolling"].isel(dim_0=window).values),
                o4_rolling=(["image_row", "window_plume"], ds_impact["O4_rolling"].isel(dim_0=window).values),

            ),
            coords=dict(
                window_plume=ds_impact["dim_0"].isel(dim_0=window).values,
                window_ref=ds_impact["dim_0"].isel(dim_0=window_ref).values,
                image_row=ds_impact["viewing_direction"].values,
            ),
            attrs=dict(
                mmsi=str(mmsi),
                t=str(t),
                plume_number=str(row["Plume_number"]),
                ref_found=str(ref_found)
            )
        )
    if df_lp is not None:
        lp_window = ((df_lp.index >= t - pd.Timedelta(minutes=window_minutes[0])) & (df_lp.index < t + pd.Timedelta(minutes=window_minutes[1])))
        lp_window_ref = ((df_lp.index >= ref_start) & (df_lp.index < ref_end))
        lp_no2_enhancement = df_lp['Fit Coefficient (NO2)'][lp_window] - df_lp['Fit Coefficient (NO2)'][lp_window_ref].mean()


        #add coord lp_window = np.where(lp_window)[0]
        ds = ds.assign_coords(
            lp_window=np.where(lp_window)[0],
            lp_window_ref=np.where(lp_window_ref)[0]
        )
        #add vars enhancement and time and background
        ds = ds.assign(
            lp_no2=(["lp_window"], df_lp['Fit Coefficient (NO2)'][lp_window].values),
            lp_rms=(["lp_window"], df_lp['RMS'][lp_window].values),
            lp_no2_enhancement=(["lp_window"], lp_no2_enhancement.values),
            lp_times_window=(["lp_window"], df_lp.index[lp_window]),
            lp_no2_ref=(["lp_window_ref"], df_lp['Fit Coefficient (NO2)'][lp_window_ref].values),
            lp_times_window_ref=(["lp_window_ref"], df_lp.index[lp_window_ref])
        )
    return ds

def upwind_downwind_interp_background_enh(ds, row, ds_impact, measurement_times, ship_passes, window_minutes=(1, 3), ref_search_minutes=60, ref_window_minutes=1,  do_lp=False, df_lp = None):
    """
    Subtracts background for a single ship pass (row from ship_passes).
    Returns: dict with keys: mmsi, t, no2_data, times_window, window, window_ref, ref_found
    """
    mmsi = row["MMSI"]
    t = pd.to_datetime(row.name)#.tz_localize("UTC")
    time_diff = row["Closest_Impact_Measurement_Time_Diff"].total_seconds()
    if time_diff > 60:
        return None
    window = ((measurement_times >= t - pd.Timedelta(minutes=window_minutes[0])) & (measurement_times < t + pd.Timedelta(minutes=window_minutes[1])))
    # Find reference window
    ref_found = False
    ref_offset = 3
    while not ref_found and ref_offset < ref_search_minutes:
        ref_start = t - pd.Timedelta(minutes=ref_offset)
        ref_end = t - pd.Timedelta(minutes=ref_offset - ref_window_minutes)
        window_ref = ((measurement_times >= ref_start) & (measurement_times < ref_end))
        ref_times = measurement_times[window_ref]
        other_ships_in_window = False
        for idx2, row2 in ship_passes.iterrows():
            if row2["MMSI"] == mmsi:
                continue
            other_t = pd.to_datetime(idx2).tz_localize("UTC") if pd.to_datetime(idx2).tzinfo is None else pd.to_datetime(idx2)
            if any(abs((ref_times - other_t).total_seconds()) < 5*60):
                other_ships_in_window = True
                break
        if not other_ships_in_window and window_ref.sum() > 0:
            ref_found = True
        else:
            ref_offset += 1
    if not ref_found:
        print(f"No clean upwind reference window found for MMSI {mmsi} at {t}")
        return ds
    
    downwind_ref_found = False
    downwind_ref_offset = window_minutes[1] + 3
    while not downwind_ref_found and downwind_ref_offset < ref_search_minutes:
        downwind_ref_start = t + pd.Timedelta(minutes=downwind_ref_offset)
        downwind_ref_end = t + pd.Timedelta(minutes=downwind_ref_offset + ref_window_minutes)
        downwind_window_ref = ((measurement_times >= downwind_ref_start) & (measurement_times < downwind_ref_end))
        downwind_ref_times = measurement_times[downwind_window_ref]
        other_ships_in_window = False
        for idx2, row2 in ship_passes.iterrows():
            if row2["MMSI"] == mmsi:
                continue
            other_t = pd.to_datetime(idx2).tz_localize("UTC") if pd.to_datetime(idx2).tzinfo is None else pd.to_datetime(idx2)
            if any(abs((downwind_ref_times - other_t).total_seconds()) < 5*60):
                other_ships_in_window = True
                break
        if not other_ships_in_window and downwind_window_ref.sum() > 0:
            downwind_ref_found = True
        else:
            downwind_ref_offset += 1
    if not downwind_ref_found:
        print(f"No clean downwind reference window found for MMSI {mmsi} at {t}")
        return ds


    up_mean = ds_impact["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
    down_mean = ds_impact["a[NO2]"].isel(dim_0=downwind_window_ref).mean(dim="dim_0")

    # reference center times
    t_ref_center = pd.to_datetime(measurement_times[window_ref]).mean()
    t_down_center = pd.to_datetime(measurement_times[downwind_window_ref]).mean()

    # times inside the plume window
    times_window = pd.to_datetime(measurement_times[window])
    # normalized interpolation factor (0 -> upwind, 1 -> downwind)
    denom = (t_down_center - t_ref_center).total_seconds()
    alpha = ((times_window - t_ref_center).total_seconds() / denom).astype(float)
    alpha = np.clip(alpha, 0.0, 1.0)

    # make alpha an xarray aligned to dim_0 of the selected window
    dim0_coords = ds_impact["dim_0"].isel(dim_0=window).values
    alpha_da = xr.DataArray(alpha, dims=("dim_0",), coords={"dim_0": dim0_coords})

    # expand up/down means to include dim_0 so broadcasting works
    up_exp = up_mean.expand_dims(dim_0=alpha_da.coords["dim_0"])
    down_exp = down_mean.expand_dims(dim_0=alpha_da.coords["dim_0"])

    # interpolated background for each time step in window
    interp_bg = (1 - alpha_da) * up_exp + alpha_da * down_exp

    # final enhancement: observed minus interpolated background
    no2_enhancement_interp = ds_impact["a[NO2]"].isel(dim_0=window) - interp_bg
    vertically_integrated_no2_interp = no2_enhancement_interp.sum(dim="viewing_direction")

    # also compute O4 enhancement using upwind mean -> downwind mean interpolation (same procedure)
    up_mean_o4 = ds_impact["a[O4]"].isel(dim_0=window_ref).mean(dim="dim_0")
    down_mean_o4 = ds_impact["a[O4]"].isel(dim_0=downwind_window_ref).mean(dim="dim_0")
    up_o4_exp = up_mean_o4.expand_dims(dim_0=alpha_da.coords["dim_0"])
    down_o4_exp = down_mean_o4.expand_dims(dim_0=alpha_da.coords["dim_0"])
    interp_bg_o4 = (1 - alpha_da) * up_o4_exp + alpha_da * down_o4_exp
    o4_enhancement_interp = ds_impact["a[O4]"].isel(dim_0=window) - interp_bg_o4
    #introduce new_coord window_ref_down
    ds = ds.assign_coords(
        window_ref_down=np.where(downwind_window_ref)[0],
    )
    ds = ds.assign(
        no2_enhancement_interp=(["image_row", "window_plume"], no2_enhancement_interp.values),
        vertically_integrated_no2_enhancement_interp=(["window_plume"], vertically_integrated_no2_interp.values),
        o4_enhancement_interp=(["image_row", "window_plume"], o4_enhancement_interp.values),
        times_ref_down=(["window_ref_down"], np.array(pd.to_datetime(measurement_times[downwind_window_ref]), dtype='datetime64[ns]')),
        no2_ref_down=(["image_row", "window_ref_down"], ds_impact["a[NO2]"].isel(dim_0=downwind_window_ref).values),
        o4_ref_down=(["image_row", "window_ref_down"], ds_impact["a[O4]"].isel(dim_0=downwind_window_ref).values),
    )
    return ds

def polynomial_background_enh(ds_impact_masked, degree=8): 
    time_diff = pd.to_datetime(ds_impact_masked["datetime"]) - pd.to_datetime(ds_impact_masked["datetime"])[0]
    x = time_diff.total_seconds().astype(float)
    y = ds_impact_masked["a[NO2]"]

    # Fit 6th degree polynomial
    coeffs = np.polyfit(x, y, degree)
    poly_fit = np.polyval(coeffs, x)

    ds_impact_masked["NO2_polynomial"] = xr.DataArray(
    poly_fit,
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )
    ds_impact_masked["NO2_enhancement_polynomial"] = xr.DataArray(
    y - poly_fit,
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )

    return ds_impact_masked

def polynomial_background_enh_lp_doas(df_lp_doas, degree = 8):

    x_lp = (df_lp_doas.index - df_lp_doas.index[0]).total_seconds().astype(float)
    y_lp = df_lp_doas['Fit Coefficient (NO2)']

    coeffs_lp = np.polyfit(x_lp, y_lp, degree)
    poly_fit_lp = np.polyval(coeffs_lp, x_lp)

    # Calculate enhancement (detrended)
    lpdoas_enhancement = y_lp - poly_fit_lp
    df_lp_doas['NO2_polynomial'] = poly_fit_lp
    df_lp_doas['NO2_enhancement_polynomial'] = lpdoas_enhancement

    return df_lp_doas

def fft_background_enh(ds_impact_masked, t_cut = 3000):

    time_diff = pd.to_datetime(ds_impact_masked["datetime"]) - pd.to_datetime(ds_impact_masked["datetime"])[0]
    x = time_diff.total_seconds().astype(float)
    dt = np.median(np.diff(x)) #!!! todo: this fft assumes evenly spaced data, creating it by interpolation should be valid
    N = len(x)

    # FFT
    Y = fft(ds_impact_masked["a[NO2]"])
    freqs = fftfreq(N, d=dt)  # in Hz

    f_cut = 1/t_cut  # Hz

    # Zero out frequencies with |f| < f_cut (i.e., periods > 10 min)
    Y_filtered = Y.copy()
    Y_filtered[np.abs(freqs) < f_cut] = 0

    # Inverse FFT to get filtered signal
    ds_impact_masked["NO2_fft_filter"] = xr.DataArray(
    np.real(ifft(Y_filtered)),
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )
    ds_impact_masked["NO2_enhancements_fft_filter"] = xr.DataArray(
    ds_impact_masked["a[NO2]"] - ds_impact_masked["NO2_fft_filter"],
    dims=ds_impact_masked["dim_0"].dims,  # or the appropriate dims
    coords=ds_impact_masked["dim_0"].coords,  # or other matching coords
    )

    return ds_impact_masked

def fft_background_enh_lp_doas(df_lp_doas, t_cut = 3000):
    x_lp = (df_lp_doas.index - df_lp_doas.index[0]).total_seconds().astype(float)
    y_lp = df_lp_doas['Fit Coefficient (NO2)']
    # Sampling interval in seconds for LP-DOAS (assumes x_lp is in seconds and evenly spaced)
    dt_lp = np.median(np.diff(x_lp))
    N_lp = len(y_lp)

    # FFT
    Y_lp = fft(y_lp)
    freqs_lp = fftfreq(N_lp, d=dt_lp)  # in Hz

    # 10 min period = 600 s, so frequency = 1/600 Hz
    f_cut_lp = 1/t_cut  # Hz

    # Zero out frequencies with |f| < f_cut_lp (i.e., periods > 10 min)
    Y_lp_filtered = Y_lp.copy()
    Y_lp_filtered[np.abs(freqs_lp) < f_cut_lp] = 0

    # Inverse FFT to get filtered signal
    df_lp_doas["NO2_fft_filter"] = np.real(ifft(Y_lp_filtered))

    # Plot original and filtered signals
    df_lp_doas["NO2_enhancements_fft_filter"] = y_lp - df_lp_doas["NO2_fft_filter"]

    return df_lp_doas