import numpy as np
import pandas as pd
import xarray as xr
from numpy.fft import fft, ifft, fftfreq
import SEICOR.plumes


def rolling_background_enh(ds, window_size=500): 
    ds["NO2_enhancement_rolling_back"] = ds["a[NO2]"] - ds["a[NO2]"].rolling(dim_0=window_size).mean()
    ds["NO2_rolling_background"] = ds["a[NO2]"].rolling(dim_0=window_size).mean()
    ds["O4_enhancement_rolling_back"] = ds["a[O4]"] - ds["a[O4]"].rolling(dim_0=window_size).mean()
    ds["O4_rolling_background"] = ds["a[O4]"].rolling(dim_0=window_size).mean()
    return ds

def find_reference_window(t, mmsi, ds_impact, measurement_times, ship_passes, direction="up",
                          start_offset=3, ref_search_minutes=60, ref_window_minutes=1,
                          ref_min_span_seconds=40, ref_min_count=100, other_ship_minutes=5,
                          ztest_p_threshold=0.15, ztest_min_cluster_size=20):
    """Searches a clean reference window of `ref_window_minutes` before (direction="up") or
    after (direction="down") the ship pass at `t`, stepping 1 min per try from `start_offset`
    up to `ref_search_minutes`.

    A window is accepted if no other ship pass lies within `other_ship_minutes` of it, it holds at
    least `ref_min_count` measurements covering at least `ref_min_span_seconds`, and the z-test
    finds no plume in it.

    Returns (window_ref, ref_start, ref_end) with window_ref a boolean mask over measurement_times,
    or None if no clean window was found.
    """
    other_times = pd.DatetimeIndex(pd.to_datetime(ship_passes.index[ship_passes["MMSI"] != mmsi]))
    if other_times.tz is None:
        other_times = other_times.tz_localize("UTC")
    other_times = other_times.tz_convert("UTC").values
    min_separation = np.timedelta64(int(other_ship_minutes * 60), "s")

    offset = start_offset
    while offset < ref_search_minutes:
        if direction == "up":
            ref_start = t - pd.Timedelta(minutes=offset)
            ref_end = t - pd.Timedelta(minutes=offset - ref_window_minutes)
        else:
            ref_start = t + pd.Timedelta(minutes=offset)
            ref_end = t + pd.Timedelta(minutes=offset + ref_window_minutes)
        offset += 1

        window_ref = (measurement_times >= ref_start) & (measurement_times < ref_end)
        n_ref = int(window_ref.sum())
        if n_ref == 0:
            continue
        ref_times = measurement_times[window_ref]
        # other ships passing close to the reference window
        if other_times.size and (np.abs(ref_times.values[:, None] - other_times[None, :]) < min_separation).any():
            continue
        # minimum number of samples and timespan
        span_seconds = (ref_times.max() - ref_times.min()).total_seconds()
        if n_ref < ref_min_count or span_seconds < float(ref_min_span_seconds):
            continue
        # make sure there is no plume in the reference window
        no2_ref = ds_impact["a[NO2]"].isel(dim_0=window_ref)
        mask = SEICOR.plumes.detect_plume_ztest((no2_ref - no2_ref.mean(dim="dim_0")).values,
                                                p_threshold=ztest_p_threshold, min_cluster_size=ztest_min_cluster_size)
        if mask.sum() == 0:
            return window_ref, ref_start, ref_end
    return None

def upwind_constant_background_enh(row, ds_impact, measurement_times, ship_passes,
                                    window_minutes=(1, 3), ref_search_minutes=60,
                                    ref_window_minutes=1, ref_min_span_seconds=40,
                                    ref_min_count=100, do_lp=False, df_lp_310=None, df_lp_365=None,
                                    ref_start_offset=3, other_ship_minutes=5,
                                    ref_ztest_p_threshold=0.15, ref_ztest_min_cluster_size=20):
    """Subtracts background for a single ship pass (row from ship_passes).

    This variant enforces minimum quality for the chosen reference window: it must
    contain at least `ref_min_count` measurements and cover at least
    `ref_min_span_seconds` seconds.
    """

    mmsi = row["MMSI"]
    t = pd.to_datetime(row.name)
    time_diff = row["Closest_Impact_Measurement_Time_Diff"].total_seconds()
    if time_diff > 60:
        return None

    window = ((measurement_times >= t - pd.Timedelta(minutes=window_minutes[0])) &
              (measurement_times < t + pd.Timedelta(minutes=window_minutes[1])))

    # Find upwind reference window
    upwind = find_reference_window(t, mmsi, ds_impact, measurement_times, ship_passes, direction="up",
                                   start_offset=ref_start_offset, ref_search_minutes=ref_search_minutes,
                                   ref_window_minutes=ref_window_minutes,
                                   ref_min_span_seconds=ref_min_span_seconds, ref_min_count=ref_min_count,
                                   other_ship_minutes=other_ship_minutes,
                                   ztest_p_threshold=ref_ztest_p_threshold,
                                   ztest_min_cluster_size=ref_ztest_min_cluster_size)
    if upwind is None:
        print(f"No clean reference window found for MMSI {mmsi} at {t}")
        return None
    window_ref, ref_start, ref_end = upwind
    ref_found = True

    no2_enhancement = ds_impact["a[NO2]"].isel(dim_0=window) - ds_impact["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
    vertically_integrated_no2 = no2_enhancement.sum(dim="viewing_direction")
    o4_enhancement = ds_impact["a[O4]"].isel(dim_0=window) - ds_impact["a[O4]"].isel(dim_0=window_ref).mean(dim="dim_0")

    ds = xr.Dataset(
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
            vea=(["image_row"], ds_impact.los[:, 0].values),
            vaa=(["window"], ds_impact["viewing-azimuth-angle"].isel(viewing_direction=0).values),
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
            ref_found=str(ref_found),
        ),
    )
    ds["vea"] = np.round(ds.vea - 90.0, 1)
    # rolling background (only if calculated with rolling_background_enh)
    if "NO2_enhancement_rolling_back" in ds_impact:
        ds = ds.assign(
            no2_enhancement_rolling_back=(["image_row", "window_plume"], ds_impact["NO2_enhancement_rolling_back"].isel(dim_0=window).values),
            o4_enhancement_rolling_back=(["image_row", "window_plume"], ds_impact["O4_enhancement_rolling_back"].isel(dim_0=window).values),
            no2_rolling_background=(["image_row", "window_plume"], ds_impact["NO2_rolling_background"].isel(dim_0=window).values),
            o4_rolling_background=(["image_row", "window_plume"], ds_impact["O4_rolling_background"].isel(dim_0=window).values),
        )
    # add an extended plume window: 10 minutes before t to 15 minutes after t
    ext_start = t - pd.Timedelta(minutes=10)
    ext_end = t + pd.Timedelta(minutes=15)
    plume_window_extended = ((measurement_times >= ext_start) & (measurement_times < ext_end))
    if plume_window_extended.sum() > 0:
        no2_ext = ds_impact["a[NO2]"].isel(dim_0=plume_window_extended) - ds_impact["a[NO2]"].isel(dim_0=window_ref).mean(dim="dim_0")
        vert_no2_ext = no2_ext.sum(dim="viewing_direction")
        o4_ext = ds_impact["a[O4]"].isel(dim_0=plume_window_extended) - ds_impact["a[O4]"].isel(dim_0=window_ref).mean(dim="dim_0")

        ds = ds.assign_coords(plume_window_extended=ds_impact["dim_0"].isel(dim_0=plume_window_extended).values)
        ds = ds.assign(
            no2_extended=(["image_row", "plume_window_extended"], ds_impact["a[NO2]"].isel(dim_0=plume_window_extended).values),
            o4_extended=(["image_row", "plume_window_extended"], ds_impact["a[O4]"].isel(dim_0=plume_window_extended).values),
            times_plume_extended=(["plume_window_extended"], np.array(pd.to_datetime(measurement_times[plume_window_extended]), dtype='datetime64[ns]')),
            no2_enhancement_c_back_extended=(["image_row", "plume_window_extended"], no2_ext.values),
            o4_enhancement_c_back_extended=(["image_row", "plume_window_extended"], o4_ext.values),
            vertically_integrated_no2_enhancement_c_back_extended=(["plume_window_extended"], vert_no2_ext.values),
        )

    if df_lp_365 is not None:
        lp_window = ((df_lp_365.index >= t - pd.Timedelta(minutes=window_minutes[0])) & (df_lp_365.index < t + pd.Timedelta(minutes=window_minutes[1])))
        lp_window_ref = ((df_lp_365.index >= ref_start) & (df_lp_365.index < ref_end))
        lp_no2_enhancement = df_lp_365['Fit Coefficient (NO2)'][lp_window] - df_lp_365['Fit Coefficient (NO2)'][lp_window_ref].mean()

        lp_idx = df_lp_365.index
        if isinstance(lp_idx, pd.DatetimeIndex) and lp_idx.tz is not None:
            lp_idx = lp_idx.tz_convert("UTC").tz_localize(None)
        lp_idx_window = lp_idx[lp_window]
        lp_idx_window_ref = lp_idx[lp_window_ref]

        ds = ds.assign_coords(
            lp_window_365=np.where(lp_window)[0],
            lp_window_ref_365=np.where(lp_window_ref)[0],
        )

        ds = ds.assign(
            lp_no2=(["lp_window_365"], df_lp_365['Fit Coefficient (NO2)'][lp_window].values),
            lp_rms=(["lp_window_365"], df_lp_365['RMS'][lp_window].values),
            lp_no2_enhancement=(["lp_window_365"], lp_no2_enhancement.values),
            lp_times_window_365=(["lp_window_365"], np.array(lp_idx_window, dtype="datetime64[ns]")),
            lp_no2_ref=(["lp_window_ref_365"], df_lp_365['Fit Coefficient (NO2)'][lp_window_ref].values),
            lp_times_window_ref_365=(["lp_window_ref_365"], np.array(lp_idx_window_ref, dtype="datetime64[ns]")),
        )

    if df_lp_310 is not None:
        lp_window = ((df_lp_310.index >= t - pd.Timedelta(minutes=window_minutes[0])) & (df_lp_310.index < t + pd.Timedelta(minutes=window_minutes[1])))
        lp_window_ref = ((df_lp_310.index >= ref_start) & (df_lp_310.index < ref_end))
        lp_o3_enhancement = df_lp_310['Fit Coefficient (O3)'][lp_window] - df_lp_310['Fit Coefficient (O3)'][lp_window_ref].mean()
        lp_so2_enhancement = df_lp_310['Fit Coefficient (SO2)'][lp_window] - df_lp_310['Fit Coefficient (SO2)'][lp_window_ref].mean()
        lp_idx = df_lp_310.index
        if isinstance(lp_idx, pd.DatetimeIndex) and lp_idx.tz is not None:
            lp_idx = lp_idx.tz_convert("UTC").tz_localize(None)
        lp_idx_window = lp_idx[lp_window]
        lp_idx_window_ref = lp_idx[lp_window_ref]

        ds = ds.assign_coords(
            lp_window_310=np.where(lp_window)[0],
            lp_window_ref_310=np.where(lp_window_ref)[0],
        )

        ds = ds.assign(
            lp_o3=(["lp_window_310"], df_lp_310['Fit Coefficient (O3)'][lp_window].values),
            lp_so2=(["lp_window_310"], df_lp_310['Fit Coefficient (SO2)'][lp_window].values),
            #lp_rms=(["lp_window_310"], df_lp_310['RMS'][lp_window].values), already assigned a value in the lp_365 dataset, probably sufficient
            lp_o3_enhancement=(["lp_window_310"], lp_o3_enhancement.values),
            lp_so2_enhancement=(["lp_window_310"], lp_so2_enhancement.values),
            lp_times_window_310=(["lp_window_310"], np.array(lp_idx_window, dtype="datetime64[ns]")),
            lp_o3_ref=(["lp_window_ref_310"], df_lp_310['Fit Coefficient (O3)'][lp_window_ref].values),
            lp_so2_ref=(["lp_window_ref_310"], df_lp_310['Fit Coefficient (SO2)'][lp_window_ref].values),
            lp_times_window_ref_310=(["lp_window_ref_310"], np.array(lp_idx_window_ref, dtype="datetime64[ns]")),
        )


    return ds

def upwind_downwind_interp_background_enh(ds, row, 
                                          ds_impact, measurement_times, 
                                          ship_passes, window_minutes=(1, 3), 
                                          ref_search_minutes=60, ref_window_minutes=1, 
                                          ref_min_span_seconds=40, ref_min_count=100,
                                          ref_start_offset=3, other_ship_minutes=5,
                                          ref_ztest_p_threshold=0.15, ref_ztest_min_cluster_size=20):
    """
    Subtracts a background interpolated in time between an upwind and a downwind reference
    window for a single ship pass (row from ship_passes).

    `ds` is the plume dataset of upwind_constant_background_enh or None. If it holds the upwind
    reference (coordinate 'window_ref'), that window is reused; otherwise the upwind reference is
    searched here and stored as no2_ref, o4_ref and times_ref. With ds=None, a new plume dataset
    is created, so the function also runs on its own.

    Returns the plume dataset with the interpolated enhancement added, the dataset without it if
    no clean upwind/downwind reference was found, or None if no measurement is close to the pass.
    """
    mmsi = row["MMSI"]
    t = pd.to_datetime(row.name)#.tz_localize("UTC")
    time_diff = row["Closest_Impact_Measurement_Time_Diff"].total_seconds()
    if time_diff > 60:
        return None
    window = ((measurement_times >= t - pd.Timedelta(minutes=window_minutes[0])) & (measurement_times < t + pd.Timedelta(minutes=window_minutes[1])))
    search_kwargs = dict(ref_search_minutes=ref_search_minutes, ref_window_minutes=ref_window_minutes,
                         ref_min_span_seconds=ref_min_span_seconds, ref_min_count=ref_min_count,
                         other_ship_minutes=other_ship_minutes, ztest_p_threshold=ref_ztest_p_threshold,
                         ztest_min_cluster_size=ref_ztest_min_cluster_size)

    if ds is None:
        ds = xr.Dataset(
            data_vars=dict(
                times_plume=(["window_plume"], np.array(pd.to_datetime(measurement_times[window]), dtype='datetime64[ns]')),
                vea=(["image_row"], np.round(ds_impact.los[:, 0].values - 90.0, 1)),
            ),
            coords=dict(
                window_plume=ds_impact["dim_0"].isel(dim_0=window).values,
                image_row=ds_impact["viewing_direction"].values,
            ),
            attrs=dict(mmsi=str(mmsi), t=str(t), plume_number=str(row["Plume_number"])),
        )

    # Upwind reference: reuse the one of upwind_constant_background_enh, else search it
    if "window_ref" in ds.coords:
        window_ref = np.isin(ds_impact["dim_0"].values, ds["window_ref"].values)
    else:
        upwind = find_reference_window(t, mmsi, ds_impact, measurement_times, ship_passes, direction="up",
                                       start_offset=ref_start_offset, **search_kwargs)
        if upwind is None:
            print(f"No clean upwind reference window found for MMSI {mmsi} at {t}")
            return ds
        window_ref = upwind[0]
        ds = ds.assign_coords(window_ref=ds_impact["dim_0"].isel(dim_0=window_ref).values)
        ds = ds.assign(
            no2_ref=(["image_row", "window_ref"], ds_impact["a[NO2]"].isel(dim_0=window_ref).values),
            o4_ref=(["image_row", "window_ref"], ds_impact["a[O4]"].isel(dim_0=window_ref).values),
            times_ref=(["window_ref"], np.array(pd.to_datetime(measurement_times[window_ref]), dtype='datetime64[ns]')),
        )
        ds.attrs["ref_found"] = "True"

    # Downwind reference: starts ref_start_offset minutes after the end of the plume window
    downwind = find_reference_window(t, mmsi, ds_impact, measurement_times, ship_passes, direction="down",
                                     start_offset=window_minutes[1] + ref_start_offset, **search_kwargs)
    if downwind is None:
        print(f"No clean downwind reference window found for MMSI {mmsi} at {t}")
        return ds
    downwind_window_ref = downwind[0]


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
    # add an extended plume window: 10 minutes before t to 15 minutes after t
    ext_start = t - pd.Timedelta(minutes=10)
    ext_end = t + pd.Timedelta(minutes=15)
    plume_window_extended = ((measurement_times >= ext_start) & (measurement_times < ext_end))
    if plume_window_extended.sum() > 0:
        times_ext = pd.to_datetime(measurement_times[plume_window_extended])
        denom = (t_down_center - t_ref_center).total_seconds()
        alpha_ext = ((times_ext - t_ref_center).total_seconds() / denom).astype(float)
        alpha_ext = np.clip(alpha_ext, 0.0, 1.0)
        dim0_coords_ext = ds_impact["dim_0"].isel(dim_0=plume_window_extended).values
        alpha_da_ext = xr.DataArray(alpha_ext, dims=("dim_0",), coords={"dim_0": dim0_coords_ext})

        up_exp_ext = up_mean.expand_dims(dim_0=alpha_da_ext.coords["dim_0"])
        down_exp_ext = down_mean.expand_dims(dim_0=alpha_da_ext.coords["dim_0"])
        interp_bg_ext = (1 - alpha_da_ext) * up_exp_ext + alpha_da_ext * down_exp_ext
        no2_enh_ext = ds_impact["a[NO2]"].isel(dim_0=plume_window_extended) - interp_bg_ext
        vert_no2_ext = no2_enh_ext.sum(dim="viewing_direction")

        up_o4_exp_ext = up_mean_o4.expand_dims(dim_0=alpha_da_ext.coords["dim_0"])
        down_o4_exp_ext = down_mean_o4.expand_dims(dim_0=alpha_da_ext.coords["dim_0"])
        interp_bg_o4_ext = (1 - alpha_da_ext) * up_o4_exp_ext + alpha_da_ext * down_o4_exp_ext
        o4_enh_ext = ds_impact["a[O4]"].isel(dim_0=plume_window_extended) - interp_bg_o4_ext

        ds = ds.assign_coords(plume_window_extended=ds_impact["dim_0"].isel(dim_0=plume_window_extended).values)
        ds = ds.assign(
            no2_enhancement_interp_extended=(["image_row", "plume_window_extended"], no2_enh_ext.values),
            vertically_integrated_no2_enhancement_interp_extended=(["plume_window_extended"], vert_no2_ext.values),
            o4_enhancement_interp_extended=(["image_row", "plume_window_extended"], o4_enh_ext.values),
            times_plume_extended=(["plume_window_extended"], np.array(pd.to_datetime(measurement_times[plume_window_extended]), dtype='datetime64[ns]')),
        )
    return ds

def polynomial_background_enh(ds_impact_masked, degree=8): 
    time_diff = pd.to_datetime(ds_impact_masked["datetime"]) - pd.to_datetime(ds_impact_masked["datetime"])[0]
    x = time_diff.total_seconds().astype(float)
    y = ds_impact_masked["a[NO2]"]

    # Fit polynomial of given degree (Polynomial.fit maps x to [-1, 1], which keeps the fit well conditioned)
    poly_fit = np.polynomial.Polynomial.fit(x, y, degree)(x)

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

    # Polynomial.fit maps x to [-1, 1], which keeps the fit well conditioned
    poly_fit_lp = np.polynomial.Polynomial.fit(x_lp, y_lp, degree)(x_lp)

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

    # Background = low-pass: keep only frequencies with |f| < f_cut (i.e., periods > t_cut)
    Y_filtered = Y.copy()
    Y_filtered[np.abs(freqs) >= f_cut] = 0

    # Inverse FFT to get the background, enhancement = signal - background (high-pass)
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

    f_cut_lp = 1/t_cut  # Hz

    # Background = low-pass: keep only frequencies with |f| < f_cut_lp (i.e., periods > t_cut)
    Y_lp_filtered = Y_lp.copy()
    Y_lp_filtered[np.abs(freqs_lp) >= f_cut_lp] = 0

    # Inverse FFT to get the background
    df_lp_doas["NO2_fft_filter"] = np.real(ifft(Y_lp_filtered))

    # enhancement = signal - background (high-pass)
    df_lp_doas["NO2_enhancements_fft_filter"] = y_lp - df_lp_doas["NO2_fft_filter"]

    return df_lp_doas