"""
Download ERA5 model-level wind (approx. 10-700 m above ground) around Wedel
and the Wettermast Hamburg-Billwerder.
 
Saves per month to D:\\SEICOR\\ERA5:
  era5_ml_YYYYMM.grib      u, v, T, q on model levels 121-137
  era5_lnsp_z_YYYYMM.grib  log surface pressure + surface geopotential
  era5_sl_YYYYMM.grib      single-level fields (PBL height, fluxes, 10/100 m wind, ...)
T, q, lnsp and z are needed to compute the actual height of each level.

Up to MAX_PARALLEL requests run at once.  Files already present are skipped and failed
requests are listed at the end, so the script can simply be run again until all are there.

Resume: before submitting anything, the script lists your queued, running and finished CDS
requests.  If one has exactly the same content as a missing file, its result is downloaded
instead of submitting a duplicate, so an interrupted run loses no place in the queue.

Requirements: pip install cdsapi   (brings the ecmwf-datastores-client used here)
CDS account, API key in %USERPROFILE%\\.cdsapirc, ERA5 licence accepted.
"""

import calendar
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import ecmwf.datastores as eds

# ---------------------------------------------------------------- settings
OUTDIR = Path(r"D:\SEICOR\ERA5")
FIRST_MONTH = (2025, 3)                    # first and last month to download (UTC), inclusive
LAST_MONTH = (2026, 6)
YEARS_MONTHS = [(y, m) for y in range(FIRST_MONTH[0], LAST_MONTH[0] + 1) for m in range(1, 13)
                if FIRST_MONTH <= (y, m) <= LAST_MONTH]
# N/W/S/E: covers Wedel (~53.58N, 9.70E) and Wettermast Hamburg-Billwerder
# (~53.52N, 10.10E) with one grid point margin around each site
AREA = "54.0/9.25/53.25/10.5"

GRID = "0.25/0.25"
LEVELS = "121/to/137"                      # ~715 m down to ~10 m
MAX_PARALLEL = 8                           # requests sent to the CDS at once; the CDS queues
                                           # anything above its own per-user limit anyway
 
SINGLE_LEVEL_VARS = [
    # boundary layer and stability
    "boundary_layer_height",
    "friction_velocity",
    "instantaneous_surface_sensible_heat_flux",
    "instantaneous_moisture_flux",
    "surface_sensible_heat_flux",        # accumulated over previous hour, J/m2
    "surface_latent_heat_flux",          # accumulated over previous hour, J/m2
    "forecast_surface_roughness",
    "forecast_logarithm_of_surface_roughness_for_heat",
    "instantaneous_eastward_turbulent_surface_stress",
    "instantaneous_northward_turbulent_surface_stress",
    # inversions / strong gradient layers
    "trapping_layer_base_height",
    "trapping_layer_top_height",
    "duct_base_height",
    # water surface temperature (air-water stability)
    "lake_mix_layer_temperature",
    "lake_cover",
    "sea_surface_temperature",
    "convective_available_potential_energy",
    "convective_inhibition",
    "k_index",
    "total_totals_index",
    # near-surface state
    "2m_temperature",
    "2m_dewpoint_temperature",
    "skin_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    # clouds and radiation
    "total_cloud_cover",
    "low_cloud_cover",
    "surface_solar_radiation_downwards",
    # invariants
    "land_sea_mask",
    "geopotential",
]
 
 
def month_jobs(year, month):
    """The three requests of one month as (dataset, request, target file)."""
    last = calendar.monthrange(year, month)[1]
    common = {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "an",
        "levtype": "ml",
        "date": f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last:02d}",
        "time": "00/to/23/by/1",
        "area": AREA,
        "grid": GRID,
    }
    n, w, s, e = (float(x) for x in AREA.split("/"))
    return {
        # single levels: PBL height, stability, near-surface state (fast CDS disks)
        "sl": ("reanalysis-era5-single-levels",
               {"product_type": ["reanalysis"],
                "variable": SINGLE_LEVEL_VARS,
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": [f"{d:02d}" for d in range(1, last + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [n, w, s, e],
                # GRIB: the CDS charges 6x per field for NetCDF, which puts a full month of these
                # variables over the per-request cost limit (147k vs 121k); GRIB costs ~25k
                "data_format": "grib",
                "download_format": "unarchived"},
               OUTDIR / f"era5_sl_{year}{month:02d}.grib"),
        # model levels from the MARS tape archive (slow)
        "ml": ("reanalysis-era5-complete",
               {**common, "levelist": LEVELS, "param": "130/131/132/133"},   # T, u, v, q
               OUTDIR / f"era5_ml_{year}{month:02d}.grib"),
        "sfc": ("reanalysis-era5-complete",
                {**common, "levelist": "1", "param": "129/152"},            # surface geopotential, lnsp
                OUTDIR / f"era5_lnsp_z_{year}{month:02d}.grib"),
    }


_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def make_client():
    """CDS client configured from %USERPROFILE%\\.cdsapirc (url + key)."""
    cfg = {}
    for line in (Path.home() / ".cdsapirc").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return eds.Client(url=cfg["url"], key=cfg["key"])


def _norm(v):
    """Comparable form of a request value: the CDS stores e.g. 54.0 as 54 and may reorder lists."""
    if isinstance(v, (list, tuple)):
        return tuple(sorted(_norm(x) for x in v))
    try:
        return format(float(v), "g")
    except (TypeError, ValueError):
        return str(v)


def request_key(dataset, request):
    return dataset, tuple(sorted((k, _norm(v)) for k, v in request.items()))


def existing_cds_jobs(client):
    """{request_key: (status, request id)} of your queued, running and finished CDS requests.
    Where several share a request, a finished one is preferred, then running, then the newest."""
    rank = {"successful": 0, "running": 1, "accepted": 2}
    found = {}
    page = client.get_jobs(limit=100, sortby="-created", status=list(rank))
    while page is not None:
        for j in page.json["jobs"]:
            key = request_key(j["processID"], client.get_remote(j["jobID"]).request)
            if key not in found or rank[j["status"]] < rank[found[key][0]]:
                found[key] = (j["status"], j["jobID"])
        page = page.next
    return found


def run_job(dataset, request, target, request_id=None):
    """Retrieve one request, or wait for and download an existing CDS request (request_id).

    The file is written as *.part and renamed when complete, so an interrupted download is
    not mistaken for a finished one on the next run.
    """
    part = target.with_name(target.name + ".part")
    t0 = time.monotonic()
    client = make_client()                             # one client per thread
    if request_id is not None:
        remote = client.get_remote(request_id)
        log(f"resuming   {target.name}  (CDS request {remote.status})")
        try:
            remote.download(str(part))
        except Exception as exc:                       # e.g. results of an old request expired
            log(f"resume of  {target.name} failed ({exc}), submitting it again")
            request_id = None
    if request_id is None:
        log(f"requesting {target.name}")
        client.submit(dataset, request).download(str(part))
    part.replace(target)
    log(f"done       {target.name}  ({(time.monotonic() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)   # keep the log readable
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [month_jobs(y, m) for y, m in YEARS_MONTHS]
    # the fast single-level requests first, so they do not wait behind the slow tape requests
    queue = [job[kind] for kind in ("sl", "ml", "sfc") for job in jobs]
    todo = [j for j in queue if not j[2].exists()]

    log("looking for matching requests at the CDS ...")
    cds_jobs = existing_cds_jobs(make_client())
    resume_ids = {j[2]: cds_jobs.get(request_key(j[0], j[1]), (None, None))[1] for j in todo}
    n_resume = sum(rid is not None for rid in resume_ids.values())
    log(f"{len(queue) - len(todo)} of {len(queue)} files already present, {len(todo)} missing: "
        f"{n_resume} resumed from existing CDS requests, {len(todo) - n_resume} new "
        f"({MAX_PARALLEL} in parallel)")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_job, *j, resume_ids[j[2]]): j[2].name for j in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:                   # keep the other requests running
                failed.append(futures[fut])
                log(f"FAILED     {futures[fut]}: {exc}")

    if failed:
        log(f"{len(failed)} request(s) failed, run the script again to retry: {', '.join(sorted(failed))}")
    else:
        log("Done.")