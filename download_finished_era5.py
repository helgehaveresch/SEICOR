"""
Complement to download_era_5.py: download the results of finished CDS requests whose
file is still missing in OUTDIR (e.g. because download_era_5.py was interrupted after
the CDS had finished the request).

Only successful requests are considered, nothing new is submitted.  Requests are matched
to target files by their exact content, using the settings of download_era_5.py.
Run download_era_5.py afterwards for anything still missing (queued, running or failed).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from download_era_5 import (MAX_PARALLEL, OUTDIR, YEARS_MONTHS, log, make_client,
                            month_jobs, request_key)


def finished_cds_jobs(client):
    """{request_key: request id} of your successful CDS requests (newest wins)."""
    found = {}
    page = client.get_jobs(limit=100, sortby="-created", status=["successful"])
    while page is not None:
        for j in page.json["jobs"]:
            key = request_key(j["processID"], client.get_remote(j["jobID"]).request)
            found.setdefault(key, j["jobID"])
        page = page.next
    return found


def download(request_id, target):
    """Download a finished request as *.part and rename it when complete."""
    part = target.with_name(target.name + ".part")
    log(f"downloading {target.name}")
    make_client().get_remote(request_id).download(str(part))   # one client per thread
    part.replace(target)
    log(f"done        {target.name}")


if __name__ == "__main__":
    logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    expected = [job for y, m in YEARS_MONTHS for job in month_jobs(y, m).values()]
    missing = [j for j in expected if not j[2].exists()]
    log(f"{len(expected) - len(missing)} of {len(expected)} files present, {len(missing)} missing")

    log("listing finished requests at the CDS ...")
    finished = finished_cds_jobs(make_client())
    todo = [(finished[request_key(ds, req)], target) for ds, req, target in missing
            if request_key(ds, req) in finished]
    log(f"{len(todo)} missing file(s) have a finished CDS request, "
        f"{len(missing) - len(todo)} have none")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(download, rid, target): target.name for rid, target in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:                   # e.g. results expired at the CDS
                failed.append(futures[fut])
                log(f"FAILED      {futures[fut]}: {exc}")

    if failed:
        log(f"{len(failed)} download(s) failed: {', '.join(sorted(failed))}. "
            f"Run download_era_5.py to request them again.")
    else:
        log("Done.")
