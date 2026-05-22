#!/usr/bin/env python3
"""Copy plume .nc files from a source root to a destination root.

For each source file found under SOURCE_ROOT, compute the destination path
under DEST_ROOT preserving relative structure. If the destination file
already exists, copy any attributes present in the destination but missing
from the source into the source dataset, then write the (updated)
source dataset to the destination (overwriting). If the destination file
does not exist, the source file is copied as-is.

Usage:
  python SEICOR/copy_plumes_with_attrs.py --src "Q:/BREDOM/SEICOR/plumes_4" --dst "P:/data/SEICOR/plumes_2"

"""
from pathlib import Path
import argparse
import os
import shutil
import re
# disable HDF5 file locking on network drives (Windows)
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import xarray as xr
import sys


def copy_and_merge_attrs(src_root: Path, dst_root: Path, dst_modif_root: Path, pattern='*.nc', dry_run=False, min_plumes: str = None):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_modif_root = Path(dst_modif_root)
    if not src_root.exists():
        raise FileNotFoundError(f'source root not found: {src_root}')
    dst_root.mkdir(parents=True, exist_ok=True)

    # parse min_plumes if provided (accept 'plumes_YYMMDD' or 'YYMMDD')
    min_date_int = None
    if min_plumes:
        mm = re.match(r'^(?:plumes_)?(\d{6})$', str(min_plumes))
        if not mm:
            raise ValueError(f"min_plumes must be 'plumes_YYMMDD' or 'YYMMDD', got: {min_plumes}")
        min_date_int = int(mm.group(1))

    summary = {'copied_new': 0, 'overwritten_with_merged_attrs': 0, 'errors': 0}

    for src in sorted(src_root.rglob(pattern)):
        try:
            rel = src.relative_to(src_root)
        except Exception:
            # fallback: use name only
            rel = src.name

        # if min_date_int set: only process files under plumes_YYMMDD with date > min
        if min_date_int is not None:
            has_plumes = False
            greater = False
            for part in Path(rel).parts:
                m = re.match(r'^plumes_(\d{6})$', part)
                if m:
                    has_plumes = True
                    folder_date = int(m.group(1))
                    if folder_date > min_date_int:
                        greater = True
                    break
            if (not has_plumes) or (not greater):
                continue

        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst_modif = dst_modif_root / rel
        dst_modif.parent.mkdir(parents=True, exist_ok=True)
        
        if not dst.exists():
            if dry_run:
                print(f'[DRY] copy new: {src} -> {dst}')
                summary['copied_new'] += 1
            else:
                try:
                    shutil.copy2(src, dst)
                    print(f'Copied new: {src} -> {dst}')
                    summary['copied_new'] += 1
                except Exception as e:
                    print(f'Failed to copy {src} -> {dst}: {e}')
                    summary['errors'] += 1
            continue

        # destination exists: open both datasets and merge missing attrs from dst into src
        try:
            ds_src_orig = xr.open_dataset(src).load()
            ds_src = ds_src_orig.copy(deep=True)
            ds_src_orig.close()
            ds_dst_orig = xr.open_dataset(dst).load()
            ds_dst = ds_dst_orig.copy(deep=True)
            ds_src_orig.close()
        except Exception as e:
            print(f'Failed to open datasets for {src} or {dst}: {e}')
            summary['errors'] += 1
            try:
                ds_src_orig.close()
            except Exception:
                pass
            try:
                ds_dst_orig.close()
            except Exception:
                pass
            continue

        try:
            # copy missing attributes from destination into source (but do not overwrite existing keys)
            copied_keys = []
            for k, v in ds_dst.attrs.items():
                if k not in ds_src.attrs:
                    ds_src.attrs[k] = v
                    copied_keys.append(k)

            if not copied_keys:
                # nothing to merge; still copy source to dst (overwrite) to ensure identical file
                if dry_run:
                    print(f'[DRY] overwrite (no attr merge): {src} -> {dst_modif}')
                    summary['overwritten_with_merged_attrs'] += 1
                else:
                    try:
                        ds_src.load()
                        ds_src.to_netcdf(dst_modif, mode='w')
                        print(f'Overwrote (no attr merge needed): {dst_modif}')
                        summary['overwritten_with_merged_attrs'] += 1
                    except Exception as e:
                        print(f'Failed to overwrite {dst_modif} with {src}: {e}')
                        summary['errors'] += 1
            else:
                if dry_run:
                    print(f'[DRY] merged attrs {copied_keys} from {dst} into {src}, then overwrite')
                    summary['overwritten_with_merged_attrs'] += 1
                else:
                    try:
                        # load into memory and write out
                        ds_src.load()
                        ds_src.to_netcdf(dst_modif, mode='w')
                        print(f'Merged attrs {copied_keys} from {dst} into {dst_modif} (from {src})')
                        summary['overwritten_with_merged_attrs'] += 1
                    except Exception as e:
                        print(f'Failed to write merged dataset to {dst_modif}: {e}')
                        summary['errors'] += 1
        finally:
            try:
                ds_src.close()
            except Exception:
                pass
            try:
                ds_dst.close()
            except Exception:
                pass

    print('\nDone. Summary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')


def main():
    p = argparse.ArgumentParser(description='Copy plume .nc files and merge missing attributes from destination into source before copying')
    p.add_argument('--src', required=True, help='Source root (e.g. Q:/BREDOM/SEICOR/plumes_4)')
    p.add_argument('--dst', required=True, help='Destination root (e.g. P:/data/SEICOR/plumes_2)')
    p.add_argument('--dst-modif', required=True, help='Destination root for modified files (e.g. P:/data/SEICOR/plumes_2_modified)')
    p.add_argument('--pattern', default='*.nc', help='Filename glob pattern to match (default: *.nc)')
    p.add_argument('--dry-run', action='store_true', help='Show planned actions without performing copies')
    p.add_argument('--min-plumes', dest='min_plumes', required=False, help="Only process files under subfolders 'plumes_YYMMDD' with date > YYMMDD (accept 'plumes_250728' or '250728')")
    args = p.parse_args()

    try:
        copy_and_merge_attrs(Path(args.src), Path(args.dst), Path(args.dst_modif), pattern=args.pattern, dry_run=args.dry_run, min_plumes=args.min_plumes)
    except Exception as e:
        print('Error:', e)
        sys.exit(2)


if __name__ == '__main__':
    main()
