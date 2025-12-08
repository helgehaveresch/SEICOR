import pandas as pd
import os
import shutil
import re
import zipfile
import numpy as np

def parse_img_time(fname):
    
    """
    Parse the timestamp from the image filename.

    Handles formats:
    1. yymmdd_hhmmssms.JPG
    2. yymmdd_hhmmssmms-??_???-??.JPG
    """

    base = os.path.splitext(fname)[0]
    # Try old format first
    try:
        dt = pd.to_datetime(base, format="%y%m%d_%H%M%S%f").tz_localize("CET")
        return dt
    except Exception:
        pass
    # Try extended format: yymmdd_hhmmssmms-??_???-??
    match = re.match(r"(\d{6}_\d{8,9})", base)
    if match:
        try:
            dt = pd.to_datetime(match.group(1), format="%y%m%d_%H%M%S%f").tz_localize("CET")
            return dt.tz_convert("UTC")
        except Exception:
            pass
    return None

def get_closest_image(img_files, img_times, target_time):
    """
    Returns the filename and time difference (in seconds) of the image closest to target_time.

    Parameters
    ----------
    img_files : list of str
        List of image filenames.
    img_times : list of datetime
        List of datetimes corresponding to img_files.
    target_time : datetime
        The time to which the closest image is sought.

    Returns
    -------
    closest_file : str
        Filename of the closest image.
    time_diff : float
        Absolute time difference in seconds between image and target_time.
    """
    closest_file, closest_time = min(zip(img_files, img_times), key=lambda x: abs(x[1] - target_time))
    time_diff = abs((closest_time - target_time).total_seconds())
    return closest_file, time_diff

def assign_video_images_to_ship_pass(df_closest, img_dir, date):
    """
    For each row in df_closest, find and assign the closest image file and time difference.
    Adds columns 'closest_image_file' and 'image_time_diff'.

    img_dir may be either:
    - a directory path (expected to contain a subfolder "{date}_video"), or
    - a zip archive file path containing images (files may be in subfolders).
    """
    target_folder = "{}_video".format(date)
    target_zip_folder = "{}_video.zip".format(date)
    dirpath = os.path.join(img_dir, target_folder)
    zip_dirpath = os.path.join(img_dir, target_zip_folder)
    is_zip = os.path.isfile(zip_dirpath) and zipfile.is_zipfile(zip_dirpath)

    if is_zip:
        # list image members inside the zip archive
        with zipfile.ZipFile(zip_dirpath, "r") as zf:
            members = [
                m for m in zf.namelist()
                if not m.endswith("/") and m.lower().endswith(".jpg")
            ]
            imgs = []
            for m in members:
                parsed = parse_img_time(os.path.basename(m))
                if parsed is not None:
                    imgs.append((m, parsed))
    else:
        imgs = []
        # If directory does not exist, do not abort — mark rows as having no image
        if not os.path.isdir(dirpath):
            df_out = df_closest.copy()
            df_out["closest_image_file"] = "no image"
            df_out["image_time_diff"] = np.nan
            return df_out

        for f in os.listdir(dirpath):
            if f.lower().endswith(".jpg"):
                parsed = parse_img_time(f)
                if parsed is not None:
                    imgs.append((f, parsed))

    if not imgs:
        # no valid images found; mark all rows as having no image and return
        df_out = df_closest.copy()
        df_out["closest_image_file"] = "no image"
        df_out["image_time_diff"] = np.nan
        return df_out

    img_files, img_times = zip(*imgs)

    closest_files = []
    time_diffs = []
    for t in df_closest.index:
        try:
            closest_file, time_diff = get_closest_image(img_files, img_times, t)
            # store the archive member path (for zip) or the full filepath (for dir)
            if is_zip:
                # keep zip member path as stored value
                closest_files.append(closest_file)
            else:
                closest_files.append(os.path.join(dirpath, closest_file))
            time_diffs.append(time_diff)
        except Exception:
            # could not find/parse a closest image for this timestamp
            closest_files.append("no image")
            time_diffs.append(np.nan)

    df_closest = df_closest.copy()
    df_closest["closest_image_file"] = closest_files
    df_closest["image_time_diff"] = time_diffs
    return df_closest

def copy_ship_images(df_closest, img_dir, date, img_out_dir, time_threshold=30):
    """
    Copies images for each ship in df_closest to img_out_dir, renaming them to include MMSI.
    Supports img_dir being a directory (with subfolder "{date}_video") or a zip archive.
    Only copies if image_time_diff <= time_threshold (seconds).
    """
    target_folder = "{}_video".format(date)
    target_zip_folder = "{}_video.zip".format(date)
    dirpath = os.path.join(img_dir, target_folder)
    zip_dirpath = os.path.join(img_dir, target_zip_folder)
    is_zip = os.path.isfile(zip_dirpath) and zipfile.is_zipfile(zip_dirpath)

    if is_zip:
        with zipfile.ZipFile(zip_dirpath, "r") as zf:
            for ship_time, ship in df_closest.iterrows():
                mmsi = ship["MMSI"]
                closest_file = ship["closest_image_file"]  # this is the member path in zip or 'no image'
                time_diff = ship["image_time_diff"]

                # skip if marked 'no image' or time diff not numeric or too large
                if closest_file == "no image" or not (isinstance(time_diff, (int, float)) and time_diff <= time_threshold):
                    print(f"No suitable image found for MMSI {mmsi} at time {ship_time}, closest image was {closest_file} with time difference {time_diff}")
                    df_closest.at[ship_time, "closest_image_file"] = "no image"
                    continue

                base = os.path.basename(closest_file)
                name_base, ext = os.path.splitext(base)
                new_name = f"{name_base}_{mmsi}{ext}"
                dst = os.path.join(img_out_dir, new_name)
                try:
                    with zf.open(closest_file) as srcf, open(dst, "wb") as dstf:
                        shutil.copyfileobj(srcf, dstf)
                except KeyError:
                    print(f"Image {closest_file} not found inside zip archive {zip_dirpath}")
                    df_closest.at[ship_time, "closest_image_file"] = "no image"
                except Exception as e:
                    print(f"Error extracting {closest_file} from {zip_dirpath}: {e}")
                    df_closest.at[ship_time, "closest_image_file"] = "no image"
    else:
        for ship_time, ship in df_closest.iterrows():
            mmsi = ship["MMSI"]
            closest_file = ship["closest_image_file"]
            time_diff = ship["image_time_diff"]
            # skip if no image marked or time diff invalid or too large
            if closest_file == "no image" or not (isinstance(time_diff, (int, float)) and time_diff <= time_threshold):
                print(f"No suitable image found for MMSI {mmsi} at time {ship_time}, closest image was {closest_file} with time difference {time_diff}")
                df_closest.at[ship_time, "closest_image_file"] = "no image"
                continue

            # determine source path: absolute path or path inside dirpath
            if os.path.isabs(closest_file) and os.path.exists(closest_file):
                src = closest_file
            else:
                src = os.path.join(dirpath, os.path.basename(closest_file))

            base, ext = os.path.splitext(os.path.basename(closest_file))
            new_name = f"{base}_{mmsi}.jpg"
            dst = os.path.join(img_out_dir, new_name)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Source image not found: {src} (for MMSI {mmsi})")
                df_closest.at[ship_time, "closest_image_file"] = "no image"
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")
                df_closest.at[ship_time, "closest_image_file"] = "no image"
    print("Copied Images of ships")
    return df_closest