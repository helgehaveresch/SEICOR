import pandas as pd
import os
import shutil
import re


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
    """
    img_dir = os.path.join(img_dir, "{}_video".format(date))

    img_files, img_times = zip(*[
        (f, parse_img_time(f))
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") and parse_img_time(f) is not None
    ])
    closest_files = []
    time_diffs = []
    for t in df_closest.index:
        closest_file, time_diff = get_closest_image(img_files, img_times, t)
        closest_files.append(os.path.join(img_dir, closest_file))
        time_diffs.append(time_diff)
    df_closest = df_closest.copy()
    df_closest["closest_image_file"] = closest_files
    df_closest["image_time_diff"] = time_diffs
    return df_closest

def copy_ship_images(df_closest, img_dir, img_out_dir, time_threshold=30):
    """
    Copies images for each ship in df_closest to img_out_dir, renaming them to include MMSI.
    Only copies if image_time_diff <= time_threshold (seconds).
    """

    os.makedirs(img_out_dir, exist_ok=True)

    for ship_time, ship in df_closest.iterrows():
        mmsi = ship["MMSI"]
        closest_file = ship["closest_image_file"]
        time_diff = ship["image_time_diff"]
        if time_diff <= time_threshold:
            src = closest_file if os.path.isabs(closest_file) else os.path.join(img_dir, closest_file)
            base, ext = os.path.splitext(os.path.basename(closest_file))
            new_name = f"{base}_{mmsi}.jpg"
            dst = os.path.join(img_out_dir, new_name)
            shutil.copy2(src, dst)
        else:
            print(f"No suitable image found for MMSI {mmsi} at time {ship_time}, closest image was {closest_file} with time difference {time_diff:.2f} seconds")
    print("Copied Images of ships")