#%%
import sys
sys.path.append(r"C:\Users\hhave\Documents\Promotion\scripts")
from doas_tools.file_handling import read_SC_file_imaging
from imaging_tools.process_SC import process_SC_img_data
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import pandas as pd

#%%
date_str = "251010"
variable = "O4"
slant_folder= r"P:\data\data_tmp\SLANT_25\OKT2025"
azimuths = np.arange(160,251.2,1.2)
ds_slant = read_SC_file_imaging(slant_folder, date_str, mode="UD.NO2_VIS")
ds_slant = process_SC_img_data(ds_slant, structure="image_predefined", vaa_reference=azimuths)
# %%
variable_str = f"a[{variable}]"
title = f"Scanning geometry {variable}"
image_num = 11

ds_image = ds_slant.isel(image=image_num)
s_freq = int(len(ds_slant["viewing-azimuth-angle"])/25)
if s_freq == 0:
    s_freq = 1

date = ds_image["datetime"].values[np.where(~pd.isna(ds_image.datetime.values))[0][0]].astype('datetime64[D]')
time = ds_image["time_str"].values[np.where(~pd.isna(ds_image.time_str.values))[0][0]]
los = ds_image["los"].values[np.where(~pd.isna(ds_image["los"].values))[0][0]] - 90

plt.figure(figsize=(10,5))
plt.imshow(ds_image[variable_str].T, aspect="auto", origin="lower")
plt.ylabel("Viewing Elevation Angle / °")
plt.xlabel("Viewing Azimuth Angle / °")
plt.xticks(np.arange(0, len(ds_image["viewing-azimuth-angle"]), s_freq), ds_image["viewing-azimuth-angle"].values[::s_freq], rotation=45)
plt.yticks(np.arange(0, len(los),2), [f"{v:.1f}" for v in los[::2]], rotation=45)
plt.title( "{} on {} at {} UTC".format(title,date, time))
plt.tight_layout()
plt.colorbar()# %%

# %%
save_dir = r"P:\data\data_tmp\SEICOR_scanning"
os.makedirs(save_dir, exist_ok=True)

title = f"Scanning geometry {variable}"
n_images = ds_slant.sizes.get("image", 0)
s_freq = max(1, int(len(ds_slant["viewing-azimuth-angle"]) / 25))

for image_num in range(n_images):
    plt.figure(figsize=(10,5))
    ds_image = ds_slant.isel(image=image_num)
    date = ds_image["datetime"].values[np.where(~pd.isna(ds_image.datetime.values))[0][0]].astype('datetime64[D]')
    time = ds_image["time_str"].values[np.where(~pd.isna(ds_image.time_str.values))[0][0]]
    los = ds_image["los"].values[np.where(~pd.isna(ds_image["los"].values))[0][0]] - 90 

    plt.figure(figsize=(10,5))
    plt.imshow(ds_image[variable_str].T, aspect="auto", origin="lower")
    plt.ylabel("Viewing Elevation Angle / °")
    plt.xlabel("Viewing Azimuth Angle / °")
    plt.xticks(np.arange(0, len(ds_image["viewing-azimuth-angle"]), s_freq), ds_image["viewing-azimuth-angle"].values[::s_freq], rotation=45)
    plt.yticks(np.arange(0, len(los),2), [f"{v:.1f}" for v in los[::2]], rotation=45)
    plt.title( "{} on {} at {} UTC".format(title,date, time))
    plt.tight_layout()
    plt.colorbar()# %%
    time_str_safe = time.replace(":", "-").replace(" ", "_")
    variable_str_safe = variable_str.replace("[", "").replace("]", "").replace("a", "")

    fname = f"{date}_{time_str_safe}_image_{image_num}_{variable_str_safe}.png"
    save_path = os.path.join(save_dir,variable_str_safe , fname)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure directory exists
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {save_path}")
# %%
import imageio
from pathlib import Path

date_filter = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"  # example; change to desired date or None
# build list of PNGs ending with O4.png and create a GIF
save_dir = r"P:\data\data_tmp\SEICOR_scanning\{}".format(variable)
out_gif = Path(save_dir) / "{}_{}_images.gif".format(variable, date_filter)

# Set date_filter to a string like "2025-10-21" (matches filenames that contain the date).
# Set to None to use all dates.

files = sorted(Path(save_dir).glob("*{}.png".format(variable)))

# filter by date string in filename if requested
if date_filter:
    # match either the literal date substring or date with no dashes (just in case)
    date_no_dash = date_filter.replace("-", "")
    files = [f for f in files if (date_filter in f.name) or (date_no_dash in f.name)]

if len(files) == 0:
    print(f"No files matching '*O4.png' (and date={date_filter}) found in {save_dir}")
else:
    images = []
    for f in files:
        try:
            images.append(imageio.imread(str(f)))
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    if images:
        # fps controls speed; change duration or fps as desired
        imageio.mimsave(str(out_gif), images, fps=2)
        print(f"Saved GIF: {out_gif}")

# %%

# %%
import os
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def make_video_from_folder_with_filenames(
    src_folder=r"P:\data\data_tmp\ship_picture",
    out_video_name="ship_picture_video.mp4",
    fps=5,
    font_size_ratio=0.04,
    text_padding=10,
):
    src = Path(src_folder)
    out_path = src / out_video_name
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"No image files found in {src}")
        return

    # try to load a TTF font, fall back to default
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 12)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    # use first image size as video frame size
    with Image.open(files[0]) as im0:
        frame_w, frame_h = im0.size

    # adjust font size relative to image height if possible
    if isinstance(font, ImageFont.FreeTypeFont):
        font_size = max(12, int(frame_h * font_size_ratio))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

    for p in files:
        try:
            im = Image.open(p).convert("RGBA")
        except Exception as e:
            print(f"Warning: cannot open {p}: {e}")
            continue

        # resize to match first image if needed
        if im.size != (frame_w, frame_h):
            im = im.resize((frame_w, frame_h), Image.LANCZOS)

        draw = ImageDraw.Draw(im)

        # build text (filename)
        text = p.name

        # measure text size and draw semi-transparent rectangle behind it
        try:
            # Pillow >= 8: textbbox available
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            try:
                # older Pillow: textsize
                text_w, text_h = draw.textsize(text, font=font)
            except Exception:
                # fallback to ImageFont methods
                try:
                    text_w, text_h = font.getsize(text)
                except Exception:
                    # last resort: approximate
                    text_w = len(text) * (font.size if hasattr(font, "size") else 12)
                    text_h = int((font.size if hasattr(font, "size") else 12) * 1.2)

        # compute rectangle and text positions
        rect_x0 = text_padding
        rect_x1 = rect_x0 + text_w + text_padding * 2
        rect_y1 = frame_h - text_padding
        rect_y0 = rect_y1 - (text_h + text_padding * 2)
        if rect_y0 < 0:
            rect_y0 = 0
        # draw rectangle (semi-transparent)
        rect_color = (0, 0, 0, 180)  # dark translucent
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=rect_color)

        # composite overlay and draw text
        im = Image.alpha_composite(im, overlay)
        draw = ImageDraw.Draw(im)
        text_pos = (rect_x0 + text_padding, rect_y0 + text_padding)
        text_color = (255, 255, 255, 255)
        draw.text(text_pos, text, font=font, fill=text_color)

        frames.append(np.asarray(im.convert("RGB")))

    if not frames:
        print("No frames to write to video.")
        return

    try:
        # write mp4 using ffmpeg backend (requires ffmpeg on PATH)
        imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
        print(f"Saved video: {out_path}")
    except Exception as e:
        print(f"Failed to write mp4 with imageio: {e}. Trying GIF fallback.")
        try:
            gif_path = out_path.with_suffix(".gif")
            imageio.mimsave(str(gif_path), frames, fps=fps)
            print(f"Saved GIF fallback: {gif_path}")
        except Exception as e2:
            print(f"Failed to write GIF fallback: {e2}")

# Example usage
if __name__ == "__main__":
    make_video_from_folder_with_filenames(
        src_folder=r"P:\data\data_tmp\ship_picture",
        out_video_name="ship_picture_video.mp4",
        fps=5,
    )
# %%
