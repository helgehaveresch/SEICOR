#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import re



def read_ais(file):
    """
    Input is file and path of ais datafile.
    """
    KNOTS_TO_MS = 1/1.944

    # open file line by line
    with open(file, "r", encoding="cp1252") as f:
        content = [line.split(";") for line in f]

    # sort content by length, different message types have different length
    # and different information content
    content5 = pd.DataFrame([line for line in content if len(line) == 5],
                            columns=["MMSI",
                                     "Interogration",
                                     "Message",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content6 = pd.DataFrame([line for line in content if len(line) == 6],
                            columns=["MMSI",
                                     "Class",
                                     "Part",
                                     "Name",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content7 = pd.DataFrame([line for line in content if len(line) == 7],
                            columns=["MMSI",
                                     "Voyage data",
                                     "Unkown",
                                     "Dimensions",
                                     "ERI",
                                     "Draught load UTC Time",
                                     "Serial"],
                            dtype=str)

    content7["len"] = ""
    content7["beam"] = ""
    content7["draught"] = ""
    content7["load"] = ""
    content7["UTC Time"] = ""

    for i, index in enumerate(content7.index):
        content7.loc[index, "len"] = content7.loc[index, "Dimensions"].split(" ")[1]
        content7.loc[index, "beam"] = content7.loc[index, "Dimensions"].split(" ")[3]
        content7.loc[index, "draught"] = content7.loc[index, "Draught load UTC Time"].split(" ")[1]
        content7.loc[index, "load"] = content7.loc[index, "Draught load UTC Time"].split(" ")[3]
        content7.loc[index, "UTC Time"] = content7.loc[index, "Draught load UTC Time"].split(" ")[4]+" "+content7.loc[index, "Draught load UTC Time"].split(" ")[5]

    content9 = pd.DataFrame([line for line in content if len(line) == 9],
                            columns=["MMSI",
                                     "Class",
                                     "Unknown",
                                     "Name",
                                     "Type",
                                     "Destination",
                                     "Dimensions",
                                     "UTC Time",
                                     "Serial"],
                            dtype=str)

    content9["Length_in_m"] = ""
    content9["Width_in_m"] = ""
    content9["Draught_in_m"] = ""

    for i, index in enumerate(content9.index):
        content9.loc[index, "Length_in_m"] = content9.loc[index, "Dimensions"].split(" ")[0]
        content9.loc[index, "Width_in_m"] = content9.loc[index, "Dimensions"].split(" ")[1]
        content9.loc[index, "Draught_in_m"] = content9.loc[index, "Dimensions"].split(" ")[2]

    # content 11 is most common message and includes positional data
    content11 = pd.DataFrame([line for line in content if len(line) == 11],
                             columns=["MMSI",
                                      "Navigation_status",
                                      "Rate of turn",
                                      "Speed in knots",
                                      "Latitude",
                                      "Longitude",
                                      "Course_in_deg",
                                      "True_heading_in_deg",
                                      "UTC seconds",
                                      "UTC Time",
                                      "Serial"],
                             dtype=str)

    content11 = content11.assign(Name="", Type="", Length_in_m="", Width_in_m="",
                                 Draught_in_m="", Destination="", ERI="")

    content11["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content11["UTC Time"]])
    content5["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content5["UTC Time"]])
    content6["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content6["UTC Time"]])
    content7["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content7["UTC Time"]])
    content9["UTC Time"] = convert_timestamp([re.findall(r"\d\d\d\d\d\d \d\d\d\d\d\d", time)[0] for time in content9["UTC Time"]])

    # combine all information from different message types
    for i, mmsi in enumerate(content11["MMSI"].unique()):
        index11 = np.where(mmsi == content11["MMSI"])[0]

        if len(content7) > 0:
            try:
                index7 = np.where(mmsi == content7["MMSI"])[0]
                content11.loc[index11, "ERI"] = str(np.unique(content7.loc[index7, "ERI"])[0])
            except:
                pass

        if len(content6) > 0:
            index6 = np.where(mmsi == content6["MMSI"])[0]
        if len(content9) > 0:
            index9 = np.where(mmsi == content9["MMSI"])[0]
            if len(index9) > 0:
                content11.loc[index11, "Name"] = str(np.unique(content9.loc[index9, "Name"])[0])
                content11.loc[index11, "Type"] = str(np.unique(content9.loc[index9, "Type"])[0])
                content11.loc[index11, "Destination"] = str(np.unique(content9.loc[index9, "Destination"])[0])
                content11.loc[index11, "Length_in_m"] = str(np.unique(content9.loc[index9, "Length_in_m"])[0])
                content11.loc[index11, "Width_in_m"] = str(np.unique(content9.loc[index9, "Width_in_m"])[0])
                content11.loc[index11, "Draught_in_m"] = str(np.unique(content9.loc[index9, "Draught_in_m"])[0])

        if all(content11.loc[index11, "Name"] == "Part B"):
            try:
                content11.loc[index11, "Name"] = str(np.unique(content6.loc[index6, "Name"])[0])
            except:
                content11.loc[index11, "Name"] = str(" ")

    # convert to appropiate data types and change some units
    content11 = content11.loc[["kt" in c for c in content11.loc[:, "Speed in knots"]], :]
    content11.loc[:, "Speed in knots"] = [float(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Speed in knots"]]
    content11.loc[:, "Latitude"] = [np.float64(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Latitude"]]
    content11.loc[:, "Longitude"] = [np.float64(re.sub("[a-zA-Z]", "", c)) for c in content11.loc[:, "Longitude"]]
    content11.loc[:, "Name"] = [str(str(c).rstrip()) for c in content11.loc[:, "Name"]]
    content11.loc[:, "Navigation_status"] = [ str(c.rstrip()) for c in content11.loc[:, "Navigation_status"]]
    content11.loc[:, "Type"] = [str(str(c).rstrip()) for c in content11.loc[:, "Type"]]
    content11.drop_duplicates(inplace=True)

    content11.loc[:, "True_heading_in_deg"] = [str(i) for i in content11.loc[:, "True_heading_in_deg"]]
    content11.loc[:, "True_heading_in_deg"] = [float(i.replace("°", "")) if i is not isinstance(i, float) else np.nan for i in content11.loc[:, "True_heading_in_deg"]]
    content11.loc[:, "True_heading_in_deg"] = [float(i) if float(i) <= 359 else np.nan for i in content11.loc[:, "True_heading_in_deg"]]

    content11.loc[:, "Speed_in_m/s"] = [i * KNOTS_TO_MS for i in content11.loc[:, "Speed in knots"]]

    content11.loc[:, "Course_in_deg"] = [str(i) for i in content11.loc[:, "Course_in_deg"]]
    content11.loc[content11.loc[:, "Course_in_deg"] == " unk. ", "Course_in_deg"] = 9999
    content11.loc[:, "Course_in_deg"] = pd.to_numeric(content11.loc[:, "Course_in_deg"], errors='coerce')

    content11.loc[:, "Length_in_m"] = pd.to_numeric(content11.loc[:, "Length_in_m"].values, errors="coerce")
    content11.loc[:, "Width_in_m"] = pd.to_numeric(content11.loc[:, "Width_in_m"].values, errors="coerce")
    content11.loc[:, "Draught_in_m"] = pd.to_numeric(content11.loc[:, "Draught_in_m"].values, errors="coerce")

    content11.set_index("UTC Time", inplace=True)
    content11.index.name = "UTC_Time"
    # subset only important information
    content11 = content11[["MMSI", "ERI", "Navigation_status", "Speed_in_m/s", "Latitude", "Longitude",
                           "Course_in_deg", "True_heading_in_deg", "Serial", "Name", "Type",
                           "Length_in_m", "Width_in_m", "Draught_in_m", "Destination"]]
    # prepare output with appropiate data types
    content11 = content11.astype({"MMSI": int,
                                  "ERI": str,
                                  "Navigation_status": str,
                                  "Speed_in_m/s": float,
                                  "Latitude": float,
                                  "Longitude": float,
                                  "Course_in_deg": float,
                                  "True_heading_in_deg": float,
                                  "Serial": str,
                                  "Name": str,
                                  "Type": str,
                                  "Length_in_m": float,
                                  "Width_in_m": float,
                                  "Draught_in_m": float,
                                  "Destination": str})

    content11 = content11.replace({np.nan: None})

    return content11

def convert_timestamp(time):

    """
    Convert timestamps to pandas.datetime objects.
    """
    time = np.asarray([str(t.replace(" ", "")) for t in time])

    y = [str("20"+y[:2]) for y in time]
    m = [str(m[2:4]) for m in time]
    d = [str(d[4:6]) for d in time]
    H = [str(H[6:8]) for H in time]
    M = [str(M[8:10]) for M in time]
    S = [str(S[10:12]) for S in time]

    time = [pd.to_datetime(y+"-"+m+"-"+d+" "+H+":"+M+":"+S, yearfirst=True, utc=True) for y, m, d, H, M, S
            in zip(y, m, d, H, M, S)]
    return time
#%%
ais_dir = r"P:\data\data_tmp\AIS"
files = glob.glob(os.path.join(ais_dir, "shipplotter25[04-05]*.log"))

all_positions = []
for file in files:
    print(f"Reading {file}")
    df = read_ais(file)
    # Filter for ships longer than 80 meters
    df = df[df["Length_in_m"].notnull() & (df["Length_in_m"] > 80)]
    all_positions.append(df)

if not all_positions:
    print("No AIS files found!")
    exit()

positions = pd.concat(all_positions, ignore_index=True)
lons = positions["Longitude"].values
lats = positions["Latitude"].values

#%% Plot heatmap
min_lon, max_lon = 9.63, 9.73
min_lat, max_lat = 53.555, 53.585

tiler = cimgt.GoogleTiles(url="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}@2x.png?api_key=42af7c69-c3f8-430c-abd2-4cacad752133")
mercator = tiler.crs

# Create transparent colormap from 'hot'
base_cmap = cm.get_cmap('hot', 256)
colors = base_cmap(np.linspace(0, 1, 256))
colors[:10, -1] = np.linspace(0.6, 0.6, 10)  # Make lowest 100 alpha-transparent
transparent_hot = ListedColormap(colors)

# Plot
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=mercator)
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
ax.add_image(tiler, 13, interpolation='bilinear', alpha=1.0)
ax.set_facecolor('white')

# Compute heatmap
heatmap, xedges, yedges = np.histogram2d(
    lons, lats, bins=300, range=[[min_lon, max_lon], [min_lat, max_lat]]
)
extent = [min_lon, max_lon, min_lat, max_lat]

# Show heatmap with custom colormap
ax.imshow(
    np.log1p(heatmap.T),  # log1p to compress dynamic range
    extent=extent,
    origin='lower',
    cmap=transparent_hot,
    alpha=0.6,
    transform=ccrs.PlateCarree(),
    zorder=10
)

gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.MultipleLocator(0.05)
gl.ylocator = mticker.MultipleLocator(0.02)

# Label + export
plt.title("Ship Trajectories in Wedel", size=20)
plt.savefig(os.path.join(ais_dir, "ship_trajectory_heatmap_apr_may_80m.png"), dpi=200)
plt.show()
#%%