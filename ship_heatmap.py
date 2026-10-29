#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle, ConnectionPatch, Polygon
import pyproj
from PIL import ImageEnhance
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from concurrent.futures import ThreadPoolExecutor



def read_ais_positions(file):
    """
    Lean AIS reader for the heatmap: returns only MMSI, Latitude, Longitude and
    Length_in_m (from the static type-9 messages) of the position messages.
    Much faster than ais.read_ais, which builds all columns and timestamps.
    """
    pos_lines, static_lines = [], []
    with open(file, "r", encoding="cp1252") as f:
        for line in f:
            parts = line.split(";")
            if len(parts) == 11 and "kt" in parts[3]:
                pos_lines.append(parts)
            elif len(parts) == 9:
                static_lines.append(parts)

    # drop exact duplicate messages (as read_ais does)
    pos = pd.DataFrame(pos_lines, columns=range(11)).drop_duplicates()
    pos = pd.DataFrame({
        "MMSI": pos[0].values,
        "Latitude": pd.to_numeric(pos[4].str.replace(r"[a-zA-Z]", "", regex=True), errors="coerce").values,
        "Longitude": pd.to_numeric(pos[5].str.replace(r"[a-zA-Z]", "", regex=True), errors="coerce").values,
    })

    static = pd.DataFrame(static_lines, columns=range(9))
    length = pd.to_numeric(static[6].str.split(" ").str[0], errors="coerce")
    length_per_mmsi = length.groupby(static[0].values).max()
    pos["Length_in_m"] = pos["MMSI"].map(length_per_mmsi)
    return pos

#%%
ais_dir = r"Q:\BREDOM\SEICOR\AIS"
min_lon, max_lon = 9.63, 9.715
min_lat, max_lat = 53.5525, 53.5825
# overview map extent; AIS positions are kept for this larger area to show the receiver coverage there
ov_min_lon, ov_max_lon = 9.56, 10.21
ov_min_lat, ov_max_lat = 53.43, 53.645

# YYMM of the AIS log files to use: every month from April 2025 to June 2026
MONTHS = pd.period_range("2025-04", "2026-06", freq="M").strftime("%y%m").tolist()
MIN_LENGTH_M = 150         # only ships longer than this
files = sorted(f for month in MONTHS
               for f in glob.glob(os.path.join(ais_dir, f"shipplotter{month}*.log")))
out_file = r"C:\Users\hhave\Nextcloud_neu\Promotion\publications\SEICOR_emissions_paper\figures\setup_map_ais.png"

def load_filtered(file):
    df = read_ais_positions(file)
    # Filter for ships longer than MIN_LENGTH_M inside the overview window
    return df[(df["Length_in_m"] > MIN_LENGTH_M)
              & df["Longitude"].between(ov_min_lon, ov_max_lon)
              & df["Latitude"].between(ov_min_lat, ov_max_lat)]

with ThreadPoolExecutor(max_workers=8) as pool:
    all_positions = list(pool.map(load_filtered, files))

if not all_positions:
    print("No AIS files found!")
    exit()

ov_positions = pd.concat(all_positions, ignore_index=True)
positions = ov_positions[ov_positions["Longitude"].between(min_lon, max_lon)
                         & ov_positions["Latitude"].between(min_lat, max_lat)]
print(f"{len(files)} files, {len(positions)} positions in the detail map, {len(ov_positions)} in the overview")
lons = positions["Longitude"].values
lats = positions["Latitude"].values

#%% Plot: (a) overview of Hamburg, (b) AIS density around the instrument, joined by zoom lines
LAYOUT = "horizontal"  # "vertical": (a) above (b); "horizontal": (a) left of (b)
out_file = (r"C:\Users\hhave\Nextcloud_neu\Promotion\publications\SEICOR_emissions_paper\figures"
            rf"\setup_map_combined_{LAYOUT}.pdf")

# Final printed size (double-column journal width), so font sizes are what readers see
FIG_WIDTH_IN = 7.2
# side by side, each map is only ~1.7 in tall, so text has to be smaller
FONT_SIZE = 9 if LAYOUT == "vertical" else 7
TICK_FONT_SIZE = FONT_SIZE - 1  # longitude/latitude labels at the map edges
YTICK_ROTATION = 90  # latitude labels along the axis; 0 = horizontal
BIN_SIZE_M = 10  # heatmap cell size on the ground in metres
TILE_ZOOM = 12     # detail map (b)
OV_TILE_ZOOM = 12  # overview map (a)

LOS_COLOR = "#1f3b73"        # dark blue: site markers, labels, port area
# IMPACT gold as in the setup sketch and the IMPACT photo figure
IMPACT_COLOR = "#ffc000"
IMPACT_ALT_COLOR = "#ffe699"  # lighter gold for the alternative (noon) LOS
# redder and darker, so the two stay distinguishable (also for colour-blind readers)
LONGPATH_COLOR = "#e8590c"
INSITU_COLOR = "black"
OV_BIN_SIZE_M = 100  # heatmap cell size in the overview map
SHOW_OV_AIS = True   # AIS density in the overview map too (shows how far the receiver reaches)

stadia_key = "42af7c69-c3f8-430c-abd2-4cacad752133"
MAP_SATURATION = 0.3  # base map colour strength: 0 = greyscale, 1 = original colours

class GreyTiles(cimgt.GoogleTiles):
    """Tiles with reduced colour saturation, so the base map doesn't compete with the coloured data."""
    def get_image(self, tile):
        img, extent, origin = super().get_image(tile)
        # 0 = greyscale, 1 = original colours
        return ImageEnhance.Color(img.convert("RGB")).enhance(MAP_SATURATION), extent, origin

tiler = GreyTiles(url="https://tiles.stadiamaps.com/tiles/alidade_bright/{z}/{x}/{y}@2x.png?api_key=" + stadia_key)
mercator = tiler.crs

# Histogram directly in the map projection, so the image needs no reprojection
xy = mercator.transform_points(ccrs.PlateCarree(), lons, lats)
x0, y0 = mercator.transform_point(min_lon, min_lat, ccrs.PlateCarree())
x1, y1 = mercator.transform_point(max_lon, max_lat, ccrs.PlateCarree())
# Mercator stretches by 1/cos(lat); scale the bin size so cells are BIN_SIZE_M on the ground
merc_scale = 1 / np.cos(np.deg2rad((min_lat + max_lat) / 2))
bin_merc = BIN_SIZE_M * merc_scale
xedges = np.arange(x0, x1 + bin_merc, bin_merc)
yedges = np.arange(y0, y1 + bin_merc, bin_merc)
heatmap, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xedges, yedges])

# Cells with fewer than MIN_COUNT positions (empty or single stray messages) are masked and drawn transparent
MIN_COUNT = 2
heatmap = np.ma.masked_less(heatmap, MIN_COUNT)
# light red -> dark red (the near-white start of "Reds" would vanish on the map)
cmap = ListedColormap(plt.get_cmap("Reds")(np.linspace(0.35, 1, 256)))
cmap.set_bad(alpha=0)
# cap at the 99th percentile so single hotspots (berths) don't wash out the lane
norm = Normalize(vmin=MIN_COUNT, vmax=np.percentile(heatmap.compressed(), 99))

ox0, oy0 = mercator.transform_point(ov_min_lon, ov_min_lat, ccrs.PlateCarree())
ox1, oy1 = mercator.transform_point(ov_max_lon, ov_max_lat, ccrs.PlateCarree())
ov_merc_scale = 1 / np.cos(np.deg2rad((ov_min_lat + ov_max_lat) / 2))


def lonlat_ticks(ax, lon_min, lon_max, lat_min, lat_max, lon_step, lat_step):
    """Faint grid + normal axis tick labels (cartopy's gridline labels can drop the outermost ones).
    Ticks exactly on the map edges are skipped, they collide with the other axis."""
    lon_t = np.arange(np.ceil(lon_min / lon_step) * lon_step, lon_max + 1e-9, lon_step)
    lat_t = np.arange(np.ceil(lat_min / lat_step) * lat_step, lat_max + 1e-9, lat_step)
    lon_t = lon_t[(lon_t > lon_min + 1e-6) & (lon_t < lon_max - 1e-6)]
    lat_t = lat_t[(lat_t > lat_min + 1e-6) & (lat_t < lat_max - 1e-6)]
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=lon_t, ylocs=lat_t,
                 linewidth=0.5, color='gray', alpha=0.3, linestyle='-')
    ax.set_xticks(lon_t, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_t, crs=ccrs.PlateCarree())
    # AMT style: space between degree sign and direction ("53.58° N")
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="° "))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="° "))
    plt.setp(ax.get_yticklabels(), rotation=YTICK_ROTATION, va="center")
    ax.tick_params(labelsize=TICK_FONT_SIZE, length=2.5, width=0.6)  # smaller labels and tick marks


MAP_ATTRIBUTION = "© Stadia Maps © OpenMapTiles © OpenStreetMap contributors"

def add_attribution(ax):
    """Tile provider credit, which AMT requires to be visible on the map itself."""
    ax.text(0.995, 0.008, MAP_ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_SIZE - 3, color="0.25", zorder=30,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))


def add_scalebar(ax, length_m, label, scale, size_vertical):
    """Scale bar; axes are in Mercator metres, which are stretched by 1/cos(lat)."""
    sb = AnchoredSizeBar(ax.transData, length_m * scale, label, loc="lower left", pad=0.4,
                         borderpad=0.6, sep=3, size_vertical=size_vertical, color="black",
                         frameon=True, fontproperties={"size": FONT_SIZE})
    sb.patch.set_alpha(0.85)
    sb.zorder = 30
    ax.add_artist(sb)


# Figure: both maps share the width, heights follow their aspect ratios
# TrueType fonts in the PDF (AMT asks for embedded fonts; matplotlib's default Type 3 can cause typesetting issues)
plt.rcParams.update({"font.size": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42})
r_ov = (oy1 - oy0) / (ox1 - ox0)
r_det = (y1 - y0) / (x1 - x0)
AX_WIDTH_IN = FIG_WIDTH_IN * 0.85
if LAYOUT == "vertical":
    fig = plt.figure(figsize=(FIG_WIDTH_IN, AX_WIDTH_IN * (r_ov + r_det) + 1.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[r_ov, r_det], hspace=0.12)
    ax2 = fig.add_subplot(gs[0], projection=mercator)  # (a) overview
    ax = fig.add_subplot(gs[1], projection=mercator)   # (b) detail
else:
    # same height for both maps, widths follow their aspect ratios
    ax_height_in = AX_WIDTH_IN * 0.92 / (1 / r_ov + 1 / r_det)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, ax_height_in + 1.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1 / r_det, 1 / r_ov], wspace=0.08)
    ax = fig.add_subplot(gs[0], projection=mercator)   # (a) detail, left
    ax2 = fig.add_subplot(gs[1], projection=mercator)  # (b) overview, right

# --- (b) detail map with AIS density
ax.set_extent([x0, x1, y0, y1], crs=mercator)
ax.add_image(tiler, TILE_ZOOM, interpolation='bilinear', alpha=0.6)  # faded base map

ax.imshow(
    heatmap.T,
    norm=norm,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin='lower',
    cmap=cmap,
    alpha=0.6,
    interpolation='nearest',
    transform=mercator,
    zorder=10
)
# subtle colorbar: same transparency as the heatmap, thin grey outline, small grey text
# (own axes, positioned after drawing, so it doesn't change the map widths):
# vertical layout: upright bar right of the detail map; horizontal layout: flat bar below the legend
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0, 0, 0.01, 0.01]), alpha=0.6,
                    orientation="vertical" if LAYOUT == "vertical" else "horizontal")
cbar.outline.set_edgecolor("0.6")
cbar.outline.set_linewidth(0.5)
cbar.set_label("AIS signal density", size=FONT_SIZE - 1, color="0.3")
cbar.set_ticks([])
# "low"/"high" at the ends of the bar instead of tick labels
if LAYOUT == "vertical":
    cbar_end_labels = [("high", (0.5, 1.03), "center", "bottom"), ("low", (0.5, -0.03), "center", "top")]
else:
    cbar_end_labels = [("low", (-0.06, 0.5), "right", "center"), ("high", (1.06, 0.5), "left", "center")]
for text, xy_txt, ha, va in cbar_end_labels:
    cbar.ax.text(*xy_txt, text, transform=cbar.ax.transAxes, ha=ha, va=va,
                 fontsize=FONT_SIZE - 1, color="0.3")

# IMPACT lines of sight: geodesics from the instrument, cut at the map boundary
impact_lat, impact_lon = 53.56958522848946, 9.69174249821205
# (azimuth in degrees from north, clockwise; color; opacity; legend label)
LOS_LINES = [(192.3, IMPACT_COLOR, 1.0, "IMPACT 2 LOS", "-"),
             (242.3, IMPACT_ALT_COLOR, 0.9, "IMPACT 2 alt. LOS (noon)", "--")]  # last: line style
geod = pyproj.Geod(ellps="WGS84")
dist = np.arange(0, 10000, 5.0)  # metres, longer than the map extent
legend_handles = []
# Instrument site; the in-situ instrument is co-located with IMPACT
site, = ax.plot(impact_lon, impact_lat, marker="o", linestyle="none", markersize=5.5,
                markerfacecolor=LOS_COLOR, markeredgecolor=INSITU_COLOR, markeredgewidth=1.2,
                transform=ccrs.PlateCarree(), zorder=22, label="Instrument location")
legend_handles.append(site)
impact_xy = mercator.transform_point(impact_lon, impact_lat, ccrs.PlateCarree())
ax.annotate("IMPACT 2", xy=impact_xy, xytext=(7, 4), textcoords="offset points",
            color=LOS_COLOR, fontweight="bold", va="bottom", zorder=22,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# River label in open water south of the shipping lane, rotated along the river
ax.text(9.6405, 53.5795, "Elbe", transform=ccrs.PlateCarree(), rotation=-12,
        ha="center", va="center", fontsize=FONT_SIZE + 3, fontstyle="italic",
        color="#4a6f96", zorder=15)

# Town label on land north of the river (upright, dark grey, like a place name on a map)
ax.text(9.697, 53.576, "Wedel", transform=ccrs.PlateCarree(), ha="center", va="center",
        fontsize=FONT_SIZE + 3, fontweight="bold", color="0.3", zorder=15,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


for azimuth, los_color, los_alpha, los_label, los_style in LOS_LINES:
    los_lons, los_lats, _ = geod.fwd(np.full_like(dist, impact_lon), np.full_like(dist, impact_lat),
                                     np.full_like(dist, azimuth), dist)
    inside = (los_lons >= min_lon) & (los_lons <= max_lon) & (los_lats >= min_lat) & (los_lats <= max_lat)
    n_inside = np.argmin(inside) if not inside.all() else len(inside)
    line, = ax.plot(los_lons[:n_inside], los_lats[:n_inside], color=los_color, alpha=los_alpha, linewidth=1.5,
                    linestyle=los_style,
                    # above the LP-DOAS path (zorder 21), below the instrument marker (22)
                    transform=ccrs.PlateCarree(), zorder=21.5,
                    label=los_label)
    legend_handles.append(line)

# 870 m long path parallel to the main LOS, shifted sideways by PARALLEL_OFFSET_M
# (positive = to the left when looking along the LOS, i.e. east for 192.3°)
PARALLEL_LENGTH_M = 870
PARALLEL_OFFSET_M = 40
los_azimuth = LOS_LINES[0][0]
start_lon, start_lat, _ = geod.fwd(impact_lon, impact_lat, los_azimuth - 90, PARALLEL_OFFSET_M)
d = np.linspace(0, PARALLEL_LENGTH_M, 100)
par_lons, par_lats, _ = geod.fwd(np.full_like(d, start_lon), np.full_like(d, start_lat),
                                 np.full_like(d, los_azimuth), d)
# black end bars (telescope and reflector) across both ends of the path, no outline along the sides.
# Marker (2, 2, angle) is a short line; rotated so it stands perpendicular to the path on the map
# (Mercator keeps angles, so the azimuth translates directly into the on-screen direction)
LP_END_BAR = dict(markersize=4, markeredgewidth=1.5, markeredgecolor="black")
ax.plot(par_lons, par_lats, color=LONGPATH_COLOR, linewidth=1.8, transform=ccrs.PlateCarree(), zorder=21,
        marker=(2, 2, 90 - los_azimuth), markevery=[0, -1], **LP_END_BAR)
# legend symbol: plain orange line, without the end bars
legend_handles.append(Line2D([], [], color=LONGPATH_COLOR, linewidth=1.8,
                             label=f"LP-DOAS light path"))
LEGEND_HANDLER_MAP = {}

# --- schematic ship with exhaust plume and velocity vectors (illustration, not data) -------------
SHIP_LON = 9.667             # position along the lane, west of the IMPACT LOS
ship_lat = 53.569
SHIP_LENGTH_M, SHIP_BEAM_M = 500, 90         # drawn slightly larger than a real ship for visibility
SHIP_SPEED, WIND_SPEED = 6.0, 5.0          # m/s, illustrative
WIND_TOWARDS_DEG = 150.0                   # direction the wind blows towards (from north, clockwise)
PLUME_LENGTH_M = 1900
SHIP_ARROW_M, WIND_ARROW_M = 450, 450      # drawn arrow lengths
WIND_ARROW_START = (9.677, 53.5765)        # (lon, lat), open area north of the lane
VECTOR_COLOR = "black"
PLUME_COLOR = "#505050"                    # grey exhaust



def merc(lon, lat):
    return np.array(mercator.transform_point(lon, lat, ccrs.PlateCarree()))

# heading downstream (westwards) along the lane, as unit vectors in local east/north metres
p_ship = merc(SHIP_LON, ship_lat)
d_lane = merc(SHIP_LON - 0.003, 53.5693) - p_ship
heading = d_lane / np.linalg.norm(d_lane)               # Mercator is conformal: same direction in metres
across = np.array([-heading[1], heading[0]])
wind_dir = np.array([np.sin(np.deg2rad(WIND_TOWARDS_DEG)), np.cos(np.deg2rad(WIND_TOWARDS_DEG))])
to_merc = lambda metres: metres * merc_scale            # local metres -> Mercator units

# hull outline (pointed bow), in Mercator coordinates
L, B = SHIP_LENGTH_M, SHIP_BEAM_M
hull_local = [(-L / 2, -B / 2), (L / 2 - B, -B / 2), (L / 2, 0), (L / 2 - B, B / 2), (-L / 2, B / 2)]
hull = np.array([p_ship + to_merc(a * heading + c * across) for a, c in hull_local])

# plume: puffs emitted tau seconds ago sit at funnel + (v_wind - v_ship) * tau
funnel = p_ship + to_merc(-0.25 * L * heading)
drift = WIND_SPEED * wind_dir - SHIP_SPEED * heading
drift_dir = drift / np.linalg.norm(drift)
drift_across = np.array([-drift_dir[1], drift_dir[0]])
n_seg = 40
s = np.linspace(0, PLUME_LENGTH_M, n_seg + 1)
half_width = 15 + 0.12 * s                              # widening plume, metres
for i in range(n_seg):
    corners = [funnel + to_merc(s[j] * drift_dir + side * half_width[j] * drift_across)
               for j, side in [(i, -1), (i + 1, -1), (i + 1, 1), (i, 1)]]
    ax.add_patch(Polygon(corners, closed=True, transform=mercator, facecolor=PLUME_COLOR,
                         edgecolor="none", alpha=0.9 * (1 - i / n_seg) ** 0.5, zorder=21.8))  # above the LOS lines (21-21.5), below the instrument marker (22)
ax.add_patch(Polygon(hull, closed=True, transform=mercator, facecolor="0.15", edgecolor="black",
                     linewidth=0.5, zorder=21.9))

# velocity vectors with labels
VEC_HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

def vector(start, direction, length_m, text, text_offset):
    end = start + to_merc(length_m * direction)
    ax.annotate("", xy=end, xytext=start, xycoords=ax.transData, textcoords=ax.transData,
                arrowprops=dict(arrowstyle="-|>", color=VECTOR_COLOR, lw=1.3, shrinkA=0, shrinkB=0,
                                mutation_scale=9), zorder=23)
    ax.annotate(text, xy=end, xytext=text_offset, textcoords="offset points", ha="center", va="center",
                fontsize=FONT_SIZE + 1, color=VECTOR_COLOR, zorder=23, path_effects=VEC_HALO)

# ship vector parallel to the hull, on its north side (the plume drifts to the south-east)
SHIP_ARROW_OFFSET_M = 170
vector(p_ship + to_merc(-0.5 * SHIP_ARROW_M * heading - SHIP_ARROW_OFFSET_M * across), heading, SHIP_ARROW_M,
       r"$\vec{v}_\mathrm{ship}$", (0, 7))
vector(merc(*WIND_ARROW_START), wind_dir, WIND_ARROW_M, r"$\vec{v}_\mathrm{wind}$", (-14, 2))

# legend order: instrument, IMPACT LOS, alternative IMPACT LOS, LP-DOAS
legend_handles = [legend_handles[i] for i in (0, 1, 2, 3)]
# legend below the map, so it doesn't cover the shipping lane
# matplotlib fills legend columns top-to-bottom; reorder so entries read left-to-right
LEGEND_NCOL = 2
nrows = int(np.ceil(len(legend_handles) / LEGEND_NCOL))
row_major = [legend_handles[r * LEGEND_NCOL + c] for c in range(LEGEND_NCOL) for r in range(nrows)
             if r * LEGEND_NCOL + c < len(legend_handles)]
# horizontal layout: the legend spans both maps and is placed after drawing (see below)
if LAYOUT == "vertical":
    ax.legend(handles=row_major, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=LEGEND_NCOL,
              frameon=False, fontsize=FONT_SIZE - 1, handler_map=LEGEND_HANDLER_MAP)

add_scalebar(ax, 500, "500 m", merc_scale, size_vertical=8)
add_attribution(ax)
lonlat_ticks(ax, min_lon, max_lon, min_lat, max_lat, 0.05, 0.02)  # lon 0.05°: 9.65°, 9.7° E; lat 0.02°: 53.56°, 53.58° N

# --- (a) overview map of Hamburg
ax2.set_extent([ox0, ox1, oy0, oy1], crs=mercator)
ax2.add_image(tiler, OV_TILE_ZOOM, interpolation='bilinear', alpha=1.0)

# (name, lat, lon, marker, label offset in points, horizontal / vertical label alignment)
if LAYOUT == "vertical":
    SITES = [
        # label right of the AIS-map frame, so it doesn't cross it
        ("IMPACT 2", impact_lat, impact_lon, "o", (28, 0), "left", "center"),
        # two lines, centred below the marker, so it stays clear of the port area
        ("Meteomast\nBillwerder", 53.51922389077999, 10.10283134530118, "^", (0, -8), "center", "top"),
    ]
else:
    # small map: instrument label above the frame (the LOS runs south of it), Billwerder below its marker,
    # so the labels don't stack
    SITES = [
        # above right of the marker, same placement as in the detail map
        ("IMPACT 2", impact_lat, impact_lon, "o", (7, 6), "left", "bottom"),  # just above the frame
        # two lines, centred, so it stays clear of the port area
        ("Meteomast\nBillwerder", 53.51922389077999, 10.10283134530118, "^", (0, -7), "center", "top"),
    ]

# Port of Hamburg as an area: approximate outline around the harbour basins
# (Waltershof, Altenwerder, Steinwerder, Kleiner Grasbrook), drawn below the markers.
# Corners as (lat, lon), the order Google Maps copies them in
PORT_AREA_LATLON = [(53.465221308125145, 9.994057621750427), (53.477778635730985, 9.93157928913051), (53.496025256878035, 9.928513731680528), (53.50504230455625, 9.8865820397531), (53.53156171225776, 9.885861990220409), (53.54304685076283, 9.860533543409668), 
             (53.54674463226084, 9.915074171060953),  (53.542854955452, 9.990405117026345),(53.522744707172976, 10.061646219712818),(53.51126623867092, 10.059198778049351),(53.51225652970764, 9.99279099957341),
             (53.48886771009989, 9.991436940004371)]
port_lat, port_lon = np.array(PORT_AREA_LATLON).T
port_xy = mercator.transform_points(ccrs.PlateCarree(), port_lon, port_lat)[:, :2]
ax2.add_patch(Polygon(port_xy, closed=True, transform=ax2.transData, facecolor=to_rgba(LOS_COLOR, 0.15),
                      edgecolor=LOS_COLOR, linewidth=1.0, zorder=17))
# label centred above the area
# label inside the area (centre of its corners), so "Hamburg" can sit north of the river
ax2.annotate("Port of\nHamburg", xy=(port_xy[:, 0].mean(), port_xy[:, 1].mean()),
             xytext=(0, 0), textcoords="offset points", ha="center", va="center",
             color=LOS_COLOR, fontweight="bold", zorder=21,
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Town names, same style as "Wedel" in the detail map; placed north of the river,
# clear of the instrument label and the port label
# (name, lat, lon, font size, grey level): Wedel smaller and lighter as the smaller town
for town, town_lat, town_lon, town_size, town_grey in [("Wedel", 53.605, 9.685, FONT_SIZE + 1, "0.55"),
                                                       ("Hamburg", 53.566, 9.990, FONT_SIZE + 3, "0.3")]:
    ax2.text(town_lon, town_lat, town, transform=ccrs.PlateCarree(), ha="center", va="center",
             fontsize=town_size, fontweight="bold", color=town_grey, zorder=21,
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

for name, lat, lon, marker, offset, ha, va in SITES:
    ax2.plot(lon, lat, marker=marker, linestyle="none", markersize=6,
             markerfacecolor=LOS_COLOR, markeredgecolor="black", markeredgewidth=1.0,
             transform=ccrs.PlateCarree(), zorder=20)
    ax2.annotate(name, xy=mercator.transform_point(lon, lat, ccrs.PlateCarree()),
                 xytext=offset, textcoords="offset points", ha=ha, va=va,
                 color=LOS_COLOR, fontweight="bold", zorder=21,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# AIS density over the whole overview, same colours as the detail map, coarser cells
if SHOW_OV_AIS:
    ov_xy = mercator.transform_points(ccrs.PlateCarree(), ov_positions["Longitude"].values,
                                      ov_positions["Latitude"].values)
    ov_bin_merc = OV_BIN_SIZE_M * ov_merc_scale
    ov_xedges = np.arange(ox0, ox1 + ov_bin_merc, ov_bin_merc)
    ov_yedges = np.arange(oy0, oy1 + ov_bin_merc, ov_bin_merc)
    ov_heatmap, _, _ = np.histogram2d(ov_xy[:, 0], ov_xy[:, 1], bins=[ov_xedges, ov_yedges])
    ov_heatmap = np.ma.masked_less(ov_heatmap, MIN_COUNT)
    ax2.imshow(ov_heatmap.T, norm=Normalize(vmin=MIN_COUNT, vmax=np.percentile(ov_heatmap.compressed(), 99)),
               extent=[ov_xedges[0], ov_xedges[-1], ov_yedges[0], ov_yedges[-1]], origin='lower',
               cmap=cmap, alpha=0.6, interpolation='nearest', transform=mercator, zorder=10)

# IMPACT lines of sight, fading out towards OV_LOS_LENGTH_M (below frame and markers)
OV_LOS_LENGTH_M = 15000
d_ov = np.linspace(0, OV_LOS_LENGTH_M, 300)
OV_DASH_M = 600  # dash and gap length for dashed lines in the overview
for azimuth, los_color, los_alpha, _, los_style in LOS_LINES:
    ov_lons, ov_lats, _ = geod.fwd(np.full_like(d_ov, impact_lon), np.full_like(d_ov, impact_lat),
                                   np.full_like(d_ov, azimuth), d_ov)
    pts = mercator.transform_points(ccrs.PlateCarree(), ov_lons, ov_lats)[:, :2]
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    colors = np.tile(to_rgba(los_color), (len(segments), 1))
    colors[:, 3] = los_alpha * (1 - d_ov[:-1] / OV_LOS_LENGTH_M)  # linear fade to transparent
    if los_style == "--":
        # dashes made by leaving out every other stretch: matplotlib's own dash pattern would
        # restart on each of the short fading segments and look solid
        keep = (d_ov[:-1] // OV_DASH_M) % 2 == 0
        segments, colors = segments[keep], colors[keep]
    ax2.add_collection(LineCollection(segments, colors=colors, linewidths=1.2,
                                      transform=ax2.transData, zorder=18))

# direction to the North Sea: arrow along the Elbe (centre of the AIS lane) to the western map edge,
# text rotated parallel to the arrow on its south-west side
NS_TAIL_LON, NS_TIP_LON = 9.620, 9.563
ov_lane_bins = np.arange(ov_min_lon, NS_TAIL_LON + 0.01, 0.005)
ov_lane_lat = (ov_positions.groupby(pd.cut(ov_positions["Longitude"], ov_lane_bins), observed=False)["Latitude"]
               .median().to_numpy())
ov_ok = np.isfinite(ov_lane_lat)
ov_lane_at = lambda lon: np.interp(lon, (0.5 * (ov_lane_bins[1:] + ov_lane_bins[:-1]))[ov_ok], ov_lane_lat[ov_ok])
ns_tail = merc(NS_TAIL_LON, ov_lane_at(NS_TAIL_LON))
ns_tip = merc(NS_TIP_LON, ov_lane_at(NS_TIP_LON))
ns_dir = (ns_tip - ns_tail) / np.linalg.norm(ns_tip - ns_tail)
ns_angle = np.degrees(np.arctan2(ns_dir[1], ns_dir[0])) - 180  # text reads left-to-right, pointing upriver-down
ns_normal = np.array([-ns_dir[1], ns_dir[0]])
if ns_normal[1] > 0:  # south-west side, away from the "Wedel" label
    ns_normal = -ns_normal
# arrow beside the river (not on the AIS lane), text further out on the same side
NS_ARROW_OFFSET_M = 900
ns_shift = to_merc(NS_ARROW_OFFSET_M * ns_normal) * ov_merc_scale / merc_scale
ax2.annotate("", xy=ns_tip + ns_shift, xytext=ns_tail + ns_shift, xycoords=ax2.transData,
             textcoords=ax2.transData,
             arrowprops=dict(arrowstyle="-|>", color="#4a6f96", lw=1.2, shrinkA=0, shrinkB=0), zorder=21)
ax2.annotate("North Sea", xy=0.35 * ns_tip + 0.65 * ns_tail + ns_shift, xycoords=ax2.transData,
             xytext=tuple(4 * ns_normal), textcoords="offset points", rotation=ns_angle,
             rotation_mode="anchor", ha="center", va="top", fontsize=FONT_SIZE, fontstyle="italic",
             color="#4a6f96", zorder=21, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# frame showing the extent of the detailed map
ax2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=mercator, fill=False,
                        edgecolor="black", linewidth=1.0, zorder=19))

add_scalebar(ax2, 5000, "5 km", ov_merc_scale, size_vertical=80)
add_attribution(ax2)
# lon step 0.2°: 9.8° and 10° E; lat step 0.1°: 53.5° and 53.6° N
lonlat_ticks(ax2, ov_min_lon, ov_max_lon, ov_min_lat, ov_max_lat, 0.2, 0.1)
# longitude labels on top, so the gap between the maps stays free for the zoom lines
# tick labels of the inner edges moved outward, so the gap between the maps stays free for the zoom lines
if LAYOUT == "vertical":
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
else:
    ax2.yaxis.set_ticks_position("right")
    ax2.yaxis.set_label_position("right")
    # the right-hand labels are separate text objects and don't inherit the rotation from lonlat_ticks
    plt.setp([t.label2 for t in ax2.yaxis.get_major_ticks()], rotation=YTICK_ROTATION, va="center")

# --- zoom lines, panel letters, colorbar position, save
# letters in reading order: the top/left panel is (a)
panel_letters = [(ax2, "(a)"), (ax, "(b)")] if LAYOUT == "vertical" else [(ax, "(a)"), (ax2, "(b)")]
for panel_ax, letter in panel_letters:
    panel_ax.text(0.99, 0.97, letter, transform=panel_ax.transAxes, ha="right", va="top",
                  fontweight="bold", fontsize=FONT_SIZE + 1, zorder=30,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

# positions are only final after drawing, because cartopy shrinks the maps to keep their aspect
fig.canvas.draw()

# zoom lines from the frame in (a) to the facing corners of (b):
# faint and below markers/labels inside (a), solid in the gap between the panels
ZOOM_LINE_COLOR = "0.25"
if LAYOUT == "vertical":
    # bottom corners of the frame -> top corners of (b), leaving (a) through its bottom edge
    corner_pairs = [((x0, y0), (0, 1)), ((x1, y0), (1, 1))]
    edge_dim, edge_point = 1, (0, 0)
else:
    # left corners of the frame -> right corners of the detail map, leaving the overview through its left edge
    corner_pairs = [((x0, y1), (1, 1)), ((x0, y0), (1, 0))]
    edge_dim, edge_point = 0, (0, 0)
edge_a = ax2.transAxes.transform(edge_point)[edge_dim]  # exit edge of (a), display coords
for frame_corner, panel_corner in corner_pairs:
    p_frame = ax2.transData.transform(frame_corner)
    p_panel = ax.transAxes.transform(panel_corner)
    t = (p_frame[edge_dim] - edge_a) / (p_frame[edge_dim] - p_panel[edge_dim])
    p_edge = ax2.transData.inverted().transform(p_frame + t * (p_panel - p_frame))
    ax2.plot([frame_corner[0], p_edge[0]], [frame_corner[1], p_edge[1]], transform=mercator,
             color=ZOOM_LINE_COLOR, linewidth=0.6, alpha=0.5, zorder=15)
    fig.add_artist(ConnectionPatch(xyA=p_edge, coordsA=ax2.transData,
                                   xyB=panel_corner, coordsB=ax.transAxes,
                                   color=ZOOM_LINE_COLOR, linewidth=0.8, zorder=25))
ax2.set_extent([ox0, ox1, oy0, oy1], crs=mercator)  # plotting must not change the extent

pos = ax.get_position()
if LAYOUT == "vertical":
    # colorbar right of the detail map: thin and 60 % of the map height, vertically centred
    cbar_height = 0.6 * pos.height
    cbar.ax.set_position([pos.x1 + 0.015, pos.y0 + (pos.height - cbar_height) / 2, 0.012, cbar_height])
else:
    # one row below both maps (under the longitude labels), centred under both:
    # [legend entries]  AIS signal density  low [colorbar] high
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    def fig_width(artist):
        """Width of a drawn artist in figure fractions."""
        return np.ptp(to_fig.transform(artist.get_window_extent(renderer))[:, 0])

    labels_y0 = min(to_fig.transform(lbl.get_window_extent(renderer))[0, 1]
                    for a in (ax, ax2) for lbl in a.get_xticklabels() if lbl.get_visible() and lbl.get_text())
    pos2 = ax2.get_position()
    x_center = (pos.x0 + pos2.x1) / 2

    # tighter spacing than the default, so the row isn't wider than the two maps
    leg = fig.legend(handles=legend_handles, loc="center left", ncol=len(legend_handles),
                     frameon=False, fontsize=FONT_SIZE - 1, columnspacing=1.2, handlelength=1.6,
                     handletextpad=0.5, handler_map=LEGEND_HANDLER_MAP)
    cbar.set_label("")  # title is drawn left of the bar instead of below it
    cbar_title = fig.text(0, 0, "AIS signal density", ha="left", va="center",
                          fontsize=FONT_SIZE - 1, color="0.3")
    fig.canvas.draw()
    end_labels = {t.get_text(): t for t in cbar.ax.texts}  # "low" / "high" from above

    gap, pad = 0.015, 0.008
    cbar_width, cbar_thickness = 0.1, 0.025
    fixed = gap + gap / 2 + 2 * pad + cbar_width  # parts of the row that don't scale with the font

    def text_width():
        return (fig_width(leg) + fig_width(cbar_title) + fig_width(end_labels["low"])
                + fig_width(end_labels["high"]))

    # scale the row's text so the row spans the two maps (legend handles scale with the font size)
    base_size = FONT_SIZE - 1
    row_size = float(np.clip(base_size * (pos2.x1 - pos.x0 - fixed) / text_width(), 5, FONT_SIZE))
    for t in [*leg.get_texts(), cbar_title, *end_labels.values()]:
        t.set_fontsize(row_size)
    leg.prop.set_size(row_size)  # handle lengths and spacing are in font-size units
    leg._fontsize = leg.prop.get_size_in_points()
    fig.canvas.draw()

    leg_height = np.ptp(to_fig.transform(leg.get_window_extent(renderer))[:, 1])
    total = fixed + text_width()
    x = x_center - total / 2
    y_row = labels_y0 - 0.02 - leg_height / 2

    leg.set_bbox_to_anchor((x, y_row), transform=fig.transFigure)
    x += fig_width(leg) + gap
    cbar_title.set_position((x, y_row))
    x += fig_width(cbar_title) + gap / 2 + fig_width(end_labels["low"]) + pad
    cbar.ax.set_position([x, y_row - cbar_thickness / 2, cbar_width, cbar_thickness])

plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.show()
#%%
