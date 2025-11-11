#%%
import osmnx as ox
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# Download boundary of Wedel
wedel = ox.geocode_to_gdf("Wedel, Germany")
#wedel_buffered = wedel.to_crs(epsg=4326).buffer(0.03)  # buffer in degrees for lat/lon

# Get bounds for the plot
#minx, miny, maxx, maxy = wedel_buffered.total_bounds

# Use Stamen Terrain or OSM as a background
tiler = cimgt.OSM()
mercator = tiler.crs

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=mercator)
minx, maxx, miny, maxy=9.63, 9.74, 53.54, 53.59
# Set extent in lat/lon
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

# Add OSM tiles
map_image =cimgt.GoogleTiles(url="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}@2x.png?api_key=42af7c69-c3f8-430c-abd2-4cacad752133")

ax.add_image(map_image,12)
ax.plot([9.65, 9.72], [53.55, 53.58], color='red', linewidth=2, marker='o', transform=ccrs.PlateCarree())

# Add latitude/longitude grid and labels
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

ax.set_title("Wedel and Surroundings (OSM Basemap with Cartopy)")
plt.show()
# %%
