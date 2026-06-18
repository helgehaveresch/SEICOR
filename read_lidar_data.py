#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import xarray as xr
import os
from pathlib import Path
import zipfile


def read_csv_from_zip(zip_path: str | os.PathLike, inner_csv_name: str | None = None, **read_csv_kwargs) -> pd.DataFrame:
	"""Read a CSV inside a .zip archive into a pandas DataFrame.

	Parameters
	----------
	zip_path:
		Path to the .zip file.
	inner_csv_name:
		File name inside the zip (e.g. "Wind10_1082@Y2025_M06_D01.CSV").
		If None, the first member ending with .csv (case-insensitive) is used.
	**read_csv_kwargs:
		Forwarded to pandas.read_csv (delimiter/sep, encoding, skiprows, etc.).
	"""
	zip_path = Path(zip_path)
	if not zip_path.exists():
		raise FileNotFoundError(f"Zip file not found: {zip_path}")

	with zipfile.ZipFile(zip_path, mode="r") as zf:
		members = zf.namelist()
		if not members:
			raise FileNotFoundError(f"Zip archive is empty: {zip_path}")
		with zf.open(inner_csv_name, mode="r") as f:
			return pd.read_csv(f, **read_csv_kwargs)


def read_wind_ranger(
	start_date_yymmdd: str,
	base_dir: str | os.PathLike = r"D:\SEICOR\wind_lidar",
	station: str = "Wind_1082",
	end_date_yymmdd: str | None = None,
	max_days: int = 400,
	missing: str = "break",
	**read_csv_kwargs,
	) -> pd.DataFrame:
	"""Read daily Wind10 lidar CSVs from zipped archives and concatenate.

	Expected file naming pattern per day:
	- Zip:  {station}@Y{YYYY}_M{MM}_D{DD}.CSV.zip
	- CSV:  {station}@Y{YYYY}_M{MM}_D{DD}.CSV

	Parameters
	----------
	start_date_yymmdd:
		Start date as "yymmdd".
	base_dir:
		Directory that contains the daily zip files.
	station:
		Prefix in the file name (default: "Wind_1082").
	end_date_yymmdd:
		Optional end date as "yymmdd" (inclusive). If None, reads until first missing day.
	max_days:
		Safety cap on number of days to try when end_date is None.
	missing:
		What to do if a day is missing: "break" (default), "skip", or "raise".
	**read_csv_kwargs:
		Passed to pandas.read_csv (via read_csv_from_zip).
	"""
	start = pd.to_datetime(start_date_yymmdd, format="%y%m%d")
	end = None if end_date_yymmdd is None else pd.to_datetime(end_date_yymmdd, format="%y%m%d")

	frames: list[pd.DataFrame] = []
	missing_days: list[str] = []

	for i in range(max_days if end is None else 1000):
		day = start + pd.Timedelta(days=i)
		if end is not None and day > end:
			break

		zip_name = f"{station}@Y{day:%Y}_M{day:%m}_D{day:%d}.CSV.zip"
		csv_name = f"{station}@Y{day:%Y}_M{day:%m}_D{day:%d}.CSV"
		zip_path = Path(base_dir) / zip_name

		try:
			df_day = read_csv_from_zip(zip_path, csv_name, **read_csv_kwargs)
			frames.append(df_day)
		except FileNotFoundError:
			missing_days.append(f"{day:%Y-%m-%d}")
			if missing == "raise":
				raise
			if missing == "skip":
				continue
			# default: break
			break

	if not frames:
		raise FileNotFoundError(
			f"No data loaded starting at {start_date_yymmdd} from {base_dir}. Missing days: {missing_days[:10]}"
		)

	return pd.concat(frames, ignore_index=True)


def percent_nans(data, *, by_variable=True, exclude=None):
	"""Calculate percentage of NaNs in various container types.

	Parameters
	----------
	data : pandas.DataFrame
		Input data to inspect.
	by_variable : bool
		If True and `data` is a Dataset/DataFrame, return per-variable percentages
		in addition to the overall percentage.

	Returns
	-------
	dict
		Dictionary with keys:
		- 'overall': float percentage (0-100) of NaNs across the provided data
		- 'by_variable': dict mapping variable/column name -> percent NaNs (only when applicable)
	"""

	exclude_names = set(exclude) if exclude is not None else set()
	cols = [col for col in data.columns if col not in exclude_names]
	total = sum(data[col].size for col in cols)
	by_var = {col: 100.0 * data[col].isna().sum() / data[col].size if data[col].size else 0.0 for col in cols}
	overall = 100.0 * sum(data[col].isna().sum() for col in cols) / total
	return {'overall': overall, 'by_variable': by_var if by_variable else None}



def plot_height_series(ax, x_values, column_template: str, *, heights=[0,10], label_template: str, y_transform=None):
	for height in heights:
		series = df[column_template.format(height=height)]
		if y_transform is not None:
			series = y_transform(series, height)
		ax.plot(x_values, series, label=label_template.format(height=height))
#%%

path = r"D:\Lidar_IUP_20260608"
station="Wind10_1082" #switch to "Wind_1082" for 1s data
start_date = "260518"  # yymmdd
end_date = "260608"   # yymmdd


df = read_wind_ranger(
	start_date,
	base_dir=path,
	station=station,
	end_date_yymmdd=end_date,
	sep=",",
	skiprows=1,
)

# replace fill values
df = df.replace(9998.0, np.nan)
df = df.replace(9999.0, np.nan)
#%%
# Parse timestamp column (dd/mm/yyyy hh:mm:ss)
df["Time and Date"] = pd.to_datetime(
	df["Time and Date"],
	format="%d/%m/%Y %H:%M:%S",
	errors="coerce",
)


HEIGHT_LEVELS = [10, 24, 39, 59, 79, 109, 139, 174, 219, 279] #Wedel
#HEIGHT_LEVELS = [300, 270, 241, 212, 183, 154, 125, 96, 67, 38, 10] #test data

exclude = ['Reference','Time and Date','Timestamp (s)','Info. Flags','Status Flags','Battery (V)', 'Generator (V)','Upper Temp. (C)', 'Lower Temp. (C)', 'Pod Humidity (%)','GPS','Met Compass Bearing (deg)','Met Tilt (deg)','Met Air Temp. (C)','Met Pressure (mbar)','Met Humidity (%)','Met Wind Speed (m/s)','Met Wind Direction (deg)','Raining','Fog']
percentages = percent_nans(df, exclude=exclude)
percentages
# %%

plt.figure(figsize=(30, 6))
plot_height_series(
	plt.gca(),
	df["Time and Date"],
	"Horizontal Wind Speed (m/s) at {height}m",
	heights=HEIGHT_LEVELS,
	label_template="ws at {height}m",
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Horizontal Wind Speed (m/s)")
plt.title("Wind Speed at Different Heights")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=5, maxticks=30)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()

# %%
plt.figure(figsize=(30, 6))

plot_height_series(
	plt.gca(),
	df["Time and Date"],
	"Horizontal Wind Speed (m/s) at {height}m",
	heights=HEIGHT_LEVELS,
	label_template="ws at {height}m",
	y_transform=lambda series, height: series / df["Horizontal Wind Speed (m/s) at 10m"],
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Scale factor (m/s)")
plt.title("Wind Speed Normalized by 10m")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=5, maxticks=30)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()

# %%
#plot Wind Direction (deg) at at different heights
plt.figure(figsize=(30, 6))
plot_height_series(
	plt.gca(),
	df["Time and Date"],
	"Wind Direction (deg) at {height}m",
	heights=HEIGHT_LEVELS,
	label_template="wd at {height}m",
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Wind Direction (deg)")
plt.title("Wind Direction at Different Heights")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=5, maxticks=30)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()

# %%
