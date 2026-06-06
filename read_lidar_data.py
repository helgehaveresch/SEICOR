#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import xarray as xr
import os
import sys

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

		if inner_csv_name is None:
			csv_members = [m for m in members if m.lower().endswith(".csv")]
			if not csv_members:
				raise FileNotFoundError(
					f"No .csv files found in {zip_path}. Contents: {members}"
				)
			inner_csv_name = csv_members[0]
		else:
			# Allow passing just the base name even if the zip stores it in a folder
			if inner_csv_name not in members:
				matches = [m for m in members if Path(m).name == inner_csv_name]
				if len(matches) == 1:
					inner_csv_name = matches[0]
				elif len(matches) > 1:
					raise FileNotFoundError(
						f"Multiple matches for '{inner_csv_name}' in {zip_path}: {matches}"
					)
				else:
					raise FileNotFoundError(
						f"'{inner_csv_name}' not found in {zip_path}. Contents: {members}"
					)

		with zf.open(inner_csv_name, mode="r") as f:
			return pd.read_csv(f, **read_csv_kwargs)


def read_wind_range(
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
		Prefix in the file name (default: "Wind10_1082").
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

	for i in range(max_days if end is None else 10_000_000):
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
	data : numpy.ndarray | pandas.DataFrame | pandas.Series | xarray.DataArray | xarray.Dataset
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

	Examples
	--------
	>>> percent_nans(np.array([1.0, np.nan, 3.0]))
	{'overall': 33.33333333333333, 'by_variable': None}
	"""
	# numpy array
	if isinstance(data, np.ndarray):
		total = data.size
		if total == 0:
			return {'overall': 0.0, 'by_variable': None}
		nans = np.count_nonzero(np.isnan(data))
		return {'overall': 100.0 * nans / total, 'by_variable': None}

	# pandas
	if isinstance(data, (pd.DataFrame, pd.Series)):
		if isinstance(data, pd.Series):
			total = data.size
			nans = data.isna().sum()
			return {'overall': 100.0 * int(nans) / int(total) if total else 0.0, 'by_variable': None}
		# DataFrame: allow excluding columns from analysis
		exclude_names = set(exclude) if exclude is not None else set()
		cols = [col for col in data.columns if col not in exclude_names]
		if not cols:
			return {'overall': 0.0, 'by_variable': {} if by_variable else None}
		total = sum(data[col].size for col in cols)
		if total == 0:
			return {'overall': 0.0, 'by_variable': {col: 0.0 for col in cols} if by_variable else None}
		by_var = {col: 100.0 * data[col].isna().sum() / data[col].size if data[col].size else 0.0 for col in cols}
		overall = 100.0 * sum(data[col].isna().sum() for col in cols) / total
		return {'overall': overall, 'by_variable': by_var if by_variable else None}

	# xarray
	if isinstance(data, (xr.DataArray, xr.Dataset)):
		exclude_names = set(exclude) if exclude is not None else set()
		if isinstance(data, xr.DataArray):
			arr = data
			total = arr.size
			nans = np.count_nonzero(np.isnan(arr.values))
			return {'overall': 100.0 * nans / total if total else 0.0, 'by_variable': None}
		# Dataset: per-variable breakdown, allow excluding variables
		by_var = {}
		total_all = 0
		nans_all = 0
		for name, var in data.data_vars.items():
			if name in exclude_names:
				continue
			vals = var.values
			total = vals.size
			nans = int(np.count_nonzero(np.isnan(vals)))
			by_var[name] = 100.0 * nans / total if total else 0.0
			total_all += total
			nans_all += nans
		overall = 100.0 * nans_all / total_all if total_all else 0.0
		return {'overall': overall, 'by_variable': by_var if by_variable else None}

	# fallback: try to convert to numpy
	try:
		arr = np.asarray(data)
		total = arr.size
		if total == 0:
			return {'overall': 0.0, 'by_variable': None}
		nans = np.count_nonzero(np.isnan(arr))
		return {'overall': 100.0 * nans / total, 'by_variable': None}
	except Exception:
		raise TypeError(f"Unsupported data type for percent_nans: {type(data)}")


def plot_height_series(ax, x_values, column_template: str, *, heights=[0,10], label_template: str, y_transform=None):
	for height in heights:
		series = df[column_template.format(height=height)]
		if y_transform is not None:
			series = y_transform(series, height)
		ax.plot(x_values, series, label=label_template.format(height=height))


start_date = "250601"  # yymmdd
end_date = "250725"   # yymmdd 

df = read_wind_range(
	start_date,
	base_dir=r"D:\SEICOR\wind_lidar",
	station="Wind10_1082",
	end_date_yymmdd=end_date,
	sep=",",
	skiprows=1,
)

# If you need other parsing options, e.g. semicolon separator:
# df = read_csv_from_zip(zip_file, inner_name, sep=';')
df = df.replace(9998.0, np.nan)
df = df.replace(9999.0, np.nan)
#%%
# Parse timestamp column (dd/mm/yyyy hh:mm:ss)
df["Time and Date"] = pd.to_datetime(
	df["Time and Date"],
	format="%d/%m/%Y %H:%M:%S",
	errors="coerce",
)


HEIGHT_LEVELS = [11, 25, 40, 60, 80, 110, 140, 175, 220, 280] #Wedel
HEIGHT_LEVELS = [300, 270, 241, 212, 183, 154, 125, 96, 67, 38, 10] #test data

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
""""
# if the wind direction is changing by 180° +/- 5° between to measurements following each other, it is likely a 180° ambiguity. We can correct it by adding 180° to the second measurement.
def correct_wind_direction_ambiguity(wd_series: pd.Series, threshold: float = 10.0) -> pd.Series:
	wd_corrected = wd_series.copy()
	for i in range(1, len(wd_series)):
		current = wd_corrected.iloc[i]
		previous = wd_corrected.iloc[i-1]
		if pd.isna(current) or pd.isna(previous):
			continue
		# Use circular angular distance so wrap-around at 0/360 is handled correctly.
		diff = abs((current - previous + 180) % 360 - 180)
		if abs(diff - 180) < threshold:
			wd_corrected.iloc[i] = (current + 180) % 360
	return wd_corrected
# Apply the correction to each height
for height in HEIGHT_LEVELS:
	df[f"Wind Direction (deg) at {height}m"] = correct_wind_direction_ambiguity(df[f"Wind Direction (deg) at {height}m"])
# Re-plot the corrected wind direction
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
plt.title("Corrected Wind Direction at Different Heights")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()
"""
# %%
