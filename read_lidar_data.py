#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import sys
#%%

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


def read_wind10_range(
	start_date_yymmdd: str,
	base_dir: str | os.PathLike = r"D:\SEICOR\wind_lidar",
	station: str = "Wind10_1082",
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


# --- Example: read a date range and append into one df ---
start_date = "250601"  # yymmdd
end_date = "250725"   # yymmdd (optional)

df = read_wind10_range(
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
# %%
plt.figure(figsize=(12, 6))
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 300m"], label="ws at 300m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 270m"], label="ws at 270m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 241m"], label="ws at 241m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 212m"], label="ws at 212m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 183m"], label="ws at 183m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 154m"], label="ws at 154m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 125m"], label="ws at 125m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 96m"], label="ws at 96m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 67m"], label="ws at 67m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 38m"], label="ws at 38m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 10m")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Horizontal Wind Speed (m/s)")
plt.title("Wind Speed at Different Heights")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()

# %%
plt.figure(figsize=(12, 6))
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 300m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 300m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 270m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 270m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 241m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 241m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 212m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 212m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 183m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 183m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 154m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 154m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 125m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 125m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 96m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 96m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 67m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 67m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 38m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 38m")
plt.plot(df["Time and Date"], df["Horizontal Wind Speed (m/s) at 10m"]/df["Horizontal Wind Speed (m/s) at 10m"], label="ws at 10m")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Scale factor (m/s)")
plt.title("Wind Speed Normalized by 10m")

# Show only a reasonable number of datetime ticks
ax = plt.gca()
locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.gcf().autofmt_xdate()

# %%
