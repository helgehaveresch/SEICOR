#%%
from datetime import datetime
from meteostat import Hourly
from pathlib import Path
import pandas as pd

# Set time period
start = datetime(2025, 3, 1)
end = datetime(2025, 10, 31)

data = Hourly('D2465', start, end)
data = data.fetch()


# %%
Path(r"Q:\BREDOM\SEICOR\weatherstations\York-Moorende").mkdir(parents=True, exist_ok=True)
data.to_csv(r"Q:\BREDOM\SEICOR\weatherstations\York-Moorende\weatherdata_hourly.csv")
# %%
from datetime import datetime
import meteostat as ms

# Define time period
start = datetime(2024, 1, 1)
end   = datetime(2024, 1, 7)

# Create a daily time series for station D2465
ts = ms.Daily("D2465", start, end)


# %%
import meteostat as ms


# %%
