#%%
import requests
import pandas as pd

station = "HANSKALBSAND"
param = "W"

url = f"https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/{station}/{param}/measurements.json"
response = requests.get(url, params=param)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# nur letztes Jahr filtern
df = df[df["timestamp"].dt.year == 2025]

df.to_csv("elbe_hanskalbsand_wasserstand_2025.csv", index=False)

print(df.head())
# %%
