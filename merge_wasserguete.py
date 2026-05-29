#%%
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import pytz
#%%
# Path to the folder containing the CSV files
folder = r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese"
pattern = os.path.join(folder, "Wasserguete_2*.csv")

all_files = sorted(glob.glob(pattern))
dfs = []

for file in all_files:
    # Skip the first 6 lines (metadata), then read the data
    df = pd.read_csv(
        file,
        sep=';',
        skiprows=6,
        names=['Zeitstempel', 'Wert', 'empty'],
        usecols=[0, 1],
        encoding='utf-8',
        decimal=','
    )
    # Parse datetime
    df['Zeitstempel'] = pd.to_datetime(df['Zeitstempel'], format='%d.%m.%Y %H:%M')
    # Convert Wert to float (comma as decimal)
    df['Wert'] = df['Wert'].astype(str).str.replace(',', '.').astype(float)
    df['file'] = os.path.basename(file)  # Optional: keep track of source file
    dfs.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)
#rename Wert to T and Zeitstempel to time_LT
merged_df.rename(columns={'Wert': 'T', 'Zeitstempel': 'time_LT'}, inplace=True)

# Convert time from CET to UTC before any filling
cet = pytz.timezone('Europe/Berlin')
merged_df['time_LT'] = merged_df['time_LT'].dt.tz_localize(cet, ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC').dt.tz_localize(None)
# Rename time to time_UTC
df = merged_df.rename(columns={'time_LT': 'time_UTC'})

# Save to a new CSV (optional)
df.to_csv(r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese\Wasserguete_merged.csv", index=False, sep=';')

#%%

df = pd.read_csv(r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese\Wasserguete_merged.csv", sep=';', parse_dates=['time_UTC'])

df['hour'] = df['time_UTC'].dt.hour + df['time_UTC'].dt.minute / 60.0
diurnal_mean = df.groupby('hour')['T'].mean()

def get_diurnal_value(dt):
    hour = dt.hour + dt.minute / 60.0
    return np.interp(hour, diurnal_mean.index, diurnal_mean.values)

df = df.sort_values('time_UTC')
full_time = pd.date_range(df['time_UTC'].min(), df['time_UTC'].max(), freq='10min')
filled = pd.DataFrame({'time_UTC': full_time})
filled = filled.merge(df[['time_UTC', 'T']], on='time_UTC', how='left')

# --- Fill the ice gap with 0 only where T is NaN ---
ice_start = pd.Timestamp('2026-01-23 00:00')
ice_end = pd.Timestamp('2026-02-18 00:00')
mask_ice = (filled['time_UTC'] >= ice_start) & (filled['time_UTC'] < ice_end) & (filled['T'].isna())
filled.loc[mask_ice, 'T'] = 0.0

# --- Fill missing values 
not_ice = ~mask_ice
is_nan = filled['T'].isna() & not_ice
nan_groups = (is_nan != is_nan.shift()).cumsum()
for group, group_df in filled[is_nan].groupby(nan_groups):
    idx = group_df.index
    if len(idx) == 0:
        continue
    gap_start = idx[0] - 1 if idx[0] > 0 else idx[0]
    gap_end = idx[-1] + 1 if idx[-1] < len(filled)-1 else idx[-1]
    gap_len = (gap_end - gap_start - 1) * 10  # in minutes
    if gap_start < 0 or gap_end >= len(filled):
        continue
    t0 = filled.loc[gap_start, 'T']
    t1 = filled.loc[gap_end, 'T']
    if gap_len <= 240:  # <4h
        vals = np.linspace(t0, t1, len(idx)+2)[1:-1]
    else:
        vals = []
        for j, i in enumerate(idx):
            frac = (j+1)/(len(idx)+1)
            diurnal = get_diurnal_value(filled.loc[i, 'time_UTC'])
            lin = t0 + frac*(t1-t0)
            vals.append(diurnal + (lin - get_diurnal_value(filled.loc[gap_start, 'time_UTC'])))
    # Only fill where T is NaN (do not overwrite existing data)
    for i, v in zip(idx, vals):
        if pd.isna(filled.loc[i, 'T']):
            filled.loc[i, 'T'] = v

# --- Resample filled dataset to hourly resolution ---
filled_hourly = filled.set_index('time_UTC').resample('1h').mean().reset_index()
filled_hourly.to_csv(r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese\Wasserguete_merged_filled_hourly.csv", index=False, sep=';')
print("Hourly DataFrame shape:", filled_hourly.shape)
filled_hourly = pd.read_csv(r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese\Wasserguete_merged_filled_hourly.csv", sep=';', parse_dates=['time_UTC'])

# Optionally plot
plt.figure(figsize=(30, 5))
plt.plot(filled_hourly['time_UTC'], filled_hourly['T'], label='Hourly (mean)')
plt.xlabel('Time (UTC)')
plt.ylabel('Temperature (°C)')
plt.title('Water temperature filled, hourly')
plt.tight_layout()
plt.grid()
plt.savefig(r"Q:\BREDOM\SEICOR\Wasserguete_Blankenese\Wasserguete_merged_filled_hourly.png", dpi=600)
plt.show()


# %%
