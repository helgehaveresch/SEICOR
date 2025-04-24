#%%
from datetime import datetime, timedelta
import pvlib
import pytz

def find_time_for_azimuth(latitude, longitude, date, target_azimuth, timezone='UTC', step_minutes=1):
    """
    Find the time at which the sun reaches a specified azimuth angle.

    Parameters:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        date (str): Date in the format 'YYYY-MM-DD'.
        target_azimuth (float): Desired solar azimuth angle in degrees.
        timezone (str): Timezone string (default: 'UTC').
        step_minutes (int): Time step in minutes for iteration (default: 1).

    Returns:
        str: Time at which the sun reaches the target azimuth, or None if not found.
    """
    local_tz = pytz.timezone(timezone)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    dt_start = local_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0))
    dt_end = dt_start + timedelta(days=1)
    
    current_dt = dt_start
    closest_time = None
    closest_diff = float('inf')
    
    while current_dt < dt_end:
        solpos = pvlib.solarposition.get_solarposition(current_dt, latitude, longitude)
        current_azimuth = solpos['azimuth'].values[0]
        
        diff = abs(current_azimuth - target_azimuth)
        if diff < closest_diff:
            closest_diff = diff
            closest_time = current_dt.strftime('%H:%M:%S')
        
        current_dt += timedelta(minutes=step_minutes)
    
    if closest_time:
        return datetime.strptime(closest_time, "%H:%M:%S")
    return None

def find_time_for_vza(latitude, longitude, date, target_vza, timezone='UTC', step_minutes=1):
    """
    Find the times at which the sun reaches a specified view zenith angle (VZA).

    Parameters:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        date (str): Date in the format 'YYYY-MM-DD'.
        target_vza (float): Desired solar view zenith angle (VZA) in degrees.
        timezone (str): Timezone string (default: 'UTC').
        step_minutes (int): Time step in minutes for iteration (default: 1).

    Returns:
        tuple: (morning time, evening time) when the sun reaches the target VZA.
    """
    local_tz = pytz.timezone(timezone)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    dt_start = local_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0))
    dt_end = dt_start + timedelta(days=1)
    
    current_dt = dt_start
    morning_time = None
    evening_time = None
    
    while current_dt < dt_end:
        solpos = pvlib.solarposition.get_solarposition(current_dt, latitude, longitude)
        vza = 90 - solpos['elevation'].values[0]  # Calculate VZA from elevation
        
        if abs(vza - target_vza) < 0.5:  # Allow small margin of error
            if morning_time is None:
                morning_time = current_dt.strftime('%H:%M:%S')
            else:
                evening_time = current_dt.strftime('%H:%M:%S')
        
        current_dt += timedelta(minutes=step_minutes)
    
    return (
        datetime.strptime(morning_time, "%H:%M:%S") if morning_time else None,
        datetime.strptime(evening_time, "%H:%M:%S") if evening_time else None,
    )


from datetime import datetime, timedelta
import pvlib
import pytz

def find_time_for_azimuth(latitude, longitude, date, target_azimuth, timezone='UTC', step_minutes=1):
    """
    Find the time at which the sun reaches a specified azimuth angle.

    Parameters:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        date (str): Date in the format 'YYYY-MM-DD'.
        target_azimuth (float): Desired solar azimuth angle in degrees.
        timezone (str): Timezone string (default: 'UTC').
        step_minutes (int): Time step in minutes for iteration (default: 1).

    Returns:
        str: Time at which the sun reaches the target azimuth, or None if not found.
    """
    local_tz = pytz.timezone(timezone)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    dt_start = local_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0))
    dt_end = dt_start + timedelta(days=1)
    
    current_dt = dt_start
    closest_time = None
    closest_diff = float('inf')
    
    while current_dt < dt_end:
        solpos = pvlib.solarposition.get_solarposition(current_dt, latitude, longitude)
        current_azimuth = solpos['azimuth'].values[0]
        
        diff = abs(current_azimuth - target_azimuth)
        if diff < closest_diff:
            closest_diff = diff
            closest_time = current_dt.strftime('%H:%M:%S')
        
        current_dt += timedelta(minutes=step_minutes)
    
    if closest_time:
        return datetime.strptime(closest_time, "%H:%M:%S")
    return None

def find_time_for_vza(latitude, longitude, date, target_vza, timezone='UTC', step_minutes=1):
    """
    Find the times at which the sun reaches a specified view zenith angle (VZA).

    Parameters:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        date (str): Date in the format 'YYYY-MM-DD'.
        target_vza (float): Desired solar view zenith angle (VZA) in degrees.
        timezone (str): Timezone string (default: 'UTC').
        step_minutes (int): Time step in minutes for iteration (default: 1).

    Returns:
        tuple: (morning time, evening time) when the sun reaches the target VZA.
    """
    local_tz = pytz.timezone(timezone)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    dt_start = local_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0))
    dt_end = dt_start + timedelta(days=1)
    
    current_dt = dt_start
    morning_time = None
    evening_time = None
    
    while current_dt < dt_end:
        solpos = pvlib.solarposition.get_solarposition(current_dt, latitude, longitude)
        vza = 90 - solpos['elevation'].values[0]  # Calculate VZA from elevation
        
        if abs(vza - target_vza) < 0.5:  # Allow small margin of error
            if morning_time is None:
                morning_time = current_dt.strftime('%H:%M:%S')
            else:
                evening_time = current_dt.strftime('%H:%M:%S')
        
        current_dt += timedelta(minutes=step_minutes)
    
    return (
        datetime.strptime(morning_time, "%H:%M:%S") if morning_time else None,
        datetime.strptime(evening_time, "%H:%M:%S") if evening_time else None,
    )


latitude = 53.08475580
longitude = 8.82082790
vza_start_end = 96  # Target View Zenith Angle
azimuth_target = 295  # Looking for when the sun is directly south
date = "2025-03-31"
# Define the start and end times
noon_start = find_time_for_azimuth(latitude, longitude, date, azimuth_target-5)
noon_end = find_time_for_azimuth(latitude, longitude, date, azimuth_target+5)
start_time, end_time = find_time_for_vza(latitude, longitude, date, vza_start_end)
time_step = timedelta(seconds=12)  # Increment time by 12 seconds
special_interval = timedelta(minutes=30)  # Every 30 minutes
special_duration = timedelta(minutes=1)  # Special line duration (1 min)

# File content header
header = """**********************************************************
* SEICOR IMPACT_ISO time file from 31.03.
**********************************************************
* dark measurement
**********************************************************
00:30:00   90    180   N
**********************************************************
* Ship measurement
**********************************************************
"""

# Footer block (copy data to server)
footer = """**********************************************************
* copy data to server
**********************************************************
22:30:00   15.9    180    B
"""

# Generate time entries
entries = []
current_time = start_time
next_special_time = start_time  # Initialize first special event

while current_time <= end_time:
    if current_time == next_special_time:
        # Insert the special line for 1 minute (5 times for 12-second steps)
        for _ in range(5):
            entries.append(f"{current_time.strftime('%H:%M:%S')}   90    270    _")
            current_time += time_step  # Move forward by 12 seconds
        
        # Schedule next special event
        next_special_time += special_interval

    # Check if current time falls within the noon window
    if noon_start <= current_time < noon_end:
        vza_value = azimuth_target+10  # Modify viewing azimuth angle for 10 minutes
    else:
        vza_value = azimuth_target  # Default value

    # Normal ship measurement line
    entries.append(f"{current_time.strftime('%H:%M:%S')}   15.9    {vza_value}    S")
    current_time += time_step  # Move forward by 12 seconds

# Combine all lines with header and footer
content = header + "\n".join(entries) + "\n" + footer + "\n"

# Write to an .asc file
#with open(r"C:\Users\hhave\Desktop\amaxoma_times_SEICOR_IMPACT_ISO_v2.asc", "w") as file:
#    file.write(content)

print("File 'output.asc' has been created successfully.")


# %%
find_time_for_azimuth(latitude, longitude, "2025-03-12" , 170+5.0)

# %%
