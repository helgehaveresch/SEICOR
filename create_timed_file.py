from datetime import datetime, timedelta

# Define the start and end times
start_time = datetime.strptime("06:40:01", "%H:%M:%S")
end_time = datetime.strptime("19:00:00", "%H:%M:%S")
time_step = timedelta(seconds=1)  # Increment time by 1 second
special_interval = timedelta(minutes=30)  # Every 30 minutes
special_duration = timedelta(minutes=1)  # Special line duration (1 min)
noon_start = datetime.strptime("12:00:00", "%H:%M:%S")  # Start of noon window
noon_end = datetime.strptime("12:10:00", "%H:%M:%S")  # End of noon window

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
        # Insert the special line for 1 minute (60 times)
        for _ in range(60):
            entries.append(f"{current_time.strftime('%H:%M:%S')}   90    270    _")
            current_time += time_step  # Move forward by 1 second
        
        # Schedule next special event
        next_special_time += special_interval

    # Check if current time falls within the noon window
    if noon_start <= current_time < noon_end:
        vza_value = 175  # Modify viewing azimuth angle for 10 minutes
    else:
        vza_value = 180  # Default value

    # Normal ship measurement line
    entries.append(f"{current_time.strftime('%H:%M:%S')}   15.9    {vza_value}    S")
    current_time += time_step  # Move forward by 1 second

# Combine all lines with header and footer
content = header + "\n".join(entries) + "\n" + footer + "\n"

# Write to an .asc file
with open(r"C:\Users\hhave\Desktop\amaxoma_times_SEICOR_IMPACT_ISO_v1", "w") as file:
    file.write(content)

print("File 'output.asc' has been created successfully.")

#%%

import pvlib
import datetime
import pytz

def get_solar_azimuth(latitude, longitude, date_time, timezone='UTC'):
    """
    Calculate the solar azimuth angle given latitude, longitude, and datetime.

    Parameters:
        latitude (float): Latitude in decimal degrees (positive for North, negative for South).
        longitude (float): Longitude in decimal degrees (positive for East, negative for West).
        date_time (str): Date and time in the format 'YYYY-MM-DD HH:MM:SS'.
        timezone (str): Timezone string (default: 'UTC').

    Returns:
        float: Solar azimuth angle in degrees, where 0° is North.
    """
    # Convert date_time string to a localized datetime object
    local_tz = pytz.timezone(timezone)
    dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
    dt = local_tz.localize(dt)

    # Get solar position
    solpos = pvlib.solarposition.get_solarposition(dt, latitude, longitude)

    # Extract azimuth angle
    azimuth = solpos['azimuth'].values[0]

    return azimuth

# Example usage:
latitude = 40.7128  # New York City
longitude = -74.0060
date_time = "2025-03-08 12:00:00"
timezone = "America/New_York"

azimuth_angle = get_solar_azimuth(latitude, longitude, date_time, timezone)
print(f"Solar Azimuth Angle: {azimuth_angle:.2f}°")