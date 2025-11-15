#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:39:57 2025

@author: sebinjohn
"""

import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from obspy.clients.fdsn import Client
import pandas as pd

# ---- Connect to IRIS ----
client = Client("IRIS")

# ---- Networks to query ----
networks = ["AK", "AV", "CN","AT"]

# ---- Time range ----
# You can adjust these if you want older or newer stations.
starttime = "2000-01-01"
endtime = "2025-12-31"

records = []

for net in networks:
    print(f"Fetching stations for network {net} ...")
    inv = client.get_stations(network=net,
                              starttime=starttime,
                              endtime=endtime,
                              level="station")
    for network in inv:
        for station in network:
            records.append({
                "Network": network.code,
                "Station": station.code,
                "Latitude": station.latitude,
                "Longitude": station.longitude,
                "Elevation_m": station.elevation,
                "Site": station.site.name if station.site.name else "",
            })

# ---- Convert to DataFrame ----
df = pd.DataFrame(records)


station_list_file = "/Users/sebinjohn/vel_proj/unique_stations.txt"
# ---- READ UNIQUE STATION LIST ----
with open(station_list_file, "r") as f:
    all_station_lines = f.readlines()

# Clean and normalize station list
all_station_list = [
    line.strip().split(".")[-1] for line in all_station_lines if line.strip()
]
# ---- EXTRACT STATION NAMES ----

missing_stations = sorted(
    [st for st in all_station_list]
)

print(f"\nTotal stations in text file: {len(all_station_list)}")
print(f"Stations found in STATION0.HYP: {len(existing_names)}")
print(f"Missing stations: {len(missing_stations)}")
print(missing_stations)




# ---- Helper to convert decimal degrees to deg/min + hemisphere ----
def dec_to_degmin(value, is_lat=True):
    hemi = "N" if (is_lat and value >= 0) else "S" if is_lat else "E" if value >= 0 else "W"
    value = abs(value)
    deg = int(value)
    minutes = (value - deg) * 60
    return deg, minutes, hemi

# ---- Generate HYP-format lines ----
missing_lines = []
skipped = []

for st in missing_stations:
    if len(st) <= 4:
        # Look up from IRIS data
        row = df.loc[df["Station"] == st]
        if not row.empty:
            lat = float(row["Latitude"].values[0])
            lon = float(row["Longitude"].values[0])
            elev = float(row["Elevation_m"].values[0])

            dlat, mlat, ns = dec_to_degmin(lat, True)
            dlon, mlon, ew = dec_to_degmin(lon, False)

            # Format:  "  A19K7012.24N16104.26W  24"
            line = f"  {st:<4}{dlat:02d}{mlat:05.2f}{ns}{dlon:03d}{mlon:05.2f}{ew}{elev:4.0f}"
            missing_lines.append(line)
        else:
            skipped.append(st)
    else:
        skipped.append(st)


out_file = "/Users/sebinjohn/vel_proj/missing_station_lines.txt"
with open(out_file, "w") as f:
    for line in missing_lines:
        f.write(line + "\n")