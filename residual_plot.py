#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 16:14:58 2025

@author: sebinjohn
"""

import os
from obspy import read_events
from obspy import Catalog
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#cat = read_events("/Users/sebinjohn/Downloads/combined_events_AK_w_phasehints.xml")
cat = read_events("/Users/sebinjohn/vel_proj/vmodel_final_clean.xml")

###################
import pandas as pd
from obspy.clients.fdsn import Client

# === File to cache station metadata ===
cache_file = "/Users/sebinjohn/vel_proj/station_metadata.csv"

# === Check if cached file exists ===
if os.path.exists(cache_file):
    print("Loading cached station metadata ...")
    df = pd.read_csv(cache_file)
else:
    print("No cache found — fetching from IRIS ...")
    client = Client("IRIS")
    networks = ["AK", "AV", "CN", "AT"]
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

    df = pd.DataFrame(records)
    # Save for next time
    df.to_csv(cache_file, index=False)
    print(f"Saved station metadata to {cache_file}")
################

# === Load RES summary text file ===
import re


res_file = "/Users/sebinjohn/vel_proj/data/res_556.txt"
stations = []
res_vals = {f"RES{i if i>0 else ''}": [] for i in range(5)}
nobs_vals = {f"NobsRES{i if i>0 else ''}": [] for i in range(5)}

with open(res_file, "r") as f:
    for line in f:
        # Skip header or empty lines
        if line.strip().startswith("Stn") or not line.strip():
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        station = parts[1].strip()

        # Extract all patterns like -7.83( 53)
        matches = re.findall(r"([-+]?\d*\.\d+|\d+)\s*\(\s*(\d+)\)", line)

        # Expect 5 groups (RES, RES1, RES2, RES3, RES4)
        if len(matches) < 5:
            # pad with NaNs if incomplete
            matches += [(None, None)] * (5 - len(matches))

        stations.append(station)

        # Split into RES & Nobs arrays
        for i, (val, nobs) in enumerate(matches):
            key_r = f"RES{i if i>0 else ''}"
            key_n = f"NobsRES{i if i>0 else ''}"

            res_vals[key_r].append(float(val) if val is not None else None)
            nobs_vals[key_n].append(int(nobs) if nobs is not None else None)

# === Combine into one DataFrame ===
df_res = pd.DataFrame({"Stn": stations, **res_vals, **nobs_vals})

print("Loaded residuals for", len(df_res), "stations.")
print(df_res.head(10))

# === Merge residuals with station metadata ===
df_merged = pd.merge(df, df_res, left_on="Station", right_on="Stn", how="inner")
print("Merged stations:", len(df_merged))

# === Ensure unique station merge ===
df_unique = df.drop_duplicates(subset=["Station"], keep="first")
df_merged = pd.merge(df_unique, df_res, left_on="Station", right_on="Stn", how="inner")
print("Merged unique stations:", len(df_merged))




# === Map setup ===
proj = ccrs.LambertConformal(central_longitude=-150, central_latitude=63)
extent = [-185, -131, 50, 72]  # Alaska region

fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))
ax.set_extent(extent, crs=ccrs.PlateCarree())

# === Add map features ===
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')

# === Compute signed residuals ===
residuals = df_merged["RES"]
res_min = residuals.min()
res_max = residuals.max()
abs_max = max(abs(res_min), abs(res_max))  # symmetric colorbar
#res_max=7.85
# === Scatter stations (signed residuals, blue–red colormap) ===
sc = ax.scatter(
    df_merged["Longitude"],
    df_merged["Latitude"],
    c=residuals,
    cmap="RdBu_r",           # blue→white→red
    vmin=res_min,
    vmax=res_max,
    s=60,
    edgecolor="k",
    transform=ccrs.PlateCarree(),
    marker="^",
    zorder=4
)

# === Add colorbar ===
cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
cb.set_label("Station Residual (s)")
cb.set_ticks(np.linspace(res_min, res_max, 7))

# === Title & save ===
plt.title("VELEST Station Residuals", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/station_residual_map.png",
            dpi=300, bbox_inches="tight")
plt.show()
