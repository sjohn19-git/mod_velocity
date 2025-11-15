#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 04:34:38 2025

@author: sebinjohn
"""

import pandas as pd
import re
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import matplotlib.pyplot as plt


def load_catalog(file_path):
    """
    Parse VELEST-style output file with columns:
    eq, origin-time, latitude, longitude, depth, x, y, z, mag, ifxOBS
    """
    pattern = re.compile(
        r'^\s*(\d+)\s+'                 # eq id
        r'(\d{6})\s+'                   # date (YYMMDD)
        r'(\d{4})\s+'                   # time (HHMM)
        r'([\d\.]+)\s+'                 # origin-time or gap
        r'(\d+\.\d+)N\s+'               # latitude
        r'(\d+\.\d+)W\s+'               # longitude
        r'(-?\d+\.?\d*|\*+)\s*'         # depth (can be ******)
        r'(-?\d+\.?\d*|\*+)\s*'         # x
        r'(-?\d+\.?\d*|\*+)\s*'         # y
        r'(-?\d+\.?\d*|\*+)\s*'         # z
        r'(-?\d+\.?\d*|\*+)\s+'         # mag
        r'(\d+)\s+'                     # ifx flag
        r'(\d+)'                        # OBS count
    )
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                data.append(m.groups())
    
    cols = [
        "id", "YYMMDD", "HHMM", "SS.ss", "latN", "lonW", 
        "depth_km", "x_km", "y_km", "z_km", "mag", "ifx", "nobs"
    ]
    df = pd.DataFrame(data, columns=cols)

    # --- Clean numeric columns ---
    num_cols = ["depth_km", "x_km", "y_km", "z_km", "mag"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].replace("*******", np.nan), errors='coerce')
    
    df["Latitude"] = df["latN"].astype(float)
    df["Longitude"] = df["lonW"].astype(float)

    df = df.drop(columns=["latN", "lonW"])
        # --- Combine into datetime ---
    # Build string: "YYMMDDHHMMSS.ss"
    df["datetime_str"] = df["YYMMDD"].astype(str) + df["HHMM"].astype(str)
    df["datetime_str"] = df["datetime_str"] + df["SS.ss"].astype(str).str.zfill(5)
    
    # Convert to datetime — handle fractional seconds properly
    def parse_time(x):
        try:
            return pd.to_datetime(x, format="%y%m%d%H%M%S.%f", errors="coerce")
        except Exception:
            return pd.NaT
    
    df["OriginTime"] = df["datetime_str"].apply(parse_time)
    
    # Drop helper string
    df = df.drop(columns=["datetime_str"])
    return df

# --- Load both catalogs ---
orig_df = load_catalog("/Users/sebinjohn/vel_proj/data/loc_org.txt")
inv_df  = load_catalog("/Users/sebinjohn/vel_proj/data/loc_invers.txt")



print(orig_df.head())
print(inv_df.head())


# Merge based on event ID (or nearest lat/lon if needed)
merged = pd.merge(orig_df, inv_df, on="id", suffixes=("_orig", "_inv"))

merged["Longitude_orig"]= -merged["Longitude_orig"] 
merged["Longitude_inv"] = -merged["Longitude_inv"] 




plt.figure(figsize=(6,6))
plt.scatter(merged["depth_km_orig"], merged["depth_km_inv"], c='steelblue', alpha=0.7, edgecolor='k')
plt.plot([0, max(merged["depth_km_orig"].max(), merged["depth_km_inv"].max())],
         [0, max(merged["depth_km_orig"].max(), merged["depth_km_inv"].max())],
         'r--', label="1:1 line")
plt.xlabel("Original Depth (km)")
plt.ylabel("Inverted Depth (km)")
plt.title("Depth Comparison (Original vs Inverted)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/depth_change.png")
plt.show()

merged["ΔOriginTime_s"] = (merged["OriginTime_inv"] - merged["OriginTime_orig"]).dt.total_seconds()

# --- Plot difference vs. Original Origin Time ---
plt.figure(figsize=(8,5))
plt.scatter(merged["OriginTime_orig"], merged["ΔOriginTime_s"],
            c='royalblue', alpha=0.7, edgecolor='k')

plt.axhline(0, color='r', linestyle='--', label="No change")
plt.xlabel("Original Origin Time")
plt.ylabel("Δ Origin Time (s) [Inverted − Original]")
plt.title("Difference in Origin Time Between Inverted and Original Catalogs")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

# Format x-axis to show readable dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.savefig("/Users/sebinjohn/Downloads/results/origin_time_change.png")
plt.show()


def calc_shift(row):
    p1 = (row["Latitude_orig"], row["Longitude_orig"])
    p2 = (row["Latitude_inv"], row["Longitude_inv"])
    return geodesic(p1, p2).km

merged["ΔLoc_km"] = merged.apply(calc_shift, axis=1)

proj = ccrs.LambertConformal(central_longitude=-150, central_latitude=63)
extent = [-185, -130, 50, 72]

# === Plot 1: Original Locations ===
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')

ax.scatter(merged["Longitude_orig"], merged["Latitude_orig"],
           s=35, color="blue", edgecolor="white",
           transform=ccrs.PlateCarree(), label="Original")

plt.legend()
plt.title("Original Event Locations", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/original_location.png")
plt.show()

# === Plot 2: Inverted Locations ===
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')

ax.scatter(merged["Longitude_inv"], merged["Latitude_inv"],
           s=35, color="red", edgecolor="white",
           transform=ccrs.PlateCarree(), label="Inverted")

plt.legend()
plt.title("Inverted Event Locations", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/inversion_location.png")
plt.show()


import matplotlib.colors as mcolors

# === Plot 3: Original Locations Colored by Location Shift (ΔLoc_km, Linear Scale) ===
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')

# --- Linear normalization ---
vmin = merged["ΔLoc_km"].min()
vmax = merged["ΔLoc_km"].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# --- Scatter plot ---
sc = ax.scatter(
    merged["Longitude_orig"], merged["Latitude_orig"],
    c=merged["ΔLoc_km"], cmap="plasma", norm=norm,
    s=55, edgecolor="k", transform=ccrs.PlateCarree(), zorder=5
)

# --- Colorbar ---
cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
cb.set_label("Location Shift (km)")
cb.ax.tick_params(labelsize=9)

# --- Final polish ---
plt.title("Original Event Locations Colored by Shift Distance (km)", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/location_change.png")
plt.show()
