import os
from obspy import read_events
from obspy import Catalog
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import numpy as np


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



stations = set()

for event in cat:
    for pick in event.picks:
        if pick.waveform_id and pick.waveform_id.network_code and pick.waveform_id.station_code:
            net_sta = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
            stations.add(net_sta)
        else:
            print("spurious pick",pick)

# sort for consistency
stations_list = sorted(stations)

# write each network.station on a new line
with open("/Users/sebinjohn/Downloads/unique_stations.txt", "w") as f:
    for net_sta in stations_list:
        f.write(net_sta + "\n")

print(f"Number of unique network.station codes in picks: {len(stations_list)}")
print("Saved to /Users/sebinjohn/Downloads/unique_stations.txt")


def extract_stations(file_path):
    stations = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty or comment lines
            if not line or line.startswith("RESET") or line.startswith("!"):
                continue

            # Only take first 4 non-space characters (station code)
            station_code = line[:4].strip()
            if station_code:
                stations.append(station_code)
    return stations


file_path = "/Users/sebinjohn/vel_proj/STATION0.HYP"
accepted_stas = extract_stations(file_path)


# ---- Define stations to remove ----
remove_stas = [
    "BLKN", "CBB", "CLRS", "EDM", "EUNU", "FCC", "FNSB", "GRNB", "HG4B",
    "ILON", "JEDB", "KUKN", "LLLB", "OZB", "PGC", "POIN", "RES", "SCHQ",
    "ULM", "YKAB2", "YKAW1", "YKAW3"
]

# ---- Filter out unwanted stations ----
accepted_stas = [sta for sta in accepted_stas if sta not in remove_stas]



total_before = sum(len(event.picks) for event in cat)

catn = Catalog(events=cat)

for event in catn:
    # keep only picks from accepted stations
    event.picks = [
        pick for pick in event.picks
        if pick.waveform_id.station_code in accepted_stas
    ]

def pick_count(catn):
    # Count total number of picks after filtering
    total_after = sum(len(event.picks) for event in catn)
    
    # Calculate how many were removed
    removed = total_before - total_after
    
    print(f"Total picks before filtering: {total_before}")
    print(f"Total picks after filtering:  {total_after}")
    print(f"Picks removed:                {removed}")
    print(f"Percentage removed:           {removed / total_before * 100:.2f}%")
    

removed_count = 0

for event in catn:
    # Count how many picks have no phase_hint
    missing_before = sum(1 for pick in event.picks if pick.phase_hint is None)
    removed_count += missing_before

    # Keep only picks that have a phase_hint
    event.picks = [pick for pick in event.picks if pick.phase_hint is not None]

print(f"Total picks removed due to missing phase_hint: {removed_count}")

pick_count(catn)


# Clean invalid picks (no time)
for event in catn:
    event.picks = [pick for pick in event.picks if pick.time is not None]


pick_count(catn)


events_with_zero_picks = [i for i, event in enumerate(catn) if len(event.picks) == 0]

if events_with_zero_picks:
    print("\nEvents with zero picks (indices in catn):", events_with_zero_picks)
else:
    print("\nNo events with zero picks.")

# filter out events with zero picks
catn = Catalog(events=[event for event in catn if len(event.picks) > 0])
pick_count(catn)
print(f"Remaining events after removing events with zero picks: {len(catn)}")



events_over_180 = [i for i, event in enumerate(catn) if len(event.picks) > 180]
print(f"\nNumber of events with more than 180 picks: {len(events_over_180)}")
print("Indices of such events:", events_over_180)
catn = Catalog(events=[event for event in catn if len(event.picks) <= 180])
print(f"Remaining events in catn after removing events that had picks more than 180: {len(catn)}")


for event in catn:
    if len(event.focal_mechanisms) > 0:
        print(f"Removing {len(event.focal_mechanisms)} focal mechanisms from event")
        event.focal_mechanisms = []



removed_events = []
kept_events = []

for i, event in enumerate(catn):
    origin = event.preferred_origin()
    lon = origin.longitude
    # Remove events that are WEST of the 180° meridian (i.e., lon < 0 if we define 180°E as cutoff)
    if lon > 0:
        removed_events.append((i, lon))
    else:
        kept_events.append(event)

catn = Catalog(events=kept_events)

print(f"\nRemoved {len(removed_events)} events west of 180° meridian.")
print(f"Remaining events in catalog: {len(catn)}")

# Optional: list what was removed
for idx, lon in removed_events:
    print(f"  Event {idx} → longitude {lon:.2f}")

pick_count(catn)


sta_lon = {f"{row.Network}.{row.Station}": row.Longitude for _, row in df.iterrows()}

removed_phase_count = 0
removed_stas = set()
for event in catn:
    new_picks = []
    for pick in event.picks:
        net = pick.waveform_id.network_code
        sta = pick.waveform_id.station_code
        key = f"{net}.{sta}"
        if key in sta_lon:
            if sta_lon[key] > 0:
                removed_phase_count += 1
                removed_stas.add(key)
                continue  # skip this pick
        else:
            print(f"{key} no metadata")
            continue
        new_picks.append(pick)
    event.picks = new_picks            

print(f"\nRemoved {removed_phase_count} picks recorded at stations east of 180° meridian.")
print(f"Remaining events in catalog: {len(catn)}")
print(f"Unique stations removed: {len(removed_stas)}")

pick_count(catn)

#save to Nordic
catn.write(
    "/Users/sebinjohn/OrbStack/seiscomp-assess/home/sysop/velest/ev.nor",
    format="NORDIC"
)
catcnv=catn.copy()

from obspy import read_events

# Assuming catcnv already exists
# catcnv = catn.copy()

def clean_duplicate_picks(catalog):
    for event in catalog:
        picks = event.picks
        cleaned_picks = []
        # Dictionary to track earliest P and S for each station
        seen = {}

        for pick in sorted(picks, key=lambda p: p.time):  # sort by arrival time
            station = pick.waveform_id.station_code
            phase = pick.phase_hint.upper() if pick.phase_hint else ""

            # classify as P or S type
            if phase.startswith("P"):
                phase_type = "P"
            elif phase.startswith("S"):
                phase_type = "S"
            else:
                continue  # skip other types (e.g., Lg, Rg, etc.)

            key = (station, phase_type)

            # keep first (earliest) occurrence only
            if key not in seen:
                seen[key] = pick
                cleaned_picks.append(pick)

        # replace old picks
        event.picks = cleaned_picks

    return catalog


# Apply the cleaning
catcnv = clean_duplicate_picks(catcnv)
pick_count(catcnv)

events_with_zero_picks = [i for i, event in enumerate(catcnv) if len(event.picks) == 0]

if events_with_zero_picks:
    print("\nEvents with zero picks (indices in catn):", events_with_zero_picks)
else:
    print("\nNo events with zero picks.")


catcnv.write(
    "/Users/sebinjohn/OrbStack/seiscomp-assess/home/sysop/velest/ev.cnv",
    format="CNV"
)

####################

# Extract lon/lat for events
lons, lats = [], []
for event in catcnv:
    try:
        o = event.preferred_origin() or event.origins[0]
        lons.append(float(o.longitude))
        lats.append(float(o.latitude))
    except Exception:
        continue

# --- Map Setup (focused on Alaska) ---
proj = ccrs.LambertConformal(central_longitude=-150, central_latitude=63)
extent = [-185, -131, 50, 72]   # Focus on Alaska

fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))

ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')
ax.plot([180, 180], [extent[2], extent[3]], color='black', linewidth=1.2,
    linestyle='--', transform=ccrs.PlateCarree(), label="180° meridian")
# Scatter events
sc = ax.scatter(lons, lats, color='red', s=25, edgecolor='k', transform=ccrs.PlateCarree())
plt.title("Select Earthquakes in Alaska — Double-click to Close Polygon", fontsize=13)

selected_indices = []

def onselect(verts):
    """Callback: select earthquakes within drawn polygon"""
    global selected_indices
    xy = ax.projection.transform_points(ccrs.PlateCarree(), 
                                        np.array(lons), np.array(lats))
    points = np.column_stack((xy[:, 0], xy[:, 1]))
    polygon = Path(verts)
    mask = polygon.contains_points(points)
    selected_indices = [i for i, inside in enumerate(mask) if inside]

    print(f"Selected {len(selected_indices)} events inside polygon.")

    # Highlight selected points in blue
    if selected_indices:
        ax.scatter(np.array(lons)[selected_indices],
                   np.array(lats)[selected_indices],
                   color='blue', s=40, edgecolor='white',
                   transform=ccrs.PlateCarree(), zorder=4)
        plt.draw()

selector = PolygonSelector(ax, onselect, useblit=True, props=dict(color='orange', linewidth=2))
plt.show()



# --- Subset and save ---
if selected_indices:
    subset_events = [catcnv[i] for i in selected_indices]
    cat_sub = Catalog(events=subset_events)
    print(f"Subset contains {len(cat_sub)} events.")

    out_nor = "/Users/sebinjohn/OrbStack/seiscomp-assess/home/sysop/aleutian_velest/ev_sub.nor"
    out_cnv="/Users/sebinjohn/OrbStack/seiscomp-assess/home/sysop/aleutian_velest/ev_sub.cnv"
    cat_sub.write(out_nor, format="NORDIC")
    catcnv.write(out_cnv,format="CNV")

    print(f"Saved to {out_nor}")
else:
    print("No events selected.")

print(f"Remaining events in catalog: {len(catn)}")
pick_count(cat_sub)



###################
# --- Plot events + recording stations ---
###################
from obspy import UTCDateTime

if selected_indices:
    # --- Collect stations that recorded phases for selected events ---
    recorded_stations = set()
    for event in cat_sub:
        for pick in event.picks:
            wid = pick.waveform_id
            if wid and wid.station_code:
                recorded_stations.add(wid.station_code)
        
    rec_df = df[df["Station"].isin(recorded_stations)]
    # Filter station metadata from STATION0.HYP parsing
    stn_lons = rec_df["Longitude"].values
    stn_lats = rec_df["Latitude"].values
    stn_names = rec_df["Station"].values

    # --- Map setup ---
    proj = ccrs.LambertConformal(central_longitude=-150, central_latitude=63)
    extent = [-185, max(stn_lons), 50, 72]

    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(11, 8))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')

    # --- Plot events ---
    ax.scatter(lons, lats, color='lightcoral', s=25, edgecolor='k',
               transform=ccrs.PlateCarree(), label="All events", alpha=0.5)

    sel_lons = [lons[i] for i in selected_indices]
    sel_lats = [lats[i] for i in selected_indices]
    ax.plot([180, 180], [extent[2], extent[3]], color='black', linewidth=1.2,
        linestyle='--', transform=ccrs.PlateCarree(), label="180° meridian")
    # --- Plot stations ---
    ax.scatter(stn_lons, stn_lats, color='blue', s=45, edgecolor='white',
               transform=ccrs.PlateCarree(), zorder=5, label="Stations with picks",marker="^")
    
    ax.scatter(sel_lons, sel_lats, color='red', s=40, edgecolor='white',
               transform=ccrs.PlateCarree(), zorder=4, label="Selected events")

    # # Optional: label stations
    # for name, lon, lat in zip(stn_names, stn_lons, stn_lats):
    #     ax.text(lon + 0.25, lat + 0.15, name, fontsize=7,
    #             transform=ccrs.PlateCarree(), zorder=6)

    plt.legend(loc="lower left")
    plt.title("Selected Events and Recording Stations", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("No selected events to plot with stations.")


#########plot single event

###############################################
# --- Plot one event (by index) + recording stations
###############################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# --- Select event index ---
event_index = 779   # <-- change this to any valid index in cat_sub

if event_index < len(cat_sub):
    event = cat_sub[event_index]
    print(f"Plotting event index {event_index} with {len(event.picks)} picks")

    # --- Extract event origin coordinates ---
    try:
        o = event.preferred_origin() or event.origins[0]
        ev_lon, ev_lat = float(o.longitude), float(o.latitude)
        print(f"Event location: lon={ev_lon:.3f}, lat={ev_lat:.3f}")
    except Exception as e:
        print(f"Error getting event origin: {e}")
        ev_lon, ev_lat = None, None

    # --- Collect stations that recorded picks for this event ---
    recorded_stations = set()
    for pick in event.picks:
        wid = pick.waveform_id
        if wid and wid.station_code:
            recorded_stations.add(wid.station_code)
    
    rec_df = df[df["Station"].isin(recorded_stations)]
    # Filter station metadata from STATION0.HYP parsing
    stn_lons = rec_df["Longitude"].values
    stn_lats = rec_df["Latitude"].values
    stn_names = rec_df["Station"].values


    # --- Map setup ---
    proj = ccrs.LambertConformal(central_longitude=-150, central_latitude=63)
    extent = [-185, -130, 50, 72]

    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 8))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--')
    # --- Draw 180° meridian (dateline) ---
    ax.plot([180, 180], [extent[2], extent[3]], color='black', linewidth=1.2,
        linestyle='--', transform=ccrs.PlateCarree(), label="180° meridian")
    # --- Plot event location ---
    if ev_lon is not None and ev_lat is not None:
        ax.scatter(ev_lon, ev_lat, color='red', s=20, edgecolor='white',
                   transform=ccrs.PlateCarree(), zorder=5, label=f"Event #{event_index}")

    # --- Plot recording stations ---
    ax.scatter(stn_lons, stn_lats, marker='^', color='blue', s=50, edgecolors='white',
               transform=ccrs.PlateCarree(), zorder=6, label="Stations with picks")

    # Optional: label stations
    for name, lon, lat in zip(stn_names, stn_lons, stn_lats):
        ax.text(lon + 0.25, lat + 0.15, name, fontsize=7,
                transform=ccrs.PlateCarree(), zorder=7)

    plt.legend(loc="lower left")
    plt.title(f"Event #{event_index} and Stations with Picks", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

else:
    print(f"Invalid event index: {event_index}. Catalog has {len(cat_sub)} events.")



# event_id_rm=[214,294,539]   


# # First, get event IDs (resource_ids) for those indices
# event_ids_to_remove = [str(catn[i].resource_id) for i in event_id_rm if i < len(catn)]
# print("Event IDs to remove:")
# for eid in event_ids_to_remove:
#     print(eid)
# catn = Catalog(events=[event for i, event in enumerate(catn) if i not in event_id_rm])

# print(f"Remaining events after removing specified indices ({event_id_rm}): {len(catn)}")


#save to Nordic
catn.write(
    "/Users/sebinjohn/OrbStack/seiscomp-assess/home/sysop/velest/ev.nor",
    format="NORDIC"
)