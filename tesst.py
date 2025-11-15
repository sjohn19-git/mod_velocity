#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:06:18 2025

@author: sebinjohn
"""

import obspy

ev=obspy.read_events("/Users/sebinjohn/Downloads/us100076ty_cleaned.xml")



# Loop through all events, picks, and show phase hints
for event in ev:
    print(f"Event ID: {event.resource_id}")
    for pick in event.picks:
        print(pick.phase_hint,pick.waveform_id)
        
        
event
        
phase_hints = [pick.phase_hint for event in ev for pick in event.picks]
print(phase_hints)

from collections import Counter

phase_counts = Counter([pick.phase_hint for event in ev for pick in event.picks])
print(phase_counts)

# Loop through events and print their IDs
for event in ev:
    print(event.resource_id)
    ide=event.resource_id
event_id_to_find = ev[0].resource_id.id.split("/")[-1]  # for example, first event
print(f"\nSearching for Event ID: {event_id_to_find}")
   

cat = obspy.read_events("/Users/sebinjohn/vel_proj/vmodel_final_clean.xml")

found_event = None
for event in cat:
    if event_id_to_find in event.resource_id.id:
        found_event = event
        break


# Loop through all events, picks, and show phase hints
for event in ev:
    print(f"Event ID: {event.resource_id}")
    for pick in event.picks:
        print(pick.phase_hint,pick.waveform_id)
        