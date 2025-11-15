#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:06:36 2025

@author: sebinjohn
"""
import numpy as np
import matplotlib.pyplot as plt
import re

float_re = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

def parse_velest_mod(filename):
    vp, d_vp = [], []
    vs, d_vs = [], []

    reading_p = False
    reading_s = False

    with open(filename, "r") as f:
        for line in f:

            # Detect section headers — do NOT continue, we want to parse this line also
            if "P-VELOCITY MODEL" in line.upper():
                reading_p = True
                reading_s = False
            elif "S-VELOCITY MODEL" in line.upper():
                reading_s = True
                reading_p = False

            # Extract numbers from this line
            nums = float_re.findall(line)

            # Must have ≥ 2 numbers to be valid (velocity + depth)
            if len(nums) < 2:
                continue

            vel = float(nums[0])
            depth = float(nums[1])

            # Add to correct arrays
            if reading_p:
                vp.append(vel)
                d_vp.append(depth)
            elif reading_s:
                vs.append(vel)
                d_vs.append(depth)

    return (
        np.array(d_vp), np.array(vp),
        np.array(d_vs), np.array(vs)
    )



# ------------------ MAIN -------------------------
input_file = "/Users/sebinjohn/vel_proj/data/model visualization/input4.mod"
output_file = "/Users/sebinjohn/vel_proj/data/model visualization/output4.mod"

d_in_vp, vp_in, d_in_vs, vs_in = parse_velest_mod(input_file)
d_out_vp, vp_out, d_out_vs, vs_out = parse_velest_mod(output_file)

print("INPUT VP (vel, depth):")
print(np.column_stack([vp_in, d_in_vp]))

print("\nINPUT VS (vel, depth):")
print(np.column_stack([vs_in, d_in_vs]))

print("\nOUTPUT VP (vel, depth):")
print(np.column_stack([vp_out, d_out_vp]))

print("\nOUTPUT VS (vel, depth):")
print(np.column_stack([vs_out, d_out_vs]))


# ----------- P VELOCITY MODEL (Figure 1) -----------
plt.figure(figsize=(6,9))

plt.plot(vp_in,  d_in_vp,  "o--", label="VP input")
plt.plot(vp_out, d_out_vp, "o-",  label="VP output")

plt.gca().invert_yaxis()
plt.xlabel("Velocity (km/s)", fontsize=14)
plt.ylabel("Depth (km)", fontsize=14)
plt.title("P-wave Velocity Model", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/Vp_model.png")
plt.show()


# ----------- S VELOCITY MODEL (Figure 2) -----------
plt.figure(figsize=(6,9))

plt.plot(vs_in,  d_in_vs,  "s--", label="VS input")
plt.plot(vs_out, d_out_vs, "s-",  label="VS output")

plt.gca().invert_yaxis()
plt.xlabel("Velocity (km/s)", fontsize=14)
plt.ylabel("Depth (km)", fontsize=14)
plt.title("S-wave Velocity Model", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/sebinjohn/Downloads/results/Vs_model.png")
plt.show()



print("----- INPUT MODEL VP/VS -----")
for i in range(len(vp_in)):
    ratio = vp_in[i] / vs_in[i]
    print(f"Layer {i+1:2d}:  VP = {vp_in[i]:5.2f}   VS = {vs_in[i]:5.2f}   VP/VS = {ratio:5.3f}")

print("\n----- OUTPUT MODEL VP/VS -----")
for i in range(len(vp_out)):
    ratio = vp_out[i] / vs_out[i]
    print(f"Layer {i+1:2d}:  VP = {vp_out[i]:5.2f}   VS = {vs_out[i]:5.2f}   VP/VS = {ratio:5.3f}")
