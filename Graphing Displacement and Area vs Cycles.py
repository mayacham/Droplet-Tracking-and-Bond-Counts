# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:48:31 2025

@author: Kari
"""

import pandas as pd
import matplotlib.pyplot as plt
#**********************************should probably make a new csv file for different data sets I want
# Load your summary CSV
DisplacementAreaDF = pd.read_csv("D:/Maya/Dataframes/Displacement and Area/Displacement_and_Area.csv")

# Toggle between all trials and selected trials
use_all_trials = True  # Change to True to plot all trials

# Define the trials you want to include
included_trials = [
    '250717 Irregular Shape Start 50 Equal Droplets',
    '250717 Irregular Shape Start 50 Equal Droplets 2'
]


# Function to check if a trial should be included
def trial_is_included(trial_name):
    return use_all_trials or trial_name in included_trials


# --- Plot Average Displacement ---
plt.figure(figsize=(10, 6))



for trial_name, group in DisplacementAreaDF.groupby('trial name'):
    if not trial_is_included(trial_name):
        continue
    plt.plot(group['cycle number'], group['average displacement']/group['average displacement'].iloc[0], marker='o', label=trial_name)

plt.xlabel("Cycle")
plt.ylabel("Average Displacement")
plt.title("Average Displacement vs. Cycle Number")
plt.legend(title="Trial Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

# --- Plot Average Area ---
plt.figure(figsize=(10, 6))          # Start a new figure

for trial_name, group in DisplacementAreaDF.groupby('trial name'):
    if not trial_is_included(trial_name):
        continue
    plt.plot(group['cycle number'], group['average area']/group['average area'].iloc[0], marker='s', label=trial_name)

plt.xlabel("Cycle")
plt.ylabel("Average Area")
plt.title("Average Area vs. Cycle Number")
plt.legend(title="Trial Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

