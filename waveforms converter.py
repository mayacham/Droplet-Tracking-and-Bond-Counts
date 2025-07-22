# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:14:27 2025

@author: Kari
"""

import numpy as np

# Replace with your actual file path
input_file = "D:/Maya/Codes/custom wave.raw"
output_file = "D:/Maya/Codes/wave.csv"

# Read and convert
data = np.fromfile(input_file, dtype=np.int16)
normalized = data / 32768.0
np.savetxt(output_file, normalized, delimiter=",")

print(f"Saved CSV to {output_file}")

