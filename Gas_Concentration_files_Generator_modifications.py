
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the data
H2S = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\No_Gas_01.csv")
H2S_e = H2S.to_numpy()
WL_H2S = H2S_e[:, 0]
H2S_abs = abs(H2S_e[:, 1])  # cm^2/mol

# Set the output folder
output_folder = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Data train\No_Gas\with noise\No_Gas_01"

# Parameters for noise
base_noise_level = 0.000013
relative_noise_factor = 0.0013

# Generate files with varying transmission values and noise
num_files = 100
transmission_start = 0.1
transmission_end = 1.0
transmission_step = (transmission_end - transmission_start) / (num_files - 1)

for i in range(num_files):
    transmission_value = transmission_start + i * transmission_step
    transmission = np.full(WL_H2S.shape, transmission_value)
    
    # Add uniform noise
    relative_noise = np.random.uniform(-relative_noise_factor, relative_noise_factor, size=transmission.shape)
    noise = relative_noise * transmission
    noise[noise > 0] = np.fmax(noise[noise > 0], base_noise_level)
    noise[noise < 0] = np.fmin(noise[noise < 0], -base_noise_level)
    noisy_transmission = transmission + noise
    
    # Save to CSV
    df = pd.DataFrame({'Wavelength': WL_H2S, 'Transmission': noisy_transmission})
    file_name = f'Regular_{i+1:03d}.csv'
    df.to_csv(os.path.join(output_folder, file_name), index=False)