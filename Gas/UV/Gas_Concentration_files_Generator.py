import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

epsilon = 2.504e15  # 1/(ppm meter)
CL = 1000  # ppm meter
# H2S
H2S = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Sulfur dioxide.csv")
H2S_e = H2S.to_numpy()
WL_H2S = H2S_e[:, 0]
H2S_abs = abs(H2S_e[:, 1])  # cm^2/mol

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.title("H2S")
plt.xlabel("Wavelength[nm]")
plt.ylabel("Transmission")

output_folder = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Data train\gasses with noise\Sulfur dioxide"
# Plot transmission for all ppm meter values from 0 to 1000
base_noise_level = 0.002
relative_noise_factor = 0.03  

for ppm_meter in range(10, 1011, 10):
    transmission = np.exp(-H2S_abs * epsilon * ppm_meter)
    relative_noise = np.random.uniform(-relative_noise_factor, relative_noise_factor, size=transmission.shape)
    # Ensure minimum noise level
    noise = relative_noise * transmission
    noise[noise > 0] = np.fmax(noise[noise > 0], base_noise_level)
    noise[noise < 0] = np.fmin(noise[noise < 0], -base_noise_level)
    noisy_transmission = transmission + noise
    
    
    
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({'Transmission': transmission})
    file_path = os.path.join(output_folder, f'Sulfur_dioxide_{ppm_meter}_ppm_meter_noisy.csv')
    df.to_csv(file_path, index=False)
# plt.plot(WL_H2S, noisy_transmission, label=f'{ppm_meter} ppm meter')
# plt.show()