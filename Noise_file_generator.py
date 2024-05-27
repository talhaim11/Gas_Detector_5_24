import pandas as pd
import numpy as np
import os

# Set the number of samples and files
num_samples = 311
num_files = 100

# Set the output directory
output_dir = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Data train\Noise"
os.makedirs(output_dir, exist_ok=True)

# Generate and save files for random noise
for i in range(num_files):
    random_noise = np.random.random(size=num_samples)
    df_random_noise = pd.DataFrame({'transmission': random_noise})
    df_random_noise.to_csv(os.path.join(output_dir, f'random_noise_{i+1}.csv'), index=False)

# Generate and save files for Gaussian noise
for i in range(num_files):
    mean = np.random.uniform(0, 1)  # Randomly choose a mean between -5 and 5
    std_dev = np.random.uniform(0.5, 2)  # Randomly choose a standard deviation between 0.5 and 2
    gaussian_noise = np.random.normal(mean, std_dev, size=num_samples)
    gaussian_noise=np.abs(gaussian_noise/(np.max(gaussian_noise)))
    df_gaussian_noise = pd.DataFrame({'transmission': gaussian_noise})
    df_gaussian_noise.to_csv(os.path.join(output_dir, f'gaussian_noise_{i+1}.csv'), index=False)

# Generate and save files for white noise
for i in range(num_files):
    scale = np.random.uniform(0.1, 0.5)  # Randomly choose a scale between 0.1 and 1
    white_noise = np.random.normal(0.5, scale, size=num_samples)
    white_noise=np.abs(white_noise/(np.max(white_noise)))
    df_white_noise = pd.DataFrame({'transmission': white_noise})
    df_white_noise.to_csv(os.path.join(output_dir, f'white_noise_{i+1}.csv'), index=False)
