import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Constants - define these outside of the main function
epsilon = 2.504e15  # 1/(ppm meter)
CL = 1000  # ppm meter
siz_factor = 0.15
# These indices might need to be adjusted after we look at the structure of the interpolated spectra
spectrum_start_index = 0
spectrum_end_index = -1

def read_h2s_spectrum(file_path):
    """Reads the H2S absorption spectrum from an Excel file and returns the wavelength and absorption arrays."""
    h2s_data = pd.read_excel(file_path)
    wl_h2s = h2s_data.iloc[:, 0].to_numpy()
    abs_h2s = np.abs(h2s_data.iloc[:, 1].to_numpy())
    return wl_h2s, abs_h2s

def read_experimental_data(directory_path):
    """Reads experimental spectra from Excel files in a directory and returns an array of spectra."""
    df = pd.read_excel(directory_path)
    # Assuming the first row after the headers contains the wavelengths
    wl_exp = df.columns[3:].astype(float)
    # Replace 'inf' values with NaN, interpolate NaNs linearly, then backfill and forward fill as needed
    spectra_exp = df.iloc[1:, 3:].replace([np.inf, -np.inf], np.nan)
    spectra_exp = spectra_exp.interpolate(method='linear', axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    return wl_exp.to_numpy(), spectra_exp.to_numpy()

def interpolate_spectra(wl_exp, spectra_exp, wl_h2s):
    """Interpolates the experimental spectra to match the wavelength points of the H2S spectrum."""
    interpolated_data = []
    for spectrum in spectra_exp:
        interp_spectra = np.interp(wl_h2s, wl_exp, spectrum)
        interpolated_data.append(interp_spectra)
    return np.array(interpolated_data)

def normalize_spectra(interpolated_data, wl_h2s):
    """Normalizes spectra based on minimum and maximum values in specific ranges."""
    normalized_data = []
    for spectra in interpolated_data:
        siz = int(siz_factor * len(wl_h2s))
        min_val = np.median(spectra[:siz])
        max_val = np.average(spectra[-siz:])
        normalized_spectra = (spectra - min_val) / (max_val - min_val)
        normalized_data.append(normalized_spectra)
    return np.array(normalized_data)

def estimate_concentration(normalized_data, wl_h2s, abs_h2s):
    """Estimates H2S concentration based on the normalized spectra."""
    conc_min = 0
    conc_max = 1200
    conc_res = 5
    conc_vec = np.linspace(conc_min, conc_max, int((conc_max - conc_min) / conc_res))
    concentrations = []
    for spectra in normalized_data:
        rms_vec = []
        for conc in conc_vec:
            t_sim = np.exp(-conc * epsilon * abs_h2s)[spectrum_start_index:spectrum_end_index]
            rms = np.linalg.norm(spectra - t_sim)
            rms_vec.append(rms)
        best_conc = conc_vec[np.argmin(rms_vec)]
        concentrations.append(best_conc)
    return concentrations

def plot_spectra(wl_h2s, spectra_list, labels):
    """Plots the given spectra with labels."""
    plt.figure(figsize=(12, 6))
    for i, spectra in enumerate(spectra_list):
        plt.plot(wl_h2s, spectra, label=f'Sample {labels[i]}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Experimental Spectra')
    plt.legend()
    plt.show()

def main():
    h2s_file_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\H2S.xlsx"
    data_directory = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas cells check\H2S 6000\05_03.xlsx"

    wl_h2s, abs_h2s = read_h2s_spectrum(h2s_file_path)
    wl_exp, spectra_exp = read_experimental_data(data_directory)

    # Align the H2S data to the experimental data's wavelength range
    # Find the indices where the H2S data overlaps with the experimental data
    wl_overlap_indices = np.where((wl_h2s >= min(wl_exp)) & (wl_h2s <= max(wl_exp)))[0]
    wl_h2s_aligned = wl_h2s[wl_overlap_indices]
    abs_h2s_aligned = abs_h2s[wl_overlap_indices]

    # Update the global indices for concentration estimation
    global spectrum_start_index, spectrum_end_index
    spectrum_start_index = 0  # Start index should remain 0 after trimming wl_h2s
    spectrum_end_index = len(wl_h2s_aligned)  # End index is the length of the aligned H2S data

    interpolated_data = interpolate_spectra(wl_exp, spectra_exp, wl_h2s_aligned)
    normalized_data = normalize_spectra(interpolated_data, wl_h2s_aligned)
    concentrations = estimate_concentration(normalized_data, wl_h2s_aligned, abs_h2s_aligned)

    # Get labels for plotting (sample names/numbers)
    labels = [f'{i+1}' for i in range(spectra_exp.shape[0])]

    # Plotting the first spectrum for simplicity; you can plot all if needed
    plot_spectra(wl_h2s_aligned, [normalized_data[0]], [labels[0]])

    print("Estimated H2S Concentrations:", concentrations)

if __name__ == "__main__":
    main()
