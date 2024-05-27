import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_and_plot_peaks(file_path):
    # Load the Excel file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Analyzing file: {file_path}")

    # Define the interval between pulses (in numbers)
    interval = 500  # 500 numbers apart

    # Initialize lists to store peaks and ratios
    peaks = {'Sensor S Signal': [], 'Sensor R1 Signal': [], 'Sensor R2 Signal': [], 'Sensor 4 Signal': []}
    ratios_s_R2 = []
    ratios_R1_4 = []
    ratios_R2_4 = []

    # Iterate through each interval to find peaks
    for i in range(0, len(df), interval):
        for sensor in peaks.keys():
            pulse_data = df[sensor][i:i + interval]
            peak_value = pulse_data.max()
            peaks[sensor].append(peak_value)

        # Calculate ratios for each interval
        ratios_s_R2.append(peaks['Sensor S Signal'][-1] / peaks['Sensor R2 Signal'][-1])
        ratios_R1_4.append(peaks['Sensor R1 Signal'][-1] / peaks['Sensor 4 Signal'][-1])
        ratios_R2_4.append(peaks['Sensor R2 Signal'][-1] / peaks['Sensor 4 Signal'][-1])

    # Plot the peaks for each sensor
    plt.figure(1, figsize=(12, 8))
    for sensor, peak_values in peaks.items():
        plt.plot(peak_values, label=f'Peaks - {sensor}')
    plt.xlabel('Interval Number')
    plt.ylabel('Peak Value')
    plt.title('Peaks for Sensors')
    plt.legend()

    # Plot the peak ratios
    plt.figure(2, figsize=(12, 8))
    plt.plot(ratios_s_R2, label='Peak Ratio S/R2')
    plt.plot(ratios_R1_4, label='Peak Ratio R1/4')
    plt.plot(ratios_R2_4, label='Peak Ratio R2/4')
    plt.xlabel('Interval Number')
    plt.ylabel('Ratio Value')
    plt.title('Peak Ratios for Sensors')
    plt.legend()

    
    plt.show()

# Specify the folder containing the files
folder_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Two entries\Modified\hot fog"

# List all files in the folder and filter for CSV files
all_files = os.listdir(folder_path)
csv_files = [file for file in all_files if file.endswith('.csv')]

# Analyze and plot peaks for each CSV file
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    analyze_and_plot_peaks(file_path)
