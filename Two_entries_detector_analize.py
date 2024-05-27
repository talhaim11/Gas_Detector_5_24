import pandas as pd
import numpy as np
import os

def analyze_peaks_and_ratios(file_path, output_folder):
    # Load the Excel file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Analyzing file: {file_path}")
    # Define the interval between pulses (in numbers) and the threshold
    interval = 500  # 500 numbers apart
    threshold = 0  # Threshold for detecting a signal

    # Initialize lists to store peaks and ratios
    peaks = {'Sensor S Signal': [], 'Sensor R1 Signal': [], 'Sensor R2 Signal': [], 'Sensor 4 Signal': []}
    ratios_s_R2 = []
    ratios_R1_4 = []
    ratios_R2_4 = [] 
    # Iterate through each interval to find peaks and calculate ratios
    for i in range(0, len(df), interval):
        # Find peaks for each sensor in the current interval
        current_peaks = {}
        for sensor in ['Sensor S Signal', 'Sensor R1 Signal', 'Sensor R2 Signal', 'Sensor 4 Signal']:
            pulse_data = df[sensor][i:i + interval]
            peak_value = pulse_data.max()
            if peak_value > threshold:
                current_peaks[sensor] = peak_value
            else:
                current_peaks[sensor] = "no signal in this pulse"
            peaks[sensor].append(current_peaks[sensor])

        # Calculate ratios for each interval
        if isinstance(current_peaks['Sensor S Signal'], (np.int64, float)) and isinstance(current_peaks['Sensor R2 Signal'], (np.int64, float)):
            ratios_s_R2.append(current_peaks['Sensor S Signal'] / current_peaks['Sensor R2 Signal'])
        else:
            ratios_s_R2.append("insufficient signal")

        if isinstance(current_peaks['Sensor R1 Signal'], (np.int64, float)) and isinstance(current_peaks['Sensor 4 Signal'], (np.int64, float)):
            ratios_R1_4.append(current_peaks['Sensor R1 Signal'] / current_peaks['Sensor 4 Signal'])
        else:
            ratios_R1_4.append("insufficient signal")
            
        if isinstance(current_peaks['Sensor R2 Signal'], (np.int64, float)) and isinstance(current_peaks['Sensor 4 Signal'], (np.int64, float)):
            ratios_R2_4.append(current_peaks['Sensor R2 Signal'] / current_peaks['Sensor 4 Signal'])
        else:
            ratios_R2_4.append("insufficient signal")

    # Convert the peaks dictionary to a DataFrame for easier analysis
    peaks_df = pd.DataFrame(peaks)

    # Convert 'no signal in this pulse' to NaN for mean calculation and add ratio columns
    for sensor in ['Sensor S Signal', 'Sensor R1 Signal', 'Sensor R2 Signal', 'Sensor 4 Signal']:
        peaks_df[sensor] = pd.to_numeric(peaks_df[sensor], errors='coerce')
    peaks_df['Ratio_Sensor S_Sensor R2'] = pd.to_numeric(ratios_s_R2, errors='coerce')
    peaks_df['Ratio_Sensor R1_Sensor 4'] = pd.to_numeric(ratios_R1_4, errors='coerce')
    peaks_df['Ratio_Sensor R2_Sensor 4'] = pd.to_numeric(ratios_R2_4, errors='coerce') 


    # Calculate means for the sensors and ratios
    avg_s = np.mean(peaks_df['Sensor S Signal'].dropna())
    avg_r1 = np.mean(peaks_df['Sensor R1 Signal'].dropna())
    avg_r2 = np.mean(peaks_df['Sensor R2 Signal'].dropna())
    avg_4 = np.mean(peaks_df['Sensor 4 Signal'].dropna())
    mean_s_R2 = np.mean(peaks_df['Ratio_Sensor S_Sensor R2'].dropna())
    mean_R1_4 = np.mean(peaks_df['Ratio_Sensor R1_Sensor 4'].dropna())
    mean_R2_4 = np.mean(peaks_df['Ratio_Sensor R2_Sensor 4'].dropna())

   # Add a new row at the end of the DataFrame to store the mean values
    new_row = {
        'Sensor S Signal': avg_s,
        'Sensor R1 Signal': avg_r1,
        'Sensor R2 Signal': avg_r2,
        'Sensor 4 Signal': avg_4,
        'AVG Sensor S Signal': avg_s,
        'AVG Sensor R1 Signal': avg_r1,
        'AVG Sensor R2 Signal': avg_r2,
        'AVG Sensor 4 Signal': avg_4,
        'S-R2 ratio mean': mean_s_R2,
        'R1-4 ratio mean': mean_R1_4,
        'R2-4 ratio mean': mean_R2_4
    }
    peaks_df = peaks_df._append(new_row, ignore_index=True)

    # Calculate the average ratios and add them to the DataFrame
    avg_ratio_s_R2 = avg_s / avg_r2 if avg_r2 != 0 else "undefined"
    avg_ratio_R1_4 = avg_r1 / avg_4 if avg_4 != 0 else "undefined"

    # Create empty columns for the average ratios
    peaks_df['AVG Ratio_Sensor S_Sensor R2'] = np.nan
    peaks_df['AVG Ratio_Sensor R1_Sensor 4'] = np.nan

    # Set the last row of the average ratio columns to the calculated values
    peaks_df.at[len(peaks_df) - 1, 'AVG Ratio_Sensor S_Sensor R2'] = avg_ratio_s_R2
    peaks_df.at[len(peaks_df) - 1, 'AVG Ratio_Sensor R1_Sensor 4'] = avg_ratio_R1_4
    # Save the peaks and ratios to a new CSV file in the output folder
    output_path = os.path.join(output_folder, 'analyzed_' + os.path.basename(file_path))
    peaks_df.to_csv(output_path, index=False)

    print(f'Peak analysis and ratio calculations complete. Results saved to {output_path}.')


# Specify the folder containing the files
folder_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Two entries\Modified\Angles\right to left - angles calculation - 17_03\2"

# Specify the folder where you want to save the new files
output_folder = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Two entries\Modified\Angles\right to left - angles calculation - 17_03\2\Analyzed"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the folder and filter for CSV files
all_files = os.listdir(folder_path)
csv_files = [file for file in all_files if file.endswith('.csv')]

# Analyze each CSV file
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    analyze_peaks_and_ratios(file_path, output_folder)
