import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Specify the folder containing the files
folder_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Two entries\Modified\Angles\right to left - angles calculation - 17_03\2\Analyzed"
# List all files in the folder and filter for CSV files
all_files = os.listdir(folder_path)
csv_files = [file for file in all_files if file.endswith('.csv')]

# Initialize lists to store the values
avg_s_sensor = []
avg_r1_sensor = []
avg_r2_sensor = []
avg_4_sensor = []
avg_ratio_s_r2 = []
avg_ratio_r1_4 = []
ratio_s_r2 = []
ratio_r1_4 = []
ratios_R2_4 = [] 

# Read each file and extract the values
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    avg_s_sensor.append(df['AVG Sensor S Signal'].iloc[-1])
    avg_r1_sensor.append(df['AVG Sensor R1 Signal'].iloc[-1])
    avg_r2_sensor.append(df['AVG Sensor R2 Signal'].iloc[-1])
    avg_4_sensor.append(df['AVG Sensor 4 Signal'].iloc[-1])
    avg_ratio_s_r2.append(df['AVG Ratio_Sensor S_Sensor R2'].iloc[-1])
    avg_ratio_r1_4.append(df['AVG Ratio_Sensor R1_Sensor 4'].iloc[-1])
    ratio_s_r2.append(df['S-R2 ratio mean'].iloc[-1])
    ratio_r1_4.append(df['R1-4 ratio mean'].iloc[-1])
    ratios_R2_4.append(df['R2-4 ratio mean'].iloc[-1])


# Plot the average values for each sensor
plt.figure(1, figsize=(10, 6))
plt.plot(avg_s_sensor/np.nanmax(avg_s_sensor), label='AVG Sensor S Signal')
plt.plot(avg_r1_sensor/np.nanmax(avg_r1_sensor), label='AVG Sensor R1 Signal')
plt.plot(avg_r2_sensor/np.nanmax(avg_r2_sensor), label='AVG Sensor R2 Signal')
plt.plot(avg_4_sensor/np.nanmax(avg_4_sensor), label='AVG Sensor 4 Signal')
plt.xlabel('File Number')
plt.ylabel('Average Signal')
plt.title('Average Sensor Signals Across Files')
plt.legend()


# Plot the ratios for each file
plt.figure(2, figsize=(10, 6))
plt.plot(ratio_s_r2, label='Ratio S-R2')
plt.plot(ratio_r1_4, label='Ratio R1-4')
plt.plot(ratios_R2_4, label='Ratio R2-4')
plt.xlabel('File Number')
plt.ylabel('Ratio')
plt.title('Ratios Across Files')
plt.legend()


# Plot the average ratios across files
plt.figure(3, figsize=(10, 6))
plt.plot(avg_ratio_s_r2, label='AVG Ratio S-R2')
plt.plot(avg_ratio_r1_4, label='AVG Ratio R1-4')
plt.xlabel('File Number')
plt.ylabel('Average Ratio')
plt.title('Average Ratios Across Files')
plt.legend()




plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Specify the folder containing the files
# folder_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Two entries\Modified\Angles\LeftRight\Analyzed"

# # List all files in the folder and filter for CSV files
# all_files = os.listdir(folder_path)
# csv_files = [file for file in all_files if file.startswith('analyzed_') and file.endswith('.csv')]

# # Initialize lists to store the values
# avg_s_sensor = []
# avg_r1_sensor = []
# avg_r2_sensor = []
# avg_4_sensor = []
# avg_ratio_s_r2 = []
# avg_ratio_r1_4 = []
# ratio_s_r2 = []
# ratio_r1_4 = []

# # Read each file and extract the values
# for file_name in csv_files:
#     file_path = os.path.join(folder_path, file_name)
#     df = pd.read_csv(file_path)
#     avg_s_sensor.append(df['AVG Sensor S Signal'].iloc[-1])
#     avg_r1_sensor.append(df['AVG Sensor R1 Signal'].iloc[-1])
#     avg_r2_sensor.append(df['AVG Sensor R2 Signal'].iloc[-1])
#     avg_4_sensor.append(df['AVG Sensor 4 Signal'].iloc[-1])
#     avg_ratio_s_r2.append(df['AVG Ratio_Sensor S_Sensor R2'].iloc[-1])
#     avg_ratio_r1_4.append(df['AVG Ratio_Sensor R1_Sensor 4'].iloc[-1])
#     ratio_s_r2.append(df['S-R2 ratio mean'].iloc[-1])
#     ratio_r1_4.append(df['R1-4 ratio mean'].iloc[-1])

# # Plot the average values for each sensor
# plt.figure(1, figsize=(10, 6))
# plt.plot(avg_s_sensor, label='AVG Sensor S Signal')
# plt.plot(avg_r1_sensor, label='AVG Sensor R1 Signal')
# plt.plot(avg_r2_sensor, label='AVG Sensor R2 Signal')
# plt.plot(avg_4_sensor, label='AVG Sensor 4 Signal')
# plt.xlabel('File Number')
# plt.ylabel('Average Signal')
# plt.title('Average Sensor Signals Across Files')
# plt.legend()

# # Plot the ratios for each file
# plt.figure(2, figsize=(10, 6))
# plt.plot(ratio_s_r2, label='Ratio S-R2')
# plt.plot(ratio_r1_4, label='Ratio R1-4')
# plt.xlabel('File Number')
# plt.ylabel('Ratio')
# plt.title('Ratios Across Files')
# plt.legend()

# # Plot the average ratios across files
# plt.figure(3, figsize=(10, 6))
# plt.plot(avg_ratio_s_r2, label='AVG Ratio S-R2')
# plt.plot(avg_ratio_r1_4, label='AVG Ratio R1-4')
# plt.xlabel('File Number')
# plt.ylabel('Average Ratio')
# plt.title('Average Ratios Across Files')
# plt.legend()

# plt.show()
