import ctypes
from ctypes import *
import os
import math
import matplotlib.pyplot as plt
import time
import keyboard
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models, callbacks
from keras.callbacks import EarlyStopping
from datetime import datetime




# Definitions
STRING = c_char_p
DWORD = c_ulong
ULONG = c_ulong
WORD = c_ushort
USHORT = c_ushort
SHORT = c_short
UCHAR = c_ubyte
BYTE = c_byte
UINT = c_uint
LPBYTE = POINTER(c_ubyte)
CHAR = c_char
LPBOOL = POINTER(c_int)
PUCHAR = POINTER(c_ubyte)
PCHAR = STRING
PVOID = c_void_p
INT = c_int
LPTSTR = STRING
LPDWORD = POINTER(DWORD)
LPWORD = POINTER(WORD)
PULONG = POINTER(ULONG)
LPVOID = PVOID
VOID = None
ULONGLONG = c_ulonglong
HANDLE = c_void_p
BOOL = c_bool

# Handle and status
FT_STATUS = ULONG
FT_HANDLE = PVOID
FT4222_STATUS = ULONG

ftstatus = FT_STATUS(0)
numOfDevices = ULONG(0)

# Load libraries FT4222 and ftd2xx
ftd2xx = windll.LoadLibrary("ftd2xx.dll")
FT4222 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + "/LibFT4222-64.dll")

# Define functions of ftd2xx
ftd2xx.FT_CreateDeviceInfoList.restype = FT_STATUS
ftd2xx.FT_CreateDeviceInfoList.argtypes = [LPDWORD]

ftd2xx.FT_GetDeviceInfoDetail.restype = FT_STATUS
ftd2xx.FT_GetDeviceInfoDetail.argtypes = [DWORD, LPDWORD, LPDWORD, LPDWORD, LPDWORD, LPVOID, LPVOID, LPVOID]

ftd2xx.FT_OpenEx.restype = FT_STATUS
ftd2xx.FT_OpenEx.argtypes = [STRING, INT, HANDLE]

ftd2xx.FT_SetLatencyTimer.restype = FT_STATUS
ftd2xx.FT_SetLatencyTimer.argtypes = [HANDLE, BYTE]

ftd2xx.FT_SetUSBParameters.restype = FT_STATUS
ftd2xx.FT_SetUSBParameters.argtypes = [HANDLE, ULONG, ULONG]

ftd2xx.FT_Close.restype = FT_STATUS
ftd2xx.FT_Close.argtypes = [HANDLE]

# FT4222_SetClock
FT4222.FT4222_SetClock.restype = FT4222_STATUS
FT4222.FT4222_SetClock.argtypes = [HANDLE, INT]

# FT4222_SPIMaster_Init
FT4222.FT4222_SPIMaster_Init.restype = FT4222_STATUS
FT4222.FT4222_SPIMaster_Init.argtypes = [HANDLE, INT, INT, INT, INT, BYTE]

# FT4222_SPIMaster_SingleReadWrite
FT4222.FT4222_SPIMaster_SingleReadWrite.restype = FT4222_STATUS


# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________



################FUNCTIONS##################
# ----------------------------------------#





# Convert returned byte array to a decimal value
def bytes2decimal(Array):
    if len(Array) == 3:
        decimal = Array[1] * 256 + Array[2]
        return decimal
    else:
        print("Wrong array length")

# Read a register value
def ReadRegister(Handle, RegisterValue):
    # Appending address to fit with DISB format.
    AppendedRegAddres = RegisterValue * 4 + 2

    # Initializing different inputs for ReadWrite function
    Length = ctypes.c_ushort(3)
    TransferSize = ctypes.c_ushort()
    TRUE = ctypes.c_bool(1)
    readbuf = (c_ubyte * 3)()
    writebuf = (c_ubyte * 3)()
    # Create a writebuffer, requesting data read from chosen address
    writebuf[0] = AppendedRegAddres
    writebuf[1] = 0
    writebuf[2] = 0
    # Read DATA
    ft4222Status = FT4222.FT4222_SPIMaster_SingleReadWrite(Handle, ctypes.byref(readbuf), ctypes.byref(writebuf), Length, ctypes.byref(TransferSize), TRUE)
    # Convert the 2 byte value to single decimal value if Read successful. Otherwise print error message
    if ft4222Status == 0:
        DecimalValue = bytes2decimal(readbuf)
        return DecimalValue
    else:
        print("Something went wrong during the transfer process", ft4222Status)

# Set a new register value
def SetRegister(Handle, RegisterValue, NewRegValue):
    # Appending address to fit with DISB format.
    AppendedRegAddres = RegisterValue * 4

    # Initializing different inputs for ReadWrite function
    Length = ctypes.c_ushort(3)
    TransferSize = ctypes.c_ushort()
    TRUE = ctypes.c_bool(1)
    readbuf = (c_ubyte * 3)()
    writebuf = (c_ubyte * 3)()
    # Create a writebuffer, choosing the correct register and transferring data
    writebuf[0] = AppendedRegAddres
    writebuf[1] = 0
    writebuf[2] = 0
    # If the new value cannot be contained within a single uByte, split it into two bytes.
    if NewRegValue > 255:
        writebuf[1] = math.floor(NewRegValue / 256)
        writebuf[2] = NewRegValue - 256 * math.floor(NewRegValue / 255)
    else:
        writebuf[1] = 0
        writebuf[2] = NewRegValue

    # Write the new register value to the chosen register.
    ft4222Status = FT4222.FT4222_SPIMaster_SingleReadWrite(Handle, ctypes.byref(readbuf), ctypes.byref(writebuf), Length, ctypes.byref(TransferSize), TRUE)
    # Check if the register was successfully set.
    if ft4222Status != 0:
        print("Something went wrong during the transfer process", ft4222Status)

# Get a spectrum from the spectrometer
def GetSpectrum(HandleREG, HandleStream):
    # Perform soft reset of image buffer
    SetRegister(HandleREG, 8, 16)
    # Trigger an exposure
    SetRegister(HandleREG, 8, 1)

    # Readout value of first and last pixel to determine framebuffer length
    FirstPixel = ReadRegister(HandleREG, 23)
    LastPixel = ReadRegister(HandleREG, 24)

    # Initializing different inputs for ReadWrite function
    Datalength = (LastPixel - FirstPixel + 1) * 2
    readbuf = (c_ubyte * Datalength)()
    writebuf = (c_ubyte * Datalength)()
    Length = ctypes.c_ushort(Datalength)
    TransferSize = ctypes.c_ushort()
    TRUE = ctypes.c_bool(1)
    # Create an empty array for the spectrum to be written to.
    Spectrum = []

    # Check if the framebuffer is full and spectrum is ready to be read.
    NumberOfPixReady = 0
    while (LastPixel - FirstPixel) > NumberOfPixReady:
        NumberOfPixReady = ReadRegister(HandleREG, 12)

    # Read spectrum
    ft4222Status = FT4222.FT4222_SPIMaster_SingleReadWrite(HandleStream, ctypes.byref(readbuf), ctypes.byref(writebuf), Length, ctypes.byref(TransferSize), TRUE)

    # If successful combine the ubyte values to 16 bit values (65535). Else display error message.
    if ft4222Status == 0:
        for i in range(0, Datalength, 2):
            Spectrum.append(readbuf[i] * 256 + readbuf[i + 1])
        return Spectrum
    else:
        print("Something went wrong during the transfer process", ft4222Status)

# Device class
def list_devices():
    numOfDevices = ctypes.c_ulong()
    ftd2xx.FT_CreateDeviceInfoList(ctypes.byref(numOfDevices))

    flags = ctypes.c_ulong()
    typ = ctypes.c_ulong()
    ID = ctypes.c_ulong()
    LocID = ctypes.c_ulong()
    serial = ctypes.create_string_buffer(20)
    desc = ctypes.create_string_buffer(64)
    handle = ctypes.c_void_p()

    ret = []
    for i in range(numOfDevices.value):
        ftStatus = ftd2xx.FT_GetDeviceInfoDetail(i, ctypes.byref(flags), ctypes.byref(typ),
                                                 ctypes.byref(ID), ctypes.byref(LocID), serial,
                                                 desc, ctypes.byref(handle))
        ret.append((i, flags.value, typ.value,
                    ID.value, LocID.value,
                    serial.value.decode("utf-8"), desc.value.decode("utf-8")))

    return ret




# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________


##############IBSEN PROGRAM################
# ----------------------------------------#


# Check for available device and collect information on them.
ret = list_devices()

# Set up 3 handles used for register R/W, bulk reads and GPIO
ftHandleOutStream = ctypes.c_void_p()
ftHandleOutCom = ctypes.c_void_p()
ftHandleOutGPIO = ctypes.c_void_p()

SerialNumberA = STRING("A".encode('utf-8'))
SerialNumberB = STRING("B".encode('utf-8'))
SerialNumberC = STRING("C".encode('utf-8'))
# Open 3 devices with their associated handle based on serial number.
OPEN_BY_SERIAL = INT(1)
ftStatus = ftd2xx.FT_OpenEx(SerialNumberA, OPEN_BY_SERIAL, ctypes.byref(ftHandleOutStream))
print("A ", ftStatus)
ftStatus = ftd2xx.FT_OpenEx(SerialNumberB, OPEN_BY_SERIAL, ctypes.byref(ftHandleOutCom))
print("B ", ftStatus)
ftStatus = ftd2xx.FT_OpenEx(SerialNumberC, OPEN_BY_SERIAL, ctypes.byref(ftHandleOutGPIO))
print("C ", ftStatus)

# Set latency and USB parameter for data transfer
UcLatency = BYTE(20)
ftStatus = ftd2xx.FT_SetLatencyTimer(ftHandleOutStream, UcLatency)
print("FT_SetLatencyTimer ", ftStatus)

OutTransferSize = ULONG(4096)
InTransferSize = ULONG(65536)
ftStatus = ftd2xx.FT_SetUSBParameters(ftHandleOutStream, InTransferSize, OutTransferSize)
print("FT_SetUSBParameters ", ftStatus)

# Init MASTER SPI
# Initialize a SPI connection with the R/W and bulk read handles.
ft4222Status = FT4222.FT4222_SetClock(ftHandleOutStream, 0)

ft4222Status = FT4222.FT4222_SPIMaster_Init(ftHandleOutCom, 1, 2, 0, 1, 2)
ft4222Status = FT4222.FT4222_SPIMaster_Init(ftHandleOutStream, 1, 2, 0, 1, 2)

# READ FROM DISB REGISTER
# Read different register values using the ReadRegister function
PCB_SN = ReadRegister(ftHandleOutCom, 1)
print("\nPCB SERIAL NUMBER ", PCB_SN)

HardwareVersion = ReadRegister(ftHandleOutCom, 2)
print("Hardware Version ", HardwareVersion)

FirmwareVersion = ReadRegister(ftHandleOutCom, 3)
print("Firmware Version ", FirmwareVersion)

DetectorType = ReadRegister(ftHandleOutCom, 4)
print("Detector Type ", DetectorType)

PixelPerImage = ReadRegister(ftHandleOutCom, 5)
print("Pixels per image ", PixelPerImage)

NumberofCaliChars = ReadRegister(ftHandleOutCom, 6)
print("Number of Calibration Chars ", NumberofCaliChars)

print("Wavelength Calibration values")
for i in range((int)(NumberofCaliChars / 14)):
    combinedCaliChar = []
    for j in range(14):
        caliChar = ReadRegister(ftHandleOutCom, 7)
        combinedCaliChar.append(chr(caliChar))
    CalibrationValue = ''.join(map(str, combinedCaliChar))
    print(CalibrationValue)

# SET DISB REGISTER
# Set integrations times of the DISB registers, using the SetRegister function.
# Set to 10 ms. = 50.000 * 200 ns = 512ms
SetRegister(ftHandleOutCom, 10, 38)
SetRegister(ftHandleOutCom, 9, 50000)

# Read registers to confirm values were set correctly
IntMSB = ReadRegister(ftHandleOutCom, 10)
print("Integration time IntMSB ", IntMSB)
IntLSB = ReadRegister(ftHandleOutCom, 9)
print("Integration time IntMSB ", IntLSB)

# Set up the plot for live data streaming
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel('wavelength')
ax.set_ylabel('Intensity')
ax.set_title('Live Spectrum')




# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________





##############FGD FUNCTIONS################
# ----------------------------------------#

# Load gas reference data
def load_reference_data():
    df_h2s = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\H2S.csv")
    df_ammonia = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Ammonia.csv")
    df_benzene = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Benzene.csv")
    df_toluene = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Toluene.csv")
    df_xylene = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Xylene.csv")
    df_sulfur_dioxide = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Interpolated data\Sulfur Dioxide.csv")

    # Organize the reference data in a dictionary for easy access
    reference_data = {
        'H2S': {
            'wavelengths': df_h2s['Wavelength'].values,
            'Transmission': df_h2s['Transmission'].values
        },
        'Ammonia': {
            'wavelengths': df_ammonia['Wavelength'].values,
            'Transmission': df_ammonia['Transmission'].values
        },
        'Benzene': {
            'wavelengths': df_benzene['Wavelength'].values,
            'Transmission': df_benzene['Transmission'].values
        },
        'Toluene': {
            'wavelengths': df_toluene['Wavelength'].values,
            'Transmission': df_toluene['Transmission'].values
        },
        'Xylene': {
            'wavelengths': df_xylene['Wavelength'].values,
            'Transmission': df_xylene['Transmission'].values
        },
        'Sulfur Dioxide': {
            'wavelengths': df_sulfur_dioxide['Wavelength'].values,
            'Transmission': df_sulfur_dioxide['Transmission'].values
        }
    }
    return reference_data

def update_plot(raw_data):
    global reference_data_mode, dark_mode


    # Ensure the X_axis and raw_data are the same length
    if len(X_axis) != len(raw_data):
        print("Warning: X_axis and raw_data lengths do not match. Truncating to the shortest length.")
        min_length = min(len(X_axis), len(raw_data))
        X_axis_truncated = X_axis[:min_length]
        raw_data_truncated = raw_data[:min_length]
    else:
        X_axis_truncated = X_axis
        raw_data_truncated = raw_data

    if reference_data_mode is not None and dark_mode is not None:

        transmission_data = [((raw - dark) / (ref - dark)) if (ref-dark) != 0 else 0 for raw, dark, ref in zip(raw_data, dark_mode, reference_data_mode)]
        
        siz_factor=0.2
        siz = int(siz_factor * len(transmission_data[start:end]))

        max_val=np.median(np.sort(transmission_data[start:end])[-siz:])

        min_val=np.median(np.sort(transmission_data[start:end])[0:siz])

        # min_val = np.median(transmission_data[start:end][:siz])
        # max_val = np.median(transmission_data[start:end][-siz:])
        print(max_val)
        print(min_val)
        # if abs(min_val-max_val)>0.1:

        #     transmission_data = (transmission_data - min_val) / (max_val - min_val)
        
        
        predicted_gas=predict_gas(transmission_data)
        text=LABELS[predicted_gas]
        line.set_xdata(X_axis[:len(transmission_data)])
        line.set_ydata(transmission_data)
        ax.set_ylim([-0.1,1.1])
        ax.set_ylabel('Transmission')
        ax.set_title(text)
        if text == LABELS[7] or text == LABELS[8]:  # Assuming index 0 corresponds to "no gas"
            print("No gas detected.")

        else:
            if text in reference_data:
                wl_gas, abs_gas = reference_data[text]['wavelengths'], reference_data[text]['Transmission']
                concentration = estimate_concentration(transmission_data[start:end], wl_gas, abs_gas)
                print(f"Concentration of {text}: {concentration}")
                line.set_label(f'{text} Concentration: {concentration:.2f}')
                ax.legend(loc='upper right') 

            else:
                print(f"No concentration calculation for {text}.")


    else:
        line.set_xdata(X_axis[:len(raw_data)])
        line.set_ydata(raw_data)
        ax.set_ylabel('Intensity')
        line.set_label('Raw Intensity')
        ax.legend(loc='upper right')  # Adjust the location as necessary

    ax.set_xlim([200, 300])  # Set the x-axis scale from 200 to 300
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    fig.canvas.flush_events()

def preprocess_data(data):
    data=np.array(data)
    data = data[start:end].astype('float32')
    data=np.expand_dims(data, axis=0)
    return data

def predict_gas(spectrum):
    preprocessed_spectrum = preprocess_data(spectrum)
    prediction = model.predict([preprocessed_spectrum])
    predicted_gas = np.argmax(prediction)
    return predicted_gas

def estimate_concentration(spectra, wl_gas, abs_gas):
    """Estimates gas concentration based on the normalized spectra and the type of gas detected."""

    conc_min = 0
    conc_max = 2000
    conc_res = 5
    conc_vec = np.linspace(conc_min, conc_max, int((conc_max - conc_min) / conc_res))
    siz_factor=0.2
    siz = int(siz_factor * len(spectra))
    # min_val = np.median(spectra[:siz])
    # max_val = np.average(spectra[-siz:])
    max_val=np.median(np.sort(spectra)[-siz:])

    min_val=np.median(np.sort(spectra)[0:siz])
    spectra = (spectra - min_val) / (max_val - min_val)

    concentrations = []

    # Ensure spectra length and absorption data length match
    #if len(spectra) != len(abs_gas):
    #    spectra = np.interp(wl_gas, np.arange(len(spectra)), spectra)
    for conc in conc_vec:
        t_sim = np.exp(-conc * epsilon * abs_gas)
        rms = np.linalg.norm(spectra - t_sim)
        concentrations.append((conc, rms))
    
    best_conc = min(concentrations, key=lambda x: x[1])[0]
    return best_conc



# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

##################MODEL####################
# ----------------------------------------#


LABELS = [ 'Ammonia','Benzene','H2S','Sulfur','Ozone','Toluene','Xylene','Regular','noise']

# Load the model architecture
model = models.Sequential([
    layers.Dense(250, activation='tanh', input_shape=(311,)),
    layers.Dense(180, activation='tanh'),
    layers.Dense(100, activation='tanh'),
    layers.Dense(50, activation='tanh'),
    layers.Dense(20, activation='tanh'),
    layers.Dense(len(LABELS), activation='softmax')
])

# Load the weights
model.load_weights(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\s1.weights.h5")


# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________


##############Main Program################
# ----------------------------------------#
epsilon = 2.504e15  # 1/(ppm meter)
CL = 1000  # ppm meter
siz_factor = 0.15
# These indices might need to be adjusted after we look at the structure of the interpolated spectra
spectrum_start_index = 100
spectrum_end_index = 350

start = 647
end = 958


reference_data = load_reference_data()

# Read the X_axis values from a CSV file and creating reference vector variable of the spectrum needed
df = pd.read_csv(r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\General_Codes\Ibsen_spectrum_analyzer\Python Software Example\x64\X_axis_file.csv", header=None)
X_axis = df.iloc[0, :].values  # numpy array of values
reference_data_mode = None
dark_mode = None

# Initialize the flags for dark mode and reference data mode
dark_mode_captured = False
reference_data_captured = False

# Initialize the recording flag and recorded data list
is_recording = False
recorded_data = []
recording = False 

# Ask the user for a filename
filename = input("Enter a filename for the recorded data (without extension): ")


# Continuous data stream and live plotting
while True:
    recorded_data = []
    new_spectrum = GetSpectrum(ftHandleOutCom, ftHandleOutStream)
    update_plot(new_spectrum)
    time.sleep(0.1)

    # Check if both dark mode and reference data mode have been captured
    if dark_mode_captured and reference_data_captured:
        # Start recording if 'q' is pressed and not currently recording
        if keyboard.is_pressed('w') and not is_recording:
            is_recording = True
            print("Recording started.")
        # Pause recording if 'a' is pressed and currently recording
        elif keyboard.is_pressed('a') and is_recording:
            is_recording = False
            print("Recording paused.")
        # Continue recording if 'q' is pressed and currently paused
        elif keyboard.is_pressed('q') and not is_recording:
            is_recording = True
            print("Recording resumed.")
        # Stop and save if 'z' is pressed
        elif keyboard.is_pressed('z'):
            is_recording = False
            print("Recording stopped. Saving data...")
            # Save the recorded data to a file
            time.sleep(0.5)
            pd.DataFrame(recorded_data).to_csv(f"{filename}.csv", index=False, header=False)
            print(f"Recording finished and saved as {filename}.csv")
            recorded_data = []  # Clear recorded data after saving
            recording = False  # Stop recording



    # Capture dark mode and set the flag
    if keyboard.is_pressed('d') and not dark_mode_captured:
        dark_mode = new_spectrum
        dark_mode_captured = True
        print("Dark mode data captured.")
    # Capture reference data mode and set the flag
    if keyboard.is_pressed('r') and not reference_data_captured:
        reference_data_mode = new_spectrum
        reference_data_captured = True
        print("Reference data captured.")

    # Record the data if recording is active
    if is_recording:
        recorded_data.append(new_spectrum[start:end])


# Close devices
ftd2xx.FT_Close(ftHandleOutCom)
ftd2xx.FT_Close(ftHandleOutStream)
ftd2xx.FT_Close(ftHandleOutGPIO)



# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________

# _______________________________________________________________________________________________





