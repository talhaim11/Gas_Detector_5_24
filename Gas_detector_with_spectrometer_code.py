import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import ctypes
from ctypes import *
import math
import time
import keyboard

class GasAnalyzer:
    def __init__(self, spectrometer_code, model_weights, ppm_calculation_code):
        # Initialize spectrometer
        self.spectrometer = spectrometer_code

        # Load the pre-trained model
        self.model = load_model(model_weights)

        # Load the PPM calculation setup
        self.ppm_calculation = ppm_calculation_code

    def update_plot(self, raw_data):
            global reference_data_mode, dark_mode
            print(f"Length of new data: {len(raw_data)}")
            print(f"Length of X_axis: {len(X_axis)}")

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
                # Ensure no division by zero and subtract dark mode before division
                transmission_data = [((raw - dark) / (ref - dark)) if (ref-dark) != 0 else 0 for raw, dark, ref in zip(raw_data, dark_mode, reference_data_mode)]
                line.set_xdata(X_axis[:len(transmission_data)])
                line.set_ydata(transmission_data)
                ax.set_ylim([0,1.1])
                ax.set_ylabel('Transmission')
            else:
                line.set_xdata(X_axis[:len(raw_data)])
                line.set_ydata(raw_data)
                ax.set_ylabel('Intensity')

            ax.set_xlim([200, 300])  # Set the x-axis scale from 100 to 300
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()
            pass

    def classify_gas(self, spectrum):
        # Preprocess the spectrum for the model if necessary
        # Return the classification results
        return self.model.predict(spectrum)

    def calculate_ppm(self, spectrum, gas_type):
        # Use the PPM calculation code with the spectrum to calculate PPM
        pass

    def run(self):
        # The main loop for data acquisition
        while True:
            spectrum = self.spectrometer.GetSpectrum()
            self.update_plot(spectrum)

            gas_type = self.classify_gas(spectrum)
            if gas_type != "No interrupt" and gas_type != "Noise":
                ppm = self.calculate_ppm(spectrum, gas_type)
                print(f"Detected {gas_type} at {ppm} ppm")

# Then you would create an instance of the GasAnalyzer and run it:
# analyzer = GasAnalyzer(spectrometer_code, model_weights, ppm_calculation_code)
# analyzer.run()
