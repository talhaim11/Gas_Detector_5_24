import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models, callbacks
from keras.callbacks import EarlyStopping
import ctypes
from ctypes import *

import math
import matplotlib.pyplot as plt
import time
import keyboard

folder_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Data train\All_train_files"
all_data = []
labels = []
LABELS = ['H2S', 'Ammonia', 'Benzene', 'Ozone', 'Sulfur', 'Toluene', 'Xylene', 'Regular', 'noise']

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        t = df.to_numpy()

        # Determine the gas type based on the filename
        for i, gas in enumerate(LABELS):
            modified_gas = gas.replace('_', ' ')  # Replace underscores with spaces for comparison
            if modified_gas in filename:
                labels.append(i)
                break

        all_data.append(t[:,0])

X = np.array(all_data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Build the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(LABELS), activation='softmax')
])

model.summary()

# model.load_weights('s1.weights.h5')

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(X_train, y_train, epochs=60, batch_size=10,
          validation_split=0.2, callbacks=[early_stopping])

model.evaluate(X_test, y_test)