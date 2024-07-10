import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load and prepare data
def load_and_prepare_data(directory, target_label):
    all_data = []
    all_labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, skiprows=1)
            
            # Extract the fire or non-fire label from the filename
            is_fire_file = 'fire' in filename.lower()
            
            # Filter rows with the target label
            target_df = df[df.iloc[:, -1] == target_label]
            non_target_df = df[df.iloc[:, -1] != target_label]
            
            # Extract features (all columns except the last one)
            X_target = target_df.iloc[:, :-1].values
            X_non_target = non_target_df.iloc[:, :-1].values
            
            # Labels: 1 for fire and 0 for non-fire
            y_target = np.ones(len(X_target)) if is_fire_file else np.zeros(len(X_target))
            y_non_target = np.zeros(len(X_non_target)) if is_fire_file else np.ones(len(X_non_target))
            
            all_data.append(X_target)
            all_data.append(X_non_target)
            all_labels.extend(y_target)
            all_labels.extend(y_non_target)
    
    # Combine all data
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    # Ensure labels are floats
    y = y.astype(float)
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Use float type for BCELoss
    
    # Split the data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    # Create DataLoaders for the training and validation sets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, X_test, y_test


