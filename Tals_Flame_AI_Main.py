# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Import your model and data preparation modules
from models.Tals_Flame_AI_Model import Tals_Flame_AI_Model
from utils.data_preparation import load_and_prepare_data

# Training the model
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            # Print batch loss and accuracy for debugging
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}')
        
        epoch_loss /= len(train_loader)
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Validate the model
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = correct / total
    return val_loss, accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model weights saved to {path}')

# Evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = (outputs >= 0.5).float()
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.4f}')

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        df_cm = pd.DataFrame(cm, index=["Non-Fire", "Fire"], columns=["Non-Fire", "Fire"])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        return accuracy

# Initialize the model
model = Tals_Flame_AI_Model(input_size=20)  # Use all 20 features

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.BCELoss()

# Load and prepare data with the specified target label
data_directory = r"C:\Users\thaim\OneDrive\Desktop\test for flame ai\test_model"
target_label = 2
train_loader, val_loader, X_test, y_test = load_and_prepare_data(data_directory, target_label)

# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50)

# Save weights
model_save_path = r'C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\General_Codes\Gas_Detector_5_24\Model_weights.pth'
save_model(model, model_save_path)

# Evaluate the model
evaluate_model(model, X_test, y_test)