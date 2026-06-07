# ==========================================================
# IMPORTS
# ==========================================================
#
# torch:
# Core deep learning framework used for tensor operations,
# neural network creation and GPU acceleration.
#
# Dataset & DataLoader:
# Used to create custom datasets and efficient mini-batch
# loading during training.
#
# nn:
# Contains neural network layers and loss functions.
#
# optim:
# Provides optimization algorithms such as SGD.
#
# pandas:
# Used for reading CSV datasets and generating submission files.
#
# matplotlib:
# Used for visualizing training loss and accuracy.
#
# ==========================================================


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd
# ==========================================================
# STEP 1: DEVICE CONFIGURATION
# ==========================================================
#
# Use GPU if CUDA is available, otherwise fall back to CPU.
# This allows the code to run on different systems without
# modification.
#
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# reproducibility
torch.manual_seed(42)
# ==========================================================
# STEP 2: DATA LOADING AND PREPROCESSING
# ==========================================================
#
# Fashion-MNIST dataset is loaded from CSV files.
# Pixel values are normalized to the range [0,1] by dividing
# by 255 to improve training stability.
# ==========================================================

train_df = pd.read_csv(r"C:\Users\Ayush\Desktop\dataset\fashion-mnist_train.csv")

test_df = pd.read_csv(r"C:\Users\Ayush\Desktop\dataset\fashion-mnist_test.csv")

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

X_train = X_train / 255.0
X_test = X_test / 255.0
# ==========================================================
# STEP 3: CUSTOM DATASET CREATION
# ==========================================================
#
# A custom Dataset class is created so that Fashion-MNIST
# data can be used with PyTorch DataLoaders.
#
# Features are convrted into float tensors.
# Label are convrted into long tensor for use with
# CrossEntropyLoss.
#
# ==========================================================


class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
# ==========================================================
# STEP 4: DATALOADER CREATION
# ==========================================================
# DataLoader divides the dataset into mini-batches.
# batch_size = 32 ,Number of samples processed at one time.
# shuffle = True gives-Randomizes training data each epoch to improve generalization.
# ==========================================================

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)
# ==========================================================
# STEP 5: MODEL ARCHITECTURE
# ==========================================================
# Custom Multi-Branch Neural Network
# Architecture:
#                       Input (784)
#                           |
#                    Shared Layer (16)
#                           |
#                 -----------------------
#                 |                     |
#              Left Branch       Right Branch
#               16->8->8          16->12->8
#                 |                     |
#                 Skip Connection       |
#                 -----------------------
#                            |
#                     Concatenation
#                            |
#                    Output Layer (10)
#
# A skip connection is used in the left branch to improve
# gradient flow and feature reuse.
#
# ==========================================================


class MyNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        # 1. Base Shared Layer: Input (784) -> Out (16)
        self.hidden1 = nn.Linear(num_features, 16)

        # --- LEFT BRANCH LAYERS ---
        # Input (16) -> Out (8)
        self.left_hidden1 = nn.Linear(16, 8)
        # Input (8) -> Out (8)
        self.left_hidden2 = nn.Linear(8, 8)

        # --- RIGHT BRANCH LAYERS ---
        # Input (16) -> Out (12)
        self.right_hidden1 = nn.Linear(16, 12)
        # Input (12) -> Out (8)
        self.right_hidden2 = nn.Linear(12, 8)

        # --- FINAL OUTPUT LAYER ---
        # The two 8-unit outputs concatenate into 16 units. Out (10) for classes.
        self.output_layer = nn.Linear(16, 10)

    def forward(self, x):
        # Pass through the first hidden layer and apply ReLU activation
        x_shared = torch.relu(self.hidden1(x))

        # --- LEFT BRANCH TRACKING ---
        # Step A: First left layer
        left_branch_input = torch.relu(self.left_hidden1(x_shared))
        # Step B: Second left layer
        left_branch_out = torch.relu(self.left_hidden2(left_branch_input))
        # Step C: SKIP CONNECTION (Add the input of the layer back to its output)
        left_final = left_branch_out + left_branch_input

        # --- RIGHT BRANCH TRACKING ---
        right_branch_1 = torch.relu(self.right_hidden1(x_shared))
        right_final = torch.relu(self.right_hidden2(right_branch_1))

        # --- CONCATENATE BRANCHES ---
        # Combine the 8 outputs from left and 8 from right side-by-side to get 16
        combined = torch.cat((left_final, right_final), dim=1)

        # --- FINAL OUTPUT ---
        out = self.output_layer(combined)
        return out

# ==========================================================
# STEP 6: TRAINING CONFIGURATION
# ==========================================================
epochs = 150
learning_rate = 0.001
model = MyNN(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# ==========================================================
# STEP 7: MODEL TRAINING AND EVALUATION
# ==========================================================
# Training Phase:
#     Forward Pass
#       Loss Calculation
#     Backpropagation
#      Weight Update
# Evaluation Phase:
#     Model performance is measured on the test dataset.
# Metrics recorded:
#     Average Training Loss
#     Test Accuracy
# ==========================================================
loss_history = []
accuracy_history = []

final_predictions = []
best_accuracy = 0


for epoch in range(epochs):

    total_epoch_loss = 0

    for batch_features, batch_labels in train_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader)

    model.eval()

    total = 0
    correct = 0

    epoch_predictions = []

    with torch.no_grad():

        for batch_features, batch_labels in test_loader:

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)

            _, predicted = torch.max(outputs, 1)

            epoch_predictions.extend(
                predicted.cpu().numpy()
            )

            total += batch_labels.size(0)

            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    #best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        

        
     # Save predictions only from the FINAL epoch
    if epoch == epochs - 1:
        final_predictions = epoch_predictions

    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    model.train()

print("\n==========================")
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print("==========================")


# ==========================================================
# Save trained model weights
# ==========================================================

torch.save(model.state_dict(),"base_task/save_models/fashion_mnist_model.pth")

print("Model weights saved successfully.")


# ==========================================================
# Create submission.csv
# ==========================================================

submission_df = pd.DataFrame({"Id": range(len(final_predictions)),"Label": final_predictions})

submission_df.to_csv("base_task/submission.csv", index=False)

print("submission.csv generated successfully.")


# ==========================================================
# Save training plots
# ==========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(accuracy_history)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("base_task/plots/training_plots.png")

print("Plots saved successfully.")

plt.show()