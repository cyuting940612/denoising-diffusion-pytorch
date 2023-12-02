import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import DataProcess_Classifier as dpc
import tensorflow as tf
from tensorflow import keras
import Classifier
import Encoder
import DataProcess_user as dpu

df_all = dpc.DataProcess_Classifier()

# X = np.zeros((365,3,96))
# y = np.zeros((365))
# for i in range(365):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]
X = df_all[:,0:100,:]
y = df_all[:,100,0]
# X = np.random.randn(100, 3, 96)  # 100 data points with 3 channels and 96 time steps
# y = np.random.randint(0, 2, 100)  # Binary classification (0 or 1)
X = np.transpose(X,(0,2,1))
X = Encoder.encoder(X)
X = np.transpose(X,(0,2,1))

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Classifier.SimpleClassifier()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
# Save the trained model
torch.save(model.state_dict(), '2d_classifier_model.pth')
print("Finished Training")
