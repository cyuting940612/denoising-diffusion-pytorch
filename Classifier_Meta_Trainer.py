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
import DataProfess_Classifier_High_Low as dpchl

df_all_week = dpc.data_process()
df_all_highlow = dpchl.data_process()

# X = np.zeros((365,3,96))
# y = np.zeros((365))
# for i in range(365):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]
X_week = df_all_week[:,0:1,:]
y_week = df_all_week[:,1,0]
reshaped_arr_week = X_week.reshape(36500, 96)
X_tensor_week = torch.tensor(reshaped_arr_week, dtype=torch.float32)
y_tensor_week = torch.tensor(y_week, dtype=torch.float32)

X_highlow = df_all_highlow[:,0:96]
y_highlow = df_all_highlow[:,96]
X_tensor_highlow = torch.tensor(X_highlow, dtype=torch.float32)
y_tensor_highlow = torch.tensor(y_week, dtype=torch.float32)

model_cls_week = Classifier.SimpleNN()
# Load the trained model
model_cls_week.load_state_dict(torch.load('2d_classifier_model_week_weekend.pth'))
model_cls_week.eval()

with torch.no_grad():
    _,features_week = model_cls_week(X_tensor_week)


model_cls_week = Classifier.SimpleNN()
# Load the trained model
model_cls_week.load_state_dict(torch.load('2d_classifier_model_high_low.pth'))
model_cls_week.eval()

with torch.no_grad():
    _,features_highlow = model_cls_week(X_tensor_highlow)



y  = y_week * 2 + y_highlow
X = np.dstack((features_week,features_highlow))
# X = np.random.randn(100, 3, 96)  # 100 data points with 3 channels and 96 time steps
# y = np.random.randint(0, 2, 100)  # Binary classification (0 or 1)
# X = np.transpose(X,(0,2,1))
# X = Encoder.encoder(X)
# X = np.transpose(X,(0,2,1))

# reshaped_arr = X.reshape(36500, 96)
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Classifier.MetaClassifier()
criterion = nn.CrossEntropyLoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output,_ = model(X_batch)
        output_64 = output.to(dtype=torch.float64)
        y_batch_64 = y_batch.to(dtype=torch.float64)
        loss = criterion(output_64, output_64)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
# Save the trained model
torch.save(model.state_dict(), '2d_classifier_model_meta.pth')

print("Finished Training")
