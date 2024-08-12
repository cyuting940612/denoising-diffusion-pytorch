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


model_cls_meta = Classifier.MetaClassifier()
# Load the trained model
model_cls_meta.load_state_dict(torch.load('2d_classifier_model_meta.pth'))
model_cls_meta.eval()

with torch.no_grad():
    _,features_week = model_cls_meta(X_tensor)

output = features_week.numpy()

user1_week = output[0,:]
user1_week_2 = output[100,:]
user1_weekend = output[600,:]
user2_weekend = output[607,:]
epsilon = 1e-8
mape_same_user_week = np.mean(np.abs((user1_week - user1_week_2) / (user1_week+epsilon)))
mape_same_user_week_weekend = np.mean(np.abs((user1_week - user1_weekend) / (user1_week+epsilon)))
mape_different_user_week_weekend = np.mean(np.abs((user1_week - user2_weekend) / (user1_week+epsilon)))

print(output)