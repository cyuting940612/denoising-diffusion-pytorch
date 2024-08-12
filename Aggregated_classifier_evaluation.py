import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import Aggregated_classifier as ac
import DataProfess_Classifier_High_Low as dpchl
import DataProcess_Classifier as dpc
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import pandas as pd



df_all_week = dpc.data_process()
# df_all_highlow = dpchl.data_process()

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
y_highlow = df_all_week[:,2,0]
y_min =df_all_week[:,3,0]
y_max =df_all_week[:,4,0]
y_range =df_all_week[:,5,0]
y_var =df_all_week[:,6,0]
y_peaklength = df_all_week[:,7,0]
y_lowlength = df_all_week[:,8,0]
y_peak07 = df_all_week[:,9,0]
y_peak815 = df_all_week[:,10,0]
y_peak1623 = df_all_week[:,11,0]
y_low07 = df_all_week[:,12,0]
y_low815 = df_all_week[:,13,0]
y_low1623 = df_all_week[:,14,0]

y_combine = np.column_stack((y_week, y_highlow,y_min,y_max,y_range,y_var,y_peaklength,y_lowlength,y_peak07,y_peak815,y_peak1623,y_low07,y_low815,y_low1623))
reshaped_arr_week = X_week.reshape(36500, 96)
X_tensor_week = torch.tensor(reshaped_arr_week, dtype=torch.float32)
y_tensor_week = torch.tensor(y_combine, dtype=torch.float32)

network = ac.Aggregated_Classifier()
dataset = TensorDataset(X_tensor_week, y_tensor_week)
testing_dataloader = DataLoader(dataset,batch_size=16, shuffle=True)

correct = 0
total = 0
network = ac.Aggregated_Classifier()
network.load_state_dict(torch.load('aggregated_classifier_model_144.pth'))
# Assuming your model is named 'model' and your test DataLoader is named 'test_loader'
network.eval()
with torch.no_grad():
    for inputs, labels in testing_dataloader:
        outputs,_ = network(inputs)  # Get model predictions

        predictions = (outputs > 0).int()

        # Calculate accuracy
        correct_predictions = (predictions == labels).all(dim=1).float()
        correct += correct_predictions.sum().item()
        total += labels.size(0)

print(f'Accuracy of the model on the test images: {100 * correct / total}%')