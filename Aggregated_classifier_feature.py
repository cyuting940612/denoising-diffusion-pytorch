import math

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import Aggregated_classifier as ac
import DataProfess_Classifier_High_Low as dpchl
import DataProcess_Classifier as dpc
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm


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
y_combine = np.column_stack((y_week, y_highlow))
reshaped_arr_week = X_week.reshape(36500, 96)
X_tensor_week = torch.tensor(reshaped_arr_week, dtype=torch.float32)
y_tensor_week = torch.tensor(y_combine, dtype=torch.float32)




model = ac.Aggregated_Classifier()

model.load_state_dict(torch.load('aggregated_classifier_model.pth'))
model.eval()  # Set the model to evaluation mode


with torch.no_grad():
    _,aggregated_features = model(X_tensor_week)

output = aggregated_features.numpy()
user1_week = output[0,:]
user1_week_2 = output[100,:]
user1_weekend = output[26100,:]
user2_weekend = output[26107,:]
epsilon = 1e-8
mape_same_user_week = np.mean(np.abs((user1_week - user1_week_2) / (user1_week+epsilon)))
mape_same_user_week_weekend = np.mean(np.abs((user1_week - user1_weekend) / (user1_week+epsilon)))
mape_different_user_weekend = np.mean(np.abs((user1_weekend - user2_weekend) / (user1_weekend+epsilon)))
mape_different_user_week_weekend = np.mean(np.abs((user1_week - user2_weekend) / (user1_week+epsilon)))


def calculate_fid(images1, images2):
    # Calculate activations
    act1 = images1.reshape(10,1)
    act2 = images2.reshape(10,1)

    act1 = np.atleast_2d(act1)
    act2 = np.atleast_2d(act2)

    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Check for the single-feature edge case
    if sigma1.shape == ():
        sigma1 = np.array([[sigma1]])
    if sigma2.shape == ():
        sigma2 = np.array([[sigma2]])

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Take the real part if complex

    # Calculate FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

fid = calculate_fid(user1_week, user1_week)
fid_same_user_week = calculate_fid(user1_week , user1_week_2)
fid_same_user_week_weekend = calculate_fid(user1_week , user1_weekend)
fid_different_user_weekend = calculate_fid(user1_weekend , user2_weekend)
fid_different_user_week_weekend = calculate_fid(user1_week , user2_weekend)

print(fid_same_user_week)
print(fid_same_user_week_weekend)
print(fid_different_user_weekend)
print(fid_different_user_week_weekend)