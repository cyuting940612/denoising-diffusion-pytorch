
from scipy.linalg import sqrtm
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy.stats import entropy
import pandas as pd
import DataProcess_filterfree as dp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel




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
# from CNN_classifier import Conv1DModel
from scipy.interpolate import interp1d
import Generalized_classifier as gc


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU.")

def lanczos_kernel(x, a):
    if x == 0:
        return 1
    elif -a < x < a:
        sinc1 = np.sinc(x)
        sinc2 = np.sinc(x / a)
        return a * sinc1 * sinc2
    else:
        return 0


def lanczos_resample(image, new_size, a=3):
    height, width = image.shape
    new_height, new_width = new_size

    # Calculate the scale factors
    scale_x = new_width / width
    scale_y = new_height / height

    # Create the resampled image
    resampled_image = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            x = j / scale_x
            y = i / scale_y

            # Calculate the integer and fractional parts
            x_int = int(x)
            y_int = int(y)
            x_frac = x - x_int
            y_frac = y - y_int

            # Apply the Lanczos kernel
            value = 0
            for m in range(-a + 1, a):
                for n in range(-a + 1, a):
                    x_idx = min(max(x_int + m, 0), width - 1)
                    y_idx = min(max(y_int + n, 0), height - 1)
                    lanczos_kernel_1 = lanczos_kernel(x_frac - m, a)
                    lanczos_kernel_2 = lanczos_kernel(y_frac - n, a)
                    value += image[y_idx, x_idx] * lanczos_kernel_1 * lanczos_kernel_2

            resampled_image[i, j] = value

    return resampled_image

# Create a simple gradient image
new_size = (1, 96)
a = 3
filename_load_2022 = 'ERCOT load - 2022.csv'
filename_load_2018 = 'ERCOT load - 2019.csv'
df_load_2022 = pd.read_csv(filename_load_2022)
df_load_2018 = pd.read_csv(filename_load_2018)

load_origin_2022 = df_load_2022['ERCOT']/1000
load_origin_2018 = df_load_2018['ERCOT']/1000

load_numpy_2022 = load_origin_2022.to_numpy()
load_numpy_2018 = load_origin_2018.to_numpy()

array_2d_2022 = load_numpy_2022.reshape((365, 24))
array_2d_2018 = load_numpy_2018.reshape((365, 24))


# Number of days in each month (non-leap year)
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Initialize a list to store the monthly arrays
monthly_data_2022 = []
monthly_data_2018 = []

start_day = 0
for days in days_in_month:
    end_day = start_day + days
    # Extract data for the current month
    month_data_2022 = array_2d_2022[start_day:end_day, :].reshape(-1)
    monthly_data_2022.append(month_data_2022)
    start_day = end_day

# for index in range(1,73):
#     end_day = start_day + 5
#     # Extract data for the current month
#     month_data_2022 = array_2d_2022[start_day:end_day, :].reshape(-1)
#     monthly_data_2022.append(month_data_2022)
#     start_day = end_day

# Now, monthly_data is a list of 12 arrays, each containing the concatenated hourly data for a month
start_day = 0
for days in days_in_month:
    end_day = start_day + days
    # Extract data for the current month
    month_data_2018 = array_2d_2018[start_day:end_day, :].reshape(-1)
    monthly_data_2018.append(month_data_2018)
    start_day = end_day
# for index in range(1,73):
#     end_day = start_day + 5
#     # Extract data for the current month
#     month_data_2018 = array_2d_2018[start_day:end_day, :].reshape(-1)
#     monthly_data_2018.append(month_data_2018)
#     start_day = end_day

max_interval = 96*31
resampled_image_arr_2022 = np.zeros((365,max_interval))
resampled_image_arr_2018 = np.zeros((365,max_interval))

resampled_month_arr_2022 = np.zeros((12,max_interval))
resampled_month_arr_2018 = np.zeros((12,max_interval))
def expand_array(input_array, target_length=2976):
    # Initialize the target array with zeros
    expanded_array = np.zeros(target_length)

    # Copy the input array into the beginning of the expanded array
    expanded_array[:len(input_array)] = input_array

    return expanded_array

for i in range(array_2d_2022.shape[0]):
    origin_image_2022 = array_2d_2022[i].reshape(1,array_2d_2022.shape[1])
    repeated_array_2d_2022 = np.repeat(array_2d_2022[i],4)
    resampled_image_2022 = expand_array(repeated_array_2d_2022)
    resampled_image_arr_2022[i] = resampled_image_2022

for j in range(array_2d_2018.shape[0]):
    origin_image_2018 = array_2d_2018[j].reshape(1,array_2d_2018.shape[1])
    repeated_array_2d_2018 = np.repeat(array_2d_2018[j],4)
    resampled_image_2018 = expand_array(repeated_array_2d_2018)
    resampled_image_arr_2018[j] = resampled_image_2018





for m,month_2022 in enumerate(monthly_data_2022):
    origin_monthly_2022 = month_2022.reshape(1,month_2022.shape[0])
    # resampled_monthly_2022 = lanczos_resample(origin_monthly_2022, new_size, a)
    repeated_month_2022 = np.repeat(month_2022,4)

    resampled_monthly_2022 = expand_array(repeated_month_2022)
    resampled_month_arr_2022[m] = resampled_monthly_2022

for n, month_2018 in enumerate(monthly_data_2018):
    origin_monthly_2018 = month_2018.reshape(1, month_2018.shape[0])
    # resampled_monthly_2018 = lanczos_resample(origin_monthly_2018, new_size, a)
    repeated_month_2018 = np.repeat(month_2018,4)
    resampled_monthly_2018 = expand_array(repeated_month_2018)
    resampled_month_arr_2018[n] = resampled_monthly_2018
# Resize to 10 using Lanczos resampling



# def normalize_to_range(data, a=0, b=255):
#     # Calculate min and max for each column
#     col_mins = np.min(data, axis=1, keepdims=True)
#     col_maxs = np.max(data, axis=1, keepdims=True)
#
#     # col_mins = np.min(data, axis=0)
#     # col_maxs = np.max(data, axis=0)
#
#     # Avoid division by zero
#     col_ranges = col_maxs - col_mins
#     col_ranges[col_ranges == 0] = 1
#
#     # Normalize each column to range [a, b]
#     normalized_data = a + ((data - col_mins) / col_ranges) * (b - a)
#     return normalized_data
def expand_array_2D(input_array, target_length=2976):
    # Initialize the target array with zeros
    expanded_array = np.zeros((input_array.shape[0],target_length))
    for i in range(input_array.shape[0]):


    # Copy the input array into the beginning of the expanded array
        expanded_array[i,:input_array.shape[1]] = input_array[i,:]

    return expanded_array

# Path to your CSV file
# week_file_path = 'filter_free_weekday_testing.csv'
week_file_path = 'filter_free_weekday_5000_s1.csv'

# Read the CSV file into a pandas DataFrame
generated_week_data = pd.read_csv(week_file_path, header=None)  # Assuming the file has no header

# Convert the DataFrame to a numpy array and then reshape it
# generated_week_reshaped_array = generated_week_data.values.reshape(8, 96)
generated_week_reshaped_array = generated_week_data.values.reshape(5000, 96)

# normalized_generated_week_reshaped_array = normalize_to_range(generated_week_reshaped_array, a=0, b=1)
expanded_generated_weekend_reshaped_array = expand_array_2D(generated_week_reshaped_array)

# Convert the numpy array to a list of lists
generated_week_images = expanded_generated_weekend_reshaped_array.tolist()
generated_week_images_numpy = np.array(generated_week_images)
X_tensor_generated_week_stack = torch.tensor(generated_week_images_numpy, dtype=torch.float32)
X_tensor_generated_week = torch.tensor(generated_week_images_numpy[:,:96], dtype=torch.float32)

# Path to your CSV file
weekend_file_path = 'filter_free_weekend_5000_s1.csv'
# Read the CSV file into a pandas DataFrame
generated_weekend_data = pd.read_csv(weekend_file_path, header=None)  # Assuming the file has no header


# Convert the DataFrame to a numpy array and then reshape it
generated_weekend_reshaped_array = generated_weekend_data.values.reshape(5000, 96)
# normalized_generated_weekend_reshaped_array = normalize_to_range(generated_weekend_reshaped_array, a=0, b=1)
expanded_generated_weekend_reshaped_array = expand_array_2D(generated_weekend_reshaped_array)
# Convert the numpy array to a list of lists
generated_weekend_images = expanded_generated_weekend_reshaped_array.tolist()
generated_weekend_images_numpy = np.array(generated_weekend_images)
X_tensor_generated_weekend_stack = torch.tensor(generated_weekend_images_numpy, dtype=torch.float32)
X_tensor_generated_weekend = torch.tensor(generated_weekend_images_numpy[:,0:96], dtype=torch.float32)



# Path to your CSV file
# week_file_path = 'filter_free_weekday_testing.csv'
transformer_week_file_path = 'transformer_weekday.csv'

# Read the CSV file into a pandas DataFrame
transformer_generated_week_data = pd.read_csv(transformer_week_file_path, header=None)  # Assuming the file has no header

# Convert the DataFrame to a numpy array and then reshape it
# generated_week_reshaped_array = generated_week_data.values.reshape(8, 96)
transformer_generated_week_reshaped_array = transformer_generated_week_data.values.reshape(100, 96)

# normalized_generated_week_reshaped_array = normalize_to_range(generated_week_reshaped_array, a=0, b=1)
transformer_expanded_generated_weekend_reshaped_array = expand_array_2D(transformer_generated_week_reshaped_array)

# Convert the numpy array to a list of lists
transformer_generated_week_images = transformer_expanded_generated_weekend_reshaped_array.tolist()
transformer_generated_week_images_numpy = np.array(transformer_generated_week_images)
X_tensor_generated_transformer_week = torch.tensor(transformer_generated_week_images_numpy[:,:96], dtype=torch.float32)

# Path to your CSV file
transformer_weekend_file_path = 'transformer_weekend.csv'
# Read the CSV file into a pandas DataFrame
transformer_generated_weekend_data = pd.read_csv(transformer_weekend_file_path, header=None)  # Assuming the file has no header


# Convert the DataFrame to a numpy array and then reshape it
transformer_generated_weekend_reshaped_array = transformer_generated_weekend_data.values.reshape(100, 96)
# normalized_generated_weekend_reshaped_array = normalize_to_range(generated_weekend_reshaped_array, a=0, b=1)
transformer_expanded_generated_weekend_reshaped_array = expand_array_2D(transformer_generated_weekend_reshaped_array)
# Convert the numpy array to a list of lists
transformer_generated_weekend_images = transformer_expanded_generated_weekend_reshaped_array.tolist()
transformer_generated_weekend_images_numpy = np.array(transformer_generated_weekend_images)
X_tensor_generated_transformer_weekend = torch.tensor(transformer_generated_weekend_images_numpy[:,0:96], dtype=torch.float32)



df_all_week,_ = dpc.data_process()
# df_all_highlow = dpchl.data_process()

# X = np.zeros((365,3,96))
# y = np.zeros((365))
# for i in range(365):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]
X_daily = df_all_week[:36500,:96]
# total_interval = 36500+1200+5200
# normalized_arr_week = normalize_to_range(reshaped_arr_week, a=0, b=1)
X_tensor_stack = torch.tensor(df_all_week, dtype=torch.float32)
X_tensor_daily = torch.tensor(X_daily, dtype=torch.float32)

# num_samples = 36500+12
# input_shape = (96, 1)
# num_classes = 32

model_stack = gc.Generalized_Classifier().to(device)
model_stack.load_state_dict(torch.load('generalized_classifier_model_monthly.pth'))
model_stack.eval()

model_daily = ac.Daily_Classifier().to(device)
model_daily.load_state_dict(torch.load('aggregated_classifier_daily.pth'))
model_daily.eval()  # Set the model to evaluation mode

model_weekly = ac.Weekly_Classifier().to(device)
model_weekly.load_state_dict(torch.load('aggregated_classifier_weekly.pth'))
model_weekly.eval()  # Set the model to evaluation mode

model_monthly = ac.Monthly_Classifier().to(device)
model_monthly.load_state_dict(torch.load('aggregated_classifier_monthly.pth'))
model_monthly.eval()  # Set the model to evaluation mode

X_tensor_winter_2022 = torch.tensor(resampled_image_arr_2022[0:100,0:96], dtype=torch.float32)
X_tensor_summer_2022 = torch.tensor(resampled_image_arr_2022[150:250,0:96], dtype=torch.float32)
X_tensor_winter_2018 = torch.tensor(resampled_image_arr_2018[0:100,0:96], dtype=torch.float32)
X_tensor_summer_2018 = torch.tensor(resampled_image_arr_2018[150:250,0:96], dtype=torch.float32)

X_tensor_winter_2022_stack = torch.tensor(resampled_image_arr_2022[0:100], dtype=torch.float32)
X_tensor_summer_2022_stack = torch.tensor(resampled_image_arr_2022[150:250], dtype=torch.float32)
X_tensor_winter_2018_stack = torch.tensor(resampled_image_arr_2018[0:100], dtype=torch.float32)
X_tensor_summer_2018_stack = torch.tensor(resampled_image_arr_2018[150:250], dtype=torch.float32)


X_tensor_spring_monthly_2022_stack = torch.tensor(resampled_month_arr_2022[0:4],dtype=torch.float32)
X_tensor_spring_monthly_2018_stack = torch.tensor(resampled_month_arr_2018[0:4],dtype=torch.float32)
X_tensor_summer_monthly_2022_stack = torch.tensor(resampled_month_arr_2022[4:8],dtype=torch.float32)
X_tensor_summer_monthly_2018_stack = torch.tensor(resampled_month_arr_2018[4:8],dtype=torch.float32)

array_2d_2022_repeated = np.repeat(array_2d_2022, 4, axis=1)
array_2d_2018_repeated = np.repeat(array_2d_2018, 4, axis=1)
X_tensor_daily_2022 = torch.tensor(array_2d_2022_repeated,dtype=torch.float32, device=device)
X_tensor_daily_2018 = torch.tensor(array_2d_2018_repeated,dtype=torch.float32, device=device)



with torch.no_grad():
    _,_,_,month_feature_daily_phase_2022 = model_daily(X_tensor_daily_2022.to(device))
    _,_,_,month_feature_daily_phase_2018 = model_daily(X_tensor_daily_2018.to(device))

start_index = 0
week_feature_2022 = []
week_feature_2018 = []
for i in range(52):

    month_feature_daily_phase_2022_day1 = month_feature_daily_phase_2022[start_index,:]
    month_feature_daily_phase_2022_day2 = month_feature_daily_phase_2022[start_index+1,:]
    month_feature_daily_phase_2022_day3 = month_feature_daily_phase_2022[start_index+2,:]
    month_feature_daily_phase_2022_day4 = month_feature_daily_phase_2022[start_index+3,:]
    month_feature_daily_phase_2022_day5 = month_feature_daily_phase_2022[start_index+4,:]
    month_feature_daily_phase_2022_day6 = month_feature_daily_phase_2022[start_index+5,:]
    month_feature_daily_phase_2022_day7 = month_feature_daily_phase_2022[start_index+6,:]
    single_week_feature_2022 = torch.cat((month_feature_daily_phase_2022_day1,month_feature_daily_phase_2022_day2,month_feature_daily_phase_2022_day3,month_feature_daily_phase_2022_day4,month_feature_daily_phase_2022_day5,month_feature_daily_phase_2022_day6,month_feature_daily_phase_2022_day7),dim=0)
    week_feature_2022.append(single_week_feature_2022)

    month_feature_daily_phase_2018_day1 = month_feature_daily_phase_2018[start_index,:]
    month_feature_daily_phase_2018_day2 = month_feature_daily_phase_2018[start_index+1,:]
    month_feature_daily_phase_2018_day3 = month_feature_daily_phase_2018[start_index+2,:]
    month_feature_daily_phase_2018_day4 = month_feature_daily_phase_2018[start_index+3,:]
    month_feature_daily_phase_2018_day5 = month_feature_daily_phase_2018[start_index+4,:]
    month_feature_daily_phase_2018_day6 = month_feature_daily_phase_2018[start_index+5,:]
    month_feature_daily_phase_2018_day7 = month_feature_daily_phase_2018[start_index+6,:]
    single_week_feature_2018 = torch.cat((month_feature_daily_phase_2018_day1,month_feature_daily_phase_2018_day2,month_feature_daily_phase_2018_day3,month_feature_daily_phase_2018_day4,month_feature_daily_phase_2018_day5,month_feature_daily_phase_2018_day6,month_feature_daily_phase_2018_day7),dim=0)
    week_feature_2018.append(single_week_feature_2018)
    start_index+=7

week_feature_2022_torch = torch.stack(week_feature_2022, dim=0)
week_feature_2018_torch = torch.stack(week_feature_2018, dim=0)

with torch.no_grad():
    _,_,_,month_feature_weekly_phase_2022 = model_weekly(week_feature_2022_torch.to(device))
    _,_,_,month_feature_weekly_phase_2018 = model_weekly(week_feature_2018_torch.to(device))

start_index = 0
month_feature_2022 = []
month_feature_2018 = []
for i in range(12):

    month_feature_weekly_phase_2022_week1 = month_feature_weekly_phase_2022[start_index,:]
    month_feature_weekly_phase_2022_week2 = month_feature_weekly_phase_2022[start_index+1,:]
    month_feature_weekly_phase_2022_week3 = month_feature_weekly_phase_2022[start_index+2,:]
    month_feature_weekly_phase_2022_week4 = month_feature_weekly_phase_2022[start_index+3,:]
    single_month_feature_2022 = torch.cat((month_feature_weekly_phase_2022_week1,month_feature_weekly_phase_2022_week2,month_feature_weekly_phase_2022_week3,month_feature_weekly_phase_2022_week4),dim=0)
    month_feature_2022.append(single_month_feature_2022)

    month_feature_weekly_phase_2018_week1 = month_feature_weekly_phase_2018[start_index,:]
    month_feature_weekly_phase_2018_week2 = month_feature_weekly_phase_2018[start_index+1,:]
    month_feature_weekly_phase_2018_week3 = month_feature_weekly_phase_2018[start_index+2,:]
    month_feature_weekly_phase_2018_week4 = month_feature_weekly_phase_2018[start_index+3,:]

    single_month_feature_2018 = torch.cat((month_feature_weekly_phase_2018_week1,month_feature_weekly_phase_2018_week2,month_feature_weekly_phase_2018_week3,month_feature_weekly_phase_2018_week4),dim=0)
    month_feature_2018.append(single_month_feature_2018)
    start_index+=4

month_feature_2022_torch = torch.stack(month_feature_2022, dim=0)
month_feature_2018_torch = torch.stack(month_feature_2018, dim=0)

with torch.no_grad():
    X_tensor_daily = X_tensor_daily.to(device)
    _,_,_,aggregated_features = model_daily(X_tensor_daily)
    _,_,_,generated_week_aggregated_features = model_daily(X_tensor_generated_week.to(device))
    _,_,_,generated_weekend_aggregated_features = model_daily(X_tensor_generated_weekend.to(device))
    _,_,_,generated_week_transformer_aggregated_features = model_daily(X_tensor_generated_transformer_week.to(device))
    _,_,_,generated_weekend_transformer_aggregated_features = model_daily(X_tensor_generated_transformer_weekend.to(device))
    # _,_,_,generated_winter_2022_aggregated_features = model_daily(X_tensor_winter_2022.to(device))
    # _,_,_,generated_summer_2022_aggregated_features = model_daily(X_tensor_summer_2022.to(device))
    # _,_,_,generated_winter_2018_aggregated_features = model_daily(X_tensor_winter_2018.to(device))
    # _,_,_,generated_summer_2018_aggregated_features = model_daily(X_tensor_summer_2018.to(device))
    # _,_,_,generated_month_spring_2022_aggregated_features = model_monthly(month_feature_2022_torch[0:3].to(device))
    # _,_,_,generated_month_spring_2018_aggregated_features = model_monthly(month_feature_2018_torch[0:3].to(device))
    # _,_,_,generated_month_summer_2022_aggregated_features = model_monthly(month_feature_2022_torch[5:8].to(device))
    # _,_,_,generated_month_summer_2018_aggregated_features = model_monthly(month_feature_2018_torch[5:8].to(device))

    # _,_,_,aggregated_features = model_stack(X_tensor_stack.to(device))
    # _,_,_,generated_week_aggregated_features = model_stack(X_tensor_generated_week_stack.to(device))
    # _,_,_,generated_weekend_aggregated_features = model_stack(X_tensor_generated_weekend_stack.to(device))
    # _,_,_,generated_week_transformer_aggregated_features = model_stack(X_tensor_generated_transformer_week.to(device))
    # _,_,_,generated_weekend_transformer_aggregated_features = model_stack(X_tensor_generated_transformer_weekend.to(device))
    _,_,_,generated_winter_2022_aggregated_features = model_stack(X_tensor_winter_2022_stack.to(device))
    _,_,_,generated_summer_2022_aggregated_features = model_stack(X_tensor_summer_2022_stack.to(device))
    _,_,_,generated_winter_2018_aggregated_features = model_stack(X_tensor_winter_2018_stack.to(device))
    _,_,_,generated_summer_2018_aggregated_features = model_stack(X_tensor_summer_2018_stack.to(device))
    _,_,_,generated_month_spring_2022_aggregated_features = model_stack(X_tensor_spring_monthly_2022_stack.to(device))
    _,_,_,generated_month_spring_2018_aggregated_features = model_stack(X_tensor_spring_monthly_2018_stack.to(device))
    _,_,_,generated_month_summer_2022_aggregated_features = model_stack(X_tensor_summer_monthly_2022_stack.to(device))
    _,_,_,generated_month_summer_2018_aggregated_features = model_stack(X_tensor_summer_monthly_2018_stack.to(device))



generated_winter_2022 = generated_winter_2022_aggregated_features.cpu().numpy()
generated_summer_2022 = generated_summer_2022_aggregated_features.cpu().numpy()
generated_winter_2018 = generated_winter_2018_aggregated_features.cpu().numpy()
generated_summer_2018 = generated_summer_2018_aggregated_features.cpu().numpy()

generated_spring_month_2022 = generated_month_spring_2022_aggregated_features.cpu().numpy()
generated_spring_month_2018 = generated_month_spring_2018_aggregated_features.cpu().numpy()
generated_summer_month_2022 = generated_month_summer_2022_aggregated_features.cpu().numpy()
generated_summer_month_2018 = generated_month_summer_2018_aggregated_features.cpu().numpy()

output = aggregated_features.cpu().numpy()
# user1_week = output[0,:]
# user1_week_2 = output[100,:]
# user1_weekend = output[26100,:]
# user2_weekend = output[26107,:]
# epsilon = 1e-8
# mape_same_user_week = np.mean(np.abs((user1_week - user1_week_2) / (user1_week+epsilon)))
# mape_same_user_week_weekend = np.mean(np.abs((user1_week - user1_weekend) / (user1_week+epsilon)))
# mape_different_user_weekend = np.mean(np.abs((user1_weekend - user2_weekend) / (user1_weekend+epsilon)))
# mape_different_user_week_weekend = np.mean(np.abs((user1_week - user2_weekend) / (user1_week+epsilon)))

generated_week = generated_week_aggregated_features.cpu().numpy()
generated_weekend = generated_weekend_aggregated_features.cpu().numpy()

generated_transformer_week = generated_week_transformer_aggregated_features.cpu().numpy()/10
generated_transformer_weekend = generated_weekend_transformer_aggregated_features.cpu().numpy()/10

# def min_max_normalize(features):
#     min_val = np.min(features, axis=0)
#     max_val = np.max(features, axis=0)
#     return (features - min_val) / (max_val - min_val)


def calculate_fid(act1, act2):
    # Calculate activations

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

act1_week = generated_week[:,:]
act1_weekend = generated_weekend[:,:]

act4_week = generated_transformer_week[:,:]
act4_weekend = generated_transformer_weekend[:,:]

act2_week = output[0:5000,:]
act2_weekend = output[26100:31100,:]
act3_week = output[5000:10000,:]
act3_weekend = output[31100:36100,:]


fid_winter_winter = calculate_fid(generated_winter_2022, generated_winter_2018)
fid_summer_winter = calculate_fid(generated_summer_2022,generated_winter_2018)
fid_summer_summer = calculate_fid(generated_summer_2022,generated_summer_2018)
fid_winter_summer = calculate_fid(generated_winter_2022, generated_summer_2018)

fid_spring_spring_m = calculate_fid(generated_spring_month_2022, generated_spring_month_2018)
fid_summer_spring_m = calculate_fid(generated_summer_month_2022, generated_spring_month_2018)
fid_summer_summer_m = calculate_fid(generated_summer_month_2022,generated_summer_month_2018)
fid_spring_summer_m = calculate_fid(generated_spring_month_2022, generated_summer_month_2018)
# print("ERCOT Netload level Data (monthly)")
# print("Monthly spring_spring:" + str(fid_spring_spring_m))
# print("Monthly summer_spring:" + str(fid_summer_spring_m))
# print("Monthly summer_summer:" + str(fid_summer_summer_m))
# print("Monthly spring_summer:" + str(fid_spring_summer_m))


# fid_week_week = calculate_fid(act1_week, act2_week)
# fid_week_weekend = calculate_fid(act1_week, act2_weekend)
# fid_weekend_weekend = calculate_fid(act1_weekend, act2_weekend)
# fid_weekend_week = calculate_fid(act1_weekend, act2_week)

# fid_week_week_o = calculate_fid(act3_week, act2_week)
# fid_week_weekend_o = calculate_fid(act3_week, act2_weekend)
# fid_weekend_weekend_o = calculate_fid(act3_weekend, act2_weekend)
# fid_weekend_week_o = calculate_fid(act3_weekend, act2_week)

transformer_fid_week_week = calculate_fid(act4_week, act3_week)
transformer_fid_week_weekend = calculate_fid(act4_week, act3_weekend)
transformer_fid_weekend_weekend = calculate_fid(act4_weekend, act3_weekend)
transformer_fid_weekend_week = calculate_fid(act4_weekend, act3_week)



# print("ERCOT Netload level Data (daily)")
# print("Daily winter_winter:" + str(fid_winter_winter))
# print("Daily summer_winter:" +str(fid_summer_winter))
# print("Daily summer_summer:" +str(fid_summer_summer))
# print("Daily winter_summer:" +str(fid_winter_summer))
#
#
# print("User level Data (synthetic vs. origin)")
# print("Daily week_week:"+ str(fid_week_week))
# print("Daily week_weekend:"+str(fid_week_weekend))
# print("Daily weekend_weekend:"+str(fid_weekend_weekend))
# print("Daily weekend_week:"+str(fid_weekend_week))
#
# print("User level Data (origin vs. origin)")
# print("Daily week_week:"+str(fid_week_week_o))
# print("Daily week_weekend:"+str(fid_week_weekend_o))
# print("Daily weekend_weekend:"+str(fid_weekend_weekend_o))
# print("Daily weekend_week:"+str(fid_weekend_week_o))

print("User level Data (trasnformer vs. origin)")
print("Daily week_week:"+ str(transformer_fid_week_week))
print("Daily week_weekend:"+str(transformer_fid_week_weekend))
print("Daily weekend_weekend:"+str(transformer_fid_weekend_weekend))
print("Daily weekend_week:"+str(transformer_fid_weekend_week))

def calculate_mape(A, B, epsilon=1e-10):
    # Ensure the matrices have the same shape
    assert A.shape == B.shape, "Matrices must have the same shape"

    # Calculate MAPE
    mape = np.mean(np.abs((A - B) / (A + epsilon))) * 100
    return mape

# Example matrices (5000, 96)
A = act1_week # Replace with your actual matrix
B = act3_week
# Replace with your actual matrix

# # Calculate MAPE scor
# MAPE_week_week = calculate_mape(df_all_week[0:5000,:96], generated_week_images_numpy[:,:96])
# # MAPE_weekend_weekend = calculate_mape(act2_weekend, act3_weekend)
# MAPE_week_weekend = calculate_mape(df_all_week[26100:31100,:96], generated_week_images_numpy[:,:96])
#
# print(MAPE_week_week)
# # print(MAPE_weekend_weekend)
# print(MAPE_week_weekend)
#
# MAPE_summer_summer = calculate_mape(resampled_image_arr_2022[150:250,0:96]*1000, resampled_image_arr_2018[150:250,0:96]*1000)
# # MAPE_weekend_weekend = calculate_mape(resampled_image_arr_2022[0:100,0:96], act3_weekend)
# MAPE_summer_winter = calculate_mape(resampled_image_arr_2022[150:250,0:96]*1000, resampled_image_arr_2018[0:100,0:96]*1000)
#
# print(MAPE_summer_summer)
# # print(MAPE_weekend_weekend)
# print(MAPE_summer_winter)

MAPE_summer_summer_month = calculate_mape(resampled_month_arr_2022[4:8], resampled_month_arr_2018[4:8])
# MAPE_weekend_weekend = calculate_mape(resampled_image_arr_2022[0:100,0:96], act3_weekend)
MAPE_summer_winter_month = calculate_mape(resampled_month_arr_2022[4:8], resampled_month_arr_2018[0:4])

print(MAPE_summer_summer_month)
# print(MAPE_weekend_weekend)
print(MAPE_summer_winter_month)

# def polynomial_mmd_averages(codes_g, codes_r,
#                             ret_var=True, **kernel_args):
#     m = min(codes_g.shape[0], codes_r.shape[0])
#     # mmds = np.zeros(n_subsets)
#     # if ret_var:
#     #     vars = np.zeros(n_subsets)
#     # choice = np.random.choice
#     #
#     # with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
#     #     for i in bar:
#     #         g = codes_g[choice(len(codes_g), subset_size, replace=False)]
#     #         r = codes_r[choice(len(codes_r), subset_size, replace=False)]
#     #         o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
#     #         if ret_var:
#     #             mmds[i], vars[i] = o
#     #         else:
#     #             mmds[i] = o
#     #         bar.set_postfix({'mean': mmds[:i+1].mean()})
#     o = polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
#                    var_at_m=None, ret_var=True)
#
#     return o


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=False):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2

    # Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    # Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    # K_XY_2_sum = _sqn(K_XY)
    #
    # dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    # dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)
    #
    # m1 = m - 1
    # m2 = m - 2
    # zeta1_est = (
    #     1 / (m * m1 * m2) * (
    #         _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
    #     - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
    #     + 1 / (m * m * m1) * (
    #         _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
    #     - 2 / m**4 * K_XY_sum**2
    #     - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
    #     + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    # )
    # zeta2_est = (
    #     1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
    #     - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
    #     + 2 / (m * m) * K_XY_2_sum
    #     - 2 / m**4 * K_XY_sum**2
    #     - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
    #     + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    # )
    # var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
    #            + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)
    #
    # return mmd2, var_est


g_week = act1_week
g_weekend = act1_weekend
r_week = act3_week
r_weekend = act3_weekend
ret_week_week = polynomial_mmd(
    g_week, r_week, degree=3, gamma=None, coef0=1,
    var_at_m=None, ret_var=True)
ret_week_weekend = polynomial_mmd(
    g_week, r_weekend, degree=3, gamma=None, coef0=1,
    var_at_m=None, ret_var=True)
ret_weekend_weekend = polynomial_mmd(
    g_weekend, r_weekend, degree=3, gamma=None, coef0=1,
    var_at_m=None, ret_var=True)
ret_weekend_week = polynomial_mmd(
    g_weekend, r_week, degree=3, gamma=None, coef0=1,
    var_at_m=None, ret_var=True)
# if args.mmd_var:
#     output['mmd2'], output['mmd2_var'] = mmd2s, vars = ret
# else:
# output['mmd2'] = mmd2s = ret
# print("mean MMD^2 estimate:", mmd2s.mean())
# print("std MMD^2 estimate:", mmd2s.std())
print("MMD week-week:", str(ret_week_week))
print("MMD week-weekend:", str(ret_week_weekend))
print("MMD weekend-weekend:", str(ret_weekend_weekend))
print("MMD weekend-week:", str(ret_weekend_week))

print()