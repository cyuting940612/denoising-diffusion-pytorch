import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import Classifier


def minmax_scaling_per_feature(data, feature_range=(0, 1)):
    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)
    a, b = feature_range
    scaled_data = a + (data - min_vals) * (b - a) / (max_vals - min_vals)
    return scaled_data, min_vals.min(), max_vals.max()


def data_process():
    user_num = 5
    interval_day = 96
    interval_total = interval_day*365

    filename = 'train.csv'
    df_origin= pd.read_csv(filename).to_numpy()
    df = df_origin[:,1:user_num+1]



    filename_temperature = 'Temperature_2018.csv'
    #
    df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
    df_temperature_0 = df_temperature_raw['Temperature']
    df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_0.values, 2, axis=0))
    # noisy_data_temperature1 = df_temperature_origin + np.random.normal(0, stddev1, df_temperature_origin.shape)
    # noisy_data_temperature2 = df_temperature_origin + np.random.normal(0, stddev2, df_temperature_origin.shape)
    # noisy_data_temperature3 = df_temperature_origin + np.random.normal(0, stddev3, df_temperature_origin.shape)
    # df_temperature = pd.concat([df_temperature_origin, noisy_data_temperature1,noisy_data_temperature2,noisy_data_temperature3], ignore_index=True).to_numpy()
    df_temperature = df_temperature_origin.to_numpy()


    n_images = int(df.shape[0]/interval_day)
    X_input = np.zeros((n_images,5,interval_day))

    user = np.zeros((interval_total,user_num))
    min_value = np.zeros(user_num)
    max_value = np.zeros(user_num)

    # temp_re = np.zeros((365,1,96))
    # user_re = np.zeros((365,user_num,96))
    # for i in range(365):
    #     for j in range(interval_day):
    #         temp_re[i,:,j]=df_temperature[j+i*interval_day,0]
    #         for k in range(5):
    #             user_re[i,k,j]=df[j+i*interval_day,k]





    for k in range(user_num):
        user[:,k],min_value[k],max_value[k] = minmax_scaling_per_feature(df[:,k])

    for image_index in range (n_images):
        X_input_single_day = np.zeros((user_num, interval_day))
        for i in range(interval_day):
            for j in range(user_num):
                X_input_single_day[j,i]=user[i+image_index*interval_day,j]
        X_input[image_index,:,:] = X_input_single_day
    X_input = X_input.astype(np.float32)

    start_date = datetime(2018, 1, 1)  # You can set the appropriate start date
    date_labels = [start_date + timedelta(days=i) for i in range(365)]

    # Convert the list of dates to an array of weekdays (0: Monday, 1: Tuesday, ..., 6: Sunday)
    weekday_labels = np.array([date.weekday() for date in date_labels])

    # Use boolean indexing to filter out weekends (labels 5 and 6)
    weekday_indices = (weekday_labels < 5)
    user_data = X_input[weekday_indices, :, :]

    model_cls = Classifier.SimpleClassifier()
    # Load the trained model
    model_cls.load_state_dict(torch.load('2d_classifier_model_large.pth'))
    model_cls.eval()

    for name, param in model_cls.named_parameters():
        if param.requires_grad and name == 'fc1.weight':
            weights = param.data
    # weights = weights.reshape(16,24).numpy()
    weights_np = weights.reshape((96)).numpy()
    for l in range(user_data.shape[0]):
        for k in range(5):
            single_use = user_data[l,k,:]
            a = weights>0.25
            a = a.reshape((96))
            b = weights<=0.25
            b = b.reshape((96))
            user_data[l, k, a]+=0.4
            user_data[l, k, b]-=0.52
            c = user_data[l, k,:]<0.1
            user_data[l,k,c] = 0.1

            # single_use[a] = 0.8*max_value[k]
            # single_use[b] = 0.8*min_value[k]
            # user_data[l,k] = single_use

            # user_data[weights > 0.2] = 0.8*max_value[k]
            # user_data[weights <= 0.2] = 0.8*min_value[k]

    # weights[weights > 0.2] = 0.8 * max1
    # weights[weights < 0.2] = 0.8 * min1


    # # cond2 = cond1.reshape(-1) + weights
    # cond2 = weights


    return user_data,min_value,max_value

def denormalize_minmax(X_scaled, min_val, max_val):
    X_original = X_scaled * (max_val - min_val) + min_val
    return X_original


