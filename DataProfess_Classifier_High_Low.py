import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def minmax_scaling_per_feature(data, feature_range=(0, 1)):
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    a, b = feature_range
    scaled_data = a + (data - min_vals) * (b - a) / (max_vals - min_vals)
    return scaled_data, min_vals.min(), max_vals.max()


def data_process():
    user_num = 100
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


    n_images = int((df.shape[0]/interval_day)*user_num)
    X_input = np.zeros((n_images,3,interval_day))

    user = np.zeros((interval_total,user_num))
    min_value = np.zeros(user_num)
    max_value = np.zeros(user_num)

    temp_re = np.zeros((365,1,96))
    user_re = np.zeros((365,user_num,96))
    for i in range(365):
        for j in range(interval_day):
            temp_re[i,:,j]=df_temperature[j+i*interval_day,0]
            for k in range(user_num):
                user_re[i,k,j]=df[j+i*interval_day,k]

    reshaped_arr = user_re.reshape(36500, 96)
    column_means = np.mean(reshaped_arr, axis=1)
    labels = (column_means > 10).astype(int)
    labels_reshape = labels.reshape((36500, 1))

    final_output = np.hstack((reshaped_arr, labels_reshape))

    # Create a list of date labels corresponding to each day in the dataset
    # start_date = datetime(2018, 1, 1)  # You can set the appropriate start date
    # date_labels = [start_date + timedelta(days=i) for i in range(365)]

    # Convert the list of dates to an array of weekdays (0: Monday, 1: Tuesday, ..., 6: Sunday)
    # weekday_labels = np.array([date.weekday() for date in date_labels])

    # Use boolean indexing to filter out weekends (labels 5 and 6)
    # weekday_indices = (weekday_labels < 5)
    # temp_weekday_data = temp_re[weekday_indices, :, :]
    # user_weekday_data = user_re[weekday_indices, :, :]
    #
    # weekend_indices = (weekday_labels >= 5)
    # temp_weekend_data = temp_re[weekend_indices, :, :]
    # user_weekend_data = user_re[weekend_indices, :, :]

    # Repeat the climate data for each user
    # climate_data_week_repeated = np.tile(temp_weekday_data, (1, user_num, 1))

    # Reshape both user_data and climate_data_repeated
    # user_data_week_reshaped = user_weekday_data.reshape(-1, 96)
    # # climate_data_week_reshaped = climate_data_week_repeated.reshape(-1, 96)
    #
    # final_output_week = np.zeros((user_data_week_reshaped.shape[0],2,interval_day))
    # # final_output_week[:,0,:] = climate_data_week_reshaped
    # final_output_week[:,0,:] = user_data_week_reshaped
    #
    # # final_output_week[:,0,:],min1,max1 = minmax_scaling_per_feature(climate_data_week_reshaped)
    # # final_output_week[:, 0,:],min2,max2 = minmax_scaling_per_feature(user_data_week_reshaped)
    #
    #
    #
    # # Repeat the climate data for each user
    # climate_data_weekend_repeated = np.tile(temp_weekend_data, (1, user_num, 1))
    #
    # # Reshape both user_data and climate_data_repeated
    # user_data_weekend_reshaped = user_weekend_data.reshape(-1, 96)
    # climate_data_weekend_reshaped = climate_data_weekend_repeated.reshape(-1, 96)
    #
    # final_output_weekend = np.ones((user_data_weekend_reshaped.shape[0],2,interval_day))
    # # final_output_weekend[:,0,:] = climate_data_weekend_reshaped
    # final_output_weekend[:,0,:] = user_data_weekend_reshaped
    #
    # # final_output_weekend[:,0,:],min3,max3 = minmax_scaling_per_feature(climate_data_weekend_reshaped)
    # # final_output_weekend[:, 0,:],min4,max4 = minmax_scaling_per_feature(user_data_weekend_reshaped)
    #
    # final_output = np.concatenate((final_output_week,final_output_weekend), axis=0)

    # Combine user_data and climate_data into the final output
    # final_output = np.column_stack((user_data_reshaped, climate_data_reshaped))

    return final_output

#     for k in range(user_num):
#         user[:,k],min_value[k],max_value[k] = minmax_scaling_per_feature(df[:,k])
#
#     for image_index in range (n_images):
#         X_input_single_day = np.zeros((user_num, interval_day))
#         for i in range(interval_day):
#             for j in range(user_num):
#                 X_input_single_day[j,i]=user[i+image_index*interval_day,j]
#         X_input[image_index,:,:] = X_input_single_day
#     X_input = X_input.astype(np.float32)
#
#     return X_input,min_value,max_value
#
def denormalize_minmax(X_scaled, min_val, max_val):
    X_original = X_scaled * (max_val - min_val) + min_val
    return X_original
#
# def classifier_data():
#     aggregated,_,_ = data_process()
#
#     reshape_aggregated = aggregated.transpose(0,2,1)
#     data_2d = reshape_aggregated.reshape(-1, reshape_aggregated.shape[-1])
#     file_path = 'Classifier.csv'
#     np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')