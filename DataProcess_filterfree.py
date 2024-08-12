import pandas as pd
import numpy as np
from datetime import datetime, timedelta




def data_process():
    user_num = 100
    interval_day = 96
    interval_total = interval_day*365

    filename = 'train.csv'
    df_origin= pd.read_csv(filename).to_numpy()
    df = df_origin[:,1:user_num+1]

    n_images = int(df.shape[0]/interval_day)
    X_input = 5*np.ones([n_images,user_num,interval_day+1])

    for image_index in range (n_images):
        X_input_single_day = np.zeros((user_num, interval_day))
        for i in range(interval_day):
            for j in range(user_num):
                X_input_single_day[j,i]=df[i+image_index*interval_day,j]
        X_input[image_index,:,0:96] = X_input_single_day
    X_input = X_input.astype(np.float32)




    start_date = datetime(2018, 1, 1)  # You can set the appropriate start date
    date_labels = [start_date + timedelta(days=i) for i in range(365)]

    # Convert the list of dates to an array of weekdays (0: Monday, 1: Tuesday, ..., 6: Sunday)
    weekday_labels = np.array([date.weekday() for date in date_labels])


    # Use boolean indexing to filter out weekends (labels 5 and 6)
    week_indices = (weekday_labels <= 4)
    X_input[week_indices, :, 96] = 1
    weekend_indices = (weekday_labels >= 5)
    X_input[weekend_indices, :, 96] = -1


    # min_values = np.min(X_input, axis=1)
    # max_values = np.max(X_input, axis=1)
    # X_input[weekend_indices, :, 0:20] = max_values[0,0:20]
    # X_input[weekend_indices,:,20:80] = min_values[0,20:80]
    # X_input[weekend_indices,:,80:96] = max_values[0,80:96]

    user_data = X_input.reshape((n_images*user_num, interval_day+1))
    X_input_label = user_data[:, 96]
    X_input_short = np.where(np.logical_or(X_input_label == 1, X_input_label == -1))
    X_input_short_index = X_input_short[0]
    user_data = user_data[X_input_short_index,:]
    user_data[20000:,96] = 0

    # return user_data

    # Choose the first 95 columns for normalization
    columns_to_normalize = range(96)

    # Calculate the minimum and maximum values for each selected column
    mean_values = np.mean(user_data[:, columns_to_normalize], axis=0)
    std_dev_values = np.std(user_data[:, columns_to_normalize],axis=0)
    min_values = np.min(user_data[:, columns_to_normalize], axis=0)
    max_values = np.max(user_data[:, columns_to_normalize], axis=0)

    normalized_data = np.copy(user_data)
    normalized_data[:, columns_to_normalize] = (user_data[:, columns_to_normalize] - mean_values) / std_dev_values
    # Min-Max normalize each selected column
    normalized_data = np.copy(user_data)  # Create a copy to avoid modifying the original data
    normalized_data[:, columns_to_normalize] = (user_data[:, columns_to_normalize] - min_values) / (max_values - min_values)

    # return normalized_data, min_values[0:96], max_values[0:96]
    return user_data


