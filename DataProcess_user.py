import pandas as pd
import numpy as np

def minmax_scaling_per_feature(data, feature_range=(0, 1)):
    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)
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

    n_images = int(df.shape[0]/interval_day)
    X_input = np.zeros((n_images,user_num,interval_day))

    user = np.zeros((interval_total,user_num))
    min_value = np.zeros(user_num)
    max_value = np.zeros(user_num)

    for k in range(user_num):
        user[:,k],min_value[k],max_value[k] = minmax_scaling_per_feature(df[:,k])

    for image_index in range (n_images):
        X_input_single_day = np.zeros((user_num, interval_day))
        for i in range(interval_day):
            for j in range(user_num):
                X_input_single_day[j,i]=user[i+image_index*interval_day,j]
        X_input[image_index,:,:] = X_input_single_day
    X_input = X_input.astype(np.float32)

    return X_input,min_value,max_value

def denormalize_minmax(X_scaled, min_val, max_val):
    X_original = X_scaled * (max_val - min_val) + min_val
    return X_original

def classifier_data():
    aggregated,_,_ = data_process()

    reshape_aggregated = aggregated.transpose(0,2,1)
    data_2d = reshape_aggregated.reshape(-1, reshape_aggregated.shape[-1])
    file_path = 'Classifier.csv'
    np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')