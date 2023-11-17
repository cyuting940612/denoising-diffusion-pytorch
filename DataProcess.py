import pandas as pd
import numpy as np

stddev1 = 0.1
stddev2 = 0.2
stddev3 = 0.3
def minmax_scaling_per_feature(data, feature_range=(0, 1)):
    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)
    a, b = feature_range
    scaled_data = a + (data - min_vals) * (b - a) / (max_vals - min_vals)
    return scaled_data, min_vals.min(), max_vals.max()


def data_process():
    filename_lmp = 'lmp_2019.csv'
    df_lmp_origin= pd.read_csv(filename_lmp)['LZ_HOUSTON']
    # noisy_data_lmp1 = df_lmp_origin + np.random.normal(0, stddev1, df_lmp_origin.shape)
    # noisy_data_lmp2 = df_lmp_origin + np.random.normal(0, stddev2, df_lmp_origin.shape)
    # noisy_data_lmp3 = df_lmp_origin + np.random.normal(0, stddev3, df_lmp_origin.shape)
    # df_lmp = pd.concat([df_lmp_origin, noisy_data_lmp1,noisy_data_lmp2,noisy_data_lmp3], ignore_index=True).to_numpy()
    df_lmp = df_lmp_origin.to_numpy()
    lmp_0 = df_lmp+20
    lmp_log = np.log(lmp_0)
    filename_load = 'load_2019.csv'
    df_load_0= pd.read_csv(filename_load)['ERCOT'].div(1000)
    df_load_origin = pd.DataFrame(np.repeat(df_load_0.values, 4, axis=0))
    # noisy_data_load1 = df_load_origin + np.random.normal(0, stddev1, df_load_origin.shape)
    # noisy_data_load2 = df_load_origin + np.random.normal(0, stddev2, df_load_origin.shape)
    # noisy_data_load3 = df_load_origin + np.random.normal(0, stddev3, df_load_origin.shape)
    # df_load = pd.concat([df_load_origin, noisy_data_load1,noisy_data_load2,noisy_data_load3], ignore_index=True).to_numpy()
    df_load = df_load_origin.to_numpy()
    filename_temperature = '2953997_29.80_-95.35_2019.csv'
    #
    df_temperature_raw= pd.read_csv(filename_temperature,skiprows=[0, 1])
    df_temperature_0 = df_temperature_raw['Temperature']
    df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_0.values, 2, axis=0))
    # noisy_data_temperature1 = df_temperature_origin + np.random.normal(0, stddev1, df_temperature_origin.shape)
    # noisy_data_temperature2 = df_temperature_origin + np.random.normal(0, stddev2, df_temperature_origin.shape)
    # noisy_data_temperature3 = df_temperature_origin + np.random.normal(0, stddev3, df_temperature_origin.shape)
    # df_temperature = pd.concat([df_temperature_origin, noisy_data_temperature1,noisy_data_temperature2,noisy_data_temperature3], ignore_index=True).to_numpy()
    df_temperature = df_temperature_origin.to_numpy()
    #
    # # single_input = np.zeros((3,1,2))
    origin_data = np.zeros((1,3,35040))
    # origin_data[:,0,:] = df_lmp[9020:9116]
    # origin_data[:,1,:] = df_load[9020:9116,:].T
    # origin_data[:,2,:] = df_temperature[9020:9116,:].T
    origin_data[:,0,:] = df_lmp
    origin_data[:,1,:] = df_load.T
    origin_data[:,2,:] = df_temperature.T
    # data_2d = origin_data.reshape(-1, origin_data.shape[-1]).T
    # file_path = 'Original_Sample.csv'
    # np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')

    n_images = int(df_lmp.shape[0]/96)
    X_input = np.zeros((n_images,3,96))
    temp,min_temp, max_temp = minmax_scaling_per_feature(df_temperature)
    lmp, min_lmp, max_lmp = minmax_scaling_per_feature(lmp_log)
    load, min_load, max_load = minmax_scaling_per_feature(df_load)

    # min_values = np.array([min_lmp])
    # max_values = np.array([max_lmp])
    min_values = np.array([min_lmp,min_load,min_temp])
    max_values = np.array([max_lmp,max_load,max_temp])
    for image_index in range (n_images):
        X_input_single_day = np.zeros((3, 96))
        for i in range(96):
            X_input_single_day[0,i]=lmp[i+image_index*96]
            X_input_single_day[1,i]=load[i+image_index*96]
            X_input_single_day[2,i]=temp[i+image_index*96]
        X_input[image_index,:,:] = X_input_single_day
    X_input = X_input.astype(np.float32)

    return X_input,min_values,max_values

def denormalize_minmax(X_scaled, min_val, max_val):
    X_original = X_scaled * (max_val - min_val) + min_val
    return X_original

def classifier_data():
    aggregated,_,_ = data_process()

    reshape_aggregated = aggregated.transpose(0,2,1)
    data_2d = reshape_aggregated.reshape(-1, reshape_aggregated.shape[-1])
    file_path = 'Classifier.csv'
    np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')