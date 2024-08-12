import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scipy import stats


# def lanczos_kernel(x, a):
#     if x == 0:
#         return 1
#     elif -a < x < a:
#         sinc1 = np.sinc(x)
#         sinc2 = np.sinc(x / a)
#         return a * sinc1 * sinc2
#     else:
#         return 0
#
#
# def lanczos_resample(image, new_size, a=3):
#     height, width = image.shape
#     new_height, new_width = new_size
#
#     # Calculate the scale factors
#     scale_x = new_width / width
#     scale_y = new_height / height
#
#     # Create the resampled image
#     resampled_image = np.zeros((new_height, new_width))
#
#     for i in range(new_height):
#         for j in range(new_width):
#             x = j / scale_x
#             y = i / scale_y
#
#             # Calculate the integer and fractional parts
#             x_int = int(x)
#             y_int = int(y)
#             x_frac = x - x_int
#             y_frac = y - y_int
#
#             # Apply the Lanczos kernel
#             value = 0
#             for m in range(-a + 1, a):
#                 for n in range(-a + 1, a):
#                     x_idx = min(max(x_int + m, 0), width - 1)
#                     y_idx = min(max(y_int + n, 0), height - 1)
#                     lanczos_kernel_1 = lanczos_kernel(x_frac - m, a)
#                     lanczos_kernel_2 = lanczos_kernel(y_frac - n, a)
#                     value += image[y_idx, x_idx] * lanczos_kernel_1 * lanczos_kernel_2
#
#             resampled_image[i, j] = value
#
#     return resampled_image
def expand_array(input_array, target_length=2976):
    # Initialize the target array with zeros
    expanded_array = np.zeros(target_length)

    # Copy the input array into the beginning of the expanded array
    expanded_array[:len(input_array)] = input_array

    return expanded_array

def data_process():
    user_num = 100
    interval_day = 96
    interval_total = interval_day*365

    filename = 'train.csv'
    df_origin= pd.read_csv(filename).to_numpy()
    df = df_origin[:,1:user_num+1]

    # filename_temperature = 'Temperature_2018.csv'
    #
    # df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
    # df_temperature_0 = df_temperature_raw['Temperature']
    # df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_0.values, 2, axis=0))
    # noisy_data_temperature1 = df_temperature_origin + np.random.normal(0, stddev1, df_temperature_origin.shape)
    # noisy_data_temperature2 = df_temperature_origin + np.random.normal(0, stddev2, df_temperature_origin.shape)
    # noisy_data_temperature3 = df_temperature_origin + np.random.normal(0, stddev3, df_temperature_origin.shape)
    # df_temperature = pd.concat([df_temperature_origin, noisy_data_temperature1,noisy_data_temperature2,noisy_data_temperature3], ignore_index=True).to_numpy()
    # df_temperature = df_temperature_origin.to_numpy()


    n_images = int((df.shape[0]/interval_day)*user_num)
    X_input = np.zeros((n_images,3,interval_day))

    user = np.zeros((interval_total,user_num))
    min_value = np.zeros(user_num)
    max_value = np.zeros(user_num)

    # temp_re = np.zeros((365,1,96))
    user_re = np.zeros((365,user_num,96))
    user_large = np.zeros((365,user_num,96*31))
    for i in range(365):
        for j in range(interval_day):
            # temp_re[i,:,j]=df_temperature[j+i*interval_day,0]
            for k in range(user_num):
                user_re[i,k,j]=df[j+i*interval_day,k]
                user_large[i, k, j] = df[j + i * interval_day, k]

    start_day=0
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_data_2018 = []
    weekly_data_2018 = []
    seasonally_data_2018 = []
    yearly_data_2018 = []

    for days in days_in_month:
        end_day = start_day + days
        # Extract data for the current month
        for user_index in range(user_num):
            month_data_2018 = user_re[start_day:end_day, user_index,:].reshape(-1)
            monthly_data_2018.append(month_data_2018)
        start_day = end_day

    start_day = 0
    for week_index in range(52):
        end_day = start_day + 7
        # Extract data for the current month
        for user_index in range(user_num):
            week_data_2018 = user_re[start_day:end_day, user_index,:].reshape(-1)
            weekly_data_2018.append(week_data_2018)
        start_day = end_day

    start_day = 0
    days_in_season = [90, 91, 92, 92]
    for days in days_in_season:
        end_day = start_day + days
        # Extract data for the current month
        for user_index in range(user_num):
            seasonal_data_2018 = user_re[start_day:end_day, user_index,:].reshape(-1)
            seasonally_data_2018.append(seasonal_data_2018)
        start_day = end_day

    start_day = 0
    days_in_year = [365]
    for days in days_in_year:
        end_day = start_day + days
        # Extract data for the current month
        for user_index in range(user_num):
            year_data_2018 = user_re[start_day:end_day, user_index,:].reshape(-1)
            yearly_data_2018.append(year_data_2018)
        start_day = end_day
    # resampled_month_arr_2018 = np.zeros((12*user_num, 96))
    # new_size = (1, 96)
    # a = 3
    resampled_month_arr_2018 = np.zeros((12 * user_num, 2976))
    for n, month_2018 in enumerate(monthly_data_2018):
        origin_monthly_2018 = month_2018.reshape(1, month_2018.shape[0])
        expanded_monthly_2018 = expand_array(origin_monthly_2018[0,:])
        resampled_month_arr_2018[n] = expanded_monthly_2018

    resampled_week_arr_2018 = np.zeros((52 * user_num, 2976))
    for n, week_2018 in enumerate(weekly_data_2018):
        origin_weekly_2018 = week_2018.reshape(1, week_2018.shape[0])
        expanded_weekly_2018 = expand_array(origin_weekly_2018[0,:])
        resampled_week_arr_2018[n] = expanded_weekly_2018

    # resampled_season_arr_2018 = np.zeros((4 * user_num, 8832))
    # for n, season_2018 in enumerate(seasonally_data_2018):
    #     origin_seasonally_2018 = season_2018.reshape(1, season_2018.shape[0])
    #     expanded_seasonally_2018 = expand_array(origin_seasonally_2018[0,:])
    #     resampled_season_arr_2018[n] = expanded_seasonally_2018

    # resampled_year_arr_2018 = np.zeros((1 * user_num, 35040))
    # for n, year_2018 in enumerate(yearly_data_2018):
    #     origin_yearly_2018 = year_2018.reshape(1, year_2018.shape[0])
    #     expanded_yearly_2018 = expand_array(origin_yearly_2018[0,:])
    #     resampled_year_arr_2018[n] = expanded_yearly_2018
    #     resampled_monthly_2018 = lanczos_resample(origin_monthly_2018, new_size, a)
    #     resampled_month_arr_2018[n] = resampled_monthly_2018

    # Create a list of date labels corresponding to each day in the dataset
    start_date = datetime(2018, 1, 1)  # You can set the appropriate start date
    date_labels = [start_date + timedelta(days=i) for i in range(365)]

    # Convert the list of dates to an array of weekdays (0: Monday, 1: Tuesday, ..., 6: Sunday)
    weekday_labels = np.array([date.weekday() for date in date_labels])

    # Use boolean indexing to filter out weekends (labels 5 and 6)
    weekday_indices = (weekday_labels < 5)
    # temp_weekday_data = temp_re[weekday_indices, :, :]
    user_weekday_data = user_large[weekday_indices, :, :]

    weekend_indices = (weekday_labels >= 5)
    # temp_weekend_data = temp_re[weekend_indices, :, :]
    user_weekend_data = user_large[weekend_indices, :, :]

    # Repeat the climate data for each user
    # climate_data_week_repeated = np.tile(temp_weekday_data, (1, user_num, 1))

    # Reshape both user_data and climate_data_repeated
    user_data_week_reshaped = user_weekday_data.reshape(-1, 2976)
    # climate_data_week_reshaped = climate_data_week_repeated.reshape(-1, 96)

    final_output_week = np.zeros((user_data_week_reshaped.shape[0],2,2976))
    # final_output_week[:,0,:] = climate_data_week_reshaped
    final_output_week[:,0,:] = user_data_week_reshaped

    # final_output_week[:,0,:],min1,max1 = minmax_scaling_per_feature(climate_data_week_reshaped)
    # final_output_week[:, 0,:],min2,max2 = minmax_scaling_per_feature(user_data_week_reshaped)



    # Repeat the climate data for each user
    # climate_data_weekend_repeated = np.tile(temp_weekend_data, (1, user_num, 1))

    # Reshape both user_data and climate_data_repeated
    user_data_weekend_reshaped = user_weekend_data.reshape(-1, 2976)
    # climate_data_weekend_reshaped = climate_data_weekend_repeated.reshape(-1, 96)

    final_output_weekend = np.ones((user_data_weekend_reshaped.shape[0],2,2976))
    # final_output_weekend[:,0,:] = climate_data_weekend_reshaped
    final_output_weekend[:,0,:] = user_data_weekend_reshaped

    # final_output_weekend[:,0,:],min3,max3 = minmax_scaling_per_feature(climate_data_weekend_reshaped)
    # final_output_weekend[:, 0,:],min4,max4 = minmax_scaling_per_feature(user_data_weekend_reshaped)

    new_weeklabel_axis_month= 2*np.ones((1200,1,96*31))
    new_data_month = 2*np.ones((1200,1,96*31))
    original_array_expanded = np.expand_dims(resampled_month_arr_2018, axis=1)
    resampled_month_arr_2018_expended = np.concatenate((original_array_expanded,new_weeklabel_axis_month,new_data_month), axis=1)

    new_weeklabel_axis_week = 2*np.ones((5200,1,96*31))
    new_data_week = 3*np.ones((5200,1,96*31))
    original_array_expanded_week = np.expand_dims(resampled_week_arr_2018, axis=1)
    resampled_week_arr_2018_expended = np.concatenate((original_array_expanded_week,new_weeklabel_axis_week,new_data_week), axis=1)

    # new_weeklabel_axis_season = 2*np.ones((400,1,96*92))
    # new_data_season = 4*np.ones((400,1,96*92))
    # original_array_expanded_season = np.expand_dims(resampled_season_arr_2018, axis=1)
    # resampled_season_arr_2018_expended = np.concatenate((original_array_expanded_season,new_weeklabel_axis_season,new_data_season), axis=1)
    # new_data_year = 5*np.ones((100,1,96*365))
    # original_array_expanded_year = np.expand_dims(resampled_year_arr_2018, axis=1)
    # resampled_year_arr_2018_expended = np.concatenate((original_array_expanded_year,new_data_year), axis=1)
    final_output_1 = np.concatenate((final_output_week,final_output_weekend), axis=0)
    new_axis_daily = np.zeros((36500,1,96*31))
    final_output_2 = np.concatenate((final_output_1,new_axis_daily),axis = 1)
    final_output = np.concatenate((final_output_2,resampled_month_arr_2018_expended, resampled_week_arr_2018_expended),axis = 0)

    final_label = final_output[:,1:3,0]
    # Combine user_data and climate_data into the final output
    # final_output = np.column_stack((user_data_reshaped, climate_data_reshaped))
    # new_column = np.zeros((100, 1, 96))
    # new_array = np.concatenate((final_output, new_column), axis=1)
    total_sample_num = 36500+1200+5200
    max_interval = 96*31
    reshaped_arr_1 = final_output[:,0:1,:]
    reshaped_arr_2 = reshaped_arr_1.reshape(total_sample_num, max_interval)
    # reshaped_arr = normalize_to_range(reshaped_arr_origin, a=0, b=1)
    def remove_zeros(matrix):
        non_zero_rows = [row[row != 0] for row in matrix]
        return non_zero_rows

    reshaped_arr = remove_zeros(reshaped_arr_2)

    # 2. mean value
    column_max_list = [np.mean(row) if row.size > 0 else None for row in reshaped_arr]
    column_means = np.array(column_max_list)
    column_means = column_means.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_means),axis = 1)


    # labels = (column_means > 122).astype(int)
    # # labels_reshape = labels.reshape((36500+1200, 1))
    # labels_reshape = column_means.reshape((total_sample_num, 1))
    #
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_mean = np.concatenate((final_output, expanded_array_reshape), axis=1)

    # 3. min value
    column_min_list = [np.min(row) if row.size > 0 else None for row in reshaped_arr]
    # column_mins = np.min(reshaped_arr, axis=1)
    column_mins = np.array(column_min_list)
    column_mins = column_mins.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_mins),axis = 1)

    # column_mins_avg = np.average(column_mins)
    # labels = (column_mins > column_mins_avg).astype(int)
    # # labels_reshape = labels.reshape((36500+1200, 1))
    # labels_reshape = column_mins.reshape((total_sample_num, 1))
    #
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_min = np.concatenate((new_array_mean, expanded_array_reshape), axis=1)

    # 4. max value
    column_max_list = [np.max(row) if row.size > 0 else None for row in reshaped_arr]

    # column_maxs = np.max(reshaped_arr, axis=1)
    column_maxs = np.array(column_max_list)
    column_maxs = column_maxs.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_maxs),axis = 1)

    # column_maxs_avg = np.average(column_maxs)
    # labels = (column_maxs > column_maxs_avg).astype(int)
    # # labels_reshape = labels.reshape((36500+1200, 1))
    # labels_reshape = column_maxs.reshape((total_sample_num, 1))
    #
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_max = np.concatenate((new_array_min, expanded_array_reshape), axis=1)

    # 5. range value
    # column_maxs = np.max(reshaped_arr, axis=1)
    # column_mins = np.min(reshaped_arr, axis=1)
    column_range = column_maxs-column_mins
    # column_range = column_maxs.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_range),axis = 1)

    # column_range_avg = np.average(column_range)
    # labels = (column_range > column_range_avg).astype(int)
    # # labels_reshape = labels.reshape((36500+1200, 1))
    # labels_reshape = column_range.reshape((total_sample_num, 1))
    #
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_range = np.concatenate((new_array_max, expanded_array_reshape), axis=1)

    # 6. varience value
    column_var_list = [np.var(row) if row.size > 0 else None for row in reshaped_arr]
    column_var = np.array(column_var_list)
    column_var = column_var.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_var),axis = 1)
    # # column_var = np.var(reshaped_arr, axis=1)
    # column_var_avg = np.average(column_var)
    # labels = (column_var > column_var_avg).astype(int)
    # # labels_reshape = labels.reshape((36500+1200, 1))
    # labels_reshape = column_var.reshape((total_sample_num, 1))
    #
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_var = np.concatenate((new_array_range, expanded_array_reshape), axis=1)

    # 7. length of peak
    def is_half_greater_than_average(arr):
        # if len(arr) != 96:
        #     raise ValueError("Array length must be 96")

        average = np.mean(arr)
        count_greater_than_average = np.sum(arr > average)

        return count_greater_than_average
        # return 1 if count_greater_than_average >= 48 else 0

    samples = reshaped_arr

    labels = np.array([is_half_greater_than_average(sample) for sample in samples])
    labels_reshape = labels.reshape((total_sample_num, 1))
    final_label = np.concatenate((final_label,labels_reshape),axis = 1)

    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_peaklength = np.concatenate((new_array_var, expanded_array_reshape), axis=1)

    # 8. length of low
    def is_half_less_than_average(arr):
        # if len(arr) != 96:
        #     raise ValueError("Array length must be 96")

        average = np.mean(arr)
        count_less_than_average = np.sum(arr < average)

        return count_less_than_average

        # return 1 if count_less_than_average <= 48 else 0

    samples = reshaped_arr

    labels = np.array([is_half_less_than_average(sample) for sample in samples])
    labels_reshape = labels.reshape((total_sample_num, 1))
    final_label = np.concatenate((final_label,labels_reshape),axis = 1)

    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_lowlength = np.concatenate((new_array_peaklength, expanded_array_reshape), axis=1)

    #9. Peak at 0-7
    # def is_peak_in_first_32(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     peak_index = np.argmax(arr)
    #
    #     return 1 if peak_index < 32 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_peak_in_first_32(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_peak07 = np.concatenate((new_array_lowlength, expanded_array_reshape), axis=1)

    #10. Peak at 8-15
    # def is_peak_in_first_64(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     peak_index = np.argmax(arr)
    #
    #     return 1 if 32 <= peak_index < 64 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_peak_in_first_64(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_peak815 = np.concatenate((new_array_peak07, expanded_array_reshape), axis=1)

    #11. Peak at 16-23
    # def is_peak_in_first_96(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     peak_index = np.argmax(arr)
    #
    #     return 1 if peak_index > 64 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_peak_in_first_96(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num,1,max_interval))
    # new_array_peak1623 = np.concatenate((new_array_peak815, expanded_array_reshape), axis=1)

    # 12. Low at 0-7
    # def is_low_in_first_32(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     low_index = np.argmin(arr)
    #
    #     return 1 if low_index < 32 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_low_in_first_32(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num, 1, max_interval))
    # new_array_low07 = np.concatenate((new_array_peak1623, expanded_array_reshape), axis=1)

    # 13. Low at 8-15
    # def is_low_in_first_64(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     low_index = np.argmin(arr)
    #
    #     return 1 if 32 <= low_index < 64 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_low_in_first_64(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num, 1, max_interval))
    # new_array_low815 = np.concatenate((new_array_low07, expanded_array_reshape), axis=1)

    # 14. Low at 16-23
    # def is_low_in_first_96(arr):
    #     if len(arr) != 96:
    #         raise ValueError("Array length must be 96")
    #
    #     low_index = np.argmin(arr)
    #
    #     return 1 if low_index > 64 else 0
    #
    # samples = reshaped_arr
    #
    # labels = np.array([is_low_in_first_96(sample) for sample in samples])
    # labels_reshape = labels.reshape((total_sample_num, 1))
    # expanded_array = np.repeat(labels_reshape, max_interval, axis=1)
    # expanded_array_reshape = expanded_array.reshape((total_sample_num, 1, max_interval))
    # new_array_low1623 = np.concatenate((new_array_low815, expanded_array_reshape), axis=1)

    #15 skewness
    column_skew_list = [stats.skew(row) if row.size > 0 else None for row in reshaped_arr]
    column_skew = np.array(column_skew_list)
    column_skew = column_skew.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_skew),axis = 1)

    # column_skew = scipy.stats.skew(reshaped_arr, axis=1)
    # labels_reshape = column_skew.reshape((36500+1200, 1))
    #
    # expanded_array = np.repeat(labels_reshape, 96, axis=1)
    # expanded_array_reshape = expanded_array.reshape((36500+1200,1,96))
    # new_array_skew = np.concatenate((new_array_low1623, expanded_array_reshape), axis=1)


    #16 kurtosis
    column_kurtosis_list = [stats.kurtosis(row) if row.size > 0 else None for row in reshaped_arr]
    column_kurt= np.array(column_kurtosis_list)
    column_kurt = column_kurt.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_kurt),axis = 1)

    # column_kurt = scipy.stats.kurtosis(reshaped_arr, axis=1)
    # labels_reshape = column_kurt.reshape((36500+1200, 1))
    #
    # expanded_array = np.repeat(labels_reshape, 96, axis=1)
    # expanded_array_reshape = expanded_array.reshape((36500+1200,1,96))
    # new_array_kurtosis = np.concatenate((new_array_skew, expanded_array_reshape), axis=1)

    #17 Interquartile Range (IQR)
    column_iqr_list = [stats.iqr(row) if row.size > 0 else None for row in reshaped_arr]
    column_iqr= np.array(column_iqr_list)
    column_iqr = column_iqr.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_iqr),axis = 1)

    # column_iqr = scipy.stats.iqr(reshaped_arr, axis=1)
    # labels_reshape = column_iqr.reshape((36500+1200, 1))
    #
    # expanded_array = np.repeat(labels_reshape, 96, axis=1)
    # expanded_array_reshape = expanded_array.reshape((36500+1200,1,96))
    # new_array_iqr = np.concatenate((new_array_kurtosis, expanded_array_reshape), axis=1)

    #18 Coefficient of Variation (CV)
    column_cv_list = [np.std(row)/ np.mean(row) if row.size > 0 else None for row in reshaped_arr]
    column_cv= np.array(column_cv_list)
    column_cv = column_cv.reshape((total_sample_num,1))
    final_label = np.concatenate((final_label,column_cv),axis = 1)


    # column_cv = np.std(reshaped_arr,axis=1) / np.mean(reshaped_arr,axis=1)
    # labels_reshape = column_cv.reshape((36500+1200, 1))
    #
    # expanded_array = np.repeat(labels_reshape, 96, axis=1)
    # expanded_array_reshape = expanded_array.reshape((36500+1200,1,96))
    # new_array_cv = np.concatenate((new_array_iqr, expanded_array_reshape), axis=1)


    return reshaped_arr_2,final_label

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