import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# filename_lmp = 'lmp_2019.csv'
# df_lmp_origin = pd.read_csv(filename_lmp)['LZ_HOUSTON']
# df_lmp = df_lmp_origin.to_numpy()
# lmp_0 = df_lmp + 18
# lmp_log = np.log(lmp_0)
# filename_load = 'load_2019.csv'
# df_load_0 = pd.read_csv(filename_load)['ERCOT'].div(1000)
# df_load_origin = pd.DataFrame(np.repeat(df_load_0.values, 4, axis=0))
# df_load = df_load_origin.to_numpy()
# filename_temperature = '2953997_29.80_-95.35_2019.csv'
#
# df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
# df_temperature_0 = df_temperature_raw['Temperature']
# df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_0.values, 2, axis=0))
# df_temperature = df_temperature_origin.to_numpy()

filename_syn = 'Unconditional_Whole_2.csv'
lmp = pd.read_csv(filename_syn)['LMP'].to_numpy()
load = pd.read_csv(filename_syn)['Load'].to_numpy()
temp = pd.read_csv(filename_syn)['Temperature'].to_numpy()


# filename_mid = 'Generated_mid.csv'
# filename_high = 'Generated_high.csv'
# delimiter = ';'
# lmp_low = pd.read_csv(filename_low)['LMP']
# load_low = pd.read_csv(filename_low)['Load']
# temp_low = pd.read_csv(filename_low)['Temperature']
# lmp_mid = pd.read_csv(filename_mid)['LMP']
# load_mid = pd.read_csv(filename_mid)['Load']
# temp_mid = pd.read_csv(filename_mid)['Temperature']
# lmp_high = pd.read_csv(filename_high)['LMP']
# load_high = pd.read_csv(filename_high)['Load']
# temp_high = pd.read_csv(filename_high)['Temperature']
# start_index = [98,962,1346,1538,1730,2114,3362,3842]
# end_index = [193, 1057, 1441,1633, 1825, 2209, 3457, 3937]
# start_index = [1730, 2114, 4418, 4226, 4514]
# end_index = [1825, 2209, 4513,4321,4609]
# start_index = [674,1154,4226,5666,386]
# end_index = [769,1249,4321,5761,481]
start_index = [2,194,770]
end_index = [97,289,865]


# Sample data (replace these with your actual data)
# real_start_index = [ 13628, 15356, 26492, 27740, 18428 ]
# real_end_index = [ 13724, 15452, 26588, 27836,18524]
real_start_index = [ 20158]
real_end_index = [ 20254]
time_points = np.arange(96)
real_lmp_low = df_lmp_origin[1248:1344].to_numpy()
real_load_low = df_load[1248:1344]
real_temperature_low = df_temperature[1248:1344]
real_lmp_mid = df_lmp_origin[4224:4320].to_numpy()
real_load_mid = df_load[4224:4320]
real_temperature_mid = df_temperature[4224:4320]
real_lmp_high = df_lmp_origin[18428:18524].to_numpy()
real_load_high = df_load[18428:18524]
real_temperature_high = df_temperature[18428:18524]

# Create the first y-axis on the left
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

axes[0].set_ylabel('Temperature')
axes[0].set_title('Real Data')
axes[1].set_xlabel('Time (15-min intervals)')
axes[1].set_ylabel('Temperature')
axes[1].set_title('Synthetic Data')
for i in range(len(start_index)):
    temper = load[start_index[i]-2:end_index[i]-1]
    # axes[0].plot(time_points, real_temperature_high, label = 'Actual')
    axes[1].plot(time_points, temper, label = 'Synthetic')

for j in range(len(real_start_index)):
    real_temper = df_load[real_start_index[j]:real_end_index[j]]
    axes[0].plot(time_points, real_temper, label = 'Actual')
# Create a second y-axis on the right
# ax2 = ax1.twinx()
# ax2.set_ylabel('Y2', color='tab:red')
# ax2.plot(time_points, lmp_mid, color='tab:red')
# Show the plot
# plt.show()
#
# ax = plt.axes(projection='3d')
#
# # Data for a three-dimensional line
# for i in range(len(start_index)):
#     temper1 = temp[start_index[i]-2:end_index[i]-1]
#     temper2 = load[start_index[i]-2:end_index[i]-1]
#     # temper2_re = temper2.reshape(96)
#     xline = time_points
#     yline = temper1
#     # zline= temper2_re
#     zline = temper2
#     # axes[0].plot(time_points, real_temperature_high, label = 'Actual')
#     # axes[1].plot(time_points, temper, label = 'Synthetic')
#     ax.plot3D(xline, yline, zline)
#
# # Add labels
# ax.set_title('Synthetic Data')
# ax.set_xlabel('Time')
# ax.set_ylabel('Temperature')
# ax.set_zlabel('Load')

# Show the 3D plot
plt.show()











