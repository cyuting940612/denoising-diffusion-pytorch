from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
from scipy.stats import entropy
import DataProcess_Classifier as dpc

df_synthetic = pd.read_csv('4CP_Sample_1.csv').to_numpy()
#
# # Mapping ERCOT data
# filename_lmp = 'lmp_2019.csv'
# df_lmp_origin = pd.read_csv(filename_lmp)['LZ_HOUSTON']
# df_lmp_1 = df_lmp_origin.to_numpy()
# df_lmp_1 = df_lmp_1.reshape((35040, 1))
# # lmp_0 = df_lmp_new + 18
# # lmp_log = np.log(lmp_0)
# filename_load = 'load_2019.csv'
# df_load_new = pd.read_csv(filename_load)['ERCOT'].div(1000)
# df_load_origin = pd.DataFrame(np.repeat(df_load_new.values, 4, axis=0))
# df_load_1 = df_load_origin.to_numpy()
# filename_temperature = '2953997_29.80_-95.35_2019.csv'
# df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
# df_temperature_new = df_temperature_raw['Temperature']
# df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_new.values, 2, axis=0))
# df_temperature_1 = df_temperature_origin.to_numpy()
#
# df_label_1 = np.ones((35040, 1))
# stacked_1 = np.concatenate((df_lmp_1, df_load_1, df_temperature_1, df_label_1), axis=1)
#
#
# synthetic_lmp = df_synthetic[0:96,0]
# synthetic_load =df_synthetic[0:96,1]
# synthetic_temp = df_synthetic[0:96,2]
synthetic_lmp = df_synthetic[:,0]
synthetic_load =df_synthetic[:,1]
synthetic_temp = df_synthetic[:,2]

synthetic_lmp_mean = np.mean(synthetic_lmp)
synthetic_load_mean =np.mean(synthetic_load)
synthetic_temp_mean = np.mean(synthetic_temp)

synthetic_lmp_std = np.std(synthetic_lmp)
synthetic_load_std =np.std(synthetic_load)
synthetic_temp_std = np.std(synthetic_temp)

_,cp_ER,non_cp_ER = dpc.DataProcess_Classifier()

origin_lmp = cp_ER[:,0]
origin_load = cp_ER[:,1]
origin_temp = cp_ER[:,2]

# origin_lmp = non_cp_ER[:,0]
# origin_load = non_cp_ER[:,1]
# origin_temp = non_cp_ER[:,2]

cp_lmp_mean = np.mean(origin_lmp)
cp_load_mean =np.mean(origin_load)
cp_temp_mean = np.mean(origin_temp)

cp_lmp_std = np.std(origin_lmp)
cp_load_std =np.std(origin_load)
cp_temp_std = np.std(origin_temp)

# non4cp_lmp_mean = np.mean(origin_lmp)
# non4cp_load_mean =np.mean(origin_load)
# non4cp_temp_mean = np.mean(origin_temp)
#
# non4cp_lmp_std = np.std(origin_lmp)
# non4cp_load_std =np.std(origin_load)
# non4cp_temp_std = np.std(origin_temp)


# origin_lmp = stacked_1[20156:20252,0] #07/30/2019
# origin_lmp = stacked_1[0:96,0]   #01/01/2019


p_lmp = gaussian_kde(synthetic_lmp).pdf
q_lmp = gaussian_kde(origin_lmp).pdf
p_load = gaussian_kde(synthetic_load).pdf
q_load = gaussian_kde(origin_load).pdf
p_temp = gaussian_kde(synthetic_temp).pdf
q_temp = gaussian_kde(origin_temp).pdf

# Create a range of x-values to represent the PDF
# p = np.linspace(min(synthetic_lmp), max(synthetic_lmp), 100)
# p = np.linspace(min(synthetic_load), max(synthetic_load), 1000)
p = np.linspace(min(synthetic_temp), max(synthetic_temp), 1000)

# Calculate the estimated PDF values
# p_pdf = p_lmp(p)
# p_pdf = p_load(p)
p_pdf = p_temp(p)


# Create a range of x-values to represent the PDF
# q = np.linspace(min(origin_lmp), max(origin_lmp), 100)
# q = np.linspace(min(origin_load), max(origin_load), 1000)
q = np.linspace(min(origin_temp), max(origin_temp), 1000)

# Calculate the estimated PDF values
# q_pdf = q_lmp(q)
# q_pdf = q_load(q)
q_pdf = q_temp(q)

# Compute the KL Divergence
kl_divergence = entropy(p_pdf, q_pdf)

print("KL Divergence:", kl_divergence)

