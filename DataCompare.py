import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

# Generate example data
filename_lmp = 'lmp_2019.csv'
df_lmp = pd.read_csv(filename_lmp)['LZ_HOUSTON'].to_numpy()
lmp_0 = df_lmp + 18
lmp_log = np.log(lmp_0)
filename_load = 'load_2019.csv'
df_load = pd.read_csv(filename_load)['ERCOT'].div(1000)
load_0 = pd.DataFrame(np.repeat(df_load.values, 4, axis=0)).to_numpy()
filename_temperature = '2953997_29.80_-95.35_2019.csv'

df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
df_temperature = df_temperature_raw['Temperature']
temp_0 = pd.DataFrame(np.repeat(df_temperature.values, 2, axis=0)).to_numpy() # Dependent variable

load_0 = np.reshape(load_0,(35040,))
temp_0 = np.reshape(temp_0,(35040,))
# Calculate the Pearson correlation coefficient and p-value
corr_coefficient, p_value = pearsonr(load_0, lmp_log)

# Display the results
print(f"Pearson Correlation Coefficient: {corr_coefficient:.2f}")
print(f"P-Value: {p_value:.4f}")

# Visualize the data and the linear relationship
plt.scatter(load_0, lmp_log, label='Data Points')
# plt.plot(load_0, load_0 * (corr_coefficient), color='red', linestyle='--', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(2,10)
plt.legend()
plt.title('Scatter Plot with Linear Fit')
plt.show()

# Interpret the correlation coefficient
if corr_coefficient > 0:
    print("The variables are positively correlated.")
elif corr_coefficient < 0:
    print("The variables are negatively correlated.")
else:
    print("There is no linear correlation between the variables.")