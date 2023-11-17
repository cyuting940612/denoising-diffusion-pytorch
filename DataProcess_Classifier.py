import pandas as pd
import numpy as np
import DataProcess as dp

def DataProcess_Classifier():
    #Mapping ERCOT data
    filename_lmp = 'lmp_2019.csv'
    df_lmp_origin = pd.read_csv(filename_lmp)['LZ_HOUSTON']
    df_lmp_1 = df_lmp_origin.to_numpy()
    df_lmp_1 = df_lmp_1.reshape((35040,1))
    # lmp_0 = df_lmp_new + 18
    # lmp_log = np.log(lmp_0)
    filename_load = 'load_2019.csv'
    df_load_new = pd.read_csv(filename_load)['ERCOT'].div(1000)
    df_load_origin = pd.DataFrame(np.repeat(df_load_new.values, 4, axis=0))
    df_load_1 = df_load_origin.to_numpy()
    filename_temperature = '2953997_29.80_-95.35_2019.csv'
    df_temperature_raw = pd.read_csv(filename_temperature, skiprows=[0, 1])
    df_temperature_new = df_temperature_raw['Temperature']
    df_temperature_origin = pd.DataFrame(np.repeat(df_temperature_new.values, 2, axis=0))
    df_temperature_1 = df_temperature_origin.to_numpy()

    df_label_1 = np.ones((35040,1))
    stacked_1 = np.concatenate((df_lmp_1, df_load_1, df_temperature_1,df_label_1), axis=1)

    cp_ER = stacked_1[14492:26204,:]
    non_cp_ER_1 = stacked_1[0:14491,:]
    non_cp_ER_2 = stacked_1[26203:35040,:]
    non_cp_ER = np.concatenate((non_cp_ER_1,non_cp_ER_2), axis=0)
    non_cp_ER[:,3] = -1
    ER_classifier = np.concatenate((cp_ER,non_cp_ER), axis=0)

    # #Mapping CAISO data
    # df_temp = pd.read_csv('CAISO_temperature_2019.csv')
    # np_temp = df_temp.values
    # df_lmp =  pd.read_csv('CAISO_lmp_2019.csv')
    # np_lmp = df_lmp.values
    # df_load = pd.read_csv('CAISO_load_2019.csv')
    # np_load = df_load.values
    #
    #
    # # new_temp = np.zeros((24,365))
    # temp = np_temp[1:26,1:367].astype(float)
    # lmp = np_lmp[0:25,0:366]
    # load = np_load[0:25,0:366]
    #
    # stacked = np.zeros((365*24,4))
    # for i in range(365):
    #     for j in range(24):
    #         stacked[j+i*24,0] = lmp[j,i]
    #         stacked[j+i*24,1] = load[j,i]/1000
    #         stacked[j+i*24,2] = temp[j,i]
    #         stacked[j+i*24,3] = -1
    #
    # repeated_data = np.repeat(stacked, 4, axis=0)
    # classifier_data = np.concatenate((repeated_data,stacked_1),axis=0)
    return ER_classifier,cp_ER,non_cp_ER