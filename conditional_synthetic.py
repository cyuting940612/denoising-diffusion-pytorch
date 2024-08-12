import torch

import DataProcess_user
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
# from denoising_diffusion_pytorch import denoising_diffusion_pytorch_linear as dpl

import DataProcess
import numpy as np
import PCA
import Encoder
from torch import nn
from scipy.optimize import minimize
import DataProcess_user as dpu
import pandas as pd
import DataProcess as dp
import DataProcess_user as dpu
import DataProcess_filterfree as dp
from tensorflow.keras.models import load_model
import tensorflow as tf


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU.")


# model = dpl.LinearModel(
#      input_size=96,
#     output_size=96
# )

model = Unet1D(
    dim = 4,
    dim_mults = (1, 4,8,16),
    channels = 1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 96,
    timesteps = 1000,
    objective = 'pred_x0',
    # sampling_timesteps = 300
)

# origin_data,min_values,max_values=dp.data_process()
# origin_data=dp.data_process()


# data_0,min_values,max_values = dp.data_process()
# data_user,min_values_user,max_values_user = dpu.data_process()
# data_user,min,max = dpuu.data_process()

# data_1,_ = PCA.data_PCA(data_0,0)

# data_1_cp,_ = PCA.data_PCA(data_0[151:273],0)
# data = data_1_cp.reshape(data_1_cp.shape[0],1,50)

# data_0_nocp = np.concatenate((data_0[0:151],data_0[273:366]),axis=0)
# data_0_cp = data_0[151:273]
# data_1_nocp= np.transpose(data_0, (0, 2, 1))
# data_2_nocp = Encoder.encoder(data_1_nocp)
# data = np.transpose(data_2_nocp, (0, 2, 1))

# data_0_cp = data_0[151:273]
# data_1_cp= np.transpose(data_0_cp, (0, 2, 1))
# data_2_cp = Encoder.encoder(data_1_cp)
# data = np.transpose(data_2_cp, (0, 2, 1))
# data_1_nocp,_ = PCA.data_PCA(data_0_nocp,0)
# data = data_1_nocp.reshape(data_1_nocp.shape[0],1,50)


# model_cls = Classifier.SimpleClassifier()
# # Load the trained model
# model_cls.load_state_dict(torch.load('2d_classifier_model.pth'))
# model_cls.eval()

# for name, param in model_cls.named_parameters():
#     if param.requires_grad:
#         print(f"Layer: {name}, Shape: {param.data.shape}")
#         print(param.data)
synthetic_n = 1
feature_n = 1
label = -1
# stacked_sample = np.zeros((synthetic_n,1,20))
for k in range(synthetic_n):
    num_samples = 5000
    # cond1 = torch.tensor(data[200:201,:,:])
    # # condition1 = data[200,:,:]
    # # condition2 = data[0,:,:]
    #
    # for name, param in model_cls.named_parameters():
    #     if param.requires_grad and name == 'fc1.weight':
    #         weights = param.data
    # # weights = weights.reshape(16,24).numpy()
    # cond2 = cond1.reshape(-1)-0.2*weights
    # # cond2 = cond_tensor.reshape(-1)
    #
    #
    # cond2 = cond2.reshape(1,64,24)
    model_checkpoint_path = 'classifier_free_test_200k_linear_s1.pth'
    loaded_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    generated_samples_origin = diffusion.sample(batch_size=num_samples, label = label)

    # mean = final_mean.numpy()
    # df = pd.DataFrame(mean[0,:,:])
    # excel_filename = '4CP_mean.xlsx'
    # df.to_excel(excel_filename, index=False)

    # generated_samples_single = np.zeros((96,feature_n))

    generated_samples_np = generated_samples_origin.numpy()
    # denormalized_data = generated_samples_np * (max_values-min_values) + min_values

    # generated_samples_np_re = np.transpose(generated_samples_np, (0, 2, 1))
    # for i in range(5):
    #     generated_samples_single[:,i] = dpuu.denormalize_minmax(generated_samples_np_re[:,:,i],min[i],max[i])
    # generated_samples_single[:,0] = dpuu.denormalize_minmax(generated_samples_np_re[:,:,0],min1,max1)
    # generated_samples_single[:,1] = dpuu.denormalize_minmax(generated_samples_np_re[:,:,1],min2,max2)
    # # generated_samples_2d = Encoder.decoder(generated_samples_np_re)
    # # generated_samples_2d = generated_samples_origin
    # # generated_samples = generated_samples_2d.reshape(1,3,96)
    # generated_single = np.zeros((96,feature_n))
    # generated_single[:,0]= DataProcess.denormalize_minmax(generated_samples_np_re[:,:,0],min_values[0],max_values[0])
    # generated_single[:,1]= DataProcess.denormalize_minmax(generated_samples_np_re[:,:,1],min_values[2],max_values[2])
    #
    # for j in range(2,7):
    #     generated_single[:,j] = DataProcess_user.denormalize_minmax(generated_samples_np_re[:,:,j],min_values_user[j-2],max_values_user[j-2])
    # generated_single[:,0] = np.exp(generated_single[:,0])-20
    # generated_single[:,0] = generated_single[:,0]-20

    # generated_lmp_log = DataProcess.denormalize_minmax(generated_samples_2d[:,:,0],min_values[0],max_values[0])
    # generated_lmp = np.exp(generated_lmp_log)-20
    # generated_load = DataProcess.denormalize_minmax(generated_samples_2d[:,:,1],min_values[1],max_values[1])
    # generated_temp = DataProcess.denormalize_minmax(generated_samples_2d[:,:,2],min_values[2],max_values[2])
    #
    # final_lmp = generated_lmp[:,:]
    # final_load = generated_load[:,:]
    # final_temp = generated_temp[:,:]
    # final_sample = np.stack((final_lmp,final_load,final_temp),axis=1)

    # stacked_sample[k,:,:] = denormalized_data

# data_2d = np.transpose(stacked_sample,(0,2,1))

# final_data = denormalized_data.reshape(96*num_samples, feature_n)
# loaded_decoder = tf.keras.models.load_model('decoder_model.h5')
# generated_samples_np_reshape = generated_samples_np.reshape(num_samples, 20)
# encoded_data_origin = loaded_decoder.predict(generated_samples_np_reshape)
# encoded_data=encoded_data_origin*110.6338
# final_data = encoded_data.reshape(96*num_samples, feature_n)

final_data = generated_samples_np.reshape(96*num_samples, feature_n)

# final_data = stacked_sample.reshape(96*synthetic_n, feature_n)
file_path = 'filter_free_weekend_5000_s1.csv'
np.savetxt(file_path, final_data, delimiter=',', fmt='%.2f')
# final_sample =
# print(generated_samples.shape)
# print(generated_samples)
# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 1)
# print (sampled_seq)
