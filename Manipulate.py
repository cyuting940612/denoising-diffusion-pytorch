import math

import torch

import Classifier
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataProcess
import numpy as np
import PCA
import Encoder
from torch import nn
from scipy.optimize import minimize
import DataProcess_user as dpu
import pandas as pd
import DataProcess_user_conditional as dpuc
import DataProcess_user_manipulate as dpum


if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')



model = Unet1D(
    dim = 8,
    dim_mults = (1, 2, 4, 8),
    channels = 5,
    self_condition=True
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 96,
    timesteps = 1000,
    objective = 'pred_x0',
    # sampling_timesteps = 300
)

data_user,min,max = dpum.data_process()

# data_1,_ = PCA.data_PCA(data_0,0)

# data_1_cp,_ = PCA.data_PCA(data_0[151:273],0)
# data = data_1_cp.reshape(data_1_cp.shape[0],1,50)

# data_0_nocp = np.concatenate((data_0[0:151],data_0[273:366]),axis=0)
# data_0_cp = data_0[151:273]
# data_1_nocp= np.transpose(data_0, (0, 2, 1))
# data_2_nocp = Encoder.encoder(data_0)
# data = np.transpose(data_2_nocp, (0, 2, 1))

# data_0_cp = data_0[151:273]
# data_1_cp= np.transpose(data_0_cp, (0, 2, 1))
# data_2_cp = Encoder.encoder(data_1_cp)
# data = np.transpose(data_2_cp, (0, 2, 1))
# data_1_nocp,_ = PCA.data_PCA(data_0_nocp,0)
# data = data_1_nocp.reshape(data_1_nocp.shape[0],1,50)


# model_cls = Classifier.SimpleClassifier()
# # Load the trained model
# model_cls.load_state_dict(torch.load('2d_classifier_model_large.pth'))
# model_cls.eval()

# for name, param in model_cls.named_parameters():
#     if param.requires_grad:
#         print(f"Layer: {name}, Shape: {param.data.shape}")
#         print(param.data)
synthetic_n = 5
stacked_sample = np.zeros((synthetic_n,96,5))
for k in range(synthetic_n):
    num_samples = 1
    # cond1 = torch.tensor(data_user_32[0:1,:,:])
    # cond1_np = cond1.numpy()
    # # condition1 = data[200,:,:]
    # # condition2 = data[0,:,:]
    #
    # for name, param in model_cls.named_parameters():
    #     if param.requires_grad and name == 'fc1.weight':
    #         weights = param.data
    # # weights = weights.reshape(16,24).numpy()
    # weights_np = weights.numpy()
    # weights[weights > 0.2] = 0.8*max1
    # weights[weights < 0.2] = 0.8*min1
    # # cond2 = cond1.reshape(-1) + weights
    # cond2 = weights
    #
    # # cond2 = cond_tensor.reshape(-1)
    #
    #
    # cond2 = cond2.reshape(1,1,96)
    # cond2_np = cond2.numpy()
    model_checkpoint_path = 'diffusion_model_checkpoint_manipulate_testing.pth'
    loaded_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    generated_samples_origin = diffusion.sample(batch_size=num_samples)
    # generated_samples_origin = diffusion.sample_inter(cond1=cond1,cond2=cond1)


    # mean = final_mean.numpy()
    # df = pd.DataFrame(mean[0,:,:])
    # excel_filename = '4CP_mean.xlsx'
    # df.to_excel(excel_filename, index=False)

    generated_samples_single = np.zeros((96, 5))
    generated_samples_np = generated_samples_origin.numpy()
    generated_samples_np_re = np.transpose(generated_samples_np, (0, 2, 1))
    for i in range(5):
        generated_samples_single[:,i] = dpum.denormalize_minmax(generated_samples_np_re[:,:,i],min[i],max[i])
    # generated_samples_single[:, 0] = dpuc.denormalize_minmax(generated_samples_np_re[:, :, 0], min1, max1)
    # generated_samples_single[:, 1] = dpuc.denormalize_minmax(generated_samples_np_re[:, :, 1], min2, max2)
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

    stacked_sample[k, :, :] = generated_samples_single

# data_2d = np.transpose(stacked_sample,(0,2,1))
final_data = stacked_sample.reshape(96 * synthetic_n, 5)
file_path = 'Manipulate_User.csv'
np.savetxt(file_path, final_data, delimiter=',', fmt='%.2f')