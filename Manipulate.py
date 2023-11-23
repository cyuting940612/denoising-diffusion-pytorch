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

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')



model = Unet1D(
    dim = 8,
    dim_mults = (1, 2, 4, 8),
    channels = 64,
    self_condition=True
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 24,
    timesteps = 1000,
    objective = 'pred_x0'
)

data_0,min_values,max_values = dpu.data_process()

# data_1,_ = PCA.data_PCA(data_0,0)

# data_1_cp,_ = PCA.data_PCA(data_0[151:273],0)
# data = data_1_cp.reshape(data_1_cp.shape[0],1,50)

data_0_nocp = np.concatenate((data_0[0:151],data_0[273:366]),axis=0)
data_1_nocp= np.transpose(data_0_nocp, (0, 2, 1))
data_2_nocp = Encoder.encoder(data_1_nocp)
data = np.transpose(data_2_nocp, (0, 2, 1))

# data_0_cp = data_0[151:273]
# data_1_cp= np.transpose(data_0_cp, (0, 2, 1))
# data_2_cp = Encoder.encoder(data_1_cp)
# data = np.transpose(data_2_cp, (0, 2, 1))
# data_1_nocp,_ = PCA.data_PCA(data_0_nocp,0)
# data = data_1_nocp.reshape(data_1_nocp.shape[0],1,50)

training_seq = torch.from_numpy(data)
dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 5,
    train_lr = 8e-4,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

model_cls = Classifier.SimpleClassifier()
# Load the trained model
model_cls.load_state_dict(torch.load('2d_classifier_model.pth'))
model_cls.eval()

# for name, param in model_cls.named_parameters():
#     if param.requires_grad:
#         print(f"Layer: {name}, Shape: {param.data.shape}")
#         print(param.data)
synthetic_n = 1
stacked_sample = np.zeros((synthetic_n,96,100))
for k in range(synthetic_n):
    num_samples = 1
    cond_tensor = torch.zeros(1, 64, 24)
    condition = torch.tensor(data[0,:,:])
    condition1 = data[200,:,:]
    condition2 = data[0,:,:]
    cond_tensor[0,:,:] = condition

    for name, param in model_cls.named_parameters():
        if param.requires_grad and name == 'fc1.weight':
            weights = param.data
    # weights = weights.reshape(16,24).numpy()
    cond2 = cond_tensor.reshape(-1)
    # cond2 = cond_tensor.reshape(-1)


    cond2 = cond2.reshape(1,64,24)
    generated_samples_origin = diffusion.sample(batch_size=num_samples, cond=cond2)
    # generated_samples_origin = diffusion.condition_sample(batch_size=num_samples, cond=cond2)
    generated_samples_np = generated_samples_origin.numpy()
    generated_samples_np_re = np.transpose(generated_samples_np, (0, 2, 1))
    generated_samples_2d = Encoder.decoder(generated_samples_np_re)
    # generated_samples_2d = generated_samples_origin
    # generated_samples = generated_samples_2d.reshape(1,3,96)
    generated_single = np.zeros((96,100))
    for i in range (100):
        generated_single[:,i]= DataProcess.denormalize_minmax(generated_samples_2d[:,:,i],min_values[i],max_values[i])
    # generated_lmp_log = DataProcess.denormalize_minmax(generated_samples_2d[:,:,0],min_values[0],max_values[0])
    # generated_lmp = np.exp(generated_lmp_log)-20
    # generated_load = DataProcess.denormalize_minmax(generated_samples_2d[:,:,1],min_values[1],max_values[1])
    # generated_temp = DataProcess.denormalize_minmax(generated_samples_2d[:,:,2],min_values[2],max_values[2])
    #
    # final_lmp = generated_lmp[:,:]
    # final_load = generated_load[:,:]
    # final_temp = generated_temp[:,:]
    # final_sample = np.stack((final_lmp,final_load,final_temp),axis=1)

    stacked_sample[k,:,:] = generated_single

# data_2d = np.transpose(stacked_sample,(0,2,1))
final_data = stacked_sample.reshape(96*synthetic_n, 100)
file_path = 'Non_4CP_Sample_User.csv'
np.savetxt(file_path, final_data, delimiter=',', fmt='%.2f')
# final_sample =
# print(generated_samples.shape)
# print(generated_samples)
# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 1)
# print (sampled_seq)
