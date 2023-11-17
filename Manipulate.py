import torch

import Classifier
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataProcess
import numpy as np
import PCA
import Encoder

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')



model = Unet1D(
    dim = 8,
    dim_mults = (1, 2, 4, 8),
    channels = 6,
    self_condition=None
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 24,
    timesteps = 1000,
    objective = 'pred_x0'
)

data_0,min_values,max_values = DataProcess.data_process()

# data_1,_ = PCA.data_PCA(data_0,0)

# data_1_cp,_ = PCA.data_PCA(data_0[151:273],0)
# data = data_1_cp.reshape(data_1_cp.shape[0],1,50)

data_0_nocp = np.concatenate((data_0[0:151],data_0[273:366]),axis=0)
data_1_nocp= np.transpose(data_0_nocp, (0, 2, 1))
data_2_nocp = Encoder.encoder(data_1_nocp)
data = np.transpose(data_2_nocp, (0, 2, 1))
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
    train_lr = 8e-5,
    train_num_steps = 6000,         # total training steps
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
stacked_sample = np.zeros((1,96,3))
for k in range(1):
    num_samples = 1
    # cond_tensor = torch.zeros(1, 1, 50)
    # condition = torch.tensor(data[94,:,:])
    # cond_tensor[0,:,:] = condition

    for name, param in model_cls.named_parameters():
        if param.requires_grad:
            weights = param.data

    # cond2 = cond_tensor.reshape(-1)
    # cond2 = cond_tensor.reshape(-1)


    # cond2 = cond2.reshape(1,1,50)
    generated_samples_origin = diffusion.sample(batch_size=num_samples)
    generated_samples_tensor = generated_samples_origin.reshape(1,24,6)
    generated_samples_np = generated_samples_tensor.numpy()
    generated_samples_2d = Encoder.decoder(generated_samples_np)
    generated_samples = generated_samples_2d.reshape(1,3,96)
    generated_lmp_log = DataProcess.denormalize_minmax(generated_samples[:,0,:],min_values[0],max_values[0])
    generated_lmp = np.exp(generated_lmp_log)-20
    generated_load = DataProcess.denormalize_minmax(generated_samples[:,1,:],min_values[1],max_values[1])
    generated_temp = DataProcess.denormalize_minmax(generated_samples[:,2,:],min_values[2],max_values[2])

    final_lmp = generated_lmp[:,:]
    final_load = generated_load[:,:]
    final_temp = generated_temp[:,:]
    final_sample = np.stack((final_lmp,final_load,final_temp),axis=2)

    stacked_sample[k,:,:] = final_sample


data_2d = stacked_sample.reshape(-1, stacked_sample.shape[-1])
file_path = 'Non_4CP_Sample.csv'
np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')
# final_sample =
# print(generated_samples.shape)
# print(generated_samples)
# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 1)
# print (sampled_seq)
