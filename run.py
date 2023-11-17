import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataProcess
import numpy as np

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')



model = Unet1D(
    dim = 8,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    self_condition=True
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 96,
    timesteps = 1000,
    objective = 'pred_x0'
)

data,min_values,max_values = DataProcess.data_process()
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
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

num_samples = 1
cond_tensor = torch.zeros(1, 3, 96)# Specify the number of samples you want to generate
condition = torch.tensor(data[0,:,:])
cond_tensor[0,:,:] = condition

generated_samples = diffusion.sample(batch_size=num_samples, cond = cond_tensor)

generated_lmp_log = DataProcess.denormalize_minmax(generated_samples[:,0,:],min_values[0],max_values[0])
generated_lmp = np.exp(generated_lmp_log)-18
generated_load = DataProcess.denormalize_minmax(generated_samples[:,1,:],min_values[1],max_values[1])
generated_temp = DataProcess.denormalize_minmax(generated_samples[:,2,:],min_values[2],max_values[2])

final_lmp = generated_lmp[:,:]
final_load = generated_load[:,:]
final_temp = generated_temp[:,:]
final_sample = np.stack((final_lmp,final_load,final_temp),axis=2)

data_2d = final_sample.reshape(-1, final_sample.shape[-1])
file_path = 'Generated_Sample.csv'
np.savetxt(file_path, data_2d, delimiter=',', fmt='%.2f')
# final_sample =
# print(generated_samples.shape)
# print(generated_samples)
# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 1)
# print (sampled_seq)
