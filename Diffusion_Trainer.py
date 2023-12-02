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
    objective = 'pred_x0',
    # sampling_timesteps = 300
)

data_0,min_values,max_values = dpu.data_process()

# data_1,_ = PCA.data_PCA(data_0,0)

# data_1_cp,_ = PCA.data_PCA(data_0[151:273],0)
# data = data_1_cp.reshape(data_1_cp.shape[0],1,50)

data_0_nocp = np.concatenate((data_0[0:151],data_0[273:366]),axis=0)
data_0_cp = data_0[151:273]
data_1_nocp= np.transpose(data_0, (0, 2, 1))
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
    train_lr = 8e-5,
    train_num_steps = 8000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()
# Specify the file path for saving and loading the model

model_checkpoint_path = 'diffusion_model_checkpoint.pth'

# Save the model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    # Add other information you want to save
}, model_checkpoint_path)