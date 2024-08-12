import torch

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
import DataProcess_filterfree as dp
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.models import load_model

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
#
model = Unet1D(
    dim = 4,
    dim_mults = (1, 4,8,16),
    channels = 1
)

# model = dpl.LinearModel(
#      input_size=96,
#     output_size=96
# )

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 96,
    timesteps = 1000,
    objective = 'pred_x0',
    # sampling_timesteps = 300
)

# data_0,min_values,max_values = dp.data_process()
# data_user,min_values_user,max_values_user = dpuc.data_process()



# data_user,_,_= dp.data_process()

# data_user= dp.data_process()


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# weekend_mask = (data_user[:, -1] == 0)
# weekend = data_user[weekend_mask]

# loaded_encoder = load_model('encoder_model.h5')
data_user = dp.data_process()
# data = data_user[:,0:96]
# data = data / np.max(data)
# encoded_data = loaded_encoder.predict(data)

data_origin = data_user[:,0:96]

data = data_origin[:, np.newaxis, :]

label = data_user[:,96]
data_set = CustomDataset(data,label)
# data_user_32 = data_user.astype(np.float32)

# data1 = data_0[:,0,:]
# data2 = data_user[:,0:5,:]
#
# aggregate = np.concatenate((data_0[:,0:1,:],data_0[:,2:3,:],data_user[:,0:5,:]),axis=1)

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

# training_seq = torch.from_numpy(data_user)
# dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = data_set,
    train_batch_size = 8,
    train_lr = 8e-4,
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()
# Specify the file path for saving and loading the model

# model_checkpoint_path = 'diffusion_model_unconditional_checkpoint_weekend_testing.pth'
model_checkpoint_path = 'classifier_free_test_200k_linear_s1.pth'


# Save the model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    # Add other information you want to save
}, model_checkpoint_path)