import torch

import Classifier
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataProcess_Classifier as dpc
import numpy as np
import PCA
import Encoder
import DataProcess_user_conditional as dpuc



model_cls = Classifier.SimpleNN()
# Load the trained model
model_cls.load_state_dict(torch.load('2d_classifier_model_meta.pth'))
model_cls.eval()

# Access the weights and biases of the first layer
first_layer_weights = model_cls.fc3.weight.data
first_layer_biases = model_cls.fc3.bias.data

print("First layer weights shape:", first_layer_weights.shape)
print("First layer biases shape:", first_layer_biases.shape)
# data,_,_ = DataProcess.data_process()
# data = np.transpose(data,(0,2,1))
# # Convert the input to a PyTorch tensor
# encoded_data = Encoder.encoder(data)
# encoded_data = np.transpose(encoded_data,(0,2,1))
# encoded_data =
data_user = dpc.data_process()

reshaped_arr = data_user[:,0:1,:].reshape(36500, 96)
data_user_32 = reshaped_arr.astype(np.float32)

new_input = torch.tensor(data_user_32, dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    model_output,features = model_cls(new_input)

output_np = model_output.numpy()
predicted_label = np.zeros(365)
for i in range(365):
    predicted_label[i] = 1 if model_output[i] > 0 else 0
print(predicted_label)
