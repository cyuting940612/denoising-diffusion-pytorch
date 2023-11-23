import torch

import Classifier
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataProcess
import numpy as np
import PCA
import Encoder


model_cls = Classifier.SimpleClassifier()
# Load the trained model
model_cls.load_state_dict(torch.load('2d_classifier_model.pth'))
model_cls.eval()

data,_,_ = DataProcess.data_process()
data = np.transpose(data,(0,2,1))
# Convert the input to a PyTorch tensor
encoded_data = Encoder.encoder(data)
encoded_data = np.transpose(encoded_data,(0,2,1))
# encoded_data =

new_input = torch.tensor(encoded_data, dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    model_output = model_cls(new_input)

predicted_label = np.zeros(365)
for i in range(365):
    predicted_label[i] = 1 if model_output[i] > 0.5 else 0
print(predicted_label)
