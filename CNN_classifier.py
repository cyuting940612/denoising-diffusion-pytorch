import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import DataProcess_Classifier as dpc

# Example data generation
num_samples = 36500
input_shape = (96, 1)
num_classes = 32

df_all_week = dpc.data_process()
# df_all_highlow = dpchl.data_process()

# X = np.zeros((365,3,96))
# y = np.zeros((365))
# for i in range(365):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]
X_week = df_all_week[:,0:1,:]
y_week = df_all_week[:,1,0]
y_highlow = df_all_week[:,2,0]
y_min =df_all_week[:,3,0]
y_max =df_all_week[:,4,0]
y_range =df_all_week[:,5,0]
y_var =df_all_week[:,6,0]
y_peaklength = df_all_week[:,7,0]
y_lowlength = df_all_week[:,8,0]
y_peak07 = df_all_week[:,9,0]
y_peak815 = df_all_week[:,10,0]
y_peak1623 = df_all_week[:,11,0]
y_low07 = df_all_week[:,12,0]
y_low815 = df_all_week[:,13,0]
y_low1623 = df_all_week[:,14,0]

y_combine = np.column_stack((y_week,y_highlow,y_min,y_var,y_range))
# reshaped_array = y_combine.reshape((36500, 1))# y_combine = np.column_stack((y_week, y_highlow,y_min,y_max,y_range,y_var,y_peaklength,y_lowlength,y_peak07,y_peak815,y_peak1623,y_low07,y_low815,y_low1623))
reshaped_arr_week = X_week.reshape(36500, 96)
X_tensor_week_origin = torch.tensor(reshaped_arr_week, dtype=torch.float32)
X_tensor_week = X_tensor_week_origin.unsqueeze(1)


def binary_array_to_decimal(binary_array):
    # Ensure the binary array is of length 5
    if len(binary_array) != 5 or not np.all((binary_array == 0) | (binary_array == 1)):
        raise ValueError("Input must be a numpy array of length 5 containing only 0s and 1s.")

    # Convert the binary array to decimal
    decimal_value = 0
    for i, digit in enumerate(reversed(binary_array)):
        decimal_value += digit * (2 ** i)

    return decimal_value

y = np.random.randint(0, num_classes, y_combine.shape[0])
for i in range(y_combine.shape[0]):
    y[i] = binary_array_to_decimal(y_combine[i])

y = np.eye(num_classes)[y].astype(np.float32)
y_tensor_week = torch.tensor(y, dtype=torch.float32)

# # Generate random data
# X = np.random.random((num_samples, *input_shape)).astype(np.float32)
# y = np.random.randint(0, num_classes, num_samples)
# y = np.eye(num_classes)[y].astype(np.float32)
#
# # Convert data to PyTorch tensors
# X_tensor = torch.tensor(X)
# y_tensor = torch.tensor(y)

# Create DataLoader
dataset = TensorDataset(X_tensor_week, y_tensor_week)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the model
class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 48, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.flatten(x)
        feature_output = self.fc1(x)
        x = torch.relu(feature_output)
        x = torch.softmax(self.fc2(x), dim=1)
        return x,feature_output

# Create the model, define the loss function and the optimizer
model = Conv1DModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'modified_inceptive_v3_model.pth')

# # Load the model
# loaded_model = Conv1DModel()
# loaded_model.load_state_dict(torch.load('modified_inceptive_v3_model.pth'))
# loaded_model.eval()
#
# # Evaluation
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in dataloader:
#         outputs = loaded_model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels.argmax(dim=1)).sum().item()
#
# print(f'Loaded model accuracy: {100 * correct / total:.2f}%')