import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


def generate_matrix_1(rows, cols):
    # Initialize the matrix with zeros
    matrix_1 = np.zeros((rows, cols), dtype=int)

    # Fill the matrix according to the pattern
    for i in range(0,rows,8):
        start_index = int(i/2)
        for j in range(8):
            if start_index + 3 < cols:
                matrix_1[i+j, start_index] = 1
                matrix_1[i+j, start_index + 1] = 1
                matrix_1[i+j, start_index + 2] = 1
                matrix_1[i+j, start_index + 3] = 1
                # matrix_1[i+j, start_index + 4] = 1
                # matrix_1[i+j, start_index + 5] = 1
                # matrix_1[i+j, start_index + 6] = 1
                # matrix_1[i+j, start_index + 7] = 1

    return matrix_1

def generate_matrix_2(rows, cols):
    # Initialize the matrix with zeros
    matrix_2 = np.zeros((rows, cols), dtype=int)

    # Fill the matrix according to the pattern
    for i in range(0,rows,2):
        start_index = i*4
        for j in range(2):
            if start_index + 1 < cols:
                matrix_2[i+j, start_index] = 1
                matrix_2[i+j, start_index + 1] = 1
                matrix_2[i + j, start_index + 2] = 1
                matrix_2[i + j, start_index + 3] = 1
                matrix_2[i + j, start_index + 4] = 1
                matrix_2[i + j, start_index + 5] = 1
                matrix_2[i + j, start_index + 6] = 1
                matrix_2[i + j, start_index + 7] = 1
    return matrix_2

def generate_matrix_3(rows, cols):
    # Initialize the matrix with zeros
    matrix_3 = np.zeros((rows, cols), dtype=int)

    # Fill the matrix according to the pattern
    for i in range(rows):
        start_index = 2*i
        if start_index < cols:
            matrix_3[i, start_index] = 1
            matrix_3[i, start_index+1] = 1


    return matrix_3

# Define the number of rows and columns
rows_1 = 96*31*2
cols_1 = 96*31
rows_2 = 1488
cols_2 = 96*31*2
rows_3 = 744
cols_3 = 1488

# Generate the matrix
matrix_1 = generate_matrix_1(rows_1, cols_1)
matrix_2 = generate_matrix_2(rows_2, cols_2)
matrix_3 = generate_matrix_3(rows_3, cols_3)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU.")


class Generalized_Classifier(nn.Module):
    def __init__(self):
        super(Generalized_Classifier, self).__init__()

        self.input_size = 96*31
        self.hidden_size_1 = 96*31*2
        self.hidden_size_2 = 1488
        self.output_size = 744

        # Define the layers of the neural network
        self.input_layer = nn.Linear(self.input_size, self.hidden_size_1, bias=False)
        self.hidden_layer_1 = nn.Linear(self.hidden_size_1, self.hidden_size_2, bias=False)
        self.hidden_layer_2 = nn.Linear(self.hidden_size_2, self.output_size, bias=False)
        self.relu = nn.ReLU()

        # Define the custom connections for the fifth layer
        self.output_layer1 = nn.Linear(in_features=self.output_size, out_features=3)
        self.output_layer2 = nn.Linear(in_features=self.output_size, out_features=10)
        self.output_layer3 = nn.Linear(in_features=self.output_size, out_features=4)
        self.softmax = nn.Softmax(dim=-1)

        self.input_mask = torch.tensor(matrix_1, dtype=torch.float32, device=device)
        self.hidden_mask = torch.tensor(matrix_2, dtype=torch.float32, device=device)
        self.feature_mask = torch.tensor(matrix_3, dtype=torch.float32, device=device)
    def forward(self, x):
        a = self.input_layer.weight
        b = self.input_mask
        input_weights = a*b
        # tensor_detached = input_weights.T.detach()
        # weights_numpy = tensor_detached.numpy()
        # Forward pass for the input and hidden layers
        x = F.leaky_relu(F.linear(x, input_weights))
        hidden_weights = self.hidden_layer_1.weight * self.hidden_mask
        # monitor_1 = F.linear(x, hidden_weights).detach().numpy()
        # monitor_2 = x.detach().numpy()
        x = F.leaky_relu(F.linear(x, hidden_weights))
        feature_weights = self.hidden_layer_2.weight * self.feature_mask
        # monitor_3 = F.linear(x, feature_weights).detach().numpy()
        feature_output = F.linear(x, feature_weights)
        # x=self.relu(feature_output)

        # Split the tensor into two parts for the custom connections of the fifth layer
        # Forward pass through the custom connections
        clf_output_1 = F.leaky_relu(self.output_layer1(feature_output))
        clf_output_1 = clf_output_1.view(-1, 1, 3)  # Reshape to (batch_size, 14, 2)
        clf_output_1 = self.softmax(clf_output_1)


        clf_output_2 = F.leaky_relu(self.output_layer3(feature_output))
        clf_output_2 = clf_output_2.view(-1, 1, 4)  # Reshape to (batch_size, 14, 2)
        clf_output_2 = self.softmax(clf_output_2)

        # Concatenate the outputs from the fifth layer to form the final output
        reg_output = F.leaky_relu(self.output_layer2(feature_output))

        # return reg_output,clf_output_1, feature_output

        return reg_output,clf_output_1,clf_output_2, feature_output

# network = Aggregated_Classifier()
# # criterion = nn.CrossEntropyLoss()
# regression_criterion = nn.MSELoss()
# classification_criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(network.parameters(), lr=0.001)
# data = 0
# # Forward pass
# # outputs,_ = network(input)
# output = network(data)
