import torch
import torch.nn as nn
import torch.nn.functional as F





class Daily_Classifier(nn.Module):
    def __init__(self):
        super(Daily_Classifier, self).__init__()

        # Define the layers of the neural network
        self.input_layer = nn.Linear(in_features=96, out_features=192)
        self.hidden_layer1 = nn.Linear(in_features=192, out_features=48)
        self.hidden_layer2 = nn.Linear(in_features=48, out_features=20)
        self.relu = nn.ReLU()

        # Define the custom connections for the fifth layer
        self.output_layer1 = nn.Linear(in_features=20, out_features=3)
        self.output_layer2 = nn.Linear(in_features=20, out_features=10)
        self.output_layer3 = nn.Linear(in_features=20, out_features=3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Forward pass for the input and hidden layers
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer1(x)
        x= self.relu(x)
        feature_output = self.hidden_layer2(x)
        # x=self.relu(feature_output)

        # Split the tensor into two parts for the custom connections of the fifth layer
        # Forward pass through the custom connections
        clf_output_1 = F.relu(self.output_layer1(feature_output))
        clf_output_1 = clf_output_1.view(-1, 1, 3)  # Reshape to (batch_size, 14, 2)
        clf_output_1 = self.softmax(clf_output_1)


        clf_output_2 = F.relu(self.output_layer3(feature_output))
        clf_output_2 = clf_output_2.view(-1, 1, 3)  # Reshape to (batch_size, 14, 2)
        clf_output_2 = self.softmax(clf_output_2)

        # Concatenate the outputs from the fifth layer to form the final output
        reg_output = F.relu(self.output_layer2(feature_output))

        # return reg_output,clf_output_1, feature_output

        return reg_output,clf_output_1,clf_output_2, feature_output


class Weekly_Classifier(nn.Module):
    def __init__(self):
        super(Weekly_Classifier, self).__init__()

        # Define the layers of the neural network
        self.input_layer = nn.Linear(in_features=140, out_features=280)
        self.hidden_layer1 = nn.Linear(in_features=280, out_features=70)
        self.hidden_layer2 = nn.Linear(in_features=70, out_features=20)
        self.relu = nn.ReLU()

        # Define the custom connections for the fifth layer
        self.output_layer1 = nn.Linear(in_features=20, out_features=3)
        self.output_layer2 = nn.Linear(in_features=20, out_features=10)
        self.output_layer3 = nn.Linear(in_features=20, out_features=2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Forward pass for the input and hidden layers
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer1(x)
        x= self.relu(x)
        feature_output = self.hidden_layer2(x)
        # x=self.relu(feature_output)

        # Split the tensor into two parts for the custom connections of the fifth layer
        # Forward pass through the custom connections
        clf_output_1 = F.relu(self.output_layer1(feature_output))
        clf_output_1 = clf_output_1.view(-1, 1, 3)  # Reshape to (batch_size, 14, 2)
        clf_output_1 = self.softmax(clf_output_1)


        clf_output_2 = F.relu(self.output_layer3(feature_output))
        clf_output_2 = clf_output_2.view(-1, 1, 2)  # Reshape to (batch_size, 14, 2)
        clf_output_2 = self.softmax(clf_output_2)

        # Concatenate the outputs from the fifth layer to form the final output
        reg_output = F.relu(self.output_layer2(feature_output))

        # return reg_output,clf_output_1, feature_output

        return reg_output,clf_output_1,clf_output_2, feature_output


class Monthly_Classifier(nn.Module):
    def __init__(self):
        super(Monthly_Classifier, self).__init__()

        # Define the layers of the neural network
        self.input_layer = nn.Linear(in_features=80, out_features=160)
        self.hidden_layer1 = nn.Linear(in_features=160, out_features=40)
        self.hidden_layer2 = nn.Linear(in_features=40, out_features=20)
        self.relu = nn.ReLU()

        # Define the custom connections for the fifth layer
        self.output_layer1 = nn.Linear(in_features=20, out_features=3)
        self.output_layer2 = nn.Linear(in_features=20, out_features=10)
        self.output_layer3 = nn.Linear(in_features=20, out_features=2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Forward pass for the input and hidden layers
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer1(x)
        x= self.relu(x)
        feature_output = self.hidden_layer2(x)
        # x=self.relu(feature_output)

        # Split the tensor into two parts for the custom connections of the fifth layer
        # Forward pass through the custom connections
        clf_output_1 = F.relu(self.output_layer1(feature_output))
        clf_output_1 = clf_output_1.view(-1, 1, 3)  # Reshape to (batch_size, 14, 2)
        clf_output_1 = self.softmax(clf_output_1)


        clf_output_2 = F.relu(self.output_layer3(feature_output))
        clf_output_2 = clf_output_2.view(-1, 1, 2)  # Reshape to (batch_size, 14, 2)
        clf_output_2 = self.softmax(clf_output_2)

        # Concatenate the outputs from the fifth layer to form the final output
        reg_output = F.relu(self.output_layer2(feature_output))

        # return reg_output,clf_output_1, feature_output

        return reg_output,clf_output_1,clf_output_2, feature_output