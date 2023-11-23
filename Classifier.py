import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import DataProcess_Classifier as dpc
import tensorflow as tf
from tensorflow import keras


# filename_lmp = 'Classifier.csv'
# df_lmp = pd.read_csv(filename_lmp)['LMP'].to_numpy()
# df_load = pd.read_csv(filename_lmp)['Load'].to_numpy()
# df_temp = pd.read_csv(filename_lmp)['Temperature'].to_numpy()
# df_label = pd.read_csv(filename_lmp)['Label']
#


# X_train1 = X_tensor[0:301,:,:]
# X_train2 = X_tensor[366:700,:,:]
# X_train = tf.concat([X_train1, X_train2], axis=0)
# y_train1 = y_tensor[0:301]
# y_train2 = y_tensor[366:700]
# y_train = tf.concat([y_train1, y_train2], axis=0)
# X_test1 = X_tensor[301:366,:,:]
# X_test2 = X_tensor[700:730,:,:]
# X_test = tf.concat([X_test1, X_test2], axis=0)
# y_test1 = y_tensor[301:366]
# y_test2 = y_tensor[700:730]
# y_test = tf.concat([y_test1, y_test2], axis=0)
#
#
# # Define a Sequential model
# model = keras.Sequential()
#
# # Add the input layer (Flatten layer to convert input into a 1D array)
# model.add(keras.layers.Flatten(input_shape=(3, 96)))  # Example input shape for 28x28 images
#
# # Add one or more fully connected hidden layers
# model.add(keras.layers.Dense(units=32, activation='relu'))
#
# # Add the output layer with the number of classes you have
# num_classes = 1  # Change this to the number of classes in your classification task
# model.add(keras.layers.Dense(units=num_classes, activation='sigmoid'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Print a summary of the model architecture
# model.summary()
#
# num_epochs = 500 # You can adjust the number of epochs
# batch_size = 8  # You can adjust the batch size
#
# model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
#
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_accuracy}")
# Define a simple 2D classifier model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(24*16, 1)

    def forward(self, x):
        x = x.contiguous().view (-1,24*16)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return x


# Create a DataLoader for the data

# df_all = dpc.DataProcess_Classifier()
#
# X = np.zeros((730,3,96))
# y = np.zeros((730))
# for i in range(730):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]
#
# # X = np.random.randn(100, 3, 96)  # 100 data points with 3 channels and 96 time steps
# # y = np.random.randint(0, 2, 100)  # Binary classification (0 or 1)
#
# # Convert data to PyTorch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# # Initialize the model, loss function, and optimizer
# model = SimpleClassifier()
# criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss for binary classification
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
#
# # Training loop
# epochs = 1000
# for epoch in range(epochs):
#     running_loss = 0.0
#     for inputs, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels.view(-1, 1))
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
# # Save the trained model
# torch.save(model.state_dict(), '2d_classifier_model.pth')
# print("Finished Training")
#
# # Load the trained model
# model = SimpleClassifier()  # Assuming you've defined your model
# model.load_state_dict(torch.load('2d_classifier_model.pth'))
# model.eval()
#
# # Inference
# X_tensor_test = torch.tensor(X[200,:,:], dtype=torch.float32)
# with torch.no_grad():
#     test_inputs = X_tensor_test
#     predicted = model(test_inputs)
#     probability = torch.sigmoid(predicted)
#     print(f"Predicted Probability: {probability.item()}")
