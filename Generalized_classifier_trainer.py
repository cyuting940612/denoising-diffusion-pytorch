import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import DataProcess_Classifier as dpc
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import Generalized_classifier as gc

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU.")

def normalize_to_range(data, a=0, b=255):
    # Calculate min and max for each column
    col_mins = np.min(data, axis=1, keepdims=True)
    col_maxs = np.max(data, axis=1, keepdims=True)
    #
    # col_mins = np.min(data, axis=0)
    # col_maxs = np.max(data, axis=0)

    # Avoid division by zero
    col_ranges = col_maxs - col_mins
    col_ranges[col_ranges == 0] = 1

    # Normalize each column to range [a, b]
    normalized_data = a + ((data - col_mins) / col_ranges) * (b - a)
    return normalized_data


data, label = dpc.data_process()

# df_all_highlow = dpchl.data_process()

# X = np.zeros((365,3,96))
# y = np.zeros((365))
# for i in range(365):
#     for j in range(96):
#         X[i,0,j] = df_all[i*96+j,0]
#         X[i, 1, j] = df_all[i * 96 + j,1]
#         X[i, 2, j] = df_all[i * 96 + j,2]
#         y[i] = df_all[i*96,3]

# X_week = df_all_week[:,0:1,:]
# y_week = df_all_week[:,1,0]
# y_month = df_all_week[:,2,0]
# y_highlow = df_all_week[:,2+1,0]
# y_min =df_all_week[:,3+1,0]
# y_max =df_all_week[:,4+1,0]
# y_range =df_all_week[:,5+1,0]
# y_var =df_all_week[:,6+1,0]
# y_peaklength = df_all_week[:,7+1,0]
# y_lowlength = df_all_week[:,8+1,0]
# y_peak07 = df_all_week[:,9+1,0]
# y_peak815 = df_all_week[:,10+1,0]
# y_peak1623 = df_all_week[:,11+1,0]
# y_low07 = df_all_week[:,12+1,0]
# y_low815 = df_all_week[:,13+1,0]
# y_low1623 = df_all_week[:,14+1,0]
# y_skew =df_all_week[:,15+1,0]
# y_kurt =df_all_week[:,16+1,0]
# y_iqr =df_all_week[:,17+1,0]
# y_cv =df_all_week[:,18+1,0]


# y_combine = np.column_stack((y_week))
# y_combine = np.column_stack((y_peaklength,y_lowlength,y_peak07,y_peak815,y_peak1623,y_low07,y_low815,y_low1623,y_skew,y_kurt,y_iqr,y_cv,y_min,y_max,y_range,y_var))


# y_combine = np.column_stack((y_peaklength,y_lowlength,y_skew,y_kurt,y_iqr,y_cv))

# y_combine = np.column_stack((y_week, y_highlow,y_min,y_max,y_range,y_var,y_peaklength,y_lowlength,y_peak07,y_peak815,y_peak1623,y_low07,y_low815,y_low1623))
# reshaped_array = y_combine.reshape((36500+1200, 16))# y_combine = np.column_stack((y_week, y_highlow,y_min,y_max,y_range,y_var,y_peaklength,y_lowlength,y_peak07,y_peak815,y_peak1623,y_low07,y_low815,y_low1623))
# reshaped_arr_week = X_week.reshape(36500+1200, 96)
# normalized_generated_week_reshaped_array = normalize_to_range(reshaped_arr_week, a=0, b=1)

X_tensor_week = torch.tensor(data, dtype=torch.float32)
# y_tensor_week = torch.tensor(label[:,0], dtype=torch.float32)
y_reg = torch.tensor(label[:,2:13], dtype=torch.float32)
y_clf_1 = torch.tensor(label[:,0],dtype=torch.long)
y_clf_2 = torch.tensor(label[:,1],dtype=torch.long)


# Define the sizes for your training and test sets

# Assuming we're dealing with a regression task and using Mean Squared Error loss
# network = ac.Aggregated_Classifier()
# criterion = nn.CrossEntropyLoss()
network = gc.Generalized_Classifier().to(device)
regression_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

# dataset = TensorDataset(X_tensor_week, y_tensor_week)
dataset = TensorDataset(X_tensor_week, y_reg, y_clf_1,y_clf_2)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

training_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testing_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Dummy dataset: 10 samples with 96 features and random targets
# Replace this with your actual dataset
# Assuming the output is of size 2

# Training loop
for epoch in range(1000):
    running_loss = 0.0
    # Number of epochs
    for input, batch_y_reg, batch_y_clf_1, batch_y_clf_2 in training_dataloader:  # Iterate over each sample
        # Zero the parameter gradients
        optimizer.zero_grad()

        input = input.to(device)
        batch_y_reg = batch_y_reg.to(device)
        batch_y_clf_1 = batch_y_clf_1.to(device)
        batch_y_clf_2 = batch_y_clf_2.to(device)
        # Forward pass
        # outputs,_ = network(input)
        reg_output, clf_output_1,clf_output_2,_ = network(input)
        #
        # # label = label.long()
        # # Compute loss
        # loss = 0
        # for i in range(6):
        #     loss += criterion(outputs[:, i, :], label[:, i])
        # # loss = criterion(outputs, label)
        # # Backward pass and optimize
        # loss.backward()
        # optimizer.step()
        # running_loss += loss.item()
        # Calculate losses
        loss_reg = regression_criterion(reg_output, batch_y_reg)
        loss_clf_1 = classification_criterion(clf_output_1[:, 0, :], batch_y_clf_1[:])
        loss_clf_2 = classification_criterion(clf_output_2[:, 0, :], batch_y_clf_2[:])


        # Combine losses (you can adjust the weights)
        total_loss = loss_reg + loss_clf_1+loss_clf_2

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()


    # Print statistics
    print(f'Epoch {epoch+1}/{100}, Loss: {running_loss/len(training_dataloader)}')
    # print(f'Epoch {epoch+1}/{100}, Loss: {running_loss/len(training_dataloader)}')


torch.save(network.state_dict(), 'generalized_classifier_model_monthly.pth')


# model = ac.Aggregated_Classifier()
# model.load_state_dict(torch.load('aggregated_classifier_model_Combined.pth'))
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     incorrect_predict = []
#     incorrect_label = []
#     for inputs, batch_y_reg, batch_y_clf in testing_dataloader:  # Assuming test_loader is a DataLoader for your test data
#         # Forward pass
#         reg_output, clf_output,_ = model(inputs)
#
#         # Reshape labels to match the output shape
#         # labels = labels.long()
#
#         # Compute accuracy
#         predicted_clf = torch.argmax(clf_output, dim=-1)
#         # total += labels.numel()  # Total number of elements
#         # correct_index = predicted == labels
#         # correct += (predicted == labels).sum().item()
#         predicted_clf_numpy = predicted_clf.numpy()
#         labels_numpy = batch_y_clf.numpy()
#
#         for i in range(predicted_clf_numpy.shape[0]):
#             are_rows_equal = np.array_equal(predicted_clf_numpy[i,:], labels_numpy[i])
#             if are_rows_equal is False:
#                 incorrect_predict.append(predicted_clf_numpy[i,:])
#                 incorrect_label.append(labels_numpy[i])
#                 total += 1
#             else:
#                 correct += 1
#                 total += 1
#
#     # index = np.zeros((incorrect_label[0].shape[0]))
#     # for j in range(len(incorrect_label)):
#     #     for k in range(incorrect_label[0].shape[0]):
#     #         if incorrect_label[j][k] != incorrect_predict[j][k]:
#     #             index[k]+=1
#
#     index = 0
#     for j in range(len(incorrect_label)):
#         if incorrect_label[j] != incorrect_predict[j]:
#             index += 1
#
#     print(f'Accuracy: {100 * correct / total:.2f}%')
#     print(index)

# weekend_file_path = 'filter_free_weekend_testing_large.csv'
# # Read the CSV file into a pandas DataFrame
# generated_weekend_data = pd.read_csv(weekend_file_path, header=None)  # Assuming the file has no header
#
# # Convert the DataFrame to a numpy array and then reshape it
# generated_weekend_reshaped_array = generated_weekend_data.values.reshape(100, 96)
#
# # Convert the numpy array to a list of lists
# generated_weekend_images = generated_weekend_reshaped_array.tolist()
# generated_weekend_images_numpy = np.array(generated_weekend_images)
# X_tensor_generated_weekend = torch.tensor(generated_weekend_images_numpy, dtype=torch.float32)
#
# dataset = TensorDataset(X_tensor_generated_weekend, y_tensor_week)
# testing_dataloader = DataLoader(dataset,batch_size=16, shuffle=True)
#
# correct = 0
# total = 0
# network = ac.Aggregated_Classifier()
# network.load_state_dict(torch.load('aggregated_classifier_model_10.pth'))
# # Assuming your model is named 'model' and your test DataLoader is named 'test_loader'
# network.eval()
# with torch.no_grad():
#     for inputs, labels in testing_dataloader:
#         outputs,_ = network(inputs)  # Get model predictions
#
#         predictions = (outputs > 0.5).int()
#
#         # Calculate accuracy
#         correct_predictions = (predictions == labels).all(dim=1).float()
#         correct += correct_predictions.sum().item()
#         total += labels.size(0)
#
# print(f'Accuracy of the model on the test images: {100 * correct / total}%')