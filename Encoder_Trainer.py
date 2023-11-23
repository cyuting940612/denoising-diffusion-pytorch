import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, Flatten, Reshape, MaxPooling1D, UpSampling1D
from keras.models import Model
import DataProcess as dp
import DataProcess_user as dpu
# Generate or load your dataset (e.g., grayscale images)
# For simplicity, we'll create a simple dataset here.
# Replace this with your actual data loading code.
image_size = (96, 100)

# x_train = np.random.random(size=(num_samples, *image_size))  # Replace with your data

x_train_origin, _, _ = dpu.data_process()

x_train = np.transpose(x_train_origin, (0, 2, 1))  # Replace with your data

# Define the autoencoder architecture
input_img = Input(shape=image_size)

# Encoder
x = Conv1D(32, 3, activation='relu', padding='same')(input_img)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(64, 3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x)

# Decoder
x = Conv1D(64, 3, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(100, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=1000, batch_size=4, validation_split=0.2)

autoencoder.save('autoencoder_model.h5')
