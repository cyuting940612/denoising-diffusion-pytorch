import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import DataProcess_filterfree as dp


# Step 1: Importing necessary libraries

# Step 2: Preparing the data
# Let's assume `data` is your 36500 x 96 dataset



data_user = dp.data_process()
data = data_user[:,0:96]
max = np.max(data)
print(max)
data = data / np.max(data)
# Replace this with your actual data
#
# input_dim = data.shape[1]  # 96
# encoding_dim = 20
#
# # Encoder
# input_layer = Input(shape=(input_dim,))
# x = Dense(128, activation='relu')(input_layer)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# x = Dense(64, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# encoded = Dense(encoding_dim, activation='relu')(x)
#
# encoder = Model(inputs=input_layer, outputs=encoded)
#
# # Decoder
# encoded_input = Input(shape=(encoding_dim,))
# x = Dense(64, activation='relu')(encoded_input)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# decoded = Dense(input_dim, activation='sigmoid')(x)  # Using sigmoid for [0, 1] range output
#
# decoder = Model(inputs=encoded_input, outputs=decoded)
#
# # Autoencoder (encoder + decoder)
# autoencoder_input = Input(shape=(input_dim,))
# encoded_repr = encoder(autoencoder_input)
# reconstructed_output = decoder(encoded_repr)
#
# autoencoder = Model(inputs=autoencoder_input, outputs=reconstructed_output)
#
# # Compile the autoencoder
# autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#
# # Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
#
# # Step 4: Training the model
# epochs = 1000
# batch_size = 8
#
# autoencoder.fit(data, data,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 shuffle=True,
#                 validation_split=0.2,
#                 callbacks=[reduce_lr])

# # Step 5: Saving the models
# autoencoder.save('autoencoder_model.h5')
# encoder.save('encoder_model.h5')
# decoder.save('decoder_model.h5')

# Loading the models
loaded_autoencoder = tf.keras.models.load_model('autoencoder_model.h5')
loaded_encoder = tf.keras.models.load_model('encoder_model.h5')
loaded_decoder = tf.keras.models.load_model('decoder_model.h5')

# Step 6: Testing using the loaded models
# encoded_data = loaded_encoder.predict(data_normalized)
# decoded_data = loaded_decoder.predict(encoded_data)

# print("Original data shape:", data.shape)
# print("Encoded data shape:", encoded_data.shape)
# print("Decoded data shape:", decoded_data.shape)
decoded_data = loaded_autoencoder.predict(data)

print(data[0])
print(decoded_data[0])