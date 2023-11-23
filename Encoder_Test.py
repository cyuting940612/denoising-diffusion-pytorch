import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, Flatten, Reshape, MaxPooling1D, UpSampling1D
from keras.models import Model
import DataProcess as dp
from tensorflow.keras.models import load_model
import DataProcess_user as dpu

loaded_autoencoder = load_model('autoencoder_model.h5')

# After training, the encoder part can be used to extract features from raw data
encoder = Model(inputs=loaded_autoencoder.input,
                outputs=loaded_autoencoder.layers[4].output)  # Assuming Conv1D layer is at index 1
decoder_input = Input(shape=(24, 64))
decoder_layer = loaded_autoencoder.layers[5](decoder_input)  # Assuming Conv1D layer is at index -3
decoder_layer = loaded_autoencoder.layers[6](decoder_layer)  # Assuming Conv1D layer is at index -3
decoder_layer = loaded_autoencoder.layers[7](decoder_layer)  # Assuming Conv1D layer is at index -3
decoder_layer = loaded_autoencoder.layers[8](decoder_layer)
decoder_layer = loaded_autoencoder.layers[9](decoder_layer)
decoder = Model(inputs=decoder_input, outputs=decoder_layer)

test,_,_ = dpu.data_process()
test_np = np.transpose(test,(0,2,1))
test_seq = test_np[0:1,:,:]
encoded_features = encoder.predict(test_seq)
decoded_features = decoder.predict(encoded_features)
print(decoded_features)


