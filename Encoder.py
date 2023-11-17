import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, Flatten, Reshape, MaxPooling1D, UpSampling1D
from keras.models import Model
import DataProcess as dp
from tensorflow.keras.models import load_model


def encoder(input_seq):
    loaded_autoencoder = load_model('autoencoder_model.h5')

    # After training, the encoder part can be used to extract features from raw data
    encoder = Model(loaded_autoencoder.input, outputs=loaded_autoencoder.layers[4].output)

    encoded_features = encoder.predict(input_seq)


    return encoded_features

def decoder(input_seq):
    loaded_autoencoder = load_model('autoencoder_model.h5')

    # After training, the encoder part can be used to extract features from raw data
    # Extract the encoder and decoder
    encoder = Model(inputs=loaded_autoencoder.input,
                    outputs=loaded_autoencoder.layers[4].output)  # Assuming Conv1D layer is at index 1
    decoder_input = Input(shape=(24,6))
    decoder_layer = loaded_autoencoder.layers[5](decoder_input)  # Assuming Conv1D layer is at index -3
    decoder_layer = loaded_autoencoder.layers[6](decoder_layer)  # Assuming Conv1D layer is at index -3
    decoder_layer = loaded_autoencoder.layers[7](decoder_layer)  # Assuming Conv1D layer is at index -3
    decoder_layer = loaded_autoencoder.layers[8](decoder_layer)
    decoder_layer = loaded_autoencoder.layers[9](decoder_layer)
    decoder = Model(inputs=decoder_input, outputs=decoder_layer)

    decoded_features = decoder.predict(input_seq)

    return decoded_features
