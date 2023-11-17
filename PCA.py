import numpy as np
from sklearn.decomposition import PCA
import DataProcess as dp


def data_PCA(data,reduced):

    # Reshape the data to (1000, 96)
    data_2d = data.reshape(data.shape[0], -1)

    # Apply PCA
    n_components = 50  # Number of principal components to retain
    pca = PCA(n_components=n_components)
    pca.fit(data_2d)

    # Transform the data to the reduced dimension
    reduced_data = pca.transform(data_2d)
    approximate_data = pca.inverse_transform(reduced)
    return reduced_data, approximate_data
