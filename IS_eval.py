import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy.stats import entropy
import pandas as pd

# Path to your CSV file
file_path = 'filter_free_weekday_testing.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path, header=None)  # Assuming the file has no header

# Convert the DataFrame to a numpy array and then reshape it
reshaped_array = data.values.reshape(8, 96)

# Convert the numpy array to a list of lists
generated_images = reshaped_array.tolist()


def preprocess_images(one_channel_images):
    # Assuming one_channel_images is a list of (1, 1, 96) shaped arrays
    processed_images = []
    for image_list in one_channel_images:
        # Resize and replicate to create a (299, 299, 3) image
        image = np.array(image_list)

        image_square = np.tile(image, (96, 1))  # This creates a (96, 96) image

        # Convert to 3 channels by repeating the array
        image_3ch = np.repeat(image_square[:, :, np.newaxis], 3, axis=2)

        # Convert to a PyTorch tensor and ensure it's float32
        image_tensor = torch.tensor(image_3ch, dtype=torch.float32)

        image_tensor = image_tensor.permute(2, 0, 1)
        # Resize to (299, 299) for the Inception model
        resized_image = TF.resize(image_tensor, [299, 299])

        # Adjust the tensor to be in (C, H, W) format as expected by PyTorch models
        resized_image = resized_image.unsqueeze(0)

        processed_images.append(resized_image)

        # Stack the list of tensors into a single tensor
    return torch.cat(processed_images, dim=0)

def calculate_inception_score(images, cuda=False, batch_size=4, resize=False, splits=10):
    N = len(images)
    assert N > batch_size

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    if cuda:
        inception_model.cuda()

    def get_preds(x):
        if resize:
            x = TF.resize(x, [299, 299])
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.cuda() if cuda else batch
        preds[i * batch_size:i * batch_size + batch.size(0)] = get_preds(batch)

    scores = []

    py = np.mean(preds, axis=0)
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))
    return np.exp(np.mean(scores))

# Example usage
# Assuming `generated_images` is your list of (1, 1, 96) images
preprocessed_images = preprocess_images(generated_images)
inception_score = calculate_inception_score(preprocessed_images, cuda=False, batch_size=4, resize=True, splits=10)
print("Inception Score: ", inception_score)