import torch
from GAN import Generator

# Assume the trained generator model is loaded or available in memory
# generator = ... (your trained generator)

def generate_samples(generator, num_samples, label, latent_dim=100):
    """
    Generate samples using the trained generator conditioned on a specific label.

    Args:
        generator (nn.Module): Trained generator model.
        num_samples (int): Number of samples to generate.
        label (int): The label to condition the generation on (-1 or 1).
        latent_dim (int): Dimensionality of the latent space.

    Returns:
        torch.Tensor: Generated samples.
    """
    # Convert label to tensor
    label_tensor = torch.full((num_samples,), label, dtype=torch.long)

    # Sample random noise
    noise = torch.randn(num_samples, latent_dim)

    # Generate samples
    with torch.no_grad():
        generated_samples = generator(noise, label_tensor)

    return generated_samples

# Example usage
num_samples = 10  # Number of samples to generate
label = 1  # Label to condition the generation on (can be -1 or 1)

batch_size = 64
learning_rate = 0.0002
num_epochs = 100
latent_dim = 100
sample_length = 96
num_classes = 2
label_dim = 1

generator = Generator(latent_dim, sample_length, label_dim)
generator.load_state_dict(torch.load('generator_final.pth'))
generator.eval()

# Generate samples conditioned on the specified label
generated_samples = generate_samples(generator, num_samples, label)