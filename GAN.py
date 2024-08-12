import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
latent_dim = 100
sample_length = 96
num_classes = 2
label_dim = 1


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, sample_length, label_dim):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, sample_length),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels).view(labels.size(0), -1)
        gen_input = torch.cat((noise, label_input), -1)
        sample = self.model(gen_input)
        return sample


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, sample_length, label_dim):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.Linear(sample_length + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, sample, labels):
        label_input = self.label_emb(labels).view(labels.size(0), -1)
        disc_input = torch.cat((sample, label_input), -1)
        validity = self.model(disc_input)
        return validity


# Initialize models
generator = Generator(latent_dim, sample_length, label_dim)
discriminator = Discriminator(sample_length, label_dim)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Load your data here
# X_train should be your array of samples (36500, 96)
# y_train should be your labels (36500,)
X_train = torch.randn(36500, sample_length)  # Example data, replace with your actual data
y_train = torch.randint(0, 2, (36500,))  # Example labels, replace with your actual labels

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(dataloader):
        # Adversarial ground truths
        valid = torch.ones(samples.size(0), 1)
        fake = torch.zeros(samples.size(0), 1)

        # ---------------------
        #  Train Generator
        # ---------------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(samples.size(0), latent_dim)
        gen_labels = torch.randint(0, num_classes, (samples.size(0),))

        # Generate a batch of samples
        gen_samples = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_samples, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real samples
        real_pred = discriminator(samples, labels)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake samples
        fake_pred = discriminator(gen_samples.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

torch.save(generator.state_dict(), "generator_final.pth")

print("Training complete!")