import torch
import torch.nn as nn

# Latent space dimension
Z_DIM = 100

class Generator(nn.Module):
    """
    Generator network as specified in Assignment 3.
    Input: (BATCH_SIZE, 100) noise vector
    Output: (BATCH_SIZE, 1, 28, 28) image
    """
    def __init__(self, z_dim=Z_DIM):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # FC layer to 7x7x128, then reshape
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            
            # ConvTranspose2D: 128 -> 64 (output 14x14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # ConvTranspose2D: 64 -> 1 (output 28x28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """
    Discriminator network as specified in Assignment 3.
    Input: (BATCH_SIZE, 1, 28, 28) image
    Output: Single probability (real/fake)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Conv2D: 1 -> 64 (output 14x14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2D: 64 -> 128 (output 7x7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten and apply Linear layer
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid() # To get a single probability output
        )

    def forward(self, input):
        return self.main(input)

def get_model(model_name: str):
    """
    Returns the GAN models.
    """
    if model_name == "GAN":
        generator = Generator(Z_DIM)
        discriminator = Discriminator()
        return generator, discriminator
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'GAN' is supported.")