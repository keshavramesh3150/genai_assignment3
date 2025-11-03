import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

@torch.no_grad()
def generate_samples(generator: nn.Module, noise: torch.Tensor, device: str, save_path: str):
    """
    Generates a grid of images from the generator and saves them to a file.
    """
    generator.to(device)
    generator.eval() # Set model to evaluation mode
    
    # Generate fake images
    fake_images = generator(noise)
    
    # Un-normalize images from [-1, 1] to [0, 1]
    fake_images = (fake_images * 0.5) + 0.5
    
    # Save a grid of images
    save_image(fake_images, save_path, nrow=8)
    
    generator.train() # Set model back to training mode