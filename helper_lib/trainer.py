import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from helper_lib.image_generator import generate_samples

def train_gan(generator: nn.Module, discriminator: nn.Module, data_loader: DataLoader, 
                criterion: nn.Module, opt_gen: optim.Optimizer, opt_disc: optim.Optimizer, 
                z_dim: int, device: str = 'cpu', epochs: int = 10, sample_interval: int = 1):
    """
    Trains a GAN model.
    """
    generator.to(device)
    discriminator.to(device)
    
    # Create fixed noise for visualizing progress
    fixed_noise = torch.randn(64, z_dim).to(device)

    print("Starting training...")
    
    for epoch in range(epochs):
        # Track epoch losses
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0

        for batch_idx, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_images)
            loss_d_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_disc.step()
            epoch_loss_d += loss_d.item()

            # Train Generator
            generator.zero_grad()
            
            # We "fool" the discriminator
            fake_output_for_g = discriminator(fake_images)
            loss_g = criterion(fake_output_for_g, real_labels) # Use real labels
            
            loss_g.backward()
            opt_gen.step()
            epoch_loss_g += loss_g.item()

        # Calculate and print average losses for the epoch
        avg_loss_d = epoch_loss_d / len(data_loader)
        avg_loss_g = epoch_loss_g / len(data_loader)
        
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"[Avg D loss: {avg_loss_d:.4f}] [Avg G loss: {avg_loss_g:.4f}]"
        )

        # After each epoch, generate and save sample images
        if (epoch + 1) % sample_interval == 0:
            print(f"Saving sample images for epoch {epoch+1}...")
            generate_samples(generator, fixed_noise, device, f"epoch_{epoch+1}_samples.png")

    print("Finished Training")
    return generator, discriminator