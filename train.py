import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model, Z_DIM
from helper_lib.trainer import train_gan
import os

def main():
    # Parameters
    BATCH_SIZE = 128
    EPOCHS = 50  
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    DATA_DIR = './data'
    GENERATOR_SAVE_PATH = 'gan_generator.pth'
    DISCRIMINATOR_SAVE_PATH = 'gan_discriminator.pth'

    # Check for device (MPS > CUDA > CPU)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Get Data Loader
    print("Loading MNIST data...")
    train_loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE)
    print("Data loaded.")

    # Get Models
    generator, discriminator = get_model("GAN")

    # Define Loss and Optimizers
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Train the GAN
    trained_generator, trained_discriminator = train_gan(
        generator=generator,
        discriminator=discriminator,
        data_loader=train_loader,
        criterion=criterion,
        opt_gen=opt_gen,
        opt_disc=opt_disc,
        z_dim=Z_DIM,
        device=device,
        epochs=EPOCHS
    )

    # Save the Models
    print(f"Saving models...")
    torch.save(trained_generator.state_dict(), GENERATOR_SAVE_PATH)
    torch.save(trained_discriminator.state_dict(), DISCRIMINATOR_SAVE_PATH)
    print(f"Generator saved to {GENERATOR_SAVE_PATH}")
    print(f"Discriminator saved to {DISCRIMINATOR_SAVE_PATH}")
    print("Training complete.")

if __name__ == "__main__":
    main()