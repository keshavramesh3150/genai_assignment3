import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir: str, batch_size: int = 128, train: bool = True, download: bool = True):
    """
    Creates a data loader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] to match the Generator's Tanh output
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return loader