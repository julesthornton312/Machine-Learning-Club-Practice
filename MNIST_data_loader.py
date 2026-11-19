from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_MNIST():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root= "./data", 
        train = True, 
        download = True, 
        transform = transform
    )

    test_dataset= datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = 32, 
        shuffle = True, 
        drop_last = True)

    test_loader = DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle = True,
        drop_last = True
    )

    return train_loader, test_loader
