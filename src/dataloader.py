import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

def get_dataloaders(data_path, batch_size):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    transform_test = T.Compose([
        T.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=False, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=False, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
