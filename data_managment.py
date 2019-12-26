'''
    File: data_management.py
    Author: Adam Good
    Description: Contains functions for easily managing data loaders
'''
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(train_batch_size, test_batch_size, transform=transforms.ToTensor(), dataset='mnist'):
    '''
        Create MNIST dataloaders with specified batch sizes and transforms
    '''
    if dataset == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST('./data', train=True, transform=transform, download=True),
            batch_size=train_batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            datasets.MNIST('./data', train=False, transform=transform, download=True),
            batch_size=test_batch_size,
            shuffle=True
        )

        return train_loader, test_loader

    return None, None
