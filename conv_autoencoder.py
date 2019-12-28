'''
    file: conv_autoencoder.py
    author: Adam Good
    Description: A Convolutional Auto Encoder Example
'''
import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from data_managment import get_dataloaders
from autoencoder_utils import train_model

class ConvAutoEncoder(nn.Module):
    '''
        This gon be Convolutional Auto Encoder
    '''
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.ec1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.em1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ec2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.em2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.efc1 = nn.Linear(8*7*7, 32)

        self.dfc1 = nn.Linear(32, 8*7*7)
        self.dm1 = nn.Upsample(scale_factor=2, mode='nearest') #nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dc1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.dm2 = nn.Upsample(scale_factor=2, mode='nearest') #nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dc2 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=2)
        



    def encode(self, x):
        x = F.elu(self.ec1(x))
        x = self.em1(x)
        x = F.elu(self.ec2(x))
        x = self.em2(x)
        x = x.view(-1, 1, 8*7*7)
        x = self.efc1(x)
        return x

    def decode(self, x):
        x = F.elu(self.dfc1(x))
        x = x.view(-1, 8, 7, 7)
        x = self.dm1(x)
        x = F.elu(self.dc1(x))
        x = self.dm2(x)
        x = torch.sigmoid(self.dc2(x))
        return x

    def forward(self, x):
        x = self.decode(self.encode(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def main():
    parser = argparse.ArgumentParser(description="Configure Training of Auto Encoder")
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs for Training')
    parser.add_argument('-bs', '--batchsize', type=int, help='Training Batch Size')
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('-v', '--verify', action='store_true', help='Run Verification Epochs')
    parser.add_argument('-p', '--trainingpath', type=str, default='./models', help='Directory for Saving State During Training')
    parser.add_argument('-o', '--output', type=str, help='Name for Output File(s)')
    parser.add_argument('-C', '--clean', action='store_true', help='Cleans out Training Data')
    args = parser.parse_args()

    batch_size = args.batchsize
    num_epochs = args.epochs
    path = args.trainingpath
    output = args.output

    if not os.path.exists(path):
        os.makedirs(path)

    train_loader, test_loader = get_dataloaders(batch_size, batch_size)
    autoencoder = ConvAutoEncoder()
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.learningrate)

    train_model(autoencoder, optimizer, num_epochs, train_loader, test_loader, verify=args.verify, trainingpath=path, name=output)
    if args.clean:
        from shutil import rmtree
        rmtree(path)

if __name__ == '__main__':
    main()
