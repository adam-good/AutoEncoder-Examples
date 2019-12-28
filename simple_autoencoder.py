'''
    File: simple_autoencoder.py
    Author: Adam Good
    Description: An example of an autoencoder on the MNIST dataset
'''
import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from data_managment import get_dataloaders
from autoencoder_utils import train_model

SAE_LAYERS = [28*28, 512, 256, 128, 64, 32] 

class SimpleAutoEncoder(nn.Module):
    '''
        A basic autoencoder
        input -> hidden -> bottleneck -> hidden -> output
    '''
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.input_size = SAE_LAYERS[0]
        self.bottleneck_size = SAE_LAYERS[-1]

        left = SAE_LAYERS[:-1]
        right = SAE_LAYERS[1:]
        encoding_layers = [nn.Linear(l, r) for l, r in zip(left, right)]
        self.bottleneck_layer = encoding_layers.pop()
        self.encoding_layers = nn.ModuleList(encoding_layers)

        left, right = right[::-1], left[::-1]
        decoding_layers = [nn.Linear(l, r) for l, r in zip(left, right)]
        self.output_layer = decoding_layers.pop()
        self.decoding_layers = nn.ModuleList(decoding_layers)

    def encode(self, original):
        '''
            Encode x to bottleneck
        '''
        encoding = original.view(-1, 1, self.input_size)
        for layer in self.encoding_layers:
            encoding = F.relu(layer(encoding))
        encoding = self.bottleneck_layer(encoding)
        return encoding

    def decode(self, encoding):
        '''
            Decode x back into the original data
        '''
        reconstruction = encoding
        for layer in self.decoding_layers:
            reconstruction = F.elu(layer(reconstruction))
        reconstruction = torch.sigmoid(self.output_layer(reconstruction))
        reconstruction = reconstruction.view(-1, 1, 28, 28)
        return reconstruction

    def forward(self, x):
        '''
            Encode and reconstruct the input x
        '''
        return self.decode(self.encode(x))

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
    autoencoder = SimpleAutoEncoder()
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.learningrate)

    train_model(autoencoder, optimizer, num_epochs, train_loader, test_loader, verify=args.verify, trainingpath=path, name=output)
    if args.clean:
        from shutil import rmtree
        rmtree(path)

if __name__ == '__main__':
    main()
