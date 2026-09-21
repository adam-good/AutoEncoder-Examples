'''
    File: simple_autoencoder.py
    Author: Adam Good
    Description:    An example of a variational autoencoder on the MNIST dataset
                    VAE Paper: https://arxiv.org/abs/1312.6114
'''
import time
import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from data_managment import get_dataloaders
from autoencoder_utils import train_model, print_epoch

INPUT_SIZE = 28*28
H1_SIZE = 512
H2_SIZE = 128
BOTTLENECK_SIZE = 32

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()

        self.eh1 = nn.Linear(INPUT_SIZE, H1_SIZE)
        self.eh2 = nn.Linear(H1_SIZE, H2_SIZE)
        self.mean_layer = nn.Linear(H2_SIZE, BOTTLENECK_SIZE)
        self.logvar_layer = nn.Linear(H2_SIZE, BOTTLENECK_SIZE)
        self.dh2 = nn.Linear(BOTTLENECK_SIZE, H2_SIZE)
        self.dh1 = nn.Linear(H2_SIZE, H1_SIZE)
        self.output = nn.Linear(H1_SIZE, INPUT_SIZE)

    def encode(self, x):
        x = F.elu(self.eh1(x))
        x = F.elu(self.eh2(x))
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    #TODO: Look up why
    def sample(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        z = F.elu(self.dh2(z))
        z = F.elu(self.dh1(z))
        z = torch.sigmoid(self.output(z))
        return z

    def forward(self, x, retArgs=True, *argv):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        y = self.decode(z)
        if retArgs:
            return y, mean, logvar
        else:
            return y

    def run_epoch(self, optimizer, loss_fn, data_loader, epoch, log_interval=1, training=True):
        if training:
            self.train()
            epoch_type = "Training"
        else:
            self.eval()
            epoch_type = "Validation"
        
        dataset_size = len(data_loader.dataset)
        epoch_loss = 0.0
        num_items = 0
        batch_idx = 0
        start = time.time()

        for batch_idx, (data, _) in enumerate(data_loader):
            if training:
                optimizer.zero_grad()

            data = data.reshape(-1, 1, INPUT_SIZE)

            output, mean, logvar = self(data)
            loss = loss_fn(data, output, mean, logvar)
            epoch_loss += loss.item()
            num_items += len(data)
            if training:
                loss.backward()
                optimizer.step()

            if batch_idx % log_interval == 0:
                elapsed_time = time.time() - start
                print_epoch(epoch, epoch_loss, num_items, dataset_size, elapsed_time, epoch_type)

        elapsed_time = time.time() - start
        print_epoch(epoch, epoch_loss, num_items, dataset_size, elapsed_time, epoch_type, end='\n')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

# Mean Squared Error + KL Divergence
def vae_loss_fn(x, y, mean, logvar):
    # mse = F.mse_loss(x, y)
    bce = F.binary_cross_entropy(y, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp() )
    return bce + kld

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
    autoencoder = VAE()
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.learningrate)

    train_model(autoencoder, optimizer, vae_loss_fn, num_epochs, train_loader, test_loader, verify=args.verify, trainingpath=path, name=output)
    if args.clean:
        from shutil import rmtree
        rmtree(path)

if __name__ == '__main__':
    main()
