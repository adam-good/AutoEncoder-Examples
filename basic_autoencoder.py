'''
    File: basic_autoencoder.py
    Author: Adam Good
    Description: An example of an autoencoder on the MNIST dataset
'''
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import argparse
import os

from data_managment import get_dataloaders

SEED = 8675309
torch.random.manual_seed(SEED)

class AutoEncoder(nn.Module):
    '''
        A basic autoencoder
        input -> hidden -> bottleneck -> hidden -> output
    '''
    def __init__(self, input_size, h1_size, h2_size, bottleneck_size):
        super(AutoEncoder, self).__init__()
        self.h1e = nn.Linear(input_size, h1_size)
        self.h2e = nn.Linear(h1_size, h2_size)
        self.enc = nn.Linear(h2_size, bottleneck_size)
        self.h2d = nn.Linear(bottleneck_size, h2_size)
        self.h1d = nn.Linear(h2_size, h1_size)
        self.out = nn.Linear(h1_size, input_size)


    def encode(self, original):
        '''
            Encode x to bottleneck
        '''
        encoding = original.view(-1, 1, 28*28)
        encoding = F.elu(self.h1e(encoding))
        encoding = F.elu(self.h2e(encoding))
        encoding = self.enc(encoding)
        return encoding

    def decode(self, encoding):
        '''
            Decode x back into the original data
        '''
        reconstruction = F.elu(self.h2d(encoding))
        reconstruction = F.elu(self.h1d(reconstruction))
        reconstruction = torch.sigmoid(self.out(reconstruction))
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

def train_epoch(model, optimizer, data_loader, epoch, log_interval=1):
    model.train()
    dataset_size = len(data_loader.dataset)
    epoch_loss = 0.0
    num_items = 0
    batch_idx = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        epoch_loss += loss.item()
        num_items += len(data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            percent = (batch_idx+1) / len(data_loader) * 100
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"Epoch {epoch}: {num_items}/{dataset_size} ({percent:.2f}%) -- Loss={avg_loss}", end='\r')

    percent = (batch_idx+1) / len(data_loader)*100
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch}: {num_items}/{dataset_size} ({percent:.2f}%) -- Loss={avg_loss}")

def test_epoch(model, data_loader, epoch, log_interval=1):
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        epoch_loss = 0.0
        num_items = 0
        batch_idx = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            output = model(data)
            loss = F.mse_loss(output, data)
            epoch_loss += loss.item()
            num_items += len(data)

            if batch_idx % log_interval == 0:
                percent_done = (batch_idx+1)/len(data_loader)*100
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Verification: {num_items}/{dataset_size} ({percent_done:.2f}%) -- Loss={avg_loss}", end='\r')

        percent_done = (batch_idx+1) / len(data_loader)*100
        avg_loss = epoch_loss / len(data_loader)
        print(f"Verification: {num_items}/{dataset_size} ({percent_done:.2f}%) -- Loss={avg_loss}")

def train_model(model, optimizer, n_epochs, train_loader, test_loader, path='./models', name='model'):
    for epoch in range(n_epochs):
        train_epoch(model, optimizer, train_loader, epoch)
        test_epoch(model, test_loader, epoch)
        model.save(f'{path}/{name}_{epoch}.state')
    model.save(f'{name}')

def main():
    parser = argparse.ArgumentParser(description="Configure Training of Auto Encoder")
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs for Training')
    parser.add_argument('-bs', '--batchsize', type=int, help='Training Batch Size')
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
    autoencoder = AutoEncoder(28*28, 512, 256, 128)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    train_model(autoencoder, optimizer, num_epochs, train_loader, test_loader, path=path, name=output)
    if args.clean:
        from shutil import rmtree
        rmtree(path)

if __name__ == '__main__':
    main()
