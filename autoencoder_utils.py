'''
    Functions used by multiple autoencoder files
'''
import time
import torch
import torch.nn
from torch.nn import functional as F

def _train_epoch(model, optimizer, data_loader, epoch, log_interval=1):
    model.train()
    dataset_size = len(data_loader.dataset)
    epoch_loss = 0.0
    num_items = 0
    batch_idx = 0

    start = time.time()
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
            print(f"Epoch {epoch}: {num_items}/{dataset_size} ({percent:.2f}%) -- Loss={avg_loss} | {time.time() - start:.2f}s", end='\r')

    percent = (batch_idx+1) / len(data_loader)*100
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch}: {num_items}/{dataset_size} ({percent:.2f}%) -- Loss={avg_loss} | {time.time() - start:.2f}s")

def _test_epoch(model, data_loader, epoch, log_interval=1):
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
                print(f"[{epoch}]\tVerification: {num_items}/{dataset_size} ({percent_done:.2f}%) -- Loss={avg_loss}", end='\r')

        percent_done = (batch_idx+1) / len(data_loader)*100
        avg_loss = epoch_loss / len(data_loader)
        print(f"[{epoch}]\tVerification: {num_items}/{dataset_size} ({percent_done:.2f}%) -- Loss={avg_loss}")

def train_model(model, optimizer, n_epochs, train_loader, test_loader, verify=False, trainingpath='./models', name='model'):
    start = time.time()
    for epoch in range(n_epochs):
        _train_epoch(model, optimizer, train_loader, epoch)
        if verify:
            _test_epoch(model, test_loader, epoch)
        model.save(f'{trainingpath}/{name}_{epoch}.state')
    model.save(f'./models/{name}')
    elapsed_time = time.time() - start

    # Some really nice formatting
    # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training Complete: {hours}:{minutes}:{seconds}")
