'''
    Functions used by multiple autoencoder files
'''
import time
import torch
import torch.nn
from torch.nn import functional as F

def _print_epoch(epoch, loss, num_items, dataset_size, elapsed_time, epoch_type="Training", end='\r'):
    percent = num_items / dataset_size * 100
    avg_loss = loss / num_items
    print(f'[{epoch:02}] {epoch_type:12} : {num_items:05}/{dataset_size} ({percent:6.2f}%) -- Loss={avg_loss:.2E} | {elapsed_time:.2f}s', end=end)
    

def _train_epoch(model, optimizer, loss_fn, data_loader, epoch, log_interval=1):
    model.train()
    dataset_size = len(data_loader.dataset)
    epoch_loss = 0.0
    num_items = 0
    batch_idx = 0
    start = time.time()

    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        epoch_loss += loss.item()
        num_items += len(data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            _print_epoch(epoch, epoch_loss, num_items, dataset_size, time.time()-start)

    _print_epoch(epoch, epoch_loss, num_items, dataset_size, time.time()-start, end='\n')

def _test_epoch(model, data_loader, epoch, log_interval=1):
    epoch_type = "Verification"
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        epoch_loss = 0.0
        num_items = 0
        batch_idx = 0

        start = time.time()
        for batch_idx, (data, _) in enumerate(data_loader):
            output = model(data)
            loss = F.mse_loss(output, data)
            epoch_loss += loss.item()
            num_items += len(data)

            if batch_idx % log_interval == 0:
                elapsed_time = time.time() - start
                _print_epoch(epoch, epoch_loss, num_items, dataset_size, elapsed_time, epoch_type="Verification")

        elapsed_time = time.time() - start
        _print_epoch(epoch, epoch_loss, num_items, dataset_size, elapsed_time, epoch_type="Verification", end='\n')

def train_model(model, optimizer, loss_fn, n_epochs, train_loader, test_loader, verify=False, trainingpath='./models', name='model'):
    start = time.time()
    for epoch in range(n_epochs):
        _train_epoch(model, optimizer, loss_fn, train_loader, epoch)
        if verify:
            _test_epoch(model, test_loader, epoch)
        model.save(f'{trainingpath}/{name}_{epoch}.state')
    model.save(f'./models/{name}')
    elapsed_time = time.time() - start

    # Some really nice formatting
    # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training Complete: {hours:.0f}:{minutes:.0f}:{seconds:2.0f}")
