import torch
from basic_autoencoder import AutoEncoder
from data_managment import get_dataloaders
from matplotlib import pyplot as plt

def main():
    ''' main '''
    batch_size = 2
    _, test_loader = get_dataloaders(batch_size, batch_size)

    autoencoder = AutoEncoder(28*28, 512, 256, 128)
    autoencoder.load_state_dict(torch.load('./autoencoder'))

    with torch.no_grad():
        batch, _ = next(iter(test_loader))
        x = batch
        y = autoencoder(x)

        figure, ax = plt.subplots(batch_size, 2)
        for i, (orig, recon) in enumerate(zip(x,y)):
            ocm = ax[i, 0].imshow(orig.squeeze(), cmap='gray')
            ax[i, 0].set_title("Original")

            rcm = ax[i, 1].imshow(recon.squeeze(), cmap='gray')
            ax[i, 1].set_title("Reconstruction")

            figure.colorbar(ocm, ax=ax[i, 0], shrink=0.5)
            figure.colorbar(rcm, ax=ax[i, 1], shrink=0.5)

        plt.tight_layout()
        plt.savefig(f'output/out.png')
        plt.show()

if __name__ == '__main__':
    main()