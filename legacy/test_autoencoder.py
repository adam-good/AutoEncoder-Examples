import argparse
import torch
from data_managment import get_dataloaders
from matplotlib import pyplot as plt

from simple_autoencoder import SimpleAutoEncoder
from conv_autoencoder import ConvAutoEncoder
from vae import VAE

def main():
    ''' main '''
    parser = argparse.ArgumentParser("Configure Testing of Auto Encoder")
    parser.add_argument('-T', '--type', type=str, help='Type of autoencoder: simple, conv')
    parser.add_argument('-i', '--input', type=str, help="Name of File Containing the Model State")
    parser.add_argument('-bs', '--batchsize', type=int, default=1, help="Batch Size for Testing")
    parser.add_argument('-o', '--output', type=str, default='out.png', help='Output Image')
    parser.add_argument('-S', '--seed', type=int, )
    args = parser.parse_args()

    if args.seed:
        torch.random.manual_seed(args.seed)

    model_path = args.input
    batch_size = args.batchsize
    _, test_loader = get_dataloaders(batch_size, batch_size)

    # autoencoder = None
    if args.type == 'simple':
        autoencoder = SimpleAutoEncoder()
    elif args.type == 'conv':
        autoencoder = ConvAutoEncoder()
    elif args.type == 'vae':
        autoencoder = VAE()
    else:
        autoencoder = SimpleAutoEncoder()
    autoencoder.load_state_dict(torch.load(model_path))
    
    with torch.no_grad():
        batch, _ = next(iter(test_loader))
        x = batch
        if args.type == 'simple' or args.type == 'vae':
            y = autoencoder(x.view(-1, 1, 28*28), False).reshape(-1, 1, 28, 28)
        elif args.type == 'conv':
            y = autoencoder(x)

        if batch_size > 1:
            figure, ax = plt.subplots(batch_size, 2)
            for i, (orig, recon) in enumerate(zip(x, y)):
                ocm = ax[i, 0].imshow(orig.squeeze(), cmap='gray')
                ax[i, 0].set_title("Original")

                rcm = ax[i, 1].imshow(recon.squeeze(), cmap='gray')
                ax[i, 1].set_title("Reconstruction")

                figure.colorbar(ocm, ax=ax[i, 0], shrink=0.5)
                figure.colorbar(rcm, ax=ax[i, 1], shrink=0.5)
        elif batch_size == 1:
            figure, ax = plt.subplots(batch_size, 2)
            ocm = ax[0].imshow(x.squeeze(), cmap='gray')
            ax[0].set_title("Original")

            rcm = ax[1].imshow(y.squeeze(), cmap='gray')
            ax[1].set_title("Reconstruction")

            figure.colorbar(ocm, ax=ax[0], shrink=0.5)
            figure.colorbar(rcm, ax=ax[1], shrink=0.5)

        plt.tight_layout()
        plt.savefig(f'output/{args.output}')
        plt.show()

if __name__ == '__main__':
    main()