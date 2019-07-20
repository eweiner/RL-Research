from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.Linear(self.latent_dim, 5 * out_dim),
            nn.ReLU(),
            nn.Linear(out_dim * 5, out_dim * 5),
            nn.ReLU(),
            nn.Linear(out_dim * 5, out_dim),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, out_dim):
        super(Discriminator, self).__init__()
        self.out_dim = out_dim
        self.main = nn.Sequential(
            nn.Linear(out_dim, 10 * out_dim),
            nn.ReLU(),
            nn.Linear(out_dim * 10, out_dim * 10),
            nn.ReLU(),
            nn.Linear(out_dim * 10, out_dim * 10),
            nn.ReLU(),
            nn.Linear(10 * out_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        return self.main(input)


class GAN1d(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(GAN1d, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.generator = Generator(latent_dim, out_dim)
        self.discriminator = Discriminator(out_dim)

    def _create_loss_and_optimizer(self, net, learning_rate=0.001):

        # Loss function
        loss = torch.nn.BCELoss()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        return (loss, optimizer)

    def generate_noisy_labels(self, real: bool, b_size: int, noise: float = 0.0):
        """
        In order for GANs to train better, it can help to add noise to the true labels
        so the discriminator gives a path to improvement
        """
        noise_vec = torch.distributions.Uniform(
            torch.tensor([0.0]), torch.tensor([noise])
        ).sample()
        if real:
            return torch.cat(
                (
                    torch.ones((b_size, 1)) - noise_vec,
                    torch.zeros((b_size, 1)) + noise_vec,
                ),
                dim=1,
            )
        return torch.cat(
            (torch.zeros((b_size, 1)) + noise_vec, torch.ones((b_size, 1)) - noise_vec),
            dim=1,
        )


def train_gan(
    gan,
    x_train: list,
    epochs: int,
    train_noise: float = 0.1,
    freeze_generator: bool = False,
) -> tuple:
    criterion = torch.nn.BCELoss()
    optimizerD = optim.Adam(gan.discriminator.parameters(), lr=0.001)
    optimizerG = optim.Adam(gan.generator.parameters(), lr=0.0001)
    g_losses = []
    d_losses = []
    for e in range(epochs):
        for batch in x_train:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            b_size = batch.shape[0]
            ## Train with all-real batch
            gan.discriminator.zero_grad()
            label = gan.generate_noisy_labels(True, b_size, noise=train_noise)
            output = gan.discriminator(batch)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with all-fake batch
            latent_noise = torch.randn(b_size, gan.latent_dim)
            fake = gan.generator(latent_noise)
            label = gan.generate_noisy_labels(False, b_size, noise=train_noise)
            output = gan.discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gan.generator.zero_grad()
            label = gan.generate_noisy_labels(True, b_size, noise=train_noise)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = gan.discriminator(fake)
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            if not freeze_generator:
                optimizerG.step()
            # Save Losses for plotting later
            g_losses.append(errG.item())
            d_losses.append(errD.item())
            # TODO: Is there a way to look at gan to make sure generated observations <-> real states

    return g_losses, d_losses
