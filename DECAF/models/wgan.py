import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from collections import OrderedDict



class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, data_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        dim = latent_dim
        seq = []
        for item in list(hidden_dims):
            seq+= block(dim, item)
            dim = item
        self.net = nn.Sequential(*seq, nn.Linear(dim, data_dim), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, input):
        return self.net(input)

class WGAN(LightningModule):
    def __init__(
        self,
        data_dim,
        latent_dim: int = 128,
        generator_dims=(256,256),
        discriminator_dims=(256,256),
        generator_lr: float = 2E-4,
        discriminator_lr: float = 2E-4,
        generator_decay: float = 2E-6,
        discriminator_decay: float = 2E-6,
        lambda_gp: float = 5,
        batch_size: int = 256,
        b1: float = 0.5,
        b2: float = 0.999,
        d_updates: int = 5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        #Initialize Generator and Discriminator networks
        self._generator = Generator(latent_dim=self.hparams.latent_dim, hidden_dims=self.hparams.generator_dims, data_dim=self.hparams.data_dim)
        self._discriminator = Discriminator(input_dim=self.hparams.data_dim, hidden_dims=self.hparams.discriminator_dims)

    def forward(self, z):
        return self._generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.type_as(real_samples)  
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self._discriminator(interpolates)
        fake = torch.ones(real_samples.size(0), 1)
        fake = fake.type_as(real_samples)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        batch_size = batch.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(batch)
    
        if optimizer_idx == 0:
            real_loss = torch.mean(self._discriminator(batch))
            generated_batch = self(z)
            fake_loss = torch.mean(self._discriminator(generated_batch.detach()))
            d_loss = fake_loss - real_loss

            d_loss += self.hparams.lambda_gp * self.compute_gradient_penalty(batch, generated_batch)

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        if optimizer_idx == 1:
            g_loss = -torch.mean(self._discriminator(self(z)))
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self._generator.parameters(),
            lr=self.hparams.generator_lr,
            betas=(b1, b2),
            weight_decay=self.hparams.generator_decay)

        opt_d = torch.optim.Adam(self._discriminator.parameters(),
        lr=self.hparams.discriminator_lr,
        betas=(b1, b2),
        weight_decay=self.hparams.discriminator_decay)

        return (
            {"optimizer": opt_d, "frequency": self.hparams.d_updates},
            {"optimizer": opt_g, "frequency": 1}
        )

    def sample(self, n):
        z = torch.randn(n, self.hparams.latent_dim)
        return self._generator(z).detach().numpy()