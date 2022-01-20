from collections import OrderedDict

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import scipy.linalg as slin
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim, use_mask=False, f_scale=0.1, dag_seed= []):
        super().__init__()
        self.x_dim = x_dim

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim))

        if use_mask:
            if len(dag_seed) > 0:
                M_init = torch.rand(x_dim, x_dim) * 0.0
                M_init[torch.eye(x_dim, dtype=bool)] = 0
                M_init = torch.rand(x_dim, x_dim) * 0.0
                for pair in dag_seed:
                    M_init[pair[0], pair[1]] = 1

                self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False)
                print("Initialised adjacency matrix as parsed:\n", self.M)
            else:
                M_init = torch.rand(x_dim, x_dim) * 0.2
                M_init[torch.eye(x_dim, dtype=bool)] = 0
                self.M = torch.nn.parameter.Parameter(M_init)
        else:
            self.M = torch.ones(x_dim, x_dim)

        self.fc_i = nn.ModuleList( [nn.Linear(x_dim + 1, h_dim) for i in range(self.x_dim)])
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for i in range(self.x_dim)])

        for layer in self.shared.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:, i] = 1e-16

        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def sequential(self, x, z, gen_order= None, biased_edges={}):
        out = x.clone().detach()
        if gen_order is None:
            gen_order = list(range(self.x_dim))

        for i in gen_order:
            x_masked = out.clone() * self.M[:, i]
            x_masked[:, i] = 0.0
            if i in biased_edges:
                for j in biased_edges[i]:
                    x_j = x_masked[:, j].detach().numpy()
                    np.random.shuffle(x_j)
                    x_masked[:, j] = torch.from_numpy(x_j)
            out_i = nn.ReLU(inplace=True)(self.fc_i[i](torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1)))
            out[:, i] = self.fc_f[i](self.shared(out_i)).squeeze()
        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat):
        return self.model(x_hat)


class CausalGAN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dag_seed: list = [],
        h_dim: int = 200,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        lambda_gp: float = 10,
        lambda_privacy: float = 1,
        d_updates: int = 5,
        eps: float = 1e-8,
        alpha: float = 1,
        rho: float = 1,
        weight_decay: float = 1e-2,
        grad_dag_loss: bool = False,
        l1_g: float = 0,
        l1_W: float = 1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iterations_d = 0
        self.iterations_g = 0

        self.x_dim = input_dim
        self.z_dim = self.x_dim

        # networks
        self.generator = Generator(z_dim=self.z_dim, x_dim=self.x_dim, h_dim=h_dim, use_mask=use_mask, dag_seed=dag_seed)
        self.discriminator = Discriminator(x_dim=self.x_dim, h_dim=h_dim)

        self.dag_seed = dag_seed

    def forward(self, x, z):
        return self.generator(x, z)

    def sample_z(self, n):
        return torch.rand(n, self.z_dim) * 2 - 1

    def gen_synthetic(self, x, gen_order= None, biased_edges={}):
        return self.generator.sequential( x, self.sample_z(x.shape[0]).type_as(x), gen_order=gen_order, biased_edges=biased_edges)   

    def get_gen_order(self):
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0
        G = nx.from_numpy_matrix(dense_dag, create_using=nx.DiGraph)
        gen_order = list(nx.algorithms.dag.topological_sort(G))
        return gen_order

    def get_dag(self) -> np.ndarray:
        return np.round(self.get_W().detach().numpy(), 3)

    def get_W(self):
        if self.hparams.use_mask:
            return self.generator.M
        else:
            W_0 = []
            for i in range(self.x_dim):
                weights = self.generator.fc_i[i].weight[
                    :, :-1
                ]  # don't take the noise variable's weights
                W_0.append(
                    torch.sqrt(
                        torch.sum((weights) ** 2, axis=0, keepdim=True)
                        + self.hparams.eps
                    )
                )
            return torch.cat(W_0, axis=0).T

    def training_step( self, batch, batch_idx, optimizer_idx):
        # sample noise
        z = self.sample_z(batch.shape[0])
        z = z.type_as(batch)

        generated_batch = self.generator.sequential(batch, z, self.get_gen_order())  
        # train generator
        if optimizer_idx == 0:
            self.iterations_d += 1
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_loss = torch.mean(self.discriminator(batch))
            fake_loss = torch.mean(self.discriminator(generated_batch.detach()))

            # discriminator loss
            d_loss = fake_loss - real_loss

            tqdm_dict = {"d_loss": d_loss.detach()}
            output = OrderedDict( {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        elif optimizer_idx == 1:
            # sanity check: keep track of G updates
            self.iterations_g += 1

            # adversarial loss (negative D fake loss)
            g_loss = -torch.mean( self.discriminator(generated_batch))  

            tqdm_dict = {"g_loss": g_loss.detach()}

            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        else:
            raise ValueError("should not get here")

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

        return ({"optimizer": opt_d, "frequency": self.hparams.d_updates}, {"optimizer": opt_g, "frequency": 1})
