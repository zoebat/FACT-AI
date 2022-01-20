import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        data = np.array(data, dtype="float32")
        self.x = torch.from_numpy(data)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.n_samples


class DataModule(pl.LightningDataModule):
    def __init__(self, data, data_dir="", batch_size=64, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = Dataset(data)
        self.dims = self.dataset.x.shape[1:]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
