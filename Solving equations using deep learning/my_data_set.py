import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_file, label_file, train_frac=0.7, val_frac=0.15, test_frac=0.15, batch_size=32):
        self.data = torch.tensor(load_data(data_file), dtype=torch.float32)
        self.labels = torch.tensor(load_data(label_file), dtype=torch.float32)
        self.num_samples = len(self.labels)

        train_size = int(train_frac * self.num_samples)
        val_size = int(val_frac * self.num_samples)
        test_size = self.num_samples - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self, [train_size, val_size, test_size]
        )

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data
        y = self.labels
        return x, y

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_val_dataloader(self):
        return self.val_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_data_size(self):
        return self.data.size()[1], self.labels.size()[1]


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip().split()
            data.append(list(map(float, line)))
        return data
