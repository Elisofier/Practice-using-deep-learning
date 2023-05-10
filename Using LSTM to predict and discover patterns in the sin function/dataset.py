import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    def __init__(self, data_file, seq_len):
        self.data = torch.tensor(load_data(data_file), dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(float(line))
        return data
