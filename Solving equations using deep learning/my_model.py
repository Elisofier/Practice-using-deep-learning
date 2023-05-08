import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # 将c转换为PyTorch张量
        x = torch.tensor(x)
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out
