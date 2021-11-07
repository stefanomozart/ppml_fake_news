import torch
from torch import nn
import torch.nn.functional as func

class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        dropout = .25
        channels1 = 2
        channels2 = 4
        channels3 = 6
        kernels1 = [4, 8, 16]
        kernels2 = [2, 3, 4]
        kernels3 = [3, 4, 5]
        classes = 2

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[0]),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[0]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[0]),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[1]),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[1]),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[2]),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[2]),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Sequential(
            nn.Linear(204, 32),
            nn.Linear(32, classes),
        )

    def forward(self, batch):
        x = torch.reshape(batch, (-1, 1, 768))
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.view(len(batch), -1)
        x = self.dropout(x)
        logits = self.lin(x)
        return logits
