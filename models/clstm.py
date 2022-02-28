import torch
from torch import nn
import torch.nn.functional as func

# Souza et al - Convoluutional Long-Short Memory Neural Network for fake news detection
class CLSTM(nn.Module):
    def __init__(self, config=None):
        super(CLSTM, self).__init__()

        # Hyper-parameters
        self.config = config if config is not None else {
            'num_classes': 2, # binary classification: fake vs real
            'convolution_channels': [4, 16, 32],
            'convolution_kernels': [4, 6, 8],
            'embeddings_size': 768, # default BERT embeddings size
            'hidden_size': 128,
            'hidden_layers': 3,
            'linear_size': 256,
            'dropout': 0.42, 
        }

        # Fist step: convolution
        self.convolution = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.config['convolution_channels'][0], 
                kernel_size=self.config['convolution_kernels'][0]
            ),
            nn.ReLU(),
            nn.MaxPool1d(self.config['convolution_kernels'][0]),
            nn.Conv1d(
                in_channels=self.config['convolution_channels'][0],
                out_channels=self.config['convolution_channels'][1],
                kernel_size=self.config['convolution_kernels'][1]
            ),
            nn.ReLU(),
            nn.MaxPool1d(self.config['convolution_kernels'][1]),
            nn.Conv1d(
                in_channels=self.config['convolution_channels'][1],
                out_channels=self.config['convolution_channels'][2],
                kernel_size=self.config['convolution_kernels'][2]
            ),
            nn.ReLU(),
            nn.MaxPool1d(self.config['convolution_kernels'][2])
        )

        # Second step: Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size = self.config['embeddings_size'],
            hidden_size = self.config['hidden_size'],
            num_layers = self.config['hidden_layers'],
            dropout = self.config['dropout'],
            bidirectional = True,
            batch_first=True
        )

        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.linear1 = nn.Linear(
            self.config['embeddings_size'] + 2*self.config['hidden_size'],
            self.config['linear_size']
        )

        # Tanh non-linearity
        self.tanh = nn.Tanh()   

        # Dropout
        self.dropout = nn.Dropout(self.config['dropout'])

        # Fully-Connected Layer
        self.linear2 = nn.Sequential(
            nn.Linear(self.config['linear_size'], 32),
            nn.Linear(32, self.config['num_classes']),
        )

    def forward(self, batch):
        # Reshap batch
        X = torch.reshape(batch, (-1, 1, self.config['embeddings_size']))
        #print('X = torch.reshape(batch, (-1, 1, 768)) ', X.shape)

        X1 = self.convolution(X)
        #print('X1 = convolution(X) ', X1.shape)

        # LSTM
        X1, (hidden, cell) = self.lstm(X)

        # Convolution
        X2 = torch.cat([X1, X], 2).permute(1, 0, 2)

        # First linear layer
        X4 = self.tanh(self.linear1(X2)).permute(1, 2, 0)
        X5 = func.max_pool1d(X4, X4.shape[2]).squeeze(2)

        # Last linear layer
        return self.linear2(self.dropout(X5))
