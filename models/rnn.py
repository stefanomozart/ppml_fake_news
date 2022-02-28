import torch
from torch import nn
import torch.nn.functional as func

# Souza et al - Recurrent Convoluutional Neural Network for fake news detection
class RNN(nn.Module):
    def __init__(self, config=None):
        super(RNN, self).__init__()
        
        # Hyper-parameters
        self.config = config if config is not None else {
            'num_classes': 2, # binary classification: fake vs real
            'embeddings_size': 768, # default BERT embeddings size
            'hidden_size': 128,
            'hidden_layers': 3,
            'linear_size': 256,
            'dropout': 0.42, 
        }
        
        # Bi-directional LSTM
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
        
        # LSTM
        X1, (hidden, cell) = self.lstm(X)
        
        # Convolution
        X2 = torch.cat([X1, X], 2).permute(1, 0, 2)
        
        # First linear layer
        X4 = self.tanh(self.linear1(X2)).permute(1, 2, 0)
        X5 = func.max_pool1d(X4, X4.shape[2]).squeeze(2)

        # Last linear layer
        return self.linear2(self.dropout(X5))
