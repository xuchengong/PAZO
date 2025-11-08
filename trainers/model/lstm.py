import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, dropout=0.2):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.LSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=False, batch_first=True, num_layers=2)
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        """
        input : [bs, vocab_size]
        output: [bs, 2]
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.LSTM(x)
        x = self.dropout(x)
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        out = self.fc(x)  
        return out
    