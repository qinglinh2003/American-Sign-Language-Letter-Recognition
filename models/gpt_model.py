import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, num_layers=8, dropout=0.1, seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, dropout=dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.token_embedding(x)              # (batch, seq_len, embed_dim)
        x = self.pos_embedding(x)                # (batch, seq_len, embed_dim)
        x = x.transpose(0, 1)                    # (seq_len, batch, embed_dim)
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)       # (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)                    # (batch, seq_len, embed_dim)
        logits = self.fc_out(x)                  # (batch, seq_len, vocab_size)
        return logits