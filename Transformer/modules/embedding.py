import torch
import torch.nn as nn
import torch.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        # Initialize position encoding matrix (shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of shape [max_len, 1] with position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term (shape: [d_model//2]) for the sin and cos functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin/cos to even/odd indices in the position encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, not a parameter (no gradients needed)
        self.register_buffer("pe", pe)

    def forward(self, x):
        '''
        x: [bs, seq_len]
        '''
        B, seq_len = x.size()

        return self.pe[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, p_dropout):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_emb = PositionalEncoding(max_len=max_len, d_model=d_model)
        self.dropout = nn.Dropout(p=p_dropout)
        self.d_model = d_model
    
    def forward(self, x):
        '''
        x: [bs, seq_len]
        '''
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)

        scale = math.sqrt(self.d_model)

        out = self.dropout(token_emb * scale + pos_emb)

        return out