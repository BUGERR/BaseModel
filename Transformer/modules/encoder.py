import torch
import torch.nn as nn
import torch.functional as F

from .layers import MultiheadAttention, PositionwiseFeedForward, LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, p_dropout):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model=d_model, n_head=n_head, p_dropout=p_dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, p_dropout=p_dropout)
        self.ln1 = LayerNorm(normalized_shape=d_model)
        self.ln2 = LayerNorm(normalized_shape=d_model)

    def forward(self, x, src_mask):
        '''
        x: [bs, seq_len, d_model]
        '''
        residual = x
        x = self.self_attn(x, x, x, src_mask)
        
        x = residual + x
        x = self.ln1(x)

        residual = x
        x = self.ffn(x)
        
        x = residual + x
        x = self.ln2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, p_dropout, n_layer):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_hidden=d_hidden,
                    p_dropout=p_dropout
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x