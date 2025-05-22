import torch
import torch.nn as nn
import torch.functional as F

from .layers import MultiheadAttention, PositionwiseFeedForward, LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, p_dropout):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model=d_model, n_head=n_head, p_dropout=p_dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, p_dropout=p_dropout)
        self.ln1 = LayerNorm(normalized_shape=d_model)
        self.ln2 = LayerNorm(normalized_shape=d_model)
        self.ln3 = LayerNorm(normalized_shape=d_model)

    def forward(self, tgt, enc, tgt_mask, src_mask):

        # decoder self-attn
        residual = tgt
        x = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        
        # add & norm
        x = residual + x
        x = self.ln1(x)

        # decoder q, encoder kv, cross-attn
        residual = x
        x = self.self_attn(x, enc, enc, mask=src_mask)
        
        # add & norm
        x = residual + x
        x = self.ln2(x)

        # MLP
        residual = x
        x = self.ffn(x)
        
        # add & norm
        residual = x
        x = self.ln3(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, p_dropout, n_layer):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_hidden=d_hidden,
                    p_dropout=p_dropout
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(tgt, enc_src, tgt_mask, src_mask)

        return x