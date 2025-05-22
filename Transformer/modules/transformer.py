import torch.nn as nn
import torch.functional as F
import torch

from .embedding import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder

def make_pad_mask(seq, pad_idx):
    '''
    seq: [bs, seq_len]

    mask: [bs, seq_len] --> [bs, 1, 1, seq_len]

    attn_srcore: [bs, n_head, seq_len, seq_len]
    '''
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def make_causal_mask(seq):
    '''
    seq: [bs, seq_len]

    mask: [seq_len, seq_len] --> [1, 1, seq_len, seq_len]

    attn_score: [bs, n_head, seq_len, seq_len]
    '''
    batch_size, seq_len = seq.size()
    mask = torch.tril(torch.ones(seq_len, seq_len, device = seq.device)).bool().unsqueeze(0).unsqueeze(0)
    return mask

def make_tgt_mask(tgt_seq, pad_idx):
    pad_mask = make_pad_mask(tgt_seq, pad_idx)
    sub_mask = make_causal_mask(tgt_seq)
    tgt_mask = pad_mask & sub_mask
    return tgt_mask

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, vocab_size, max_len, d_model, d_hidden, n_head, n_layer, p_dropout):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            p_dropout=p_dropout
        )

        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            d_hidden=d_hidden,
            p_dropout=p_dropout,
            n_layer=n_layer
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            d_hidden=d_hidden,
            p_dropout=p_dropout,
            n_layer=n_layer
        )

        self.logits_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_parameters()

        # weight tying
        self.logits_head.weight = self.emb.token_emb.weight


    def forward(self, src, tgt):
        return self.decode(
            tgt, self.encode(src), self.make_src_mask(src)
        )
    
    def make_src_mask(self, src):
        return make_pad_mask(src, self.src_pad_idx)
    
    def make_tgt_mask(self, tgt):
        return make_tgt_mask(tgt, self.tgt_pad_idx)
    
    def encode(self, src):
        return self.encoder(self.emb(src), self.make_src_mask(src))
    
    def decode(self, tgt, enc_src, src_mask):
        return self.logits_head(
            self.decoder(tgt, enc_src, self.make_tgt_mask(tgt), src_mask)
        )
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    