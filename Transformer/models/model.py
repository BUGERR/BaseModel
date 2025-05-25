import torch.nn as nn
import torch
import math
from .layer import EncoderLayer, DecoderLayer

def make_pad_mask(seq, pad_idx):
    # [batch_size, 1, 1, src_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(seq.device)

def make_sub_mask(tgt):
    batch_size, seq_len = tgt.size()
    # [seq_len, seq_len]
    mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    return mask

def make_tgt_mask(tgt, pad_idx):
    batch_size, tgt_len = tgt.shape
    # [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = make_pad_mask(tgt, pad_idx)
    # [tgt_len, tgt_len]
    tgt_sub_mask = make_sub_mask(tgt)
    # [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    return tgt_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Initialize position encoding matrix (shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of shape [max_len, 1] with position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )

        # Compute the div_term (shape: [d_model//2]) for the sin and cos functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin/cos to even/odd indices in the position encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, not a parameter (no gradients needed)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # [seq_len, d_model]
        return self.pe[:seq_len, :]
    

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, src_pad_idx, src_vocab_size, d_model, max_len,
            n_layers, n_head, d_hidden, p_drop=0.1):

        super().__init__()

        self.src_word_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=p_drop)
        self.d_model = d_model
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_hidden, n_head, p_drop)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        src_emb = self.src_word_emb(src) * math.sqrt(self.d_model)
        pos_enc = self.pos_enc(src)
        x = self.dropout(src_emb + pos_enc)
        x = self.layer_norm(x)

        for layer in self.layer_stack:
            x = layer(x, src_mask)
        return x
    
class Decoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, tgt_pad_idx, tgt_vocab_size, d_model, max_len,
            n_layers, n_head, d_hidden, p_drop=0.1):

        super().__init__()

        self.tgt_word_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=p_drop)
        self.d_model = d_model
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_hidden, n_head, p_drop)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, src, tgt_mask, src_mask):
        tgt_emb = self.tgt_word_emb(tgt) * math.sqrt(self.d_model)
        pos_enc = self.pos_enc(tgt)
        x = self.dropout(tgt_emb + pos_enc)
        x = self.layer_norm(x)

        for layer in self.layer_stack:
            x = layer(x, src, tgt_mask, src_mask)
        return x
    

class Transformer(nn.Module):

    def __init__(
        self,
        pad_idx,
        vocab_size,
        d_model,
        n_head,
        max_len,
        d_hidden,
        n_layer,
        p_dropout,
    ):
        """
        Constructor for the Transformer model.

        :param src_pad_idx: Padding index for the source sequences.
        :param tgt_pad_idx: Padding index for the target sequences.
        :param tgt_sos_idx: Start-of-sequence index for the target sequences.
        :param enc_voc_size: Vocabulary size of the encoder.
        :param dec_voc_size: Vocabulary size of the decoder.
        :param d_model: Dimensionality of the model.
        :param n_head: Number of attention heads.
        :param max_len: Maximum sequence length.
        :param d_hidden: Dimensionality of the feed-forward network.
        :param n_layers: Number of layers in the encoder and decoder.
        :param p_drop: Dropout probability.
        """

        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_pad_idx=pad_idx,
            src_vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            n_layers=n_layer,
            n_head=n_head,
            d_hidden=d_hidden,
            p_drop=p_dropout,
        )

        self.decoder = Decoder(
            tgt_pad_idx=pad_idx,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            n_layers=n_layer,
            n_head=n_head,
            d_hidden=d_hidden,
            p_drop=p_dropout
        )

        self.projection = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()

        # Share the weight between two word embeddings & last dense layer
        self.projection.weight = self.decoder.tgt_word_emb.weight

        self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), self.make_src_mask(src))

    def make_src_mask(self, src):
        return make_pad_mask(src, self.pad_idx)

    def make_tgt_mask(self, tgt):
        return make_tgt_mask(tgt, self.pad_idx)

    def encode(self, src):
        return self.encoder(src, self.make_src_mask(src))

    def decode(self, tgt, memory, memory_mask):
        return self.projection(
            self.decoder(
                tgt, memory, self.make_tgt_mask(tgt), memory_mask
            )
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)