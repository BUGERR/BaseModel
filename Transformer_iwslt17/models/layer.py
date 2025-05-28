import torch.nn as nn
from .sublayer import MultiHeadAttention, PositionwiseFeedForward
    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_hidden, n_head, p_drop=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_head, d_model, p_drop)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hidden, p_drop)

    def forward(self, enc_input, src_mask=None):
        enc_output = self.enc_self_attn(enc_input, enc_input, enc_input, src_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output
    
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_hidden, n_head, p_drop=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_head, d_model, p_drop)
        self.dec_enc_attn = MultiHeadAttention(n_head, d_model, p_drop)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hidden, p_drop)

    def forward(self, dec_input, enc_output, tgt_mask=None, src_mask=None):
        dec_output = self.dec_self_attn(dec_input, dec_input, dec_input, tgt_mask)
        dec_output = self.dec_enc_attn(dec_input, enc_output, enc_output, src_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output