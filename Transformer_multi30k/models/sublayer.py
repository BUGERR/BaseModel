import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.1) -> torch.Tensor:
    '''
    q: [batch_size, n_head, src_len, d_k]
    k: [batch_size, n_head, src_len or tgt_len, d_k]
    v: [batch_size, n_head, src_len or tgt_len, d_v]
    
    mask: [src_len, src_len or tgt_len]
    '''
    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight_drop = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight_drop @ value, attn_weight

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, p_drop=0.1):
        super().__init__()

        self.n_head = n_head
        self.p_drop = p_drop

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=p_drop)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = None


    def forward(self, q, k, v, mask=None):
        '''
        q: [batch_size, src_len, d_model]
        k: [batch_size, src_len or tgt_len, d_model]
        v: [batch_size, src_len or tgt_len, d_model]
        
        mask: [src_len, src_len or tgt_len]
        '''
        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        d_k = d_v = d_model // self.n_head

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_q(q).view(batch_size, len_q, self.n_head, d_k)
        k = self.w_k(k).view(batch_size, len_k, self.n_head, d_k)
        v = self.w_v(v).view(batch_size, len_k, self.n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


        q, attn = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_drop)
        self.attention = attn
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_model, d_hidden, p_drop=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden) # position-wise
        self.w_2 = nn.Linear(d_hidden, d_model) # position-wise
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
    

# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''

#     def __init__(self, p_drop=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=p_drop)

#     def forward(self, q, k, v, mask=None):
#         '''
#         q: [batch_size, n_head, len_q, d_k]
#         k: [batch_size, n_head, len_k, d_k]
#         v: [batch_size, n_head, len_k, d_v]
        
#         mask: [1, len_q, len_k]
#         '''
#         batch_size, n_head, len_q, d_k = q.size()

#         k_t = k.transpose(-2, -1)

#         attn = torch.matmul(q, k_t) / math.sqrt(d_k)

#         if mask is not None:
#             attn = attn.masked_fill(mask.logical_not(), float("-inf"))

#         attn = self.dropout(F.softmax(attn, dim=-1))

#         output = torch.matmul(attn, v)

#         return output, attn