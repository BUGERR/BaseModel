import torch
from torch import nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # Learnable parameters
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # x: [batch_size, ..., normalized_shape]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, q, k, v, mask=None):
        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, mask)
        else:
            # calculate attention manually
            batch_size, num_attention_heads, seq_len, d_key = k.size()

            # 1. Compute the dot product between query and key^T
            k_t = k.transpose(-2, -1)
            scores = q @ k_t / math.sqrt(d_key)

            # 2. Apply mask (optional)
            if mask is not None:
                scores = scores.masked_fill(mask.logical_not(), float("-inf"))

            # 3. Apply softmax to get attention weights
            attn = F.softmax(scores, dim=-1)

            # 4. Compute the weighted sum of values
            out = attn @ v

        return out


class MultiheadAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_concat = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # 1. Linear projections, [batch_size, length, hidden_size]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Split tensor by number of heads, [batch_size, length, num_attention_heads, d_key]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Apply attention
        out = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer, [batch_size, length, hidden_size]
        out = self.concat(out)
        out = self.w_concat(out)

        return self.dropout(out)

    def split(self, tensor):
        batch_size, length, hidden_size = tensor.size()

        d_key = hidden_size // self.num_attention_heads
        return tensor.view(
            batch_size, length, self.num_attention_heads, d_key
        ).transpose(1, 2)

    def concat(self, tensor):
        batch_size, head, length, d_key = tensor.size()
        hidden_size = head * d_key

        tensor = (
            tensor.transpose(1, 2).contiguous().view(batch_size, length, hidden_size)
        )
        return tensor


# class MultiheadAttention(nn.Module):
#     ''' Multi-Head Attention module '''
#
#     def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
#         super().__init__()
#
#         self.n_head = num_attention_heads
#         self.p_drop = dropout
#
#         self.w_q = nn.Linear(hidden_size, hidden_size)
#         self.w_k = nn.Linear(hidden_size, hidden_size)
#         self.w_v = nn.Linear(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, hidden_size)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.attention = ScaledDotProductAttention()
#
#     def forward(self, q, k, v, mask=None):
#         '''
#         q: [batch_size, src_len, d_model]
#         k: [batch_size, src_len or tgt_len, d_model]
#         v: [batch_size, src_len or tgt_len, d_model]
#
#         mask: [src_len, src_len or tgt_len]
#         '''
#         batch_size, len_q, d_model = q.size()
#         batch_size, len_k, d_model = k.size()
#         d_k = d_v = d_model // self.n_head
#
#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_q(q).view(batch_size, len_q, self.n_head, d_k)
#         k = self.w_k(k).view(batch_size, len_k, self.n_head, d_k)
#         v = self.w_v(v).view(batch_size, len_k, self.n_head, d_v)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         q = self.attention(q, k, v, mask=mask)
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
#         q = self.dropout(self.fc(q))
#
#         return q


class PositionwiseFeedForward(nn.Module):

    def __init__(self, hidden_size, hidden, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden)
        self.linear2 = nn.Linear(hidden, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
