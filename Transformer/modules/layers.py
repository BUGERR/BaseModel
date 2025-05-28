import torch
import torch.nn as nn
import math
import torch.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_dropout):
        super().__init__()

        self.flash = hasattr(F, 'scaled_dot_product_attention')

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q: [bs, n_head, seq_len_n, d_k]
        k: [bs, n_head, seq_len_m, d_k]
        v: [bs, n_head, seq_len_m, d_v]
        
        d_k = d_v

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, mask)
        '''
        scale = 1 / math.sqrt(q.size(-1))

        qk = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            qk.masked_fill_(mask.logical_not(), float('-inf'))

        attn_weight = F.softmax(qk, dim=-1)

        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v

        return out, attn_weight
    

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, p_dropout):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(p_dropout)

        self.W_concat = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=p_dropout)

        self.n_head = n_head

        self.attn_score = None
        

    def forward(self, q, k, v, mask=None):
        '''
        q: [bs, seq_len_n, d_model]
        k: [bs, seq_len_m, d_model]
        v: [bs, seq_len_m, d_model]

        split_q: [bs, n_head, seq_len, d_k]
        '''
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        attn_out, attn_score = self.attention(q, k, v, mask)

        self.attn_score = attn_score

        out = self.concat(attn_out)

        out = self.W_concat(out)

        out = self.dropout(out)
        
        return out
    
    def split(self, tensor):
        '''
        [bs, seq_len, d_model] --> [bs, n_head, seq_len, d_k]
        '''
        batch_size, seq_len, d_model = tensor.size()
        d_k = d_model // self.n_head

        return tensor.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        
    def concat(self, tensor):
        '''
        [bs, n_head, seq_len, d_k] --> [bs, seq_len, d_model]
        '''
        batch_size, n_head, seq_len, d_k = tensor.size()
        d_model = n_head * d_k

        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, p_dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.dropout(x)
        return out
    

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # Learnable parameters
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


# if __name__ == '__main__':
#     x = nn.Parameter(torch.ones(3, 3))
#     x_T = x.transpose(-2, -1)
#     print(x, x_T)
