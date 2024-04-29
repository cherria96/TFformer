#%%
import torch
from torch import nn
import math
import torch.nn.functional as F




class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, 
                 positional_encoding_k = None,
                 positional_encoding_v = None,
                 attn_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v

    def forward(self, q, k, v, mask=None, dropout = None, one_direction = None):
        # q, k (32, 1287, 5)
        d_k = k.size(-1) # Assume q, k, v have same last dimension size , 5
        scores = torch.matmul(q, k.transpose(-2, -1)) # (32, 1287, 1287)
        if self.positional_encoding_k is not None:
            R_k = self.positional_encoding_k(k) # (32, 1287, 5)
            scores = scores + torch.einsum('b q d, q k d -> b q k', q, R_k)

        scores = scores / d_k ** 0.5
        
        if one_direction:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        if dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v) # (32, 1287, 5)

        if self.positional_encoding_v is not None:
            R_v = self.positional_encoding_v(q.size(-2), v.size(-2))
            output = output + torch.einsum('b q v, q v d -> b q d', attn, R_v)

        return output, attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, head_size=None, dropout=0.0, positional_encoding_k=None, positional_encoding_v=None,
                 final_layer=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn = None
        self.num_heads = num_heads
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // num_heads
        self.q_layers = []
        self.k_layers = []
        self.v_layers = []
        v_layer = nn.Linear(d_model, self.head_size)
        for _ in range(self.num_heads):
            self.q_layers.append(nn.Linear(d_model, self.head_size))
            self.k_layers.append(nn.Linear(d_model, self.head_size))
            self.v_layers.append(v_layer)
        

        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, self.num_heads * self.head_size) for _ in range(2)])
        self.attention = ScaledDotProductAttention(positional_encoding_k, positional_encoding_v)
        self.dropout = nn.Dropout(p=dropout)
        if final_layer:
            self.final_layer = nn.Linear(self.num_heads * self.head_size, d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, one_direction=True):
        batch_size = query.size(0)
        heads = []
        attns = []
        for i in range(self.num_heads):
            query_ = self.q_layers[i](query).view(batch_size, -1, self.head_size) # (1287,32,5)
            key_ = self.k_layers[i](key).view(batch_size, -1, self.head_size)
            value_ = self.v_layers[i](value).view(batch_size, -1, self.head_size)
            head, attn = self.attention(query_, key_, value_, 
                                        mask = mask, dropout = self.dropout, 
                                        one_direction = one_direction) 
            heads.append(head) 
            attns.append(attn)
        head = torch.stack(heads) if self.num_heads > 1 else heads[0] # (num_heads, batch_size, -1, head_size)
        self.attn = torch.stack(attns) # (num_heads, batch_size, seq_len, seq_len)

        outputs = torch.mean(head, axis = 0) if self.num_heads > 1 else head
        outputs = torch.stack([outputs]*self.num_heads)
        # outputs = nn.Linear(outputs.shape[-1], self.head_size * self.num_heads)(outputs)
        outputs = outputs.transpose(0,1).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        if hasattr(self, 'final_layer'):
            outputs = self.final_layer(outputs)
        return self.layer_norm(outputs + query)



class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, final_layer=True, **kwargs):
        super().__init__()
        # self.layer_norm = LayerNorm(hidden)   - already used in MultiHeadedAttention and PositionwiseFeedForward
        self.self_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                   dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                   positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, causal_mask):
        # self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        x = self.self_attention(x, x, x, causal_mask, True)
        x = self.feed_forward(x)
        return x

if __name__ == "__main__":

    from TFformer import PositionalEncoding
    def _generate_causal_mask(timestep, dim_total):
        causal_mask = torch.tril(torch.ones(timestep, timestep))
        causal_mask = causal_mask.repeat_interleave(dim_total, dim=1)
        _remainder_mask = torch.ones(timestep * (dim_total - 1), timestep * dim_total)
        causal_mask = torch.cat((causal_mask, _remainder_mask))
        return causal_mask

    embedded = torch.load('embedded.pt') # (32, 1287, 10) -> (batch, timestep * 13, seq_hidden)

    seq_hidden_units = 10
    num_heads = 2
    head_size = seq_hidden_units // num_heads
    dropout_rate = 0.2
    num_layer = 2
    causal_mask = _generate_causal_mask(99, 13) # (1287, 1287) -> (timestep * 13, timestep * 13)



    basic_block_cls = TransformerEncoderBlock
    positional_encoding = PositionalEncoding(d_model = head_size)
    transformer_encoder = nn.ModuleList([basic_block_cls(
        hidden = seq_hidden_units, attn_heads = num_heads, head_size = head_size,
        feed_forward_hidden = seq_hidden_units, dropout = dropout_rate,
        self_positional_encoding_k=None, self_positional_encoding_v=None
        ) for _ in range(num_layer)])
    for block in transformer_encoder:
        hidden = block(embedded, causal_mask)

    

# %%
