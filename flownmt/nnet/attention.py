from overrides import overrides
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

from flownmt.nnet.layer_norm import LayerNorm


class GlobalAttention(nn.Module):
    """
    Global Attention between encoder and decoder
    """

    def __init__(self, key_features, query_features, value_features, hidden_features=None, dropout=0.0):
        """

        Args:
            key_features: int
                dimension of keys
            query_features: int
                dimension of queries
            value_features: int
                dimension of values (outputs)
            hidden_features: int
                dimension of hidden states (default value_features)
            dropout: float
                dropout rate
        """
        super(GlobalAttention, self).__init__()
        if hidden_features is None:
            hidden_features = value_features
        self.key_proj = nn.Linear(key_features, 2 * hidden_features, bias=True)
        self.query_proj = nn.Linear(query_features, hidden_features, bias=True)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_features, value_features)
        self.hidden_features = hidden_features
        self.reset_parameters()

    def reset_parameters(self):
        # key proj
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0)
        # query proj
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0)
        # fc
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    @overrides
    def forward(self, query, key, key_mask=None):
        """

        Args:
            query: Tensor
                query tensor [batch, query_length, query_features]
            key: Tensor
                key tensor [batch, key_length, key_features]
            key_mask: ByteTensor or None
                binary ByteTensor [batch, src_len] padding elements are indicated by 1s.

        Returns: Tensor
            value tensor [batch, query_length, value_features]

        """
        bs, timesteps, _ = key.size()
        dim = self.hidden_features
        # [batch, query_length, dim]
        query = self.query_proj(query)

        # [batch, key_length, 2 * dim]
        c = self.key_proj(key)
        # [batch, key_length, 2, dim]
        c = c.view(bs, timesteps, 2, dim)
        # [batch, key_length, dim]
        key = c[:, :, 0]
        value = c[:, :, 1]

        # attention weights [batch, query_length, key_length]
        attn_weights = torch.bmm(query, key.transpose(1, 2))
        if key_mask is not None:
            attn_weights = attn_weights.masked_fill(key_mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_weights.float(), dim=-1,
                                 dtype=torch.float32 if attn_weights.dtype == torch.float16 else attn_weights.dtype)

        # values [batch, query_length, dim]
        out = torch.bmm(attn_weights, value)
        out = F.dropout(self.fc(out), p=self.dropout, training=self.training)
        return out

    def init(self, query, key, key_mask=None, init_scale=1.0):
        with torch.no_grad():
            return self(query, key, key_mask=key_mask)


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    """
    def __init__(self, model_dim, heads, dropout=0.0, mask_diag=False):
        """

        Args:
            model_dim: int
                the input dimension for keys, queries and values
            heads: int
                number of heads
            dropout: float
                dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // heads
        self.heads = heads
        self.dropout = dropout
        self.mask_diag = mask_diag
        assert self.head_dim * heads == self.model_dim, "model_dim must be divisible by number of heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = Parameter(torch.empty(3 * model_dim, model_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * model_dim))
        self.layer_norm = LayerNorm(model_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # in proj
        nn.init.xavier_uniform_(self.in_proj_weight[:self.model_dim, :])
        nn.init.xavier_uniform_(self.in_proj_weight[self.model_dim:(self.model_dim * 2), :])
        nn.init.xavier_uniform_(self.in_proj_weight[(self.model_dim * 2):, :])
        nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, key_mask=None):
        """

        Args:
            query: Tenfor
                [batch, tgt_len, model_dim]
            key: Tensor
                [batch, src_len, model_dim]
            value: Tensor
                [batch, src_len, model_dim]
            key_mask: ByteTensor or None
                binary ByteTensor [batch, src_len] padding elements are indicated by 1s.

        Returns:

        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        bs, src_len, model_dim = key.size()
        tgt_len = query.size(1)
        heads = self.heads
        residual = query

        # k, v: [bs, src_len, model_dim]
        # q: [bs, tgt_len, model_dim]
        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        q *= self.scaling

        model_dim = q.size(2)
        dim = model_dim // heads

        # [len, batch, model_dim] -> [len, batch * heads, dim] -> [batch * heads, len, dim]
        q = q.transpose(0, 1).contiguous().view(tgt_len, bs * heads, dim).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(src_len, bs * heads, dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(src_len, bs * heads, dim).transpose(0, 1)

        # attention weights [batch * heads, tgt_len, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if key_mask is not None:
            attn_weights = attn_weights.view(bs, heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bs * heads, tgt_len, src_len)

        if self.mask_diag:
            assert tgt_len == src_len
            # [1, tgt_len, tgt_len]
            diag_mask = torch.eye(tgt_len, device=query.device, dtype=torch.uint8).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(diag_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights.float(), dim=-1,
                                 dtype=torch.float32 if attn_weights.dtype == torch.float16 else attn_weights.dtype)


        # outputs [batch * heads, tgt_len, dim]
        out = torch.bmm(attn_weights, v)
        # merge heads
        # [batch, heads, tgt_len, dim] -> [batch, tgt_len, heads, dim]
        # -> [batch, tgt_len, model_dim]
        out = out.view(bs, heads, tgt_len, dim).transpose(1, 2).contiguous().view(bs, tgt_len, model_dim)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.layer_norm(out + residual)
        return out

    def init(self, query, key, value, key_mask=None, init_scale=1.0):
        with torch.no_grad():
            return self(query, key, value, key_mask=key_mask)

    def _in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        return self._in_proj(key, start=self.model_dim).chunk(2, dim=-1)

    def _in_proj_q(self, query):
        return self._in_proj(query, end=self.model_dim)

    def _in_proj_k(self, key):
        return self._in_proj(key, start=self.model_dim, end=2 * self.model_dim)

    def _in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.model_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, features, hidden_features, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(features, hidden_features)
        self.dropout = dropout
        self.linear2 = nn.Linear(hidden_features, features)
        self.layer_norm = LayerNorm(features)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x), inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.dropout(self.linear2(x), p=self.dropout, training=self.training)
        x = self.layer_norm(residual + x)
        return x

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)
