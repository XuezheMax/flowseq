import torch.nn as nn

from flownmt.nnet.attention import MultiHeadAttention, PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, heads, dropout=0.0, mask_diag=False):
        super(TransformerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(model_dim, heads, dropout=dropout, mask_diag=mask_diag)
        self.pos_ffn = PositionwiseFeedForward(model_dim, hidden_dim, dropout=dropout)

    def forward(self, x, mask):
        out = self.slf_attn(x, x, x, key_mask=mask)
        out = self.pos_ffn(out)
        return out

    def init(self, x, mask, init_scale=1.0):
        out = self.slf_attn.init(x, x, x, key_mask=mask, init_scale=init_scale)
        out = self.pos_ffn.init(out, init_scale=init_scale)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, heads, dropout=0.0, mask_diag=False):
        super(TransformerDecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(model_dim, heads, dropout=dropout, mask_diag=mask_diag)
        self.enc_attn = MultiHeadAttention(model_dim, heads, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(model_dim, hidden_dim, dropout=dropout)

    def forward(self, x, mask, src, src_mask):
        out = self.slf_attn(x, x, x, key_mask=mask)
        out = self.enc_attn(out, src, src, key_mask=src_mask)
        out = self.pos_ffn(out)
        return out

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        out = self.slf_attn.init(x, x, x, key_mask=mask, init_scale=init_scale)
        out = self.enc_attn.init(out, src, src, key_mask=src_mask, init_scale=init_scale)
        out = self.pos_ffn.init(out, init_scale=init_scale)
        return out
