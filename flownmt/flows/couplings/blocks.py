import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from flownmt.nnet.weightnorm import Conv1dWeightNorm, LinearWeightNorm
from flownmt.nnet.attention import GlobalAttention, MultiHeadAttention
from flownmt.nnet.positional_encoding import PositionalEncoding
from flownmt.nnet.transformer import TransformerDecoderLayer


class NICEConvBlock(nn.Module):
    def __init__(self, src_features, in_features, out_features, hidden_features, kernel_size, dropout=0.0):
        super(NICEConvBlock, self).__init__()
        self.conv1 = Conv1dWeightNorm(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = Conv1dWeightNorm(hidden_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.activation = nn.ELU(inplace=True)
        self.attn = GlobalAttention(src_features, hidden_features, hidden_features, dropout=dropout)
        self.linear = LinearWeightNorm(hidden_features * 2, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        """

        Args:
            x: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                x mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: Tensor
            out tensor [batch, length, out_features]

        """
        out = self.activation(self.conv1(x.transpose(1, 2)))
        out = self.activation(self.conv2(out)).transpose(1, 2) * mask.unsqueeze(2)
        out = self.attn(out, src, key_mask=src_mask.eq(0))
        out = self.linear(torch.cat([x, out], dim=2))
        return out

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        out = self.activation(self.conv1.init(x.transpose(1, 2), init_scale=init_scale))
        out = self.activation(self.conv2.init(out, init_scale=init_scale)).transpose(1, 2) * mask.unsqueeze(2)
        out = self.attn.init(out, src, key_mask=src_mask.eq(0), init_scale=init_scale)
        out = self.linear.init(torch.cat([x, out], dim=2), init_scale=0.0)
        return out


class NICERecurrentBlock(nn.Module):
    def __init__(self, rnn_mode, src_features, in_features, out_features, hidden_features, dropout=0.0):
        super(NICERecurrentBlock, self).__init__()
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(in_features, hidden_features // 2, batch_first=True, bidirectional=True)
        self.attn = GlobalAttention(src_features, hidden_features, hidden_features, dropout=dropout)
        self.linear = LinearWeightNorm(in_features + hidden_features, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        lengths = mask.sum(dim=1).long()
        packed_out = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_out)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=mask.size(1))
        # [batch, length, out_features]
        out = self.attn(out, src, key_mask=src_mask.eq(0))
        out = self.linear(torch.cat([x, out], dim=2))
        return out

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        lengths = mask.sum(dim=1).long()
        packed_out = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_out)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=mask.size(1))
        # [batch, length, out_features]
        out = self.attn.init(out, src, key_mask=src_mask.eq(0), init_scale=init_scale)
        out = self.linear.init(torch.cat([x, out], dim=2), init_scale=0.0)
        return out


class NICESelfAttnBlock(nn.Module):
    def __init__(self, src_features, in_features, out_features, hidden_features, heads, dropout=0.0,
                 pos_enc='add', max_length=100):
        super(NICESelfAttnBlock, self).__init__()
        assert pos_enc in ['add', 'attn']
        self.src_proj = nn.Linear(src_features, in_features, bias=False) if src_features != in_features else None
        self.pos_enc = PositionalEncoding(in_features, padding_idx=None, init_size=max_length + 1)
        self.pos_attn = MultiHeadAttention(in_features, heads, dropout=dropout) if pos_enc == 'attn' else None
        self.transformer = TransformerDecoderLayer(in_features, hidden_features, heads, dropout=dropout)
        self.linear = LinearWeightNorm(in_features, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer(x, key_mask, src, src_mask.eq(0))
        return self.linear(x)

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer.init(x, key_mask, src, src_mask.eq(0), init_scale=init_scale)
        x = x * mask.unsqueeze(2)
        return self.linear.init(x, init_scale=0.0)
