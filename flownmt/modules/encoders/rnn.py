from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.encoders.encoder import Encoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RecurrentCore(nn.Module):
    def __init__(self, embed, rnn_mode, num_layers, latent_dim, hidden_size, dropout=0.0):
        super(RecurrentCore, self).__init__()
        self.embed = embed

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        assert hidden_size % 2 == 0
        self.rnn = RNN(embed.embedding_dim, hidden_size // 2,
                       num_layers=num_layers, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Sequential(nn.Linear(hidden_size, latent_dim), nn.ELU())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.enc_proj[0].bias, 0.)

    @overrides
    def forward(self, src_sents, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embed = F.dropout(self.embed(src_sents), p=0.2, training=self.training)

        lengths = masks.sum(dim=1).long()
        packed_embed = pack_padded_sequence(word_embed, lengths, batch_first=True, enforce_sorted=False)
        packed_enc, _ = self.rnn(packed_embed)
        # [batch, length, hidden_size]
        src_enc, _ = pad_packed_sequence(packed_enc, batch_first=True, total_length=masks.size(1))
        # [batch, length, latent_dim]
        src_enc = self.enc_proj(src_enc).mul(masks.unsqueeze(2))

        # [batch, latent_dim]
        batch = src_sents.size(0)
        idx = lengths - 1
        batch_idx = torch.arange(0, batch).long().to(idx.device)
        ctx = src_enc[batch_idx, idx]
        return src_enc, ctx


class RecurrentEncoder(Encoder):
    """
    Src Encoder to encode source sentence with Recurrent Neural Networks
    """

    def __init__(self, vocab_size, embed_dim, padding_idx, rnn_mode, num_layers, latent_dim, hidden_size, dropout=0.0):
        super(RecurrentEncoder, self).__init__(vocab_size, embed_dim, padding_idx)
        self.core = RecurrentCore(self.embed, rnn_mode, num_layers, latent_dim, hidden_size, dropout=dropout)

    @overrides
    def forward(self, src_sents, masks=None) -> Tuple[torch.Tensor, torch.Tensor]:
        src_enc, ctx = self.core(src_sents, masks=masks)
        return src_enc, ctx

    def init(self, src_sents, masks=None, init_scale=1.0) -> torch.Tensor:
        with torch.no_grad():
            src_enc, _ = self.core(src_sents, masks=masks)
            return src_enc

    @classmethod
    def from_params(cls, params: Dict) -> "RecurrentEncoder":
        return RecurrentEncoder(**params)


RecurrentEncoder.register('rnn')
