from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from flownmt.modules.decoders.decoder import Decoder
from flownmt.nnet.attention import GlobalAttention


class RecurrentDecoder(Decoder):
    """
    Decoder with Recurrent Neural Networks
    """
    def __init__(self, vocab_size, latent_dim, rnn_mode, num_layers, hidden_size, bidirectional=True,
                 dropout=0.0, dropword=0.0, label_smoothing=0., _shared_weight=None):
        super(RecurrentDecoder, self).__init__(vocab_size, latent_dim,
                                               label_smoothing=label_smoothing,
                                               _shared_weight=_shared_weight)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        assert hidden_size % 2 == 0
        # RNN for processing latent variables zs
        if bidirectional:
            self.rnn = RNN(latent_dim, hidden_size // 2, num_layers=num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = RNN(latent_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

        self.attn = GlobalAttention(latent_dim, hidden_size, latent_dim, hidden_features=hidden_size)
        self.ctx_proj = nn.Sequential(nn.Linear(latent_dim + hidden_size, latent_dim), nn.ELU())
        self.dropout = dropout
        self.dropout2d = nn.Dropout2d(dropword) if dropword > 0. else None # drop entire tokens

    def forward(self, z, mask, src, src_mask):
        lengths = mask.sum(dim=1).long()
        if self.dropout2d is not None:
            z = self.dropout2d(z)

        packed_z = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        packed_enc, _ = self.rnn(packed_z)
        enc, _ = pad_packed_sequence(packed_enc, batch_first=True, total_length=mask.size(1))

        ctx = self.attn(enc, src, key_mask=src_mask.eq(0))
        ctx = torch.cat([ctx, enc], dim=2)
        ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
        return self.readout(ctx)

    @overrides
    def init(self, z, mask, src, src_mask, init_scale=1.0):
        with torch.no_grad():
            return self(z, mask, src, src_mask)

    @overrides
    def decode(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            z: Tensor
                latent code [batch, length, hidden_size]
            mask: Tensor
                mask [batch, length]
            src: Tensor
                src encoding [batch, src_length, hidden_size]
            src_mask: Tensor
                source mask [batch, src_length]

        Returns: Tensor1, Tensor2
            Tenser1: decoded word index [batch, length]
            Tensor2: log probabilities of decoding [batch]

        """
        # [batch, length, vocab_size]
        log_probs = F.log_softmax(self(z, mask, src, src_mask), dim=2)
        # [batch, length]
        log_probs, dec = log_probs.max(dim=2)
        dec = dec * mask.long()
        # [batch]
        log_probs = log_probs.mul(mask).sum(dim=1)
        return dec, log_probs

    @overrides
    def loss(self, z: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            z: Tensor
                latent codes [batch, length, hidden_size]
            target: LongTensor
                target translations [batch, length]
            mask: Tensor
                masks for target sentence [batch, length]
            src: Tensor
                src encoding [batch, src_length, hidden_size]
            src_mask: Tensor
                source mask [batch, src_length]

        Returns: Tensor
            tensor for loss [batch]

        """
        # [batch, length, vocab_size] -> [batch, vocab_size, length]
        logits = self(z, mask, src, src_mask).transpose(1, 2)
        # [batch, length]
        loss = self.criterion(logits, target).mul(mask)
        return loss.sum(dim=1)

    @classmethod
    def from_params(cls, params: Dict) -> "RecurrentDecoder":
        return RecurrentDecoder(**params)


RecurrentDecoder.register('rnn')
