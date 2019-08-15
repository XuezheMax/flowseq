from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.decoders.decoder import Decoder
from flownmt.nnet.attention import MultiHeadAttention
from flownmt.nnet.transformer import TransformerDecoderLayer
from flownmt.nnet.positional_encoding import PositionalEncoding


class TransformerDecoder(Decoder):
    """
    Decoder with Transformer
    """
    def __init__(self, vocab_size, latent_dim, num_layers, hidden_size, heads, label_smoothing=0.,
                 dropout=0.0, dropword=0.0, max_length=100, _shared_weight=None):
        super(TransformerDecoder, self).__init__(vocab_size, latent_dim,
                                                 label_smoothing=label_smoothing,
                                                 _shared_weight=_shared_weight)
        self.pos_enc = PositionalEncoding(latent_dim, None, max_length + 1)
        self.pos_attn = MultiHeadAttention(latent_dim, heads, dropout=dropout)
        layers = [TransformerDecoderLayer(latent_dim, hidden_size, heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.dropword = dropword # drop entire tokens

    def forward(self, z, mask, src, src_mask):
        z = F.dropout2d(z, p=self.dropword, training=self.training)
        # [batch, length, latent_dim]
        pos_enc = self.pos_enc(z) * mask.unsqueeze(2)

        key_mask = mask.eq(0)
        ctx = self.pos_attn(pos_enc, z, z, key_mask)

        src_mask = src_mask.eq(0)
        for layer in self.layers:
            ctx = layer(ctx, key_mask, src, src_mask)

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
    def from_params(cls, params: Dict) -> "TransformerDecoder":
        return TransformerDecoder(**params)


TransformerDecoder.register('transformer')
