from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.decoders.decoder import Decoder
from flownmt.nnet.attention import GlobalAttention

class SimpleDecoder(Decoder):
    """
     Simple Decoder to predict translations from latent z
    """

    def __init__(self, vocab_size, latent_dim, hidden_size, dropout=0.0, label_smoothing=0., _shared_weight=None):
        super(SimpleDecoder, self).__init__(vocab_size, latent_dim,
                                            label_smoothing=label_smoothing,
                                            _shared_weight=_shared_weight)
        self.attn = GlobalAttention(latent_dim, latent_dim, latent_dim, hidden_features=hidden_size)
        ctx_features = latent_dim * 2
        self.ctx_proj = nn.Sequential(nn.Linear(ctx_features, latent_dim), nn.ELU())
        self.dropout = dropout

    @overrides
    def forward(self, z, src, src_mask):
        ctx = self.attn(z, src, key_mask=src_mask.eq(0))
        ctx = F.dropout(self.ctx_proj(torch.cat([ctx, z], dim=2)), p=self.dropout, training=self.training)
        return self.readout(ctx)

    @overrides
    def init(self, z, mask, src, src_mask, init_scale=1.0):
        with torch.no_grad():
            self(z, src, src_mask)

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
        log_probs = F.log_softmax(self(z, src, src_mask), dim=2)
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
        logits = self(z, src, src_mask).transpose(1, 2)
        # [batch, length]
        loss = self.criterion(logits, target).mul(mask)
        return loss.sum(dim=1)

    @classmethod
    def from_params(cls, params: Dict) -> "SimpleDecoder":
        return SimpleDecoder(**params)


SimpleDecoder.register('simple')
