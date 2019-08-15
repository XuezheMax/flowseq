from overrides import overrides
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.encoders.encoder import Encoder
from flownmt.nnet.transformer import TransformerEncoderLayer
from flownmt.nnet.positional_encoding import PositionalEncoding


class TransformerCore(nn.Module):
    def __init__(self, embed, num_layers, latent_dim, hidden_size, heads, dropout=0.0, max_length=100):
        super(TransformerCore, self).__init__()
        self.embed = embed
        self.padding_idx = embed.padding_idx

        embed_dim = embed.embedding_dim
        self.embed_scale = math.sqrt(embed_dim)
        assert embed_dim == latent_dim
        layers = [TransformerEncoderLayer(latent_dim, hidden_size, heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.pos_enc = PositionalEncoding(latent_dim, self.padding_idx, max_length + 1)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    @overrides
    def forward(self, src_sents, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, leagth, embed_dim]
        x = self.embed_scale * self.embed(src_sents)
        x += self.pos_enc(src_sents)
        x = F.dropout(x, p=0.2, training=self.training)

        # [batch, leagth, latent_dim]
        key_mask = masks.eq(0)
        if not key_mask.any():
            key_mask = None

        for layer in self.layers:
            x = layer(x, key_mask)

        x *= masks.unsqueeze(2)
        # [batch, latent_dim]
        batch = src_sents.size(0)
        idx = masks.sum(dim=1).long() - 1
        batch_idx = torch.arange(0, batch).long().to(idx.device)
        ctx = x[batch_idx, idx]
        return x, ctx


class TransformerEncoder(Encoder):
    """
    Src Encoder to encode source sentence with Transformer
    """

    def __init__(self, vocab_size, embed_dim, padding_idx, num_layers, latent_dim, hidden_size, heads, dropout=0.0, max_length=100):
        super(TransformerEncoder, self).__init__(vocab_size, embed_dim, padding_idx)
        self.core = TransformerCore(self.embed, num_layers, latent_dim, hidden_size, heads, dropout=dropout, max_length=max_length)

    @overrides
    def forward(self, src_sents, masks=None) -> Tuple[torch.Tensor, torch.Tensor]:
        src_enc, ctx = self.core(src_sents, masks=masks)
        return src_enc, ctx

    def init(self, src_sents, masks=None, init_scale=1.0) -> torch.Tensor:
        with torch.no_grad():
            src_enc, _ = self.core(src_sents, masks=masks)
            return src_enc

    @classmethod
    def from_params(cls, params: Dict) -> "TransformerEncoder":
        return TransformerEncoder(**params)


TransformerEncoder.register('transformer')
