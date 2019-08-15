from overrides import overrides
from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.nnet.weightnorm import LinearWeightNorm
from flownmt.nnet.transformer import TransformerDecoderLayer
from flownmt.nnet.positional_encoding import PositionalEncoding
from flownmt.modules.posteriors.posterior import Posterior


class TransformerCore(nn.Module):
    def __init__(self, embed, num_layers, latent_dim, hidden_size, heads, dropout=0.0, dropword=0.0, max_length=100):
        super(TransformerCore, self).__init__()
        self.tgt_embed = embed
        self.padding_idx = embed.padding_idx

        embed_dim = embed.embedding_dim
        self.embed_scale = math.sqrt(embed_dim)
        assert embed_dim == latent_dim
        layers = [TransformerDecoderLayer(latent_dim, hidden_size, heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.pos_enc = PositionalEncoding(latent_dim, self.padding_idx, max_length + 1)
        self.dropword = dropword # drop entire tokens
        self.mu = LinearWeightNorm(latent_dim, latent_dim, bias=True)
        self.logvar = LinearWeightNorm(latent_dim, latent_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    @overrides
    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        x = self.embed_scale * self.tgt_embed(tgt_sents)
        x = F.dropout2d(x, p=self.dropword, training=self.training)
        x += self.pos_enc(tgt_sents)
        x = F.dropout(x, p=0.2, training=self.training)

        mask = tgt_masks.eq(0)
        key_mask = src_masks.eq(0)
        for layer in self.layers:
            x = layer(x, mask, src_enc, key_mask)

        mu = self.mu(x) * tgt_masks.unsqueeze(2)
        logvar = self.logvar(x) * tgt_masks.unsqueeze(2)
        return mu, logvar

    def init(self, tgt_sents, tgt_masks, src_enc, src_masks, init_scale=1.0, init_mu=True, init_var=True):
        with torch.no_grad():
            x = self.embed_scale * self.tgt_embed(tgt_sents)
            x = F.dropout2d(x, p=self.dropword, training=self.training)
            x += self.pos_enc(tgt_sents)
            x = F.dropout(x, p=0.2, training=self.training)

            mask = tgt_masks.eq(0)
            key_mask = src_masks.eq(0)
            for layer in self.layers:
                x = layer.init(x, mask, src_enc, key_mask, init_scale=init_scale)

            x = x * tgt_masks.unsqueeze(2)
            mu = self.mu.init(x, init_scale=0.05 * init_scale) if init_mu else self.mu(x)
            logvar = self.logvar.init(x, init_scale=0.05 * init_scale) if init_var else self.logvar(x)
            mu = mu * tgt_masks.unsqueeze(2)
            logvar = logvar * tgt_masks.unsqueeze(2)
            return mu, logvar


class TransformerPosterior(Posterior):
    """
    Posterior with Transformer
    """
    def __init__(self, vocab_size, embed_dim, padding_idx, num_layers, latent_dim, hidden_size, heads,
                 dropout=0.0, dropword=0.0, max_length=100, _shared_embed=None):
        super(TransformerPosterior, self).__init__(vocab_size, embed_dim, padding_idx, _shared_embed=_shared_embed)
        self.core = TransformerCore(self.tgt_embed, num_layers, latent_dim, hidden_size, heads,
                                    dropout=dropout, dropword=dropword, max_length=max_length)

    def target_embed_weight(self):
        if isinstance(self.core, nn.DataParallel):
            return self.core.module.tgt_embedd.weight
        else:
            return self.core.tgt_embed.weight

    @overrides
    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        return self.core(tgt_sents, tgt_masks, src_enc, src_masks)

    @overrides
    def sample(self, tgt_sents: torch.Tensor, tgt_masks: torch.Tensor,
               src_enc: torch.Tensor, src_masks: torch.Tensor,
               nsamples: int =1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.core(tgt_sents, tgt_masks, src_enc, src_masks)
        z, eps = Posterior.reparameterize(mu, logvar, tgt_masks, nsamples=nsamples, random=random)
        log_probs = Posterior.log_probability(z, eps, mu, logvar, tgt_masks)
        return z, log_probs

    @overrides
    def init(self, tgt_sents, tgt_masks, src_enc, src_masks, init_scale=1.0, init_mu=True, init_var=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.core.init(tgt_sents, tgt_masks, src_enc, src_masks,
                                    init_scale=init_scale, init_mu=init_mu, init_var=init_var)
        z, eps = Posterior.reparameterize(mu, logvar, tgt_masks, random=True)
        log_probs = Posterior.log_probability(z, eps, mu, logvar, tgt_masks)
        z = z.squeeze(1)
        log_probs = log_probs.squeeze(1)
        return z, log_probs

    @classmethod
    def from_params(cls, params: Dict) -> "TransformerPosterior":
        return TransformerPosterior(**params)


TransformerPosterior.register('transformer')
