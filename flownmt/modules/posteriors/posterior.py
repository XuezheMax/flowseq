import math
from typing import Dict, Tuple
import torch
import torch.nn as nn


class Posterior(nn.Module):
    """
    posterior class
    """
    _registry = dict()

    def __init__(self, vocab_size, embed_dim, padding_idx, _shared_embed=None):
        super(Posterior, self).__init__()
        if _shared_embed is None:
            self.tgt_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            self.reset_parameters()
        else:
            self.tgt_embed = _shared_embed

    def reset_parameters(self):
        nn.init.uniform_(self.tgt_embed.weight, -0.1, 0.1)
        if self.tgt_embed.padding_idx is not None:
            with torch.no_grad():
                self.tgt_embed.weight[self.tgt_embed.padding_idx].fill_(0)

    def target_embed_weight(self):
        raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar, mask, nsamples=1, random=True):
        # [batch, length, dim]
        size = mu.size()
        std = logvar.mul(0.5).exp()
        # [batch, nsamples, length, dim]
        if random:
            eps = torch.randn(size[0], nsamples, *size[1:], device=mu.device)
            eps *= mask.view(size[0], 1, size[1], 1)
        else:
            eps = mu.new_zeros(size[0], nsamples, *size[1:])
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1)), eps


    @staticmethod
    def log_probability(z, eps, mu, logvar, mask):
        size = eps.size()
        nz = size[3]
        # [batch, nsamples, length, nz]
        log_probs = logvar.unsqueeze(1) + eps.pow(2)
        # [batch, 1]
        cc = mask.sum(dim=1, keepdim=True) * (math.log(math.pi * 2.) * nz)
        # [batch, nsamples, length * nz] --> [batch, nsamples]
        log_probs = log_probs.view(size[0], size[1], -1).sum(dim=2) + cc
        return log_probs * -0.5

    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        raise  NotImplementedError

    def init(self, tgt_sents, tgt_masks, src_enc, src_masks, init_scale=1.0, init_mu=True, init_var=True) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(self, tgt_sents: torch.Tensor, tgt_masks: torch.Tensor,
               src_enc: torch.Tensor, src_masks: torch.Tensor,
               nsamples: int =1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            tgt_masks: Tensor [batch, tgt_length]
                tensor for target masks
            src_enc: Tensor [batch, src_length, hidden_size]
                tensor for source encoding
            src_masks: Tensor [batch, src_length]
                tensor for source masks
            nsamples: int
                number of samples
            random: bool
                if True, perform random sampling. Otherwise, return mean.

        Returns: Tensor1, Tensor2
            Tensor1: samples from the posterior [batch, nsamples, tgt_length, nz]
            Tensor2: log probabilities [batch, nsamples]

        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Posterior._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Posterior._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
