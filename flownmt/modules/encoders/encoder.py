from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Src Encoder to encode source sentence
    """
    _registry = dict()

    def __init__(self, vocab_size, embed_dim, padding_idx):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        if self.embed.padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[self.embed.padding_idx].fill_(0)

    @overrides
    def forward(self, src_sents, masks=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoding src sentences into src encoding representations.
        Args:
            src_sents: Tensor [batch, length]
            masks: Tensor or None [batch, length]

        Returns: Tensor1, Tensor2
            Tensor1: tensor for src encoding [batch, length, hidden_size]
            Tensor2: tensor for global state [batch, hidden_size]

        """
        raise NotImplementedError

    def init(self, src_sents, masks=None, init_scale=1.0) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Encoder._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Encoder._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
