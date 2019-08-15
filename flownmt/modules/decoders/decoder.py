from typing import Dict, Tuple
import torch
import torch.nn as nn

from flownmt.nnet.criterion import LabelSmoothedCrossEntropyLoss


class Decoder(nn.Module):
    """
    Decoder to predict translations from latent z
    """
    _registry = dict()

    def __init__(self, vocab_size, latent_dim, label_smoothing=0., _shared_weight=None):
        super(Decoder, self).__init__()
        self.readout = nn.Linear(latent_dim, vocab_size, bias=True)
        if _shared_weight is not None:
            self.readout.weight = _shared_weight
            nn.init.constant_(self.readout.bias, 0.)
        else:
            self.reset_parameters(latent_dim)

        if label_smoothing < 1e-5:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif 1e-5 < label_smoothing < 1.0:
            self.criterion = LabelSmoothedCrossEntropyLoss(label_smoothing)
        else:
            raise ValueError('label smoothing should be less than 1.0.')

    def reset_parameters(self, dim):
        # nn.init.normal_(self.readout.weight, mean=0, std=dim ** -0.5)
        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        nn.init.constant_(self.readout.bias, 0.)

    def init(self, z, mask, src, src_mask, init_scale=1.0):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Decoder._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Decoder._registry[name]

    @classmethod
    def from_params(cls, params: Dict) -> "Decoder":
        raise NotImplementedError


Decoder.register('simple')
