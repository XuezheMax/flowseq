from typing import Dict, Tuple
import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing Flow base class
    """
    _registry = dict()

    def __init__(self, inverse):
        super(Flow, self).__init__()
        self.inverse = inverse

    def forward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def backward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def init(self, *input, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fwdpass(self, x: torch.Tensor, *h, init=False, init_scale=1.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The random variable before flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial y / \partial x`
            Then the density :math:`\log(p(y)) = \log(p(x)) - logdet`

        """
        if self.inverse:
            if init:
                raise RuntimeError('inverse flow shold be initialized with backward pass')
            else:
                return self.backward(x, *h, **kwargs)
        else:
            if init:
                return self.init(x, *h, init_scale=init_scale, **kwargs)
            else:
                return self.forward(x, *h, **kwargs)

    def bwdpass(self, y: torch.Tensor, *h, init=False, init_scale=1.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y: Tensor
                The random variable after flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        if self.inverse:
            if init:
                return self.init(y, *h, init_scale=init_scale, **kwargs)
            else:
                return self.forward(y, *h, **kwargs)
        else:
            if init:
                raise RuntimeError('forward flow should be initialzed with forward pass')
            else:
                return self.backward(y, *h, **kwargs)

    @classmethod
    def register(cls, name: str):
        Flow._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Flow._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
