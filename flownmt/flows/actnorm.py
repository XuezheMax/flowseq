from overrides import overrides
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from flownmt.flows.flow import Flow


class ActNormFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(ActNormFlow, self).__init__(inverse)
        self.in_features = in_features
        self.log_scale = Parameter(torch.Tensor(in_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    @overrides
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = input * self.log_scale.exp() + self.bias
        out = out * mask.unsqueeze(dim - 1)
        logdet = self.log_scale.sum(dim=0, keepdim=True)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = (input - self.bias) * mask.unsqueeze(dim - 1)
        out = out.div(self.log_scale.exp() + 1e-8)
        logdet = self.log_scale.sum(dim=0, keepdim=True) * -1.0
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data: input: Tensor
                input tensor [batch, N1, N2, ..., in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]
            init_scale: float
                initial scale

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        with torch.no_grad():
            out, _ = self.forward(data, mask)
            mean = out.view(-1, self.in_features).mean(dim=0)
            std = out.view(-1, self.in_features).std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, mask)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "ActNormFlow":
        return ActNormFlow(**params)


ActNormFlow.register('actnorm')
