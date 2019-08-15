from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from flownmt.flows.flow import Flow


class InvertibleLinearFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(InvertibleLinearFlow, self).__init__(inverse)
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

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
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight)
        _, logdet = torch.slogdet(self.weight)
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
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight_inv)
        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleLinearFlow":
        return InvertibleLinearFlow(**params)


class InvertibleMultiHeadFlow(Flow):
    @staticmethod
    def _get_heads(in_features):
        units = [32, 16, 8]
        for unit in units:
            if in_features % unit == 0:
                return in_features // unit
        assert in_features < 8, 'features={}'.format(in_features)
        return 1

    def __init__(self, in_features, heads=None, type='A', inverse=False):
        super(InvertibleMultiHeadFlow, self).__init__(inverse)
        self.in_features = in_features
        if heads is None:
            heads = InvertibleMultiHeadFlow._get_heads(in_features)
        self.heads = heads
        self.type = type
        assert in_features % heads == 0, 'features ({}) should be divided by heads ({})'.format(in_features, heads)
        assert type in ['A', 'B'], 'type should belong to [A, B]'
        self.weight = Parameter(torch.Tensor(in_features // heads, in_features // heads))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

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
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
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
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight_inv)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, mask)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}, heads={}, type={}'.format(self.inverse, self.in_features, self.heads, self.type)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleMultiHeadFlow":
        return InvertibleMultiHeadFlow(**params)


InvertibleLinearFlow.register('invertible_linear')
InvertibleMultiHeadFlow.register('invertible_multihead')
