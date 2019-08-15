from overrides import overrides
from typing import Tuple, Dict
import torch

from flownmt.flows.couplings.blocks import NICEConvBlock, NICERecurrentBlock, NICESelfAttnBlock
from flownmt.flows.flow import Flow
from flownmt.flows.couplings.transform import Transform, Additive, Affine, NLSQ


class NICE(Flow):
    """
    NICE Flow
    """
    def __init__(self, src_features, features, hidden_features=None, inverse=False, split_dim=2, split_type='continuous', order='up', factor=2,
                 transform='affine', type='conv', kernel=3, rnn_mode='LSTM', heads=1, dropout=0.0, pos_enc='add', max_length=100):
        super(NICE, self).__init__(inverse)
        self.features = features
        assert split_dim in [1, 2]
        assert split_type in ['continuous', 'skip']
        if split_dim == 1:
            assert split_type == 'skip'
        if factor != 2:
            assert split_type == 'continuous'
        assert order in ['up', 'down']
        self.split_dim = split_dim
        self.split_type = split_type
        self.up = order == 'up'
        if split_dim == 2:
            out_features = features // factor
            in_features = features - out_features
            self.z1_channels = in_features if self.up else out_features
        else:
            in_features = features
            out_features = features
            self.z1_channels = None
        assert transform in ['additive', 'affine', 'nlsq']
        if transform == 'additive':
            self.transform = Additive
        elif transform == 'affine':
            self.transform = Affine
            out_features = out_features * 2
        elif transform == 'nlsq':
            self.transform = NLSQ
            out_features = out_features * 5
        else:
            raise ValueError('unknown transform: {}'.format(transform))

        if hidden_features is None:
            hidden_features = min(2 * in_features, 1024)
        assert type in ['conv', 'self_attn', 'rnn']
        if type == 'conv':
            self.net = NICEConvBlock(src_features, in_features, out_features, hidden_features, kernel_size=kernel, dropout=dropout)
        elif type == 'rnn':
            self.net = NICERecurrentBlock(rnn_mode, src_features, in_features, out_features, hidden_features, dropout=dropout)
        else:
            self.net = NICESelfAttnBlock(src_features, in_features, out_features, hidden_features,
                                         heads=heads, dropout=dropout, pos_enc=pos_enc, max_length=max_length)

    def split(self, z, mask):
        split_dim = self.split_dim
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim), mask
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            if split_dim == 1:
                mask = mask.index_select(split_dim, idx1)
            return (z1, z2), mask
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = self.split_dim
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        params = self.net(z, mask, src, src_mask)
        return params

    def init_net(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0):
        params = self.net.init(z, mask, src, src_mask, init_scale=init_scale)
        return params

    @overrides
    def forward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, length, in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(input, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    @overrides
    def backward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, length, in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(input, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.bwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    @overrides
    def init(self, data: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(data, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.init_net(z, mask, src, src_mask, init_scale=init_scale)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}, scale={}'.format(self.inverse, self.in_channels, self.scale)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE":
        return NICE(**params)


NICE.register('nice')
