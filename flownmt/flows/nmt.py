from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from flownmt.flows.flow import Flow
from flownmt.flows.actnorm import ActNormFlow
from flownmt.flows.linear import InvertibleMultiHeadFlow
from flownmt.flows.couplings.coupling import NICE
from flownmt.utils import squeeze, unsqueeze, split, unsplit


class NMTFlowPOSAttnUnit(Flow):
    """
    Unit for POS Attention
    """
    def __init__(self, features, src_features, hidden_features=None, inverse=False,
                 transform='affine', heads=1, max_length=100, dropout=0.0):
        super(NMTFlowPOSAttnUnit, self).__init__(inverse)
        self.actnorm = ActNormFlow(features, inverse=inverse)
        self.coupling_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                split_dim=2, split_type='continuous', order='up',
                                transform=transform, type='self_attn', heads=heads,
                                dropout=dropout, pos_enc='attn', max_length=max_length)

        self.coupling_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                  split_dim=2, split_type='continuous', order='down',
                                  transform=transform, type='self_attn', heads=heads,
                                  dropout=dropout, pos_enc='add', max_length=max_length)

    @overrides
    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input, tgt_mask)

        out, logdet = self.coupling_up.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling_down.backward(input, tgt_mask, src, src_mask)

        out, logdet = self.coupling_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, tgt_mask, init_scale=init_scale)

        out, logdet = self.coupling_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlowUnit(Flow):
    """
    One Unit of NMTFlowStep
    """

    def __init__(self, features, src_features, hidden_features=None, inverse=False, transform='affine',
                 coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, split_timestep=True):
        super(NMTFlowUnit, self).__init__(inverse)
        # dim=2, type=continuous
        self.coupling1_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                 split_dim=2, split_type='continuous', order='up',
                                 transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.coupling1_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                   split_dim=2, split_type='continuous', order='down',
                                   transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                   heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)
        self.actnorm1 = ActNormFlow(features, inverse=inverse)

        # dim=2, type=skip
        self.coupling2_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                 split_dim=2, split_type='skip', order='up',
                                 transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.coupling2_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                   split_dim=2, split_type='skip', order='down',
                                   transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                   heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.split_timestep = split_timestep
        if split_timestep:
            self.actnorm2 = ActNormFlow(features, inverse=inverse)
            # dim=1, type=skip
            self.coupling3_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                     split_dim=1, split_type='skip', order='up',
                                     transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                     heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

            self.coupling3_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                       split_dim=1, split_type='skip', order='down',
                                       transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                       heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)
        else:
            self.actnorm2 = None
            self.coupling3_up = None
            self.coupling3_down = None

    @overrides
    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling1_up.forward(input, tgt_mask, src, src_mask)

        out, logdet = self.coupling1_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_up.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        if self.split_timestep:
            # ================================================================================

            out, logdet = self.actnorm2.forward(out, tgt_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            # block3 dim=1, type=skip
            out, logdet = self.coupling3_up.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling3_down.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split_timestep:
            # block3 dim=1, type=skip
            out, logdet_accum = self.coupling3_down.backward(input, tgt_mask, src, src_mask)

            out, logdet = self.coupling3_up.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            out, logdet = self.actnorm2.backward(out, tgt_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================
        else:
            out, logdet_accum = input, 0

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_down.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block1 dim=2, type=continuous
        out, logdet = self.coupling1_down.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling1_up.init(data, tgt_mask, src, src_mask, init_scale=init_scale)

        out, logdet = self.coupling1_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        if self.split_timestep:
            # ================================================================================

            out, logdet = self.actnorm2.init(out, tgt_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            # block3 dim=1, type=skip
            out, logdet = self.coupling3_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling3_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlowStep(Flow):
    """
    One step of NMTFlow
    """

    def __init__(self, features, src_features, hidden_features=None, inverse=False, transform='affine',
                 coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, split_timestep=True):
        super(NMTFlowStep, self).__init__(inverse)
        self.actnorm1 = ActNormFlow(features, inverse=inverse)
        self.linear1 = InvertibleMultiHeadFlow(features, type='A', inverse=inverse)
        self.unit1 = NMTFlowUnit(features, src_features, hidden_features=hidden_features, inverse=inverse,
                                 transform=transform, coupling_type=coupling_type, kernel_size=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, max_length=max_length, split_timestep=split_timestep)
        self.actnorm2 = ActNormFlow(features, inverse=inverse)
        self.linear2 = InvertibleMultiHeadFlow(features, type='B', inverse=inverse)
        self.unit2 = NMTFlowUnit(features, src_features, hidden_features=hidden_features, inverse=inverse,
                                 transform=transform, coupling_type=coupling_type, kernel_size=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, max_length=max_length, split_timestep=split_timestep)

    def sync(self):
        self.linear1.sync()
        self.linear2.sync()

    @overrides
    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm1.forward(input, tgt_mask)

        out, logdet = self.linear1.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear2.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit2.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.unit2.backward(input, tgt_mask, src, src_mask)

        out, logdet = self.linear2.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm1.init(data, tgt_mask, init_scale=init_scale)

        out, logdet = self.linear1.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear2.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit2.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class NMTFlowBlock(Flow):
    """
    NMT Flow Block
    """

    def __init__(self, num_steps, features, src_features, hidden_features=None, inverse=False, prior=False, factor=2,
                 transform='affine', coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, pos_attn=False):
        super(NMTFlowBlock, self).__init__(inverse)
        if pos_attn:
            self.pos_attn = NMTFlowPOSAttnUnit(features, src_features,hidden_features=hidden_features,
                                               inverse=inverse, transform=transform, heads=heads,
                                               max_length=max_length, dropout=dropout)
        else:
            self.pos_attn = None

        steps = [NMTFlowStep(features, src_features, hidden_features=hidden_features, inverse=inverse,
                             transform=transform, coupling_type=coupling_type, kernel_size=kernel_size,
                             rnn_mode=rnn_mode, heads=heads, max_length=max_length,
                             dropout=dropout, split_timestep=prior) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        if prior:
            assert features % factor == 0, 'features {} should divide factor {}'.format(features, factor)
            self.prior = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                              split_dim=2, split_type='continuous', order='up', factor=factor,
                              transform=transform, type=coupling_type, kernel=kernel_size,
                              heads=heads, rnn_mode=rnn_mode, pos_enc='add', max_length=max_length, dropout=dropout)
            self.z_features = features - features // factor
            assert self.z_features == self.prior.z1_channels
        else:
            self.prior = None
            self.z_features = features

    def sync(self):
        for step in self.steps:
            step.sync()

    @overrides
    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        if self.pos_attn is None:
            logdet_accum = input.new_zeros(input.size(0))
            out = input
        else:
            out, logdet_accum = self.pos_attn.forward(input, tgt_mask, src, src_mask)

        for step in self.steps:
            out, logdet = step.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        if self.prior is not None:
            out, logdet = self.prior.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prior is None:
            logdet_accum = input.new_zeros(input.size(0))
            out = input
        else:
            out, logdet_accum = self.prior.backward(input, tgt_mask, src, src_mask)

        for step in reversed(self.steps):
            out, logdet = step.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        if self.pos_attn is not None:
            out, logdet = self.pos_attn.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        if self.pos_attn is None:
            logdet_accum = data.new_zeros(data.size(0))
            out = data
        else:
            out, logdet_accum = self.pos_attn.init(data, tgt_mask, src, src_mask, init_scale=init_scale)

        for step in self.steps:
            out, logdet = step.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        if self.prior is not None:
            out, logdet = self.prior.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlow(Flow):
    """
    NMT Flow
    """

    def __init__(self, levels, num_steps, features, src_features, factors, hidden_features=None, inverse=False,
                 transform='affine', coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, pos_enc='add', max_length=100, dropout=0.0):
        super(NMTFlow, self).__init__(inverse)
        assert levels == len(num_steps)
        assert levels == len(factors) + 1
        blocks = []
        self.levels = levels
        self.features = features
        pos_attn = coupling_type == 'self_attn' and pos_enc == 'attn'

        for level in range(levels):
            if level == levels - 1:
                block = NMTFlowBlock(num_steps[level], features, src_features, hidden_features=hidden_features,
                                     inverse=inverse, prior=False, coupling_type=coupling_type, transform=transform,
                                     kernel_size=kernel_size, rnn_mode=rnn_mode, heads=heads, max_length=max_length,
                                     dropout=dropout, pos_attn=pos_attn)
            else:
                factor = factors[level]
                block = NMTFlowBlock(num_steps[level], features, src_features, hidden_features=hidden_features,
                                     inverse=inverse, prior=True, factor=factor, coupling_type=coupling_type,
                                     transform=transform,kernel_size=kernel_size, rnn_mode=rnn_mode, heads=heads,
                                     max_length=max_length, dropout=dropout, pos_attn=pos_attn)
                features = block.z_features * 2
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def sync(self):
        for block in self.blocks:
            block.sync()

    @overrides
    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                out1, out2 = split(out, block.z_features)
                outputs.append(out2)
                out, tgt_mask = squeeze(out1, tgt_mask)

        for _ in range(self.levels - 1):
            out2 = outputs.pop()
            out = unsqueeze(out)
            out = unsplit([out, out2])
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        masks = []
        out = input
        for i in range(self.levels - 1):
            out1, out2 = split(out, self.blocks[i].z_features)
            outputs.append(out2)
            masks.append(tgt_mask)
            out, tgt_mask = squeeze(out1, tgt_mask)

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if i > 0:
                out2 = outputs.pop()
                tgt_mask = masks.pop()
                out1 = unsqueeze(out)
                out = unsplit([out1, out2])
            out, logdet = block.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet
        assert len(outputs) == 0
        assert len(masks) == 0

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                out1, out2 = split(out, block.z_features)
                outputs.append(out2)
                out, tgt_mask = squeeze(out1, tgt_mask)

        for _ in range(self.levels - 1):
            out2 = outputs.pop()
            out = unsqueeze(out)
            out = unsplit([out, out2])
        assert len(outputs) == 0
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "NMTFlow":
        return NMTFlow(**params)


NMTFlow.register('nmt')
