from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.priors.length_predictors.predictor import LengthPredictor
from flownmt.modules.priors.length_predictors.utils import discretized_mix_logistic_loss, discretized_mix_logistic_topk


class DiffDiscreteMixLogisticLengthPredictor(LengthPredictor):
    def __init__(self, features, max_src_length, diff_range, nmix=1, dropout=0.0):
        super(DiffDiscreteMixLogisticLengthPredictor, self).__init__()
        self.max_src_length = max_src_length
        self.range = diff_range
        self.nmix = nmix
        self.features = features
        self.dropout = dropout
        self.ctx_proj = None
        self.diff = None

    def set_length_unit(self, length_unit):
        self.length_unit = length_unit
        self.ctx_proj = nn.Sequential(nn.Linear(self.features, self.features), nn.ELU())
        self.diff = nn.Linear(self.features, 3 * self.nmix)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.ctx_proj[0].bias, 0.)
        nn.init.uniform_(self.diff.weight, -0.1, 0.1)
        nn.init.constant_(self.diff.bias, 0.)

    def forward(self, ctx):
        ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
        # [batch, 3 * nmix]
        coeffs = self.diff(ctx)
        # [batch, nmix]
        logit_probs = F.log_softmax(coeffs[:, :self.nmix], dim=1)
        mu = coeffs[:, self.nmix:self.nmix * 2]
        log_scale = coeffs[:, self.nmix * 2:]
        return mu, log_scale, logit_probs

    @overrides
    def loss(self, ctx: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx: Tensor
                tensor [batch, features]
            src_mask: Tensor
                tensor for source mask [batch, src_length]
            tgt_mask: Tensor
                tensor for target mask [batch, tgt_length]
        Returns: Tensor
            tensor for loss [batch]
        """
        src_lengths = src_mask.sum(dim=1).float()
        tgt_lengths = tgt_mask.sum(dim=1).float()
        mu, log_scale, logit_probs = self(ctx, src_lengths.long())
        x = (tgt_lengths - src_lengths).div(self.range).clamp(min=-1, max=1)
        bin_size = 0.5 / self.range
        lower = bin_size - 1.0
        upper = 1.0 - bin_size
        loss = discretized_mix_logistic_loss(x, mu, log_scale, logit_probs, bin_size, lower, upper)
        return loss

    @overrides
    def predict(self, ctx: torch.Tensor, src_mask:torch.Tensor, topk: int = 1) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            ctx: Tensor
                tensor [batch, features]
            src_mask: Tensor
                tensor for source mask [batch, src_length]
            topk: int (default 1)
                return top k length candidates for each src sentence
        Returns: Tensor1, LongTensor2
            Tensor1: log probs for each length
            LongTensor2: tensor for lengths [batch, topk]
        """
        bin_size = 0.5 / self.range
        lower = bin_size - 1.0
        upper = 1.0 - bin_size
        # [batch]
        src_lengths = src_mask.sum(dim=1).long()
        mu, log_scale, logit_probs = self(ctx, src_lengths)
        # [batch, topk]
        log_probs, diffs = discretized_mix_logistic_topk(mu, log_scale, logit_probs,
                                                         self.range, bin_size, lower, upper, topk=topk)
        lengths = (diffs + src_lengths.unsqueeze(1)).clamp(min=self.length_unit)
        res = lengths.fmod(self.length_unit)
        padding = (self.length_unit - res).fmod(self.length_unit)
        lengths = lengths + padding
        return log_probs, lengths

    @classmethod
    def from_params(cls, params: Dict) -> 'DiffDiscreteMixLogisticLengthPredictor':
        return DiffDiscreteMixLogisticLengthPredictor(**params)


DiffDiscreteMixLogisticLengthPredictor.register('diff_logistic')
