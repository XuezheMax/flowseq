from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flownmt.modules.priors.length_predictors.predictor import LengthPredictor
from flownmt.nnet.criterion import LabelSmoothedCrossEntropyLoss


class DiffSoftMaxLengthPredictor(LengthPredictor):
    def __init__(self, features, max_src_length, diff_range, dropout=0.0, label_smoothing=0.):
        super(DiffSoftMaxLengthPredictor, self).__init__()
        self.max_src_length = max_src_length
        self.range = diff_range
        self.features = features
        self.dropout = dropout
        self.ctx_proj = None
        self.diff = None
        if label_smoothing < 1e-5:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif 1e-5 < label_smoothing < 1.0:
            self.criterion = LabelSmoothedCrossEntropyLoss(label_smoothing)
        else:
            raise ValueError('label smoothing should be less than 1.0.')

    def set_length_unit(self, length_unit):
        self.length_unit = length_unit
        self.ctx_proj = nn.Sequential(nn.Linear(self.features, self.features), nn.ELU(),
                                      nn.Linear(self.features, self.features), nn.ELU())
        self.diff = nn.Linear(self.features, 2 * self.range + 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.ctx_proj[0].bias, 0.)
        nn.init.constant_(self.ctx_proj[2].bias, 0.)
        nn.init.uniform_(self.diff.weight, -0.1, 0.1)
        nn.init.constant_(self.diff.bias, 0.)

    def forward(self, ctx):
        ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
        return self.diff(ctx)

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
        # [batch]
        src_lengths = src_mask.sum(dim=1).long()
        tgt_lengths = tgt_mask.sum(dim=1).long()
        # [batch, 2 * range + 1]
        logits = self(ctx)
        # [1, 2 * range + 1]
        mask = torch.arange(0, logits.size(1), device=logits.device).unsqueeze(0)
        # [batch, 2 * range + 1]
        mask = (mask + src_lengths.unsqueeze(1) - self.range).fmod(self.length_unit).ne(0)
        logits = logits.masked_fill(mask, float('-inf'))

        # handle tgt < src - range
        x = (tgt_lengths - src_lengths + self.range).clamp(min=0)
        tgt = x + src_lengths - self.range
        res = tgt.fmod(self.length_unit)
        padding = (self.length_unit - res).fmod(self.length_unit)
        tgt = tgt + padding
        # handle tgt > src + range
        x = (tgt - src_lengths + self.range).clamp(max=2 * self.range)
        tgt = x + src_lengths - self.range
        tgt = tgt - tgt.fmod(self.length_unit)

        x = tgt - src_lengths + self.range
        loss_length = self.criterion(logits, x)
        return loss_length

    @overrides
    def predict(self, ctx: torch.Tensor, src_mask:torch.Tensor, topk: int = 1) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Args:
            ctx: Tensor
                tensor [batch, features]
            src_mask: Tensor
                tensor for source mask [batch, src_length]
            topk: int (default 1)
                return top k length candidates for each src sentence
        Returns: LongTensor1, Tensor2
            LongTensor1: tensor for lengths [batch, topk]
            Tensor2: log probs for each length
        """
        # [batch]
        src_lengths = src_mask.sum(dim=1).long()
        # [batch, 2 * range + 1]
        logits = self(ctx)
        # [1, 2 * range + 1]
        x = torch.arange(0, logits.size(1), device=logits.device).unsqueeze(0)
        # [batch, 2 * range + 1]
        tgt = x + src_lengths.unsqueeze(1) - self.range
        mask = tgt.fmod(self.length_unit).ne(0)
        logits = logits.masked_fill(mask, float('-inf'))
        # [batch, 2 * range + 1]
        log_probs = F.log_softmax(logits, dim=1)
        # handle tgt length <= 0
        mask = tgt.le(0)
        log_probs = log_probs.masked_fill(mask, float('-inf'))
        # [batch, topk]
        log_probs, x = log_probs.topk(topk, dim=1)
        lengths = x + src_lengths.unsqueeze(1) - self.range
        return lengths, log_probs

    @classmethod
    def from_params(cls, params: Dict) -> 'DiffSoftMaxLengthPredictor':
        return DiffSoftMaxLengthPredictor(**params)


DiffSoftMaxLengthPredictor.register('diff_softmax')
