import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    For training, the loss is smoothed with parameter eps, while for evaluation, the smoothing is disabled.
    """
    def __init__(self, label_smoothing):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.eps = label_smoothing

    def forward(self, input, target):
        # [batch, c, d1, ..., dk]
        loss = F.log_softmax(input, dim=1) * -1.
        # [batch, d1, ..., dk]
        nll_loss = loss.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        if self.training:
            # [batch, c, d1, ..., dk]
            inf_mask = loss.eq(float('inf'))
            # [batch, d1, ..., dk]
            smooth_loss = loss.masked_fill(inf_mask, 0.).sum(dim=1)
            eps_i = self.eps / (1.0 - inf_mask.float()).sum(dim=1)
            return nll_loss * (1. - self.eps) + smooth_loss * eps_i
        else:
            return nll_loss
