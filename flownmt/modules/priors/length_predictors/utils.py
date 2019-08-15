from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def discretized_mix_logistic_loss(x, means, logscales, logit_probs,
                                  bin_size, lower, upper) -> torch.Tensor:
    """
    loss for discretized mixture logistic distribution
    Args:
        x: [batch, ]
        means: [batch, nmix]
        logscales: [batch, nmix]
        logit_probs:, [batch, nmix]
        bin_size: float
            The segment for cdf is [x-binsize, x+binsize]
        lower: float
        upper: float
    Returns:
        loss [batch]
    """
    eps = 1e-12
    # [batch, 1]
    x = x.unsqueeze(1)
    # [batch, nmix]
    centered_x = x - means
    if isinstance(logscales, float):
        inv_stdv = np.exp(-logscales)
    else:
        inv_stdv = torch.exp(-logscales)

    # [batch, nmix]
    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    # [batch, nmix]
    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)
    # lower < x < upper
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta + eps)
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    # x < lower
    log_cdf_low = plus_in - F.softplus(plus_in)

    # x > upper
    log_cdf_up = -F.softplus(min_in)

    # [batch, nmix]
    log_cdf = torch.where(cdf_delta.gt(1e-5), log_cdf_mid, log_cdf_approx)
    log_cdf = torch.where(x.ge(lower), log_cdf, log_cdf_low)
    log_cdf = torch.where(x.le(upper), log_cdf, log_cdf_up)

    # [batch]
    loss = torch.logsumexp(log_cdf + logit_probs, dim=1) * -1.
    return loss


def discretized_mix_logistic_topk(means, logscales, logit_probs,
                                  range, bin_size, lower, upper, topk=1) -> Tuple[torch.Tensor, torch.LongTensor]:
    """
    topk for discretized mixture logistic distribution
    Args:
        means: [batch, nmix]
        logscales: [batch, nmix]
        logit_probs:, [batch, nmix]
        range: int
            range of x
        bin_size: float
            The segment for cdf is [x-binsize, x+binsize]
        lower: float
        upper: float
        topk: int
    Returns: Tensor1, Tensor2
        Tensor1: log probs [batch, topk]
        Tensor2: indexes for top k [batch, topk]

    """
    eps = 1e-12
    # [batch, 1, nmix]
    means = means.unsqueeze(1)
    logscales = logscales.unsqueeze(1)
    logit_probs = logit_probs.unsqueeze(1)
    # [1, 2 * range + 1, 1]
    x = torch.arange(-range, range + 1, 1., device=means.device).unsqueeze(0).unsqueeze(2)
    x = x.div(range)
    # [batch, 2 * range + 1, nmix]
    centered_x = x - means
    if isinstance(logscales, float):
        inv_stdv = np.exp(-logscales)
    else:
        inv_stdv = torch.exp(-logscales)

    # [batch, 2 * range + 1, nmix]
    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    # [batch, 2 * range + 1, nmix]
    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)
    # lower < x < upper
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta + eps)
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    # x < lower
    log_cdf_low = plus_in - F.softplus(plus_in)

    # x > upper
    log_cdf_up = -F.softplus(min_in)

    # [batch, 2 * range + 1, nmix]
    log_cdf = torch.where(cdf_delta.gt(1e-5), log_cdf_mid, log_cdf_approx)
    log_cdf = torch.where(x.ge(lower), log_cdf, log_cdf_low)
    log_cdf = torch.where(x.le(upper), log_cdf, log_cdf_up)
    # [batch, 2 * range + 1]
    log_probs = torch.logsumexp(log_cdf + logit_probs, dim=2)
    log_probs, idx = log_probs.topk(topk, dim=1)

    return log_probs, idx - range
