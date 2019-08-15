import logging
import sys
from typing import Tuple, List
import torch
from torch._six import inf


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def norm(p: torch.Tensor, dim: int):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return norm(p.transpose(0, dim), 0).transpose(0, dim)


def exponentialMovingAverage(original, shadow, decay_rate, init=False):
    params = dict()
    for name, param in shadow.named_parameters():
        params[name] = param
    for name, param in original.named_parameters():
        shadow_param = params[name]
        if init:
            shadow_param.data.copy_(param.data)
        else:
            shadow_param.data.add_((1 - decay_rate) * (param.data - shadow_param.data))


def logPlusOne(x):
    """
    compute log(x + 1) for small x
    Args:
        x: Tensor
    Returns: Tensor
        log(x+1)
    """
    eps = 1e-4
    mask = x.abs().le(eps).type_as(x)
    return x.mul(x.mul(-0.5) + 1.0) * mask + (x + 1.0).log() * (1.0 - mask)


def gate(x1, x2):
    return x1 * x2.sigmoid_()


def total_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


def squeeze(x: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        mask: Tensor
            mask tensor [batch, length]
        factor: int
            squeeze factor (default 2)
    Returns: Tensor1, Tensor2
        squeezed x [batch, length // factor, factor * features]
        squeezed mask [batch, length // factor]
    """
    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert length % factor == 0
    # [batch, length // factor, factor * features]
    x = x.contiguous().view(batch, length // factor, factor * features)
    mask = mask.view(batch, length // factor, factor).sum(dim=2).clamp(max=1.0)
    return x, mask


def unsqueeze(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        factor: int
            unsqueeze factor (default 2)
    Returns: Tensor
        squeezed tensor [batch, length * 2, features // 2]
    """
    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert features % factor == 0
    # [batch, length * factor, features // factor]
    x = x.view(batch, length * factor, features // factor)
    return x


def split(x: torch.Tensor, z1_features) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        z1_features: int
            the number of features of z1
    Returns: Tensor, Tensor
        split tensors [batch, length, z1_features], [batch, length, features-z1_features]
    """
    z1 = x[:, :, :z1_features]
    z2 = x[:, :, z1_features:]
    return z1, z2


def unsplit(xs: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        xs: List[Tensor]
            tensors to be combined
    Returns: Tensor
        combined tensor
    """
    return torch.cat(xs, dim=2)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask


# def prepare_rnn_seq(rnn_input, lengths, batch_first=False):
#     '''
#     Args:
#         rnn_input: [seq_len, batch, input_size]: tensor containing the features of the input sequence.
#         lengths: [batch]: tensor containing the lengthes of the input sequence
#         batch_first: If True, then the input and output tensors are provided as [batch, seq_len, feature].
#     Returns:
#     '''
#
#     def check_decreasing(lengths):
#         lens, order = torch.sort(lengths, dim=0, descending=True)
#         if torch.ne(lens, lengths).sum() == 0:
#             return None
#         else:
#             _, rev_order = torch.sort(order)
#             return lens, order, rev_order
#
#     check_res = check_decreasing(lengths)
#
#     if check_res is None:
#         lens = lengths
#         rev_order = None
#     else:
#         lens, order, rev_order = check_res
#         batch_dim = 0 if batch_first else 1
#         rnn_input = rnn_input.index_select(batch_dim, order)
#     lens = lens.tolist()
#     seq = pack_padded_sequence(rnn_input, lens, batch_first=batch_first)
#     return seq, rev_order
#
# def recover_rnn_seq(seq, rev_order, batch_first=False, total_length=None):
#     output, _ = pad_packed_sequence(seq, batch_first=batch_first, total_length=total_length)
#     if rev_order is not None:
#         batch_dim = 0 if batch_first else 1
#         output = output.index_select(batch_dim, rev_order)
#     return output
#
#
# def recover_order(tensors, rev_order):
#     if rev_order is None:
#         return tensors
#     recovered_tensors = [tensor.index_select(0, rev_order) for tensor in tensors]
#     return recovered_tensors
#
#
# def decreasing_order(lengths, tensors):
#     def check_decreasing(lengths):
#         lens, order = torch.sort(lengths, dim=0, descending=True)
#         if torch.ne(lens, lengths).sum() == 0:
#             return None
#         else:
#             _, rev_order = torch.sort(order)
#             return lens, order, rev_order
#
#     check_res = check_decreasing(lengths)
#
#     if check_res is None:
#         lens = lengths
#         rev_order = None
#         ordered_tensors = tensors
#     else:
#         lens, order, rev_order = check_res
#         ordered_tensors = [tensor.index_select(0, order) for tensor in tensors]
#
#     return lens, ordered_tensors, rev_order
