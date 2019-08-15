from typing import Dict, Tuple
import torch
import torch.nn as nn


class LengthPredictor(nn.Module):
    """
    Length Predictor
    """
    _registry = dict()

    def __init__(self):
        super(LengthPredictor, self).__init__()
        self.length_unit = None

    def set_length_unit(self, length_unit):
        self.length_unit = length_unit

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
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        LengthPredictor._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return LengthPredictor._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
