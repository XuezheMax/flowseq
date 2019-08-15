import math
import torch
import torch.nn as nn

from flownmt.utils import make_positions


class PositionalEncoding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, encoding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEncoding.get_embedding(
            init_size,
            encoding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_encodings, encoding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = encoding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_encodings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_encodings, -1)
        if encoding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_encodings, 1)], dim=1)
        emb[0, :] = 0
        return emb

    def forward(self, x):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.size()[:2]
        max_pos = seq_len + 1
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = PositionalEncoding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if self.padding_idx is None:
            return self.weights[1:seq_len + 1].detach()
        else:
            positions = make_positions(x, self.padding_idx)
            return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
