from overrides import overrides
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from flownmt.nnet.weightnorm import LinearWeightNorm
from flownmt.modules.posteriors.posterior import Posterior
from flownmt.nnet.attention import GlobalAttention


class ShiftRecurrentCore(nn.Module):
    def __init__(self, embed, rnn_mode, num_layers, latent_dim, hidden_size, bidirectional=True, use_attn=False, dropout=0.0, dropword=0.0):
        super(ShiftRecurrentCore, self).__init__()
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        assert hidden_size % 2 == 0
        self.tgt_embed = embed
        assert num_layers == 1
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = RNN(embed.embedding_dim, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.rnn = RNN(embed.embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.use_attn = use_attn
        if use_attn:
            self.attn = GlobalAttention(latent_dim, hidden_size, hidden_size)
            self.ctx_proj = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ELU())
        else:
            self.ctx_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ELU())
        self.dropout = dropout
        self.dropout2d = nn.Dropout2d(dropword) if dropword > 0. else None # drop entire tokens
        self.mu = LinearWeightNorm(hidden_size, latent_dim, bias=True)
        self.logvar = LinearWeightNorm(hidden_size, latent_dim, bias=True)

    @overrides
    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        tgt_embed = self.tgt_embed(tgt_sents)
        if self.dropout2d is not None:
            tgt_embed = self.dropout2d(tgt_embed)
        lengths = tgt_masks.sum(dim=1).long()
        packed_embed = pack_padded_sequence(tgt_embed, lengths, batch_first=True, enforce_sorted=False)
        packed_enc, _ = self.rnn(packed_embed)
        tgt_enc, _ = pad_packed_sequence(packed_enc, batch_first=True, total_length=tgt_masks.size(1))

        if self.bidirectional:
            # split into fwd and bwd
            fwd_tgt_enc, bwd_tgt_enc = tgt_enc.chunk(2, dim=2) # (batch_size, seq_len, hidden_size // 2)
            pad_vector = fwd_tgt_enc.new_zeros((fwd_tgt_enc.size(0), 1, fwd_tgt_enc.size(2)))
            pad_fwd_tgt_enc = torch.cat([pad_vector, fwd_tgt_enc], dim=1)
            pad_bwd_tgt_enc = torch.cat([bwd_tgt_enc, pad_vector], dim=1)
            tgt_enc = torch.cat([pad_fwd_tgt_enc[:, :-1], pad_bwd_tgt_enc[:, 1:]], dim=2)
        else:
            pad_vector = tgt_enc.new_zeros((tgt_enc.size(0), 1, tgt_enc.size(2)))
            tgt_enc = torch.cat([pad_vector, tgt_enc], dim=1)[:, :-1]

        if self.use_attn:
            ctx = self.attn(tgt_enc, src_enc, key_mask=src_masks.eq(0))
            ctx = torch.cat([tgt_enc, ctx], dim=2)
        else:
            ctx = tgt_enc
        ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
        mu = self.mu(ctx) * tgt_masks.unsqueeze(2)
        logvar = self.logvar(ctx) * tgt_masks.unsqueeze(2)
        return mu, logvar

    def init(self, tgt_sents, tgt_masks, src_enc, src_masks, init_scale=1.0, init_mu=True, init_var=True):
        with torch.no_grad():
            tgt_embed = self.tgt_embed(tgt_sents)
            if self.dropout2d is not None:
                tgt_embed = self.dropout2d(tgt_embed)
            lengths = tgt_masks.sum(dim=1).long()
            packed_embed = pack_padded_sequence(tgt_embed, lengths, batch_first=True, enforce_sorted=False)
            packed_enc, _ = self.rnn(packed_embed)
            tgt_enc, _ = pad_packed_sequence(packed_enc, batch_first=True, total_length=tgt_masks.size(1))

            if self.bidirectional:
                fwd_tgt_enc, bwd_tgt_enc = tgt_enc.chunk(2, dim=2)  # (batch_size, seq_len, hidden_size // 2)
                pad_vector = fwd_tgt_enc.new_zeros((fwd_tgt_enc.size(0), 1, fwd_tgt_enc.size(2)))
                pad_fwd_tgt_enc = torch.cat([pad_vector, fwd_tgt_enc], dim=1)
                pad_bwd_tgt_enc = torch.cat([bwd_tgt_enc, pad_vector], dim=1)
                tgt_enc = torch.cat([pad_fwd_tgt_enc[:, :-1], pad_bwd_tgt_enc[:, 1:]], dim=2)
            else:
                pad_vector = tgt_enc.new_zeros((tgt_enc.size(0), 1, tgt_enc.size(2)))
                tgt_enc = torch.cat([pad_vector, tgt_enc], dim=1)[:, :-1]

            if self.use_attn:
                ctx = self.attn.init(tgt_enc, src_enc, key_mask=src_masks.eq(0), init_scale=init_scale)
                ctx = torch.cat([tgt_enc, ctx], dim=2)
            else:
                ctx = tgt_enc
            ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
            mu = self.mu.init(ctx, init_scale=0.05 * init_scale) if init_mu else self.mu(ctx)
            logvar = self.logvar.init(ctx, init_scale=0.05 * init_scale) if init_var else self.logvar(ctx)
            mu = mu * tgt_masks.unsqueeze(2)
            logvar = logvar * tgt_masks.unsqueeze(2)
            return mu, logvar


class ShiftRecurrentPosterior(Posterior):
    """
    Posterior with Recurrent Neural Networks
    """
    def __init__(self, vocab_size, embed_dim, padding_idx, rnn_mode, num_layers, latent_dim, hidden_size,
                 bidirectional=True, use_attn=False, dropout=0.0, dropword=0.0, _shared_embed=None):
        super(ShiftRecurrentPosterior, self).__init__(vocab_size, embed_dim, padding_idx, _shared_embed=_shared_embed)
        self.core = ShiftRecurrentCore(self.tgt_embed, rnn_mode, num_layers, latent_dim, hidden_size,
                                       bidirectional=bidirectional, use_attn=use_attn, dropout=dropout, dropword=dropword)

    def target_embed_weight(self):
        if isinstance(self.core, nn.DataParallel):
            return self.core.module.tgt_embedd.weight
        else:
            return self.core.tgt_embed.weight

    @overrides
    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        return self.core(tgt_sents, tgt_masks, src_enc, src_masks)

    @overrides
    def sample(self, tgt_sents: torch.Tensor, tgt_masks: torch.Tensor,
               src_enc: torch.Tensor, src_masks: torch.Tensor,
               nsamples: int =1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.core(tgt_sents, tgt_masks, src_enc, src_masks)
        z, eps = Posterior.reparameterize(mu, logvar, tgt_masks, nsamples=nsamples, random=random)
        log_probs = Posterior.log_probability(z, eps, mu, logvar, tgt_masks)
        return z, log_probs

    @overrides
    def init(self, tgt_sents, tgt_masks, src_enc, src_masks, init_scale=1.0, init_mu=True, init_var=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.core.init(tgt_sents, tgt_masks, src_enc, src_masks,
                                    init_scale=init_scale, init_mu=init_mu, init_var=init_var)
        z, eps = Posterior.reparameterize(mu, logvar, tgt_masks, random=True)
        log_probs = Posterior.log_probability(z, eps, mu, logvar, tgt_masks)
        z = z.squeeze(1)
        log_probs = log_probs.squeeze(1)
        return z, log_probs

    @classmethod
    def from_params(cls, params: Dict) -> "ShiftRecurrentPosterior":
        return ShiftRecurrentPosterior(**params)


ShiftRecurrentPosterior.register('shift_rnn')
