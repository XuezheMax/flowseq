import os
import json
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from apex.parallel.distributed import flat_dist_call

from flownmt.modules import Encoder
from flownmt.modules import Posterior
from flownmt.modules import Decoder
from flownmt.modules import Prior


class FlowNMTCore(nn.Module):
    """
    core module for flow nmt model
    """

    def __init__(self, encoder: Encoder, prior: Prior, posterior: Posterior, decoder: Decoder):
        super(FlowNMTCore, self).__init__()
        self.encoder = encoder
        self.prior = prior
        self.posterior = posterior
        self.decoder = decoder

    def sync(self):
        self.prior.sync()

    def init(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        src_enc = self.encoder.init(src_sents, masks=src_masks, init_scale=init_scale)
        z, _ = self.posterior.init(tgt_sents, tgt_masks, src_enc, src_masks,
                                   init_scale=init_scale, init_mu=True, init_var=True)
        self.prior.init(z, tgt_masks, src_enc, src_masks, init_scale=init_scale)
        self.decoder.init(z, tgt_masks, src_enc, src_masks, init_scale=init_scale)

    def init_posterior(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        src_enc = self.encoder.init(src_sents, masks=src_masks, init_scale=init_scale)
        z, _ = self.posterior.init(tgt_sents, tgt_masks, src_enc, src_masks,
                                   init_scale=init_scale, init_mu=True, init_var=False)
        self.decoder.init(z, tgt_masks, src_enc, src_masks, init_scale=init_scale)

    def init_prior(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        with torch.no_grad():
            src_enc, _ = self.encoder(src_sents, masks=src_masks)
        z, _ = self.posterior.init(tgt_sents, tgt_masks, src_enc, src_masks,
                                   init_scale=init_scale, init_mu=False, init_var=True)
        self.prior.init(z.squeeze(1), tgt_masks, src_enc, src_masks, init_scale=init_scale)

    def sample_from_prior(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                          nlengths: int = 1, nsamples: int = 1, tau: float = 0.0,
                          include_zero=False) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        sampling from prior distribution
        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            nlengths: int (default 1)
                number of length candidates
            nsamples: int (default 1)
                number of samples per src per length candidate
            tau: float (default 0.0)
                temperature

        Returns: (Tensor1, Tensor2, Tensor3), (Tensor4, Tensor5), (Tensor6, Tensor7)
            Tensor1: samples from the prior [batch * nlengths * nsamples, tgt_length, nz]
            Tensor2: log probabilities [batch * nlengths * nsamples]
            Tensor3: target masks [batch * nlengths * nsamples, tgt_length]
            Tensor4: lengths [batch * nlengths]
            Tensor5: log probabilities of lengths [batch * nlengths]
            Tensor6: source encoding with shape [batch * nlengths * nsamples, src_length, hidden_size]
            Tensor7: tensor for global state [batch * nlengths * nsamples, hidden_size]
            Tensor8: source masks with shape [batch * nlengths * nsamples, src_length]

        """
        src_enc, ctx = self.encoder(src_sents, masks=src_masks)
        # [batch, nsamples, tgt_length, nz]
        return self.prior.sample(nlengths, nsamples, src_enc, ctx, src_masks, tau=tau,
                                 include_zero=include_zero)

    def sample_from_posterior(self, tgt_sents: torch, tgt_masks: torch.Tensor,
                              src_enc: torch.Tensor, src_masks: torch.Tensor,
                              nsamples: int = 1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sampling from posterior distribution
        Args:
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            tgt_masks: Tensor [batch, tgt_length]
                tensor for target masks
            src_enc: Tensor [batch, src_length, hidden_size]
                tensor for source encoding
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            nsamples: int
                number of samples
            random: bool
                if True, perform random sampling. Otherwise, return mean.

        Returns: Tensor1, Tensor2
            Tensor1: samples from the posterior [batch, nsamples, tgt_length, nz]
            Tensor2: log probabilities [batch, nsamples]

        """
        return self.posterior.sample(tgt_sents, tgt_masks, src_enc, src_masks, nsamples=nsamples, random=random)

    def reconstruct(self, src_sents: torch.Tensor, tgt_sents: torch.Tensor,
                    src_masks: torch.Tensor, tgt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_enc, ctx = self.encoder(src_sents, masks=src_masks)
        z, _ = self.sample_from_posterior(tgt_sents, tgt_masks, src_enc, src_masks, random=False)
        z = z.squeeze(1)
        recon, _ = self.decoder.decode(z, tgt_masks, src_enc, src_masks)
        recon_err = self.decoder.loss(z, tgt_sents, tgt_masks, src_enc, src_masks)
        loss_length = self.prior.length_loss(ctx, src_masks, tgt_masks)
        lengths, log_probs = self.prior.predict_length(ctx, src_masks, topk=1)
        return recon, recon_err, loss_length, lengths.squeeze(1), log_probs.squeeze(1) * -1.

    def translate_argmax(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                         n_tr: int = 1, tau: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch, tgt_length]
            Tensor2: lengths [batch]

        """
        batch = src_sents.size(0)
        # [batch * n_tr, tgt_length, nz]
        (z, log_probs, tgt_masks), (lengths, _), (src, _, _) = self.sample_from_prior(src_sents, src_masks, nlengths=1, nsamples=n_tr, tau=tau)
        if n_tr > 1:
            nbatch, length, nz = z.size()
            # [batch, n_tr, tgt_length, nz]
            z = z.view(batch, n_tr, length, nz)
            # [batch, n_tr]
            log_probs = log_probs.view(batch, n_tr)
            # [batch, n_tr, tgt_length]
            tgt_masks = tgt_masks.view(batch, n_tr, length)
            # [batch, n_tr, src_length, dim]
            src = src.view(batch, n_tr, *src.size()[1:])
            # [batch]
            idx = log_probs.argmax(dim=1)
            batch_idx = torch.arange(0, batch).long().to(idx.device)
            # [batch, tgt_length, nz]
            z = z[batch_idx, idx]
            # [batch, tgt_length]
            tgt_masks = tgt_masks[batch_idx, idx]
            # [batch, src_length, n_tr]
            src = src[:, 0]

        # [batch, tgt_length]
        trans, _ = self.decoder.decode(z, tgt_masks, src, src_masks)
        return trans, lengths

    def translate_iw(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                     n_len: int = 1, n_tr: int = 1,
                     tau: float = 0.0, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_len: int (default 1)
                number of length candidates
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature
            k: int (default 1)
                number of samples for importance weighted sampling

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch, tgt_length]
            Tensor2: lengths [batch]

        """
        batch = src_sents.size(0)
        # [batch * n_len * n_tr, tgt_length, nz]
        (z, _, tgt_masks), \
        (lengths, log_probs_length), \
        (src, ctx, src_masks) = self.sample_from_prior(src_sents, src_masks,
                                                       nlengths=n_len, nsamples=n_tr,
                                                       tau=tau, include_zero=True)
        # [batch, n_len]
        lengths = lengths.view(batch, n_len)
        log_probs_length = log_probs_length.view(batch, n_len)
        # [batch * n_len * n_tr, tgt_length]
        trans, _ = self.decoder.decode(z, tgt_masks, src, src_masks)
        # [batch, n_len * n_tr, tgt_length]
        trans_org = trans.view(batch, n_len * n_tr, trans.size(1))

        # [batch * n_len * n_tr, k, tgt_length, nz]
        z, log_probs_posterior = self.sample_from_posterior(trans, tgt_masks, src, src_masks, nsamples=k, random=True)
        nbatch, _, length, nz = z.size()
        if k > 1:
            # [batch * n_len * n_tr, k, src_length, hidden_size]
            src = src.unsqueeze(1) + src.new_zeros(nbatch, k, *src.size()[1:])
            # [batch * n_len * n_tr * k, src_length, hidden_size]
            src = src.view(nbatch * k, *src.size()[2:])
            # [batch * n_len * n_tr, k, hidden_size]
            ctx = ctx.unsqueeze(1) + ctx.new_zeros(nbatch, k, ctx.size(1))
            # [batch * n_len * n_tr * k, hidden_size]
            ctx = ctx.view(nbatch * k, ctx.size(2))
            # [batch * n_len * n_tr, k, src_length]
            src_masks = src_masks.unsqueeze(1) + src_masks.new_zeros(nbatch, k, src_masks.size(1))
            # [batch * n_len * n_tr * k, src_length]
            src_masks = src_masks.view(nbatch * k, src_masks.size(2))
            # [batch * n_len * n_tr, k, tgt_length]
            tgt_masks = tgt_masks.unsqueeze(1) + tgt_masks.new_zeros(nbatch, k, tgt_masks.size(1))
            # [batch * n_len * n_tr * k, src_length]
            tgt_masks = tgt_masks.view(nbatch * k, tgt_masks.size(2))
            # [batch * n_len * n_tr, k, tgt_length]
            trans = trans.unsqueeze(1) + trans.new_zeros(nbatch, k, trans.size(1))
            # [batch * n_len * n_tr * k, tgt_length]
            trans = trans.view(nbatch * k, trans.size(2))

        # [batch * n_len * n_tr * k, tgt_length, nz]
        z = z.view(-1, length, nz)
        # [batch * n_len * n_tr * k]
        log_probs_prior, _ = self.prior.log_probability(z, tgt_masks, src, ctx, src_masks, length_loss=False)
        # [batch * n_len * n_tr, k]
        log_probs_prior = log_probs_prior.view(nbatch, k)
        minus_log_prob_decode = self.decoder.loss(z, trans, tgt_masks, src, src_masks).view(nbatch, k)
        log_iw = log_probs_prior - minus_log_prob_decode - log_probs_posterior
        # [batch,  n_len, n_tr]
        nlprobs = math.log(k) - torch.logsumexp(log_iw, dim=1).view(batch, n_len, n_tr)
        # [batch, n_len, n_tr]
        nlprobs = nlprobs - log_probs_length.unsqueeze(2)
        nlprobs = nlprobs / lengths.unsqueeze(2).float()

        idx = nlprobs.view(batch, -1).argmin(dim=1)
        batch_idx = torch.arange(0, batch).long().to(idx.device)

        trans = trans_org[batch_idx, idx]
        lengths = lengths[batch_idx, idx.div(n_tr)]
        return trans, lengths

    def translate_sample(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                         n_len: int = 1, n_tr: int = 1, tau: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_len: int (default 1)
                number of length candidates
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch * n_len * n_tr, tgt_length]
            Tensor2: lengths [batch * n_len * n_tr]

        """
        batch = src_sents.size(0)
        # [batch * n_len * n_tr, tgt_length, nz]
        (z, _, tgt_masks), \
        (lengths, _), \
        (src, _, src_masks) = self.sample_from_prior(src_sents, src_masks,
                                                     nlengths=n_len, nsamples=n_tr,
                                                     tau=tau, include_zero=False)
        # [batch * n_len * n_tr, tgt_length]
        trans, _ = self.decoder.decode(z, tgt_masks, src, src_masks)
        # [batch, n_len]
        lengths = lengths.view(batch, n_len, 1).expand(batch, n_len, n_tr).contiguous()
        lengths = lengths.view(batch * n_len * n_tr)
        return trans, lengths

    def reconstruct_loss(self, src_sents: torch.Tensor, tgt_sents: torch,
                          src_masks: torch.Tensor, tgt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            tgt_masks: Tensor [batch, tgt_length] or None
                tensor for target masks

        Returns: Tensor1, Tensor2
            Tensor1: reconstruction error [batch]
            Tensor2: length loss [batch]

        """
        src_enc, ctx = self.encoder(src_sents, masks=src_masks)
        z, _ = self.sample_from_posterior(tgt_sents, tgt_masks, src_enc, src_masks, random=False)
        # [batch, tgt_length, nz]
        z = z.squeeze(1)

        loss_length = self.prior.length_loss(ctx, src_masks, tgt_masks)
        recon_err = self.decoder.loss(z, tgt_sents, tgt_masks, src_enc, src_masks)
        return recon_err, loss_length

    def translate_loss(self, src_sents: torch.Tensor, tgt_sents: torch,
                       src_masks: torch.Tensor, tgt_masks: torch.Tensor,
                       nsamples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            tgt_masks: Tensor [batch, tgt_length] or None
                tensor for target masks
            nsamples: int
                number of samples

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: reconstruction error [batch]
            Tensor2: KL [batch]
            Tensor3: length loss [batch]

        """
        src_enc, ctx = self.encoder(src_sents, masks=src_masks)
        z, log_probs_posterior = self.sample_from_posterior(tgt_sents, tgt_masks, src_enc, src_masks,
                                                            nsamples=nsamples, random=True)
        batch, _, length, nz = z.size()
        if nsamples > 1:
            # [batch, nsamples, src_length, hidden_size]
            src_enc = src_enc.unsqueeze(1) + src_enc.new_zeros(batch, nsamples, *src_enc.size()[1:])
            # [batch * nsamples, src_length, hidden_size]
            src_enc = src_enc.view(batch * nsamples, *src_enc.size()[2:])
            # [batch, nsamples, hidden_size]
            ctx = ctx.unsqueeze(1) + ctx.new_zeros(batch, nsamples, ctx.size(1))
            ctx = ctx.view(batch * nsamples, ctx.size(2))
            # [batch, nsamples, src_length]
            src_masks = src_masks.unsqueeze(1) + src_masks.new_zeros(batch, nsamples, src_masks.size(1))
            # [batch * nsamples, src_length]
            src_masks = src_masks.view(batch * nsamples, src_masks.size(2))
            # [batch, nsamples, tgt_length]
            tgt_masks = tgt_masks.unsqueeze(1) + tgt_masks.new_zeros(batch, nsamples, tgt_masks.size(1))
            # [batch * nsamples, src_length]
            tgt_masks = tgt_masks.view(batch * nsamples, tgt_masks.size(2))
            # [batch, nsamples, tgt_length]
            tgt_sents = tgt_sents.unsqueeze(1) + tgt_sents.new_zeros(batch, nsamples, tgt_sents.size(1))
            tgt_sents = tgt_sents.view(batch * nsamples, tgt_sents.size(2))
        # [batch * nsamples, tgt_length, nz]
        z = z.view(-1, length, nz)
        # [batch * nsamples] -> [batch, nsamples]
        log_probs_prior, loss_length = self.prior.log_probability(z, tgt_masks, src_enc, ctx, src_masks, length_loss=True)
        log_probs_prior = log_probs_prior.view(batch, nsamples)
        loss_length = loss_length.view(batch, nsamples)
        # [batch]
        KL = (log_probs_posterior - log_probs_prior).mean(dim=1)
        loss_length = loss_length.mean(dim=1)
        # [batch * nsamples] -> [batch, nsamples] -> [batch]
        recon_err = self.decoder.loss(z, tgt_sents, tgt_masks, src_enc, src_masks).view(batch, nsamples).mean(dim=1)
        return recon_err, KL, loss_length

    def forward(self, src_sents: torch.Tensor, tgt_sents: torch, src_masks: torch.Tensor, tgt_masks: torch.Tensor,
                nsamples: int = 1, only_recon_loss=False):
        if only_recon_loss:
            return self.reconstruct_loss(src_sents, tgt_sents, src_masks, tgt_masks)
        else:
            return self.translate_loss(src_sents, tgt_sents, src_masks, tgt_masks, nsamples=nsamples)


class FlowNMT(nn.Module):
    """
    NMT model with Generative Flow.
    """

    def __init__(self, core: FlowNMTCore):
        super(FlowNMT, self).__init__()
        self.core = core
        self.length_unit = self.core.prior.length_unit
        self.distribured_enabled = False

    def _get_core(self):
        return self.core.module if self.distribured_enabled else self.core

    def sync(self):
        core = self._get_core()
        core.prior.sync()

    def init(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        core = self._get_core()
        core.init(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=init_scale)

    def init_posterior(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        core = self._get_core()
        core.init_posterior(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=init_scale)

    def init_prior(self, src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0):
        core = self._get_core()
        core.init_prior(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=init_scale)

    def reconstruct(self, src_sents: torch.Tensor, tgt_sents: torch.Tensor,
                    src_masks: torch.Tensor, tgt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._get_core().reconstruct(src_sents, tgt_sents, src_masks, tgt_masks)

    def translate_argmax(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                         n_tr: int = 1, tau: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch, tgt_length]
            Tensor2: lengths [batch]

        """
        return self._get_core().translate_argmax(src_sents, src_masks, n_tr=n_tr, tau=tau)

    def translate_iw(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                     n_len: int = 1, n_tr: int = 1,
                     tau: float = 0.0, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_len: int (default 1)
                number of length candidates
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature
            k: int (default 1)
                number of samples for importance weighted sampling

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch, tgt_length]
            Tensor2: lengths [batch]

        """
        return self._get_core().translate_iw(src_sents, src_masks, n_len=n_len, n_tr=n_tr,
                                             tau=tau, k=k)

    def translate_sample(self, src_sents: torch.Tensor, src_masks: torch.Tensor,
                         n_len: int = 1, n_tr: int = 1, tau: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            n_len: int (default 1)
                number of length candidates
            n_tr: int (default 1)
                number of translations per sentence per length candidate
            tau: float (default 0.0)
                temperature

        Returns: Tensor1, Tensor2
            Tensor1: tensor for translations [batch * n_len * n_tr, tgt_length]
            Tensor2: lengths [batch * n_len * n_tr]

        """
        return self._get_core().translate_sample(src_sents, src_masks, n_len=n_len, n_tr=n_tr, tau=tau)

    def reconstruct_error(self, src_sents: torch.Tensor, tgt_sents: torch,
                          src_masks: torch.Tensor, tgt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            tgt_masks: Tensor [batch, tgt_length] or None
                tensor for target masks

        Returns: Tensor1, Tensor2
            Tensor1: reconstruction error [batch]
            Tensor2: length loss [batch]

        """
        return self.core(src_sents, tgt_sents, src_masks, tgt_masks, only_recon_loss=True)

    def loss(self, src_sents: torch.Tensor, tgt_sents: torch,
             src_masks: torch.Tensor, tgt_masks: torch.Tensor,
             nsamples: int = 1, eval=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            src_sents: Tensor [batch, src_length]
                tensor for source sentences
            tgt_sents: Tensor [batch, tgt_length]
                tensor for target sentences
            src_masks: Tensor [batch, src_length] or None
                tensor for source masks
            tgt_masks: Tensor [batch, tgt_length] or None
                tensor for target masks
            nsamples: int
                number of samples
            eval: bool
                if eval, turn off distributed mode

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: reconstruction error [batch]
            Tensor2: KL [batch]
            Tensor3: length loss [batch]

        """
        core = self._get_core() if eval else self.core
        return core(src_sents, tgt_sents, src_masks, tgt_masks, nsamples=nsamples)

    def init_distributed(self, rank, local_rank):
        assert not self.distribured_enabled
        self.distribured_enabled = True
        print("Initializing Distributed, rank {}, local rank {}".format(rank, local_rank))
        dist.init_process_group(backend='nccl', rank=rank)
        torch.cuda.set_device(local_rank)
        self.core = DistributedDataParallel(self.core)

    def sync_params(self):
        assert self.distribured_enabled
        core = self._get_core()
        flat_dist_call([param.data for param in core.parameters()], dist.all_reduce)
        self.core.needs_refresh = True

    def enable_allreduce(self):
        assert self.distribured_enabled
        self.core.enable_allreduce()

    def disable_allreduce(self):
        assert self.distribured_enabled
        self.core.disable_allreduce()

    def save(self, model_path):
        model = {'core': self._get_core().state_dict()}
        model_name = os.path.join(model_path, 'model.pt')
        torch.save(model, model_name)

    def save_core(self, path):
        core = self._get_core()
        model = {'prior': core.prior.state_dict(),
                 'encoder': core.encoder.state_dict(),
                 'decoder': core.decoder.state_dict(),
                 'posterior': core.posterior.state_dict()}
        torch.save(model, path)

    def load_core(self, path, device, load_prior=True):
        model = torch.load(path, map_location=device)
        core = self._get_core()
        core.posterior.load_state_dict(model['posterior'])
        core.encoder.load_state_dict(model['encoder'])
        core.decoder.load_state_dict(model['decoder'])
        if load_prior:
            core.prior.load_state_dict(model['prior'])

    @classmethod
    def load(cls, model_path, device):
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        flownmt = FlowNMT.from_params(params).to(device)
        model_name = os.path.join(model_path, 'model.pt')
        model = torch.load(model_name, map_location=device)
        flownmt.core.load_state_dict(model['core'])
        return flownmt

    @classmethod
    def from_params(cls, params: Dict) -> "FlowNMT":
        src_vocab_size = params.pop('src_vocab_size')
        tgt_vocab_size = params.pop('tgt_vocab_size')
        embed_dim = params.pop('embed_dim')
        latent_dim = params.pop('latent_dim')
        hidden_size = params.pop('hidden_size')
        max_src_length = params.pop('max_src_length')
        max_tgt_length = params.pop('max_tgt_length')
        src_pad_idx = params.pop('src_pad_idx')
        tgt_pad_idx = params.pop('tgt_pad_idx')

        share_embed = params.pop('share_embed')
        tie_weights = params.pop('tie_weights')

        # prior
        prior_params = params.pop('prior')
        prior_params['flow']['features'] = latent_dim
        prior_params['flow']['src_features'] = latent_dim
        prior_params['length_predictor']['features'] = latent_dim
        prior_params['length_predictor']['max_src_length'] = max_src_length
        prior = Prior.by_name(prior_params.pop('type')).from_params(prior_params)
        # eocoder
        encoder_params = params.pop('encoder')
        encoder_params['vocab_size'] = src_vocab_size
        encoder_params['embed_dim'] = embed_dim
        encoder_params['padding_idx'] = src_pad_idx
        encoder_params['latent_dim'] = latent_dim
        encoder_params['hidden_size'] = hidden_size
        encoder = Encoder.by_name(encoder_params.pop('type')).from_params(encoder_params)
        # posterior
        posterior_params = params.pop('posterior')
        posterior_params['vocab_size'] = tgt_vocab_size
        posterior_params['embed_dim'] = embed_dim
        posterior_params['padding_idx'] = tgt_pad_idx
        posterior_params['latent_dim'] = latent_dim
        posterior_params['hidden_size'] = hidden_size
        _shared_embed = encoder.embed if share_embed else None
        posterior_params['_shared_embed'] = _shared_embed
        posterior = Posterior.by_name(posterior_params.pop('type')).from_params(posterior_params)
        # decoder
        decoder_params = params.pop('decoder')
        decoder_params['vocab_size'] = tgt_vocab_size
        decoder_params['latent_dim'] = latent_dim
        decoder_params['hidden_size'] = hidden_size
        _shared_weight = posterior.tgt_embed.weight if tie_weights else None
        decoder_params['_shared_weight'] = _shared_weight
        decoder = Decoder.by_name(decoder_params.pop('type')).from_params(decoder_params)

        return FlowNMT(FlowNMTCore(encoder, prior, posterior, decoder))
