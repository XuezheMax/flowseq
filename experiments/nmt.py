import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import random
import math
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from flownmt.data import NMTDataSet, DataIterator
from flownmt import FlowNMT
from flownmt.utils import total_grad_norm
from flownmt.optim import AdamW, InverseSquareRootScheduler, ExponentialScheduler

from experiments.options import parse_args


def logging(info, logfile):
    print(info)
    print(info, file=logfile)
    logfile.flush()


def get_optimizer(learning_rate, parameters, betas, eps, amsgrad, weight_decay, lr_decay, warmup_steps, init_lr):
    optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
    if lr_decay == 'inv_sqrt':
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps, init_lr)
    elif lr_decay == 'expo':
        step_decay = 0.999995
        scheduler = ExponentialScheduler(optimizer, step_decay, warmup_steps, init_lr)
    else:
        raise ValueError('unknown lr decay method: %s' % lr_decay)
    return optimizer, scheduler


def calc_bleu(fref, fmt, result_path):
    script = os.path.join(current_path, 'scripts/multi-bleu.perl')
    temp = os.path.join(result_path, 'tmp')
    os.system("perl %s %s < %s > %s" % (script, fref, fmt, temp))
    bleu = open(temp, 'r').read().strip()
    bleu = bleu.split(",")[0].split("=")
    if len(bleu) < 2:
        return 0.0
    bleu = float(bleu[1].strip())
    return bleu


def translate(epoch, dataset, dataloader, flownmt, result_path, log):
    flownmt.eval()
    taus = [0.0,]
    bleu = 0
    logging('argmax translating...', log)
    for tau in taus:
        n_tr = 8 if tau > 1e-4 else 1
        translations = []
        lengths = []
        length_err = 0
        num_insts = 0
        start_time = time.time()
        for src, tgt, src_masks, tgt_masks in dataloader:
            trans, lens = flownmt.translate_argmax(src, src_masks, n_tr=n_tr, tau=tau)
            translations.append(trans)
            lengths.append(lens)
            length_err += (lens.float() - tgt_masks.sum(dim=1)).abs().sum().item()
            num_insts += src.size(0)

        time_cost = time.time() - start_time
        outfile = os.path.join(result_path, 'trans{}.t{:.1f}.mt'.format(epoch, 0.0))
        dataset.dump_to_file(translations, lengths, outfile)
        b = calc_bleu(dataloader.tgt_sort_origin_path, outfile, result_path)
        logging('#SENT: {}, Tau: {:.1f}, Length Err: {:.1f}, BLEU: {:.2f}, time: {:.1f}s'.format(
            num_insts, tau, length_err / num_insts, b, time_cost), log)
        if bleu < b:
            bleu = b

    taus = []
    if len(taus) > 0:
        logging('importance weighted translating...', log)
    n_len = 3
    iwk = 4
    for tau in taus:
        n_tr = 8 if tau > 1e-4 else 1
        translations = []
        lengths = []
        length_err = 0
        num_insts = 0
        start_time = time.time()
        for src, tgt, src_masks, tgt_masks in dataloader:
            trans, lens = flownmt.translate_iw(src, src_masks, n_len=n_len, n_tr=n_tr, tau=tau, k=iwk)
            translations.append(trans)
            lengths.append(lens)
            length_err += (lens.float() - tgt_masks.sum(dim=1)).abs().sum().item()
            num_insts += src.size(0)

        time_cost = time.time() - start_time
        outfile = os.path.join(result_path, 'trans{}.t{:.1f}.mt'.format(epoch, tau))
        dataset.dump_to_file(translations, lengths, outfile)
        b = calc_bleu(dataloader.tgt_sort_origin_path, outfile, result_path)
        logging('Temperature: {:.1f}, Length Err: {:.1f}, BLEU: {:.2f}, time: {:.1f}s'.format(tau, length_err / num_insts, b, time_cost), log)

        if bleu < b:
            bleu = b

    return bleu


def reconstruct(epoch, dataset, dataloader, flownmt, result_path, log):
    flownmt.eval()
    recons = []
    lengths = []
    recon_loss = 0.
    length_loss = 0.
    length_loss_pred = 0.
    length_err = 0.
    num_insts = 0
    num_words = 0
    start_time = time.time()
    for src, tgt, src_masks, tgt_masks in dataloader:
        recon, recon_err, llen, lens, llen_pred = flownmt.reconstruct(src, tgt, src_masks, tgt_masks)
        recon_loss += recon_err.sum().item()
        length_loss += llen.sum().item()
        length_loss_pred += llen_pred.sum().item()
        length_err += (lens.float() - tgt_masks.sum(dim=1)).abs().sum().item()
        num_insts += src.size(0)
        num_words += tgt_masks.sum().item()
        recons.append(recon)
        lengths.append(tgt_masks.sum(dim=1).long())

    logging('reconstruct time: {:.1f}s'.format(time.time() - start_time), log)
    outfile = os.path.join(result_path, 'reconstruct{}.mt'.format(epoch))
    dataset.dump_to_file(recons, lengths, outfile)
    bleu = calc_bleu(dataloader.tgt_sort_origin_path, outfile, result_path)
    recon_loss_per_word = recon_loss / num_words
    recon_loss = recon_loss / num_insts
    length_loss = length_loss / num_insts
    length_loss_pred = length_loss_pred / num_insts
    length_err = length_err / num_insts
    logging('Reconstruct BLEU: {:.2f}, NLL: {:.2f} ({:.2f}), Length NLL: {:.2f} ({:.2f}), Err: {:.1f}'.format(
        bleu, recon_loss, recon_loss_per_word, length_loss, length_loss_pred, length_err), log)


def eval(args, epoch, dataset, dataloader, flownmt):
    flownmt.eval()
    flownmt.sync()
    # reconstruct
    reconstruct(epoch, dataset, dataloader, flownmt, args.result_path, args.log)
    # translate
    bleu = translate(epoch, dataset, dataloader, flownmt, args.result_path, args.log)
    recon_loss = 0.
    kl_loss = 0.
    length_loss = 0.
    num_insts = 0
    num_words = 0
    test_k = 3
    for src, tgt, src_masks, tgt_masks in dataloader:
        recon, kl, llen = flownmt.loss(src, tgt, src_masks, tgt_masks, nsamples=test_k, eval=True)
        recon_loss += recon.sum().item()
        kl_loss += kl.sum().item()
        length_loss += llen.sum().item()
        num_insts += src.size(0)
        num_words += tgt_masks.sum().item()
    kl_loss = kl_loss / num_insts
    recon_loss = recon_loss / num_insts
    length_loss = length_loss / num_insts
    nll = kl_loss + recon_loss
    ppl = np.exp(nll * num_insts / num_words)
    logging('Ave  NLL: {:.2f} (recon: {:.2f}, kl: {:.2f}), len: {:.2f}, PPL: {:.2f}, BLEU: {:.2f}'.format(
        nll, recon_loss, kl_loss, length_loss, ppl, bleu), args.log)
    logging('-' * 100, args.log)
    return bleu, nll, recon_loss, kl_loss, length_loss, ppl


def setup(args):
    args.cuda = torch.cuda.is_available()
    random_seed = args.seed + args.rank if args.rank >= 0 else args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device('cuda', args.local_rank) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = False

    model_path = args.model_path
    args.checkpoint_name = os.path.join(model_path, 'checkpoint')

    result_path = os.path.join(model_path, 'translations')
    args.result_path = result_path

    vocab_path = os.path.join(model_path, 'vocab')

    data_path = args.data_path
    args.world_size = int(os.environ["WORLD_SIZE"]) if args.rank >=0 else 0
    print("Rank {}".format(args.rank), args)

    if args.rank <= 0:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)
        args.log = open(os.path.join(model_path, 'log.txt'), 'w')

    if args.recover > 0:
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        src_lang = params['src']
        tgt_lang = params['tgt']
        src_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(src_lang))
        tgt_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(tgt_lang))
        src_vocab_size = params['src_vocab_size']
        tgt_vocab_size = params['tgt_vocab_size']

        args.max_src_length = params['max_src_length']
        args.max_tgt_length = params['max_tgt_length']
        dataset = NMTDataSet(data_path, src_lang, tgt_lang,
                             src_vocab_path, tgt_vocab_path,
                             src_vocab_size, tgt_vocab_size,
                             subword=args.subword, create_vocab=False)
        assert src_vocab_size == dataset.src_vocab_size
        assert tgt_vocab_size == dataset.tgt_vocab_size
    else:
        params = json.load(open(args.config, 'r'))
        src_lang = args.src
        tgt_lang = args.tgt
        src_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(src_lang))
        tgt_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(tgt_lang))
        create_vocab = args.create_vocab

        src_max_vocab = params.pop('{}_vocab_size'.format(src_lang))
        tgt_max_vocab = params.pop('{}_vocab_size'.format(tgt_lang))
        args.max_src_length = params.pop('max_{}_length'.format(src_lang))
        args.max_tgt_length = params.pop('max_{}_length'.format(tgt_lang))

        dataset = NMTDataSet(data_path, src_lang, tgt_lang,
                             src_vocab_path, tgt_vocab_path,
                             src_max_vocab, tgt_max_vocab,
                             subword=args.subword, create_vocab=create_vocab)

        params['src'] = src_lang
        params['tgt'] = tgt_lang
        params['src_vocab_size'] = dataset.src_vocab_size
        params['tgt_vocab_size'] = dataset.tgt_vocab_size
        params['max_src_length'] = args.max_src_length
        params['max_tgt_length'] = args.max_tgt_length
        params['src_pad_idx'] = dataset.src_pad_idx
        params['tgt_pad_idx'] = dataset.tgt_pad_idx

        if args.share_all_embeddings:
            assert 'share_embed' not in params or params['share_embed'], 'share embedding args conflicts'
            assert 'tie_weights' not in params or params['tie_weights'], 'tie weights args conflicts'
            params['share_embed'] = True
            params['tie_weights'] = True
        else:
            params.setdefault('share_embed', False)
            params.setdefault('tie_weights', False)

        json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)

    flownmt = FlowNMT.from_params(params)
    flownmt.to(device)
    args.length_unit = flownmt.length_unit
    args.device = device
    args.steps_per_epoch = 1000

    return args, dataset, flownmt


def init_dataloader(args, dataset):
    batch_by_tokens = args.loss_type == 'token'
    train_iter = DataIterator(dataset, args.batch_size, batch_by_tokens, args.max_src_length, args.max_tgt_length,
                              5000, args.device, args.result_path, multi_scale=args.length_unit,
                              corpus="train", bucket_data=args.bucket_batch, rank=args.rank,
                              num_replicas=args.world_size)

    if args.rank <= 0:
        eval_batch = args.eval_batch_size
        val_iter = DataIterator(dataset, eval_batch, batch_by_tokens, args.max_src_length, args.max_tgt_length,
                                1000, args.device, args.result_path, corpus="dev",
                                bucket_data=args.bucket_batch, multi_scale=args.length_unit)
        test_iter = DataIterator(dataset, eval_batch, batch_by_tokens, args.max_src_length, args.max_tgt_length,
                                 1000, args.device, args.result_path, corpus="test",
                                 bucket_data=args.bucket_batch, multi_scale=args.length_unit)
    else:
        val_iter, test_iter = None, None
    return train_iter, val_iter, test_iter


def init_model(args, train_iter, flownmt):
    flownmt.eval()
    init_batch_size = args.init_batch_size
    if args.rank <= 0:
        logging('Rank {}, init model: {} instances'.format(args.rank, init_batch_size), args.log)
    else:
        print('Rank {}, init model: {} instances'.format(args.rank, init_batch_size))
    src_sents, tgt_sents, src_masks, tgt_masks = train_iter.get_batch(init_batch_size)
    if args.rank <= 0:
        logging("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)), args.log)
    else:
        print("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)))
    flownmt.init(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0)


def init_posterior(args, train_iter, flownmt):
    flownmt.eval()
    init_batch_size = args.init_batch_size
    if args.rank <= 0:
        logging('Rank {}, init posterior: {} instances'.format(args.rank, init_batch_size), args.log)
    else:
        print('Rank {}, init posterior: {} instances'.format(args.rank, init_batch_size))
    src_sents, tgt_sents, src_masks, tgt_masks = train_iter.get_batch(init_batch_size)
    if args.rank <= 0:
        logging("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)), args.log)
    else:
        print("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)))
    flownmt.init_posterior(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0)


def init_prior(args, train_iter, flownmt):
    flownmt.eval()
    init_batch_size = args.init_batch_size
    if args.rank <= 0:
        logging('Rank {}, init prior: {} instances'.format(args.rank, init_batch_size), args.log)
    else:
        print('Rank {}, init prior: {} instances'.format(args.rank, init_batch_size))
    src_sents, tgt_sents, src_masks, tgt_masks = train_iter.get_batch(init_batch_size)
    if args.rank <= 0:
        logging("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)), args.log)
    else:
        print("maximum sentence length (src, tgt): {}, {}".format(src_sents.size(1), tgt_sents.size(1)))
    flownmt.init_prior(src_sents, tgt_sents, src_masks, tgt_masks, init_scale=1.0)


def pretrain_model(args, dataset, train_iter, val_iter, flownmt, zero_steps):
    device = args.device
    steps_per_epoch = args.steps_per_epoch
    loss_ty_token = args.loss_type == 'token'
    lr_decay = args.lr_decay
    betas = (args.beta1, args.beta2)
    eps = args.eps
    amsgrad = args.amsgrad
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip

    batch_steps = max(1, args.batch_steps // 2)
    log = args.log if args.rank <=0 else None

    warmup_steps = min(4000, zero_steps)
    optimizer, scheduler = get_optimizer(args.lr, flownmt.parameters(), betas, eps, amsgrad, weight_decay, lr_decay,
                                         warmup_steps, init_lr=1e-7)
    lr = scheduler.get_lr()[0]

    recon_loss = torch.Tensor([0.]).to(device)
    length_loss = torch.Tensor([0.]).to(device)
    num_insts = torch.Tensor([0.]).to(device)
    num_words = torch.Tensor([0.]).to(device)
    num_nans = 0
    num_back = 0

    flownmt.train()
    start_time = time.time()
    if args.rank <= 0:
        logging('Init Epoch: %d, lr=%.6f (%s), betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s, l2=%.1e' % (
            1, lr, lr_decay, betas[0], betas[1], eps, amsgrad, weight_decay), log)

    for step, (src_sents, tgt_sents, src_masks, tgt_masks) in enumerate(train_iter):
        batch_size = src_sents.size(0)
        words = tgt_masks.sum().item()
        recon_batch = 0.
        llen_batch = 0.
        optimizer.zero_grad()
        src_sents = [src_sents, ] if batch_steps == 1 else src_sents.chunk(batch_steps, dim=0)
        tgt_sents = [tgt_sents, ] if batch_steps == 1 else tgt_sents.chunk(batch_steps, dim=0)
        src_masks = [src_masks, ] if batch_steps == 1 else src_masks.chunk(batch_steps, dim=0)
        tgt_masks = [tgt_masks, ] if batch_steps == 1 else tgt_masks.chunk(batch_steps, dim=0)
        # disable allreduce for accumulated gradient.
        if args.rank >= 0:
            flownmt.disable_allreduce()
        for src, tgt, src_mask, tgt_mask in zip(src_sents[:-1], tgt_sents[:-1], src_masks[:-1], tgt_masks[:-1]):
            recon, llen = flownmt.reconstruct_error(src, tgt, src_mask, tgt_mask)
            recon = recon.sum()
            llen = llen.sum()
            if loss_ty_token:
                loss = (recon + llen).div(words)
            else:
                loss = (recon + llen).div(batch_size)
            loss.backward()
            with torch.no_grad():
                recon_batch += recon.item()
                llen_batch += llen.item()

        # enable allreduce for the last step.
        if args.rank >= 0:
            flownmt.enable_allreduce()
        src, tgt, src_mask, tgt_mask = src_sents[-1], tgt_sents[-1], src_masks[-1], tgt_masks[-1]
        recon, llen = flownmt.reconstruct_error(src, tgt, src_mask, tgt_mask)
        recon = recon.sum()
        llen = llen.sum()
        if loss_ty_token:
            loss = (recon + llen).div(words)
        else:
            loss = (recon + llen).div(batch_size)
        loss.backward()
        with torch.no_grad():
            recon_batch += recon.item()
            llen_batch += llen.item()

        if grad_clip > 0:
            grad_norm = clip_grad_norm_(flownmt.parameters(), grad_clip)
        else:
            grad_norm = total_grad_norm(flownmt.parameters())

        if math.isnan(grad_norm):
            num_nans += 1
        else:
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                num_insts += batch_size
                num_words += words
                recon_loss += recon_batch
                length_loss += llen_batch

        if step % 10 == 0:
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 and args.rank <= 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            nums = num_insts.item()
            train_recon = recon_loss.item() / nums if nums > 0 else 0
            recon_per_word = recon_loss.item() / num_words.item() if nums > 0 else 0
            train_llen = length_loss.item() / nums if nums > 0 else 0
            curr_step = step % steps_per_epoch
            curr_lr = scheduler.get_lr()[0]
            log_info = '[{}/{} ({:.0f}%) lr={:.6f} {}] recon: {:.2f} ({:.2f}), len: {:.2f}'.format(
                curr_step, steps_per_epoch, 100. * curr_step / steps_per_epoch, curr_lr, num_nans,
                train_recon, recon_per_word,
                train_llen)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

        if step % steps_per_epoch == 0 and step > 0 or step == zero_steps:
            # new epoch
            epoch = step // steps_per_epoch
            lr = scheduler.get_lr()[0]

            if args.rank >= 0:
                dist.reduce(recon_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(length_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(num_insts, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(num_words, dst=0, op=dist.ReduceOp.SUM)

            if args.rank <= 0:
                nums = num_insts.item()
                train_recon = recon_loss.item() / nums if nums > 0 else 0
                recon_per_word = recon_loss.item() / num_words.item() if nums > 0 else 0
                train_llen = length_loss.item() / nums if nums > 0 else 0

                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                logging('Average recon: {:.2f}, ({:.2f}), len: {:.2f}, time: {:.1f}s'.format(
                    train_recon, recon_per_word, train_llen, time.time() - start_time), log)
                logging('-' * 100, log)
                with torch.no_grad():
                    reconstruct(epoch, dataset, val_iter, flownmt, args.result_path, log)
                    logging('-' * 100, log)

            if step == zero_steps:
                optimizer.zero_grad()
                break

            if args.rank <= 0:
                logging('Init Epoch: %d, lr=%.6f (%s), betas=(%.1f, %.3f), eps=%.1e amsgrad=%s, l2=%.1e' % (
                        epoch + 1, lr, lr_decay, betas[0], betas[1], eps, amsgrad, weight_decay), log)

            recon_loss = torch.Tensor([0.]).to(device)
            length_loss = torch.Tensor([0.]).to(device)
            num_insts = torch.Tensor([0.]).to(device)
            num_words = torch.Tensor([0.]).to(device)
            num_nans = 0
            num_back = 0
            flownmt.train()
            start_time = time.time()


def train(args, dataset, train_iter, val_iter, test_iter, flownmt):
    epochs = args.epochs
    loss_ty_token = args.loss_type == 'token'
    steps_per_epoch = args.steps_per_epoch
    train_k = args.train_k
    grad_clip = args.grad_clip
    batch_steps = args.batch_steps

    device = args.device
    log = args.log if args.rank <=0 else None

    kl_warmups = args.kl_warmup_steps
    kl_annealing = lambda step: min(1.0, (step + 1) / float(kl_warmups)) if kl_warmups > 0 else 1.0
    lr_decay = args.lr_decay
    init_lr = args.lr
    if lr_decay == 'expo':
        lr_warmups = 0
    elif lr_decay == 'inv_sqrt':
        lr_warmups = 10000
    else:
        raise ValueError('unknown lr decay method: %s' % lr_decay)

    betas = (args.beta1, args.beta2)
    eps = args.eps
    amsgrad = args.amsgrad
    weight_decay = args.weight_decay

    if args.recover > 0:
        checkpoint_name = args.checkpoint_name + '{}.tar'.format(args.recover)
        print(f"Rank = {args.rank}, loading from checkpoint {checkpoint_name}")
        optimizer, scheduler = get_optimizer(args.lr, flownmt.parameters(), betas, eps, amsgrad=amsgrad,
                                             weight_decay=weight_decay, lr_decay=lr_decay,
                                             warmup_steps=lr_warmups, init_lr=init_lr)

        checkpoint = torch.load(checkpoint_name, map_location=args.device)
        epoch = checkpoint['epoch']
        last_step = checkpoint['step']
        flownmt.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        best_epoch = checkpoint['best_epoch']
        best_bleu, test_bleu = checkpoint['best_bleu']
        best_nll, test_nll = checkpoint['best_nll']
        best_recon, test_recon = checkpoint['best_recon']
        best_kl, test_kl = checkpoint['best_kl']
        best_llen, test_llen = checkpoint['best_llen']
        best_ppl, test_ppl = checkpoint['best_ppl']
        del checkpoint

        if args.rank <= 0:
            with torch.no_grad():
                logging('Evaluating after resuming model...', log)
                eval(args, epoch, dataset, val_iter, flownmt)
    else:
        optimizer, scheduler = get_optimizer(args.lr, flownmt.parameters(), betas, eps, amsgrad=amsgrad,
                                             weight_decay=weight_decay, lr_decay=lr_decay,
                                             warmup_steps=lr_warmups, init_lr=init_lr)
        epoch = 0
        best_epoch = 0
        best_bleu = 0.0
        best_nll = 0.0
        best_recon = 0.0
        best_kl = 0.0
        best_llen = 0.0
        best_ppl = 0.0

        last_step = -1

    lr = scheduler.get_lr()[0]

    recon_loss = torch.Tensor([0.]).to(device)
    kl_loss = torch.Tensor([0.]).to(device)
    length_loss = torch.Tensor([0.]).to(device)
    num_insts = torch.Tensor([0.]).to(device)
    num_words = torch.Tensor([0.]).to(device)
    num_nans = 0
    num_back = 0

    flownmt.train()
    start_time = time.time()
    if args.rank <= 0:
        logging('Epoch: %d (lr=%.6f (%s), betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s, l2=%.1e, train_k=%d)' % (
            epoch + 1, lr, lr_decay, betas[0], betas[1], eps, amsgrad, weight_decay, train_k), log)

    for step, (src_sents, tgt_sents, src_masks, tgt_masks) in enumerate(train_iter):
        if step <= last_step:
            continue
        optimizer.zero_grad()
        batch_size = src_sents.size(0)
        words = tgt_masks.sum().item()
        recon_batch = 0
        kl_batch = 0
        llen_batch = 0
        kl_weight = kl_annealing(step)
        src_sents = [src_sents, ] if batch_steps == 1 else src_sents.chunk(batch_steps, dim=0)
        tgt_sents = [tgt_sents, ] if batch_steps == 1 else tgt_sents.chunk(batch_steps, dim=0)
        src_masks = [src_masks, ] if batch_steps == 1 else src_masks.chunk(batch_steps, dim=0)
        tgt_masks = [tgt_masks, ] if batch_steps == 1 else tgt_masks.chunk(batch_steps, dim=0)
        # disable allreduce for accumulated gradient.
        if args.rank >= 0:
            flownmt.disable_allreduce()
        for src, tgt, src_mask, tgt_mask in zip(src_sents[:-1], tgt_sents[:-1], src_masks[:-1], tgt_masks[:-1]):
            recon, kl, llen = flownmt.loss(src, tgt, src_masks=src_mask, tgt_masks=tgt_mask,
                                           nsamples=train_k)
            recon = recon.sum()
            kl = kl.sum()
            llen = llen.sum()
            if loss_ty_token:
                loss = (recon + kl * kl_weight + llen).div(words)
            else:
                loss = (recon + kl * kl_weight + llen).div(batch_size)
            loss.backward()
            with torch.no_grad():
                recon_batch += recon.item()
                kl_batch += kl.item()
                llen_batch += llen.item()

        # enable allreduce for the last step.
        if args.rank >= 0:
            flownmt.enable_allreduce()
        src, tgt, src_mask, tgt_mask = src_sents[-1], tgt_sents[-1], src_masks[-1], tgt_masks[-1]
        recon, kl, llen = flownmt.loss(src, tgt, src_masks=src_mask, tgt_masks=tgt_mask,
                                       nsamples=train_k)
        recon = recon.sum()
        kl = kl.sum()
        llen = llen.sum()
        if loss_ty_token:
            loss = (recon + kl * kl_weight + llen).div(words)
        else:
            loss = (recon + kl * kl_weight + llen).div(batch_size)
        loss.backward()
        with torch.no_grad():
            recon_batch += recon.item()
            kl_batch += kl.item()
            llen_batch += llen.item()

        if grad_clip > 0:
            grad_norm = clip_grad_norm_(flownmt.parameters(), grad_clip)
        else:
            grad_norm = total_grad_norm(flownmt.parameters())

        if math.isnan(grad_norm):
            num_nans += 1
        else:
            optimizer.step()
            scheduler.step()
            num_insts += batch_size
            num_words += words
            kl_loss += kl_batch
            recon_loss += recon_batch
            length_loss += llen_batch

        if step % 10 == 0:
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 and args.rank <= 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            nums = num_insts.item()
            train_recon = recon_loss.item() / nums if nums > 0 else 0
            train_kl = kl_loss.item() / nums if nums > 0 else 0
            train_llen = length_loss.item() / nums if nums > 0 else 0
            train_nll = train_recon + train_kl
            train_ppl = np.exp(train_nll * nums / num_words.item()) if nums > 0 else 0
            train_ppl = float('inf') if train_ppl > 10000 else train_ppl
            curr_lr = scheduler.get_lr()[0]
            curr_step = step if step == steps_per_epoch else step % steps_per_epoch
            log_info = '[{}/{} ({:.0f}%) lr={:.6f}, klw={:.2f} {}] NLL: {:.2f} (recon: {:.2f}, kl: {:.2f}), len: {:.2f}, PPL: {:.2f}'.format(
                curr_step, steps_per_epoch, 100. * curr_step / steps_per_epoch, curr_lr, kl_weight, num_nans,
                train_nll, train_recon, train_kl, train_llen, train_ppl)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

        if step % steps_per_epoch == 0 and step > 0:
            # new epoch
            epoch = step // steps_per_epoch
            lr = scheduler.get_lr()[0]

            if args.rank >= 0:
                dist.reduce(recon_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(kl_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(length_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(num_insts, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(num_words, dst=0, op=dist.ReduceOp.SUM)

            if args.rank <= 0:
                nums = num_insts.item()
                train_recon = recon_loss.item() / nums if nums > 0 else 0
                train_kl = kl_loss.item() / nums if nums > 0 else 0
                train_llen = length_loss.item() / nums if nums > 0 else 0
                train_nll = train_recon + train_kl
                train_ppl = np.exp(train_nll * nums / num_words.item()) if nums > 0 else 0
                train_ppl = float('inf') if train_ppl > 10000 else train_ppl

                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                logging('Average NLL: {:.2f} (recon: {:.2f}, kl: {:.2f}), len: {:.2f}, PPL: {:.2f}, time: {:.1f}s'.format(
                    train_nll, train_recon, train_kl, train_llen, train_ppl, time.time() - start_time), log)
                logging('-' * 100, log)

                with torch.no_grad():
                    logging('Evaluating validation data...', log)
                    bleu, nll, recon, kl, llen, ppl = eval(args, epoch, dataset, val_iter, flownmt)
                    if bleu > best_bleu or best_epoch == 0 or ppl < best_ppl:
                        flownmt.save(args.model_path)
                        best_bleu = bleu
                        best_epoch = epoch
                        best_nll = nll
                        best_recon = recon
                        best_kl = kl
                        best_llen = llen
                        best_ppl = ppl

                        logging('Evaluating test data...', log)
                        test_bleu, test_nll, test_recon, test_kl, test_llen, test_ppl = eval(args, epoch, dataset, test_iter, flownmt)

                logging('Best Dev  NLL: {:.2f} (recon: {:.2f}, kl: {:.2f}), len: {:.2f}, PPL: {:.2f}, BLEU: {:.2f}, epoch: {}'.format(
                    best_nll, best_recon, best_kl, best_llen, best_ppl, best_bleu, best_epoch), log)
                logging('Best Test NLL: {:.2f} (recon: {:.2f}, kl: {:.2f}), len: {:.2f}, PPL: {:.2f}, BLEU: {:.2f}, epoch: {}'.format(
                    test_nll, test_recon, test_kl, test_llen, test_ppl, test_bleu, best_epoch), log)
                logging('=' * 100, log)

                # save checkpoint
                checkpoint_name = args.checkpoint_name + '{}.tar'.format(epoch)
                torch.save({'epoch': epoch,
                            'step': step,
                            'model': flownmt.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_bleu': [best_bleu, test_bleu],
                            'best_epoch': best_epoch,
                            'best_nll': [best_nll, test_nll],
                            'best_recon': [best_recon, test_recon],
                            'best_kl': [best_kl, test_kl],
                            'best_llen': [best_llen, test_llen],
                            'best_ppl': [best_ppl, test_ppl]}, checkpoint_name)

            if epoch == epochs:
                break

            if args.rank <= 0:
                logging('Epoch: %d (lr=%.6f (%s), betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s, l2=%.1e, train_k=%d)' % (
                    epoch + 1, lr, lr_decay, betas[0], betas[1], eps, amsgrad, weight_decay, train_k), log)

            recon_loss = torch.Tensor([0.]).to(device)
            kl_loss = torch.Tensor([0.]).to(device)
            length_loss = torch.Tensor([0.]).to(device)
            num_insts = torch.Tensor([0.]).to(device)
            num_words = torch.Tensor([0.]).to(device)
            num_nans = 0
            num_back = 0
            flownmt.train()
            start_time = time.time()


def main(args):

    args, dataset, flownmt = setup(args)

    train_iter, val_iter, test_iter = init_dataloader(args, dataset)
    pretrain = args.recover < 0 and args.init_steps > 0
    checkpoint_name = args.checkpoint_name + '{}.tar'.format(0)

    if args.rank <= 0:
        # initialize model (rank 0 or -1)
        # number of parameters
        logging('Rank %d # of Parameters: %d' % (args.rank, sum([param.numel() for param in flownmt.parameters()])), args.log)

        if args.recover == 0:
            flownmt.load_core(checkpoint_name, args.device, load_prior=True)
            with torch.no_grad():
                reconstruct(0, dataset, val_iter, flownmt, args.result_path, args.log)
                logging('-' * 100, args.log)

    if args.rank >= 0:
        flownmt.init_distributed(args.rank, args.local_rank)

    if pretrain:
        init_posterior(args, train_iter, flownmt)
    elif args.recover < 0:
        init_model(args, train_iter, flownmt)

    if args.rank >= 0:
        flownmt.sync_params()

    if pretrain:
        zero_steps = args.init_steps
        pretrain_model(args, dataset, train_iter, val_iter, flownmt, zero_steps)
        init_prior(args, train_iter, flownmt)
        if args.rank >= 0:
            flownmt.sync_params()
        if args.rank <= 0:
            flownmt.save_core(checkpoint_name)

    train(args, dataset, train_iter, val_iter, test_iter, flownmt)


if __name__ == "__main__":
    args = parse_args()
    assert args.rank == -1 and args.local_rank == 0, 'single process should have wrong rank ({}) or local rank ({})'.format(args.rank, args.local_rank)
    main(args)
