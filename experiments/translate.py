import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import random
import numpy as np

import torch

from flownmt.data import NMTDataSet, DataIterator
from flownmt import FlowNMT
from experiments.options import parse_translate_args


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


def translate_argmax(dataset, dataloader, flownmt, result_path, outfile, tau, n_tr):
    flownmt.eval()
    translations = []
    lengths = []
    length_err = 0
    num_insts = 0
    start_time = time.time()
    num_back = 0
    for step, (src, tgt, src_masks, tgt_masks) in enumerate(dataloader):
        trans, lens = flownmt.translate_argmax(src, src_masks, n_tr=n_tr, tau=tau)
        translations.append(trans)
        lengths.append(lens)
        length_err += (lens.float() - tgt_masks.sum(dim=1)).abs().sum().item()
        num_insts += src.size(0)
        if step % 10 == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = 'argmax translating (tau={:.1f}, n_tr={})...{}'.format(tau, n_tr, num_insts)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)
    print('time: {:.1f}s'.format(time.time() - start_time))
    outfile = os.path.join(result_path, outfile)
    dataset.dump_to_file(translations, lengths, outfile)
    bleu = calc_bleu(dataloader.tgt_sort_origin_path, outfile, result_path)
    print('#SENT: {}, Length Err: {:.1f}, BLEU: {:.2f}'.format(num_insts, length_err / num_insts, bleu))


def translate_iw(dataset, dataloader, flownmt, result_path, outfile, tau, n_len, n_tr):
    flownmt.eval()
    iwk = 4
    translations = []
    lengths = []
    length_err = 0
    num_insts = 0
    start_time = time.time()
    num_back = 0
    for step, (src, tgt, src_masks, tgt_masks) in enumerate(dataloader):
        trans, lens = flownmt.translate_iw(src, src_masks, n_len=n_len, n_tr=n_tr, tau=tau, k=iwk)
        translations.append(trans)
        lengths.append(lens)
        length_err += (lens.float() - tgt_masks.sum(dim=1)).abs().sum().item()
        num_insts += src.size(0)
        if step % 10 == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = 'importance weighted translating (tau={:.1f}, n_len={}, n_tr={})...{}'.format(tau, n_len, n_tr, num_insts)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)
    print('time: {:.1f}s'.format(time.time() - start_time))
    outfile = os.path.join(result_path, outfile)
    dataset.dump_to_file(translations, lengths, outfile)
    bleu = calc_bleu(dataloader.tgt_sort_origin_path, outfile, result_path)
    print('#SENT: {}, Length Err: {:.1f}, BLEU: {:.2f}'.format(num_insts, length_err / num_insts, bleu))


def sample(dataset, dataloader, flownmt, result_path, outfile, tau, n_len, n_tr):
    flownmt.eval()
    lengths = []
    translations = []
    num_insts = 0
    start_time = time.time()
    num_back = 0
    for step, (src, tgt, src_masks, tgt_masks) in enumerate(dataloader):
        trans, lens = flownmt.translate_sample(src, src_masks, n_len=n_len, n_tr=n_tr, tau=tau)
        translations.append(trans)
        lengths.append(lens)
        num_insts += src.size(0)
        if step % 10 == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = 'sampling (tau={:.1f}, n_len={}, n_tr={})...{}'.format(tau, n_len, n_tr, num_insts)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)
    print('time: {:.1f}s'.format(time.time() - start_time))
    outfile = os.path.join(result_path, outfile)
    dataset.dump_to_file(translations, lengths, outfile, post_edit=False)


def setup(args):
    args.cuda = torch.cuda.is_available()
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = False

    model_path = args.model_path
    result_path = os.path.join(model_path, 'translations')
    args.result_path = result_path
    params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))

    src_lang = params['src']
    tgt_lang = params['tgt']
    data_path = args.data_path
    vocab_path = os.path.join(model_path, 'vocab')
    src_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(src_lang))
    tgt_vocab_path = os.path.join(vocab_path, '{}.vocab'.format(tgt_lang))
    src_vocab_size = params['src_vocab_size']
    tgt_vocab_size = params['tgt_vocab_size']
    args.max_src_length = params.pop('max_src_length')
    args.max_tgt_length = params.pop('max_tgt_length')
    dataset = NMTDataSet(data_path, src_lang, tgt_lang,
                         src_vocab_path, tgt_vocab_path,
                         src_vocab_size, tgt_vocab_size,
                         subword=args.subword, create_vocab=False)
    assert src_vocab_size == dataset.src_vocab_size
    assert tgt_vocab_size == dataset.tgt_vocab_size

    flownmt = FlowNMT.load(model_path, device=device)
    args.length_unit = flownmt.length_unit
    args.device = device
    return args, dataset, flownmt


def init_dataloader(args, dataset):
    eval_batch = args.batch_size
    val_iter = DataIterator(dataset, eval_batch, 0, args.max_src_length, args.max_tgt_length, 1000, args.device, args.result_path,
                            bucket_data=args.bucket_batch, multi_scale=args.length_unit, corpus="dev")
    test_iter = DataIterator(dataset, eval_batch, 0, args.max_src_length, args.max_tgt_length, 1000, args.device, args.result_path,
                             bucket_data=args.bucket_batch, multi_scale=args.length_unit, corpus="test")
    return val_iter, test_iter


def main(args):
    args, dataset, flownmt = setup(args)
    print(args)

    val_iter, test_iter = init_dataloader(args, dataset)

    result_path = args.result_path
    if args.decode == 'argmax':
        tau = args.tau
        n_tr = args.ntr
        outfile = 'argmax.t{:.1f}.ntr{}.dev.mt'.format(tau, n_tr)
        translate_argmax(dataset, val_iter, flownmt, result_path, outfile, tau, n_tr)
        outfile = 'argmax.t{:.1f}.ntr{}.test.mt'.format(tau, n_tr)
        translate_argmax(dataset, test_iter, flownmt, result_path, outfile, tau, n_tr)
    elif args.decode == 'iw':
        tau = args.tau
        n_len = args.nlen
        n_tr = args.ntr
        outfile = 'iw.t{:.1f}.nlen{}.ntr{}.dev.mt'.format(tau, n_len, n_tr)
        translate_iw(dataset, val_iter, flownmt, result_path, outfile, tau, n_len, n_tr)
        outfile = 'iw.t{:.1f}.nlen{}.ntr{}.test.mt'.format(tau, n_len, n_tr)
        translate_iw(dataset, test_iter, flownmt, result_path, outfile, tau, n_len, n_tr)
    else:
        assert not args.bucket_batch
        tau = args.tau
        n_len = args.nlen
        n_tr = args.ntr
        outfile = 'sample.t{:.1f}.nlen{}.ntr{}.dev.mt'.format(tau, n_len, n_tr)
        sample(dataset, val_iter, flownmt, result_path, outfile, tau, n_len, n_tr)
        outfile = 'sample.t{:.1f}.nlen{}.ntr{}.test.mt'.format(tau, n_len, n_tr)
        sample(dataset, test_iter, flownmt, result_path, outfile, tau, n_len, n_tr)


if __name__ == "__main__":
    args = parse_translate_args()
    with torch.no_grad():
        main(args)