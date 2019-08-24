import codecs
import math
import random
from collections import defaultdict
import numpy as np
import torch
import os


def get_sorted_wordlist(path):
    freqs = defaultdict(lambda: 0)

    with codecs.open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            words = line.strip().split()
            for word in words:
                freqs[word] += 1

    sorted_words = sorted(freqs, key=freqs.get, reverse=True)

    wordlist = [word for word in sorted_words]
    return wordlist


UNK = "<unk>"
EOS = "<eos>"
PAD = "<pad>"

SRC_PAD = PAD
TGT_PAD = PAD


class NMTDataSet():
    def __init__(self, data_path, src_lang, tgt_lang, src_vocab_path, tgt_vocab_path, src_max_vocab, tgt_max_vocab,
                 subword, create_vocab):
        self.train_src_path = os.path.join(data_path, 'train.{}'.format(src_lang))
        self.train_tgt_path = os.path.join(data_path, 'train.{}'.format(tgt_lang))
        self.dev_src_path = os.path.join(data_path, 'dev.{}'.format(src_lang))
        self.dev_tgt_path = os.path.join(data_path, 'dev.{}'.format(tgt_lang))
        self.test_src_path = os.path.join(data_path, 'test.{}'.format(src_lang))
        self.test_tgt_path = os.path.join(data_path, 'test.{}'.format(tgt_lang))

        self.subword = subword
        if "bpe" in subword:
            self.dev_tgt_path_ori = os.path.join(data_path, 'dev.{}.ori'.format(tgt_lang))
            self.test_tgt_path_ori = os.path.join(data_path, 'test.{}.ori'.format(tgt_lang))
        else:
            self.dev_tgt_path_ori = self.dev_tgt_path
            self.test_tgt_path_ori = self.test_tgt_path

        if not create_vocab:
            assert src_vocab_path is not None and tgt_vocab_path is not None and os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path)
            self.src_word2id, self.src_id2word = self.load_vocab(src_vocab_path)
            self.tgt_word2id, self.tgt_id2word = self.load_vocab(tgt_vocab_path)
        else:
            if subword == "joint-bpe":
                joint_path = os.path.join(data_path, "joint.tmp")
                os.system("cat %s %s > %s" % (self.train_src_path, self.train_tgt_path, joint_path))
                assert src_max_vocab == tgt_max_vocab, "src max vocab size != tgt max vocab size"
                word2id, id2word = self.get_vocab(joint_path, src_max_vocab, has_pad=True)
                os.remove(joint_path)
                self.src_word2id = self.tgt_word2id = word2id
                self.src_id2word = self.tgt_id2word = id2word
            else:
                if subword == "sep-bpe":
                    assert src_max_vocab == tgt_max_vocab, "src max vocab size != tgt max vocab size"
                self.src_word2id, self.src_id2word = self.get_vocab(self.train_src_path, src_max_vocab, has_pad=True)
                self.tgt_word2id, self.tgt_id2word = self.get_vocab(self.train_tgt_path, tgt_max_vocab, has_pad=True)

            if src_vocab_path is not None and tgt_vocab_path is not None:
                self.save_vocab(self.src_id2word, src_vocab_path)
                self.save_vocab(self.tgt_id2word, tgt_vocab_path)

        self.src_vocab_size = len(self.src_word2id)
        self.tgt_vocab_size = len(self.tgt_word2id)
        self.src_pad_idx = self.src_word2id[SRC_PAD]
        self.tgt_pad_idx = self.tgt_word2id[TGT_PAD]
        print(f"Source vocab size={len(self.src_word2id)}, target vocab size={len(self.tgt_word2id)}")

    def load_vocab(self, path):
        word2id = {}
        i = 0
        with codecs.open(path, "r", "utf-8") as fin:
            for line in fin:
                word2id[line.strip()] = i
                i += 1
        id2word = {v: k for k, v in word2id.items()}
        return word2id, id2word

    def save_vocab(self, id2word, path):
        print(f"Saving vocab to {path}")
        with codecs.open(path, "w", encoding="utf-8") as fout:
            for i in range(len(id2word)):
                fout.write(id2word[i] + "\n")

    def get_vocab(self, path, max_vocab=-1, has_pad=True):
        if max_vocab > 0:
            max_vocab = max_vocab - 3 if has_pad else max_vocab - 2
        wordlist = get_sorted_wordlist(path)
        if max_vocab > 0:
            wordlist = wordlist[:max_vocab]
        word2id = {}
        if has_pad:
            word2id[PAD] = 0
        word2id[UNK] = len(word2id)
        word2id[EOS] = len(word2id)
        for word in wordlist:
            word2id[word] = len(word2id)
        id2word = {i: word for word, i in word2id.items()}
        return word2id, id2word

    def dump_to_file(self, ms, lengths, path, post_edit=True):
        # ms: list of (batch_size, sent_len)
        with codecs.open(path, "w", encoding="utf-8") as fout:
            for m, length in zip(ms, lengths):
                m = m.cpu().numpy()
                length = length.cpu().numpy()
                for line, l in zip(m, length):
                    sent = []
                    for w in line[:l]:
                        word = self.tgt_id2word[w]
                        if word == EOS:
                            break
                        sent.append(word)
                    if post_edit and (self.subword == "sep-bpe" or self.subword == "joint-bpe"):
                        line = ' '.join(sent)
                        line = line.replace('@@ ', '').strip()
                        if line.endswith("@@"):
                            line = line[-2:]
                    elif post_edit and (self.subword == "joint-spm"):
                        line = ''.join(sent)
                        line = line.replace('▁', ' ').strip()
                    else:
                        line = " ".join(sent)
                    fout.write(line + "\n")


def max_tok_len(example, count):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(example[0]) + 1)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(example[1]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def batch_iter(data, batch_size, batch_size_fn=None, shuffle=False):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch))
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                yield minibatch[:-overflowed]
                minibatch = minibatch[-overflowed:]
                size_so_far = 0
                for i, ex in enumerate(minibatch):
                    size_so_far = batch_size_fn(ex, i + 1)
    if minibatch:
        yield minibatch


def bucket_batch_iter(data, batch_size, batch_size_fn=None, shuffle=False):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.
    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count):
            return count

    buckets = [20, 40, 60, 80]
    bucket_data = [[] for _ in buckets]
    outliers = []
    for ex in data:
        tgt_len = len(ex[1])
        if tgt_len > buckets[-1]:
            outliers.append(ex)
            continue
        for bid, bl in enumerate(buckets):
            if tgt_len <= bl:
                bucket_data[bid].append(ex)
                break
    if len(outliers) > 0:
        bucket_data.append(outliers)

    batches, minibatch, size_so_far = [], [], 0

    for bucket in bucket_data:
        if shuffle:
            random.shuffle(bucket)
        for ex in bucket:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch))
            if size_so_far >= batch_size:
                overflowed = 0
                if size_so_far > batch_size:
                    overflowed += 1
                if overflowed == 0:
                    batches.append(minibatch)
                    minibatch, size_so_far = [], 0
                else:
                    batches.append(minibatch[:-overflowed])
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1)

    if minibatch:
        batches.append(minibatch)

    if shuffle:
        random.shuffle(batches)

    for minibatch in batches:
        yield minibatch


class DataIterator():
    def __init__(self, dataset, batch_size, batch_by_tokens, max_src_length, max_tgt_length, buffer_multiple_size,
                 device, model_path, len_diff=-1, len_ratio=-1, multi_scale=1, corpus="train",
                 bucket_data=True, rank=-1, num_replicas=0):

        self.train = False  # need shuffle and sort
        self.device = device

        if corpus == "train":
            self.src_path = dataset.train_src_path
            self.tgt_path = dataset.train_tgt_path
            self.tgt_path_ori = None
            self.train = True
        elif corpus == "dev":
            self.src_path = dataset.dev_src_path
            self.tgt_path = dataset.dev_tgt_path
            self.tgt_path_ori = dataset.dev_tgt_path_ori
        elif corpus == "test":
            self.src_path = dataset.test_src_path
            self.tgt_path = dataset.test_tgt_path
            self.tgt_path_ori = dataset.test_tgt_path_ori
        else:
            raise ValueError

        self.corpus = corpus

        self.batch_size = batch_size
        self.batch_size_fn = max_tok_len if batch_by_tokens else None
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.len_diff = len_diff
        self.len_ratio = len_ratio
        self.multi_scale = multi_scale

        self.src_word2id = dataset.src_word2id
        self.tgt_word2id = dataset.tgt_word2id

        if rank < 0:
            assert num_replicas == 0
        else:
            assert corpus == 'train'
            assert rank < num_replicas
            assert self.tgt_path_ori is None
        self.rank = rank
        self.num_replicas = num_replicas

        self.data_size, self.data = self.get_dataset()
        self.batches = None
        if self.train:
            self.buffer_size = buffer_multiple_size * self.batch_size
            assert buffer_multiple_size > 0
        else:
            self.buffer_size = -1

        self.src_pad_idx = self.src_word2id[SRC_PAD]
        self.tgt_pad_idx = self.tgt_word2id[TGT_PAD]

        self.bucket = bucket_data
        self.sents_num = 0
        self.tgt_sort_origin_path = os.path.join(model_path, os.path.basename(self.tgt_path) + ".sort")

    def filter_sents(self, s_tokens, t_tokens):
        if self.max_tgt_length > 0 and self.max_src_length > 0:
            if len(s_tokens) + 1 > self.max_src_length or len(t_tokens) + 1 > self.max_tgt_length:
                return True

        if self.len_diff > 0:
            if abs(len(s_tokens) - len(t_tokens)) > self.len_diff:
                return True

        if self.len_ratio > 0:
            ratio = len(t_tokens) / len(s_tokens)
            if ratio > self.len_ratio or ratio < (1. / self.len_ratio):
                return True

        return False

    def pad_tgt(self, tgt):
        scale = self.multi_scale
        tgt_len = len(tgt)
        res = tgt_len % scale if tgt_len % scale > 0 else scale
        tgt_len = (scale - res) + tgt_len
        tgt = tgt + [self.tgt_word2id[EOS]] * (tgt_len - len(tgt))
        return tgt

    def get_dataset(self):
        count = 0
        data = []
        outliers = 0

        src_path, tgt_path = self.src_path, self.tgt_path
        tgt_ori_path = self.tgt_path_ori
        ftgt_ori = None if tgt_ori_path is None else codecs.open(tgt_ori_path, "r", encoding="utf-8")
        with codecs.open(src_path, "r", encoding="utf-8") as fsrc, codecs.open(tgt_path, "r", encoding="utf-8") as ftgt:
            for id, (s, t) in enumerate(zip(fsrc, ftgt)):
                if self.num_replicas > 0 and id % self.num_replicas != self.rank:
                    continue
                s_tokens = s.strip().split()
                t_tokens = t.strip().split()

                t_ori = ftgt_ori.readline().strip() if ftgt_ori is not None else None

                src = [self.src_word2id[word] if word in self.src_word2id else self.src_word2id[UNK] for word in s_tokens] + [self.src_word2id[EOS]]
                tgt = [self.tgt_word2id[word] if word in self.tgt_word2id else self.tgt_word2id[UNK] for word in t_tokens] #+ [self.tgt_word2id[EOS]]
                tgt = self.pad_tgt(tgt)

                if self.train and self.filter_sents(src, tgt):
                    outliers += 1
                    continue
                else:
                    if not self.train:
                        data.append((src, tgt, t_ori))
                        if self.filter_sents(src, tgt):
                            outliers += 1
                    else:
                        data.append((src, tgt))
                    count += 1
        print(f"Load total {count} sentences pairs, {outliers} are out of maximum sentence length!")
        return count, data

    def batch(self, batch_size):
        """Yield elements from data in chunks of batch_size."""
        batch_size_fn = self.batch_size_fn
        if batch_size_fn is None:
            def batch_size_fn(new, count):
                return count
        minibatch, size_so_far = [], 0
        for ex in self.data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1)

        if minibatch:
            yield minibatch

    def process_batch(self, minibatch):
        # padding and make mask of minibatch
        # return: batch_size x max_len
        # minibatch = sorted(minibatch, key=lambda x: len(x[1]), reverse=True)
        src_max_len = max([len(d[0]) for d in minibatch])
        tgt_max_len = max([len(d[1]) for d in minibatch])

        padded_src, padded_tgt = [], []
        src_mask = []
        tgt_mask = []
        for d in minibatch:
            s, t = d[0], d[1]
            padded_src.append(s + [self.src_pad_idx] * (src_max_len - len(s)))
            padded_tgt.append(t + [self.tgt_pad_idx] * (tgt_max_len - len(t)))
            src_mask.append([1.] * len(s) + [0.] * (src_max_len - len(s)))
            tgt_mask.append([1.] * len(t) + [0.] * (tgt_max_len - len(t)))
        padded_src = torch.from_numpy(np.array(padded_src)).long().to(self.device)
        padded_tgt = torch.from_numpy(np.array(padded_tgt)).long().to(self.device)
        src_mask = torch.from_numpy(np.array(src_mask)).float().to(self.device)
        tgt_mask = torch.from_numpy(np.array(tgt_mask)).float().to(self.device)
        return padded_src, padded_tgt, src_mask, tgt_mask

    def init_epoch(self):
        # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
        # be sorted by decreasing order, which requires reversing
        # relative to typical sort keys
        if self.train:
            def _pool():
                for p in self.batch(self.buffer_size):
                    if self.bucket:
                        p_batch = bucket_batch_iter(p,
                                                    self.batch_size,
                                                    batch_size_fn=self.batch_size_fn, shuffle=True)
                    else:
                        p_batch = batch_iter(random.sample(p, len(p)),
                                             self.batch_size,
                                             batch_size_fn=self.batch_size_fn)
                    p_batch = list(p_batch)
                    for b in p_batch:
                        yield b

            self.batches = _pool()
        else:
            if self.batches is None:
                self.batches = []
            else:
                self.batches.clear()
            iter_func = bucket_batch_iter if self.bucket else batch_iter
            for b in iter_func(
                    self.data,
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn):
                # self.batches.append(sorted(b, key=lambda x: len(x[1]), reverse=True))
                self.batches.append(b)

    def __iter__(self):
        while True:
            self.init_epoch()
            tgt_ori_sents = []
            for idx, minibatch in enumerate(self.batches):
                self.sents_num += len(minibatch)
                if not self.train:
                    tgt_ori_sents.append([d[2] for d in minibatch])
                src_batch, tgt_batch, src_mask, tgt_mask = self.process_batch(minibatch)
                yield src_batch, tgt_batch, src_mask, tgt_mask

            if not self.train:
                with codecs.open(self.tgt_sort_origin_path, "w", encoding="utf-8") as fout:
                    for b in tgt_ori_sents:
                        for sent in b:
                            fout.write(sent + "\n")
                return

    def get_batch(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return self.process_batch(batch)

    @property
    def epoch(self):
        return self.sents_num * 1. / self.data_size

    def __len__(self):
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(self.data_size / self.batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src_path", type=str, default=None)
    parser.add_argument("--train_tgt_path", type=str, default=None)
    parser.add_argument("--dev_src_path", type=str, default=None)
    parser.add_argument("--dev_tgt_path", type=str, default=None)
    parser.add_argument("--test_src_path", type=str, default=None)
    parser.add_argument("--test_tgt_path", type=str, default=None)
    parser.add_argument("--src_vocab_path", type=str, default="src.vocab")
    parser.add_argument("--tgt_vocab_path", type=str, default="tgt.vocab")

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--batch_by_tokens", type=int, default=1, help="0 is False")
    parser.add_argument("--max_src_length", type=int, default=80)
    parser.add_argument("--max_tgt_length", type=int, default=80)
    parser.add_argument("--buffer_multiple_size", type=int, default=3)

    parser.add_argument("--src_max_vocab", type=int, default=50000)
    parser.add_argument("--tgt_max_vocab", type=int, default=50000)

    parser.add_argument("--create_vocab", type=int, default=0)

    args = parser.parse_args()

    model_path = "debug"
    dataset = NMTDataSet(args.train_src_path, args.train_tgt_path, args.dev_src_path, args.dev_tgt_path,
                         args.test_src_path, args.test_tgt_path, args.src_vocab_path, args.tgt_vocab_path,
                         args.src_max_vocab, args.tgt_max_vocab, args.create_vocab)
    train_iterator = DataIterator(dataset, args.batch_size, args.batch_by_tokens, args.max_src_length, args.max_tgt_length,
                                  args.buffer_multiple_size, device="cpu", model_path=model_path, corpus="train")
    dev_iterator = DataIterator(dataset, args.batch_size, args.batch_by_tokens, args.max_src_length, args.max_tgt_length,
                                args.buffer_multiple_size, device="cpu", model_path=model_path, corpus="dev")


    # test_iterator = DataIterator(dataset, args, device="cpu", corpus="test")

    def _print(batch, id2word):
        for sent in batch:
            if id2word is None:
                print(" ".join([str(i) for i in sent]) + "\n")
            else:
                print(" ".join([id2word[w] for w in sent]) + "\n")


    step = 0
    for src_batch, tgt_batch, src_mask in train_iterator:
        print("Epoch = %f\n" % train_iterator.epoch)
        print("---src batch %d ----" % step)
        _print(src_batch.numpy(), dataset.src_id2word)
        print("---tgt batch %d ----" % step)
        _print(tgt_batch.numpy(), dataset.tgt_id2word)
        print("---src mask %d ----" % step)
        _print(src_mask.numpy(), None)

        step += 1

        if step % 10 == 0:
            break

    print("###############  Dev ###############")
    step = 0
    for src_batch, tgt_batch, src_mask in dev_iterator:
        print("Epoch = %f\n" % dev_iterator.epoch)
        print("---src batch %d ----" % step)
        _print(src_batch.numpy(), dataset.src_id2word)
        print("---tgt batch %d ----" % step)
        _print(tgt_batch.numpy(), dataset.tgt_id2word)
        print("---src mask %d ----" % step)
        _print(src_mask.numpy(), None)

        step += 1
