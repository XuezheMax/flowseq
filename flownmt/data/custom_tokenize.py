# tokenize and apply BPE
import os
import argparse
from shutil import copyfile
from pytorch_pretrained_bert import BertTokenizer
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--src_lang", type=str)
parser.add_argument("--tgt_lang", type=str)

# src and tgt have the same number of bpe size
parser.add_argument("--bpe_size", type=int, default=32000, help="bpe size")
parser.add_argument("--use_joint", type=int, default=0, help="if true, train bpe on the joint corpus of src and tgt")

parser.add_argument("--use_bert_tok", type=int, default=0)

args = parser.parse_args()


def create_dir_name(type):
    bpe_size = args.bpe_size
    bpe_size = str(bpe_size // 1000) + "k"
    if type != "bert":
        dir_name = type + "-spm-" + bpe_size
    else:
        dir_name = type + "-bpe"
    return dir_name


def create_model_prefix(type):
    bpe_size = args.bpe_size
    if type == "source":
        lang = args.src_lang
    elif type == "target":
        lang = args.tgt_lang
    elif type == "joint":
        lang = "joint"
    else:
        raise ValueError
    bpe_size = str(bpe_size // 1000) + "k"
    model_prefix = lang + "." + bpe_size + ".spm"
    return model_prefix


data_path = args.data_path
src_lang = args.src_lang
tgt_lang = args.tgt_lang
bpe_size = args.bpe_size

train_src_path = os.path.join(data_path, 'train.{}'.format(src_lang))
train_tgt_path = os.path.join(data_path, 'train.{}'.format(tgt_lang))
dev_src_path = os.path.join(data_path, 'dev.{}'.format(src_lang))
dev_tgt_path = os.path.join(data_path, 'dev.{}'.format(tgt_lang))
test_src_path = os.path.join(data_path, 'test.{}'.format(src_lang))
test_tgt_path = os.path.join(data_path, 'test.{}'.format(tgt_lang))

# path to saved bpe models
bpe_output_path = os.path.join(data_path, "spm-bpe-codes")

parent_dir = os.path.abspath(os.path.join(data_path, os.path.pardir))
if args.use_joint:
    prefix = "joint"
elif args.use_bert_tok:
    prefix = "bert"
else:
    prefix = "sep"

dir_name = create_dir_name(prefix)
# path to bpe train/dev/test files
output_dir = os.path.join(parent_dir, dir_name)

opt_train_src_path = os.path.join(output_dir, 'train.{}'.format(src_lang))
opt_train_tgt_path = os.path.join(output_dir, 'train.{}'.format(tgt_lang))
opt_dev_src_path = os.path.join(output_dir, 'dev.{}'.format(src_lang))
opt_dev_tgt_path = os.path.join(output_dir, 'dev.{}'.format(tgt_lang))
opt_test_src_path = os.path.join(output_dir, 'test.{}'.format(src_lang))
opt_test_tgt_path = os.path.join(output_dir, 'test.{}'.format(tgt_lang))

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

copyfile(dev_src_path, opt_dev_src_path + ".ori")
copyfile(test_src_path, opt_test_src_path + ".ori")
copyfile(dev_tgt_path, opt_dev_tgt_path + ".ori")
copyfile(test_tgt_path, opt_test_tgt_path + ".ori")

if not os.path.isdir(bpe_output_path):
    os.mkdir(bpe_output_path)

if args.use_bert_tok:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    paths_list = [train_src_path, dev_src_path, test_src_path, train_tgt_path, dev_tgt_path, test_tgt_path]
    opt_list = [opt_train_src_path, opt_dev_src_path, opt_test_src_path, opt_train_tgt_path, opt_dev_tgt_path, opt_test_tgt_path]
    print("Tokenize with multilingual BERT tokenizer!")
    for path, opt_path in zip(paths_list, opt_list):
        with open(path, "r", encoding="utf-8") as fin, open(opt_path, "w", encoding="utf-8") as fout:
            for line in fin:
                tokens = tokenizer.tokenize(line.strip())
                fout.write(" ".join(tokens) + "\n")
            print("------ Tokenization done on {} -----".format(path))

    unique_bpes = set()
    for path in [opt_train_src_path, opt_train_tgt_path]:
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                for token in line.strip().split():
                    unique_bpes.add(token)
    print("BPE vocab size of multilingual BERT after tokenization: {}".format(len(unique_bpes)))

else:
    def encode(path_list: list, opt_list: list, sp: spm.SentencePieceProcessor):
        for path, opt_path in zip(path_list, opt_list):
            with open(path, "r", encoding="utf-8") as fin, open(opt_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    sent = sp.EncodeAsPieces(line.strip())
                    fout.write(" ".join(sent) + "\n")
                print("------ Encoding done on {} ------".format(path))

    if args.use_joint:
        train_joint_path = os.path.join(data_path, "train.joint")
        os.system("cat %s %s > %s" % (train_src_path, train_tgt_path, train_joint_path))
        model_prefix = os.path.join(bpe_output_path, create_model_prefix("joint"))

        print("Train bpe on the joint corpus!")
        spm.SentencePieceTrainer.Train("--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false".
                                       format(train_joint_path, model_prefix, bpe_size))

        sp = spm.SentencePieceProcessor()
        sp.Load(model_prefix + ".model")
        # sp.Load_vocabulary(model_prefix + ".vocab", 2)
        print("Encode with learned bpe!")
        encode([train_src_path, dev_src_path, test_src_path, train_tgt_path, dev_tgt_path, test_tgt_path],
               [opt_train_src_path, opt_dev_src_path, opt_test_src_path, opt_train_tgt_path, opt_dev_tgt_path, opt_test_tgt_path],
               sp)
    else:
        src_model_prefix = os.path.join(bpe_output_path, create_model_prefix("source"))
        tgt_model_prefix = os.path.join(bpe_output_path, create_model_prefix("target"))
        print("Train bpe on the source corpus!")
        spm.SentencePieceTrainer.Train("--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false".
                                       format(train_src_path, src_model_prefix, bpe_size))
        # "--input={} --model_prefix={} --vocab_size={} --model_type=bpe"
        sp = spm.SentencePieceProcessor()
        sp.Load(src_model_prefix + ".model")
        # sp.Load(src_model_prefix + ".vocab", 2)s
        print("Encode source with learned bpe!")
        encode([train_src_path, dev_src_path, test_src_path], [opt_train_src_path, opt_dev_src_path, opt_test_src_path], sp)

        print("Train bpe on the target corpus!")
        spm.SentencePieceTrainer.Train("--input={} --model_prefix={} --vocab_size={} --model_type=bpe".
                                       format(train_tgt_path, tgt_model_prefix, bpe_size))
        sp = spm.SentencePieceProcessor()
        sp.Load(tgt_model_prefix + ".model")
        sp.load(tgt_model_prefix + ".vocab", 2)
        print("Encode target with learned bpe!")
        encode([train_tgt_path, dev_tgt_path, test_tgt_path], [opt_train_tgt_path, opt_dev_tgt_path, opt_test_tgt_path], sp)