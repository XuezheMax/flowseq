import os, sys
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='FlowNMT')
    parser.add_argument('--rank', type=int, default=-1, metavar='N', help='rank of the process in all distributed processes')
    parser.add_argument("--local_rank", type=int, default=0, metavar='N', help='rank of the process in the machine')
    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--eval_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for eval (default: 4)')
    parser.add_argument('--batch_steps', type=int, default=1, metavar='N',
                        help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
    parser.add_argument('--init_batch_size', type=int, default=1024, metavar='N',
                        help='number of instances for model initialization (default: 1024)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train')
    parser.add_argument('--kl_warmup_steps', type=int, default=10000, metavar='N', help='number of steps to warm up KL weight(default: 10000)')
    parser.add_argument('--init_steps', type=int, default=5000, metavar='N', help='number of steps to train decoder (default: 5000)')
    parser.add_argument('--seed', type=int, default=65537, metavar='S', help='random seed (default: 65537)')
    parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence',
                        help='loss type (default: sentence)')
    parser.add_argument('--train_k', type=int, default=1, metavar='N', help='training K (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr_decay', choices=['inv_sqrt', 'expo'], help='lr decay method', default='inv_sqrt')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    parser.add_argument('--eps', type=float, default=1e-6, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', default=None)

    parser.add_argument('--src', type=str, help='source language code', required=True)
    parser.add_argument('--tgt', type=str, help='target language code', required=True)
    parser.add_argument('--create_vocab', action='store_true', help='create vocabulary.')
    parser.add_argument('--share_all_embeddings', action='store_true', help='share source, target and output embeddings')
    parser.add_argument("--subword", type=str, default="joint-bpe", choices=['joint-bpe', 'sep-bpe', 'word', 'bert-bpe', 'joint-spm'])
    parser.add_argument('--recover', type=int, default=-1, help='recover the model from disk.')
    parser.add_argument("--bucket_batch", type=int, default=0, help="whether bucket data based on tgt length in batching")

    return parser.parse_args()


def parse_translate_args():
    parser = ArgumentParser(description='FlowNMT')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
    parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 65537)')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', default=None)
    parser.add_argument("--subword", type=str, default="joint-bpe", choices=['joint-bpe', 'sep-bpe', 'word', 'bert-bpe', 'joint-spm'])
    parser.add_argument("--bucket_batch", type=int, default=0, help="whether bucket data based on tgt length in batching")
    parser.add_argument('--decode', choices=['argmax', 'iw', 'sample'], help='decoding algorithm', default='argmax')
    parser.add_argument('--tau', type=float, default=0.0, metavar='S', help='temperature for iw decoding (default: 0.)')
    parser.add_argument('--nlen', type=int, default=3, help='number of length candidates.')
    parser.add_argument('--ntr', type=int, default=1, help='number of samples per length candidate.')
    return parser.parse_args()


def parse_distributed_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Dist FlowNMT")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    # arguments for flownmt model
    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--eval_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for eval (default: 4)')
    parser.add_argument('--init_batch_size', type=int, default=1024, metavar='N',
                        help='number of instances for model initialization (default: 1024)')
    parser.add_argument('--batch_steps', type=int, default=1, metavar='N',
                        help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train')
    parser.add_argument('--kl_warmup_steps', type=int, default=10000, metavar='N',
                        help='number of steps to warm up KL weight(default: 10000)')
    parser.add_argument('--init_steps', type=int, default=5000, metavar='N',
                        help='number of steps to train decoder (default: 5000)')
    parser.add_argument('--seed', type=int, default=65537, metavar='S', help='random seed (default: 524287)')
    parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence',
                        help='loss type (default: sentence)')
    parser.add_argument('--train_k', type=int, default=1, metavar='N', help='training K (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr_decay', choices=['inv_sqrt', 'expo'], help='lr decay method', default='inv_sqrt')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    parser.add_argument('--eps', type=float, default=1e-6, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', default=None)

    parser.add_argument('--src', type=str, help='source language code', required=True)
    parser.add_argument('--tgt', type=str, help='target language code', required=True)
    parser.add_argument('--create_vocab', action='store_true', help='create vocabulary.')
    parser.add_argument('--share_all_embeddings', action='store_true', help='share source, target and output embeddings')
    parser.add_argument("--subword", type=str, default="joint-bpe",
                        choices=['joint-bpe', 'sep-bpe', 'word', 'bert-bpe'])
    parser.add_argument("--bucket_batch", type=int, default=0,
                        help="whether bucket data based on tgt length in batching")
    parser.add_argument('--recover', type=int, default=-1, help='recover the model from disk.')

    return parser.parse_args()
