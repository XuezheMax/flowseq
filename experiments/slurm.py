import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import torch.multiprocessing as mp

import experiments.options as options
from experiments.nmt import main as single_process_main


def main():
    args = options.parse_distributed_args()
    args_dict = vars(args)

    args_dict.pop('master_addr')
    str(args_dict.pop('master_port'))
    args_dict.pop('nnodes')
    args_dict.pop('nproc_per_node')
    args_dict.pop('node_rank')

    current_env = os.environ
    nnodes = int(current_env['SLURM_NNODES'])
    dist_world_size = int(current_env['SLURM_NTASKS'])
    args.rank = int(current_env['SLURM_PROCID'])
    args.local_rank = int(current_env['SLURM_LOCALID'])


    print('start process: rank={}({}), master addr={}, port={}, nnodes={}, world size={}'.format(
        args.rank, args.local_rank, current_env["MASTER_ADDR"], current_env["MASTER_PORT"], nnodes, dist_world_size))
    current_env["WORLD_SIZE"] = str(dist_world_size)

    create_vocab = args_dict.pop('create_vocab')
    assert not create_vocab
    args.create_vocab = False

    batch_size = args.batch_size // dist_world_size
    args.batch_size = batch_size

    single_process_main(args)


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    main()
