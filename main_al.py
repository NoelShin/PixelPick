import os
import random
import numpy as np
import torch
from model import Model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_ids[0])

    # set seed
    [func(args.seed) for func in [random.seed, np.random.seed, torch.manual_seed]]
    torch.backends.cudnn.benchmark = True

    Model(args)()


if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    main(args)
