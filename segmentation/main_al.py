import os
import torch
from model import Model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids[0]
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args)
    model()


if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    main(args)
