import torch
from model import Model
from utils.utils import write_log, send_file, Visualiser, zip_dir


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args)
    model()

    zip_file = zip_dir(args.dir_checkpoints)
    # send_file(zip_file)


if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    main(args)
