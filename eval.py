import os
from copy import deepcopy
from typing import Optional
from math import ceil
import torch
import torch.functional as F
from tqdm import tqdm

from query import UncertaintySampler
from utils.utils import get_model, write_log
from utils.metrics import RunningScore


@torch.no_grad()
def evaluate(
        model,
        dataloader,
        experim_name: str,
        epoch: Optional[int] = None,
        dir_ckpt: Optional[str] = None,
        visualizer: Optional[callable] = None,
        visualize_interval: Optional[int] = 100,
        stride_total: int = 8,
        device: torch.device = torch.device("cuda:0"),
        debug: bool = False
):
    """
    :param model: a model to be evaluated
    :param dataloader: a dataloader for a validation set
    :param miou_tracker:
    :param epoch:
    :param fp: If given, the model weights will be saved in the given file path if the current mIoU is larger than
    previous best mIoU.
    :param prev_best_miou: If fp is not None, save the model if current mIoU is larger than prev_best_miou.
    :param debug: forward a single iteration for debugging.
    :return:
    """
    if dir_ckpt is not None:
        dir_ckpt = f"{dir_ckpt}/e{epoch:02d}/val" if epoch is not None else f"{dir_ckpt}/val"
        os.makedirs(dir_ckpt, exist_ok=True)

    dataloader_iter, tbar = iter(dataloader), tqdm(range(len(dataloader)))
    model.eval()
    miou_tracker = RunningScore(n_classes=dataloader.dataset.n_classes)
    for num_iter in tbar:
        dict_data = next(dataloader_iter)
        x, y = dict_data['x'].to(device), dict_data['y'].to(device)

        if dataloader.dataset.dataset_name == "voc":
            h, w = y.shape[1:]
            pad_h = ceil(h / stride_total) * stride_total - x.shape[2]
            pad_w = ceil(w / stride_total) * stride_total - x.shape[3]
            x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode='reflect')
            dict_outputs = model(x)
            dict_outputs['pred'] = dict_outputs['pred'][:, :, :h, :w]

        else:
            dict_outputs = model(x)

        logits = dict_outputs['pred']
        prob, pred = torch.softmax(logits.detach(), dim=1), logits.argmax(dim=1)

        miou_tracker.update(y.cpu().numpy(), pred.cpu().numpy())
        scores = miou_tracker.get_scores()[0]
        miou, pixel_acc = scores['Mean IoU'], scores['Pixel Acc']
        tbar.set_description(f"mIoU: {miou:.3f} | pixel acc.: {pixel_acc:.3f}")

        if dir_ckpt is not None and visualizer is not None and num_iter % visualize_interval == 0:
            if visualizer is not None:
                ent, lc, ms, = [getattr(UncertaintySampler, f"_{uc}")(prob)[0].cpu() for uc in
                                ["entropy", "least_confidence", "margin_sampling"]]
                dict_tensors = {
                    'input': dict_data['x'][0].cpu(),
                    'target': dict_data['y'][0].cpu(),
                    'pred': pred[0].detach().cpu(),
                    'confidence': lc,
                    'margin': -ms,  # minus sign is to draw smaller margin part brighter
                    'entropy': ent
                }

                visualizer(dict_tensors, fp=f"{dir_ckpt}/{num_iter}.png")

        if debug:
            break

    if dir_ckpt is not None:
        header = ["epoch", "miou", "pixel_acc"]
        write_log(f"{dir_ckpt}/log_val.txt", list_entities=[epoch, miou, pixel_acc], header=header)

    print(f"\n{'=' * 100}"
          f"\nExperim name: {experim_name}"
          f"\nEpoch {epoch} | miou: {miou:.3f} | pixel_acc.: {pixel_acc:.3f}"
          f"\n{'=' * 100}\n")
    return miou


if __name__ == '__main__':
    import os
    from args import Arguments
    from utils.utils import Visualiser, get_dataloader

    arguments_parser = Arguments()

    arguments_parser.parser.add_argument(
        "--p_state_dict",
        type=str,
        default='',
        help="path to a state_dict file"
    )

    args = arguments_parser.parse_args(verbose=True)

    dataloader = get_dataloader(
        deepcopy(args), val=True, query=False, shuffle=False, batch_size=1, n_workers=args.n_workers
    )
    visualizer = Visualiser(args.dataset_name)

    os.makedirs(args.dir_checkpoints, exist_ok=True)

    device = torch.device("cuda:0")
    model = get_model(args).to(device)
    state_dict: dict = torch.load(args.p_state_dict)["model"]
    model.load_state_dict(state_dict)

    evaluate(
        model=model,
        dataloader=dataloader,
        dir_ckpt=args.dir_checkpoints,
        experim_name=args.experim_name,
        stride_total=args.stride_total,
        visualizer=visualizer,
        visualize_interval=100,
        device=device
    )
