from typing import Optional, List
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval import evaluate
from query import UncertaintySampler, gather_previous_query_files, merge_previous_query_files
from utils.metrics import AverageMeter, RunningScore
from utils.utils import write_log, get_dataloader, get_model, get_optimizer, get_lr_scheduler


def train_epoch(
        epoch,
        dataloader,
        model,
        optimizer,
        lr_scheduler,
        loss_tracker,
        experim_name: str,
        dir_ckpt: Optional[str] = None,
        visualizer: Optional[callable] = None,
        visualize_interval: Optional[int] = 100,
        human_labels: bool = False,
        device: torch.device = torch.device("cuda:0"),
        debug: bool = False
):
    if dir_ckpt is not None:
        dir_ckpt = f"{dir_ckpt}/e{epoch:02d}"
        os.makedirs(dir_ckpt, exist_ok=True)

    ignore_index: int = dataloader.dataset.ignore_index
    miou_tracker = RunningScore(dataloader.dataset.n_classes)

    dataloader_iter, tbar = iter(dataloader), tqdm(range(len(dataloader)))
    model.train()

    for num_iter in tbar:
        dict_data = next(dataloader_iter)
        x: torch.Tensor = dict_data['x'].to(device)

        # if human-labelled annotations are given
        if human_labels:
            y: torch.Tensor = dict_data["labelled_queries"].to(device, torch.int64)

        else:
            y: torch.Tensor = dict_data['y'].to(device)
            mask: torch.Tensor = dict_data['queries'].to(device, torch.bool)
            y.flatten()[~mask.flatten()] = ignore_index

        # forward pass
        dict_outputs = model(x)

        logits = dict_outputs["pred"]
        dict_losses = {"ce": F.cross_entropy(logits, y, ignore_index=ignore_index)}

        # backward pass
        loss = sum(dict_losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prob, pred = torch.softmax(logits.detach(), dim=1), logits.argmax(dim=1)
        miou_tracker.update(y.cpu().numpy(), pred.cpu().numpy())
        loss_tracker.update(loss.detach().item())

        scores = miou_tracker.get_scores()[0]
        miou, pixel_acc = scores['Mean IoU'], scores['Pixel Acc']

        # description
        description = f"({experim_name}) Epoch {epoch} | mIoU.: {miou:.3f} | pixel acc.: {pixel_acc:.3f} | " \
                      f"avg loss: {loss_tracker.avg:.3f}"
        for loss_k, loss_v in dict_losses.items():
            description += f" | {loss_k}: {loss_v.detach().cpu().item():.3f}"
        tbar.set_description(description)

        lr_scheduler.step(epoch=epoch - 1)

        if dir_ckpt is not None and visualizer is not None and num_iter % visualize_interval == 0:
            if visualizer is not None:
                ent, lc, ms, = [getattr(UncertaintySampler, f"_{uc}")(prob)[0].cpu() for uc in
                                ["entropy", "least_confidence", "margin_sampling"]]

                y = dict_data.get('y', None)
                dict_tensors = {
                    'input': dict_data['x'][0].cpu(),
                    'target': y[0].cpu() if y is not None else None,
                    'pred': pred[0].detach().cpu(),
                    'confidence': lc,
                    'margin': -ms,  # minus sign is to draw smaller margin part brighter
                    'entropy': ent
                }

                visualizer(dict_tensors, fp=f"{dir_ckpt}/{num_iter}.png")

        if debug:
            break

    if dir_ckpt is not None:
        header = ["epoch", "miou", "pixel_acc", "loss"] if epoch == 1 else None
        write_log(f"{dir_ckpt}/log_train.txt", list_entities=[epoch, miou, pixel_acc, loss_tracker.avg], header=header)
    return model, optimizer, lr_scheduler


def train(
        args,
        dataloader: torch.utils.data.DataLoader,
        eval_interval: int = 0,
        dir_ckpt: Optional[str] = None,
        visualizer: Optional[callable] = None,
        visualize_interval: int = 100,
        human_labels: bool = False,
        device: torch.device = torch.device("cuda:0")
):
    """
    :param args:
    :param dataloader: a dataloader instance
    :param eval_interval: If > 0, evaluate the model every evel_interval epochs. If 0, no evaluation is made during
    training.
    :param device:
    :return:
    """
    debug: bool = args.debug
    experim_name: str = args.experim_name
    n_epochs: int = args.n_epochs
    print(f"\n({experim_name}) training...\n")

    model = get_model(args).to(device)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader))

    loss_tracker = AverageMeter()

    best_miou = -1.0
    for e in range(1, 1 + n_epochs):
        model, optimizer, lr_scheduler = train_epoch(
            epoch=e,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_tracker=loss_tracker,
            experim_name=experim_name,
            dir_ckpt=dir_ckpt,
            human_labels=human_labels,
            visualizer=visualizer,
            visualize_interval=visualize_interval,
            device=device,
            debug=debug
        )

        if eval_interval > 0 and e % eval_interval == 0:
            eval_dataloader = get_dataloader(
                deepcopy(args), val=True, query=False, shuffle=False, batch_size=1, n_workers=args.n_workers
            )

            current_miou = evaluate(
                model=model,
                dataloader=eval_dataloader,
                experim_name=experim_name,
                epoch=e,
                dir_ckpt=dir_ckpt,
                visualizer=visualizer,
                visualize_interval=visualize_interval,
                stride_total=args.stride_total,
                device=device,
                debug=debug
            )

            if current_miou > best_miou:
                torch.save({"model": model.state_dict()}, f"{dir_ckpt}/best_model.pt")

        if debug:
            break
    return model


if __name__ == '__main__':
    import os
    from typing import Dict
    from torch.utils.data import DataLoader
    from args import Arguments
    from utils.utils import Visualiser, get_dataloader

    arguments_parser = Arguments()
    arguments_parser.parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="how frequently the model is evaluated in epoch. Set 0 for not evaluating the model during training."
    )
    arguments_parser.parser.add_argument(
        "--p_dataset_config", '-pdc', type=str, default="/Users/noel/Desktop/pixelpick/datasets/configs/custom.yaml"
    )

    args = arguments_parser.parse_args(verbose=True)

    list_prev_query_files: Optional[List[str]] = None
    if args.dir_checkpoints == '':
        dataloader = get_dataloader(args=args, batch_size=args.batch_size, shuffle=True, n_workers=args.n_workers)
        nth_query = 0

    else:
        list_prev_query_files = gather_previous_query_files(args.dir_checkpoints)
        img_path_to_merged_query: Dict[str, np.ndarray] = merge_previous_query_files(
            list_prev_query_files, ignore_index=args.ignore_index
        )

        # change the path
        list_inputs: List[str] = list()
        list_merged_queries: List[np.ndarray] = list()
        for p_img, merged_query in sorted(img_path_to_merged_query.items()):
            filename: str = p_img.split('/')[-1]
            p_img = f"{args.dir_dataset}/train/{filename}"
            assert os.path.exists(p_img)
            list_inputs.append(p_img)
            list_merged_queries.append(merged_query)

        # change the images that the dataloader loads depending on the images that have annotation.
        dataset = get_dataloader(
            args=args, batch_size=args.batch_size, shuffle=True, n_workers=args.n_workers, generate_init_queries=False
        ).dataset

        dataset.list_inputs = list_inputs
        dataset.update_labelled_queries(list_merged_queries)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
            drop_last=len(dataset) % args.batch_size == 1
        )

        nth_query: int = len(list_prev_query_files) - 1

    visualizer = Visualiser(args.dataset_name)

    dir_ckpt = args.dir_checkpoints
    args.dir_checkpoints = f"{dir_ckpt}/{nth_query}_query" if args.n_pixels_by_us > 0 else dir_ckpt

    os.makedirs(args.dir_checkpoints, exist_ok=True)

    train(
        args,
        dataloader,
        eval_interval=args.eval_interval,
        dir_ckpt=args.dir_checkpoints,
        visualizer=visualizer,
        visualize_interval=100,
        human_labels=list_prev_query_files is not None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )