import os
from typing import Any, Dict, Union, List
import json
import pickle as pkl
from PIL import Image
from time import time


def read_via_annotation(fp: str, verbose: bool = True) -> dict:
    via_annot: dict = json.load(open(fp, 'r'))
    if verbose:
        for k, v in via_annot.items():
            print(f"=========={k}==========")
            for _k, _v in v.items():
                print(_k, _v)
            print('========================\n')
    return via_annot


def convert_annotation(
        via_annot: dict,
        k_to_category: Dict[str, str],
        k_to_category_id: Dict[str, int],
        verbose: bool = True
) -> dict:
    """
    :return converted_annot: {
        filepath (str): {
            "height": int,
            "width": int,
            "x_coords": List[int],
            "y_coords": List[int],
            "category": List[str],
            "category_id": List[int]
        }
    }
    """
    file_info: Dict[int, Dict[str, Union[int, str]]] = via_annot["file"]
    metadata: Dict[str, Dict[str, Union[dict, int, list, str]]] = via_annot["metadata"]

    converted_annot: dict = dict()
    for identifier, annot in metadata.items():
        vid: str = annot["vid"]
        filepath: str = file_info[vid]["src"]
        x: int = annot["xy"][1]
        y: int = annot["xy"][2]

        try:
            key: str = list(annot["av"].values())[0]
        except IndexError:
            if verbose:
                print(f"WARNING: A pixel label at [{x}, {y}] (x, y) for {filepath} was not entered.")
            continue

        try:
            converted_annot[filepath]["x_coords"].append(x)
            converted_annot[filepath]["y_coords"].append(y)
            converted_annot[filepath]["category"].append(k_to_category[key].lower())
            converted_annot[filepath]["category_id"].append(k_to_category_id[key])

        except KeyError:
            w, h = Image.open(filepath).size
            converted_annot.update({
                filepath: {
                    "height": h,
                    "width": w,
                    "x_coords": [x],
                    "y_coords": [y],
                    "category": [k_to_category[key].lower()],
                    "category_id": [k_to_category_id[key]]
                }
            })
    return converted_annot


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    import yaml
    from pprint import pprint
    # from dataset_conf import get_keyword_to_category_mapping
    parser = ArgumentParser("A Format Converter from VIA annotation to PixelPick annotation")
    parser.add_argument(
        "--via_annot_file", '-vaf', type=str, help="a path to VIA annotation file with json format",
        default="../via_project_13Sep2021_23h05m22s.json"
    )
    parser.add_argument(
        "--p_dataset_config", '-pdc', type=str, default="/Users/noel/projects/PixelPick/datasets/configs/custom.yaml"
    )
    parser.add_argument("--converted_file", '-cf', type=str, help="a resulting file name after conversion")
    parser.add_argument("--dataset_name", '-dn', type=str, default="camvid", help="a dataset name")
    parser.add_argument("--verbose", '-v', action="store_true", help="verbosity")

    args = parser.parse_args()

    assert os.path.exists(args.p_dataset_config), FileNotFoundError(args.p_dataset_config)
    # args: Namespace = parser.parse_args()
    dataset_config = yaml.safe_load(open(f"{args.p_dataset_config}", 'r'))
    args: dict = vars(args)
    args.update(dataset_config)
    args: Namespace = Namespace(**args)

    st = time()

    via_annot: dict = read_via_annotation(args.via_annot_file, verbose=args.verbose)

    k_to_category, k_to_category_id = args.mapping, args.k_to_category_id
    # k_to_category, k_to_category_id = get_keyword_to_category_mapping(dataset_name=args.dataset_name)
    converted_annot: dict = convert_annotation(
        via_annot,
        k_to_category=k_to_category,
        k_to_category_id=k_to_category_id,
        verbose=args.verbose
    )

    if args.converted_file is not None:
        fp: str = args.converted_file
    else:
        fp: str = args.via_annot_file.replace("json", "pkl")
    pkl.dump(converted_annot, open(fp, "wb"))

    if args.verbose:
        print()
        print('=' * 80)
        print("Converted annotations...")
        for k, v in converted_annot.items():
            print("\nFile path:", k)
            pprint(v)
        print('=' * 80)

    print(f"Conversion completed ({time() - st:.3f} sec.) and saved to '{os.path.abspath(fp)}'.")
