import os, random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import TrainerPadim
from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split

BATCH_SIZE = 8
IMAGE_INPUT_SIZE = (256, 256)
OUTPUT_SIZE = (256, 256)


def main_train_padim(train_dataset: Dataset, test_dataset: Dataset, category: str, backbone: str, ad_layers: list,
                     device: torch.device, model_checkpoint_save_path: str, diagonal_convergence: bool = False,
                     results_dirpath: str = None):
    padim = Padim(
        backbone,
        category,
        device=device,
        diag_cov=diagonal_convergence,
        layers_idxs=ad_layers,
    )
    padim.to(device)
    trainer = TrainerPadim(
        model=padim,
        device=device,
        save_path=model_checkpoint_save_path,
        data_path=None,
        class_name=category,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True
    )

    trainer.train(train_dataloader)

    # evaluate the model
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    print("Evaluation performances:")
    print(f"""
            img_roc: {img_roc}
            pxl_roc: {pxl_roc}
            f1_img: {f1_img}
            f1_pxl: {f1_pxl}
            img_pr: {img_pr}
            pxl_pr: {pxl_pr}
            pxl_pro: {pxl_pro}
            """)


def test_padim(test_dataset: Dataset, category: str, backbone: str, ad_layers: tuple, device: torch.device,
               model_checkpoint_path: str, results_dirpath: str = None):
    padim = Padim(
        backbone,
        category,
        device=device,
        layers_idxs=ad_layers,
    )
    path = padim.get_model_savepath(model_checkpoint_path)
    padim.load_state_dict(
        torch.load(path, map_location=device), strict=False
    )
    padim.to(device)
    print(f"Loaded model from path: {path}")

    # Evaluator
    padim.eval()

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # evaluate the model
    evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    print("Evaluation performances:")
    print(f"""
        img_roc: {img_roc}
        pxl_roc: {pxl_roc}
        f1_img: {f1_img}
        f1_pxl: {f1_pxl}
        img_pr: {img_pr}
        pxl_pr: {pxl_pr}
        pxl_pro: {pxl_pro}
        """)



def save_results(results_dirpath: str, category: str, seed: int, scores: tuple, backbone: str, ad_layers: tuple,
                 img_input_size: tuple, output_size: tuple):
    metrics_savefile = Path(
        results_dirpath, f"metrics_{backbone}.csv"
    )
    # check if the metrics path exists
    dirpath = os.path.dirname(metrics_savefile)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # save the scores
    append_results(
        metrics_savefile,
        category,
        seed,
        *scores,
        "padim",  # ad_model
        ad_layers,
        backbone,
        "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
        None,  # bootstrap_layer
        -1,  # epochs (not used)
        img_input_size,
        output_size,
    )


def main(args):
    batch_size = args.batch_size  # 32
    save_path = args.save_path  # "output/padim/"
    data_path = args.data_path  # "../datasets/mvtec/"
    device = args.device  # "cuda:1"  # cuda:0, cuda:1, cuda:2, cpu
    backbone_model_name = args.backbone_model_name  # "resnet18"
    save_figures = args.save_figures  # False
    results_dirpath = args.results_dirpath  # "metrics/padim/"
    categories = args.categories
    seeds = args.seeds
    img_input_size = args.img_input_size
    output_size = args.output_size
    ad_layers_idxs = args.ad_layers_idxs

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed_all(seed)

        diag_covs = [False] * len(categories)

        for category_name, diag_cov in zip(categories, diag_covs):

            print(
                "class name:",
                category_name,
                " | diagonal covariance:",
                diag_cov,
                " | save figures:",
                save_figures,
            )

            if args.train:
                print("---- PaDiM train ----")

                padim = Padim(
                    backbone_model_name,
                    category_name,
                    device=device,
                    diag_cov=diag_cov,
                    layers_idxs=ad_layers_idxs,
                )
                padim.to(device)
                trainer = TrainerPadim(
                    model=padim,
                    device=device,
                )

                train_dataset = BMAD(
                    TaskType.SEGMENTATION,
                    data_path,
                    category_name,
                    Split.TRAIN,
                    img_size=img_input_size,
                )

                train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, pin_memory=True
                )

                trainer.train(train_dataloader)

            if args.test:
                print("---- PaDiM test ----")

                # load the model if it was not trained in this run
                if not args.train:
                    padim = Padim(
                        backbone_model_name,
                        category_name,
                        device=device,
                        layers_idxs=ad_layers_idxs,
                    )
                    path = padim.get_model_savepath(save_path)
                    padim.load_state_dict(
                        torch.load(path, map_location=device, weights_only=False), strict=False
                    )
                    padim.to(device)
                    print(f"Loaded model from path: {path}")

                # Evaluator
                padim.eval()

                test_dataset = BMAD(
                    TaskType.SEGMENTATION,
                    data_path,
                    category_name,
                    Split.TEST,
                    image_size=img_input_size,
                )

                test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=True
                )

                # evaluate the model
                evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
                scores = evaluator.evaluate(padim)

                if results_dirpath is not None:
                    metrics_savefile = Path(
                        results_dirpath, f"metrics_{backbone_model_name}.csv"
                    )
                    # check if the metrics path exists
                    dirpath = os.path.dirname(metrics_savefile)
                    if not os.path.exists(dirpath):
                        os.makedirs(dirpath)

                    # save the scores
                    append_results(
                        metrics_savefile,
                        category_name,
                        seed,
                        *scores,
                        "padim",  # ad_model
                        ad_layers_idxs,
                        backbone_model_name,
                        "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
                        None,  # bootstrap_layer
                        -1,  # epochs (not used)
                        args.img_input_size,
                        args.output_size,
                    )


if __name__ == "__main__":
    import argparse

    categories = [
        "brain",
        "chest",
        "histopathology",
        "liver",
        "retinaoct",
        "retinaresc"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figures", action="store_true")
    parser.add_argument("--save_logs", action="store_true")
    parser.add_argument(
        "--backbone_model_name",
        type=str,
        help="resnet18, wide_resnet50_2, mobilenet_v2, mcunet-in3",
    )
    parser.add_argument(
        "--img_input_size",
        type=int,
        default=(256, 256),
        help="input image size, if None, default is used",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=(256, 256),
        help="output image size, if None, default is used",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_path", type=str, default=None, help="where to save the model checkpoint"
    )
    parser.add_argument("--data_path", type=str, default="../../datasets/mvtec/")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--results_dirpath", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--categories", type=str, nargs="+", default=categories)
    parser.add_argument(
        "--ad_layers_idxs",
        type=int,
        nargs="+",
        required=True,
        help="list of layers idxs to use for feature extraction",
    )

    args = parser.parse_args()

    log_filename = "padim.log"
    s = "DEBUG " if args.debug else ""

    try:
        main(args)

        if args.save_logs:
            # create a log file if it does not exist
            if not os.path.exists(log_filename):
                with open(log_filename, "w") as f:
                    f.write("")
            # write the args as a string to the log file
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_filename, "a") as f:
                f.write(s + "finished " + "\t" + now_str + "\t" + str(args) + "\n")

    except Exception as e:
        if args.save_logs:
            # write the args as a string to the log file
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_filename, "a") as f:
                f.write(s + "** FAILED **" + "\t" + now_str + "\t" + str(args) + "\n")
        raise e
