import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.common.common_utils import obsolete
from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator


def train_patchcore(dataset_path: str, category: str, backbone: str, ad_layers: list, save_path: str,
                    device: torch.device, max_dataset_size: int = 1300):
    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)

    print(f"Training Pathcore for category: {category} \n")

    # define training and test datasets
    train_dataset = BMAD(task_type=TaskType.SEGMENTATION, root_dir=dataset_path, category=category, split="train")
    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = BMAD(task_type=TaskType.SEGMENTATION, root_dir=dataset_path, category=category, split="test")
    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # define the model
    patchcore = PatchCore(device, input_size=(256, 256), feature_extractor=feature_extractor)
    patchcore.to(device)
    patchcore.train()

    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device)
    trainer.train()

    # save the model
    if save_path:
        torch.save(patchcore.state_dict(), save_path)

        # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def test_patchcore(dataset_path: str, category: str, backbone: str, ad_layers: list, model_checkpoint_path: str,
                   device: torch.device, max_dataset_size: int = 800, visual_test_path: str = None):
    test_dataset = BMAD(task_type=TaskType.SEGMENTATION, root_dir=dataset_path, category=category, split="test")

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    patchcore = PatchCore(device, input_size=(256, 256), feature_extractor=feature_extractor)
    patchcore.load_model(model_checkpoint_path)
    patchcore.to(device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore)

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

    # chek for the visual test
    if visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = patchcore(images.to(device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                patchcore.save_anomaly_map(visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                           labels[i], masks[i])



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    if args.mode == "train":
        train_patchcore(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device)
    elif args.mode == "test":
        test_patchcore(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, device, None, args.visual_test_path)


if __name__ == "__main__":
    main()
