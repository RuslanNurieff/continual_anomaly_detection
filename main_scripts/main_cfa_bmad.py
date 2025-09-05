import random
import argparse
import pathlib

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.cfa.cfa import CFA
from moviad.trainers.trainer_cfa import TrainerCFA
from moviad.utilities.configurations import TaskType
from moviad.utilities.evaluator import Evaluator


def main_train_cfa(dataset_path: str, category: str, backbone: str, ad_layers: list,
                   epochs: int, save_path: str, device: torch.device):
    gamma_c = 1
    gamma_d = 1

    print(f"Training CFA for category: {category} \n")

    train_dataset = BMAD(TaskType.SEGMENTATION, dataset_path, category, "train")
    train_dataset.load_dataset()
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

    test_dataset = BMAD(TaskType.SEGMENTATION, dataset_path, category, "test")
    test_dataset.load_dataset()
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)

    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device)

    cfa_model = CFA(feature_extractor, backbone, device)
    cfa_model.initialize_memory_bank(train_dataloader)
    cfa_model = cfa_model.to(device)

    trainer = TrainerCFA(cfa_model, feature_extractor, train_dataloader, test_dataloader, device)
    trainer.train(epochs)

    # save the model
    if save_path:
        torch.save(cfa_model.state_dict(), save_path)


def main_test_cfa(dataset_path: str, category: str, backbone: str, ad_layers: list, model_checkpoint_path: str,
                  visual_test_path: str, device: torch.device):
    gamma_c = 1
    gamma_d = 1

    test_dataset = BMAD(TaskType.SEGMENTATION, dataset_path, category, "test")
    test_dataset.load_dataset()
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    cfa_model = CFA(feature_extractor, backbone, device)
    cfa_model.load_model(model_checkpoint_path)
    cfa_model.to(device)
    cfa_model.eval()

    evaluator = Evaluator(test_dataloader, device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(cfa_model)

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
            anomaly_maps, pred_scores = cfa_model(images.to(device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                cfa_model.save_anomaly_map(dirpath, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i], labels[i],
                                           masks[i])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
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
        main_train_cfa(args.dataset_path, args.category, args.backbone, args.ad_layers, args.epochs, args.save_path, device)
    elif args.mode == "test":
        main_test_cfa(args.dataset_path, args.category, args.backbone, args.ad_layers, args.save_path, args.visual_test_path,
                      device)


if __name__ == "__main__":
    main()
