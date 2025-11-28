from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.trainers.trainer_supersimplenet import TrainerSuperSimpleNet
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.models

import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb
import numpy as np

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.trainers.trainer_rd4ad import TrainerRD4AD
from moviad.utilities.configurations import TaskType, Split

def train_ssn(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
        # define training and test datasets
    train_dataset = BMAD(TaskType.SEGMENTATION, "/mnt/disk1/ruslan_nuriev/bmad", "liver", "train")
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = BMAD(TaskType.SEGMENTATION, "/mnt/disk1/ruslan_nuriev/bmad", "liver", "test")
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = "cuda:2"
    xtrctr = CustomFeatureExtractor("wide_resnet50_2", ["layer2", "layer3"], device).model
    # backbone = getattr(torchvision.models, "wide_resnet50_2")(weights = "IMAGENET1K_V1")
    # extractor = create_feature_extractor(backbone, return_nodes=["layer2", "layer3"])

    # define the model
    model = SuperSimpleNet(xtrctr)
    model.to(device)
    model.train()

    trainer = TrainerSuperSimpleNet(model, train_dataloader, test_dataloader, device)
    trainer.train(epochs=10, evaluation_epoch_interval=10)
    
    del model
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    train_ssn()