from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.models.draem.augmentation import DRAEMTrain
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES

from moviad.models.draem.draem import DRAEM
from moviad.trainers.trainer_draem import TrainerDRAEM

import torch

import pandas as pd

def set_up_draem(train_loader, test_loader, device):
    model = DRAEM(device)
    trainer = TrainerDRAEM(model, train_loader, test_loader, device)
    return trainer

def set_up_dataset(category):
    mvtec_train = MVTecDataset("segmentation", "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec", category, "train", norm=False, img_size=(256, 256))
    mvtec_train.load_dataset()
    mvtec_test = MVTecDataset("segmentation", "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec", category, "test", norm=False, img_size=(256, 256))
    mvtec_test.load_dataset()

    draem_train = DRAEMTrain(mvtec_train, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", split="train", resize_shape=[256, 256])
    draem_train = torch.utils.data.DataLoader(draem_train, batch_size=8, shuffle=True)
    draem_test = DRAEMTrain(mvtec_test, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", split="test", resize_shape=[256, 256])
    draem_test = torch.utils.data.DataLoader(draem_test, batch_size=8)
    return draem_train, draem_test


def main(epochs=50):

    results = pd.DataFrame(columns=[category for category in CATEGORIES])

    for category in CATEGORIES:
        print(f"Training for {category}")
        draem_train, draem_test = set_up_dataset(category)
        trainer = set_up_draem(draem_train, draem_test, "cuda:0")
        _, best_results = trainer.train(epochs, 10)

        row_to_add = {category: {
        "img_roc_auc": best_results.img_roc_auc,
        "pxl_roc_auc": best_results.pxl_roc_auc,
        "img_f1": best_results.img_f1,
        "pxl_f1": best_results.pxl_f1,
        "img_pr_auc": best_results.img_pr_auc,
        "pxl_pr_auc": best_results.pxl_pr_auc,
        "pxl_au_pro": best_results.pxl_au_pro
        }}
        
        row_to_add = pd.DataFrame(row_to_add).T
        results = pd.concat([results, row_to_add], axis=0)
    
    results.to_csv("final_results.csv")
    
if __name__ == "__main__":
    main()