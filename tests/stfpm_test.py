from moviad.trainers.trainer_stfpm import TrainerSTFPM
from moviad.models.stfpm.stfpm import STFPM
from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from torch.utils.data import DataLoader
import torch
import gc
import os
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
import numpy as np
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
import torch

import json

import torch.nn.functional as F

def train(dataset_path="/mnt/disk1/ruslan_nuriev/bmad"):
    params = {
        "dataset_path": dataset_path,
        "ad_layers": ["layer1", "layer2", "layer3"],
        "epochs": 10,
        "batch_size": 32,
        "backbone_model_name": "resnet18",
        "device": "cuda:0",
    }
    best_results = {}
    for category in CATEGORIES:

        print(f"Training with params: {params} and category of {category}")
        train_data = BMAD("segmentation", params["dataset_path"], category, "train", image_size=(224, 224))
        train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        test_data = BMAD("segmentation", params["dataset_path"], category, "test", image_size=(224, 224))
        test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

        student = CustomFeatureExtractor(params["backbone_model_name"], params["ad_layers"], device=params['device'], frozen=False)
        teacher = CustomFeatureExtractor(params['backbone_model_name'], params["ad_layers"], device=params['device'])

        model = STFPM(teacher, student)
        model.train()
        trainer = TrainerSTFPM(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=test_dataloader,
            device=params['device'],
            logger=None,
            )
        
        _, best_result = trainer.train(params['epochs'])
        best_results[category] = {"img_roc_auc": best_result.img_roc_auc,
                                  "pxl_roc_auc": best_result.pxl_roc_auc,
                                  "pxl_pro": best_result.pxl_au_pro}
        torch.save(model.student.model.state_dict(), f"/home/ruslan/thesis/checkpoints/model_weights_student_{category}.pth")
        torch.save(model.teacher.model.state_dict(), f"/home/ruslan/thesis/checkpoints/model_weights_teacher_{category}.pth")

        print(f"For '{category}' category, best results are:\nImage AUROC: {best_results[category]['img_roc_auc']}\nPixel AUROC: {best_results[category]['pxl_roc_auc']}\nPixel PRO: {best_results[category]['pxl_pro']}")
        del model
        del test_data
        del train_data
        del train_dataloader
        del test_dataloader
        torch.cuda.empty_cache()
        gc.collect()

    with open('/home/ruslan/thesis/tests/file.txt', 'w') as file:
        file.write(json.dumps(best_results))

def create_binary_anomaly_map_f1(anomaly_maps, ground_truths):
    if isinstance(anomaly_maps, torch.Tensor):
        anomaly_maps = anomaly_maps.cpu().detach()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.cpu().detach()
    
    binary_maps = []
    
    for (anomaly_map, ground_truth) in zip(anomaly_maps, ground_truths):
        scores_flat = anomaly_map.flatten()
        gt_flat = ground_truth.flatten()

        precision, recall, thresholds = precision_recall_curve(gt_flat, scores_flat)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # might be zero, supposed to handle it
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        binary_map = (anomaly_map >= best_threshold).numpy().astype(int)
        binary_maps.append(binary_map)
    
    return binary_maps

def visualize_anomaly_comparison(predicted_maps, gt_maps, original_images=None, n_samples=3):
    if isinstance(predicted_maps, torch.Tensor):
        predicted_maps = predicted_maps.cpu().detach()
    if isinstance(gt_maps, torch.Tensor):
        gt_maps = gt_maps.cpu().detach()
    
    if len(predicted_maps.shape) == 4:
        predicted_maps = predicted_maps.squeeze(1)
    if len(gt_maps.shape) == 4:
        gt_maps = gt_maps.squeeze(1)
    
    n_samples = min(n_samples, predicted_maps.shape[0])
    
    # 5 columns: original, predicted, GT, pred overlay, GT overlay
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        pred_map = predicted_maps[i].numpy()
        gt_map = gt_maps[i].numpy()
        
        # Original
        if original_images is not None:
            img = original_images[i].cpu().detach()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = img.squeeze().numpy()
            
            axes[i, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
        
        # Predicted
        im1 = axes[i, 1].imshow(pred_map, cmap='jet')
        axes[i, 1].set_title('Predicted')
        axes[i, 1].axis('off')
        
        # Ground Truth
        im2 = axes[i, 2].imshow(gt_map, cmap='jet')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Overlay on original - Predicted
        if original_images is not None:
            axes[i, 3].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i, 3].imshow(pred_map, cmap='jet', alpha=0.5)
            axes[i, 3].set_title('Pred Overlay')
            axes[i, 3].axis('off')
            
            # Overlay on original - GT
            axes[i, 4].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i, 4].imshow(gt_map, cmap='jet', alpha=0.5)
            axes[i, 4].set_title('GT Overlay')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig("/home/ruslan/thesis/tests/ex.png")
    return fig

def visualize_binary_map(predicted_maps, gt_maps, n_samples):
    if isinstance(predicted_maps, torch.Tensor):
        predicted_maps = predicted_maps.cpu().detach()
    if isinstance(gt_maps, torch.Tensor):
        gt_maps = gt_maps.cpu().detach()


    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_samples):
        pred_map = predicted_maps[i]
        gt_map = gt_maps[i].numpy()
        
        # Predicted
        im1 = axes[i, 0].imshow(pred_map.squeeze(0), cmap='jet')
        axes[i, 0].set_title('Predicted')
        axes[i, 0].axis('off')
        
        # Ground Truth
        im2 = axes[i, 1].imshow(gt_map.squeeze(0), cmap='jet')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/home/ruslan/thesis/tests/ex_binary.png")
    return fig


def test_on_some_instances(dataset_path="/mnt/disk1/ruslan_nuriev/bmad"):
    layers = ["layer1", "layer2", "layer3"]
    student = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0") #128 - 512
    teacher = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0")

    test_data = BMAD("segmentation", dataset_path, "retinaresc", "test", image_size=(224, 224))
    test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
    img = next(iter(test_dataloader))


    model = STFPM(teacher, student)
    model.student.model.load_state_dict(torch.load("thesis/checkpoints/model_weights_student_retinaresc.pth", weights_only=False))
    model.teacher.model.load_state_dict(torch.load("thesis/checkpoints/model_weights_teacher_retinaresc.pth", weights_only=False))
    model.eval()

    output, anomaly_scores = model(img[0].to("cuda:0"))

    visualize_anomaly_comparison(predicted_maps=output, gt_maps=img[2], original_images=img[0], n_samples=5)

    bin_maps = create_binary_anomaly_map_f1(output, img[2])

    visualize_binary_map(bin_maps, img[2], n_samples=5)
    print(f"Anomaly scores: {anomaly_scores}")

def visualize_train_data(dataset_path="/mnt/disk1/ruslan_nuriev/bmad"):
    layers = ["layer1", "layer2", "layer3"]
    student = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0") #128 - 512
    teacher = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0")

    test_data = BMAD("segmentation", dataset_path, "retinaresc", "train", image_size=(224, 224))
    test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
    img = next(iter(test_dataloader))


    model = STFPM(teacher, student)
    model.student.model.load_state_dict(torch.load("thesis/checkpoints/model_weights_student_retinaresc.pth", weights_only=False))
    model.teacher.model.load_state_dict(torch.load("thesis/checkpoints/model_weights_teacher_retinaresc.pth", weights_only=False))
    model.eval()

    anomaly_map, anomaly_score = model(img.to("cuda:0"))

    visualize_anomaly_comparison(predicted_maps=anomaly_map, gt_maps=img.permute(0, 2, 3, 1), original_images=None, n_samples=5)
    print(f"Anomaly scores: {anomaly_score}")

    # bin_maps = create_binary_anomaly_map_f1(output, img[2])


def test():
    pass

if __name__ == "__main__":
    train()
    # test_on_some_instances()
    # visualize_train_data()
