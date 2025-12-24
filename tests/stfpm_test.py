from moviad.trainers.trainer_stfpm import TrainerSTFPM
from moviad.models.stfpm.stfpm import STFPM
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from torch.utils.data import DataLoader
from moviad.models.fastflow.fastflow import create_fastflow
from anomalib.models.image.reverse_distillation.torch_model import ReverseDistillationModel
from torch.utils.data import Subset
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
        "epochs": 30,
        "batch_size": 32,
        "backbone_model_name": "resnet18",
        "device": "cuda:0",
    }
    category = "liver"
    best_results = {}

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
                                "pxl_pro": best_result.pxl_au_pro} #/home/ruslan/thesis/tests/stfpm_tst
    torch.save(model.student.model.state_dict(), f"/home/ruslan/thesis/tests/stfpm_tst/model_weights_student_{category}.pth")
    torch.save(model.teacher.model.state_dict(), f"/home/ruslan/thesis/tests/stfpm_tst/model_weights_teacher_{category}.pth")

    print(f"For '{category}' category, best results are:\nImage AUROC: {best_results[category]['img_roc_auc']}\nPixel AUROC: {best_results[category]['pxl_roc_auc']}\nPixel PRO: {best_results[category]['pxl_pro']}")
    del model
    del test_data
    del train_data
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def create_binary_anomaly_map_f1(anomaly_maps, ground_truths):
    if isinstance(anomaly_maps, torch.Tensor):
        anomaly_maps = anomaly_maps.cpu().detach()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.cpu().detach()
    
    binary_maps = []
    
    # for (anomaly_map, ground_truth) in zip(anomaly_maps, ground_truths):
    #     scores_flat = anomaly_map.flatten()
    #     gt_flat = ground_truth.flatten()

    #     precision, recall, thresholds = precision_recall_curve(gt_flat, scores_flat)
    #     f1_scores = 2 * (precision * recall) / (precision + recall)
    #     f1_scores = np.nan_to_num(f1_scores)  # might be zero, supposed to handle it
        
    #     best_idx = np.argmax(f1_scores)
    #     best_threshold = thresholds[best_idx]

    #     binary_map = (anomaly_map >= best_threshold).numpy().astype(int)
    #     binary_maps.append(binary_map)
    def min_max_scale(anomaly_map):
        min_val, max_val = anomaly_map.min(), anomaly_map.max()
        return (anomaly_map - min_val) / (max_val - min_val)

    # Inside your loop, before flattening:
    # anomaly_map = min_max_scale(anomaly_map)
    for (anomaly_map, ground_truth) in zip(anomaly_maps, ground_truths):
        anomaly_map = min_max_scale(anomaly_map)
        scores_flat = anomaly_map.flatten()
        gt_flat = ground_truth.flatten()

        precision, recall, thresholds = precision_recall_curve(gt_flat, scores_flat)
        
        beta = 0.8
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        f_beta = np.nan_to_num(f_beta)
        
        best_idx = np.argmax(f_beta)
        best_threshold = thresholds[best_idx]

        binary_map = (anomaly_map >= best_threshold).numpy().astype(int)
        binary_maps.append(binary_map)

    return binary_maps

def visualize_anomaly_comparison(predicted_maps, gt_maps, original_images=None, n_samples=3):
    """
    Visualizes anomaly detection results with 5 columns:
    1. Original image
    2. Ground truth mask
    3. Predicted heatmap (mask)
    4. Predicted binary map
    5. Overlay of predicted binary map onto the original image
    """
    if isinstance(predicted_maps, torch.Tensor):
        predicted_maps = predicted_maps.cpu().detach()
    if isinstance(gt_maps, torch.Tensor):
        gt_maps = gt_maps.cpu().detach()
    
    if len(predicted_maps.shape) == 4:
        predicted_maps = predicted_maps.squeeze(1)
    if len(gt_maps.shape) == 4:
        gt_maps = gt_maps.squeeze(1)
    
    n_samples = min(n_samples, predicted_maps.shape[0])
    
    # Create binary maps using F1-optimal threshold
    binary_maps = create_binary_anomaly_map_f1(predicted_maps, gt_maps)
    
    # 5 columns: original, GT mask, predicted heatmap, predicted binary map, binary overlay
    fig, axes = plt.subplots(n_samples, 5, figsize=(25, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        pred_heatmap = predicted_maps[i].numpy()
        gt_map = gt_maps[i].numpy()
        binary_map = binary_maps[i]
        
        # Column 1: Original Image
        if original_images is not None:
            img = original_images[i].cpu().detach()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = img.squeeze().numpy()
            
            axes[i, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
        else:
            axes[i, 0].text(0.5, 0.5, 'No Image', ha='center', va='center')
            axes[i, 0].axis('off')
        
        # Column 2: Ground Truth Mask
        im1 = axes[i, 1].imshow(gt_map, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Column 3: Predicted Heatmap (normalized)
        im2 = axes[i, 2].imshow(pred_heatmap, cmap='jet')
        axes[i, 2].set_title('Predicted Heatmap')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Column 4: Predicted Binary Map
        im3 = axes[i, 3].imshow(binary_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('Predicted Binary Map')
        axes[i, 3].axis('off')
        
        # Column 5: Overlay of Binary Map on Original
        if original_images is not None:
            axes[i, 4].imshow(img, cmap='gray' if img.ndim == 2 else None)
            # Create a masked array for the binary map overlay
            masked_binary = np.ma.masked_where(binary_map == 0, binary_map)
            axes[i, 4].imshow(masked_binary, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[i, 4].set_title('Binary Map Overlay')
            axes[i, 4].axis('off')
        else:
            axes[i, 4].text(0.5, 0.5, 'No Image', ha='center', va='center')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig("/home/ruslan/thesis/tests/ex.png", dpi=150, bbox_inches='tight')
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
    # layers = ["layer1", "layer2", "layer3"]
    torch.manual_seed(100) #5 101 123 31


    test_data = BMAD("segmentation", dataset_path, "liver", "test", image_size=(224, 224))
    positive_indices = [i for i in range(len(test_data)) if test_data[i][1] == 1]
    positive_data = Subset(test_data, positive_indices)

    test_dataloader = DataLoader(positive_data, batch_size=2, shuffle=True)
    img = next(iter(test_dataloader))

    student = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0", frozen=False) #128 - 512
    teacher = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device="cuda:0")
    model = STFPM(teacher, student)
    model.student.model.load_state_dict(torch.load("/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_student_cl_cont.pth", weights_only=True))
    model.teacher.model.load_state_dict(torch.load("/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_teacher_cl_cont.pth", weights_only=True))

    # # model = ReverseDistillationModel("wide_resnet50_2", (224, 224), ["layer1", "layer2", "layer3"], "multiply")
    # # model.to("cuda:0")
    # # model.load_state_dict(torch.load("/home/ruslan/thesis/tests/rd4ad_wide/brain.pth", weights_only=False))
    # model = create_fastflow((224, 224), "wide_resnet50_2", "cuda:0")
    # model.load_state_dict(torch.load("/home/ruslan/thesis/tests/checkpoints/fastflow_continual_learning.pth", weights_only=False))
    model.eval()

    output, anomaly_scores = model(img[0].to("cuda:0"))

    visualize_anomaly_comparison(predicted_maps=output, gt_maps=img[2], original_images=img[0], n_samples=5)

    bin_maps = create_binary_anomaly_map_f1(output, img[2])

    visualize_binary_map(bin_maps, img[2], n_samples=2)
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

def train_mvtec_fastflow():
    from tqdm import tqdm
    from moviad.trainers.trainer_fastflow import TrainerFastFlow
    epochs = 400
    model = create_fastflow((224, 224), "resnet18", "cuda:0")
    # model.load_state_dict(torch.load("/home/ruslan/thesis/tests/checkpoints/fastflow_single_brain.pth", weights_only=False))
    model.train()

    data_train = MVTecDataset('segmentation', '/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec', "bottle", "train")
    data_train.load_dataset()
    data_train = torch.utils.data.DataLoader(data_train, batch_size=16)
    data_test = MVTecDataset('segmentation', '/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec', "bottle", "test")
    data_test.load_dataset()
    data_test = torch.utils.data.DataLoader(data_test, batch_size=5)

    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=1e-5)

    for epoch in range(epochs):
            epoch_losses = []
            with tqdm(data_train, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch

                    images = images.to("cuda:0")

                    hidden_variables, jacobians = model(images)
                    loss = TrainerFastFlow.fastflow_loss(hidden_variables, jacobians)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "/home/ruslan/thesis/tests/fast_mv.pth")

def test_mvtec(dataset_path="/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec"):
    data_test = MVTecDataset('segmentation', '/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec', "bottle", "test")
    data_test.load_dataset()
    data_test = torch.utils.data.DataLoader(data_test, batch_size=5)
    img = next(iter(data_test))
    model = create_fastflow((224, 224), "resnet18", "cuda:0")
    model.load_state_dict(torch.load("/home/ruslan/thesis/tests/fast_mv.pth", weights_only=False))
    model.eval()

    output, anomaly_scores = model(img[0].to("cuda:0"))

    visualize_anomaly_comparison(predicted_maps=output, gt_maps=img[2], original_images=img[0], n_samples=5)

    bin_maps = create_binary_anomaly_map_f1(output, img[2])

    visualize_binary_map(bin_maps, img[2], n_samples=5)
    print(f"Anomaly scores: {anomaly_scores}")

def evalll():
    from moviad.utilities.evaluator import Evaluator
    test_data = BMAD("segmentation", "/mnt/disk1/ruslan_nuriev/bmad", "liver", "test", image_size=(224, 224))
    test_loader = torch.utils.data.DataLoader(test_data, 32)
    model = create_fastflow((224, 224), "resnet18", "cuda:0")
    model.load_state_dict(torch.load("/home/ruslan/thesis/tests/checkpoints/fastflow_single_liver_exp.pth", weights_only=False))
    model.eval()

    evaluator = Evaluator(test_dataloader=test_loader, device="cuda:0")
    results = evaluator.evaluate(model)
    print("evaluating")
    print(results)



if __name__ == "__main__":
    # train()
    test_on_some_instances()
    # visualize_train_data()
    # train_mvtec_fastflow()
    # test_mvtec()
    # evalll()
