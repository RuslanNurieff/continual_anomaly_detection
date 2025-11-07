from datasets.full_dataset import CombinedDataset

import json

from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.utilities.evaluator import Evaluator

from moviad.models.draem.augmentation import DRAEMContinualDataset

from tqdm import tqdm
import numpy as np
import torch
from trainers.models import STFPMModel, RD4ADModel, DRAEMModel, FastFlowModel

import torch.nn.functional as F
import wandb


def STFPM():

        # Initialize wandb
    wandb.init(
        project="joint-training",
        name="stfpm-joint",
        config={
            "model": "STFPM",
            "backbone": "wide_resnet50_2",
            "epochs": 50,
            "device": "cuda:1",
            "input_size": (224, 224),
            "dataset": "BMAD"
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    data_test = data_full.load_test()
    # combined_test_loader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False)
    categories = data_full.categories
    # print(f"test batch: {next(iter(combined_test_loader))}")

    # Create model and strategy
    model_stfpm = STFPMModel("cuda:1", 'wide_resnet50_2', ["layer1", "layer2", "layer3"])
    model_stfpm.load_model()
    model_stfpm.ad_model.train()

    for epoch in range(50):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{50}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_stfpm.device)

                teacher_features, student_features = model_stfpm.ad_model(images)
                loss = 0
                for i in range(len(student_features)):
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)
                    loss += model_stfpm.loss(teacher_features[i], student_features[i])
                
                model_stfpm.optimizer.zero_grad()
                loss.backward()
                model_stfpm.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{50}, Loss: {avg_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

    model_stfpm.ad_model.eval()

    results = {}
    for task_id, test_task in enumerate(data_test):
        cat = categories[task_id]
        test_dl = torch.utils.data.DataLoader(test_task, batch_size=32, shuffle=False)
        evaluator = Evaluator(test_dl, model_stfpm.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_stfpm.ad_model)
        results[cat] = metrics

        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    with open('/home/ruslan/thesis/tests/joint_stfpm.txt', 'w') as file:
        file.write(json.dumps(results))
    
def RD4AD():
    
    # Initialize wandb
    wandb.init(
        project="joint-training",
        name="rd4ad-joint",
        config={
            "model": "RD4AD",
            "backbone": "wide_resnet50_2",
            "epochs": 50,
            "device": "cuda:1",
            "input_size": (224, 224),
            "dataset": "BMAD"
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    data_test = data_full.load_test()
    # combined_test_loader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False)
    categories = data_full.categories
    # print(f"test batch: {next(iter(combined_test_loader))}")

    # Create model and strategy
    model_rd4ad = RD4ADModel("cuda:1", 'wide_resnet50_2', (224, 224))
    model_rd4ad.load_model()
    model_rd4ad.ad_model.train()

    for epoch in range(50):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{50}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_rd4ad.device)

                cos_loss = torch.nn.CosineSimilarity()

                teacher_features, bn_features, student_features = model_rd4ad.ad_model(images)

                loss = 0
                for i in range(len(teacher_features)):
                    loss += torch.mean(
                        1 - cos_loss(
                            teacher_features[i].view(teacher_features[i].shape[0],-1),
                            student_features[i].view(student_features[i].shape[0],-1)
                        )
            )
                
                model_rd4ad.optimizer.zero_grad()
                loss.backward()
                model_rd4ad.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{50}, Loss: {avg_loss:.4f}")
        
        # Log training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

    model_rd4ad.ad_model.eval()

    results = {}
    for task_id, test_task in enumerate(data_test):
        cat = categories[task_id]
        test_dl = torch.utils.data.DataLoader(test_task, batch_size=32, shuffle=False)
        evaluator = Evaluator(test_dl, model_rd4ad.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_rd4ad.ad_model)
        results[cat] = metrics

        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    with open('/home/ruslan/thesis/tests/joint_rd4ad.txt', 'w') as file:
        file.write(json.dumps(results))
    
    # Finish wandb run
    wandb.finish()

def FastFlow():
    
    # Initialize wandb
    wandb.init(
        project="joint-training",
        name="fastflow-joint",
        config={
            "model": "FastFlow",
            "backbone": "wide_resnet50_2",
            "epochs": 50,
            "device": "cuda:1",
            "input_size": (224, 224),
            "dataset": "BMAD"
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    data_test = data_full.load_test()
    # combined_test_loader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False)
    categories = data_full.categories
    # print(f"test batch: {next(iter(combined_test_loader))}")

    # Create model and strategy
    model_fastflow = FastFlowModel('cuda:1', "wide_resnet50_2", (224, 224))
    model_fastflow.load_model()
    model_fastflow.ad_model.train()

    for epoch in range(50):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{50}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_fastflow.device)

                hidden_variables, jacobians = model_fastflow.ad_model(images)
                loss = model_fastflow.loss(hidden_variables, jacobians)
                
                model_fastflow.optimizer.zero_grad()
                loss.backward()
                model_fastflow.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{50}, Loss: {avg_loss:.4f}")
        
        # Log training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

    model_fastflow.ad_model.eval()

    results = {}
    for task_id, test_task in enumerate(data_test):
        cat = categories[task_id]
        test_dl = torch.utils.data.DataLoader(test_task, batch_size=32, shuffle=False)
        evaluator = Evaluator(test_dl, model_fastflow.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_fastflow.ad_model)
        results[cat] = metrics

        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    with open('/home/ruslan/thesis/tests/joint_fastflow.txt', 'w') as file:
        file.write(json.dumps(results))
    
    # Finish wandb run
    wandb.finish()

def DRAEM():
    
    # Initialize wandb
    wandb.init(
        project="joint-training",
        name="draem-joint",
        config={
            "model": "DRAEM",
            "epochs": 50,
            "device": "cuda:1",
            "batch_size": 8,
            "input_size": [224, 224],
            "dataset": "BMAD"
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir="/mnt/disk1/ruslan_nuriev/bmad", draem=True, norm=False)
    categories = data_full.categories
    data_train_tuples = data_full.load_train()
    data_test = data_full.load_test()
    
    class ImageOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, tuple_dataset):
            self.dataset = tuple_dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            return self.dataset[idx][0]
    
    data_train_images = ImageOnlyDataset(data_train_tuples)
    # data_test_images = ImageOnlyDataset(data_test_tuples)
    
    data_train = DRAEMContinualDataset(data_train_images, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", split="train", resize_shape=[224, 224])
    # data_test = DRAEMContinualDataset(data_test_tuples, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", split="test", resize_shape=[224, 224])
    data_train = torch.utils.data.DataLoader(data_train, batch_size=8, shuffle=True)
    # data_test = torch.utils.data.DataLoader(data_test, batch_size=8, shuffle=False)
    
    model_draem = DRAEMModel("cuda:1")
    model_draem.load_model()
    model_draem.ad_model.train()

    for epoch in range(50):
        epoch_losses = []
        with tqdm(data_train, desc=f"Epoch {epoch+1}/{50}") as pbar:
            for batch in pbar:
                gray_batch = batch[0].to(model_draem.device)
                aug_gray_batch = batch[1].to(model_draem.device)
                anomaly_mask = batch[2].to(model_draem.device)

                gray_rec = model_draem.ad_model.model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_draem.ad_model.model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = model_draem.l2_loss(gray_rec,gray_batch)
                ssim_loss = model_draem.ssim_loss(gray_rec, gray_batch)

                segment_loss = model_draem.focal_loss(out_mask_sm, anomaly_mask)
                # print(f"L2 Loss: {l2_loss}, SSIM Loss: {ssim_loss}, Segment Loss: {segment_loss}")
                loss = l2_loss + ssim_loss + segment_loss
                
                model_draem.optimizer.zero_grad()
                loss.backward()
                model_draem.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{50}, Loss: {avg_loss:.4f}")
        
        # Log training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

    model_draem.ad_model.eval()
    torch.save(model_draem.ad_model.state_dict(), "/home/ruslan/thesis/tests/draem_weights.pth")

    results = {}
    for task_id, task_dataset in enumerate(data_test):
        cat = categories[task_id]
        data_test = DRAEMContinualDataset(task_dataset, "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/anomaly_dataset/images", split="test", resize_shape=[224, 224])
        data_test = torch.utils.data.DataLoader(data_test, batch_size=8, shuffle=False)
        evaluator = Evaluator(data_test, model_draem.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_draem.ad_model)
        results[cat] = metrics
        
        # Log evaluation metrics to wandb
        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    with open('/home/ruslan/thesis/tests/joint_draem.txt', 'w') as file:
        file.write(json.dumps(results))
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")
    FastFlow()
    STFPM()
    RD4AD()
    DRAEM()