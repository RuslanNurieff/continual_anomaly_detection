from datasets.full_dataset import CombinedDataset

from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES
from moviad.utilities.evaluator import Evaluator

from memories.replay_strategy import ReplayModel
from trainers.models import SuperSimpleNetModel
from trainers.continual_trainer import ContinualTrainer
import wandb
from moviad.models.components.simplenet.loss import SSNLoss

from tqdm import tqdm
import numpy as np
import torch
from memories.memory_stream import StreamManager
import pandas as pd
from scipy.stats import tmean
import torch.nn.functional as F
import wandb

import random


def joint_training(backbone, device, root_dir, epochs, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize wandb
    wandb.init(
        project="ssn-tests",
        name="ssn-joint-training",
        config={
            "model": "SuperSimpleNet",
            "backbone": backbone,
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir=root_dir, norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=8, shuffle=False)
    data_test = data_full.load_test()
    categories = data_full.categories

    # Create model and strategy
    model_ssn = SuperSimpleNetModel(device, backbone, ["layer2", "layer3"])
    model_ssn.load_model()
    model_ssn.ad_model.train()
    
    loss_fn = SSNLoss()

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_ssn.device)
                B,C,H,W = images.shape
                masks = torch.zeros(B, 1, H, W).to(model_ssn.device)
                labels = torch.zeros(B).to(model_ssn.device)

                anomaly_map, anomaly_score, masks, labels = model_ssn.ad_model(
                    images=images,
                    masks=masks,
                    labels=labels,
                )
                loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
                
                model_ssn.optimizer.zero_grad()
                loss.backward()
                model_ssn.optimizer.step()                

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        model_ssn.scheduler.step()
        
        # Log training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })
    try:
        torch.save(model_ssn.ad_model.state_dict(), "/home/ruslan/thesis/tests/checkpoints/ssn_joint_training.pth")
    except:
        pass

    model_ssn.ad_model.eval()

    results = {}
    for task_id, test_task in enumerate(data_test):
        cat = categories[task_id]
        test_dl = torch.utils.data.DataLoader(test_task, batch_size=8, shuffle=False)
        evaluator = Evaluator(test_dl, model_ssn.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_ssn.ad_model)
        results[cat] = metrics

        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            # f"{cat}/img_f1": metrics.get('img_f1', 0),
            # f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            # f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            # f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            # f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    # Finish wandb run
    wandb.finish()

def fine_tuning(backbone, device, root_dir, epochs, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    wandb.init(
        project="ssn-tests",
        name="ssn-fine-tuning",
        config={
            "model": "SuperSimpleNet",
            "backbone": backbone,
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        }
    )

    model_ssn = SuperSimpleNetModel(device, backbone, ["layer2", "layer3"])
    model_ssn.load_model()
    model_ssn.ad_model.train()

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    test_task_loaders = []
    
    loss_fn = SSNLoss()

    for task_id in range(data_seq.num_categories):
        model_ssn.ad_model.train()
    
        train_loader, test_loader = data_seq.get_current_task_loaders(batch_size=8)
        test_task_loaders.append({
            'task_id': task_id,
            'test': test_loader,
            'category': data_seq.get_task_info()['category']
        })

        for epoch in range(epochs):
            epoch_losses = []
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch

                    images = images.to(model_ssn.device)
                    B,C,H,W = images.shape
                    masks = torch.zeros(B, 1, H, W).to(model_ssn.device)
                    labels = torch.zeros(B).to(model_ssn.device)

                    anomaly_map, anomaly_score, masks, labels = model_ssn.ad_model(
                        images=images,
                        masks=masks,
                        labels=labels,
                    )
                    loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
                    
                    model_ssn.optimizer.zero_grad()
                    loss.backward()
                    model_ssn.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")
                    

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            model_ssn.scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

        results = {}
        model_ssn.ad_model.eval()
        for task in test_task_loaders:
            to_log = str(len(test_task_loaders) - 1) + "_seq"
            cat = task['category']
            test_dl = task['test']
            evaluator = Evaluator(test_dl, model_ssn.device)
            print(f"Evaluating on category \"{cat}\"...")
            metrics = evaluator.evaluate(model_ssn.ad_model)
            results[cat] = metrics
            wandb.log({
                f"{to_log}/{cat}/img_roc_auc": metrics.get("img_roc_auc", 0),
                f"{to_log}/{cat}/pxl_roc_auc": metrics.get("pxl_roc_auc", 0),
            })
            print(f"Results for '{cat}': \n IMG AUROC: {metrics.get("img_roc_auc", 0)}\n Pixel AUROC: {metrics.get("pxl_roc_auc", 0)}")
        
        df = pd.DataFrame(results).T
        averages = df.apply(lambda x: tmean(x, (0.01, 1), nan_policy="omit")).to_dict()
        wandb.log({
            f"{task_id}_avg/img_roc_auc": averages.get("img_roc_auc", 0),
            f"{task_id}_avg/pxl_roc_auc": averages.get("pxl_roc_auc", 0),
        })
        print(f"Average values so far (until task_id {task_id}): \n IMG AUROC: {averages.get("img_roc_auc", 0)}\nPixel AUROC: {averages.get("pxl_roc_auc", 0)}")
            
        if not data_seq.to_next_task():
            break

    try:
        torch.save(model_ssn.ad_model.state_dict(), "/home/ruslan/thesis/tests/checkpoints/ssn_fine_tuning.pth")
    except:
        pass

    wandb.finish()



def continual_learning(backbone, device, root_dir, epochs, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    continual_dataset = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    replay_strategy = ReplayModel(
        model_conf=SuperSimpleNetModel(device, backbone, ["layer2", "layer3"]),
        buffer_size=1000
    )

    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=True,
        wandb_config={"project": "ssn-tests",
                      "name": "ssn-continual-learning",
                      "config": {
                        "model": "SuperSimpleNet",
                        "backbone": backbone,
                        "epochs": epochs,
                        "device": device,
                        "input_size": (224, 224),
                        "dataset": "BMAD",
                        "seed": seed
                    }
                },
        device=device
        )

    trainer.train(
        continual_dataset,
        epochs_per_task=epochs,
        batch_size=8
    )

    try:
        torch.save(trainer.strategy.model.state_dict(), "/home/ruslan/thesis/tests/checkpoints/ssn_continual_learning.pth")
    except:
        pass

def single_model(backbone, device, root_dir, epochs, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    wandb.init(
        project="ssn-tests",
        name="ssn-single-model",
        config={
            "model": "SuperSimpleNet",
            "backbone": backbone,
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        }
    )

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))


    for task_id in range(data_seq.num_categories):
        model_ssn = SuperSimpleNetModel(device, backbone, ["layer2", "layer3"])
        model_ssn.load_model()
        model_ssn.ad_model.train()
    
        train_loader, test_loader = data_seq.get_current_task_loaders(batch_size=8)
        cat = data_seq.get_task_info()['category']
        print(f"Training a new model on {cat}")

        loss_fn = SSNLoss()

        for epoch in range(epochs):
            epoch_losses = []
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch

                    images = images.to(model_ssn.device)
                    B,C,H,W = images.shape
                    masks = torch.zeros(B, 1, H, W).to(model_ssn.device)
                    labels = torch.zeros(B).to(model_ssn.device)

                    anomaly_map, anomaly_score, masks, labels = model_ssn.ad_model(
                        images=images,
                        masks=masks,
                        labels=labels,
                    )
    
                    loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
                    
                    model_ssn.optimizer.zero_grad()
                    loss.backward()
                    model_ssn.optimizer.step()
                    

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            model_ssn.scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

        model_ssn.ad_model.eval()
        evaluator = Evaluator(test_loader, model_ssn.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_ssn.ad_model)
        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            # f"{cat}/img_f1": metrics.get('img_f1', 0),
            # f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            # f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            # f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            # f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })

        try:
            torch.save(model_ssn.ad_model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/ssn_single_{cat}.pth")
        except:
            pass
            
        if not data_seq.to_next_task():
            break

    wandb.finish()

def single_liver(epochs, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # wandb.init(
    #     project="ssn-tests",
    #     name="ssn-single-model",
    #     config={
    #         "model": "SuperSimpleNet",
    #         "backbone": "resnet18",
    #         "epochs": epochs,
    #         "device": device,
    #         "input_size": (224, 224),
    #         "dataset": "BMAD",
    #         "seed": seed
    #     }
    # )

    data_train = BMAD("segmentation", '/mnt/disk1/ruslan_nuriev/bmad', 'brain', 'train')
    data_train = torch.utils.data.DataLoader(data_train, 8, True)
    data_test = BMAD("segmentation", '/mnt/disk1/ruslan_nuriev/bmad', 'brain', 'test')
    data_test = torch.utils.data.DataLoader(data_test, 8, False)

    model_ssn = SuperSimpleNetModel("cuda:2", "wide_resnet50_2", ["layer2", "layer3"])
    model_ssn.load_model()
    model_ssn.ad_model.train()

    loss_fn = SSNLoss()

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(data_train, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_ssn.device)
                B,C,H,W = images.shape
                masks = torch.zeros(B, 1, H, W).to(model_ssn.device)
                labels = torch.zeros(B).to(model_ssn.device)

                anomaly_map, anomaly_score, masks, labels = model_ssn.ad_model(
                    images=images,
                    masks=masks,
                    labels=labels,
                )

                loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
                
                model_ssn.optimizer.zero_grad()
                loss.backward()
                model_ssn.optimizer.step()
                

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        model_ssn.scheduler.step()

    model_ssn.ad_model.eval()
    evaluator = Evaluator(data_test, model_ssn.device)
    metrics = evaluator.evaluate(model_ssn.ad_model)
    print(metrics)

if __name__ == "__main__":
    wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")
    single_model("wide_resnet50_2", "cuda:2", "/mnt/disk1/ruslan_nuriev/bmad", 4)
    fine_tuning("wide_resnet50_2", "cuda:2", "/mnt/disk1/ruslan_nuriev/bmad", 4)
    joint_training("wide_resnet50_2", "cuda:2", "/mnt/disk1/ruslan_nuriev/bmad", 4)
    continual_learning("wide_resnet50_2", "cuda:2", "/mnt/disk1/ruslan_nuriev/bmad", 4)
    # single_liver(5)