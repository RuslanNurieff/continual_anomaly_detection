from datasets.full_dataset import CombinedDataset

from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES
from moviad.utilities.evaluator import Evaluator

from memories.replay_strategy import ReplayModel
from trainers.models import RD4ADModel
from trainers.continual_trainer import ContinualTrainer
import wandb

from tqdm import tqdm
import numpy as np
import torch
from memories.memory_stream import StreamManager

import torch.nn.functional as F
import wandb


def joint_training(device, root_dir, epochs):
    # Initialize wandb
    wandb.init(
        project="rd4ad-tests",
        name="rd4ad-joint-training",
        config={
            "model": "RD4AD",
            "backbone": "wide_resnet50_2",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD"
        }
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir=root_dir, norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    data_test = data_full.load_test()
    categories = data_full.categories

    # Create model and strategy
    model_rd4ad = RD4ADModel(device, 'wide_resnet50_2', (224, 224))
    model_rd4ad.load_model()
    model_rd4ad.ad_model.train()

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
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
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
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
            f"{cat}/img_f1": metrics.get('img_f1', 0),
            f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    # Finish wandb run
    wandb.finish()

def fine_tuning(device, root_dir, epochs):
    wandb.init(
        project="rd4ad-tests",
        name="rd4ad-fine-tuning",
        config={
            "model": "RD4AD",
            "backbone": "wide_resnet50_2",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD"
        }
    )

    model_rd4ad = RD4ADModel(device, 'wide_resnet50_2', (224, 224))
    model_rd4ad.load_model()
    model_rd4ad.ad_model.train()

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    test_task_loaders = []

    for task_id in range(data_seq.num_categories):
    
        train_loader, test_loader = data_seq.get_current_task_loaders()
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
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })
            
        if not data_seq.to_next_task():
            break
    
    model_rd4ad.ad_model.eval()

    results = {}
    for test_task in test_task_loaders:
        cat = test_task['category']
        test_dl = test_task['test']
        evaluator = Evaluator(test_dl, model_rd4ad.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_rd4ad.ad_model)
        results[cat] = metrics

        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/img_f1": metrics.get('img_f1', 0),
            f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })

    wandb.finish()



def continual_learning(device, root_dir, epochs):
    continual_dataset = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    replay_strategy = ReplayModel(
        model_conf=RD4ADModel(device, "wide_resnet50_2", (224, 224)),
        buffer_size=1000
    )

    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=True,
        wandb_config={"project": "rd4ad-tests",
                      "name": "rd4ad-continual-learning",
                      "config": {
                        "model": "RD4AD",
                        "backbone": "wide_resnet50_2",
                        "epochs": epochs,
                        "device": device,
                        "input_size": (224, 224),
                        "dataset": "BMAD"
                    }
                }
        )

    avg_metrics = trainer.train(
        continual_dataset,
        epochs_per_task=epochs
    )

def single_model(device, root_dir, epochs):
    wandb.init(
        project="rd4ad-tests",
        name="rd4ad-single-model",
        config={
            "model": "RD4AD",
            "backbone": "wide_resnet50_2",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD"
        },
        reinit=True
    )

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    for task_id in range(data_seq.num_categories):
        model_rd4ad = RD4ADModel(device, 'wide_resnet50_2', (224, 224))
        model_rd4ad.load_model()
        model_rd4ad.ad_model.train()
    
        train_loader, test_loader = data_seq.get_current_task_loaders()
        cat = data_seq.get_task_info()['category']
        print(f"Training a new model on {cat}")

        for epoch in range(epochs):
            epoch_losses = []
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
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
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

        model_rd4ad.ad_model.eval()
        evaluator = Evaluator(test_loader, model_rd4ad.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_rd4ad.ad_model)
        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            f"{cat}/img_f1": metrics.get('img_f1', 0),
            f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
            
        if not data_seq.to_next_task():
            break

    wandb.finish()

if __name__ == "__main__":
    wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")
    # fine_tuning("cuda:0", "/mnt/disk1/ruslan_nuriev/bmad", 50)
    # joint_training("cuda:0", "/mnt/disk1/ruslan_nuriev/bmad", 50)
    # continual_learning("cuda:0", "/mnt/disk1/ruslan_nuriev/bmad", 50)
    single_model("cuda:0", "/mnt/disk1/ruslan_nuriev/bmad", 50)