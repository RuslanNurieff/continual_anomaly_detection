from datasets.full_dataset import CombinedDataset

from moviad.datasets.bmad.bmad_dataset import BMAD, CATEGORIES
from moviad.utilities.evaluator import Evaluator

from memories.replay_strategy import ReplayModel
from trainers.models import STFPMModel
from moviad.trainers.trainer_stfpm import TrainerSTFPM
from trainers.continual_trainer import ContinualTrainer
import wandb

from tqdm import tqdm
import numpy as np
import torch
from memories.memory_stream import StreamManager

import torch.nn.functional as F
import wandb
import pandas as pd
from scipy.stats import tmean
import json
import random

def joint_training(device, root_dir, epochs, seed=42):
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
        project="stfpm-tests",
        name="stfpm-joint-training",
        config={
            "model": "STFPM",
            "backbone": "resnet18",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        },
        reinit=True
    )

    data_full = CombinedDataset(BMAD, task_type="segmentation", root_dir=root_dir, norm=True)
    data_train = data_full.load_train()
    combined_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    data_test = data_full.load_test()
    categories = data_full.categories

    # Create model and strategy
    model_stfpm = STFPMModel(device, 'resnet18', ['layer1', 'layer2', 'layer3'])
    model_stfpm.load_model()
    model_stfpm.ad_model.train()

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
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
                    loss += TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])

                model_stfpm.optimizer.zero_grad()
                loss.backward()
                model_stfpm.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Log training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

    try:
        torch.save(model_stfpm.ad_model.student.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_student_joint_training.pth")
        torch.save(model_stfpm.ad_model.teacher.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_teacher_joint_training.pth")
    except:
        pass

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
            # f"{cat}/img_f1": metrics.get('img_f1', 0),
            # f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            # f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            # f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            # f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
    
    # Finish wandb run
    wandb.finish()

def fine_tuning(device, root_dir, epochs, seed=42):
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
        project="stfpm-tests",
        name="stfpm-fine-tuning",
        config={
            "model": "STFPM",
            "backbone": "wide_resnet50_2",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        },
        reinit=True
    )

    model_stfpm = STFPMModel(device, 'wide_resnet50_2', ['layer1', 'layer2', 'layer3'])
    model_stfpm.load_model()
    

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    test_task_loaders = []

    for task_id in range(data_seq.num_categories):

        model_stfpm.ad_model.train()
    
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

                    images = images.to(model_stfpm.device)

                    teacher_features, student_features = model_stfpm.ad_model(images)

                    loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)
                        loss += TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])
                    
                    model_stfpm.optimizer.zero_grad()
                    loss.backward()
                    model_stfpm.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

        results = {}
        model_stfpm.ad_model.eval()
        for task in test_task_loaders:
            to_log = str(len(test_task_loaders) - 1) + "_seq"
            cat = task['category']
            test_dl = task['test']
            evaluator = Evaluator(test_dl, model_stfpm.device)
            print(f"Evaluating on category \"{cat}\"...")
            metrics = evaluator.evaluate(model_stfpm.ad_model)
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

    wandb.finish()
    try:
        torch.save(model_stfpm.ad_model.student.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_student_fine_tuning.pth")
        torch.save(model_stfpm.ad_model.teacher.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_teacher_fine_tuning.pth")
    except:
        pass



def continual_learning(device, root_dir, epochs, seed=42):
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
        model_conf=STFPMModel(device, 'wide_resnet50_2', ['layer1', 'layer2', 'layer3']),
        buffer_size=1000
    )

    trainer = ContinualTrainer(
        strategy=replay_strategy,
        logger=True,
        wandb_config={"project": "stfpm-tests",
                      "name": "stfpm-continual-learning",
                      "config": {
                        "model": "STFPM",
                        "backbone": "resnet18",
                        "epochs": epochs,
                        "device": device,
                        "input_size": (224, 224),
                        "dataset": "BMAD",
                        "seed": seed
                    },
                        "reinit": True
                },
        device=device
        )

    avg_metrics = trainer.train(
        continual_dataset,
        epochs_per_task=epochs
    )

    try:
        torch.save(trainer.strategy.model.student.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_student_cl.pth")
        torch.save(trainer.strategy.model.teacher.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_teacher_cl.pth")
    except:
        pass

def single_model(device, root_dir, epochs, seed=42):
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
        project="stfpm-tests",
        name="stfpm-single-model",
        config={
            "model": "STFPM",
            "backbone": "resnet18",
            "epochs": epochs,
            "device": device,
            "input_size": (224, 224),
            "dataset": "BMAD",
            "seed": seed
        },
        reinit=True
    )

    data_seq = StreamManager(BMAD, task_type="segmentation", root_dir=root_dir, categories=list(CATEGORIES))

    for task_id in range(data_seq.num_categories):
        model_stfpm = STFPMModel(device, 'resnet18', ['layer1', 'layer2', 'layer3'])
        model_stfpm.load_model()
        model_stfpm.ad_model.train()
    
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

                    images = images.to(model_stfpm.device)

                    teacher_features, student_features = model_stfpm.ad_model(images)

                    loss = 0
                    for i in range(len(student_features)):
                        teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                        student_features[i] = F.normalize(student_features[i], dim=1)
                        loss += TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])
                        
                    model_stfpm.optimizer.zero_grad()
                    loss.backward()
                    model_stfpm.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

        try:
            torch.save(model_stfpm.ad_model.student.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_student_single_model.pth")
            torch.save(model_stfpm.ad_model.teacher.model.state_dict(), f"/home/ruslan/thesis/tests/checkpoints/stfpm/stfpm_teacher_single_model.pth")
        except:
            pass

        model_stfpm.ad_model.eval()
        evaluator = Evaluator(test_loader, model_stfpm.device)
        print(f"Evaluating on category \"{cat}\"...")
        metrics = evaluator.evaluate(model_stfpm.ad_model)
        wandb.log({
            f"{cat}/img_roc_auc": metrics.get('img_roc_auc', 0),
            f"{cat}/pxl_roc_auc": metrics.get('pxl_roc_auc', 0),
            # f"{cat}/img_f1": metrics.get('img_f1', 0),
            # f"{cat}/pxl_f1": metrics.get('pxl_f1', 0),
            # f"{cat}/img_pr_auc": metrics.get('img_pr_auc', 0),
            # f"{cat}/pxl_pr_auc": metrics.get('pxl_pr_auc', 0),
            # f"{cat}/pxl_au_pro": metrics.get('pxl_au_pro', 0),
        })
            
        if not data_seq.to_next_task():
            break

    wandb.finish()

def single_brain(device, epochs, seed=42):
    # Set random seeds for reproducibility
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_train = BMAD("segmentation", "/mnt/disk1/ruslan_nuriev/bmad", "liver", "train")
    train_loader = torch.utils.data.DataLoader(data_train, 32, True)

    data_test = BMAD("segmentation", "/mnt/disk1/ruslan_nuriev/bmad", "liver", "test")
    test_loader = torch.utils.data.DataLoader(data_test, 32, False)

    model_rd4ad = STFPMModel("cuda:0", 'resnet18', ['layer1', 'layer2', 'layer3'])
    model_rd4ad.load_model()
    model_rd4ad.ad_model.eval()

    print("Training a new model on brain")

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(model_rd4ad.device)

                teacher_features, student_features = model_rd4ad.ad_model(images)

                loss = 0
                for i in range(len(student_features)):
                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)
                    loss += TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])
                
                model_rd4ad.optimizer.zero_grad()
                loss.backward()
                model_rd4ad.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model_rd4ad.ad_model.eval()
    evaluator = Evaluator(test_loader, model_rd4ad.device)
    cat = "brain"
    print(f"Evaluating on category \"{cat}\"...")
    metrics = evaluator.evaluate(model_rd4ad.ad_model)

    wandb.finish()

if __name__ == "__main__":
    # wandb.login(key="4f6d843a12185b07fd5f95d3e42b35c1a9f90a51")
    # fine_tuning("cuda:1", "/mnt/disk1/ruslan_nuriev/bmad", 25)
    # single_model("cuda:1", "/mnt/disk1/ruslan_nuriev/bmad", 25)
    # joint_training("cuda:1", "/mnt/disk1/ruslan_nuriev/bmad", 25)
    # continual_learning("cuda:0", "/mnt/disk1/ruslan_nuriev/bmad", 25)
    single_brain("cuda:0", 2)

