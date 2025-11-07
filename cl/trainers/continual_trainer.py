from cl_ad_model import ContinualADModel
from memories.memory_stream import ContinualDataset

import wandb
import torch
from torch import optim
from tqdm import tqdm
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader

from memories.replay_strategy import ReplayModel
from collections import defaultdict

from moviad.utilities.evaluator import Evaluator

### TODO: For each task, evaluate them sequentially by taking their averages of the metrics.
### And log them using wandb.

class ContinualTrainer:
    """Manages the continual learning training process"""
    
    def __init__(self, 
                 strategy: ContinualADModel,
                 device: str = "cuda:0",
                 logger: bool = False,
                 wandb_config: dict = None,
                 **kwargs
                ):
        
        self.strategy = strategy
        self.device = device
        self.use_wandb = logger
        self.global_step = 0

        # Initialize wandb if logging is enabled
        if self.use_wandb:
            if wandb_config:
                wandb.init(**wandb_config)
            else:
                wandb.init(project="continual-learning-ad")
            wandb.config.update({"device": device})

    def _augment_with_replay(self, batch):
        replay_dataset = self.strategy.replay_buffer.get_buffer_dataset()
        batch_size = len(batch[0]) if isinstance(batch, tuple) else len(batch)
        
        replay_samples = [item for item in replay_dataset]
        indices = torch.randperm(len(replay_samples))[:batch_size]
        selected_replay_samples = [replay_samples[i] for i in indices]
        
        if isinstance(selected_replay_samples[0], tuple) and len(selected_replay_samples[0]) > 2:
            # DRAEM case: each sample is (image, augmented_image, anomaly_mask, has_anomaly, idx, task_id)
            # Need to stack each element separately and concatenate with batch (coz torch.stack doesnn't let you stack tuples)
            num_elements = len(selected_replay_samples[0])
            result = []
            
            for elem_idx in range(num_elements):
                elements = [sample[elem_idx] for sample in selected_replay_samples]
                
                if isinstance(elements[0], torch.Tensor):
                    stacked_elem = torch.stack(elements).to(batch[0].device).type(batch[0].dtype)
                else:
                    stacked_elem = torch.from_numpy(np.array(elements)).to(batch[0].device) # deals with the last three elements (scalars, like has_anomaly)
                
                # Concatenate with current batch element
                concatenated = torch.cat([batch[elem_idx], stacked_elem])
                result.append(concatenated)
            
            return tuple(result)
        else:
            # Other models case: (image, task_id)
            # Extract just the images (first element)
            replay_images = torch.stack([sample[0] for sample in selected_replay_samples])
            replay_images = replay_images.to(batch[0].device).type(batch[0].dtype)
            
            # batch[0] is the images tensor, batch[1] is task_id
            concatenated_images = torch.cat([batch[0], replay_images])
            
            # For task_id, we can either keep the original or extend it
            # Since we're mixing tasks, we'll keep the replay task_ids too
            replay_task_ids = torch.tensor([sample[1] for sample in selected_replay_samples]).to(batch[0].device)
            concatenated_task_ids = torch.cat([batch[1], replay_task_ids])
            
            return (concatenated_images, concatenated_task_ids)
        
    def train(self, continual_dataset: ContinualDataset, epochs_per_task: int = 10, ratio: float = 0.5):
        """Train on all tasks sequentially"""
        
        self.all_task_loaders = []  # Keep for evaluation later

        self.strategy._init_model()
        
        # Log training configuration
        if self.use_wandb:
            wandb.config.update({
                "epochs_per_task": epochs_per_task,
                "replay_ratio": ratio,
                "num_tasks": continual_dataset.num_categories
            })
        
        for task_id in range(continual_dataset.num_categories):
            print(f"\n{'='*50}")
            print(f"Task {task_id}: {continual_dataset.get_task_info()['category']}")
            print('='*50)
            
            # Get data loaders
            train_loader, test_loader = continual_dataset.get_current_task_loaders()
            self.all_task_loaders.append({
                'task_id': task_id,
                'test': test_loader,
                'category': continual_dataset.get_task_info()['category']
            })
            
            # Begin task
            self.strategy.begin_task(task_id)
            has_replay = False
            
            # Log task start
            if self.use_wandb:
                wandb.log({
                    "task/current_task_id": task_id,
                    "task/category": continual_dataset.get_task_info()['category']
                }, step=self.global_step)

            # Prepare training data (current task + replay buffer)
            if self.strategy.replay_buffer.buffer: #self.strategy.replay_buffer.buffer
                # Combine current task data with replay buffer
                has_replay = True
                batch_size = train_loader.batch_size
                task_size = int((1 - ratio) * batch_size)
                replay_size = batch_size - task_size
                replay_dataset = self.strategy.replay_buffer.get_buffer_dataset()
                current_task_dataset = train_loader.dataset

                loader = DataLoader(
                    dataset=current_task_dataset,
                    batch_size=task_size,
                    shuffle=True)
                
                print(f"Training with {len(current_task_dataset)} current samples + "
                    f"{len(replay_dataset)} replay samples")
                training_loader = loader
                
            else:
                print(f"Training with {len(train_loader.dataset)} current samples (no replay)")
                training_loader = train_loader

            if self.strategy.scheduler:
                self.strategy.scheduler = optim.lr_scheduler.MultiStepLR(self.strategy.optimizer,[epochs_per_task*0.8,epochs_per_task*0.9],gamma=0.2, last_epoch=-1)
    
            for epoch in range(epochs_per_task):
                epoch_losses = []
                with tqdm(training_loader, desc=f"Epoch {epoch+1}/{epochs_per_task}") as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        # batch is a tuple/list from the DataLoader
                        images = batch
                        
                        if has_replay:
                            images = self._augment_with_replay(images)
                        
                        loss = self.strategy.partial_update(images)
                        epoch_losses.append(loss)
                        pbar.set_postfix(loss=f"{loss:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({
                                f"task_{task_id}/batch_loss": loss,
                                "train/batch_loss": loss,
                                "train/global_step": self.global_step
                            }, step=self.global_step)
                        
                        self.global_step += 1
                
                if self.strategy.scheduler:
                    self.strategy.scheduler.step()

                avg_loss = np.mean(epoch_losses)
                print(f"  Epoch {epoch+1}/{epochs_per_task}, Loss: {avg_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        f"task_{task_id}/epoch_loss": avg_loss,
                        f"task_{task_id}/epoch": epoch + 1,
                        "train/epoch_loss": avg_loss,
                        "train/min_batch_loss": np.min(epoch_losses),
                        "train/max_batch_loss": np.max(epoch_losses),
                        "train/std_batch_loss": np.std(epoch_losses)
                    }, step=self.global_step)
            
            # End task - update replay buffer
            # NOTE: For DRAEM, this stores the full tuple including pre-computed augmentations.
            # Ideally, we should store only original images and re-augment during replay,
            # but this works as a functional solution.
            self.strategy.end_task(train_loader)
            
            # Sequential evaluation
            print(f"\nEvaluating after Task {task_id}...")
            _, avg_metrics = self._sequential_evaluation(task_id)

            eval_log = {"eval/task_id": task_id}

            for metric_name, metric in avg_metrics.items():
                eval_log[f"eval/{metric_name}"] = metric
                
            if self.use_wandb:
                wandb.log(eval_log, step=self.global_step)
        
            # Move to next task
            if not continual_dataset.to_next_task():
                break
        
        return avg_metrics
    
    def _sequential_evaluation(self, current_task_id):
        
        print(f"Evaluating on tasks 0 to {current_task_id}...")
        
        all_task_metrics = {}
        for eval_task_id in range(current_task_id + 1):
            loader_info = self.all_task_loaders[eval_task_id]
            category = loader_info['category']
            test_loader = loader_info['test']
            
            print(f"Evaluating on Task {eval_task_id} ({category})...")
            
            evaluator = Evaluator(test_loader, self.device)
            results = evaluator.evaluate(self.strategy.model)
            # add to a temp dict and then average after finishing the evaluation
            # this way we'll keep the original/initial values of the first task
            self.strategy.model.train()
            
            all_task_metrics[eval_task_id] = {
                'category': category,
                'metrics': results
            }
            
            print(f"Task {eval_task_id} ({category}) Results: {results}")
        
        avg_metrics = self._calculate_average_metrics(all_task_metrics)
        
        print(f"\nAverage metrics across tasks 0-{current_task_id}: {avg_metrics}")
        
        
        return all_task_metrics, avg_metrics
    
    def _calculate_average_metrics(self, all_task_metrics):
        from scipy.stats import tmean
        metric_values = defaultdict(list)
        
        for _, task_info in all_task_metrics.items():
            metrics = task_info['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.number)):
                    metric_values[metric_name].append(metric_value)
        
        avg_metrics = {}
        for metric_name, values in metric_values.items():
            # values = np.array(values)
            avg_metrics[f"{metric_name}"] = tmean(values, (0.01, 1), nan_policy="omit") #np.true_divide(values.sum(), np.isfinite(values).sum()), I guess none of the models will output 0, but jic
                    
        return avg_metrics
