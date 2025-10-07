from cl_ad_model import ContinualADModel
from memories.memory_stream import ContinualDataset

import wandb
import torch
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
                 wandb_config: dict = None
                ): # evaluator: ContinualEvaluator = None)
        
        self.strategy = strategy
        # self.evaluator = evaluator or ContinualEvaluator()
        # self.history = defaultdict(list)
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

    def _augment_with_replay(self, images):
        """Augment current batch with samples from replay buffer."""
        replay_dataset = self.strategy.replay_buffer.get_buffer_dataset()
        replay_size = len(images)  # Match the current batch size
        
        replay_samples = [item[0] for item in replay_dataset]
        indices = torch.randperm(len(replay_samples))[:replay_size]
        replay_samples = torch.stack([replay_samples[i] for i in indices])
        replay_samples = replay_samples.to(images.device).type(images.dtype)
        
        return torch.cat([images, replay_samples])
        
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
    
            for epoch in range(epochs_per_task):
                epoch_losses = []
                with tqdm(training_loader, desc=f"Epoch {epoch+1}/{epochs_per_task}") as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        if isinstance(batch, (list, tuple)):
                            images = batch[0]
                        else:
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
                
                avg_loss = np.mean(epoch_losses)
                print(f"  Epoch {epoch+1}/{epochs_per_task}, Loss: {avg_loss:.4f}")
                # self.history[f'task_{task_id}_loss'].append(avg_loss)
                
                # Log epoch metrics
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
            self.strategy.end_task(train_loader)
            
            # Log task completion
            if self.use_wandb:
                final_buffer_size = len(self.strategy.replay_buffer.buffer) if self.strategy.replay_buffer.buffer else 0
                wandb.log({
                    f"task_{task_id}/final_loss": avg_loss,
                    f"task_{task_id}/buffer_size_after": final_buffer_size,
                    "task/completed_tasks": task_id + 1
                }, step=self.global_step)
            
            # Sequential evaluation: evaluate on all tasks seen so far
            print(f"\nEvaluating after Task {task_id}...")
            self._sequential_evaluation(task_id)
            
            # Move to next task
            if not continual_dataset.to_next_task():
                break
        
        # return self.history
    
    def _sequential_evaluation(self, current_task_id):
        """
        Evaluate on all tasks seen so far (tasks 0 to current_task_id).
        Calculate and log average metrics across all evaluated tasks.
        """
        print(f"Evaluating on tasks 0 to {current_task_id}...")
        
        all_task_metrics = {}
        
        for eval_task_id in range(current_task_id + 1):
            loader_info = self.all_task_loaders[eval_task_id]
            category = loader_info['category']
            test_loader = loader_info['test']
            
            print(f"  Evaluating on Task {eval_task_id} ({category})...")
            
            evaluator = Evaluator(test_loader, self.device)
            results = evaluator.evaluate(self.strategy.model)
            
            self.strategy.model.train()
            
            all_task_metrics[eval_task_id] = {
                'category': category,
                'metrics': results
            }
            
            print(f"Task {eval_task_id} ({category}) Results: {results}")
            
            if self.use_wandb:
                log_dict = {}
                for metric_name, metric_value in results.items():
                    log_dict[f"sequential_eval/after_task_{current_task_id}/task_{eval_task_id}_{category}/{metric_name}"] = metric_value
                wandb.log(log_dict, step=self.global_step)
        
        avg_metrics = self._calculate_average_metrics(all_task_metrics)
        
        print(f"\n  Average metrics across tasks 0-{current_task_id}: {avg_metrics}")
        
        if self.use_wandb:
            avg_log_dict = {}
            for metric_name, metric_value in avg_metrics.items():
                avg_log_dict[f"sequential_eval/after_task_{current_task_id}/average/{metric_name}"] = metric_value
            wandb.log(avg_log_dict, step=self.global_step)
        
        return all_task_metrics, avg_metrics
    
    def _calculate_average_metrics(self, all_task_metrics):
        """
        Calculate average of each metric across all evaluated tasks.
        
        Args:
            all_task_metrics: Dict mapping task_id to {'category': str, 'metrics': dict}
        
        Returns:
            Dict of averaged metrics
        """
        metric_values = defaultdict(list)
        
        for task_id, task_info in all_task_metrics.items():
            metrics = task_info['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.number)):
                    metric_values[metric_name].append(metric_value)
        
        avg_metrics = {}
        for metric_name, values in metric_values.items():
            values = np.array(values)
            avg_metrics[f"{metric_name}_mean"] = np.true_divide(values.sum(), (values != 0).sum())
            # avg_metrics[f"{metric_name}_std"] = np.std(values)
        
        return avg_metrics

    #### TODO - use evaluation logging in this function, not in training
    def evaluate(self, model):
        categories = [item["category"] for item in self.all_task_loaders]
        metrics = {k: {} for k in categories}
        for loader in self.all_task_loaders:
            evaluator = Evaluator(loader['test'], self.device)
            results = evaluator.evaluate(model)
            metrics[loader['category']] = results

        for category, result in metrics.items():
            print(f"For {category} category, the results are like: {result}")
            
        # Log evaluation metrics per category
        if self.use_wandb:
            log_dict = {}
            for metric_name, metric_value in result.items():
                log_dict[f"eval/{category}/{metric_name}"] = metric_value
            wandb.log(log_dict, step=self.global_step)
        
        return metrics
