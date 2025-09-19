from cl_ad_model import ContinualADModel
from memories.memory_stream import ContinualDataset

from typing import List, Dict
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader

from trainers.models import create_vad_model
from trainers.trainers import create_trainer

from memories.replay_strategy import ReplayModel
from collections import defaultdict

class ContinualTrainer:
    """Manages the continual learning training process"""
    
    def __init__(self, 
                 strategy: ContinualADModel,
                ): # evaluator: ContinualEvaluator = None)
        
        self.strategy = strategy
        # self.evaluator = evaluator or ContinualEvaluator()
        self.history = defaultdict(list)
        
    def train(self, continual_dataset: ContinualDataset, epochs_per_task: int = 10):
        """Train on all tasks sequentially"""
        
        self.all_task_loaders = []  # Keep for evaluation later

        self.strategy._init_model()
        
        for task_id in range(continual_dataset.num_categories):
            print(f"\n{'='*50}")
            print(f"Task {task_id}: {continual_dataset.get_task_info()['category']}")
            print('='*50)
            
            # Get data loaders
            train_loader, test_loader = continual_dataset.get_current_task_loaders()
            self.all_task_loaders.append({
                'task_id': task_id,
                'train': train_loader,
                'test': test_loader,
                'category': continual_dataset.get_task_info()['category']
            })
            
            # Begin task
            self.strategy.begin_task(task_id)

            ################################
            # Prepare training data (current task + replay buffer)
            if self.strategy.replay_buffer.buffer: #self.strategy.replay_buffer.buffer
                # Combine current task data with replay buffer
                replay_dataset = self.strategy.replay_buffer.get_buffer_dataset()
                current_task_dataset = train_loader.dataset
                
                # Create combined dataset
                combined_dataset = ConcatDataset([current_task_dataset, replay_dataset])
                combined_loader = DataLoader(
                    combined_dataset, 
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=getattr(train_loader, 'num_workers', 0)
                )
                
                print(f"Training with {len(current_task_dataset)} current samples + "
                    f"{len(replay_dataset)} replay samples")
                training_loader = combined_loader
            else:
                print(f"Training with {len(train_loader.dataset)} current samples (no replay)")
                training_loader = train_loader

            ################################
            
            # Training loop
            for epoch in range(epochs_per_task):
                epoch_losses = []
                for batch in training_loader:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                    
                    loss = self.strategy.partial_update(images)
                    epoch_losses.append(loss)
                
                avg_loss = np.mean(epoch_losses)
                print(f"  Epoch {epoch+1}/{epochs_per_task}, Loss: {avg_loss:.4f}")
                self.history[f'task_{task_id}_loss'].append(avg_loss)
            
            # End task - update replay buffer
            self.strategy.end_task(train_loader)
            
            # Move to next task
            if not continual_dataset.to_next_task():
                break
        
        return self.history
