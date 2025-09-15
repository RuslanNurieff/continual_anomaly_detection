import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from typing import List, Tuple, Dict, Any
import random
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.trainers.trainer_fastflow import TrainerFastFlow

from moviad.datasets.bmad.bmad_dataset import BMAD
from memory_stream import ContinualDataset
from moviad.utilities.configurations import TaskType


class ReplayBuffer:
    def __init__(self, buffer_size: int = 1000):
        """
        Args:
            buffer_size: Maximum number of samples to store
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.task_counts = {}

    def add_samples(self, samples: List[Tuple], task_id: int, num_samples: int = None):
        """
        Add samples to the replay buffer.
        
        Args:
            samples: List of (image, label, task_id) tuples from current task
            task_id: Current task identifier
            num_samples: Number of samples to add (if None, add all available up to buffer_size)
        """
        if num_samples is None:
            num_samples = min(len(samples), self.buffer_size)
        
        # Randomly select samples to add
        if len(samples) > num_samples:
            selected_indices = random.sample(range(len(samples)), num_samples)
            samples_to_add = [samples[i] for i in selected_indices]
        else:
            samples_to_add = samples.copy()
        
        if task_id == 0:  # First task
            # Simply add samples up to buffer size
            self.buffer = samples_to_add[:self.buffer_size]
            self.task_counts[task_id] = len(self.buffer)
            print(f"Added {len(self.buffer)} samples from task {task_id} to buffer")
            
        else:  # Subsequent tasks
            # Calculate how many samples to add from current task
            samples_per_task = self.buffer_size // (task_id + 1)
            samples_to_add_count = min(samples_per_task, len(samples_to_add))
            
            # Remove random samples from previous tasks to make space
            samples_to_remove = samples_to_add_count
            if samples_to_remove > 0:
                # Get indices of samples to remove (randomly from all previous tasks)
                remove_indices = random.sample(range(len(self.buffer)), 
                                             min(samples_to_remove, len(self.buffer)))
                
                # Remove samples (in reverse order to maintain indices)
                for idx in sorted(remove_indices, reverse=True):
                    removed_sample = self.buffer.pop(idx)
                    removed_task_id = removed_sample[1]
                    self.task_counts[removed_task_id] -= 1
            
            # Add new samples
            new_samples = samples_to_add[:samples_to_add_count]
            self.buffer.extend(new_samples)
            self.task_counts[task_id] = samples_to_add_count
            
            print(f"Added {samples_to_add_count} samples from task {task_id} to buffer")
            print(f"Buffer: {self.task_counts}")
    
    def get_replay_samples(self):
        """Get all samples from replay buffer."""
        return self.buffer.copy()
    
    def get_buffer_dataset(self):
        """Get replay buffer as a Dataset object."""
        return ReplayDataset(self.buffer)
    
    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def get_buffer_info(self):
        """Get information about current buffer state."""
        return {
            'total_samples': len(self.buffer),
            'buffer_size': self.buffer_size,
            'task_distribution': self.task_counts.copy(),
            'utilization': len(self.buffer) / self.buffer_size
        }


class ReplayDataset(Dataset):
    """Dataset wrapper for replay buffer samples."""
    
    def __init__(self, buffer_samples: List[Tuple]):
        """
        Initialize replay dataset.
        
        Args:
            buffer_samples: List of (image, label, task_id) tuples
        """
        self.samples = buffer_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item from replay dataset."""
        return self.samples[idx]  #Returns (image, task_id)


class REPLAY_AD_MODEL:
    """
    Abstract base class for Replay-based Anomaly Detection Models.
    Implements the replay memory for continual learning.
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 model_config: Dict = None,
                ):
        """
        Args:
            buffer_size: Size of replay buffer
            model_config: Configuration for the underlying AD model
        """
        self.buffer_size = buffer_size
        self.model_config = model_config

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize underlying AD model (to be implemented by subclasses)
        self.ad_model = None
        self.current_task = 0
        
        # Training history
        self.training_history = {}

        # Load the dataset
        self.continual_dataset = ContinualDataset(
            bmad_class=BMAD,
            task_type=TaskType.SEGMENTATION,
            root_dir='/mnt/disk1/ruslan_nuriev/bmad',
        )
        
    def _initialize_model(self):
        self.ad_model = create_fastflow((256, 256), "mobilenet_v2", 'cuda:0')
        self.optimizer = torch.optim.Adam(
            self.ad_model.parameters()
        )
        self.loss = TrainerFastFlow.fastflow_loss
    
    def _train_on_batch(self, batch_data, **kwargs):
        batch_data = batch_data.to(self.ad_model.device)
        hidden_variables, jacobians = self.ad_model(batch_data)
        batch_loss = self.loss(hidden_variables, jacobians)
        return batch_loss
    
    def begin_task(self, task_id: int):
        """Called at the beginning of each task."""
        self.current_task = task_id
        print(f"\n=== Beginning Task {task_id} ===")
        
        if self.ad_model is None:
            self._initialize_model()
            self.ad_model.train()
    
    def train_task(self, current_task_loader: DataLoader, 
                   num_epochs: int = 10, **training_kwargs):
        """
        Train the model on current task with replay.
        
        Args:
            current_task_loader: DataLoader for current task
            num_epochs: Number of training epochs
            **training_kwargs: Additional training parameters
        """
        # print(f"Training on task {self.current_task}")
        if self.model_config is not None:
            self.ad_model.load_state_dict(self.model_config)
        
        # Prepare training data (current task + replay buffer)
        if not self.replay_buffer.is_empty():
            # Combine current task data with replay buffer
            replay_dataset = self.replay_buffer.get_buffer_dataset()
            current_task_dataset = current_task_loader.dataset
            
            # Create combined dataset
            combined_dataset = ConcatDataset([current_task_dataset, replay_dataset])
            combined_loader = DataLoader(
                combined_dataset, 
                batch_size=current_task_loader.batch_size,
                shuffle=True,
                num_workers=getattr(current_task_loader, 'num_workers', 0)
            )
            
            print(f"Training with {len(current_task_dataset)} current samples + "
                  f"{len(replay_dataset)} replay samples")
            training_loader = combined_loader
        else:
            print(f"Training with {len(current_task_loader.dataset)} current samples (no replay)")
            training_loader = current_task_loader
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for _, batch_data in enumerate(training_loader):
                batch_loss = self._train_on_batch(batch_data, **training_kwargs)
                epoch_loss += batch_loss
                num_batches += 1

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Store training history
        self.training_history[self.current_task] = {
            'epochs': num_epochs,
            'final_loss': avg_loss,
            'replay_samples': len(self.replay_buffer.buffer)
        }

        self.model_config = self.ad_model.state_dict()
        self.end_task(current_task_loader)
        self.current_task += 1

    
    def partial_update(self):
        """
        Perform partial update on a single batch.
        Used for online/incremental learning scenarios.
        """
        self.begin_task(self.continual_dataset.get_task_info()['task_id'])



        for task in range(self.continual_dataset.num_categories):
            print(f"Training on task {task}: {self.continual_dataset.get_task_info()['category']}")
            
            # Get data loaders for current task
            train_loader, _ = self.continual_dataset.get_current_task_loaders()

            self.train_task(train_loader)
            torch.save(self.ad_model.state_dict(), "trained_model.pth")
            
            # Move to next task
            if not self.continual_dataset.to_next_task():
                break
    
    def end_task(self, current_task_loader: DataLoader):
        """
        Called at the end of each task.
        Updates replay buffer with samples from current task.
        
        Args:
            current_task_loader: DataLoader for the task that just finished
        """
        print(f"\nEnding task {self.current_task}")
        
        # Collect all samples from current task
        current_task_samples = []
        for batch_data in current_task_loader:
            images = batch_data
            for i in range(len(images)):
                current_task_samples.append(
                    images[i]
                )
        
        # Add samples to replay buffer
        if self.current_task == 0:
            # First task: add up to buffer_size samples
            self.replay_buffer.add_samples(current_task_samples, self.current_task, self.buffer_size)
        else:
            # Subsequent tasks: add samples according to the strategy
            samples_to_add = self.buffer_size // (self.current_task + 1)
            self.replay_buffer.add_samples(current_task_samples, self.current_task, samples_to_add)
        
        # Print buffer status
        buffer_info = self.replay_buffer.get_buffer_info()
        print(f"Replay buffer updated: {buffer_info}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data."""
        return self._evaluate_model(test_loader)
    
    def get_model_state(self):
        """Get current model state for saving/loading."""
        return {
            'ad_model_state': self.ad_model.state_dict() if self.ad_model else None,
            'current_task': self.current_task,
            'replay_buffer': self.replay_buffer.buffer.copy(),
            'training_history': self.training_history.copy()
        }
    
    def load_model_state(self, state_dict: Dict[str, Any]):
        """Load model state."""
        if state_dict.get('ad_model_state') and self.ad_model:
            self.ad_model.load_state_dict(state_dict['ad_model_state'])
        self.current_task = state_dict.get('current_task', 0)
        self.replay_buffer.buffer = state_dict.get('replay_buffer', [])
        self.training_history = state_dict.get('training_history', {})