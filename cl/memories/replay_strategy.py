import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from cl_ad_model import ContinualADModel

from trainers.vad_models import VADModelBase
from memories.replay_buffer import ReplayBuffer

from trainers.models import create_vad_model
from trainers.trainers import create_trainer


class ReplayModel(ContinualADModel):
    """
    Abstract base class for Replay-based Anomaly Detection Models.
    Implements the replay memory for continual learning.
    """
    
    def __init__(self,
                model_conf: dict,
                buffer_size: int = 1000,
                device: str = "cuda:0",
                ):
        """
        Args:
            buffer_size: Size of replay buffer
            model_config: Configuration for the underlying AD model
        """
        # Initialize underlying AD model (to be implemented by subclasses)
        self.model_conf = model_conf

        self.ad_model = None
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.current_task = 0
        self.device = device
        
        # Training history
        self.training_history = {}

    def _init_model(self):
        self.ad_model = self.model_conf['stfpm']
        self.ad_model.load_model()
        self.model = self.ad_model.ad_model
        self.model.train()
        self.optimizer = self.ad_model.optimizer
        # self.loss = self.ad_model.loss
        self.train_on_batch = create_trainer(self.ad_model).train_on_batch

    def begin_task(self, task_id: int):
        """Called at the beginning of each task."""

        self.current_task = task_id
        print(f"\n=== Beginning Task {task_id} ===")
    
    def partial_update(self, batch: torch.Tensor) -> float:
        """Single training step with replay"""
        # Forward pass
        # outputs = self.model(batch)
        
        # Compute loss
        # loss = self.loss(outputs[0], outputs[1])
        loss = self.train_on_batch(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def end_task(self, current_task_loader: DataLoader):
        """
        Called at the end of each task.
        Updates replay buffer with samples from current task.
        
        Args:
            current_task_loader: DataLoader for the task that just finished
        """
        print(f"\nEnding task {self.current_task}")
        
        # Calculate how many samples we need upfront
        if self.current_task == 0:
            samples_needed = self.replay_buffer.buffer_size
        else:
            samples_needed = self.replay_buffer.buffer_size // (self.current_task + 1)
        
        # Get dataset and sample efficiently
        dataset = current_task_loader.dataset
        total_samples = len(dataset)
        
        if samples_needed >= total_samples:
            # Need all samples - just iterate through dataset directly
            current_task_samples = [dataset[i] for i in range(total_samples)]
        else:
            # Sample randomly without loading everything into memory
            import random
            random_indices = random.sample(range(total_samples), samples_needed)
            current_task_samples = [dataset[i] for i in random_indices]
        
        # Add samples to replay buffer
        self.replay_buffer.add_samples(current_task_samples, self.current_task, len(current_task_samples))
        
        # Print buffer status
        buffer_info = self.replay_buffer.get_buffer_info()
        print(f"Replay buffer updated: {buffer_info}")

    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data."""
        return self._evaluate_model(test_loader)
    
    def get_model_state(self, ad_model: VADModelBase):
        """Get current model state for saving/loading."""
        return {
            'ad_model_state': ad_model.state_dict() if ad_model else None,
            'current_task': self.current_task,
            'replay_buffer': self.replay_buffer.buffer.copy(),
            'training_history': self.training_history.copy()
        }
    
    def load_model_state(self, ad_model: VADModelBase, state_dict: Dict[str, Any]):
        """Load model state."""
        if state_dict.get('ad_model_state') and ad_model:
            ad_model.load_state_dict(state_dict['ad_model_state'])
        self.current_task = state_dict.get('current_task', 0)
        self.replay_buffer.buffer = state_dict.get('replay_buffer', [])
        self.training_history = state_dict.get('training_history', {})