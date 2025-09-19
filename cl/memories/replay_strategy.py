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
                model_name: str,
                buffer_size: int = 1000,
                device: str = "cuda:0",
                backbone: Optional[str | None] = None,
                ad_layers: Optional[str | None] = None
                ):
        """
        Args:
            buffer_size: Size of replay buffer
            model_config: Configuration for the underlying AD model
        """
        # Initialize underlying AD model (to be implemented by subclasses)
        self.model_name = model_name

        self.ad_model = None
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.current_task = 0

        self.backbone = backbone
        self.ad_layers = ad_layers
        self.device = device
        
        # Training history
        self.training_history = {}

    def _init_model(self):
        self.ad_model = create_vad_model(self.model_name, self.device, backbone=self.backbone)
        self.ad_model.load_model()
        self.model = self.ad_model.ad_model
        self.optimizer = self.ad_model.optimizer
        self.loss = self.ad_model.loss
        self.train_on_batch = create_trainer(self.ad_model).train_on_batch

    def begin_task(self, task_id: int):
        """Called at the beginning of each task."""

        self.current_task = task_id
        print(f"\n=== Beginning Task {task_id} ===")
    
    def partial_update(self, batch: torch.Tensor) -> float:
        """Single training step with replay"""
        self.model.train()
        # Forward pass
        outputs = self.model(batch)
        
        # Compute loss
        loss = self.loss(outputs[0], outputs[1])
        
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

        # TODO: this sjould also evaluate the previous tasks with the current state of the model
        # If there are 3 tasks trained, this method should take all of them separately and evaluate them at each step
        # Also whether image-level or pixel-level, it should return the metrics for each task prettily
        # But I don't know, I might also create another function called evaluate just for this purpose under the continual trainer
        # I hope it's gonna work

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
            self.replay_buffer.add_samples(current_task_samples, self.current_task, self.replay_buffer.buffer_size)
        else:
            # Subsequent tasks: add samples according to the strategy
            samples_to_add = self.replay_buffer.buffer_size // (self.current_task + 1)
            self.replay_buffer.add_samples(current_task_samples, self.current_task, samples_to_add)
        
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