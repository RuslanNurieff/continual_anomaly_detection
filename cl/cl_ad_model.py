from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import torch

class ContinualADModel(ABC):
    """
    Abstract base class for continual learning anomaly detection models.
    
    This class defines the interface that all continual learning models
    must implement to work with the training framework.
    """

    @abstractmethod
    def begin_task(self, task_id: int) -> None:
        """
        Called at the beginning of each new task.
        
        Args:
            task_id: Identifier for the current task
            task_data: Data or metadata for the current task
        """
        pass
    
    @abstractmethod
    def partial_update(self) -> Dict[str, float]:
        """
        Update the model with a batch of data during training.
        
        Args:
            batch_data: A batch of training data
            **kwargs: Additional parameters for the update
            
        Returns:
            Dictionary containing training metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def end_task(self, task_id: int) -> None:
        """
        Called at the end of each task for cleanup/consolidation.
        
        Args:
            task_id: Identifier for the completed task
        """
        pass