from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import torch

from moviad.trainers import (
    trainer_cfa,
    trainer_fastflow,
    trainer_ganomaly,
    trainer_padim,
    trainer_paste,
    trainer_patchcore,
    trainer_rd4ad,
    trainer_stfpm,
    trainer_supersimplenet
)

from moviad.models.cfa.cfa import *
from moviad.models.fastflow.fastflow import *
from moviad.models.ganomaly.ganomaly import *
from moviad.models.padim.padim import *
from moviad.models.paste.stfpm import *
from moviad.models.patchcore.patchcore import *
from moviad.models.rd4ad.rd4ad import *
from moviad.models.supersimplenet.supersimplenet import *


class ContinualADModel(ABC):
    """
    Abstract base class for continual learning anomaly detection models.
    
    This class defines the interface that all continual learning models
    must implement to work with the training framework.
    """
    
    def __init__(self, current_task_id, **kwargs):
        """Initialize the model with configuration parameters."""
        self.current_task_id = current_task_id
        self.config = kwargs
    
    @abstractmethod
    def begin_task(self, task_id: int, task_data: Any) -> None:
        """
        Called at the beginning of each new task.
        
        Args:
            task_id: Identifier for the current task
            task_data: Data or metadata for the current task
        """
        pass
    
    @abstractmethod
    def partial_update(self, batch_data: Any, **kwargs) -> Dict[str, float]:
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
    
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save_model(self, path: str) -> None:
        """Save the model state. Override if custom saving is needed."""
        if hasattr(self, 'state_dict'):
            torch.save(self.state_dict(), path)
        else:
            raise NotImplementedError("save_model not implemented")
    
    def load_model(self, model_name) -> None:
        """Load the model state. Override if custom loading is needed."""
        if model_name == "cfa":
            pass