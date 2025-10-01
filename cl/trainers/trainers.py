from trainers.vad_models import VADModelBase
from abc import ABC, abstractmethod

import torch.nn.functional as F

class BaseTrainer(ABC):
    """Base trainer class for all VAD models."""
    
    def __init__(self, vad_model: VADModelBase):
        if not vad_model:
            raise RuntimeError("Model must be loaded before creating trainer")
        self.vad_model = vad_model
    
    @abstractmethod
    def train_on_batch(self, batch_data):
        """Train on a single batch."""
        pass
    
    def get_model_name(self) -> str:
        """Get the model name for this trainer."""
        return self.vad_model.model_name


class FastFlowTrainer(BaseTrainer):
    """Trainer for FastFlow models."""
    
    def train_on_batch(self, batch_data):
        batch_data = batch_data.to(self.vad_model.device)
        hidden_variables, jacobians = self.vad_model.ad_model(batch_data)
        batch_loss = self.vad_model.loss(hidden_variables, jacobians)
        return batch_loss


class RD4ADTrainer(BaseTrainer):
    """Trainer for RD4AD models."""
    
    def train_on_batch(self, batch_data):
        batch_data = batch_data.to(self.vad_model.device)
        teacher_features, student_features = self.vad_model.ad_model(batch_data)
        batch_loss = self.vad_model.loss(teacher_features, student_features)
        return batch_loss


class STFPMTrainer(BaseTrainer):
    """Trainer for STFPM models."""
    
    def train_on_batch(self, batch_data):
        batch_data = batch_data.to(self.vad_model.device)
        teacher_features, student_features = self.vad_model.ad_model(batch_data)
        batch_loss = 0
        for i in range(len(student_features)):
            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
            student_features[i] = F.normalize(student_features[i], dim=1)
            batch_loss += self.vad_model.loss(teacher_features[i], student_features[i])
    
        return batch_loss


# Trainer factory
def create_trainer(vad_model: VADModelBase) -> BaseTrainer:
    """Factory function to create appropriate trainer for the model."""
    model_name = vad_model.model_name
    
    if model_name == "fastflow":
        return FastFlowTrainer(vad_model)
    elif model_name == "rd4ad":
        return RD4ADTrainer(vad_model)
    elif model_name == "stfpm":
        return STFPMTrainer(vad_model)
    else:
        raise ValueError(f"No trainer available for model: {model_name}")