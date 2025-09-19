from trainers.vad_models import VADModelBase
from typing import Optional, Tuple, Dict, Any

import torch

from moviad.models.fastflow.fastflow import create_fastflow
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.models import STFPM
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.trainers.trainer_rd4ad import TrainerRD4AD
from moviad.trainers.trainer_stfpm import TrainerSTFPM

class FastFlowModel(VADModelBase):
    """FastFlow model implementation."""
    
    def __init__(self, device: str, backbone: str, image_size: Tuple[int, int] = (256, 256)):
        super().__init__(device, image_size)
        if backbone is None:
            raise ValueError("You have to define a backbone for FastFlow!")
        self.backbone = backbone
        self.device = device
        self.image_size = image_size
    
    def load_model(self, optimizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Load FastFlow model with configurable optimizer."""
        try:
            self.ad_model = create_fastflow(self.image_size, self.backbone, self.device)
            
            # Use default Adam if no config provided
            if optimizer_config is None:
                self.optimizer = torch.optim.Adam(self.ad_model.parameters())
            else:
                self.optimizer = self._create_optimizer(optimizer_config)
            
            self.loss = TrainerFastFlow.fastflow_loss
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FastFlow model: {e}")
    
    def _create_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer for FastFlow."""
        optimizer_type = config.get("type", "adam").lower()
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 0)
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.ad_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            momentum = config.get("momentum", 0.9)
            return torch.optim.SGD(
                self.ad_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


class RD4ADModel(VADModelBase):
    """RD4AD (Reverse Distillation for Anomaly Detection) model implementation."""
    
    def __init__(self, device: str, backbone: str, image_size: Tuple[int, int] = (256, 256)):
        super().__init__(device, image_size)
        if backbone is None:
            raise ValueError("You have to define a backbone for Reverse Distillation!")
        self.backbone = backbone
    
    def load_model(self, optimizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Load RD4AD model with its specific optimizer configuration."""
        try:
            self.ad_model = RD4AD(self.backbone, self.device, self.image_size)
            
            # RD4AD has specific optimizer requirements - use defaults or override
            if optimizer_config is None:
                self.optimizer = torch.optim.Adam(
                    list(self.ad_model.decoder.parameters()) + list(self.ad_model.bn.parameters()),
                    lr=RD4AD.DEFAULT_PARAMETERS["learning_rate"],
                    betas=RD4AD.DEFAULT_PARAMETERS["betas"],
                )
            else:
                self.optimizer = self._create_optimizer(optimizer_config)
            
            self.loss = TrainerRD4AD.loss_function
            
        except Exception as e:
            raise RuntimeError(f"Failed to load RD4AD model: {e}")
    
    def _create_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer for RD4AD with specific parameter groups."""
        optimizer_type = config.get("type", "adam").lower()
        lr = config.get("lr", RD4AD.DEFAULT_PARAMETERS["learning_rate"])
        
        # RD4AD requires specific parameter groups
        parameters = list(self.ad_model.decoder.parameters()) + list(self.ad_model.bn.parameters())
        
        if optimizer_type == "adam":
            betas = config.get("betas", RD4AD.DEFAULT_PARAMETERS["betas"])
            return torch.optim.Adam(parameters, lr=lr, betas=betas)
        elif optimizer_type == "sgd":
            momentum = config.get("momentum", 0.9)
            return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


class STFPMModel(VADModelBase):
    """STFPM (Student-Teacher Feature Pyramid Matching) model implementation."""
    
    def __init__(self, device: str, backbone: str, ad_layers: str, image_size: Tuple[int, int] = (256, 256)):
        super().__init__(device, image_size)
        self.backbone = backbone
        self.ad_layers = ad_layers
    
    def load_model(self, optimizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Load STFPM model with its specific optimizer configuration."""
        try:
            self.ad_model = STFPM(self.backbone, self.image_size, self.image_size, self.ad_layers)
            
            # STFPM uses SGD by default
            if optimizer_config is None:
                self.optimizer = torch.optim.SGD(
                    self.ad_model.student.model.parameters(),
                    STFPM.DEFAULT_PARAMETERS["learning_rate"],
                    momentum=STFPM.DEFAULT_PARAMETERS["momentum"],
                    weight_decay=STFPM.DEFAULT_PARAMETERS["weight_decay"]
                )
            else:
                self.optimizer = self._create_optimizer(optimizer_config)
            
            self.loss = TrainerSTFPM.stfpm_loss
            
        except Exception as e:
            raise RuntimeError(f"Failed to load STFPM model: {e}")
    
    def _create_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer for STFPM."""
        optimizer_type = config.get("type", "sgd").lower()
        lr = config.get("lr", STFPM.DEFAULT_PARAMETERS["learning_rate"])
        
        # STFPM optimizes only student parameters
        parameters = self.ad_model.student.model.parameters()
        
        if optimizer_type == "sgd":
            momentum = config.get("momentum", STFPM.DEFAULT_PARAMETERS["momentum"])
            weight_decay = config.get("weight_decay", STFPM.DEFAULT_PARAMETERS["weight_decay"])
            return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == "adam":
            weight_decay = config.get("weight_decay", 0)
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_vad_model(model_name: str, device: str, **kwargs) -> VADModelBase:
    """Factory function to create VAD models."""
    model_name = model_name.lower()
    
    if model_name == "fastflow":
        if "backbone" not in kwargs:
            raise ValueError("FastFlow requires 'backbone' parameter")
        return FastFlowModel(device, **kwargs)
    
    elif model_name == "rd4ad":
        if "backbone" not in kwargs:
            raise ValueError("RD4AD requires 'backbone' parameter")
        return RD4ADModel(device, **kwargs)
    
    elif model_name == "stfpm":
        required_params = ["backbone", "ad_layers"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"STFPM requires '{param}' parameter")
        return STFPMModel(device, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: fastflow, rd4ad, stfpm")