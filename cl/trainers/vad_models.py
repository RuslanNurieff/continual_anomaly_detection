from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


class VADModelBase(ABC):
    """Base class for all VAD (Visual Anomaly Detection) (specifically, the ones without memory bank approach) models."""
    
    def __init__(self, device: str, image_size: Tuple[int, int] = (256, 256)):        
        # Model components - will be set by subclasses
        self.ad_model = None
        self.optimizer = None
        self.loss = None
        self.model_name = self.__class__.__name__.replace('Model', '').lower()
    
    @abstractmethod
    def load_model(self, optimizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Load the specific model architecture and components."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded."""
        return all([
            self.ad_model is not None, 
            self.optimizer is not None, 
            self.loss is not None
        ])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "image_size": self.image_size,
            "is_loaded": self.is_loaded(),
            "num_parameters": sum(p.numel() for p in self.ad_model.parameters()) if self.ad_model else 0
        }