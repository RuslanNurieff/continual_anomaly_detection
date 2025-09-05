from cl_ad_model import CLADModel
from typing import Any, Dict
import torch

class ReplayMemory(CLADModel):
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.replay_memory = []

    def begin_task(self, task_id: int, task_data: Any) -> None:
        # Initialize or reset replay memory for the new task
        self.replay_memory = []

    def partial_update(self, batch_data: Any, **kwargs) -> Dict[str, float]:
        # Add new data to replay memory
        self.replay_memory.extend(batch_data)
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory = self.replay_memory[-self.memory_size:]

        # Perform training using both current batch and replay memory
        combined_data = batch_data + self.replay_memory
        # Placeholder for actual training logic
        loss = self.train_on_data(combined_data)
        
        return {"loss": loss}

    def end_task(self, task_id: int) -> None:
        # Optionally consolidate or prune replay memory at the end of the task
        tasks = [i for i in range(self.current_task_id)]
        