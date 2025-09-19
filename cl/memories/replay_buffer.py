from torch.utils.data import Dataset
from typing import List, Tuple
import random


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
            samples: List of (image, task_id) tuples from current task
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
            buffer_samples: List of images
        """
        self.samples = buffer_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item from replay dataset."""
        return self.samples[idx]  #Returns (image, task_id)