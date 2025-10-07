import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

import numpy as np

from moviad.datasets.iad_dataset import IadDataset


class ContinualDataset:
    def __init__(self, 
                 dataset: IadDataset,
                 task_type: str,
                 root_dir: str,
                 categories: List[str] = None,
                 split: str = 'train',
                 random_seed: int | None = None,
                 **kwargs):
        """
        Initialize the Continual Dataset using IADDataset class.
        
        Args:
            dataset: Dataset class
            task_type: Task type for dataset initialization
            root_dir: Root path for the dataset
            categories: List of category names (default: 6 BMAD categories)
            split: Split type ('train' or 'test')
            **kwargs: Additional arguments for dataset initialization
        """
        self.dataset = dataset
        self.task_type = task_type
        self.root_dir = root_dir
        self.split = split
        self.kwargs = kwargs
        
        # Default BMAD categories
        if (categories is None):
            self.categories = [
                'liver',
                'chest', 
                'histopathology',
                'brain',
                'retinaoct',
                'retinaresc'
            ]
        else:
            self.categories = categories

        if random_seed is not None:
            permutations = np.random.default_rng(seed=random_seed).permutation(len(self.categories)).tolist()
            self.categories = [self.categories[i] for i in permutations]
            
        self.num_categories = len(self.categories)
        self.current_task = 0
        
        # Initialize category datasets
        self.category_datasets = {}
        self._load_category_datasets()
        
    def _load_category_datasets(self):   
        for task_id, category in enumerate(self.categories):
            try:
                # Initialize dataset for this specific category
                dataset = self.dataset(
                    task_type=self.task_type,
                    root_dir=self.root_dir,
                    category=category,
                    split=self.split,
                    **self.kwargs
                )
                
                self.category_datasets[task_id] = dataset
                print(f"({self.split}) Task {task_id} ({category}): {len(dataset)} samples")
                
            except Exception as e:
                print(f"Error loading category {category}: {e}")
                self.category_datasets[task_id] = None
    
    def get_task_dataset(self, task_id: int):
        """
        Get dataset for a specific task.
        
        Args:
            task_id: Task identifier (0 to num_categories-1)
            
        Returns:
            TaskDatasetWrapper object for the specified task
        """
        if task_id >= self.num_categories:
            raise ValueError(f"Task ID {task_id} exceeds number of categories {self.num_categories}")
            
        if self.category_datasets[task_id] is None:
            raise ValueError(f"Dataset for task {task_id} ({self.categories[task_id]}) is not available")
            
        return TaskDatasetWrapper(
            self.category_datasets[task_id],
            task_id,
            self.split
        )
    
    def get_current_task_dataset(self):
        """Get dataset for the current task."""
        return self.get_task_dataset(self.current_task)
    
    def next_task(self) -> bool:
        """
        Move to the next task
        
        Returns:
            True if successfully moved to next task, False if no more tasks
        """
        if self.current_task < self.num_categories - 1:
            self.current_task += 1
            print(f"Moved to task {self.current_task}: {self.categories[self.current_task]}")
            return True
        else:
            print("No more tasks available")
            return False
    
    def reset_tasks(self):
        """Reset to the first task"""
        self.current_task = 0
        print(f"Reset to task 0: {self.categories[0]}")
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task."""
        return {
            'task_id': self.current_task,
            'category': self.categories[self.current_task],
            'total_tasks': self.num_categories,
            'remaining_tasks': self.num_categories - self.current_task - 1,
            'split': self.split
        }
    
    def get_task_dataloader(self, task_id: int, batch_size: int = 32, 
                           shuffle: bool = True, **kwargs):
        """
        Get DataLoader for a specific task
        
        Args:
            task_id: Task identifier
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
        """
        task_dataset = self.get_task_dataset(task_id)
        return DataLoader(task_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_current_task_dataloader(self, batch_size: int = 32, 
                                   shuffle: bool = True, **kwargs) -> DataLoader:
        """Get DataLoader for the current task."""
        return self.get_task_dataloader(self.current_task, batch_size, shuffle, **kwargs)


class TaskDatasetWrapper(Dataset):
    """Dataset wrapper for a single continual learning task."""
    
    def __init__(self, base_dataset: Dataset, task_id: int, split: str):
        """
        Initialize task dataset wrapper.
        
        Args:
            base_dataset: dataset for specific category
            task_id: Task identifier
            task_name: Name of the task/category
        """
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.split = split
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from the task dataset.
        Returns:    (image) --> train
                    (image, label, mask, image path) --> test
        """

        if self.split == "train":
            image = self.base_dataset[idx]
            return image, self.task_id
        
        image, label, mask, image_path = self.base_dataset[idx]
            
        return image, label, mask, image_path
