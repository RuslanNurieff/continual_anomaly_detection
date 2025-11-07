from typing import List

from moviad.datasets.iad_dataset import IadDataset
from datasets.continual_dataset import ContinualDataset

class StreamManager:
    """
    Main class to handle both train and test continual datasets
    """
    def __init__(self, 
                 dataset: IadDataset,
                 task_type: str,
                 root_dir: str,
                 categories: List[str] = None,
                 random_seed: int | None = None,
                 **kwargs):
        """
        Initialize manager for both train and test datasets
        
        Args:
            dataset: Dataset class
            task_type: Task type for dataset
            root_dir: Root path for dataset
            categories: List of category names
            **bmad_kwargs: Additional dataset arguments
        """
        self.train_dataset = ContinualDataset(
            dataset=dataset,
            task_type=task_type,
            root_dir=root_dir,
            categories=categories,
            split='train',
            random_seed=random_seed,
            **kwargs
        )
        
        self.test_dataset = ContinualDataset(
            dataset=dataset,
            task_type=task_type,
            root_dir=root_dir,
            categories=categories, # randomly choose (if needed) the tasks [0, 5, etc.] by choosing a seed.
            split='test',
            random_seed=random_seed,
            **kwargs
        )

        self.num_categories = self.train_dataset.num_categories
        self.category_ds = self.train_dataset.category_datasets
        
    def get_current_task_loaders(self, batch_size: int = 32, 
                                shuffle_train: bool = True,
                                shuffle_test: bool = False,
                                **kwargs):
        """
        Get train and test  loaders for current task,
        
        Returns:
            (train_loader, test_loader)
        """
        train_loader = self.train_dataset.get_current_task_dataloader(
            batch_size=batch_size, shuffle=shuffle_train, **kwargs
        )
        test_loader = self.test_dataset.get_current_task_dataloader(
            batch_size=batch_size, shuffle=shuffle_test, **kwargs
        )
        
        return train_loader, test_loader
    
    def get_task_by_id(self, task_id: int,
                       batch_size: int = 32,
                       shuffle_train: bool = True,
                       shuffle_test: bool = False,
                       **kwargs):
        """
        Get train and test  loaders for a specified task,
        
        Returns:
            (train_loader, test_loader)
        """        
        train_loader = self.train_dataset.get_task_dataloader(
            task_id=task_id, batch_size=batch_size, shuffle=shuffle_train, **kwargs
            )
        test_loader = self.test_dataset.get_task_dataloader(
            task_id=task_id, batch_size=batch_size, shuffle=shuffle_test, **kwargs
            )

        return train_loader, test_loader
    
    def to_next_task(self):
        """Move both datasets to next task."""
        train_next = self.train_dataset.next_task()
        test_next = self.test_dataset.next_task()
        return train_next and test_next
    
    def reset_tasks(self):
        """Reset both datasets to first task."""
        self.train_dataset.reset_tasks()
        self.test_dataset.reset_tasks()
    
    def get_task_info(self):
        """Get current task information."""
        return self.train_dataset.get_task_info()