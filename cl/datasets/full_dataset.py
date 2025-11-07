from typing import List

import torch
from moviad.datasets.iad_dataset import IadDataset
from datasets.continual_dataset import ContinualDataset

class CombinedDataset:
    """
    For training a model on all the available categories (to create a baseline)
    """
    def __init__(self, 
                 dataset: IadDataset,
                 task_type: str,
                 root_dir: str,
                 categories: List[str] = None,
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
            **kwargs
        )

        self.test_dataset = ContinualDataset(
            dataset=dataset,
            task_type=task_type,
            root_dir=root_dir,
            categories=categories,
            split='test',
            **kwargs
        )

        self.categories = self.train_dataset.categories

    def load_train(self, batch_size: int = 32):
        """
        Combines the category datasets from the given dataset,
        then creates train loader from it.
        """
        combined_dataset = []
        
        for task_id in range(len(self.categories)):
            combined_dataset.append(self.train_dataset.get_task_dataset(task_id))
        
        combined_dataset = torch.utils.data.ConcatDataset(combined_dataset)
        # combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
        return combined_dataset
    
    def load_test(self):
        """
        Combines the category datasets from the given dataset,
        then creates test dataset from it.
        Returns the combined dataset with category information preserved.
        """
        combined_datasets = []
        
        for task_id in range(len(self.categories)):
            task_dataset = self.test_dataset.get_task_dataset(task_id)
            combined_datasets.append(task_dataset)
        
        return combined_datasets


