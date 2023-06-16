"""
Contains functionality to create pytorch DataLoaders from image classification usecases
"""

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List

NUM_WORKERS = os.cpu_count()
def create_dataloader(train_dir:str,
                      test_dir: str,
                      train_transforms: transforms.Compose,
                      test_transforms: transforms.Compose,
                      batch_size: int,
                      num_workers: int=NUM_WORKERS) -> Tuple[DataLoader, DataLoader, List[str]]:
  """
   Creates train and test dataloaders

    Takes in training directory and testing directory and creates Pytorch Datasets which are then 
    used to create Pytorch DataLoaders

    Args:
      train_dr(str) : Folder containing images to be used for training the model
      test_dir(str) : Folder containing images to be used for testing the model
      train_transforms ( transforms.Compose) : Transformation to be applied on the images used for training
      test_transform ( transforms.Compose) : Transformation to be applied on the images used for testing
      batch_size(int) : Size of mini batch or Number of samples per batch in each DataLoaders
      num_workers (int) : Number of workers per DataLoader ( mostly equals to the number of CPU)

    Returns:
      A Tuple containing 
        train_dataloader ( torch.utils.data.DataLoader): A pytorch DataLoader for training dataset
        test_dataloader ( torch.utils.data.DataLoader) : A pytorch DataLoader for testing dataset
        classes ( List[str]) : A list of string representing the image classes.
    
    Example usage:
      train_dataloader, test_dataloader, classes = create_dataloaders(train_dir = path/to/train_img/folder,
                                                                      test_dir = path/to/train_img/folder,
                                                                      train_transforms= some_tranforms,
                                                                      test_transforms = some_transforms,
                                                                      batch_size= 32,
                                                                      num_workers = 2)
  """

  # Create pytorch datasets using images in train and test folder
  train_dataset = datasets.ImageFolder(root=train_dir,
                                       transform=train_transforms)
  test_dataset = datasets.ImageFolder(root=test_dir,
                                      transform=test_transforms)
  
  ## Get Image classes
  classes = train_dataset.classes

  # Create pytorch dataloaders from datasets
  train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=NUM_WORKERS)
  
  test_dataloader = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=NUM_WORKERS)
  
  return train_dataloader, test_dataloader, classes
  
