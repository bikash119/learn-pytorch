"""
  Contains various utility functions for pytorch model training and saving
"""

import torch
from torch import nn
from pathlib import Path

def save_model(model: nn.Module,
              target_dir: str,
              model_name: str):
  """
    Saves a pytorch model to a target directory
    Args:
      model (nn.Module): a pytorch model to be saved
      target_dir (str): A directory for saving the model
      model_name (str): The filename to given to the model. Should include either 
                        .pt or .pth as the file extension 

    Returns : None

    Example usage
      save_model(model=model_0,
                 target_dir="models",
                 model_name="deep_learning_classification.pth")
  """

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,exist_ok=True)

  # Create model save path
  assert model_name.endswith('.pt') or model_name.endswith('pth'),"model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path/model_name

  # Save the model state_dict()
  print(f'[INFO] Saving model to : {model_save_path}')
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  

