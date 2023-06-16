"""
  Contains function to display the loss curve given the results from 
  model training
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
def plot_loss_curve(results:Dict[str,List[float]]):
  """
    This function plots the loss and accuracy curve.

    Args:
      results(Dict[String,List[float]]) : A dictionary of the form
      {
        "train_loss":[]
        ,"train_acc":[]
        "test_loss":[]
        "test_acc":[]
      }
    Returns: None
  """

  ## Loss values
  train_loss = results['train_loss']
  test_loss = results['test_loss']

  ## Accuracy values
  train_acc = results['train_acc']
  test_acc = results['test_acc']

  epochs = range(len(train_loss))

  ## Setup the plot
  plt.figure(figsize=(15,7))

  ## plot loss
  plt.subplot(1,2,1)
  plt.plot(epochs,train_loss, label='train')
  plt.plot(epochs,test_loss, label='test')
  plt.title('Loss')
  plt.xlabel('epochs')
  plt.legend()

  ## Plot the accuracy
  plt.subplot(1,2,2)
  plt.plot(epochs,train_acc, label='train')
  plt.plot(epochs,test_acc, label='test')
  plt.title('Accuracy')
  plt.xlabel('epochs')
  plt.legend()
