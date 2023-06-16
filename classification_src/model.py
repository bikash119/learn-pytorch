"""
  Contains the model architecture code
"""


import torch
from torch import nn

class TinyVGG(nn.Module):

  """
    Create the TinyVGG architecure

    Replicates the TinyVGG architecture from the CNN Explainer website in pytorch.
    See the original architecture here: https://poloclub.github.io

    Args:
      input_shape(int) : An integer indicating number of input channels.
      hidden_units(int) : An integer indicating number of hidden units between layers
      output_shape(int) : An integer indicating number of classes

    Example Usage
      model_v0 = TinyVGG(input_shape=3,
                         hidden_units=10,
                         output_units=3)
  """
  def __init__(self, 
               input_shape: int,
               hidden_units:int,
               output_shape: int) -> None:
    
    super().__init__()

    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier= nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units * 13 * 13,
                  out_features=output_shape)
    )

  def forward(self,x)-> torch.Tensor:
    return self.classifier(self.block_2(self.block_1(x)))
