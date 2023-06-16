"""
  Trains a pytorch classification model using device agnostic code
"""

import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
import data_setup, engine, model, utils

## Set up hyperparameters
NUM_EPOCHS= 5
BATCH_SIZE= 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

## Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device used : {device}')

## Setup transforms
train_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

## Setup the directories
train_dir = "/content/deep_learning_classification/data/pizza_steak_sushi/train"
test_dir = "/content/deep_learning_classification/data/pizza_steak_sushi/test"

train_dataloader, test_dataloader, classes = data_setup.create_dataloader(train_dir= train_dir,
                                                                test_dir = test_dir,
                                                                train_transforms=train_transforms,
                                                                test_transforms=test_transforms,
                                                                batch_size=32)
print(f'Target classes : {classes}')

## Setup model
torch.manual_seed(42)
model = model.TinyVGG(input_shape=3, 
                   hidden_units=HIDDEN_UNITS, 
                   output_shape=len(classes)).to(device)

## Setup the loss function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

## Train the model
results = engine.train(model= model,
            train_dataloader= train_dataloader,
            test_dataloader= test_dataloader,
            loss_fn= loss_fn,
            optimizer= optimizer,
            device= device,
            epochs= NUM_EPOCHS)

## Save the model
utils.save_model(model= model,
                 target_dir= '/content/deep_learning_classification/models' ,
                 model_name= 'script_mode_tinyvgg.pth')
