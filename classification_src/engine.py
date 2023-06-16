"""
  Contains function for training and testing a pytorch model
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device
               )-> Tuple[float, float]:
  """
    Trains a pytorch model for single Epoch

    Turns a target Pytorch model to training mode and then executes all the training steps
    forward pass,
    loss calculation,
    zero gradient,
    backpropagation
    update parameters
    Args:
      model(nn.Module): Model to be trained
      dataloader(torch.utils.data.DataLoader) : A DataLoader to be used for training the model
      loss_fn(nn.Module) : A function to calculate the loss
      optimizer(torch.optim.Optimzer) : A pytorch optimizer to help minimize the loss
      device(torch. device) : A target device to compute on 'cpu' or 'cuda'
    
    Returns:
      A tuple of training loss and training accuracy metrics in the form of
      (training_loss, training_acc)

    Example Usage : 
      train_loss, train_acc = train_step(model=model_v0, 
                                         dataloader=train_dataloader,
                                         loss_fn= nn.CrossEntropyLoss(),
                                         optimizer=torch.optim.Adam(params=model_v0.parameters(),lr=0.001),
                                         device='cuda'
      )
    
  """

  #Put the model to train mode
  model.train()

  #Setup the train loss and train accuracy
  train_loss, train_acc = 0,0

  # Iterate over the dataloader
  for batch,(X,y) in enumerate(dataloader):

    #send data to device
    X,y = X.to(device),y.to(device)

    #Forward pass
    pred_logits = model(X)

    # Calculate Loss
    loss = loss_fn(pred_logits, y)
    train_loss += loss.item()

    # Calculate Acc
    pred_label = torch.argmax(torch.softmax(pred_logits,dim=1),dim=1)
    train_acc += (pred_label == y).sum().item()/ len(pred_logits)

    # Zero gradient
    optimizer.zero_grad()

    # Back prop
    loss.backward()

    # update parameters
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device:torch.device)-> Tuple[float,float]:

    """
    Tests a pytorch model for single Epoch

    Turns a target Pytorch model to eval mode and then executes all the testing steps
    forward pass,
    loss calculation,
    
    Args:
      model(nn.Module): Model to be trained
      dataloader(torch.utils.data.DataLoader) : A DataLoader to be used for testing the model
      loss_fn(nn.Module) : A function to calculate the loss
      device(torch. device) : A target device to compute on 'cpu' or 'cuda'
    
    Returns:
      A tuple of testing loss and testing accuracy metrics in the form of
      (testing_loss, testing_acc)

    Example Usage : 
      test_loss, test_acc = test_step(model=model_v0, 
                                         dataloader=train_dataloader,
                                         loss_fn= nn.CrossEntropyLoss(),
                                         device='cuda'
      )
    
  """

    #Put the model in eval mode
    model.eval()

    # Intialize the test_loss and test_acc
    test_loss, test_acc = 0,0
    
    #Turn on the inference context manager
    with torch.inference_mode():
      
      # Loop over the dataloader
      for batch,(X,y) in enumerate(dataloader):
        
        # Send data to target device
        X,y = X.to(device),y.to(device)

        
        # Forward pass
        pred_logits = model(X)

        # Calculate Loss
        loss = loss_fn(pred_logits,y)
        test_loss += loss.item()
        
        # Calculate acc
        pred_labels = torch.argmax(torch.softmax(pred_logits,dim=1),dim=1)
        test_acc += (pred_labels == y).sum().item()/ len(pred_labels) 

      # Calculate the avg loss and accuracy across all batches.  
      test_loss /= len(dataloader)
      test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int) -> Dict[str,List[float]]:
  
  """
    Trains and Tests the pytorch model

    Passes a pytorch model through the train_step and the test_step function for
    a epochs number of times.
    Args:
      model(nn.Module): Model to be trained
      train_dataloader(torch.utils.data.DataLoader) : A DataLoader to be used for training the model
      test_dataloader(torch.utils.data.DataLoader) : A DataLoader to be used for testing the model
      loss_fn(nn.Module) : A function to calculate the loss
      optimizer(torch.optim.Optimzer) : A pytorch optimizer to help minimize the loss
      device(torch.device) : A target device to compute on 'cpu' or 'cuda'
      epochs(int): Number of times the model should pass over the training and the testing dataset.
    
    Returns:
      A dictionary containing the train_loss, train_acc, test_loss & test_acc. 
      Each metrics have a value in list for each epoch

      The output form :
        {
          train_loss : List[float]
          train_acc : List[float]
          test_loss : List[float]
          test_acc : List[float]
        }
  
  """
  # Create a empty dictionary to hold the results

  results = {
      'train_loss':[]
      ,'train_acc':[]
      ,'test_loss':[]
      ,'test_acc':[]
  }

  

  for epoch in tqdm(range(epochs)):
    train_loss , train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    print(f'Epoch : {epoch+1} |'
          f'train_loss : {train_loss:.4f} |'
          f'train_acc : {train_acc:.4f} |'
          f'test_loss : {test_loss:.4f} |'
          f'test_acc : {test_acc:.4f} |' 
    )
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)
  
  return results



