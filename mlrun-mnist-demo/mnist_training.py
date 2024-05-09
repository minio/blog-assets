import json
import os
from time import time
from typing import List, Dict, Any, Tuple

import mlrun
import mlrun.frameworks.pytorch as mlrun_torch
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch import optim

import data_utilities as du
import torch_utilities as tu


def epoch_loop(model: nn.Module, loader: DataLoader, params: Dict[str, Any]) -> Dict[str, Any]:
    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

    for epoch in range(params['epochs']):
        total_loss = 0
        for images, labels in loader:
            images = images.to(training_parameters['device'])
            labels = labels.to(training_parameters['device'])
            # Flatten MNIST images into a 784 long vector.
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)

            loss = loss_func(output, labels)
            
            # This is where the model learns by backpropagating
            loss.backward()
            
            # And optimizes its weights here
            optimizer.step()
            
            total_loss += loss.item()

        print("Epoch {} - Training loss: {}".format(epoch+1, total_loss/len(loader)))

        
@mlrun.handler()
def train_model(context: mlrun.MLClientCtx, loader_type: str='memory', data_bucket: str=None, training_parameters: Dict=None):
    logger = du.create_logger()
    logger.info(loader_type)
    logger.info(data_bucket)
    logger.info(training_parameters)

    # Load the data and log loading metrics.
    if loader_type == 'memory':
        train_loader, test_loader, _ = tu.create_memory_data_loaders(training_parameters['batch_size'])
    elif loader_type == 'minio-by-batch':
        train_loader, test_loader, _ = tu.create_minio_data_loaders(training_parameters['batch_size'])
    else:
        raise Exception('Unknown loader type. Must be "memory" or "minio-by-batch"')
    
    # Train the model and log training metrics.
    logger.info('Creating the model.')    
    model = tu.MNISTModel(training_parameters['input_size'], training_parameters['hidden_sizes'], training_parameters['output_size'])
    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=training_parameters['lr'], momentum=training_parameters['momentum'])

    # Train the model.
    logger.info('Training the model.')    
    mlrun_torch.train(
        model=model,
        training_set=train_loader,
        loss_function=loss_func,
        optimizer=optimizer,
        validation_set=test_loader,
        #metric_functions=[accuracy],
        epochs=training_parameters['epochs'],
        custom_objects_map={"torch_utilities.py": "MNISTModel"},
        custom_objects_directory=os.getcwd(),
        context=context,
    )

    #start_time = time()
    #epoch_loop(model, train_loader, training_parameters)
    #training_time_sec = (time()-start_time)
    #print('\nTraining Time (in seconds) =', training_time_sec)

    # Test the model and log the accuracy as a metric.
    testing_metrics = tu.test_model_local(model, test_loader, training_parameters['device'])
    print(testing_metrics)


def accuracy(y_true, y_pred):
    ps = torch.exp(y_pred)
    probab = list(ps[0])
    pred_label = probab.index(max(probab))
    return (sum(pred_label == y_true) / y_true.size()[0]).item()


if __name__ == "__main__":

    # training configuration
    training_parameters = {
        'batch_size': 32,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'dropout_input': 0.2,
        'dropout_hidden': 0.5,
        'epochs': 35,
        'input_size': 784,
        'hidden_sizes': [1024, 1024, 1024, 1024],
        'lr': 0.025,
        'momentum': 0.5,
        'output_size': 10,
        'smoke_test_size': -1
        }

    train_model(loader_type='memory', training_parameters=training_parameters)