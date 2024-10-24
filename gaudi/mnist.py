import argparse
import os
import time
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import habana_frameworks.torch.core as htcore
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import data_utilities as du
import torch_utilities as tu

# Load the credentials and connection information.
load_dotenv('minio.env')
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
MINIO_SECRET_KEY = os.environ['MINIO_SECRET_KEY']
if os.environ['MINIO_SECURE']=='true': MINIO_SECURE = True 
else: MINIO_SECURE = False 
BUCKET_NAME = os.environ['MNIST_BUCKET_NAME']


def train_model(model: nn.Module, loader: DataLoader, training_parameters: Dict[str, Any]) -> List[float]:
    logger = du.create_logger()

    device = torch.device('cpu')
    if training_parameters['use_gpu']:
        if torch.hpu.is_available(): device = torch.device('hpu')
        if torch.cuda.is_available(): device = torch.device('cuda')
    logger.info(f'Using device: {device}')

    model.to(device)
    logger.info(f'Model moved to device: {device}')

    # Define a transform to normalize the images and turn them into tensors
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=training_parameters['lr'], momentum=training_parameters['momentum'])

    # Epoch loop
    compute_time_by_epoch = []
    for epoch in range(training_parameters['epochs']):
        total_loss = 0
        batch_count = 0
        epoch_compute_time = 0
        epoch_start = time.perf_counter()
        # Batch loop
        for images, labels in loader:
            if isinstance(images, tuple):
                image_list = du.get_images_from_minio(training_parameters['bucket_name'], images)
                images = torch.stack([transform(x) for x in image_list], dim=0)

            # Move to the specified device.
            # shape = [32, 1, 28, 28]
            images, labels = images.to(device), labels.to(device)

            # Start of compute time for the batch.
            compute_start = time.perf_counter()

            # Flatten MNIST images into a 784 long vector.
            # shape = [32, 784]
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Loss calculations            
            total_loss += loss.item()
            batch_count +=1

            # Track compute time
            batch_compute_time = time.perf_counter() - compute_start
            epoch_compute_time += batch_compute_time
        
        compute_time_by_epoch.append(epoch_compute_time)
        epoch_io_time = (time.perf_counter() - epoch_start) - epoch_compute_time
        logger.info(f'Epoch {epoch+1} - Training loss: {total_loss/batch_count} - Compute time: {epoch_compute_time} IO time: {epoch_io_time}.')

    return compute_time_by_epoch


def setup_local_training(training_parameters: Dict[str, Any], loader_type: str):
    logger = du.create_logger()
    logger.info(f'PyTorch Version: {torch.__version__}')
    experiment_start_time = time.perf_counter()

    #train_data, test_data, load_time_sec = ru.get_ray_dataset(training_parameters)
    train_loader, load_time_sec = tu.create_mnist_training_loader(training_parameters['bucket_name'], loader_type, training_parameters['batch_size'], 
                                                                  num_workers=training_parameters['num_workers'], 
                                                                  smoke_test_size=training_parameters['smoke_test_size'])
    logger.info(f'Data Loader Creation Time (in seconds) = {load_time_sec}')
    logger.info(f'Data Set Size: {len(train_loader.dataset)} samples.')

    # Create the model.
    model = tu.MNISTModel(training_parameters['input_size'], training_parameters['hidden_sizes'], training_parameters['output_size'])
    
    training_start_time = time.perf_counter()
    compute_time_by_epoch = train_model(model, train_loader, training_parameters)
    training_time_sec = time.perf_counter() - training_start_time
    experiment_time_sec = time.perf_counter() - experiment_start_time

    compute_time_sec = 0
    for epoch_time in compute_time_by_epoch:
        compute_time_sec += epoch_time

    logger.info(f'Compute Time (in seconds) = {compute_time_sec}')
    logger.info(f'I/O Time (in seconds) = {training_time_sec - compute_time_sec}')
    logger.info(f'Total Training Time (in seconds) = {training_time_sec}')
    logger.info(f'Total Experiment Time (in seconds) = {experiment_time_sec}')

    #test_loader, load_time_sec = tu.create_mnist_testing_loader(training_parameters['bucket_name'], loader_type, training_parameters['batch_size'])
    #tu.test_model_local(model, test_loader, training_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Command line interface.')
    parser.add_argument('-lb', '--list_buckets', help='List all buckets.', action='store_true')
    parser.add_argument('-lo', '--list_objects', help='List all objects in the specified bucket.')
    parser.add_argument('-eb', '--empty_bucket', help='Remove all objects in the specified bucket.')
    parser.add_argument('-load', '--load_bucket', help='Load the MNIST dataset into the specified bucket.')
    parser.add_argument('-train', '--train', help='Train model.', action='store_true')
    parser.add_argument('-lt', '--loader_type', help='Type of loader to use for loading training and test sets (batch, list, file, full or s3map).')
    args = parser.parse_args()

    if args.list_buckets:
        bucket_list = du.get_bucket_list()
        print(bucket_list)
    if args.list_objects:
        object_list = du.get_object_list(args.list_objects)
        print(f'Number of objects in {args.list_objects}:', len(object_list))
    if args.empty_bucket:
        count = du.empty_bucket(args.empty_bucket)
        print(f'Number of objects removed in {args.empty_bucket}:', count)
    if args.load_bucket:
        train_count, test_count = du.load_mnist_to_minio(args.load_bucket)
        print(f'MNIST training images added to {args.load_bucket}:', train_count)
        print(f'MNIST testing images added to {args.load_bucket}:', test_count)
    if args.train:
        # Hyperparameters
        training_parameters = {
            'batch_size': 32,
            'bucket_name': BUCKET_NAME,
            'epochs': 3,
            'hidden_sizes': [1024, 1024, 1024, 1024],
            'input_size': 784,
            'lr': 0.025,
            'momentum': 0.5,
            'num_workers': 4,
            'output_size': 10,
            'smoke_test_size': 0,
            'use_gpu': False,
            }
        setup_local_training(training_parameters, args.loader_type)