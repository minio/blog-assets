import json
import logging
import os
import sys
from time import time
from typing import List, Dict, Any, Tuple

import mlflow
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow import MlflowClient
import numpy as np
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch import optim
from torchvision import datasets, transforms

import data_utilities as du


def initialize_ray():
    runtime_env = {
        "pip": ["minio"]
    }
    ray.init(runtime_env=runtime_env)


def get_minio_run_config():
    import s3fs
    import pyarrow.fs

    s3_fs = s3fs.S3FileSystem(
        key='miniokey...',
        secret='asecretkey...',
        endpoint_url='https://...'
    )
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

    run_config = train.RunConfig(storage_path="minio_bucket", storage_filesystem=custom_fs)    
    return run_config


def train_func_per_worker(training_parameters):

    # Get the dataset shard for the training worker.
    train_data_shard = train.get_dataset_shard('train')

    # Train the model and log training metrics.
    model = du.MNISTModel(training_parameters['input_size'], training_parameters['hidden_sizes'], training_parameters['output_size'])
    model = ray.train.torch.prepare_model(model)

    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=training_parameters['lr'], momentum=training_parameters['momentum'])

    for epoch in range(training_parameters['epochs']):
        total_loss = 0
        batch_count = 0
        for batch in train_data_shard.iter_torch_batches(batch_size=training_parameters['batch_size_per_worker']):
            # Get the images and labels from the batch.
            images, labels = batch['X'], batch['y']
            
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
            batch_count +=1

        ray.train.report({'training_loss': total_loss/batch_count})


def distributed_training(training_parameters, num_workers: int, use_gpu: bool):
    logger = du.create_logger()

    logger.info('Initializing Ray.')
    initialize_ray()

    logger.info('Getting training and testing data.')
    start_time = time()

    X_train, y_train, X_test, y_test, load_time_sec = du.get_train_test_data(
        smoke_test_size=training_parameters['smoke_test_size'])

    train_data = ray.data.from_items([{'X': X_train[i], 'y': y_train[i]} for i in range(len(X_train))])
    test_data = ray.data.from_items([{'X': X_test[i], 'y': y_test[i]} for i in range(len(X_test))])
    train_data = train_data.map_batches(du.preprocess_batch, fn_kwargs={'bucket_name':'mnist'}, batch_size=training_parameters['batch_size_per_worker'])
    test_data = test_data.map_batches(du.preprocess_batch, fn_kwargs={'bucket_name':'mnist'}, batch_size=training_parameters['batch_size_per_worker'])
    load_time_sec = (time()-start_time)


    # Initialize a Ray TorchTrainer
    logger.info('Initializing Ray TorchTrainer.')
    start_time = time()

    # Scaling configuration
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=training_parameters,
        datasets={'train': train_data},
        scaling_config=scaling_config,
        run_config=train.RunConfig(
            storage_path=os.getcwd(),
            name="ray_experiments")
    )
    result = trainer.fit()
    training_time_sec = (time()-start_time)

    logger.info(result)
    logger.info(f'Load Time (in seconds) = {load_time_sec}')
    logger.info(f'Training Time (in seconds) = {training_time_sec}')


if __name__ == "__main__":

    with open('credentials.json') as f:
        credentials = json.load(f)

    os.environ['MINIO_URL'] = credentials['url']
    os.environ['MINIO_ACCESS_KEY'] = credentials['accessKey']
    os.environ['MINIO_SECRET_ACCESS_KEY'] = credentials['secretKey']

    num_workers = 4
    use_gpu = False

    # training configuration
    training_parameters = {
        'batch_size_per_worker': 100,
        'epochs': 5,
        'input_size': 784,
        'hidden_sizes': [1024, 1024, 1024, 1024],
        'lr': 0.025,
        'momentum': 0.5,
        'output_size': 10,
        'smoke_test_size': 0
        }

    distributed_training(training_parameters, num_workers, use_gpu)

    ray.shutdown()
