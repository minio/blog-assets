import io
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from time import time

from minio import Minio
from minio.error import S3Error
import numpy as np
import PIL
import ray
import torch
from torchvision import transforms

device = torch.device('cpu')  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RAY_FILE_PATH = 'ray_logs.log'
RAY_LOGGER_NAME = 'ray'
RAY_LOGGING_LEVEL = logging.INFO


def create_logger() -> None:
    logger = logging.getLogger(RAY_LOGGER_NAME)

    #if not logger.hasHandlers(): 
    logger.setLevel(RAY_LOGGING_LEVEL)
    formatter = logging.Formatter('%(process)s %(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(RAY_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def bytes_to_image(image_bytes) -> PIL.Image.Image:
    '''
    This function will transform a image byte stream into a PIL image.
    '''
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image


def get_train_test_data(smoke_test_size: float=0) -> Tuple[Any]:
    logger = create_logger()
    logger.info(f'get_train_test_data called. smoke_test_size: {smoke_test_size}')

    # Start of load time.
    start_time = time()

    # Get a list of objects and split them according to train and test.    
    object_list = get_object_list('mnist')
    X_train, y_train, X_test, y_test = split_train_test(object_list)

    if smoke_test_size > 0:
        train_size = int(smoke_test_size*len(X_train))
        test_size = int(smoke_test_size*len(X_test))
        X_train = X_train[0:train_size]
        y_train = y_train[0:train_size]
        X_test = X_test[0:test_size]
        y_test = y_test[0:test_size]

    return  X_train, y_train, X_test, y_test, (time()-start_time)    


def split_train_test(objects: List[str]) -> Tuple[List[str], List[int], List[str], List[int]]:
    '''
    This function will parse the results from get_object_list and create a training set 
    and a test set.
    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for obj in objects:
        if obj[:5] == 'train':
            X_train.append(obj)
            label = int(obj[6])
            y_train.append(label)
        if obj[:4] == 'test':
            X_test.append(obj)
            label = int(obj[5])
            y_test.append(label)
    return X_train, y_train, X_test, y_test


def preprocess_batch(batch: Dict[str, str], bucket_name: str) -> Dict[str, np.ndarray]:
    logger = create_logger()
    logger.info(f'preprocess_batch called. bucket_name: {bucket_name}, batch_size: {len(batch["X"])}')

    url = os.environ['MINIO_URL']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_ACCESS_KEY']

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=False)

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Look through all the object names in the batch.
        for i in range(len(batch['X'])):
            response = client.get_object(bucket_name, batch['X'][i])
            image = PIL.Image.open(io.BytesIO(response.data))  #bytes_to_image(response.data)
            batch['X'][i] = transform(image)

        logger.info(f'Batch retrieval successful for bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    finally:
        response.close()
        response.release_conn()

    return batch


def get_image_from_minio(bucket_name: str, object_name: str) -> PIL.Image.Image:
    logger = create_logger()

    url = os.environ['MINIO_URL']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_ACCESS_KEY']

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=False)

        response = client.get_object(bucket_name, object_name)
        image = bytes_to_image(response.data)
        
        logger.info(f'Object: {object_name} has been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    finally:
        response.close()
        response.release_conn()
    return image


def get_object_list(bucket_name: str) -> List[str]:
    '''
    Gets a list of objects from a bucket.
    '''
    logger = create_logger()

    url = os.environ['MINIO_URL']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_ACCESS_KEY']

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=False)

        object_list = []
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            object_list.append(obj.object_name)
    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return object_list


class ProcessBatch:

    def __init__(self, bucket_name: str):
        self.logger = create_logger()
        self.logger.info(f'ProcessBatch object created. bucket_name: {bucket_name}.')
        self.bucket_name = bucket_name
        
        url = os.environ['MINIO_URL']
        access_key = os.environ['MINIO_ACCESS_KEY']
        secret_key = os.environ['MINIO_SECRET_ACCESS_KEY']

        # Create client with access and secret key
        self.client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=False)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Look through all the object names in the batch.
        for i in range(len(batch['X'])):
            response = self.client.get_object(self.bucket_name, batch['X'][i])
            image = PIL.Image.open(io.BytesIO(response.data))  #bytes_to_image(response.data)
            batch['X'][i] = transform(image)

        self.logger.info(f'Batch retrieval successful for bucket: {self.bucket_name} in MinIO object storage.')

        return batch


def get_ray_dataset(training_parameters: Dict[str, Any]) -> ray.data.dataset.Dataset:
    logger = create_logger()

    logger.info('Getting training and testing data.')
    start_time = time()

    X_train, y_train, X_test, y_test, load_time_sec = get_train_test_data(training_parameters['smoke_test_size'])

    train_data = ray.data.from_items([{'X': X_train[i], 'y': y_train[i]} for i in range(len(X_train))]).random_shuffle()
    test_data = ray.data.from_items([{'X': X_test[i], 'y': y_test[i]} for i in range(len(X_test))]).random_shuffle()
    train_data = train_data.map_batches(preprocess_batch, fn_kwargs={'bucket_name':'mnist'}, batch_size=training_parameters['batch_size_per_worker'])
    test_data = test_data.map_batches(preprocess_batch, fn_kwargs={'bucket_name':'mnist'}, batch_size=training_parameters['batch_size_per_worker'])

    load_time_sec = (time()-start_time)
    return train_data, test_data, load_time_sec
    