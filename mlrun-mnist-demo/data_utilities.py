import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from time import time
import uuid

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
import numpy as np
import PIL
import torch
from torchvision import datasets, transforms

device = torch.device('cpu')  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOGGER_NAME = 'train'
LOGGING_LEVEL = logging.INFO

load_dotenv('minio.env')
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
MINIO_SECRET_KEY = os.environ['MINIO_SECRET_KEY']
if os.environ['MINIO_SECURE']=='true': MINIO_SECURE = True 
else: MINIO_SECURE = False 


def create_logger() -> None:
    logger = logging.getLogger(LOGGER_NAME)

    #if not logger.hasHandlers(): 
    logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter('%(process)s %(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(stdout_handler)

    return logger


def bytes_to_image(image_bytes) -> PIL.Image.Image:
    '''
    This function will transform a image byte stream into a PIL image.
    '''
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image


def get_train_test_data(bucket_name: str, smoke_test_size: float=0) -> Tuple[Any]:
    logger = create_logger()
    logger.info(f'get_train_test_data called. smoke_test_size: {smoke_test_size}')

    # Get a list of objects and split them according to train and test.    
    object_list = get_object_list(bucket_name)
    X_train, y_train, X_test, y_test = split_train_test(object_list)

    if smoke_test_size > 0:
        train_size = int(smoke_test_size*len(X_train))
        test_size = int(smoke_test_size*len(X_test))
        X_train = X_train[0:train_size]
        y_train = y_train[0:train_size]
        X_test = X_test[0:test_size]
        y_test = y_test[0:test_size]

    return  X_train, y_train, X_test, y_test


def image_to_byte_stream(image: PIL.Image.Image) -> io.BytesIO:
    '''
    This function will transform a PIL image to a byte stream so that it can be sent 
    to MinIO.
    '''
    # BytesIO is a file-like buffer stored in memory
    image_byte_array = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(image_byte_array, format='jpeg')
    # Turn the BytesIO object back into a bytes object
    #img_byte_array = img_byte_array.getvalue()
    #print(len(img_byte_array))
    #print(type(img_byte_array))
    image_byte_array.seek(0)
    return image_byte_array


def get_bucket_list() -> List[str]:
    '''
    Returns a list of buckets in MinIO.
    '''
    logger = logging.getLogger('mnist_logger')
    logger.setLevel(logging.INFO)

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

        buckets = client.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return buckets


def load_mnist_to_minio(bucket_name: str) -> Tuple[int,int]:
    ''' Download and load the training and test samples.'''

    train = datasets.MNIST('./mnistdata/', download=True, train=True)
    test = datasets.MNIST('./mnistdata/', download=True, train=False)

    train_count = 0
    for sample in train:
        random_uuid = uuid.uuid4()
        object_name = f'/train/{sample[1]}/{random_uuid}.jpeg'
        put_image_to_minio(bucket_name, object_name, sample[0])
        train_count += 1

    test_count = 0
    for sample in test:
        random_uuid = uuid.uuid4()
        object_name = f'/test/{sample[1]}/{random_uuid}.jpeg'
        put_image_to_minio(bucket_name, object_name, sample[0])
        test_count += 1

    return train_count, test_count


def put_image_to_minio(bucket_name: str, object_name: str, image: PIL.Image.Image) -> None:
    '''
    Puts an image byte stream to MinIO.
    '''
    logger = logging.getLogger('mnist_logger')
    logger.setLevel(logging.INFO)

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

        #image_byte_array = io.BytesIO()
        #image.save(image_byte_array, format='JPEG')
        
        # This worked but gave me a format that was not recognized when downloaded.
        #image_byte_array = io.BytesIO(image.tobytes())
        #print(type(image_byte_array))
        image_byte_array = image_to_byte_stream(image)
        content_type = 'application/octet-stream' # 'application/binary'
        response = client.put_object(bucket_name, object_name, image_byte_array, -1, content_type, part_size = 1024*1024*5)
        #response = client.fput_object(bucket_name, object_name, file_path, content_type)

        logger.info(f'Object: {response.object_name} has been loaded into bucket: {bucket_name} in MinIO object storage.')
        logger.info(f'Etag: {response.etag}')
        logger.info(f'Version ID: {response.version_id}')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err
    

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

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

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

    return batch


def get_image_from_minio(bucket_name: str, object_name: str, client: Minio=None) -> PIL.Image.Image:
    logger = create_logger()

    # Get data of an object.
    try:
        if client is None:
            # Create client with access and secret key
            client = Minio(MINIO_URL,  # host.docker.internal
                        MINIO_ACCESS_KEY,  
                        MINIO_SECRET_KEY, 
                        secure=MINIO_SECURE)

        response = client.get_object(bucket_name, object_name)
        image = bytes_to_image(response.data)
        
        logger.debug(f'Object: {object_name} has been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return image


def get_object_from_minio(bucket_name: str, object_name: str, client: Minio=None) -> PIL.Image.Image:
    logger = create_logger()

    # Get data of an object.
    try:
        if client is None:
            # Create client with access and secret key
            client = Minio(MINIO_URL,  # host.docker.internal
                        MINIO_ACCESS_KEY,  
                        MINIO_SECRET_KEY, 
                        secure=MINIO_SECURE)

        response = client.get_object(bucket_name, object_name)
        
        logger.debug(f'Object: {object_name} has been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return response.data


def get_object_list(bucket_name: str) -> List[str]:
    '''
    Gets a list of objects from a bucket.
    '''
    logger = create_logger()

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

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
