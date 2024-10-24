from io import BytesIO
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from time import time
import uuid

import PIL.Image
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
import numpy as np
import PIL
from torchvision import datasets, transforms


LOG_FILE_PATH = 'training_logs.log'
LOGGER_NAME = 'training'
LOGGING_LEVEL = logging.INFO

#load_dotenv()
#MINIO_URL = os.environ['MINIO_URL']
#MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
#MINIO_SECRET_KEY = os.environ['MINIO_SECRET_KEY']
#if os.environ['MINIO_SECURE']=='true': MINIO_SECURE = True 
#else: MINIO_SECURE = False 


def create_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    #if not logger.hasHandlers(): 
    logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter('%(process)s %(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    #file_handler = logging.FileHandler(RAY_FILE_PATH)
    #file_handler.setLevel(logging.DEBUG)
    #file_handler.setFormatter(formatter)

    logger.handlers = []
    #logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def empty_bucket(bucket_name: str) -> int:
    '''
    Removes all objects from the specified bucket.
    '''
    logger = create_logger()
    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)
        
        count = 0
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            client.remove_object(bucket_name, obj.object_name)
            count += 1
            if count % 100 == 0:
                logger.info(f'{count} objects deleted from {bucket_name}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return count


def create_object_inventory(bucket_name: str, verbose: bool=False) -> str:
    '''
    Creates an object of JSON format that contains an inventory of all objects 
    in the specified bucket.
    '''
    logger = create_logger()
    inventory_object_name = bucket_name + '-inventory.json'

    try:
        # Create client with access and secret key
        url, access_key, secret_key, secure = get_minio_credentials()
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)
        
        logger.info(f'Calling list_objects for bucket: {bucket_name}.')
        object_list = []
        objects = client.list_objects(bucket_name, recursive=True)

        logger.info(f'Generating object inventory for bucket: {bucket_name}.')
        for obj in objects:
            if obj.object_name != inventory_object_name:
                if verbose:
                    object_data = {'object_name': obj.object_name,
                                   'content_type': obj.content_type,
                                   'etag': obj.etag,
                                   'is_dir': obj.is_dir,
                                   'last_modified': obj.last_modified,
                                   'metadata': obj.metadata,
                                   'owner_id': obj.owner_id,
                                   'owner_name': obj.owner_name,
                                   'size': obj.size,
                                   'storage_class': obj.storage_class,
                                   'tags': obj.tags,
                                   'version_id': obj.version_id,
                    }
                else:
                    object_data = {'object_name': obj.object_name}

                object_list.append(object_data)

        logger.info(f'Saving object inventory for bucket: {bucket_name}.')
        # Setting the default parameter to the str function allows the last_modified date to be turned into
        # a string. In it original form it is not JSON serializable.
        json_string = json.dumps(object_list, default=str)
        encoded_json = json_string.encode('utf-8')
        response = client.put_object(bucket_name, inventory_object_name, data=BytesIO(encoded_json), 
                                    length=len(encoded_json), content_type='application/json')
        logger.info(f'Inventory object successfully created and saved for bucket: {bucket_name}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return inventory_object_name, object_list


def get_object_inventory(bucket_name: str) -> List[Any]:
    '''
    Get the object inventory object for the specified bucket.
    '''

    # Get data of an object.
    try:
        inventory_object_name = bucket_name + '-inventory.json'
        url, access_key, secret_key, secure = get_minio_credentials()
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)
        response = client.get_object(bucket_name, inventory_object_name)
        
        # Both of the techniques below work if object list is a bytes object.
        object_list = json.loads(response.data.decode('utf-8'))

    finally:
        response.close()
        response.release_conn()
    return object_list


def get_mnist_lists(bucket_name: str, split: str='train', smoke_test_size: int=0) -> Tuple[Any]:
    logger = create_logger()
    logger.debug(f'get_mnist_lists called. bucket_name: {bucket_name} smoke_test_size: {smoke_test_size}')

    # Get a list of objects and split them according to train and test.    
    object_list = get_object_list(bucket_name, split)

    X = []
    y = []
    for path in object_list:
        X.append(path)
        label = int(path.split('/')[1])  #int(obj[6])
        y.append(label)

    if smoke_test_size > 0:
        X = X[0:smoke_test_size]
        y = y[0:smoke_test_size]

    #X_train, y_train, X_test, y_test = split_train_test(object_list)

    #if smoke_test_size > 0:
    #    train_size = int(smoke_test_size*len(X_train))
    #    test_size = int(smoke_test_size*len(X_test))
    #    X_train = X_train[0:train_size]
    #    y_train = y_train[0:train_size]
    #    X_test = X_test[0:test_size]
    #    y_test = y_test[0:test_size]

    return X, y


def image_to_byte_stream(image: PIL.Image.Image) -> BytesIO:
    '''
    This function will transform a PIL image to a byte stream so that it can be sent 
    to MinIO.
    '''
    # BytesIO is a file-like buffer stored in memory
    image_byte_array = BytesIO()
    # image.save expects a file-like as a argument
    image.save(image_byte_array, format='jpeg')
    # Turn the BytesIO object back into a bytes object
    #img_byte_array = img_byte_array.getvalue()
    #print(len(img_byte_array))
    #print(type(img_byte_array))
    image_byte_array.seek(0)
    return image_byte_array


def get_minio_credentials() -> Tuple[str]:
    url = os.environ['MINIO_URL']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_KEY']
    if os.environ['MINIO_SECURE']=='true': secure = True 
    else: secure = False 
    return (url, access_key, secret_key, secure)


def get_bucket_list() -> List[str]:
    logger = create_logger()
    
    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

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
    logger = create_logger()

    train = datasets.MNIST('./mnistdata/', download=True, train=True)
    test = datasets.MNIST('./mnistdata/', download=True, train=False)

    # Create client with access and secret key
    url, access_key, secret_key, secure = get_minio_credentials()
    client = Minio(url,  # host.docker.internal
                access_key,  
                secret_key, 
                secure=secure)
    # Make the bucket if it does not exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        print(f'Bucket {bucket_name} already exists.')

    train_count = 0
    for sample in train:
        random_uuid = uuid.uuid4()
        object_name = f'/train/{sample[1]}/{random_uuid}.jpeg'
        put_image_to_minio(bucket_name, object_name, sample[0], client)
        train_count += 1
        if train_count % 100 == 0:
            logger.info(f'{train_count} training objects added to {bucket_name}.')

    test_count = 0
    for sample in test:
        random_uuid = uuid.uuid4()
        object_name = f'/test/{sample[1]}/{random_uuid}.jpeg'
        put_image_to_minio(bucket_name, object_name, sample[0], client)
        test_count += 1
        if test_count % 100 == 0:
            logger.info(f'{test_count} testing objects added to {bucket_name}.')

    return train_count, test_count


def put_image_to_minio(bucket_name: str, object_name: str, image: PIL.Image.Image, client: Minio=None) -> None:
    '''
    Puts an image byte stream to MinIO.
    '''
    logger = create_logger()


    try:
        if client == None:
            # Create client with access and secret key
            url, access_key, secret_key, secure = get_minio_credentials()
            client = Minio(url,  # host.docker.internal
                        access_key,  
                        secret_key, 
                        secure=secure)

        #image_byte_array = BytesIO()
        #image.save(image_byte_array, format='JPEG')
        
        # This worked but gave me a format that was not recognized when downloaded.
        #image_byte_array = BytesIO(image.tobytes())
        #print(type(image_byte_array))
        image_byte_array = image_to_byte_stream(image)
        content_type = 'application/octet-stream' # 'application/binary'
        response = client.put_object(bucket_name, object_name, image_byte_array, -1, content_type, part_size = 1024*1024*5)

        logger.debug(f'Object: {response.object_name} has been loaded into bucket: {bucket_name} in MinIO object storage.')
        logger.debug(f'Etag: {response.etag}')
        logger.debug(f'Version ID: {response.version_id}')

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
            label = int(obj.split('/')[1])  #int(obj[6])
            y_train.append(label)
        if obj[:4] == 'test':
            X_test.append(obj)
            label = int(obj.split('/')[1])  #int(obj[5])
            y_test.append(label)
    return X_train, y_train, X_test, y_test


def preprocess_batch(batch: Dict[str, str], bucket_name: str) -> Dict[str, np.ndarray]:
    logger = create_logger()
    logger.debug(f'preprocess_batch called. bucket_name: {bucket_name}, batch_size: {len(batch["X"])}')

    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Look through all the object names in the batch.
        for i in range(len(batch['X'])):
            response = client.get_object(bucket_name, batch['X'][i])
            image = PIL.Image.open(BytesIO(response.data))
            batch['X'][i] = transform(image)

        logger.info(f'Batch retrieval successful for bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return batch


def get_images_from_minio(bucket_name: str, object_names: Tuple[str], client: Minio=None) -> List[PIL.Image.Image]:
    logger = create_logger()

    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        images = []
        for object_name in object_names:
            response = client.get_object(bucket_name, object_name)
            # Convert the bytes object to a Pil.Image.Image object.
            image = PIL.Image.open(BytesIO(response.data))
            images.append(image)

        response.close()
        response.release_conn()
        logger.debug(f'Objects: {object_names} have been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return images


def get_image_from_minio(bucket_name: str, object_name: str, client: Minio=None) -> PIL.Image.Image:
    logger = create_logger()

    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        response = client.get_object(bucket_name, object_name)
        # Convert the bytes object to a Pil.Image.Image object.
        image = PIL.Image.open(BytesIO(response.data))
        response.close()
        response.release_conn()
        
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

    url, access_key, secret_key, secure = get_minio_credentials()

    # Get data of an object.
    try:
        if client is None:
            # Create client with access and secret key
            client = Minio(url,  # host.docker.internal
                        access_key,  
                        secret_key, 
                        secure=secure)

        response = client.get_object(bucket_name, object_name)
        
        logger.debug(f'Object: {object_name} has been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return response.data


def get_object_list(bucket_name: str, prefix: str=None) -> List[str]:
    '''
    Gets a list of objects from a bucket.
    '''
    logger = create_logger()

    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        object_list = []
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        for obj in objects:
            object_list.append(obj.object_name)
    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return object_list
