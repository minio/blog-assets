import io
import json
import logging
from typing import Dict, List

from minio import Minio
from minio.datatypes import Object
from minio.error import S3Error
from minio.helpers import ObjectWriteResult
import pandas as pd


def get_object(bucket_name: str, object_name: str, file_path: str) -> Object:
    '''
    This function will download an object from MinIO to the specified file_path
    and return the object_info.
    '''

    # Load the credentials and connection information.
    with open('credentials.json') as f:
        credentials = json.load(f)

    # Create client with access and secret key
    client = Minio(credentials['url'],  # host.docker.internal
                credentials['accessKey'],  
                credentials['secretKey'], 
                secure=False)
    
    # Get data of an object.
    object_info = client.fget_object(bucket_name, object_name, file_path)

    return object_info


def get_object_list(bucket_name: str, prefix: str=None) -> List[str]:
    '''
    Gets a list of objects from a bucket.
    '''
    logger = logging.getLogger('gsod_logger')
    logger.setLevel(logging.INFO)

    # Load the credentials and connection information.
    with open('credentials.json') as f:
        credentials = json.load(f)

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(credentials['url'],  # host.docker.internal
                    credentials['accessKey'],  
                    credentials['secretKey'], 
                    secure=False)

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


def put_file(bucket_name: str, object_name: str, file_path: str) -> ObjectWriteResult:
    '''
    This function will upload a file to MinIO and return the object_info.
    '''

    # Load the credentials and connection information.
    with open('credentials.json') as f:
        credentials = json.load(f)

    # Create client with access and secret key
    client = Minio(credentials['url'],  # host.docker.internal
                credentials['accessKey'],  
                credentials['secretKey'], 
                secure=False)
    
    # Make sure bucket exists.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)

    # Upload the file.
    object_write_result = client.fput_object(bucket_name, object_name, file_path)

    return object_write_result


def put_text(bucket_name: str, object_name: str, text: str, metadata: Dict=None) -> ObjectWriteResult:
    '''
    This function will upload a file to MinIO and return the object_info.
    '''

    # Load the credentials and connection information.
    with open('credentials.json') as f:
        credentials = json.load(f)

    # Create client with access and secret key
    client = Minio(credentials['url'],  # host.docker.internal
                credentials['accessKey'],  
                credentials['secretKey'], 
                secure=False)
    
    # Make sure bucket exists.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)

    # Upload the text.
    # Upload the dataframe as an object.
    #encoded_df = df.to_csv(index=False).encode('utf-8')
    #client.put_object(bucket, object_name, data=io.BytesIO(encoded_df), length=len(encoded_df), content_type='application/csv')
    text_as_bytes = str.encode(text)
    object_write_result = client.put_object(bucket_name, object_name, io.BytesIO(text_as_bytes), len(text_as_bytes), metadata=metadata)
    return object_write_result


def get_text(bucket_name: str, object_name: str) -> str:
    '''
    This function will download an object from MinIO as text.
    '''

    # Load the credentials and connection information.
    with open('credentials.json') as f:
        credentials = json.load(f)

    # Create client with access and secret key
    client = Minio(credentials['url'],  # host.docker.internal
                credentials['accessKey'],  
                credentials['secretKey'], 
                secure=False)
    
    # Get data of an object.
    response = client.get_object(bucket_name, object_name)
    text = response.data.decode()
    return text
