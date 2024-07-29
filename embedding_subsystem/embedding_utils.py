import glob
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from minio import Minio
from minio.error import S3Error
import psycopg2

LOGGER_NAME = 'embedding_subsystem'
LOGGING_LEVEL = logging.DEBUG
LOGGER_FILE_PATH = 'embedding_logs.log'

load_dotenv('minio.env')
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
MINIO_SECRET_KEY = os.environ['MINIO_SECRET_KEY']
if os.environ['MINIO_SECURE']=='true': MINIO_SECURE = True 
else: MINIO_SECURE = False 
PGVECTOR_HOST = os.environ['PGVECTOR_HOST']
PGVECTOR_DATABASE = os.environ['PGVECTOR_DATABASE']
PGVECTOR_USER = os.environ['PGVECTOR_USER']
PGVECTOR_PASSWORD = os.environ['PGVECTOR_PASSWORD']
PGVECTOR_PORT = os.environ['PGVECTOR_PORT']


def create_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    #if not logger.hasHandlers(): 
    logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter('%(process)s %(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOGGER_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def upload_model_to_minio(bucket_name: str, full_model_name: str, revision: str) -> None:
    '''
    Download a model from Hugging Face and upload it to MinIO. This function will use
    the current systems temp directory to temporarily save the model. 
    '''

    # Create a local directory for the model.
    #home = str(Path.home())
    temp_dir = tempfile.gettempdir()
    base_path = f'{temp_dir}{os.sep}hf-models'
    os.makedirs(base_path, exist_ok=True)

    # Get the user name and the model name.
    tmp = full_model_name.split('/') 
    user_name = tmp[0]
    model_name = tmp[1]

    # The snapshot_download will use this pattern for the path name.
    model_path_name=f'models--{user_name}--{model_name}' 
    # The full path on the local drive. 
    full_model_local_path = base_path + os.sep + model_path_name + os.sep + 'snapshots' + os.sep + revision 
    # The path used by MinIO. 
    full_model_object_path = model_path_name + '/snapshots/' + revision

    print(f'Starting download from HF to {full_model_local_path}.')
    snapshot_download(repo_id=full_model_name, revision=revision, cache_dir=base_path)

    print('Uploading to MinIO.')
    upload_local_directory_to_minio(full_model_local_path, bucket_name, full_model_object_path)
    #shutil.rmtree(full_model_local_path)


def upload_local_directory_to_minio(local_path:str, bucket_name:str , minio_path:str) -> None:
    assert os.path.isdir(local_path)

    # Create client with access and secret key
    client = Minio(MINIO_URL,
                MINIO_ACCESS_KEY,  
                MINIO_SECRET_KEY, 
                secure=MINIO_SECURE)

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, '/') # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(local_file, bucket_name, minio_path + '/' + os.path.basename(local_file))
        else:
            remote_path = os.path.join(minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(os.sep, '/')  # Replace \ with / on Windows
            client.fput_object(bucket_name, remote_path, local_file)


def download_model_from_minio(bucket_name: str, full_model_name: str, revision: str) -> str:
    # Create a local directory for the model.
    #home = str(Path.home())
    temp_dir = tempfile.gettempdir()
    base_path = f'{temp_dir}{os.sep}hf-models'
    os.makedirs(base_path, exist_ok=True)

    # Get the user name and the model name.
    tmp = full_model_name.split('/') 
    user_name = tmp[0]
    model_name = tmp[1]

    # The snapshot_download will use this patter for the path name.
    model_path_name=f'models--{user_name}--{model_name}' 
    # The full path on the local drive. 
    full_model_local_path = base_path + os.sep + model_path_name + os.sep + 'snapshots' + os.sep + revision 
    # The path used by MinIO. 
    full_model_object_path = model_path_name + '/snapshots/' + revision 
    print(full_model_local_path)

    # Create client with access and secret key
    client = Minio(MINIO_URL,
                MINIO_ACCESS_KEY,  
                MINIO_SECRET_KEY, 
                secure=MINIO_SECURE)

    for obj in client.list_objects(bucket_name, prefix=full_model_object_path, recursive=True):
        file_path = os.path.join(base_path, obj.object_name)
        client.fget_object(bucket_name, obj.object_name, file_path)
    return full_model_local_path


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


def get_document_from_minio(bucket_name: str, object_name: str) -> str:
    '''
    Retrieves an object from MinIO, saves it in a temp file and retiurns the 
    path to the temp file.
    '''
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

        # Generate a temp file.
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, object_name)
        # Save object to file.
        client.fget_object(bucket_name, object_name, temp_file)
        
    except S3Error as s3_err:
        raise s3_err
    except Exception as err:
        raise err

    return temp_file


def save_embeddings_to_vectordb(chunks, embeddings) -> None:
    '''
    This function will write an embeeding along with the embeddings text
    to the vector db.
    '''

    try:
        connection = psycopg2.connect(host=PGVECTOR_HOST, database=PGVECTOR_DATABASE, 
                                      user=PGVECTOR_USER, password=PGVECTOR_PASSWORD, 
                                      port=PGVECTOR_PORT)
        cursor = connection.cursor()
    
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting", error)

    try:
        for text, embedding in zip(chunks, embeddings):
            cursor.execute(
                "INSERT INTO embeddings (embedding, text) VALUES (%s, %s)",
                (embedding, text)
            )
        connection.commit()
    except (Exception, psycopg2.Error) as error:
        print("Error while writing to DB", error)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()