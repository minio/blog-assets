import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error


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


def get_bucket_list() -> List[str]:
    logger = create_logger()

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(MINIO_URL,  # host.docker.internal
                    MINIO_ACCESS_KEY,  
                    MINIO_SECRET_KEY, 
                    secure=MINIO_SECURE)

        buckets = client.list_buckets()

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logger.error(f'Error occurred: {err}.')
        raise err

    return buckets
