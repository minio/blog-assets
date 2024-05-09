from typing import Dict

import mlrun
import minio_utilities as mu

@mlrun.handler()
def train_model(data_bucket: str=None, training_parameters: Dict=None):
    logger = mu.create_logger()
    logger.info(data_bucket)
    logger.info(training_parameters)
    bucket_list = mu.get_bucket_list()
    logger.info(bucket_list)