'''
Use the command below to create a Python Containerized Component:

    kfp component build src/ --component-filepattern model_training.py --no-push-image

'''
import logging
from typing import List

from kfp import dsl
from kfp.dsl import Dataset
from kfp.dsl import Model
from kfp.dsl import Markdown
from kfp.dsl import Input
from kfp.dsl import Output
from kfp.client import Client
import pandas as pd
import torch

import data_utilities as du
import model_utilities as mu

kfp_endpoint = 'http://localhost:8080'


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['pandas==1.3.5', 'minio==7.1.14'])
def get_raw_data(bucket: str, object_name: str, table_df: Output[Dataset]):
    '''
    Return an object as a Pandas DataFrame.
    '''
    df = du.get_object(bucket, object_name)
    df.to_csv(table_df.path, index=False)


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['numpy==1.24.3', 'pandas==1.3.5', 'scikit-learn==1.0.2'])
def preprocess(in_df: Input[Dataset], out_df: Output[Dataset]) -> None:
    '''
    Preprocess the dataframe.
    '''
    df = pd.read_csv(in_df.path)
    df = du.preprocess(df)
    df.to_csv(out_df.path, index=False)


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['numpy==1.24.3', 'pandas==1.3.5', 'scikit-learn==1.0.2'])
def feature_engineering(pre: Input[Dataset], 
                        train_X: Output[Dataset], train_y: Output[Dataset], 
                        valid_X: Output[Dataset], valid_y: Output[Dataset], 
                        test_X: Output[Dataset], test_y: Output[Dataset],
          validation_size: int=1, test_size: int=1) -> None:
    '''
    Feature engineering.
    '''
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    df = pd.read_csv(pre.path)
    df_train_X, df_train_y, df_valid_X, df_valid_y, df_test_X, df_test_y = du.feature_engineering(df, validation_size, test_size)
    df_train_X.to_csv(train_X.path, index=False)
    df_train_y.to_csv(train_y.path, index=False)
    df_valid_X.to_csv(valid_X.path, index=False)
    df_valid_y.to_csv(valid_y.path, index=False)
    df_test_X.to_csv(test_X.path, index=False)
    df_test_y.to_csv(test_y.path, index=False)
    logger.info('Feature engineering complete.')


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['numpy==1.24.3', 'pandas==1.3.5', 'torch==1.13.1'])
def train_model(train_X: Input[Dataset], train_y: Input[Dataset],
                valid_X: Input[Dataset], valid_y: Input[Dataset], 
                model: Output[Model], training_results: Output[Markdown]) -> None:
    df_train_X = pd.read_csv(train_X.path)
    df_train_y = pd.read_csv(train_y.path)
    df_valid_X = pd.read_csv(valid_X.path)
    df_valid_y = pd.read_csv(valid_y.path)
    mbd_model, results = mu.train_model(df_train_X, df_train_y, df_valid_X, df_valid_y)
    #embedding_sizes = [(3135, 50), (51, 26)]
    #mbd_model = mu.MBDModel(embedding_sizes, 1)
    torch.save(mbd_model.state_dict(), model.path)
    with open(training_results.path, 'w') as f:
        for result in results:
            epoch_result = f'Training Loss: {result[0]}, Validation Loss: {result[1]}, Validation accuracy: {result[2]}.<br>'
            f.write(epoch_result)


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['numpy==1.24.3', 'pandas==1.3.5', 'torch==1.13.1'])
def test_model(test_X: Input[Dataset], test_y: Input[Dataset], model: Input[Model], test_results: Output[Markdown]) -> None:
    df_test_X = pd.read_csv(test_X.path)
    df_test_y = pd.read_csv(test_y.path)
    #mbd_model = mu.MBDModel([(3135, 50), (51, 26)], 1)
    mbd_model = mu.create_model()
    mbd_model.load_state_dict(torch.load(model.path))
    smape = mu.test_model(df_test_X, df_test_y, mbd_model)
    with open(test_results.path, 'w') as f:
        f.write('Symetric Mean Absolute Percent Error (SMAPE): **' + str(smape) + '**')


@dsl.component(base_image='python:3.10.0',
               target_image='docker/model-training-project/model-training:v1', 
               packages_to_install=['minio==7.1.14', 'numpy==1.24.3', 'pandas==1.3.5', 'scikit-learn==1.0.2', 'torch==1.13.1'])
def monolith(bucket: str, object_name: str, test_results: Output[Markdown]) -> None:
    df = du.get_object(bucket, object_name)
    df = du.preprocess(df)
    df_train, df_valid, df_test = du.feature_engineering(df)
    mbd_model, results = mu.train_model(df_train, df_valid)
    smape = mu.test_model(df_test, mbd_model)
    with open(test_results.path, 'w') as f:
        f.write('Symetric Mean Absolute Percent Error (SMAPE): **' + str(smape) + '**')


@dsl.pipeline(
    name='model-training-pipeline',
    description='Pipeline that will train a Nueral Network.'
)
def training_pipeline(bucket: str, object_name: str) -> Markdown:
    raw_dataset = get_raw_data(bucket=bucket, object_name=object_name)
    processed_dataset = preprocess(in_df=raw_dataset.outputs['table_df'])
    final_datasets = feature_engineering(pre=processed_dataset.outputs['out_df'])
    training_results = train_model(train_X=final_datasets.outputs['train_X'], train_y=final_datasets.outputs['train_y'],
                                   valid_X=final_datasets.outputs['valid_X'], valid_y=final_datasets.outputs['valid_y'])
    testing_results = test_model(test_X=final_datasets.outputs['test_X'], test_y=final_datasets.outputs['test_y'], 
                                 model=training_results.outputs['model'])
    return testing_results.outputs['test_results']


def start_training_pipeline_run():
    client = Client()
    run = client.create_run_from_pipeline_func(training_pipeline, experiment_name='Containerized Python Components', enable_caching=False,
    arguments={
        'bucket': 'microbusiness-density',
        'object_name': 'train.csv'
        }
    )
    url = f'{kfp_endpoint}/#/runs/details/{run.run_id}'
    print(url)


if __name__ == '__main__':
    start_training_pipeline_run()