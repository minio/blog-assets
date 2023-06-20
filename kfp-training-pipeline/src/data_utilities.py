import io
import json
import logging
from typing import List, Tuple, Dict, Any

from minio import Minio
from minio.error import S3Error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch


def get_device():
    '''See if there is a GPU available.'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_object(bucket: str, object_name: str, host: str=None) -> pd.DataFrame:
    '''
    object_paths must be a list of path strings. Ex. bucket/train.csv.
    The first part of the object path must be the bucket name. The delimer is '/'.
    Return an object as a Pandas DataFrame.
    '''
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    logger.info(bucket)
    logger.info(object_name)
    logger.info(host)

    with open('credentials.json') as f:
        credentials = json.load(f)

    if host == None: host = credentials['url']

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio(host,  # host.docker.internal
                    credentials['accessKey'],  
                    credentials['secretKey'], 
                    secure=False)

        response = client.get_object(bucket, object_name)
        df = pd.read_csv(io.BytesIO(response.data))

        logger.info(f'Object: {object_name} has been retrieved from bucket: {bucket} in MinIO object storage.')
        logger.info(f'Object length: {len(df)}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

    finally:
        response.close()
        response.release_conn()
    return df


def get_config() -> Dict[Any, Any]:
    config = {
        'categorical_cols': ['cfips', 'state'],
        'continuous_cols': ['unix_elapsed_seconds'],  #, 'pct_bachelors_degree', 'pct_graduate_degree'
        'target': ['microbusiness_density'],
        'embedding_sizes': [(3135, 50), (51, 26)]
    }
    return config


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    for col in get_config()['categorical_cols']:
        df[col] = LabelEncoder().fit_transform(df[col])
        df[col] = df[col].astype('category')

    # Create date fields.
    df['measurement_date'] = pd.to_datetime(df['first_day_of_month'])
    df['year'] = pd.DatetimeIndex(df['measurement_date']).year

    field = df['measurement_date']
    # Get a boolean mask indicating missing values.
    mask = ~field.isna()
    # Missing values will be np.nan.
    # Otherwise calculate the number of seconds since the unix epoch start date.
    # This is done by converting th field to an int64 which gives the number of nanoseconds (billionths of a second) since 1/1/1970. 
    # Floor division by 1B will give number of seconds dropping decimal components which should not result in a loss of 
    # data since this field does not contain a time component and there are more than a billion nanoseconds in a day.
    df['unix_elapsed_seconds'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, np.nan)
    df['unix_elapsed_seconds'] = normalize(df['unix_elapsed_seconds'])

    df['microbusiness_density'] = df['microbusiness_density'].astype(np.float32)
    df['unix_elapsed_seconds'] = df['unix_elapsed_seconds'].astype(np.float32)

    logger.info('Preprocessing complete.')
    return df


def normalize(col: pd.Series) -> pd.Series:
    return (col - col.min())/(col.max()-col.min())


def create_year(df: pd.DataFrame, source_year: int, destination_year: int) -> pd.DataFrame:
    df_temp = df.loc[df['year'] == source_year].copy()
    df_temp['year'] = destination_year
    return pd.concat([df, df_temp])


def feature_engineering(df: pd.DataFrame, validation_size: int=1, test_size: int=1, host: str=None) -> Tuple[pd.DataFrame]:
    '''
    Columns Used:
        S1501_C01_001E Population 18 to 24 years
        S1501_C01_005E Bachelor's degree or higher 18-24
        S1501_C01_006E Population 25 years and over
        S1501_C01_012E Bachelor's degree 25+
        S1501_C01_013E  Graduate or Professional Degree 25+
    '''

    bucket = 'census-data'
    df2019 = get_object(bucket, 'S1501-2019.csv', host=host)
    df2019['year'] = 2019
    df2020 = get_object(bucket, 'S1501-2020.csv', host=host)
    df2020['year'] = 2020
    df2021 = get_object(bucket, 'S1501-2021.csv', host=host)
    df2021['year'] = 2021
    df_s1501 = pd.concat([df2019, df2020, df2021])
    df_s1501['cfips'] = df_s1501['GEO_ID'].str[-5:].astype('Int64')

    df_s1501['total_population'] = df_s1501['S1501_C01_001E'] + df_s1501['S1501_C01_006E']
    df_s1501['pct_bachelors_degree'] = (df_s1501['S1501_C01_005E'] + df_s1501['S1501_C01_012E']) / df_s1501['total_population']
    df_s1501['pct_graduate_degree'] = (df_s1501['S1501_C01_013E']) / df_s1501['total_population']
    df_s1501 = df_s1501[['cfips', 'year', 'pct_bachelors_degree', 'pct_graduate_degree']]
    df_s1501 = create_year(df_s1501, 2021, 2022)
    
    df_s1501.set_index(['cfips', 'year'], inplace=True, verify_integrity=True)
    df = df.join(df_s1501, how='left', on=['cfips', 'year'], lsuffix='_train', rsuffix='_census')
    df.reset_index(inplace=True)
    df.fillna(0)
    df['pct_bachelors_degree'] = df['pct_bachelors_degree'].astype(np.float32)
    df['pct_graduate_degree'] = df['pct_graduate_degree'].astype(np.float32)

    df_train, df_valid, df_test = split(df, validation_size, test_size)

    #X_cols = ['cfips', 'state', 'unix_elapsed_seconds', 'pct_bachelors_degree', 'pct_graduate_degree']
    #y_col = ['microbusiness_density']
    config = get_config()
    categorical_cols = config['categorical_cols']
    continuous_cols = config['continuous_cols']
    y_col = config['target']
    X_cols = categorical_cols + continuous_cols

    for col in categorical_cols:
        df_train[col] = df_train[col].astype('category')
        df_valid[col] = df_valid[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    # Create features and targets
    df_train_X, df_train_y = df_train[X_cols], df_train[y_col]
    df_valid_X, df_valid_y = df_valid[X_cols], df_valid[y_col]
    df_test_X, df_test_y = df_test[X_cols], df_test[y_col]

    return df_train_X, df_train_y, df_valid_X, df_valid_y, df_test_X, df_test_y


def split(df: pd.DataFrame, validation_size: int=1, test_size: int=1) -> pd.DataFrame:
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    unique_dates_sorted = sorted(df['measurement_date'].unique())

    training_dates = unique_dates_sorted[:-(validation_size+test_size)]
    validation_dates = unique_dates_sorted[-(validation_size+test_size):-test_size]
    test_dates = unique_dates_sorted[-test_size:]

    df_test = df[df['measurement_date'].isin(test_dates)].copy()
    df_valid = df[df['measurement_date'].isin(validation_dates)].copy()
    df_train = df[df['measurement_date'].isin(training_dates)].copy()

    del df_test['measurement_date']
    del df_valid['measurement_date']
    del df_train['measurement_date']
    logger.info('Training, validation, and test sets have been created.')
    return df_train, df_valid, df_test
