'''
Module that contains the Base class for training and testing models locally and distributed.
'''
from typing import List, Dict, Any

from torch import nn
from torch.utils.data.dataloader import DataLoader


class TrainingBase:
    '''
    Base class for training and testing models locally and distributed.
    '''
    def __init__(self):
        pass

    def train_model_dist(self, model: nn.Module, loader: DataLoader, training_parameters: Dict[str, Any]) -> List[float]:
        '''
        Trains a model distributed with the specified data loader.
        '''

    def train_model_local(self, model: nn.Module, loader: DataLoader, training_parameters: Dict[str, Any]) -> List[float]:
        '''
        Trains a model locally with the specified data loader.
        '''

    def setup_local_training(self, training_parameters: Dict[str, Any], loader_type: str):
        '''
        Sets up the model and the data loader for local training against a training set.
        '''

    def test_model_local(self, model: nn.Module, loader: DataLoader, training_parameters: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Tests the model locally against the test set.
        '''

    def setup_local_testing(self, training_parameters: Dict[str, Any], loader_type: str):
        '''
        Sets up the model and the data loader for testing the model locally against the test set.
        '''
