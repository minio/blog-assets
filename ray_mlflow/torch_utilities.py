import io
import json
import logging
import os
from typing import Any, Dict, List, Tuple
from time import time

from minio import Minio
from minio.error import S3Error
import numpy as np
import PIL
import ray
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import data_utilities as du

device = torch.device('cpu')  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MNISTModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        self.lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.lin4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.lin5 = nn.Linear(hidden_sizes[3], output_size)
        self.activation = nn.ReLU()
        self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.activation(out)
        out = self.lin2(out)
        out = self.activation(out)
        out = self.lin3(out)
        out = self.activation(out)
        out = self.lin4(out)
        out = self.activation(out)
        out = self.lin5(out)
        out = self.output_activation(out)
        return out


def test_model(model: MNISTModel, test_data) -> Dict[str, Any]:
    correct_count, total_count = 0, 0

    for batch in test_data.iter_torch_batches(batch_size=100):
        # Get the images and labels from the batch.
        images, labels = batch['X'], batch['y']
        labels = labels.type(torch.LongTensor)   # casting to long
        images, labels = images.to(device), labels.to(device)

        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            total_count += 1
    
    testing_metrics = {
        'incorrect_count': total_count-correct_count,
        'correct_count': correct_count,
        'accuracy': (correct_count/total_count)
    }
    print("\nNumber Of Images Tested =", total_count)
    print('Incorrect predictions: ', total_count-correct_count)
    print('Model Accuracy = ', (correct_count/total_count))
    return testing_metrics


class ImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = get_image_from_minio('mnist', self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


def create_minio_data_loaders(batch_size:int, smoke_test_size: float=None) -> Tuple[Any]:
    # Start of load time.
    start_time = time()

    # Get a list of objects and split them according to train and test.    
    object_list = du.get_object_list('mnist')
    X_train, y_train, X_test, y_test = du.create_train_test(object_list)

    if smoke_test_size > 0:
        train_size = int(smoke_test_size*len(X_train))
        test_size = int(smoke_test_size*len(X_test))
        X_train = X_train[0:train_size]
        y_train = y_train[0:train_size]
        X_test = X_test[0:test_size]
        y_test = y_test[0:test_size]

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_dataset = ImageDataset(X_train, y_train, transform=transform)
    test_dataset = ImageDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, len(train_dataset), len(test_dataset), (time()-start_time)    


def create_memory_data_loaders(batch_size: int, smoke_test_size: float=None) -> Tuple[Any]:
    # Start of load time.
    start_time = time()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    train_dataset = datasets.MNIST('./mnistdata', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('./mnistdata', download=True, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, len(train_dataset), len(test_dataset), (time()-start_time)

