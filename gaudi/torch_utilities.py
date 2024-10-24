from io import BytesIO
import os
import time
from typing import Any, Dict, List, Tuple

from PIL import Image
from s3torchconnector import S3MapDataset, S3IterableDataset, S3ClientConfig, S3Reader
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import data_utilities as du


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNISTModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        self.lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.lin4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.lin5 = nn.Linear(hidden_sizes[3], output_size)
        #self.activation = F.relu()
        #self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.relu(self.lin4(out))
        out = F.log_softmax(self.lin5(out), dim=1)
        return out


def test_model_local(model: nn.Module, loader: DataLoader, training_parameters: Dict[str, Any]) -> Dict[str, Any]:
    logger = du.create_logger()

    device = torch.device('cpu')
    if training_parameters['use_gpu']:
        if torch.xpu.is_available(): device = torch.device('xpu')
        if torch.cuda.is_available(): device = torch.device('cuda')

    # Define a transform to normalize the images and turn them into tensors
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    correct_count, total_count = 0, 0
    
    for images, labels in loader:
        #image_list = du.get_images_from_minio(training_parameters['bucket_name'], image_names)
        #images = torch.stack([transform(x) for x in image_list], dim=0)
        images = images.to(device)
        labels = labels.to(device)

        for i in range(len(labels)):
            img = images[i].view(1, 784)
            #img = images[i]

            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps[0])
            pred_label = probab.index(max(probab))
            true_label = labels[i]
            if(true_label == pred_label):
                correct_count += 1
            total_count += 1
    
    testing_metrics = {
        'incorrect_count': total_count-correct_count,
        'correct_count': correct_count,
        'accuracy': (correct_count/total_count)
    }
    logger.info(f'Number Of Images Tested: {total_count}.')
    logger.info(f'Model Accuracy: {correct_count/total_count}.')
    return testing_metrics


def test_model(model: MNISTModel, test_data, device: str='cpu') -> Dict[str, Any]:
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


class ImageDatasetFull(Dataset):
    def __init__(self, bucket_name: str, X, y, transform=None):
        self.bucket_name = bucket_name
        self.y = y
        self.transform = transform

        raw_images = du.get_images_from_minio(bucket_name, X)
        images = torch.stack([transform(x) for x in raw_images], dim=0)
        self.X = images

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class ImageDatasetBatch(Dataset):
    def __init__(self, bucket_name: str, image_list: List[str], y: List[int], transform=None):
        self.bucket_name = bucket_name
        self.X = image_list
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        img = du.get_image_from_minio(self.bucket_name, self.X[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[index]


class ImageDatasetList(Dataset):
    def __init__(self, bucket_name: str, X, y, transform=None):
        self.bucket_name = bucket_name
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_mnist_training_loader(bucket_name: str, loader_type:str, batch_size:int, num_workers: int=1, smoke_test_size: float=0) -> Tuple[Any]:
    # Start of load time.
    start_time = time.perf_counter()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # The file loader type will download and load the training data directly into a dataset.
    if loader_type == 'file':
        train_dataset = datasets.MNIST('./mnistdata', download=True, train=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, (time.perf_counter()-start_time)    

    # The remaining loader types load from MinIO.
    # Get a list of objects and split them according to train and test.    
    X_train, y_train = du.get_mnist_lists(bucket_name, 'train')

    if smoke_test_size > 0:
        #train_size = int(smoke_test_size*len(X_train))
        X_train = X_train[0:smoke_test_size]
        y_train = y_train[0:smoke_test_size]

    if loader_type == 'list':
        train_dataset = ImageDatasetList(bucket_name, X_train, y_train, transform=transform)
    elif loader_type == 'full':
        train_dataset = ImageDatasetFull(bucket_name, X_train, y_train, transform=transform)
    elif loader_type == 'batch':
        train_dataset = ImageDatasetBatch(bucket_name, X_train, y_train, transform=transform)
    elif loader_type == 's3map':
        uri = f's3://{bucket_name}/train'
        aws_region = os.environ['AWS_REGION']
        train_dataset = S3MapDataset.from_prefix(uri, region=aws_region, transform=MNISTTransform(transform))
    else:
        raise ValueError('loader_type must be either file, list, batch, full or s3map.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, (time.perf_counter()-start_time)    


def transform_s3_object(object: S3Reader) -> torch.Tensor:
    '''
    This is a transform function that passed to S3MapDataset.from_prefix() and is called for every object return
    from object storage.

    TODO: Turn this function into a callable class so the pil transform can be passed in and not hard coded.
    '''
    pil_transform: transforms.Compose = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    content = object.read()
    image_pil = Image.open(BytesIO(content))
    # For some reason the transform below returns a uint8 when a float is needed to normalize.
    #image_tensor = torchvision.transforms.functional.pil_to_tensor(image_pil)
    image_tensor = pil_transform(image_pil)

    if object.key[:5] == 'train':
        label = int(object.key[6])
    if object.key[:4] == 'test':
        label = int(object.key[5])

    return (image_tensor, label)


class MNISTTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, object: S3Reader) -> torch.Tensor:
        content = object.read()
        image_pil = Image.open(BytesIO(content))
        # For some reason the transform below returns a uint8 when a float is needed to normalize.
        #image_tensor = torchvision.transforms.functional.pil_to_tensor(image_pil)
        image_tensor = self.transform(image_pil)
        label = int(object.key.split('/')[1])

        return (image_tensor, label)


def create_mnist_testing_loader(bucket_name: str, loader_type:str, batch_size:int, smoke_test_size: float=0) -> Tuple[Any]:
    # Start of load time.
    start_time = time.perf_counter()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # The file loader type will download and load the training data directly into a dataset.
    if loader_type == 'file':
        test_dataset = datasets.MNIST('./mnistdata', download=True, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_loader, (time.perf_counter()-start_time)    

    # The remaining loader types load from MinIO.
    # Get a list of objects and split them according to train and test.    
    _, _, X_test, y_test = du.get_mnist_lists(bucket_name)

    if smoke_test_size > 0:
        test_size = int(smoke_test_size*len(X_test))
        X_test = X_test[0:test_size]
        y_test = y_test[0:test_size]

    if loader_type == 'list':
        test_dataset = ImageDatasetList(bucket_name, X_test, y_test, transform=transform)
    elif loader_type == 'full':
        test_dataset = ImageDatasetFull(bucket_name, X_test, y_test, transform=transform)
    elif loader_type == 'batch':
        test_dataset = ImageDatasetBatch(bucket_name, X_test, y_test, transform=transform)
    #elif loader_type == 's3map':
    #    uri = f's3://{bucket_name}'
    #    test_dataset = S3MapDataset.from_prefix(uri, region=AWS_REGION)
    #    item = map_dataset[0]
    #    content = item.read()
    #    print(item.bucket)
    #    print(item.key)
    #    print(len(content))

    else:
        raise ValueError('loader_type must be either file, list, batch, or full.')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    return test_loader, (time.perf_counter()-start_time)
