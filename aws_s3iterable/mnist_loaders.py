from io import BytesIO
import math
import os
import tarfile
import time
from typing import Any, Dict, List, Tuple

import PIL
from PIL import Image
from s3torchconnector import S3MapDataset, S3IterableDataset, S3ClientConfig, S3Reader
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import data_utilities as du


class MNISTMemory(Dataset):
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


class MNISTIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, bucket_name: str, tar_list: List[str], transform=None):
        logger = du.create_logger()
        logger.info('MNISTIterableDataset.__init__() called.')
        self.bucket_name = bucket_name
        self.tar_list = tar_list
        self.transform = transform

    def __iter__(self):
        # The logger needs to be recreated here since this function will run in
        # another process.
        logger = du.create_logger()
        logger.info('MNISTIterableDataset.__iter__() called.')
        worker_info = torch.utils.data.get_worker_info()
        tar_count = len(self.tar_list)

        if worker_info is None:  # Single process data loading, return the full iterator.
            start = 0
            end = tar_count
            worker_id = -1
        else:  # Split the workload for each worker process.
            worker_id = worker_info.id
            per_worker = int(math.ceil(tar_count / float(worker_info.num_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, tar_count)

        #images = []
        count = 0
        for tar_path in self.tar_list[start:end]:
            tar_byte_array = du.get_object_from_minio(self.bucket_name, tar_path)        
            # Open the tar stream.
            #tar_buffer = BytesIO(tar_byte_array)
            with tarfile.open(fileobj=BytesIO(tar_byte_array), mode='r') as tar:
                for member in tar.getmembers():
                    # Skip directories or non-file members
                    if member.isfile():
                        # Extract the file-like object
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            # Read content as bytes
                            label = int(member.name.split('/')[0])
                            image_stream = BytesIO(file_obj.read())
                            image = Image.open(image_stream)
                            count += 1
                            yield self.transform((image, label))
                            #images.append((image, label))

        logger.info(f'Worker ID: {worker_id} tar index start: {start} tar index end: {end}.')
        logger.info(f'Worker ID: {worker_id} Number of images: {count}.')
        #return map(self.transform, images)


class MNISTMap(Dataset):
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


class MNISTList(Dataset):
    def __init__(self, bucket_name: str, X, y, transform=None):
        self.bucket_name = bucket_name
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class IterTarTransform:
    def __init__(self, transform, bucket_name: str):
        self.transform = transform
        self.bucket_name = bucket_name

    def __call__(self, image: Tuple[PIL.Image.Image, int]) -> Tuple[torch.Tensor, int]:
        if self.transform is not None:
            label = image[1]
            image_tensor = self.transform(image[0])

        return (image_tensor, label)


class S3IterTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, object: S3Reader) -> Tuple[torch.Tensor, int]:
        content = object.read()
        image_pil = Image.open(BytesIO(content))
        image_tensor = self.transform(image_pil)
        label = int(object.key.split('/')[1])

        return (image_tensor, label)


class IterTransform:
    def __init__(self, transform, bucket_name: str):
        self.transform = transform
        self.bucket_name = bucket_name

    def __call__(self, object_path: str) -> Tuple[torch.Tensor, int]:
        img = du.get_image_from_minio(self.bucket_name, object_path)
        if self.transform is not None:
            image_tensor = self.transform(img)
        label = int(object_path.split('/')[1])
        return (image_tensor, label)


class S3MapTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, object: S3Reader) -> Tuple[torch.Tensor, int]:
        content = object.read()
        image_pil = Image.open(BytesIO(content))
        # For some reason the transform below returns a uint8 when a float is needed to normalize.
        #image_tensor = torchvision.transforms.functional.pil_to_tensor(image_pil)
        image_tensor = self.transform(image_pil)
        label = int(object.key.split('/')[1])

        return (image_tensor, label)


def create_mnist_loader(bucket_name: str, split: str, loader_type:str, batch_size:int, num_workers: int=1, smoke_test_count: int=0) -> Tuple[Any]:
    # Start of load time.
    start_time = time.perf_counter()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    if loader_type == 's3map':
        uri = f's3://{bucket_name}/{split}'
        aws_region = os.environ['AWS_REGION']
        dataset = S3MapDataset.from_prefix(uri, region=aws_region, transform=S3MapTransform(transform))
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return loader, (time.perf_counter()-start_time)    

    if loader_type == 's3iter':
        uri = f's3://{bucket_name}/{split}'
        aws_region = os.environ['AWS_REGION']
        dataset = S3IterableDataset.from_prefix(uri, region=aws_region, enable_sharding=True, transform=S3IterTransform(transform))
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return loader, (time.perf_counter()-start_time)    

    elif loader_type == 'itertar':
        # Get a list of objects and split them according to train and test.    
        tar_list = du.get_object_list(bucket_name, split)
        if smoke_test_count > 0:
            tar_list = tar_list[0:smoke_test_count]
        dataset = MNISTIterableDataset(bucket_name, tar_list, transform=IterTarTransform(transform, bucket_name))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return loader, (time.perf_counter()-start_time)    

    # The file loader type will download and load the training data directly into a dataset.
    if loader_type == 'file':
        dataset = datasets.MNIST('./mnistdata', download=True, 
                                 train=True if split == 'train' else False, 
                                 transform=transform)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if split == 'train' else False)
        return loader, (time.perf_counter()-start_time)    

    # The remaining loader types load from MinIO.
    # Get a list of objects and split them according to train and test.    
    X, y = du.get_mnist_list(bucket_name, split)

    if smoke_test_count > 0:
        X = X[0:smoke_test_count]
        y = y[0:smoke_test_count]

    loader = None
    if loader_type == 'list':
        dataset = MNISTList(bucket_name, X, y, transform=transform)
    elif loader_type == 'memory':
        dataset = MNISTMemory(bucket_name, X, y, transform=transform)
    elif loader_type == 'map':
        dataset = MNISTMap(bucket_name, X, y, transform=transform)
    elif loader_type == 'iter':
        dataset = MNISTIterableDataset(bucket_name, X, y, transform=IterTransform(transform, bucket_name))
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        raise ValueError('loader_type must be either file, list, full, map or s3map.')

    if loader == None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return loader, (time.perf_counter()-start_time)    
