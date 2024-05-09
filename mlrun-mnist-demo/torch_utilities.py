from typing import Any, Dict, List, Tuple
from time import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import data_utilities as du

device = torch.device('cpu')  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        if not self.training:
            ps = torch.exp(out)
            return torch.argmax(ps, dim=1)
        else:
            return out


def test_model_local(model: nn.Module, loader: DataLoader, device='cpu') -> Dict[str, Any]:
    correct_count, total_count = 0, 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Put the model in evaluation mode.
            # Turn off gradients to speed up this part
            model.eval()
            with torch.no_grad():
                pred_label = model(img)
                pred_label = pred_label.numpy()[0]
            true_label = labels.numpy()[i]

            if(true_label == pred_label):
                correct_count += 1
            total_count += 1
    
    testing_metrics = {
        'incorrect_count': total_count-correct_count,
        'correct_count': correct_count,
        'accuracy': (correct_count/total_count)
    }
    print("Number Of Images Tested =", total_count)
    print("\nModel Accuracy =", (correct_count/total_count))
    return testing_metrics


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
        img = du.get_image_from_minio('mnist', self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        img = img.view(1, 784)
        return img, self.y[idx]


def create_minio_data_loaders(batch_size:int, smoke_test_size: float=-1) -> Tuple[Any]:
    # Start of load time.
    start_time = time()

    # Get a list of objects and split them according to train and test.    
    X_train, y_train, X_test, y_test = du.get_train_test_data(smoke_test_size)

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

    return train_loader, test_loader, (time()-start_time)    


def create_memory_data_loaders(batch_size: int) -> Tuple[Any]:
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
    return train_loader, test_loader, (time()-start_time)
