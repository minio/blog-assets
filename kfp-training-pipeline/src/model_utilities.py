from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F

import data_utilities as du

def test_model(df_test_X, df_test_y, model) -> float:
    categorical_cols = du.get_config()['categorical_cols']

    test_dataset = CountyDataset(df_test_X, df_test_y, categorical_cols)
    batch_size = 1000
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    predictions = torch.tensor([])
    targets = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for x1, x2, y in test_loader:
            batch_pred = model(x1, x2)
            predictions = torch.cat((predictions, batch_pred), 0)
            targets = torch.cat((targets, y))
    
    smape_val = smape(targets.squeeze().numpy(), predictions.squeeze().numpy())
    return smape_val


def smape(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def create_model():
    config = du.get_config()
    embedding_sizes = config['embedding_sizes']
    continuous_cols = config['continuous_cols']
    #model = MBDModel(embedding_sizes, len(continuous_cols))
    model = RegressionModel(embedding_sizes, len(continuous_cols))
    return model


def train_model(df_train_X, df_train_y, df_valid_X, df_valid_y, epochs=5, lr=0.01, wd=0.0) -> Tuple[nn.Module, List]:
    config = du.get_config()
    categorical_cols = config['categorical_cols'] 
    #continuous_cols = config['continuous_cols']
    #embedding_sizes = [(3135, 50), (51, 26)]

    #model = MBDModel(embedding_sizes, len(continuous_cols))
    #model = RegressionModel(embedding_sizes, len(continuous_cols))
    model = create_model()

    # Create the datasets
    train_dataset = CountyDataset(df_train_X, df_train_y, categorical_cols)
    valid_dataset = CountyDataset(df_valid_X, df_valid_y, categorical_cols)
    # Create the loaders
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    optim = get_optimizer(model, lr = lr, wd = wd)
    results = []
    for i in range(epochs): 
        loss = train_epoch(model, optim, train_loader)
        print('Training loss: ', loss)
        val_loss, val_accuracy = validate_model(model, valid_loader)
        results.append((loss, val_loss, val_accuracy))
    return model, results


class CountyDataset(Dataset):
    def __init__(self, X, y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = y.values.astype(np.float32)  #to_numpy()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


class RegressionModel(nn.Module):

    def __init__(self, embedding_sizes, n_cont):
        super(RegressionModel, self).__init__()

        # Set up the embeddings.
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont

        # Set up the rest of the network.
        self.linear1 = nn.Linear(self.n_emb + self.n_cont, 20, bias=True)
        self.linear2 = nn.Linear(20, 20, bias=True)
        self.linear3 = nn.Linear(20, 1, bias=True)

    def forward(self, x_cat, x_cont):
        out = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        out = torch.cat(out, 1)
        out = torch.cat([out, x_cont], 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out


class MBDModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 1)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        out = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        out = torch.cat(out, 1)
        out = self.emb_drop(out)
        out2 = self.bn1(x_cont)
        out = torch.cat([out, out2], 1)
        out = F.relu(self.lin1(out))
        out = self.drops(out)
        out = self.bn2(out)
        out = F.relu(self.lin2(out))
        out = self.drops(out)
        out = self.bn3(out)
        out = self.lin3(out)
        return out


def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


def train_epoch(model, optimizer, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        model.zero_grad()
        optimizer.zero_grad()
        batch_size = y.shape[0]
        output = model(x1, x2)
        loss = F.mse_loss(output, y)
        #print(batch_size)
        #print('Output:', output)
        #print('Target:', y)
        #print(loss.item())
        #bra
        loss.backward()
        optimizer.step()
        total += batch_size
        sum_loss += batch_size * (loss.item())
    return float(sum_loss/total)


def validate_model(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    sum_l1_loss = 0
    for x1, x2, y in valid_dl:
        batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.mse_loss(out, y)
        sum_loss += batch_size * (loss.item())
        total += batch_size
        l1_loss = F.l1_loss(out, y)
        sum_l1_loss += batch_size * l1_loss
    print('Validation loss (MSE) %.3f - Accuracy (MAE) %.3f' % (sum_loss/total, sum_l1_loss/total))
    return sum_loss/total, sum_l1_loss/total