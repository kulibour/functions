import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

class compile_dataset(Dataset):
    def __init__(self, dataset, features, y_name):
        self.dataset = dataset
        self.features = features
        self.y_name = y_name

    def __len__(self):
        self.size = self.dataset.shape[0]
        return self.size

    def __getitem__(self, idx):
        x = np.array(self.dataset[self.features].values[idx], dtype=np.float32)
        y = np.array(self.dataset[self.y_name].values[idx], dtype=np.float32)
        return x, y

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.ReLU, normalization=nn.BatchNorm1d, num_layer=5, dropout=0.5):
        super(MLP, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim,hidden_dim))
        for n in range(num_layer-2):
            m = int(2*n) if n!=0 else 1
            if normalization:
                layers.append(normalization(hidden_dim//m))
            layers.append(nn.Linear(hidden_dim//m,hidden_dim//(m*2)))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim//(m*2),output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

    def train_loop(self, loader, model, loss_func, optimizer, DEVICE, progress=False):
        model = model.train()
        if progress:
            loader = tqdm(loader)
        loss_list = []
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_func(logits.flatten(), y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        return sum(loss_list)

    def eval_loop(self, loader, model, DEVICE, progress=False):
        model = model.eval()
        if progress:
            loader = tqdm(loader)
        y_exp_list,y_pred_list = [],[]
        for x, y in loader:
            with torch.no_grad():
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                y_exp_list.append(y.cpu())
                y_pred_list.append(logits.cpu().flatten())
        return torch.cat(y_exp_list), torch.cat(y_pred_list)
