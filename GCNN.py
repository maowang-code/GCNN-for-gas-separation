import os
import time
import warnings
import numpy as np
import pandas as pd

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

# use IPythonConsole for pretty drawings
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F 

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")



class GCN(torch.nn.Module):
    def __init__(self,embedding_size, dropout, linear_layers):  ## linear_size = embedding_size*2
        # Init parent
        super(GCN, self).__init__()

        # GCN layers
        self.initial_conv = GCNConv(9, embedding_size)    ## the number of node features is 9
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        
        ##--dense layer and dropout------
        self.dropout = nn.Dropout(dropout)
        
        self.linears = nn.ModuleList(
          [nn.Linear(embedding_size*2, embedding_size*2) for i in range(linear_layers)])
        
        self.dropouts = nn.ModuleList(
          [nn.Dropout(dropout) for i in range(linear_layers)])
        

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)
          
        # Global Pooling
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        
        for l, d in zip(self.linears, self.dropouts):
            hidden = l(hidden)
            hidden = F.relu(hidden)
            hidden = d(hidden)      
            
        out = self.out(hidden) 
        return out, hidden


    
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
    


def train(data):
    # Enumerate over the data
    model.train()
    for batch in data:
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred, _ = model(batch.x.float(), batch.edge_index, batch.batch) 
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()  
        # Update using the gradients
        optimizer.step() 
        

def tes_loss(test_loader, model, device):    
    model.eval()
    loss_all = 0
    for test_batch in test_loader:
        with torch.no_grad():
            test_batch.to(device)
            pred, _ = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            loss = torch.sqrt(loss_fn(pred, test_batch.y))  ## RMSE
            loss_all = loss_all+loss.detach().cpu().item() * len(pred)      ## length of batched data
    
    loss_all_mean = loss_all/len(test_loader.dataset)   ## length of whole dataset
    return loss_all_mean



def pred(test_loader, model, device):
    # Analyze the results for one batch
    # out put 1 dim prediciton of data
    model.eval()
    prediction = []
    for test_batch in test_loader:
        with torch.no_grad():
            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            prediction = prediction + pred.t().tolist()[0]   
    return np.array(prediction)


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
            
