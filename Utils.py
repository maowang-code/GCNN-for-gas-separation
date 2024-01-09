import os
import time
import warnings
import numpy as np
import pandas as pd
#import imageio as iio

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



def get_atom_features(mol):
    """ 
    This will return a matrix / 2d array of the shape
    [Number of Nodes, Node Feature size]
    
    mol: rdkit mol object
    """
    all_node_feats = []

    for atom in mol.GetAtoms():
        node_feats = []
        # Feature 1: Atomic number        
        node_feats.append(atom.GetAtomicNum())
        # Feature 2: Atom degree
        node_feats.append(atom.GetDegree())
        # Feature 3: Formal charge
        node_feats.append(atom.GetFormalCharge())
        # Feature 4: Hybridization
        node_feats.append(atom.GetHybridization())
        # Feature 5: Aromaticity
        node_feats.append(atom.GetIsAromatic())
        # Feature 6: Total Num Hs
        node_feats.append(atom.GetTotalNumHs())
        # Feature 7: Radical Electrons
        node_feats.append(atom.GetNumRadicalElectrons())
        # Feature 8: In Ring
        node_feats.append(atom.IsInRing())
        # Feature 9: Chirality
        node_feats.append(atom.GetChiralTag())

        # Append node features to matrix
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)




def get_edge_index(mol):
    row, col = [], []
    
    for bond in mol.GetBonds():          ##an unweighted and undirected graph, both side connection
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        
    return torch.tensor([row, col], dtype=torch.long)


def prepare_dataloader(smiles_list,target,train=True):
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    data_list = []

    for i, mol in enumerate(mol_list):

        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)
        
        if train:
            y = torch.tensor([[target[i]]],dtype=torch.float)         
            data =torch_geometric.data.Data(x=x, edge_index=edge_index,y=y,smiles=mol_list[i])
            data_list.append(data)
        else:
            data =torch_geometric.data.Data(x=x, edge_index=edge_index,smiles=mol_list[i])
            data_list.append(data)
            
    return  data_list  


            