
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





def gen_uncertainty(test_loader,model,number_repeat): ## unlabeled x and trained model
    '''
    predict the test_loader's uncertainty.
    
    Args:
        test_loader: the test_loader.
        model: the trained model.

    Returns:
        the y predict value, and uncertainty.
    '''    
    test_pred_y_all = []
    model.eval()    ## close all the dropout and Batch Normalization
    enable_dropout(model)  ## open the droppout 

    for i in range(number_repeat):   
        test_gt_y = []
        test_pred_y = []
        for test_batch in test_loader:
            with torch.no_grad():
                test_batch.to(device)
                pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
                test_pred_y= test_pred_y +pred.t().tolist()[0]    
        test_pred_y_all.append(test_pred_y)   
    test_pred_y_mean = np.mean(test_pred_y_all,axis=0)
    test_pred_y_std = np.std(test_pred_y_all,axis=0)

    return test_pred_y_mean,test_pred_y_std   



def acquisition(current_train_max,smiles_list_out,yhat,std,xi=0.01):
    '''
    Computes the EI at yhat based on existing samples y_train_pred_mean
    and out_pred_y_std.
    
    Args:
        current_train_max: predicted max trained values
        smiles_list_out: the smiles list of the predicted polymer
        yhat:  the predicted mean ffv of test 
        std:  the predicted std ffv of test 
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        ranked smiles list of the input data, EI index with descending sequence, EI values with descending sequence.
    '''
    
    from scipy.stats import norm  
    mu_sample_opt = current_train_max 
    with np.errstate(divide='warn'):
        imp = yhat - mu_sample_opt - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + yhat * norm.pdf(Z)
        ei[std == 0.0] = 0.0
        ei_descend = np.sort(ei)[::-1] 
        ei_idx_descend = np.argsort(ei)[::-1] 
        ranked_smiles = [smiles_list_out[i] for i in ei_idx_descend]

    return ranked_smiles,ei_idx_descend,ei_descend
