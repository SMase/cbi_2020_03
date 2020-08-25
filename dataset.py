from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import utils
import numpy as np
import torch
import random

from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, Get3DDistanceMatrix
import pickle

random.seed(0)

# N_atom_features = 28
N_atom_features = 21

def get_atom_feature(m, is_ligand=True):
    x = utils.get_atom_types(m)
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(x[i])
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n, N_atom_features))], 1)
    else:
        H = np.concatenate([np.zeros((n, N_atom_features)), H], 1)
    return H        

"""
def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(utils.atom_feature(m, i))
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n, N_atom_features))], 1)
    else:
        H = np.concatenate([np.zeros((n, N_atom_features)), H], 1)
    return H        
"""

"""
class MolDataset(Dataset):
    def __init__(self, keys, pKd, data_dir):
        self.data_dir = data_dir
        self.keys, self.pKd = self.check_data(keys, pKd)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        pocket_fname = self.data_dir + '/' + key + '/' + key + '_pocket.pdb'
        for f in os.listdir(self.data_dir + '/' + key):
            if f.endswith('.sdf'):
                ligand_name = f[:3]
                ligand_fname = self.data_dir + '/' + key + '/' + ligand_name + '.sdf'
                break
        for m1 in Chem.SDMolSupplier(ligand_fname): break
        m2 = Chem.MolFromPDBFile(pocket_fname)

        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        H1 = get_atom_feature(m1, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        H2 = get_atom_feature(m2, False)

        #aggregation
        H = np.concatenate([H1, H2], 0)
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(d1,d2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        #node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1

        Y = self.pKd[idx]

        sample = {
                  'H':H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'Y': Y, \
                  'V': valid, \
                  'key': key, \
                  }

        return sample

    def check_data(self, keys, val):
        checked_pdb = []
        checked_pKd = []
        for pdb, pkd in zip(keys, val):
            checked_pdb.append(pdb)
            checked_pKd.append(pkd)
        return checked_pdb, checked_pKd
"""

class MolDataset(Dataset):
    def __init__(self, keys, pKd, data_dir):
        self.data_dir = data_dir
        self.keys, self.pKd = self.check_data(keys, pKd)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        pocket_fname = self.data_dir + '/' + key + '/' + key + '_pocket.pdb'
        for f in os.listdir(self.data_dir + '/' + key):
            if f.endswith('.sdf'):
                ligand_name = f[:3]
                ligand_fname = self.data_dir + '/' + key + '/' + ligand_name + '.sdf'
                break
        for m1 in Chem.SDMolSupplier(ligand_fname): break
        m2 = Chem.MolFromPDBFile(pocket_fname)

        # with open(self.data_dir+'{0}_pair.pkl'.format(key), 'rb') as f:
        #     m1, m2 = pickle.load(f)

        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        H1 = get_atom_feature(m1, True)

        dis1 = Get3DDistanceMatrix(m1)
        Adj1 = (dis1 < 4.0).astype(int)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        H2 = get_atom_feature(m2, False)
        
        dis2 = Get3DDistanceMatrix(m2)
        Adj2 = (dis2 < 4.0).astype(int)

        #aggregation
        H = np.concatenate([H1, H2], 0)
        dm = distance_matrix(d1, d2)

        agg_Adj1 = np.zeros((n1+n2, n1+n2))
        agg_Adj1[:n1, :n1] = Adj1
        agg_Adj1[n1:, n1:] = Adj2
        agg_Adj2 = np.copy(agg_Adj1)
        dm = distance_matrix(d1, d2)
        clo = (dm < 4.0).astype(int)
        agg_Adj2[:n1, n1:] = clo
        agg_Adj2[n1:, :n1] = clo.T
        agg_Adj2 = agg_Adj2.astype(int)

        #node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1

        Y = self.pKd[idx]

        sample = {
                  'H':H, \
                  'A1': agg_Adj1, \
                  'A2': agg_Adj2, \
                  'Y': Y, \
                  'V': valid, \
                  'key': key, \
                  }

        return sample

    def check_data(self, keys, val):
        checked_pdb = []
        checked_pKd = []
        for pdb, pkd in zip(keys, val):
            checked_pdb.append(pdb)
            checked_pKd.append(pkd)
        return checked_pdb, checked_pKd

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    
    H = np.zeros((len(batch), max_natoms, 2*N_atom_features))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))
    keys = []
    n_atom = []
    
    for i in range(len(batch)):
        natom = len(batch[i]['H'])

        H[i,:natom] = batch[i]['H']
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        Y[i] = batch[i]['Y']
        V[i,:natom] = batch[i]['V']
        n_atom.append(natom)
        keys.append(batch[i]['key'])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()
    
    return H, A1, A2, Y, V, keys, n_atom
