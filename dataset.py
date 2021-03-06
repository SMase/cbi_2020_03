from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import utils
import numpy as np
import torch
import random

from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle

random.seed(0)

def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(utils.atom_feature(m, i, None, None))
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,28))], 1)
    else:
        H = np.concatenate([np.zeros((n,28)), H], 1)
    return H        

class MolDataset(Dataset):

    def __init__(self, keys, pKd, data_dir, distance):
        self.data_dir = data_dir
        # self.data_df = pd.read_csv('data.csv')
        self.distance = distance
        self.keys, self.pKd = self.check_data(keys, pKd)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with open(self.data_dir+'{0}/{0}.pair_{1}.pkl'.format(key, self.distance), 'rb') as f:
            m1, m2 = pickle.load(f)

        # with open(self.data_dir+'{0}_pair.pkl'.format(key), 'rb') as f:
        #     m1, m2 = pickle.load(f)

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

        #Y = float(np.loadtxt(self.data_dir+'/{0}/value'.format(key)))
        #Y = self.data_df[self.data_df['pdbid'] == key]['pvalue'].values[0]
        
        #pIC50 to class
        # Y = 1 if 'CHEMBL' in key else 0

        #if n1+n2 > 300 : return None
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
            chk_dir = os.path.join(self.data_dir, pdb)
            if not os.path.isdir(chk_dir):
                print('Warnings: There is no directory. ({})'.format(chk_dir))
                continue
            chk_file = os.path.join(self.data_dir, pdb, "{0}.pair_{1}.pkl".format(pdb, self.distance))
            # chk_file = os.path.join(self.data_dir, '{0}_pair.pkl'.format(pdb))
            if not os.path.isfile(chk_file):
                print('Warnings: There is no file. ({})'.format(chk_file))
                continue

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
        #return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    
    H = np.zeros((len(batch), max_natoms, 56))
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
