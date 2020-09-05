from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os, random, pickle
import utils
import numpy as np
import torch

from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, Get3DDistanceMatrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

# random.seed(0)

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

N_atom_features = 21
ARG_ANUM = {6:0, 7:1, 8:2, 9:3, 15:4, 16:5, 17:6, 35:7}
ARG_HYB = {Chem.HybridizationType.SP3:10, Chem.HybridizationType.SP2:11, Chem.HybridizationType.SP:12}
ARG_FEAT = {'Aro':9, 'Acc':14, 'Don':15, 'Hyd':16, 'Lum':17, 'Neg':18, 'Pos':19, 'ZnB':20}

def get_atom_types(mol):
    """
    C N O F P S Cl Br X Aro sp3 sp2 sp Ring Acc Don Hyd Lum Neg Pos ZnB
    """
    vecs = np.zeros((mol.GetNumAtoms(), N_atom_features), dtype=int)
    feats = factory.GetFeaturesForMol(mol)

    for feat in feats:
        sym = feat.GetFamily()[:3]
        if sym not in ARG_FEAT:
            continue
        for atomidx in feat.GetAtomIds():
            vecs[atomidx, ARG_FEAT[sym]] = 1
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        atomidx = atom.GetIdx()
        vecs[atomidx, ARG_ANUM.get(an, 8)] = 1
        hyb = atom.GetHybridization()
        if hyb in ARG_HYB:
            vecs[atomidx, ARG_HYB[hyb]] = 1
        if atom.IsInRing():
            vecs[atomidx, 13] = 1
    return vecs

def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = np.array(get_atom_types(m))
    if is_ligand:
        H = np.concatenate([H, np.zeros((n, N_atom_features), dtype=int)], 1)
    else:
        H = np.concatenate([np.zeros((n, N_atom_features), dtype=int), H], 1)
    return H

class MolDataset(Dataset):
    def __init__(self, keys, pKd, data_dir):
        self.data_dir = data_dir
        self.keys, self.pKd = self.check_data(keys, pKd)
        self.cachedir = '/tmp/moldata'
        os.makedirs(self.cachedir, exist_ok=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        caching mechanism is added
        """
        key = self.keys[idx]
        keyfname = f'{self.cachedir}/{key}.pkl'
        if os.path.exists(keyfname):
            sample = pickle.load(open(keyfname, 'rb'))
            return sample

        print(key, '... caching')

        pocket_fname = self.data_dir + '/' + key + '/' + key + '_pocket.pdb'
        for f in os.listdir(self.data_dir + '/' + key):
            if f.endswith('.sdf'):
                ligand_name = f[:3]
                ligand_fname = self.data_dir + '/' + key + '/' + ligand_name + '.sdf'
                break
        for m1 in Chem.SDMolSupplier(ligand_fname): break
        m2 = Chem.MolFromPDBFile(pocket_fname)
        if not m2:
            print(key)

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
        #clo = (dm < 4.0).astype(int)
        #agg_Adj2[:n1,n1:] = clo
        #agg_Adj2[n1:,:n1] = clo.T
        agg_Adj2[:n1,n1:] = dm
        agg_Adj2[n1:,:n1] = dm.T

        #node indice for aggregation
        valid = np.zeros((n1+n2,), dtype=int)
        valid[:n1] = 1

        Y = int(self.pKd[idx] >= 8.5)

        # agg_Adj1 = np.zeros((1, 1))

        sample = {
                  'H':H, \
                  'A1': agg_Adj1, \
                  'A2': agg_Adj2, \
                  'Y': Y, \
                  'V': valid, \
                  'key': key, \
                  }

        pickle.dump(sample, open(keyfname, 'wb'), protocol=4)
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
    # A1 = np.zeros((len(batch), 1, 1))
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

