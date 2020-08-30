from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os, random, pickle
import utils
import numpy as np
import torch

from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, Get3DDistanceMatrix

random.seed(0)

N_atom_features = 21

def get_atom_types(mol):
    """
    Family Acceptor => Acc
    Family Donor = Don
    Family Hydrophobe => Hph
    Family LumpedHydrophobe = LHp
    Family NegIonizable => Neg
    Family PosIonizable => Pos
    Family ZnBinder => Znb
    Family Aromatic => Aro

    C, N, O, F, P, S, Cl, Br, X,
    Aromatic, sp3, sp2, sp, Ring,
    Acceptor, Donor, Hydrophobe, LumpedHydrophobe, NegIonizable, PosIonizable, ZnBinder

    21 dimension
    """
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(mol)
    features = {}
    for sym in ['Acc', 'Don', 'Hyd', 'Lum', 'Neg', 'Pos', 'ZnB', 'Aro']:
        features[sym] = set()
    for feat in feats:
        sym = feat.GetFamily()[:3]
        [features[sym].add(i) for i in feat.GetAtomIds()]
    features_lb = {}
    for k, v in features.items():
        for i in v:
            if i not in features_lb:
                features_lb[i] = []
            features_lb[i].append(k)

    vecs = []
    for atom in mol.GetAtoms():
        atom_feat = np.zeros(21, dtype=int)
        an = atom.GetAtomicNum()
        if an == 6:
            atom_feat[0] = 1
        elif an == 7:
            atom_feat[1] = 1
        elif an == 8:
            atom_feat[2] = 1
        elif an == 9:
            atom_feat[3] = 1
        elif an == 15:
            atom_feat[4] = 1
        elif an == 16:
            atom_feat[5] = 1
        elif an == 17:
            atom_feat[6] = 1
        elif an == 35:
            atom_feat[7] = 1
        else:
            atom_feat[8] = 1
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP3:
            atom_feat[10] = 1
        elif hyb == Chem.HybridizationType.SP2:
            atom_feat[11] = 1
        elif hyb == Chem.HybridizationType.SP:
            atom_feat[12] = 1
        if atom.IsInRing():
            atom_feat[13] = 1
        idx = atom.GetIdx()
        if idx in features_lb:
            ff = features_lb[idx]
            if 'Aro' in ff:
                atom_feat[9] = 1
            if 'Acc' in ff:
                atom_feat[14] = 1
            if 'Don' in ff:
                atom_feat[15] = 1
            if 'Hyd' in ff:
                atom_feat[16] = 1
            if 'Lum' in ff:
                atom_feat[17] = 1
            if 'Neg' in ff:
                atom_feat[18] = 1
            if 'Pos' in ff:
                atom_feat[19] = 1
            if 'ZnB' in ff:
                atom_feat[20] = 1
        vecs.append(atom_feat)
    vecs = np.array(vecs)
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

        Y = self.pKd[idx]

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

