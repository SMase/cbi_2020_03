import numpy as np
import torch
from scipy import sparse
import os.path
import time
import torch.nn as nn
from ase import Atoms, Atom
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

# N_atom_features = 21
# N_atom_features = 28


def set_cuda_visible_device(ngpus):
    # import subprocess
    # import os
    # empty = []
    # for i in range(8):
    #     command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
    #     output = subprocess.check_output(command, shell=True).decode("utf-8")
    #     #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
    #     if int(output)==1:
    #         empty.append(i)
    # if len(empty)<ngpus:
    #     print ('avaliable gpus are less than required')
    #     exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28

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
    
