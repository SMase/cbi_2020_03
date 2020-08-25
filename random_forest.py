from utils import get_atom_types
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import argparse, time, pickle, os, gzip
import multiprocessing as mp

class Timer:
    def __init__(self, dt=10):
        self.dt = dt
        self.start = time.time()
        self.t0 = self.start

    def mark(self):
        self.t = time.time()
        if self.dt < self.t - self.t0:
            self.t0  = self.t
            return True
        return False

    def lapse(self):
        return time.time() - self.start

def get_distance_matrix(mol1, mol2):
    coords1 = mol1.GetConformer(0).GetPositions()
    coords2 = mol2.GetConformer(0).GetPositions()
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    c1 = np.tile(coords1, n2).reshape(n1, n2, 3)
    c2 = np.tile(coords2.T, n1).T.reshape(n1, n2, 3)
    return np.linalg.norm(c1-c2, axis=2)

def get_A(ligand, pocket, n_atom_features, max_ligand_size, max_pocket_size):
    ligand_atom_types = get_atom_types(ligand)
    pocket_atom_types = get_atom_types(pocket)

    M = ligand.GetNumAtoms()
    N = pocket.GetNumAtoms()

    z = np.zeros((max_ligand_size - M, 21), dtype=int)
    A1 = np.concatenate((ligand_atom_types, z), axis=0)

    z = np.zeros((max_pocket_size - N, 21), dtype=int)
    A2 = np.concatenate((pocket_atom_types, z), axis=0)

    A = np.concatenate((A1, A2), axis=0)

    return A

def get_D(ligand, pocket, thres, max_ligand_size, max_pocket_size):
    M = ligand.GetNumAtoms()
    N = pocket.GetNumAtoms()

    Dl = np.array(get_distance_matrix(ligand, ligand) < thres, dtype=int)
    Dp = np.array(get_distance_matrix(pocket, pocket) < thres, dtype=int)
    Di = np.array(get_distance_matrix(ligand, pocket) < thres, dtype=int)

    D = np.zeros((max_ligand_size+max_pocket_size, max_ligand_size+max_pocket_size), dtype=int)
    D[:M,:M] = Dl
    D[max_ligand_size:max_ligand_size+N, max_ligand_size:max_ligand_size+N] = Dp
    D[:M, max_ligand_size:max_ligand_size+N] = Di
    D[max_ligand_size:max_ligand_size+N, :M] = Di.T

    return D

def serialize(A, D):
    return np.concatenate((np.ravel(A), np.ravel(D)))

def gen(fname, intermol_dist_thres=4):
    for line in open(fname, 'rt'):
        it = line.rstrip().split()
        pdb_code = it[0]
        ligand_name = it[1]
        pKd = float(it[3])
        yield dict(pdb_code=pdb_code, ligand_name=ligand_name, pKd=pKd, intermol_dist_thres=intermol_dist_thres)

def worker(args):
    pdb_code = args['pdb_code']
    ligand_name = args['ligand_name']
    intermol_dist_thres = args['intermol_dist_thres']

    ligand = Chem.MolFromMolFile(f'cbidata/{pdb_code}/{ligand_name}.sdf')
    pocket = Chem.MolFromPDBFile(f'cbidata/{pdb_code}/{pdb_code}_pocket.pdb')

    ligand_atom_types = get_atom_types(ligand)
    pocket_atom_types = get_atom_types(pocket)

    A = get_A(ligand, pocket, 21, 60, 240)
    D = get_D(ligand, pocket, intermol_dist_thres, 60, 240)

    data = serialize(A, D)

    ret = dict(args)
    ret['data'] = data

    return ret

def load_data(fname, limit=0):
    print(f'Loading {fname}')

    X = []
    y = []
    count = 0

    timer = Timer(10)

    pool = mp.Pool(os.cpu_count())

    for ret in pool.imap_unordered(worker, gen(fname, 4.5)):
        count += 1
        data =  ret['data']
        pKd = ret['pKd']
        X.append(data)
        y.append(pKd)

        if timer.mark():
            print(f'{timer.lapse():.2f} {count}')

        if 0 < limit and limit <= count:
            break

    print(f'{timer.lapse():.2f} {count}')

    X = np.array(X)
    y = np.array(y)

    return X, y

def main(args):
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(s)

    if not os.path.exists('zzz.pkl.gz'):
        train_X, train_y = load_data(args.train_keys)
        test_X, test_y = load_data(args.test_keys)
        d = train_X, train_y, test_X, test_y
        pickle.dump(d, gzip.open('zzz.pkl.gz', 'wb'), protocol=4)
    else:
        train_X, train_y, test_X, test_y = pickle.load(gzip.open('zzz.pkl.gz', 'rb'))

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    regr = RandomForestRegressor(max_depth=10, random_state=0, verbose=2, n_jobs=-1).fit(train_X, train_y)
    print(regr.score(test_X, test_y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_keys", help="train keys", type=str, default='keys/train_keys.txt')
    parser.add_argument("--test_keys", help="test keys", type=str, default='keys/val_keys.txt')
    parser.add_argument("--data_fpath", help="file path of dude data", type=str, default='data_select/')
    args = parser.parse_args()
    print(args)

    main(args)
