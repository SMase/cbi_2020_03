from gnn import gnn
import torch.nn as nn
import torch
from torch.utils.data import DataLoader                                     
import numpy as np
import os, argparse, time, shutil, sys, random
import utils
from rdkit import Chem
from rdkit.Chem import QED

def read_keyfile(fname, local=False):
    keys = []
    for line in open(fname):
        it = line.rstrip().split('\t')
        if local:
            pdb_code, value = it[0], float(it[1])
        else:
            pdb_code, ligand_name, year, value = it[0], it[1], int(it[2]), float(it[3])
        if os.path.exists(f'cbidata/{pdb_code}'):
            keys.append((pdb_code, value))
    return keys

def read_ligand(dirname):
    sdf_found = False
    for f in os.listdir(dirname):
        if f.endswith('.sdf'):
            sdf_found = True
            sdf_fname = f'{dirname}/{f}'
            for ligand_mol in Chem.SDMolSupplier(sdf_fname):
                break
            break
    return ligand_mol if sdf_found else None

def filter_and_stratify(*dataset, random_stratify=False):
    all_data = []
    for _ in dataset:
        all_data += _
    if random_stratify:
        random.shuffle(all_data)
    else:
        data = sorted(all_data, key=lambda x: x[1], reverse=True)
    train = []
    test = []
    test2 = []

    count = 0
    for key, pkd in all_data:
        if pkd < 2 or 11 < pkd:
            test2.append((key, pkd))
            continue

        ligand_mol = read_ligand(f'cbidata/{key}')

        n_atoms = ligand_mol.GetNumAtoms()
        if n_atoms < 10 or 45 < n_atoms:
            test2.append((key, pkd))
            continue
        if QED.qed(ligand_mol) < 0.4:
            test2.append((key, pkd))
            continue

        count += 1

        if count % 5 == 0:
            test.append((key, pkd))
        else:
            train.append((key, pkd))
    return train, test, test2

def write_keys(keys, oname):
    with open(oname, 'wt') as out:
        for key, pkd in keys:
            print(key, pkd, sep='\t', file=out)

def main(args):
    import datetime
    print(datetime.datetime.now().isoformat())

    # hyper parameters
    num_epochs = args.epoch
    lr = args.lr
    ngpu = args.ngpu
    batch_size = args.batch_size
    data_fpath = args.data_fpath
    save_dir = args.save_dir

    if args.clear_cache:
        try:
            shutil.rmtree('/tmp/moldata')
        except:
            pass

    os.makedirs(save_dir, exist_ok=True)

    train = read_keyfile(args.train_keys)
    test = read_keyfile(args.test_keys)
    train_keys, test_keys, test2_keys = filter_and_stratify(train, test, random_stratify=args.random_stratify)

    write_keys(train_keys, 'train.local.key')
    write_keys(test_keys, 'test.local.key')
    write_keys(test2_keys, 'test2.local.key')

    print(f'Number of train data: {len(train_keys)}')
    print(f'Number of test data: {len(test_keys)}')

    # initialize model
    device = "cpu"
    if 0 < args.ngpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'CUDA available: {torch.cuda.is_available()} {device}')

    train_dataset = ds.MolDataset([k for k, v in train_keys], [v for k, v in train_keys], args.data_fpath)
    test_dataset = ds.MolDataset([k for k, v in train_keys], [v for k, v in test_keys], args.data_fpath)

    N_atom_features = train_dataset[0]['H'].shape[1]//2
    args.N_atom_features = N_atom_features

    train_dataloader = DataLoader(train_dataset, args.batch_size, \
        shuffle=True, num_workers=args.num_workers, collate_fn=ds.collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=True, num_workers=args.num_workers, collate_fn=ds.collate_fn)

    model = gnn(args)
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = utils.initialize_model(model, device)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loss function
    # loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()

    best_train_loss, best_val_loss = np.inf, np.inf

    for epoch in range(num_epochs):
        st = time.time()
        #collect losses of each iteration
        train_losses = [] 
        test_losses = [] 

        #collect true label of each iteration
        train_true = []
        test_true = []
        
        #collect predicted label of each iteration
        train_pred = []
        test_pred = []
        
        model.train()
        for sample in train_dataloader:
            model.zero_grad()
            H, A1, A2, Y, V, keys, _ = sample 
            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)
            
            #train neural network
            pred = model.train_model((H, A1, A2, V))

            loss = loss_fn(pred, Y) 
            loss.backward()
            optimizer.step()
            
            #collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(Y.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())
        
        model.eval()
        for sample in test_dataloader:
            model.zero_grad()
            H, A1, A2, Y, V, keys, _ = sample 
            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)
            
            #train neural network
            pred = model.train_model((H, A1, A2, V))

            loss = loss_fn(pred, Y) 
            
            #collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            
        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))
        end = time.time()
        lapse = end - st

        best_train_loss = train_losses if train_losses < best_train_loss else best_train_loss

        if test_losses < best_val_loss:
            best_val_loss = test_losses
            torch.save(model.state_dict(), f'{save_dir}/save_{epoch}.pt')
            torch.save(model.state_dict(), f'{save_dir}/model_weights.pt')

        ls = [f'Epoch: {epoch}',
              f'Lapse: {lapse:.1f}s',
              f'Losses: ({train_losses:.3f}, {test_losses:.3f})',
              f'Best: ({best_train_loss:.3f}, {best_val_loss:.3f})']
        print('\t'.join(ls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", '-l', help="learning rate", type=float, default=0.0001)
    parser.add_argument("--epoch", '-E', help="epoch", type=int, default=1000)
    parser.add_argument("--ngpu", '-g', help="number of gpu", type=int, default=1)
    parser.add_argument("--batch_size", '-B', help="batch_size", type=int, default=8)
    parser.add_argument("--num_workers", '-n', help="number of workers", type=int, default=0)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
    parser.add_argument("--data_fpath", '-d', help="file path of dude data", type=str, default='data_select/')
    parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default='./save/')
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=4.461085466198279)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=0.19818493842903845)
    parser.add_argument("--dropout_rate", '-D', help="dropout_rate", type=float, default=0.3)
    parser.add_argument("--train_keys", '-T', help="train keys", type=str, default='keys/train_keys.txt')
    parser.add_argument("--test_keys", '-t', help="test keys", type=str, default='keys/val_keys.txt')
    parser.add_argument('--clear_cache', '-0', help='clear cache directory', action='store_true')
    parser.add_argument('--dataset_version', '-v', help='dataset version', type=int, default=2)
    parser.add_argument('--random_stratify', '-R', help='random stratify', action='store_true')
    args = parser.parse_args()
    if args.dataset_version == 1:
        import dataset as ds
    elif args.dataset_version == 2:
        import dataset2 as ds
    else:
        raise Exception('Wrong dataset version')
    print(args)
    main(args)
