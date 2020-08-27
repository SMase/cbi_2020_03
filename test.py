import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import numpy as np
from gnn import gnn
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import argparse, itertools
from saliency import VanillaGrad

def yyplot(y_obs, y_pred):
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    plt.savefig('figure.png')

parser = argparse.ArgumentParser()
parser.add_argument("--ngpu", '-g', help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", '-B', help="batch_size", type=int, default=8)
parser.add_argument("--num_workers", help="number of workers", type=int, default=7)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
parser.add_argument("--data_fpath", '-d', help="file path of dude data", type=str, default='data_diff/')
parser.add_argument("--save_dir", '-S', help="save directory of model parameter", type=str, default='./save/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=4.461085466198279)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=0.19818493842903845)
parser.add_argument("--dropout_rate", '-D', help="dropout_rate", type=float, default=0.3)
parser.add_argument("--test_keys", '-t', help="test keys", type=str, default='keys/test_keys.txt')
parser.add_argument("--map_path", '-P', help="savepath for saliency map", type=str, default='')
parser.add_argument("--map_th_max", help="max th value for saliency map", type=int, default=20)
parser.add_argument("--map_th_min", help="min th value for saliency map", type=int, default=-20)
parser.add_argument('--dataset_version', '-v', help='dataset version', type=int, default=2)
args = parser.parse_args()
if args.dataset_version == 1:
    import dataset as ds
elif args.dataset_version == 2:
    import dataset2 as ds
else:
    raise Exception('Wrong dataset version')

print(args)

prm = edict()
prm["n_graph_layer"] = args.n_graph_layer
prm["d_graph_layer"] = args.d_graph_layer
prm["n_FC_layer"] = args.n_FC_layer
prm["d_FC_layer"] = args.d_FC_layer
prm["dropout_rate"] = args.dropout_rate
prm["initial_mu"] = args.initial_mu
prm["initial_dev"] = args.initial_dev

test_keys = []
test_pkd = []
for line in open(args.test_keys):
    it = line.rstrip().split('\t')
    pdb_code, ligand_name, year, value = it[0], it[1], int(it[2]), float(it[3])
    test_keys.append(pdb_code)
    test_pkd.append(value)

dataset_path = args.data_fpath
test_dataset = ds.MolDataset(test_keys, test_pkd, dataset_path)

N_atom_features = test_dataset[0]['H'].shape[1]//2
args.N_atom_features = N_atom_features

prm['N_atom_features'] = args.N_atom_features

model_path = args.save_dir + "model_weights.pt"
model = gnn(prm)
model.load_state_dict(torch.load(model_path))

test_dataloader = DataLoader(test_dataset, 8, shuffle=True, num_workers=8, collate_fn=ds.collate_fn)

test_true = []
test_pred = []
test_label = []

saliency_list = []
n_atom_list = []

model.eval()
for i_batch, sample in enumerate(test_dataloader):
    model.zero_grad()
    H, A1, A2, Y, V, keys, n_atom = sample

    #train neural network
    embed = model.embede(H)
    model.zero_grad()
    pred = model.test_model((embed, A1, A2, V))

    out = torch.sum(pred)
    embed.retain_grad()
    out.backward()
    saliency = embed.grad.clone()
    saliency *= embed.data.clone()

    #collect loss, true label and predicted label
    test_true.append(Y.data.cpu().numpy())
    test_pred.append(pred.data.cpu().numpy())
    test_label.append(keys)
    saliency_list.append(saliency)
    n_atom_list.append(n_atom)

test_pred = np.concatenate(np.array(test_pred), 0)
test_true = np.concatenate(np.array(test_true), 0)

rmse = mean_squared_error(test_true, test_pred)**0.5
mae = mean_absolute_error(test_true, test_pred)
r2_p = stats.pearsonr(test_true, test_pred)
print("rmse: {}\nmae: {}\nr2: {}".format(rmse, mae, r2_p[0]))
yyplot(test_true, test_pred)

pdb_list = list(itertools.chain.from_iterable(test_label))
df = pd.DataFrame({
    'PDB': pdb_list,
    'pKd': list(test_true),
    'predicted': list(test_pred)
})

df.round(3).to_csv("result.tsv", sep="\t")

v_grad = VanillaGrad(th_max=args.map_th_max, th_min=args.map_th_min)
if args.map_path != '':
    v_grad.save_saliency_map(args, n_atom_list, saliency_list, pdb_list)
