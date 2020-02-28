import torch
from dataset import MolDataset, collate_fn, DTISampler
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import pickle
import numpy as np
from gnn import gnn
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

prm = edict()
prm["n_graph_layer"] = 4
prm["d_graph_layer"] = 140
prm["n_FC_layer"] = 4
prm["d_FC_layer"] = 128
prm["dropout_rate"] = 0.3
prm["initial_mu"] = 0.19818493842903845
prm["initial_dev"] = 4.461085466198279

path = "./save/model_weights.pt"
model = gnn(prm)
model.load_state_dict(torch.load(path))

with open ("./keys/test_keys.pkl", 'rb') as fp:
    test_keys = pickle.load(fp)

dataset_path = "./data/data_diff/"
test_dataset = MolDataset(test_keys, dataset_path)

test_dataloader = DataLoader(test_dataset, 10, shuffle=True, num_workers=10, collate_fn=collate_fn)

test_true = []
test_pred = []
model.eval()
for i_batch, sample in enumerate(test_dataloader):
    model.zero_grad()
    H, A1, A2, Y, V, keys = sample

    #train neural network
    pred = model.train_model((H, A1, A2, V))

    #collect loss, true label and predicted label
    test_true.append(Y.data.cpu().numpy())
    test_pred.append(pred.data.cpu().numpy())

test_pred = np.concatenate(np.array(test_pred), 0)
test_true = np.concatenate(np.array(test_true), 0)

rmse = mean_squared_error(test_true, test_pred)**0.5
mae = mean_absolute_error(test_true, test_pred)
r2_p = stats.pearsonr(test_true, test_pred)
print("rmse: {}\nmae: {}\nr2: {}".format(rmse, mae, r2_p[0]))
yyplot(test_true, test_pred)

df = pd.DataFrame({
    'pKd': list(test_true),
    'predicted': list(test_pred)
})
df.to_csv("result.tsv", sep="\t")
