from gnn import gnn
import numpy as np
import utils
import torch.nn as nn
import torch
import os
import argparse
import time
from torch.utils.data import DataLoader                                     
from dataset import MolDataset, collate_fn
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print(s)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=8)
parser.add_argument("--num_workers", help="number of workers", type=int, default=7)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
parser.add_argument("--data_fpath", help="file path of dude data", type=str, default='data_select/')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default='./save/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=4.461085466198279)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=0.19818493842903845)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.3)
parser.add_argument("--train_keys", help="train keys", type=str, default='keys/train_keys.txt')
parser.add_argument("--test_keys", help="test keys", type=str, default='keys/val_keys.txt')
args = parser.parse_args()
print(args)

# hyper parameters
num_epochs = args.epoch
lr = args.lr
ngpu = args.ngpu
batch_size = args.batch_size
data_fpath = args.data_fpath
save_dir = args.save_dir

# make save dir if it doesn't exist
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)

# read data. data is stored in format of dictionary.
# Each key has information about protein-ligand complex.
train_keys = []
train_pkd = []
for line in open(args.train_keys):
    it = line.rstrip().split('\t')
    pdb_code, ligand_name, year, value = it[0], it[1], int(it[2]), float(it[3])
    train_keys.append(pdb_code)
    train_pkd.append(value)

test_keys = []
test_pkd = []
for line in open(args.test_keys):
    it = line.rstrip().split('\t')
    pdb_code, ligand_name, year, value = it[0], it[1], int(it[2]), float(it[3])
    test_keys.append(pdb_code)
    test_pkd.append(value)

# print simple statistics about dude data and pdbbind data
print(f'Number of train data: {len(train_keys)}')
print(f'Number of test data: {len(test_keys)}')

# initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# if args.ngpu>0:
    # cmd = utils.set_cuda_visible_device(args.ngpu)
    # os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]


model = gnn(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

model = utils.initialize_model(model, device)

# train and test dataset
#train_dataset = MolDataset(train_keys, train_pkd, args.data_fpath, args.distance)
#test_dataset = MolDataset(test_keys, test_pkd, args.data_fpath, args.distance)

train_dataset = MolDataset(train_keys, train_pkd, args.data_fpath)
test_dataset = MolDataset(test_keys, test_pkd, args.data_fpath)

train_dataloader = DataLoader(train_dataset, args.batch_size, \
     shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

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
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                            Y.to(device), V.to(device)
        
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
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                          Y.to(device), V.to(device)
        
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

        # ベスト値のときのみ保存
        name = save_dir + '/save_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), name)

        name = save_dir + '/model_weights.pt'
        torch.save(model.state_dict(), name)

    print(f"Epoch: {epoch}\t{lapse:.1f}s\ttrain loss: {train_losses:.3f}\tval loss: {test_losses:.3f}\ttrain best loss: {best_train_loss:.3f}\tval best loss: {best_val_loss:.3f}")
