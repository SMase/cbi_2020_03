# cbi_2020

CBI講演会(2020.3)共有用

# 韓国版PotentialNet

https://github.com/jaechanglim/GNN_DTI

# Instruction

```
(Download cbidata_gnn_dti_refined.tar.bz2 from https://drive.google.com/drive/u/0/folders/1q9YtZTTYr1ZI9Vu9ddunT2Gl16Pmy4S9)
tar jxvf cbidata_gnn_dti_refiend.tar.bz2

python train.py -l 0.0002 -D 0.3 -E 1000 -g 0 -B 64 -v 2 -d cbidata -T keys/keys_refined
python test.py -g 0 -v 2 -d cbidata -S save/ -t test.local.keys
```

