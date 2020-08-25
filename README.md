# cbi_2020

CBI講演会(2020.3)共有用

# 韓国版PotentialNet

https://github.com/jaechanglim/GNN_DTI

# Instruction

```
(Download cbidata_gnn_dti.tar.bz2 from https://drive.google.com/drive/u/0/folders/1q9YtZTTYr1ZI9Vu9ddunT2Gl16Pmy4S9)
tar jxvf cbidata_gnn_dti.tar.bz2
mkdir -p cbidata
mv 2018/* select/* cbidata/
rm -rf 2018 select

python -u train.py --dropout_rate=0.3 --epoch=1000 --ngpu=0 --batch_size=256 --num_workers=0 --train_keys keys/keys_select --test_keys keys/keys_2018 --data_fpath cbidata
python test.py --save_dir save/ --data_fpath cbidata --test_keys keys/keys_2018 --ngpu=0
```

