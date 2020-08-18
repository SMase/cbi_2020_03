# cbi_2020
CBI講演会(2020.3)共有用

# 韓国版PotentialNet
https://github.com/jaechanglim/GNN_DTI

```
tar jxvf bidata_gnn_dti.tar.bz2
mkdir -p cbidata
mv 2018/* select/* cbidata/
rm -rf 2018 select

python -u train.py --dropout_rate=0.3 --epoch=1000 --ngpu=0 --batch_size=256 --num_workers=0 --train_keys keys/keys_select --test_keys keys/keys_2018 --data_fpath cbidata
python test.py --save_dir save/ --data_fpath cbidata --test_keys keys/keys_2018 --ngpu=0
```

## Environment
Please copy .env_sample to .env and edit for fitting to your environment.

## Container
Environnement created by docker & docker-compose. Please exec below commands, according to your purpose.
```
# To up the docker network and all containers.
$ docker-compose up -d

# To stop/restart/rm the specific container.
$ docker-compose <stop/restart/rm> <service name>

# To build the specific image.
$ docker-compose build <service name>

# To destroy docker-network and all containers.
$ docker-compose down

# To destroy docker-network, all containers and all images.
$ docker-compose down --rmi all
```

## Run
```
# To up the docker container
$ docker-compose up -d

# To enter into the container
$ docker-compose exec gnn_dti bash

# To set conda env
(base) ~ $ conda activate dti

# To run 
(dti) ~ $ python -u train.py --dropout_rate=0.3 --epoch=1000 --ngpu=1 --batch_size=256 --num_workers=0

```
