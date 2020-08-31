run:
	python train.py -R -l 0.0001 -D 0.2 -E 30 -g 0 -B 64 -v 2 -d cbidata -T keys/keys_select -t keys/keys_2018
	python test.py -g 0 -v 2 -d cbidata -t train.local.key -S save
	mv result.tsv train.tsv
	mv figure.png train.png
	python test.py -g 0 -v 2 -d cbidata -t test.local.key -S save
	mv result.tsv test.tsv
	mv figure.png test.png
