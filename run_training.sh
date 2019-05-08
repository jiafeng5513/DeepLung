#!/bin/bash
set -e

# python prepare.py
cd detector
maxeps=150
f=9
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/anaconda2/bin/python main.py --model dpn3d26 -b 16 --save-dir dpn3d26/training_config_$f/ --epochs $maxeps --config config_training$f
for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/anaconda2/bin/python main.py --model dpn3d26 -b 16 --resume results/dpn3d26/training_config_$f/00$i.ckpt --test 0 --save-dir dpn3d26/training_config_$f/ --config config_training$f
	elif [ $i -lt 100 ]; then
	    CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/anaconda2/bin/python main.py --model dpn3d26 -b 16 --resume results/dpn3d26/training_config_$f/0$i.ckpt --test 0 --save-dir dpn3d26/training_config_$f/ --config config_training$f
	elif [ $i -lt 1000 ]; then
	    CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/anaconda2/bin/python main.py --model dpn3d26 -b 16 --resume results/dpn3d26/training_config_$f/$i.ckpt --test 0 --save-dir dpn3d26/training_config_$f/ --config config_training$f
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/dpn3d26/training_config_$f/val$i/" ]; then
        mkdir results/dpn3d26/training_config_$f/val$i/
    fi
    mv results/dpn3d26/training_config_$f/bbox/*.npy results/dpn3d26/training_config_$f/val$i/
done 