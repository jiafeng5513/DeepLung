#!/usr/bin/env bash
# test detector
set -e
cd detector
f=9
maxeps=1
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/anaconda2/bin/python main.py --model dpn3d26 -b 16 --save-dir dpn3d26/test_config_$f/ --test 1 --resume results/dpn3d26/training_config_$f/150.ckpt --epochs $maxeps --config config_training$f