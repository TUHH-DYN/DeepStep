#!/bin/bash

# it might take a couple of days/weeks to complete the training
# run the script using: 
# nohup ./train_waveeqn.sh > nohup_train_waveeqn.txt&

# make sure to use the right conda env
# source /home/jakobo/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf

python train_waveeqn.py 'u_p_net' 64 64 64 0
python train_continue_waveeqn.py 'u_p_net' 64 64 64 0

python train_waveeqn.py 'u_p_net' 29 58 116 0
python train_continue_waveeqn.py 'u_p_net' 29 58 116 0


python train_waveeqn.py 'u_net' 96 96 96 0
python train_continue_waveeqn.py 'u_net' 96 96 96 0

python train_waveeqn.py 'u_net' 46 92 184 0
python train_continue_waveeqn.py 'u_net' 46 92 184 0


python train_waveeqn.py 'u_net_noskip' 106 106 106 0
python train_continue_waveeqn.py 'u_net_noskip' 106 106 106 0

python train_waveeqn.py 'u_net_noskip' 48 96 192 0
python train_continue_waveeqn.py 'u_net_noskip' 48 96 192 0