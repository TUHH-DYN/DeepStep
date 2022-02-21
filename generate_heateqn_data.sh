#!/bin/bash

# it might take a couple of hours to complete the data generation
# run the script using: 
# nohup ./generate_heateqn_data.sh > nohup_generate_heateqn_data.txt&

# make sure to use the right conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fenics

python generate_heateqn_training_data.py
python generate_heateqn_validation_data_f.py
python generate_heateqn_validation_data_g.py
python generate_heateqn_validation_data_h.py
python generate_heateqn_validation_data_i.py
