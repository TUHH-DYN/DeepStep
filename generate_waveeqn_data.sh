#!/bin/bash

# it might take a couple of hours to complete the data generation
# run the script using: 
# nohup ./generate_waveeqn_data.sh > nohup_generate_waveeqn_data.txt&

# make sure to use the right conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fenics

python generate_waveeqn_training_data_a_to_e.py
python generate_waveeqn_validation_data_f.py
python generate_waveeqn_validation_data_g.py
python generate_waveeqn_validation_data_h.py
python generate_waveeqn_validation_data_i.py
