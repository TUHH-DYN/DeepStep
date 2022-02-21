import os
import sys
import json
from datetime import datetime
import numpy as np
from numpy.lib.function_base import append

from modules.prediction import process_prediction

##
## Use a trained model to predict future timesteps
##


validation_case = 'f' # rect. domain + rect. boundary + rect. inhomog.
# validation_case = 'g' # rect. domain + left third inhomog.
# validation_case = 'h' # rect. domain + rounded edges
# validation_case = 'i' # rect. domain + zigzag edges 


model_config = {'name': 'u_p_net_64',
                'filename_base': 'model_u_p_net_64_after_350epochs/model_u_p_net_64',
                'trained_epochs': 350}

# model_config = {'name': 'u_p_net_x2',
#                 'filename_base': 'model_u_p_net_x2_after_350epochs/model_u_p_net_x2',
#                 'trained_epochs': 350}

# model_config = {'name': 'u_net_96',
#                 'filename_base': 'model_u_net_96_after_350epochs/model_u_net_96',
#                 'trained_epochs': 350}

# model_config = {'name': 'u_net_x2',
#                 'filename_base': 'model_u_net_x2_after_350epochs/model_u_net_x2',
#                 'trained_epochs': 350}

# model_config = {'name': 'u_net_noskip_106',
#                 'filename_base': 'model_u_net_noskip_106_after_350epochs/model_u_net_noskip_106',
#                 'trained_epochs': 350}

# model_config = {'name': 'u_net_noskip_x2',
#                 'filename_base': 'model_u_net_noskip_x2_after_350epochs/model_u_net_noskip_x2',
#                 'trained_epochs': 350}

now = datetime.now()
path_name = os.path.dirname(sys.argv[0])   
abs_path = os.path.abspath(path_name) 

config = {}

config['prediciton'] = {}

# config['prediciton']['timestep_start']        = 0
config['prediciton']['timestep_start']        = 10

# config['prediciton']['timesteps']             = 3
# config['prediciton']['timesteps']             = 20
config['prediciton']['timesteps']             = 5
# config['prediciton']['timesteps']             = 200
# config['prediciton']['timesteps']             = 797   

config['prediciton']['iterative_feedback']    = False 

config['prediciton']['compute_error_metrics'] = True
config['prediciton']['save_input_fields']     = True
config['prediciton']['save_predicted_fields'] = True
config['prediciton']['save_reference_fields'] = True
config['prediciton']['save_difference']       = True
config['prediciton']['save_config']           = True
config['prediciton']['save_error_metrics']    = True
config['prediciton']['export_images']         = False
config['prediciton']['export_video']          = False

config['prediciton']['output_folder']         = 'data/prediction_waveequation'

config['prediciton']['export_image_scale'] = 1.0
config['prediciton']['camera_projection'] = "perspective" # perspective or orthographic
config['prediciton']['export_err_region'] = 'domain'
config['prediciton']['export_err_class'] = 'raw'
config['prediciton']['export_err_type'] = 'rmse'
config['prediciton']['zLimitMin'] = -0.1
config['prediciton']['zLimitMax'] = 0.1
config['prediciton']['fps'] = 10
config['prediciton']['zLimitScaleMin'] = config['prediciton']['zLimitMin']
config['prediciton']['zLimitScaleMax'] = config['prediciton']['zLimitMax']
config['prediciton']['zLimitErrMin'] = -0.02
config['prediciton']['zLimitErrMax'] = 0.02
config['prediciton']['zLimitScaleErrMin'] = 0
config['prediciton']['zLimitScaleErrMax'] = 0.02
config['prediciton']['camera'] = dict(
                # up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0.0, z=1.75)
            )
config['prediciton']['camera'] = dict(
                                    up=dict(x=0, y=0, z=1),
                                    center=dict(x=0, y=0, z=0),
                                    eye=dict(x=1.25, y=1.25, z=1.25)
                                )
         

config['model'] = {}
config['model']['name'] = model_config['name']
config['model']['filename_base'] = model_config['filename_base']
config['model']['trained_epochs'] = str(model_config['trained_epochs'])
config['model']['folder'] = 'data/model_waveequation'
config['model']['filename_post_model'] = '_best_loss'
config['model']['filename_post_weights'] = '_best_loss_only_weights'
# config['model']['filename_post_model'] = '_best_val_loss'
# config['model']['filename_post_weights'] = '_best_val_loss_only_weights'
# config['model']['filename_post_model'] = '_final'
# config['model']['filename_post_weights'] = '_final_only_weights'


config['data'] = {}
config['data']['folder'] = 'data/datasets_test_waveequation'
config['data']['domain_mask_input_id'] = 1
config['data']['output_crop'] = 0
config['data']['filename_base'] =  "dataset_validation_"+validation_case 

model_filename_str = config['model']['filename_base'].split('/')
model_filename_str = model_filename_str[-1]
if not config['prediciton']['iterative_feedback']:
    config['prediciton']['filename_base'] = 'prediction_'+validation_case+'_{}_{}_no_feedback_{}_{}'.format(config['prediciton']['timestep_start'], config['prediciton']['timesteps'], model_filename_str[6:25], config['model']['filename_post_model'][1:])
else:
    config['prediciton']['filename_base'] = 'prediction_'+validation_case+'_{}_{}_{}_{}'.format(config['prediciton']['timestep_start'], config['prediciton']['timesteps'], model_filename_str[6:25], config['model']['filename_post_model'][1:])


error_metrics_all_item = process_prediction( abs_path, config)