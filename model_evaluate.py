import os
import sys
import json
from datetime import datetime
import numpy as np
from numpy.core import numerictypes
from numpy.lib.function_base import append
from tensorflow.python.ops.gen_batch_ops import batch

from modules.prediction import process_prediction

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow import keras

import modules.error_metrics as err_metrics
import modules.data as data
import modules.loss as loss

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

problem_type = 'heat'
# problem_type = 'wave'

path_name = os.path.dirname(sys.argv[0])   
abs_path = os.path.abspath(path_name) 
output_path = abs_path+'/data/prediction_'+problem_type+'equation/'

model_configs = []

model_configs.append({'name': 'u_p_net_64',
                'filename_base': 'model_u_p_net_64_after_350epochs/model_u_p_net_64',
                'trained_epochs': 350})

# model_configs.append({'name': 'u_p_net_x2',
#                 'filename_base': 'model_u_p_net_x2_after_350epochs/model_u_p_net_x2',
#                 'trained_epochs': 350})

model_configs.append({'name': 'u_net_96',
                'filename_base': 'model_u_net_96_after_350epochs/model_u_net_96',
                'trained_epochs': 350})

# model_configs.append({'name': 'u_net_x2',
#                 'filename_base': 'model_u_net_x2_after_350epochs/model_u_net_x2',
#                 'trained_epochs': 350})

model_configs.append({'name': 'u_net_noskip_106',
                'filename_base': 'model_u_net_noskip_106_after_350epochs/model_u_net_noskip_106',
                'trained_epochs': 350})

# model_configs.append({'name': 'u_net_noskip_x2',
#                 'filename_base': 'model_u_net_noskip_x2_after_350epochs/model_u_net_noskip_x2',
#                 'trained_epochs': 350})


# filenames_post_model = ['_best_loss', '_best_val_loss', '_final']
# filenames_post_weights = ['_best_loss_only_weights', '_best_val_loss_only_weights', '']
# filenames_post_model = ['_best_loss']
# filenames_post_weights = ['_best_loss_only_weights']
# filenames_post_model = ['_best_val_loss']
# filenames_post_weights = ['_best_val_loss_only_weights']
filenames_post_model = ['_final']
filenames_post_weights = ['_final_only_weights']

config = {}

config['data'] = {}
config['data']['folder'] = {}
config['data']['folder']['train'] = 'data/datasets_train_'+problem_type+'equation'
config['data']['folder']['test'] = 'data/datasets_test_'+problem_type+'equation'
config['data']['input_sizes'] = [3,1,1]
config['data']['output_size'] = 1

if problem_type == 'heat':
    config['data']['input_file_prefixes'] = ['field_dataset_', 'domain_surf_dataset_', 'material_dataset_']
    config['data']['augmentation'] = 'rotate' 
elif problem_type == 'wave':
    config['data']['input_file_prefixes'] = ['field_dataset_', 'domain_dataset_', 'material_dataset_']
    config['data']['augmentation'] = 'mirror_and_flip' 

# config['data']['batch_size'] = 1
config['data']['batch_size'] = 8
config['data']['file_batch_size'] = 2
config['data']['reference_input'] = 0
# config['data']['output_crop'] = 20
config['data']['output_crop'] = 0
config['data']['randomize_files'] = False

data_path_train = abs_path+"/"+config['data']['folder']['train']+"/"
data_path_test = abs_path+"/"+config['data']['folder']['test']+"/"
train_files, val_files = data.get_files(data_path_train, data_path_test, config['data']['input_file_prefixes'])


train_data, val_data = data.get_dataset_from_generators(train_files,
                                                        val_files,
                                                        config['data']['input_sizes'],
                                                        config['data']['output_size'],
                                                        config['data']['batch_size'],
                                                        config['data']['file_batch_size'],
                                                        config['data']['reference_input'],
                                                        config['data']['output_crop'],
                                                        config['data']['augmentation'],
                                                        config['data']['randomize_files'])

error_metrics_all = []

for model_config in model_configs:


    model_config['normalization_handling'] = 'custom_sample_norm'

    error_metrics_model = {}
    error_metrics_model['model_name'] = model_config['name']
    error_metrics_model['model_trained_epochs'] = model_config['trained_epochs']

    config['model'] = {}
    config['model']['name'] = model_config['name']
    config['model']['filename_base'] = model_config['filename_base']
    config['model']['trained_epochs'] = str(model_config['trained_epochs'])

    assert len(filenames_post_model) == len(filenames_post_weights)
    for fi in range(len(filenames_post_model)):


        config['model']['filename_post_model'] = filenames_post_model[fi]
        config['model']['filename_post_weights'] = filenames_post_weights[fi]

        config['model']['folder'] = 'data/model_'+problem_type+'equation'

        print('-------------------------------')
        print('-------------------------------')
        print('-------------------------------')
        print('-------------------------------')
        print('-------------------------------')

        print("dataset after applying take() method")
        
        print("loading model: " + abs_path+"/"+config['model']['folder'] +"/"+config['model']['filename_base'] + config['model']['filename_post_model'] + ".hdf5")
        model = load_model(abs_path+"/"+config['model']['folder'] +"/"+config['model']['filename_base'] + config['model']['filename_post_model'] + ".hdf5")

        counter = 0
        scores = []

        error_types = ['mae', 'mse', 'rmse', 'me', 'mes', 'ssp']
        error_metrics = {}
        error_metrics['overall'] = {}
        error_metrics['overall']['raw'] = {}
        error_metrics['overall']['normed'] = {}
        error_metrics['overall']['normed']['norm_val'] = []
        error_metrics['domain'] = {}
        error_metrics['domain']['raw'] = {}
        error_metrics['domain']['normed'] = {}
        error_metrics['domain']['normed']['norm_val'] = []
        for error_type in error_types:
            error_metrics['overall']['raw'][error_type] = []
            error_metrics['overall']['normed'][error_type] = []
            error_metrics['domain']['raw'][error_type] = []
            error_metrics['domain']['normed'][error_type] = []

        for i in val_data:

            output = model.predict(i[0])

            # print(i)

            # print('Inputs:')
            # print(i[0]['input_0'].numpy().shape)
            # print(np.mean(i[0]['input_0'].numpy()))
            # print(i[0]['input_1'].numpy().shape)
            # print(np.mean(i[0]['input_1'].numpy()))
            # print(i[0]['input_2'].numpy().shape)
            # print(np.mean(i[0]['input_2'].numpy()))
            # print('')
            # print('Label-Input:')
            # print(i[0]['label'].numpy().shape)
            # print(np.mean(i[0]['label'].numpy()))
            # print('')

            # print('Outputs:')
            # print(i[1].numpy().shape)
            # print(np.mean(i[1].numpy()))
            # print(output.shape)
            # print(np.mean(output))
            # print('')

            # print('diff(Output, Label-Input):')
            # print(np.mean(i[0]['label'].numpy()-i[1].numpy()))
            # print('')

            # print('Domain:')
            # print(i[0]['input_1'].numpy().shape)
            # print(np.unique(i[0]['input_1'].numpy()))
            # print('')

            def to_tensor(arg):
                arg = tf.convert_to_tensor(arg, dtype=tf.float32)
                return arg

            y_true_tf = to_tensor(i[1].numpy())
            y_pred_tf = to_tensor(output)
            mask_tf = to_tensor(i[0]['input_1'].numpy())

            # Norm reference and output
            def max_val_func(inp_x):
                return tf.reduce_max(tf.math.abs(inp_x), axis=(1,2,3))
            # max_val_in_0 = keras.layers.Lambda(max_val_func,)(i[0]['input_0'].numpy()) 
            max_val_in_0 = keras.layers.Lambda(max_val_func,)(i[0]['input_0']) 

            if model_config['normalization_handling'] == 'custom_sample_norm':
                def norm_func(inp_x):
                    max_val = inp_x[0]
                    inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                    inp_x_norm = tf.math.add(tf.constant(0.5), tf.math.divide(inp_x_norm, tf.math.multiply(tf.constant(2.0), max_val)))
                    return tf.transpose(inp_x_norm, [3, 1, 2, 0])
            elif model_config['normalization_handling'] == 'custom_sample_norm_zero_symm':
                def inv_norm_func(inp_x):
                    max_vall = inp_x[0]
                    inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                    inp_x_norm = tf.math.multiply(inp_x_norm, tf.math.multiply(tf.constant(2.0), max_vall))
                    return tf.transpose(inp_x_norm, [3, 1, 2, 0])
            else:
                assert 0

            y_true_norm_tf = keras.layers.Lambda(norm_func, name="lamda_norm_func")([max_val_in_0, y_true_tf]) 
            y_pred_norm_tf = keras.layers.Lambda(norm_func, name="lamda_norm_func")([max_val_in_0, y_pred_tf]) 

            error_metrics['overall']['normed']['norm_val'] += max_val_in_0.numpy().tolist()
            error_metrics['domain']['normed']['norm_val'] += max_val_in_0.numpy().tolist()

            error_metrics['overall']['raw']['mae'].append(loss.mae(y_true_tf, y_pred_tf).numpy().item())
            error_metrics['overall']['normed']['mae'].append(loss.mae(y_true_norm_tf, y_pred_norm_tf).numpy().item())
            error_metrics['domain']['raw']['mae'].append(loss.domain_mae(y_true_tf, y_pred_tf, mask_tf, 0).numpy().item())
            error_metrics['domain']['normed']['mae'].append(loss.domain_mae(y_true_norm_tf, y_pred_norm_tf, mask_tf, 0).numpy().item())

            error_metrics['overall']['raw']['mse'].append(loss.mse(y_true_tf, y_pred_tf).numpy().item())
            error_metrics['overall']['normed']['mse'].append(loss.mse(y_true_norm_tf, y_pred_norm_tf).numpy().item())
            error_metrics['domain']['raw']['mse'].append(loss.domain_mse(y_true_tf, y_pred_tf, mask_tf, 0).numpy().item())
            error_metrics['domain']['normed']['mse'].append(loss.domain_mse(y_true_norm_tf, y_pred_norm_tf, mask_tf, 0).numpy().item())

            error_metrics['overall']['raw']['rmse'].append(loss.rmse(y_true_tf, y_pred_tf).numpy().item())
            error_metrics['overall']['normed']['rmse'].append(loss.rmse(y_true_norm_tf, y_pred_norm_tf).numpy().item())
            error_metrics['domain']['raw']['rmse'].append(loss.domain_rmse(y_true_tf, y_pred_tf, mask_tf, 0).numpy().item())
            error_metrics['domain']['normed']['rmse'].append(loss.domain_rmse(y_true_norm_tf, y_pred_norm_tf, mask_tf, 0).numpy().item())

            error_metrics['overall']['raw']['me'].append(loss.me(y_true_tf, y_pred_tf).numpy().item())
            error_metrics['overall']['normed']['me'].append(loss.me(y_true_norm_tf, y_pred_norm_tf).numpy().item())
            error_metrics['domain']['raw']['me'].append(loss.domain_me(y_true_tf, y_pred_tf, mask_tf, 0).numpy().item())
            error_metrics['domain']['normed']['me'].append(loss.domain_me(y_true_norm_tf, y_pred_norm_tf, mask_tf, 0).numpy().item())

            error_metrics['overall']['raw']['mes'].append(loss.mes(y_true_tf, y_pred_tf).numpy().item())
            error_metrics['overall']['normed']['mes'].append(loss.mes(y_true_norm_tf, y_pred_norm_tf).numpy().item())
            error_metrics['domain']['raw']['mes'].append(loss.domain_mes(y_true_tf, y_pred_tf, mask_tf, 0).numpy().item())
            error_metrics['domain']['normed']['mes'].append(loss.domain_mes(y_true_norm_tf, y_pred_norm_tf, mask_tf, 0).numpy().item())

            overall_raw_ssp = []
            overall_normed_ssp = []
            domain_raw_ssp = []
            domain_normed_ssp = []

            for sspi in range(y_pred_tf.numpy().shape[0]):
                overall_raw_ssp.append(err_metrics.ssp_numpy(y_true_tf.numpy()[sspi,:,:,:], y_pred_tf.numpy()[sspi,:,:,:]))   
                overall_normed_ssp.append(err_metrics.ssp_numpy(y_true_norm_tf.numpy()[sspi,:,:,:], y_pred_norm_tf.numpy()[sspi,:,:,:]))
                domain_raw_ssp.append(err_metrics.domain_ssp_numpy(y_true_tf.numpy()[sspi,:,:,:], y_pred_tf.numpy()[sspi,:,:,:], mask_tf.numpy()[sspi,:,:,:]))   
                domain_normed_ssp.append(err_metrics.domain_ssp_numpy(y_true_norm_tf.numpy()[sspi,:,:,:], y_pred_norm_tf.numpy()[sspi,:,:,:], mask_tf.numpy()[sspi,:,:,:]))   

            error_metrics['overall']['raw']['ssp'].append(np.mean(overall_raw_ssp))
            error_metrics['overall']['normed']['ssp'].append(np.mean(overall_normed_ssp))
            error_metrics['domain']['raw']['ssp'].append(np.mean(domain_raw_ssp))
            error_metrics['domain']['normed']['ssp'].append(np.mean(domain_normed_ssp))

            print(model_config['name'] + ' ' + str(model_config['trained_epochs']) + ' ' + filenames_post_model[fi][1:] + ': ' + str(counter))
            counter += 1

            # break

        for error_type in error_types:
            error_metrics['overall']['raw'][error_type+'_avg'] = np.mean(error_metrics['overall']['raw'][error_type])
            error_metrics['overall']['normed'][error_type+'_avg'] = np.mean(error_metrics['overall']['normed'][error_type])
            error_metrics['domain']['raw'][error_type+'_avg'] = np.mean(error_metrics['domain']['raw'][error_type])
            error_metrics['domain']['normed'][error_type+'_avg'] = np.mean(error_metrics['domain']['normed'][error_type])
        
        error_metrics_model[filenames_post_model[fi][1:]] = error_metrics

    error_metrics_all.append(error_metrics_model)

    with open(output_path+'model_evaluation_error_metrics.json', 'w') as fp:
        json.dump(error_metrics_all, fp, sort_keys=True, indent=4)