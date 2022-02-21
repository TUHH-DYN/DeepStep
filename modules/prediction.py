import numpy as np

import json
import os
import sys
import glob

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import cv2

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

import modules.preprocess as preprocess
import modules.error_metrics
import modules.net as net

import time 


def process_prediction(abs_path, config):

    output_path = abs_path+"/"+config['prediciton']['output_folder']+"/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, mode=0o755)

    # load model
    model = load_model(abs_path+"/"+config['model']['folder'] +"/"+config['model']['filename_base'] + config['model']['filename_post_model'] + ".hdf5")

    # load model config
    model_config = None
    
    with open(abs_path+"/"+config['model']['folder'] +"/"+config['model']['filename_base']+"_config.json") as json_file:
        model_config = json.load(json_file)

    print('--------')
    num_inputs = len(model_config['data']['input_sizes'])
    
    filename_base_proc = config['data']['filename_base']
    if 'dataset_' in filename_base_proc:
        filename_base_proc = filename_base_proc.replace("dataset_", "")

    data_src = []
    for i in range(num_inputs):
        data_src.append(np.load(abs_path+"/"+config['data']['folder']+"/"+model_config['data']['input_file_prefixes'][i]+filename_base_proc+".npy"))

    print('filename_base: ' + config['data']['filename_base'])
    print('timestep_start: {}'.format(config['prediciton']['timestep_start']))
    print('timestep_end: {}'.format(config['prediciton']['timestep_start']+config['prediciton']['timesteps']+model_config['data']['input_sizes'][model_config['data']['reference_input']]-1))

    a = config['prediciton']['timestep_start']
    b = config['prediciton']['timestep_start']+model_config['data']['input_sizes'][model_config['data']['reference_input']]
    data_ts = data_src[model_config['data']['reference_input']][a:b, :, :]
    data_ts = np.expand_dims(np.transpose(data_ts, (1, 2, 0)), axis=0)

    print('--------')

    ##
    ## zero-input for first (non-predictable) steps
    ##

    input_dict = {}
    for j in range(num_inputs):
        if j == model_config['data']['reference_input']:
            input_dict['input_'+str(j)] = np.zeros_like(data_ts)
        else:
            input_dict['input_'+str(j)] = np.zeros_like(np.expand_dims(data_src[j], axis=0))
    
    if config['data']['output_crop'] <= 0:
        input_dict['label'] = np.zeros_like(data_ts[:,:,:,0])
    else:
        input_dict['label'] = np.zeros_like(data_ts[:,config['data']['output_crop']:-config['data']['output_crop'],config['data']['output_crop']:-config['data']['output_crop'],0])

    inputs_out = []
    for i in range(num_inputs):
        inputs_out.append([])
    outputs_out = []
    reference_outputs_out = []
    output_diffs_out = []

    
    print('--------')

    ##
    ## set input data for first prediction
    ##

    domain = data_src[config['data']['domain_mask_input_id']]
    no_domain = 1-domain
    
    for j in range(num_inputs):
        if j == model_config['data']['reference_input']:
            input_dict['input_'+str(j)] = data_ts.copy()
        else:
            input_dict['input_'+str(j)] = np.expand_dims(data_src[j].copy(), axis=0)
    
    print('--------')
    print('filename_base: '+str(config['data']['filename_base']))

    if config['prediciton']['compute_error_metrics']:

        normalization = False

        if model_config['model']['normalization_handling'] == 'custom_sample_norm':
            normalization = True
        elif model_config['model']['normalization_handling'] == 'custom_sample_norm_zero_symm':
            normalization = True

        error_types = ['mae', 'mse', 'rmse', 'me', 'mes', 'ssp']
        error_metrics = {}
        error_metrics['filename_base'] = str(config['data']['filename_base'])
        error_metrics['step_overall'] = []
        error_metrics['step_prediction'] = []
        error_metrics['overall'] = {}
        error_metrics['overall']['raw'] = {}
        error_metrics['overall']['raw']['mad'] = []
        if normalization:
            error_metrics['overall']['normed'] = {}
            error_metrics['overall']['normed']['norm_val'] = []
        error_metrics['domain'] = {}
        error_metrics['domain']['raw'] = {}
        error_metrics['domain']['raw']['mad'] = []
        if normalization:
            error_metrics['domain']['normed'] = {}
            error_metrics['domain']['normed']['norm_val'] = []
        for error_type in error_types:
            error_metrics['overall']['raw'][error_type] = []
            error_metrics['domain']['raw'][error_type] = []
            if normalization:
                error_metrics['overall']['normed'][error_type] = []
                error_metrics['domain']['normed'][error_type] = []

    for t in range(config['prediciton']['timesteps']):
        
        for j in range(num_inputs):
            if j == model_config['data']['reference_input']:
                inputs_out[j].append(input_dict['input_'+str(j)][0,:,:,:].copy())
            else:
                inputs_out[j].append(input_dict['input_'+str(j)][0,:,:].copy())

        reference_step = t+config['prediciton']['timestep_start']+model_config['data']['input_sizes'][model_config['data']['reference_input']]

        start = time.time()
        output = model.predict(input_dict)
        end = time.time()
        print('# {} | single step prediction (model) took {:.4E}s'.format(reference_step, end - start))

        output = np.squeeze(output)
        output_postproc = output

        if config['prediciton']['compute_error_metrics']:

            reference_output = data_src[model_config['data']['reference_input']][reference_step, :, :].copy()

            normalization = True

            norm_val_from_input = max(abs(np.max(input_dict['input_'+str(model_config['data']['reference_input'])])), abs(np.min(input_dict['input_'+str(model_config['data']['reference_input'])])))
            if model_config['model']['normalization_handling'] == 'custom_sample_norm':
                output_postproc_normalize = 0.5 + output_postproc/(norm_val_from_input*2.0)
                reference_output_normalize = 0.5 + reference_output/(norm_val_from_input*2.0)
            elif model_config['model']['normalization_handling'] == 'custom_sample_norm_zero_symm':
                output_postproc_normalize = output_postproc/(norm_val_from_input*2.0)
                reference_output_normalize = reference_output/(norm_val_from_input*2.0)
            else:
                normalization = False
                # assert 0

            output_postproc_raw = output_postproc
            reference_output_raw = reference_output
            
            error_metrics['step_overall'].append(reference_step-1)
            error_metrics['step_prediction'].append(t+1)
            
            if normalization:
                error_metrics['overall']['normed']['norm_val'].append(norm_val_from_input)
                error_metrics['domain']['normed']['norm_val'].append(norm_val_from_input)

            error_metrics['overall']['raw']['mad'].append(modules.error_metrics.mae_numpy(data_src[model_config['data']['reference_input']][reference_step-1, :, :], data_src[model_config['data']['reference_input']][reference_step, :, :]))
            error_metrics['domain']['raw']['mad'].append(modules.error_metrics.domain_mae_numpy(data_src[model_config['data']['reference_input']][reference_step-1, :, :], data_src[model_config['data']['reference_input']][reference_step, :, :], domain))

            error_metrics['overall']['raw']['mae'].append(modules.error_metrics.mae_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['mae'].append(modules.error_metrics.domain_mae_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['mae'].append(modules.error_metrics.mae_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['mae'].append(modules.error_metrics.domain_mae_numpy(output_postproc_normalize, reference_output_normalize, domain))
            
            error_metrics['overall']['raw']['mse'].append(modules.error_metrics.mse_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['mse'].append(modules.error_metrics.domain_mse_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['mse'].append(modules.error_metrics.mse_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['mse'].append(modules.error_metrics.domain_mse_numpy(output_postproc_normalize, reference_output_normalize, domain))
            
            error_metrics['overall']['raw']['rmse'].append(modules.error_metrics.rmse_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['rmse'].append(modules.error_metrics.domain_rmse_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['rmse'].append(modules.error_metrics.rmse_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['rmse'].append(modules.error_metrics.domain_rmse_numpy(output_postproc_normalize, reference_output_normalize, domain))
            
            error_metrics['overall']['raw']['me'].append(modules.error_metrics.me_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['me'].append(modules.error_metrics.domain_me_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['me'].append(modules.error_metrics.me_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['me'].append(modules.error_metrics.domain_me_numpy(output_postproc_normalize, reference_output_normalize, domain))
            
            error_metrics['overall']['raw']['mes'].append(modules.error_metrics.mes_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['mes'].append(modules.error_metrics.domain_mes_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['mes'].append(modules.error_metrics.mes_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['mes'].append(modules.error_metrics.domain_mes_numpy(output_postproc_normalize, reference_output_normalize, domain))
            
            error_metrics['overall']['raw']['ssp'].append(modules.error_metrics.ssp_numpy(output_postproc_raw, reference_output_raw))
            error_metrics['domain']['raw']['ssp'].append(modules.error_metrics.domain_ssp_numpy(output_postproc_raw, reference_output_raw, domain))
            if normalization:
                error_metrics['overall']['normed']['ssp'].append(modules.error_metrics.ssp_numpy(output_postproc_normalize, reference_output_normalize))
                error_metrics['domain']['normed']['ssp'].append(modules.error_metrics.domain_ssp_numpy(output_postproc_normalize, reference_output_normalize, domain))

            print('# {} | overall | raw    | mae: {:.4E} \t mse: {:.4E} \t rmse: {:.4E} \t ssp: {:.4E} \t me: {:.4E} \t mes: {:.4E} \t mad: {:.4E}'.format(reference_step,
                                                                                                                                                              error_metrics['overall']['raw']['mae'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['mse'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['rmse'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['ssp'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['me'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['mes'][-1],
                                                                                                                                                              error_metrics['overall']['raw']['mad'][-1]))
            print('# {} | domain  | raw    | mae: {:.4E} \t mse: {:.4E} \t rmse: {:.4E} \t ssp: {:.4E} \t me: {:.4E} \t mes: {:.4E} \t mad: {:.4E}'.format(reference_step,
                                                                                                                                                              error_metrics['domain']['raw']['mae'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['mse'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['rmse'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['ssp'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['me'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['mes'][-1],
                                                                                                                                                              error_metrics['domain']['raw']['mad'][-1]))
            if normalization:
                print('# {} | overall | normed | mae: {:.4E} \t mse: {:.4E} \t rmse: {:.4E} \t ssp: {:.4E} \t me: {:.4E} \t mes: {:.4E}'.format(reference_step,
                                                                                                                                                                error_metrics['overall']['normed']['mae'][-1],
                                                                                                                                                                error_metrics['overall']['normed']['mse'][-1],
                                                                                                                                                                error_metrics['overall']['normed']['rmse'][-1],
                                                                                                                                                                error_metrics['overall']['normed']['ssp'][-1],
                                                                                                                                                                error_metrics['overall']['normed']['me'][-1],
                                                                                                                                                                error_metrics['overall']['normed']['mes'][-1]))

                print('# {} | domain  | normed | mae: {:.4E} \t mse: {:.4E} \t rmse: {:.4E} \t ssp: {:.4E} \t me: {:.4E} \t mes: {:.4E} \t'.format(reference_step,
                                                                                                                                                                error_metrics['domain']['normed']['mae'][-1],
                                                                                                                                                                error_metrics['domain']['normed']['mse'][-1],
                                                                                                                                                                error_metrics['domain']['normed']['rmse'][-1],
                                                                                                                                                                error_metrics['domain']['normed']['ssp'][-1],
                                                                                                                                                                error_metrics['domain']['normed']['me'][-1],
                                                                                                                                                                error_metrics['domain']['normed']['mes'][-1]))
            print('')

            output_diff = np.abs(output_postproc-reference_output)
            output_diffs_out.append(output_diff.copy())
            reference_outputs_out.append(reference_output.copy())

        outputs_out.append(output_postproc.copy())
        
        # add output/reference to inputs
        if config['prediciton']['iterative_feedback'] == True:
            input_dict['input_'+str(model_config['data']['reference_input'])] = np.roll(input_dict['input_'+str(model_config['data']['reference_input'])], -1, axis=3)
            input_dict['input_'+str(model_config['data']['reference_input'])][0, :, :, -1] = output_postproc.copy()
        else:
            a = reference_step-model_config['data']['input_sizes'][model_config['data']['reference_input']]+1
            b = reference_step+1
            input_dict['input_'+str(model_config['data']['reference_input'])] = data_src[model_config['data']['reference_input']][a:b, :, :].copy()
            input_dict['input_'+str(model_config['data']['reference_input'])] = np.expand_dims(np.transpose(input_dict['input_'+str(model_config['data']['reference_input'])], (1, 2, 0)), axis=0)

    if config['prediciton']['compute_error_metrics']:
        for error_type in error_types:
            error_metrics['overall']['raw'][error_type+'_avg'] = np.mean(error_metrics['overall']['raw'][error_type])
            error_metrics['domain']['raw'][error_type+'_avg'] = np.mean(error_metrics['domain']['raw'][error_type])
            if normalization:
                error_metrics['overall']['normed'][error_type+'_avg'] = np.mean(error_metrics['overall']['normed'][error_type])
                error_metrics['domain']['normed'][error_type+'_avg'] = np.mean(error_metrics['domain']['normed'][error_type])
        
        error_metrics['overall']['raw']['mad_avg'] = np.mean(error_metrics['overall']['raw']['mad'])
        error_metrics['domain']['raw']['mad_avg'] = np.mean(error_metrics['domain']['raw']['mad'])

        print('# Averages')
        print('# {} | overall | raw    | mae_avg: {:.4E} \t mse_avg: {:.4E} \t rmse_avg: {:.4E} \t ssp_avg: {:.4E} \t me_avg: {:.4E} \t mes_avg: {:.4E} \t mad_avg: {:.4E}'.format(reference_step,
                                                                                                                                                        error_metrics['overall']['raw']['mae_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['mse_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['rmse_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['ssp_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['me_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['mes_avg'],
                                                                                                                                                        error_metrics['overall']['raw']['mad_avg']))
        if normalization:
            print('# {} | overall | normed | mae_avg: {:.4E} \t mse_avg: {:.4E} \t rmse_avg: {:.4E} \t ssp_avg: {:.4E} \t me_avg: {:.4E} \t mes_avg: {:.4E}'.format(reference_step,
                                                                                                                                                            error_metrics['overall']['normed']['mae_avg'],
                                                                                                                                                            error_metrics['overall']['normed']['mse_avg'],
                                                                                                                                                            error_metrics['overall']['normed']['rmse_avg'],
                                                                                                                                                            error_metrics['overall']['normed']['ssp_avg'],
                                                                                                                                                            error_metrics['overall']['normed']['me_avg'],
                                                                                                                                                            error_metrics['overall']['normed']['mes_avg']))

        print('# {} | domain  | raw    | mae_avg: {:.4E} \t mse_avg: {:.4E} \t rmse_avg: {:.4E} \t ssp_avg: {:.4E} \t me_avg: {:.4E} \t mes_avg: {:.4E} \t mad_avg: {:.4E}'.format(reference_step,
                                                                                                                                                        error_metrics['domain']['raw']['mae_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['mse_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['rmse_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['ssp_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['me_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['mes_avg'],
                                                                                                                                                        error_metrics['domain']['raw']['mad_avg']))
        if normalization:
            print('# {} | domain  | normed | mae_avg: {:.4E} \t mse_avg: {:.4E} \t rmse_avg: {:.4E} \t ssp_avg: {:.4E} \t me_avg: {:.4E} \t mes_avg: {:.4E}'.format(reference_step,
                                                                                                                                                            error_metrics['domain']['normed']['mae_avg'],
                                                                                                                                                            error_metrics['domain']['normed']['mse_avg'],
                                                                                                                                                            error_metrics['domain']['normed']['rmse_avg'],
                                                                                                                                                            error_metrics['domain']['normed']['ssp_avg'],
                                                                                                                                                            error_metrics['domain']['normed']['me_avg'],
                                                                                                                                                            error_metrics['domain']['normed']['mes_avg']))
            
        print('# Longterm')
        print('# {} | overall | raw    | mae[-1]: {:.4E} \t mse[-1]: {:.4E} \t rmse[-1]: {:.4E} \t ssp[-1]: {:.4E} \t me[-1]: {:.4E} \t mes[-1]: {:.4E} \t mad[-1]: {:.4E}'.format(reference_step,
                                                                                                                                                        error_metrics['overall']['raw']['mae'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['mse'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['rmse'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['ssp'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['me'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['mes'][-1],
                                                                                                                                                        error_metrics['overall']['raw']['mad'][-1]))
        if normalization:
            print('# {} | overall | normed | mae[-1]: {:.4E} \t mse[-1]: {:.4E} \t rmse[-1]: {:.4E} \t ssp[-1]: {:.4E} \t me[-1]: {:.4E} \t mes[-1]: {:.4E}'.format(reference_step,
                                                                                                                                                            error_metrics['overall']['normed']['mae'][-1],
                                                                                                                                                            error_metrics['overall']['normed']['mse'][-1],
                                                                                                                                                            error_metrics['overall']['normed']['rmse'][-1],
                                                                                                                                                            error_metrics['overall']['normed']['ssp'][-1],
                                                                                                                                                            error_metrics['overall']['normed']['me'][-1],
                                                                                                                                                            error_metrics['overall']['normed']['mes'][-1]))

        print('# {} | domain  | raw    | mae[-1]: {:.4E} \t mse[-1]: {:.4E} \t rmse[-1]: {:.4E} \t ssp[-1]: {:.4E} \t me[-1]: {:.4E} \t mes[-1]: {:.4E} \t mad[-1]: {:.4E}'.format(reference_step,
                                                                                                                                                        error_metrics['domain']['raw']['mae'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['mse'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['rmse'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['ssp'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['me'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['mes'][-1],
                                                                                                                                                        error_metrics['domain']['raw']['mad'][-1]))
        if normalization:
            print('# {} | domain  | normed | mae[-1]: {:.4E} \t mse[-1]: {:.4E} \t rmse[-1]: {:.4E} \t ssp[-1]: {:.4E} \t me[-1]: {:.4E} \t mes[-1]: {:.4E}'.format(reference_step,
                                                                                                                                                            error_metrics['domain']['normed']['mae'][-1],
                                                                                                                                                            error_metrics['domain']['normed']['mse'][-1],
                                                                                                                                                            error_metrics['domain']['normed']['rmse'][-1],
                                                                                                                                                            error_metrics['domain']['normed']['ssp'][-1],
                                                                                                                                                            error_metrics['domain']['normed']['me'][-1],
                                                                                                                                                            error_metrics['domain']['normed']['mes'][-1]))

    if config['prediciton']['save_input_fields']:
        for j in range(num_inputs):
            np.save(output_path+config['prediciton']['filename_base']+'_inputs_'+str(j)+'_out.npy', inputs_out[j])

    if config['prediciton']['save_predicted_fields']:
        np.save(output_path+config['prediciton']['filename_base']+'_outputs.npy', outputs_out)
    
    if config['prediciton']['save_reference_fields']:
        np.save(output_path+config['prediciton']['filename_base']+'_reference_outputs.npy', reference_outputs_out)

    if config['prediciton']['save_difference']:
        np.save(output_path+config['prediciton']['filename_base']+'_output_diffs.npy', output_diffs_out)

    if config['prediciton']['save_config']:
        with open(output_path+config['prediciton']['filename_base']+'_config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)

    if config['prediciton']['compute_error_metrics'] and config['prediciton']['save_error_metrics']:
        with open(output_path+config['prediciton']['filename_base']+'_error_metrics.json', 'w') as fp:
            json.dump(error_metrics, fp, sort_keys=True, indent=4)
        
    #
    # save images as video
    #

    if config['prediciton']['export_video'] or config['prediciton']['export_images']:

        if not os.path.exists(output_path+config['prediciton']['filename_base']+'_img/'):
            os.mkdir(output_path+config['prediciton']['filename_base']+'_img/')
            os.chmod(output_path+config['prediciton']['filename_base']+'_img/', mode=0o755)

        assert len(reference_outputs_out) == len(outputs_out)

        reference_outputs_out =  np.asarray(reference_outputs_out)
        outputs_out =  np.asarray(outputs_out)

        for i in range(len(outputs_out)):

            assert reference_outputs_out.shape[2] == outputs_out.shape[2]
            assert reference_outputs_out.shape[1] == outputs_out.shape[1]
            aspect = reference_outputs_out.shape[2]/reference_outputs_out.shape[1]

            # 2d Plot
            # surf_ref = go.Heatmap(z=reference_outputs_out[i,:,:], zmin=config['prediciton']['zLimitScaleMin'], zmax=config['prediciton']['zLimitScaleMax'])

            # 3d Plot
            surf_ref = go.Surface(z=reference_outputs_out[i,:,:], cmin=config['prediciton']['zLimitScaleMin'], cmax=config['prediciton']['zLimitScaleMax'])
            fig_ref = go.Figure(data=[surf_ref])
            fig_ref.update_layout(title='Reference')

            fig_ref.update_layout(scene_aspectmode='manual',
                                  scene_aspectratio=dict(x=aspect, y=1, z=1))
            fig_ref.update_layout(scene = dict(
                            xaxis = dict(nticks=4, range=[0,reference_outputs_out.shape[2]],),
                            yaxis = dict(nticks=4, range=[0,reference_outputs_out.shape[1]],),
                            zaxis = dict(nticks=4, range=[config['prediciton']['zLimitMin'],config['prediciton']['zLimitMax']],),),)
            if config['prediciton']['camera_projection'] == "orthographic":
                fig_ref.layout.scene.camera.projection.type = "orthographic"

            fig_ref.update_layout(scene_camera=config['prediciton']['camera'])
            fig_ref.write_image(output_path+config['prediciton']['filename_base']+'_img/{}_ref.png'.format(i), scale=config['prediciton']['export_image_scale'] )
            
            # 2d Plot
            # surf_pred = go.Heatmap(z=outputs_out[i,:,:], zmin=config['prediciton']['zLimitMin'], zmax=config['prediciton']['zLimitMax'])

            # 3d Plot
            surf_pred = go.Surface(z=outputs_out[i,:,:], cmin=config['prediciton']['zLimitMin'], cmax=config['prediciton']['zLimitMax'])
            fig_pred = go.Figure(data=[surf_pred])
            fig_pred.update_layout(title='Prediction')
            fig_pred.update_layout(scene_aspectmode='manual',
                                  scene_aspectratio=dict(x=aspect, y=1, z=1))
            fig_pred.update_layout(scene = dict(
                            xaxis = dict(nticks=4, range=[0,outputs_out.shape[2]],),
                            yaxis = dict(nticks=4, range=[0,outputs_out.shape[1]],),
                            zaxis = dict(nticks=4, range=[config['prediciton']['zLimitMin'],config['prediciton']['zLimitMax']],),),)
            if config['prediciton']['camera_projection'] == "orthographic":
                fig_pred.layout.scene.camera.projection.type = "orthographic"

            fig_pred.update_layout(scene_camera=config['prediciton']['camera'])
            fig_pred.write_image(output_path+config['prediciton']['filename_base']+'_img/{}_pred.png'.format(i), scale=config['prediciton']['export_image_scale'] )

            # 2d Plot
            # surf_diff = go.Heatmap(z=np.abs(reference_outputs_out[i,:,:]-outputs_out[i,:,:]), zmin=config['prediciton']['zLimitScaleErrMin'], zmax=config['prediciton']['zLimitScaleErrMax'], colorscale='Teal')

            # 3d Plot
            surf_diff = go.Surface(z=np.abs(reference_outputs_out[i,:,:]-outputs_out[i,:,:]), cmin=config['prediciton']['zLimitScaleErrMin'], cmax=config['prediciton']['zLimitScaleErrMax'], colorscale='Teal')
            fig_diff = go.Figure(data=[surf_diff])
            fig_diff.update_layout(title='Absolute Difference')
            fig_diff.update_layout(scene_aspectmode='manual',
                                  scene_aspectratio=dict(x=aspect, y=1, z=1))
            fig_diff.update_layout(scene = dict(
                            xaxis = dict(nticks=4, range=[0,outputs_out.shape[2]],),
                            yaxis = dict(nticks=4, range=[0,outputs_out.shape[1]],),
                            zaxis = dict(nticks=4, range=[config['prediciton']['zLimitErrMin'],config['prediciton']['zLimitErrMax']],),),)
            if config['prediciton']['camera_projection'] == "orthographic":
                fig_diff.layout.scene.camera.projection.type = "orthographic"

            fig_diff.update_layout(scene_camera=config['prediciton']['camera'])
            fig_diff.write_image(output_path+config['prediciton']['filename_base']+'_img/{}_diff.png'.format(i), scale=config['prediciton']['export_image_scale'] )

            max_steps = len(outputs_out)-1
            fig_err = go.Figure()

            err = error_metrics[config['prediciton']['export_err_region']][config['prediciton']['export_err_class']][config['prediciton']['export_err_type']]

            fig_err.add_trace(go.Scatter(x=list(range(0,len(err[:i+1]))), y=err[:i+1], name="linear", line_shape='linear'))
            fig_err.update_layout(scene_aspectmode='manual',
                                scene_aspectratio=dict(x=aspect, y=1))
            fig_err.update_traces(mode='lines')
            fig_err.update_layout(title=config['prediciton']['export_err_type'])
            fig_err.update_layout(xaxis_range=[0,max_steps])
            fig_err.update_layout(yaxis_range=[0,max(err)*1.05])
            fig_err.write_image(output_path+config['prediciton']['filename_base']+'_img/{}_err.png'.format(i), scale=config['prediciton']['export_image_scale'] )


            image1 = Image.open(output_path+config['prediciton']['filename_base']+'_img/{}_ref.png'.format(i))
            image2 = Image.open(output_path+config['prediciton']['filename_base']+'_img/{}_pred.png'.format(i))
            image3 = Image.open(output_path+config['prediciton']['filename_base']+'_img/{}_diff.png'.format(i))
            image4 = Image.open(output_path+config['prediciton']['filename_base']+'_img/{}_err.png'.format(i))

            #resize, first image
            image1_size = image1.size
            image2_size = image2.size
            image3_size = image3.size
            image4_size = image4.size

            assert image1_size[0] == image2_size[0]
            assert image1_size[0] == image3_size[0]
            assert image1_size[1] == image2_size[1]
            assert image1_size[1] == image3_size[1]

            image4 = image4.resize((image1_size[0], image1_size[1]))

            new_image = Image.new('RGB',(2*image1_size[0], 2*image1_size[1]), (250,250,250))
            new_image.paste(image1,(0,0))
            new_image.paste(image2,(image1_size[0],0))
            new_image.paste(image3,(0,image1_size[1]))
            new_image.paste(image4,(image1_size[0],image1_size[1]))
            new_image.save(output_path+config['prediciton']['filename_base']+'_img/{}_merged.png'.format(i),"PNG")

        if config['prediciton']['export_video']:

            image_folder = output_path+config['prediciton']['filename_base']+'_img/'
            video_name = output_path+config['prediciton']['filename_base']+'_video.mp4'

            images = [img for img in os.listdir(image_folder) if img.endswith("_merged.png")]
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            print('Video height: {}'.format(height))
            print('Video width: {}'.format(width))
            
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter(video_name, fourcc, config['prediciton']['fps'], (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            cv2.destroyAllWindows()
            video.release()

    print(' ')
    print('# Filename base:')
    print('# | ' + config['prediciton']['filename_base'])
    
    if config['prediciton']['compute_error_metrics']:
        return error_metrics
    else:
        return []