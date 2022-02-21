import numpy as np
import pickle
import random
import os
import sys
import glob
import json
import math
import shutil

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger

import modules.preprocess as preprocess
import modules.net as net
import modules.data as data
import modules.loss as loss

def process_training(abs_path, config):

    # make sure the output folder exists
    data_path_train = abs_path+"/"+config['data']['folder']['train']+"/"
    data_path_test = abs_path+"/"+config['data']['folder']['test']+"/"
    output_path = abs_path+"/"+config['training']['output_folder']+"/"
    if 'load_folder' in config['training']:
        load_path = abs_path+"/"+config['training']['load_folder']+"/"
    else:
        load_path = output_path

    if not os.path.exists(output_path):
        # os.mkdir(output_path)
        os.makedirs(output_path, exist_ok=True)
        os.chmod(output_path, mode=0o755)

    print('data_path_train: {}'.format(data_path_train))
    print('data_path_test: {}'.format(data_path_test))
    print('output_path: {}'.format(output_path))

    # create a backup in case we continue the training
    if 'backup' in config['training'] and config['training']['backup'] == True:
        i=0
        backup_path = load_path + config['training']['filename_base']+'_bu_'+str(i)+'/'
        while os.path.exists(backup_path):
            i += 1
            backup_path = load_path + config['training']['filename_base']+'bu_'+str(i)+'/'
        os.makedirs(backup_path, exist_ok=True)
        os.chmod(backup_path, mode=0o755)
        for filename in glob.glob(os.path.join(load_path, config['training']['filename_base']+'*.*')):
            shutil.copy(filename, backup_path)

    # tensorflow gpu settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print('GPUs:')
            print(gpus)
            if config['training']['gpu'] == 'all':
                tf.config.experimental.set_visible_devices(gpus[:], 'GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_visible_devices(gpus[config['training']['gpu']], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[config['training']['gpu']], True)
        except RuntimeError as e:
            print(e)

    # make file/folder based train/validation split (and dump the file list to a json file) or
    # read a config list from the config file
    train_files, val_files = data.get_files(data_path_train, data_path_test, config['data']['input_file_prefixes'])
    files = {}
    files['train_files'] = train_files
    files['val_files'] = val_files
    print("Loaded and randomized training and validation files from path")
    with open(output_path+config['training']['filename_base']+'_files.json', 'w') as fp:
        json.dump(files, fp, indent=4)
        print("Saved training and validation files as json")

    if 'activation_function' not in config['model']:
        config['model']['activation_function'] = 'relu'

    if 'activation_function_out' not in config['model']:
        config['model']['activation_function_out'] = 'relu'
    
    input_shapes = []
    for i in range(len(config['data']['input_sizes'])):
        input_shapes.append((None, None, config['data']['input_sizes'][i]))
    
    print(f"input_sizes = {config['data']['input_sizes']}")
    print(f"inputs_dangling = {config['model']['inputs_dangling']}")
    print(f"num_input_filters = {config['model']['num_input_filters']}")

    # get the tensorflow model and layers for loss function
    [model, output_crop, model_output_layer, model_label_layer] = net.get_u_p_net_model(input_shapes = input_shapes,
                                                inputs_dangling             = config['model']['inputs_dangling'],
                                                label_shape_for_loss        = config['model']['label_shape_for_loss'],
                                                input_ids_last_timestep     = config['model']['input_ids_last_timestep'],
                                                num_output_frames           = config['data']['output_size'],
                                                num_input_filters           = config['model']['num_input_filters'],
                                                conv_kernel_sizes           = config['model']['conv_kernel_sizes'],
                                                pooling_size                = config['model']['pooling_size'],
                                                depth                       = config['model']['depth'],
                                                concatenate_inputs          = config['model']['concatenate_inputs'],
                                                padding                     = config['model']['padding'],
                                                skip_connection_handling    = config['model']['skip_connection_handling'],
                                                pooling_handling            = config['model']['pooling_handling'],
                                                normalization_handling      = config['model']['normalization_handling'],
                                                domain_injection            = config['model']['domain_injection'],
                                                domain_injection_ids        = config['model']['domain_injection_ids'],
                                                activation_function         = config['model']['activation_function'],
                                                activation_function_out     = config['model']['activation_function_out']
                                                )

    if 'weights_from_file' in config['model'] and config['model']['weights_from_file'] == True:
        model.load_weights(load_path+config['training']['filename_base'] + "_final.hdf5") 
        print('Loaded weights from: ' + config['training']['filename_base'] + "_final.hdf5")
    else:
        print('Filename base: ' + config['training']['filename_base'])

    # print the automatically determined output crop (due to valid convolutions)
    config['data']['output_crop'] = output_crop
    print('output_crop: {}'.format(output_crop))
    print('activation_function: '+ config['model']['activation_function'])
    print('activation_function_out: '+ config['model']['activation_function_out'])

    # print model summary
    model.summary()


    # save the model architecture as png file
    model_img = output_path+config['training']['filename_base']+'_architecture.png'
    print(model_img)
    tf.keras.utils.plot_model(model, to_file=model_img, show_shapes=True, show_layer_names=True,)


    # get layers for loss functions
    model_input_layer_domain_mask = model.inputs[config['training']['domain_mask']]
    
    if 'learningrate' not in config['training']:
        config['training']['learningrate'] = 0.001

    if ('optimizer' not in config['training']) or (config['training']['optimizer'].lower() == 'adam'):

        if 'beta_1' not in config['training']:
            config['training']['beta_1'] = 0.9

        if 'beta_2' not in config['training']:
            config['training']['beta_2'] = 0.999

        if 'epsilon' not in config['training']:
            config['training']['epsilon'] = 1e-7

        if 'amsgrad' not in config['training']:
            config['training']['amsgrad'] = False

        if config['training']['amsgrad']:
            print('Training with Adam (amsgrad) optimizer')
        else:
            print('Training with Adam optimizer')

        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learningrate'], beta_1=config['training']['beta_1'], beta_2=config['training']['beta_2'], epsilon=config['training']['epsilon'], amsgrad=config['training']['amsgrad'] )

    elif config['training']['optimizer'].lower() == 'sgd':

        print('Training with SGD optimizer')

        if 'momentum' not in config['training']:
            config['training']['momentum'] = 0

        if 'nesterov' not in config['training']:
            config['training']['nesterov'] = False

        optimizer = tf.keras.optimizers.SGD(learning_rate=config['training']['learningrate'], momentum=config['training']['momentum'], nesterov=config['training']['nesterov'])

    # set the loss function and
    # compile the model
    run_eagerly = config['model']['run_eagerly']
    mask_crop = config['data']['output_crop']
    metrics = []
    model.add_metric(loss.mse(model_label_layer, model_output_layer), name='mse', aggregation='mean')
    model.add_metric(loss.mae(model_label_layer, model_output_layer), name='mae', aggregation='mean')
    model.add_metric(loss.rmse(model_label_layer, model_output_layer), name='rmse', aggregation='mean')
    model.add_metric(loss.me(model_label_layer, model_output_layer), name='me', aggregation='mean')
    model.add_metric(loss.mes(model_label_layer, model_output_layer), name='mes', aggregation='mean')
    model.add_metric(loss.domain_mse(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop), name='domain_mse', aggregation='mean')
    model.add_metric(loss.domain_mae(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop), name='domain_mae', aggregation='mean')
    model.add_metric(loss.domain_rmse(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop), name='domain_rmse', aggregation='mean')
    model.add_metric(loss.domain_me(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop), name='domain_me', aggregation='mean')
    model.add_metric(loss.domain_mes(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop), name='domain_mes', aggregation='mean')

    if config['training']['loss'] == 'domain_mse':
        model.add_loss(loss.domain_mse(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'domain_mae':
        model.add_loss(loss.domain_mae(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'domain_rmse':
        model.add_loss(loss.domain_rmse(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'domain_ssp':
        model.add_loss(loss.domain_ssp(model_label_layer, model_output_layer, model_input_layer_domain_mask, mask_crop))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)      
    elif config['training']['loss'] == 'mse':
        model.add_loss(loss.mse(model_label_layer, model_output_layer))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'mae':
        model.add_loss(loss.mae(model_label_layer, model_output_layer))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'rmse':
        model.add_loss(loss.rmse(model_label_layer, model_output_layer))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)
    elif config['training']['loss'] == 'ssp':
        model.add_loss(loss.ssp(model_label_layer, model_output_layer))
        model.compile(metrics=metrics, run_eagerly=run_eagerly, optimizer=optimizer)    
    else: 
        assert 0
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['patience'])

    model_checkpoint_loss_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_best_loss.hdf5',
                save_weights_only=False,
                monitor='loss',
                mode='min',
                save_best_only=True)

    model_checkpoint_loss_callback_weights_only = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_best_loss_only_weights.hdf5',
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)

    model_checkpoint_val_loss_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_best_val_loss.hdf5',
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

    model_checkpoint_val_loss_callback_weights_only = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_best_val_loss_only_weights.hdf5',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

    model_checkpoint_freq_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_epoch_{epoch:08d}.hdf5',
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                # save_freq='epoch',
                save_freq=20)

    model_checkpoint_freq_callback_weights_only = tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path+config['training']['filename_base']+'_only_weights_epoch_{epoch:08d}.hdf5',
                save_weights_only=True,
                # save_freq='epoch',
                save_freq=20)

    model_csv_logger_callback = CSVLogger(output_path+config['training']['filename_base']+'_history_log.csv', append=True)


    if config['model']['domain_injection']:
        config['data']['output_crop'] = 0


    # get data generators
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

    print('Train with loss: ' + config['training']['loss'])
    print('Train with LearningRate: ' + str(config['training']['learningrate']))

    callbacks=[early_stopping_callback,
               model_checkpoint_loss_callback,
               model_checkpoint_loss_callback_weights_only,
               model_checkpoint_val_loss_callback,
               model_checkpoint_val_loss_callback_weights_only,
               model_checkpoint_freq_callback,
               model_checkpoint_freq_callback_weights_only,
               model_csv_logger_callback]

    if ('learningrate_scheduler' in config['training']) and (config['training']['learningrate_scheduler'].lower() == 'step_decay'):
        assert 'learningrate_drop_rate' in config['training']
        assert 'learningrate_epochs_drop' in config['training']
        assert 'learningrate_scheduler_verbose' in config['training']
        def lrStepDecay(epoch, lr):
            drop_rate = config['training']['learningrate_drop_rate']
            epochs_drop = config['training']['learningrate_epochs_drop']
            return config['training']['learningrate'] * math.pow(drop_rate, math.floor(epoch/epochs_drop))
        LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lrStepDecay, verbose=config['training']['learningrate_scheduler_verbose'])
        callbacks.append(LearningRateScheduler)
    elif ('learningrate_scheduler' in config['training']) and (config['training']['learningrate_scheduler'].lower() == 'exp_decay'):
        assert 'learningrate_k' in config['training']
        assert 'learningrate_scheduler_verbose' in config['training']
        def lrExpDecay(epoch, lr):
            k = config['training']['learningrate_k']
            return config['training']['learningrate'] * math.exp(-k*epoch)
        LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lrExpDecay, verbose=config['training']['learningrate_scheduler_verbose'])
        callbacks.append(LearningRateScheduler)

    # dump config dict as json file
    with open(output_path+config['training']['filename_base']+'_config.json', 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)

    # train the model
    history = model.fit(train_data,
                        epochs=config['training']['epochs'],
                        batch_size=config['data']['batch_size'],
                        shuffle=config['training']['shuffle'],
                        validation_data=val_data,
                        verbose=config['training']['verbose'],
                        callbacks=callbacks)    
    
    model.save(output_path+config['training']['filename_base']+'_final.hdf5')
    model.save_weights(output_path+config['training']['filename_base']+'_final_only_weights.hdf5')

    history_dict = history.history

    if 'lr' in history_dict:
        history_dict.pop('lr', None)

    if 'history_from_file' in config['model'] and config['model']['history_from_file'] == True:
        print('Loading model.fit-history from file and appending latest model.fit-data.')
        with open(load_path+config['training']['filename_base']+'_history.json') as json_file:
            prev_history = json.load(json_file)
            for key in prev_history:
                print('Concatenating history[\''+key+'\']...')
                assert key in history_dict
                history_dict[key] = prev_history[key] + history_dict[key]

    history_loss = np.asarray(history_dict['loss'])
    history_val_loss = np.asarray(history_dict['val_loss'])                                      
    history_min_loss = np.amin(history_loss)
    history_min_val_loss = np.amin(history_val_loss)
    history_min_loss_epoch = np.argmin(history_loss)
    history_min_val_loss_epoch = np.argmin(history_val_loss)

    json.dump(history_dict, open(output_path+config['training']['filename_base']+'_history.json', 'w'))

    return [history_min_loss, history_min_val_loss, history_min_loss_epoch, history_min_val_loss_epoch]