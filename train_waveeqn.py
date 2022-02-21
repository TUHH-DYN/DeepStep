import sys, os
import json

from modules.training import process_training

##
## Train an u-net configuration for 300 epochs
## with a learningrate decreasing from 1E-4 to 1E-5
##

def main():

    print("Total arguments passed:", len(sys.argv))

    filter_size = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]

    num_input_filters = {
        'u_p_net': [[filter_size[0],filter_size[1],filter_size[2]],[filter_size[0],filter_size[1],filter_size[2]],[filter_size[0],filter_size[1],filter_size[2]]],
        'u_net': [[filter_size[0],filter_size[1],filter_size[2]]],
        'u_net_noskip': [[filter_size[0],filter_size[1],filter_size[2]]],
    }

    concatenate_inputs = {
        'u_p_net': False,
        'u_net': True,
        'u_net_noskip': True,
    }

    name = str(sys.argv[1])
    if name in num_input_filters:
        num_input_filters = num_input_filters[name]
    else:
        num_input_filters = num_input_filters['u_p_net']
    
    if name in concatenate_inputs:
        concatenate_inputs = concatenate_inputs[name]
    else:
        concatenate_inputs = concatenate_inputs['u_p_net']

    if filter_size[0] == filter_size[1]/2 and filter_size[1] == filter_size[2]/2:
        name += '_x2'
    elif filter_size[0] == filter_size[1] and filter_size[1] == filter_size[2]:
        name += '_'+str(filter_size[0])
    else:
        name += '_'+str(filter_size[0])+'_'+str(filter_size[1])+'_'+str(filter_size[2])

    if len(sys.argv) >= 6:
        gpu = int(sys.argv[5])
    else:       
        gpu = 0

    print(f'name: {name}')
    print(f'num_input_filters: {num_input_filters}')
    print(f'concatenate_inputs: {concatenate_inputs}')
    print(f'gpu: {gpu}')

    # exit()

    path_name = os.path.dirname(sys.argv[0])   
    abs_path = os.path.abspath(path_name) 

    config = {}

    config['training'] = {}
    config['training']['filename_base'] = 'model_'+name
    config['training']['loss'] = 'domain_mse'
    config['training']['domain_mask'] = 1
    config['training']['optimizer'] = 'adam'
    config['training']['amsgrad'] = False
    config['training']['learningrate'] = 1E-4 
    config['training']['learningrate_scheduler'] = 'exp_decay'
    config['training']['learningrate_scheduler_verbose'] = True
    config['training']['learningrate_k'] = 0.0077
    config['training']['epochs'] = 300
    config['training']['patience'] = 300
    config['training']['shuffle'] = True
    config['training']['verbose'] = 2   # 2 = just epoch evaluations | 1 = progress bar for each epoch
    config['training']['gpu'] = gpu       # 0 or 1 or 'all'
    config['training']['output_folder'] = 'data/model_waveequation/'+config['training']['filename_base']+'_after_'+str(config['training']['epochs'])+'epochs'

    config['data'] = {}
    config['data']['folder'] = {}
    config['data']['folder']['train'] = 'data/datasets_train_waveequation'
    config['data']['folder']['test'] = 'data/datasets_test_waveequation'
    config['data']['input_sizes'] = [3,1,1]
    config['data']['output_size'] = 1
    config['data']['input_file_prefixes'] = ['field_dataset_', 'domain_dataset_', 'material_dataset_']
    config['data']['file_batch_size'] = 2
    config['data']['batch_size'] = 128
    config['data']['reference_input'] = 0
    config['data']['augmentation'] = 'mirror_and_flip' 
    config['data']['randomize_files'] = True

    config['model'] = {}
    config['model']['concatenate_inputs']       = concatenate_inputs
    config['model']['padding']                  = 'valid' 
    config['model']['skip_connection_handling'] = 'concatenate'
    config['model']['pooling_handling']         = 'max_pooling'
    config['model']['normalization_handling']   = 'custom_sample_norm'
    config['model']['domain_injection']         = True
    config['model']['domain_injection_ids']     = [1,0]
    config['model']['run_eagerly']              = False # False or True
    config['model']['input_ids_last_timestep']  = [0,2]
    config['model']['num_input_filters']        = num_input_filters
    config['model']['conv_kernel_sizes']        = [3,3,3] # number for each depth-layer
    config['model']['pooling_size']             = 2
    config['model']['depth']                    = 3
    config['model']['activation_function']      = 'relu'
    config['model']['activation_function_out']  = 'linear'
    config['model']['inputs_dangling']          = [0, 0, 0]
    config['model']['label_shape']              = (None, None, config['data']['output_size'])
    config['model']['label_shape_for_loss']     = (None, None, config['data']['output_size'])

    print(config['training']['filename_base'])

    [min_loss, min_val_loss, min_loss_epoch, min_val_loss_epoch] = process_training(abs_path, config)

    result = {}
    result['min_loss'] = min_loss
    result['min_val_loss'] = min_val_loss
    result['min_loss_epoch'] = int(min_loss_epoch)
    result['min_val_loss_epoch'] = int(min_val_loss_epoch)
    result['filename_base'] = config['training']['filename_base']
    result['iteration'] = 0
    result['padding'] = config['model']['padding'] 
    result['pooling_size'] = config['model']['pooling_size'] 
    result['depth'] = config['model']['depth'] 
    result['num_input_filters'] = config['model']['num_input_filters'][0]
    result['conv_kernel_sizes'] = config['model']['conv_kernel_sizes']
    result['loss_function'] = config['training']['loss']
    result['learningrate'] = config['training']['learningrate']

    json.dump(result, open(abs_path+'/'+config['training']['output_folder']+'/'+config['training']['filename_base']+'_result.json', 'w'))

if __name__ == "__main__":
    main()
